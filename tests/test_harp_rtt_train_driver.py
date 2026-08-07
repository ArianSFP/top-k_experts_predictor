from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from harp_rtt.geometry import build_centered_router_geometry
from harp_rtt.static_artifacts import StaticTargetArtifacts
from harp_rtt.train import (
    EFFECTIVE_BATCH_SIZE,
    MAX_PROCESS_PEAK_BYTES,
    MICROBATCH_CANDIDATES,
    _conservative_adamw_reservation_bytes,
    _epoch_seed,
    _expected_parent_phase,
    _fresh_or_resume_output,
    _optimizer_steps_per_epoch,
    autotune_microbatch_size,
    build_parser,
    forward_model_batch,
    loss_dimensions,
    prepare_model_batch,
    production_config,
    request_complete_subset,
    require_formal_adaptive_index_inventory,
    resolve_phase_spec,
    verify_index_inventory,
    verify_router_numerics_audit,
)
from harp_rtt.training import sha256_file


REQUIRED = [
    "--index-root", "index",
    "--corpus-root", "corpus",
    "--static-dir", "static",
    "--router-numerics-audit", "router_audit.json",
    "--router-numerics-audit-sha256", "b" * 64,
    "--anchor-checkpoint", "anchor.pt",
    "--anchor-sha256", "a" * 64,
    "--target-preprocessing", "target.pt",
    "--mtp-preprocessing", "mtp.pt",
    "--output-dir", "out",
]


def test_parser_exposes_only_train_validation_driver_paths() -> None:
    parser = build_parser()
    args = parser.parse_args([*REQUIRED, "--phase", "phase2"])
    assert args.index_root.name == "index"
    assert args.anchor_sha256 == "a" * 64
    assert args.phase == "phase2"
    options = parser._option_string_actions
    assert "--allow-test" not in options
    assert "--split" not in options
    with pytest.raises(SystemExit):
        parser.parse_args([*REQUIRED, "--phase", "phase2", "--anchor-sha256", "bad"])


def test_phase_resolution_requires_a_parent_after_phase2() -> None:
    parser = build_parser()
    phase2 = parser.parse_args([*REQUIRED, "--phase", "phase2", "--epochs", "2"])
    assert resolve_phase_spec(phase2).epochs == 2
    phase3 = parser.parse_args([*REQUIRED, "--phase", "phase3"])
    with pytest.raises(ValueError, match="requires --initialize-from or --resume"):
        resolve_phase_spec(phase3)
    phase3.initialize_from = phase3.anchor_checkpoint
    spec = resolve_phase_spec(phase3)
    assert spec.name == "phase3"
    assert spec.legacy_learning_rate <= spec.new_learning_rate / 5


class _FakeRichDataset:
    def __init__(self, split: str = "train") -> None:
        self.split = split
        self.segments = [
            SimpleNamespace(
                sequences=[
                    {"request_id": "r2"},
                    {"request_id": "r0"},
                    {"request_id": "r1"},
                ]
            )
        ]
        # Deliberately interleave request rows.
        self.records = [
            SimpleNamespace(segment=0, sequence=0),
            SimpleNamespace(segment=0, sequence=1),
            SimpleNamespace(segment=0, sequence=0),
            SimpleNamespace(segment=0, sequence=2),
            SimpleNamespace(segment=0, sequence=1),
            SimpleNamespace(segment=0, sequence=2),
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index}


def test_request_subsetting_is_deterministic_and_request_complete() -> None:
    dataset = _FakeRichDataset()
    first, first_manifest = request_complete_subset(dataset, 2, seed=42)  # type: ignore[arg-type]
    second, second_manifest = request_complete_subset(dataset, 2, seed=42)  # type: ignore[arg-type]
    assert first_manifest == second_manifest
    assert first.indices == second.indices  # type: ignore[attr-defined]
    selected = set(first_manifest["selected_request_ids"])
    expected = [
        index
        for index, record in enumerate(dataset.records)
        if dataset.segments[0].sequences[record.sequence]["request_id"] in selected
    ]
    assert first.indices == expected  # type: ignore[attr-defined]
    assert first_manifest["selected_rows"] == 4
    with pytest.raises(PermissionError, match="never accepts"):
        request_complete_subset(_FakeRichDataset("test"), 1, seed=42)  # type: ignore[arg-type]


def test_microbatch_autotune_uses_fixed_order_and_20_gib_cap() -> None:
    seen: list[int] = []

    def probe(candidate: int) -> int:
        seen.append(candidate)
        if candidate == 8:
            raise torch.cuda.OutOfMemoryError("synthetic out of memory")
        return 21 * 1024**3 if candidate == 4 else 19 * 1024**3

    selected, trials = autotune_microbatch_size(probe)
    assert MICROBATCH_CANDIDATES == (8, 4, 2, 1)
    assert MAX_PROCESS_PEAK_BYTES == 20 * 1024**3
    assert selected == 2
    assert seen == [8, 4, 2]
    assert trials[0]["oom"] is True
    assert trials[-1]["within_limit"] is True
    with pytest.raises(ValueError, match="exactly 8,4,2,1"):
        autotune_microbatch_size(lambda _: 0, candidates=(4, 2, 1))


def _static(hidden: int = 8, layers: int = 2, experts: int = 64) -> StaticTargetArtifacts:
    geometry = build_centered_router_geometry(
        torch.randn(layers, experts, hidden), relative_rank_threshold=0.0
    )
    return StaticTargetArtifacts(
        geometry=geometry,
        token_embedding=torch.randn(19, hidden, dtype=torch.bfloat16),
        final_rmsnorm_weight=torch.randn(hidden, dtype=torch.bfloat16),
        final_rmsnorm_epsilon=1e-5,
        manifest={"contract": {"experts_per_token": 8}},
    )


def test_production_config_is_derived_from_verified_static_and_anchor() -> None:
    static = _static()
    anchor = SimpleNamespace(
        config=SimpleNamespace(layers=2, experts=64, horizons=8, route_history=8)
    )
    config = production_config(static, anchor)  # type: ignore[arg-type]
    assert config.router_rank == static.geometry.maximum_rank
    assert config.exact_token_width == 8
    assert config.candidate_width == 64
    dimensions = loss_dimensions(config)
    assert dimensions.selected_experts == 8
    assert dimensions.candidates == 64


class _CaptureForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prepared: dict[str, object] | None = None
        self.anchor_batch: dict[str, object] | None = None

    def forward(
        self,
        *,
        batch: dict[str, object],
        anchor_inputs: dict[str, dict[str, object]],
    ) -> dict[str, torch.Tensor]:
        self.prepared = batch
        self.anchor_batch = anchor_inputs["batch"]
        return {"active_scores": torch.zeros(1)}


def test_final_hidden_is_bf16_rmsnormalized_before_model_and_anchor_call() -> None:
    static = _static(hidden=4)
    raw = torch.randn(2, 4, dtype=torch.float32)
    batch = {"inputs": {"final_hidden": raw}, "targets": {}}
    prepared = prepare_model_batch(batch, static)
    expected = static.reconstruct_final_hidden(raw, activation_dtype=torch.bfloat16)
    assert torch.equal(prepared["inputs"]["final_hidden"], expected)
    assert batch["inputs"]["final_hidden"] is raw

    model = _CaptureForward()
    forward_model_batch(model, batch, static)
    assert model.prepared is model.anchor_batch
    assert torch.equal(model.prepared["inputs"]["final_hidden"], expected)


def test_effective_batch_accumulation_is_fixed_at_32() -> None:
    assert EFFECTIVE_BATCH_SIZE == 32
    assert _optimizer_steps_per_epoch(65, 8) == (3, 4)
    assert _optimizer_steps_per_epoch(65, 4) == (3, 8)
    assert _optimizer_steps_per_epoch(65, 2) == (3, 16)
    assert _optimizer_steps_per_epoch(65, 1) == (3, 32)


def test_fresh_output_refuses_even_an_empty_existing_directory(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    args = SimpleNamespace(output_dir=output, resume=None)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _fresh_or_resume_output(args)  # type: ignore[arg-type]


def test_restart_seed_and_adamw_reservation_are_deterministic() -> None:
    assert _epoch_seed(42, "phase2", 3) == _epoch_seed(42, "phase2", 3)
    assert _epoch_seed(42, "phase2", 3) != _epoch_seed(42, "phase2", 4)
    assert _expected_parent_phase("phase3") == "phase2"
    parameter = nn.Parameter(torch.zeros(7))
    optimizer = torch.optim.AdamW([parameter])
    assert _conservative_adamw_reservation_bytes(optimizer) == 7 * 12


def test_resume_uses_a_new_hash_linked_lineage(tmp_path) -> None:
    source = tmp_path / "source"
    checkpoints = source / "checkpoints"
    checkpoints.mkdir(parents=True)
    source_manifest = source / "run_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema": "harp_rtt_training_driver_v1",
                "phase": {"name": "phase2"},
            }
        )
    )
    resume = checkpoints / "epoch_001_best.manifest.json"
    resume.write_text(
        json.dumps(
            {
                "schema": "harp_rtt_checkpoint_manifest_v1",
                "provenance": {
                    "driver_schema": "harp_rtt_training_driver_v1",
                    "run_manifest": "run_manifest.json",
                    "run_manifest_sha256": sha256_file(source_manifest),
                },
            }
        )
    )
    output = tmp_path / "continuation"
    resolved, manifest = _fresh_or_resume_output(
        SimpleNamespace(output_dir=output, resume=resume, phase="phase2")
    )
    assert resolved == output.resolve()
    assert manifest is not None
    assert manifest["phase"]["name"] == "phase2"
    assert not output.exists()


def test_index_inventory_verifies_every_ledger_file(tmp_path) -> None:
    root = tmp_path / "index"
    segment = root / "seg0"
    segment.mkdir(parents=True)
    summary = {
        "schema": "harp_rtt_rich_event_index_collection_v1",
        "segments": 1,
        "split_manifest_sha256": "b" * 64,
    }
    (root / "INDEX_SUMMARY.json").write_text(json.dumps(summary))
    manifest = {"schema": "harp_rtt_rich_event_index_v1", "segment": "seg0"}
    (segment / "index_manifest.json").write_text(json.dumps(manifest))
    (segment / "rows.npy").write_bytes(b"immutable")
    hashes = {
        name: sha256_file(segment / name)
        for name in ("index_manifest.json", "rows.npy")
    }
    (segment / "INDEX_SHA256SUMS.json").write_text(json.dumps(hashes))
    inventory = verify_index_inventory(root)
    assert inventory["segments"] == 1
    assert inventory["adaptive_contract"] == {
        "required_decoding_profile": "exact_h1_native_mtp_adaptive_h2_h4",
        "required_anchor_spine_schema": "harp_rtt_legacy_anchor_spine_v1",
        "required_anchor_spine_depth": 6,
        "fully_adaptive_segments": 0,
        "complete_anchor_spine_segments": 0,
        "legacy_or_incomplete_segments": 1,
        "all_segments_fully_adaptive": False,
        "all_segments_complete_anchor_spines": False,
    }
    with pytest.raises(ValueError, match="requires every rich-index segment"):
        require_formal_adaptive_index_inventory(inventory)
    (segment / "rows.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_index_inventory(root)


def test_formal_training_accepts_only_complete_adaptive_index_contract() -> None:
    inventory = {
        "segments": 2,
        "adaptive_contract": {
            "required_decoding_profile": "exact_h1_native_mtp_adaptive_h2_h4",
            "required_anchor_spine_schema": "harp_rtt_legacy_anchor_spine_v1",
            "required_anchor_spine_depth": 6,
            "fully_adaptive_segments": 2,
            "complete_anchor_spine_segments": 2,
            "legacy_or_incomplete_segments": 0,
            "all_segments_fully_adaptive": True,
            "all_segments_complete_anchor_spines": True,
        },
    }
    assert require_formal_adaptive_index_inventory(inventory) == inventory[
        "adaptive_contract"
    ]
    for broken in (
        {"segments": 0, "adaptive_contract": inventory["adaptive_contract"]},
        {"segments": 2},
        {
            "segments": 2,
            "adaptive_contract": {
                **inventory["adaptive_contract"],
                "legacy_or_incomplete_segments": 1,
            },
        },
    ):
        with pytest.raises(ValueError, match="adaptive"):
            require_formal_adaptive_index_inventory(broken)


def test_router_numerics_audit_is_hash_bound_to_static_geometry(tmp_path) -> None:
    geometry_sha = "c" * 64
    checkpoint_index_sha = "d" * 64
    revision = "e" * 40
    report = {
        "schema": "harp_rtt_router_numerics_diagnostic_v2",
        "split": "validation",
        "test_accessed": False,
        "valid_endpoints": 640,
        "captured_tied_boundary_endpoints": 70,
        "source_identity": {
            "static_geometry_sha256": geometry_sha,
            "checkpoint_revision": revision,
            "checkpoint_index_sha256": checkpoint_index_sha,
        },
        "gate_assessment": {
            "canonical_factorization_passed": True,
            "captured_bf16_strict_boundary_passed": True,
            "training_supported_by_static_v1": True,
            "cross_architecture_bf16_exact_replay_required": False,
            "canonical_factorization_observed": {
                "exact_set": 1.0,
                "slot_recall": 1.0,
                "max_abs_error": 1e-5,
            },
            "captured_bf16_strict_boundary_observed": {
                "exact_set": 1.0,
                "slot_recall": 1.0,
                "max_abs_error": 0.03,
            },
        },
    }
    path = tmp_path / "router_audit.json"
    path.write_text(json.dumps(report))
    static_manifest = {
        "files": {"router_geometry.safetensors": {"sha256": geometry_sha}},
        "model": {
            "repository_revision": revision,
            "index_sha256": checkpoint_index_sha,
        },
    }
    binding = verify_router_numerics_audit(
        path, sha256_file(path), static_manifest
    )
    assert binding["sha256"] == sha256_file(path)
    assert binding["test_accessed"] is False
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_router_numerics_audit(path, "0" * 64, static_manifest)
