#!/usr/bin/env python3
"""Export a compact no-recapture B3.1 factor bundle from frozen B3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.anchor import LegacyHARPAnchorBridge  # noqa: E402
from harp_rtt.b31 import anchor_spine_prefix_matches  # noqa: E402
from harp_rtt.dataset import HarpRTTDataset  # noqa: E402
from harp_rtt.exact_k import stable_topk  # noqa: E402
from harp_rtt.model import HARPRTTTeacher  # noqa: E402
from harp_rtt.model.heads import exact_projected_marginals  # noqa: E402
from harp_rtt.node_counterfactual import (  # noqa: E402
    NodeCounterfactualDatasetAdapter,
    load_node_counterfactual_companion,
)
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.train import (  # noqa: E402
    prepare_model_batch,
    production_config,
    runtime_static_artifacts,
    verify_router_numerics_audit,
)
from harp_rtt.training import move_to_device, seed_everything, sha256_file  # noqa: E402
from harp_rtt.v2_losses import node_target_branch_distribution  # noqa: E402
from runpod.evaluate_harp_rtt_b31_factorial import BUNDLE_SCHEMA  # noqa: E402
from runpod.train_harp_rtt_b2_translator import verify_probe_corpus_binding  # noqa: E402
from runpod.train_harp_rtt_b3 import (  # noqa: E402
    ACTIVE_CANDIDATE_SOURCES,
    BUDGET_INDEX,
    EXPECTED_PROBE_ROWS,
    GENERATOR_SCHEMA,
    load_non_anchor_state,
    make_loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-index", type=Path, required=True)
    parser.add_argument("--probe-corpus", type=Path, required=True)
    parser.add_argument("--probe-companion", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--router-numerics-audit", type=Path, required=True)
    parser.add_argument("--router-numerics-audit-sha256", required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--capture-source-commit", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def checkpoint_and_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != GENERATOR_SCHEMA:
        raise ValueError("B3.1 export requires a completed B3 generator checkpoint")
    manifest_path = path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("B3 checkpoint has no adjacent run manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if checkpoint.get("run_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("B3 checkpoint/run-manifest binding mismatch")
    result_path = path.parent / "B3_GENERATOR_RESULT.json"
    if not result_path.is_file():
        raise FileNotFoundError("B3 checkpoint has no adjacent terminal generator result")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("checkpoint_sha256") != sha256_file(path):
        raise ValueError("B3 terminal result/checkpoint binding mismatch")
    for record in (manifest, result):
        for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
            if record.get(key) is not False:
                raise PermissionError(f"B3 initializer violates {key}")
    return checkpoint, manifest


def _append(storage: dict[str, list[Tensor]], name: str, value: Tensor, dtype: torch.dtype) -> None:
    storage.setdefault(name, []).append(value.detach().to(device="cpu", dtype=dtype))


def _frozen_b3_forward(
    model: HARPRTTTeacher,
    prepared: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor]]:
    """Reproduce the BF16-transform execution contract used by B3."""

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        anchor_output = model.anchor(batch=prepared)
        outputs = model(
            batch=prepared,
            anchor_inputs={"batch": prepared},
            candidate_training_progress=1.0,
            candidate_active_sources=ACTIVE_CANDIDATE_SOURCES,
            random_anytime_truncation=False,
        )
    return anchor_output, outputs


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite B3.1 factor bundle {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    seed_everything(42, deterministic=True)
    checkpoint, training_manifest = checkpoint_and_manifest(args.checkpoint)
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    router_audit = verify_router_numerics_audit(
        args.router_numerics_audit,
        args.router_numerics_audit_sha256,
        static.manifest,
    )
    bridge, anchor_provenance = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint,
        args.target_preprocessing,
        args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    model = HARPRTTTeacher(
        bridge,
        production_config(static, bridge),
        static.geometry,
        token_embedding=static.token_embedding,
    ).to(device).eval()
    load_non_anchor_state(model, checkpoint)
    runtime_static = runtime_static_artifacts(static, device)
    del static

    labels, companion_manifest = load_node_counterfactual_companion(
        args.probe_companion,
        split="train",
        training=True,
        expected_bindings={"source_commit": args.capture_source_commit},
    )
    binding = verify_probe_corpus_binding(args.probe_corpus, companion_manifest)
    base = HarpRTTDataset(
        args.probe_index, "train", corpus_root=args.probe_corpus, max_tree_nodes=32
    )
    probe = NodeCounterfactualDatasetAdapter(
        base, labels, split="train", training=True
    )
    if len(probe) != EXPECTED_PROBE_ROWS or len(labels) != EXPECTED_PROBE_ROWS:
        raise ValueError("B3.1 export requires the frozen 2,048-position probe")

    dense: dict[str, list[Tensor]] = {}
    sources: dict[str, list[Tensor]] = {}
    posteriors: dict[str, dict[str, list[Tensor]]] = {
        "learned": {"captured": [], "other": []},
        "target": {"captured": [], "other": []},
    }
    for host in make_loader(
        probe, batch_size=args.batch_size, shuffle=False, seed=42,
        workers=args.num_workers, device=device,
    ):
        batch = move_to_device(host, device)
        prepared = prepare_model_batch(batch, runtime_static)
        anchor_output, outputs = _frozen_b3_forward(
            model, prepared, device=device
        )
        anchor_scores = anchor_output["future_router_scores"][:, :4].float()
        _, anchor_marginals, _ = exact_projected_marginals(anchor_scores, 8)
        counterfactual = prepared["targets"]["counterfactual"]
        selection = counterfactual["budget_node_masks"][:, BUDGET_INDEX].bool()
        nodes = selection.shape[-1]
        label_valid = counterfactual["valid"].bool().any(-1)
        branch_mask = (
            outputs["branch_mask"][..., :nodes].bool()
            & selection[:, None]
            & label_valid[:, None]
        )
        target, _ = node_target_branch_distribution(
            counterfactual, captured_nodes=nodes, selection_mask=selection
        )
        learned = outputs["branch_weights"].float()
        # H1 has no counterfactual target replay. Keep the target-posterior
        # condition equal to the learned causal posterior at H1 so B3.1 does
        # not fabricate a target label for the independent exact-root task.
        target[:, 0] = learned[:, 0]

        _append(dense, "anchor_scores", anchor_scores, torch.bfloat16)
        _append(dense, "anchor_marginals", anchor_marginals, torch.bfloat16)
        _append(
            dense, "target_ids", prepared["targets"]["future_selected_ids"], torch.int16
        )
        _append(dense, "branch_mask", branch_mask, torch.bool)
        greedy_match = anchor_spine_prefix_matches(
            prepared["anchor_inputs"]["mtp_spine"]["exact_prefix_hashes"],
            prepared["targets"]["future_prefix_hashes"],
        )
        first_h2 = ~greedy_match[:, 1]
        first_h3 = greedy_match[:, 1] & ~greedy_match[:, 2]
        first_h4 = greedy_match[:, 1] & greedy_match[:, 2] & ~greedy_match[:, 3]
        fully = greedy_match[:, 1:4].all(-1)
        _append(dense, "strata_prefix_matched", greedy_match, torch.bool)
        _append(dense, "strata_prefix_mismatch", ~greedy_match, torch.bool)
        for name, row_mask in (
            ("first_divergence_h2", first_h2),
            ("first_divergence_h3", first_h3),
            ("first_divergence_h4", first_h4),
            ("fully_matched_h4", fully),
        ):
            _append(
                dense, f"strata_{name}",
                row_mask[:, None].expand(-1, 4), torch.bool,
            )
        posteriors["learned"]["captured"].append(learned[..., :nodes].cpu())
        posteriors["learned"]["other"].append(learned[..., nodes].cpu())
        posteriors["target"]["captured"].append(target[..., :nodes].cpu())
        posteriors["target"]["other"].append(target[..., nodes].cpu())

        score_sources = {
            "semantic": outputs["branch_semantic_scores"][..., :nodes, :],
            "geometry": outputs["branch_geometry_scores"][..., :nodes, :],
            "free": outputs["branch_free_scores"][..., :nodes, :],
            "learned_branch": outputs["branch_scores"][..., :nodes, :],
        }
        for name, score in score_sources.items():
            _append(sources, name, stable_topk(score.float(), 8), torch.int16)

        native = counterfactual["selected_ids"].long().permute(0, 2, 1, 3)
        native = native[:, None].expand(-1, 4, -1, -1, -1).clone()
        semantic_ids = stable_topk(
            outputs["branch_semantic_scores"][..., :nodes, :].float(), 8
        )
        native[:, 0] = semantic_ids[:, 0]
        _append(sources, "oracle_native", native, torch.int16)

        mixture_ids = stable_topk(outputs["branch_mixture_marginals"].float(), 8)
        trajectory_ids = stable_topk(outputs["trajectory_round_scores"][:, -1].float(), 8)
        for name, ids in (("mixture", mixture_ids), ("trajectory", trajectory_ids)):
            expanded = ids[..., None, :].expand(-1, -1, -1, nodes, -1)
            _append(sources, name, expanded, torch.int16)

    bundle: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "outer_split": "train",
            "partition": "diagnostic_probe",
            "rows": EXPECTED_PROBE_ROWS,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_source_commit": checkpoint["source_commit"],
            "export_source_commit": args.source_commit,
            "training_manifest_sha256": sha256_file(args.checkpoint.parent / "run_manifest.json"),
            "companion_manifest_sha256": sha256_file(args.probe_companion / "manifest.json"),
            "router_numerics_audit_sha256": sha256_file(args.router_numerics_audit),
            "router_audit_schema": router_audit.get("schema"),
            "probe_binding": binding,
            "anchor_provenance": anchor_provenance,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        },
        "anchor_scores": torch.cat(dense["anchor_scores"]),
        "anchor_marginals": torch.cat(dense["anchor_marginals"]),
        "target_ids": torch.cat(dense["target_ids"]),
        "branch_mask": torch.cat(dense["branch_mask"]),
        "branch_selected_ids": {
            name: torch.cat(values) for name, values in sorted(sources.items())
        },
        "posteriors": {
            name: {
                key: torch.cat(values) for key, values in condition.items()
            }
            for name, condition in posteriors.items()
        },
        "strata_masks": {
            name.removeprefix("strata_"): torch.cat(values)
            for name, values in dense.items()
            if name.startswith("strata_")
        },
        "exact_k": 8,
        "candidate_width": 64,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        torch.save(bundle, handle)
        handle.flush(); os.fsync(handle.fileno())
    result = {
        "schema": "harp_rtt_b31_factor_bundle_export_v1",
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "bytes": args.output.stat().st_size,
        "rows": EXPECTED_PROBE_ROWS,
        "sources": sorted(sources),
        "training_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
