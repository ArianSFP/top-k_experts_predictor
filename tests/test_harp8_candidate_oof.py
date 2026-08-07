from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from harp8.candidates import (
    DIRECT_EXTENSION_PREFIXES,
    POOL_SCHEMA,
    POOL_SCHEMA_V2,
    REQUEST_IDS_HASH_ENCODING,
    _load_checkpoint,
    export_candidate_pool,
    request_ids_sha256,
)
from harp8.config import HARPConfig
from harp8.data import CompactHARPData
from harp8.jspace_router_data import (
    _read_pool_manifest,
    assert_disjoint_level2_request_sets,
    validate_level2_pool_request_contract,
)
from harp8.model import HARP8Teacher
from harp8.train_jspace_reranker import (
    _candidate_pool_split_lineage,
    _read_pool_manifest as _read_reranker_pool_manifest,
)


def _fixture(tmp_path: Path) -> tuple[CompactHARPData, CompactHARPData, Path, Path]:
    capture = tmp_path / "capture"
    mtp = tmp_path / "mtp"
    capture.mkdir()
    mtp.mkdir()
    config = HARPConfig(
        experts=12,
        layers=2,
        horizons=2,
        route_history=2,
        mtp_depths=2,
        target_state_channels=1,
        target_state_width=4,
        mtp_state_channels=1,
        mtp_state_width=4,
        mtp_state_projection_width=8,
        mtp_metadata_width=4,
        mtp_vocab_width=0,
        route_width=16,
        state_width=16,
        mtp_width=16,
        model_width=16,
        route_ffn_width=32,
        mtp_ffn_width=32,
        fusion_ffn_width=32,
        attention_heads=4,
        temporal_blocks=1,
        layer_blocks=1,
        state_blocks=1,
        mtp_cross_blocks=1,
        fusion_blocks=1,
        dropout=0.0,
        future_latent_width=0,
    )
    requests = [
        {"request_id": 11, "offline_split": "train", "domain": "a"},
        {"request_id": 22, "offline_split": "train", "domain": "b"},
        {"request_id": 33, "offline_split": "validation", "domain": "a"},
        {"request_id": 44, "offline_split": "test", "domain": "b"},
    ]
    (capture / "requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8"
    )
    rows_per_request = 4
    rows = len(requests) * rows_per_request
    rng = np.random.default_rng(9)
    logits = rng.normal(size=(rows, config.layers, config.experts)).astype(np.float32)
    np.save(capture / "raw_router_logits.npy", logits)
    np.save(
        capture / "top8_expert_ids.npy",
        np.argsort(-logits, axis=-1)[..., :8].astype(np.uint16),
    )
    np.save(
        mtp / "mtp_router_logits_depths.npy",
        rng.normal(size=(rows, 2, config.experts)).astype(np.float32),
    )
    target = rng.normal(size=(rows, config.layers, 4)).astype(np.float16)
    mtp_states = rng.normal(size=(rows, 2, 4)).astype(np.float16)
    inner_path = tmp_path / "inner_split_manifest.json"
    inner_path.write_text(
        json.dumps(
            {
                "schema": "harp8_inner_split_v1",
                "source_offline_split": "train",
                "assignments": {"11": "train", "22": "validation"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    inner = CompactHARPData(
        capture,
        mtp,
        target,
        mtp_states,
        config,
        rows_per_request=rows_per_request,
        split_manifest=inner_path,
    )
    outer = CompactHARPData(
        capture,
        mtp,
        target,
        mtp_states,
        config,
        rows_per_request=rows_per_request,
    )
    checkpoint = tmp_path / "best.pt"
    model = HARP8Teacher(config).eval()
    torch.save(
        {
            "schema": "harp8t_training_v1",
            "model_config": config.to_dict(),
            "model_state": model.state_dict(),
            "epoch": 1,
            "seed": 42,
        },
        checkpoint,
    )
    return inner, outer, checkpoint, inner_path


def test_inner_heldout_rows_are_explicit_oof_level2_training(tmp_path: Path) -> None:
    inner, _outer, checkpoint, inner_path = _fixture(tmp_path)
    output = tmp_path / "oof-train"
    manifest = export_candidate_pool(
        inner,
        checkpoint,
        output,
        source_data_split="validation",
        level2_split="train",
        source_offline_split="train",
        base_fold_id="inner-fold-0",
        base_fit_excluded=True,
        base_fit_split_manifest=inner_path,
        candidate_count=10,
        batch_size=2,
        device="cpu",
    )
    assert manifest["schema"] == POOL_SCHEMA_V2
    assert manifest["source_data_split"] == "validation"
    assert manifest["source_offline_split"] == "train"
    assert manifest["level2_split"] == "train"
    assert manifest["split"] == "train"
    assert manifest["base_train_request_ids"] == [11]
    assert manifest["base_train_request_ids_sha256"] == request_ids_sha256([11])
    assert manifest["exported_request_ids_sha256"] == request_ids_sha256([22])
    assert manifest["request_ids_hash_encoding"] == REQUEST_IDS_HASH_ENCODING
    assert manifest["base_fit_overlap_count"] == 0
    assert manifest["source_row_split_manifest"]["path"] == str(inner_path)
    assert manifest["base_fit_split_manifest"]["path"] == str(inner_path)
    accepted = _read_pool_manifest(output, "train")
    assert accepted["base_fit_excluded"] is True


def test_pre_direct_checkpoint_export_is_bitwise_equivalent(tmp_path: Path) -> None:
    inner, _outer, checkpoint, inner_path = _fixture(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model_state"] = {
        key: value
        for key, value in payload["model_state"].items()
        if not key.startswith(DIRECT_EXTENSION_PREFIXES)
    }
    legacy_checkpoint = tmp_path / "pre-direct.pt"
    torch.save(payload, legacy_checkpoint)

    strict_output = tmp_path / "strict-output"
    legacy_output = tmp_path / "legacy-output"
    common = {
        "source_data_split": "validation",
        "level2_split": "train",
        "source_offline_split": "train",
        "base_fold_id": "inner-fold-0",
        "base_fit_excluded": True,
        "base_fit_split_manifest": inner_path,
        "candidate_count": 10,
        "batch_size": 2,
        "device": "cpu",
    }
    strict_manifest = export_candidate_pool(
        inner, checkpoint, strict_output, **common
    )
    legacy_manifest = export_candidate_pool(
        inner, legacy_checkpoint, legacy_output, **common
    )
    assert strict_manifest["checkpoint_load_profile"] == "strict"
    assert (
        legacy_manifest["checkpoint_load_profile"]
        == "zero_initialized_direct_extension"
    )
    for filename in (
        "candidate_scores.f32",
        "candidate_ids.u2",
        "target_membership.u1",
        "teacher_candidate_scores.f32",
        "valid_future.u1",
        "current_scores.f32",
        "current_rank.f32",
        "source_gates.f16",
        "copy_gates.f16",
        "generator_context.f16",
    ):
        assert (strict_output / filename).read_bytes() == (
            legacy_output / filename
        ).read_bytes()


def test_pre_direct_compatibility_rejects_other_state_mismatch(
    tmp_path: Path,
) -> None:
    _inner, _outer, checkpoint, _inner_path = _fixture(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model_state"].pop("route_to_model.weight")
    payload["model_state"]["undeclared_extension.weight"] = torch.zeros(1)
    incompatible = tmp_path / "incompatible.pt"
    torch.save(payload, incompatible)
    with pytest.raises(ValueError, match="complete zero-initialized direct"):
        _load_checkpoint(incompatible, "cpu")


def test_pre_direct_compatibility_rejects_partial_direct_missing_state(
    tmp_path: Path,
) -> None:
    _inner, _outer, checkpoint, _inner_path = _fixture(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    one_direct_key = next(
        key
        for key in payload["model_state"]
        if key.startswith(DIRECT_EXTENSION_PREFIXES)
    )
    payload["model_state"].pop(one_direct_key)
    partial = tmp_path / "partial-direct.pt"
    torch.save(payload, partial)
    with pytest.raises(ValueError, match="complete zero-initialized direct"):
        _load_checkpoint(partial, "cpu")


def test_outer_validation_can_use_distinct_base_fit_manifest(tmp_path: Path) -> None:
    _inner, outer, checkpoint, inner_path = _fixture(tmp_path)
    output = tmp_path / "outer-validation"
    manifest = export_candidate_pool(
        outer,
        checkpoint,
        output,
        source_data_split="validation",
        level2_split="validation",
        source_offline_split="validation",
        base_fold_id="inner-fold-0",
        base_fit_excluded=True,
        base_fit_split_manifest=inner_path,
        candidate_count=10,
        batch_size=2,
        device="cpu",
    )
    assert manifest["source_row_split_manifest"] is None
    assert manifest["split_manifest"] is None
    assert manifest["base_fit_split_manifest"]["path"] == str(inner_path)
    assert manifest["exported_request_ids_sha256"] == request_ids_sha256([33])
    assert manifest["base_train_request_ids_sha256"] == request_ids_sha256([11])
    assert _read_pool_manifest(output, "validation")["level2_split"] == "validation"


def test_level2_train_fails_closed_without_exclusion_proof(tmp_path: Path) -> None:
    inner, _outer, checkpoint, _inner_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="base_fit_excluded"):
        export_candidate_pool(
            inner,
            checkpoint,
            tmp_path / "unsafe",
            source_data_split="validation",
            level2_split="train",
            source_offline_split="train",
            base_fold_id="inner-fold-0",
            base_fit_excluded=False,
            candidate_count=10,
            batch_size=2,
            device="cpu",
        )
    assert not (tmp_path / "unsafe").exists()


def test_claimed_exclusion_rejects_request_overlap(tmp_path: Path) -> None:
    inner, _outer, checkpoint, _inner_path = _fixture(tmp_path)
    ids = tmp_path / "base_ids.json"
    ids.write_text(json.dumps({"request_ids": [11, 22]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        export_candidate_pool(
            inner,
            checkpoint,
            tmp_path / "overlap",
            source_data_split="validation",
            level2_split="train",
            source_offline_split="train",
            base_fold_id="inner-fold-0",
            base_fit_excluded=True,
            base_train_request_ids=ids,
            candidate_count=10,
            batch_size=2,
            device="cpu",
        )
    assert not (tmp_path / "overlap").exists()


def test_source_offline_split_is_verified_not_relabelled(tmp_path: Path) -> None:
    inner, _outer, checkpoint, inner_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="source_offline_split"):
        export_candidate_pool(
            inner,
            checkpoint,
            tmp_path / "relabelled",
            source_data_split="validation",
            level2_split="train",
            source_offline_split="validation",
            base_fold_id="inner-fold-0",
            base_fit_excluded=True,
            base_fit_split_manifest=inner_path,
            candidate_count=10,
            batch_size=2,
            device="cpu",
        )


def test_v1_validation_is_readable_but_v1_train_requires_legacy_override(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "manifest.json").write_text(
        json.dumps(
            {
                "schema": POOL_SCHEMA,
                "split": "validation",
                "allow_test": False,
                "store_context": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert _read_pool_manifest(validation, "validation")["schema"] == POOL_SCHEMA

    train = tmp_path / "train"
    train.mkdir()
    (train / "manifest.json").write_text(
        json.dumps(
            {
                "schema": POOL_SCHEMA,
                "split": "train",
                "allow_test": False,
                "store_context": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot be used for level-2 training"):
        _read_pool_manifest(train, "train")
    assert (
        _read_pool_manifest(train, "train", allow_legacy_level2_train=True)["schema"]
        == POOL_SCHEMA
    )


def test_sealed_test_source_is_refused_before_output_creation(tmp_path: Path) -> None:
    _inner, outer, checkpoint, inner_path = _fixture(tmp_path)
    with pytest.raises(PermissionError, match="sealed test"):
        export_candidate_pool(
            outer,
            checkpoint,
            tmp_path / "test-output",
            source_data_split="test",
            level2_split="validation",
            source_offline_split="test",
            base_fold_id="inner-fold-0",
            base_fit_excluded=True,
            base_fit_split_manifest=inner_path,
            candidate_count=10,
            batch_size=2,
            device="cpu",
        )
    assert not (tmp_path / "test-output").exists()


def test_loader_rejects_tampered_v2_request_hash(tmp_path: Path) -> None:
    root = tmp_path / "tampered"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": POOL_SCHEMA_V2,
                "split": "train",
                "split_field_semantics": "deprecated_alias_of_level2_split",
                "level2_split": "train",
                "source_data_split": "validation",
                "source_offline_split": "train",
                "base_fold_id": "fold-0",
                "base_fit_excluded": True,
                "base_fit_overlap_count": 0,
                "base_train_request_ids": [11],
                "base_train_request_ids_sha256": request_ids_sha256([11]),
                "exported_request_ids_sha256": "not-a-sha",
                "request_ids_hash_encoding": REQUEST_IDS_HASH_ENCODING,
                "allow_test": False,
                "store_context": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exported request-ID hash"):
        _read_pool_manifest(root, "train")


def _write_v2_preflight_manifest(
    root: Path,
    *,
    role: str = "train",
    source_data_split: str = "validation",
    source_offline_split: str = "train",
    schema: str = POOL_SCHEMA_V2,
    allow_test: bool = False,
) -> dict[str, object]:
    root.mkdir()
    manifest: dict[str, object] = {
        "schema": schema,
        "split": role,
        "split_field_semantics": "deprecated_alias_of_level2_split",
        "level2_split": role,
        "source_data_split": source_data_split,
        "source_offline_split": source_offline_split,
        "base_fold_id": "inner-holdout-0",
        "base_fit_excluded": True,
        "base_fit_overlap_count": 0,
        "base_train_request_ids": [11],
        "base_train_request_ids_sha256": request_ids_sha256([11]),
        "exported_request_ids_sha256": request_ids_sha256([22]),
        "request_ids_hash_encoding": REQUEST_IDS_HASH_ENCODING,
        "allow_test": allow_test,
        "store_context": True,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return manifest


def test_reranker_preflight_admits_v2_role_and_rejects_wrong_role(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v2"
    _write_v2_preflight_manifest(root)
    accepted = _read_reranker_pool_manifest(root, "train")
    assert accepted["level2_split"] == "train"
    with pytest.raises(ValueError, match="expected 'validation' pool"):
        _read_reranker_pool_manifest(root, "validation")


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"schema": "unknown_pool_v9"}, ValueError, "incompatible schema"),
        ({"split": "validation"}, ValueError, "compatibility split"),
        ({"allow_test": True}, PermissionError, "sealed test"),
        ({"source_data_split": "test"}, PermissionError, "test source"),
        ({"source_offline_split": "test"}, PermissionError, "test offline"),
    ],
)
def test_reranker_preflight_rejects_mixed_or_test_backed_v2_pool(
    tmp_path: Path,
    changes: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    root = tmp_path / f"case-{len(list(tmp_path.iterdir()))}"
    manifest = _write_v2_preflight_manifest(root)
    manifest.update(changes)
    (root / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    with pytest.raises(error, match=match):
        _read_reranker_pool_manifest(root, "train")


def test_loaded_v2_metadata_recomputes_hash_and_base_fit_exclusion() -> None:
    manifest = {
        "schema": POOL_SCHEMA_V2,
        "exported_request_ids_sha256": request_ids_sha256([22]),
        "base_train_request_ids": [11],
        "base_train_request_ids_sha256": request_ids_sha256([11]),
        "base_fit_excluded": True,
    }
    validate_level2_pool_request_contract(
        manifest, np.asarray([22, 22]), expected_split="train"
    )
    with pytest.raises(ValueError, match="exported request-ID hash"):
        validate_level2_pool_request_contract(
            manifest, np.asarray([33]), expected_split="train"
        )
    overlapping = dict(manifest)
    overlapping["exported_request_ids_sha256"] = request_ids_sha256([11])
    with pytest.raises(ValueError, match="overlaps the declared base-fit"):
        validate_level2_pool_request_contract(
            overlapping, np.asarray([11]), expected_split="train"
        )


def test_level2_train_validation_request_sets_must_be_disjoint() -> None:
    train = SimpleNamespace(
        pool=SimpleNamespace(request_ids=np.asarray([11, 11, 22], dtype=np.int64))
    )
    validation = SimpleNamespace(
        pool=SimpleNamespace(request_ids=np.asarray([22, 33], dtype=np.int64))
    )
    with pytest.raises(ValueError, match="train/validation request sets overlap"):
        assert_disjoint_level2_request_sets(train, validation)

    validation.pool.request_ids = np.asarray([33, 33], dtype=np.int64)
    assert_disjoint_level2_request_sets(train, validation)


def _scientific_oof_manifests() -> tuple[dict[str, object], dict[str, object]]:
    source = {"path": "inner.json", "sha256": "a" * 64}
    base = {"path": "inner.json", "sha256": "a" * 64}
    common: dict[str, object] = {
        "schema": POOL_SCHEMA_V2,
        "base_fold_id": "inner-holdout-0",
        "base_fit_excluded": True,
        "base_fit_overlap_count": 0,
        "base_train_request_ids_sha256": "b" * 64,
        "base_fit_split_manifest": base,
    }
    train = {
        **common,
        "level2_split": "train",
        "source_offline_split": "train",
        "source_row_split_manifest": source,
    }
    validation = {
        **common,
        "level2_split": "validation",
        "source_offline_split": "validation",
        "source_row_split_manifest": None,
    }
    return train, validation


def test_scientific_lineage_accepts_distinct_oof_source_split_manifests() -> None:
    train, validation = _scientific_oof_manifests()
    lineage = _candidate_pool_split_lineage(train, validation)
    assert lineage["mode"] == "level2_oof_v2"
    assert lineage["base_fit_split_manifest_sha256"] == "a" * 64
    assert lineage["source_row_split_manifest_sha256"] == {
        "train": "a" * 64,
        "validation": None,
    }


@pytest.mark.parametrize(
    ("side", "field", "value", "match"),
    [
        ("validation", "base_fold_id", "other", "base-fold"),
        ("validation", "base_train_request_ids_sha256", "c" * 64, "base-fit request"),
        ("train", "source_offline_split", "validation", "offline split"),
        ("validation", "level2_split", "train", "level-2 role"),
    ],
)
def test_scientific_oof_lineage_rejects_identity_or_role_mismatch(
    side: str, field: str, value: object, match: str
) -> None:
    train, validation = _scientific_oof_manifests()
    selected = train if side == "train" else validation
    selected[field] = value
    with pytest.raises(ValueError, match=match):
        _candidate_pool_split_lineage(train, validation)


def test_scientific_oof_lineage_rejects_mixed_pool_schemas() -> None:
    train, validation = _scientific_oof_manifests()
    validation["schema"] = POOL_SCHEMA
    with pytest.raises(ValueError, match="mixed candidate-pool schemas"):
        _candidate_pool_split_lineage(train, validation)
