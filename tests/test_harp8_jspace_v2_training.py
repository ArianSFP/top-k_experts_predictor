from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from harp8.jspace_loss_profiles import CUSTOM_LOSS_PROFILE, PREREGISTERED_V1
from harp8.jspace_reranker import JSpaceRerankerLossConfig
from harp8.jspace_v2_data import AlignedJContextCandidateData
from harp8.train import sha256_file
from harp8.train_jspace_reranker import JSpaceTrainingConfig
from harp8.train_jspace_v2_reranker import (
    JSPACE_V2_TRAINING_MANIFEST_SCHEMA,
    JSPACE_V2_TRAINING_SCHEMA,
    build_parser,
    derive_v2_model_config,
    load_composite_jspace_v2_checkpoint,
    prepare_v2_model_batch,
    train_jspace_v2_reranker,
)


def _write(root: Path, name: str, dtype: str, values: np.ndarray) -> dict[str, object]:
    path = root / name
    output = np.memmap(path, mode="w+", dtype=dtype, shape=values.shape)
    output[...] = values
    output.flush()
    del output
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _pool(root: Path, *, split: str, request_id: int) -> None:
    root.mkdir()
    rows, h, layers, candidates, experts, width = 2, 8, 2, 10, 12, 8
    shape = (rows, h, layers, candidates)
    base = np.broadcast_to(
        np.linspace(2, -2, candidates, dtype=np.float32), shape
    ).copy()
    ids = np.broadcast_to(np.arange(candidates, dtype=np.uint16), shape).copy()
    membership = np.zeros(shape, np.uint8)
    membership[..., :8] = 1
    arrays = [
        _write(root, "candidate_scores.f32", "<f4", base),
        _write(root, "candidate_ids.u2", "<u2", ids),
        _write(root, "target_membership.u1", "u1", membership),
        _write(root, "teacher_candidate_scores.f32", "<f4", base + membership * 3),
        _write(root, "valid_future.u1", "u1", np.ones((rows, h), np.uint8)),
        _write(root, "current_scores.f32", "<f4", base),
        _write(root, "current_rank.f32", "<f4", np.zeros(shape, np.float32)),
        _write(
            root,
            "source_gates.f16",
            "<f2",
            np.full((rows, h, layers, 3), 1 / 3, np.float16),
        ),
        _write(root, "copy_gates.f16", "<f2", np.full(shape, 0.5, np.float16)),
        _write(
            root,
            "generator_context.f16",
            "<f2",
            np.random.default_rng(request_id)
            .normal(size=(rows, h, layers, width))
            .astype(np.float16),
        ),
    ]
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "request_ids": [request_id] * rows,
                "within": [0, 1],
                "domains": ["unit", "unit"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "harp8_candidate_pool_v1",
                "split": split,
                "rows": rows,
                "horizons": h,
                "layers": layers,
                "experts": experts,
                "native_k": 8,
                "candidate_count": candidates,
                "model_width": width,
                "store_context": True,
                "allow_test": False,
                "checkpoint": {"sha256": "same-generator"},
                "split_manifest": None,
                "arrays": arrays,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _artifacts(
    tmp_path: Path,
) -> tuple[
    AlignedJContextCandidateData, AlignedJContextCandidateData, Path, torch.Tensor
]:
    capture = tmp_path / "capture"
    mtp = tmp_path / "mtp"
    capture.mkdir()
    mtp.mkdir()
    requests = [101, 202]
    (capture / "requests.jsonl").write_text(
        "".join(json.dumps({"request_id": value}) + "\n" for value in requests),
        encoding="utf-8",
    )
    rng = np.random.default_rng(7)
    rows, layers, experts = 4, 2, 12
    logits = rng.normal(size=(rows, layers, experts)).astype(np.float32)
    np.save(capture / "raw_router_logits.npy", logits)
    np.save(
        capture / "top8_expert_ids.npy",
        np.argsort(-logits, axis=-1)[..., :8].astype(np.uint16),
    )
    np.save(
        mtp / "mtp_hidden_depths.npy",
        rng.normal(size=(rows, 3, 5)).astype(np.float16),
    )
    np.save(
        mtp / "mtp_router_logits_depths.npy",
        rng.normal(size=(rows, 3, experts)).astype(np.float32),
    )
    features = tmp_path / "features.npy"
    np.save(features, rng.normal(size=(rows, layers, 6)).astype(np.float16))
    train_pool = tmp_path / "train_pool"
    validation_pool = tmp_path / "validation_pool"
    _pool(train_pool, split="train", request_id=101)
    _pool(validation_pool, split="validation", request_id=202)
    options = {
        "capture_dir": capture,
        "mtp_dir": mtp,
        "target_features": features,
        "rows_per_request": 2,
        "history": 3,
    }
    train = AlignedJContextCandidateData(train_pool, **options)
    validation = AlignedJContextCandidateData(validation_pool, **options)
    keys = torch.from_numpy(
        rng.normal(size=(layers, experts, experts)).astype(np.float32)
    )
    return train, validation, features, keys


def test_v2_cli_and_data_derived_config_are_explicit(tmp_path: Path) -> None:
    train, _validation, _features, keys = _artifacts(tmp_path)
    parser = build_parser()
    assert parser.get_default("model_width") == 384
    assert parser.get_default("router_query_rank") == 64
    assert parser.get_default("loss_profile") == PREREGISTERED_V1
    config = derive_v2_model_config(
        train,
        keys,
        overrides={
            "model_width": 16,
            "attention_heads": 4,
            "feedforward_width": 32,
            "expert_embedding_width": 4,
            "router_query_rank": 4,
            "temporal_blocks": 1,
            "axial_blocks": 1,
            "mtp_hidden_blocks": 1,
            "mtp_router_blocks": 1,
            "set_blocks": 1,
            "inducing_points": 2,
            "dropout": 0.0,
        },
    )
    assert config.horizons == 4
    assert config.pool_horizons == 8
    assert config.generator_context_width == 8
    assert config.router_query_rank == 4
    batch = prepare_v2_model_batch(
        train.batch(np.asarray([0]), "cpu", active_horizons=4, compact=True),
        include_candidate_features=True,
    )
    assert batch["generator_context"].shape == (1, 4, 2, 8)


def test_tiny_v2_training_checkpoint_and_composite_contract(tmp_path: Path) -> None:
    train, validation, features, keys = _artifacts(tmp_path)
    config = derive_v2_model_config(
        train,
        keys,
        overrides={
            "model_width": 16,
            "attention_heads": 4,
            "feedforward_width": 32,
            "expert_embedding_width": 4,
            "router_query_rank": 4,
            "temporal_blocks": 1,
            "axial_blocks": 1,
            "mtp_hidden_blocks": 1,
            "mtp_router_blocks": 1,
            "set_blocks": 1,
            "inducing_points": 2,
            "dropout": 0.0,
        },
    )
    training = JSpaceTrainingConfig(
        epochs=1,
        minimum_epochs=1,
        patience=1,
        microbatch_size=1,
        evaluation_batch_size=1,
        gradient_accumulation=2,
        bootstrap_replicates=20,
        max_train_rows=2,
        max_validation_rows=2,
        active_horizons=4,
    )
    output = tmp_path / "output"
    manifest = train_jspace_v2_reranker(
        train,
        validation,
        output,
        keys,
        config,
        JSpaceRerankerLossConfig(
            horizon_weights=(1.0, 1.0, 1.25, 1.5),
            hard_negative_count=2,
        ),
        training,
        target_features=features,
        target_feature_rms=None,
        router_provenance={"fixture": True},
        device="cpu",
        strict_lineage=False,
        ordered_group_prefetch=True,
    )
    assert manifest["schema"] == JSPACE_V2_TRAINING_MANIFEST_SCHEMA
    assert manifest["loss_profile"] == CUSTOM_LOSS_PROFILE
    assert manifest["epoch0_base_score_assertion"]["exact_score_equality"]
    train_arrays = manifest["input_provenance"]["pools"]["train"]["arrays"]
    context_record = next(
        record
        for record in train_arrays
        if Path(record["path"]).name == "generator_context.f16"
    )
    assert context_record["sha256"] == sha256_file(
        train.pool.root / "generator_context.f16"
    )
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["schema"] == JSPACE_V2_TRAINING_SCHEMA
    assert checkpoint["loss_profile"] == CUSTOM_LOSS_PROFILE
    assert checkpoint["input_provenance"]["execution_contract"]["loss_profile"] == (
        CUSTOM_LOSS_PROFILE
    )
    assert (
        checkpoint["input_provenance"]["execution_contract"]["resolved_loss_config"]
        == checkpoint["loss_config"]
    )
    assert checkpoint["inference_contract"]["generator_context_required"]
    assert checkpoint["inference_contract"]["layer_aware_mtp_queries"]

    deployed = load_composite_jspace_v2_checkpoint(output / "best.pt")
    batch = validation.batch(np.asarray([0]), "cpu")
    with torch.inference_mode():
        result = deployed(batch)
    assert torch.equal(result["scores"][:, 4:], batch["candidate_scores"][:, 4:])
    assert manifest["composite_output_contract"] == {
        "pool_horizons": 8,
        "reranked_horizons": [1, 2, 3, 4],
        "frozen_harp_passthrough_horizons": [5, 6, 7, 8],
        "passthrough_is_exact_base_score": True,
    }
