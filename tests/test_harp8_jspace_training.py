from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from harp8.jspace_data import AlignedJCandidateData
from harp8.jspace_loss_profiles import PREREGISTERED_V1
from harp8.jspace_reranker import JSpaceCandidateReranker, JSpaceRerankerLossConfig
from harp8.train import sha256_file
from harp8.train_jspace_reranker import (
    JSPACE_TRAINING_MANIFEST_SCHEMA,
    JSPACE_TRAINING_SCHEMA,
    JSpaceTrainingConfig,
    _coverage_gate,
    _load_router_keys,
    _optimizer_groups,
    _read_pool_manifest,
    build_parser,
    candidate_feature_width,
    composite_horizon_scores,
    derive_model_config,
    load_composite_jspace_checkpoint,
    prepare_model_batch,
    slice_horizon_batch,
    train_jspace_reranker,
)


def _write_array(path: Path, dtype: str, values: np.ndarray) -> dict[str, object]:
    output = np.memmap(path, mode="w+", dtype=dtype, shape=values.shape)
    output[...] = values
    output.flush()
    del output
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_pool(
    root: Path,
    *,
    split: str,
    request_id: int,
    rows: int = 2,
    horizons: int = 8,
    layers: int = 2,
    experts: int = 12,
    candidates: int = 10,
) -> None:
    root.mkdir()
    shape = (rows, horizons, layers, candidates)
    ids = np.broadcast_to(np.arange(candidates, dtype=np.uint16), shape).copy()
    scores = np.broadcast_to(
        np.linspace(2.0, -2.0, candidates, dtype=np.float32), shape
    ).copy()
    membership = np.zeros(shape, dtype=np.uint8)
    membership[..., :8] = 1
    teacher = scores.copy()
    teacher[..., :8] += 4.0
    arrays = []
    arrays.append(_write_array(root / "candidate_scores.f32", "<f4", scores))
    arrays.append(_write_array(root / "candidate_ids.u2", "<u2", ids))
    arrays.append(_write_array(root / "target_membership.u1", "u1", membership))
    arrays.append(_write_array(root / "teacher_candidate_scores.f32", "<f4", teacher))
    arrays.append(
        _write_array(
            root / "valid_future.u1",
            "u1",
            np.ones((rows, horizons), dtype=np.uint8),
        )
    )
    arrays.append(_write_array(root / "current_scores.f32", "<f4", scores))
    arrays.append(
        _write_array(
            root / "current_rank.f32",
            "<f4",
            np.broadcast_to(
                np.arange(candidates, dtype=np.float32) / (experts - 1), shape
            ).copy(),
        )
    )
    arrays.append(
        _write_array(
            root / "source_gates.f16",
            "<f2",
            np.full((rows, horizons, layers, 3), 1 / 3, dtype=np.float16),
        )
    )
    arrays.append(
        _write_array(
            root / "copy_gates.f16",
            "<f2",
            np.full(shape, 0.5, dtype=np.float16),
        )
    )
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "request_ids": [request_id] * rows,
                "within": list(range(rows)),
                "domains": ["unit"] * rows,
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
                "horizons": horizons,
                "layers": layers,
                "experts": experts,
                "native_k": 8,
                "candidate_count": candidates,
                "model_width": 16,
                "store_context": False,
                "allow_test": split == "test",
                "arrays": arrays,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    capture = tmp_path / "capture"
    mtp = tmp_path / "mtp"
    capture.mkdir()
    mtp.mkdir()
    rows_per_request, layers, experts = 2, 2, 12
    request_ids = [101, 202]
    (capture / "requests.jsonl").write_text(
        "".join(json.dumps({"request_id": value}) + "\n" for value in request_ids),
        encoding="utf-8",
    )
    rows = rows_per_request * len(request_ids)
    generator = np.random.default_rng(7)
    logits = generator.normal(size=(rows, layers, experts)).astype(np.float32)
    np.save(capture / "raw_router_logits.npy", logits)
    np.save(
        capture / "top8_expert_ids.npy",
        np.argsort(-logits, axis=-1)[..., :8].astype(np.uint16),
    )
    np.save(
        mtp / "mtp_hidden_depths.npy",
        generator.normal(size=(rows, 3, 5)).astype(np.float16),
    )
    np.save(
        mtp / "mtp_router_logits_depths.npy",
        generator.normal(size=(rows, 3, experts)).astype(np.float32),
    )
    target = tmp_path / "j_features.npy"
    rms = tmp_path / "j_feature_rms.npy"
    np.save(
        target,
        generator.normal(size=(rows, layers, 6)).astype(np.float16),
    )
    np.save(rms, np.ones((rows, layers), dtype=np.float32))
    train_pool = tmp_path / "train_pool"
    validation_pool = tmp_path / "validation_pool"
    _write_pool(train_pool, split="train", request_id=101)
    _write_pool(validation_pool, split="validation", request_id=202)
    router = tmp_path / "router_keys.npy"
    np.save(
        router,
        generator.normal(size=(layers, experts, experts)).astype(np.float32),
    )
    return {
        "capture": capture,
        "mtp": mtp,
        "target": target,
        "rms": rms,
        "train_pool": train_pool,
        "validation_pool": validation_pool,
        "router": router,
    }


def _data(
    paths: dict[str, Path]
) -> tuple[AlignedJCandidateData, AlignedJCandidateData]:
    options = {
        "capture_dir": paths["capture"],
        "mtp_dir": paths["mtp"],
        "target_features": paths["target"],
        "target_feature_rms": paths["rms"],
        "rows_per_request": 2,
        "history": 3,
    }
    return (
        AlignedJCandidateData(paths["train_pool"], **options),
        AlignedJCandidateData(paths["validation_pool"], **options),
    )


def _tiny_model_config(data: AlignedJCandidateData, keys: torch.Tensor):
    return derive_model_config(
        data,
        keys,
        include_feature_rms=True,
        overrides={
            "model_width": 16,
            "attention_heads": 4,
            "feedforward_width": 32,
            "expert_embedding_width": 4,
            "temporal_blocks": 1,
            "axial_blocks": 1,
            "mtp_blocks": 1,
            "set_blocks": 1,
            "inducing_points": 2,
            "dropout": 0.0,
        },
    )


def test_data_derived_config_and_batch_include_rms_and_causal_features(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    train, _validation = _data(paths)
    keys, _provenance = _load_router_keys(paths["router"])
    config = _tiny_model_config(train, keys)
    assert config.j_width == train.target_width == 7
    assert config.candidate_feature_width == candidate_feature_width(3) == 13
    batch = prepare_model_batch(
        train.batch(np.asarray([0]), "cpu"),
        include_feature_rms=True,
        include_candidate_features=True,
        active_horizons=4,
    )
    assert batch["j_states"].shape == (1, 3, 2, 7)
    assert batch["j_mask"].shape == (1, 3, 2)
    assert batch["j_mask"][0, 0].all()
    assert not batch["j_mask"][0, 1:].any()
    assert batch["candidate_features"].shape == (1, 4, 2, 10, 13)
    assert batch["candidate_scores"].shape[1] == 4
    assert batch["valid_future"].shape[1] == 4
    assert batch["mtp_mask"].shape == (1, 3)


def test_model_config_requires_full_router_rank_unless_ablation_is_explicit(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    train, _validation = _data(paths)
    full_keys, _provenance = _load_router_keys(paths["router"])
    truncated_keys = full_keys[..., :-1].contiguous()

    with pytest.raises(ValueError, match="full router rank|ablation"):
        derive_model_config(
            train,
            truncated_keys,
            include_feature_rms=True,
        )

    config = derive_model_config(
        train,
        truncated_keys,
        include_feature_rms=True,
        allow_router_rank_ablation=True,
    )
    assert config.router_key_width == config.experts - 1
    assert not config.uses_full_router_rank


def test_epoch_zero_is_exact_and_optimizer_excludes_norm_bias_embeddings(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    train, _validation = _data(paths)
    keys, _provenance = _load_router_keys(paths["router"])
    config = _tiny_model_config(train, keys)
    model = JSpaceCandidateReranker(config, keys).eval()
    raw = train.batch(np.asarray([1]), "cpu")
    batch = prepare_model_batch(
        raw,
        include_feature_rms=True,
        include_candidate_features=True,
        active_horizons=config.horizons,
    )
    with torch.inference_mode():
        output = model(batch)
    assert torch.equal(output.scores, batch["candidate_scores"])
    groups = _optimizer_groups(model, 0.01)
    decay_ids = {id(value) for value in groups[0]["params"]}
    no_decay_ids = {id(value) for value in groups[1]["params"]}
    assert decay_ids.isdisjoint(no_decay_ids)
    for name, parameter in model.named_parameters():
        if (
            parameter.ndim == 1
            or name.lower().endswith("bias")
            or any(word in name.lower() for word in ("embedding", "norm"))
        ):
            assert id(parameter) in no_decay_ids


def test_composite_scores_leave_inactive_horizons_exactly_unchanged() -> None:
    base = torch.arange(1 * 8 * 2 * 10, dtype=torch.float32).reshape(1, 8, 2, 10)
    active = torch.full((1, 4, 2, 10), -17.0)
    composite = composite_horizon_scores(active, base)
    assert torch.equal(composite[:, :4], active)
    assert torch.equal(composite[:, 4:], base[:, 4:])
    assert torch.equal(
        base[:, 4:], torch.arange(80, 160, dtype=torch.float32).reshape(1, 4, 2, 10)
    )


def test_coverage_gate_and_test_pool_fail_closed(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    _train, validation = _data(paths)
    gate = _coverage_gate(validation)
    assert gate["passes"]
    assert gate["mean_h1_h4_coverage"] == pytest.approx(1.0)
    test_pool = tmp_path / "test_pool"
    _write_pool(test_pool, split="test", request_id=303)
    with pytest.raises(ValueError, match="forbidden|expected"):
        _read_pool_manifest(test_pool, "validation")


def test_tiny_training_checkpoint_resumes_and_emits_evaluation(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    train, validation = _data(paths)
    keys, provenance = _load_router_keys(paths["router"])
    config = _tiny_model_config(train, keys)
    training = JSpaceTrainingConfig(
        epochs=2,
        minimum_epochs=1,
        patience=2,
        microbatch_size=1,
        evaluation_batch_size=1,
        gradient_accumulation=2,
        bootstrap_replicates=40,
        max_train_rows=2,
        max_validation_rows=1,
        active_horizons=4,
    )
    assert build_parser().get_default("ordered_group_prefetch") is False
    assert build_parser().get_default("loss_profile") == PREREGISTERED_V1
    output = tmp_path / "training"
    paused = train_jspace_reranker(
        train,
        validation,
        output,
        keys,
        config,
        JSpaceRerankerLossConfig(
            horizon_weights=JSpaceRerankerLossConfig().horizon_weights[:4]
        ),
        training,
        target_features=paths["target"],
        target_feature_rms=paths["rms"],
        router_provenance=provenance,
        device="cpu",
        stop_after_epoch=1,
        strict_lineage=False,
        ordered_group_prefetch=True,
    )
    assert paused["status"] == "paused"
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["schema"] == JSPACE_TRAINING_SCHEMA
    assert checkpoint["completed_epoch"] == 1
    assert checkpoint["loss_profile"] == PREREGISTERED_V1
    assert checkpoint["input_provenance"]["execution_contract"]["loss_profile"] == (
        PREREGISTERED_V1
    )
    assert (
        checkpoint["input_provenance"]["execution_contract"]["resolved_loss_config"]
        == checkpoint["loss_config"]
    )
    manifest = train_jspace_reranker(
        train,
        validation,
        output,
        keys,
        config,
        JSpaceRerankerLossConfig(
            horizon_weights=JSpaceRerankerLossConfig().horizon_weights[:4]
        ),
        training,
        target_features=paths["target"],
        target_feature_rms=paths["rms"],
        router_provenance=provenance,
        device="cpu",
        resume=output / "last.pt",
        strict_lineage=False,
        ordered_group_prefetch=True,
    )
    assert manifest["schema"] == JSPACE_TRAINING_MANIFEST_SCHEMA
    assert manifest["loss_profile"] == PREREGISTERED_V1
    assert manifest["sealed_test_accessed"] is False
    assert manifest["epoch0_base_score_assertion"]["exact_score_equality"]
    assert manifest["composite_output_contract"] == {
        "pool_horizons": 8,
        "reranked_horizons": [1, 2, 3, 4],
        "frozen_harp_passthrough_horizons": [5, 6, 7, 8],
        "passthrough_is_exact_base_score": True,
    }
    assert (output / "best.pt").exists()
    assert (output / "last.pt").exists()
    assert (output / "training_history.csv").exists()
    assert (output / "validation_metrics.json").exists()
    assert (output / "validation_h1_h4_paired_bootstrap.json").exists()
    final = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert final["completed_epoch"] == 2
    assert len(final["history"]) == 2
    assert final["input_provenance"]["execution_contract"]["io_pipeline"] == {
        "schema": "harp8_ordered_group_prefetch_v1",
        "opt_in": True,
        "rows_per_materialization": 2,
        "cpu_read_ahead_groups": 1,
        "compact_model_only_transfer": True,
        "pinned_nonblocking_h2d_on_cuda": True,
    }

    uninterrupted_output = tmp_path / "training_uninterrupted"
    train_jspace_reranker(
        train,
        validation,
        uninterrupted_output,
        keys,
        config,
        JSpaceRerankerLossConfig(
            horizon_weights=JSpaceRerankerLossConfig().horizon_weights[:4]
        ),
        training,
        target_features=paths["target"],
        target_feature_rms=paths["rms"],
        router_provenance=provenance,
        device="cpu",
        strict_lineage=False,
        ordered_group_prefetch=True,
    )
    uninterrupted = torch.load(
        uninterrupted_output / "last.pt", map_location="cpu", weights_only=False
    )
    assert final["history"] == uninterrupted["history"]
    for name, value in final["model_state"].items():
        assert torch.equal(value, uninterrupted["model_state"][name]), name

    deployed = load_composite_jspace_checkpoint(output / "best.pt", device="cpu")
    inference_batch = validation.batch(np.asarray([0], dtype=np.int64), "cpu")
    with torch.inference_mode():
        deployed_output = deployed(inference_batch)
    assert deployed.pool_horizons == validation.pool.horizons == 8
    assert deployed.reranker.config.horizons == 4
    assert torch.equal(
        deployed_output["scores"][:, :4],
        deployed_output["active_scores"],
    )
    assert torch.equal(
        deployed_output["scores"][:, 4:],
        inference_batch["candidate_scores"][:, 4:].float(),
    )
    with pytest.raises(ValueError, match="pool horizons|checkpoint contract"):
        load_composite_jspace_checkpoint(
            output / "best.pt",
            device="cpu",
            pool_horizons=7,
        )
