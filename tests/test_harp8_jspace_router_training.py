from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import pytest

import harp8.train_jspace_router_forecaster as router_training_module

from harp8.candidates import POOL_SCHEMA_V2, REQUEST_IDS_HASH_ENCODING, request_ids_sha256
from harp8.jspace_router_data import FullRouterForecastData
from harp8.jspace_router_forecaster import (
    FullRouterLossConfig,
    JSpaceFullRouterForecaster,
)
from harp8.train import sha256_file
from harp8.train_jspace_router_forecaster import (
    FullRouterTrainingConfig,
    _aggregate_deferred_training_metrics,
    JSPACE_ROUTER_TRAINING_MANIFEST_SCHEMA,
    JSPACE_ROUTER_TRAINING_SCHEMA,
    build_parser,
    derive_router_forecaster_config,
    load_jspace_router_forecaster_checkpoint,
    train_jspace_router_forecaster,
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
    rows, horizons, layers, candidates, experts, width = 2, 8, 2, 10, 12, 8
    shape = (rows, horizons, layers, candidates)
    scores = np.broadcast_to(
        np.linspace(2.0, -2.0, candidates, dtype=np.float32), shape
    ).copy()
    ids = np.broadcast_to(np.arange(candidates, dtype=np.uint16), shape).copy()
    arrays = [
        _write(root, "candidate_scores.f32", "<f4", scores),
        _write(root, "candidate_ids.u2", "<u2", ids),
        _write(root, "target_membership.u1", "u1", np.zeros(shape, np.uint8)),
        _write(root, "teacher_candidate_scores.f32", "<f4", np.zeros(shape, np.float32)),
        _write(root, "valid_future.u1", "u1", np.ones((rows, horizons), np.uint8)),
        _write(root, "current_scores.f32", "<f4", np.zeros(shape, np.float32)),
        _write(root, "current_rank.f32", "<f4", np.zeros(shape, np.float32)),
        _write(root, "source_gates.f16", "<f2", np.zeros((rows, horizons, layers, 3), np.float16)),
        _write(root, "copy_gates.f16", "<f2", np.zeros(shape, np.float16)),
        _write(
            root,
            "generator_context.f16",
            "<f2",
            np.random.default_rng(request_id).normal(
                size=(rows, horizons, layers, width)
            ).astype(np.float16),
        ),
    ]
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "request_ids": [request_id, request_id],
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
                "schema": POOL_SCHEMA_V2,
                "split": split,
                "split_field_semantics": "deprecated_alias_of_level2_split",
                "level2_split": split,
                "source_data_split": split,
                "source_offline_split": split,
                "base_fold_id": "synthetic-fold-0",
                "base_fit_excluded": True,
                "base_fit_overlap_count": 0,
                "base_train_request_ids": [999],
                "base_train_request_ids_sha256": request_ids_sha256([999]),
                "exported_request_ids_sha256": request_ids_sha256(
                    [request_id, request_id]
                ),
                "request_ids_hash_encoding": REQUEST_IDS_HASH_ENCODING,
                "allow_test": False,
                "rows": rows,
                "horizons": horizons,
                "layers": layers,
                "experts": experts,
                "native_k": 8,
                "candidate_count": candidates,
                "model_width": width,
                "store_context": True,
                "checkpoint": {"sha256": "same"},
                "split_manifest": None,
                "arrays": arrays,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _artifacts(tmp_path: Path, *, dual_stream: bool = False):
    capture = tmp_path / "capture"
    mtp = tmp_path / "mtp"
    capture.mkdir()
    mtp.mkdir()
    requests = [101, 202]
    (capture / "requests.jsonl").write_text(
        "".join(json.dumps({"request_id": value}) + "\n" for value in requests),
        encoding="utf-8",
    )
    rng = np.random.default_rng(91)
    rows, layers, experts = 20, 2, 12
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
    secondary = tmp_path / "secondary.npy"
    secondary_rms = tmp_path / "secondary_rms.npy"
    if dual_stream:
        np.save(
            secondary,
            rng.normal(size=(rows, layers, 7)).astype(np.float16),
        )
        np.save(
            secondary_rms,
            rng.normal(size=(rows, layers)).astype(np.float16),
        )
    train_pool = tmp_path / "train_pool"
    validation_pool = tmp_path / "validation_pool"
    _pool(train_pool, split="train", request_id=101)
    _pool(validation_pool, split="validation", request_id=202)
    common = {
        "capture_dir": capture,
        "mtp_dir": mtp,
        "target_features": features,
        "secondary_target_features": secondary if dual_stream else None,
        "secondary_target_feature_rms": secondary_rms if dual_stream else None,
        "rows_per_request": 10,
        "history": 3,
    }
    train = FullRouterForecastData(
        train_pool, expected_split="train", **common
    )
    validation = FullRouterForecastData(
        validation_pool, expected_split="validation", **common
    )
    return train, validation, features, (
        secondary if dual_stream else None
    ), (secondary_rms if dual_stream else None)


def test_full_router_cli_is_validation_only_and_24gib_sized() -> None:
    parser = build_parser()
    assert parser.get_default("model_width") == 192
    assert parser.get_default("microbatch_size") == 4
    assert parser.get_default("output_rank") == 64
    assert parser.get_default("top8_swap_weight") == 0.0
    assert parser.get_default("base_kl_weight") == 0.0
    assert parser.get_default("delta_l2_weight") == 0.0
    assert parser.get_default("relative_regret_weight") == 0.0
    assert parser.get_default("freeze_output_bias") is False
    assert any(action.dest == "secondary_target_features" for action in parser._actions)
    assert any(
        action.dest == "secondary_target_feature_rms" for action in parser._actions
    )
    assert not any(action.dest == "test_pool" for action in parser._actions)

    parsed = parser.parse_args(
        [
            "--train-pool", "train",
            "--validation-pool", "validation",
            "--capture-dir", "capture",
            "--mtp-dir", "mtp",
            "--target-features", "features.npy",
            "--output", "output",
            "--base-kl-weight", "0.7",
            "--delta-l2-weight", "0.2",
            "--relative-regret-weight", "0.4",
            "--freeze-output-bias",
        ]
    )
    assert parsed.base_kl_weight == 0.7
    assert parsed.delta_l2_weight == 0.2
    assert parsed.relative_regret_weight == 0.4
    assert parsed.freeze_output_bias is True


def test_training_config_omits_disabled_freeze_for_legacy_resume() -> None:
    assert "freeze_output_bias" not in FullRouterTrainingConfig().to_dict()
    assert FullRouterTrainingConfig(freeze_output_bias=True).to_dict()[
        "freeze_output_bias"
    ] is True


def test_tiny_training_checkpoint_roundtrip_and_validation_contract(tmp_path: Path) -> None:
    train, validation, features, _, _ = _artifacts(tmp_path)
    model_config = derive_router_forecaster_config(
        train,
        overrides={
            "model_width": 16,
            "attention_heads": 4,
            "feedforward_width": 32,
            "layer_blocks": 1,
            "mtp_blocks": 1,
            "fusion_blocks": 1,
            "output_rank": 4,
            "dropout": 0.0,
        },
    )
    training_config = FullRouterTrainingConfig(
        epochs=1,
        minimum_epochs=1,
        patience=1,
        microbatch_size=1,
        evaluation_batch_size=1,
        gradient_accumulation=2,
        max_train_rows=2,
        max_validation_rows=2,
    )
    loss_config = FullRouterLossConfig(
        hard_negative_end_rank=12,
        predicted_negative_count=4,
    )
    output = tmp_path / "output"
    manifest = train_jspace_router_forecaster(
        train,
        validation,
        output,
        model_config,
        loss_config,
        training_config,
        target_features=features,
        target_feature_rms=None,
        device="cpu",
        strict_lineage=False,
    )
    assert manifest["schema"] == JSPACE_ROUTER_TRAINING_MANIFEST_SCHEMA
    assert manifest["sealed_test_accessed"] is False
    assert manifest["epoch0_baseline_assertion"] == {
        "exact_score_equality": True,
        "zero_residual": True,
    }
    payload = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert payload["schema"] == JSPACE_ROUTER_TRAINING_SCHEMA
    assert payload["completed_epoch"] == 1
    assert "optimizer_state" in payload and "rng_state" in payload
    assert payload["inference_contract"]["future_labels_cross_model_boundary"] is False

    contract = payload["baseline_expansion_contract"]
    assert contract["base_floor_margin"] == 1.0
    assert contract["candidate_count"] == 10
    assert contract["native_k"] == 8
    assert (
        payload["inference_contract"]["baseline_expansion_contract"]
        == contract
    )
    deployed = load_jspace_router_forecaster_checkpoint(output / "best.pt")
    batch = validation.batch(np.asarray([0]), "cpu")
    with torch.inference_mode():
        first = deployed(batch).future_router_scores
    reloaded = load_jspace_router_forecaster_checkpoint(output / "best.pt")
    with torch.inference_mode():
        second = reloaded(batch).future_router_scores
    assert torch.equal(first, second)
    assert first.shape == (1, 8, 2, 12)
    metrics = json.loads((output / "validation_metrics.json").read_text())
    assert len(metrics["horizon_metrics"]) == 8
    assert metrics["sealed_test_accessed"] is False
    assert manifest["validation_metrics_source"] == "persisted_best_epoch_evaluation"


def test_epoch_zero_remains_best_under_deliberate_validation_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, validation, features, _, _ = _artifacts(tmp_path)
    model_config = derive_router_forecaster_config(
        train,
        overrides={
            "model_width": 16,
            "attention_heads": 4,
            "feedforward_width": 32,
            "layer_blocks": 1,
            "mtp_blocks": 1,
            "fusion_blocks": 1,
            "output_rank": 4,
            "dropout": 0.0,
        },
    )
    training_config = FullRouterTrainingConfig(
        epochs=1,
        minimum_epochs=1,
        patience=1,
        microbatch_size=1,
        evaluation_batch_size=1,
        gradient_accumulation=2,
        max_train_rows=2,
        max_validation_rows=2,
    )
    loss_config = FullRouterLossConfig(
        hard_negative_end_rank=12,
        predicted_negative_count=4,
    )
    scores = iter((0.9, 0.1))
    observed_scores: list[float] = []

    def fake_evaluation(*_args: object, **_kwargs: object) -> dict[str, object]:
        score = next(scores)
        observed_scores.append(score)
        return {
            "mean_h1_h4_recall_at_8": score,
            "h2_request_macro_recall_at_8": score,
            "mean_h1_h8_recall_at_8": score,
            "horizon_metrics": [],
            "request_metrics": [],
            "layer_metrics": [],
            "domain_metrics": [],
            "sealed_test_accessed": False,
        }

    monkeypatch.setattr(
        router_training_module,
        "evaluate_full_router_forecaster",
        fake_evaluation,
    )
    output = tmp_path / "epoch0_best"
    manifest = train_jspace_router_forecaster(
        train,
        validation,
        output,
        model_config,
        loss_config,
        training_config,
        target_features=features,
        target_feature_rms=None,
        device="cpu",
        strict_lineage=False,
    )
    assert manifest["best_epoch"] == 0
    assert observed_scores == [0.9, 0.1]
    assert manifest["validation_metrics"]["mean_h1_h4_recall_at_8"] == 0.9
    assert manifest["validation_metrics_source"] == "persisted_best_epoch_evaluation"
    persisted = json.loads((output / "validation_metrics.json").read_text())
    assert persisted["mean_h1_h4_recall_at_8"] == 0.9
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert best["completed_epoch"] == 0
    assert last["completed_epoch"] == 1
    model = load_jspace_router_forecaster_checkpoint(output / "best.pt")
    with torch.inference_mode():
        result = model(validation.batch(np.asarray([0]), "cpu"))
    assert torch.count_nonzero(result.delta) == 0


def test_resume_rejects_changed_baseline_floor_margin(tmp_path: Path) -> None:
    train, validation, features, _, _ = _artifacts(tmp_path)
    model_config = derive_router_forecaster_config(
        train,
        overrides={
            "model_width": 16,
            "attention_heads": 4,
            "feedforward_width": 32,
            "layer_blocks": 1,
            "mtp_blocks": 1,
            "fusion_blocks": 1,
            "output_rank": 4,
            "dropout": 0.0,
        },
    )
    training_config = FullRouterTrainingConfig(
        epochs=1,
        minimum_epochs=1,
        patience=1,
        microbatch_size=1,
        evaluation_batch_size=1,
        gradient_accumulation=2,
        max_train_rows=2,
        max_validation_rows=2,
    )
    loss_config = FullRouterLossConfig(
        hard_negative_end_rank=12,
        predicted_negative_count=4,
    )
    output = tmp_path / "resume_margin"
    train_jspace_router_forecaster(
        train,
        validation,
        output,
        model_config,
        loss_config,
        training_config,
        target_features=features,
        target_feature_rms=None,
        device="cpu",
        strict_lineage=False,
    )
    train.base_floor_margin = 2.0
    validation.base_floor_margin = 2.0
    changed = replace(training_config, base_floor_margin=2.0)
    with pytest.raises(ValueError, match="training_config|baseline_expansion"):
        train_jspace_router_forecaster(
            train,
            validation,
            output,
            model_config,
            loss_config,
            changed,
            target_features=features,
            target_feature_rms=None,
            device="cpu",
            resume=output / "last.pt",
            strict_lineage=False,
        )


def test_dual_stream_checkpoint_roundtrip_provenance_and_resume_contract(
    tmp_path: Path,
) -> None:
    train, validation, features, secondary, secondary_rms = _artifacts(
        tmp_path, dual_stream=True
    )
    assert secondary is not None and secondary_rms is not None
    model_config = derive_router_forecaster_config(
        train,
        overrides={
            "model_width": 16,
            "attention_heads": 4,
            "feedforward_width": 32,
            "layer_blocks": 1,
            "mtp_blocks": 1,
            "fusion_blocks": 1,
            "output_rank": 4,
            "dropout": 0.0,
        },
    )
    assert model_config.secondary_j_width == 8
    training_config = FullRouterTrainingConfig(
        epochs=1,
        minimum_epochs=1,
        patience=1,
        microbatch_size=1,
        evaluation_batch_size=1,
        gradient_accumulation=2,
        max_train_rows=2,
        max_validation_rows=2,
    )
    loss_config = FullRouterLossConfig(
        hard_negative_end_rank=12,
        predicted_negative_count=4,
    )
    output = tmp_path / "dual_output"
    manifest = train_jspace_router_forecaster(
        train,
        validation,
        output,
        model_config,
        loss_config,
        training_config,
        target_features=features,
        target_feature_rms=None,
        secondary_target_features=secondary,
        secondary_target_feature_rms=secondary_rms,
        device="cpu",
        strict_lineage=False,
    )
    contract = manifest["target_state_stream_contract"]
    assert contract["schema"] == "harp8_full_router_target_state_streams_v1"
    assert contract["enabled"] is True
    assert [stream["slot"] for stream in contract["streams"]] == [
        "primary",
        "secondary",
    ]
    assert all(stream["features"]["sha256"] for stream in contract["streams"])
    assert contract["streams"][1]["rms"]["sha256"]
    assert contract["fusion"]["input_concatenation_before_projection"] is False

    payload = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert payload["target_state_stream_contract"] == contract
    assert payload["inference_contract"]["target_state_stream_contract"] == contract
    reloaded = load_jspace_router_forecaster_checkpoint(output / "best.pt")
    batch = validation.batch(np.asarray([0]), "cpu")
    with torch.inference_mode():
        first = reloaded(batch).future_router_scores
        second = load_jspace_router_forecaster_checkpoint(output / "best.pt")(
            batch
        ).future_router_scores
    assert torch.equal(first, second)

    resumed = train_jspace_router_forecaster(
        train,
        validation,
        output,
        model_config,
        loss_config,
        training_config,
        target_features=features,
        target_feature_rms=None,
        secondary_target_features=secondary,
        secondary_target_feature_rms=secondary_rms,
        device="cpu",
        resume=output / "last.pt",
        strict_lineage=False,
    )
    assert resumed["completed_epoch"] == 1
    assert resumed["target_state_stream_contract"] == contract

    alternate = tmp_path / "alternate_secondary.npy"
    np.save(alternate, np.asarray(train.secondary_target_features))
    train.secondary_target_features_path = alternate
    validation.secondary_target_features_path = alternate
    with pytest.raises(ValueError, match="input_provenance|target_state_stream_contract"):
        train_jspace_router_forecaster(
            train,
            validation,
            output,
            model_config,
            loss_config,
            training_config,
            target_features=features,
            target_feature_rms=None,
            secondary_target_features=alternate,
            secondary_target_feature_rms=secondary_rms,
            device="cpu",
            resume=output / "last.pt",
            strict_lineage=False,
        )


def test_old_single_stream_checkpoint_config_and_state_layout_still_load(
    tmp_path: Path,
) -> None:
    train, _, _, _, _ = _artifacts(tmp_path)
    config = derive_router_forecaster_config(
        train,
        overrides={
            "model_width": 16,
            "attention_heads": 4,
            "feedforward_width": 32,
            "layer_blocks": 1,
            "mtp_blocks": 1,
            "fusion_blocks": 1,
            "output_rank": 4,
            "dropout": 0.0,
        },
    )
    # Disabled optional geometry is omitted, exactly matching pre-dual configs.
    assert "secondary_j_width" not in config.to_dict()
    for name in (
        "mtp_diagonal_from_local",
        "mtp_null_only_when_all_missing",
        "mtp_horizon_depth_attention_bias",
    ):
        assert name not in config.to_dict()
    original = JSpaceFullRouterForecaster(config).eval()
    assert not any(
        "horizon_depth_attention_bias" in name
        for name in original.state_dict()
    )
    baseline_contract = {
        "schema": "harp8_full_router_baseline_expansion_v1",
    }
    legacy = tmp_path / "legacy_single.pt"
    torch.save(
        {
            "schema": JSPACE_ROUTER_TRAINING_SCHEMA,
            "model_config": config.to_dict(),
            "model_state": original.state_dict(),
            "baseline_expansion_contract": baseline_contract,
            "inference_contract": {
                "baseline_expansion_contract": baseline_contract,
            },
        },
        legacy,
    )
    loaded = load_jspace_router_forecaster_checkpoint(legacy)
    assert loaded.config.secondary_j_width is None
    assert loaded.config.mtp_diagonal_from_local is False
    assert loaded.config.mtp_null_only_when_all_missing is False
    assert loaded.config.mtp_horizon_depth_attention_bias is False
    assert loaded.history_encoder.secondary_target_projection is None
    assert loaded.history_encoder.target_stream_gate is None
    assert set(loaded.state_dict()) == set(original.state_dict())


def test_deferred_metric_aggregation_exactly_matches_legacy_logging_order() -> None:
    names = ("router_kl", "top8_swap", "loss")
    rows = [
        torch.tensor([0.1, 0.3, 0.7], dtype=torch.float32),
        torch.tensor([0.2, -0.4, 0.9], dtype=torch.float32),
        torch.tensor([1.0, 0.5, -0.2], dtype=torch.float32),
    ]
    legacy = {name: 0.0 for name in names}
    for row in rows:
        values = row.detach().cpu().tolist()
        for name, value in zip(names, values, strict=True):
            legacy[name] += float(value)
    actual = _aggregate_deferred_training_metrics(names, rows)
    assert actual == legacy
    assert all(not row.requires_grad for row in rows)


def test_output_bias_freezes_before_optimizer_and_is_resume_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, validation, features, _, _ = _artifacts(tmp_path)
    model_config = derive_router_forecaster_config(
        train,
        overrides={
            "model_width": 16,
            "attention_heads": 4,
            "feedforward_width": 32,
            "layer_blocks": 1,
            "mtp_blocks": 1,
            "fusion_blocks": 1,
            "output_rank": 4,
            "dropout": 0.0,
        },
    )
    training_config = FullRouterTrainingConfig(
        epochs=1,
        minimum_epochs=1,
        patience=1,
        microbatch_size=1,
        evaluation_batch_size=1,
        gradient_accumulation=2,
        max_train_rows=2,
        max_validation_rows=2,
        freeze_output_bias=True,
    )
    loss_config = FullRouterLossConfig(
        router_kl=0.0,
        boundary=0.0,
        predicted_boundary=0.0,
        top8_swap=1.0,
        full_membership=0.0,
        centered_score=0.0,
        base_kl=0.1,
        delta_l2=0.2,
        relative_regret=0.3,
        hard_negative_end_rank=12,
        predicted_negative_count=4,
    )
    original_groups = router_training_module._optimizer_groups
    observed: dict[str, object] = {}

    def checking_groups(
        model: torch.nn.Module, weight_decay: float
    ) -> list[dict[str, object]]:
        bias = model.output_head.bias
        observed["requires_grad"] = bias.requires_grad
        groups = original_groups(model, weight_decay)
        optimizer_ids = {
            id(parameter)
            for group in groups
            for parameter in group["params"]
        }
        observed["in_optimizer"] = id(bias) in optimizer_ids
        return groups

    monkeypatch.setattr(router_training_module, "_optimizer_groups", checking_groups)
    output = tmp_path / "frozen_bias"
    manifest = train_jspace_router_forecaster(
        train,
        validation,
        output,
        model_config,
        loss_config,
        training_config,
        target_features=features,
        target_feature_rms=None,
        device="cpu",
        strict_lineage=False,
    )
    assert observed == {"requires_grad": False, "in_optimizer": False}

    contract = manifest["output_head_training_contract"]
    expected_frozen = (
        model_config.horizons * model_config.layers * model_config.experts
    )
    assert contract["frozen_parameter_count"] == expected_frozen
    assert contract["frozen_before_optimizer_creation"] is True
    assert contract["state_dict_parameter_retained"] is True
    assert manifest["training_config"]["freeze_output_bias"] is True
    assert (
        manifest["input_provenance"]["execution_contract"][
            "output_head_training_contract"
        ]
        == contract
    )

    payload = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert payload["output_head_training_contract"] == contract
    assert "output_head.bias" in payload["model_state"]
    assert torch.count_nonzero(payload["model_state"]["output_head.bias"]) == 0
    unfrozen = JSpaceFullRouterForecaster(model_config)
    assert set(unfrozen.state_dict()) == set(payload["model_state"])
    expected_trainable = (
        sum(parameter.numel() for parameter in unfrozen.parameters())
        - expected_frozen
    )
    assert manifest["trainable_parameters"] == expected_trainable

    resumed = train_jspace_router_forecaster(
        train,
        validation,
        output,
        model_config,
        loss_config,
        training_config,
        target_features=features,
        target_feature_rms=None,
        device="cpu",
        resume=output / "last.pt",
        strict_lineage=False,
    )
    assert resumed["output_head_training_contract"] == contract

    tampered = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    tampered["output_head_training_contract"] = {
        **contract,
        "trainable": True,
    }
    tampered_path = tmp_path / "tampered_freeze.pt"
    torch.save(tampered, tampered_path)
    with pytest.raises(ValueError, match="output_head_training_contract"):
        train_jspace_router_forecaster(
            train,
            validation,
            output,
            model_config,
            loss_config,
            training_config,
            target_features=features,
            target_feature_rms=None,
            device="cpu",
            resume=tampered_path,
            strict_lineage=False,
        )
