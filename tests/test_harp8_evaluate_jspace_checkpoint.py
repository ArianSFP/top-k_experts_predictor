from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from harp8.evaluate_jspace_checkpoint import (
    DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_REPLICATES,
    DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_SEED,
    JSPACE_CANDIDATE_RESIDUAL_SCALE_DIAGNOSTIC_SCHEMA,
    JSPACE_CHECKPOINT_EVALUATION_SCHEMA,
    _paired_scale_bootstrap,
    build_parser,
    evaluate_saved_jspace_checkpoint,
    parse_residual_scale_grid,
)
from harp8.jspace_reranker import JSpaceRerankerLossConfig
from harp8.jspace_v2_reranker import CompositeJSpaceV2Inference
from harp8.train_jspace_reranker import (
    JSpaceTrainingConfig,
    _load_router_keys,
    train_jspace_reranker,
)
from harp8.train_jspace_v2_reranker import (
    derive_v2_model_config,
    train_jspace_v2_reranker,
)


def _fixture_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


V1_FIXTURES = _fixture_module(
    "test_harp8_jspace_training.py", "_jspace_v1_evaluator_fixtures"
)
V2_FIXTURES = _fixture_module(
    "test_harp8_jspace_v2_training.py", "_jspace_v2_evaluator_fixtures"
)


def _v1_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    tmp_path.mkdir()
    paths = V1_FIXTURES._artifacts(tmp_path)
    train, validation = V1_FIXTURES._data(paths)
    keys, provenance = _load_router_keys(paths["router"])
    config = V1_FIXTURES._tiny_model_config(train, keys)
    training = JSpaceTrainingConfig(
        epochs=1,
        minimum_epochs=1,
        patience=1,
        microbatch_size=1,
        evaluation_batch_size=1,
        gradient_accumulation=2,
        bootstrap_replicates=10,
        max_train_rows=2,
        max_validation_rows=2,
        active_horizons=4,
    )
    output = tmp_path / "training"
    train_jspace_reranker(
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
        target_features=paths["target"],
        target_feature_rms=paths["rms"],
        router_provenance=provenance,
        device="cpu",
        strict_lineage=False,
    )
    return output / "last.pt", paths


def _v2_checkpoint(tmp_path: Path):
    tmp_path.mkdir()
    train, validation, features, keys = V2_FIXTURES._artifacts(tmp_path)
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
        bootstrap_replicates=10,
        max_train_rows=2,
        max_validation_rows=2,
        active_horizons=4,
    )
    output = tmp_path / "training"
    train_jspace_v2_reranker(
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
    )
    return output / "last.pt", train, validation, features


def test_v1_exact_checkpoint_evaluation_and_paired_bootstraps(
    tmp_path: Path,
) -> None:
    checkpoint, paths = _v1_checkpoint(tmp_path / "v1")
    output = tmp_path / "v1_evaluation"
    manifest = evaluate_saved_jspace_checkpoint(
        checkpoint,
        output,
        comparison_checkpoint=checkpoint,
        paired_against_base=True,
        device="cpu",
        bootstrap_replicates=20,
        strict_lineage=False,
    )
    assert manifest["schema"] == JSPACE_CHECKPOINT_EVALUATION_SCHEMA
    assert manifest["completed_epoch"] == 1
    assert manifest["validation_rows"] == 2
    assert manifest["sealed_test_accessed"] is False
    assert manifest["optimizer_constructed"] is False
    assert manifest["training_resume_invoked"] is False
    assert manifest["paired_against_checkpoint"]["paired_gain"] == 0.0
    assert manifest["paired_against_checkpoint"]["ci95_lower"] == 0.0
    assert manifest["paired_against_checkpoint"]["ci95_upper"] == 0.0
    assert manifest["paired_against_base"]["requests"] == 1
    assert "residual_scale_diagnostic" not in manifest
    metrics = json.loads(
        (output / "validation_metrics.json").read_text(encoding="utf-8")
    )
    assert len(metrics["horizon_metrics"]) == 8
    assert (output / "validation_request_metrics.csv").exists()
    assert (output / "validation_layer_metrics.csv").exists()

    with pytest.raises(ValueError, match="strict scientific lineage"):
        evaluate_saved_jspace_checkpoint(
            checkpoint,
            tmp_path / "strict_rejected",
            device="cpu",
            bootstrap_replicates=5,
        )

    test_pool = tmp_path / "test_pool"
    V1_FIXTURES._write_pool(
        test_pool,
        split="test",
        request_id=303,
    )
    with pytest.raises(ValueError, match="forbidden|expected"):
        evaluate_saved_jspace_checkpoint(
            checkpoint,
            tmp_path / "test_rejected",
            validation_pool=test_pool,
            train_pool=paths["train_pool"],
            capture_dir=paths["capture"],
            mtp_dir=paths["mtp"],
            target_features=paths["target"],
            target_feature_rms=paths["rms"],
            device="cpu",
            bootstrap_replicates=5,
            strict_lineage=False,
        )


def test_v2_exact_checkpoint_uses_context_and_infers_paths(tmp_path: Path) -> None:
    checkpoint, _train, _validation, _features = _v2_checkpoint(tmp_path / "v2")
    output = tmp_path / "v2_evaluation"
    manifest = evaluate_saved_jspace_checkpoint(
        checkpoint,
        output,
        paired_against_base=True,
        device="cpu",
        bootstrap_replicates=20,
        strict_lineage=False,
    )
    assert manifest["checkpoint_schema"] == "harp8_jspace_v2_reranker_training_v1"
    assert manifest["completed_epoch"] == 1
    metrics = json.loads(
        (output / "validation_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["candidate_count"] == 10
    assert metrics["native_k"] == 8
    assert len(metrics["layer_metrics"]) == 16


def test_cli_is_validation_only_and_has_no_lineage_bypass() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "validation_pool" in destinations
    assert "test_pool" not in destinations
    assert "resume" not in destinations
    assert "strict_lineage" not in destinations
    assert parser.get_default("device") == "cuda:0"


def test_comparison_bootstrap_rejects_nonmatching_request_sets() -> None:
    from harp8.evaluate_jspace_checkpoint import _paired_checkpoint_bootstrap

    left = [
        {"request_id": 1, "horizon": horizon, "recall_at_8": 0.5}
        for horizon in range(1, 5)
    ]
    right = [
        {"request_id": 2, "horizon": horizon, "recall_at_8": 0.5}
        for horizon in range(1, 5)
    ]
    with pytest.raises(ValueError, match="identical H1-H4 requests"):
        _paired_checkpoint_bootstrap(
            left, right, native_k=8, replicates=10, seed=42
        )


def test_v2_residual_scale_grid_has_exact_anchors_and_h5_h8_passthrough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _train, _validation, _features = _v2_checkpoint(tmp_path / "v2")
    exact_output = tmp_path / "exact"
    evaluate_saved_jspace_checkpoint(
        checkpoint,
        exact_output,
        device="cpu",
        bootstrap_replicates=20,
        strict_lineage=False,
    )
    forward_calls = 0
    original_forward = CompositeJSpaceV2Inference.forward

    def counted_forward(self, batch):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(self, batch)

    monkeypatch.setattr(CompositeJSpaceV2Inference, "forward", counted_forward)
    scaled_output = tmp_path / "scaled"
    manifest = evaluate_saved_jspace_checkpoint(
        checkpoint,
        scaled_output,
        residual_scale_grid=(0.0, 0.5, 1.0),
        batch_size=2,
        device="cpu",
        bootstrap_replicates=20,
        scale_bootstrap_replicates=128,
        scale_bootstrap_seed=17,
        strict_lineage=False,
    )
    assert forward_calls == 1
    exact = json.loads(
        (exact_output / "validation_metrics.json").read_text(encoding="utf-8")
    )
    scaled_exact = json.loads(
        (scaled_output / "validation_metrics.json").read_text(encoding="utf-8")
    )
    assert scaled_exact == exact
    diagnostic = json.loads(
        (scaled_output / "residual_scale_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["schema"] == JSPACE_CANDIDATE_RESIDUAL_SCALE_DIAGNOSTIC_SCHEMA
    assert diagnostic["requested_scales"] == [0.0, 0.5, 1.0]
    assert diagnostic["paired_reference_residual_scale"] == 0.0
    assert diagnostic["exact_checkpoint_residual_scale"] == 1.0
    assert diagnostic["h5_h8_exact_frozen_passthrough"] is True
    assert diagnostic["creates_trained_checkpoint"] is False
    by_scale = {
        float(row["residual_scale"]): row for row in diagnostic["scale_metrics"]
    }
    assert by_scale[1.0]["horizon_metrics"] == exact["horizon_metrics"]
    inactive = by_scale[0.0]["horizon_metrics"][4:]
    for scale in (0.5, 1.0):
        assert by_scale[scale]["horizon_metrics"][4:] == inactive
    for row in by_scale[0.0]["horizon_metrics"]:
        assert row["recall_at_8"] == row["base_recall_at_8"]
        assert row["request_macro_recall_at_8"] == row[
            "request_macro_base_recall_at_8"
        ]
    paired = by_scale[0.0]["paired_request_h1_h4_gain_vs_alpha0"]
    assert paired["paired_gain_at_8"] == 0.0
    assert paired["ci95_low"] == 0.0
    assert paired["ci95_high"] == 0.0
    assert manifest["residual_scale_diagnostic"][
        "h5_h8_exact_frozen_passthrough"
    ] is True
    assert (scaled_output / "residual_scale_summary.csv").is_file()
    assert (scaled_output / "residual_scale_horizon_metrics.csv").is_file()


def test_v2_scale_bootstrap_and_diagnostic_are_deterministic(tmp_path: Path) -> None:
    values = (0.0, 0.25, -0.125, 0.5)
    first = _paired_scale_bootstrap(values, replicates=1_000, seed=91)
    second = _paired_scale_bootstrap(values, replicates=1_000, seed=91)
    assert first == second
    assert first["paired_gain_at_8"] == np.mean(values)

    checkpoint, _train, _validation, _features = _v2_checkpoint(tmp_path / "v2")
    diagnostics = []
    for name in ("first", "second"):
        output = tmp_path / name
        evaluate_saved_jspace_checkpoint(
            checkpoint,
            output,
            residual_scale_grid=(0.0, 0.5, 1.0),
            device="cpu",
            scale_bootstrap_replicates=128,
            scale_bootstrap_seed=23,
            strict_lineage=False,
        )
        diagnostics.append(json.loads(
            (output / "residual_scale_diagnostic.json").read_text(
                encoding="utf-8"
            )
        ))
    assert diagnostics[0] == diagnostics[1]


def test_v2_train_scale_selection_and_validation_oracle_designations(
    tmp_path: Path,
) -> None:
    checkpoint, _train, _validation, _features = _v2_checkpoint(tmp_path / "v2")
    train_output = tmp_path / "train_scale"
    train_manifest = evaluate_saved_jspace_checkpoint(
        checkpoint,
        train_output,
        split="train",
        residual_scale_grid=(0.0, 0.5, 1.0),
        device="cpu",
        scale_bootstrap_replicates=64,
        strict_lineage=False,
    )
    train = json.loads(
        (train_output / "residual_scale_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert train_manifest["split"] == "train"
    assert train["designation"] == "level2_meta_train_scale_selection"
    assert train["eligible_for_model_selection"] is True
    assert train["uses_validation_labels_for_selection"] is False

    validation_output = tmp_path / "validation_scale"
    evaluate_saved_jspace_checkpoint(
        checkpoint,
        validation_output,
        residual_scale_grid=(0.0, 0.5, 1.0),
        device="cpu",
        scale_bootstrap_replicates=64,
        strict_lineage=False,
    )
    validation = json.loads(
        (validation_output / "residual_scale_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["designation"] == "post_hoc_validation_only_oracle_diagnostic"
    assert validation["eligible_for_model_selection"] is False
    assert validation["uses_validation_labels_for_selection"] is True


def test_v2_scale_grid_is_v2_only_and_refuses_sealed_test_before_open(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="train/validation"):
        evaluate_saved_jspace_checkpoint(
            tmp_path / "does_not_exist.pt",
            tmp_path / "refused",
            split="test",
            residual_scale_grid=(0.0, 1.0),
            device="cpu",
        )
    checkpoint, _paths = _v1_checkpoint(tmp_path / "v1")
    with pytest.raises(ValueError, match="only for v2"):
        evaluate_saved_jspace_checkpoint(
            checkpoint,
            tmp_path / "v1_refused",
            residual_scale_grid=(0.0, 1.0),
            device="cpu",
            strict_lineage=False,
        )
    parser = build_parser()
    split_action = next(action for action in parser._actions if action.dest == "split")
    assert tuple(split_action.choices) == ("train", "validation")
    assert not any(action.dest == "test_pool" for action in parser._actions)
    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["scale_bootstrap_replicates"] == (
        DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_REPLICATES
    )
    assert defaults["scale_bootstrap_seed"] == DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_SEED


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "0,,1", "0,0"])
def test_v2_scale_grid_parser_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_residual_scale_grid(value)
