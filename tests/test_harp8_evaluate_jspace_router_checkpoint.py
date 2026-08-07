from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from harp8.evaluate_jspace_router_checkpoint import (
    FULL_ROUTER_CHECKPOINT_EVALUATION_SCHEMA,
    FULL_ROUTER_RESIDUAL_SCALE_DIAGNOSTIC_SCHEMA,
    _paired_request_bootstrap,
    build_parser,
    evaluate_saved_full_router_checkpoint,
    parse_residual_scale,
    parse_residual_scale_grid,
)
from harp8.jspace_router_forecaster import FullRouterLossConfig
from harp8.train_jspace_router_forecaster import (
    FullRouterTrainingConfig,
    derive_router_forecaster_config,
    train_jspace_router_forecaster,
)


def _fixture_module():
    path = Path(__file__).with_name("test_harp8_jspace_router_training.py")
    specification = importlib.util.spec_from_file_location(
        "_full_router_evaluator_fixtures", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


FIXTURES = _fixture_module()


def _checkpoint(tmp_path: Path, *, dual_stream: bool = False):
    tmp_path.mkdir()
    train, validation, features, secondary, secondary_rms = FIXTURES._artifacts(
        tmp_path, dual_stream=dual_stream
    )
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
    output = tmp_path / "training"
    train_jspace_router_forecaster(
        train,
        validation,
        output,
        model_config,
        FullRouterLossConfig(
            hard_negative_end_rank=12,
            predicted_negative_count=4,
        ),
        training_config,
        target_features=features,
        target_feature_rms=None,
        secondary_target_features=secondary,
        secondary_target_feature_rms=secondary_rms,
        device="cpu",
        strict_lineage=False,
    )
    return output / "last.pt", train, validation, features, secondary, secondary_rms


def _evaluate(
    checkpoint: Path,
    output: Path,
    train,
    validation,
    features: Path,
    secondary: Path | None,
    secondary_rms: Path | None,
    split: str = "validation",
    **kwargs,
):
    return evaluate_saved_full_router_checkpoint(
        checkpoint,
        output,
        split=split,
        train_pool=train.pool.root,
        validation_pool=validation.pool.root,
        capture_dir=train.aligned.capture_dir,
        mtp_dir=train.aligned.mtp_dir,
        target_features=features,
        secondary_target_features=secondary,
        secondary_target_feature_rms=secondary_rms,
        rows_per_request=10,
        batch_size=1,
        device="cpu",
        strict_lineage=False,
        **kwargs,
    )


def test_last_checkpoint_evaluation_and_prediction_only_export(
    tmp_path: Path,
) -> None:
    checkpoint, train, validation, features, secondary, secondary_rms = _checkpoint(
        tmp_path / "fixture"
    )
    output = tmp_path / "evaluation"
    manifest = _evaluate(
        checkpoint,
        output,
        train,
        validation,
        features,
        secondary,
        secondary_rms,
        export_row_top8=True,
    )
    assert manifest["schema"] == FULL_ROUTER_CHECKPOINT_EVALUATION_SCHEMA
    assert manifest["completed_epoch"] == 1
    assert manifest["split"] == "validation"
    assert manifest["rows"] == 2
    assert manifest["sealed_test_accessed"] is False
    assert manifest["optimizer_constructed"] is False
    assert manifest["trainer_invoked"] is False
    assert manifest["input_provenance_verified"] is True
    assert manifest["target_state_stream_contract_verified"] is True
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["split"] == "validation"
    assert len(metrics["horizon_metrics"]) == 8
    with np.load(output / "predicted_top8_rows.npz", allow_pickle=False) as rows:
        assert rows["predicted_top8"].shape == (2, 8, 2, 8)
        assert rows["predicted_top8"].dtype == np.uint16
        assert rows["request_id"].tolist() == [202, 202]
        assert float(rows["residual_scale"].item()) == 1.0
        assert not bool(rows["contains_teacher_router_scores"].item())
        assert not bool(rows["contains_target_top8"].item())
        assert "teacher_router_scores" not in rows.files
        assert "target_top8" not in rows.files
    assert manifest["row_top8_export"]["residual_scale"] == 1.0
    assert manifest["residual_scale_diagnostic"] is None


def test_residual_scale_grid_preserves_baseline_and_checkpoint_anchors(
    tmp_path: Path,
) -> None:
    checkpoint, train, validation, features, secondary, secondary_rms = _checkpoint(
        tmp_path / "fixture"
    )
    exact_output = tmp_path / "exact"
    _evaluate(
        checkpoint,
        exact_output,
        train,
        validation,
        features,
        secondary,
        secondary_rms,
    )
    scaled_output = tmp_path / "scaled"
    manifest = _evaluate(
        checkpoint,
        scaled_output,
        train,
        validation,
        features,
        secondary,
        secondary_rms,
        residual_scale_grid=(0.0, 0.5, 1.0),
        export_row_top8=True,
        export_residual_scale=0.0,
        bootstrap_replicates=128,
        bootstrap_seed=17,
    )
    # metrics.json always remains the exact alpha=1 checkpoint evaluation.
    exact_metrics = json.loads(
        (exact_output / "metrics.json").read_text(encoding="utf-8")
    )
    scaled_exact_metrics = json.loads(
        (scaled_output / "metrics.json").read_text(encoding="utf-8")
    )
    assert scaled_exact_metrics == exact_metrics

    diagnostic = json.loads(
        (scaled_output / "residual_scale_diagnostic.json").read_text(encoding="utf-8")
    )
    assert diagnostic["schema"] == FULL_ROUTER_RESIDUAL_SCALE_DIAGNOSTIC_SCHEMA
    assert (
        diagnostic["designation"]
        == "post_hoc_validation_only_oracle_diagnostic"
    )
    assert diagnostic["uses_same_split_labels_for_selection"] is True
    assert diagnostic["uses_validation_labels_for_selection"] is True
    assert diagnostic["eligible_for_model_selection"] is False
    assert diagnostic["creates_trained_checkpoint"] is False
    assert diagnostic["requested_scales"] == [0.0, 0.5, 1.0]
    assert diagnostic["best_residual_scale"] in diagnostic["requested_scales"]
    assert diagnostic["paired_reference_residual_scale"] == 0.0
    assert diagnostic["bootstrap"] == {
        "confidence_level": 0.95,
        "interval": "percentile",
        "pairing": "same_request_and_horizon",
        "replicates": 128,
        "seed": 17,
        "unit": "request",
    }
    by_scale = {
        float(row["residual_scale"]): row for row in diagnostic["scale_metrics"]
    }
    for horizon in by_scale[0.0]["horizon_metrics"]:
        assert horizon["recall_at_8"] == horizon["base_recall_at_8"]
        assert (
            horizon["request_macro_recall_at_8"]
            == horizon["request_macro_base_recall_at_8"]
        )
    alpha0_paired = by_scale[0.0]["paired_request_h1_h4_gain_vs_alpha0"]
    assert alpha0_paired["paired_gain_at_8"] == 0.0
    assert alpha0_paired["ci95_low"] == 0.0
    assert alpha0_paired["ci95_high"] == 0.0
    for horizon in by_scale[0.0]["paired_request_horizon_gains_vs_alpha0"]:
        assert horizon["paired_gain_at_8"] == 0.0
        assert horizon["ci95_low"] == 0.0
        assert horizon["ci95_high"] == 0.0
    assert (
        by_scale[1.0]["mean_h1_h4_recall_at_8"]
        == exact_metrics["mean_h1_h4_recall_at_8"]
    )
    assert (
        by_scale[1.0]["mean_h1_h8_recall_at_8"]
        == exact_metrics["mean_h1_h8_recall_at_8"]
    )
    assert (
        by_scale[1.0]["h2_request_macro_recall_at_8"]
        == exact_metrics["h2_request_macro_recall_at_8"]
    )
    expected_paired_gain = float(
        np.mean(
            [
                candidate["request_macro_recall_at_8"]
                - baseline["request_macro_recall_at_8"]
                for candidate, baseline in zip(
                    by_scale[1.0]["horizon_metrics"][:4],
                    by_scale[0.0]["horizon_metrics"][:4],
                    strict=True,
                )
            ]
        )
    )
    assert by_scale[1.0]["paired_request_h1_h4_gain_vs_alpha0"][
        "paired_gain_at_8"
    ] == pytest.approx(expected_paired_gain, abs=1e-15)
    assert manifest["residual_scale_diagnostic"]["creates_trained_checkpoint"] is False
    with np.load(scaled_output / "predicted_top8_rows.npz", allow_pickle=False) as rows:
        assert float(rows["residual_scale"].item()) == 0.0
    assert manifest["row_top8_export"]["residual_scale"] == 0.0
    assert (
        "paired_h1_h4_ci95_low"
        in (scaled_output / "residual_scale_summary.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert (
        "paired_gain_ci95_low"
        in (scaled_output / "residual_scale_horizon_metrics.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    repeat_output = tmp_path / "scaled_repeat"
    _evaluate(
        checkpoint,
        repeat_output,
        train,
        validation,
        features,
        secondary,
        secondary_rms,
        residual_scale_grid=(0.0, 0.5, 1.0),
        bootstrap_replicates=128,
        bootstrap_seed=17,
    )
    repeated = json.loads(
        (repeat_output / "residual_scale_diagnostic.json").read_text(encoding="utf-8")
    )
    assert repeated == diagnostic


def test_train_scale_grid_is_selection_eligible_but_validation_is_not(
    tmp_path: Path,
) -> None:
    checkpoint, train, validation, features, secondary, secondary_rms = _checkpoint(
        tmp_path / "fixture"
    )
    train_output = tmp_path / "train_scale"
    train_manifest = _evaluate(
        checkpoint,
        train_output,
        train,
        validation,
        features,
        secondary,
        secondary_rms,
        split="train",
        residual_scale_grid=(0.0, 0.5, 1.0),
        bootstrap_replicates=128,
        bootstrap_seed=19,
    )
    train_diagnostic = json.loads(
        (train_output / "residual_scale_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert train_manifest["split"] == "train"
    assert train_diagnostic["split"] == "train"
    assert train_diagnostic["designation"] == "level2_meta_train_scale_selection"
    assert train_diagnostic["uses_same_split_labels_for_selection"] is True
    assert train_diagnostic["uses_validation_labels_for_selection"] is False
    assert train_diagnostic["eligible_for_model_selection"] is True

    validation_output = tmp_path / "validation_scale"
    validation_manifest = _evaluate(
        checkpoint,
        validation_output,
        train,
        validation,
        features,
        secondary,
        secondary_rms,
        split="validation",
        residual_scale_grid=(0.0, 0.5, 1.0),
        bootstrap_replicates=128,
        bootstrap_seed=19,
    )
    validation_diagnostic = json.loads(
        (validation_output / "residual_scale_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation_manifest["split"] == "validation"
    assert (
        validation_diagnostic["designation"]
        == "post_hoc_validation_only_oracle_diagnostic"
    )
    assert validation_diagnostic["eligible_for_model_selection"] is False
    assert validation_manifest["residual_scale_diagnostic"][
        "eligible_for_model_selection"
    ] is False


def test_paired_request_bootstrap_is_deterministic_and_point_exact() -> None:
    values = (0.0, 0.25, -0.125, 0.5)
    first = _paired_request_bootstrap(values, replicates=1_000, seed=91)
    second = _paired_request_bootstrap(values, replicates=1_000, seed=91)
    assert first == second
    assert first["paired_gain_at_8"] == np.mean(values)
    assert first["ci95_low"] <= first["paired_gain_at_8"]
    assert first["ci95_high"] >= first["paired_gain_at_8"]


def test_residual_scale_grid_requires_explicit_export_scale(tmp_path: Path) -> None:
    checkpoint, train, validation, features, secondary, secondary_rms = _checkpoint(
        tmp_path / "fixture"
    )
    with pytest.raises(ValueError, match="explicit selected scale"):
        _evaluate(
            checkpoint,
            tmp_path / "rejected_export",
            train,
            validation,
            features,
            secondary,
            secondary_rms,
            residual_scale_grid=(0.0, 1.0),
            export_row_top8=True,
        )
    for kwargs, message in (
        ({"bootstrap_replicates": 0}, "replicates"),
        ({"bootstrap_seed": -1}, "seed"),
    ):
        with pytest.raises(ValueError, match=message):
            _evaluate(
                checkpoint,
                tmp_path / f"rejected_{message}",
                train,
                validation,
                features,
                secondary,
                secondary_rms,
                residual_scale_grid=(0.0, 1.0),
                **kwargs,
            )


@pytest.mark.parametrize(
    "value",
    ["-0.1", "nan", "inf", "0,,1", "0,0"],
)
def test_residual_scale_grid_parser_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_residual_scale_grid(value)


@pytest.mark.parametrize("value", ["-1", "nan", "-inf"])
def test_export_residual_scale_parser_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_residual_scale(value)


def test_evaluator_refuses_test_split_before_data_open(tmp_path: Path) -> None:
    checkpoint, train, validation, features, secondary, secondary_rms = _checkpoint(
        tmp_path / "fixture"
    )
    with pytest.raises(PermissionError, match="train/validation"):
        evaluate_saved_full_router_checkpoint(
            checkpoint,
            tmp_path / "refused",
            split="test",
            train_pool=train.pool.root,
            validation_pool=validation.pool.root,
            capture_dir=train.aligned.capture_dir,
            mtp_dir=train.aligned.mtp_dir,
            target_features=features,
            secondary_target_features=secondary,
            secondary_target_feature_rms=secondary_rms,
            rows_per_request=10,
            device="cpu",
            residual_scale_grid=(0.0, 1.0),
            strict_lineage=False,
        )
    parser = build_parser()
    split_action = next(action for action in parser._actions if action.dest == "split")
    assert tuple(split_action.choices) == ("train", "validation")
    assert not any(action.dest == "test_pool" for action in parser._actions)
    assert not any(action.dest == "resume" for action in parser._actions)
    assert not any(action.dest == "strict_lineage" for action in parser._actions)
    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["bootstrap_replicates"] == 5_000
    assert defaults["bootstrap_seed"] == 20_260_807


def test_evaluator_rejects_changed_dual_stream_contract(tmp_path: Path) -> None:
    checkpoint, train, validation, features, secondary, secondary_rms = _checkpoint(
        tmp_path / "fixture", dual_stream=True
    )
    assert secondary is not None and secondary_rms is not None
    alternate = tmp_path / "alternate_secondary.npy"
    changed = np.asarray(train.secondary_target_features).copy()
    changed[0, 0, 0] += np.float16(1.0)
    np.save(alternate, changed)
    with pytest.raises(
        ValueError,
        match="input_provenance|target_state_stream_contract",
    ):
        _evaluate(
            checkpoint,
            tmp_path / "contract_rejected",
            train,
            validation,
            features,
            alternate,
            secondary_rms,
        )
