from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "runpod" / "train_harp_delta_v3.py"
SPEC = importlib.util.spec_from_file_location("train_harp_delta_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_reuse_partition_is_explicitly_diagnostic_outer_train(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "partition.json",
        {
            "schema": MODULE.REUSE_PARTITION_SCHEMA,
            "outer_split": "train",
            "positions": 4096,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        },
    )
    value = MODULE._manifest(path, "b2_reuse_4096")
    assert value["positions"] == 4096
    with pytest.raises(ValueError, match="schema mismatch"):
        MODULE._manifest(path, "delta20k")


def test_reuse_split_requires_frozen_224_32_request_partition(tmp_path: Path) -> None:
    training = [f"train-{index}" for index in range(224)]
    tuning = [f"tune-{index}" for index in range(32)]
    path = write_json(
        tmp_path / "split.json",
        {
            "schema": MODULE.REUSE_SPLIT_SCHEMA,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
            "inner_split": {
                "request_disjoint": True,
                "training_requests": training,
                "tuning_requests": tuning,
                "training_rows": 3584,
                "tuning_rows": 512,
            },
        },
    )
    assert MODULE._reuse_split_manifest(path)["inner_split"]["request_disjoint"]


def test_b31_gate_requires_native_routes_with_learned_posterior(tmp_path: Path) -> None:
    native = {"coverage_h2": 0.988, "coverage_h3": 0.986, "coverage_h4": 0.982}
    path = write_json(
        tmp_path / "b31.json",
        {
            "schema": MODULE.B31_REPORT_SCHEMA,
            "conditions": 56,
            "training_started": False,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
            "bundle_sha256": "a" * 64,
            "factorial": {"oracle_native__learned__quota_32_32": native},
        },
    )
    gate = MODULE._b31_gate(path)
    assert gate["native_learned_quota32_mean_h2_h4"] > 0.985

    native["coverage_h3"] = 0.96
    write_json(path, json.loads(path.read_text()) | {
        "factorial": {"oracle_native__learned__quota_32_32": native}
    })
    with pytest.raises(PermissionError, match="route information"):
        MODULE._b31_gate(path)


def test_semantic_checkpoint_selection_uses_metric_aligned_c64() -> None:
    metrics = {
        "loss": 99.0,
        "semantic_candidate_coverage_at_64_quota32": 0.91,
    }
    assert MODULE.selection_value("semantic", metrics) == pytest.approx(0.91)
