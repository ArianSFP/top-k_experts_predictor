from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "runpod" / "evaluate_harp_rtt_b31_factorial.py"
SPEC = importlib.util.spec_from_file_location("evaluate_harp_rtt_b31_factorial", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def bundle() -> dict[str, object]:
    anchor = torch.zeros(1, 4, 1, 80)
    anchor[..., :8] = 2
    marginal = torch.zeros_like(anchor)
    marginal[..., :8] = 1
    branches = anchor[..., None, :].repeat(1, 1, 1, 2, 1)
    branches[..., 0, 8:16] = 5
    branches[..., 1, 16:24] = 5
    captured = torch.full((1, 4, 2), 0.4)
    return {
        "schema": MODULE.BUNDLE_SCHEMA,
        "provenance": {
            "outer_split": "train",
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        },
        "anchor_scores": anchor,
        "anchor_marginals": marginal,
        "target_ids": torch.arange(8, 16).reshape(1, 1, 1, 8).expand(1, 4, 1, 8),
        "branch_scores": {"semantic": branches},
        "posteriors": {"learned": {"captured": captured, "other": torch.full((1, 4), 0.2)}},
        "branch_mask": torch.ones(1, 4, 2, dtype=torch.bool),
        "exact_k": 8,
        "candidate_width": 64,
    }


def test_b31_bundle_evaluator_reports_all_horizons(tmp_path: Path) -> None:
    path = tmp_path / "factor.pt"
    torch.save(bundle(), path)
    loaded = MODULE.load_bundle(path)
    report = MODULE.evaluate_bundle(loaded)
    assert report["conditions"] == 4
    condition = next(iter(report["factorial"].values()))
    assert all(f"coverage_h{h}" in condition for h in range(1, 5))
    assert report["training_started"] is False


def test_b31_bundle_refuses_non_train_provenance(tmp_path: Path) -> None:
    value = bundle()
    value["provenance"]["outer_split"] = "validation"  # type: ignore[index]
    path = tmp_path / "bad.pt"
    torch.save(value, path)
    with pytest.raises(PermissionError, match="outer-train"):
        MODULE.load_bundle(path)


def test_b31_bundle_accepts_compact_selected_ids(tmp_path: Path) -> None:
    value = bundle()
    scores = value.pop("branch_scores")["semantic"]  # type: ignore[index,union-attr]
    value["branch_selected_ids"] = {
        "semantic": torch.argsort(
            scores, dim=-1, descending=True, stable=True
        )[..., :8]
    }
    path = tmp_path / "compact.pt"
    torch.save(value, path)
    report = MODULE.evaluate_bundle(MODULE.load_bundle(path))
    assert report["route_sources"] == ["semantic"]
    assert report["conditions"] == 4
