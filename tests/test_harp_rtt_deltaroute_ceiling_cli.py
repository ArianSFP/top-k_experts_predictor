from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "runpod" / "evaluate_harp_deltaroute_v4_ceiling.py"
SPEC = importlib.util.spec_from_file_location("evaluate_harp_deltaroute_v4_ceiling", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


PARENT = "2" * 64


def _bundle() -> dict[str, object]:
    batch, horizons, layers, nodes, experts, exact_k = 4, 4, 2, 2, 80, 8
    anchor = torch.zeros(batch, horizons, layers, experts)
    anchor[..., :exact_k] = 5.0
    anchor_marginals = torch.zeros_like(anchor)
    anchor_marginals[..., :exact_k] = 1.0
    native = torch.empty(batch, nodes, layers, exact_k, dtype=torch.long)
    native[:, 0] = torch.arange(8, 16).reshape(1, 1, 8)
    native[:, 1] = torch.arange(16, 24).reshape(1, 1, 8)
    target = torch.arange(8, 16).reshape(1, 1, 1, 8).expand(
        batch, horizons, layers, exact_k
    ).clone()
    branch_scores = torch.zeros(batch, horizons, layers, nodes, experts)
    branch_scores[..., 0, 8:16] = 6.0
    branch_scores[..., 1, 16:24] = 6.0
    marginals = torch.zeros_like(branch_scores)
    marginals[..., 0, 8:16] = 1.0
    marginals[..., 1, 16:24] = 1.0
    captured = torch.tensor([0.8, 0.1]).reshape(1, 1, 2).expand(batch, horizons, 2)
    return {
        "schema": MODULE.BUNDLE_SCHEMA,
        "provenance": {
            "outer_split": "train",
            "parent_checkpoint_sha256": PARENT,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        },
        "request_ids": ["r0", "r0", "r1", "r1"],
        "anchor_scores": anchor,
        "anchor_marginals": anchor_marginals,
        "target_ids": target,
        "future_valid": torch.ones(batch, horizons, layers, dtype=torch.bool),
        "node_native_ids": native,
        "branch_mask": torch.ones(batch, horizons, nodes, dtype=torch.bool),
        "factual_branch_indices": torch.zeros(batch, horizons, dtype=torch.long),
        "posteriors": {
            "target": {"captured": captured, "other": torch.full((batch, horizons), 0.1)},
            "learned": {"captured": captured, "other": torch.full((batch, horizons), 0.1)},
            "mtp": {"captured": captured, "other": torch.full((batch, horizons), 0.1)},
        },
        "learned_node_scores": branch_scores,
        "learned_node_marginals": marginals,
        "prefix_mismatch": torch.ones(batch, horizons, dtype=torch.bool),
        "exact_k": exact_k,
        "candidate_width": 64,
    }


def test_ceiling_reports_factual_native_top8_swap_caps_and_no_training() -> None:
    report, sidecar = MODULE.evaluate_bundle(_bundle())
    factual = report["conditions"]["native_factual_branch_with_anchor_fallback"]
    assert factual["request_macro_h2_h4"] == 1.0
    assert report["native_factual_h2_h4_gate"]["passed"] is True
    swap = report["swap_cap_oracle"]
    assert swap["1"]["request_macro"] <= swap["8"]["request_macro"]
    assert report["training_started"] is False
    assert report["optimizer_constructed"] is False
    assert sidecar["available"] is True


def test_ceiling_bundle_rejects_wrong_parent_and_nontrain(tmp_path: Path) -> None:
    path = tmp_path / "ceiling.pt"
    torch.save(_bundle(), path)
    with pytest.raises(ValueError, match="checkpoint hash"):
        MODULE.load_bundle(path, expected_parent_sha256="3" * 64)
    value = _bundle()
    value["provenance"]["outer_split"] = "validation"  # type: ignore[index]
    torch.save(value, path)
    with pytest.raises(PermissionError, match="outer-train"):
        MODULE.load_bundle(path, expected_parent_sha256=PARENT)
