import torch

from runpod.analyze_harp_resident_complementarity import (
    branch_compatibility_grid, build_cells, information_gate, summarize,
)
from runpod.train_harp_resident_fusion import FEATURE_WIDTH, causal_features


def _mass(ids: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(*ids.shape[:-1], 256)
    return result.scatter(-1, ids, 1.0)


def _synthetic_pair() -> tuple[dict, dict]:
    target = torch.arange(8).reshape(1, 1, 1, 8).expand(1, 4, 40, 8)
    shadow_ids = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]).reshape(
        1, 1, 8
    ).expand(4, 40, 8)
    harp_ids = torch.tensor([4, 5, 6, 7, 12, 13, 14, 15]).reshape(
        1, 1, 8
    ).expand(4, 40, 8)
    captured = torch.zeros(4, 32)
    captured[:, 0] = 1.0
    node_ids = torch.full((32, 40, 8), -1, dtype=torch.long)
    node_ids[0] = shadow_ids[0]
    record = {
        "request_id": "row-request",
        "source_request_id": "source-request",
        "ceiling_row": 0,
        "final_shadow_ids": shadow_ids,
        "final_shadow_marginals": _mass(shadow_ids),
        "captured_shadow_marginals": _mass(shadow_ids),
        "captured_probability": torch.ones(4),
        "shadow_lm_captured_probabilities": captured,
        "shadow_lm_other_probabilities": torch.zeros(4),
        "branch_mask": captured.bool(),
        "selected_ids": node_ids,
        "prior_missing_mass": torch.full((32, 40), 0.3),
        "prior_missing_count": torch.full((32, 40), 3.0),
        "available_experts": torch.zeros(40, 256, dtype=torch.bool),
    }
    ceiling = {
        "request_ids": ["row-request"],
        "anchor_marginals": _mass(harp_ids)[None],
        "target_ids": target,
        "future_valid": torch.ones(1, 4, 40, dtype=torch.bool),
        "prefix_mismatch": torch.tensor([[False, False, True, True]]),
        "factual_branch_indices": torch.zeros(1, 4, dtype=torch.long),
    }
    return {"records": [record]}, ceiling


def test_synthetic_overlap_gate_finds_large_complementarity() -> None:
    sidecar, ceiling = _synthetic_pair()
    cells = build_cells(sidecar, ceiling)
    summary = summarize(cells)
    assert len(cells) == 160
    assert summary["shadow_recall_h1_h4"] == 0.5
    assert summary["harp_recall_h1_h4"] == 0.5
    assert summary["harp_unique_h1_h4"] == 0.5
    assert summary["expert_merge_k32_h1_h4"] == 1.0
    assert summary["harp_unique_cold_h1_h4"] == 0.5
    assert information_gate(summary, reference=0.916573)["passed"] is True


def test_causal_feature_map_ignores_diagnostic_labels() -> None:
    sidecar, ceiling = _synthetic_pair()
    record = sidecar["records"][0]
    harp = ceiling["anchor_marginals"][0]
    baseline = causal_features(record, harp)
    poisoned = dict(record)
    poisoned["prefix_mismatch"] = torch.ones(4, dtype=torch.bool)
    poisoned["factual_branch_indices"] = torch.full((4,), 31)
    repeated = causal_features(poisoned, harp)
    assert baseline.shape == (4, 40, 256, FEATURE_WIDTH)
    assert torch.equal(baseline, repeated)


def test_branch_compatibility_preserves_captured_and_other_contract() -> None:
    sidecar, ceiling = _synthetic_pair()
    grid = branch_compatibility_grid(sidecar, ceiling, eta_values=(0.0, 1.0))
    assert grid["diagnostic_only"] is True
    assert grid["conditions"]["0.0"]["recall_h1_h4"] == 0.5
    assert grid["conditions"]["1.0"]["recall_h1_h4"] == 0.5
    assert grid["conditions"]["1.0"]["factual_branch_top1"] == 1.0
