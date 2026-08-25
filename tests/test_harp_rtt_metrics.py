from __future__ import annotations

import pytest
import torch
from torch import nn

from harp_rtt.metrics import (
    RequestMetricAccumulator,
    cache_set_counts,
    candidate_coverage_at_k,
    candidate_coverage_gate,
    evaluate_harp_rtt,
    model_selection_tuple,
    paired_complete_request_bootstrap,
    slot_recall_at_k,
    summarize_harp_rtt_predictions,
)


def test_slot_recall_uses_lower_expert_id_for_exact_ties() -> None:
    scores = torch.zeros(1, 4, 1, 6)
    targets = torch.tensor([[[[0, 2]], [[0, 2]], [[0, 2]], [[0, 2]]]])
    recall = slot_recall_at_k(scores, targets, k=2)
    # Stable top-2 is [0,1], so exactly one of the two target slots is found.
    assert torch.equal(recall, torch.full((1, 4, 1), 0.5))


def test_cacheset_counts_true_current_future_intersection_and_prediction() -> None:
    current = torch.tensor([[[1, 2], [3, 4]]])
    target = torch.tensor([[
        [[1, 5], [3, 4]],
        [[6, 7], [3, 8]],
        [[1, 2], [4, 9]],
        [[2, 9], [3, 4]],
    ]])
    predicted = torch.tensor([[
        [[1, 6], [3, 7]],
        [[6, 7], [8, 9]],
        [[1, 2], [4, 5]],
        [[9, 8], [3, 4]],
    ]])
    valid = torch.tensor([[
        [True, True],
        [True, True],
        [True, False],
        [False, True],
    ]])

    correct, true, cells = cache_set_counts(predicted, current, target, valid)

    assert correct.tolist() == [[2, 0, 2, 2]]
    assert true.tolist() == [[3, 1, 2, 2]]
    assert cells.tolist() == [[2, 2, 1, 1]]


def test_cacheset_counts_rejects_misaligned_current_geometry() -> None:
    predicted = torch.zeros(1, 4, 2, 2, dtype=torch.long)
    target = torch.zeros_like(predicted)
    valid = torch.ones(1, 4, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="current IDs"):
        cache_set_counts(predicted, torch.zeros(1, 3, 2), target, valid)


def test_request_macro_summary_does_not_weight_long_requests_more() -> None:
    scores = torch.full((3, 4, 1, 6), -10.0)
    scores[:2, ..., 0] = 2.0
    scores[:2, ..., 1] = 1.0
    scores[2, ..., 4] = 2.0
    scores[2, ..., 5] = 1.0
    targets = torch.tensor([0, 1]).expand(3, 4, 1, 2).clone()
    candidates = torch.tensor([0, 1, 2, 3]).expand(3, 4, 1, 4).clone()
    report = summarize_harp_rtt_predictions(
        scores,
        targets,
        ["long", "long", "short"],
        target_router_logits=scores.clone(),
        candidate_ids=candidates,
        k=2,
    )
    assert report["mean_h1_h4_request_macro_slot_recall_at_2"] == pytest.approx(0.5)
    assert report["h4_request_macro_slot_recall_at_2"] == pytest.approx(0.5)
    assert report["mean_h1_h4_request_macro_candidate_coverage_at_4"] == 1.0
    assert report["mean_h1_h4_request_macro_router_kl"] == pytest.approx(
        0.0, abs=1e-7
    )
    assert report["complete_requests"] == 2
    selection = model_selection_tuple(report, k=2, candidate_width=4)
    assert selection[:3] == pytest.approx((0.5, 0.5, 0.5))
    assert selection[4] == 1.0


def test_candidate_coverage_treats_duplicate_ids_as_a_set() -> None:
    candidates = torch.tensor([[[[0, 0, 0, 0]]]])
    targets = torch.tensor([[[[0, 1]]]])
    coverage = candidate_coverage_at_k(candidates, targets, experts=4, k=2)
    assert coverage.item() == pytest.approx(0.5)


def test_candidate_padding_cannot_overwrite_real_expert_zero() -> None:
    candidates = torch.tensor([[[[0, -1, -1, -1]]]])
    mask = torch.tensor([[[[True, False, False, False]]]])
    targets = torch.tensor([[[[0, 1]]]])
    coverage = candidate_coverage_at_k(
        candidates, targets, mask, experts=4, k=2
    )
    assert coverage.item() == pytest.approx(0.5)


def test_c64_gate_uses_formal_inclusive_thresholds() -> None:
    report = {
        "candidate_width": 64,
        "mean_h1_h4_request_macro_candidate_coverage_at_64": 0.985,
        "h4_request_macro_candidate_coverage_at_64": 0.970,
    }
    result = candidate_coverage_gate(report)
    assert result["passed"] is True
    assert result["mean_margin"] == pytest.approx(0.0)


def _paired_rows(gain: float) -> list[dict[str, float | int | str]]:
    return [
        {
            "request_id": request,
            "horizon": horizon,
            "slot_recall_at_8": 0.5 + gain,
        }
        for request in ("a", "b", "c")
        for horizon in range(1, 5)
    ]


def test_paired_bootstrap_requires_1000_and_complete_matched_requests() -> None:
    with pytest.raises(ValueError, match="at least 1,000"):
        paired_complete_request_bootstrap(_paired_rows(0.1), _paired_rows(0.0), replicates=999)
    result = paired_complete_request_bootstrap(
        _paired_rows(0.1), _paired_rows(0.0), replicates=1_000, seed=7
    )
    assert result["requests"] == 3
    assert result["paired_gain"] == pytest.approx(0.1)
    assert result["lower_bound"] > 0
    with pytest.raises(ValueError, match="complete H1--H4"):
        paired_complete_request_bootstrap(
            _paired_rows(0.1)[:-1], _paired_rows(0.0), replicates=1_000
        )


class _NestedOutputModel(nn.Module):
    def forward(self, *, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        inputs = batch["inputs"]
        assert isinstance(inputs, dict)
        return {
            "active_scores": inputs["predicted_scores"],
            "candidate_ids": inputs["candidate_ids"],
            "candidate_mask": torch.ones_like(inputs["candidate_ids"], dtype=torch.bool),
        }


def test_evaluator_consumes_nested_contract_and_keeps_test_sealed() -> None:
    scores = torch.randn(2, 4, 1, 6)
    targets = torch.tensor([0, 1]).expand(2, 4, 1, 2).clone()
    candidates = torch.arange(4).expand(2, 4, 1, 4).clone()
    batch = {
        "metadata": {"request_id": ["r0", "r1"]},
        "inputs": {"predicted_scores": scores, "candidate_ids": candidates},
        "targets": {
            "future_selected_ids": targets,
            "future_router_logits": scores.clone(),
            "future_available": torch.ones(2, 4, 1, dtype=torch.bool),
        },
    }
    model = _NestedOutputModel().train()
    report = evaluate_harp_rtt(
        model, [batch], device="cpu", split="validation", autocast=False, k=2
    )
    assert report["requests"] == 2
    assert model.training is True
    with pytest.raises(PermissionError, match="sealed"):
        evaluate_harp_rtt(model, [batch], device="cpu", split="test", k=2)


def test_accumulator_requires_candidates_consistently() -> None:
    scores = torch.randn(1, 4, 1, 6)
    targets = torch.tensor([0, 1]).expand(1, 4, 1, 2).clone()
    accumulator = RequestMetricAccumulator(k=2)
    accumulator.update(scores, targets, ["r"], target_router_logits=scores)
    with pytest.raises(ValueError, match="every batch or none"):
        accumulator.update(
            scores,
            targets,
            ["r"],
            target_router_logits=scores,
            candidate_ids=torch.arange(4).expand(1, 4, 1, 4),
        )
