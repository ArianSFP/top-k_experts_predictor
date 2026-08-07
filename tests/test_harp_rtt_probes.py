from __future__ import annotations

import pytest
import torch

from harp_rtt.geometry import build_centered_router_geometry
from harp_rtt.probes import (
    C64OverfitHook,
    candidate_union_ids,
    compare_tree_probes,
    exact_token_paired_probe,
    future_router_reconstruction_probe,
    generator_candidate_width_probe,
    prefix_conditional_bayes_ceiling,
    tree_availability_probe,
    two_layer_future_state_ceiling_probe,
)


def test_candidate_width_ladder_and_stable_union() -> None:
    scores = -torch.arange(160, dtype=torch.float32).view(1, 1, 1, 160)
    scores = scores.expand(2, 4, 1, 160).clone()
    targets = torch.arange(8).expand(2, 4, 1, 8).clone()
    report = generator_candidate_width_probe(
        scores, targets, ["a", "b"], split="validation"
    )
    assert set(report["widths"]) == {"C32", "C64", "C128"}
    assert report["widths"]["C64"]["candidate_gate"]["passed"] is True
    tied = candidate_union_ids(torch.zeros(1, 4, 1, 10), 4)
    assert tied[0, 0, 0].tolist() == [0, 1, 2, 3]


def _condition_report(value: float) -> dict[str, object]:
    rows = [
        {"request_id": request, "horizon": horizon, "slot_recall_at_8": value}
        for request in ("a", "b")
        for horizon in range(1, 5)
    ]
    return {
        "split": "validation",
        "mean_h1_h4_request_macro_slot_recall_at_8": value,
        "request_metrics": rows,
    }


def test_exact_token_probe_reports_paired_lower_bound() -> None:
    result = exact_token_paired_probe(
        _condition_report(0.8), _condition_report(0.7), replicates=1_000
    )
    assert result["passed"] is True
    assert result["paired"]["lower_bound"] > 0
    test_report = _condition_report(0.8)
    test_report["split"] = "test"
    with pytest.raises(PermissionError, match="sealed"):
        exact_token_paired_probe(test_report, test_report, replicates=1_000)


def test_true_router_input_reconstruction_is_exact() -> None:
    torch.manual_seed(5)
    weights = torch.randn(2, 10, 4)
    geometry = build_centered_router_geometry(weights, relative_rank_threshold=0.0)
    inputs = torch.randn(1, 4, 2, 4)
    logits = torch.einsum("bhld,led->bhle", inputs, weights)
    selected = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :3]
    report = future_router_reconstruction_probe(
        geometry,
        inputs,
        logits,
        selected_ids=selected,
        k=3,
        maximum_absolute_logit_error=1e-4,
    )
    assert report["passed"] is True
    assert report["exact_topk_set_agreement"] == 1.0
    assert report["selected_label_slot_recall_at_3"] == 1.0


def _future_state_probe_fixture():
    torch.manual_seed(17)
    geometry = build_centered_router_geometry(
        torch.randn(2, 10, 4), relative_rank_threshold=0.0
    )
    train_pre_attention = torch.zeros(2, 4, 2, 3)
    validation_pre_attention = torch.zeros(2, 4, 2, 3)
    router_template = torch.randn(1, 1, 2, 4)
    train_router = router_template.expand(2, 4, 2, 4).clone()
    validation_router = router_template.expand(2, 4, 2, 4).clone()
    validation_ids = torch.argsort(
        geometry.centered_logits(validation_router),
        dim=-1,
        descending=True,
        stable=True,
    )[..., :3]
    return {
        "geometry": geometry,
        "train_pre_attention_states": train_pre_attention,
        "train_router_inputs": train_router,
        "train_request_ids": ["train-a", "train-b"],
        "validation_pre_attention_states": validation_pre_attention,
        "validation_router_inputs": validation_router,
        "validation_selected_ids": validation_ids,
        "validation_request_ids": ["validation-a", "validation-b"],
        "hidden_width": 4,
        "optimization_steps": 1,
        "k": 3,
    }


def test_two_layer_future_state_ceiling_is_request_safe_and_exact() -> None:
    report = two_layer_future_state_ceiling_probe(
        **_future_state_probe_fixture()
    )
    assert report["architecture"]["affine_layers"] == 2
    assert report["architecture"]["layer_specific"] is True
    assert report["request_separation"] == {
        "train_requests": 2,
        "validation_requests": 2,
        "overlap_requests": 0,
        "request_disjoint": True,
    }
    assert report["fitting"]["validation_labels_used_for_fit"] is False
    assert report["mean_h1_h4_request_macro_slot_recall_at_3"] == 1.0
    assert report["mean_h1_h4_request_macro_exact_set_agreement"] == 1.0


def test_two_layer_future_state_ceiling_rejects_request_overlap_and_test() -> None:
    overlap = _future_state_probe_fixture()
    overlap["validation_request_ids"] = ["train-a", "validation-b"]
    with pytest.raises(ValueError, match="requests overlap"):
        two_layer_future_state_ceiling_probe(**overlap)

    sealed = _future_state_probe_fixture()
    sealed["validation_split"] = "test"
    with pytest.raises(PermissionError, match="sealed"):
        two_layer_future_state_ceiling_probe(**sealed)


def _expand_bayes_sets(values: list[list[int]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64)[:, None, None].expand(
        len(values), 4, 1, 2
    ).clone()


def test_prefix_conditional_deterministic_bayes_ceiling_is_one() -> None:
    report = prefix_conditional_bayes_ceiling(
        [[10, 11], [10, 11], [20, 21]],
        _expand_bayes_sets([[0, 1], [1, 0], [2, 3]]),
        decoding_mode="deterministic",
        experts=5,
        k=2,
    )
    metric = "mean_h1_h4_prefix_macro_expected_marginal_optimal_slot_recall_at_2"
    assert report[metric] == 1.0
    assert report["coverage"]["eligible_prefix_layer_group_fraction"] == 1.0
    assert report["uncertainty"]["mean_inclusion_binary_entropy_nats"] == 0.0
    assert report["grouping_contract"]["additional_conditioning_fields"] == []


def test_prefix_conditional_stochastic_ceiling_and_repeat_coverage() -> None:
    report = prefix_conditional_bayes_ceiling(
        [[10], [10], [10], [10], [99]],
        _expand_bayes_sets([[0, 1], [0, 2], [0, 1], [0, 2], [3, 4]]),
        decoding_mode="stochastic",
        experts=5,
        k=2,
    )
    metric = "mean_h1_h4_prefix_macro_expected_marginal_optimal_slot_recall_at_2"
    assert report[metric] == pytest.approx(0.75)
    assert report["coverage"]["eligible_prefix_fraction"] == pytest.approx(0.5)
    assert report["coverage"]["eligible_prefix_layer_group_fraction"] == pytest.approx(
        0.5
    )
    assert report["coverage"]["eligible_observation_endpoint_fraction"] == pytest.approx(
        0.8
    )
    assert report["uncertainty"]["plugin_estimate_is_finite_sample_optimistic"] is True
    assert report["uncertainty"]["minimum_optimal_topk_boundary_gap"] == 0.0


def test_prefix_conditional_bayes_ceiling_fails_closed_on_mode_or_split() -> None:
    inconsistent = _expand_bayes_sets([[0, 1], [0, 2]])
    with pytest.raises(ValueError, match="deterministic decoding produced multiple"):
        prefix_conditional_bayes_ceiling(
            [[10], [10]],
            inconsistent,
            decoding_mode="deterministic",
            experts=5,
            k=2,
        )
    with pytest.raises(PermissionError, match="sealed"):
        prefix_conditional_bayes_ceiling(
            [[10], [10]],
            inconsistent,
            decoding_mode="stochastic",
            experts=5,
            k=2,
            split="test",
        )


def _tree_batch(*, branch: bool = True, accept_h3: bool = True) -> dict[str, object]:
    parents = torch.tensor([[-1, 0, 0 if branch else 1, 1]])
    horizon_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    horizon_mask[0, 0, 0] = True
    horizon_mask[0, 1, 1:3] = True
    horizon_mask[0, 2, 3] = True
    acceptance = torch.tensor([[True, True, False, accept_h3]])
    return {
        "metadata": {"request_id": ["r"]},
        "inputs": {
            "tree": {
                "mask": torch.ones(1, 4, dtype=torch.bool),
                "parent": parents,
                "horizon_mask": horizon_mask,
                "exact_root_match": torch.tensor([[True, False, False, False]]),
                "tensor_available": torch.ones(1, 4, 3, dtype=torch.bool),
            }
        },
        "targets": {
            "tree_acceptance": acceptance,
            "tree_acceptance_valid": torch.ones(1, 4, dtype=torch.bool),
        },
    }


def test_tree_probe_reports_branching_and_full_path_acceptance() -> None:
    adaptive = tree_availability_probe(_tree_batch())
    assert adaptive["branching_sample_fraction"] == 1.0
    assert adaptive["exact_h1_root_sample_fraction"] == 1.0
    assert adaptive["horizon_metrics"][2]["request_macro_path_acceptance"] == 1.0
    chain = tree_availability_probe(_tree_batch(branch=False, accept_h3=False))
    comparison = compare_tree_probes(adaptive, chain)
    assert comparison["mean_h3_h4_path_acceptance_delta"] > 0


def _overfit_report() -> dict[str, object]:
    return {
        "split": "train",
        "complete_requests": 2,
        "candidate_width": 64,
        "mean_h1_h4_request_macro_slot_recall_at_8": 0.985,
        "min_h1_h4_request_macro_slot_recall_at_8": 0.98,
        "h4_request_macro_slot_recall_at_8": 0.98,
        "mean_h1_h4_request_macro_exact_set_nll": 0.1,
        "mean_h1_h4_request_macro_candidate_coverage_at_64": 0.99,
        "h4_request_macro_candidate_coverage_at_64": 0.98,
        "mean_h1_h4_request_macro_router_kl": 0.01,
    }


def test_c64_overfit_hook_tracks_formal_best_and_gap() -> None:
    hook = C64OverfitHook(expected_requests=2, maximum_gap_to_coverage=0.01)
    hook(10, _overfit_report())
    result = hook.report()
    assert result["best_step"] == 10
    assert result["overfit_passed"] is True
    assert result["candidate_gate"]["passed"] is True
