from __future__ import annotations

import pytest
import torch

from harp_rtt.losses import (
    COMPONENT_NAMES,
    GradientAudit,
    HARPRTTLossConfig,
    HARPRTTLossSchedule,
    HARPRTTObjective,
    boundary_loss_per_endpoint,
    hard_negative_mask,
    horizon_balanced_reduce,
)
from harp_rtt.schema import HARPRTTDimensions


def dimensions() -> HARPRTTDimensions:
    return HARPRTTDimensions(
        layers=2,
        experts=12,
        primary_horizons=4,
        legacy_horizons=8,
        selected_experts=2,
        history_tokens=2,
        tree_nodes=4,
        candidates=6,
        trajectory_rounds=2,
    )


def synthetic_objective_inputs() -> tuple[
    dict[str, torch.Tensor], dict[str, object], list[torch.Tensor]
]:
    torch.manual_seed(31)
    dims = dimensions()
    batch, horizons, layers, experts = 2, 4, 2, 12
    nodes, rank = 4, 3
    active_scores = torch.randn(
        batch, horizons, layers, experts, requires_grad=True
    )
    teacher = torch.randn(batch, horizons, layers, experts)
    labels = torch.topk(teacher, dims.selected_experts, dim=-1).indices
    trajectory_scores = torch.randn(
        batch, 2, horizons, layers, experts, requires_grad=True
    )
    corrections = torch.randn(
        batch, 2, horizons, layers, experts, requires_grad=True
    )
    queries = torch.randn(
        batch, horizons, layers, nodes, rank, requires_grad=True
    )
    branch_logits = torch.tensor(
        [
            [[-1.0, 0.3, 0.7, 1.1]] * horizons,
            [[0.8, -0.5, 1.2, -0.9]] * horizons,
        ],
        requires_grad=True,
    )
    branch_weights = torch.softmax(branch_logits.detach(), dim=-1)
    branch_mask = torch.eye(horizons, nodes, dtype=torch.bool)[None].expand(
        batch, -1, -1
    )
    query_targets = torch.randn(batch, horizons, layers, rank)
    outputs = {
        "active_scores": active_scores,
        "trajectory_round_scores": trajectory_scores,
        "trajectory_corrections": corrections,
        "router_queries": queries,
        "branch_weights": branch_weights,
        "branch_posterior_logits": branch_logits,
        "branch_mask": branch_mask,
    }
    complete_batch: dict[str, object] = {
        "inputs": {"tree": {"horizon_mask": branch_mask}},
        "targets": {
            "future_selected_ids": labels,
            "future_centered_router_logits": teacher,
            "future_router_queries": query_targets,
            "future_available": torch.ones(
                batch, horizons, layers, dtype=torch.bool
            ),
            "tree_acceptance": torch.tensor(
                [[False, True, False, True], [True, False, True, False]]
            ),
            "tree_acceptance_valid": torch.ones(batch, nodes, dtype=torch.bool),
        },
    }
    leaves = [
        active_scores,
        trajectory_scores,
        corrections,
        queries,
        branch_logits,
    ]
    return outputs, complete_batch, leaves


def test_loss_schedule_reaches_formal_weights_and_tau_endpoints() -> None:
    schedule = HARPRTTLossSchedule(
        recall_start_step=100,
        recall_ramp_steps=200,
        temperature_anneal_steps=400,
    )
    before = schedule.at(99)
    start = schedule.at(100)
    ramped = schedule.at(300)
    finished = schedule.at(500)

    assert before.weights["recall"] == 0.0
    assert start.weights["recall"] == 0.0
    assert start.temperature == pytest.approx(1.0)
    assert ramped.weights["recall"] == pytest.approx(0.5)
    assert finished.temperature == pytest.approx(0.1)
    assert finished.weights == {
        "set": 1.0,
        "recall": 0.5,
        "boundary": 0.25,
        "router_kl": 0.15,
        "query": 0.10,
        "branch": 0.10,
        "trajectory": 0.05,
    }


def test_horizon_reduce_is_equal_h1_h4_plus_quarter_worst() -> None:
    values = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    valid = torch.ones_like(values, dtype=torch.bool)
    loss, horizons = horizon_balanced_reduce(values, valid)
    assert torch.equal(horizons, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert loss.item() == pytest.approx(2.5 + 0.25 * 4.0)

    valid[:, -1] = False
    loss, _ = horizon_balanced_reduce(values, valid)
    assert loss.item() == pytest.approx(2.0 + 0.25 * 3.0)


def test_boundary_mines_deduplicated_requested_rank_union() -> None:
    # Model ranks 6--12 are IDs 5--11.  Teacher ranks 9--12 are IDs 3--0.
    scores = torch.arange(12, 0, -1, dtype=torch.float32)[None]
    teacher = torch.arange(12, dtype=torch.float32)[None]
    labels = torch.tensor([[0, 5]])
    mask = hard_negative_mask(scores, teacher, labels)
    expected = torch.tensor(
        [[False, True, True, True, False, False, True, True, True, True, True, True]]
    )
    assert torch.equal(mask, expected)

    actual = boundary_loss_per_endpoint(scores, teacher, labels)
    false_lse = torch.logsumexp(scores[0, expected[0]], dim=0)
    selected = scores[0, labels[0]]
    soft_minimum = -torch.logsumexp(-selected, dim=0)
    wanted = torch.nn.functional.softplus(1.0 + false_lse - soft_minimum)
    assert actual.item() == pytest.approx(wanted.item())


def test_combined_objective_has_formal_components_and_gradients() -> None:
    outputs, batch, leaves = synthetic_objective_inputs()
    config = HARPRTTLossConfig(
        dimensions=dimensions(),
        schedule=HARPRTTLossSchedule(
            recall_ramp_steps=0,
            temperature_anneal_steps=0,
        ),
    )
    objective = HARPRTTObjective(config)
    result = objective(outputs, batch, step=1)

    assert set(result.components) == set(COMPONENT_NAMES)
    expected = sum(result.weighted_components.values())
    assert torch.allclose(result.total, expected)
    assert result.temperature == pytest.approx(0.1)
    assert result.scheduled_weights["recall"] == pytest.approx(0.5)
    assert all(value.shape == (4,) for value in result.horizon_components.values())
    assert torch.isfinite(result.total)

    result.total.backward()
    for leaf in leaves:
        assert leaf.grad is not None
        assert torch.isfinite(leaf.grad).all()
        assert leaf.grad.abs().sum() > 0


def test_acceptance_labels_change_only_branch_component() -> None:
    outputs, batch, _ = synthetic_objective_inputs()
    config = HARPRTTLossConfig(
        dimensions=dimensions(),
        schedule=HARPRTTLossSchedule(recall_ramp_steps=0),
    )
    objective = HARPRTTObjective(config)
    first = objective(outputs, batch, step=20_000)
    flipped = {
        **batch,
        "targets": {
            **batch["targets"],
            "tree_acceptance": ~batch["targets"]["tree_acceptance"],
        },
    }
    second = objective(outputs, flipped, step=20_000)

    for name in COMPONENT_NAMES:
        if name == "branch":
            assert not torch.allclose(first.components[name], second.components[name])
        else:
            assert torch.allclose(first.components[name], second.components[name])


def test_query_target_can_be_projected_through_frozen_router_basis() -> None:
    outputs, batch, _ = synthetic_objective_inputs()
    dims = dimensions()
    torch.manual_seed(4)
    router_inputs = torch.randn(2, 4, dims.layers, 5)
    basis = torch.randn(dims.layers, 5, 3)
    target = torch.einsum("bhld,ldr->bhlr", router_inputs, basis)
    outputs["router_queries"] = target[:, :, :, None].expand(-1, -1, -1, 4, -1)
    batch["targets"] = {
        **batch["targets"],
        "future_router_inputs": router_inputs,
    }
    del batch["targets"]["future_router_queries"]
    objective = HARPRTTObjective(
        HARPRTTLossConfig(dimensions=dims), router_input_basis=basis
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        projected = objective._target_queries(batch["targets"], rank=3)
        result = objective(outputs, batch, step=1)
    assert projected.dtype == torch.float32
    assert torch.equal(projected, target)
    assert result.components["query"].item() == pytest.approx(0.0, abs=1e-6)


def test_objective_rejects_any_trajectory_round_count_other_than_two() -> None:
    outputs, batch, _ = synthetic_objective_inputs()
    outputs["trajectory_round_scores"] = outputs["trajectory_round_scores"][:, :1]
    outputs["trajectory_corrections"] = outputs["trajectory_corrections"][:, :1]
    objective = HARPRTTObjective(HARPRTTLossConfig(dimensions=dimensions()))
    with pytest.raises(ValueError, match="R=2"):
        objective(outputs, batch, step=0)


def test_gradient_audit_requires_ten_consecutive_fifty_step_passes() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    audit = GradientAudit()
    assert audit.audit(
        step=49,
        primary_loss=parameter.square().sum(),
        auxiliary_losses={"recall": 0.1 * parameter.square().sum()},
        parameters=[parameter],
    ) is None

    result = None
    for index, step in enumerate(range(50, 501, 50), start=1):
        primary = parameter.square().sum()
        auxiliaries = {
            "recall": 0.10 * parameter.square().sum(),
            "boundary": 0.05 * parameter.square().sum(),
        }
        result = audit.audit(
            step=step,
            primary_loss=primary,
            auxiliary_losses=auxiliaries,
            parameters=[parameter],
        )
        assert result is not None
        assert result.passed
        assert result.consecutive_passes == index
        assert result.promotion_ready is (index == 10)
        assert parameter.grad is None
    assert result is not None and result.auxiliary_to_primary_ratio == pytest.approx(0.15)

    failure = audit.audit(
        step=550,
        primary_loss=parameter.square().sum(),
        auxiliary_losses={"recall": parameter.square().sum()},
        parameters=[parameter],
    )
    assert failure is not None
    assert not failure.passed
    assert failure.consecutive_passes == 0
    assert not failure.promotion_ready
    assert failure.recommended_auxiliary_scale == pytest.approx(0.5)


def test_gradient_audit_missed_interval_breaks_consecutive_window() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    audit = GradientAudit(required_consecutive_passes=2)
    first = audit.audit(
        step=50,
        primary_loss=parameter.square().sum(),
        auxiliary_losses={"recall": 0.1 * parameter.square().sum()},
        parameters=[parameter],
    )
    assert first is not None and first.consecutive_passes == 1
    after_gap = audit.audit(
        step=150,
        primary_loss=parameter.square().sum(),
        auxiliary_losses={"recall": 0.1 * parameter.square().sum()},
        parameters=[parameter],
    )
    assert after_gap is not None and after_gap.consecutive_passes == 1
    assert not after_gap.promotion_ready
