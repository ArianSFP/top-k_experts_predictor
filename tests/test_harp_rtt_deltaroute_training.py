from __future__ import annotations

import pytest
import torch
from torch import nn

from harp_rtt.deltaroute_training import (
    configure_deltaroute_stage,
    counterfactual_trajectory_loss,
    depth_balanced_node_weights,
    factual_alignment_loss,
    joint_factual_mixture_loss,
    sampled_teacher_force_mask,
    teacher_forcing_probability,
)
from harp_rtt.factual_branch_attention import FactualAlignmentOutput


def _alignment(scores: torch.Tensor) -> FactualAlignmentOutput:
    candidates = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :4]
    return FactualAlignmentOutput(
        scores=scores,
        marginals=torch.sigmoid(scores),
        correction=torch.zeros_like(scores),
        expert_branch_weights=None,
        coherence_kl=scores.sum() * 0.0,
        candidate_ids=candidates,
    )


def test_alignment_loss_excludes_h1_and_promotes_missing_true() -> None:
    scores = torch.zeros(1, 4, 2, 6, requires_grad=True)
    anchor = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).reshape(1, 1, 1, 6)
    anchor = anchor.expand_as(scores)
    labels = torch.tensor([0, 1]).reshape(1, 1, 1, 2).expand(1, 4, 2, 2).clone()
    labels[:, 1:, :, 1] = 5
    targets = {
        "future_selected_ids": labels,
        "future_available": torch.ones(1, 4, dtype=torch.bool),
    }
    first = factual_alignment_loss(_alignment(scores), anchor, targets)
    first.total.backward()
    assert scores.grad is not None
    assert torch.equal(scores.grad[:, 0], torch.zeros_like(scores.grad[:, 0]))
    assert float(scores.grad[:, 1:, :, 5].mean()) < 0.0


def test_stage_ownership_freezes_parent_and_rejects_missing_subsystem() -> None:
    parent = nn.Linear(2, 2)
    aligner = nn.Linear(2, 2)
    trajectory = nn.Linear(2, 2)
    ownership = configure_deltaroute_stage(
        parent, aligner, trajectory, "align_m0"
    )
    assert all(not parameter.requires_grad for parameter in parent.parameters())
    assert all(parameter.requires_grad for parameter in aligner.parameters())
    assert all(not parameter.requires_grad for parameter in trajectory.parameters())
    assert ownership.trainable_parameters == sum(p.numel() for p in aligner.parameters())
    with pytest.raises(ValueError, match="requires trajectory"):
        configure_deltaroute_stage(parent, aligner, None, "transition_r0")


def test_teacher_forcing_schedule_has_closed_loop_tail() -> None:
    assert teacher_forcing_probability(0, 100) == 1.0
    assert teacher_forcing_probability(60, 100) == 0.0
    assert teacher_forcing_probability(100, 100) == 0.0
    mask = sampled_teacher_force_mask(
        (2, 3), 4, 1.0, device=torch.device("cpu")
    )
    assert mask.shape == (2, 3, 3) and bool(mask.all())


def test_counterfactual_loss_uses_router_induced_query_error() -> None:
    torch.manual_seed(3)
    queries = torch.randn(2, 3, 4, 3)
    keys = torch.randn(4, 7, 3)
    logits = torch.einsum("bnlr,ler->bnle", queries, keys)
    ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :2]
    valid = torch.ones(2, 3, 4, dtype=torch.bool)
    exact = counterfactual_trajectory_loss(
        queries, logits, queries, logits, ids, valid, keys,
    )
    assert float(exact.components["induced_logit_huber"]) == 0.0
    shifted = counterfactual_trajectory_loss(
        queries + 0.5, logits, queries, logits, ids, valid, keys,
    )
    assert float(shifted.components["induced_logit_huber"]) > 0.0


def test_counterfactual_loss_neutralizes_masked_sentinel_labels() -> None:
    torch.manual_seed(13)
    queries = torch.randn(1, 2, 3, 4, requires_grad=True)
    keys = torch.randn(3, 7, 4)
    logits = torch.einsum("bnlr,ler->bnle", queries, keys)
    ids = torch.argsort(logits.detach(), dim=-1, descending=True, stable=True)[..., :2]
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    valid[:, 1] = False
    ids[:, 1] = 65535
    target_queries = queries.detach().clone()
    target_logits = logits.detach().clone()
    target_queries[:, 1] = torch.nan
    target_logits[:, 1] = torch.nan
    loss = counterfactual_trajectory_loss(
        queries, logits, target_queries, target_logits, ids, valid, keys,
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()


def test_joint_factual_mixture_backpropagates_to_branch_scores() -> None:
    branch = torch.randn(1, 4, 2, 2, 6, requires_grad=True)
    anchor = torch.randn(1, 4, 2, 6)
    posterior = torch.tensor([[[0.5, 0.25, 0.25]]]).expand(1, 4, 3).clone()
    mask = torch.ones(1, 4, 2, dtype=torch.bool)
    labels = torch.tensor([0, 1]).reshape(1, 1, 1, 2).expand(1, 4, 2, 2).clone()
    targets = {
        "future_selected_ids": labels,
        "future_available": torch.ones(1, 4, dtype=torch.bool),
    }
    loss = joint_factual_mixture_loss(
        branch_scores=branch,
        anchor_scores=anchor,
        branch_probabilities=posterior,
        branch_mask=mask,
        targets=targets,
    )
    loss.total.backward()
    assert branch.grad is not None
    assert torch.equal(branch.grad[:, 0], torch.zeros_like(branch.grad[:, 0]))
    assert float(branch.grad[:, 1:].abs().sum()) > 0.0


def test_depth_balancing_gives_each_present_depth_equal_mass() -> None:
    depth = torch.tensor([[2, 2, 3, 4, 4, 4]])
    mask = torch.ones_like(depth, dtype=torch.bool)
    weights = depth_balanced_node_weights(depth, mask)
    assert weights.sum() == pytest.approx(1.0)
    for value in (2, 3, 4):
        assert weights[depth == value].sum() == pytest.approx(1 / 3)
