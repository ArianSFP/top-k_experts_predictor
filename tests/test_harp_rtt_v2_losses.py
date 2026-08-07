from __future__ import annotations

import torch

from harp_rtt.counterfactual import empty_counterfactual_tensors
from harp_rtt.v2_losses import (
    counterfactual_posterior_loss,
    counterfactual_semantic_loss,
    swap_loss,
    target_branch_distribution,
)


def synthetic_counterfactual() -> dict[str, torch.Tensor]:
    labels = empty_counterfactual_tensors(layers=2, rank=3, experts=12)
    labels["path_mask"][:2] = True
    labels["path_depths"][:2] = 4
    labels["node_local_indices"][0] = torch.tensor([0, 1, 2, 3])
    labels["node_local_indices"][1] = torch.tensor([0, 1, 4, 5])
    labels["target_path_logp"][0] = torch.tensor([-0.1, -0.2, -0.4, -0.7])
    labels["target_path_logp"][1] = torch.tensor([-0.1, -0.2, -0.6, -0.9])
    labels["valid"][:2, 1:] = True
    torch.manual_seed(8)
    labels["query_coordinates"][:2, 1:] = torch.randn(2, 3, 2, 3)
    labels["router_logits"][:2, 1:] = torch.randn(
        2, 3, 2, 12
    ).to(torch.bfloat16)
    ids = torch.argsort(
        labels["router_logits"][:2, 1:].float(),
        dim=-1,
        descending=True,
        stable=True,
    )[..., :2]
    labels["selected_ids"][:2, 1:, :, :2] = ids.to(torch.int32)
    # The loss below uses exact_k=2, so retain the contract's remaining six
    # slots as harmless valid expert IDs.
    labels["selected_ids"][:2, 1:, :, 2:] = torch.tensor(
        [2, 3, 4, 5, 6, 7], dtype=torch.int32
    )
    return {name: value.unsqueeze(0) for name, value in labels.items()}


def test_counterfactual_semantic_loss_reaches_queries_and_route_scores() -> None:
    labels = synthetic_counterfactual()
    torch.manual_seed(11)
    queries = torch.randn(1, 4, 2, 7, 3, requires_grad=True)
    geometry_scores = torch.randn(1, 4, 2, 7, 12, requires_grad=True)
    free_scores = torch.randn(1, 4, 2, 7, 12, requires_grad=True)
    # exact_k=2 consumes the leading two authoritative IDs.
    labels["selected_ids"] = labels["selected_ids"][..., :2]
    result = counterfactual_semantic_loss(
        {
            "router_queries": queries,
            "branch_semantic_scores": geometry_scores + free_scores,
            "branch_mask": torch.ones(1, 4, 7, dtype=torch.bool),
        },
        labels,
        exact_k=2,
    )
    assert result.active_path_depth_cells == 6
    assert torch.isfinite(result.total)
    result.total.backward()
    assert queries.grad is not None and queries.grad.abs().sum() > 0
    assert geometry_scores.grad is not None and geometry_scores.grad.abs().sum() > 0
    assert free_scores.grad is not None and free_scores.grad.abs().sum() > 0


def test_counterfactual_target_posterior_deduplicates_prefix_and_adds_other() -> None:
    labels = synthetic_counterfactual()
    target, valid = target_branch_distribution(labels, captured_nodes=6)
    assert target.shape == (1, 4, 7)
    # The two selected paths share node one at H2; it receives probability
    # exp(-.2) once, with all uncaptured mass assigned to OTHER.
    assert torch.allclose(target[0, 1, 1], torch.tensor(-0.2).exp())
    assert torch.allclose(target.sum(-1)[valid], torch.ones_like(target.sum(-1)[valid]))
    logits = torch.zeros_like(target, requires_grad=True)
    visibility = torch.ones_like(target, dtype=torch.bool)
    visibility[0, 1, 1] = False
    loss = counterfactual_posterior_loss(
        logits, labels, branch_mask=visibility
    )
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert logits.grad[0, 1, 1] == 0


def test_swap_loss_promotes_missing_truth_and_counts_outside_pool() -> None:
    base = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    final = base.clone().requires_grad_(True)
    truth = torch.tensor([[0, 4]])
    candidate = torch.tensor([[True, True, True, True, False]])
    result = swap_loss(
        final, base, truth, exact_k=2, candidate_mask=candidate
    )
    assert result.swap_pairs == 1
    assert result.outside_candidate_misses == 1
    result.loss.backward()
    assert final.grad is not None
    assert final.grad[0, 4] < 0
    assert final.grad[0, 1] > 0
