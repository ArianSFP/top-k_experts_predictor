from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from harp_rtt.b15 import (
    add_other_anchor_mass,
    candidate_union,
    factual_branch_candidates,
    global_candidates,
    h1_root_supervision_loss,
    required_b15_free_bytes,
    select_path_budget,
    selected_set_inclusion_mass,
)
from harp_rtt.counterfactual import select_counterfactual_paths
from harp_rtt.node_counterfactual import (
    empty_node_counterfactual_tensors,
    validate_node_counterfactual_tensors,
)


@dataclass(frozen=True)
class Node:
    local_index: int
    parent_local_index: int | None
    depth: int
    token_id: int
    token_rank_under_parent: int
    token_path_ids: tuple[int, ...]
    token_path_log_probabilities: tuple[float, ...]
    path_log_probability: float


def tree() -> list[Node]:
    rows = [
        (None, 1, 10, 0, (10,), (0.0,), 0.0),
        (0, 2, 20, 0, (10, 20), (0.0, -0.1), -0.1),
        (0, 2, 21, 1, (10, 21), (0.0, -0.2), -0.2),
        (1, 3, 30, 0, (10, 20, 30), (0.0, -0.1, -0.1), -0.2),
        (1, 3, 31, 1, (10, 20, 31), (0.0, -0.1, -0.3), -0.4),
        (2, 3, 32, 0, (10, 21, 32), (0.0, -0.2, -0.2), -0.4),
        (3, 4, 40, 0, (10, 20, 30, 40), (0.0, -0.1, -0.1, -0.1), -0.3),
        (3, 4, 41, 1, (10, 20, 30, 41), (0.0, -0.1, -0.1, -0.4), -0.6),
        (4, 4, 42, 0, (10, 20, 31, 42), (0.0, -0.1, -0.3, -0.1), -0.5),
        (5, 4, 43, 0, (10, 21, 32, 43), (0.0, -0.2, -0.2, -0.1), -0.5),
    ]
    return [Node(index, *row) for index, row in enumerate(rows)]


def test_budget_four_is_exactly_backward_compatible() -> None:
    nodes = tree()
    legacy = select_counterfactual_paths(nodes)
    selected = select_path_budget(nodes, 4)
    assert selected.endpoint_local_indices == tuple(
        path.endpoint_local_index for path in legacy if path is not None
    )
    assert selected.realized_budget == sum(path is not None for path in legacy)


def test_budget_eight_causally_backfills_only_informative_endpoints() -> None:
    selected = select_path_budget(tree(), 8)
    assert selected.requested_budget == 8
    assert selected.realized_budget <= 8
    assert selected.realized_budget == len(set(selected.endpoint_local_indices))
    assert selected.unique_node_count == sum(selected.node_mask)
    assert selected.node_mask[0]
    assert all(selected.node_mask[index] for index in selected.endpoint_local_indices)
    # Every endpoint after the first contributed a node not contained by prior paths.
    observed: set[int] = set()
    nodes = tree()
    for endpoint in selected.endpoint_local_indices:
        path = set()
        current: int | None = endpoint
        while current is not None:
            path.add(current)
            current = nodes[current].parent_local_index
        assert any(nodes[index].depth >= 2 and index not in observed for index in path)
        observed.update(path)


def test_path_budgets_are_explicitly_nested() -> None:
    nodes = tree()
    four = select_path_budget(nodes, 4)
    eight = select_path_budget(nodes, 8)
    sixteen = select_path_budget(nodes, 16)
    all_nodes = select_path_budget(nodes, "all")
    assert set(four.endpoint_local_indices) <= set(eight.endpoint_local_indices)
    assert set(eight.endpoint_local_indices) <= set(sixteen.endpoint_local_indices)
    assert set(sixteen.endpoint_local_indices) <= set(all_nodes.endpoint_local_indices)
    assert all(not left or right for left, right in zip(four.node_mask, eight.node_mask))
    assert all(not left or right for left, right in zip(eight.node_mask, sixteen.node_mask))
    assert all(not left or right for left, right in zip(sixteen.node_mask, all_nodes.node_mask))


def test_selected_set_inclusion_mass_is_exact_overlap_objective() -> None:
    ids = torch.tensor([[[0, 1]], [[1, 2]]])
    probability = torch.tensor([0.3, 0.4])
    valid = torch.ones(2, 1, dtype=torch.bool)
    mass, captured = selected_set_inclusion_mass(
        ids, probability, valid, experts=4
    )
    assert torch.allclose(mass[0], torch.tensor([0.3, 0.7, 0.4, 0.0]))
    assert torch.allclose(captured, torch.tensor([0.7]))
    assert torch.allclose(mass.sum(-1), 2 * captured)
    # Top two is the exact maximizer of expected selected-slot overlap.
    assert global_candidates(mass, width=2).tolist() == [[1, 2]]


def test_other_mass_uses_exact_k_anchor_marginals() -> None:
    anchor = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    branch = torch.tensor([[0.3, 0.7, 0.4, 0.0]])
    full, other, error = add_other_anchor_mass(
        branch, torch.tensor([0.7]), anchor, exact_k=2
    )
    assert torch.allclose(other, torch.tensor([0.3]))
    assert torch.allclose(full.sum(-1), torch.tensor([2.0]), atol=1e-5)
    assert torch.isfinite(error).all()


def test_candidate_quota_preserves_anchor_and_fills_distinct_branch_ids() -> None:
    anchor = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0]])
    branch = torch.tensor([[0.0, 0.1, 0.2, 4.0, 3.0]])
    candidates = candidate_union(anchor, branch, anchor_quota=2, width=4)
    assert candidates.tolist() == [[0, 1, 3, 4]]


def test_candidate_quota_all_zero_branch_exactly_reproduces_anchor() -> None:
    anchor = torch.tensor([[4.0, 9.0, 7.0, 8.0, 6.0, 5.0]])
    result = candidate_union(
        anchor, torch.zeros_like(anchor), anchor_quota=2, width=5
    )
    assert result.tolist() == [[1, 3, 2, 4, 5]]


def test_candidate_quota_falls_back_after_fewer_positive_branch_experts() -> None:
    anchor = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]])
    branch = torch.tensor([[0.3, 0.0, 0.0, 0.0, 0.4, 0.0]])
    result = candidate_union(anchor, branch, anchor_quota=2, width=5)
    # Expert 0 overlaps the anchor quota, expert 4 is the only new positive,
    # and the remaining positions fall back to anchor order.
    assert result.tolist() == [[0, 1, 4, 2, 3]]


def test_candidate_quota_positive_ties_are_stable_and_unique() -> None:
    anchor = torch.arange(8, dtype=torch.float32).flip(0).unsqueeze(0)
    branch = torch.tensor(
        [[0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]]
    )
    result = candidate_union(anchor, branch, anchor_quota=2, width=6)
    assert result.tolist() == [[0, 1, 2, 3, 4, 5]]
    assert len(set(result[0].tolist())) == 6


def test_b15_storage_preflight_uses_absolute_and_estimated_floors() -> None:
    gib = 1 << 30
    assert required_b15_free_bytes(0) == 100 * gib
    assert required_b15_free_bytes(80 * gib) == 100 * gib
    assert required_b15_free_bytes(100 * gib) == 125 * gib
    with pytest.raises(ValueError, match="non-negative"):
        required_b15_free_bytes(-1)


def test_factual_ceiling_fills_from_anchor_score_order_without_duplicates() -> None:
    anchor = torch.tensor([[0.0, 9.0, 7.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    factual = torch.tensor([[3, 7, 0, 9, 8, 6, 5, 4]])
    result = factual_branch_candidates(anchor, factual, width=10)
    assert result.tolist() == [[3, 7, 0, 9, 8, 6, 5, 4, 1, 2]]


def test_node_contract_is_parent_before_child_and_h1_masked() -> None:
    tensors = empty_node_counterfactual_tensors(
        nodes=4, layers=2, rank=3, experts=12
    )
    tensors["node_mask"][:] = True
    tensors["node_local_indices"][:] = torch.arange(4)
    tensors["parent_local_indices"][:] = torch.tensor([-1, 0, 1, 1])
    tensors["depth"][:] = torch.tensor([1, 2, 3, 3])
    tensors["first_divergence_depth"][:] = torch.tensor([-1, -1, -1, 3])
    tensors["budget_node_masks"][:] = True
    tensors["budget_endpoint_masks"][:, 2] = True
    tensors["budget_realized"][:] = 1
    tensors["budget_category_counts"][:, 0] = 1
    tensors["source_edge_logp"][:] = torch.tensor([0.0, -0.1, -0.2, -0.3])
    tensors["source_path_logp"][:] = torch.tensor([0.0, -0.1, -0.3, -0.4])
    tensors["target_edge_logp"][1:] = torch.tensor([-0.15, -0.25, -0.35])
    tensors["target_path_logp"][1:] = torch.tensor([-0.15, -0.4, -0.5])
    tensors["target_next_token_ids"][:] = torch.tensor([20, 30, 40, 41])
    tensors["target_next_token_logp"][:] = torch.tensor([-0.1, -0.2, -0.3, -0.4])
    tensors["target_next_token_valid"][:] = True
    tensors["valid"][1:] = True
    tensors["selected_ids"][1:] = torch.arange(8, dtype=torch.int32)
    validate_node_counterfactual_tensors(
        tensors, nodes=4, layers=2, rank=3, experts=12
    )
    tensors["valid"][0] = True
    with pytest.raises(ValueError, match="H1"):
        validate_node_counterfactual_tensors(
            tensors, nodes=4, layers=2, rank=3, experts=12
        )


def test_h1_contract_backpropagates_without_starting_an_optimizer() -> None:
    scores = torch.randn(2, 3, 12, requires_grad=True)
    queries = torch.randn(2, 3, 5, requires_grad=True)
    target_logits = torch.randn(2, 3, 12)
    target_ids = torch.argsort(
        target_logits, dim=-1, descending=True, stable=True
    )[..., :2]
    target_queries = torch.randn(2, 3, 5)
    result = h1_root_supervision_loss(
        scores,
        queries,
        target_ids,
        target_queries,
        target_logits,
        exact_k=2,
    )
    assert torch.isfinite(result.total)
    result.total.backward()
    assert scores.grad is not None and scores.grad.abs().sum() > 0
    assert queries.grad is not None and queries.grad.abs().sum() > 0
