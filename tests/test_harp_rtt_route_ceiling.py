from __future__ import annotations

import torch

from harp_rtt.route_ceiling import (
    factual_branch_topk,
    posterior_native_topk,
    slot_recall_at_k,
    stable_ranks,
    swap_cap_oracle_recall,
    true_expert_support_audit,
)


def test_factual_branch_uses_native_node_and_other_falls_back() -> None:
    nodes = torch.tensor(
        [[[[4, 5]], [[6, 7]], [[8, 9]]]], dtype=torch.long
    )
    anchor = torch.tensor([[[[0, 1]], [[2, 3]]]])
    factual = torch.tensor([[1, 3]])
    result = factual_branch_topk(nodes, factual, anchor)
    assert result.tolist() == [[[[6, 7]], [[2, 3]]]]


def test_posterior_native_topk_includes_other_anchor_mass() -> None:
    ids = torch.tensor([[[[2, 3]], [[4, 5]]]])
    captured = torch.tensor([[[0.6, 0.0]]])
    other = torch.tensor([[0.4]])
    mask = torch.tensor([[[True, False]]])
    anchor = torch.zeros(1, 1, 1, 8)
    anchor[..., 0] = 1.0
    anchor[..., 1] = 1.0
    top, mass = posterior_native_topk(
        ids, captured, other, mask, anchor, exact_k=2
    )
    assert top.tolist() == [[[[2, 3]]]]
    assert torch.isclose(mass.sum(), torch.tensor(2.0))


def test_swap_cap_oracle_is_monotone_and_exact() -> None:
    anchor = torch.tensor([[[0, 1, 2, 3]]])
    target = torch.tensor([[[0, 4, 5, 6]]])
    candidates = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]])
    values = swap_cap_oracle_recall(anchor, candidates, target, (0, 1, 2, 4))
    assert [float(values[key]) for key in (0, 1, 2, 4)] == [0.25, 0.5, 0.75, 1.0]


def test_stable_ranks_break_ties_by_expert_id() -> None:
    ranks = stable_ranks(torch.tensor([[1.0, 1.0, 0.5, 1.0]]))
    assert ranks.tolist() == [[0, 1, 3, 2]]


def test_true_expert_support_audit_reports_branch_and_fallback_ranks() -> None:
    anchor = torch.tensor([[[[4.0, 3.0, 2.0, 1.0]]]])
    targets = torch.tensor([[[[2, 3]]]])
    node_scores = torch.tensor(
        [[[[[0.0, 1.0, 4.0, 3.0], [3.0, 2.0, 1.0, 0.0]]]]]
    )
    node_marginals = torch.softmax(node_scores, dim=-1) * 2.0
    learned = torch.tensor([[[0.8, 0.2]]])
    mtp = torch.tensor([[[0.2, 0.8]]])
    mask = torch.tensor([[[True, True]]])
    factual = torch.tensor([[0]])
    audit = true_expert_support_audit(
        target_ids=targets,
        anchor_scores=anchor,
        node_scores=node_scores,
        node_marginals=node_marginals,
        learned_probabilities=learned,
        mtp_probabilities=mtp,
        branch_mask=mask,
        factual_branch_indices=factual,
    )
    assert audit.anchor_rank.tolist() == [[[[2, 3]]]]
    assert audit.best_branch_rank.tolist() == [[[[0, 1]]]]
    assert audit.factual_branch_rank.tolist() == [[[[0, 1]]]]
    assert slot_recall_at_k(torch.tensor([[[[2, 3]]]]), targets).item() == 1.0
