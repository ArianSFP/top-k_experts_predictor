from __future__ import annotations

import pytest
import torch

from harp_rtt.b31 import (
    anchor_spine_prefix_matches,
    complete_other_mass,
    evaluate_factorial,
    evaluate_selected_factorial,
    exact_root_greedy_spine_indices,
    legacy_b3_divergence_depths,
    quota_candidate_union,
    selected_set_inclusion_mass,
    selected_ids_inclusion_mass,
    slot_coverage_at_k,
)


def test_greedy_spine_ignores_nonzero_rank_of_exact_h1_root() -> None:
    tree = {
        "mask": torch.tensor([[True, True, True, True, True, True]]),
        "depth": torch.tensor([[1, 2, 2, 3, 4, 3]]),
        "parent": torch.tensor([[-1, 0, 0, 1, 3, 2]]),
        # The forced exact H1 token was not MTP top-1.  Greedy branching for
        # this experiment begins below that observed root.
        "child_ranks": torch.tensor([[7, 0, 1, 0, 0, 0]]),
        "exact_committed_h1_root": torch.tensor(
            [[True, False, False, False, False, False]]
        ),
    }
    assert exact_root_greedy_spine_indices(tree).tolist() == [[0, 1, 3, 4]]


def test_anchor_spine_prefix_match_continues_through_tree_eos() -> None:
    spine = torch.zeros(1, 6, 32, dtype=torch.uint8)
    for depth in range(6):
        spine[:, depth] = depth + 1
    factual = spine[:, :4].clone()
    factual[:, 2] = 99
    factual[:, 3] = 100
    assert anchor_spine_prefix_matches(spine, factual).tolist() == [
        [True, True, False, False]
    ]


def test_legacy_b3_compatibility_reconstructs_root_rank_divergence() -> None:
    tree = {
        "mask": torch.tensor([[True, True, True, True]]),
        "depth": torch.tensor([[1, 2, 3, 4]]),
        "parent": torch.tensor([[-1, 0, 1, 2]]),
        "child_ranks": torch.tensor([[7, 0, 1, 0]]),
    }
    assert legacy_b3_divergence_depths(tree).tolist() == [[1, 1, 1, 1]]


def test_selected_set_mass_uses_membership_not_router_softmax() -> None:
    scores = torch.full((1, 1, 1, 2, 6), -10.0)
    scores[..., 0, :2] = torch.tensor([5.0, 4.0])
    scores[..., 1, 2:4] = torch.tensor([5.0, 4.0])
    weights = torch.tensor([[[0.75, 0.25]]])
    mask = torch.ones_like(weights, dtype=torch.bool)
    mass = selected_set_inclusion_mass(scores, weights, mask, exact_k=2)
    assert mass.shape == (1, 1, 1, 6)
    assert mass.sum().item() == pytest.approx(2.0)
    assert mass[0, 0, 0, :4].tolist() == pytest.approx([0.75, 0.75, 0.25, 0.25])


def test_other_mass_uses_exact_k_anchor_marginals() -> None:
    captured = torch.zeros(1, 2, 1, 4)
    anchor = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]], [[0.5, 0.5, 0.5, 0.5]]]])
    result = complete_other_mass(captured, torch.tensor([[0.5, 0.25]]), anchor)
    assert result[0, 0, 0].tolist() == pytest.approx([0.5, 0.5, 0.0, 0.0])
    assert result[0, 1, 0].sum().item() == pytest.approx(0.5)


def test_absolute_captured_mass_and_other_preserve_exact_cardinality() -> None:
    scores = torch.zeros(1, 1, 1, 2, 6)
    scores[..., 0, :2] = 2.0
    scores[..., 1, 2:4] = 2.0
    weights = torch.tensor([[[0.4, 0.2]]])
    mask = torch.ones_like(weights, dtype=torch.bool)
    captured = selected_set_inclusion_mass(
        scores, weights, mask, exact_k=2, normalize=False
    )
    anchor = torch.tensor([[[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]]])
    completed = complete_other_mass(captured, torch.tensor([[0.4]]), anchor)
    assert captured.sum().item() == pytest.approx(1.2)
    assert completed.sum().item() == pytest.approx(2.0)


def test_absolute_captured_mass_rejects_probability_above_one() -> None:
    scores = torch.zeros(1, 1, 1, 2, 6)
    weights = torch.tensor([[[0.8, 0.4]]])
    with pytest.raises(ValueError, match="exceeds one"):
        selected_set_inclusion_mass(
            scores, weights, torch.ones_like(weights, dtype=torch.bool),
            exact_k=2, normalize=False,
        )


def test_precomputed_selected_ids_match_dense_selected_set_mass() -> None:
    scores = torch.randn(1, 2, 1, 3, 10)
    weights = torch.tensor([[[0.2, 0.3, 0.1], [0.1, 0.2, 0.4]]])
    mask = torch.ones_like(weights, dtype=torch.bool)
    ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :2]
    dense = selected_set_inclusion_mass(
        scores, weights, mask, exact_k=2, normalize=False
    )
    compact = selected_ids_inclusion_mass(
        ids, weights, mask, experts=10, normalize=False
    )
    assert torch.equal(dense, compact)


def test_zero_branch_mass_reproduces_anchor_top64_exactly() -> None:
    anchor = torch.arange(80, dtype=torch.float32).reshape(1, 1, 1, 80)
    result = quota_candidate_union(
        anchor, torch.zeros_like(anchor), anchor_quota=40, width=64
    )
    expected = torch.argsort(anchor, dim=-1, descending=True, stable=True)[..., :64]
    assert torch.equal(result.expert_ids, expected)
    assert result.dense_mask.sum().item() == 64


def test_quota_union_uses_only_positive_branch_evidence_then_anchor_fallback() -> None:
    anchor = torch.arange(10, dtype=torch.float32).reshape(1, 1, 1, 10)
    branch = torch.zeros_like(anchor)
    branch[..., 0] = 1.0
    branch[..., 1] = 0.5
    result = quota_candidate_union(anchor, branch, anchor_quota=3, width=6)
    assert result.expert_ids.flatten().tolist() == [9, 8, 7, 0, 1, 6]
    assert len(set(result.expert_ids.flatten().tolist())) == 6


def test_factorial_reports_route_posterior_and_policy_axes() -> None:
    anchor = torch.zeros(1, 1, 1, 80)
    anchor[..., :8] = 2.0
    anchor_marginals = torch.zeros_like(anchor)
    anchor_marginals[..., :8] = 1.0
    branch = anchor[:, :, :, None].repeat(1, 1, 1, 2, 1)
    branch[..., 0, 8:16] = 4.0
    branch[..., 1, 16:24] = 4.0
    posterior = torch.tensor([[[0.6, 0.4]]])
    mask = torch.ones_like(posterior, dtype=torch.bool)
    target = torch.arange(8, 16).reshape(1, 1, 1, 8)
    report = evaluate_factorial(
        anchor_scores=anchor,
        anchor_marginals=anchor_marginals,
        target_ids=target,
        branch_scores={"semantic": branch},
        posteriors={"learned": (posterior, torch.zeros(1, 1))},
        branch_mask=mask,
        strata_masks={"only": torch.ones(1, 1, dtype=torch.bool)},
    )
    assert set(report) == {
        "semantic__learned__quota_48_16",
        "semantic__learned__quota_40_24",
        "semantic__learned__quota_32_32",
        "semantic__learned__global",
    }
    assert all(value["stratum_only_coverage"] == 1.0 for value in report.values())
    candidates = quota_candidate_union(
        anchor,
        selected_set_inclusion_mass(branch, posterior, mask),
        anchor_quota=40,
        width=64,
    )
    assert slot_coverage_at_k(candidates.expert_ids, target).item() == 1.0
