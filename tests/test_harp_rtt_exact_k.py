from __future__ import annotations

import itertools

import pytest
import torch

from harp_rtt.exact_k import (
    cardinality_project_marginals,
    exact_k_logz_marginals_fast,
    exact_set_nll,
    soft_cardinality_topk,
    soft_recall_loss,
    stable_topk,
    validate_exact_set_labels,
)


def _brute_force(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    subsets = list(itertools.combinations(range(scores.numel()), k))
    weights = torch.stack([scores[list(subset)].sum() for subset in subsets])
    probabilities = weights.softmax(0)
    marginals = torch.zeros_like(scores)
    for probability, subset in zip(probabilities, subsets, strict=True):
        marginals[list(subset)] += probability
    return torch.logsumexp(weights, 0), marginals


def test_exact_k_wrapper_matches_brute_force_and_repairs_mass() -> None:
    scores = torch.tensor([0.4, -0.7, 1.2, 0.1, 0.8, -1.5])
    expected_log_z, expected_marginals = _brute_force(scores, 3)
    log_z, raw_marginals = exact_k_logz_marginals_fast(scores, 3)
    marginals, pre_error = cardinality_project_marginals(raw_marginals, 3)
    assert torch.allclose(log_z, expected_log_z, atol=1e-6, rtol=1e-6)
    assert torch.allclose(marginals, expected_marginals, atol=1e-6, rtol=1e-6)
    assert torch.allclose(marginals.sum(), torch.tensor(3.0), atol=1e-7, rtol=0)
    assert pre_error < 2e-6


def test_exact_set_nll_rejects_malformed_active_labels() -> None:
    scores = torch.randn(2, 12)
    with pytest.raises(ValueError, match="distinct"):
        exact_set_nll(scores, torch.tensor([[0, 1, 1], [3, 4, 5]]), k=3)
    with pytest.raises(ValueError, match="out-of-range"):
        exact_set_nll(scores, torch.tensor([[0, 1, 12], [3, 4, 5]]), k=3)
    with pytest.raises(TypeError, match="integer"):
        exact_set_nll(scores, torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), k=3)
    with pytest.raises(ValueError, match="expected"):
        exact_set_nll(scores, torch.tensor([[0, 1], [3, 4]]), k=3)
    with pytest.raises(ValueError, match="non-negative"):
        exact_set_nll(
            scores,
            torch.tensor([[0, 1, 2], [3, 4, 5]]),
            valid=torch.tensor([1.0, -1.0]),
            k=3,
        )


def test_inactive_capture_sentinels_are_safely_ignored() -> None:
    scores = torch.randn(2, 10, requires_grad=True)
    labels = torch.tensor([[1, 3, 7], [-1, -1, -1]])
    valid = torch.tensor([True, False])
    safe, weights = validate_exact_set_labels(scores, labels, valid=valid, k=3)
    assert safe[1].tolist() == [0, 1, 2]
    assert weights.tolist() == [1.0, 0.0]
    loss = exact_set_nll(scores, labels, valid, k=3)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(scores.grad).all()
    assert torch.equal(scores.grad[1], torch.zeros(10))


@pytest.mark.parametrize("temperature", [1.0, 0.3, 0.1])
def test_soft_top8_has_fp32_exact_cardinality_and_finite_gradients(
    temperature: float,
) -> None:
    generator = torch.Generator().manual_seed(90)
    scores = (1000 * torch.randn(2, 3, 256, generator=generator)).to(torch.bfloat16)
    scores = scores.requires_grad_(True)
    memberships = soft_cardinality_topk(scores, temperature=temperature)
    assert memberships.dtype == torch.float32
    assert memberships.shape == scores.shape
    assert torch.isfinite(memberships).all()
    assert torch.all((memberships >= 0) & (memberships <= 1))
    assert torch.allclose(
        memberships.sum(-1),
        torch.full((2, 3), 8.0),
        atol=2e-5,
        rtol=0,
    )
    memberships[..., :8].sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_soft_top8_is_affine_invariant_after_standardization() -> None:
    scores = torch.randn(4, 64)
    first = soft_cardinality_topk(scores, temperature=0.4)
    second = soft_cardinality_topk(7.5 * scores + 123.0, temperature=0.4)
    assert torch.allclose(first, second, atol=2e-6, rtol=2e-6)


def test_soft_recall_rewards_true_experts_and_validates_sets() -> None:
    labels = torch.tensor([[0, 1, 2, 3]])
    poor = torch.tensor([[-2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0]])
    good = -poor
    assert soft_recall_loss(good, labels, k=4, temperature=0.1) < soft_recall_loss(
        poor, labels, k=4, temperature=0.1
    )


def test_stable_topk_breaks_ties_by_lower_expert_id() -> None:
    scores = torch.tensor([[1.0, 2.0, 2.0, 0.0, 2.0]])
    assert stable_topk(scores, 3).tolist() == [[1, 2, 4]]
