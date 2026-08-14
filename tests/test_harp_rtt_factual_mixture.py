from __future__ import annotations

import itertools

import torch

from harp_rtt.factual_mixture import exact_factual_mixture_nll


def _brute_log_probability(scores: torch.Tensor, truth: tuple[int, ...]) -> torch.Tensor:
    sets = list(itertools.combinations(range(scores.numel()), len(truth)))
    values = torch.stack([scores[list(indices)].sum() for indices in sets])
    return scores[list(truth)].sum() - torch.logsumexp(values, dim=0)


def test_exact_factual_mixture_matches_brute_force_and_has_gradients() -> None:
    branch = torch.tensor(
        [[[[[1.0, 0.0, -1.0, 0.5], [0.0, 1.0, 0.5, -1.0]]]]],
        requires_grad=True,
    )
    anchor = torch.tensor([[[[0.5, -0.5, 1.5, 0.0]]]], requires_grad=True)
    posterior = torch.tensor([[[0.25, 0.50, 0.25]]])
    mask = torch.tensor([[[True, True]]])
    truth = torch.tensor([[[[0, 2]]]])
    result = exact_factual_mixture_nll(
        branch, anchor, posterior, mask, truth, k=2
    )
    expected = -torch.logsumexp(
        torch.stack(
            [
                torch.log(torch.tensor(0.25)) + _brute_log_probability(branch[0, 0, 0, 0], (0, 2)),
                torch.log(torch.tensor(0.50)) + _brute_log_probability(branch[0, 0, 0, 1], (0, 2)),
                torch.log(torch.tensor(0.25)) + _brute_log_probability(anchor[0, 0, 0], (0, 2)),
            ]
        ),
        dim=0,
    )
    assert torch.allclose(result.loss, expected, atol=1e-6, rtol=1e-6)
    result.loss.backward()
    assert branch.grad is not None and torch.isfinite(branch.grad).all()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()


def test_exact_factual_mixture_masks_branches_and_normalizes_other() -> None:
    branch = torch.zeros(1, 1, 1, 2, 4)
    anchor = torch.tensor([[[[5.0, 4.0, 0.0, -1.0]]]])
    result = exact_factual_mixture_nll(
        branch,
        anchor,
        torch.tensor([[[0.9, 0.0, 0.1]]]),
        torch.tensor([[[False, False]]]),
        torch.tensor([[[[0, 1]]]]),
        k=2,
    )
    assert result.normalized_probabilities.tolist() == [[[[0.0, 0.0, 1.0]]]]
    assert torch.isfinite(result.loss)
