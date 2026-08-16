from __future__ import annotations

import pytest
import torch

from harp_rtt.resident_codebook import (
    fit_resident_proxy,
    validate_codebook_tables,
)


def test_fit_resident_proxy_recovers_stable_convex_pair():
    candidate_ids = torch.tensor([7, 3, 9])
    left = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    right = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    third = torch.full_like(left, 3.0)
    candidates = torch.stack([left, right, third], dim=1)
    target = 0.25 * left + 0.75 * right
    weights = torch.tensor([0.4, 0.6])
    top1 = fit_resident_proxy(
        target, candidates, weights, candidate_ids, proxies=1
    )
    top2 = fit_resident_proxy(
        target, candidates, weights, candidate_ids, proxies=2
    )
    assert top1.proxy_count == 1
    assert top2.proxy_count == 2
    assert top2.proxy_ids.tolist() == [3, 7]
    assert top2.coefficients.float().tolist() == pytest.approx([0.75, 0.25])
    assert top2.weighted_error < 1e-6
    assert top2.weighted_error < top1.weighted_error


def test_fit_resident_proxy_uses_lower_id_on_exact_tie():
    target = torch.ones(2, 3)
    candidates = torch.ones(2, 2, 3)
    result = fit_resident_proxy(
        target,
        candidates,
        torch.ones(2),
        torch.tensor([9, 4]),
        proxies=1,
    )
    assert result.proxy_ids.tolist() == [4, -1]


def test_validate_codebook_tables_rejects_external_proxy():
    ids = torch.tensor([[0, -1], [0, -1], [2, -1], [0, 2]])
    coefficients = torch.tensor([
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.5, 0.5],
    ])
    counts = torch.tensor([1, 1, 1, 2])
    validate_codebook_tables(torch.tensor([0, 2]), ids, coefficients, counts, experts=4)
    ids[1, 0] = 1
    with pytest.raises(ValueError, match="outside"):
        validate_codebook_tables(
            torch.tensor([0, 2]), ids, coefficients, counts, experts=4
        )
