from __future__ import annotations

import torch

from harp_rtt.cachetrajectory_c32_setrank import (
    EquivariantCacheTrajectory32,
    continuous_policy_score,
    parameter_and_mac_contract,
    ranking_loss,
    select_mean_constrained_policies,
)


def test_set_ranker_geometry_permutation_equivariance_and_contract() -> None:
    torch.manual_seed(4)
    model = EquivariantCacheTrajectory32()
    features = torch.randn(3, 32, 28)
    candidate_ids = torch.stack(
        tuple(torch.randperm(256)[:32] for _ in range(3))
    )
    horizons = torch.tensor([0, 1, 3])
    layers = torch.tensor([2, 11, 39])
    factual, survivor = model(features, candidate_ids, horizons, layers)
    permutation = torch.randperm(32)
    inverse = torch.argsort(permutation)
    permuted = model(
        features[:, permutation], candidate_ids[:, permutation], horizons, layers
    )
    assert factual.shape == survivor.shape == (3, 32)
    assert torch.allclose(factual, permuted[0][:, inverse], atol=1e-6, rtol=1e-6)
    assert torch.allclose(survivor, permuted[1][:, inverse], atol=1e-6, rtol=1e-6)
    contract = parameter_and_mac_contract()
    assert contract["linear_macs_per_cell"] == 42_240
    assert contract["parameters"] == 83_514
    assert contract["deployment_bf16_bytes"] == 167_028
    assert contract["linear_macs_per_cell"] <= 623_360
    assert contract["fixed_survivor_quota"] is False
    assert parameter_and_mac_contract(24)["linear_macs_per_cell"] == 31_680


def test_continuous_score_zero_scales_reproduce_base_and_loss_is_finite() -> None:
    base = torch.randn(5, 32)
    factual = torch.randn(5, 32, requires_grad=True)
    survivor = torch.randn(5, 32, requires_grad=True)
    current = torch.zeros(5, 32, dtype=torch.bool)
    current[:, :8] = True
    target = torch.zeros(5, 32, dtype=torch.bool)
    target[:, torch.arange(8)] = True
    score = continuous_policy_score(
        base,
        factual,
        survivor,
        current,
        residual_scale=0.0,
        survivor_scale=0.0,
        current_bias=0.0,
    )
    assert torch.equal(score, base)
    loss, pieces = ranking_loss(
        factual, survivor, target, current, base
    )
    assert torch.isfinite(loss)
    assert set(pieces) == {
        "factual_bce",
        "listwise",
        "factual_pairwise",
        "survivor_bce",
        "survivor_pairwise",
        "residual_anchor",
    }
    loss.backward()
    assert factual.grad is not None
    assert survivor.grad is not None



def test_bf16_forward_and_joint_mean_cache_calibration() -> None:
    torch.manual_seed(5)
    model = EquivariantCacheTrajectory32().to(torch.bfloat16)
    features = torch.randn(2, 32, 28)
    candidate_ids = torch.stack(tuple(torch.randperm(256)[:32] for _ in range(2)))
    factual, survivor = model(
        features,
        candidate_ids,
        torch.tensor([0, 3]),
        torch.tensor([1, 39]),
    )
    assert factual.dtype == survivor.dtype == torch.bfloat16
    assert torch.equal(factual, torch.zeros_like(factual))

    def option(
        factual_value: float,
        cache_value: float,
        residual: float,
    ) -> dict[str, float]:
        return {
            "factual": factual_value,
            "cache_set": cache_value,
            "residual_scale": residual,
            "survivor_scale": 0.0,
            "current_bias": 0.0,
        }

    selected = select_mean_constrained_policies(
        (
            (option(0.99, 0.90, 1.0), option(0.80, 0.96, 0.0)),
            (option(0.90, 1.00, 2.0), option(0.80, 0.96, 0.0)),
            (option(0.90, 0.95, 3.0),),
            (option(0.90, 0.95, 4.0),),
        )
    )
    assert selected["cache_set_mean"] >= 0.95
    assert any(
        row["cache_set"] < 0.95 for row in selected["per_horizon_metrics"]
    )
    assert selected["factual_mean"] > 0.875
