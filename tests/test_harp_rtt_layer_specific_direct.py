from __future__ import annotations

import torch

from harp_rtt.layer_specific_direct import (
    LayerSpecificDirectDeltaRouteTrajectory,
)
from harp_rtt.route_dynamics import DeltaRouteConfig
from tests.test_harp_rtt_direct_route import _inputs


def test_layer_specific_direct_parent_exact_and_gradients() -> None:
    torch.manual_seed(37)
    config = DeltaRouteConfig(
        experts=7, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=9, metadata_width=5, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    model = LayerSpecificDirectDeltaRouteTrajectory(
        config, torch.randn(4, 7, 3), torch.randn(4, 7),
        torch.ones(4, 3, dtype=torch.bool), raw_rank=3,
    )
    inputs = _inputs(config)
    output = model(**inputs)
    assert torch.equal(output.queries, inputs["parent_queries"].float())
    assert torch.equal(output.scores, inputs["parent_scores"].float())
    output.scores.square().mean().backward()
    assert model.query_output.weight.grad is not None
    assert model.free_basis.grad is not None


def test_production_layer_specific_direct_stays_under_25m() -> None:
    config = DeltaRouteConfig()
    with torch.device("meta"):
        model = LayerSpecificDirectDeltaRouteTrajectory(
            config, torch.empty(40, 256, 255), torch.empty(40, 256),
            torch.ones(40, 255, dtype=torch.bool), raw_rank=32,
        )
    assert sum(parameter.numel() for parameter in model.parameters()) < 25_000_000
