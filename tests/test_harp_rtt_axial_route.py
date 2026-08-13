from __future__ import annotations

import torch

from harp_rtt.axial_route import AxialRouteTrajectory
from harp_rtt.route_dynamics import DeltaRouteConfig
from tests.test_harp_rtt_direct_route import _inputs


def _model(config: DeltaRouteConfig) -> AxialRouteTrajectory:
    return AxialRouteTrajectory(
        config, torch.randn(config.layers, config.raw_width, config.router_rank),
        torch.randn(config.layers, config.experts, config.router_rank),
        torch.randn(config.layers, config.experts),
        torch.ones(config.layers, config.router_rank, dtype=torch.bool),
        raw_rank=3, visible_rank=3, route_width=4, axial_blocks=1,
    )


def test_axial_route_is_parent_exact_and_has_output_gradients() -> None:
    torch.manual_seed(41)
    config = DeltaRouteConfig(
        experts=7, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=9, metadata_width=5, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    model = _model(config)
    inputs = _inputs(config)
    output = model(**inputs)
    assert torch.equal(output.queries, inputs["parent_queries"].float())
    assert torch.equal(output.scores, inputs["parent_scores"].float())
    output.scores.square().mean().backward()
    assert model.query_output.weight.grad is not None
    assert model.free_basis.grad is not None


def test_axial_route_has_cross_layer_dependency_after_opening_gate() -> None:
    torch.manual_seed(43)
    config = DeltaRouteConfig(
        experts=7, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=9, metadata_width=5, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    model = _model(config).eval()
    torch.nn.init.normal_(model.query_output.weight, std=0.1)
    inputs = _inputs(config)
    first = model(**inputs).scores
    inputs["parent_queries"][:, :, 0].add_(3.0)
    inputs["parent_scores"][:, :, 0].add_(3.0)
    second = model(**inputs).scores
    assert not torch.equal(first[:, :, -1], second[:, :, -1])


def test_axial_route_masks_unavailable_node() -> None:
    torch.manual_seed(47)
    config = DeltaRouteConfig(
        experts=7, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=9, metadata_width=5, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    model = _model(config).eval()
    torch.nn.init.normal_(model.query_output.weight, std=0.1)
    inputs = _inputs(config)
    inputs["available"][:, -1] = False
    first = model(**inputs).scores
    inputs["fused"][:, -1].fill_(1e6)
    second = model(**inputs).scores
    assert torch.equal(first[:, :, :, -1], second[:, :, :, -1])


def test_production_axial_route_stays_under_25m() -> None:
    config = DeltaRouteConfig(attention_heads=8)
    with torch.device("meta"):
        model = AxialRouteTrajectory(
            config, torch.empty(40, 2048, 255),
            torch.empty(40, 256, 255), torch.empty(40, 256),
            torch.ones(40, 255, dtype=torch.bool),
            raw_rank=32, visible_rank=32, route_width=64, axial_blocks=2,
        )
    assert sum(parameter.numel() for parameter in model.parameters()) < 25_000_000
