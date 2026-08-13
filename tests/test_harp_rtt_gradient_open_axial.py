from __future__ import annotations

import torch

from harp_rtt.gradient_open_axial import GradientOpenAxialRouteTrajectory
from harp_rtt.route_dynamics import DeltaRouteConfig
from tests.test_harp_rtt_direct_route import _inputs


def test_gradient_open_axial_is_exact_but_encoder_receives_first_step_gradient() -> None:
    torch.manual_seed(53)
    config = DeltaRouteConfig(
        experts=7, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=9, metadata_width=5, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    model = GradientOpenAxialRouteTrajectory(
        config, torch.randn(4, 9, 3), torch.randn(4, 7, 3),
        torch.randn(4, 7), torch.ones(4, 3, dtype=torch.bool),
        raw_rank=3, visible_rank=3, route_width=4, axial_blocks=1,
    )
    inputs = _inputs(config)
    output = model(**inputs)
    assert torch.equal(output.queries, inputs["parent_queries"].float())
    assert torch.equal(output.scores, inputs["parent_scores"].float())
    output.scores.square().mean().backward()
    assert model.channels.raw_down.grad is not None
    assert float(model.channels.raw_down.grad.abs().sum()) > 0.0
    assert model.visible_hidden_down.grad is not None
    assert float(model.visible_hidden_down.grad.abs().sum()) > 0.0
    axial_parameter = next(model.axial.parameters())
    assert axial_parameter.grad is not None
    assert float(axial_parameter.grad.abs().sum()) > 0.0


def test_gradient_open_axial_changes_after_one_update() -> None:
    torch.manual_seed(59)
    config = DeltaRouteConfig(
        experts=7, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=9, metadata_width=5, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    model = GradientOpenAxialRouteTrajectory(
        config, torch.randn(4, 9, 3), torch.randn(4, 7, 3),
        torch.randn(4, 7), torch.ones(4, 3, dtype=torch.bool),
        raw_rank=3, visible_rank=3, route_width=4, axial_blocks=1,
    )
    inputs = _inputs(config)
    before = model(**inputs).scores.detach().clone()
    loss = model(**inputs).scores.square().mean()
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=1e-3).step()
    after = model(**inputs).scores.detach()
    assert not torch.equal(before, after)
