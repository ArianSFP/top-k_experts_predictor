from __future__ import annotations

import torch

from harp_rtt.direct_route import DirectDeltaRouteTrajectory
from harp_rtt.route_dynamics import DeltaRouteConfig


def _inputs(config: DeltaRouteConfig) -> dict[str, torch.Tensor]:
    batch = 2
    raw = torch.randn(batch, config.nodes, config.raw_width)
    return {
        "parent_queries": torch.randn(
            batch, config.horizons, config.layers, config.nodes,
            config.router_rank,
        ),
        "parent_scores": torch.randn(
            batch, config.horizons, config.layers, config.nodes,
            config.experts,
        ),
        "context_states": torch.randn(
            batch, config.horizons, config.layers, config.latent_width,
        ),
        "fused": raw,
        "post_ffn": torch.randn_like(raw),
        "router_input": torch.randn_like(raw),
        "vocabulary": torch.randn_like(raw),
        "token": torch.randn_like(raw),
        "router_logits": torch.randn(batch, config.nodes, config.experts),
        "metadata": torch.randn(batch, config.nodes, config.metadata_width),
        "parents": torch.tensor([[-1, 0, 1], [-1, 0, 0]]),
        "available": torch.ones(batch, config.nodes, dtype=torch.bool),
    }


def test_direct_route_is_parent_exact_at_zero_and_then_learns() -> None:
    torch.manual_seed(31)
    config = DeltaRouteConfig(
        experts=7, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=9, metadata_width=5, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    model = DirectDeltaRouteTrajectory(
        config, torch.randn(4, 7, 3), torch.randn(4, 7),
        torch.ones(4, 3, dtype=torch.bool),
    )
    inputs = _inputs(config)
    output = model(**inputs)
    assert torch.equal(output.queries, inputs["parent_queries"].float())
    assert torch.equal(output.scores, inputs["parent_scores"].float())
    output.scores.square().mean().backward()
    assert model.query_output.weight.grad is not None
    assert model.free_basis.grad is not None


def test_direct_route_masks_unavailable_nodes() -> None:
    config = DeltaRouteConfig(
        experts=7, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=9, metadata_width=5, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    model = DirectDeltaRouteTrajectory(
        config, torch.randn(4, 7, 3), torch.randn(4, 7),
        torch.ones(4, 3, dtype=torch.bool),
    )
    inputs = _inputs(config)
    inputs["available"][:, -1] = False
    first = model(**inputs)
    inputs["fused"][:, -1].fill_(1e6)
    second = model(**inputs)
    assert torch.equal(first.scores[:, :, :, -1], second.scores[:, :, :, -1])
