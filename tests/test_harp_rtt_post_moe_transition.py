from __future__ import annotations

import torch

from harp_rtt.content_transition import ContentTransitionConfig
from harp_rtt.post_moe_transition import LayerSpecificPostMoeProbe


def _probe() -> LayerSpecificPostMoeProbe:
    torch.manual_seed(29)
    config = ContentTransitionConfig(
        layers=4, experts=7, hidden_width=9, router_rank=3, exact_k=2,
        latent_width=8, effect_width=4, transition_width=16, dropout=0.0,
    )
    return LayerSpecificPostMoeProbe(
        config, torch.randn(4, 7, 3), torch.randn(4, 7),
        torch.ones(4, 3, dtype=torch.bool), state_rank=5,
    )


def test_post_moe_probe_geometry_content_gain_and_gradients() -> None:
    probe = _probe()
    states = torch.randn(2, 3, 4, 9)
    ids = torch.randint(0, 7, (2, 3, 4, 2))
    weights = torch.rand(2, 3, 4, 2)
    full = probe(states, ids, weights)
    ablated = probe(states, ids, weights, use_router_blind_content=False)
    assert full.predicted_queries.shape == (2, 3, 3, 3)
    assert full.predicted_scores.shape == (2, 3, 3, 7)
    assert not torch.equal(full.predicted_scores, ablated.predicted_scores)
    full.predicted_scores.square().mean().backward()
    assert probe.state_down.grad is not None
    assert probe.expert_effect.grad is not None


def test_production_post_moe_probe_stays_under_25m() -> None:
    config = ContentTransitionConfig()
    with torch.device("meta"):
        probe = LayerSpecificPostMoeProbe(
            config, torch.empty(40, 256, 255), torch.empty(40, 256),
            torch.ones(40, 255, dtype=torch.bool), state_rank=192,
        )
    assert sum(parameter.numel() for parameter in probe.parameters()) < 25_000_000
