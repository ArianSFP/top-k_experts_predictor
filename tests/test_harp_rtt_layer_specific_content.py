from __future__ import annotations

import torch

from harp_rtt.content_transition import ContentTransitionConfig
from harp_rtt.layer_specific_content import LayerSpecificFullStateProbe


def _probe() -> LayerSpecificFullStateProbe:
    torch.manual_seed(23)
    config = ContentTransitionConfig(
        layers=4, experts=7, hidden_width=9, router_rank=3, exact_k=2,
        latent_width=8, effect_width=4, transition_width=16,
        content_adapter_rank=2, dropout=0.0,
    )
    basis = torch.stack([
        torch.linalg.qr(torch.randn(9, 3)).Q for _ in range(4)
    ])
    return LayerSpecificFullStateProbe(
        config, basis, torch.randn(4, 7, 3), torch.randn(4, 7),
        torch.ones(4, 3, dtype=torch.bool), content_rank=5,
    )


def test_layer_specific_probe_geometry_and_gradients() -> None:
    probe = _probe()
    inputs = torch.randn(2, 3, 4, 9)
    ids = torch.randint(0, 7, (2, 3, 4, 2))
    weights = torch.rand(2, 3, 4, 2)
    output = probe(inputs, ids, weights)
    assert output.predicted_queries.shape == (2, 3, 3, 3)
    assert output.predicted_scores.shape == (2, 3, 3, 7)
    output.predicted_scores.square().mean().backward()
    assert probe.content_down.grad is not None
    assert probe.expert_effect.grad is not None


def test_layer_specific_router_blind_ablation_is_matched() -> None:
    probe = _probe().eval()
    inputs = torch.randn(2, 3, 4, 9)
    ids = torch.randint(0, 7, (2, 3, 4, 2))
    weights = torch.rand(2, 3, 4, 2)
    with torch.no_grad():
        full = probe(inputs, ids, weights, use_router_blind_content=True)
        ablated = probe(inputs, ids, weights, use_router_blind_content=False)
    assert torch.equal(full.router_queries, ablated.router_queries)
    assert torch.equal(full.router_blind_inputs, ablated.router_blind_inputs)
    assert not torch.equal(full.predicted_scores, ablated.predicted_scores)


def test_layer_specific_parameter_count_stays_under_cap() -> None:
    config = ContentTransitionConfig()
    # Use meta tensors to audit production geometry without materializing it.
    with torch.device("meta"):
        probe = LayerSpecificFullStateProbe(
            config,
            torch.empty(config.layers, config.hidden_width, config.router_rank),
            torch.empty(config.layers, config.experts, config.router_rank),
            torch.empty(config.layers, config.experts),
            torch.ones(config.layers, config.router_rank, dtype=torch.bool),
            content_rank=128,
        )
    assert sum(parameter.numel() for parameter in probe.parameters()) < 25_000_000
