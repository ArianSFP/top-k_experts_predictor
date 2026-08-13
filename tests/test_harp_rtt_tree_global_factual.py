from __future__ import annotations

import torch

from harp_rtt.route_dynamics import DeltaRouteConfig
from harp_rtt.tree_global_factual import TreeGlobalFactualHead


def _inputs(config: DeltaRouteConfig) -> dict[str, torch.Tensor]:
    batch = 2
    raw = torch.randn(batch, config.nodes, config.raw_width)
    posterior = torch.rand(batch, config.horizons, config.nodes + 1)
    posterior = posterior / posterior.sum(-1, keepdim=True)
    return {
        "context_states": torch.randn(
            batch, config.horizons, config.layers, config.latent_width,
        ),
        "posterior": posterior,
        "fused": raw, "post_ffn": torch.randn_like(raw),
        "router_input": torch.randn_like(raw),
        "vocabulary": torch.randn_like(raw), "token": torch.randn_like(raw),
        "router_logits": torch.randn(batch, config.nodes, config.experts),
        "metadata": torch.randn(batch, config.nodes, config.metadata_width),
        "parents": torch.tensor([[-1, 0, 1], [-1, 0, 0]]),
        "available": torch.ones(batch, config.nodes, dtype=torch.bool),
    }


def test_tree_global_head_is_zero_but_gradient_open() -> None:
    torch.manual_seed(61)
    config = DeltaRouteConfig(
        experts=7, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=9, metadata_width=5, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    model = TreeGlobalFactualHead(config, torch.randn(4, 7, 3))
    output = model(**_inputs(config))
    assert torch.count_nonzero(output.score_delta) == 0
    output.score_delta[:, 1:].square().sum().add(
        output.score_delta[:, 1:].sum()
    ).backward()
    assert model.channels.raw[0].weight.grad is not None
    assert float(model.channels.raw[0].weight.grad.abs().sum()) > 0.0
    assert model.node_key.weight.grad is not None
    assert float(model.node_key.weight.grad.abs().sum()) > 0.0
    assert model.expert_query.weight.grad is not None
    assert float(model.expert_query.weight.grad.abs().sum()) > 0.0


def test_tree_global_head_normalizes_expert_weights_including_other() -> None:
    config = DeltaRouteConfig(
        experts=7, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=9, metadata_width=5, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    output = TreeGlobalFactualHead(
        config, torch.randn(4, 7, 3)
    )(**_inputs(config))
    assert output.expert_branch_weights.shape == (2, 4, 4, 7, 4)
    assert torch.allclose(
        output.expert_branch_weights.sum(-1),
        torch.ones(2, 4, 4, 7), atol=1e-6,
    )


def test_tree_global_head_has_expert_specific_branch_weights() -> None:
    torch.manual_seed(67)
    config = DeltaRouteConfig(
        experts=7, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=9, metadata_width=5, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    output = TreeGlobalFactualHead(
        config, torch.randn(4, 7, 3)
    )(**_inputs(config))
    assert not torch.equal(
        output.expert_branch_weights[..., 0, :],
        output.expert_branch_weights[..., 1, :],
    )
