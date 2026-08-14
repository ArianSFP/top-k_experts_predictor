from __future__ import annotations

import pytest
import torch

from harp_rtt.route_dynamics import (
    DeltaRouteConfig,
    DeltaRouteTrajectory,
    LayerConditionedChannelPool,
    RouteDynamicsCore,
    centered_logit_huber,
    gather_node_horizon,
    predicted_route_distribution,
)


def _config() -> DeltaRouteConfig:
    return DeltaRouteConfig(
        experts=8,
        layers=4,
        horizons=2,
        nodes=3,
        router_rank=3,
        raw_width=6,
        metadata_width=4,
        latent_width=8,
        effect_width=4,
        transition_width=16,
        layer_adapter_rank=2,
        free_rank=2,
        attention_heads=2,
        exact_k=2,
        dropout=0.0,
    )


def _geometry(config: DeltaRouteConfig) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(11)
    keys = torch.randn(
        config.layers, config.experts, config.router_rank, generator=generator
    )
    bias = torch.randn(config.layers, config.experts, generator=generator)
    rank_mask = torch.ones(config.layers, config.router_rank, dtype=torch.bool)
    return keys, bias, rank_mask


def _channels(config: DeltaRouteConfig, batch: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(13)
    raw = [
        torch.randn(batch, config.nodes, config.raw_width, generator=generator)
        for _ in range(5)
    ]
    return {
        "fused": raw[0],
        "post_ffn": raw[1],
        "router_input": raw[2],
        "vocabulary": raw[3],
        "token": raw[4],
        "router_logits": torch.randn(
            batch, config.nodes, config.experts, generator=generator
        ),
        "metadata": torch.randn(
            batch, config.nodes, config.metadata_width, generator=generator
        ),
        "parents": torch.tensor([[-1, 0, 1]]).expand(batch, -1).clone(),
        "available": torch.ones(batch, config.nodes, dtype=torch.bool),
    }


def test_predicted_route_distribution_has_native_forward_and_dense_gradient() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]], requires_grad=True)
    route = predicted_route_distribution(scores, k=2)
    expected = torch.zeros_like(scores)
    expected.scatter_(-1, route.selected_ids, route.selected_weights)
    torch.testing.assert_close(route.dense_straight_through_weights, expected)
    torch.testing.assert_close(route.selected_weights.sum(-1), torch.ones(1))
    route.dense_straight_through_weights[..., 2].sum().backward()
    assert scores.grad is not None
    assert float(scores.grad.abs().sum()) > 0.0


def test_channel_pool_is_layer_conditioned_and_rejects_noncausal_parent() -> None:
    config = _config()
    pool = LayerConditionedChannelPool(config)
    inputs = _channels(config)
    pooled, path = pool(**inputs)
    assert pooled.shape == (
        2, config.horizons, config.layers, config.nodes, config.latent_width
    )
    assert path.shape == (2, config.nodes, config.latent_width)
    malformed = dict(inputs)
    malformed["parents"] = inputs["parents"].clone()
    malformed["parents"][:, 1] = 2
    with pytest.raises(ValueError, match="precede"):
        pool(**malformed)


def test_transition_message_depends_on_execution_weights() -> None:
    config = _config()
    core = RouteDynamicsCore(config)
    with torch.no_grad():
        core.residual_gate.fill_(1.0)
        core.expert_base[0, 0].fill_(1.0)
        core.expert_base[0, 1].fill_(-1.0)
    query = torch.randn(2, config.router_rank)
    ids = torch.tensor([[0, 1], [0, 1]])
    context = torch.randn(2, config.latent_width)
    first = core.step(query, ids, torch.tensor([[0.9, 0.1], [0.9, 0.1]]), context, layer=0)
    second = core.step(query, ids, torch.tensor([[0.1, 0.9], [0.1, 0.9]]), context, layer=0)
    assert not torch.equal(first, second)
    sequence = torch.randn(2, config.layers, config.router_rank)
    assert core.affine_control(sequence).shape == (
        2, config.layers - 1, config.router_rank
    )


def test_rollout_is_layer_causal_under_teacher_forcing() -> None:
    config = _config()
    keys, bias, _ = _geometry(config)
    core = RouteDynamicsCore(config)
    with torch.no_grad():
        core.residual_gate.fill_(1.0)
        core.expert_base.normal_()
    seed = torch.randn(2, config.router_rank)
    context = torch.randn(2, config.layers, config.latent_width)
    ids = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4]]]).expand(2, -1, -1).clone()
    weights = torch.full_like(ids, 0.5, dtype=torch.float32)
    force = torch.ones(2, config.layers - 1, dtype=torch.bool)
    first = core.rollout(
        seed, context, keys, bias,
        teacher_ids=ids, teacher_weights=weights, teacher_force_mask=force,
    )
    changed = ids.clone()
    changed[:, 2] = torch.tensor([6, 7])
    second = core.rollout(
        seed, context, keys, bias,
        teacher_ids=changed, teacher_weights=weights, teacher_force_mask=force,
    )
    torch.testing.assert_close(first.queries[:, :3], second.queries[:, :3])
    assert not torch.equal(first.queries[:, 3], second.queries[:, 3])


def test_deltaroute_epoch_zero_reproduces_parent_queries_scores_and_routes() -> None:
    config = _config()
    keys, bias, rank_mask = _geometry(config)
    model = DeltaRouteTrajectory(config, keys, bias, rank_mask)
    batch = 2
    parent_queries = torch.randn(
        batch, config.horizons, config.layers, config.nodes, config.router_rank
    )
    parent_scores = torch.randn(
        batch, config.horizons, config.layers, config.nodes, config.experts
    )
    context = torch.randn(batch, config.horizons, config.layers, config.latent_width)
    output = model(
        parent_queries=parent_queries,
        parent_scores=parent_scores,
        context_states=context,
        **_channels(config, batch),
    )
    torch.testing.assert_close(output.queries, parent_queries.float(), rtol=0, atol=0)
    torch.testing.assert_close(output.scores, parent_scores.float(), rtol=0, atol=0)
    expected = torch.argsort(
        parent_scores.float(), dim=-1, descending=True, stable=True
    )[..., : config.exact_k]
    assert torch.equal(output.selected_ids, expected)


def test_centered_logit_huber_uses_router_induced_error() -> None:
    config = _config()
    keys, _, _ = _geometry(config)
    target = torch.randn(2, config.layers, config.router_rank)
    assert float(centered_logit_huber(target, target, keys)) == 0.0
    shifted = target.clone()
    shifted[:, 1, 0] += 1.0
    assert float(centered_logit_huber(shifted, target, keys)) > 0.0


def test_gather_node_horizon_selects_each_nodes_depth() -> None:
    values = torch.arange(1 * 4 * 2 * 3 * 1).reshape(1, 4, 2, 3, 1)
    selected = gather_node_horizon(values, torch.tensor([[1, 3, 4]]))
    assert selected.shape == (1, 3, 2, 1)
    assert torch.equal(selected[:, 0], values[:, 0, :, 0])
    assert torch.equal(selected[:, 1], values[:, 2, :, 1])
    assert torch.equal(selected[:, 2], values[:, 3, :, 2])
    padded = gather_node_horizon(
        values, torch.tensor([[1, 0, 4]]), torch.tensor([[True, False, True]])
    )
    assert torch.equal(padded[:, 1], torch.zeros_like(padded[:, 1]))
