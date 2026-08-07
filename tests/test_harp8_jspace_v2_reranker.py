from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from harp8.jspace_reranker import JSpaceRerankerLossConfig
from harp8.jspace_v2_reranker import (
    CompositeJSpaceV2Inference,
    JSpaceV2CandidateReranker,
    JSpaceV2RerankerConfig,
    LayerSpecificRouterQuery,
    SeparatedMTPEncoder,
    composite_v2_horizon_scores,
    v2_reranker_loss,
)


def _config(**updates: object) -> JSpaceV2RerankerConfig:
    value = JSpaceV2RerankerConfig(
        experts=16,
        layers=3,
        horizons=4,
        pool_horizons=8,
        candidate_count=12,
        native_k=8,
        j_lags=3,
        j_width=10,
        generator_context_width=9,
        mtp_nodes=6,
        mtp_state_channels=2,
        mtp_state_width=7,
        router_key_width=16,
        router_query_rank=8,
        expert_embedding_width=8,
        candidate_feature_width=4,
        model_width=32,
        attention_heads=4,
        feedforward_width=64,
        temporal_blocks=1,
        axial_blocks=1,
        mtp_hidden_blocks=1,
        mtp_router_blocks=1,
        set_blocks=2,
        inducing_points=4,
        dropout=0.0,
    )
    return replace(value, **updates)


def _batch(
    config: JSpaceV2RerankerConfig,
    *,
    batch_size: int = 2,
    full_horizons: bool = False,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(811)
    horizons = config.pool_horizons if full_horizons else config.horizons
    shape = (batch_size, horizons, config.layers, config.candidate_count)
    ids = torch.arange(config.candidate_count).view(1, 1, 1, -1).expand(shape).clone()
    base = torch.randn(shape, generator=generator)
    membership = torch.zeros(shape, dtype=torch.bool)
    membership[..., : config.native_k] = True
    values: dict[str, torch.Tensor] = {
        "candidate_scores": base,
        "candidate_ids": ids,
        "candidate_mask": torch.ones(shape, dtype=torch.bool),
        "candidate_features": torch.randn(
            shape + (config.candidate_feature_width,), generator=generator
        ),
        "target_membership": membership,
        "teacher_candidate_scores": base + membership.float() * 3.0,
        "valid_future": torch.ones(batch_size, horizons, dtype=torch.bool),
        "generator_context": torch.randn(
            batch_size,
            horizons,
            config.layers,
            config.generator_context_width,
            generator=generator,
        ),
        "j_states": torch.randn(
            batch_size,
            config.j_lags,
            config.layers,
            config.j_width,
            generator=generator,
        ),
        "j_mask": torch.ones(
            batch_size, config.j_lags, config.layers, dtype=torch.bool
        ),
        "mtp_states": torch.randn(
            batch_size,
            config.mtp_nodes,
            config.mtp_state_channels,
            config.mtp_state_width,
            generator=generator,
        ),
        "mtp_router_logits": torch.randn(
            batch_size,
            config.mtp_nodes,
            config.experts,
            generator=generator,
        ),
        "mtp_mask": torch.ones(batch_size, config.mtp_nodes, dtype=torch.bool),
    }
    values["mtp_mask"][-1] = False
    return values


def _model(config: JSpaceV2RerankerConfig) -> JSpaceV2CandidateReranker:
    keys = torch.randn(
        config.layers,
        config.experts,
        config.router_key_width,
        generator=torch.Generator().manual_seed(19),
    )
    return JSpaceV2CandidateReranker(config, keys)


def test_v2_defaults_fail_closed_to_h1_h4_over_h1_h8_pool() -> None:
    config = JSpaceV2RerankerConfig()
    config.validate()
    assert config.horizons == 4
    assert config.pool_horizons == 8
    assert config.uses_full_router_rank
    with pytest.raises(ValueError, match="exactly H1-H4"):
        replace(config, horizons=8).validate()


def test_v2_epoch_zero_equals_base_and_uses_layer_aware_context() -> None:
    config = _config()
    model = _model(config).eval()
    batch = _batch(config)
    with torch.inference_mode():
        output = model(batch)
    assert torch.equal(output.scores, batch["candidate_scores"])
    assert torch.count_nonzero(output.delta) == 0
    assert output.horizon_layer_context.shape == (
        2,
        config.horizons,
        config.layers,
        config.model_width,
    )
    assert output.mtp_hidden_context.shape == output.horizon_layer_context.shape
    assert output.mtp_router_context.shape == output.horizon_layer_context.shape
    assert output.source_weights.shape == (
        2,
        config.horizons,
        config.layers,
        3,
    )
    assert torch.allclose(
        output.source_weights.sum(dim=-1),
        torch.ones_like(output.source_weights[..., 0]),
        atol=1e-6,
    )
    # Target-layer-specific query maps are explicit parameters, not a shared
    # linear layer conditioned only through an embedding.
    assert model.router_query.down.shape[:2] == (
        config.layers,
        config.model_width,
    )
    assert model.router_query.up.shape[0] == config.layers


def test_v2_nontrivial_scores_are_candidate_permutation_equivariant() -> None:
    config = _config()
    model = _model(config).eval()
    torch.nn.init.normal_(model.delta_head[-1].weight, std=0.04)
    torch.nn.init.normal_(model.delta_head[-1].bias, std=0.01)
    batch = _batch(config)
    permutation = torch.tensor([5, 0, 10, 2, 11, 1, 7, 4, 3, 9, 8, 6])
    inverse = torch.argsort(permutation)
    permuted = dict(batch)
    for name in (
        "candidate_scores",
        "candidate_ids",
        "candidate_mask",
        "target_membership",
        "teacher_candidate_scores",
    ):
        permuted[name] = batch[name].index_select(-1, permutation)
    permuted["candidate_features"] = batch["candidate_features"].index_select(
        -2, permutation
    )
    with torch.inference_mode():
        original = model(batch)
        reordered = model(permuted)
    assert torch.allclose(
        original.scores,
        reordered.scores.index_select(-1, inverse),
        atol=2e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        original.horizon_layer_context,
        reordered.horizon_layer_context,
        atol=1e-6,
        rtol=1e-6,
    )


def test_mtp_hidden_and_router_are_distinct_memories_until_fusion() -> None:
    config = _config()
    encoder = SeparatedMTPEncoder(config).eval()
    batch = _batch(config, batch_size=1)
    with torch.inference_mode():
        hidden, router, padding = encoder(
            batch["mtp_states"],
            batch["mtp_router_logits"],
            mask=torch.zeros(1, config.mtp_nodes, dtype=torch.bool),
        )
    expected = (1, config.mtp_nodes + 1, config.model_width)
    assert hidden.shape == router.shape == expected
    assert padding.shape == (1, config.mtp_nodes + 1)
    assert encoder.hidden_blocks is not encoder.router_blocks
    assert encoder.state_projections is not encoder.router_projection
    names = dict(encoder.named_parameters())
    assert any(name.startswith("state_projections") for name in names)
    assert any(name.startswith("router_projection") for name in names)


def test_layer_specific_query_has_no_cross_layer_parameter_application() -> None:
    config = _config()
    query = LayerSpecificRouterQuery(config)
    with torch.no_grad():
        query.down.zero_()
        query.up.zero_()
        query.bias.zero_()
        query.down[1].fill_(0.25)
        query.up[1].fill_(0.5)
    context = torch.ones(2, config.horizons, config.layers, config.model_width)
    output = query(context)
    assert torch.count_nonzero(output[:, :, 0]) == 0
    assert torch.count_nonzero(output[:, :, 2]) == 0
    assert torch.count_nonzero(output[:, :, 1]) > 0


def test_v2_composite_preserves_h5_h8_and_rejects_label_only_inputs() -> None:
    config = _config()
    model = _model(config).eval()
    composite = CompositeJSpaceV2Inference(model).eval()
    full = _batch(config, full_horizons=True)
    with torch.inference_mode():
        output = composite(full)
    assert output["scores"].shape == full["candidate_scores"].shape
    assert torch.equal(
        output["scores"][:, 4:], full["candidate_scores"][:, 4:].float()
    )
    assert torch.equal(
        output["active_scores"], full["candidate_scores"][:, :4].float()
    )
    with pytest.raises(ValueError, match="label-only"):
        model({**_batch(config), "prefix_matches_committed": torch.ones(2)})


def test_v2_shape_gates_and_loss() -> None:
    config = _config()
    model = _model(config)
    batch = _batch(config)
    wrong = dict(batch)
    wrong["generator_context"] = batch["generator_context"][:, :, :-1]
    with pytest.raises(ValueError, match="generator_context"):
        model(wrong)

    with torch.inference_mode():
        output = model(batch)
    loss = v2_reranker_loss(
        output,
        batch,
        JSpaceRerankerLossConfig(
            horizon_weights=(1.0, 1.0, 1.25, 1.5),
            hard_negative_count=3,
        ),
    )
    assert torch.isfinite(loss.total)

    base = torch.randn(1, 8, 2, 12)
    active = torch.randn(1, 4, 2, 12)
    merged = composite_v2_horizon_scores(active, base)
    assert torch.equal(merged[:, 4:], base[:, 4:])
