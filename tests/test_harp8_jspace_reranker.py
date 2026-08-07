from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from harp8.jspace_reranker import (
    JSpaceCandidateReranker,
    JSpaceRerankerConfig,
    JSpaceRerankerLossConfig,
    JSpaceRerankerOutput,
    MTPNodeEncoder,
    jspace_reranker_loss,
)


def _config(**updates: object) -> JSpaceRerankerConfig:
    config = JSpaceRerankerConfig(
        experts=16,
        layers=3,
        horizons=8,
        candidate_count=12,
        native_k=8,
        j_lags=3,
        j_width=10,
        mtp_nodes=6,
        mtp_state_channels=2,
        mtp_state_width=7,
        router_key_width=16,
        expert_embedding_width=8,
        candidate_feature_width=4,
        model_width=32,
        attention_heads=4,
        feedforward_width=64,
        temporal_blocks=1,
        axial_blocks=1,
        mtp_blocks=1,
        set_blocks=2,
        inducing_points=4,
        dropout=0.0,
    )
    return replace(config, **updates)


def _batch(config: JSpaceRerankerConfig, *, batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    shape = (
        batch_size,
        config.horizons,
        config.layers,
        config.candidate_count,
    )
    base_ids = torch.arange(config.candidate_count, dtype=torch.long)
    candidate_ids = base_ids.view(1, 1, 1, -1).expand(shape).clone()
    candidate_scores = torch.randn(shape, generator=generator)
    candidate_features = torch.randn(
        shape + (config.candidate_feature_width,), generator=generator
    )
    j_states = torch.randn(
        batch_size,
        config.j_lags,
        config.layers,
        config.j_width,
        generator=generator,
    )
    mtp_states = torch.randn(
        batch_size,
        config.mtp_nodes,
        config.mtp_state_channels,
        config.mtp_state_width,
        generator=generator,
    )
    mtp_router = torch.randn(
        batch_size,
        config.mtp_nodes,
        config.experts,
        generator=generator,
    )
    mtp_mask = torch.ones(batch_size, config.mtp_nodes, dtype=torch.bool)
    # An entirely unavailable MTP source is valid and uses the learned fallback.
    mtp_mask[-1] = False
    membership = torch.zeros(shape, dtype=torch.bool)
    membership[..., : config.native_k] = True
    teacher = candidate_scores.detach().clone()
    teacher[..., : config.native_k] += 3.0
    return {
        "candidate_scores": candidate_scores,
        "candidate_ids": candidate_ids,
        "candidate_features": candidate_features,
        "candidate_mask": torch.ones(shape, dtype=torch.bool),
        "j_states": j_states,
        "j_mask": torch.ones(
            batch_size, config.j_lags, config.layers, dtype=torch.bool
        ),
        "mtp_states": mtp_states,
        "mtp_router_logits": mtp_router,
        "mtp_mask": mtp_mask,
        "target_membership": membership,
        "teacher_candidate_scores": teacher,
        "valid_future": torch.ones(
            batch_size, config.horizons, dtype=torch.bool
        ),
    }


def _model(config: JSpaceRerankerConfig) -> JSpaceCandidateReranker:
    keys = torch.randn(
        config.layers,
        config.experts,
        config.router_key_width,
        generator=torch.Generator().manual_seed(9),
    )
    return JSpaceCandidateReranker(config, keys)


def test_production_defaults_encode_approved_geometry() -> None:
    config = JSpaceRerankerConfig()
    config.validate()
    assert config.horizons == 8
    assert config.candidate_count == 64
    assert config.mtp_nodes == 6
    assert config.set_blocks == 2
    assert config.inducing_points == 16
    assert config.uses_full_router_rank


def test_zero_initialized_forward_is_exact_base_and_router_keys_are_frozen() -> None:
    config = _config()
    model = _model(config).eval()
    batch = _batch(config)
    with torch.inference_mode():
        output = model(batch)
    assert output.scores.shape == batch["candidate_scores"].shape
    assert output.delta.shape == batch["candidate_scores"].shape
    assert output.router_dot_scores.shape == batch["candidate_scores"].shape
    assert output.horizon_layer_context.shape == (
        2,
        config.horizons,
        config.layers,
        config.model_width,
    )
    assert torch.count_nonzero(output.delta) == 0
    assert torch.equal(output.scores, batch["candidate_scores"])
    assert "router_keys" in dict(model.named_buffers())
    assert "router_keys" not in dict(model.named_parameters())


def test_nontrivial_reranker_is_candidate_permutation_equivariant() -> None:
    config = _config()
    model = _model(config).eval()
    # Move off the exact-zero initialization so this tests the architecture,
    # not merely the residual identity path.
    torch.nn.init.normal_(model.delta_head[-1].weight, std=0.05)
    torch.nn.init.normal_(model.delta_head[-1].bias, std=0.02)
    batch = _batch(config)
    permutation = torch.tensor([3, 0, 10, 4, 1, 11, 9, 2, 7, 6, 5, 8])
    inverse = torch.argsort(permutation)
    permuted = dict(batch)
    for name in (
        "candidate_scores",
        "candidate_ids",
        "candidate_features",
        "candidate_mask",
        "target_membership",
        "teacher_candidate_scores",
    ):
        permuted[name] = batch[name].index_select(-2 if name == "candidate_features" else -1, permutation)
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
        original.router_dot_scores,
        reordered.router_dot_scores.index_select(-1, inverse),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        original.horizon_layer_context,
        reordered.horizon_layer_context,
        atol=1e-6,
        rtol=1e-6,
    )


def test_mtp_encoder_is_separate_and_accepts_all_missing_nodes() -> None:
    config = _config()
    encoder = MTPNodeEncoder(config).eval()
    batch = _batch(config, batch_size=1)
    with torch.inference_mode():
        nodes, padding = encoder(
            batch["mtp_states"],
            batch["mtp_router_logits"],
            mask=torch.zeros(1, config.mtp_nodes, dtype=torch.bool),
        )
    assert nodes.shape == (1, config.mtp_nodes + 1, config.model_width)
    assert padding.shape == (1, config.mtp_nodes + 1)
    assert padding[0, :-1].all()
    assert not padding[0, -1]
    assert all("lens" not in name.lower() for name, _ in encoder.named_parameters())

    with pytest.raises(ValueError, match="mtp_states geometry"):
        encoder(
            batch["mtp_states"][:, :-1],
            batch["mtp_router_logits"][:, :-1],
        )


def test_combined_loss_rewards_correct_ranking_and_backpropagates() -> None:
    config = _config(candidate_feature_width=0)
    batch = _batch(_config())
    shape = batch["candidate_scores"].shape
    membership = batch["target_membership"]
    perfect = torch.where(
        membership,
        torch.full(shape, 5.0),
        torch.full(shape, -5.0),
    ).requires_grad_()
    inverted = (-perfect.detach()).requires_grad_()
    batch["teacher_candidate_scores"] = perfect.detach().clone()
    loss_config = JSpaceRerankerLossConfig(hard_negative_count=3)
    good = jspace_reranker_loss(perfect, batch, loss_config)
    bad = jspace_reranker_loss(inverted, batch, loss_config)
    assert torch.isfinite(good.total)
    assert good.total < bad.total
    assert set(good.components) == {
        "boundary",
        "balanced_bce",
        "listwise",
        "restricted_kl",
    }
    assert good.components["restricted_kl"].abs() < 1e-6
    assert good.per_horizon["boundary"].shape == (config.horizons,)
    good.total.backward()
    assert perfect.grad is not None
    assert torch.isfinite(perfect.grad).all()


def test_zero_weight_horizons_cannot_change_total_loss() -> None:
    config = _config()
    batch = _batch(config)
    predicted = batch["candidate_scores"].clone()
    h1_only = JSpaceRerankerLossConfig(
        hard_negative_count=4,
        horizon_weights=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    reference = jspace_reranker_loss(predicted, batch, h1_only)
    changed = predicted.clone()
    changed[:, 1:] = changed[:, 1:] * 20.0 + 100.0
    ignored = jspace_reranker_loss(changed, batch, h1_only)
    assert torch.allclose(reference.total, ignored.total, atol=1e-6, rtol=0.0)


def test_shape_contracts_fail_closed() -> None:
    config = _config()
    model = _model(config)
    batch = _batch(config)
    wrong_candidates = dict(batch)
    wrong_candidates["candidate_scores"] = batch["candidate_scores"][..., :-1]
    wrong_candidates["candidate_ids"] = batch["candidate_ids"][..., :-1]
    with pytest.raises(ValueError, match="candidate geometry"):
        model(wrong_candidates)

    wrong_keys = torch.zeros(
        config.layers, config.experts, config.router_key_width - 1
    )
    with pytest.raises(ValueError, match="router_keys geometry"):
        JSpaceCandidateReranker(config, wrong_keys)
