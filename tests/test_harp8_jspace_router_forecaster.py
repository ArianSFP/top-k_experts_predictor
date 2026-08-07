from __future__ import annotations

from dataclasses import replace
from inspect import getsource

import pytest
import torch
from torch.nn import functional as F

from harp8.jspace_router_forecaster import (
    FullRouterLossConfig,
    HorizonLayerResidualHead,
    JSpaceFullRouterForecaster,
    JSpaceRouterForecasterConfig,
    SeparatedNativeMTPMemory,
    full_router_forecaster_loss,
    _masked_horizon_mean,
    _top8_swap_rows,
    _stable_masked_topk_ids,
)
from harp8.jspace_router_metrics import slot_recall_at_k


def _config(**updates: object) -> JSpaceRouterForecasterConfig:
    config = JSpaceRouterForecasterConfig(
        experts=16,
        layers=3,
        horizons=8,
        history=3,
        j_width=10,
        generator_context_width=9,
        mtp_nodes=6,
        mtp_state_channels=1,
        mtp_state_width=7,
        model_width=24,
        attention_heads=4,
        feedforward_width=48,
        layer_blocks=1,
        mtp_blocks=1,
        fusion_blocks=1,
        output_rank=8,
        dropout=0.0,
    )
    return replace(config, **updates)


def _batch(config: JSpaceRouterForecasterConfig, batch_size: int = 2):
    generator = torch.Generator().manual_seed(808)
    teacher = torch.randn(
        batch_size,
        config.horizons,
        config.layers,
        config.experts,
        generator=generator,
    )
    values = {
        "base_router_scores": torch.randn(
            teacher.shape, generator=generator
        ),
        "route_history": torch.randn(
            batch_size,
            config.history,
            config.layers,
            config.experts,
            generator=generator,
        ),
        "route_mask": torch.ones(
            batch_size, config.history, config.layers, dtype=torch.bool
        ),
        "j_states": torch.randn(
            batch_size,
            config.history,
            config.layers,
            config.j_width,
            generator=generator,
        ),
        "j_mask": torch.ones(
            batch_size, config.history, config.layers, dtype=torch.bool
        ),
        "generator_context": torch.randn(
            batch_size,
            config.horizons,
            config.layers,
            config.generator_context_width,
            generator=generator,
        ),
        "mtp_states": torch.randn(
            batch_size,
            config.mtp_nodes,
            config.mtp_state_width,
            generator=generator,
        ),
        "mtp_router_logits": torch.randn(
            batch_size,
            config.mtp_nodes,
            config.experts,
            generator=generator,
        ),
        "mtp_mask": torch.ones(
            batch_size, config.mtp_nodes, dtype=torch.bool
        ),
        "within_request": torch.linspace(0.0, 0.5, batch_size),
        "teacher_router_scores": teacher,
        "target_top8": torch.topk(teacher, 8, dim=-1).indices,
        "valid_future": torch.ones(
            batch_size, config.horizons, dtype=torch.bool
        ),
    }
    if config.secondary_j_width is not None:
        values["secondary_target_states"] = torch.randn(
            batch_size,
            config.history,
            config.layers,
            config.secondary_j_width,
            generator=generator,
        )
        values["secondary_target_mask"] = values["j_mask"].clone()
    return values


def test_full_router_forecaster_epoch_zero_is_exact_harp_passthrough() -> None:
    config = _config()
    model = JSpaceFullRouterForecaster(config).eval()
    batch = _batch(config)
    with torch.inference_mode():
        output = model(batch)
    assert output.future_router_scores.shape == (2, 8, 3, 16)
    assert torch.equal(output.future_router_scores, batch["base_router_scores"])
    assert torch.count_nonzero(output.delta) == 0
    assert output.source_weights.shape == (2, 8, 3, 5)
    assert torch.allclose(
        output.source_weights.sum(dim=-1),
        torch.ones_like(output.source_weights[..., 0]),
        atol=1e-6,
    )


def test_future_labels_cannot_affect_forward_and_forbidden_labels_fail_closed() -> None:
    config = _config()
    model = JSpaceFullRouterForecaster(config).eval()
    batch = _batch(config)
    changed = dict(batch)
    changed["teacher_router_scores"] = torch.randn_like(
        batch["teacher_router_scores"]
    ) * 1000
    changed["target_top8"] = torch.flip(batch["target_top8"], dims=(-1,))
    with torch.inference_mode():
        first = model(batch).future_router_scores
        second = model(changed).future_router_scores
    assert torch.equal(first, second)
    with pytest.raises(ValueError, match="label-only"):
        model({**batch, "accepted_through_depth": torch.ones(2)})


def test_all_missing_mtp_is_finite_and_no_later_horizon_reuses_depth_six() -> None:
    config = _config()
    model = JSpaceFullRouterForecaster(config).eval()
    batch = _batch(config)
    batch["mtp_mask"].zero_()
    with torch.inference_mode():
        output = model(batch)
    assert torch.isfinite(output.future_router_scores).all()
    assert torch.isfinite(output.source_weights).all()
    # H7/H8 diagonal slots are explicit missing parameters, not aliases for
    # the last observed MTP depth.
    assert model.diagonal_missing.shape == (2, 8, config.model_width)


def test_corrected_diagonal_uses_pre_context_node_local_encodings() -> None:
    config = _config(mtp_diagonal_from_local=True)
    model = JSpaceFullRouterForecaster(config).eval()
    batch = _batch(config, batch_size=1)
    mask = batch["mtp_mask"]
    with torch.inference_mode():
        first = model.mtp_encoder(
            batch["mtp_states"], batch["mtp_router_logits"], mask
        )
    (
        first_hidden_context,
        first_router_context,
        _,
        first_hidden_local,
        first_router_local,
    ) = first

    changed_states = batch["mtp_states"].clone()
    changed_router = batch["mtp_router_logits"].clone()
    # Perturb depth 2 only.  The corrected H1 diagonal must remain a pure
    # function of depth 1 even though contextual memory can mix all depths.
    changed_states[:, 1].add_(1000.0 * torch.randn_like(changed_states[:, 1]))
    changed_router[:, 1].add_(1000.0 * torch.randn_like(changed_router[:, 1]))
    with torch.inference_mode():
        second = model.mtp_encoder(changed_states, changed_router, mask)
    (
        second_hidden_context,
        second_router_context,
        _,
        second_hidden_local,
        second_router_local,
    ) = second

    first_diagonal = model._diagonal_sources(
        first_hidden_local, first_router_local, mask
    )
    second_diagonal = model._diagonal_sources(
        second_hidden_local, second_router_local, mask
    )
    assert torch.equal(first_diagonal[0][:, 0], second_diagonal[0][:, 0])
    assert torch.equal(first_diagonal[1][:, 0], second_diagonal[1][:, 0])
    # This guards against accidentally making both outputs local: the
    # all-depth memories are still contextualized by their encoder blocks.
    assert not torch.equal(
        first_hidden_context[:, 0], second_hidden_context[:, 0]
    )
    assert not torch.equal(
        first_router_context[:, 0], second_router_context[:, 0]
    )


def test_diagonal_h1_h6_mapping_and_h7_h8_explicit_missing() -> None:
    config = _config(mtp_diagonal_from_local=True)
    model = JSpaceFullRouterForecaster(config).eval()
    hidden = torch.stack(
        [torch.full((config.model_width,), 10.0 + depth) for depth in range(6)]
    )[None]
    router = torch.stack(
        [torch.full((config.model_width,), 20.0 + depth) for depth in range(6)]
    )[None]
    mask = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        model.diagonal_missing[0, 6:].fill_(-7.0)
        model.diagonal_missing[1, 6:].fill_(-8.0)
    diagonal_hidden, diagonal_router = model._diagonal_sources(
        hidden, router, mask
    )
    assert torch.equal(diagonal_hidden[:, 0], hidden[:, 0])  # H1 <- depth 1
    assert torch.equal(diagonal_router[:, 0], router[:, 0])
    assert torch.equal(diagonal_hidden[:, 5], hidden[:, 5])  # H6 <- depth 6
    assert torch.equal(diagonal_router[:, 5], router[:, 5])
    assert torch.equal(
        diagonal_hidden[:, 6:],
        torch.full_like(diagonal_hidden[:, 6:], -7.0),
    )
    assert torch.equal(
        diagonal_router[:, 6:],
        torch.full_like(diagonal_router[:, 6:], -8.0),
    )


def test_corrected_null_token_is_available_only_for_all_missing_rows() -> None:
    corrected = _config(mtp_null_only_when_all_missing=True)
    encoder = SeparatedNativeMTPMemory(corrected).eval()
    batch = _batch(corrected, batch_size=2)
    mask = torch.tensor(
        [[True, False, False, False, False, False], [False] * 6],
        dtype=torch.bool,
    )
    with torch.inference_mode():
        _, _, padding, _, _ = encoder(
            batch["mtp_states"], batch["mtp_router_logits"], mask
        )
    assert padding[0].tolist() == [False, True, True, True, True, True, True]
    assert padding[1].tolist() == [True, True, True, True, True, True, False]

    legacy = _config(mtp_null_only_when_all_missing=False)
    legacy_encoder = SeparatedNativeMTPMemory(legacy).eval()
    with torch.inference_mode():
        _, _, legacy_padding, _, _ = legacy_encoder(
            batch["mtp_states"], batch["mtp_router_logits"], mask
        )
    assert legacy_padding[:, -1].tolist() == [False, False]


def test_opt_in_horizon_depth_attention_bias_is_per_head_and_finite() -> None:
    config = _config(
        mtp_diagonal_from_local=True,
        mtp_null_only_when_all_missing=True,
        mtp_horizon_depth_attention_bias=True,
    )
    model = JSpaceFullRouterForecaster(config).eval()
    expected = (config.attention_heads, config.horizons, config.mtp_nodes)
    assert model.hidden_horizon_depth_attention_bias is not None
    assert model.router_horizon_depth_attention_bias is not None
    assert model.hidden_horizon_depth_attention_bias.shape == expected
    assert model.router_horizon_depth_attention_bias.shape == expected
    assert torch.count_nonzero(model.hidden_horizon_depth_attention_bias) == 0
    batch = _batch(config)
    # Exercise both a partly available row and the all-missing null fallback.
    batch["mtp_mask"][0, 3:] = False
    batch["mtp_mask"][1].zero_()
    with torch.inference_mode():
        output = model(batch)
    assert torch.isfinite(output.future_router_scores).all()
    assert torch.isfinite(output.source_weights).all()


def test_disabled_mtp_corrections_preserve_legacy_config_and_state_layout() -> None:
    config = _config()
    payload = config.to_dict()
    for name in (
        "mtp_diagonal_from_local",
        "mtp_null_only_when_all_missing",
        "mtp_horizon_depth_attention_bias",
    ):
        assert name not in payload
    model = JSpaceFullRouterForecaster(config)
    assert model.hidden_horizon_depth_attention_bias is None
    assert model.router_horizon_depth_attention_bias is None
    assert not any(
        "horizon_depth_attention_bias" in name for name in model.state_dict()
    )


def test_layer_horizon_residual_head_is_independent_and_full_namespace() -> None:
    config = _config()
    head = HorizonLayerResidualHead(config)
    with torch.no_grad():
        head.down.zero_()
        head.up.zero_()
        head.bias.zero_()
        head.down[2, 1].fill_(0.25)
        head.up[2, 1].fill_(0.5)
    context = torch.ones(1, 8, 3, config.model_width)
    output = head(context)
    assert output.shape == (1, 8, 3, config.experts)
    assert torch.count_nonzero(output[:, 2, 1]) == config.experts
    masked = output.clone()
    masked[:, 2, 1] = 0
    assert torch.count_nonzero(masked) == 0


def test_full_router_loss_has_forward_kl_and_top8_gradients() -> None:
    config = _config()
    batch = _batch(config)
    cfg = FullRouterLossConfig(
        temperature=1.0,
        hard_negative_end_rank=16,
        predicted_negative_count=4,
    )
    exact = batch["teacher_router_scores"].clone().requires_grad_(True)
    exact_loss = full_router_forecaster_loss(exact, batch, cfg)
    reversed_scores = (-batch["teacher_router_scores"]).clone().requires_grad_(True)
    wrong_loss = full_router_forecaster_loss(reversed_scores, batch, cfg)
    assert exact_loss.components["router_kl"].abs() < 1e-6
    assert wrong_loss.components["router_kl"] > exact_loss.components["router_kl"]
    wrong_loss.total.backward()
    assert reversed_scores.grad is not None
    assert torch.isfinite(reversed_scores.grad).all()
    assert torch.count_nonzero(reversed_scores.grad) > 0


def test_slot_recall_uses_native_denominator_eight() -> None:
    scores = torch.arange(16, dtype=torch.float32).view(1, 1, 1, 16)
    target = torch.tensor([[[[15, 14, 13, 12, 11, 10, 1, 0]]]])
    assert slot_recall_at_k(scores, target, 8).item() == 0.75
    assert slot_recall_at_k(scores, target, 16).item() == 1.0


def test_stable_negative_mining_breaks_exact_ties_by_expert_id() -> None:
    values = torch.zeros(1, 1, 1, 16)
    excluded = torch.zeros_like(values, dtype=torch.bool)
    excluded[..., [1, 3, 8, 10, 11, 12, 13, 15]] = True
    selected = _stable_masked_topk_ids(values, excluded, 4)
    assert selected.tolist() == [[[[0, 2, 4, 5]]]]

    values[..., 7] = 2.0
    selected = _stable_masked_topk_ids(values, excluded, 4)
    assert selected.tolist() == [[[[7, 0, 2, 4]]]]


def test_authoritative_top8_overrides_tied_raw_logits_in_boundary_loss() -> None:
    config = _config()
    batch = _batch(config, batch_size=1)
    batch["teacher_router_scores"].zero_()
    authoritative = torch.arange(8, 16).view(1, 1, 1, 8).expand(1, 8, 3, 8)
    batch["target_top8"] = authoritative
    scores = torch.zeros_like(batch["teacher_router_scores"], requires_grad=True)
    loss = full_router_forecaster_loss(
        scores,
        batch,
        FullRouterLossConfig(
            router_kl=0.0,
            boundary=1.0,
            predicted_boundary=0.0,
            full_membership=0.0,
            centered_score=0.0,
            hard_negative_end_rank=16,
            predicted_negative_count=4,
        ),
    )
    loss.total.backward()
    assert scores.grad is not None
    # Even though the tied raw vector's stable top-8 would be IDs 0..7, the
    # captured authoritative IDs 8..15 receive the positive gradient sign.
    assert torch.all(scores.grad[..., 8:16] < 0)
    assert torch.all(scores.grad[..., 0:8] > 0)


def test_top8_swap_loss_is_offset_invariant_and_zero_for_exact_set() -> None:
    config = _config()
    batch = _batch(config, batch_size=1)
    target = torch.arange(8, 16).view(1, 1, 1, 8).expand(1, 8, 3, 8)
    batch["target_top8"] = target
    scores = torch.full_like(batch["teacher_router_scores"], -1.0)
    scores[..., 8:15] = 2.0
    scores[..., 15] = 0.0
    scores[..., 0] = 1.0
    cfg = FullRouterLossConfig(
        router_kl=0.0,
        boundary=0.0,
        predicted_boundary=0.0,
        top8_swap=1.0,
        full_membership=0.0,
        centered_score=0.0,
        hard_negative_end_rank=16,
        predicted_negative_count=4,
    )
    wrong = full_router_forecaster_loss(scores, batch, cfg)
    shifted = full_router_forecaster_loss(scores + 37.0, batch, cfg)
    assert wrong.components["top8_swap"] > 0
    assert torch.equal(
        wrong.components["top8_swap"], shifted.components["top8_swap"]
    )

    exact = scores.clone()
    exact[..., 15] = 2.0
    exact[..., 0] = -1.0
    repaired = full_router_forecaster_loss(exact, batch, cfg)
    assert repaired.components["top8_swap"].item() == 0.0
    assert repaired.total.item() == 0.0


def test_top8_swap_gradients_only_repair_missing_and_intruding_experts() -> None:
    config = _config()
    batch = _batch(config, batch_size=1)
    target = torch.arange(8, 16).view(1, 1, 1, 8).expand(1, 8, 3, 8)
    batch["target_top8"] = target
    scores = torch.full_like(batch["teacher_router_scores"], -1.0)
    scores[..., 8:15] = 2.0
    scores[..., 15] = 0.0
    scores[..., 0] = 1.0
    scores.requires_grad_()
    loss = full_router_forecaster_loss(
        scores,
        batch,
        FullRouterLossConfig(
            router_kl=0.0,
            boundary=0.0,
            predicted_boundary=0.0,
            top8_swap=1.0,
            full_membership=0.0,
            centered_score=0.0,
            hard_negative_end_rank=16,
            predicted_negative_count=4,
        ),
    )
    loss.total.backward()
    assert scores.grad is not None
    assert torch.all(scores.grad[..., 15] < 0)
    assert torch.all(scores.grad[..., 0] > 0)
    untouched = scores.grad.clone()
    untouched[..., 15] = 0
    untouched[..., 0] = 0
    assert torch.count_nonzero(untouched) == 0


def test_production_default_parameter_count_is_24gib_practical() -> None:
    config = JSpaceRouterForecasterConfig()
    model = JSpaceFullRouterForecaster(config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    # This bound is deliberately generous but prevents an accidental dense
    # D->256 head per (h,l), which would add ~63M parameters by itself.
    assert parameters < 20_000_000
    assert model.output_head.up.shape == (8, 40, 64, 256)


def test_dual_stream_gate_is_post_projection_per_cell_and_epoch0_passthrough() -> None:
    config = _config(secondary_j_width=6)
    model = JSpaceFullRouterForecaster(config).eval()
    batch = _batch(config)
    assert model.history_encoder.secondary_target_projection is not None
    assert model.history_encoder.target_stream_gate is not None
    assert model.history_encoder.j_projection[1].in_features == config.j_width
    assert (
        model.history_encoder.secondary_target_projection[1].in_features
        == config.secondary_j_width
    )
    with torch.inference_mode():
        output = model(batch)
    assert torch.equal(output.future_router_scores, batch["base_router_scores"])
    assert torch.count_nonzero(output.delta) == 0
    assert output.target_stream_weights.shape == (2, 3, 3, 2)
    assert torch.allclose(
        output.target_stream_weights,
        torch.full_like(output.target_stream_weights, 0.5),
        atol=1e-7,
    )

    gate = model.history_encoder.target_stream_gate[-1]
    with torch.no_grad():
        gate.bias.copy_(torch.tensor([8.0, -8.0]))
    with torch.inference_mode():
        primary = model(batch).target_stream_weights
    assert torch.all(primary[..., 0] > 0.999)
    with torch.no_grad():
        gate.bias.copy_(torch.tensor([-8.0, 8.0]))
    with torch.inference_mode():
        secondary = model(batch).target_stream_weights
    assert torch.all(secondary[..., 1] > 0.999)
    assert torch.allclose(secondary.sum(dim=-1), torch.ones_like(secondary[..., 0]))


def test_single_and_dual_models_fail_closed_on_wrong_stream_contract() -> None:
    single_config = _config()
    single = JSpaceFullRouterForecaster(single_config).eval()
    single_batch = _batch(single_config)
    unexpected = dict(single_batch)
    unexpected["secondary_target_states"] = torch.zeros(2, 3, 3, 6)
    unexpected["secondary_target_mask"] = single_batch["j_mask"].clone()
    with pytest.raises(ValueError, match="explicit dual-stream contract"):
        single(unexpected)

    dual_config = _config(secondary_j_width=6)
    dual = JSpaceFullRouterForecaster(dual_config).eval()
    dual_batch = _batch(dual_config)
    del dual_batch["secondary_target_states"]
    with pytest.raises(KeyError, match="secondary_target_states"):
        dual(dual_batch)

    mismatched = _batch(dual_config)
    mismatched["secondary_target_mask"][0, 0, 0] = False
    with pytest.raises(ValueError, match="share the audited mask"):
        dual(mismatched)


def test_cpu_history_mask_integrity_assertion_is_preserved() -> None:
    config = _config()
    model = JSpaceFullRouterForecaster(config).eval()
    batch = _batch(config, batch_size=1)
    batch["j_mask"].zero_()
    batch["route_mask"].zero_()
    with pytest.raises(ValueError, match="at least one causal history cell"):
        model(batch)


def _legacy_full_router_loss(
    scores: torch.Tensor,
    batch: dict[str, torch.Tensor],
    cfg: FullRouterLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact pre-optimization formula retained as a parity oracle."""

    scores = scores.float()
    teacher = batch["teacher_router_scores"].float()
    target_top8 = batch["target_top8"].long()
    valid = batch["valid_future"].bool()
    tau = float(cfg.temperature)
    kl_rows = F.kl_div(
        torch.log_softmax(scores / tau, dim=-1),
        torch.softmax(teacher / tau, dim=-1),
        reduction="none",
    ).sum(dim=-1) * tau**2
    kl_h = _masked_horizon_mean(kl_rows, valid)

    positives = scores.gather(-1, target_top8)
    membership = torch.zeros_like(scores, dtype=torch.bool)
    membership.scatter_(-1, target_top8, True)
    hard_negative_ids = _stable_masked_topk_ids(
        teacher, membership, cfg.hard_negative_end_rank - 8
    )
    hard_negatives = scores.gather(-1, hard_negative_ids)
    boundary_h = _masked_horizon_mean(
        F.softplus(
            hard_negatives[..., None, :]
            - positives[..., :, None]
            + float(cfg.margin)
        ),
        valid,
    )

    mined_negative_ids = _stable_masked_topk_ids(
        scores, membership, cfg.predicted_negative_count
    )
    mined_negatives = scores.gather(-1, mined_negative_ids)
    predicted_boundary_h = _masked_horizon_mean(
        F.softplus(
            mined_negatives[..., None, :]
            - positives[..., :, None]
            + float(cfg.margin)
        ),
        valid,
    )

    top8_swap_h = _masked_horizon_mean(
        _top8_swap_rows(scores, target_top8, membership, float(cfg.margin)),
        valid,
    )
    positive_bce = F.softplus(-scores).masked_fill(~membership, 0.0).sum(
        dim=-1
    ) / 8.0
    negative_bce = F.softplus(scores).masked_fill(membership, 0.0).sum(
        dim=-1
    ) / max(1, scores.shape[-1] - 8)
    membership_h = _masked_horizon_mean(
        0.5 * (positive_bce + negative_bce), valid
    )

    centered_scores = scores - scores.mean(dim=-1, keepdim=True)
    centered_teacher = teacher - teacher.mean(dim=-1, keepdim=True)
    centered_h = _masked_horizon_mean(
        F.smooth_l1_loss(
            centered_scores, centered_teacher, reduction="none"
        ).mean(dim=-1),
        valid,
    )
    weights = scores.new_tensor(cfg.horizon_weights)
    weights = weights / weights.sum()
    horizon_components = {
        "router_kl": kl_h,
        "boundary": boundary_h,
        "predicted_boundary": predicted_boundary_h,
        "top8_swap": top8_swap_h,
        "full_membership": membership_h,
        "centered_score": centered_h,
    }
    components = {
        name: (values * weights).sum()
        for name, values in horizon_components.items()
    }
    coefficients = {
        "router_kl": cfg.router_kl,
        "boundary": cfg.boundary,
        "predicted_boundary": cfg.predicted_boundary,
        "top8_swap": cfg.top8_swap,
        "full_membership": cfg.full_membership,
        "centered_score": cfg.centered_score,
    }
    total = sum(components[name] * coefficients[name] for name in components)
    return total, components


def test_mixed_loss_and_gradients_exactly_match_legacy_formula() -> None:
    config = _config()
    batch = _batch(config, batch_size=1)
    batch["teacher_router_scores"].zero_()
    batch["target_top8"] = torch.arange(8, 16).view(1, 1, 1, 8).expand(
        1, config.horizons, config.layers, 8
    )
    source = torch.linspace(
        -1.0, 1.0, config.experts, dtype=torch.float32
    ).view(1, 1, 1, -1).expand_as(batch["teacher_router_scores"]).clone()
    # Exact ties exercise expert-ID ordering in both predicted consumers.
    source[..., 0:4] = 0.5
    source[..., 8:12] = 0.5
    cfg = FullRouterLossConfig(
        router_kl=0.2,
        boundary=0.3,
        predicted_boundary=0.4,
        top8_swap=0.5,
        full_membership=0.6,
        centered_score=0.7,
        hard_negative_end_rank=16,
        predicted_negative_count=4,
        margin=0.125,
    )
    legacy_scores = source.clone().requires_grad_(True)
    optimized_scores = source.clone().requires_grad_(True)
    legacy_total, legacy_components = _legacy_full_router_loss(
        legacy_scores, batch, cfg
    )
    optimized = full_router_forecaster_loss(optimized_scores, batch, cfg)
    assert torch.equal(optimized.total, legacy_total)
    for name, expected in legacy_components.items():
        assert torch.equal(optimized.components[name], expected), name
    legacy_total.backward()
    optimized.total.backward()
    assert torch.equal(optimized_scores.grad, legacy_scores.grad)


def test_zero_weight_branches_are_typed_zeros_with_legacy_gradient_parity() -> None:
    config = _config()
    batch = _batch(config, batch_size=1)
    batch["target_top8"] = torch.arange(8, 16).view(1, 1, 1, 8).expand(
        1, config.horizons, config.layers, 8
    )
    source = torch.zeros_like(batch["teacher_router_scores"])
    source[..., 0:8] = 1.0
    cfg = FullRouterLossConfig(
        router_kl=0.0,
        boundary=0.0,
        predicted_boundary=0.0,
        top8_swap=1.0,
        full_membership=0.0,
        centered_score=0.0,
        hard_negative_end_rank=16,
        predicted_negative_count=4,
    )
    legacy_scores = source.clone().requires_grad_(True)
    optimized_scores = source.clone().requires_grad_(True)
    legacy_total, _legacy_components = _legacy_full_router_loss(
        legacy_scores, batch, cfg
    )
    label_only_batch = dict(batch)
    del label_only_batch["teacher_router_scores"]
    optimized = full_router_forecaster_loss(
        optimized_scores, label_only_batch, cfg
    )
    for name in (
        "router_kl",
        "boundary",
        "predicted_boundary",
        "full_membership",
        "centered_score",
    ):
        value = optimized.components[name]
        assert value.item() == 0.0
        assert value.dtype == optimized_scores.dtype
        assert value.device == optimized_scores.device
    assert not optimized.metric_values.requires_grad
    assert optimized.metric_values.device == optimized_scores.device
    assert torch.equal(optimized.total, legacy_total)
    legacy_total.backward()
    optimized.total.backward()
    assert torch.equal(optimized_scores.grad, legacy_scores.grad)


def test_shared_predicted_ranking_uses_one_stable_sort_on_adversarial_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    batch = _batch(config, batch_size=1)
    del batch["teacher_router_scores"]
    batch["target_top8"] = torch.arange(8, 16).view(1, 1, 1, 8).expand(
        1, config.horizons, config.layers, 8
    )
    scores = torch.zeros(1, config.horizons, config.layers, config.experts)
    original = torch.argsort
    orders: list[torch.Tensor] = []

    def recording_argsort(*args: object, **kwargs: object) -> torch.Tensor:
        result = original(*args, **kwargs)
        orders.append(result.detach().clone())
        return result

    monkeypatch.setattr(torch, "argsort", recording_argsort)
    loss = full_router_forecaster_loss(
        scores,
        batch,
        FullRouterLossConfig(
            router_kl=0.0,
            boundary=0.0,
            predicted_boundary=1.0,
            top8_swap=1.0,
            full_membership=0.0,
            centered_score=0.0,
            hard_negative_end_rank=16,
            predicted_negative_count=4,
        ),
    )
    assert len(orders) == 1
    expected = torch.arange(8).view(1, 1, 1, 8).expand(
        1, config.horizons, config.layers, 8
    )
    assert torch.equal(orders[0][..., :8], expected)
    assert loss.total > 0


def test_loss_hot_path_keeps_packed_metrics_on_device() -> None:
    source = getsource(full_router_forecaster_loss)
    assert ".cpu()" not in source
    config = _config()
    batch = _batch(config, batch_size=1)
    scores = batch["teacher_router_scores"].clone().requires_grad_(True)
    loss = full_router_forecaster_loss(
        scores,
        batch,
        FullRouterLossConfig(
            hard_negative_end_rank=16, predicted_negative_count=4
        ),
    )
    assert loss.metric_values.device == scores.device
    assert not loss.metric_values.requires_grad
    expected = dict(
        zip(
            loss.metric_names,
            (float(value) for value in loss.metric_values.tolist()),
        )
    )
    assert loss.metrics == expected


def test_trust_region_defaults_preserve_legacy_schema_and_validate_weights() -> None:
    cfg = FullRouterLossConfig(
        hard_negative_end_rank=16, predicted_negative_count=4
    )
    serialized = cfg.to_dict()
    for name in ("base_kl", "delta_l2", "relative_regret"):
        assert name not in serialized
        for invalid in (-0.01, float("nan"), float("inf"), -float("inf")):
            with pytest.raises(ValueError, match="finite and non-negative"):
                replace(cfg, **{name: invalid}).validate(8, 16)

    enabled = replace(cfg, base_kl=0.2, delta_l2=0.3, relative_regret=0.4)
    assert enabled.to_dict()["base_kl"] == 0.2
    assert enabled.to_dict()["delta_l2"] == 0.3
    assert enabled.to_dict()["relative_regret"] == 0.4


def test_trust_region_terms_match_explicit_formulas_masks_and_gradients() -> None:
    config = _config()
    base = torch.linspace(-1.25, 1.5, config.experts).view(1, 1, 1, -1)
    base = base.expand(2, config.horizons, config.layers, -1).clone()
    delta = (
        torch.sin(torch.arange(base.numel(), dtype=torch.float32))
        .reshape_as(base)
        .mul(0.2)
    )
    scores = (base + delta).requires_grad_(True)
    target_top8 = torch.arange(8, 16).view(1, 1, 1, 8).expand(
        2, config.horizons, config.layers, 8
    )
    valid = torch.ones(2, config.horizons, dtype=torch.bool)
    valid[1, 2] = False
    batch = {
        "base_router_scores": base,
        "target_top8": target_top8,
        "valid_future": valid,
    }
    cfg = FullRouterLossConfig(
        temperature=1.7,
        router_kl=0.0,
        boundary=0.0,
        predicted_boundary=0.0,
        top8_swap=0.0,
        full_membership=0.0,
        centered_score=0.0,
        base_kl=0.7,
        delta_l2=0.3,
        relative_regret=0.5,
        hard_negative_end_rank=16,
        predicted_negative_count=4,
        margin=0.25,
    )
    actual = full_router_forecaster_loss(scores, batch, cfg)

    tau = cfg.temperature
    expected_base_kl_h = _masked_horizon_mean(
        F.kl_div(
            torch.log_softmax(scores / tau, dim=-1),
            torch.softmax(base / tau, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        * tau**2,
        valid,
    )
    centered_delta = delta - delta.mean(dim=-1, keepdim=True)
    expected_delta_l2_h = _masked_horizon_mean(
        centered_delta.square().mean(dim=-1), valid
    )
    membership = torch.zeros_like(base, dtype=torch.bool)
    membership.scatter_(-1, target_top8, True)
    negative_ids = _stable_masked_topk_ids(
        base.detach(), membership, cfg.predicted_negative_count
    )
    new_boundary = F.softplus(
        scores.gather(-1, negative_ids)[..., None, :]
        - scores.gather(-1, target_top8)[..., :, None]
        + cfg.margin
    ).mean(dim=(-2, -1))
    old_boundary = F.softplus(
        base.gather(-1, negative_ids)[..., None, :]
        - base.gather(-1, target_top8)[..., :, None]
        + cfg.margin
    ).mean(dim=(-2, -1))
    expected_regret_h = _masked_horizon_mean(
        F.relu(new_boundary - old_boundary), valid
    )
    horizon_weights = torch.tensor(cfg.horizon_weights)
    horizon_weights /= horizon_weights.sum()
    expected = {
        "base_kl": (expected_base_kl_h * horizon_weights).sum(),
        "delta_l2": (expected_delta_l2_h * horizon_weights).sum(),
        "relative_regret": (expected_regret_h * horizon_weights).sum(),
    }
    for name, value in expected.items():
        assert torch.allclose(actual.components[name], value, atol=1e-7), name
    expected_total = sum(
        expected[name] * getattr(cfg, name) for name in expected
    )
    assert torch.allclose(actual.total, expected_total, atol=1e-7)
    assert all(name in actual.metric_names for name in expected)
    assert all(f"{name}_h8" in actual.metric_names for name in expected)
    actual.total.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad)
    assert torch.count_nonzero(scores.grad[1, 2]) == 0


def test_trust_region_is_shift_invariant_and_ignores_invalid_rows() -> None:
    config = _config()
    batch = _batch(config, batch_size=2)
    del batch["teacher_router_scores"]
    batch["valid_future"][1, 3] = False
    scores = torch.randn_like(batch["base_router_scores"])
    cfg = FullRouterLossConfig(
        router_kl=0.0,
        boundary=0.0,
        predicted_boundary=0.0,
        top8_swap=0.0,
        full_membership=0.0,
        centered_score=0.0,
        base_kl=1.0,
        delta_l2=1.0,
        relative_regret=1.0,
        hard_negative_end_rank=16,
        predicted_negative_count=4,
    )
    original = full_router_forecaster_loss(scores, batch, cfg)
    row_shift = torch.randn(*scores.shape[:-1], 1) * 5.0
    shifted = full_router_forecaster_loss(scores + row_shift, batch, cfg)
    for name in ("base_kl", "delta_l2", "relative_regret"):
        assert torch.allclose(
            original.components[name], shifted.components[name], atol=2e-6
        ), name

    corrupted = scores.clone()
    corrupted[1, 3].fill_(1.0e6)
    ignored = full_router_forecaster_loss(corrupted, batch, cfg)
    for name in ("base_kl", "delta_l2", "relative_regret"):
        assert torch.equal(original.components[name], ignored.components[name]), name


def test_relative_regret_uses_detached_base_fixed_negatives_and_repair_gradients() -> None:
    config = _config()
    base = torch.zeros(
        1,
        config.horizons,
        config.layers,
        config.experts,
        requires_grad=True,
    )
    target = torch.arange(8, 16).view(1, 1, 1, 8).expand(
        1, config.horizons, config.layers, 8
    )
    with torch.no_grad():
        base[..., 8:16] = 2.0
        base[..., :8] = 1.0
    scores = base.detach().clone()
    scores[..., 8:16] -= 2.0
    scores.requires_grad_()
    batch = {
        "base_router_scores": base,
        "target_top8": target,
        "valid_future": torch.ones(1, config.horizons, dtype=torch.bool),
    }
    cfg = FullRouterLossConfig(
        router_kl=0.0,
        boundary=0.0,
        predicted_boundary=0.0,
        full_membership=0.0,
        centered_score=0.0,
        relative_regret=1.0,
        hard_negative_end_rank=16,
        predicted_negative_count=4,
    )
    result = full_router_forecaster_loss(scores, batch, cfg)
    assert result.components["relative_regret"] > 0
    result.total.backward()
    assert base.grad is None
    assert torch.all(scores.grad[..., 8:16] < 0)
    assert torch.all(scores.grad[..., :4] > 0)
    assert torch.count_nonzero(scores.grad[..., 4:8]) == 0


def test_disabled_trust_region_does_not_read_base_or_change_legacy_metrics() -> None:
    config = _config()
    batch = _batch(config, batch_size=1)
    del batch["base_router_scores"]
    scores = batch["teacher_router_scores"].clone().requires_grad_(True)
    cfg = FullRouterLossConfig(
        hard_negative_end_rank=16, predicted_negative_count=4
    )
    result = full_router_forecaster_loss(scores, batch, cfg)
    for name in ("base_kl", "delta_l2", "relative_regret"):
        assert name not in result.components
        assert name not in result.metric_names
        assert not any(metric.startswith(f"{name}_h") for metric in result.metric_names)
    result.total.backward()
    assert scores.grad is not None

    trust_only = replace(
        cfg,
        router_kl=0.0,
        boundary=0.0,
        predicted_boundary=0.0,
        full_membership=0.0,
        centered_score=0.0,
    )
    for name in ("base_kl", "delta_l2", "relative_regret"):
        with pytest.raises(KeyError, match="base_router_scores"):
            full_router_forecaster_loss(
                scores, batch, replace(trust_only, **{name: 1.0})
            )
