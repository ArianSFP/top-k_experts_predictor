from __future__ import annotations

import pytest
import torch
from torch import nn

from harp_rtt.exact_k import exact_set_nll
from harp_rtt.geometry import build_centered_router_geometry
from harp_rtt.model import HARPRTTConfig, HARPRTTTeacher
from harp_rtt.model.heads import exact_projected_marginals
from harp_rtt.schema import HARPRTTDimensions
from harp_rtt.training import autocast_context


class _UnusedAnchor(nn.Module):
    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        scores = inputs["scores"]
        return {
            "future_router_scores": scores,
            "future_inclusion_probabilities": torch.sigmoid(scores),
        }


class _ForbiddenTargets(dict[str, object]):
    """Sentinel proving the model-side rich adapter never reads labels."""

    def __getitem__(self, key: str) -> object:
        raise AssertionError(f"model attempted to read target field {key!r}")

    def get(self, key: str, default: object = None) -> object:
        raise AssertionError(f"model attempted to read target field {key!r}")

    def __iter__(self):
        raise AssertionError("model attempted to iterate target fields")

    def __len__(self) -> int:
        raise AssertionError("model attempted to inspect target fields")


def _config() -> HARPRTTConfig:
    return HARPRTTConfig(
        experts=12,
        layers=3,
        anchor_horizons=8,
        active_horizons=4,
        route_history=3,
        exact_k=3,
        candidate_width=6,
        max_tree_nodes=4,
        max_tree_depth=4,
        max_tree_branches=8,
        router_rank=6,
        target_control_width=6,
        target_content_width=6,
        target_post_attention_width=6,
        target_post_moe_width=6,
        target_routed_width=6,
        target_shared_width=6,
        exact_token_width=6,
        final_hidden_width=6,
        tree_hidden_width=6,
        tree_fused_width=6,
        tree_router_input_width=6,
        tree_token_width=6,
        tree_metadata_width=4,
        route_summary_width=4,
        route_key_width=4,
        target_width=16,
        route_width=16,
        tree_width=16,
        model_width=16,
        reranker_width=16,
        object_residual_width=4,
        route_ffn_width=32,
        target_ffn_width=32,
        tree_ffn_width=32,
        decoder_ffn_width=32,
        reranker_ffn_width=32,
        attention_heads=4,
        temporal_blocks=1,
        route_layer_blocks=1,
        target_blocks=1,
        tree_blocks=2,
        decoder_blocks=6,
        trajectory_rounds=2,
        reranker_set_blocks=1,
        reranker_summary_blocks=1,
        dropout=0.0,
    )


def _model_and_inputs(batch: int = 2) -> tuple[HARPRTTTeacher, dict[str, object]]:
    torch.manual_seed(17)
    config = _config()
    router_weights = torch.randn(config.layers, config.experts, 6)
    geometry = build_centered_router_geometry(
        router_weights, relative_rank_threshold=1e-7
    )
    assert geometry.maximum_rank == config.router_rank
    model = HARPRTTTeacher(
        _UnusedAnchor(), config, geometry, token_embedding=torch.randn(32, 6)
    ).eval()
    anchor_scores = torch.randn(
        batch,
        config.anchor_horizons,
        config.layers,
        config.experts,
    )
    route_history = torch.randn(
        batch,
        config.layers,
        config.route_history,
        config.experts,
    )
    selected_values, selected_ids = torch.topk(
        route_history, config.exact_k, dim=-1
    )
    tree_parent = torch.tensor([-1, 0, 0, 1])[None].expand(batch, -1).clone()
    tree_depth = torch.tensor([0, 1, 1, 2])[None].expand(batch, -1).clone()
    tree_branch = torch.tensor([0, 1, 2, 1])[None].expand(batch, -1).clone()
    tree_horizon_mask = torch.ones(
        batch, config.max_tree_nodes, config.active_horizons, dtype=torch.bool
    )
    inputs: dict[str, object] = {
        "anchor_outputs": {
            "future_router_scores": anchor_scores,
            "future_inclusion_probabilities": torch.sigmoid(anchor_scores),
        },
        "route_history": route_history,
        "route_available": torch.ones(
            batch, config.layers, config.route_history, dtype=torch.bool
        ),
        "history_selected_ids": selected_ids,
        "history_selected_weights": torch.softmax(selected_values, dim=-1),
        "history_summary": torch.randn(
            batch,
            config.layers,
            config.route_history,
            config.route_summary_width,
        ),
        "target_control": torch.randn(batch, config.layers, 6),
        "target_content": torch.randn(batch, config.layers, 6),
        "target_post_attention": torch.randn(batch, config.layers, 6),
        "target_post_moe": torch.randn(batch, config.layers, 6),
        "target_routed": torch.randn(batch, config.layers, 6),
        "target_shared": torch.randn(batch, config.layers, 6),
        "exact_token_embedding": torch.randn(batch, 6),
        "final_hidden": torch.randn(batch, 6),
        "tree_hidden": torch.randn(batch, config.max_tree_nodes, 6),
        "tree_fused": torch.randn(batch, config.max_tree_nodes, 6),
        "tree_router_input": torch.randn(batch, config.max_tree_nodes, 6),
        "tree_router_logits": torch.randn(
            batch, config.max_tree_nodes, config.experts
        ),
        "tree_token_embeddings": torch.randn(batch, config.max_tree_nodes, 6),
        "tree_metadata": torch.randn(
            batch, config.max_tree_nodes, config.tree_metadata_width
        ),
        "tree_depth_ids": tree_depth,
        "tree_parent_ids": tree_parent,
        "tree_branch_ids": tree_branch,
        "tree_available": torch.ones(
            batch, config.max_tree_nodes, dtype=torch.bool
        ),
        "tree_horizon_mask": tree_horizon_mask,
    }
    return model, inputs


@torch.no_grad()
def test_shapes_exact_cardinality_and_zero_step_anchor_equality() -> None:
    model, inputs = _model_and_inputs()
    outputs = model(**inputs)
    config = model.config
    anchor = inputs["anchor_outputs"]["future_router_scores"]  # type: ignore[index]
    assert torch.equal(outputs["future_router_scores"], anchor)
    assert torch.equal(
        outputs["legacy_h5_h8_scores"], anchor[:, config.active_horizons :]
    )
    assert outputs["active_scores"].shape == (2, 4, 3, 12)
    assert outputs["active_marginals"].shape == (2, 4, 3, 12)
    assert outputs["top8_ids"].shape == (2, 4, 3, 3)
    assert outputs["branch_marginals"].shape == (2, 4, 3, 5, 12)
    assert outputs["branch_posterior_logits"].shape == (2, 4, 5)
    assert torch.isfinite(outputs["branch_posterior_logits"]).all()
    assert outputs["trajectory_round_scores"].shape == (2, 2, 4, 3, 12)
    assert outputs["candidate_ids"].shape == (2, 4, 3, 6)
    assert outputs["candidate_mask"].shape == (2, 4, 3, 6)
    assert outputs["candidate_dense_mask"].shape == (2, 4, 3, 12)
    assert outputs["candidate_causal_history_features"].shape == (2, 4, 3, 6, 13)
    assert outputs["candidate_branch_support_features"].shape == (2, 4, 3, 6, 6)
    assert torch.allclose(
        outputs["active_marginals"].sum(-1),
        torch.full((2, 4, 3), 3.0),
        atol=2e-5,
    )
    dimensions = HARPRTTDimensions(
        layers=3,
        experts=12,
        primary_horizons=4,
        legacy_horizons=8,
        selected_experts=3,
        history_tokens=3,
        tree_nodes=4,
        candidates=6,
        trajectory_rounds=2,
    )
    outputs["structured_output"].validate(dimensions)


@torch.no_grad()
def test_full_model_accepts_bf16_capture_features_under_cpu_autocast() -> None:
    """Exercise every learned input frontier with native capture dtypes."""

    model, inputs = _model_and_inputs(batch=1)
    integer_or_mask = {
        "history_selected_ids",
        "tree_depth_ids",
        "tree_parent_ids",
        "tree_branch_ids",
        "tree_available",
        "tree_horizon_mask",
        "route_available",
    }
    for name, value in tuple(inputs.items()):
        if (
            name not in integer_or_mask
            and isinstance(value, torch.Tensor)
            and value.is_floating_point()
        ):
            inputs[name] = value.to(torch.bfloat16)
    anchor_outputs = inputs["anchor_outputs"]
    assert isinstance(anchor_outputs, dict)
    for name, value in tuple(anchor_outputs.items()):
        assert isinstance(value, torch.Tensor)
        anchor_outputs[name] = value.to(torch.bfloat16)

    with autocast_context("cpu", enabled=True):
        outputs = model(**inputs)

    anchor = anchor_outputs["future_router_scores"]
    assert torch.equal(outputs["future_router_scores"], anchor)
    assert outputs["active_scores"].dtype == torch.float32
    assert outputs["candidate_corrections"].dtype == outputs["active_scores"].dtype
    assert torch.isfinite(outputs["active_scores"]).all()
    assert torch.isfinite(outputs["branch_posterior_logits"]).all()


def test_branch_exact_marginals_preserve_forward_and_train_branch_scores() -> None:
    model, _ = _model_and_inputs(batch=1)
    config = model.config
    scores = torch.randn(
        1,
        config.active_horizons,
        config.layers,
        config.max_tree_nodes,
        config.experts,
        requires_grad=True,
    )
    logits = torch.randn(
        1, config.active_horizons, config.max_tree_nodes, requires_grad=True
    )
    mask = torch.ones_like(logits, dtype=torch.bool)
    mixture = model.branch_mixture(scores, logits, mask)
    with torch.no_grad():
        _, exact, _ = exact_projected_marginals(scores, config.exact_k)
    assert torch.equal(mixture.branch_marginals, exact)
    assert torch.allclose(
        mixture.branch_marginals.sum(-1),
        torch.full_like(mixture.branch_marginals.sum(-1), float(config.exact_k)),
        atol=2e-5,
    )
    assert torch.allclose(
        mixture.mixture_marginals.sum(-1),
        torch.full_like(mixture.mixture_marginals.sum(-1), float(config.exact_k)),
        atol=2e-5,
    )
    # A non-symmetric downstream objective must reach both branch scores and
    # branch posterior logits; exact marginals are not diagnostic-only here.
    coefficients = torch.linspace(
        -1.0, 1.0, config.experts, dtype=mixture.mixture_logits.dtype
    )
    loss = (mixture.mixture_logits * coefficients).sum()
    loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum() > 0
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_primary_set_loss_reaches_hybrid_branch_gates_after_bridge_opens() -> None:
    model, inputs = _model_and_inputs(batch=1)
    with torch.no_grad():
        model.branch_residual.gate.fill_(0.1)
        model.score_head.free_gate.fill_(0.1)
    outputs = model(**inputs)
    outputs["branch_scores"].retain_grad()
    outputs["branch_posterior_logits"].retain_grad()
    labels = torch.arange(model.config.exact_k).view(1, 1, 1, -1).expand(
        1, model.config.active_horizons, model.config.layers, -1
    )
    exact_set_nll(
        outputs["active_scores"], labels, k=model.config.exact_k
    ).backward()
    for gate in (model.score_head.geometry_gate, model.score_head.free_gate):
        assert gate.grad is not None and torch.isfinite(gate.grad).all()
        assert gate.grad.abs().sum() > 0
    assert outputs["branch_scores"].grad is not None
    assert torch.isfinite(outputs["branch_scores"].grad).all()
    assert outputs["branch_scores"].grad.abs().sum() > 0
    assert outputs["branch_posterior_logits"].grad is not None
    assert torch.isfinite(outputs["branch_posterior_logits"].grad).all()
    assert outputs["branch_posterior_logits"].grad.abs().sum() > 0


@torch.no_grad()
def test_tree_node_permutation_only_permutes_branch_diagnostics() -> None:
    model, inputs = _model_and_inputs(batch=1)
    first = model(**inputs)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    second_inputs = dict(inputs)
    for name in (
        "tree_hidden",
        "tree_fused",
        "tree_router_input",
        "tree_router_logits",
        "tree_token_embeddings",
        "tree_metadata",
        "tree_depth_ids",
        "tree_branch_ids",
        "tree_available",
        "tree_horizon_mask",
    ):
        second_inputs[name] = inputs[name][:, permutation]  # type: ignore[index]
    old_parent = inputs["tree_parent_ids"][:, permutation]  # type: ignore[index]
    second_inputs["tree_parent_ids"] = torch.where(
        old_parent < 0, old_parent, inverse[old_parent]
    )
    second = model(**second_inputs)
    assert torch.allclose(first["active_scores"], second["active_scores"], atol=2e-6)
    assert torch.equal(first["candidate_ids"], second["candidate_ids"])
    expected_marginals = torch.cat(
        (
            first["branch_marginals"][..., permutation, :],
            first["branch_marginals"][..., -1:, :],
        ),
        dim=-2,
    )
    assert torch.allclose(expected_marginals, second["branch_marginals"], atol=2e-6)
    assert torch.allclose(
        first["candidate_branch_support_features"],
        second["candidate_branch_support_features"],
        atol=2e-6,
    )


@torch.no_grad()
def test_candidate_permutation_equivariance_and_outside_pool_preservation() -> None:
    model, inputs = _model_and_inputs(batch=1)
    config = model.config
    base = torch.randn(1, 4, 3, 12)
    candidate_ids = torch.arange(config.candidate_width)[None, None, None].expand(
        1, 4, 3, -1
    ).clone()
    evidence = [torch.randn_like(base) for _ in range(7)]
    history = inputs["route_history"]
    endpoint = torch.randn(1, 4, 3, config.model_width)
    mixture = torch.softmax(torch.randn_like(base), dim=-1) * config.exact_k
    support = model.reranker.build_causal_route_support(
        history,
        inputs["history_selected_ids"],
        inputs["history_selected_weights"],
        inputs["route_available"],
    )
    assert support.transition_scores.shape == base.shape
    assert torch.isfinite(support.transition_scores).all()
    assert support.transition_scores.abs().sum() > 0
    branch_scores = torch.randn(1, 4, 3, 4, 12)
    branch_marginals = torch.softmax(branch_scores, dim=-1) * config.exact_k
    branch_weights = torch.softmax(torch.randn(1, 4, 4), dim=-1)
    branch_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    model.reranker.residual_head.weight.fill_(0.1)
    model.reranker.residual_head.bias.fill_(0.2)
    causal_kwargs = {
        "causal_support": support,
        "transition_scores": support.transition_scores,
        "branch_scores": branch_scores,
        "branch_marginals": branch_marginals,
        "branch_weights": branch_weights,
        "branch_mask": branch_mask,
    }
    first = model.reranker(
        base,
        candidate_ids,
        dense_evidence=evidence,
        history_logits=history,
        endpoint_context=endpoint,
        mixture_marginals=mixture,
        **causal_kwargs,
    )
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    second = model.reranker(
        base,
        candidate_ids[..., permutation],
        dense_evidence=evidence,
        history_logits=history,
        endpoint_context=endpoint,
        mixture_marginals=mixture,
        **causal_kwargs,
    )
    assert torch.allclose(first.dense_scores, second.dense_scores, atol=2e-6)
    assert torch.allclose(
        first.candidate_corrections[..., permutation],
        second.candidate_corrections,
        atol=2e-6,
    )
    assert torch.allclose(
        first.causal_history_features[..., permutation, :],
        second.causal_history_features,
        atol=2e-6,
    )
    assert torch.allclose(
        first.branch_support_features[..., permutation, :],
        second.branch_support_features,
        atol=2e-6,
    )
    node_permutation = torch.tensor([2, 0, 3, 1])
    third = model.reranker(
        base,
        candidate_ids,
        dense_evidence=evidence,
        history_logits=history,
        endpoint_context=endpoint,
        mixture_marginals=mixture,
        causal_support=support,
        transition_scores=support.transition_scores,
        branch_scores=branch_scores[..., node_permutation, :],
        branch_marginals=branch_marginals[..., node_permutation, :],
        branch_weights=branch_weights[..., node_permutation],
        branch_mask=branch_mask[..., node_permutation],
    )
    assert torch.allclose(
        first.branch_support_features,
        third.branch_support_features,
        atol=2e-6,
    )
    assert torch.allclose(first.dense_scores, third.dense_scores, atol=2e-6)
    dense_mask = torch.zeros_like(base, dtype=torch.bool).scatter(
        -1, candidate_ids, True
    )
    assert torch.equal(first.dense_scores[~dense_mask], base[~dense_mask])


@torch.no_grad()
def test_trajectory_has_exactly_two_direct_residual_rounds() -> None:
    model, inputs = _model_and_inputs(batch=1)
    outputs = model(**inputs)
    anchor = inputs["anchor_outputs"]["future_router_scores"][:, :4]  # type: ignore[index]
    rounds = outputs["trajectory_round_scores"]
    corrections = outputs["trajectory_corrections"]
    assert rounds.shape[1] == 2
    assert torch.equal(rounds[:, 0], anchor)
    assert torch.equal(rounds[:, 1], anchor)
    assert torch.count_nonzero(corrections) == 0


@torch.no_grad()
def test_nested_collated_dataset_batch_adapter() -> None:
    model, direct = _model_and_inputs(batch=1)
    route = direct["route_history"]
    selected_ids = direct["history_selected_ids"]
    selected_weights = direct["history_selected_weights"]
    summary = direct["history_summary"]
    meta = torch.zeros(1, 4, 13, dtype=torch.long)
    meta[..., 4] = direct["tree_depth_ids"]
    meta[..., 11] = direct["tree_branch_ids"]
    roles = (
        "post_attention_residual_u",
        "normalized_target_router_input_a",
        "post_moe_residual_xplus",
        "routed_expert_output_delta_r",
        "shared_expert_output_delta_s",
        "raw_target_router_logits",
    )
    nested = {
        "inputs": {
            "history": {
                "logits": route.permute(0, 2, 1, 3),
                "selected_ids": selected_ids.permute(0, 2, 1, 3),
                "execution_weights": selected_weights.permute(0, 2, 1, 3),
                "summary": summary.permute(0, 2, 1, 3),
                "available": direct["route_available"].permute(0, 2, 1),
            },
            "current": {
                "normalized_target_router_input_a": direct["target_content"],
                "post_attention_residual_u": direct["target_post_attention"],
                "post_moe_residual_xplus": direct["target_post_moe"],
                "routed_expert_output_delta_r": direct["target_routed"],
                "shared_expert_output_delta_s": direct["target_shared"],
            },
            "current_roles": roles,
            "current_available": torch.tensor(
                [
                    [
                        [True, False, True, False, True, False],
                        [False, True, False, True, False, True],
                        [True, True, False, False, True, True],
                    ]
                ]
            ),
            "exact_next_token_id": torch.tensor([5]),
            "final_hidden": direct["final_hidden"],
            "tree": {
                "states": torch.stack(
                    [
                        direct["tree_fused"],
                        direct["tree_hidden"],
                        direct["tree_router_input"],
                        direct["tree_token_embeddings"],
                    ],
                    dim=2,
                ),
                "router_logits": direct["tree_router_logits"],
                "meta": meta,
                "scalars": direct["tree_metadata"],
                "parent": direct["tree_parent_ids"],
                "mask": direct["tree_available"],
                "horizon_mask": direct["tree_horizon_mask"].transpose(1, 2),
                "path_log_probabilities": torch.tensor(
                    [[0.0, -0.2, -0.5, -0.9]]
                ),
                "local_probabilities": torch.tensor([[1.0, 0.8, 0.7, 0.6]]),
                "source_ready": torch.tensor([[0.0, 0.25, 0.5, 1.0]]),
                "target_positions": direct["tree_depth_ids"],
                "structural_valid": direct["tree_available"],
                "conditioning_classes": torch.tensor([[2, 1, 1, 1]]),
            },
        },
        # Any access raises immediately.  Future router targets and MTP
        # acceptance remain label-only even though the complete batch carries
        # them beside causal inputs.
        "targets": _ForbiddenTargets(),
    }
    # The production adapter executes inside the BF16 training autocast
    # scope.  Frozen V projection and V q reconstruction must nevertheless
    # match the canonical FP32 geometry exactly.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        adapted = model._adapt_rich_batch(nested)
    raw_available = nested["inputs"]["current_available"]
    expected_available = raw_available[..., [1, 1, 0, 2, 3, 4]]
    assert torch.equal(adapted["target_available"], expected_available)
    router_input = nested["inputs"]["current"][
        "normalized_target_router_input_a"
    ]
    expected_control = torch.einsum(
        "bld,ldr->blr", router_input, model.input_basis
    ) * model.rank_mask[None]
    expected_blind = router_input - torch.einsum(
        "blr,ldr->bld", expected_control, model.input_basis
    )
    assert adapted["target_control"].dtype == torch.float32
    assert adapted["target_content"].dtype == torch.float32
    assert torch.equal(adapted["target_control"], expected_control)
    assert torch.equal(adapted["target_content"], expected_blind)
    assert torch.allclose(adapted["target_control"], expected_control, atol=1e-6)
    assert torch.allclose(adapted["target_content"], expected_blind, atol=1e-6)
    # The test config uses four metadata channels, so the named adaptive
    # contract contributes path log-p, local p, readiness, and normalized
    # depth in that order instead of the legacy anonymous scalar vector.
    expected_tree_metadata = torch.stack(
        (
            nested["inputs"]["tree"]["path_log_probabilities"],
            nested["inputs"]["tree"]["local_probabilities"],
            nested["inputs"]["tree"]["source_ready"],
            direct["tree_depth_ids"].float() / model.config.max_tree_depth,
        ),
        dim=-1,
    )
    assert torch.equal(adapted["tree_metadata"], expected_tree_metadata)
    nested["inputs"]["tree"]["adaptive_contract"] = torch.tensor([False])
    legacy_adapted = model._adapt_rich_batch(nested)
    assert torch.equal(legacy_adapted["tree_metadata"], direct["tree_metadata"])
    nested["inputs"]["tree"]["adaptive_contract"] = torch.tensor([True])
    outputs = model(
        batch=nested,
        anchor_outputs=direct["anchor_outputs"],
    )
    anchor = direct["anchor_outputs"]["future_router_scores"]
    assert torch.equal(outputs["future_router_scores"], anchor)
    assert outputs["branch_mask"].shape == (1, 4, 5)


@torch.no_grad()
def test_rich_adapter_compacts_colliding_branch_hashes() -> None:
    model, direct = _model_and_inputs(batch=1)
    route = direct["route_history"]
    meta = torch.zeros(1, 4, 13, dtype=torch.long)
    meta[..., 4] = direct["tree_depth_ids"]
    # These collide under the former modulo-(max_tree_branches + 1) mapping.
    meta[..., 11] = torch.tensor([[1, 10, 1, 10]])
    roles = (
        "post_attention_residual_u",
        "normalized_target_router_input_a",
        "post_moe_residual_xplus",
        "routed_expert_output_delta_r",
        "shared_expert_output_delta_s",
        "raw_target_router_logits",
        "selected_expert_ids",
        "selected_execution_weights",
    )
    nested = {
        "inputs": {
            "history": {
                "logits": route.permute(0, 2, 1, 3),
                "selected_ids": direct["history_selected_ids"].permute(0, 2, 1, 3),
                "execution_weights": direct["history_selected_weights"].permute(0, 2, 1, 3),
                "summary": direct["history_summary"].permute(0, 2, 1, 3),
                "available": direct["route_available"].permute(0, 2, 1),
            },
            "current": {
                "normalized_target_router_input_a": direct["target_content"],
                "post_attention_residual_u": direct["target_post_attention"],
                "post_moe_residual_xplus": direct["target_post_moe"],
                "routed_expert_output_delta_r": direct["target_routed"],
                "shared_expert_output_delta_s": direct["target_shared"],
            },
            "current_roles": roles,
            "current_available": torch.ones(1, 3, len(roles), dtype=torch.bool),
            "exact_next_token_id": torch.tensor([5]),
            "final_hidden": direct["final_hidden"],
            "tree": {
                "states": torch.stack(
                    [
                        direct["tree_fused"],
                        direct["tree_hidden"],
                        direct["tree_router_input"],
                        direct["tree_token_embeddings"],
                    ],
                    dim=2,
                ),
                "router_logits": direct["tree_router_logits"],
                "meta": meta,
                "scalars": direct["tree_metadata"],
                "parent": direct["tree_parent_ids"],
                "mask": direct["tree_available"],
                "horizon_mask": direct["tree_horizon_mask"].transpose(1, 2),
            },
        }
    }
    adapted = model._adapt_rich_batch(nested)
    assert adapted["tree_branch_ids"].tolist() == [[1, 2, 1, 2]]


@torch.no_grad()
def test_empty_tree_is_finite_and_function_preserving() -> None:
    model, inputs = _model_and_inputs(batch=1)
    inputs["tree_available"].zero_()
    inputs["tree_horizon_mask"].zero_()
    outputs = model(**inputs)
    anchor = inputs["anchor_outputs"]["future_router_scores"]
    assert torch.equal(outputs["future_router_scores"], anchor)
    assert torch.isfinite(outputs["active_marginals"]).all()
    assert torch.isfinite(outputs["candidate_branch_support_features"]).all()
    assert torch.equal(outputs["branch_weights"][..., -1], torch.ones(1, 4))
    assert torch.count_nonzero(outputs["branch_weights"][..., :-1]) == 0


@torch.no_grad()
def test_valid_tree_structural_ids_cannot_be_silently_clamped() -> None:
    model, inputs = _model_and_inputs(batch=1)
    inputs["tree_depth_ids"][0, 0] = model.config.max_tree_depth + 1
    with pytest.raises(ValueError, match="depth ID"):
        model(**inputs)
    inputs["tree_depth_ids"][0, 0] = 0
    inputs["tree_branch_ids"][0, 0] = model.config.max_tree_branches + 1
    with pytest.raises(ValueError, match="branch ID"):
        model(**inputs)


@torch.no_grad()
def test_public_schema_model_inputs_adapter() -> None:
    model, direct = _model_and_inputs(batch=1)
    node_ids = torch.tensor([[10, 11, 12, 13]])
    public_inputs = {
        "route_history_logits": direct["route_history"].permute(0, 2, 1, 3),
        "route_history_selected_ids": direct["history_selected_ids"].permute(
            0, 2, 1, 3
        ),
        "route_history_selected_weights": direct[
            "history_selected_weights"
        ].permute(0, 2, 1, 3),
        "route_history_features": direct["history_summary"].permute(0, 2, 1, 3),
        "route_history_mask": direct["route_available"].permute(0, 2, 1),
        "current_router_coordinates": direct["target_control"],
        "current_content_sketch": direct["target_content"],
        "post_attention_states": direct["target_post_attention"],
        "post_moe_states": direct["target_post_moe"],
        "routed_residuals": direct["target_routed"],
        "shared_residuals": direct["target_shared"],
        "exact_next_token_embeddings": direct["exact_token_embedding"],
        "final_hidden_states": direct["final_hidden"],
        "harp_scores": direct["anchor_outputs"]["future_router_scores"],
        "tree": {
            "node_ids": node_ids,
            "parent_ids": torch.tensor([[-1, 10, 10, 11]]),
            "branch_ids": direct["tree_branch_ids"],
            "depths": direct["tree_depth_ids"],
            "target_positions": torch.tensor([[1, 2, 3, 4]]),
            "path_log_probabilities": torch.zeros(1, 4),
            "local_probabilities": torch.ones(1, 4),
            "source_ready": torch.zeros(1, 4),
            "valid": direct["tree_available"],
            "token_embeddings": direct["tree_token_embeddings"],
            "hidden_states": direct["tree_hidden"],
            "fused_states": direct["tree_fused"],
            "router_inputs": direct["tree_router_input"],
            "router_logits": direct["tree_router_logits"],
        },
    }
    outputs = model(batch=public_inputs)
    anchor = direct["anchor_outputs"]["future_router_scores"]
    assert torch.equal(outputs["future_router_scores"], anchor)
    assert outputs["branch_mask"].sum(-1).tolist() == [[2, 2, 2, 2]]

@torch.no_grad()
def test_anytime_budget_masks_later_sibling_features_before_encoding() -> None:
    model, inputs = _model_and_inputs(batch=1)
    model.eval()
    first = model(**inputs, tree_visibility_budget=1)
    changed = dict(inputs)
    for name in (
        "tree_hidden",
        "tree_fused",
        "tree_router_input",
        "tree_router_logits",
        "tree_token_embeddings",
        "tree_metadata",
    ):
        value = inputs[name].clone()  # type: ignore[union-attr]
        value[:, 1:] = value[:, 1:] + 10_000
        changed[name] = value
    second = model(**changed, tree_visibility_budget=1)
    assert first["tree_visibility_budget"] == 1
    assert torch.equal(first["active_scores"], second["active_scores"])
    assert torch.equal(first["candidate_ids"], second["candidate_ids"])
