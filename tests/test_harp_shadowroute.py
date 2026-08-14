from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from harp_rtt.shadow_cache import (
    ShadowNodeResult,
    ShadowTreeRunner,
    validate_tree_topology,
)
from harp_rtt.shadow_capture import (
    ShadowCaptureDimensions,
    assert_shadow_labels_allowed,
    empty_shadow_capture,
    estimated_shadow_capture_bytes,
    validate_shadow_capture,
)
from harp_rtt.shadow_checkpoint import (
    EXPECTED_LAYER_TYPES,
    ShadowTargetContract,
    validate_routed_expert_inventory,
    validate_shadow_target_config,
)
from harp_rtt.shadow_expert import (
    ExactTop1PlusDraftExperts,
    IndexedShadowExperts,
    ShadowExpertConfig,
    SharedResidualExperts,
    SwiGLUDraftExpert,
    shadow_pool_parameter_count,
    target_neuron_importance,
    target_selected_expert_outputs,
)
from harp_rtt.shadow_route import raw_mtp_prior_mixture
from harp_rtt.shadow_training import (
    s0_scale_gate,
    selected_expert_distillation_loss,
    shadow_route_objective,
    teacher_state_reset_interval,
)


class TinyNativeExperts(nn.Module):
    def __init__(self, experts: int, width: int) -> None:
        super().__init__()
        self.effect = nn.Parameter(
            torch.arange(1, experts + 1, dtype=torch.float32)[:, None]
            .expand(experts, width)
            .clone()
        )

    def forward(self, hidden, ids, weights):
        values = self.effect[ids] * hidden[:, None]
        return (values * weights[..., None]).sum(1)


class ConstantFallback(nn.Module):
    def __init__(self, width: int, value: float) -> None:
        super().__init__()
        self.width = width
        self.value = value

    def forward(self, hidden, ids, weights):
        del ids, weights
        return torch.full_like(hidden, self.value)


def test_exact_top1_plus_draft_uses_only_top1_native():
    native = TinyNativeExperts(4, 3)
    draft = SwiGLUDraftExpert(3, 2)
    for parameter in draft.parameters():
        nn.init.zeros_(parameter)
    module = ExactTop1PlusDraftExperts(native, draft, experts=4)
    hidden = torch.ones(2, 3)
    ids = torch.tensor([[2, 1], [0, 3]])
    weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]])
    output = module(hidden, ids, weights)
    assert torch.allclose(output[0], torch.full((3,), 2.25))
    assert torch.allclose(output[1], torch.full((3,), 0.6))


def test_indexed_shadow_execution_and_gradient_ownership():
    config = ShadowExpertConfig(
        hidden_width=3, experts=4, exact_k=2, shadow_width=1,
        target_intermediate_width=2,
    )
    module = IndexedShadowExperts(config, fallback=ConstantFallback(3, 9.0))
    module.trained_experts[:] = True
    with torch.no_grad():
        module.gate_up_proj.fill_(1.0)
        module.down_proj.fill_(1.0)
    hidden = torch.ones(1, 3, requires_grad=True)
    ids = torch.tensor([[0, 2]])
    weights = torch.tensor([[0.25, 0.75]])
    output = module(hidden, ids, weights)
    expected_unit = torch.nn.functional.silu(torch.tensor(3.0)) * 3.0
    assert torch.allclose(output, torch.full((1, 3), expected_unit))
    output.sum().backward()
    assert module.gate_up_proj.grad is not None
    assert module.gate_up_proj.grad[0].abs().sum() > 0
    assert module.gate_up_proj.grad[2].abs().sum() > 0
    assert module.gate_up_proj.grad[1].abs().sum() == 0
    assert module.gate_up_proj.grad[3].abs().sum() == 0


def test_indexed_shadow_untrained_route_uses_whole_layer_fallback():
    config = ShadowExpertConfig(
        hidden_width=3, experts=4, exact_k=2, shadow_width=1,
        target_intermediate_width=2,
    )
    module = IndexedShadowExperts(config, fallback=ConstantFallback(3, 7.0))
    module.trained_experts[0] = True
    output = module(
        torch.ones(1, 3),
        torch.tensor([[0, 1]]),
        torch.tensor([[0.9, 0.1]]),
    )
    assert torch.equal(output, torch.full((1, 3), 7.0))


def test_target_neuron_initialization_uses_stable_importance_order():
    config = ShadowExpertConfig(
        hidden_width=2, experts=2, exact_k=1, shadow_width=1,
        target_intermediate_width=3,
    )
    module = IndexedShadowExperts(config)
    gate_up = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
    down = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)
    importance = torch.tensor([[1.0, 3.0, 3.0], [4.0, 2.0, 1.0]])
    selected = module.initialize_from_target_neurons(gate_up, down, importance)
    assert selected.tolist() == [[1], [0]]
    assert torch.equal(module.gate_up_proj[0, 0], gate_up[0, 1])
    assert torch.equal(module.gate_up_proj[0, 1], gate_up[0, 4])
    assert torch.equal(module.down_proj[0, :, 0], down[0, :, 1])


def test_target_selected_expert_outputs_match_swiglu_and_static_importance():
    hidden = torch.tensor([[1.0, 2.0]])
    ids = torch.tensor([[1, 0]])
    gate_up = torch.zeros(2, 4, 2)
    down = torch.zeros(2, 2, 2)
    gate_up[1, :2] = torch.eye(2)
    gate_up[1, 2:] = torch.eye(2)
    down[1] = torch.eye(2)
    outputs = target_selected_expert_outputs(hidden, ids, gate_up, down)
    assert torch.allclose(outputs[0, 0], torch.nn.functional.silu(hidden) * hidden)
    assert torch.equal(outputs[0, 1], torch.zeros(2))
    importance = target_neuron_importance(gate_up, down)
    assert importance.shape == (2, 2)
    assert torch.equal(importance[0], torch.zeros(2))
    assert bool((importance[1] > 0).all())


def test_shadow_parameter_counts_match_formal_plan():
    assert shadow_pool_parameter_count() == 1_006_632_960
    assert shadow_pool_parameter_count(
        layers=40, experts=1, hidden_width=2048, shadow_width=128
    ) == 31_457_280


def test_tree_runner_isolates_siblings_and_processes_h1_once():
    root = {"tokens": [10]}

    def step(token, cache, depth):
        cache["tokens"].append(token)
        value = float(sum(cache["tokens"]))
        return ShadowNodeResult(
            router_logits=torch.full((2, 5), value),
            selected_ids=torch.tensor([[0, 1], [0, 1]]),
            selected_weights=torch.tensor([[0.6, 0.4], [0.6, 0.4]]),
            hidden_state=torch.full((2, 3), float(depth)),
            cache=cache,
        )

    result = ShadowTreeRunner(step, max_depth=4).run(
        root_cache=root,
        token_ids=torch.tensor([11, 12, 13]),
        parent_indices=torch.tensor([-1, 0, 0]),
        node_mask=torch.tensor([True, True, True]),
    )
    assert root == {"tokens": [10]}
    assert result.caches[0]["tokens"] == [10, 11]
    assert result.caches[1]["tokens"] == [10, 11, 12]
    assert result.caches[2]["tokens"] == [10, 11, 13]
    assert result.valid.all()


def test_tree_topology_rejects_sibling_as_future_parent():
    with pytest.raises(ValueError, match="precede"):
        validate_tree_topology(
            torch.tensor([-1, 2, 0]), torch.ones(3, dtype=torch.bool)
        )


def test_raw_mtp_prior_preserves_uncaptured_other_mass():
    scores = torch.zeros(1, 3, 1, 5)
    scores[0, 0, 0, :2] = 8
    scores[0, 1, 0, 2:4] = 8
    scores[0, 2, 0, [1, 4]] = 8
    depths = torch.tensor([[1, 2, 2]])
    logp = torch.tensor([[0.0, torch.log(torch.tensor(0.3)), torch.log(torch.tensor(0.2))]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    anchor = torch.zeros(1, 2, 1, 5)
    anchor[..., :2] = 1.0
    output = raw_mtp_prior_mixture(
        scores, depths, logp, valid, anchor, exact_k=2, horizons=2
    )
    assert torch.allclose(output.branch_weights[0, 1], torch.tensor([0.0, 0.3, 0.2]))
    assert torch.allclose(output.other_weights[0, 1], torch.tensor(0.5))
    assert torch.allclose(output.mixture_marginals.sum(-1), torch.full((1, 2, 1), 2.0))
    assert output.selected_ids.shape == (1, 2, 1, 2)


def _valid_tiny_capture():
    dims = ShadowCaptureDimensions(
        nodes=4, layers=3, hidden_width=4, experts=5, exact_k=2,
        router_rank=3,
    )
    values = empty_shadow_capture(dims)
    values["node_mask"][:3] = True
    values["parent_local_indices"][:3] = torch.tensor([-1, 0, 1])
    values["valid"][:3] = True
    values["selected_ids"][:3] = torch.tensor([0, 1], dtype=torch.int32)
    values["selected_weights"][:3] = torch.tensor([0.6, 0.4], dtype=torch.bfloat16)
    return dims, values


def test_shadow_capture_contract_and_privacy():
    dims, values = _valid_tiny_capture()
    validate_shadow_capture(values, dims)
    assert_shadow_labels_allowed(split="train", training=True, enabled=True)
    with pytest.raises(PermissionError):
        assert_shadow_labels_allowed(split="validation", training=True, enabled=True)
    with pytest.raises(PermissionError):
        assert_shadow_labels_allowed(split="train", training=False, enabled=True)
    values["selected_ids"][3, 0, 0] = 1
    with pytest.raises(ValueError, match="padded"):
        validate_shadow_capture(values, dims)


def test_capture_storage_estimate_scales_by_source_and_nodes():
    dims = ShadowCaptureDimensions(
        nodes=2, layers=3, hidden_width=4, experts=5, exact_k=2,
        router_rank=3,
    )
    one = estimated_shadow_capture_bytes(
        1, dimensions=dims, raw_audit_fraction=0.0
    )
    assert estimated_shadow_capture_bytes(
        7, dimensions=dims, raw_audit_fraction=0.0
    ) == 7 * one


def test_shadow_objective_is_finite_and_backpropagates():
    predicted = torch.randn(1, 2, 5, requires_grad=True)
    target = torch.randn(1, 2, 5)
    predicted_state = torch.randn(1, 2, 4, requires_grad=True)
    target_state = torch.randn(1, 2, 4)
    ids = target.topk(2, dim=-1).indices
    valid = torch.tensor([[True, True]])
    output = shadow_route_objective(
        predicted_routed_delta=predicted_state,
        target_routed_delta=target_state,
        predicted_hidden=predicted_state,
        target_hidden=target_state,
        predicted_router_logits=predicted,
        target_router_logits=target,
        target_selected_ids=ids,
        valid=valid,
    )
    assert torch.isfinite(output.total)
    output.total.backward()
    assert predicted.grad is not None and predicted.grad.abs().sum() > 0
    assert predicted_state.grad is not None and predicted_state.grad.abs().sum() > 0


def test_selected_expert_distillation_uses_execution_weights():
    prediction = torch.tensor([[[1.0], [3.0]]])
    target = torch.zeros_like(prediction)
    weights = torch.tensor([[0.75, 0.25]])
    loss = selected_expert_distillation_loss(
        prediction, target, weights, torch.tensor([True])
    )
    expected = 0.75 * 0.5 + 0.25 * 2.5
    assert torch.allclose(loss, torch.tensor(expected))


def test_large_gain_and_reset_curriculum():
    assert s0_scale_gate(100_000, 0.70).passed
    assert not s0_scale_gate(100_000, 0.65).passed
    assert s0_scale_gate(500_000, 0.85).passed
    assert [teacher_state_reset_interval(value) for value in (0, .2, .4, .6, .8, 1)] == [
        1, 2, 4, 8, 40, 40,
    ]


def test_exact_target_config_and_routed_inventory():
    contract = ShadowTargetContract()
    config = {
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": .25,
        "rms_norm_eps": 1e-6,
        "layer_types": EXPECTED_LAYER_TYPES,
        "attention_bias": False,
        "attention_dropout": 0.0,
    }
    validate_shadow_target_config(config, contract)
    config["layer_types"] = tuple(reversed(EXPECTED_LAYER_TYPES))
    with pytest.raises(ValueError, match="layer_types"):
        validate_shadow_target_config(config, contract)
    inventory = {
        f"model.language_model.layers.{layer}.mlp.experts.{role}": "shard"
        for layer in range(40)
        for role in ("gate_up_proj", "down_proj")
    }
    checkpoint = SimpleNamespace(
        routed_expert_keys=tuple(sorted(inventory))
    )
    assert len(validate_routed_expert_inventory(checkpoint)) == 80
