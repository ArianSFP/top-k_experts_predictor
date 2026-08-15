from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from harp_rtt.shadow_cache import (
    ShadowNodeResult,
    ShadowTreeRunner,
    compact_visible_tree,
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
    BasisDraftConfig,
    ExactTop1PlusDraftExperts,
    IndexedShadowExperts,
    PackedInt4TopKExperts,
    PackedInt4ResidentExperts,
    RouteConditionedBasisExperts,
    RouterVisibleTailControl,
    ShadowExpertConfig,
    SharedResidualExperts,
    SwiGLUDraftExpert,
    basisdraft_parameter_count,
    resident_int4_storage_bytes,
    resident_tail_control_storage_bytes,
    shadow_pool_parameter_count,
    target_neuron_importance,
    dequantize_groupwise_int4,
    quantize_groupwise_int4,
    target_selected_expert_outputs,
)
from harp_rtt.shadow_route import raw_mtp_prior_mixture, shadow_lm_path_posterior
from harp_rtt.resident_policy import (
    allocate_resident_experts,
    allocate_resident_utility,
)
from harp_rtt.shadow_training import (
    s0_scale_gate,
    selected_expert_distillation_loss,
    shadow_route_objective,
    teacher_state_reset_interval,
)
from runpod.train_shadow_experts_local import (
    ShadowGeneratedTokenDataset,
    next_router_agreement_loss,
    objective,
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


class TinyGeneratedSegment:
    def __init__(self) -> None:
        self.target_by_position = {(0, 4): 0, (0, 5): 40, (1, 7): 80}
        self.sequences = [
            {"request_id": "train-a", "split": "train"},
            {"request_id": "holdout-b", "split": "train"},
        ]

    def read(self, kind, rows, role):
        assert kind == "target"
        width = {
            "normalized_target_router_input_a": 4,
            "selected_expert_ids": 2,
            "selected_execution_weights": 2,
            "routed_expert_output_delta_r": 4,
            "post_attention_residual_u": 4,
            "post_moe_residual_xplus": 4,
            "shared_expert_output_delta_s": 4,
            "raw_target_router_logits": 256,
        }[role]
        dtype = torch.int32 if role == "selected_expert_ids" else torch.float32
        return torch.full((len(rows), width), rows[0], dtype=dtype)


def test_generated_token_dataset_is_unique_and_request_allowlisted():
    base = SimpleNamespace(segments=[TinyGeneratedSegment()])
    dataset = ShadowGeneratedTokenDataset(
        base, layer=3, allowed_request_ids={"train-a"}
    )
    assert len(dataset) == 2
    assert dataset.requests == {"train-a"}
    first = dataset[0]
    assert first["inputs"] == {}
    assert first["metadata"]["position"] == 4
    assert first["targets"]["future_router_inputs"].shape == (1, 4)
    assert first["targets"]["future_selected_ids"].shape == (1, 2)
    assert first["targets"]["future_available"].tolist() == [True]
    routed = ShadowGeneratedTokenDataset(
        base,
        layer=3,
        allowed_request_ids={"train-a"},
        next_router_agreement=True,
    )[0]
    states = routed["targets"]["future_states"]
    assert states["next_raw_target_router_logits"].shape == (1, 256)
    assert states["next_selected_expert_ids"].shape == (1, 2)


def test_next_router_agreement_teacher_forcing_and_gradient():
    torch.manual_seed(17)
    target_routed = torch.randn(2, 1, 4)
    predicted = target_routed.clone().requires_grad_(True)
    current_u = torch.randn(2, 1, 4)
    shared = torch.randn(2, 1, 4)
    current_xplus = current_u + shared + target_routed
    attention_delta = torch.randn(2, 1, 4)
    next_u = current_xplus + attention_delta
    norm_weight = torch.randn(4)
    router_weight = torch.randn(256, 4)
    normalized = next_u * torch.rsqrt(
        next_u.square().mean(-1, keepdim=True) + 1e-6
    ) * (1.0 + norm_weight)
    teacher_logits = torch.nn.functional.linear(normalized, router_weight)
    teacher_ids = torch.argsort(
        teacher_logits, dim=-1, descending=True, stable=True
    )[..., :8]
    loss, parts, predicted_logits = next_router_agreement_loss(
        predicted,
        {
            "post_attention_residual_u": current_u,
            "post_moe_residual_xplus": current_xplus,
            "shared_expert_output_delta_s": shared,
            "next_post_attention_residual_u": next_u,
            "next_raw_target_router_logits": teacher_logits,
            "next_selected_expert_ids": teacher_ids,
        },
        torch.ones(2, 1, dtype=torch.bool),
        norm_weight=norm_weight,
        router_weight=router_weight,
    )
    assert torch.allclose(predicted_logits, teacher_logits, atol=1e-5, rtol=1e-5)
    assert float(parts["next_router_kl"].detach()) < 1e-6
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()



def test_resident_v2_objective_trains_tail_and_missing_route_control():
    torch.manual_seed(19)
    hidden = torch.randn(2, 1, 4)
    ids = torch.tensor([[[0, 1]], [[2, 3]]])
    weights = torch.tensor([[[0.6, 0.4]], [[0.55, 0.45]]])
    gate_up = torch.randn(256, 4, 4)
    down = torch.randn(256, 4, 2)
    next_router = torch.randn(256, 4)
    next_norm = torch.randn(4)
    control = RouterVisibleTailControl(
        hidden_width=4, experts=256, rank=2, dtype=torch.float32
    )
    control.bind_next_router(next_router)
    model = PackedInt4ResidentExperts.from_target(
        torch.tensor([0]),
        SharedResidualExperts(SwiGLUDraftExpert(4, 2), experts=256),
        gate_up,
        down,
        exact_k=2,
        group_size=2,
        router_control=control,
    )
    routed_target = torch.randn(2, 1, 4)
    current_u = torch.randn(2, 1, 4)
    shared = torch.randn(2, 1, 4)
    current_xplus = current_u + shared + routed_target
    next_u = current_xplus + torch.randn(2, 1, 4)
    normalized = next_u * torch.rsqrt(
        next_u.square().mean(-1, keepdim=True) + 1e-6
    ) * (1.0 + next_norm)
    teacher_logits = torch.nn.functional.linear(normalized, next_router)
    teacher_ids = torch.argsort(
        teacher_logits, dim=-1, descending=True, stable=True
    )[..., :8]
    host = {
        "metadata": {"request_id": ["a", "b"]},
        "inputs": {},
        "targets": {
            "future_router_inputs": hidden,
            "future_selected_ids": ids,
            "future_execution_weights": weights,
            "future_available": torch.ones(2, 1, dtype=torch.bool),
            "future_states": {
                "routed_expert_output_delta_r": routed_target,
                "post_attention_residual_u": current_u,
                "post_moe_residual_xplus": current_xplus,
                "shared_expert_output_delta_s": shared,
                "next_post_attention_residual_u": next_u,
                "next_raw_target_router_logits": teacher_logits,
                "next_selected_expert_ids": teacher_ids,
            },
        },
    }
    loss, parts, *_ = objective(
        model,
        host,
        mode="resident_int4_tail_control",
        layer=0,
        device=torch.device("cpu"),
        gate_up=gate_up,
        down=down,
        next_norm_weight=next_norm,
        next_router_weight=next_router,
        router_agreement_weight=1.0,
        individual_loss_weight=0.0,
    )
    assert float(parts["tail_huber"]) > 0
    assert float(parts["control_huber"]) > 0
    loss.backward()
    assert model.fallback.draft_expert.gate_up_proj.weight.grad is not None
    assert model.router_control is not None
    assert model.router_control.expert_codes.weight.grad is not None


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


def test_exact_topk_without_draft_uses_requested_native_slots():
    native = TinyNativeExperts(4, 3)
    draft = SwiGLUDraftExpert(3, 2)
    module = ExactTop1PlusDraftExperts(
        native, draft, experts=4, exact_slots=2, draft_scale=0.0
    )
    hidden = torch.ones(1, 3)
    ids = torch.tensor([[2, 1, 0]])
    weights = torch.tensor([[0.5, 0.3, 0.2]])
    expected = native(hidden, ids[:, :2], weights[:, :2])
    assert torch.equal(module(hidden, ids, weights), expected)


def test_basisdraft_split_initializer_exactly_matches_shared_expert():
    torch.manual_seed(23)
    shared = SwiGLUDraftExpert(4, 6)
    basis = RouteConditionedBasisExperts(BasisDraftConfig(
        hidden_width=4,
        experts=5,
        exact_k=3,
        basis_count=3,
        basis_width=2,
    ))
    basis.initialize_from_shared(shared)
    assert torch.count_nonzero(basis.expert_down_proj) == 0
    hidden = torch.randn(7, 4)
    ids = torch.tensor([
        [0, 1, 2], [1, 3, 4], [2, 0, 4], [3, 1, 0],
        [4, 3, 2], [0, 4, 1], [2, 3, 0],
    ])
    weights = torch.rand(7, 3)
    weights = weights / weights.sum(-1, keepdim=True)
    assert torch.allclose(
        basis(hidden, ids, weights), shared(hidden), atol=2e-6, rtol=2e-6
    )
    assert not hasattr(basis, "native_experts")


def test_basisdraft_preserves_expert_identity_weights_and_gradients():
    config = BasisDraftConfig(
        hidden_width=3, experts=4, exact_k=2, basis_count=2, basis_width=1
    )
    module = RouteConditionedBasisExperts(config)
    with torch.no_grad():
        module.gate_up_proj.fill_(1.0)
        module.down_proj.fill_(1.0)
        module.expert_gate_up_proj.fill_(0.5)
        module.expert_down_proj[0].fill_(0.25)
        module.expert_down_proj[3].fill_(0.5)
        module.expert_coefficients.copy_(torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 3.0]
        ])[..., None])
    hidden = torch.ones(1, 3, requires_grad=True)
    ids = torch.tensor([[0, 3]])
    weights = torch.tensor([[0.25, 0.75]])
    individual = module.selected_unweighted(hidden, ids)
    output = module(hidden, ids, weights)
    assert torch.allclose(
        output, (individual * weights[..., None]).sum(1), atol=1e-6
    )
    swapped = module(hidden, ids.flip(-1), weights)
    assert not torch.allclose(output, swapped)
    output.sum().backward()
    assert module.expert_coefficients.grad is not None
    assert module.expert_coefficients.grad[0].abs().sum() > 0
    assert module.expert_coefficients.grad[3].abs().sum() > 0
    assert module.expert_coefficients.grad[1:3].abs().sum() == 0
    assert module.expert_down_proj.grad is not None
    assert module.expert_down_proj.grad[0].abs().sum() > 0
    assert module.expert_down_proj.grad[3].abs().sum() > 0
    assert module.expert_down_proj.grad[1:3].abs().sum() == 0


def test_basisdraft_rejects_truncated_route():
    module = RouteConditionedBasisExperts(BasisDraftConfig(
        hidden_width=3, experts=4, exact_k=2, basis_count=2, basis_width=1
    ))
    with pytest.raises(ValueError, match="complete native top-k"):
        module(torch.ones(1, 3), torch.tensor([[0]]), torch.ones(1, 1))


def test_groupwise_int4_round_trip_and_sparse_execution():
    torch.manual_seed(7)
    gate_up = torch.randn(3, 4, 4)
    down = torch.randn(3, 4, 2)
    gate_packed, gate_scales = quantize_groupwise_int4(gate_up, group_size=2)
    down_packed, down_scales = quantize_groupwise_int4(down, group_size=2)
    reconstructed_gate = dequantize_groupwise_int4(
        gate_packed, gate_scales, group_size=2, dtype=torch.float32
    )
    reconstructed_down = dequantize_groupwise_int4(
        down_packed, down_scales, group_size=2, dtype=torch.float32
    )
    assert float((reconstructed_gate - gate_up).square().mean().sqrt()) < 0.2
    module = PackedInt4TopKExperts(
        hidden_width=4,
        intermediate_width=2,
        experts=3,
        active_slots=2,
        group_size=2,
    )
    module.gate_up_packed.copy_(gate_packed)
    module.gate_up_scales.copy_(gate_scales)
    module.down_packed.copy_(down_packed)
    module.down_scales.copy_(down_scales)
    hidden = torch.randn(2, 4)
    ids = torch.tensor([[0, 2, 1], [1, 0, 2]])
    weights = torch.tensor([[0.5, 0.3, 0.2], [0.6, 0.25, 0.15]])
    individual = target_selected_expert_outputs(
        hidden,
        ids[:, :2],
        reconstructed_gate,
        reconstructed_down,
    )
    expected = (individual * weights[:, :2, None]).sum(1)
    uncached = module(hidden, ids, weights)
    assert torch.allclose(uncached, expected, atol=1e-6, rtol=1e-6)
    module.enable_dequantized_cache()
    cached_first = module(hidden, ids, weights)
    cached_second = module(hidden, ids, weights)
    assert torch.equal(cached_first, uncached)
    assert torch.equal(cached_second, uncached)
    assert set(module._gate_up_cache) == {0, 1, 2}
    module.enable_dequantized_cache(False)
    assert not module._gate_up_cache and not module._down_cache


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


def test_target_selected_expert_top1_fast_path_matches_manual_swiglu():
    torch.manual_seed(7)
    hidden = torch.randn(3, 2, 4)
    ids = torch.tensor([[[0], [2]], [[1], [0]], [[2], [1]]])
    gate_up = torch.randn(3, 6, 4)
    down = torch.randn(3, 4, 3)
    outputs = target_selected_expert_outputs(hidden, ids, gate_up, down)
    expected = torch.empty_like(outputs)
    for batch in range(3):
        for token in range(2):
            expert = int(ids[batch, token, 0])
            gate, up = torch.nn.functional.linear(
                hidden[batch, token], gate_up[expert]
            ).chunk(2)
            expected[batch, token, 0] = torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up, down[expert]
            )
    assert torch.allclose(outputs, expected, atol=1e-6, rtol=1e-6)


def test_resident_int4_uses_exact_residents_and_mass_scaled_fallback():
    torch.manual_seed(17)
    gate_up = torch.randn(4, 4, 4)
    down = torch.randn(4, 4, 2)
    draft = SwiGLUDraftExpert(4, 2)
    fallback = SharedResidualExperts(draft, experts=4)
    module = PackedInt4ResidentExperts.from_target(
        torch.tensor([0, 2]), fallback, gate_up, down,
        exact_k=2, group_size=2,
    )
    hidden = torch.randn(3, 4)
    ids = torch.tensor([[0, 1], [3, 1], [2, 0]])
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.55, 0.45]])
    actual = module(hidden, ids, weights)
    expected = draft(hidden) * torch.tensor([[0.3], [1.0], [0.0]])
    for row, slot in ((0, 0), (2, 0), (2, 1)):
        local_id = int(module.expert_to_resident[ids[row, slot]])
        selected_gate, selected_down = module._resident_weights(
            local_id, dtype=hidden.dtype
        )
        gate, up = torch.nn.functional.linear(
            hidden[row], selected_gate
        ).chunk(2)
        value = torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up, selected_down
        )
        expected[row] += weights[row, slot] * value
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert set(module.state_dict()) >= {
        "resident_ids", "expert_to_resident", "gate_up_packed", "down_packed"
    }
    assert not hasattr(module, "native_experts")



def test_resident_tail_control_is_zero_compatible_identity_sensitive_and_nonowning():
    torch.manual_seed(29)
    gate_up = torch.randn(4, 4, 4)
    down = torch.randn(4, 4, 2)
    draft = SwiGLUDraftExpert(4, 2)
    router = torch.randn(4, 4)
    control = RouterVisibleTailControl(
        hidden_width=4, experts=4, rank=2, dtype=torch.float32
    )
    control.bind_next_router(router)
    module = PackedInt4ResidentExperts.from_target(
        torch.tensor([0, 2]),
        SharedResidualExperts(draft, experts=4),
        gate_up,
        down,
        exact_k=2,
        group_size=2,
        router_control=control,
    )
    hidden = torch.randn(2, 4, requires_grad=True)
    ids = torch.tensor([[0, 1], [2, 0]])
    weights = torch.tensor([[0.6, 0.4], [0.55, 0.45]])
    components = module.forward_components(hidden, ids, weights)
    assert torch.count_nonzero(components.control_output) == 0
    assert torch.equal(
        components.output,
        components.resident_output + components.tail_output,
    )
    assert "_next_router_weight" not in module.state_dict()
    assert control._next_router_weight.data_ptr() == router.data_ptr()
    with torch.no_grad():
        control.expert_codes.weight[1] = torch.tensor([1.0, -0.5])
    corrected = module.forward_components(hidden, ids, weights)
    assert corrected.control_output[0].abs().sum() > 0
    assert corrected.control_output[1].abs().sum() == 0
    corrected.output.sum().backward()
    assert control.expert_codes.weight.grad is not None
    assert control.expert_codes.weight.grad[1].abs().sum() > 0
    assert control.expert_codes.weight.grad[[0, 2, 3]].abs().sum() == 0


def test_resident_tail_control_storage_stays_below_six_gib():
    v1_bytes = 6_401_871_784
    control_bytes = resident_tail_control_storage_bytes(rank=128)
    assert control_bytes == 25_559_040
    assert v1_bytes + control_bytes == 6_427_430_824
    assert v1_bytes + control_bytes < 6 * 2**30


def test_resident_int4_storage_matches_five_gib_budget():
    size = resident_int4_storage_bytes(residents=80)
    assert size == 5_347_737_600
    assert size / (1024 ** 3) < 5.0
    resident_92 = resident_int4_storage_bytes(residents=92)
    shared_fallback = 3 * 2048 * 512 * 40 * 2
    assert (resident_92 + shared_fallback) / (1024 ** 3) < 6.0


def test_resident_allocation_spends_global_budget_on_largest_marginal_gain():
    counts = torch.tensor([
        [10, 9, 1, 0],
        [10, 8, 7, 0],
        [10, 2, 1, 0],
    ], dtype=torch.int64)
    allocation = allocate_resident_experts(
        counts,
        total_residents=7,
        minimum_per_layer=1,
        maximum_per_layer=3,
    )
    assert allocation.resident_counts == (2, 3, 2)
    assert sum(allocation.resident_counts) == 7
    assert torch.equal(allocation.resident_ids[0], torch.tensor([0, 1]))
    assert allocation.covered_slots == (19, 25, 12)
    with pytest.raises(ValueError, match="outside"):
        allocate_resident_experts(
            counts, total_residents=2, minimum_per_layer=1, maximum_per_layer=3
        )



def test_resident_utility_allocation_is_stable_and_equal_budget():
    utility = torch.tensor([
        [10.0, 9.0, 1.0, 0.0],
        [10.0, 8.0, 7.0, 0.0],
        [10.0, 2.0, 1.0, 0.0],
    ])
    allocation = allocate_resident_utility(
        utility,
        total_residents=7,
        minimum_per_layer=1,
        maximum_per_layer=3,
    )
    assert allocation.resident_counts == (2, 3, 2)
    assert torch.equal(allocation.resident_ids[0], torch.tensor([0, 1]))
    tied = allocate_resident_utility(
        torch.ones(2, 3),
        total_residents=3,
        minimum_per_layer=1,
        maximum_per_layer=2,
    )
    assert tied.resident_counts == (2, 1)
    with pytest.raises(ValueError, match="finite"):
        allocate_resident_utility(
            torch.tensor([[float("nan"), 1.0]]),
            total_residents=1,
            minimum_per_layer=1,
            maximum_per_layer=1,
        )


def test_shadow_parameter_counts_match_formal_plan():
    assert shadow_pool_parameter_count() == 1_006_632_960
    assert shadow_pool_parameter_count(
        layers=40, experts=1, hidden_width=2048, shadow_width=128
    ) == 31_457_280
    assert basisdraft_parameter_count() == 256_901_120
    assert basisdraft_parameter_count(expert_residual_width=16) == 1_137_704_960


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


def test_compact_visible_tree_remaps_sparse_ancestor_closed_budget():
    tokens = torch.tensor([10, 11, 12, 13, 14, 15])
    parents = torch.tensor([-1, 0, 0, 1, 2, 3])
    visible = torch.tensor([True, True, False, True, False, True])
    compact_tokens, compact_parents, original = compact_visible_tree(
        tokens, parents, visible
    )
    assert compact_tokens.tolist() == [10, 11, 13, 15]
    assert compact_parents.tolist() == [-1, 0, 1, 2]
    assert original.tolist() == [0, 1, 3, 5]
    invalid = visible.clone()
    invalid[1] = False
    with pytest.raises(ValueError, match="ancestor-closed"):
        compact_visible_tree(tokens, parents, invalid)


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


def test_shadow_lm_posterior_accumulates_causal_edge_probabilities():
    probabilities = torch.full((1, 4, 6), 0.1)
    probabilities[0, 0] = torch.tensor([0.1, 0.3, 0.2, 0.1, 0.1, 0.2])
    probabilities[0, 1] = torch.tensor([0.1, 0.1, 0.1, 0.5, 0.1, 0.1])
    branch_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    branch_mask[0, 1, 1:3] = True
    branch_mask[0, 2, 3] = True
    posterior = shadow_lm_path_posterior(
        probabilities.log(),
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[-1, 0, 0, 1]]),
        torch.tensor([[1, 2, 2, 3]]),
        torch.ones(1, 4, dtype=torch.bool),
        branch_mask,
    )
    assert torch.allclose(
        posterior.cumulative_log_probabilities[0].exp(),
        torch.tensor([1.0, 0.3, 0.2, 0.15]),
    )
    assert torch.allclose(
        posterior.other_probabilities[0], torch.tensor([1.0, 0.5, 0.85, 1.0])
    )


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
    normalized = dict(config)
    normalized.pop("rope_theta")
    normalized["rope_parameters"] = {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10_000_000.0,
        "rope_type": "default",
    }
    validate_shadow_target_config(normalized, contract)
    normalized["rope_parameters"]["mrope_section"] = [10, 11, 11]
    with pytest.raises(ValueError, match="mRoPE section"):
        validate_shadow_target_config(normalized, contract)
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
