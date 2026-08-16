from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from harp_rtt.routemtp import (
    AnytimePathPosterior,
    RouteMTPConfig,
    RouteMTPPredictor,
    SwitchableLoRALinear,
    install_cache_coherent_adapters,
)
from harp_rtt.routemtp_adapters import (
    SwitchableExpertLoRA,
    SwitchableMTPRouterResidual,
)
from harp_rtt.routemtp_batch import (
    assert_causal_routemtp_inputs,
    build_anytime_node_masks,
)


def _config() -> RouteMTPConfig:
    return RouteMTPConfig(
        hidden_width=16,
        target_layers=3,
        experts=16,
        router_rank=4,
        exact_k=2,
        max_nodes=6,
        width=8,
        route_width=4,
        direct_rank=3,
        layer_mixture_rank=2,
        lora_rank=2,
        residual_width=8,
    )


def _topology() -> tuple[torch.Tensor, ...]:
    parents = torch.tensor([[-1, 0, 1, 2, 1, 4]])
    depths = torch.tensor([[1, 2, 3, 4, 3, 4]])
    ranks = torch.tensor([[0, 0, 0, 0, 1, 0]])
    mask = torch.ones(1, 6, dtype=torch.bool)
    edge = torch.tensor([[
        0.0, math.log(0.6), math.log(0.5), math.log(0.4),
        math.log(0.25), math.log(0.3),
    ]])
    return parents, depths, ranks, mask, edge


def test_zero_path_head_recovers_raw_native_mass_and_other() -> None:
    config = _config(); torch.manual_seed(4)
    model = AnytimePathPosterior(config)
    parents, depths, ranks, mask, edge = _topology()
    output = model(
        recurrent_state=torch.randn(1, 6, 16),
        token_embedding=torch.randn(1, 6, 16),
        route_scores=torch.randn(1, 6, 3, 16),
        target_context=torch.randn(1, 4, 3, 8),
        native_edge_log_probabilities=edge,
        parent_ids=parents,
        node_depths=depths,
        child_ranks=ranks,
        node_mask=mask,
        visible_mask=mask,
    )
    expected = torch.tensor([[1.0, 0.6, 0.3, 0.12, 0.15, 0.045]])
    assert torch.allclose(output.node_path_probabilities, expected, atol=1e-6)
    assert torch.allclose(output.probabilities.sum(-1), torch.ones(1, 4))
    assert torch.allclose(
        output.probabilities[0, :, -1],
        torch.tensor([0.0, 0.4, 0.55, 0.835]),
        atol=1e-6,
    )


def test_post_execution_evidence_cannot_change_pre_execution_scores() -> None:
    config = _config(); model = AnytimePathPosterior(config)
    parents, depths, ranks, mask, edge = _topology()
    arguments = dict(
        recurrent_state=torch.randn(1, 6, 16),
        token_embedding=torch.randn(1, 6, 16),
        route_scores=torch.randn(1, 6, 3, 16),
        target_context=torch.randn(1, 4, 3, 8),
        native_edge_log_probabilities=edge,
        parent_ids=parents,
        node_depths=depths,
        child_ranks=ranks,
        node_mask=mask,
        visible_mask=mask,
    )
    before = model(**arguments, include_post_execution=False)
    with torch.no_grad():
        model.post[-1].bias.fill_(2.0)
    pre_only = model(**arguments, include_post_execution=False)
    full = model(**arguments, include_post_execution=True)
    assert torch.equal(before.pre_edge_correction, pre_only.pre_edge_correction)
    assert torch.equal(before.probabilities, pre_only.probabilities)
    assert not torch.equal(pre_only.probabilities[:, 2:], full.probabilities[:, 2:])


def test_anytime_masks_are_nested_and_fail_closed() -> None:
    node = torch.tensor([[True, True, True, True, True, True]])
    nested = torch.tensor([[[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]]).bool()
    result = build_anytime_node_masks(node, nested)
    assert result.shape == (1, 4, 6)
    assert result[0, 0].tolist() == [True, False, False, False, False, False]
    broken = nested.clone(); broken[0, 1, 1] = False
    with pytest.raises(ValueError, match="not nested"):
        build_anytime_node_masks(node, broken)


def test_route_head_predicts_only_node_depth_and_exact_k_mass() -> None:
    config = _config(); torch.manual_seed(7)
    keys = torch.randn(3, 16, 4)
    bias = torch.randn(3, 16)
    rank = torch.ones(3, 4, dtype=torch.bool)
    model = RouteMTPPredictor(config, keys, bias, rank)
    parents, depths, ranks, mask, edge = _topology()
    common_state = torch.randn(1, 6, 16)
    selected = torch.randint(0, 16, (1, 6, 2))
    weights = torch.softmax(torch.randn(1, 6, 2), -1)
    anchor = torch.full((1, 4, 3, 16), 2.0 / 16)
    output = model(
        fused_state=common_state,
        router_input=torch.randn_like(common_state),
        post_moe_hidden=torch.randn_like(common_state),
        vocabulary_head_input=torch.randn_like(common_state),
        mtp_router_logits=torch.randn(1, 6, 16),
        mtp_selected_ids=selected,
        mtp_selected_weights=weights,
        recurrent_state=torch.randn_like(common_state),
        node_token_embeddings=torch.randn_like(common_state),
        node_parent_ids=parents,
        node_depths=depths,
        node_child_ranks=ranks,
        node_mask=mask,
        visible_mask=mask,
        native_edge_log_probabilities=edge,
        current_post_layer=torch.randn(1, 3, 16),
        current_queries=torch.randn(1, 3, 4),
        current_centered_router_logits=torch.randn(1, 3, 16),
        current_selected_ids=torch.randint(0, 16, (1, 3, 2)),
        current_selected_weights=torch.softmax(torch.randn(1, 3, 2), -1),
        anchor_marginals=anchor,
    )
    assert output.route.scores.shape == (1, 6, 3, 16)
    assert torch.allclose(
        output.route.marginals.sum(-1), torch.full((1, 6, 3), 2.0), atol=2e-5
    )
    assert torch.allclose(
        output.factual.projected_marginals.sum(-1),
        torch.full((1, 4, 3), 2.0),
        atol=2e-5,
    )
    output.route.scores.sum().backward()
    assert model.route_head.query.weight.grad is not None


def test_switchable_lora_preserves_native_path_at_zero_and_when_disabled() -> None:
    torch.manual_seed(9)
    base = nn.Linear(5, 4, bias=False)
    adapter = SwitchableLoRALinear(base, 2)
    values = torch.randn(3, 5)
    expected = base(values)
    with adapter.active():
        assert torch.equal(adapter(values), expected)
    with torch.no_grad():
        adapter.up.weight.fill_(0.25)
    assert torch.equal(adapter(values), expected)
    with adapter.active():
        assert not torch.equal(adapter(values), expected)


def test_switchable_lora_inherits_bfloat16_base_dtype() -> None:
    base = nn.Linear(5, 4, bias=False, dtype=torch.bfloat16)
    adapter = SwitchableLoRALinear(base, 2)
    assert adapter.down.weight.dtype == torch.bfloat16
    assert adapter.up.weight.dtype == torch.bfloat16
    values = torch.randn(3, 5, dtype=torch.bfloat16)
    with adapter.active():
        assert torch.equal(adapter(values), base(values))


def test_switchable_lora_only_changes_speculative_tail_rows() -> None:
    torch.manual_seed(10)
    base = nn.Linear(5, 4, bias=False)
    adapter = SwitchableLoRALinear(base, 2)
    with torch.no_grad():
        adapter.up.weight.fill_(0.25)
    values = torch.randn(1, 4, 5)
    expected = base(values)
    with adapter.active(tail_rows=2):
        actual = adapter(values)
    assert torch.equal(actual[:, :2], expected[:, :2])
    assert not torch.equal(actual[:, 2:], expected[:, 2:])


def test_installation_leaves_kv_unadapted() -> None:
    attention = SimpleNamespace(
        q_proj=nn.Linear(8, 8, bias=False),
        k_proj=nn.Linear(8, 8, bias=False),
        v_proj=nn.Linear(8, 8, bias=False),
        o_proj=nn.Linear(8, 8, bias=False),
    )
    mtp = SimpleNamespace(
        fc=nn.Linear(16, 8, bias=False),
        decoder=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]),
    )
    original_k = attention.k_proj; original_v = attention.v_proj
    installed = install_cache_coherent_adapters(mtp, rank=2)
    assert mtp.fc is installed.fusion
    assert attention.q_proj is installed.query
    assert attention.o_proj is installed.output
    assert attention.k_proj is original_k and attention.v_proj is original_v


def test_model_input_contract_rejects_label_leakage() -> None:
    with pytest.raises(PermissionError, match="undeclared"):
        assert_causal_routemtp_inputs({"future_router_logits": torch.zeros(1)})


class _PackedExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.num_experts = 3; self.act_fn = torch.nn.functional.silu
        self.gate_up_proj = nn.Parameter(torch.randn(3, 8, 6), requires_grad=False)
        self.down_proj = nn.Parameter(torch.randn(3, 6, 4), requires_grad=False)

    def forward(self, hidden, ids, weights):
        flat = hidden.reshape(-1, 6); ids = ids.reshape(flat.shape[0], -1); weights = weights.reshape_as(ids)
        result = torch.zeros_like(flat)
        for expert in range(3):
            positions = (ids == expert).nonzero(as_tuple=False)
            if not positions.numel(): continue
            token, slot = positions[:, 0], positions[:, 1]
            gate, up = torch.nn.functional.linear(flat[token], self.gate_up_proj[expert]).chunk(2, -1)
            value = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, self.down_proj[expert])
            result.index_add_(0, token, value * weights[token, slot, None])
        return result.reshape_as(hidden)


def test_expert_lora_is_exact_at_zero_and_routes_gradients_only_to_selected_cells() -> None:
    torch.manual_seed(18); base = _PackedExperts(); adapter = SwitchableExpertLoRA(base, rank=2)
    hidden = torch.randn(2, 6)
    ids = torch.tensor([[0, 1], [1, 0]])
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    expected = base(hidden, ids, weights)
    with adapter.active():
        actual = adapter(hidden, ids, weights)
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert adapter.gate_up_up.grad is not None
    assert torch.equal(adapter.gate_up_up.grad[2], torch.zeros_like(adapter.gate_up_up.grad[2]))


def test_expert_lora_preserves_committed_prefix_rows() -> None:
    torch.manual_seed(20); base = _PackedExperts(); adapter = SwitchableExpertLoRA(base, rank=2)
    with torch.no_grad():
        adapter.gate_up_up.fill_(0.1); adapter.down_up.fill_(0.1)
    hidden = torch.randn(1, 4, 6)
    ids = torch.tensor([[[0, 1], [1, 0], [0, 2], [2, 1]]])
    weights = torch.full((1, 4, 2), 0.5)
    expected = base(hidden, ids, weights)
    with adapter.active(tail_rows=2):
        actual = adapter(hidden, ids, weights)
    assert torch.equal(actual[:, :2], expected[:, :2])
    assert not torch.equal(actual[:, 2:], expected[:, 2:])


class _Router(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.linear = nn.Linear(6, 5, bias=False)
        self.linear.requires_grad_(False)
    def forward(self, hidden):
        logits = self.linear(hidden)
        probability = torch.softmax(logits.float(), -1)
        weight, ids = torch.topk(probability, 2, -1)
        return logits, (weight / weight.sum(-1, keepdim=True)).to(logits.dtype), ids


def test_router_residual_zero_init_preserves_native_hard_route() -> None:
    torch.manual_seed(19); base = _Router()
    adapter = SwitchableMTPRouterResidual(base, 6, 5, rank=2, top_k=2)
    hidden = torch.randn(4, 6)
    expected = base(hidden)
    with adapter.active():
        actual = adapter(hidden)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))
    collected_trust, collected_balance = adapter.collected_regularization()
    assert torch.isfinite(collected_trust) and torch.isfinite(collected_balance)
    adapter.set_training_progress(0.1)
    assert adapter.active_top_k == 5 and 0.2 < adapter.temperature < 1.0
    adapter.set_training_progress(1.0)
    assert adapter.active_top_k == 2 and adapter.temperature == pytest.approx(0.2)
    adapter.set_inference_mode()
    assert adapter.active_top_k == 2
    trust, balance = adapter.regularization(hidden)
    assert torch.isfinite(trust) and torch.isfinite(balance)


def test_router_residual_preserves_committed_prefix_rows() -> None:
    torch.manual_seed(21); base = _Router()
    adapter = SwitchableMTPRouterResidual(base, 6, 5, rank=2, top_k=2)
    with torch.no_grad():
        adapter.up.weight.fill_(0.2)
    hidden = torch.randn(1, 4, 6)
    expected = base(hidden)
    with adapter.active(tail_rows=2):
        actual = adapter(hidden)
    assert torch.equal(actual[0][:, :2], expected[0][:, :2])
