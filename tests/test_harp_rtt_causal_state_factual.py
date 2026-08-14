from __future__ import annotations

import torch

from harp_rtt.causal_state_factual import CausalStateFactualHead
from harp_rtt.route_dynamics import DeltaRouteConfig


def _case() -> tuple[CausalStateFactualHead, dict[str, torch.Tensor]]:
    config = DeltaRouteConfig(
        experts=16, layers=3, horizons=4, nodes=5,
        router_rank=7, raw_width=12, metadata_width=8,
        latent_width=16, effect_width=8, transition_width=32,
        layer_adapter_rank=4, free_rank=4, attention_heads=4,
        exact_k=2, dropout=0.0,
    )
    model = CausalStateFactualHead(
        config, torch.randn(3, 16, 7),
        state_rank=3, route_width=5, axial_blocks=1,
    )
    batch = 2
    ids = torch.randint(0, 16, (batch, 8, 3, 2))
    weights = torch.rand(batch, 8, 3, 2).softmax(-1)
    values = {
        "context_states": torch.randn(batch, 4, 3, 16),
        "current_states": torch.randn(batch, 4, 3, 12),
        "current_queries": torch.randn(batch, 3, 7),
        "history_logits": torch.randn(batch, 8, 3, 16),
        "history_selected_ids": ids,
        "history_selected_weights": weights,
    }
    return model, values


def test_causal_state_head_is_exactly_parent_anchored() -> None:
    torch.manual_seed(42)
    model, values = _case()
    model.eval()
    output = model(**values)
    assert torch.equal(output.score_delta, torch.zeros_like(output.score_delta))
    assert torch.equal(output.score_delta[:, 0], torch.zeros_like(output.score_delta[:, 0]))


def test_causal_state_head_opens_full_encoder_gradient_on_first_step() -> None:
    torch.manual_seed(43)
    model, values = _case()
    model.train()
    output = model(**values)
    output.score_delta[:, 1:].sum().backward()
    required = (
        model.state_shared[0].weight,
        model.state_down,
        model.state_up,
        model.route_embedding,
        model.current_query.weight,
        model.history_logits.weight,
        model.axial.layers[0].self_attn.in_proj_weight,
    )
    for parameter in required:
        assert parameter.grad is not None
        assert bool((parameter.grad != 0).any())


def test_route_execution_weights_change_causal_hidden_after_learning() -> None:
    torch.manual_seed(44)
    model, values = _case()
    model.eval()
    with torch.no_grad():
        model.query_output.weight.add_(0.01)
    first = model(**values).score_delta
    changed = dict(values)
    changed["history_selected_weights"] = torch.flip(
        values["history_selected_weights"], dims=(-1,)
    )
    second = model(**changed).score_delta
    assert not torch.equal(first[:, 1:], second[:, 1:])

