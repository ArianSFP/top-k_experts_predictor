from __future__ import annotations

import torch

from harp_rtt.causal_state_axial import CausalStateAxialRouteTrajectory
from harp_rtt.route_dynamics import DeltaRouteConfig


def _case() -> tuple[CausalStateAxialRouteTrajectory, dict[str, torch.Tensor]]:
    c = DeltaRouteConfig(
        experts=16, layers=3, horizons=4, nodes=5,
        router_rank=7, raw_width=12, metadata_width=8,
        latent_width=16, effect_width=8, transition_width=32,
        layer_adapter_rank=4, free_rank=4, attention_heads=4,
        exact_k=2, dropout=0.0,
    )
    model = CausalStateAxialRouteTrajectory(
        c, torch.randn(3, 12, 7), torch.randn(3, 16, 7),
        torch.randn(3, 16), torch.ones(3, 7, dtype=torch.bool),
        raw_rank=3, visible_rank=3, route_width=5, axial_blocks=1,
    )
    b = 2; raw = lambda: torch.randn(b, 5, 12)
    available = torch.ones(b, 5, dtype=torch.bool)
    values = {
        "parent_queries": torch.randn(b, 4, 3, 5, 7),
        "parent_scores": torch.randn(b, 4, 3, 5, 16),
        "context_states": torch.randn(b, 4, 3, 16),
        "fused": raw(), "post_ffn": raw(), "router_input": raw(),
        "vocabulary": raw(), "token": raw(),
        "router_logits": torch.randn(b, 5, 16),
        "metadata": torch.randn(b, 5, 8),
        "parents": torch.tensor([[-1,0,0,1,2],[-1,0,0,1,2]]),
        "available": available,
        "current_states": torch.randn(b, 4, 3, 12),
        "current_queries": torch.randn(b, 3, 7),
        "history_logits": torch.randn(b, 8, 3, 16),
        "history_selected_ids": torch.randint(0, 16, (b, 8, 3, 2)),
        "history_selected_weights": torch.rand(b, 8, 3, 2).softmax(-1),
    }
    return model, values


def test_causal_axial_exactly_reproduces_parent() -> None:
    torch.manual_seed(42)
    model, values = _case(); model.eval()
    out = model(**values)
    assert torch.equal(out.queries, values["parent_queries"].float())
    assert torch.equal(out.scores, values["parent_scores"].float())


def test_causal_axial_opens_state_and_branch_gradients() -> None:
    torch.manual_seed(43)
    model, values = _case(); model.train()
    model(**values).scores.sum().backward()
    for parameter in (
        model.state_projection[0].weight,
        model.current_query.weight,
        model.history_logits.weight,
        model.history_route_embedding,
        model.channels.raw_down,
        model.axial.layers[0].self_attn.in_proj_weight,
    ):
        assert parameter.grad is not None
        assert bool((parameter.grad != 0).any())

