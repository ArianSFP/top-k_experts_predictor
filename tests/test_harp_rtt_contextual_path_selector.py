from __future__ import annotations

import torch

from harp_rtt.contextual_path_selector import ContextualFactualPathSelector
from harp_rtt.route_dynamics import DeltaRouteConfig


def test_contextual_selector_prior_reproduction_and_context_interaction() -> None:
    config = DeltaRouteConfig(
        experts=16, layers=3, horizons=4, nodes=5, router_rank=7,
        raw_width=12, metadata_width=8, latent_width=16,
        effect_width=4, transition_width=32, layer_adapter_rank=2,
        free_rank=2, attention_heads=4, exact_k=2, dropout=0.0,
    )
    model = ContextualFactualPathSelector(config, blocks=1).eval()
    generator = torch.Generator().manual_seed(5)
    tree = torch.randn(2, 5, 16, generator=generator)
    context = torch.randn(2, 4, 3, 16, generator=generator)
    path = torch.log(torch.tensor([
        [0.4, 0.3, 0.2, 0.08, 0.02],
        [0.5, 0.2, 0.15, 0.1, 0.05],
    ]))
    visible = torch.zeros(2, 4, 5, dtype=torch.bool)
    visible[:, 0, 0] = True
    visible[:, 1, :2] = True
    visible[:, 2, :4] = True
    visible[:, 3] = True
    available = torch.ones(2, 5, dtype=torch.bool)
    with torch.no_grad():
        initial = model(
            tree_states=tree, context_states=context,
            path_log_probabilities=path, horizon_mask=visible,
            node_available=available,
        )
    expected_captured = torch.where(
        visible, path[:, None].exp(), torch.zeros_like(path[:, None])
    )
    expected = torch.cat((
        expected_captured,
        (1.0 - expected_captured.sum(-1)).clamp_min(1e-8)[..., None],
    ), -1)
    assert torch.allclose(initial.probabilities, expected, atol=1e-6)
    assert torch.allclose(initial.probabilities.sum(-1), torch.ones(2, 4))
    assert torch.equal(initial.probabilities[..., :-1][~visible], torch.zeros_like(
        initial.probabilities[..., :-1][~visible]
    ))

    model.train(); model.zero_grad(set_to_none=True)
    output = model(
        tree_states=tree, context_states=context,
        path_log_probabilities=path, horizon_mask=visible,
        node_available=available,
    )
    (-output.logits[:, 1:, 0].mean()).backward()
    assert model.interaction[-1].weight.grad is not None
    assert model.context[1].weight.grad is not None

