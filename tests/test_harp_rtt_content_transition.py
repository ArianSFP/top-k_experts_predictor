from __future__ import annotations

import torch

from harp_rtt.content_transition import (
    ContentTransitionConfig,
    FullStateTransitionProbe,
)


def _probe() -> FullStateTransitionProbe:
    torch.manual_seed(7)
    config = ContentTransitionConfig(
        layers=4, experts=7, hidden_width=9, router_rank=3, exact_k=2,
        latent_width=8, effect_width=4, transition_width=16,
        content_adapter_rank=2, output_adapter_rank=2, dropout=0.0,
    )
    basis = torch.empty(config.layers, config.hidden_width, config.router_rank)
    for layer in range(config.layers):
        basis[layer] = torch.linalg.qr(
            torch.randn(config.hidden_width, config.router_rank)
        ).Q
    return FullStateTransitionProbe(
        config, basis, torch.randn(config.layers, config.experts, config.router_rank),
        torch.randn(config.layers, config.experts),
        torch.ones(config.layers, config.router_rank, dtype=torch.bool),
    )


def test_router_blind_decomposition_is_orthogonal() -> None:
    probe = _probe()
    inputs = torch.randn(2, 3, 4, 9)
    queries, blind = probe.decompose(inputs)
    projected_blind = torch.einsum(
        "...ld,ldr->...lr", blind, probe.input_basis
    )
    assert queries.shape == (2, 3, 4, 3)
    assert float(projected_blind.abs().max()) < 2e-5


def test_full_state_probe_predicts_next_layer_and_backpropagates() -> None:
    probe = _probe()
    inputs = torch.randn(2, 3, 4, 9)
    ids = torch.randint(0, 7, (2, 3, 4, 2))
    weights = torch.rand(2, 3, 4, 2)
    output = probe(inputs, ids, weights)
    assert output.predicted_queries.shape == (2, 3, 3, 3)
    assert output.predicted_scores.shape == (2, 3, 3, 7)
    assert output.selected_ids.shape == (2, 3, 3, 2)
    output.predicted_scores.square().mean().backward()
    assert probe.content_input.weight.grad is not None
    assert torch.isfinite(probe.content_input.weight.grad).all()


def test_content_ablation_removes_only_router_blind_path() -> None:
    probe = _probe().eval()
    inputs = torch.randn(2, 3, 4, 9)
    ids = torch.randint(0, 7, (2, 3, 4, 2))
    weights = torch.rand(2, 3, 4, 2)
    with torch.no_grad():
        full = probe(inputs, ids, weights, use_router_blind_content=True)
        ablated = probe(inputs, ids, weights, use_router_blind_content=False)
    assert torch.equal(full.router_queries, ablated.router_queries)
    assert torch.equal(full.router_blind_inputs, ablated.router_blind_inputs)
    assert not torch.equal(full.predicted_scores, ablated.predicted_scores)


def test_selected_execution_weights_change_transition() -> None:
    probe = _probe().eval()
    inputs = torch.randn(1, 2, 4, 9)
    ids = torch.tensor([[[[0, 1]] * 4, [[2, 3]] * 4]])
    first = torch.tensor([0.99, 0.01]).expand(1, 2, 4, 2)
    second = torch.tensor([0.01, 0.99]).expand(1, 2, 4, 2)
    with torch.no_grad():
        a = probe(inputs, ids, first).predicted_queries
        b = probe(inputs, ids, second).predicted_queries
    assert not torch.equal(a, b)
