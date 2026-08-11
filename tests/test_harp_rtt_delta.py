from __future__ import annotations

import torch

from harp_rtt.delta import (
    AdaptiveAnchorCandidateSelector,
    HARPDeltaConfig,
    HARPDeltaInputAdapter,
    HARPDeltaSemanticOutput,
    HARPDeltaTeacher,
    HARPDeltaTree,
    causal_position_features,
    selective_swap_decode,
)
from harp_rtt.exact_k import stable_topk


def compact_config() -> HARPDeltaConfig:
    return HARPDeltaConfig(
        experts=80,
        layers=3,
        horizons=4,
        exact_k=2,
        candidate_width=64,
        max_tree_nodes=4,
        router_rank=5,
        context_input_width=8,
        root_input_width=8,
        node_input_width=8,
        tree_width=16,
        set_width=16,
        ranker_width=16,
        tree_ffn_width=32,
        ranker_ffn_width=32,
        attention_heads=4,
        tree_blocks=2,
        ranker_blocks=1,
        free_rank=4,
        position_frequencies=2,
        maximum_swaps=2,
        dropout=0.0,
    )


def test_position_features_are_bounded_and_do_not_clamp_after_32() -> None:
    values = causal_position_features(torch.tensor([0, 32, 33, 10_000]), 3)
    assert values.shape == (4, 8)
    assert torch.isfinite(values).all()
    assert values.abs().max() <= 1.0
    assert not torch.equal(values[1], values[2])


def test_selective_swap_abstention_is_exact_anchor_topk() -> None:
    anchor = torch.arange(10, dtype=torch.float32).reshape(1, 1, 1, 10)
    candidates = torch.arange(10).reshape(1, 1, 1, 10)
    result = selective_swap_decode(
        anchor,
        candidates,
        anchor.clone(),
        torch.full_like(anchor, 0.5),
        exact_k=2,
        maximum_swaps=2,
        confidence_threshold=0.65,
    )
    assert torch.equal(result, stable_topk(anchor, 2))


def test_selective_swap_is_bounded_and_deterministic() -> None:
    anchor = torch.arange(10, dtype=torch.float32).reshape(1, 1, 1, 10)
    candidates = torch.arange(10).reshape(1, 1, 1, 10)
    scores = anchor.clone()
    scores[..., 0] = 20.0
    scores[..., 1] = 19.0
    scores[..., 2] = 18.0
    result = selective_swap_decode(
        anchor,
        candidates,
        scores,
        torch.ones_like(scores),
        exact_k=2,
        maximum_swaps=1,
        confidence_threshold=0.65,
    )
    assert set(result.flatten().tolist()) == {0, 9}


def test_deltatree_epoch_zero_is_anchor_protected() -> None:
    torch.manual_seed(4)
    config = compact_config()
    keys = torch.randn(config.layers, config.experts, config.router_rank)
    bias = torch.randn(config.layers, config.experts)
    model = HARPDeltaTree(config, keys, bias).eval()
    batch, nodes = 2, 4
    anchor = torch.randn(batch, config.horizons, config.layers, config.experts)
    context = torch.randn(
        batch, config.horizons, config.layers, config.context_input_width
    )
    root = torch.randn(batch, config.layers, config.root_input_width)
    node = torch.randn(batch, nodes, config.node_input_width)
    parents = torch.tensor([[-1, 0, 0, 1], [-1, 0, 1, 1]])
    available = torch.ones(batch, nodes, dtype=torch.bool)
    path_logp = torch.log(torch.tensor([[0.6, 0.2, 0.1, 0.05]]).repeat(batch, 1))
    horizon_mask = torch.zeros(batch, config.horizons, nodes, dtype=torch.bool)
    horizon_mask[:, 1, 0:2] = True
    horizon_mask[:, 2, 0:3] = True
    horizon_mask[:, 3] = True
    model_inputs = {
        "anchor_scores": anchor,
        "context_features": context,
        "root_features": root,
        "node_features": node,
        "node_parent_ids": parents,
        "node_available": available,
        "node_path_log_probabilities": path_logp,
        "node_horizon_mask": horizon_mask,
        "source_positions": torch.tensor([5, 500]),
    }
    output = model(**model_inputs)
    semantic = model(**model_inputs, semantic_only=True)
    assert isinstance(semantic, HARPDeltaSemanticOutput)
    for name in (
        "root_scores",
        "root_queries",
        "node_scores",
        "node_queries",
        "factual_path_logits",
        "factual_path_posterior",
        "tree_states",
        "context_states",
    ):
        assert torch.equal(getattr(semantic, name), getattr(output, name))
    assert torch.equal(output.final_ids, stable_topk(anchor, config.exact_k))
    assert bool((output.anchor_quotas == 64).all())
    assert output.candidate_ids.shape == (
        batch, config.horizons, config.layers, config.candidate_width
    )
    assert output.node_scores.shape == (
        batch, config.horizons, config.layers, nodes, config.experts
    )
    expected_mass = torch.full(
        (batch, config.horizons, config.layers), float(config.exact_k)
    )
    assert torch.allclose(output.branch_marginals.sum(-1), expected_mass, atol=1e-5)


def test_production_configuration_stays_below_parameter_cap() -> None:
    config = HARPDeltaConfig()
    keys = torch.zeros(config.layers, config.experts, config.router_rank)
    bias = torch.zeros(config.layers, config.experts)
    model = HARPDeltaTeacher(config, keys, bias)
    trainable = sum(parameter.numel() for parameter in model.parameters())
    assert trainable <= 25_000_000


def test_raw_capture_adapter_preserves_declared_axes() -> None:
    config = compact_config()
    adapter = HARPDeltaInputAdapter(
        config, raw_width=12, target_control_width=5, metadata_width=3
    )
    batch, nodes, history = 2, 4, 3
    output = adapter(
        anchor_scores=torch.randn(batch, config.horizons, config.layers, config.experts),
        route_history_logits=torch.randn(batch, config.layers, history, config.experts),
        target_control=torch.randn(batch, config.layers, 5),
        exact_token_embedding=torch.randn(batch, 12),
        final_hidden=torch.randn(batch, 12),
        tree_hidden=torch.randn(batch, nodes, 12),
        tree_fused=torch.randn(batch, nodes, 12),
        tree_router_input=torch.randn(batch, nodes, 12),
        tree_router_logits=torch.randn(batch, nodes, config.experts),
        tree_token_embeddings=torch.randn(batch, nodes, 12),
        tree_vocab_embedding=torch.randn(batch, nodes, 12),
        tree_vocab_statistics=torch.randn(batch, nodes, 6),
        tree_metadata=torch.randn(batch, nodes, 3),
    )
    assert output.context_features.shape == (
        batch, config.horizons, config.layers, config.context_input_width
    )
    assert output.root_features.shape == (batch, config.layers, config.root_input_width)
    assert output.node_features.shape == (batch, nodes, config.node_input_width)


def test_candidate_selector_with_no_positive_lift_is_exact_anchor_top64() -> None:
    config = compact_config()
    selector = AdaptiveAnchorCandidateSelector(config)
    anchor_scores = torch.randn(1, 4, config.layers, config.experts)
    anchor_marginals = torch.sigmoid(anchor_scores)
    posterior = torch.ones(1, 4, 1)
    result = selector(
        anchor_scores,
        anchor_marginals,
        anchor_marginals.clone(),
        posterior,
        forced_anchor_quota=32,
    )
    assert torch.equal(result.expert_ids, stable_topk(anchor_scores, 64))
