from __future__ import annotations

import torch

from harp_rtt.delta import HARPDeltaConfig
from harp_rtt.delta_batch import factual_branch_indices, prepare_delta_batch


def config() -> HARPDeltaConfig:
    return HARPDeltaConfig(
        experts=64, layers=2, exact_k=2, candidate_width=64,
        max_tree_nodes=4, router_rank=3, context_input_width=8,
        root_input_width=8, node_input_width=8, tree_width=16,
        set_width=16, ranker_width=16, tree_ffn_width=32,
        ranker_ffn_width=32, attention_heads=4, tree_blocks=1,
        ranker_blocks=1, free_rank=4, position_frequencies=2,
        maximum_swaps=2, dropout=0.0,
    )


def test_factual_branch_indices_use_other_only_when_prefix_absent() -> None:
    hashes = torch.zeros(1, 4, 32, dtype=torch.uint8)
    for node in range(4):
        hashes[0, node, 0] = node + 1
    future = hashes.clone()
    future[0, 2, 0] = 99
    result = factual_branch_indices(
        exact_prefix_hashes=hashes,
        node_depths=torch.tensor([[1, 2, 3, 4]]),
        node_available=torch.ones(1, 4, dtype=torch.bool),
        future_prefix_hashes=future,
    )
    assert result.tolist() == [[0, 1, 4, 3]]


def test_prepare_delta_batch_keeps_counterfactual_out_of_inputs() -> None:
    c = config()
    batch, history, nodes, hidden, vocabulary = 1, 3, 4, 6, 100
    hashes = torch.zeros(batch, nodes, 32, dtype=torch.uint8)
    for node in range(nodes):
        hashes[:, node, 0] = node + 1
    tree = {
        "states": torch.randn(batch, nodes, 4, hidden),
        "router_logits": torch.randn(batch, nodes, c.experts),
        "vocab_top64_ids": torch.arange(64)[None, None].expand(batch, nodes, -1),
        "vocab_top64_log_probabilities": torch.full((batch, nodes, 64), -5.0),
        "vocab_statistics": torch.randn(batch, nodes, 6),
        "meta": torch.zeros(batch, nodes, 13, dtype=torch.long),
        "path_log_probabilities": torch.log(torch.tensor([[0.8, 0.4, 0.2, 0.1]])),
        "local_probabilities": torch.tensor([[0.8, 0.5, 0.5, 0.5]]),
        "source_ready": torch.ones(batch, nodes),
        "depth": torch.tensor([[1, 2, 3, 4]]),
        "child_ranks": torch.zeros(batch, nodes),
        "first_divergence_depths": torch.zeros(batch, nodes),
        "cumulative_path_ranks": torch.ones(batch, nodes),
        "sibling_counts": torch.ones(batch, nodes),
        "parent": torch.tensor([[-1, 0, 1, 2]]),
        "mask": torch.ones(batch, nodes, dtype=torch.bool),
        "horizon_mask": torch.eye(4, dtype=torch.bool)[None],
        "exact_prefix_hashes": hashes,
    }
    tree["meta"][..., 7] = torch.tensor([[1, 2, 3, 4]])
    source = {
        "inputs": {
            "history": {"logits": torch.randn(batch, history, c.layers, c.experts)},
            "current": {"normalized_target_router_input_a": torch.randn(batch, c.layers, hidden)},
            "tree": tree,
            "exact_next_token_id": torch.tensor([5]),
            "within_request": torch.tensor([101.0]),
            "final_hidden": torch.randn(batch, hidden),
        },
        "targets": {
            "future_prefix_hashes": hashes.clone(),
            "counterfactual": {"label_only": torch.tensor(True)},
        },
    }
    prepared = prepare_delta_batch(
        source,
        anchor_scores=torch.randn(batch, 4, c.layers, c.experts),
        token_embedding=torch.randn(vocabulary, hidden),
        input_basis=torch.randn(c.layers, hidden, c.router_rank),
        rank_mask=torch.ones(c.layers, c.router_rank, dtype=torch.bool),
        config=c,
    )
    assert prepared.model_inputs["source_positions"].item() == 101
    assert prepared.model_inputs["node_horizon_mask"].shape == (1, 4, 4)
    assert "counterfactual" not in prepared.model_inputs
    assert prepared.factual_branch_index.tolist() == [[0, 1, 2, 3]]
