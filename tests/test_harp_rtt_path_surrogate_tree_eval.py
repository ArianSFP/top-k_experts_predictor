from __future__ import annotations

from types import SimpleNamespace

import torch

from harp_rtt.path_route_tree import reconstruct_tree_token_prefixes
from runpod.evaluate_harp_path_surrogate_on_tree import deployed_grid, predict_nodes
from tests.test_harp_rtt_path_route_surrogate import _fixture


def test_tree_prediction_uses_each_nodes_causal_prefix() -> None:
    model, fixture = _fixture()
    model.eval()
    token_table = torch.randn(64, model.config.token_width)
    tree = {
        "token_ids": torch.tensor([[11, 21, 31, 41]]),
        "parent": torch.tensor([[-1, 0, 1, 2]]),
        "depth": torch.tensor([[1, 2, 3, 4]]),
        "mask": torch.ones(1, 4, dtype=torch.bool),
    }
    queries, scores = predict_nodes(
        model,
        token_embedding=token_table,
        state_coordinates=fixture["state_coordinates"][:1],
        current_queries=fixture["current_queries"][:1],
        history_ids=fixture["history_selected_ids"][:1],
        history_weights=fixture["history_selected_weights"][:1],
        tree=tree,
        node_microbatch=2,
    )
    prefixes, _ = reconstruct_tree_token_prefixes(
        tree["token_ids"], tree["parent"], tree["depth"], tree["mask"]
    )
    for node in range(4):
        manual = model(
            state_coordinates=fixture["state_coordinates"][:1],
            current_queries=fixture["current_queries"][:1],
            history_selected_ids=fixture["history_selected_ids"][:1],
            history_selected_weights=fixture["history_selected_weights"][:1],
            path_token_embeddings=token_table[prefixes[:, node]],
        )
        horizon = int(tree["depth"][0, node]) - 1
        assert torch.allclose(
            scores[0, node], manual.scores[0, horizon].float(), atol=2e-6, rtol=2e-6
        )
        assert torch.allclose(
            queries[0, node], manual.queries[0, horizon].float(), atol=2e-6, rtol=2e-6
        )


def test_deployed_grid_changes_only_each_nodes_declared_horizon() -> None:
    node_scores = torch.randn(1, 3, 2, 5)
    baseline = torch.zeros(1, 4, 2, 3, 5)
    semantic = SimpleNamespace(node_scores=baseline)
    depth = torch.tensor([[1, 2, 4]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    result = deployed_grid(node_scores, semantic, depth, mask)
    assert torch.equal(result[0, 0, :, 0], node_scores[0, 0])
    assert torch.equal(result[0, 1, :, 1], node_scores[0, 1])
    assert torch.equal(result[0, 3, :, 2], node_scores[0, 2])
    untouched = result.clone()
    untouched[0, 0, :, 0] = 0
    untouched[0, 1, :, 1] = 0
    untouched[0, 3, :, 2] = 0
    assert torch.count_nonzero(untouched) == 0
