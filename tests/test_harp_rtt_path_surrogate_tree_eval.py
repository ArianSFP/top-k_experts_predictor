from __future__ import annotations

from types import SimpleNamespace

import torch

from harp_rtt.path_route_tree import reconstruct_tree_token_prefixes
from runpod.evaluate_harp_path_surrogate_on_tree import (
    assert_development_companion_privacy,
    blend_branch_marginals,
    deployed_grid,
    predict_nodes,
)
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


def test_legacy_outer_train_companion_privacy_is_accepted() -> None:
    assert_development_companion_privacy({
        "split": "train", "label_only": True, "sealed_test_opened": False,
    })


def test_explicit_opened_split_is_rejected() -> None:
    for field in (
        "formal_validation_opened", "calibration_opened", "sealed_test_opened",
    ):
        manifest = {
            "split": "train", "label_only": True,
            "sealed_test_opened": False, field: True,
        }
        try:
            assert_development_companion_privacy(manifest)
        except PermissionError:
            pass
        else:
            raise AssertionError(f"explicit {field}=true must fail closed")


def test_branch_marginal_blend_preserves_exact_cardinality() -> None:
    left = torch.tensor([[[[1.25, 0.75, 0.0]]]])
    right = torch.tensor([[[[0.0, 0.5, 1.5]]]])
    result = blend_branch_marginals(
        left, right, adapted_weight=0.25, exact_k=2
    )
    assert result.shape == left.shape
    assert torch.allclose(result.sum(-1), torch.full((1, 1, 1), 2.0))
    assert torch.equal(
        blend_branch_marginals(left, right, adapted_weight=1.0, exact_k=2),
        left,
    )


def test_branch_marginal_blend_rejects_invalid_contract() -> None:
    values = torch.ones(1, 1, 1, 3)
    try:
        blend_branch_marginals(values, values[..., :2], adapted_weight=0.5, exact_k=2)
    except ValueError:
        pass
    else:
        raise AssertionError("different branch shapes must fail closed")
    try:
        blend_branch_marginals(values, values, adapted_weight=1.1, exact_k=2)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid blend weight must fail closed")
