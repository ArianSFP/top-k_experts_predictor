from __future__ import annotations

import torch

from runpod.train_harp_branch_surrogate import choose_nodes, objective
from tests.test_harp_rtt_path_route_trajectory import _trajectory_fixture


def _branch_batch() -> tuple[object, dict[str, torch.Tensor], torch.Tensor]:
    model, inputs = _trajectory_fixture()
    batch, nodes = 2, 8
    depth = torch.tensor([[2, 2, 3, 3, 4, 4, 4, 4]]).expand(batch, -1).clone()
    target_queries = torch.randn(
        batch, nodes, model.config.layers, model.config.router_rank
    )
    target_scores = torch.einsum(
        "bnlr,ler->bnle", target_queries, model.expert_keys
    ) + model.centered_bias[None, None]
    branch = {
        "state_coordinates": inputs["state_coordinates"],
        "current_queries": inputs["current_queries"],
        "history_selected_ids": inputs["history_selected_ids"],
        "history_selected_weights": inputs["history_selected_weights"],
        "path_token_ids": torch.randint(0, 19, (batch, nodes, 4)),
        "node_depth": depth,
        "node_mask": torch.ones(batch, nodes, dtype=torch.uint8),
        "budget16_mask": torch.ones(batch, nodes, dtype=torch.uint8),
        "target_queries": target_queries,
        "target_selected_ids": target_scores.topk(model.config.exact_k, dim=-1).indices,
    }
    token_embedding = torch.randn(19, model.config.token_width)
    return model, branch, token_embedding


def test_counterfactual_node_sampling_is_depth_balanced() -> None:
    _, batch, _ = _branch_batch()
    rows, nodes = choose_nodes(
        batch["node_depth"], batch["budget16_mask"], nodes_per_source=8,
        generator=torch.Generator().manual_seed(42),
    )
    for row in range(2):
        chosen = rows == row
        chosen_depth = batch["node_depth"][rows[chosen], nodes[chosen]]
        assert int((chosen_depth == 2).sum()) == 2
        assert int((chosen_depth == 3).sum()) == 2
        assert int((chosen_depth == 4).sum()) == 4


def test_counterfactual_objective_reaches_rollout_parameters() -> None:
    model, batch, token_embedding = _branch_batch()
    loss, components, (_, labels, depth) = objective(
        model, batch, token_embedding=token_embedding, device=torch.device("cpu"),
        nodes_per_source=8, generator=torch.Generator().manual_seed(7),
    )
    assert torch.isfinite(loss)
    assert set(components) == {
        "exact_set", "induced_logit_huber", "top_boundary", "total",
    }
    assert labels.shape == (16, model.config.layers, model.config.exact_k)
    assert set(depth.tolist()) == {2, 3, 4}
    loss.backward()
    assert model.trajectory_cell.weight_hh.grad is not None
    assert float(model.trajectory_cell.weight_hh.grad.abs().sum()) > 0
