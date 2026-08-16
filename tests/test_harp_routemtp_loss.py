from __future__ import annotations

import torch

from harp_rtt.exact_k import exact_k_logz_with_marginals, stable_topk
from harp_rtt.routemtp import RouteMTPPathOutput, RouteMTPRouteOutput
from harp_rtt.routemtp_loss import RouteMTPObjective


def test_routemtp_objective_trains_exact_factual_mixture_and_routes() -> None:
    torch.manual_seed(12)
    batch, nodes, layers, experts, rank, k, horizons = 2, 5, 3, 16, 4, 2, 4
    keys = torch.randn(layers, experts, rank)
    bias = torch.randn(layers, experts)
    scores = torch.randn(batch, nodes, layers, experts, requires_grad=True)
    queries = torch.randn(batch, nodes, layers, rank, requires_grad=True)
    _, marginals = exact_k_logz_with_marginals(scores, k)
    route = RouteMTPRouteOutput(
        scores=scores,
        queries=queries,
        marginals=marginals,
        selected_ids=stable_topk(scores, k),
        hidden=torch.randn(batch, nodes, layers, 8),
        node_summary=torch.randn(batch, nodes, 8),
        target_context=torch.randn(batch, horizons, layers, 8),
    )
    depths = torch.tensor([[1, 2, 3, 4, 3], [1, 2, 3, 4, 4]])
    visible = torch.ones(batch, nodes, dtype=torch.bool)
    probabilities = torch.zeros(batch, horizons, nodes + 1)
    for b in range(batch):
        for h in range(horizons):
            mask = depths[b] == h + 1
            probabilities[b, h, :nodes][mask] = 0.8 / mask.sum()
            probabilities[b, h, -1] = 0.2
    path = RouteMTPPathOutput(
        probabilities=probabilities,
        logits=probabilities.clamp_min(1e-12).log(),
        mask=torch.ones_like(probabilities, dtype=torch.bool),
        node_path_probabilities=probabilities[:, :, :-1].sum(1),
        pre_edge_correction=torch.zeros(batch, nodes),
        post_edge_correction=torch.zeros(batch, nodes),
        local_other_probabilities=torch.zeros(batch, nodes),
    )
    teacher_logits = torch.randn_like(scores)
    teacher_ids = stable_topk(teacher_logits, k)
    teacher_queries = torch.randn_like(queries)
    factual_logits = torch.randn(batch, horizons, layers, experts)
    factual_ids = stable_topk(factual_logits, k)
    objective = RouteMTPObjective(keys, bias, exact_k=k)
    output = objective(
        route=route,
        path=path,
        node_depths=depths,
        visible_mask=visible,
        native_path_log_probabilities=torch.zeros(batch, nodes),
        teacher_node_logits=teacher_logits,
        teacher_node_ids=teacher_ids,
        teacher_node_queries=teacher_queries,
        branch_valid=torch.ones(batch, nodes, layers, dtype=torch.bool),
        anchor_scores=torch.randn(batch, horizons, layers, experts),
        factual_ids=factual_ids,
        factual_valid=torch.ones(batch, horizons, layers, dtype=torch.bool),
        factual_branch_indices=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]]),
    )
    assert all(torch.isfinite(value) for value in output.as_dict().values())
    output.total.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
