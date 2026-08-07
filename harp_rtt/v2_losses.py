"""HARP-RTT v2 branch-semantic, posterior, and swap objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .exact_k import exact_set_nll, stable_topk


@dataclass(frozen=True)
class CounterfactualSemanticLoss:
    total: Tensor
    exact_set: Tensor
    query: Tensor
    router_kl: Tensor
    active_path_depth_cells: int


@dataclass(frozen=True)
class SwapLoss:
    loss: Tensor
    swap_pairs: int
    outside_candidate_misses: int


def _required(mapping: Mapping[str, Tensor], name: str) -> Tensor:
    value = mapping.get(name)
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _gather_branch(
    value: Tensor,
    node_indices: Tensor,
    *,
    horizon: int,
) -> Tensor:
    """Gather [B,H,L,N,...] at one [B,P] node index."""

    if value.ndim < 5:
        raise ValueError("branch prediction must be [B,H,L,N,...]")
    batch, paths = node_indices.shape
    if value.shape[0] != batch:
        raise ValueError("counterfactual batch disagrees with branch prediction")
    selected = value[:, horizon]
    tail = selected.shape[3:]
    index = node_indices.clamp_min(0)[:, None, :, *([None] * len(tail))]
    index = index.expand(batch, selected.shape[1], paths, *tail)
    gathered = selected.gather(2, index)
    permutation = [0, 2, 1] + list(range(3, gathered.ndim))
    return gathered.permute(permutation)


def counterfactual_semantic_loss(
    outputs: Mapping[str, Tensor],
    counterfactual: Mapping[str, Tensor],
    *,
    exact_k: int = 8,
    query_weight: float = 0.2,
    router_kl_weight: float = 0.1,
    query_cosine_weight: float = 0.1,
) -> CounterfactualSemanticLoss:
    """Balance branch translation equally over available path/depth cells."""

    scores = _required(outputs, "branch_geometry_scores")
    queries = _required(outputs, "router_queries")
    node_indices = _required(counterfactual, "node_local_indices").long()
    valid = _required(counterfactual, "valid").bool()
    target_queries = _required(counterfactual, "query_coordinates").detach().float()
    target_logits = _required(counterfactual, "router_logits").detach().float()
    selected_ids = _required(counterfactual, "selected_ids").detach().long()
    if node_indices.ndim != 3 or node_indices.shape[1:] != (4, 4):
        raise ValueError("counterfactual node_local_indices must be [B,4,4]")
    if valid.shape[:3] != node_indices.shape:
        raise ValueError("counterfactual validity disagrees with path geometry")
    if valid[:, :, 0].any():
        raise ValueError("counterfactual H1 supervision must remain masked")
    if scores.shape[:2] != (node_indices.shape[0], 4):
        raise ValueError("branch scores must expose H1--H4")
    if queries.shape[:4] != scores.shape[:4]:
        raise ValueError("branch scores and queries disagree")
    if target_logits.shape[:-1] != valid.shape:
        raise ValueError("counterfactual router logits disagree with validity")
    if target_queries.shape[:-1] != valid.shape:
        raise ValueError("counterfactual queries disagree with validity")
    if selected_ids.shape[:-1] != valid.shape:
        raise ValueError("counterfactual selected IDs disagree with validity")

    zero = scores.sum() * 0.0
    set_cells: list[Tensor] = []
    query_cells: list[Tensor] = []
    kl_cells: list[Tensor] = []
    for depth in range(1, 4):
        indices = node_indices[:, :, depth]
        structural = indices >= 0
        predicted_scores = _gather_branch(scores, indices, horizon=depth)
        predicted_queries = _gather_branch(queries, indices, horizon=depth)
        cell_valid = valid[:, :, depth] & structural[..., None]
        for path in range(4):
            active = cell_valid[:, path]
            if not bool(active.any()):
                continue
            set_cells.append(
                exact_set_nll(
                    predicted_scores[:, path],
                    selected_ids[:, path, depth],
                    valid=active,
                    k=exact_k,
                )
            )
            huber = F.huber_loss(
                predicted_queries[:, path].float(),
                target_queries[:, path, depth],
                reduction="none",
                delta=1.0,
            ).mean(-1)
            cosine = 1.0 - F.cosine_similarity(
                predicted_queries[:, path].float(),
                target_queries[:, path, depth],
                dim=-1,
                eps=1e-8,
            )
            mask = active.float()
            query_cells.append(
                ((huber + query_cosine_weight * cosine) * mask).sum()
                / mask.sum().clamp_min(1.0)
            )
            probability = torch.softmax(target_logits[:, path, depth], dim=-1)
            kl = F.kl_div(
                torch.log_softmax(predicted_scores[:, path].float(), dim=-1),
                probability,
                reduction="none",
            ).sum(-1)
            kl_cells.append((kl * mask).sum() / mask.sum().clamp_min(1.0))
    if not set_cells:
        return CounterfactualSemanticLoss(zero, zero, zero, zero, 0)
    exact = torch.stack(set_cells).mean()
    query = torch.stack(query_cells).mean()
    router = torch.stack(kl_cells).mean()
    total = exact + float(query_weight) * query + float(router_kl_weight) * router
    return CounterfactualSemanticLoss(
        total=total,
        exact_set=exact,
        query=query,
        router_kl=router,
        active_path_depth_cells=len(set_cells),
    )


def target_branch_distribution(
    counterfactual: Mapping[str, Tensor],
    *,
    captured_nodes: int,
) -> tuple[Tensor, Tensor]:
    """Build target captured-path probability plus residual OTHER mass."""

    indices = _required(counterfactual, "node_local_indices").long()
    logp = _required(counterfactual, "target_path_logp").detach().float()
    valid = _required(counterfactual, "valid").bool()
    if indices.shape[1:] != (4, 4) or logp.shape != indices.shape:
        raise ValueError("counterfactual path posterior geometry is invalid")
    batch = indices.shape[0]
    target = logp.new_zeros(batch, 4, captured_nodes + 1)
    endpoint_valid = torch.zeros(batch, 4, dtype=torch.bool, device=logp.device)
    for depth in range(1, 4):
        for path in range(4):
            node = indices[:, path, depth]
            active = valid[:, path, depth].any(-1) & (node >= 0)
            endpoint_valid[:, depth] |= active
            probability = logp[:, path, depth].exp()
            for row in active.nonzero(as_tuple=False).flatten().tolist():
                node_id = int(node[row].item())
                if node_id >= captured_nodes:
                    raise ValueError("counterfactual node index exceeds captured tree")
                target[row, depth, node_id] = torch.maximum(
                    target[row, depth, node_id], probability[row]
                )
        mass = target[:, depth, :-1].sum(-1)
        scale = torch.where(mass > 1.0, 1.0 / mass.clamp_min(1e-12), torch.ones_like(mass))
        target[:, depth, :-1] *= scale[:, None]
        target[:, depth, -1] = (
            1.0 - target[:, depth, :-1].sum(-1)
        ).clamp_min(0.0)
    return target, endpoint_valid


def counterfactual_posterior_loss(
    posterior_logits: Tensor,
    counterfactual: Mapping[str, Tensor],
) -> Tensor:
    """Cross-entropy to target path probabilities plus residual OTHER."""

    if posterior_logits.ndim != 3 or posterior_logits.shape[1] != 4:
        raise ValueError("branch posterior logits must be [B,4,N+1]")
    target, valid = target_branch_distribution(
        counterfactual, captured_nodes=posterior_logits.shape[-1] - 1
    )
    if target.shape != posterior_logits.shape:
        raise ValueError("counterfactual target posterior disagrees with model")
    per_endpoint = -(
        target * torch.log_softmax(posterior_logits.float(), dim=-1)
    ).sum(-1)
    active = valid.float()
    return (per_endpoint * active).sum() / active.sum().clamp_min(1.0)


def swap_loss(
    scores: Tensor,
    base_scores: Tensor,
    true_ids: Tensor,
    *,
    exact_k: int = 8,
    margin: float = 0.125,
    candidate_mask: Tensor | None = None,
) -> SwapLoss:
    """Promote missing true experts above false members of stable base top-k."""

    if scores.shape != base_scores.shape or scores.ndim < 2:
        raise ValueError("final and base scores must share [...,E] geometry")
    if true_ids.shape != scores.shape[:-1] + (exact_k,):
        raise ValueError("swap labels disagree with score geometry")
    experts = scores.shape[-1]
    true_mask = torch.zeros_like(scores, dtype=torch.bool)
    true_mask.scatter_(-1, true_ids.long(), True)
    base_ids = stable_topk(base_scores.detach(), exact_k)
    base_mask = torch.zeros_like(true_mask)
    base_mask.scatter_(-1, base_ids, True)
    missing = true_mask & ~base_mask
    intruding = base_mask & ~true_mask
    pair_mask = missing.unsqueeze(-1) & intruding.unsqueeze(-2)
    pair_values = F.softplus(
        float(margin) + scores.unsqueeze(-2) - scores.unsqueeze(-1)
    )
    # pair_values[..., missing, intruding] currently equals
    # softplus(m + s_intruding - s_missing).
    count = int(pair_mask.sum().item())
    zero = scores.sum() * 0.0
    loss = (
        pair_values.masked_select(pair_mask).mean() if count else zero
    )
    outside = 0
    if candidate_mask is not None:
        if candidate_mask.shape != scores.shape:
            raise ValueError("candidate mask must match dense scores")
        outside = int((true_mask & ~candidate_mask.bool()).sum().item())
    return SwapLoss(loss=loss, swap_pairs=count, outside_candidate_misses=outside)


__all__ = [
    "CounterfactualSemanticLoss",
    "SwapLoss",
    "counterfactual_posterior_loss",
    "counterfactual_semantic_loss",
    "swap_loss",
    "target_branch_distribution",
]
