"""Optimizer-free information ceilings for HARP-DeltaRoute v4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from .b31 import complete_other_mass, selected_ids_inclusion_mass
from .exact_k import stable_topk


def slot_recall_at_k(predicted_ids: Tensor, target_ids: Tensor) -> Tensor:
    if predicted_ids.shape[:-1] != target_ids.shape[:-1]:
        raise ValueError("predicted and target selected sets must share leading geometry")
    return (
        target_ids.long().unsqueeze(-1) == predicted_ids.long().unsqueeze(-2)
    ).any(-1).float().mean(-1)


def factual_branch_topk(
    node_selected_ids: Tensor,
    factual_branch_indices: Tensor,
    anchor_ids: Tensor,
) -> Tensor:
    """Use the captured factual node when present, otherwise anchor/OTHER."""

    if node_selected_ids.ndim != 4:
        raise ValueError("node selected IDs must be [B,N,L,K]")
    batch, nodes, layers, k = node_selected_ids.shape
    if factual_branch_indices.ndim != 2 or factual_branch_indices.shape[0] != batch:
        raise ValueError("factual branch indices must be [B,H]")
    horizons = factual_branch_indices.shape[1]
    if anchor_ids.shape != (batch, horizons, layers, k):
        raise ValueError("anchor selected IDs have invalid geometry")
    indices = factual_branch_indices.long()
    if bool(((indices < 0) | (indices > nodes)).any()):
        raise ValueError("factual branch index lies outside nodes plus OTHER")
    present = indices < nodes
    safe = indices.clamp_max(nodes - 1)
    gathered = node_selected_ids.long()[:, None].expand(
        batch, horizons, nodes, layers, k
    ).gather(
        2, safe[..., None, None, None].expand(batch, horizons, 1, layers, k)
    ).squeeze(2)
    return torch.where(present[..., None, None], gathered, anchor_ids.long())


def posterior_native_topk(
    node_selected_ids: Tensor,
    captured_probabilities: Tensor,
    other_probability: Tensor,
    branch_mask: Tensor,
    anchor_marginals: Tensor,
    *,
    exact_k: int = 8,
) -> tuple[Tensor, Tensor]:
    """Return Top8 and inclusion mass from native routes plus anchor OTHER."""

    if node_selected_ids.ndim == 4:
        batch, nodes, layers, k = node_selected_ids.shape
        horizons = captured_probabilities.shape[1]
        ids = node_selected_ids[:, None].expand(batch, horizons, nodes, layers, k)
        ids = ids.permute(0, 1, 3, 2, 4)
    elif node_selected_ids.ndim == 5:
        ids = node_selected_ids
    else:
        raise ValueError("native selected IDs must be [B,N,L,K] or [B,H,L,N,K]")
    captured = selected_ids_inclusion_mass(
        ids, captured_probabilities, branch_mask,
        experts=anchor_marginals.shape[-1], normalize=False,
    )
    completed = complete_other_mass(captured, other_probability, anchor_marginals)
    return stable_topk(completed, exact_k), completed


def swap_cap_oracle_recall(
    anchor_ids: Tensor,
    candidate_ids: Tensor,
    target_ids: Tensor,
    swap_caps: Iterable[int] = (1, 2, 4, 6, 8),
) -> dict[int, Tensor]:
    """Maximum slot recall after at most ``m`` anchor-to-C64 swaps."""

    if anchor_ids.shape != target_ids.shape:
        raise ValueError("anchor and target TopK geometry differs")
    if candidate_ids.shape[:-1] != target_ids.shape[:-1]:
        raise ValueError("candidate and target leading geometry differs")
    k = target_ids.shape[-1]
    anchor_hit = (
        target_ids[..., None] == anchor_ids[..., None, :]
    ).any(-1)
    candidate_hit = (
        target_ids[..., None] == candidate_ids[..., None, :]
    ).any(-1)
    incumbent_hits = anchor_hit.sum(-1)
    recoverable = (candidate_hit & ~anchor_hit).sum(-1)
    result: dict[int, Tensor] = {}
    previous: Tensor | None = None
    for cap in sorted({int(value) for value in swap_caps}):
        if not 0 <= cap <= k:
            raise ValueError("swap cap must lie in [0,k]")
        recall = (incumbent_hits + recoverable.clamp_max(cap)).float() / float(k)
        if previous is not None and bool((recall < previous).any()):
            raise RuntimeError("swap-cap oracle is not monotone")
        result[cap] = recall
        previous = recall
    return result


def stable_ranks(scores: Tensor) -> Tensor:
    order = torch.argsort(scores.float(), dim=-1, descending=True, stable=True)
    return torch.argsort(order, dim=-1, stable=True)


@dataclass(frozen=True)
class TrueExpertSupportAudit:
    anchor_rank: Tensor
    best_branch_rank: Tensor
    learned_mixture_rank: Tensor
    mtp_mixture_rank: Tensor
    factual_branch_rank: Tensor
    geometry_best_rank: Tensor
    supporting_branches: Tensor


def true_expert_support_audit(
    *,
    target_ids: Tensor,
    anchor_scores: Tensor,
    node_scores: Tensor,
    node_marginals: Tensor,
    learned_probabilities: Tensor,
    mtp_probabilities: Tensor,
    branch_mask: Tensor,
    factual_branch_indices: Tensor,
    geometry_scores: Tensor | None = None,
) -> TrueExpertSupportAudit:
    """Return rank evidence only for factual experts, avoiding dense artifacts."""

    if node_scores.ndim != 5:
        raise ValueError("node scores must be [B,H,L,N,E]")
    if node_marginals.shape != node_scores.shape:
        raise ValueError("node marginal geometry differs from scores")
    batch, horizons, layers, nodes, experts = node_scores.shape
    if target_ids.shape[:3] != (batch, horizons, layers) or anchor_scores.shape != (
        batch, horizons, layers, experts
    ):
        raise ValueError("truth/anchor geometry is invalid")
    if branch_mask.shape != (batch, horizons, nodes):
        raise ValueError("branch mask has invalid geometry")
    for name, probability in (("learned", learned_probabilities), ("mtp", mtp_probabilities)):
        if probability.shape != branch_mask.shape:
            raise ValueError(f"{name} probability geometry is invalid")

    active = branch_mask[:, :, None, :, None]
    masked_scores = node_scores.float().masked_fill(~active, -torch.inf)
    ranks = stable_ranks(masked_scores)
    gather_ids = target_ids.long()[..., None, :].expand(
        batch, horizons, layers, nodes, target_ids.shape[-1]
    )
    node_true_ranks = ranks.gather(-1, gather_ids)
    large = torch.full_like(node_true_ranks, experts)
    best_rank = torch.where(
        active.expand_as(node_true_ranks), node_true_ranks, large
    ).amin(-2)

    def mixture_rank(probabilities: Tensor) -> Tensor:
        mixture = torch.einsum(
            "bhn,bhlne->bhle", probabilities.float(), node_marginals.float()
        )
        return stable_ranks(mixture).gather(-1, target_ids.long())

    safe_factual = factual_branch_indices.long().clamp(0, nodes - 1)
    factual_scores = node_scores.gather(
        3, safe_factual[..., None, None, None].expand(
            batch, horizons, layers, 1, experts
        )
    ).squeeze(3)
    factual_scores = torch.where(
        (factual_branch_indices < nodes)[..., None, None], factual_scores, anchor_scores
    )
    support = node_marginals.gather(-1, gather_ids) > 0.5
    support = (support & active.expand_as(support)).sum(-2)
    geometry = node_scores if geometry_scores is None else geometry_scores
    geometry_ranks = stable_ranks(geometry.float().masked_fill(~active, -torch.inf))
    geometry_best = torch.where(
        active.expand_as(node_true_ranks), geometry_ranks.gather(-1, gather_ids), large
    ).amin(-2)
    return TrueExpertSupportAudit(
        anchor_rank=stable_ranks(anchor_scores).gather(-1, target_ids.long()),
        best_branch_rank=best_rank,
        learned_mixture_rank=mixture_rank(learned_probabilities),
        mtp_mixture_rank=mixture_rank(mtp_probabilities),
        factual_branch_rank=stable_ranks(factual_scores).gather(-1, target_ids.long()),
        geometry_best_rank=geometry_best,
        supporting_branches=support,
    )


__all__ = [
    "TrueExpertSupportAudit", "factual_branch_topk", "posterior_native_topk",
    "slot_recall_at_k", "stable_ranks", "swap_cap_oracle_recall",
    "true_expert_support_audit",
]
