"""Candidate-native losses for HARP's anchor-plus-branch C64 contract."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .exact_k import stable_topk


def candidate_entry_loss(
    aligned_scores: Tensor,
    anchor_scores: Tensor,
    true_ids: Tensor,
    valid: Tensor,
    *,
    anchor_quota: int = 32,
    candidate_width: int = 64,
    margin: float = 0.125,
) -> Tensor:
    """Push each missing true expert across the deployed branch-slot boundary.

    The serving candidate set first fixes ``anchor_quota`` anchor experts and
    fills the remaining slots from aligned branch evidence.  This objective
    uses that exact discrete set in the forward pass.  Only factual experts
    absent from C64 receive loss, against the detached score at the current
    final branch slot.
    """

    if aligned_scores.shape != anchor_scores.shape:
        raise ValueError("candidate entry scores must share geometry")
    if true_ids.shape[:-1] != aligned_scores.shape[:-1]:
        raise ValueError("candidate entry labels disagree with scores")
    if valid.shape != aligned_scores.shape[:-1]:
        raise ValueError("candidate entry validity disagrees with scores")
    experts = aligned_scores.shape[-1]
    branch_quota = candidate_width - anchor_quota
    if not 0 < anchor_quota < candidate_width <= experts:
        raise ValueError("candidate entry quotas are invalid")

    anchor_ids = stable_topk(anchor_scores.detach().float(), anchor_quota)
    anchor_mask = torch.zeros_like(aligned_scores, dtype=torch.bool)
    anchor_mask.scatter_(-1, anchor_ids, True)
    branch_scores = aligned_scores.float().masked_fill(anchor_mask, -torch.inf)
    branch_ids = stable_topk(branch_scores.detach(), branch_quota)
    branch_mask = torch.zeros_like(anchor_mask)
    branch_mask.scatter_(-1, branch_ids, True)
    candidate_mask = anchor_mask | branch_mask

    true_mask = torch.zeros_like(anchor_mask)
    true_mask.scatter_(-1, true_ids.long(), True)
    missing = true_mask & ~candidate_mask & valid.bool()[..., None]
    if not bool(missing.any()):
        return aligned_scores.sum() * 0.0

    boundary = branch_scores.detach().gather(-1, branch_ids)[..., -1]
    violations = F.softplus(
        float(margin) + boundary[..., None] - aligned_scores.float()
    )
    return violations.masked_select(missing).mean()


__all__ = ["candidate_entry_loss"]
