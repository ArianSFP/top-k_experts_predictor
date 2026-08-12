"""Exact-cardinality factual mixture objectives for HARP-DeltaRoute v4."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .exact_k import exact_k_logz_with_marginals, validate_exact_set_labels


@dataclass(frozen=True)
class FactualMixtureResult:
    loss: Tensor
    per_row_nll: Tensor
    branch_set_log_probabilities: Tensor
    normalized_probabilities: Tensor


def exact_factual_mixture_nll(
    branch_scores: Tensor,
    anchor_scores: Tensor,
    branch_probabilities: Tensor,
    branch_mask: Tensor,
    true_ids: Tensor,
    *,
    valid: Tensor | None = None,
    k: int = 8,
) -> FactualMixtureResult:
    """Evaluate ``-log sum_b pi_b P_b(S)`` for exact-``k`` branch sets.

    ``branch_probabilities[..., -1]`` and ``anchor_scores`` represent OTHER.
    Masked captured branches are removed and the remaining probabilities are
    normalized explicitly, making this safe for both absolute and conditional
    posterior inputs.
    """

    if branch_scores.ndim < 3:
        raise ValueError("branch scores must end in [nodes, experts]")
    leading = branch_scores.shape[:-2]
    nodes, experts = branch_scores.shape[-2:]
    if anchor_scores.shape != leading + (experts,):
        raise ValueError("anchor and branch score geometry differs")
    expected_without_layer = branch_scores.shape[:-3] + (nodes + 1,)
    if branch_probabilities.shape == expected_without_layer:
        branch_probabilities = branch_probabilities.unsqueeze(-2).expand(
            *leading, nodes + 1
        )
    elif branch_probabilities.shape != leading + (nodes + 1,):
        raise ValueError("branch posterior geometry is invalid")
    if branch_mask.shape == branch_scores.shape[:-3] + (nodes,):
        branch_mask = branch_mask.unsqueeze(-2).expand(*leading, nodes)
    if branch_mask.shape != leading + (nodes,):
        raise ValueError("branch mask geometry is invalid")
    if true_ids.shape != leading + (k,):
        raise ValueError("factual selected-set geometry is invalid")
    if valid is None:
        valid = torch.ones(leading, dtype=torch.bool, device=branch_scores.device)
    safe_ids, weights = validate_exact_set_labels(
        anchor_scores, true_ids, valid=valid, k=k
    )

    scores = torch.cat(
        (branch_scores.float(), anchor_scores[..., None, :].float()), dim=-2
    )
    posterior_mask = torch.cat(
        (branch_mask.bool(), torch.ones_like(branch_mask[..., :1], dtype=torch.bool)),
        dim=-1,
    )
    probabilities = branch_probabilities.float().masked_fill(~posterior_mask, 0.0)
    if not torch.isfinite(probabilities).all() or bool((probabilities < 0).any()):
        raise ValueError("branch probabilities must be finite and non-negative")
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-12)

    log_z, _ = exact_k_logz_with_marginals(scores, k)
    gathered = scores.gather(
        -1, safe_ids[..., None, :].expand(*leading, nodes + 1, k)
    ).sum(-1)
    branch_log_p = gathered - log_z
    log_weights = probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
    mixture_log_p = torch.logsumexp(
        (log_weights + branch_log_p).masked_fill(~posterior_mask, -torch.inf), dim=-1
    )
    per_row = -mixture_log_p
    denominator = weights.sum().clamp_min(1.0)
    loss = (per_row * weights).sum() / denominator
    return FactualMixtureResult(loss, per_row, branch_log_p, probabilities)


__all__ = ["FactualMixtureResult", "exact_factual_mixture_nll"]
