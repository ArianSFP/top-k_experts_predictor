"""Raw-MTP-prior exact-cardinality aggregation for ShadowRoute."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .exact_k import cardinality_project_marginals, exact_k_marginals, stable_topk


@dataclass(frozen=True)
class ShadowRouteMixture:
    branch_marginals: Tensor
    branch_weights: Tensor
    other_weights: Tensor
    mixture_marginals: Tensor
    selected_ids: Tensor
    cardinality_error: Tensor


def raw_mtp_prior_mixture(
    branch_scores: Tensor,
    node_depths: Tensor,
    source_path_logp: Tensor,
    node_valid: Tensor,
    anchor_marginals: Tensor,
    *,
    exact_k: int = 8,
    horizons: int = 4,
    tolerance: float = 2e-5,
) -> ShadowRouteMixture:
    """Mix per-node exact-k marginals by causal MTP path mass plus OTHER.

    Shapes are ``branch_scores [B,N,L,E]``, topology/probabilities ``[B,N]``
    and ``anchor_marginals [B,H,L,E]``.  Node depth one is exact H1.  Nodes at
    depth H supply the branch distribution for horizon H.
    """

    if branch_scores.ndim != 4:
        raise ValueError("branch scores must be [B,N,L,E]")
    batch, nodes, layers, experts = branch_scores.shape
    if node_depths.shape != (batch, nodes):
        raise ValueError("node depths disagree with branch scores")
    if source_path_logp.shape != (batch, nodes) or node_valid.shape != (batch, nodes):
        raise ValueError("node probabilities/validity disagree with branch scores")
    if anchor_marginals.shape != (batch, horizons, layers, experts):
        raise ValueError("anchor marginals have invalid geometry")
    if not torch.isfinite(branch_scores).all() or not torch.isfinite(anchor_marginals).all():
        raise ValueError("route scores/marginals must be finite")
    valid = node_valid.bool()
    if not valid[:, 0].all() or not (node_depths[:, 0] == 1).all():
        raise ValueError("every tree must expose exact H1 at node zero")
    active_logp = source_path_logp[valid]
    if not torch.isfinite(active_logp).all() or bool((active_logp > tolerance).any()):
        raise ValueError("valid MTP path log probabilities are invalid")
    if torch.isfinite(source_path_logp[~valid]).any():
        raise ValueError("padded nodes contain source path probability")

    branch_marginals = exact_k_marginals(branch_scores.float(), exact_k)
    weights = branch_scores.new_zeros(batch, horizons, nodes, dtype=torch.float32)
    other = branch_scores.new_zeros(batch, horizons, dtype=torch.float32)
    mixture = branch_scores.new_zeros(batch, horizons, layers, experts, dtype=torch.float32)
    mixture[:, 0] = branch_marginals[:, 0]
    weights[:, 0, 0] = 1.0
    for horizon in range(2, horizons + 1):
        selected = valid & (node_depths == horizon)
        probability = torch.where(
            selected, source_path_logp.float().exp(), torch.zeros_like(source_path_logp.float())
        )
        mass = probability.sum(-1)
        if bool((mass > 1.0 + tolerance).any()):
            raise ValueError("captured MTP path mass exceeds one")
        # Preserve sub-unit captured probability; the residual belongs to OTHER.
        # Fixed-depth path prefixes are disjoint, so their raw causal mass must
        # not be renormalized over only the captured subset.
        residual = (1.0 - mass).clamp(0.0, 1.0)
        weights[:, horizon - 1] = probability
        other[:, horizon - 1] = residual
        mixture[:, horizon - 1] = (
            torch.einsum("bn,bnle->ble", probability, branch_marginals)
            + residual[:, None, None] * anchor_marginals[:, horizon - 1].float()
        )
    projected, error = cardinality_project_marginals(mixture, exact_k)
    selected_ids = stable_topk(projected, exact_k)
    return ShadowRouteMixture(
        branch_marginals=branch_marginals,
        branch_weights=weights,
        other_weights=other,
        mixture_marginals=projected,
        selected_ids=selected_ids,
        cardinality_error=error,
    )


__all__ = ["ShadowRouteMixture", "raw_mtp_prior_mixture"]
