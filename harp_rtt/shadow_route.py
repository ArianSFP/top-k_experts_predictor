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


@dataclass(frozen=True)
class ShadowPathPosterior:
    cumulative_log_probabilities: Tensor
    captured_probabilities: Tensor
    other_probabilities: Tensor


def shadow_lm_path_posterior(
    vocabulary_log_probabilities: Tensor,
    token_ids: Tensor,
    parent_indices: Tensor,
    node_depths: Tensor,
    node_valid: Tensor,
    branch_mask: Tensor,
    *,
    horizons: int = 4,
    tolerance: float = 2e-5,
) -> ShadowPathPosterior:
    """Causal path posterior from the frozen LM head on shadow states.

    Each node's vocabulary distribution predicts its children.  The exact H1
    root is conditioned on and therefore has cumulative log probability zero.
    Only the immutable deployed branch mask contributes captured mass; the
    residual probability is assigned to OTHER.
    """

    if vocabulary_log_probabilities.ndim != 3:
        raise ValueError("shadow vocabulary log probabilities must be [B,N,V]")
    batch, nodes, vocabulary = vocabulary_log_probabilities.shape
    expected = (batch, nodes)
    for name, value in (
        ("token IDs", token_ids),
        ("parent indices", parent_indices),
        ("node depths", node_depths),
        ("node validity", node_valid),
    ):
        if value.shape != expected:
            raise ValueError(f"shadow {name} disagree with vocabulary predictions")
    if branch_mask.shape != (batch, horizons, nodes):
        raise ValueError("shadow branch mask has invalid geometry")
    valid = node_valid.bool()
    if not valid[:, 0].all() or not (node_depths[:, 0] == 1).all():
        raise ValueError("every shadow tree must expose the exact H1 root")
    if bool(((token_ids < 0) | (token_ids >= vocabulary))[valid].any()):
        raise ValueError("shadow tree token lies outside the vocabulary")
    if not torch.isfinite(vocabulary_log_probabilities[valid]).all():
        raise ValueError("valid shadow vocabulary log probabilities must be finite")
    cumulative = vocabulary_log_probabilities.new_full((batch, nodes), float("nan"))
    cumulative[:, 0] = 0.0
    row = torch.arange(batch, device=vocabulary_log_probabilities.device)
    for node in range(1, nodes):
        active = valid[:, node]
        if not bool(active.any()):
            continue
        parent = parent_indices[:, node].long()
        if bool(((parent < 0) | (parent >= node))[active].any()):
            raise ValueError("shadow parents must precede their children")
        parent_safe = parent.clamp_min(0)
        edge = vocabulary_log_probabilities[
            row, parent_safe, token_ids[:, node].long().clamp(0, vocabulary - 1)
        ]
        value = cumulative[row, parent_safe] + edge
        if not torch.isfinite(value[active]).all():
            raise ValueError("shadow path contains an invalid parent probability")
        cumulative[:, node] = torch.where(active, value, cumulative[:, node])
    captured = vocabulary_log_probabilities.new_zeros(
        batch, horizons, nodes, dtype=torch.float32
    )
    for horizon in range(2, horizons + 1):
        selected = branch_mask[:, horizon - 1].bool() & valid & (
            node_depths == horizon
        )
        captured[:, horizon - 1] = torch.where(
            selected, cumulative.float().exp(), torch.zeros_like(cumulative.float())
        )
    mass = captured.sum(-1)
    if bool((mass > 1.0 + tolerance).any()):
        raise ValueError("shadow LM captured path mass exceeds one")
    other = (1.0 - mass).clamp(0.0, 1.0)
    return ShadowPathPosterior(cumulative, captured, other)


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


__all__ = [
    "ShadowPathPosterior", "ShadowRouteMixture", "raw_mtp_prior_mixture",
    "shadow_lm_path_posterior",
]
