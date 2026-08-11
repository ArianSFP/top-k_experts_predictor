"""Metric-aligned B3.1 source-factorial diagnostics.

The B2 translator was promoted using posterior-weighted native selected sets,
whereas B3 proposed candidates from maxima over heterogeneous raw score
vectors. This module makes that distinction explicit and provides a pure-
tensor evaluator that can be run on already captured artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from .exact_k import stable_topk


@dataclass(frozen=True)
class CandidatePolicyResult:
    """One deterministic C-width candidate policy."""

    expert_ids: Tensor
    dense_mask: Tensor
    anchor_quota: int | None


def _validate_dense_scores(scores: Tensor, *, name: str) -> None:
    if scores.ndim != 4:
        raise ValueError(f"{name} must be [B,H,L,E]")
    if not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError(f"{name} must be finite floating point")


def selected_set_inclusion_mass(
    branch_scores: Tensor,
    posterior_weights: Tensor,
    branch_mask: Tensor,
    *,
    exact_k: int = 8,
    normalize: bool = True,
) -> Tensor:
    """Return posterior selected-set mass, not router-softmax probability.

    ``normalize=True`` accepts posterior weights conditioned on the captured
    nodes. Set it to false when weights already carry absolute probability and
    an explicit ``OTHER`` residual will be added later.
    """

    if branch_scores.ndim != 5:
        raise ValueError("branch_scores must be [B,H,L,N,E]")
    expected = branch_scores.shape[:2] + (branch_scores.shape[3],)
    if posterior_weights.shape != expected or branch_mask.shape != expected:
        raise ValueError("branch posterior geometry disagrees with branch scores")
    if not 1 <= exact_k <= branch_scores.shape[-1]:
        raise ValueError("exact_k lies outside the expert namespace")
    if not posterior_weights.is_floating_point() or not torch.isfinite(
        posterior_weights
    ).all():
        raise ValueError("posterior_weights must be finite floating point")
    if (posterior_weights < 0).any():
        raise ValueError("posterior_weights must be non-negative")

    visible = branch_mask.bool()
    weights = posterior_weights.float() * visible.float()
    denominator = weights.sum(-1, keepdim=True)
    active = denominator.squeeze(-1) > 0
    if normalize:
        weights = weights / denominator.clamp_min(1e-12)
    elif bool((denominator > 1.0 + 1e-5).any()):
        raise ValueError("absolute captured posterior mass exceeds one")

    selected = stable_topk(branch_scores.float(), exact_k)
    batch, horizons, layers, nodes, _ = branch_scores.shape
    experts = branch_scores.shape[-1]
    mass = branch_scores.new_zeros(batch, horizons, layers, experts).float()
    source = weights[:, :, None, :, None].expand(
        batch, horizons, layers, nodes, exact_k
    )
    mass.scatter_add_(
        -1,
        selected.reshape(batch, horizons, layers, nodes * exact_k),
        source.reshape(batch, horizons, layers, nodes * exact_k),
    )
    return mass * active[:, :, None, None].float()


def selected_ids_inclusion_mass(
    selected_ids: Tensor,
    posterior_weights: Tensor,
    branch_mask: Tensor,
    *,
    experts: int,
    normalize: bool = False,
) -> Tensor:
    """Inclusion mass from authoritative or precomputed selected expert IDs."""

    if selected_ids.ndim != 5:
        raise ValueError("branch selected IDs must be [B,H,L,N,K]")
    batch, horizons, layers, nodes, exact_k = selected_ids.shape
    expected = (batch, horizons, nodes)
    if posterior_weights.shape != expected or branch_mask.shape != expected:
        raise ValueError("branch selected-ID posterior geometry is invalid")
    ids = selected_ids.long()
    active_ids = branch_mask[:, :, None, :, None].bool().expand_as(ids)
    if bool((((ids < 0) | (ids >= experts)) & active_ids).any()):
        raise ValueError("branch selected ID lies outside the expert namespace")
    ids = torch.where(active_ids, ids, torch.zeros_like(ids))
    weights = posterior_weights.float() * branch_mask.bool().float()
    denominator = weights.sum(-1, keepdim=True)
    active = denominator.squeeze(-1) > 0
    if normalize:
        weights = weights / denominator.clamp_min(1e-12)
    elif bool((denominator > 1.0 + 1e-5).any()):
        raise ValueError("absolute captured posterior mass exceeds one")
    mass = weights.new_zeros(batch, horizons, layers, experts)
    source = weights[:, :, None, :, None].expand(
        batch, horizons, layers, nodes, exact_k
    )
    mass.scatter_add_(
        -1,
        ids.reshape(batch, horizons, layers, nodes * exact_k),
        source.reshape(batch, horizons, layers, nodes * exact_k),
    )
    return mass * active[:, :, None, None].float()


def complete_other_mass(
    captured_mass: Tensor,
    other_probability: Tensor,
    anchor_marginals: Tensor,
) -> Tensor:
    """Complete uncaptured probability through the exact-k anchor."""

    _validate_dense_scores(captured_mass, name="captured_mass")
    _validate_dense_scores(anchor_marginals, name="anchor_marginals")
    if captured_mass.shape != anchor_marginals.shape:
        raise ValueError("captured and anchor marginals must share geometry")
    expected = captured_mass.shape[:2]
    if other_probability.shape != expected:
        raise ValueError("other_probability must be [B,H]")
    if not torch.isfinite(other_probability).all() or (
        (other_probability < 0) | (other_probability > 1)
    ).any():
        raise ValueError("other_probability must lie in [0,1]")
    return captured_mass.float() + other_probability[:, :, None, None].float() * (
        anchor_marginals.float()
    )


def quota_candidate_union(
    anchor_scores: Tensor,
    branch_mass: Tensor,
    *,
    anchor_quota: int,
    width: int = 64,
) -> CandidatePolicyResult:
    """Anchor quota plus strictly-positive branch mass with anchor fallback."""

    _validate_dense_scores(anchor_scores, name="anchor_scores")
    _validate_dense_scores(branch_mass, name="branch_mass")
    if anchor_scores.shape != branch_mass.shape:
        raise ValueError("anchor and branch mass must share geometry")
    experts = anchor_scores.shape[-1]
    if not 1 <= anchor_quota <= width <= experts:
        raise ValueError("candidate quota must satisfy 1 <= quota <= width <= E")

    rows = anchor_scores.numel() // experts
    anchor_order = torch.argsort(
        anchor_scores.float().reshape(rows, experts),
        dim=-1,
        descending=True,
        stable=True,
    )
    branch_flat = branch_mass.float().reshape(rows, experts)
    branch_order = torch.argsort(
        branch_flat, dim=-1, descending=True, stable=True
    )
    selected = torch.zeros(rows, experts, dtype=torch.bool, device=anchor_scores.device)
    ids = torch.full(
        (rows, width), -1, dtype=torch.long, device=anchor_scores.device
    )
    counts = torch.zeros(rows, dtype=torch.long, device=anchor_scores.device)
    row_ids = torch.arange(rows, device=anchor_scores.device)

    def add(values: Tensor, *, positive: Tensor | None = None) -> None:
        is_new = ~selected.gather(1, values[:, None]).squeeze(1)
        keep = is_new & (counts < width)
        if positive is not None:
            keep &= positive
        if bool(keep.any()):
            active_rows = row_ids[keep]
            active_ids = values[keep]
            slots = counts[keep]
            ids[active_rows, slots] = active_ids
            selected[active_rows, active_ids] = True
            counts[keep] += 1

    for rank in range(anchor_quota):
        add(anchor_order[:, rank])
    for rank in range(experts):
        if bool((counts >= width).all()):
            break
        candidate = branch_order[:, rank]
        positive = branch_flat.gather(1, candidate[:, None]).squeeze(1) > 0
        add(candidate, positive=positive)
    for rank in range(experts):
        if bool((counts >= width).all()):
            break
        add(anchor_order[:, rank])
    if bool((ids < 0).any()):
        raise RuntimeError("candidate union did not produce the configured width")
    leading = anchor_scores.shape[:-1]
    return CandidatePolicyResult(
        expert_ids=ids.reshape(*leading, width),
        dense_mask=selected.reshape(*leading, experts),
        anchor_quota=anchor_quota,
    )


def global_candidate_union(
    anchor_marginals: Tensor,
    branch_marginals: Tensor,
    *,
    width: int = 64,
) -> CandidatePolicyResult:
    """Globally rank calibrated exact-k evidence from anchor and branches."""

    _validate_dense_scores(anchor_marginals, name="anchor_marginals")
    _validate_dense_scores(branch_marginals, name="branch_marginals")
    if anchor_marginals.shape != branch_marginals.shape:
        raise ValueError("global candidate marginals must share geometry")
    if not 1 <= width <= anchor_marginals.shape[-1]:
        raise ValueError("global candidate width lies outside the expert namespace")
    anchor = anchor_marginals.float().clamp(0, 1)
    branch = branch_marginals.float().clamp(0, 1)
    fused = 1.0 - (1.0 - anchor) * (1.0 - branch)
    ids = stable_topk(fused, width)
    mask = torch.zeros_like(fused, dtype=torch.bool)
    mask.scatter_(-1, ids, True)
    return CandidatePolicyResult(ids, mask, None)


def slot_coverage_at_k(candidate_ids: Tensor, target_ids: Tensor) -> Tensor:
    """Per-layer true-slot coverage for candidate sets."""

    if candidate_ids.shape[:-1] != target_ids.shape[:-1]:
        raise ValueError("candidate and target leading geometry differs")
    membership = (
        target_ids.long().unsqueeze(-1) == candidate_ids.long().unsqueeze(-2)
    ).any(-1)
    return membership.float().mean(-1)


def source_diagnostics(
    anchor_scores: Tensor,
    branch_mass: Tensor,
    candidates: CandidatePolicyResult,
) -> dict[str, float]:
    """Summarize positive evidence, overlap, and branch-only contribution."""

    _validate_dense_scores(anchor_scores, name="anchor_scores")
    _validate_dense_scores(branch_mass, name="branch_mass")
    width = candidates.expert_ids.shape[-1]
    anchor_ids = stable_topk(anchor_scores.float(), width)
    anchor_mask = torch.zeros_like(anchor_scores, dtype=torch.bool)
    anchor_mask.scatter_(-1, anchor_ids, True)
    positive = branch_mass > 0
    chosen = candidates.dense_mask
    return {
        "mean_positive_branch_experts": float(positive.float().sum(-1).mean()),
        "mean_candidate_anchor_overlap": float(
            (chosen & anchor_mask).float().sum(-1).mean()
        ),
        "mean_branch_only_candidates": float(
            (chosen & ~anchor_mask).float().sum(-1).mean()
        ),
    }


def evaluate_factorial(
    *,
    anchor_scores: Tensor,
    anchor_marginals: Tensor,
    target_ids: Tensor,
    branch_scores: Mapping[str, Tensor],
    posteriors: Mapping[str, tuple[Tensor, Tensor]],
    branch_mask: Tensor,
    exact_k: int = 8,
    width: int = 64,
    strata_masks: Mapping[str, Tensor] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate route-source x posterior x candidate-policy conditions."""

    _validate_dense_scores(anchor_scores, name="anchor_scores")
    _validate_dense_scores(anchor_marginals, name="anchor_marginals")
    if target_ids.shape != anchor_scores.shape[:-1] + (exact_k,):
        raise ValueError("target selected-set geometry is invalid")
    report: dict[str, dict[str, float]] = {}
    for source_name, scores in sorted(branch_scores.items()):
        for posterior_name, (captured_weights, other) in sorted(posteriors.items()):
            captured = selected_set_inclusion_mass(
                scores,
                captured_weights,
                branch_mask,
                exact_k=exact_k,
                normalize=False,
            )
            completed = complete_other_mass(captured, other, anchor_marginals)
            conditions: list[tuple[str, CandidatePolicyResult]] = [
                (
                    f"quota_{quota}_{width - quota}",
                    quota_candidate_union(
                        anchor_scores, captured, anchor_quota=quota, width=width
                    ),
                )
                for quota in (48, 40, 32)
            ]
            conditions.append(
                (
                    "global",
                    global_candidate_union(
                        anchor_marginals, completed, width=width
                    ),
                )
            )
            for policy_name, candidates in conditions:
                coverage = slot_coverage_at_k(candidates.expert_ids, target_ids)
                key = f"{source_name}__{posterior_name}__{policy_name}"
                metrics = {
                    "coverage": float(coverage.mean()),
                    **source_diagnostics(anchor_scores, captured, candidates),
                }
                for horizon in range(coverage.shape[1]):
                    metrics[f"coverage_h{horizon + 1}"] = float(
                        coverage[:, horizon].mean()
                    )
                _add_stratified_coverage(metrics, coverage, strata_masks)
                report[key] = metrics
    return report


def evaluate_selected_factorial(
    *,
    anchor_scores: Tensor,
    anchor_marginals: Tensor,
    target_ids: Tensor,
    branch_selected_ids: Mapping[str, Tensor],
    posteriors: Mapping[str, tuple[Tensor, Tensor]],
    branch_mask: Tensor,
    width: int = 64,
    strata_masks: Mapping[str, Tensor] | None = None,
) -> dict[str, dict[str, float]]:
    """Factorial evaluator for compact/native branch selected-set bundles."""

    _validate_dense_scores(anchor_scores, name="anchor_scores")
    _validate_dense_scores(anchor_marginals, name="anchor_marginals")
    experts = anchor_scores.shape[-1]
    report: dict[str, dict[str, float]] = {}
    for source_name, ids in sorted(branch_selected_ids.items()):
        if ids.shape[:3] != anchor_scores.shape[:3] or ids.shape[3] != branch_mask.shape[-1]:
            raise ValueError(f"selected-set source {source_name} geometry is invalid")
        if ids.shape[-1] != target_ids.shape[-1]:
            raise ValueError("selected-set source exact-k differs from targets")
        for posterior_name, (captured_weights, other) in sorted(posteriors.items()):
            captured = selected_ids_inclusion_mass(
                ids, captured_weights, branch_mask, experts=experts, normalize=False
            )
            completed = complete_other_mass(captured, other, anchor_marginals)
            conditions: list[tuple[str, CandidatePolicyResult]] = [
                (
                    f"quota_{quota}_{width - quota}",
                    quota_candidate_union(
                        anchor_scores, captured, anchor_quota=quota, width=width
                    ),
                )
                for quota in (48, 40, 32)
            ]
            conditions.append(
                ("global", global_candidate_union(anchor_marginals, completed, width=width))
            )
            for policy_name, candidates in conditions:
                coverage = slot_coverage_at_k(candidates.expert_ids, target_ids)
                metrics = {
                    "coverage": float(coverage.mean()),
                    **source_diagnostics(anchor_scores, captured, candidates),
                }
                for horizon in range(coverage.shape[1]):
                    metrics[f"coverage_h{horizon + 1}"] = float(coverage[:, horizon].mean())
                _add_stratified_coverage(metrics, coverage, strata_masks)
                report[f"{source_name}__{posterior_name}__{policy_name}"] = metrics
    return report


def _add_stratified_coverage(
    metrics: dict[str, float],
    coverage: Tensor,
    strata_masks: Mapping[str, Tensor] | None,
) -> None:
    if strata_masks is None:
        return
    expected = coverage.shape[:2]
    for name, mask in sorted(strata_masks.items()):
        if mask.shape != expected:
            raise ValueError(f"stratum {name} must be [B,H]")
        active = mask.bool()[..., None].expand_as(coverage)
        count = int(active.sum().item())
        metrics[f"stratum_{name}_cells"] = float(count)
        if count:
            metrics[f"stratum_{name}_coverage"] = float(
                coverage.masked_select(active).mean()
            )


__all__ = [
    "CandidatePolicyResult",
    "complete_other_mass",
    "evaluate_factorial",
    "global_candidate_union",
    "quota_candidate_union",
    "selected_set_inclusion_mass",
    "slot_coverage_at_k",
    "source_diagnostics",
]
