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


def exact_root_greedy_spine_indices(tree: Mapping[str, Tensor]) -> Tensor:
    """Return the structural greedy spine below the exact committed H1 root.

    The H1 root is observed target state, not an MTP branch decision.  Its
    token rank under the pre-root MTP parent may therefore be nonzero and must
    not turn every descendant into a false prefix divergence.
    """

    required = (
        "mask",
        "depth",
        "parent",
        "child_ranks",
        "exact_committed_h1_root",
    )
    values: dict[str, Tensor] = {}
    for name in required:
        value = tree.get(name)
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise ValueError(f"tree.{name} must be [B,N]")
        values[name] = value
    shape = values["mask"].shape
    if any(value.shape != shape for value in values.values()):
        raise ValueError("greedy-spine tree tensors must share [B,N] geometry")

    mask = values["mask"].bool()
    depth = values["depth"].long()
    parent = values["parent"].long()
    child_ranks = values["child_ranks"].long()
    exact_root = values["exact_committed_h1_root"].bool()
    batch, _ = shape
    rows = torch.arange(batch, device=mask.device)
    result = torch.empty(batch, 4, dtype=torch.long, device=mask.device)

    active = mask & exact_root & (depth == 1)
    if bool((active.sum(-1) != 1).any()):
        raise ValueError("adaptive tree does not contain one exact committed H1 root")
    previous = active.long().argmax(-1)
    result[:, 0] = previous
    for horizon in range(2, 5):
        active = (
            mask
            & (depth == horizon)
            & (parent == previous[:, None])
            & (child_ranks == 0)
        )
        if bool((active.sum(-1) != 1).any()):
            bad = rows[active.sum(-1) != 1]
            raise ValueError(
                "adaptive tree does not contain one structural greedy child "
                f"at H{horizon}; first bad batch row {int(bad[0])}"
            )
        previous = active.long().argmax(-1)
        result[:, horizon - 1] = previous
    return result


def anchor_spine_prefix_matches(
    exact_prefix_hashes: Tensor,
    future_prefix_hashes: Tensor,
) -> Tensor:
    """Match factual H1--H4 prefixes to the complete causal MTP spine."""

    if (
        exact_prefix_hashes.ndim != 3
        or exact_prefix_hashes.shape[1] < 4
        or exact_prefix_hashes.shape[2] != 32
    ):
        raise ValueError("anchor spine exact prefix hashes must be [B,D>=4,32]")
    expected = (exact_prefix_hashes.shape[0], 4, 32)
    if future_prefix_hashes.shape != expected:
        raise ValueError("future factual prefix hashes must be [B,4,32]")
    matches = (
        exact_prefix_hashes[:, :4].to(torch.uint8)
        == future_prefix_hashes.to(torch.uint8)
    ).all(-1)
    if not bool(matches[:, 0].all()):
        raise ValueError("complete MTP spine does not begin at the exact factual H1 root")
    return matches


def legacy_b3_divergence_depths(tree: Mapping[str, Tensor]) -> Tensor:
    """Reconstruct the exact legacy B3 feature, including H1 root rank.

    This exists only to evaluate the frozen B3 checkpoint on the feature
    semantics it was trained with.  New training uses the corrected dataset
    feature, which excludes the observed H1 root from branch divergence.
    """

    for name in ("mask", "depth", "parent", "child_ranks"):
        value = tree.get(name)
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise ValueError(f"tree.{name} must be [B,N]")
    mask = tree["mask"].bool()
    depth = tree["depth"].long()
    parent = tree["parent"].long()
    child_ranks = tree["child_ranks"].long()
    if not (mask.shape == depth.shape == parent.shape == child_ranks.shape):
        raise ValueError("legacy B3 divergence tensors must share [B,N] geometry")
    result = torch.zeros_like(depth)
    for local in range(depth.shape[1]):
        parent_index = parent[:, local]
        inherited = result.gather(1, parent_index.clamp_min(0)[:, None]).squeeze(1)
        local_divergence = torch.where(
            mask[:, local] & (child_ranks[:, local] > 0),
            depth[:, local],
            torch.zeros_like(depth[:, local]),
        )
        result[:, local] = torch.where(
            (parent_index >= 0) & (inherited > 0), inherited, local_divergence
        )
    return result


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
    # The policy is three lexicographic groups: the guaranteed anchor prefix,
    # positive non-anchor branch evidence, then unused anchor fallback.  Dense
    # integer ranks implement the old row/rank insertion loop exactly while
    # avoiding hundreds of device synchronizations during evaluation.
    anchor_rank = torch.argsort(anchor_order, dim=-1, stable=True)
    branch_rank = torch.argsort(branch_order, dim=-1, stable=True)
    guaranteed = anchor_rank < anchor_quota
    positive_branch = (branch_flat > 0) & ~guaranteed
    group = torch.where(
        guaranteed,
        torch.zeros_like(anchor_rank),
        torch.where(
            positive_branch,
            torch.ones_like(anchor_rank),
            torch.full_like(anchor_rank, 2),
        ),
    )
    within_group = torch.where(positive_branch, branch_rank, anchor_rank)
    policy_key = group * experts + within_group
    ids = torch.argsort(policy_key, dim=-1, stable=True)[:, :width]
    selected = torch.zeros(
        rows, experts, dtype=torch.bool, device=anchor_scores.device
    )
    selected.scatter_(1, ids, True)
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
