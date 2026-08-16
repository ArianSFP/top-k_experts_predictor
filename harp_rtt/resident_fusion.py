"""Leakage-safe HARP/Resident-Shadow complementarity primitives.

The legacy HARP anchor produces one factual route forecast per horizon.  It is
not a counterfactual branch model.  This module consequently fuses HARP only
after the deployed Shadow-LM branch mixture has been formed.  All privileged
switch/merge functions are explicitly named as oracles and are unsuitable for
runtime use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from .exact_k import cardinality_project_marginals, stable_topk


PREDICTION_SIDECAR_SCHEMA = "harp_resident_prediction_sidecar_v1"
RESIDENT_POLICIES = ("resident_3850", "hot25", "hot25_current")


def validate_resident_policy(policy: str) -> str:
    if policy not in RESIDENT_POLICIES:
        raise ValueError(
            f"resident policy must be one of {RESIDENT_POLICIES}, got {policy!r}"
        )
    return policy


def hot_resident_subset(
    resident_ids_by_layer: Iterable[Tensor], *, per_layer: int = 64
) -> tuple[Tensor, ...]:
    """Take the frozen train-derived prefix of each resident ranking.

    The sealed Resident-Only allocation stores IDs in stable descending
    train-utility order.  A hot-25 view is therefore its first 64 IDs at every
    layer, not a development-label reallocation.
    """

    values = tuple(torch.as_tensor(ids, dtype=torch.long) for ids in resident_ids_by_layer)
    if len(values) != 40:
        raise ValueError("resident policy requires exactly forty layer namespaces")
    if not 1 <= int(per_layer) <= 256:
        raise ValueError("per-layer resident count must lie in 1..256")
    result = []
    for layer, ids in enumerate(values):
        if ids.ndim != 1 or ids.numel() < per_layer:
            raise ValueError(
                f"layer {layer} has fewer than {per_layer} ordered residents"
            )
        if ids.unique().numel() != ids.numel() or bool(((ids < 0) | (ids >= 256)).any()):
            raise ValueError(f"layer {layer} resident namespace is invalid")
        result.append(ids[:per_layer].clone())
    return tuple(result)


def expert_availability_mask(
    resident_ids_by_layer: Iterable[Tensor],
    *,
    current_selected_ids: Tensor | None = None,
    experts: int = 256,
) -> Tensor:
    """Return ``[L,E]`` availability without initiating any expert load."""

    residents = tuple(torch.as_tensor(ids, dtype=torch.long) for ids in resident_ids_by_layer)
    if len(residents) != 40:
        raise ValueError("availability requires forty resident namespaces")
    available = torch.zeros((40, experts), dtype=torch.bool)
    for layer, ids in enumerate(residents):
        if ids.ndim != 1 or bool(((ids < 0) | (ids >= experts)).any()):
            raise ValueError("resident ID lies outside the expert namespace")
        available[layer, ids] = True
    if current_selected_ids is not None:
        current = torch.as_tensor(current_selected_ids, dtype=torch.long)
        if current.ndim != 2 or current.shape[0] != 40:
            raise ValueError("current-token selected IDs must be [40,K]")
        if bool(((current < 0) | (current >= experts)).any()):
            raise ValueError("current-token route contains an invalid expert ID")
        available.scatter_(1, current, True)
    return available


def selected_route_availability(selected_ids: Tensor, availability: Tensor) -> Tensor:
    """Gather a layer/expert availability table for ``[...,L,K]`` routes."""

    ids = selected_ids.long()
    if availability.ndim not in (2, 3):
        raise ValueError("availability must be [L,E] or [B,L,E]")
    layers, experts = availability.shape[-2:]
    if ids.shape[-2] != layers:
        raise ValueError("route and availability layer axes differ")
    if bool(((ids < 0) | (ids >= experts)).any()):
        raise ValueError("selected route contains an invalid expert ID")
    if availability.ndim == 2:
        layer_shape = (1,) * (ids.ndim - 2) + (layers, experts)
        expanded = availability.reshape(layer_shape).expand(*ids.shape[:-1], experts)
    else:
        if ids.ndim < 3 or availability.shape[0] != ids.shape[0]:
            raise ValueError("batched availability does not align with routes")
        extra = ids.ndim - 3
        expanded = availability.reshape(
            availability.shape[0], *((1,) * extra), layers, experts
        ).expand(*ids.shape[:-1], experts)
    return expanded.gather(-1, ids)


def cumulative_prior_omission(
    parent_indices: Tensor,
    node_mask: Tensor,
    immediate: Tensor,
) -> Tensor:
    """Accumulate only omissions causally preceding each router decision.

    ``immediate[n,l]`` is the missing mass/count caused by executing layer
    ``l`` of node ``n``.  Router ``(n,l)`` may observe every layer of ancestor
    tokens and layers ``<l`` of its own token, never its current-layer miss.
    """

    parent = parent_indices.long()
    mask = node_mask.bool()
    if parent.ndim != 1 or mask.shape != parent.shape:
        raise ValueError("tree parent/mask tensors must be aligned vectors")
    if immediate.ndim != 2 or immediate.shape[0] != parent.numel():
        raise ValueError("immediate omission must be [N,L]")
    count = int(mask.sum())
    if count < 1 or not torch.equal(
        mask, torch.arange(mask.numel(), device=mask.device) < count
    ):
        raise ValueError("tree node mask must be a nonempty contiguous prefix")
    if int(parent[0]) != -1:
        raise ValueError("tree root parent must be -1")
    result = torch.zeros_like(immediate)
    completed = torch.zeros(
        parent.numel(), dtype=immediate.dtype, device=immediate.device
    )
    for node in range(count):
        ancestor = int(parent[node])
        if node and not 0 <= ancestor < node:
            raise ValueError("tree parents must precede children")
        inherited = immediate.new_zeros(()) if ancestor < 0 else completed[ancestor]
        within = immediate[node].cumsum(-1) - immediate[node]
        result[node] = inherited + within
        completed[node] = inherited + immediate[node].sum()
    return result


@dataclass(frozen=True)
class PairedSetOverlap:
    both: Tensor
    shadow_unique: Tensor
    harp_unique: Tensor
    neither: Tensor

    @property
    def shadow_recall(self) -> Tensor:
        return self.both + self.shadow_unique

    @property
    def harp_recall(self) -> Tensor:
        return self.both + self.harp_unique

    @property
    def union_recall(self) -> Tensor:
        return 1.0 - self.neither


def paired_set_overlap(
    shadow_ids: Tensor, harp_ids: Tensor, target_ids: Tensor
) -> PairedSetOverlap:
    """Partition true slots into shared, unique, and unrecovered evidence."""

    if shadow_ids.shape != harp_ids.shape or shadow_ids.shape != target_ids.shape:
        raise ValueError("paired overlap requires aligned TopK tensors")
    if target_ids.shape[-1] < 1:
        raise ValueError("paired overlap requires a nonempty selected set")
    shadow_hit = (target_ids[..., :, None] == shadow_ids[..., None, :]).any(-1)
    harp_hit = (target_ids[..., :, None] == harp_ids[..., None, :]).any(-1)
    scale = float(target_ids.shape[-1])
    return PairedSetOverlap(
        both=(shadow_hit & harp_hit).sum(-1).float() / scale,
        shadow_unique=(shadow_hit & ~harp_hit).sum(-1).float() / scale,
        harp_unique=(harp_hit & ~shadow_hit).sum(-1).float() / scale,
        neither=(~shadow_hit & ~harp_hit).sum(-1).float() / scale,
    )


def candidate_union(shadow_ids: Tensor, harp_ranked_ids: Tensor) -> Tensor:
    """Stable unique union with Shadow IDs ordered before HARP additions."""

    if shadow_ids.shape[:-1] != harp_ranked_ids.shape[:-1]:
        raise ValueError("candidate union leading geometry differs")
    experts = torch.cat((shadow_ids.long(), harp_ranked_ids.long()), dim=-1)
    if bool(((experts < 0) | (experts >= 256)).any()):
        raise ValueError("candidate union contains an invalid expert ID")
    flat = experts.reshape(-1, experts.shape[-1])
    rows: list[Tensor] = []
    for row in flat:
        seen: set[int] = set()
        values = []
        for value in row.tolist():
            if value not in seen:
                values.append(value); seen.add(value)
        rows.append(torch.tensor(values, dtype=torch.long, device=row.device))
    width = max(value.numel() for value in rows)
    result = torch.full(
        (len(rows), width), -1, dtype=torch.long, device=experts.device
    )
    for index, row in enumerate(rows):
        result[index, : row.numel()] = row
    return result.reshape(*experts.shape[:-1], width)


def candidate_recall(candidate_ids: Tensor, target_ids: Tensor) -> Tensor:
    if candidate_ids.shape[:-1] != target_ids.shape[:-1]:
        raise ValueError("candidate and target leading geometry differs")
    safe = candidate_ids.clamp_min(0)
    hit = (target_ids[..., :, None] == safe[..., None, :]) & (
        candidate_ids[..., None, :] >= 0
    )
    return hit.any(-1).float().mean(-1)


def oracle_model_switch_recall(
    shadow_ids: Tensor, harp_ids: Tensor, target_ids: Tensor
) -> Tensor:
    overlap = paired_set_overlap(shadow_ids, harp_ids, target_ids)
    return torch.maximum(overlap.shadow_recall, overlap.harp_recall)


def oracle_expert_merge_recall(
    shadow_ids: Tensor,
    harp_ranked_ids: Tensor,
    target_ids: Tensor,
    *,
    swap_caps: Iterable[int] = (1, 2, 4, 8),
    cold_expert_mask: Tensor | None = None,
) -> dict[int, Tensor]:
    """Privileged maximum recall from bounded HARP-for-Shadow swaps."""

    if shadow_ids.shape[:-1] != target_ids.shape[:-1]:
        raise ValueError("shadow and target leading geometry differs")
    union = candidate_union(shadow_ids, harp_ranked_ids)
    shadow_hit = (
        target_ids[..., :, None] == shadow_ids[..., None, :]
    ).any(-1)
    union_hit = (
        target_ids[..., :, None] == union.clamp_min(0)[..., None, :]
    ).any(-1)
    if cold_expert_mask is not None:
        cold = cold_expert_mask.bool()
        if cold.shape != target_ids.shape:
            raise ValueError("cold target mask must align with target IDs")
        union_hit = shadow_hit | (union_hit & cold)
    incumbent = shadow_hit.sum(-1)
    recoverable = (union_hit & ~shadow_hit).sum(-1)
    k = target_ids.shape[-1]
    result: dict[int, Tensor] = {}
    previous: Tensor | None = None
    for cap in sorted({int(value) for value in swap_caps}):
        if not 0 <= cap <= k:
            raise ValueError("swap cap must lie in [0,k]")
        value = (incumbent + recoverable.clamp_max(cap)).float() / float(k)
        if previous is not None and bool((value < previous).any()):
            raise RuntimeError("oracle swap ceilings are not monotone")
        result[cap] = value
        previous = value
    return result


def true_expert_ranks(scores: Tensor, target_ids: Tensor) -> Tensor:
    if scores.shape[:-1] != target_ids.shape[:-1]:
        raise ValueError("score and target leading geometry differs")
    order = torch.argsort(scores.float(), dim=-1, descending=True, stable=True)
    ranks = torch.argsort(order, dim=-1, stable=True)
    return ranks.gather(-1, target_ids.long()) + 1


def _validate_marginals(value: Tensor, name: str, *, k: int = 8) -> None:
    if value.ndim != 4 or value.shape[-2:] != (40, 256):
        raise ValueError(f"{name} must be [B,H,40,256]")
    if not torch.isfinite(value).all() or bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{name} must contain finite inclusion probabilities")
    error = (value.float().sum(-1) - float(k)).abs().amax()
    if float(error) > 5e-3:
        raise ValueError(f"{name} violates exact-cardinality mass by {float(error):.3e}")


def _unit_interval_straight_through(value: Tensor) -> Tensor:
    """Clamp in the forward pass while retaining a usable zero-boundary gradient."""

    clipped = value.clamp(0.0, 1.0)
    return value + (clipped - value).detach()


class ScalarHarpRescue(nn.Module):
    """Zero-baseline horizon/layer interpolation of captured evidence."""

    def __init__(self, *, horizons: int = 4, layers: int = 40) -> None:
        super().__init__()
        self.horizons = int(horizons); self.layers = int(layers)
        self.raw_gate = nn.Parameter(torch.zeros(horizons, layers))

    def forward(
        self,
        *,
        baseline_marginals: Tensor,
        captured_shadow_marginals: Tensor,
        captured_probability: Tensor,
        harp_marginals: Tensor,
    ) -> Tensor:
        _validate_marginals(baseline_marginals, "baseline marginals")
        _validate_marginals(harp_marginals, "HARP marginals")
        if captured_shadow_marginals.shape != baseline_marginals.shape:
            raise ValueError("captured Shadow marginals have invalid geometry")
        batch, horizons, layers, _ = baseline_marginals.shape
        if (horizons, layers) != (self.horizons, self.layers):
            raise ValueError("fusion geometry differs from its configuration")
        probability = captured_probability.float()
        if probability.shape == (batch, horizons):
            probability = probability[..., None].expand(batch, horizons, layers)
        if probability.shape != (batch, horizons, layers):
            raise ValueError("captured probability must be [B,H] or [B,H,L]")
        if bool(((probability < 0) | (probability > 1)).any()):
            raise ValueError("captured probability lies outside [0,1]")
        gate = _unit_interval_straight_through(self.raw_gate)[None, :, :, None]
        replacement = probability[..., None] * harp_marginals
        result = baseline_marginals + gate * (
            replacement - captured_shadow_marginals
        )
        if bool(((result < -1e-5) | (result > 1 + 1e-5)).any()):
            raise RuntimeError("scalar fusion left the marginal simplex")
        return result.clamp(0.0, 1.0)


class ExpertHarpRescue(nn.Module):
    """Small expert-wise calibrated residual with an exact zero outer gate."""

    def __init__(
        self,
        feature_width: int,
        *,
        hidden_width: int = 64,
        horizons: int = 4,
        layers: int = 40,
        clip_log_odds: float = 8.0,
    ) -> None:
        super().__init__()
        if min(feature_width, hidden_width, horizons, layers) < 1:
            raise ValueError("expert rescue dimensions must be positive")
        self.feature_width = int(feature_width)
        self.horizons = int(horizons); self.layers = int(layers)
        self.clip_log_odds = float(clip_log_odds)
        self.horizon_embedding = nn.Embedding(horizons, 8)
        self.layer_embedding = nn.Embedding(layers, 16)
        self.expert_embedding = nn.Embedding(256, 16)
        self.residual = nn.Sequential(
            nn.Linear(feature_width + 40, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )
        self.raw_open_gate = nn.Parameter(torch.zeros(horizons, layers))

    def forward(
        self,
        *,
        baseline_marginals: Tensor,
        harp_marginals: Tensor,
        features: Tensor,
        cold_mask: Tensor | None = None,
    ) -> Tensor:
        _validate_marginals(baseline_marginals, "baseline marginals")
        _validate_marginals(harp_marginals, "HARP marginals")
        batch, horizons, layers, experts = baseline_marginals.shape
        if (horizons, layers, experts) != (self.horizons, self.layers, 256):
            raise ValueError("expert fusion geometry differs from its configuration")
        if features.shape != (*baseline_marginals.shape, self.feature_width):
            raise ValueError("expert fusion features have invalid geometry")
        h = self.horizon_embedding(
            torch.arange(horizons, device=features.device)
        )[None, :, None, None].expand(batch, horizons, layers, experts, -1)
        l = self.layer_embedding(
            torch.arange(layers, device=features.device)
        )[None, None, :, None].expand(batch, horizons, layers, experts, -1)
        e = self.expert_embedding(
            torch.arange(experts, device=features.device)
        )[None, None, None].expand(batch, horizons, layers, experts, -1)
        reliability = self.residual(torch.cat((features, h, l, e), dim=-1)).squeeze(-1)
        epsilon = 1e-5
        shadow_logit = torch.logit(baseline_marginals.clamp(epsilon, 1 - epsilon))
        harp_logit = torch.logit(harp_marginals.clamp(epsilon, 1 - epsilon))
        evidence = (harp_logit - shadow_logit).clamp(
            -self.clip_log_odds, self.clip_log_odds
        )
        if cold_mask is not None:
            if cold_mask.shape != baseline_marginals.shape:
                raise ValueError("cold fusion mask has invalid geometry")
            reliability = reliability * cold_mask.to(reliability)
        proposed = torch.sigmoid(shadow_logit + reliability * evidence)
        proposed, _ = cardinality_project_marginals(proposed, 8)
        gate = _unit_interval_straight_through(self.raw_open_gate)[None, :, :, None]
        # Both terms carry mass eight, so this convex outer attachment is an
        # exact epoch-zero identity and needs no second projection.
        return baseline_marginals + gate * (proposed - baseline_marginals)


def rescue_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "ExpertHarpRescue",
    "PREDICTION_SIDECAR_SCHEMA",
    "PairedSetOverlap",
    "RESIDENT_POLICIES",
    "ScalarHarpRescue",
    "candidate_recall",
    "candidate_union",
    "cumulative_prior_omission",
    "expert_availability_mask",
    "hot_resident_subset",
    "oracle_expert_merge_recall",
    "oracle_model_switch_recall",
    "paired_set_overlap",
    "rescue_parameter_count",
    "selected_route_availability",
    "true_expert_ranks",
    "validate_resident_policy",
]
