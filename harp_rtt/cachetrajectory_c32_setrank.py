"""Factorized permutation-equivariant C32 replacement ranker.

The serving score is a conservative residual over the deployed dense score.
The second head predicts current-set survival continuously; it never imposes a
fixed survivor quota. Both heads share a lightweight per-candidate trunk.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cachetrajectory_c32 import (
    EXPERTS,
    FEATURE_WIDTH,
    HORIZONS,
    LAYERS,
    TOP_K,
)


CANDIDATE_WIDTH = 32
FROZEN_C16_LOWER_BOUND_MACS = 623_360
OBJECT_WIDTH = 8
HORIZON_WIDTH = 4
LAYER_WIDTH = 4
STATE_WIDTH = 20


class EquivariantCacheTrajectory32(nn.Module):
    """Factorized C32 ranker with factual-residual and survivor heads."""

    def __init__(
        self,
        feature_width: int = FEATURE_WIDTH,
        candidate_width: int = CANDIDATE_WIDTH,
    ) -> None:
        super().__init__()
        self.feature_width = int(feature_width)
        self.candidate_width = int(candidate_width)
        if not TOP_K < self.candidate_width <= CANDIDATE_WIDTH:
            raise ValueError("candidate width must lie in [9,32]")
        self.object_embedding = nn.Embedding(LAYERS * EXPERTS, OBJECT_WIDTH)
        self.horizon_embedding = nn.Embedding(HORIZONS, HORIZON_WIDTH)
        self.layer_embedding = nn.Embedding(LAYERS, LAYER_WIDTH)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, STATE_WIDTH),
            nn.GELU(),
        )
        trunk_input = STATE_WIDTH + OBJECT_WIDTH + HORIZON_WIDTH + LAYER_WIDTH
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input, STATE_WIDTH),
            nn.GELU(),
        )
        self.factual_residual_head = nn.Linear(STATE_WIDTH, 1)
        self.survivor_head = nn.Linear(STATE_WIDTH, 1)
        # A fresh checkpoint is exactly the deployed dense ranking when its
        # factual residual is used without the separately calibrated survivor
        # term.
        nn.init.zeros_(self.factual_residual_head.weight)
        nn.init.zeros_(self.factual_residual_head.bias)

    def forward(
        self,
        features: Tensor,
        candidate_ids: Tensor,
        horizons: Tensor,
        layers: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or features.shape[1:] != (
            self.candidate_width,
            self.feature_width,
        ):
            raise ValueError(
                "features must have [cells,candidate_width,feature_width] geometry"
            )
        if candidate_ids.shape != features.shape[:2]:
            raise ValueError("candidate IDs do not align with features")
        if horizons.shape != features.shape[:1] or layers.shape != features.shape[:1]:
            raise ValueError("horizon/layer indices do not align with cells")
        feature_state = self.feature_projection(
            features.to(self.feature_projection[0].weight.dtype)
        )
        objects = self.object_embedding(
            layers.long()[:, None] * EXPERTS + candidate_ids.long()
        )
        horizon = self.horizon_embedding(horizons.long())[:, None].expand(
            -1, self.candidate_width, -1
        )
        layer = self.layer_embedding(layers.long())[:, None].expand(
            -1, self.candidate_width, -1
        )
        output = self.trunk(
            torch.cat((feature_state, objects, horizon, layer), dim=-1)
        )
        return (
            self.factual_residual_head(output).squeeze(-1),
            self.survivor_head(output).squeeze(-1),
        )


def parameter_and_mac_contract(width: int = CANDIDATE_WIDTH) -> dict[str, Any]:
    """Return exact learned parameter bytes and linear MACs per route cell."""
    if not TOP_K < width <= CANDIDATE_WIDTH:
        raise ValueError("candidate width must lie in [9,32]")
    model = EquivariantCacheTrajectory32(candidate_width=width)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trunk_input = STATE_WIDTH + OBJECT_WIDTH + HORIZON_WIDTH + LAYER_WIDTH
    linear_macs = width * (
        FEATURE_WIDTH * STATE_WIDTH
        + trunk_input * STATE_WIDTH
        + STATE_WIDTH * 2
    )
    return {
        "parameters": parameters,
        "deployment_bf16_bytes": parameters * 2,
        "linear_macs_per_cell": linear_macs,
        "frozen_c16_lower_bound_macs_per_cell": FROZEN_C16_LOWER_BOUND_MACS,
        "mac_headroom_per_cell": FROZEN_C16_LOWER_BOUND_MACS - linear_macs,
        "mac_ratio": linear_macs / FROZEN_C16_LOWER_BOUND_MACS,
        "candidate_width": width,
        "fixed_survivor_quota": False,
    }


def centered_survivor_score(survivor_logits: Tensor, is_current: Tensor) -> Tensor:
    """Center survivor logits over the eight current candidates in each cell."""
    current_float = is_current.float()
    mean = (survivor_logits.float() * current_float).sum(-1, keepdim=True) / (
        current_float.sum(-1, keepdim=True).clamp_min(1.0)
    )
    return (survivor_logits.float() - mean) * current_float


def continuous_policy_score(
    base_score: Tensor,
    factual_residual: Tensor,
    survivor_logits: Tensor,
    is_current: Tensor,
    *,
    residual_scale: float,
    survivor_scale: float,
    current_bias: float,
) -> Tensor:
    """Combine all signals continuously; no candidate count is pinned."""
    return (
        base_score.float()
        + float(residual_scale)
        * factual_residual.float()
        * (~is_current.bool()).float()
        + float(survivor_scale) * centered_survivor_score(
            survivor_logits, is_current
        )
        + float(current_bias) * is_current.float()
    )


def _masked_bce(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    selected_logits = logits[mask]
    selected_target = target.float()[mask]
    if selected_logits.numel() == 0:
        return logits.sum() * 0.0
    positive = selected_target.sum().clamp_min(1.0)
    negative = (1.0 - selected_target).sum().clamp_min(1.0)
    positive_weight = (negative / positive).detach().clamp(1.0, 8.0)
    return F.binary_cross_entropy_with_logits(
        selected_logits, selected_target, pos_weight=positive_weight
    )


def _pairwise_softplus(
    logits: Tensor, target: Tensor, candidate_mask: Tensor
) -> Tensor:
    positive = target.bool() & candidate_mask
    negative = ~target.bool() & candidate_mask
    pair_mask = positive.unsqueeze(-1) & negative.unsqueeze(-2)
    if not bool(pair_mask.any()):
        return logits.sum() * 0.0
    difference = logits.unsqueeze(-1) - logits.unsqueeze(-2)
    return F.softplus(-difference[pair_mask]).mean()


def ranking_loss(
    factual_residual: Tensor,
    survivor_logits: Tensor,
    target: Tensor,
    is_current: Tensor,
    base_score: Tensor,
    valid: Tensor | None = None,
    *,
    factual_bce_weight: float = 0.35,
    listwise_weight: float = 1.0,
    pairwise_weight: float = 1.0,
    survivor_bce_weight: float = 0.75,
    survivor_pairwise_weight: float = 0.75,
    residual_anchor_weight: float = 0.002,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Pairwise/listwise top-k objective with explicit survivor ranking."""
    if valid is None:
        valid = torch.ones(target.shape[:-1], dtype=torch.bool, device=target.device)
    if valid.shape != target.shape[:-1]:
        raise ValueError("valid must match all target axes except candidate width")
    candidate_valid = valid.unsqueeze(-1).expand_as(target)
    novel_valid = ~is_current.bool() & candidate_valid
    combined = (
        base_score.float()
        + factual_residual.float() * (~is_current.bool()).float()
    )
    target_float = target.float()
    factual_bce = _masked_bce(combined, target, novel_valid)
    novel_target = target.bool() & novel_valid
    novel_target_float = novel_target.float()
    target_distribution = novel_target_float / novel_target_float.sum(
        -1, keepdim=True
    ).clamp_min(1.0)
    listwise_logits = combined.masked_fill(~novel_valid, -1.0e4)
    listwise_rows = -(
        target_distribution * F.log_softmax(listwise_logits, dim=-1)
    ).sum(-1)
    listwise_valid = valid & novel_target.any(-1)
    listwise = (
        listwise_rows[listwise_valid].mean()
        if bool(listwise_valid.any())
        else combined.sum() * 0.0
    )
    novel_negative = ~target.bool() & novel_valid
    negative_order = base_score.float().masked_fill(
        ~novel_negative, -1.0e4
    ).argsort(dim=-1, descending=True, stable=True)
    hard_negative = torch.zeros_like(novel_negative)
    hard_negative.scatter_(
        -1,
        negative_order[..., : min(TOP_K, target.shape[-1])],
        True,
    )
    hard_negative &= novel_negative
    factual_pairwise = _pairwise_softplus(
        combined, target, novel_target | hard_negative
    )

    current_valid = is_current.bool() & candidate_valid
    survivor_bce = _masked_bce(survivor_logits, target, current_valid)
    survivor_pairwise = _pairwise_softplus(
        survivor_logits, target, current_valid
    )
    residual_anchor = (
        factual_residual[novel_valid].square().mean()
        if bool(novel_valid.any())
        else factual_residual.sum() * 0.0
    )
    pieces = {
        "factual_bce": factual_bce,
        "listwise": listwise,
        "factual_pairwise": factual_pairwise,
        "survivor_bce": survivor_bce,
        "survivor_pairwise": survivor_pairwise,
        "residual_anchor": residual_anchor,
    }
    loss = (
        factual_bce_weight * factual_bce
        + listwise_weight * listwise
        + pairwise_weight * factual_pairwise
        + survivor_bce_weight * survivor_bce
        + survivor_pairwise_weight * survivor_pairwise
        + residual_anchor_weight * residual_anchor
    )
    return loss, pieces


POLICY_KEYS = ("residual_scale", "survivor_scale", "current_bias")


def _policy_tuple(row: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(float(row[key]) for key in POLICY_KEYS)


def _complexity(row: Mapping[str, Any]) -> float:
    return sum(abs(value) for value in _policy_tuple(row))


def pareto_calibration_options(
    options: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate and prune a deterministic factual/CacheSet frontier."""
    deduplicated: dict[tuple[float, float], dict[str, Any]] = {}
    for raw in options:
        row = dict(raw)
        key = (float(row["cache_set"]), float(row["factual"]))
        order = (_complexity(row), _policy_tuple(row))
        previous = deduplicated.get(key)
        if previous is None or order < (
            _complexity(previous),
            _policy_tuple(previous),
        ):
            deduplicated[key] = row

    rows = list(deduplicated.values())
    kept: list[dict[str, Any]] = []
    for row in rows:
        row_cache = float(row["cache_set"])
        row_factual = float(row["factual"])
        row_complexity = _complexity(row)
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_cache = float(other["cache_set"])
            other_factual = float(other["factual"])
            other_complexity = _complexity(other)
            if (
                other_cache >= row_cache
                and other_factual >= row_factual
                and other_complexity <= row_complexity
                and (
                    other_cache > row_cache
                    or other_factual > row_factual
                    or other_complexity < row_complexity
                )
            ):
                dominated = True
                break
        if not dominated:
            kept.append(row)
    return sorted(
        kept,
        key=lambda row: (
            float(row["cache_set"]),
            float(row["factual"]),
            _complexity(row),
            _policy_tuple(row),
        ),
    )


def select_mean_constrained_policies(
    options_by_horizon: Sequence[Sequence[Mapping[str, Any]]],
    cache_target: float = 0.95,
) -> dict[str, Any]:
    """Select four policies under a mean, rather than per-H, CacheSet bound."""
    if len(options_by_horizon) != HORIZONS:
        raise ValueError("exactly four horizon option sets are required")
    fronts = [pareto_calibration_options(options) for options in options_by_horizon]
    if any(not frontier for frontier in fronts):
        raise ValueError("every horizon must have at least one calibration option")

    def pairs(
        first: Sequence[Mapping[str, Any]],
        second: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for left in first:
            for right in second:
                output.append(
                    {
                        "cache_sum": float(left["cache_set"])
                        + float(right["cache_set"]),
                        "factual_sum": float(left["factual"])
                        + float(right["factual"]),
                        "complexity_sum": _complexity(left) + _complexity(right),
                        "options": (left, right),
                        "policy_tuple": _policy_tuple(left) + _policy_tuple(right),
                    }
                )
        return output

    left_pairs = pairs(fronts[0], fronts[1])
    right_pairs = sorted(
        pairs(fronts[2], fronts[3]),
        key=lambda row: (
            row["cache_sum"],
            row["factual_sum"],
            row["complexity_sum"],
            row["policy_tuple"],
        ),
    )

    def quality(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -float(row["factual_sum"]),
            -float(row["cache_sum"]),
            float(row["complexity_sum"]),
            tuple(row["policy_tuple"]),
        )

    suffix_best: list[Mapping[str, Any] | None] = [None] * len(right_pairs)
    best: Mapping[str, Any] | None = None
    for index in range(len(right_pairs) - 1, -1, -1):
        if best is None or quality(right_pairs[index]) < quality(best):
            best = right_pairs[index]
        suffix_best[index] = best

    right_cache = [float(row["cache_sum"]) for row in right_pairs]
    required = HORIZONS * float(cache_target)
    winner: tuple[tuple[Mapping[str, Any], ...], float, float] | None = None
    winner_key: tuple[Any, ...] | None = None
    for left in left_pairs:
        index = bisect_left(
            right_cache,
            required - float(left["cache_sum"]) - 1.0e-12,
        )
        if index == len(right_pairs):
            continue
        right = suffix_best[index]
        if right is None:
            continue
        selected = tuple(left["options"]) + tuple(right["options"])
        cache_sum = float(left["cache_sum"]) + float(right["cache_sum"])
        factual_sum = float(left["factual_sum"]) + float(right["factual_sum"])
        complexity = float(left["complexity_sum"]) + float(
            right["complexity_sum"]
        )
        policy_tuple = tuple(left["policy_tuple"]) + tuple(right["policy_tuple"])
        key = (-factual_sum, -cache_sum, complexity, policy_tuple)
        if winner_key is None or key < winner_key:
            winner = (selected, cache_sum, factual_sum)
            winner_key = key
    if winner is None:
        raise AssertionError("no mean-CacheSet-feasible joint policy")

    selected, cache_sum, factual_sum = winner
    return {
        "policies_by_horizon": [
            {key: float(option[key]) for key in POLICY_KEYS}
            for option in selected
        ],
        "per_horizon_metrics": [
            {
                "factual": float(option["factual"]),
                "cache_set": float(option["cache_set"]),
            }
            for option in selected
        ],
        "factual_mean": factual_sum / HORIZONS,
        "cache_set_mean": cache_sum / HORIZONS,
        "frontier_sizes": [len(frontier) for frontier in fronts],
        "left_pair_count": len(left_pairs),
        "right_pair_count": len(right_pairs),
    }


_CONTRACT = parameter_and_mac_contract()
if _CONTRACT["linear_macs_per_cell"] > FROZEN_C16_LOWER_BOUND_MACS:
    raise AssertionError("C32 set ranker exceeds the frozen C16 MAC budget")
