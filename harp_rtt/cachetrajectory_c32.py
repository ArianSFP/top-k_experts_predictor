"""Leak-safe C32 candidate features and a factorized dual-objective scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LAYERS = 40
EXPERTS = 256
HORIZONS = 4
TOP_K = 8
FEATURE_WIDTH = 28


def build_candidate_ids(scores: Tensor, current_ids: Tensor, width: int = 32) -> Tensor:
    """Return current top eight followed by dense-score-ranked novel experts."""
    if scores.ndim != 4 or scores.shape[-2:] != (LAYERS, EXPERTS):
        raise ValueError("scores must have [N,4,40,256] geometry")
    if current_ids.shape != (scores.shape[0], LAYERS, TOP_K):
        raise ValueError("current_ids must have [N,40,8] geometry")
    if not TOP_K <= width <= EXPERTS:
        raise ValueError("candidate width must lie in [8,256]")
    order = scores.float().argsort(dim=-1, descending=True, stable=True)
    current = current_ids.long()[:, None].expand(-1, scores.shape[1], -1, -1)
    ranked_is_current = (order.unsqueeze(-1) == current.unsqueeze(-2)).any(-1)
    novel = order[~ranked_is_current].reshape(*order.shape[:-1], EXPERTS - TOP_K)
    candidates = torch.cat((current, novel[..., : width - TOP_K]), dim=-1)
    if not bool((candidates.sort(dim=-1).values[..., 1:] != candidates.sort(dim=-1).values[..., :-1]).all()):
        raise ValueError("candidate construction produced duplicate experts")
    return candidates


def build_features(
    scores: Tensor,
    candidates: Tensor,
    current_ids: Tensor,
    history_ids: Tensor,
    history_weights: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build features using only deployed scores and causal route history.

    Features are raw and normalized deployed score, normalized dense rank,
    current membership, then membership/rank/weight for eight causal lags.
    No future target value is consumed here.
    """
    n, horizons, layers, width = candidates.shape
    if (n, horizons, layers) != scores.shape[:3]:
        raise ValueError("candidate and score geometry differs")
    if history_ids.shape != (n, 8, layers, TOP_K):
        raise ValueError("history_ids must have [N,8,40,8] geometry")
    if history_weights.shape != history_ids.shape:
        raise ValueError("history weights and IDs differ")

    candidates_long = candidates.long()
    gathered = scores.float().gather(-1, candidates_long)
    mean = scores.float().mean(-1, keepdim=True)
    std = scores.float().std(-1, keepdim=True, unbiased=False).clamp_min(1e-5)
    normalized = (gathered - mean) / std
    order = scores.float().argsort(dim=-1, descending=True, stable=True)
    inverse_rank = torch.empty_like(order)
    ranks = torch.arange(EXPERTS, device=order.device).view(1, 1, 1, -1)
    inverse_rank.scatter_(-1, order, ranks.expand_as(order))
    dense_rank = inverse_rank.gather(-1, candidates_long)
    is_current = (
        candidates_long.unsqueeze(-1)
        == current_ids.long()[:, None, :, None, :]
    ).any(-1)
    columns = [
        gathered,
        normalized,
        dense_rank.float() / float(EXPERTS - 1),
        is_current.float(),
    ]
    for lag in range(8):
        ids = history_ids[:, lag].long()[:, None, :, None, :]
        weights = history_weights[:, lag].float()[:, None, :, None, :]
        matches = candidates_long.unsqueeze(-1) == ids
        present = matches.any(-1)
        rank = matches.float().argmax(-1).float() / float(TOP_K - 1)
        rank = torch.where(present, rank, torch.full_like(rank, 1.25))
        weight = (matches.float() * weights).sum(-1)
        columns.extend((present.float(), rank, weight))
    features = torch.stack(columns, dim=-1)
    if features.shape != (n, horizons, layers, width, FEATURE_WIDTH):
        raise AssertionError("C32 feature width drift")
    return features, dense_rank, is_current


def target_membership(candidates: Tensor, target_ids: Tensor) -> Tensor:
    if target_ids.shape != (*candidates.shape[:3], TOP_K):
        raise ValueError("target IDs do not align with candidates")
    return (candidates.long().unsqueeze(-1) == target_ids.long().unsqueeze(-2)).any(-1)


def split_masks(request_index: Tensor) -> dict[str, Tensor]:
    request_index = request_index.long()
    even = request_index.remainder(2).eq(0)
    calibration = even & request_index.remainder(8).eq(2)
    training = even & ~calibration
    blind = ~even
    return {
        "training": training,
        "calibration": calibration,
        "design": even,
        "blind": blind,
        "full": torch.ones_like(even),
    }


class FactorizedCacheTrajectory32(nn.Module):
    """Shared per-candidate scorer with separate factual and survivor heads."""

    def __init__(self, feature_width: int = FEATURE_WIDTH) -> None:
        super().__init__()
        self.object_embedding = nn.Embedding(LAYERS * EXPERTS, 8)
        self.horizon_embedding = nn.Embedding(HORIZONS, 4)
        self.layer_embedding = nn.Embedding(LAYERS, 4)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(feature_width), nn.Linear(feature_width, 32), nn.GELU()
        )
        self.trunk = nn.Sequential(nn.Linear(48, 32), nn.GELU())
        self.factual_head = nn.Linear(32, 1)
        self.survivor_head = nn.Linear(32, 1)

    def forward(
        self,
        features: Tensor,
        candidate_ids: Tensor,
        horizons: Tensor,
        layers: Tensor,
    ) -> tuple[Tensor, Tensor]:
        width = candidate_ids.shape[-1]
        feature_state = self.feature_projection(features.float())
        objects = self.object_embedding(layers[:, None] * EXPERTS + candidate_ids.long())
        horizon = self.horizon_embedding(horizons)[:, None].expand(-1, width, -1)
        layer = self.layer_embedding(layers)[:, None].expand(-1, width, -1)
        state = self.trunk(torch.cat((feature_state, objects, horizon, layer), dim=-1))
        return self.factual_head(state).squeeze(-1), self.survivor_head(state).squeeze(-1)


def parameter_and_mac_contract(width: int = 32) -> dict[str, int | float]:
    model = FactorizedCacheTrajectory32()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    # Linear-layer multiply-accumulates per token/layer/horizon cell.
    factorized_macs = width * (FEATURE_WIDTH * 32 + 48 * 32 + 32 * 2)
    frozen_c16_macs = (
        16 * 43 * 48
        + 16 * 64 * 192
        + 16 * 64 * 64
        + 2 * 16 * 16 * 64
        + 2 * 16 * 64 * 128
        + 16 * (64 * 32 + 32)
    )
    return {
        "parameters": parameters,
        "deployment_bf16_bytes": parameters * 2,
        "linear_macs_per_cell": factorized_macs,
        "frozen_c16_lower_bound_macs_per_cell": frozen_c16_macs,
        "mac_ratio": factorized_macs / frozen_c16_macs,
    }


def dual_objective_loss(
    factual_logits: Tensor,
    survivor_logits: Tensor,
    target: Tensor,
    is_current: Tensor,
) -> Tensor:
    target_float = target.float()
    factual_bce = F.binary_cross_entropy_with_logits(
        factual_logits, target_float, pos_weight=torch.tensor(3.0, device=factual_logits.device)
    )
    survivor_target = target_float[is_current]
    survivor_bce = F.binary_cross_entropy_with_logits(
        survivor_logits[is_current],
        survivor_target,
        pos_weight=torch.tensor(3.0, device=factual_logits.device),
    )
    distribution = target_float / target_float.sum(-1, keepdim=True).clamp_min(1.0)
    listwise = -(distribution * F.log_softmax(factual_logits, dim=-1)).sum(-1).mean()
    return factual_bce + survivor_bce + 0.5 * listwise


def request_macro_metrics(
    selected_ids: Tensor,
    current_ids: Tensor,
    target_ids: Tensor,
    valid: Tensor,
    request_index: Tensor,
    row_mask: Tensor,
) -> dict[str, Any]:
    selected = selected_ids[row_mask]
    current = current_ids[row_mask]
    target = target_ids[row_mask]
    active = valid[row_mask]
    requests = request_index[row_mask]
    factual = (selected.unsqueeze(-1) == target.unsqueeze(-2)).any(-1).sum(-1).float() / TOP_K
    target_in_current = (target.unsqueeze(-1) == current[:, None, :, None, :]).any(-1)
    target_in_prediction = (target.unsqueeze(-1) == selected.unsqueeze(-2)).any(-1)
    factual_h: list[float] = []
    cache_h: list[float] = []
    for horizon in range(HORIZONS):
        factual_rows: list[Tensor] = []
        cache_rows: list[Tensor] = []
        for request in requests.unique(sorted=True):
            request_rows = requests.eq(request)[:, None] & active[:, horizon]
            factual_rows.append(factual[:, horizon][request_rows].mean())
            numerator = (
                target_in_current[:, horizon]
                & target_in_prediction[:, horizon]
                & request_rows.unsqueeze(-1)
            ).sum()
            denominator = (
                target_in_current[:, horizon] & request_rows.unsqueeze(-1)
            ).sum()
            cache_rows.append(numerator.float() / denominator.clamp_min(1))
        factual_h.append(float(torch.stack(factual_rows).mean()))
        cache_h.append(float(torch.stack(cache_rows).mean()))
    return {
        "factual_h": factual_h,
        "cache_set_h": cache_h,
        "factual_mean": sum(factual_h) / HORIZONS,
        "cache_set_mean": sum(cache_h) / HORIZONS,
    }
