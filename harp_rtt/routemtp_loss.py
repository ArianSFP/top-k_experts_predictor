"""Normalized objectives for RouteMTP v1."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .exact_k import exact_set_nll
from .factual_mixture import exact_factual_mixture_nll
from .losses import boundary_loss_per_endpoint
from .routemtp import RouteMTPPathOutput, RouteMTPRouteOutput


@dataclass(frozen=True)
class RouteMTPLossWeights:
    branch_set: float = 1.0
    router_kl: float = 0.5
    factual: float = 1.0
    boundary: float = 0.2
    query: float = 0.03
    path: float = 0.5
    native_trust: float = 0.02

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"loss weight {name} must be finite and non-negative")
        if self.branch_set <= 0 or self.factual <= 0:
            raise ValueError("branch-set and factual weights must be positive")


@dataclass(frozen=True)
class RouteMTPLossOutput:
    total: Tensor
    branch_set: Tensor
    router_kl: Tensor
    factual: Tensor
    boundary: Tensor
    query: Tensor
    path: Tensor
    native_trust: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "total": self.total,
            "branch_set": self.branch_set,
            "router_kl": self.router_kl,
            "factual": self.factual,
            "boundary": self.boundary,
            "query": self.query,
            "path": self.path,
            "native_trust": self.native_trust,
        }


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    if values.shape != weights.shape:
        raise ValueError("weighted reduction geometry differs")
    denominator = weights.float().sum()
    if bool(denominator <= 0):
        return values.sum() * 0.0
    return (values.float() * weights.float()).sum() / denominator


def _branch_weights(
    valid: Tensor,
    node_depths: Tensor,
    native_path_log_probabilities: Tensor,
    *,
    floor: float,
    horizon_weights: tuple[float, float, float, float],
) -> Tensor:
    if valid.ndim != 3:
        raise ValueError("branch validity must be [B,N,L]")
    if node_depths.shape != valid.shape[:2] or native_path_log_probabilities.shape != valid.shape[:2]:
        raise ValueError("branch weighting topology differs")
    if not 0 <= floor <= 1:
        raise ValueError("branch weight floor must lie in [0,1]")
    probability = native_path_log_probabilities.float().clamp(max=0).exp().sqrt()
    probability[:, 0] = 1.0
    probability = probability.clamp_min(float(floor))
    horizon = torch.tensor(horizon_weights, device=valid.device, dtype=torch.float32)
    depth_weight = horizon[(node_depths.clamp(1, 4) - 1).long()]
    weights = probability * depth_weight * valid.any(-1).float()
    # Equalize each request/depth cell before expanding over layers.  This
    # prevents a large H2 frontier from suppressing rare H4 divergence rows.
    normalized = torch.zeros_like(weights)
    for depth in range(1, 5):
        mask = (node_depths == depth) & valid.any(-1)
        denominator = (weights * mask.float()).sum(-1, keepdim=True).clamp_min(1e-12)
        normalized = normalized + torch.where(mask, weights / denominator, torch.zeros_like(weights))
    return normalized[..., None] * valid.float()


class RouteMTPObjective(nn.Module):
    """Direct route, exact factual mixture, and causal path objectives."""

    def __init__(
        self,
        expert_keys: Tensor,
        centered_bias: Tensor,
        *,
        exact_k: int = 8,
        weights: RouteMTPLossWeights = RouteMTPLossWeights(),
        horizon_weights: tuple[float, float, float, float] = (0.5, 1.0, 1.25, 1.5),
        branch_weight_floor: float = 0.05,
        router_temperature: float = 1.0,
        boundary_margin: float = 0.125,
    ) -> None:
        super().__init__(); weights.validate()
        if expert_keys.ndim != 3 or centered_bias.shape != expert_keys.shape[:2]:
            raise ValueError("RouteMTP objective router geometry is invalid")
        if len(horizon_weights) != 4 or any(float(v) <= 0 for v in horizon_weights):
            raise ValueError("RouteMTP requires four positive horizon weights")
        if router_temperature <= 0:
            raise ValueError("router temperature must be positive")
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.register_buffer("centered_bias", centered_bias.detach().float().clone())
        self.exact_k = int(exact_k)
        self.weights = weights
        self.horizon_weights = tuple(float(v) for v in horizon_weights)
        self.branch_weight_floor = float(branch_weight_floor)
        self.router_temperature = float(router_temperature)
        self.boundary_margin = float(boundary_margin)

    def forward(
        self,
        *,
        route: RouteMTPRouteOutput,
        path: RouteMTPPathOutput,
        node_depths: Tensor,
        visible_mask: Tensor,
        native_path_log_probabilities: Tensor,
        teacher_node_logits: Tensor,
        teacher_node_ids: Tensor,
        teacher_node_queries: Tensor,
        branch_valid: Tensor,
        anchor_scores: Tensor,
        factual_ids: Tensor,
        factual_valid: Tensor,
        factual_branch_indices: Tensor,
        branch_supervision_mask: Tensor | None = None,
        native_trust_logits: Tensor | None = None,
        native_trust_targets: Tensor | None = None,
    ) -> RouteMTPLossOutput:
        scores = route.scores
        batch, nodes, layers, experts = scores.shape
        if teacher_node_logits.shape != scores.shape:
            raise ValueError("teacher node logits disagree with RouteMTP scores")
        if teacher_node_ids.shape != (batch, nodes, layers, self.exact_k):
            raise ValueError("teacher node selected sets have invalid geometry")
        if teacher_node_queries.shape != route.queries.shape:
            raise ValueError("teacher node queries disagree with RouteMTP queries")
        if branch_valid.shape != (batch, nodes, layers):
            raise ValueError("branch validity has invalid geometry")
        horizons = path.probabilities.shape[1]
        if anchor_scores.shape != (batch, horizons, layers, experts):
            raise ValueError("anchor score geometry is invalid")
        if factual_ids.shape != (batch, horizons, layers, self.exact_k):
            raise ValueError("factual selected sets have invalid geometry")
        if factual_valid.shape != (batch, horizons, layers):
            raise ValueError("factual validity has invalid geometry")
        if factual_branch_indices.shape != (batch, horizons):
            raise ValueError("factual branch indices must be [B,H]")

        if branch_supervision_mask is None:
            branch_supervision_mask = visible_mask
        if branch_supervision_mask.shape != (batch, nodes):
            raise ValueError("branch supervision mask must be [B,N]")
        cell_weights = _branch_weights(
            branch_valid & branch_supervision_mask[..., None],
            node_depths,
            native_path_log_probabilities,
            floor=self.branch_weight_floor,
            horizon_weights=self.horizon_weights,
        )
        branch_set = exact_set_nll(
            scores, teacher_node_ids, valid=cell_weights, k=self.exact_k
        )

        temperature = self.router_temperature
        teacher_probability = torch.softmax(teacher_node_logits.float() / temperature, dim=-1)
        per_kl = F.kl_div(
            torch.log_softmax(scores.float() / temperature, dim=-1),
            teacher_probability,
            reduction="none",
        ).sum(-1) * (temperature * temperature)
        router_kl = _weighted_mean(per_kl, cell_weights)

        per_boundary = boundary_loss_per_endpoint(
            scores,
            teacher_node_logits,
            teacher_node_ids,
            margin=self.boundary_margin,
            model_rank_start=9,
            teacher_rank_start=9,
            rank_end=min(32, experts),
        )
        boundary = _weighted_mean(per_boundary, cell_weights)

        with torch.autocast(device_type=scores.device.type, enabled=False):
            predicted_geometry = torch.einsum(
                "bnlr,ler->bnle", route.queries.float(), self.expert_keys.float()
            ) + self.centered_bias[None, None].float()
            teacher_centered = torch.einsum(
                "bnlr,ler->bnle",
                teacher_node_queries.float(),
                self.expert_keys.float(),
            ) + self.centered_bias[None, None].float()
            per_query = F.smooth_l1_loss(
                predicted_geometry, teacher_centered, reduction="none"
            ).mean(-1)
        query = _weighted_mean(per_query, cell_weights)

        horizon_mask = (
            (node_depths[:, None, :] == torch.arange(
                1, horizons + 1, device=node_depths.device
            )[None, :, None])
            & visible_mask[:, None].bool()
        )
        branch_scores = scores[:, None].expand(
            batch, horizons, nodes, layers, experts
        ).permute(0, 1, 3, 2, 4)
        factual_result = exact_factual_mixture_nll(
            branch_scores,
            anchor_scores,
            path.probabilities,
            horizon_mask,
            factual_ids,
            valid=factual_valid,
            k=self.exact_k,
        )
        factual = factual_result.loss

        safe_factual = factual_branch_indices.clamp(0, nodes)
        selected_probability = path.probabilities.gather(
            -1, safe_factual[..., None]
        ).squeeze(-1)
        path_valid = (factual_branch_indices >= 0) & (factual_branch_indices <= nodes)
        horizon_weight = torch.tensor(
            self.horizon_weights, device=scores.device, dtype=torch.float32
        )[None].expand(batch, horizons)
        path_loss = -selected_probability.clamp_min(1e-12).log()
        path_component = _weighted_mean(path_loss, horizon_weight * path_valid.float())

        if (native_trust_logits is None) != (native_trust_targets is None):
            raise ValueError("native trust logits and targets must be provided together")
        if native_trust_logits is None:
            native_trust = scores.sum() * 0.0
        else:
            if native_trust_logits.shape != native_trust_targets.shape:
                raise ValueError("native trust tensors disagree")
            native_trust = F.kl_div(
                torch.log_softmax(native_trust_logits.float(), dim=-1),
                torch.softmax(native_trust_targets.float(), dim=-1),
                reduction="batchmean",
            )

        values = {
            "branch_set": branch_set,
            "router_kl": router_kl,
            "factual": factual,
            "boundary": boundary,
            "query": query,
            "path": path_component,
            "native_trust": native_trust,
        }
        total = sum(getattr(self.weights, name) * value for name, value in values.items())
        return RouteMTPLossOutput(total=total, path=path_component, **{
            key: value for key, value in values.items() if key != "path"
        })


__all__ = [
    "RouteMTPLossOutput",
    "RouteMTPLossWeights",
    "RouteMTPObjective",
]
