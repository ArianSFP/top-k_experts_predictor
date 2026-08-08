"""Formal HARP-RTT objective and objective-gradient promotion audit.

The objective is intentionally strict about the production geometry: four
active horizons, exactly two trajectory rounds, and exact-cardinality target
sets.  It accepts either the rich dataset's complete ``{"inputs", "targets"}``
batch or its ``targets`` mapping directly.  Acceptance labels are read only by
the branch auxiliary; they are never used to construct another target or a
model-side feature.

All numerically sensitive reductions (exact-k likelihood, soft top-k,
softmax/KL, and gradient norms) run in FP32 even under autocast.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .exact_k import (
    exact_set_nll,
    soft_cardinality_topk,
    validate_exact_set_labels,
)
from .schema import HARPRTTDimensions


COMPONENT_NAMES = (
    "set",
    "recall",
    "boundary",
    "router_kl",
    "query",
    "branch",
    "trajectory",
)


def _optional(mapping: Mapping[str, Any], *names: str) -> Any | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _required(mapping: Mapping[str, Any], *names: str) -> Any:
    value = _optional(mapping, *names)
    if value is None:
        raise KeyError(f"none of the required fields are present: {names}")
    return value


def _finite_tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite and floating-point")
    return value


@dataclass(frozen=True)
class HARPRTTLossWeights:
    """Post-warm-up coefficients from the formal architecture."""

    set: float = 1.0
    recall: float = 0.50
    boundary: float = 0.25
    router_kl: float = 0.15
    query: float = 0.10
    branch: float = 0.10
    trajectory: float = 0.05

    def validate(self) -> None:
        for name in COMPONENT_NAMES:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"loss weight {name!r} must be finite and non-negative")
        if self.set <= 0:
            raise ValueError("the primary exact-set weight must be positive")

    def as_dict(self) -> dict[str, float]:
        self.validate()
        return {name: float(getattr(self, name)) for name in COMPONENT_NAMES}


@dataclass(frozen=True)
class ScheduledLossState:
    step: int
    temperature: float
    recall_progress: float
    weights: Mapping[str, float]


@dataclass(frozen=True)
class HARPRTTLossSchedule:
    """Warm up soft recall while annealing its sigmoid temperature 1 -> .1.

    Exact-set likelihood and the other auxiliaries retain their formal
    coefficients.  The recall coefficient starts at zero and linearly reaches
    its configured value over ``recall_ramp_steps``.  Temperature annealing
    begins at ``recall_start_step`` and is linear in log-temperature, avoiding
    an abrupt change near the low-temperature endpoint.
    """

    recall_start_step: int = 0
    recall_ramp_steps: int = 1_000
    temperature_anneal_steps: int = 10_000
    temperature_start: float = 1.0
    temperature_end: float = 0.1

    def validate(self) -> None:
        if self.recall_start_step < 0:
            raise ValueError("recall_start_step must be non-negative")
        if self.recall_ramp_steps < 0:
            raise ValueError("recall_ramp_steps must be non-negative")
        if self.temperature_anneal_steps < 0:
            raise ValueError("temperature_anneal_steps must be non-negative")
        for name in ("temperature_start", "temperature_end"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.temperature_end > self.temperature_start:
            raise ValueError("temperature schedule must anneal rather than increase")

    @staticmethod
    def _progress(step: int, start: int, duration: int) -> float:
        if step < start:
            return 0.0
        if duration == 0:
            return 1.0
        return min(max((step - start) / float(duration), 0.0), 1.0)

    def at(
        self,
        step: int,
        weights: HARPRTTLossWeights = HARPRTTLossWeights(),
    ) -> ScheduledLossState:
        self.validate()
        weights.validate()
        if not isinstance(step, int) or step < 0:
            raise ValueError("training step must be a non-negative integer")
        recall_progress = self._progress(
            step, self.recall_start_step, self.recall_ramp_steps
        )
        temperature_progress = self._progress(
            step, self.recall_start_step, self.temperature_anneal_steps
        )
        log_start = math.log(float(self.temperature_start))
        log_end = math.log(float(self.temperature_end))
        temperature = math.exp(
            log_start + temperature_progress * (log_end - log_start)
        )
        scheduled = weights.as_dict()
        scheduled["recall"] *= recall_progress
        return ScheduledLossState(
            step=step,
            temperature=temperature,
            recall_progress=recall_progress,
            weights=scheduled,
        )


@dataclass(frozen=True)
class HARPRTTLossConfig:
    dimensions: HARPRTTDimensions = field(default_factory=HARPRTTDimensions)
    weights: HARPRTTLossWeights = field(default_factory=HARPRTTLossWeights)
    schedule: HARPRTTLossSchedule = field(default_factory=HARPRTTLossSchedule)
    horizon_worst_weight: float = 0.25
    boundary_margin: float = 1.0
    model_negative_rank_start: int = 6
    teacher_negative_rank_start: int = 9
    negative_rank_end: int = 64
    query_cosine_weight: float = 0.10
    query_huber_delta: float = 1.0
    trajectory_correction_weight: float = 0.01
    soft_topk_bisection_steps: int = 32

    def validate(self) -> None:
        self.dimensions.validate()
        self.weights.validate()
        self.schedule.validate()
        if self.dimensions.primary_horizons != 4:
            raise ValueError("the formal HARP-RTT objective requires H1--H4")
        if self.dimensions.trajectory_rounds != 2:
            raise ValueError("the formal HARP-RTT objective requires exactly two rounds")
        for name in (
            "horizon_worst_weight",
            "query_cosine_weight",
            "trajectory_correction_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(float(self.boundary_margin)):
            raise ValueError("boundary_margin must be finite")
        if not math.isfinite(float(self.query_huber_delta)) or self.query_huber_delta <= 0:
            raise ValueError("query_huber_delta must be finite and positive")
        if self.model_negative_rank_start < 1 or self.teacher_negative_rank_start < 1:
            raise ValueError("hard-negative ranks are one-indexed and must be positive")
        if self.negative_rank_end < max(
            self.model_negative_rank_start, self.teacher_negative_rank_start
        ):
            raise ValueError("negative_rank_end precedes a hard-negative start rank")
        if self.soft_topk_bisection_steps < 1:
            raise ValueError("soft_topk_bisection_steps must be positive")


def endpoint_validity(
    valid: Tensor | None,
    scores: Tensor,
) -> Tensor:
    """Return FP32 endpoint weights with shape ``[B,H,L]``."""

    shape = scores.shape[:3]
    if valid is None:
        return torch.ones(shape, dtype=torch.float32, device=scores.device)
    if not isinstance(valid, Tensor):
        raise TypeError("future validity must be a torch.Tensor")
    if valid.device != scores.device:
        raise ValueError("future validity and scores must share a device")
    if valid.shape == shape[:2]:
        valid = valid[..., None].expand(shape)
    elif valid.shape != shape:
        raise ValueError(
            f"future validity must have shape {shape[:2]} or {shape}, got {tuple(valid.shape)}"
        )
    weights = valid.float()
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("future validity must be finite and non-negative")
    return weights


def horizon_balanced_reduce(
    per_endpoint: Tensor,
    valid: Tensor,
    *,
    worst_weight: float = 0.25,
) -> tuple[Tensor, Tensor]:
    """Reduce ``[B,H,...]`` values as equal horizons plus worst horizon.

    Missing endpoints are excluded.  A horizon with no valid endpoint is
    excluded from both the equal-horizon mean and the maximum.
    """

    if per_endpoint.shape != valid.shape or per_endpoint.ndim < 2:
        raise ValueError("per_endpoint and valid must have the same [B,H,...] shape")
    if not math.isfinite(float(worst_weight)) or worst_weight < 0:
        raise ValueError("worst_weight must be finite and non-negative")
    values = per_endpoint.float().movedim(1, 0).flatten(1)
    weights = valid.float().movedim(1, 0).flatten(1)
    denominator = weights.sum(-1)
    horizon_values = (values * weights).sum(-1) / denominator.clamp_min(1.0)
    horizon_valid = denominator > 0
    if not bool(horizon_valid.any()):
        return per_endpoint.sum() * 0.0, horizon_values
    mean = horizon_values[horizon_valid].mean()
    worst = horizon_values.masked_fill(~horizon_valid, -torch.inf).max()
    return mean + float(worst_weight) * worst, horizon_values


def _balance_horizon_values(
    values: Tensor,
    valid: Tensor,
    worst_weight: float,
) -> Tensor:
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("horizon values and validity must be matching vectors")
    if not bool(valid.any()):
        return values.sum() * 0.0
    selected = values[valid]
    return selected.mean() + float(worst_weight) * selected.max()


def hard_negative_mask(
    scores: Tensor,
    teacher_scores: Tensor,
    true_ids: Tensor,
    *,
    model_rank_start: int = 6,
    teacher_rank_start: int = 9,
    rank_end: int = 64,
) -> Tensor:
    """Return the deduplicated false-expert union for the boundary loss.

    Ranks are one-indexed and inclusive: model ranks 6--64 and teacher ranks
    9--64 under the formal defaults.  Every true expert is removed after the
    union, including a true expert that appears in either rank interval.
    """

    _finite_tensor(scores, "scores")
    _finite_tensor(teacher_scores, "teacher_scores")
    if scores.shape != teacher_scores.shape:
        raise ValueError("scores and teacher_scores must have identical shapes")
    if true_ids.shape[:-1] != scores.shape[:-1]:
        raise ValueError("true_ids leading dimensions disagree with scores")
    experts = int(scores.shape[-1])
    if true_ids.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("true_ids must use an integer dtype")
    if true_ids.device != scores.device:
        raise ValueError("true_ids and scores must share a device")
    if ((true_ids < 0) | (true_ids >= experts)).any():
        raise ValueError("true_ids contain an out-of-range expert ID")
    if model_rank_start < 1 or teacher_rank_start < 1 or rank_end < 1:
        raise ValueError("hard-negative ranks must be positive")

    candidate = torch.zeros_like(scores, dtype=torch.bool)
    model_order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    teacher_order = torch.argsort(
        teacher_scores, dim=-1, descending=True, stable=True
    )

    def add_ranks(order: Tensor, start: int) -> None:
        lower = min(start - 1, experts)
        upper = min(rank_end, experts)
        if upper > lower:
            candidate.scatter_(-1, order[..., lower:upper], True)

    add_ranks(model_order, model_rank_start)
    add_ranks(teacher_order, teacher_rank_start)
    true_mask = torch.zeros_like(candidate)
    true_mask.scatter_(-1, true_ids.long(), True)
    candidate &= ~true_mask

    # The production E=256 geometry always yields candidates.  Keep small
    # synthetic geometries well-defined by falling back to all false experts.
    missing = ~candidate.any(dim=-1)
    if missing.any():
        false_experts = ~true_mask
        candidate = torch.where(missing[..., None], false_experts, candidate)
    return candidate


def boundary_loss_per_endpoint(
    scores: Tensor,
    teacher_scores: Tensor,
    true_ids: Tensor,
    *,
    margin: float = 1.0,
    model_rank_start: int = 6,
    teacher_rank_start: int = 9,
    rank_end: int = 64,
) -> Tensor:
    """Compute formal soft-min/LSE boundary loss without reducing endpoints."""

    candidates = hard_negative_mask(
        scores,
        teacher_scores,
        true_ids,
        model_rank_start=model_rank_start,
        teacher_rank_start=teacher_rank_start,
        rank_end=rank_end,
    )
    values = scores.float()
    strongest_false = torch.logsumexp(
        values.masked_fill(~candidates, -torch.inf), dim=-1
    )
    true_scores = values.gather(-1, true_ids.long())
    soft_minimum = -torch.logsumexp(-true_scores, dim=-1)
    return F.softplus(float(margin) + strongest_false - soft_minimum)


@dataclass
class HARPRTTLossOutput:
    total: Tensor
    primary: Tensor
    auxiliary: Tensor
    components: Mapping[str, Tensor]
    weighted_components: Mapping[str, Tensor]
    horizon_components: Mapping[str, Tensor]
    temperature: float
    recall_progress: float
    scheduled_weights: Mapping[str, float]

    def as_dict(self) -> dict[str, Tensor]:
        """Return scalar tensors under stable trainer/logging names."""

        result = {
            "total": self.total,
            "primary": self.primary,
            "auxiliary": self.auxiliary,
        }
        result.update(self.components)
        return result


class HARPRTTObjective(nn.Module):
    """Combined exact-8, ranking, geometry, branch, and trajectory objective."""

    def __init__(
        self,
        config: HARPRTTLossConfig = HARPRTTLossConfig(),
        *,
        router_input_basis: Tensor | None = None,
        router_rank_mask: Tensor | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dimensions = config.dimensions
        if router_input_basis is not None:
            _finite_tensor(router_input_basis, "router_input_basis")
            if router_input_basis.ndim != 3 or router_input_basis.shape[0] != dimensions.layers:
                raise ValueError("router_input_basis must be [L,D,R]")
            basis = router_input_basis.detach().float().clone()
        else:
            basis = None
        if router_rank_mask is not None:
            if basis is None:
                raise ValueError("router_rank_mask requires router_input_basis")
            if router_rank_mask.shape != (dimensions.layers, basis.shape[-1]):
                raise ValueError("router_rank_mask must be [L,R]")
            rank_mask = router_rank_mask.detach().bool().clone()
        else:
            rank_mask = None
        self.register_buffer("router_input_basis", basis)
        self.register_buffer("router_rank_mask", rank_mask)

    def _split_batch(
        self, batch_or_targets: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
        nested = batch_or_targets.get("targets")
        if nested is None:
            return batch_or_targets, None
        if not isinstance(nested, Mapping):
            raise TypeError("batch targets must be a mapping")
        return nested, batch_or_targets

    def _validate_primary(
        self, outputs: Mapping[str, Any], targets: Mapping[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dimensions = self.config.dimensions
        scores = _finite_tensor(_required(outputs, "active_scores"), "active_scores")
        expected = (
            scores.shape[0],
            dimensions.primary_horizons,
            dimensions.layers,
            dimensions.experts,
        )
        if tuple(scores.shape) != expected:
            raise ValueError(f"active_scores have shape {tuple(scores.shape)}, expected {expected}")
        labels = _required(targets, "future_selected_ids", "target_topk")
        if not isinstance(labels, Tensor):
            raise TypeError("future_selected_ids must be a torch.Tensor")
        expected_labels = expected[:-1] + (dimensions.selected_experts,)
        if tuple(labels.shape) != expected_labels:
            raise ValueError(
                f"future_selected_ids have shape {tuple(labels.shape)}, expected {expected_labels}"
            )
        teacher = _finite_tensor(
            _required(
                targets,
                "future_centered_router_logits",
                "future_router_logits",
                "target_router_logits",
            ),
            "future_router_logits",
        )
        if teacher.shape != scores.shape:
            raise ValueError("future router logits must match active_scores")
        valid = endpoint_validity(
            _optional(targets, "future_available", "future_valid"), scores
        )
        safe_labels, valid = validate_exact_set_labels(
            scores,
            labels,
            valid=valid,
            k=dimensions.selected_experts,
        )
        return scores, safe_labels, teacher, valid

    def _set_horizons(
        self, scores: Tensor, labels: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        values = []
        horizon_valid = valid.flatten(2).sum(-1).sum(0) > 0
        for horizon in range(self.config.dimensions.primary_horizons):
            values.append(
                exact_set_nll(
                    scores[:, horizon],
                    labels[:, horizon],
                    valid[:, horizon],
                    self.config.dimensions.selected_experts,
                )
            )
        horizons = torch.stack(values)
        total = _balance_horizon_values(
            horizons, horizon_valid, self.config.horizon_worst_weight
        )
        return total, horizons

    def _recall(
        self,
        scores: Tensor,
        labels: Tensor,
        valid: Tensor,
        temperature: float,
    ) -> tuple[Tensor, Tensor]:
        memberships = soft_cardinality_topk(
            scores,
            self.config.dimensions.selected_experts,
            temperature=temperature,
            bisection_steps=self.config.soft_topk_bisection_steps,
        )
        per_endpoint = 1.0 - memberships.gather(-1, labels).sum(-1) / float(
            self.config.dimensions.selected_experts
        )
        return horizon_balanced_reduce(
            per_endpoint,
            valid,
            worst_weight=self.config.horizon_worst_weight,
        )

    def _boundary(
        self,
        scores: Tensor,
        teacher: Tensor,
        labels: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        per_endpoint = boundary_loss_per_endpoint(
            scores,
            teacher,
            labels,
            margin=self.config.boundary_margin,
            model_rank_start=self.config.model_negative_rank_start,
            teacher_rank_start=self.config.teacher_negative_rank_start,
            rank_end=self.config.negative_rank_end,
        )
        return horizon_balanced_reduce(
            per_endpoint,
            valid,
            worst_weight=self.config.horizon_worst_weight,
        )

    def _router_kl(
        self, scores: Tensor, teacher: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        target_probability = torch.softmax(teacher.detach().float(), dim=-1)
        per_endpoint = F.kl_div(
            torch.log_softmax(scores.float(), dim=-1),
            target_probability,
            reduction="none",
        ).sum(-1)
        return horizon_balanced_reduce(
            per_endpoint,
            valid,
            worst_weight=self.config.horizon_worst_weight,
        )

    def _target_queries(self, targets: Mapping[str, Any], rank: int) -> Tensor:
        direct = _optional(
            targets,
            "future_router_queries",
            "future_router_coordinates",
            "router_query_targets",
        )
        if direct is not None:
            result = _finite_tensor(direct, "future_router_queries").float()
        else:
            router_inputs = _finite_tensor(
                _required(targets, "future_router_inputs"), "future_router_inputs"
            )
            if self.router_input_basis is None:
                if router_inputs.shape[-1] != rank:
                    raise ValueError(
                        "future_router_inputs are not router coordinates; construct "
                        "HARPRTTObjective with the frozen router_input_basis"
                    )
                result = router_inputs.float()
            else:
                if router_inputs.shape[2] != self.router_input_basis.shape[0]:
                    raise ValueError("future_router_inputs disagree with router basis layers")
                if router_inputs.shape[-1] != self.router_input_basis.shape[1]:
                    raise ValueError("future_router_inputs disagree with router basis width")
                # Query supervision is expressed in the same frozen V basis
                # used by the production router geometry.  The objective is
                # evaluated inside the training autocast scope, so explicitly
                # retain the canonical FP32 projection here.
                with torch.autocast(
                    device_type=router_inputs.device.type, enabled=False
                ):
                    result = torch.einsum(
                        "bhld,ldr->bhlr",
                        router_inputs.float(),
                        self.router_input_basis.float(),
                    )
                    if self.router_rank_mask is not None:
                        result = result * self.router_rank_mask[None, None].to(
                            result.dtype
                        )
        if result.shape[-1] != rank:
            raise ValueError("router-query target rank disagrees with prediction")
        return result.detach()

    def _pooled_queries(self, outputs: Mapping[str, Any]) -> Tensor:
        queries = _finite_tensor(
            _required(outputs, "router_queries", "query_coordinates"),
            "router_queries",
        )
        if queries.ndim == 4:
            return queries.float()
        if queries.ndim != 5:
            raise ValueError("router_queries must be [B,H,L,R] or [B,H,L,N,R]")
        batch, horizons, _, nodes, _ = queries.shape
        weights = _optional(outputs, "branch_weights", "branch_posteriors")
        if weights is not None:
            weights = _finite_tensor(weights, "branch_weights").float()
            if weights.shape != (batch, horizons, nodes):
                raise ValueError("branch_weights disagree with branch-specific queries")
        else:
            logits = _optional(outputs, "branch_posterior_logits")
            mask = _optional(outputs, "branch_mask")
            if logits is not None:
                logits = _finite_tensor(logits, "branch_posterior_logits").float()
                if logits.shape != (batch, horizons, nodes):
                    raise ValueError("branch posterior logits disagree with queries")
                if mask is None:
                    weights = torch.softmax(logits, dim=-1)
                else:
                    if not isinstance(mask, Tensor) or mask.shape != logits.shape:
                        raise ValueError("branch_mask disagrees with branch posterior logits")
                    safe = mask.bool().clone()
                    missing = ~safe.any(-1)
                    if missing.any():
                        safe[..., 0] |= missing
                    weights = torch.softmax(logits.masked_fill(~safe, -torch.inf), dim=-1)
            else:
                weights = torch.full(
                    (batch, horizons, nodes),
                    1.0 / float(nodes),
                    dtype=torch.float32,
                    device=queries.device,
                )
        weights = weights.detach()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        with torch.autocast(device_type=queries.device.type, enabled=False):
            return torch.einsum(
                "bhn,bhlnr->bhlr", weights.float(), queries.float()
            )

    def _query(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        predicted = self._pooled_queries(outputs)
        expected_prefix = valid.shape
        if predicted.shape[:3] != expected_prefix:
            raise ValueError("router_queries must begin [B,H,L]")
        target = self._target_queries(targets, predicted.shape[-1])
        if target.shape != predicted.shape:
            raise ValueError("router-query prediction and target shapes disagree")
        huber = F.huber_loss(
            predicted,
            target,
            reduction="none",
            delta=self.config.query_huber_delta,
        ).mean(-1)
        cosine = 1.0 - F.cosine_similarity(predicted, target, dim=-1, eps=1e-8)
        per_endpoint = huber + self.config.query_cosine_weight * cosine
        return horizon_balanced_reduce(
            per_endpoint,
            valid,
            worst_weight=self.config.horizon_worst_weight,
        )

    def _branch_mask(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
        complete_batch: Mapping[str, Any] | None,
        expected: tuple[int, int, int],
    ) -> Tensor | None:
        value = _optional(outputs, "branch_mask")
        if value is None:
            value = _optional(targets, "tree_horizon_mask", "branch_mask")
        if value is None and complete_batch is not None:
            inputs = complete_batch.get("inputs")
            if isinstance(inputs, Mapping):
                tree = inputs.get("tree")
                if isinstance(tree, Mapping):
                    value = _optional(tree, "horizon_mask")
        if value is None:
            return None
        if not isinstance(value, Tensor):
            raise TypeError("branch horizon mask must be a torch.Tensor")
        if value.shape == expected:
            return value.bool()
        transposed = (expected[0], expected[2], expected[1])
        if value.shape == transposed:
            return value.permute(0, 2, 1).bool()
        raise ValueError("branch horizon mask must be [B,H,N] or [B,N,H]")

    def _branch(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
        complete_batch: Mapping[str, Any] | None,
        zero: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = _optional(outputs, "branch_posterior_logits")
        if logits is None:
            return zero, zero.new_zeros(self.config.dimensions.primary_horizons)
        logits = _finite_tensor(logits, "branch_posterior_logits").float()
        if logits.ndim != 3:
            raise ValueError("branch_posterior_logits must be [B,H,N]")
        expected = (
            logits.shape[0],
            self.config.dimensions.primary_horizons,
            logits.shape[2],
        )
        if logits.shape != expected:
            raise ValueError("branch_posterior_logits have the wrong horizon geometry")
        labels = _required(targets, "tree_acceptance", "branch_acceptance")
        label_valid = _required(
            targets, "tree_acceptance_valid", "branch_acceptance_valid"
        )
        if not isinstance(labels, Tensor) or not isinstance(label_valid, Tensor):
            raise TypeError("branch acceptance labels and validity must be tensors")
        captured_nodes = int(labels.shape[-1])
        if expected[2] == captured_nodes + 1:
            captured_shape = (expected[0], expected[1], captured_nodes)
            if labels.shape == (expected[0], captured_nodes):
                labels = labels[:, None].expand(captured_shape)
            elif labels.shape != captured_shape:
                raise ValueError("branch acceptance labels disagree with captured nodes")
            if label_valid.shape == (expected[0], captured_nodes):
                label_valid = label_valid[:, None].expand(captured_shape)
            elif label_valid.shape != captured_shape:
                raise ValueError("branch acceptance validity disagrees with captured nodes")

            posterior_mask = self._branch_mask(
                outputs, targets, complete_batch, expected
            )
            captured_mask = (
                torch.ones(captured_shape, dtype=torch.bool, device=logits.device)
                if posterior_mask is None else posterior_mask[..., :-1]
            )
            branch_valid = label_valid.bool().clone() & captured_mask
            target = labels.detach().float()
            if not torch.isfinite(target).all() or ((target < 0) | (target > 1)).any():
                raise ValueError("branch acceptance targets must lie in [0,1]")

            acceptance_logits = _optional(outputs, "branch_acceptance_logits")
            if acceptance_logits is None:
                acceptance_logits = logits[..., :-1]
            acceptance_logits = _finite_tensor(
                acceptance_logits, "branch_acceptance_logits"
            ).float()
            if acceptance_logits.shape == (expected[0], captured_nodes):
                acceptance_logits = acceptance_logits[:, None].expand(captured_shape)
            elif acceptance_logits.shape != captured_shape:
                raise ValueError("branch acceptance logits disagree with captured nodes")
            acceptance_per_node = F.binary_cross_entropy_with_logits(
                acceptance_logits, target, reduction="none"
            )
            acceptance_loss, acceptance_horizons = horizon_balanced_reduce(
                acceptance_per_node,
                branch_valid,
                worst_weight=self.config.horizon_worst_weight,
            )

            matched = (target > 0.5) & branch_valid
            other = torch.full(
                expected[:2], captured_nodes, dtype=torch.long, device=logits.device
            )
            matching_index = matched.float().argmax(dim=-1)
            category = torch.where(matched.any(-1), matching_index, other)
            posterior_valid = label_valid.bool().any(-1)
            posterior_per_endpoint = F.cross_entropy(
                logits.flatten(0, 1), category.flatten(), reduction="none"
            ).reshape(expected[:2])
            posterior_loss, posterior_horizons = horizon_balanced_reduce(
                posterior_per_endpoint,
                posterior_valid,
                worst_weight=self.config.horizon_worst_weight,
            )
            return (
                acceptance_loss + posterior_loss,
                acceptance_horizons + posterior_horizons,
            )
        if labels.shape == (expected[0], expected[2]):
            labels = labels[:, None].expand(expected)
        elif labels.shape != expected:
            raise ValueError("branch acceptance labels must be [B,N] or [B,H,N]")
        if label_valid.shape == (expected[0], expected[2]):
            label_valid = label_valid[:, None].expand(expected)
        elif label_valid.shape != expected:
            raise ValueError("branch acceptance validity must be [B,N] or [B,H,N]")
        # ``expand`` above creates a zero-stride view; materialize before
        # intersecting it with the per-horizon structural mask.
        branch_valid = label_valid.bool().clone()
        horizon_mask = self._branch_mask(
            outputs, targets, complete_batch, expected
        )
        if horizon_mask is not None:
            branch_valid &= horizon_mask
        # Detach explicitly: acceptance is a label-only capture field.
        target = labels.detach().float()
        if not torch.isfinite(target).all() or ((target < 0) | (target > 1)).any():
            raise ValueError("branch acceptance targets must lie in [0,1]")
        per_node = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        return horizon_balanced_reduce(
            per_node,
            branch_valid,
            worst_weight=self.config.horizon_worst_weight,
        )

    def _trajectory(
        self,
        outputs: Mapping[str, Any],
        labels: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        scores = _finite_tensor(
            _required(outputs, "trajectory_round_scores", "trajectory_scores"),
            "trajectory_round_scores",
        )
        corrections = _finite_tensor(
            _required(outputs, "trajectory_corrections"),
            "trajectory_corrections",
        )
        dimensions = self.config.dimensions
        expected = (
            labels.shape[0],
            dimensions.trajectory_rounds,
            dimensions.primary_horizons,
            dimensions.layers,
            dimensions.experts,
        )
        if tuple(scores.shape) != expected:
            raise ValueError(
                "trajectory_round_scores must be batch-major [B,R=2,H,L,E]; "
                f"got {tuple(scores.shape)}, expected {expected}"
            )
        if corrections.shape != scores.shape:
            raise ValueError("trajectory_corrections must match trajectory_round_scores")
        horizon_values = scores.new_zeros(dimensions.primary_horizons).float()
        horizon_valid = valid.flatten(2).sum(-1).sum(0) > 0
        for round_index in range(dimensions.trajectory_rounds):
            _, round_horizons = self._set_horizons(
                scores[:, round_index], labels, valid
            )
            correction_per_endpoint = corrections[:, round_index].float().square().mean(-1)
            _, correction_horizons = horizon_balanced_reduce(
                correction_per_endpoint,
                valid,
                worst_weight=0.0,
            )
            horizon_values = horizon_values + round_horizons
            horizon_values = horizon_values + (
                self.config.trajectory_correction_weight * correction_horizons
            )
        total = _balance_horizon_values(
            horizon_values, horizon_valid, self.config.horizon_worst_weight
        )
        return total, horizon_values

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch_or_targets: Mapping[str, Any],
        *,
        step: int,
    ) -> HARPRTTLossOutput:
        """Evaluate the scheduled objective at an optimizer step."""

        if not isinstance(outputs, Mapping) or not isinstance(batch_or_targets, Mapping):
            raise TypeError("outputs and batch_or_targets must be mappings")
        targets, complete_batch = self._split_batch(batch_or_targets)
        scores, labels, teacher, valid = self._validate_primary(outputs, targets)
        scheduled = self.config.schedule.at(step, self.config.weights)

        set_loss, set_horizons = self._set_horizons(scores, labels, valid)
        recall, recall_horizons = self._recall(
            scores, labels, valid, scheduled.temperature
        )
        boundary, boundary_horizons = self._boundary(
            scores, teacher, labels, valid
        )
        router, router_horizons = self._router_kl(scores, teacher, valid)
        query, query_horizons = self._query(outputs, targets, valid)
        zero = scores.sum() * 0.0
        branch, branch_horizons = self._branch(
            outputs, targets, complete_batch, zero
        )
        trajectory, trajectory_horizons = self._trajectory(
            outputs, labels, valid
        )
        components: dict[str, Tensor] = {
            "set": set_loss,
            "recall": recall,
            "boundary": boundary,
            "router_kl": router,
            "query": query,
            "branch": branch,
            "trajectory": trajectory,
        }
        horizon_components = {
            "set": set_horizons,
            "recall": recall_horizons,
            "boundary": boundary_horizons,
            "router_kl": router_horizons,
            "query": query_horizons,
            "branch": branch_horizons,
            "trajectory": trajectory_horizons,
        }
        weighted = {
            name: components[name] * float(scheduled.weights[name])
            for name in COMPONENT_NAMES
        }
        primary = weighted["set"]
        auxiliary = sum(
            (weighted[name] for name in COMPONENT_NAMES if name != "set"),
            zero,
        )
        total = primary + auxiliary
        return HARPRTTLossOutput(
            total=total,
            primary=primary,
            auxiliary=auxiliary,
            components=components,
            weighted_components=weighted,
            horizon_components=horizon_components,
            temperature=scheduled.temperature,
            recall_progress=scheduled.recall_progress,
            scheduled_weights=scheduled.weights,
        )


def _gradient_norm(loss: Tensor, parameters: tuple[Tensor, ...]) -> float:
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    square = torch.zeros((), dtype=torch.float64, device=loss.device)
    for gradient in gradients:
        if gradient is not None:
            square = square + gradient.detach().double().square().sum()
    return float(square.sqrt())


@dataclass(frozen=True)
class GradientAuditResult:
    step: int
    primary_norm: float
    auxiliary_combined_norm: float
    auxiliary_norm_sum: float
    auxiliary_to_primary_ratio: float
    component_norms: Mapping[str, float]
    passed: bool
    consecutive_passes: int
    promotion_ready: bool
    recommended_auxiliary_scale: float

    def as_dict(self) -> dict[str, float | int | bool]:
        result: dict[str, float | int | bool] = {
            "gradient_audit_step": self.step,
            "gradient_norm_primary": self.primary_norm,
            "gradient_norm_auxiliary_combined": self.auxiliary_combined_norm,
            "gradient_norm_auxiliary_sum": self.auxiliary_norm_sum,
            "gradient_auxiliary_to_primary_ratio": self.auxiliary_to_primary_ratio,
            "gradient_audit_passed": self.passed,
            "gradient_audit_consecutive_passes": self.consecutive_passes,
            "gradient_audit_promotion_ready": self.promotion_ready,
            "gradient_recommended_auxiliary_scale": self.recommended_auxiliary_scale,
        }
        result.update(
            {f"gradient_norm_{name}": value for name, value in self.component_norms.items()}
        )
        return result


class GradientAudit:
    """Gate phase promotion on ten consecutive 50-step gradient audits.

    The pass condition applies to the norm of the summed, weighted auxiliary
    gradient vector.  Individual component norms and their conservative scalar
    sum are also reported so gradient cancellation remains visible.
    """

    def __init__(
        self,
        *,
        interval: int = 50,
        maximum_auxiliary_ratio: float = 0.5,
        required_consecutive_passes: int = 10,
    ) -> None:
        if interval <= 0:
            raise ValueError("gradient audit interval must be positive")
        if not math.isfinite(maximum_auxiliary_ratio) or maximum_auxiliary_ratio < 0:
            raise ValueError("maximum_auxiliary_ratio must be finite and non-negative")
        if required_consecutive_passes <= 0:
            raise ValueError("required_consecutive_passes must be positive")
        self.interval = int(interval)
        self.maximum_auxiliary_ratio = float(maximum_auxiliary_ratio)
        self.required_consecutive_passes = int(required_consecutive_passes)
        self.consecutive_passes = 0
        self.last_audit_step: int | None = None

    def due(self, step: int) -> bool:
        if not isinstance(step, int) or step < 0:
            raise ValueError("training step must be a non-negative integer")
        return step > 0 and step % self.interval == 0

    @property
    def promotion_ready(self) -> bool:
        return self.consecutive_passes >= self.required_consecutive_passes

    def state_dict(self) -> dict[str, int | float | None]:
        return {
            "interval": self.interval,
            "maximum_auxiliary_ratio": self.maximum_auxiliary_ratio,
            "required_consecutive_passes": self.required_consecutive_passes,
            "consecutive_passes": self.consecutive_passes,
            "last_audit_step": self.last_audit_step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name in (
            "interval",
            "maximum_auxiliary_ratio",
            "required_consecutive_passes",
        ):
            if state.get(name) != getattr(self, name):
                raise ValueError(f"gradient audit state disagrees on {name}")
        passes = int(state.get("consecutive_passes", 0))
        last = state.get("last_audit_step")
        if passes < 0 or (last is not None and (not isinstance(last, int) or last < 0)):
            raise ValueError("invalid gradient audit state")
        self.consecutive_passes = passes
        self.last_audit_step = last

    def audit(
        self,
        *,
        step: int,
        primary_loss: Tensor,
        auxiliary_losses: Mapping[str, Tensor],
        parameters: Iterable[Tensor],
    ) -> GradientAuditResult | None:
        """Audit a due step without mutating ``parameter.grad`` buffers."""

        if not self.due(step):
            return None
        if self.last_audit_step is not None and step <= self.last_audit_step:
            raise ValueError("gradient audit steps must increase monotonically")
        if (
            self.last_audit_step is not None
            and step != self.last_audit_step + self.interval
        ):
            # A missed due audit breaks the consecutive-passing window.
            self.consecutive_passes = 0
        if not isinstance(primary_loss, Tensor) or primary_loss.numel() != 1:
            raise ValueError("primary_loss must be a scalar tensor")
        trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
        if not trainable:
            raise ValueError("gradient audit requires shared trainable parameters")
        component_norms: dict[str, float] = {}
        auxiliary_total = primary_loss * 0.0
        for name, loss in auxiliary_losses.items():
            if not isinstance(loss, Tensor) or loss.numel() != 1:
                raise ValueError(f"auxiliary loss {name!r} must be a scalar tensor")
            auxiliary_total = auxiliary_total + loss
            component_norms[name] = _gradient_norm(loss, trainable)
        primary_norm = _gradient_norm(primary_loss, trainable)
        combined_norm = _gradient_norm(auxiliary_total, trainable)
        auxiliary_norm_sum = float(sum(component_norms.values()))
        if primary_norm > 0 and math.isfinite(primary_norm):
            ratio = combined_norm / primary_norm
            passed = math.isfinite(ratio) and ratio <= self.maximum_auxiliary_ratio
            if combined_norm > 0:
                recommended = min(
                    1.0,
                    self.maximum_auxiliary_ratio * primary_norm / combined_norm,
                )
            else:
                recommended = 1.0
        else:
            ratio = math.inf
            passed = False
            recommended = 0.0
        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        self.last_audit_step = step
        return GradientAuditResult(
            step=step,
            primary_norm=primary_norm,
            auxiliary_combined_norm=combined_norm,
            auxiliary_norm_sum=auxiliary_norm_sum,
            auxiliary_to_primary_ratio=ratio,
            component_norms=component_norms,
            passed=passed,
            consecutive_passes=self.consecutive_passes,
            promotion_ready=self.promotion_ready,
            recommended_auxiliary_scale=recommended,
        )

    def audit_output(
        self,
        *,
        step: int,
        losses: HARPRTTLossOutput,
        parameters: Iterable[Tensor],
    ) -> GradientAuditResult | None:
        auxiliaries = {
            name: loss
            for name, loss in losses.weighted_components.items()
            if name != "set"
        }
        return self.audit(
            step=step,
            primary_loss=losses.weighted_components["set"],
            auxiliary_losses=auxiliaries,
            parameters=parameters,
        )


__all__ = [
    "COMPONENT_NAMES",
    "GradientAudit",
    "GradientAuditResult",
    "HARPRTTLossConfig",
    "HARPRTTLossOutput",
    "HARPRTTLossSchedule",
    "HARPRTTLossWeights",
    "HARPRTTObjective",
    "ScheduledLossState",
    "boundary_loss_per_endpoint",
    "endpoint_validity",
    "hard_negative_mask",
    "horizon_balanced_reduce",
]
