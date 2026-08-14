"""Training objectives and large-gain gates for HARP-ShadowRoute v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .exact_k import exact_set_nll
from .losses import boundary_loss_per_endpoint


PR6_BRANCH_RECALL_H2_H4 = 0.557170


@dataclass(frozen=True)
class ShadowLossWeights:
    routed: float = 0.5
    hidden: float = 0.25
    router: float = 1.0
    exact_set: float = 1.0
    boundary: float = 0.2
    language_model: float = 0.1
    factual: float = 1.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"loss weight {name} must be finite and non-negative")
        if self.router <= 0 or self.exact_set <= 0:
            raise ValueError("router and exact-set supervision must remain primary")


@dataclass(frozen=True)
class ShadowLossOutput:
    total: Tensor
    components: Mapping[str, Tensor]

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            **{name: float(value.detach()) for name, value in self.components.items()},
        }


def teacher_state_reset_interval(progress: float) -> int:
    if not math.isfinite(float(progress)) or not 0.0 <= progress <= 1.0:
        raise ValueError("training progress must lie in [0,1]")
    index = min(int(progress * 5), 4)
    return (1, 2, 4, 8, 40)[index]


def _weighted_mean(values: Tensor, valid: Tensor) -> Tensor:
    if values.shape != valid.shape:
        raise ValueError("values and validity must align")
    weights = valid.float()
    return (values.float() * weights).sum() / weights.sum().clamp_min(1.0)


def selected_expert_distillation_loss(
    predicted_outputs: Tensor,
    target_outputs: Tensor,
    selected_weights: Tensor,
    valid: Tensor,
) -> Tensor:
    if predicted_outputs.shape != target_outputs.shape or predicted_outputs.ndim < 3:
        raise ValueError("selected expert outputs must have matching [...,K,D] geometry")
    if selected_weights.shape != predicted_outputs.shape[:-1]:
        raise ValueError("selected execution weights disagree with expert outputs")
    if valid.shape != selected_weights.shape[:-1]:
        raise ValueError("expert-output validity has invalid geometry")
    per_slot = F.huber_loss(
        predicted_outputs.float(), target_outputs.detach().float(),
        reduction="none", delta=1.0,
    ).mean(-1)
    weighted = per_slot * selected_weights.detach().float()
    per_row = weighted.sum(-1) / selected_weights.detach().float().sum(-1).clamp_min(1e-12)
    return _weighted_mean(per_row, valid)


def shadow_route_objective(
    *,
    predicted_routed_delta: Tensor,
    target_routed_delta: Tensor,
    predicted_hidden: Tensor,
    target_hidden: Tensor,
    predicted_router_logits: Tensor,
    target_router_logits: Tensor,
    target_selected_ids: Tensor,
    valid: Tensor,
    predicted_lm_logits: Tensor | None = None,
    target_lm_logits: Tensor | None = None,
    target_next_ids: Tensor | None = None,
    factual_scores: Tensor | None = None,
    factual_ids: Tensor | None = None,
    factual_valid: Tensor | None = None,
    weights: ShadowLossWeights = ShadowLossWeights(),
    temperature: float = 1.0,
) -> ShadowLossOutput:
    weights.validate()
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("distillation temperature must be finite and positive")
    if predicted_routed_delta.shape != target_routed_delta.shape:
        raise ValueError("routed residual geometry differs")
    if predicted_hidden.shape != target_hidden.shape:
        raise ValueError("hidden-state geometry differs")
    if predicted_router_logits.shape != target_router_logits.shape:
        raise ValueError("router-logit geometry differs")
    leading = predicted_router_logits.shape[:-1]
    if predicted_routed_delta.shape[:-1] != leading or predicted_hidden.shape[:-1] != leading:
        raise ValueError("state and router leading axes differ")
    if target_selected_ids.shape[:-1] != leading or valid.shape != leading:
        raise ValueError("selected sets or validity disagree with router rows")
    active = valid.bool()
    safe_ids = torch.where(active[..., None], target_selected_ids.long(), 0)

    routed_rows = F.huber_loss(
        predicted_routed_delta.float(), target_routed_delta.detach().float(),
        reduction="none", delta=1.0,
    ).mean(-1)
    routed = _weighted_mean(routed_rows, active)
    hidden_huber = F.huber_loss(
        predicted_hidden.float(), target_hidden.detach().float(),
        reduction="none", delta=1.0,
    ).mean(-1)
    hidden_cosine = 1.0 - F.cosine_similarity(
        predicted_hidden.float(), target_hidden.detach().float(), dim=-1, eps=1e-8
    )
    hidden = _weighted_mean(hidden_huber + 0.1 * hidden_cosine, active)

    tau = float(temperature)
    teacher_probability = torch.softmax(target_router_logits.detach().float() / tau, dim=-1)
    router_rows = F.kl_div(
        torch.log_softmax(predicted_router_logits.float() / tau, dim=-1),
        teacher_probability,
        reduction="none",
    ).sum(-1) * (tau * tau)
    router = _weighted_mean(router_rows, active)
    exact = exact_set_nll(
        predicted_router_logits, safe_ids, valid=active, k=target_selected_ids.shape[-1]
    )
    boundary_rows = boundary_loss_per_endpoint(
        predicted_router_logits,
        target_router_logits.detach(),
        safe_ids,
        margin=0.125,
        model_rank_start=9,
        teacher_rank_start=9,
        rank_end=32,
    )
    boundary = _weighted_mean(boundary_rows, active)

    zero = predicted_router_logits.sum() * 0.0
    language_model = zero
    if any(value is not None for value in (
        predicted_lm_logits, target_lm_logits, target_next_ids
    )):
        if any(value is None for value in (
            predicted_lm_logits, target_lm_logits, target_next_ids
        )):
            raise ValueError("LM distillation fields must be supplied together")
        assert predicted_lm_logits is not None
        assert target_lm_logits is not None
        assert target_next_ids is not None
        if predicted_lm_logits.shape != target_lm_logits.shape:
            raise ValueError("LM logit geometry differs")
        if target_next_ids.shape != predicted_lm_logits.shape[:-1]:
            raise ValueError("next-token labels disagree with LM logits")
        lm_teacher = torch.softmax(target_lm_logits.detach().float() / tau, dim=-1)
        lm_kl = F.kl_div(
            torch.log_softmax(predicted_lm_logits.float() / tau, dim=-1),
            lm_teacher, reduction="none",
        ).sum(-1) * (tau * tau)
        lm_ce = F.cross_entropy(
            predicted_lm_logits.float().flatten(0, -2),
            target_next_ids.long().flatten(),
            reduction="none",
        ).reshape(target_next_ids.shape)
        lm_valid = active
        while lm_valid.ndim > target_next_ids.ndim:
            lm_valid = lm_valid.any(-1)
        if lm_valid.shape != target_next_ids.shape:
            raise ValueError("LM validity cannot be derived from route validity")
        language_model = _weighted_mean(lm_kl + lm_ce, lm_valid)

    factual = zero
    if any(value is not None for value in (factual_scores, factual_ids, factual_valid)):
        if any(value is None for value in (factual_scores, factual_ids, factual_valid)):
            raise ValueError("factual mixture fields must be supplied together")
        assert factual_scores is not None
        assert factual_ids is not None
        assert factual_valid is not None
        safe_factual = torch.where(factual_valid.bool()[..., None], factual_ids.long(), 0)
        factual = exact_set_nll(
            factual_scores, safe_factual, valid=factual_valid.bool(),
            k=factual_ids.shape[-1],
        )

    components = {
        "routed": routed,
        "hidden": hidden,
        "router": router,
        "exact_set": exact,
        "boundary": boundary,
        "language_model": language_model,
        "factual": factual,
    }
    total = sum(float(getattr(weights, name)) * value for name, value in components.items())
    return ShadowLossOutput(total=total, components=components)


@dataclass(frozen=True)
class LargeGainGate:
    passed: bool
    value: float
    threshold: float
    gain_over_pr6: float
    reason: str


def s0_scale_gate(tokens: int, recall_h2_h4: float) -> LargeGainGate:
    if tokens not in {100_000, 500_000, 1_000_000}:
        raise ValueError("S0 scale gate is defined only at 100k/500k/1M tokens")
    if not math.isfinite(float(recall_h2_h4)):
        raise ValueError("route recall must be finite")
    threshold = 0.70 if tokens == 100_000 else 0.85
    gain = float(recall_h2_h4) - PR6_BRANCH_RECALL_H2_H4
    passed = float(recall_h2_h4) >= threshold and gain >= 0.10
    return LargeGainGate(
        passed=passed,
        value=float(recall_h2_h4),
        threshold=threshold,
        gain_over_pr6=gain,
        reason="large_gain_and_absolute_gate_pass" if passed else "stop_below_large_gain_gate",
    )


__all__ = [
    "LargeGainGate",
    "PR6_BRANCH_RECALL_H2_H4",
    "ShadowLossOutput",
    "ShadowLossWeights",
    "s0_scale_gate",
    "selected_expert_distillation_loss",
    "shadow_route_objective",
    "teacher_state_reset_interval",
]
