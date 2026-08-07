"""Endpoint losses and gradient diagnostics for HARP-8T."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn import functional as F

from .config import LossConfig


@dataclass
class HARPLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    metrics: dict[str, float]


def _masked_layer_horizon(
    rows: torch.Tensor,
    valid_future: torch.Tensor,
) -> torch.Tensor:
    """Reduce [B,H,L...] rows to one value per horizon."""

    while rows.ndim > 3:
        rows = rows.mean(dim=-1)
    if rows.ndim == 2:
        rows = rows.unsqueeze(-1)
    mask = valid_future[:, :, None].expand_as(rows)
    numerator = (rows * mask).sum(dim=(0, 2))
    denominator = mask.sum(dim=(0, 2)).clamp_min(1)
    return numerator / denominator


def _masked_token_horizon(
    rows: torch.Tensor,
    valid_future: torch.Tensor,
) -> torch.Tensor:
    while rows.ndim > 2:
        rows = rows.mean(dim=-1)
    numerator = (rows * valid_future).sum(dim=0)
    denominator = valid_future.sum(dim=0).clamp_min(1)
    return numerator / denominator


def _horizon_weights(
    horizons: int,
    config: LossConfig,
    active_horizons: Iterable[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    active = set(int(value) for value in active_horizons)
    invalid = sorted(value for value in active if not 1 <= value <= horizons)
    if invalid:
        raise ValueError(f"active horizons outside model range: {invalid}")
    weights = torch.zeros(horizons, device=device, dtype=dtype)
    configured = config.horizon_weights
    if configured is not None and len(configured) != horizons:
        raise ValueError("configured horizon weights disagree with model horizons")
    if configured is not None and any(float(value) < 0 for value in configured):
        raise ValueError("horizon weights must be non-negative")
    for horizon in active:
        weights[horizon - 1] = (
            float(configured[horizon - 1])
            if configured is not None
            else (config.h2_weight if horizon == 2 else 1.0)
        )
    if weights.sum() == 0:
        raise ValueError("at least one active horizon is required")
    return weights / weights.sum()


def endpoint_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: LossConfig,
    *,
    active_horizons: Iterable[int] = tuple(range(1, 9)),
    coefficients: dict[str, float] | None = None,
) -> HARPLossOutput:
    """Compute forward-KL, boundary, inclusion, score and latent losses."""

    predicted = outputs["future_router_scores"].float()
    teacher = batch["teacher_router_scores"].float()
    target_top8 = batch["target_top8"].long()
    valid = batch["valid_future"].bool()
    if predicted.shape != teacher.shape:
        raise ValueError("student and teacher router-score geometry differs")
    if target_top8.shape != predicted.shape[:-1] + (8,):
        raise ValueError("top-8 labels disagree with router-score geometry")
    if valid.shape != predicted.shape[:2]:
        raise ValueError("future-validity mask disagrees with horizon geometry")
    if config.hard_negative_end_rank <= 8 or config.hard_negative_end_rank > predicted.shape[-1]:
        raise ValueError("hard-negative end rank must lie in [9, E]")

    temperature = float(config.temperature)
    teacher_probability = torch.softmax(teacher / temperature, dim=-1)
    kl_rows = F.kl_div(
        torch.log_softmax(predicted / temperature, dim=-1),
        teacher_probability,
        reduction="none",
    ).sum(dim=-1) * temperature**2
    router_kl_h = _masked_layer_horizon(kl_rows, valid)

    teacher_order = torch.topk(
        teacher,
        config.hard_negative_end_rank,
        dim=-1,
        sorted=True,
    ).indices
    positive = predicted.gather(-1, target_top8)
    negative_ids = teacher_order[..., 8 : config.hard_negative_end_rank]
    negative = predicted.gather(-1, negative_ids)
    pairwise = F.softplus(-(positive[..., :, None] - negative[..., None, :]))
    boundary_h = _masked_layer_horizon(pairwise, valid)

    positive_bce = F.softplus(-positive).mean(dim=-1)
    negative_bce = F.softplus(negative).mean(dim=-1)
    inclusion_h = _masked_layer_horizon(
        0.5 * (positive_bce + negative_bce), valid
    )

    membership = torch.zeros_like(predicted, dtype=torch.bool)
    membership.scatter_(-1, target_top8, True)
    full_positive = F.softplus(-predicted).masked_fill(~membership, 0.0)
    full_negative = F.softplus(predicted).masked_fill(membership, 0.0)
    membership_rows = 0.5 * (
        full_positive.sum(dim=-1) / target_top8.shape[-1]
        + full_negative.sum(dim=-1)
        / (predicted.shape[-1] - target_top8.shape[-1])
    )
    full_membership_h = _masked_layer_horizon(membership_rows, valid)

    effective_candidate_count = min(config.candidate_count, predicted.shape[-1])
    if not target_top8.shape[-1] < effective_candidate_count:
        raise ValueError("candidate count must exceed native top-k and fit namespace")
    negative_boundary_rank = (
        effective_candidate_count - target_top8.shape[-1] + 1
    )
    negative_only = predicted.masked_fill(membership, -torch.inf)
    candidate_boundary = torch.topk(
        negative_only,
        negative_boundary_rank,
        dim=-1,
        sorted=True,
    ).values[..., -1]
    candidate_rows = F.softplus(
        candidate_boundary[..., None]
        - positive
        + float(config.candidate_margin)
    ).mean(dim=-1)
    candidate16_h = _masked_layer_horizon(candidate_rows, valid)

    predicted_centered = predicted - predicted.mean(dim=-1, keepdim=True)
    teacher_centered = teacher - teacher.mean(dim=-1, keepdim=True)
    score_rows = F.smooth_l1_loss(
        predicted_centered,
        teacher_centered,
        reduction="none",
    ).mean(dim=-1)
    centered_score_h = _masked_layer_horizon(score_rows, valid)

    if "future_latent" in outputs and "future_latent_target" in batch:
        latent_rows = F.smooth_l1_loss(
            outputs["future_latent"].float(),
            batch["future_latent_target"].float(),
            reduction="none",
        ).mean(dim=-1)
        future_latent_h = _masked_token_horizon(latent_rows, valid)
    else:
        future_latent_h = torch.zeros_like(router_kl_h)

    horizon_weights = _horizon_weights(
        predicted.shape[1],
        config,
        active_horizons,
        device=predicted.device,
        dtype=predicted.dtype,
    )
    per_component_h = {
        "router_kl": router_kl_h,
        "boundary": boundary_h,
        "inclusion": inclusion_h,
        "centered_score": centered_score_h,
        "future_latent": future_latent_h,
        "full_membership": full_membership_h,
        "candidate16": candidate16_h,
    }
    unweighted = {
        name: (values * horizon_weights).sum()
        for name, values in per_component_h.items()
    }
    weights = {
        "router_kl": config.router_kl,
        "boundary": config.boundary,
        "inclusion": config.inclusion,
        "centered_score": config.centered_score,
        "future_latent": config.future_latent,
        "full_membership": config.full_membership,
        "candidate16": config.candidate16,
    }
    if coefficients is not None:
        unknown = sorted(set(coefficients) - set(weights))
        if unknown:
            raise ValueError(f"unknown loss coefficients: {unknown}")
        weights.update({name: float(value) for name, value in coefficients.items()})
    weighted = {name: unweighted[name] * weights[name] for name in unweighted}
    total = sum(weighted.values())
    components = {
        **unweighted,
        **{f"weighted_{name}": value for name, value in weighted.items()},
    }
    metrics = {name: float(value.detach()) for name, value in components.items()}
    metrics["loss"] = float(total.detach())
    for horizon in range(predicted.shape[1]):
        metrics[f"router_kl_h{horizon + 1}"] = float(
            router_kl_h[horizon].detach()
        )
        metrics[f"boundary_h{horizon + 1}"] = float(boundary_h[horizon].detach())
    return HARPLossOutput(total=total, components=components, metrics=metrics)


def component_gradient_norms(
    components: dict[str, torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
) -> dict[str, float]:
    """Measure unweighted component gradient norms on a shared parameter set."""

    selected = [parameter for parameter in parameters if parameter.requires_grad]
    result: dict[str, float] = {}
    names = [name for name in components if not name.startswith("weighted_")]
    for index, name in enumerate(names):
        gradients = torch.autograd.grad(
            components[name],
            selected,
            retain_graph=index + 1 < len(names),
            allow_unused=True,
        )
        squared = sum(
            gradient.detach().float().square().sum()
            for gradient in gradients
            if gradient is not None
        )
        result[name] = float(torch.sqrt(squared).cpu()) if not isinstance(squared, int) else 0.0
    return result


def calibrated_coefficients(
    median_norms: dict[str, float],
    *,
    minimum: float = 0.01,
    maximum: float = 1.0,
) -> dict[str, float]:
    """Freeze coefficients at the preregistered gradient-ratio targets."""

    targets = {
        "router_kl": 1.0,
        "boundary": 0.5,
        "inclusion": 0.25,
        "centered_score": 0.1,
        "future_latent": 0.1,
        "full_membership": 0.5,
        "candidate16": 1.0,
    }
    reference = max(float(median_norms.get("router_kl", 0.0)), 1e-12)
    values: dict[str, float] = {"router_kl": 1.0}
    for name, target in targets.items():
        if name == "router_kl":
            continue
        norm = max(float(median_norms.get(name, 0.0)), 1e-12)
        values[name] = min(maximum, max(minimum, target * reference / norm))
    return values
