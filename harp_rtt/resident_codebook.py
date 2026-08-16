"""Deterministic fitting primitives for the resident functional codebook."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ResidentProxyFit:
    proxy_ids: Tensor
    coefficients: Tensor
    proxy_count: int
    weighted_error: float


def _validate_fit_inputs(
    target: Tensor,
    candidates: Tensor,
    weights: Tensor,
    candidate_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if target.ndim != 2:
        raise ValueError("resident proxy target must be [rows,width]")
    if candidates.ndim != 3 or candidates.shape[0] != target.shape[0]:
        raise ValueError("resident proxy candidates must be [rows,candidates,width]")
    if candidates.shape[2] != target.shape[1]:
        raise ValueError("resident proxy target/candidate widths differ")
    if weights.shape != (target.shape[0],):
        raise ValueError("resident proxy weights must be [rows]")
    if candidate_ids.shape != (candidates.shape[1],):
        raise ValueError("resident proxy IDs do not match candidate values")
    if target.shape[0] < 1 or candidates.shape[1] < 1:
        raise ValueError("resident proxy fit requires rows and candidates")
    if candidate_ids.dtype not in {
        torch.int16, torch.int32, torch.int64, torch.uint8,
    }:
        raise TypeError("resident proxy IDs must be integers")
    if (
        not torch.isfinite(target).all()
        or not torch.isfinite(candidates).all()
        or not torch.isfinite(weights).all()
        or bool((weights < 0).any())
        or float(weights.sum()) <= 0
    ):
        raise ValueError("resident proxy fit inputs contain invalid values")
    ids = candidate_ids.long()
    if ids.unique().numel() != ids.numel():
        raise ValueError("resident proxy candidate IDs must be unique")
    order = torch.argsort(ids, stable=True)
    return (
        target.float(),
        candidates[:, order].float(),
        weights.float(),
        ids[order],
    )


def _weighted_error(target: Tensor, prediction: Tensor, weights: Tensor) -> Tensor:
    row_scale = target.square().mean(-1).clamp_min(1e-8)
    row_error = (target - prediction).square().mean(-1) / row_scale
    return (row_error * weights).sum() / weights.sum().clamp_min(1e-8)


def fit_resident_proxy(
    target: Tensor,
    candidates: Tensor,
    weights: Tensor,
    candidate_ids: Tensor,
    *,
    proxies: int,
) -> ResidentProxyFit:
    """Fit the best stable one- or two-resident simplex approximation.

    Candidates must contain deployed INT4 resident outputs on the same
    activations as target. For two proxies every pair is considered and its
    optimal convex coefficient is solved analytically.
    """

    if proxies not in {1, 2}:
        raise ValueError("resident codebook supports exactly one or two proxies")
    target, candidates, weights, candidate_ids = _validate_fit_inputs(
        target, candidates, weights, candidate_ids
    )
    errors = torch.stack([
        _weighted_error(target, candidates[:, index], weights)
        for index in range(candidates.shape[1])
    ])
    best_index = min(
        range(errors.numel()),
        key=lambda index: (float(errors[index]), int(candidate_ids[index])),
    )
    best = ResidentProxyFit(
        proxy_ids=torch.tensor(
            [int(candidate_ids[best_index]), -1], dtype=torch.int16
        ),
        coefficients=torch.tensor([1.0, 0.0], dtype=torch.bfloat16),
        proxy_count=1,
        weighted_error=float(errors[best_index]),
    )
    if proxies == 1 or candidates.shape[1] == 1:
        return best

    pair_best: tuple[float, int, int, float] | None = None
    for left in range(candidates.shape[1] - 1):
        for right in range(left + 1, candidates.shape[1]):
            left_value = candidates[:, left]
            right_value = candidates[:, right]
            direction = left_value - right_value
            residual = target - right_value
            row_scale = target.square().mean(-1).clamp_min(1e-8)
            expanded_weights = (weights / row_scale)[:, None]
            denominator = (
                expanded_weights * direction.square()
            ).sum().clamp_min(1e-12)
            alpha = float(
                ((expanded_weights * residual * direction).sum() / denominator)
                .clamp(0.0, 1.0)
            )
            prediction = alpha * left_value + (1.0 - alpha) * right_value
            error = float(_weighted_error(target, prediction, weights))
            candidate = (
                error,
                int(candidate_ids[left]),
                int(candidate_ids[right]),
                alpha,
            )
            if pair_best is None or candidate[:3] < pair_best[:3]:
                pair_best = candidate
    assert pair_best is not None
    error, left_id, right_id, alpha = pair_best
    if error >= best.weighted_error - 1e-12 or alpha <= 1e-4 or alpha >= 1.0 - 1e-4:
        return best
    return ResidentProxyFit(
        proxy_ids=torch.tensor([left_id, right_id], dtype=torch.int16),
        coefficients=torch.tensor([alpha, 1.0 - alpha], dtype=torch.bfloat16),
        proxy_count=2,
        weighted_error=error,
    )


def validate_codebook_tables(
    resident_ids: Tensor,
    proxy_ids: Tensor,
    proxy_coefficients: Tensor,
    proxy_count: Tensor,
    *,
    experts: int = 256,
) -> None:
    """Fail closed on malformed or out-of-namespace deployment mappings."""

    residents = torch.as_tensor(resident_ids, dtype=torch.long)
    ids = torch.as_tensor(proxy_ids, dtype=torch.long)
    coefficients = torch.as_tensor(proxy_coefficients, dtype=torch.float32)
    counts = torch.as_tensor(proxy_count, dtype=torch.long)
    if (
        residents.ndim != 1
        or residents.unique().numel() != residents.numel()
        or bool(((residents < 0) | (residents >= experts)).any())
    ):
        raise ValueError("resident codebook namespace is invalid")
    if (
        ids.shape != (experts, 2)
        or coefficients.shape != (experts, 2)
        or counts.shape != (experts,)
        or bool(((counts < 1) | (counts > 2)).any())
    ):
        raise ValueError("resident codebook table geometry is invalid")
    active = torch.arange(2)[None] < counts[:, None]
    if bool((~torch.isin(ids[active], residents)).any()):
        raise ValueError("resident codebook proxy lies outside its resident namespace")
    active_coefficients = coefficients * active
    if (
        not torch.isfinite(coefficients).all()
        or bool((active_coefficients < 0).any())
        or not torch.allclose(
            active_coefficients.sum(-1), torch.ones(experts), atol=2e-3, rtol=0
        )
    ):
        raise ValueError("resident codebook coefficients are not a simplex")


__all__ = [
    "ResidentProxyFit",
    "fit_resident_proxy",
    "validate_codebook_tables",
]
