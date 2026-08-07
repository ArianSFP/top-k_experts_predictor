"""Strict exact-cardinality primitives for HARP-RTT.

The exact-set likelihood and inclusion marginals deliberately reuse the
audited FP32 implementation from :mod:`gcrp2r_training.model_core`.  This
module adds the stricter public contract required by HARP-RTT and the
cardinality-constrained sigmoid relaxation used by the soft Recall@8 loss.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gcrp2r_training.model_core import (
    cardinality_project_marginals,
    exact_k_logz_marginals_fast,
    exact_k_logz_with_marginals,
    exact_k_marginals,
    exact_set_nll as _gcrp_exact_set_nll,
    log_esp_k,
)


DEFAULT_K = 8
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_score_geometry(scores: Tensor, k: int) -> None:
    if not isinstance(scores, Tensor):
        raise TypeError("scores must be a torch.Tensor")
    if not scores.is_floating_point():
        raise TypeError("scores must have a floating-point dtype")
    if scores.ndim < 1:
        raise ValueError("scores must have an expert axis")
    experts = int(scores.shape[-1])
    if not 1 <= int(k) <= experts:
        raise ValueError(f"k must lie in 1..{experts}, got {k}")


def _validated_weights(
    scores: Tensor,
    valid: Tensor | None,
) -> Tensor:
    leading_shape = scores.shape[:-1]
    if valid is None:
        return torch.ones(leading_shape, device=scores.device, dtype=torch.float32)
    if not isinstance(valid, Tensor):
        raise TypeError("valid must be a torch.Tensor or None")
    if valid.shape != leading_shape:
        raise ValueError(
            "valid mask must match the score leading dimensions: "
            f"expected {tuple(leading_shape)}, got {tuple(valid.shape)}"
        )
    if valid.device != scores.device:
        raise ValueError("valid mask and scores must be on the same device")
    if valid.is_complex():
        raise TypeError("valid mask must be boolean or real-valued")
    weights = valid.to(torch.float32)
    if not torch.isfinite(weights).all():
        raise ValueError("valid weights contain NaN or Inf")
    if (weights < 0).any():
        raise ValueError("valid weights must be non-negative")
    return weights


def validate_exact_set_labels(
    scores: Tensor,
    true_ids: Tensor,
    *,
    valid: Tensor | None = None,
    k: int = DEFAULT_K,
) -> tuple[Tensor, Tensor]:
    """Validate exact-set labels and return safe IDs plus FP32 row weights.

    Positive-weight rows must contain exactly ``k`` distinct in-range expert
    IDs.  Zero-weight rows are allowed to contain capture sentinels because
    they do not define a target set; they are replaced with a safe canonical
    set before the underlying gather operation.
    """

    _validate_score_geometry(scores, k)
    if not isinstance(true_ids, Tensor):
        raise TypeError("true_ids must be a torch.Tensor")
    expected = scores.shape[:-1] + (int(k),)
    if true_ids.shape != expected:
        raise ValueError(
            "true_ids must match score leading dimensions and contain k IDs: "
            f"expected {tuple(expected)}, got {tuple(true_ids.shape)}"
        )
    if true_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError("true_ids must use an integer dtype")
    if true_ids.device != scores.device:
        raise ValueError("true_ids and scores must be on the same device")

    weights = _validated_weights(scores, valid)
    active = weights > 0
    ids = true_ids.to(torch.int64)
    experts = int(scores.shape[-1])
    active_ids = ids[active]
    if active_ids.numel():
        out_of_range = (active_ids < 0) | (active_ids >= experts)
        if out_of_range.any():
            bad = active_ids[out_of_range]
            raise ValueError(
                "active exact-set labels contain out-of-range expert IDs; "
                f"valid interval is [0, {experts - 1}], first invalid ID is {int(bad[0])}"
            )
        ordered = active_ids.sort(dim=-1).values
        if (ordered[..., 1:] == ordered[..., :-1]).any():
            raise ValueError("active exact-set labels must contain k distinct expert IDs")

    if active.all():
        return ids, weights
    canonical = torch.arange(int(k), device=scores.device, dtype=torch.int64)
    canonical = canonical.expand_as(ids)
    safe_ids = torch.where(active.unsqueeze(-1), ids, canonical)
    return safe_ids, weights


def exact_set_nll(
    scores: Tensor,
    true_ids: Tensor,
    valid: Tensor | None = None,
    k: int = DEFAULT_K,
) -> Tensor:
    """Return strict exact-``k`` set NLL with the audited analytic gradient."""

    safe_ids, weights = validate_exact_set_labels(
        scores, true_ids, valid=valid, k=k
    )
    return _gcrp_exact_set_nll(scores, safe_ids, weights, int(k))


def stable_topk(scores: Tensor, k: int = DEFAULT_K) -> Tensor:
    """Return score-descending IDs with lower expert ID breaking exact ties."""

    _validate_score_geometry(scores, k)
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., : int(k)]


def soft_cardinality_topk(
    scores: Tensor,
    k: int = DEFAULT_K,
    *,
    temperature: float = 1.0,
    bisection_steps: int = 32,
    standardize: bool = True,
    standardization_epsilon: float = 1e-6,
) -> Tensor:
    """Differentiable sigmoid top-``k`` relaxation constrained to mass ``k``.

    Scores are standardized independently along the expert axis by default.
    A per-row threshold is then solved in FP32 using exactly the requested
    number of differentiable bisection updates.  This is an auxiliary ranking
    surrogate: final inference still uses exact-set marginals and stable top-k.
    """

    _validate_score_geometry(scores, k)
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    if not isinstance(bisection_steps, int) or bisection_steps < 1:
        raise ValueError("bisection_steps must be a positive integer")
    if (
        not math.isfinite(float(standardization_epsilon))
        or float(standardization_epsilon) <= 0
    ):
        raise ValueError("standardization_epsilon must be finite and positive")

    values = scores.float()
    if not torch.isfinite(values).all():
        raise ValueError("scores contain NaN or Inf")
    if standardize:
        centered = values - values.mean(dim=-1, keepdim=True)
        scale = centered.square().mean(dim=-1, keepdim=True).sqrt()
        values = centered / scale.clamp_min(float(standardization_epsilon))

    tau = float(temperature)
    # At 32 tau outside the observed range, sigmoid saturation safely brackets
    # every non-degenerate k in FP32, including the all-equal score case.
    margin = 32.0 * tau
    lower = values.amin(dim=-1, keepdim=True) - margin
    upper = values.amax(dim=-1, keepdim=True) + margin
    target = float(k)
    for _ in range(bisection_steps):
        threshold = (lower + upper) * 0.5
        mass = torch.sigmoid((values - threshold) / tau).sum(
            dim=-1, keepdim=True
        )
        # Increasing the threshold decreases selected mass.
        lower = torch.where(mass > target, threshold, lower)
        upper = torch.where(mass > target, upper, threshold)
    threshold = (lower + upper) * 0.5
    return torch.sigmoid((values - threshold) / tau)


def soft_recall_loss(
    scores: Tensor,
    true_ids: Tensor,
    valid: Tensor | None = None,
    k: int = DEFAULT_K,
    *,
    temperature: float = 1.0,
    bisection_steps: int = 32,
) -> Tensor:
    """Return ``1 -`` mean soft SlotRecall@``k`` over valid target sets."""

    safe_ids, weights = validate_exact_set_labels(
        scores, true_ids, valid=valid, k=k
    )
    memberships = soft_cardinality_topk(
        scores,
        k,
        temperature=temperature,
        bisection_steps=bisection_steps,
    )
    per_row = memberships.gather(-1, safe_ids).sum(-1) / float(k)
    denominator = weights.sum()
    if bool(denominator == 0):
        return scores.sum() * 0.0
    return ((1.0 - per_row) * weights).sum() / denominator


__all__ = [
    "DEFAULT_K",
    "cardinality_project_marginals",
    "exact_k_logz_marginals_fast",
    "exact_k_logz_with_marginals",
    "exact_k_marginals",
    "exact_set_nll",
    "log_esp_k",
    "soft_cardinality_topk",
    "soft_recall_loss",
    "stable_topk",
    "validate_exact_set_labels",
]
