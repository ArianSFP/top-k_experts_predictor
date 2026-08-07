"""Numerically strict primitives shared by GCRP-2R v1.3 training phases."""

from __future__ import annotations

import hashlib
import torch
from torch import Tensor, nn


def initialize_standard_layers(module: nn.Module) -> None:
    """Apply the exact v1.3 standard initialization to ordinary modules."""
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight, gain=1.0)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Embedding):
            nn.init.normal_(child.weight, mean=0.0, std=0.02)
        elif isinstance(child, nn.RMSNorm):
            if child.elementwise_affine:
                nn.init.ones_(child.weight)
        elif isinstance(child, nn.MultiheadAttention):
            if child.in_proj_weight is not None:
                nn.init.xavier_uniform_(child.in_proj_weight, gain=1.0)
            if child.in_proj_bias is not None:
                nn.init.zeros_(child.in_proj_bias)


def model_state_sha256(module: nn.Module) -> str:
    """Hash names, shapes, dtypes, and exact bytes of a configured state dict."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def center_expert_axis(values: Tensor) -> Tensor:
    return values - values.mean(dim=-1, keepdim=True)


def deterministic_topk(scores: Tensor, k: int) -> Tensor:
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]


def log_esp_k(scores: Tensor, k: int) -> Tensor:
    """FP32 log elementary-symmetric polynomial of order ``k``."""
    x = scores.float()
    if not 0 <= k <= x.shape[-1]:
        raise ValueError(f"invalid k={k} for E={x.shape[-1]}")
    if k == 0:
        return torch.zeros_like(x[..., 0])
    shift = x.max(dim=-1, keepdim=True).values
    x = x - shift
    neg_inf = torch.full_like(x[..., 0], float("-inf"))
    dp = torch.stack(
        [torch.zeros_like(neg_inf)] + [neg_inf for _ in range(k)], dim=-1
    )
    for expert in range(x.shape[-1]):
        updated = [dp[..., 0]]
        for order in range(1, k + 1):
            include = dp[..., order - 1] + x[..., expert]
            if order > expert + 1:
                value = dp[..., order]
            elif order == expert + 1:
                value = include
            else:
                value = torch.logaddexp(dp[..., order], include)
            updated.append(value)
        dp = torch.stack(updated, dim=-1)
    return dp[..., k] + k * shift.squeeze(-1)


def exact_k_marginals(scores: Tensor, k: int) -> Tensor:
    """Recover exact inclusion marginals using log prefix/suffix recurrences.

    This routine is intended for detached metrics and candidate features.  The
    exact-set training loss differentiates through :func:`log_esp_k` directly.
    """
    x = scores.float()
    experts = x.shape[-1]
    if not 1 <= k <= experts:
        raise ValueError(f"invalid k={k} for E={experts}")
    shift = x.max(dim=-1, keepdim=True).values
    shifted = x - shift
    neg_inf = torch.full_like(x[..., 0], float("-inf"))
    initial = torch.stack(
        [torch.zeros_like(neg_inf)] + [neg_inf for _ in range(k)], dim=-1
    )
    prefix = [initial]
    for expert in range(experts):
        old = prefix[-1]
        updated = [old[..., 0]]
        for order in range(1, k + 1):
            include = old[..., order - 1] + shifted[..., expert]
            if order > expert + 1:
                value = old[..., order]
            elif order == expert + 1:
                value = include
            else:
                value = torch.logaddexp(old[..., order], include)
            updated.append(value)
        prefix.append(torch.stack(updated, dim=-1))
    suffix: list[Tensor] = [initial] * (experts + 1)
    suffix[experts] = initial
    for expert in range(experts - 1, -1, -1):
        old = suffix[expert + 1]
        processed = experts - 1 - expert
        updated = [old[..., 0]]
        for order in range(1, k + 1):
            include = old[..., order - 1] + shifted[..., expert]
            if order > processed + 1:
                value = old[..., order]
            elif order == processed + 1:
                value = include
            else:
                value = torch.logaddexp(old[..., order], include)
            updated.append(value)
        suffix[expert] = torch.stack(updated, dim=-1)
    log_z_shifted = prefix[-1][..., k]
    marginals = []
    for expert in range(experts):
        parts = [
            prefix[expert][..., selected]
            + suffix[expert + 1][..., k - 1 - selected]
            for selected in range(k)
        ]
        without = torch.logsumexp(torch.stack(parts, dim=-1), dim=-1)
        marginals.append(torch.exp(shifted[..., expert] + without - log_z_shifted))
    return torch.stack(marginals, dim=-1)


@torch.no_grad()
def exact_k_logz_marginals_fast(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Exact FP32 prefix/suffix evaluation without an autograd graph."""
    original_shape = scores.shape[:-1]
    experts = scores.shape[-1]
    flat = scores.float().reshape(-1, experts)
    shift = flat.max(-1, keepdim=True).values
    x = flat - shift
    prefix = torch.full(
        (flat.shape[0], experts + 1, k + 1),
        float("-inf"),
        device=flat.device,
        dtype=torch.float32,
    )
    suffix = torch.full_like(prefix, float("-inf"))
    prefix[:, 0, 0] = 0.0
    suffix[:, experts, 0] = 0.0
    for expert in range(experts):
        prefix[:, expert + 1, 0] = 0.0
        upper = min(k, expert + 1)
        for order in range(1, upper + 1):
            include = prefix[:, expert, order - 1] + x[:, expert]
            if order == expert + 1:
                prefix[:, expert + 1, order] = include
            else:
                prefix[:, expert + 1, order] = torch.logaddexp(
                    prefix[:, expert, order], include
                )
    for expert in range(experts - 1, -1, -1):
        suffix[:, expert, 0] = 0.0
        processed = experts - expert
        upper = min(k, processed)
        for order in range(1, upper + 1):
            include = suffix[:, expert + 1, order - 1] + x[:, expert]
            if order == processed:
                suffix[:, expert, order] = include
            else:
                suffix[:, expert, order] = torch.logaddexp(
                    suffix[:, expert + 1, order], include
                )
    log_z_shifted = prefix[:, experts, k]
    terms = torch.stack(
        [
            prefix[:, :experts, order]
            + suffix[:, 1:, k - 1 - order]
            for order in range(k)
        ],
        dim=-1,
    )
    marginals = torch.exp(x + torch.logsumexp(terms, dim=-1) - log_z_shifted[:, None])
    log_z = log_z_shifted + k * shift.squeeze(-1)
    return log_z.reshape(original_shape), marginals.reshape(*original_shape, experts)


def cardinality_project_marginals(marginals: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Repair small FP32 cardinality drift and return the auditable pre-error."""
    values = marginals.float()
    total = values.sum(-1, keepdim=True)
    pre_error = (total.squeeze(-1) - float(k)).abs()
    projected = values * (float(k) / total.clamp_min(torch.finfo(torch.float32).tiny))
    return projected, pre_error


class _ExactSetNLL(torch.autograd.Function):
    """Exact loss with analytic marginal gradient and bounded saved state."""

    @staticmethod
    def forward(ctx, scores: Tensor, true_ids: Tensor, valid: Tensor, k: int) -> Tensor:
        log_z, marginals = exact_k_logz_marginals_fast(scores, k)
        ids = true_ids.long()
        true_sum = scores.float().gather(-1, ids).sum(-1)
        weights = valid.to(torch.float32)
        denominator = weights.sum().clamp_min(1.0)
        loss = ((log_z - true_sum) * weights).sum() / denominator
        ctx.save_for_backward(marginals, ids, weights, denominator)
        ctx.input_dtype = scores.dtype
        return loss

    @staticmethod
    def backward(ctx, gradient: Tensor):
        marginals, ids, weights, denominator = ctx.saved_tensors
        score_gradient = marginals.clone()
        negative = torch.full_like(ids, -1.0, dtype=score_gradient.dtype)
        score_gradient.scatter_add_(-1, ids, negative)
        score_gradient *= (weights / denominator)[..., None]
        score_gradient *= gradient.float()
        return score_gradient.to(ctx.input_dtype), None, None, None


def exact_set_nll(
    scores: Tensor, true_ids: Tensor, valid: Tensor | None = None, k: int = 8
) -> Tensor:
    if valid is None:
        valid = torch.ones(scores.shape[:-1], device=scores.device, dtype=torch.float32)
    return _ExactSetNLL.apply(scores, true_ids, valid, k)


class _ExactLogZMarginals(torch.autograd.Function):
    """Return log Z and detached marginals from one exact-k DP pass.

    Phase 3 needs both quantities.  The log-partition backward is exactly the
    marginal vector, while the state/dependence objective intentionally treats
    marginals as stop-gradient values.
    """

    @staticmethod
    def forward(ctx, scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
        log_z, marginals = exact_k_logz_marginals_fast(scores, k)
        ctx.save_for_backward(marginals)
        ctx.input_dtype = scores.dtype
        ctx.mark_non_differentiable(marginals)
        return log_z, marginals

    @staticmethod
    def backward(ctx, gradient_log_z: Tensor, gradient_marginals: Tensor | None):
        del gradient_marginals
        (marginals,) = ctx.saved_tensors
        gradient = gradient_log_z.float().unsqueeze(-1) * marginals
        return gradient.to(ctx.input_dtype), None


def exact_k_logz_with_marginals(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Compute differentiable exact-k log Z and stop-gradient marginals once."""
    return _ExactLogZMarginals.apply(scores, k)


def informational_availability(
    source_valid: Tensor, source_retained: Tensor
) -> Tensor:
    """Availability for the approved informational-causality profile."""
    return source_valid * source_retained


def physical_availability(
    source_valid: Tensor, ready_by_view: Tensor, source_retained: Tensor
) -> Tensor:
    """Original v1.3 runtime availability, retained for auditing."""
    return source_valid[:, None, :] * ready_by_view * source_retained[:, None, :]


def frechet_joint(mu1: Tensor, mu2: Tensor, dependence: Tensor) -> Tensor:
    mu1 = mu1.float()
    mu2 = mu2.float()
    lower = torch.clamp(mu1 + mu2 - 1.0, min=0.0)
    upper = torch.minimum(mu1, mu2)
    p11 = lower + (upper - lower) * torch.sigmoid(dependence.float())
    return torch.stack(
        (1.0 - mu1 - mu2 + p11, mu1 - p11, mu2 - p11, p11), dim=-1
    )
