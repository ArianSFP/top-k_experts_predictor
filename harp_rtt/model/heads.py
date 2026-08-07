"""Hybrid router-geometry head and exact-cardinality branch mixture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from harp_rtt.exact_k import (
    cardinality_project_marginals,
    exact_k_logz_with_marginals,
    soft_cardinality_topk,
)

from .common import zero_linear
from .config import HARPRTTConfig
from .tree import TreeEncoding


def exact_projected_marginals(scores: Tensor, k: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return exact log-Z, sum-k marginals, and pre-projection mass error."""

    log_z, marginals = exact_k_logz_with_marginals(scores, k)
    projected, error = cardinality_project_marginals(marginals, k)
    return log_z, projected, error


def _exact_forward_straight_through_marginals(
    scores: Tensor,
    exact_marginals: Tensor,
    k: int,
) -> Tensor:
    """Keep exact marginals in the forward pass and supply a trainable VJP.

    The audited exact-k primitive intentionally exposes stop-gradient
    marginals, which is appropriate for diagnostics but would sever the
    branch-mixture path from the primary loss.  The exact marginal Jacobian
    is a dense expert covariance (O(E^2) per endpoint).  During training we
    therefore attach the bounded O(E * bisection_steps) cardinality sigmoid
    only for the backward pass.  It uses the same unstandardized
    exponential-family score scale at temperature one.  Inference and every
    public forward value remain the exact dynamic-program marginals.
    """

    if not torch.is_grad_enabled() or not scores.requires_grad:
        return exact_marginals
    surrogate = soft_cardinality_topk(
        scores,
        k,
        temperature=1.0,
        bisection_steps=32,
        standardize=False,
    )
    return exact_marginals + (surrogate - surrogate.detach()).to(
        exact_marginals.dtype
    )


@dataclass(frozen=True)
class HybridScoreOutput:
    branch_scores: Tensor
    branch_logits: Tensor
    geometry_scores: Tensor
    free_scores: Tensor
    transition_scores: Tensor
    predicted_queries: Tensor


@dataclass(frozen=True)
class BranchMixtureOutput:
    branch_marginals: Tensor
    branch_weights: Tensor
    mixture_marginals: Tensor
    mixture_logits: Tensor
    branch_log_z: Tensor
    cardinality_error: Tensor


class HybridRouterScoreHead(nn.Module):
    """Score every expert through frozen geometry plus free residuals."""

    def __init__(
        self,
        config: HARPRTTConfig,
        expert_keys: Tensor,
        rank_mask: Tensor,
        centered_bias: Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        expected_keys = (config.layers, config.experts, config.router_rank)
        if expert_keys.shape != expected_keys:
            raise ValueError("expert_keys have the wrong shape")
        if rank_mask.shape != (config.layers, config.router_rank):
            raise ValueError("router rank_mask has the wrong shape")
        if centered_bias.shape != (config.layers, config.experts):
            raise ValueError("centered router bias has the wrong shape")
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.register_buffer("rank_mask", rank_mask.detach().bool().clone())
        self.register_buffer("centered_bias", centered_bias.detach().float().clone())
        self.query = nn.Linear(config.model_width, config.router_rank)
        self.branch_query = nn.Linear(config.tree_width, config.router_rank)
        self.free_hidden = nn.Linear(config.model_width, config.decoder_ffn_width * 2)
        self.free_branch = nn.Linear(config.tree_width, config.decoder_ffn_width)
        self.free_output = nn.Linear(config.decoder_ffn_width, config.experts)
        # Scalar gates are zero, rather than sigmoid logits, so step zero is
        # bitwise equal to the supplied HARP score tensor.
        self.geometry_gate = nn.Parameter(
            torch.zeros(config.active_horizons, config.layers, 1)
        )
        self.free_gate = nn.Parameter(
            torch.zeros(config.active_horizons, config.layers, 1)
        )
        self.transition_gate = nn.Parameter(
            torch.zeros(config.active_horizons, config.layers, 1)
        )

    def forward(
        self,
        endpoint: Tensor,
        tree: TreeEncoding,
        anchor_scores: Tensor,
        *,
        transition_scores: Tensor | None = None,
    ) -> HybridScoreOutput:
        config = self.config
        batch, horizons, layers, _ = endpoint.shape
        nodes = tree.states.shape[1]
        expected = (batch, config.active_horizons, config.layers, config.experts)
        if anchor_scores.shape != expected:
            raise ValueError("anchor active scores have the wrong shape")
        base_query = self.query(endpoint)
        branch_delta = self.branch_query(tree.states)
        predicted_query = base_query[:, :, :, None, :] + branch_delta[:, None, None]
        predicted_query = predicted_query * self.rank_mask[None, None, :, None].to(
            predicted_query.dtype
        )
        # K q + b_c is the frozen target-router score path.  Keep the learned
        # q construction autocastable, then score its resulting values in
        # FP32 so autocast cannot perturb expert ordering at the top-8 edge.
        with torch.autocast(
            device_type=predicted_query.device.type, enabled=False
        ):
            geometry = torch.einsum(
                "bhlnr,ler->bhlne",
                predicted_query.float(),
                self.expert_keys.float(),
            ) + self.centered_bias.float()[None, None, :, None]

        gate, value = self.free_hidden(endpoint).chunk(2, dim=-1)
        free_base = F.silu(gate) * value
        free_hidden = free_base[:, :, :, None] + self.free_branch(tree.states)[
            :, None, None
        ]
        free = self.free_output(F.silu(free_hidden))
        if transition_scores is None:
            transition = anchor_scores.new_zeros(expected)
        else:
            if transition_scores.shape != expected:
                raise ValueError("transition_scores have the wrong shape")
            transition = transition_scores
        branch_scores = (
            anchor_scores[:, :, :, None]
            + self.geometry_gate[None, :, :, None] * geometry
            + self.free_gate[None, :, :, None] * free
            + self.transition_gate[None, :, :, None]
            * transition[:, :, :, None]
        )
        return HybridScoreOutput(
            branch_scores=branch_scores,
            branch_logits=tree.posterior_logits,
            geometry_scores=geometry,
            free_scores=free,
            transition_scores=transition,
            predicted_queries=predicted_query,
        )


class ExactBranchMixture(nn.Module):
    """Mix branch exact-k marginals with normalized path posteriors."""

    def __init__(self, config: HARPRTTConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        branch_scores: Tensor,
        branch_logits: Tensor,
        branch_mask: Tensor,
    ) -> BranchMixtureOutput:
        # scores [B,H,L,N,E], posteriors/mask [B,H,N]
        if branch_scores.ndim != 5:
            raise ValueError("branch scores must be [B,H,L,N,E]")
        if branch_logits.shape != branch_scores.shape[:2] + (branch_scores.shape[3],):
            raise ValueError("branch posterior logits disagree with branch scores")
        if branch_mask.shape != branch_logits.shape:
            raise ValueError("branch mask disagrees with branch posterior logits")
        safe_mask = branch_mask.bool().clone()
        missing = ~safe_mask.any(dim=-1)
        if missing.any():
            # Avoid version-dependent mixed boolean/integer advanced-indexing
            # semantics (PyTorch 2.8 selected the wrong axis here).  Every
            # empty [B,H] row gets node zero as its finite fallback.
            safe_mask[..., 0] |= missing
        safe_logits = branch_logits.masked_fill(~safe_mask, -torch.inf)
        weights = torch.softmax(safe_logits.float(), dim=-1)
        weights = weights * safe_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        log_z, marginals, error = exact_projected_marginals(
            branch_scores, self.config.exact_k
        )
        marginals = _exact_forward_straight_through_marginals(
            branch_scores, marginals, self.config.exact_k
        )
        # Exact marginals and their branch mixture are part of the FP32
        # cardinality-constrained router path, not a learned BF16 feature
        # transform.
        with torch.autocast(device_type=marginals.device.type, enabled=False):
            mixture = torch.einsum(
                "bhn,bhlne->bhle", weights.float(), marginals.float()
            )
        mixture, mixture_error = cardinality_project_marginals(
            mixture, self.config.exact_k
        )
        epsilon = torch.finfo(mixture.dtype).eps
        logits = torch.logit(mixture.clamp(epsilon, 1.0 - epsilon))
        logits = logits - logits.mean(dim=-1, keepdim=True)
        return BranchMixtureOutput(
            branch_marginals=marginals,
            branch_weights=weights,
            mixture_marginals=mixture,
            mixture_logits=logits,
            branch_log_z=log_z,
            cardinality_error=torch.maximum(
                error.amax(dim=-1), mixture_error
            ),
        )


class BranchMixtureResidual(nn.Module):
    """Zero-initialized bridge from branch marginals to the shared score path."""

    def __init__(self, config: HARPRTTConfig) -> None:
        super().__init__()
        self.gate = nn.Parameter(
            torch.zeros(config.active_horizons, config.layers, 1)
        )

    def forward(self, anchor_scores: Tensor, mixture_logits: Tensor) -> Tensor:
        anchor_centered = anchor_scores - anchor_scores.mean(dim=-1, keepdim=True)
        return anchor_scores + self.gate[None] * (mixture_logits - anchor_centered)


__all__ = [
    "BranchMixtureOutput",
    "BranchMixtureResidual",
    "ExactBranchMixture",
    "HybridRouterScoreHead",
    "HybridScoreOutput",
    "exact_projected_marginals",
]
