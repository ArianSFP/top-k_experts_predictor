"""Causal distillation of route-relevant target content from an MTP tree."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CausalContentOutput:
    content_latent: Tensor
    branch_weights: Tensor
    scores: Tensor


class CausalContentStudent(nn.Module):
    """Pool causal tree evidence and add a safe residual to parent scores.

    Captured nodes and explicit OTHER compete under the frozen path posterior.
    Unavailable nodes never acquire mass.  The score residual is zero-gated so
    epoch zero exactly reproduces the supplied parent scores.
    """

    def __init__(
        self,
        *,
        horizons: int,
        layers: int,
        nodes: int,
        experts: int,
        width: int = 256,
        score_rank: int = 16,
    ) -> None:
        super().__init__()
        if min(horizons, layers, nodes, experts, width, score_rank) < 1:
            raise ValueError("causal content student dimensions must be positive")
        self.horizons = horizons
        self.layers = layers
        self.nodes = nodes
        self.experts = experts
        self.width = width
        self.node_key = nn.Linear(width, width, bias=False)
        self.node_value = nn.Linear(width, width)
        self.other_key = nn.Linear(width, width, bias=False)
        self.other_value = nn.Linear(width, width)
        self.attention_query = nn.Parameter(torch.empty(horizons, layers, width))
        hidden = max(1, width // 2)
        self.posterior_correction = nn.Sequential(
            nn.RMSNorm(width), nn.Linear(width, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.content_head = nn.Sequential(
            nn.RMSNorm(width), nn.Linear(width, width * 2), nn.SiLU(),
            nn.Linear(width * 2, width),
        )
        self.score_coefficients = nn.Linear(width, score_rank)
        self.score_basis = nn.Parameter(torch.empty(layers, experts, score_rank))
        self.score_gate = nn.Parameter(torch.zeros(horizons, layers, 1))
        self.reliability_gate = nn.Parameter(torch.zeros(horizons, layers, 1))
        nn.init.normal_(self.attention_query, std=0.02)
        nn.init.normal_(self.score_basis, std=0.02)
        nn.init.zeros_(self.posterior_correction[-1].weight)
        nn.init.zeros_(self.posterior_correction[-1].bias)

    def forward(
        self,
        *,
        node_context: Tensor,
        other_context: Tensor,
        posterior: Tensor,
        node_mask: Tensor,
        parent_scores: Tensor,
    ) -> CausalContentOutput:
        batch = node_context.shape[0]
        if node_context.shape != (
            batch, self.horizons, self.layers, self.nodes, self.width
        ):
            raise ValueError("causal content node context has invalid geometry")
        if other_context.shape != (
            batch, self.horizons, self.layers, self.width
        ):
            raise ValueError("causal content OTHER context has invalid geometry")
        if posterior.shape != (batch, self.horizons, self.nodes + 1):
            raise ValueError("causal content posterior has invalid geometry")
        if node_mask.shape != (batch, self.horizons, self.nodes):
            raise ValueError("causal content node mask has invalid geometry")
        if parent_scores.shape != (
            batch, self.horizons, self.layers, self.experts
        ):
            raise ValueError("causal content parent scores have invalid geometry")
        if not torch.isfinite(posterior).all() or bool((posterior < 0).any()):
            raise ValueError("causal content posterior must be finite and non-negative")

        query = self.attention_query[None, ..., None, :]
        node_logits = (
            self.node_key(node_context) * query
        ).sum(-1) / math.sqrt(self.width)
        correction = self.posterior_correction(node_context).squeeze(-1)
        node_logits = node_logits + self.reliability_gate[None] * correction
        node_logits = node_logits + posterior[..., :-1].clamp_min(1e-30).log()[:, :, None]
        node_logits = node_logits.masked_fill(~node_mask[:, :, None], -torch.inf)

        other_logits = (
            self.other_key(other_context) * self.attention_query[None]
        ).sum(-1) / math.sqrt(self.width)
        other_logits = other_logits + posterior[..., -1].clamp_min(1e-30).log()[:, :, None]
        weights = torch.softmax(
            torch.cat((node_logits, other_logits[..., None]), dim=-1).float(),
            dim=-1,
        )
        if not torch.isfinite(weights).all():
            raise ValueError("causal content branch weights are non-finite")

        values = self.node_value(node_context)
        other = self.other_value(other_context)
        pooled = (
            weights[..., :-1, None].to(values.dtype) * values
        ).sum(-2) + weights[..., -1, None].to(other.dtype) * other
        latent = self.content_head(pooled)
        coefficients = self.score_coefficients(latent)
        delta = torch.einsum(
            "bhla,lea->bhle", coefficients.float(), self.score_basis.float()
        )
        scores = parent_scores.float() + self.score_gate[None] * delta
        return CausalContentOutput(latent, weights, scores)


__all__ = ["CausalContentOutput", "CausalContentStudent"]
