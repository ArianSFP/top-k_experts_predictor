"""Gradient-open, parent-exact axial DeltaRoute translator."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .axial_route import AxialRouteTrajectory
from .route_dynamics import (
    DeltaRouteConfig,
    DeltaRouteTrajectoryOutput,
    predicted_route_distribution,
)


class GradientOpenAxialRouteTrajectory(AxialRouteTrajectory):
    """Preserve the parent exactly without starving the feature encoder.

    A frozen reference of each randomly initialised output map is subtracted
    from the live map.  Reference features are stop-gradient.  Consequently
    the residual is bit-identically zero initially, while gradients reach the
    layer-specific channel projections and axial trajectory blocks on the
    first optimizer step.
    """

    def __init__(
        self,
        config: DeltaRouteConfig,
        input_basis: Tensor,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
        *,
        raw_rank: int = 32,
        visible_rank: int = 32,
        route_width: int = 64,
        axial_blocks: int = 2,
    ) -> None:
        super().__init__(
            config, input_basis, expert_keys, centered_bias, rank_mask,
            raw_rank=raw_rank, visible_rank=visible_rank,
            route_width=route_width, axial_blocks=axial_blocks,
        )
        nn.init.normal_(self.query_output.weight, std=0.02)
        nn.init.zeros_(self.query_output.bias)
        nn.init.normal_(self.query_up, std=0.02)
        nn.init.normal_(self.free_basis, std=0.02)
        self.register_buffer(
            "reference_query_weight", self.query_output.weight.detach().clone()
        )
        self.register_buffer(
            "reference_query_bias", self.query_output.bias.detach().clone()
        )
        self.register_buffer(
            "reference_query_up", self.query_up.detach().clone()
        )
        self.register_buffer(
            "reference_free_basis", self.free_basis.detach().clone()
        )

    @staticmethod
    def _anchored_linear(
        live: nn.Linear, features: Tensor, weight: Tensor, bias: Tensor
    ) -> Tensor:
        value = live(features)
        reference = F.linear(
            features.detach(), weight.to(features.dtype), bias.to(features.dtype)
        )
        return value - reference

    def forward(
        self,
        *,
        parent_queries: Tensor,
        parent_scores: Tensor,
        context_states: Tensor,
        fused: Tensor,
        post_ffn: Tensor,
        router_input: Tensor,
        vocabulary: Tensor,
        token: Tensor,
        router_logits: Tensor,
        metadata: Tensor,
        parents: Tensor,
        available: Tensor,
    ) -> DeltaRouteTrajectoryOutput:
        config = self.config
        batch, horizons, layers, nodes, rank = parent_queries.shape
        if (horizons, layers, nodes, rank) != (
            config.horizons, config.layers, config.nodes, config.router_rank
        ) or parent_scores.shape != (
            batch, horizons, layers, nodes, config.experts
        ):
            raise ValueError("parent route geometry differs from gradient-open config")
        context, pooled, path = self.causal_context(
            context_states=context_states, fused=fused, post_ffn=post_ffn,
            router_input=router_input, vocabulary=vocabulary, token=token,
            router_logits=router_logits, metadata=metadata, parents=parents,
            available=available,
        )
        context = self._axial_context(
            context, parent_queries, parent_scores, available
        )
        hidden = self.query_hidden(context)
        shared = self._anchored_linear(
            self.query_output, hidden,
            self.reference_query_weight, self.reference_query_bias,
        )
        low = torch.nn.functional.silu(torch.einsum(
            "bhlnd,lda->bhlna", context, self.query_down
        ))
        live_adapter = torch.einsum(
            "bhlna,lar->bhlnr", low, self.query_up
        )
        reference_adapter = torch.einsum(
            "bhlna,lar->bhlnr", low.detach(),
            self.reference_query_up.to(low.dtype),
        )
        delta_query = shared + live_adapter - reference_adapter
        delta_query = delta_query * self.rank_mask[None, None, :, None]
        queries = parent_queries.float() + delta_query.float()
        geometry = torch.einsum(
            "bhlnr,ler->bhlne", delta_query.float(), self.expert_keys.float()
        )
        coefficients = self.free_coefficients(context)
        live_free = torch.einsum(
            "bhlna,lea->bhlne", coefficients.float(), self.free_basis.float()
        )
        reference_free = torch.einsum(
            "bhlna,lea->bhlne", coefficients.detach().float(),
            self.reference_free_basis.float(),
        )
        free_delta = live_free - reference_free
        scores = parent_scores.float() + geometry + free_delta
        route = predicted_route_distribution(scores, k=config.exact_k)
        return DeltaRouteTrajectoryOutput(
            queries, scores, route.selected_ids, route.selected_weights,
            pooled, path,
        )


__all__ = ["GradientOpenAxialRouteTrajectory"]
