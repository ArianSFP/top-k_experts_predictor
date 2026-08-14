"""Branch translator conditioned on exact causal target state and route history."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .gradient_open_axial import GradientOpenAxialRouteTrajectory
from .route_dynamics import (
    DeltaRouteConfig,
    DeltaRouteTrajectoryOutput,
    predicted_route_distribution,
)


class CausalStateAxialRouteTrajectory(GradientOpenAxialRouteTrajectory):
    """Decode each branch after target-state × branch-path interaction.

    The parent and earlier axial diagnostics condition branches on a compact
    target context.  This variant exposes the exact completed token's four
    residual streams, router coordinates, and eight executed route histories
    to every branch/layer cell before axial refinement.  The inherited frozen
    reference maps keep the complete output bit-identical to the selected
    budget-16 parent at initialization while leaving all new paths gradient
    open.
    """

    STATE_ROLES = 4
    HISTORY = 8

    def __init__(
        self,
        config: DeltaRouteConfig,
        input_basis: Tensor,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
        *,
        raw_rank: int = 24,
        visible_rank: int = 24,
        route_width: int = 64,
        axial_blocks: int = 2,
    ) -> None:
        super().__init__(
            config, input_basis, expert_keys, centered_bias, rank_mask,
            raw_rank=raw_rank, visible_rank=visible_rank,
            route_width=route_width, axial_blocks=axial_blocks,
        )
        self.state_norm = nn.ModuleList(
            nn.RMSNorm(config.raw_width) for _ in range(self.STATE_ROLES)
        )
        self.state_projection = nn.ModuleList(
            nn.Linear(config.raw_width, config.latent_width)
            for _ in range(self.STATE_ROLES)
        )
        self.state_mixture = nn.Parameter(torch.zeros(
            config.layers, self.STATE_ROLES
        ))
        self.current_query = nn.Linear(config.router_rank, config.latent_width)
        self.history_logits = nn.Linear(config.experts, config.latent_width)
        self.history_route_embedding = nn.Parameter(torch.empty(
            config.layers, config.experts, route_width
        ))
        self.history_route_projection = nn.Linear(
            route_width, config.latent_width
        )
        self.history_lag_logits = nn.Parameter(torch.zeros(
            config.horizons, self.HISTORY
        ))
        self.current_norm = nn.RMSNorm(config.latent_width)
        nn.init.normal_(self.history_route_embedding, std=0.02)

    def _current_context(
        self,
        *,
        current_states: Tensor,
        current_queries: Tensor,
        history_logits: Tensor,
        history_selected_ids: Tensor,
        history_selected_weights: Tensor,
    ) -> Tensor:
        config = self.config
        batch = current_states.shape[0]
        if current_states.shape != (
            batch, self.STATE_ROLES, config.layers, config.raw_width
        ):
            raise ValueError("current target states must be [B,4,L,D]")
        if current_queries.shape != (
            batch, config.layers, config.router_rank
        ):
            raise ValueError("current target queries must be [B,L,R]")
        if history_logits.shape != (
            batch, self.HISTORY, config.layers, config.experts
        ):
            raise ValueError("route history logits must be [B,8,L,E]")
        expected = (
            batch, self.HISTORY, config.layers, config.exact_k
        )
        if history_selected_ids.shape != expected or history_selected_weights.shape != expected:
            raise ValueError("route history ID/weight geometry is invalid")

        role_features = torch.stack([
            projection(norm(current_states[:, role]))
            for role, (norm, projection) in enumerate(zip(
                self.state_norm, self.state_projection, strict=True
            ))
        ], dim=-2)
        role_weights = torch.softmax(self.state_mixture.float(), dim=-1)
        state = torch.einsum(
            "lq,blqw->blw", role_weights, role_features.float()
        )

        layer = torch.arange(
            config.layers, device=current_states.device
        )[None, None, :, None]
        route = self.history_route_embedding[
            layer, history_selected_ids.long()
        ]
        weights = history_selected_weights.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        route = self.history_route_projection(
            (route * weights[..., None]).sum(-2)
        )
        per_lag = route + self.history_logits(history_logits.float())
        lag_weights = torch.softmax(self.history_lag_logits.float(), dim=-1)
        history = torch.einsum("ht,btlw->bhlw", lag_weights, per_lag)
        result = (
            state[:, None] + self.current_query(current_queries.float())[:, None]
            + history
        )
        return self.current_norm(result)

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
        current_states: Tensor,
        current_queries: Tensor,
        history_logits: Tensor,
        history_selected_ids: Tensor,
        history_selected_weights: Tensor,
    ) -> DeltaRouteTrajectoryOutput:
        config = self.config
        batch, horizons, layers, nodes, rank = parent_queries.shape
        if (horizons, layers, nodes, rank) != (
            config.horizons, config.layers, config.nodes, config.router_rank
        ) or parent_scores.shape != (
            batch, horizons, layers, nodes, config.experts
        ):
            raise ValueError("parent route geometry differs from causal axial config")
        context, pooled, path = self.causal_context(
            context_states=context_states, fused=fused, post_ffn=post_ffn,
            router_input=router_input, vocabulary=vocabulary, token=token,
            router_logits=router_logits, metadata=metadata, parents=parents,
            available=available,
        )
        current = self._current_context(
            current_states=current_states, current_queries=current_queries,
            history_logits=history_logits,
            history_selected_ids=history_selected_ids,
            history_selected_weights=history_selected_weights,
        )
        context = context + current[..., None, :].to(context.dtype)
        context = self._axial_context(
            context, parent_queries, parent_scores, available
        )

        hidden = self.query_hidden(context)
        shared = self._anchored_linear(
            self.query_output, hidden,
            self.reference_query_weight, self.reference_query_bias,
        )
        low = F.silu(torch.einsum(
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


__all__ = ["CausalStateAxialRouteTrajectory"]
