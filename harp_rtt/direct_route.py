"""Direct layer-conditioned branch translation for HARP-DeltaRoute v4."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .route_dynamics import (
    DeltaRouteConfig,
    DeltaRouteTrajectoryOutput,
    LayerConditionedChannelPool,
    predicted_route_distribution,
)


class DirectDeltaRouteTrajectory(nn.Module):
    """Translate causal branch channels independently at every target layer.

    This is the high-capacity control against recurrent route rollout.  The
    seven raw MTP channels remain separate until a target-layer/horizon query
    pools them.  Both geometry and free-score residuals are exactly zero at
    initialization, so the selected budget-16 parent is reproduced without a
    warm-start discontinuity.
    """

    def __init__(
        self,
        config: DeltaRouteConfig,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
    ) -> None:
        super().__init__()
        config.validate()
        if expert_keys.shape != (config.layers, config.experts, config.router_rank):
            raise ValueError("expert keys disagree with direct-route config")
        if centered_bias.shape != (config.layers, config.experts):
            raise ValueError("centered bias disagrees with direct-route config")
        if rank_mask.shape != (config.layers, config.router_rank):
            raise ValueError("rank mask disagrees with direct-route config")
        self.config = config
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.register_buffer("centered_bias", centered_bias.detach().float().clone())
        self.register_buffer("rank_mask", rank_mask.detach().bool().clone())
        self.channels = LayerConditionedChannelPool(config)
        self.context = nn.Linear(config.latent_width, config.latent_width)
        self.query_hidden = nn.Sequential(
            nn.RMSNorm(config.latent_width),
            nn.Linear(config.latent_width, config.transition_width),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.query_output = nn.Linear(
            config.transition_width, config.router_rank
        )
        self.query_down = nn.Parameter(torch.empty(
            config.layers, config.latent_width, config.layer_adapter_rank
        ))
        self.query_up = nn.Parameter(torch.zeros(
            config.layers, config.layer_adapter_rank, config.router_rank
        ))
        self.free_coefficients = nn.Linear(
            config.latent_width, config.free_rank
        )
        self.free_basis = nn.Parameter(torch.zeros(
            config.layers, config.experts, config.free_rank
        ))
        nn.init.zeros_(self.query_output.weight)
        nn.init.zeros_(self.query_output.bias)
        nn.init.normal_(self.query_down, std=0.02)

    def causal_context(
        self,
        *,
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
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = self.config
        batch = fused.shape[0]
        if context_states.shape != (
            batch, config.horizons, config.layers, config.latent_width
        ):
            raise ValueError("direct-route context states have invalid geometry")
        pooled, path = self.channels(
            fused=fused, post_ffn=post_ffn, router_input=router_input,
            vocabulary=vocabulary, token=token, router_logits=router_logits,
            metadata=metadata, parents=parents, available=available,
        )
        return pooled + self.context(context_states)[:, :, :, None], pooled, path

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
        ):
            raise ValueError("parent query geometry differs from direct-route config")
        if parent_scores.shape != (
            batch, horizons, layers, nodes, config.experts
        ):
            raise ValueError("parent score geometry differs from direct-route queries")
        context, pooled, path = self.causal_context(
            context_states=context_states, fused=fused, post_ffn=post_ffn,
            router_input=router_input, vocabulary=vocabulary, token=token,
            router_logits=router_logits, metadata=metadata, parents=parents,
            available=available,
        )
        shared = self.query_output(self.query_hidden(context))
        low = torch.einsum(
            "bhlnd,lda->bhlna", context, self.query_down
        )
        delta_query = shared + torch.einsum(
            "bhlna,lar->bhlnr", torch.nn.functional.silu(low), self.query_up
        )
        delta_query = delta_query * self.rank_mask[None, None, :, None]
        queries = parent_queries.float() + delta_query.float()
        geometry = torch.einsum(
            "bhlnr,ler->bhlne", delta_query.float(), self.expert_keys.float()
        )
        coefficients = self.free_coefficients(context)
        free = torch.einsum(
            "bhlna,lea->bhlne", coefficients.float(), self.free_basis.float()
        )
        scores = parent_scores.float() + geometry + free
        route = predicted_route_distribution(scores, k=config.exact_k)
        return DeltaRouteTrajectoryOutput(
            queries, scores, route.selected_ids, route.selected_weights,
            pooled, path,
        )


__all__ = ["DirectDeltaRouteTrajectory"]
