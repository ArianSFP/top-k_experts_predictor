"""Whole-trajectory refinement for HARP-DeltaRoute v4.

The direct heads predict each target layer independently.  This module keeps
their causal, layer-specific source fusion, but treats the forty parent route
states for a branch as one structured object.  It first maps every parent
query through the frozen router-input basis into a small common-hidden-space
sketch, adds a layer/expert route message, and then refines the complete layer
axis with self-attention.  No target label is accepted by the serving API.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .exact_k import stable_topk
from .layer_specific_direct import LayerSpecificDirectDeltaRouteTrajectory
from .route_dynamics import (
    DeltaRouteConfig,
    DeltaRouteTrajectoryOutput,
    predicted_route_distribution,
)


class AxialRouteTrajectory(LayerSpecificDirectDeltaRouteTrajectory):
    """Refine each branch's complete parent router trajectory jointly."""

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
            config, expert_keys, centered_bias, rank_mask, raw_rank=raw_rank
        )
        if input_basis.shape != (
            config.layers, config.raw_width, config.router_rank
        ):
            raise ValueError("router input basis disagrees with axial config")
        if min(visible_rank, route_width, axial_blocks) < 1:
            raise ValueError("axial trajectory dimensions must be positive")
        self.visible_rank = visible_rank
        self.route_width = route_width
        self.register_buffer(
            "input_basis", input_basis.detach().float().clone()
        )

        # q_l coordinates live in a different SVD basis at every target layer.
        # Project V_l q_l into one learned sketch of the common hidden space
        # before asking the layer-axis model to compare adjacent states.
        self.visible_hidden_down = nn.Parameter(torch.empty(
            config.raw_width, visible_rank
        ))
        self.visible_up = nn.Linear(visible_rank, config.latent_width)

        # Expert IDs are layer-local.  Their embeddings are therefore also
        # layer-local before being added to the common trajectory token.
        self.route_embedding = nn.Parameter(torch.empty(
            config.layers, config.experts, route_width
        ))
        self.route_up = nn.Linear(route_width, config.latent_width)
        self.axial_layer = nn.Embedding(config.layers, config.latent_width)
        self.axial_horizon = nn.Embedding(config.horizons, config.latent_width)
        layer = nn.TransformerEncoderLayer(
            d_model=config.latent_width,
            nhead=config.attention_heads,
            dim_feedforward=config.transition_width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.axial = nn.TransformerEncoder(
            layer, num_layers=axial_blocks,
            norm=nn.RMSNorm(config.latent_width),
        )
        nn.init.normal_(self.visible_hidden_down, std=0.02)
        nn.init.normal_(self.route_embedding, std=0.02)

    def _parent_visible(self, parent_queries: Tensor) -> Tensor:
        basis_sketch = torch.einsum(
            "ldr,da->lra",
            self.input_basis.float(), self.visible_hidden_down.float(),
        )
        visible = torch.einsum(
            "bhlnr,lra->bhlna", parent_queries.float(), basis_sketch
        )
        return self.visible_up(visible)

    def _parent_route(self, parent_scores: Tensor) -> Tensor:
        ids = stable_topk(parent_scores.float(), self.config.exact_k)
        probabilities = torch.softmax(parent_scores.float(), dim=-1)
        weights = probabilities.gather(-1, ids)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        layer = torch.arange(
            self.config.layers, device=ids.device
        )[None, None, :, None, None]
        embedded = self.route_embedding[layer, ids.long()]
        message = (embedded * weights[..., None]).sum(-2)
        return self.route_up(message)

    def _axial_context(
        self,
        context: Tensor,
        parent_queries: Tensor,
        parent_scores: Tensor,
        available: Tensor,
    ) -> Tensor:
        config = self.config
        batch = context.shape[0]
        token = (
            context
            + self._parent_visible(parent_queries)
            + self._parent_route(parent_scores)
            + self.axial_layer.weight[None, None, :, None]
            + self.axial_horizon.weight[None, :, None, None]
        )
        sequence = token.permute(0, 1, 3, 2, 4).reshape(
            batch * config.horizons * config.nodes,
            config.layers, config.latent_width,
        )
        sequence = self.axial(sequence)
        result = sequence.reshape(
            batch, config.horizons, config.nodes,
            config.layers, config.latent_width,
        ).permute(0, 1, 3, 2, 4)
        return result * available[:, None, None, :, None].to(result.dtype)

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
            raise ValueError("parent query geometry differs from axial config")
        if parent_scores.shape != (
            batch, horizons, layers, nodes, config.experts
        ):
            raise ValueError("parent score geometry differs from axial queries")
        context, pooled, path = self.causal_context(
            context_states=context_states, fused=fused, post_ffn=post_ffn,
            router_input=router_input, vocabulary=vocabulary, token=token,
            router_logits=router_logits, metadata=metadata, parents=parents,
            available=available,
        )
        context = self._axial_context(
            context, parent_queries, parent_scores, available
        )
        shared = self.query_output(self.query_hidden(context))
        low = torch.einsum("bhlnd,lda->bhlna", context, self.query_down)
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


__all__ = ["AxialRouteTrajectory"]
