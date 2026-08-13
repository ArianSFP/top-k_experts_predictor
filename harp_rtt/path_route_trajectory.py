"""Self-conditioned layerwise router trajectory over token-path context."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .exact_k import stable_topk
from .path_route_surrogate import (
    PathRouteSurrogateConfig,
    PathRouteSurrogateOutput,
    TokenConditionedRouteSurrogate,
)


class LayerwiseTokenRouteSurrogate(TokenConditionedRouteSurrogate):
    """Decode target layers recurrently using the model's own predicted route.

    The inherited encoder produces a causal token-path/layer context. This
    decoder then rolls through the target layers. At every step, its router
    query and exact top-k route produce an execution-style expert-effect
    message which conditions the next layer. A straight-through dense route
    distribution supplies gradients while the forward pass uses only the
    selected experts, matching serving-time self-conditioning.
    """

    def __init__(
        self,
        config: PathRouteSurrogateConfig,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
    ) -> None:
        super().__init__(config, expert_keys, centered_bias, rank_mask)
        self.expert_effect_embedding = nn.Parameter(torch.empty(
            config.layers, config.experts, config.route_width
        ))
        self.feedback_query = nn.Linear(config.router_rank, config.width)
        self.feedback_route = nn.Linear(config.route_width, config.width)
        self.feedback_norm = nn.RMSNorm(config.width)
        self.rollout_norm = nn.RMSNorm(config.width)
        self.trajectory_cell = nn.GRUCell(config.width, config.width)
        nn.init.normal_(self.expert_effect_embedding, std=0.02)

    def _route_effect(self, scores: Tensor, layer: int) -> Tensor:
        config = self.config
        top_ids = stable_topk(scores, config.exact_k)
        top_scores = scores.gather(-1, top_ids)
        selected_weights = torch.softmax(top_scores.float(), dim=-1)
        hard = torch.zeros_like(scores.float()).scatter(
            -1, top_ids, selected_weights
        )
        dense = torch.softmax(scores.float(), dim=-1)
        route_weights = hard + dense - dense.detach()
        return torch.einsum(
            "bhe,er->bhr",
            route_weights,
            self.expert_effect_embedding[layer].float(),
        )

    def forward(
        self,
        *,
        state_coordinates: Tensor,
        current_queries: Tensor,
        history_selected_ids: Tensor,
        history_selected_weights: Tensor,
        path_token_embeddings: Tensor,
    ) -> PathRouteSurrogateOutput:
        encoded = super().forward(
            state_coordinates=state_coordinates,
            current_queries=current_queries,
            history_selected_ids=history_selected_ids,
            history_selected_weights=history_selected_weights,
            path_token_embeddings=path_token_embeddings,
        )
        config = self.config
        base = encoded.hidden
        state = torch.zeros(
            base.shape[0], config.horizons, config.width,
            device=base.device, dtype=base.dtype,
        )
        queries: list[Tensor] = []
        scores: list[Tensor] = []
        hidden: list[Tensor] = []
        for layer in range(config.layers):
            cell = self.rollout_norm(base[:, :, layer] + state)
            transformed = self.query_hidden(cell)
            query = self.query_output(transformed)
            low = torch.nn.functional.silu(torch.einsum(
                "bhw,wa->bha", cell, self.query_down[layer]
            ))
            query = query + torch.einsum(
                "bha,ar->bhr", low, self.query_up[layer]
            )
            query = query * self.rank_mask[layer][None, None]
            score = torch.einsum(
                "bhr,er->bhe", query.float(), self.expert_keys[layer]
            ) + self.centered_bias[layer][None, None]
            score = score + torch.einsum(
                "bha,ea->bhe", self.free_coefficients(cell).float(),
                self.free_basis[layer].float(),
            )
            queries.append(query)
            scores.append(score)
            hidden.append(cell)
            if layer + 1 < config.layers:
                effect = self._route_effect(score, layer)
                feedback = self.feedback_norm(
                    cell
                    + self.feedback_query(query.float())
                    + self.feedback_route(effect)
                )
                state = self.trajectory_cell(
                    feedback.reshape(-1, config.width),
                    state.reshape(-1, config.width),
                ).reshape(base.shape[0], config.horizons, config.width)
        query_tensor = torch.stack(queries, dim=2)
        score_tensor = torch.stack(scores, dim=2)
        return PathRouteSurrogateOutput(
            queries=query_tensor,
            scores=score_tensor,
            selected_ids=stable_topk(score_tensor, config.exact_k),
            hidden=torch.stack(hidden, dim=2),
            path_states=encoded.path_states,
        )


__all__ = ["LayerwiseTokenRouteSurrogate"]
