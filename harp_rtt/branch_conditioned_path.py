"""Branch-state-conditioned layerwise route trajectory.

The factual path surrogate is deliberately retained as the exact epoch-zero
baseline.  This module adds the causal state emitted by the selected MTP tree
node only after target-layer conditioning, avoiding the early all-layer
bottleneck of the older DeltaTree translator.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .exact_k import stable_topk
from .path_route_surrogate import (
    PathRouteSurrogateConfig,
    PathRouteSurrogateOutput,
    TokenConditionedRouteSurrogate,
)
from .path_route_trajectory import LayerwiseTokenRouteSurrogate


class BranchConditionedLayerwiseRouteSurrogate(LayerwiseTokenRouteSurrogate):
    """Roll target routes from high-rank, branch-specific causal evidence.

    Four full-width MTP state channels and the node vocabulary state are
    projected with a different low-rank map for every target layer.  Router,
    route and metadata evidence remain distinct channel tokens.  A target
    layer/horizon query pools these tokens before the self-conditioned route
    scan.  The scalar residual gate is initialized to zero, so loading a
    factual/branch-adapted trajectory checkpoint reproduces it exactly.
    """

    BRANCH_STATE_ROLES = 4
    BRANCH_CHANNELS = 8

    def __init__(
        self,
        config: PathRouteSurrogateConfig,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
        *,
        branch_state_rank: int = 32,
        branch_vocab_rank: int = 16,
    ) -> None:
        super().__init__(config, expert_keys, centered_bias, rank_mask)
        if min(branch_state_rank, branch_vocab_rank) < 1:
            raise ValueError("branch projection ranks must be positive")
        self.branch_state_rank = branch_state_rank
        self.branch_vocab_rank = branch_vocab_rank

        # These maps are target-layer specific: a single early compression is
        # not required to preserve directions for all forty router bases.
        self.branch_state_down = nn.Parameter(torch.empty(
            self.BRANCH_STATE_ROLES, config.layers, config.token_width,
            branch_state_rank,
        ))
        self.branch_state_up = nn.Parameter(torch.empty(
            self.BRANCH_STATE_ROLES, config.layers, branch_state_rank,
            config.width,
        ))
        self.branch_vocab_down = nn.Parameter(torch.empty(
            config.layers, config.token_width, branch_vocab_rank,
        ))
        self.branch_vocab_up = nn.Parameter(torch.empty(
            config.layers, branch_vocab_rank, config.width,
        ))
        self.branch_router = nn.Linear(config.experts, config.width)
        self.branch_expert_embedding = nn.Parameter(torch.empty(
            config.experts, config.route_width
        ))
        self.branch_route = nn.Linear(config.route_width, config.width)
        self.branch_statistics = nn.Sequential(
            nn.Linear(14, config.width), nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.branch_channel_embedding = nn.Embedding(
            self.BRANCH_CHANNELS, config.width
        )
        self.branch_key = nn.Linear(config.width, config.width, bias=False)
        self.branch_value = nn.Linear(config.width, config.width, bias=False)
        self.branch_query = nn.Linear(config.width, config.width, bias=False)
        self.branch_output = nn.Sequential(
            nn.RMSNorm(config.width), nn.Linear(config.width, config.width)
        )
        self.branch_gate = nn.Parameter(torch.zeros(config.layers))
        nn.init.normal_(self.branch_state_down, std=0.02)
        nn.init.normal_(self.branch_state_up, std=0.02)
        nn.init.normal_(self.branch_vocab_down, std=0.02)
        nn.init.normal_(self.branch_vocab_up, std=0.02)
        nn.init.normal_(self.branch_expert_embedding, std=0.02)

    def branch_parameters(self) -> list[nn.Parameter]:
        names = (
            "branch_state_down", "branch_state_up", "branch_vocab_down",
            "branch_vocab_up", "branch_router", "branch_expert_embedding",
            "branch_route", "branch_statistics", "branch_channel_embedding",
            "branch_key", "branch_value", "branch_query", "branch_output",
            "branch_gate",
        )
        result: list[nn.Parameter] = []
        for name in names:
            value = getattr(self, name)
            if isinstance(value, nn.Parameter):
                result.append(value)
            else:
                result.extend(value.parameters())
        return result

    def _branch_tokens(
        self,
        *,
        branch_states: Tensor,
        branch_router_logits: Tensor,
        branch_selected_ids: Tensor,
        branch_selected_weights: Tensor,
        branch_vocab_embedding: Tensor,
        branch_vocab_statistics: Tensor,
        branch_scalars: Tensor,
    ) -> Tensor:
        config = self.config
        batch = branch_states.shape[0]
        if branch_states.shape != (
            batch, self.BRANCH_STATE_ROLES, config.token_width
        ):
            raise ValueError("branch states must be [B,4,D]")
        if branch_router_logits.shape != (batch, config.experts):
            raise ValueError("branch router logits must be [B,E]")
        if branch_selected_ids.shape != (batch, config.exact_k):
            raise ValueError("branch selected IDs must be [B,K]")
        if branch_selected_weights.shape != (batch, config.exact_k):
            raise ValueError("branch selected weights must be [B,K]")
        if branch_vocab_embedding.shape != (batch, config.token_width):
            raise ValueError("branch vocabulary embedding must be [B,D]")
        if branch_vocab_statistics.shape != (batch, 6) or branch_scalars.shape != (batch, 8):
            raise ValueError("branch statistics/scalars have invalid geometry")

        raw_low = torch.einsum(
            "bqd,qlda->bqla", branch_states.float(),
            self.branch_state_down.float(),
        )
        raw = torch.einsum(
            "bqla,qlaw->blqw", torch.nn.functional.silu(raw_low),
            self.branch_state_up.float(),
        )
        vocab_low = torch.einsum(
            "bd,lda->bla", branch_vocab_embedding.float(),
            self.branch_vocab_down.float(),
        )
        vocab = torch.einsum(
            "bla,law->blw", torch.nn.functional.silu(vocab_low),
            self.branch_vocab_up.float(),
        )
        router = self.branch_router(branch_router_logits.float())[:, None].expand(
            batch, config.layers, config.width
        )
        weights = branch_selected_weights.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        route = self.branch_route((
            self.branch_expert_embedding[branch_selected_ids.long()]
            * weights[..., None]
        ).sum(-2))[:, None].expand(batch, config.layers, config.width)
        statistics = self.branch_statistics(torch.cat(
            (branch_vocab_statistics.float(), branch_scalars.float()), dim=-1
        ))[:, None].expand(batch, config.layers, config.width)
        # Four raw state tokens plus vocabulary, router, route, and metadata.
        tokens = torch.cat((
            raw, vocab[..., None, :], router[..., None, :],
            route[..., None, :], statistics[..., None, :],
        ), dim=-2)
        return tokens + self.branch_channel_embedding.weight[None, None]

    def _condition(self, base: Tensor, tokens: Tensor) -> Tensor:
        config = self.config
        heads = config.attention_heads
        head_width = config.width // heads
        query = self.branch_query(base.float()).reshape(
            base.shape[0], config.horizons, config.layers, heads, head_width
        )
        key = self.branch_key(tokens).reshape(
            base.shape[0], config.layers, self.BRANCH_CHANNELS,
            heads, head_width,
        )
        value = self.branch_value(tokens).reshape_as(key)
        attention = torch.einsum(
            "bhlad,blcad->bhlac", query, key
        ) / math.sqrt(head_width)
        attention = torch.softmax(attention.float(), dim=-1).to(value.dtype)
        pooled = torch.einsum(
            "bhlac,blcad->bhlad", attention, value
        ).flatten(-2)
        residual = self.branch_output(pooled)
        gate = torch.tanh(self.branch_gate.float())[None, None, :, None]
        return base + gate.to(residual.dtype) * residual

    def _rollout(self, base: Tensor, path_states: Tensor) -> PathRouteSurrogateOutput:
        config = self.config
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
            queries.append(query); scores.append(score); hidden.append(cell)
            if layer + 1 < config.layers:
                effect = self._route_effect(score, layer)
                feedback = self.feedback_norm(
                    cell + self.feedback_query(query.float())
                    + self.feedback_route(effect)
                )
                state = self.trajectory_cell(
                    feedback.reshape(-1, config.width),
                    state.reshape(-1, config.width),
                ).reshape(base.shape[0], config.horizons, config.width)
        query_tensor = torch.stack(queries, dim=2)
        score_tensor = torch.stack(scores, dim=2)
        return PathRouteSurrogateOutput(
            queries=query_tensor, scores=score_tensor,
            selected_ids=stable_topk(score_tensor, config.exact_k),
            hidden=torch.stack(hidden, dim=2), path_states=path_states,
        )

    def forward(
        self,
        *,
        state_coordinates: Tensor,
        current_queries: Tensor,
        history_selected_ids: Tensor,
        history_selected_weights: Tensor,
        path_token_embeddings: Tensor,
        branch_states: Tensor,
        branch_router_logits: Tensor,
        branch_selected_ids: Tensor,
        branch_selected_weights: Tensor,
        branch_vocab_embedding: Tensor,
        branch_vocab_statistics: Tensor,
        branch_scalars: Tensor,
    ) -> PathRouteSurrogateOutput:
        # Call the non-recurrent ancestor explicitly to obtain the same causal
        # base tensor used by the factual trajectory initializer.
        encoded = TokenConditionedRouteSurrogate.forward(
            self,
            state_coordinates=state_coordinates,
            current_queries=current_queries,
            history_selected_ids=history_selected_ids,
            history_selected_weights=history_selected_weights,
            path_token_embeddings=path_token_embeddings,
        )
        tokens = self._branch_tokens(
            branch_states=branch_states,
            branch_router_logits=branch_router_logits,
            branch_selected_ids=branch_selected_ids,
            branch_selected_weights=branch_selected_weights,
            branch_vocab_embedding=branch_vocab_embedding,
            branch_vocab_statistics=branch_vocab_statistics,
            branch_scalars=branch_scalars,
        )
        return self._rollout(
            self._condition(encoded.hidden, tokens), encoded.path_states
        )


__all__ = ["BranchConditionedLayerwiseRouteSurrogate"]
