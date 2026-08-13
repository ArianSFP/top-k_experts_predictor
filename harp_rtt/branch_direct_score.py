"""High-rank branch endpoint decoder with direct expert-boundary scores."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .exact_k import stable_topk
from .path_route_surrogate import PathRouteSurrogateConfig, PathRouteSurrogateOutput
from .path_route_trajectory import LayerwiseTokenRouteSurrogate


class DirectHighRankBranchRouteSurrogate(LayerwiseTokenRouteSurrogate):
    """Correct a frozen trajectory directly from full MTP endpoint content.

    This is intentionally not another recurrent wrapper.  The frozen parent
    first emits its complete trajectory.  Each target layer then receives its
    own rank-64 view of all four 2,048-dimensional branch states and predicts
    both a router-coordinate correction and an unconstrained expert-boundary
    correction.  Zero output matrices preserve the initializer exactly while
    leaving a short path from exact-set loss to the deployed scores.
    """

    STATE_ROLES = 4

    def __init__(
        self,
        config: PathRouteSurrogateConfig,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
        *, raw_rank: int = 64,
    ) -> None:
        super().__init__(config, expert_keys, centered_bias, rank_mask)
        if raw_rank < 1:
            raise ValueError("direct branch raw rank must be positive")
        self.raw_rank = raw_rank
        self.branch_state_down = nn.Parameter(torch.empty(
            self.STATE_ROLES, config.layers, config.token_width, raw_rank
        ))
        self.branch_state_up = nn.Parameter(torch.empty(
            self.STATE_ROLES, config.layers, raw_rank, config.width
        ))
        self.branch_router = nn.Linear(config.experts, config.width)
        self.branch_route_embedding = nn.Parameter(torch.empty(
            config.experts, config.route_width
        ))
        self.branch_route = nn.Linear(config.route_width, config.width)
        self.branch_vocab = nn.Sequential(
            nn.RMSNorm(config.token_width),
            nn.Linear(config.token_width, config.width), nn.SiLU(),
        )
        self.branch_statistics = nn.Sequential(
            nn.Linear(14, config.width), nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.branch_norm = nn.RMSNorm(config.width)
        self.query_down_direct = nn.Parameter(torch.empty(
            config.layers, config.width, raw_rank
        ))
        self.query_up_direct = nn.Parameter(torch.zeros(
            config.layers, raw_rank, config.router_rank
        ))
        self.score_down_direct = nn.Parameter(torch.empty(
            config.layers, config.width, raw_rank
        ))
        self.score_up_direct = nn.Parameter(torch.zeros(
            config.layers, raw_rank, config.experts
        ))
        nn.init.normal_(self.branch_state_down, std=0.02)
        nn.init.normal_(self.branch_state_up, std=0.02)
        nn.init.normal_(self.branch_route_embedding, std=0.02)
        nn.init.normal_(self.query_down_direct, std=0.02)
        nn.init.normal_(self.score_down_direct, std=0.02)

    def branch_parameters(self) -> list[nn.Parameter]:
        names = (
            "branch_state_down", "branch_state_up", "branch_router",
            "branch_route_embedding", "branch_route", "branch_vocab",
            "branch_statistics", "branch_norm", "query_down_direct",
            "query_up_direct", "score_down_direct", "score_up_direct",
        )
        result: list[nn.Parameter] = []
        for name in names:
            value = getattr(self, name)
            if isinstance(value, nn.Parameter): result.append(value)
            else: result.extend(value.parameters())
        return result

    def _branch_context(
        self, *, parent_hidden: Tensor, branch_states: Tensor,
        branch_router_logits: Tensor, branch_selected_ids: Tensor,
        branch_selected_weights: Tensor, branch_vocab_embedding: Tensor,
        branch_vocab_statistics: Tensor, branch_scalars: Tensor,
    ) -> Tensor:
        config = self.config; batch = parent_hidden.shape[0]
        if branch_states.shape != (batch, self.STATE_ROLES, config.token_width):
            raise ValueError("direct branch states must be [B,4,D]")
        low = torch.einsum(
            "bqd,qlda->bqla", branch_states.float(),
            self.branch_state_down.float(),
        )
        state = torch.einsum(
            "bqla,qlaw->blw", torch.nn.functional.silu(low),
            self.branch_state_up.float(),
        )
        weights = branch_selected_weights.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        route = self.branch_route((
            self.branch_route_embedding[branch_selected_ids.long()]
            * weights[..., None]
        ).sum(-2))
        shared = (
            self.branch_router(branch_router_logits.float())
            + self.branch_vocab(branch_vocab_embedding.float())
            + self.branch_statistics(torch.cat(
                (branch_vocab_statistics.float(), branch_scalars.float()), -1
            ))
            + route
        )
        return self.branch_norm(
            parent_hidden.float() + state[:, None] + shared[:, None, None]
        )

    def forward(
        self, *, state_coordinates: Tensor, current_queries: Tensor,
        history_selected_ids: Tensor, history_selected_weights: Tensor,
        path_token_embeddings: Tensor, branch_states: Tensor,
        branch_router_logits: Tensor, branch_selected_ids: Tensor,
        branch_selected_weights: Tensor, branch_vocab_embedding: Tensor,
        branch_vocab_statistics: Tensor, branch_scalars: Tensor,
    ) -> PathRouteSurrogateOutput:
        config = self.config
        parent = super().forward(
            state_coordinates=state_coordinates, current_queries=current_queries,
            history_selected_ids=history_selected_ids,
            history_selected_weights=history_selected_weights,
            path_token_embeddings=path_token_embeddings,
        )
        context = self._branch_context(
            parent_hidden=parent.hidden, branch_states=branch_states,
            branch_router_logits=branch_router_logits,
            branch_selected_ids=branch_selected_ids,
            branch_selected_weights=branch_selected_weights,
            branch_vocab_embedding=branch_vocab_embedding,
            branch_vocab_statistics=branch_vocab_statistics,
            branch_scalars=branch_scalars,
        )
        query_low = torch.nn.functional.silu(torch.einsum(
            "bhlw,lwa->bhla", context, self.query_down_direct
        ))
        delta_query = torch.einsum(
            "bhla,lar->bhlr", query_low, self.query_up_direct
        ) * self.rank_mask[None, None]
        score_low = torch.nn.functional.silu(torch.einsum(
            "bhlw,lwa->bhla", context, self.score_down_direct
        ))
        delta_score = torch.einsum(
            "bhla,lae->bhle", score_low, self.score_up_direct
        )
        queries = parent.queries.float() + delta_query.float()
        scores = parent.scores.float() + delta_score.float() + torch.einsum(
            "bhlr,ler->bhle", delta_query.float(), self.expert_keys.float()
        )
        return PathRouteSurrogateOutput(
            queries=queries, scores=scores,
            selected_ids=stable_topk(scores, config.exact_k),
            hidden=context, path_states=parent.path_states,
        )


__all__ = ["DirectHighRankBranchRouteSurrogate"]
