"""Token-conditioned target route surrogate for large existing-data pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .exact_k import stable_topk


@dataclass(frozen=True)
class PathRouteSurrogateConfig:
    experts: int = 256
    layers: int = 40
    horizons: int = 4
    history: int = 8
    exact_k: int = 8
    router_rank: int = 255
    state_roles: int = 4
    state_rank: int = 128
    token_width: int = 2048
    width: int = 256
    ffn_width: int = 768
    attention_heads: int = 8
    blocks: int = 2
    route_width: int = 64
    output_adapter_rank: int = 16
    free_rank: int = 16
    dropout: float = 0.05

    def validate(self) -> None:
        values = (
            self.experts, self.layers, self.horizons, self.history,
            self.exact_k, self.router_rank, self.state_roles, self.state_rank,
            self.token_width, self.width, self.ffn_width, self.attention_heads,
            self.blocks, self.route_width, self.output_adapter_rank,
            self.free_rank,
        )
        if any(value < 1 for value in values):
            raise ValueError("path surrogate dimensions must be positive")
        if self.width % self.attention_heads or self.exact_k > self.experts:
            raise ValueError("path surrogate attention/expert geometry is invalid")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("path surrogate dropout must lie in [0,1)")

    def to_dict(self) -> dict[str, int | float]:
        self.validate(); return asdict(self)


@dataclass(frozen=True)
class PathRouteSurrogateOutput:
    queries: Tensor
    scores: Tensor
    selected_ids: Tensor
    hidden: Tensor
    path_states: Tensor


class TokenConditionedRouteSurrogate(nn.Module):
    """Predict target router trajectories from current state and token path."""

    def __init__(
        self,
        config: PathRouteSurrogateConfig,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
    ) -> None:
        super().__init__()
        config.validate(); self.config = config
        if expert_keys.shape != (
            config.layers, config.experts, config.router_rank
        ) or centered_bias.shape != (config.layers, config.experts):
            raise ValueError("path surrogate router geometry is invalid")
        if rank_mask.shape != (config.layers, config.router_rank):
            raise ValueError("path surrogate rank mask is invalid")
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.register_buffer("centered_bias", centered_bias.detach().float().clone())
        self.register_buffer("rank_mask", rank_mask.detach().float().clone())

        self.state_norm = nn.ModuleList(
            nn.RMSNorm(config.state_rank) for _ in range(config.state_roles)
        )
        self.state_projection = nn.ModuleList(
            nn.Linear(config.state_rank, config.width)
            for _ in range(config.state_roles)
        )
        self.state_mixture = nn.Parameter(torch.zeros(
            config.layers, config.state_roles
        ))
        self.current_query = nn.Linear(config.router_rank, config.width)

        self.route_embedding = nn.Parameter(torch.empty(
            config.layers, config.experts, config.route_width
        ))
        self.route_projection = nn.Linear(config.route_width, config.width)
        self.history_lag_logits = nn.Parameter(torch.zeros(
            config.horizons, config.history
        ))
        self.token_projection = nn.Sequential(
            nn.RMSNorm(config.token_width),
            nn.Linear(config.token_width, config.width), nn.SiLU(),
        )
        self.path_cell = nn.GRUCell(config.width, config.width)

        self.layer = nn.Embedding(config.layers, config.width)
        self.horizon = nn.Embedding(config.horizons, config.width)
        self.input_norm = nn.RMSNorm(config.width)
        block = nn.TransformerEncoderLayer(
            d_model=config.width, nhead=config.attention_heads,
            dim_feedforward=config.ffn_width, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.axial = nn.TransformerEncoder(
            block, num_layers=config.blocks, norm=nn.RMSNorm(config.width)
        )
        cell_horizons = torch.arange(config.horizons).repeat_interleave(
            config.layers
        )
        # A route at horizon h may use the complete current target state and
        # the proposed token prefix through h, but never a later draft token.
        # Rows are attention queries and columns are keys; True is masked.
        self.register_buffer(
            "axial_causal_mask",
            cell_horizons[None, :] > cell_horizons[:, None],
            persistent=False,
        )
        self.query_hidden = nn.Sequential(
            nn.RMSNorm(config.width),
            nn.Linear(config.width, config.width), nn.SiLU(),
        )
        self.query_output = nn.Linear(config.width, config.router_rank)
        self.query_down = nn.Parameter(torch.empty(
            config.layers, config.width, config.output_adapter_rank
        ))
        self.query_up = nn.Parameter(torch.empty(
            config.layers, config.output_adapter_rank, config.router_rank
        ))
        self.free_coefficients = nn.Linear(config.width, config.free_rank)
        self.free_basis = nn.Parameter(torch.empty(
            config.layers, config.experts, config.free_rank
        ))
        nn.init.normal_(self.route_embedding, std=0.02)
        nn.init.normal_(self.query_down, std=0.02)
        nn.init.normal_(self.query_up, std=0.02)
        nn.init.zeros_(self.free_basis)

    def _path_states(self, token_embeddings: Tensor) -> Tensor:
        config = self.config
        if token_embeddings.shape[-2:] != (
            config.horizons, config.token_width
        ):
            raise ValueError("path token embeddings must end in [H,D]")
        batch = token_embeddings.shape[0]
        state = torch.zeros(
            batch, config.width, device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        )
        values = []
        projected = self.token_projection(token_embeddings)
        for horizon in range(config.horizons):
            state = self.path_cell(
                projected[:, horizon] + self.horizon.weight[horizon], state
            )
            values.append(state)
        return torch.stack(values, dim=1)

    def forward(
        self,
        *,
        state_coordinates: Tensor,
        current_queries: Tensor,
        history_selected_ids: Tensor,
        history_selected_weights: Tensor,
        path_token_embeddings: Tensor,
    ) -> PathRouteSurrogateOutput:
        config = self.config
        batch = state_coordinates.shape[0]
        if state_coordinates.shape != (
            batch, config.state_roles, config.layers, config.state_rank
        ):
            raise ValueError("state-coordinate geometry is invalid")
        if current_queries.shape != (
            batch, config.layers, config.router_rank
        ):
            raise ValueError("current-query geometry is invalid")
        expected_history = (
            batch, config.history, config.layers, config.exact_k
        )
        if history_selected_ids.shape != expected_history or history_selected_weights.shape != expected_history:
            raise ValueError("history route geometry is invalid")

        roles = torch.stack([
            projection(norm(state_coordinates[:, role]))
            for role, (norm, projection) in enumerate(zip(
                self.state_norm, self.state_projection, strict=True
            ))
        ], dim=-2)
        state = torch.einsum(
            "lq,blqw->blw",
            torch.softmax(self.state_mixture.float(), dim=-1), roles.float(),
        )
        layer = torch.arange(config.layers, device=state.device)[
            None, None, :, None
        ]
        route = self.route_embedding[layer, history_selected_ids.long()]
        weights = history_selected_weights.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        route = self.route_projection((route * weights[..., None]).sum(-2))
        history = torch.einsum(
            "ht,btlw->bhlw",
            torch.softmax(self.history_lag_logits.float(), dim=-1), route,
        )
        path = self._path_states(path_token_embeddings)
        token = (
            state[:, None] + self.current_query(current_queries.float())[:, None]
            + history + path[:, :, None]
            + self.layer.weight[None, None] + self.horizon.weight[None, :, None]
        )
        hidden = self.axial(
            self.input_norm(token).reshape(
                batch, config.horizons * config.layers, config.width
            ),
            mask=self.axial_causal_mask,
        ).reshape(batch, config.horizons, config.layers, config.width)
        transformed = self.query_hidden(hidden)
        queries = self.query_output(transformed)
        low = torch.nn.functional.silu(torch.einsum(
            "bhlw,lwa->bhla", hidden, self.query_down
        ))
        queries = queries + torch.einsum(
            "bhla,lar->bhlr", low, self.query_up
        )
        queries = queries * self.rank_mask[None, None]
        scores = torch.einsum(
            "bhlr,ler->bhle", queries.float(), self.expert_keys
        ) + self.centered_bias[None, None]
        free = torch.einsum(
            "bhla,lea->bhle", self.free_coefficients(hidden).float(),
            self.free_basis.float(),
        )
        scores = scores + free
        return PathRouteSurrogateOutput(
            queries, scores, stable_topk(scores, config.exact_k), hidden, path
        )


__all__ = [
    "PathRouteSurrogateConfig", "PathRouteSurrogateOutput",
    "TokenConditionedRouteSurrogate",
]
