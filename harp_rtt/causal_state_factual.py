"""Direct factual route prediction from exact serving-time target state.

The branch models preserve rich MTP tensors but compress the completed target
token to router coordinates and a few route-logit summaries before H2--H4.
This head keeps four exact causal residual streams layer-local, retains the
identities and execution weights of the eight-token route history, and models
the complete horizon/layer grid jointly.  It never accepts a future target or
counterfactual tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .route_dynamics import DeltaRouteConfig


@dataclass(frozen=True)
class CausalStateFactualOutput:
    score_delta: Tensor
    hidden: Tensor


class CausalStateFactualHead(nn.Module):
    """Forecast factual H1--H4 routes from information available at token end.

    The output is reference-anchored: it is bit-identically zero at
    initialization, while every causal encoder receives a first-step gradient.
    H1 is deliberately held at the parent because it has a separate gate.
    """

    STATE_ROLES = 4

    def __init__(
        self,
        config: DeltaRouteConfig,
        expert_keys: Tensor,
        *,
        state_rank: int = 16,
        route_width: int = 64,
        axial_blocks: int = 2,
    ) -> None:
        super().__init__()
        config.validate()
        if expert_keys.shape != (
            config.layers, config.experts, config.router_rank
        ):
            raise ValueError("expert keys disagree with causal-state config")
        if min(state_rank, route_width, axial_blocks) < 1:
            raise ValueError("causal-state dimensions must be positive")
        self.config = config
        self.state_rank = int(state_rank)
        self.route_width = int(route_width)
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())

        # A shared path captures global semantics while low-rank layer adapters
        # preserve the rotation of route-relevant content across target layers.
        self.state_norm = nn.ModuleList(
            nn.RMSNorm(config.raw_width) for _ in range(self.STATE_ROLES)
        )
        self.state_shared = nn.ModuleList(
            nn.Linear(config.raw_width, config.latent_width)
            for _ in range(self.STATE_ROLES)
        )
        self.state_down = nn.Parameter(torch.empty(
            self.STATE_ROLES, config.layers, config.raw_width, state_rank
        ))
        self.state_up = nn.Parameter(torch.empty(
            self.STATE_ROLES, config.layers, state_rank, config.latent_width
        ))
        self.state_gate = nn.Parameter(torch.ones(
            self.STATE_ROLES, config.layers, 1
        ))

        self.current_query = nn.Linear(config.router_rank, config.latent_width)
        self.history_logits = nn.Linear(config.experts, config.latent_width)
        self.route_embedding = nn.Parameter(torch.empty(
            config.layers, config.experts, route_width
        ))
        self.route_projection = nn.Linear(route_width, config.latent_width)
        self.lag_logits = nn.Parameter(torch.zeros(config.horizons, 8))
        self.parent_context = nn.Linear(config.latent_width, config.latent_width)
        self.layer = nn.Embedding(config.layers, config.latent_width)
        self.horizon = nn.Embedding(config.horizons, config.latent_width)
        self.input_norm = nn.RMSNorm(config.latent_width)

        block = nn.TransformerEncoderLayer(
            d_model=config.latent_width,
            nhead=config.attention_heads,
            dim_feedforward=config.transition_width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.axial = nn.TransformerEncoder(
            block, num_layers=axial_blocks,
            norm=nn.RMSNorm(config.latent_width),
        )

        self.query_hidden = nn.Sequential(
            nn.RMSNorm(config.latent_width),
            nn.Linear(config.latent_width, config.latent_width), nn.SiLU(),
        )
        self.query_output = nn.Linear(config.latent_width, config.router_rank)
        self.query_down = nn.Parameter(torch.empty(
            config.layers, config.latent_width, config.layer_adapter_rank
        ))
        self.query_up = nn.Parameter(torch.empty(
            config.layers, config.layer_adapter_rank, config.router_rank
        ))
        self.free_coefficients = nn.Linear(config.latent_width, config.free_rank)
        self.free_basis = nn.Parameter(torch.empty(
            config.layers, config.experts, config.free_rank
        ))
        self.horizon_bias = nn.Parameter(torch.zeros(
            config.horizons, config.layers, config.experts
        ))

        nn.init.normal_(self.state_down, std=0.02)
        nn.init.normal_(self.state_up, std=0.02)
        nn.init.normal_(self.route_embedding, std=0.02)
        nn.init.normal_(self.query_down, std=0.02)
        nn.init.normal_(self.query_up, std=0.02)
        nn.init.normal_(self.free_basis, std=0.02)
        self._snapshot_reference()

    def _snapshot_reference(self) -> None:
        for name, value in (
            ("ref_query_weight", self.query_output.weight),
            ("ref_query_bias", self.query_output.bias),
            ("ref_query_down", self.query_down),
            ("ref_query_up", self.query_up),
            ("ref_free_weight", self.free_coefficients.weight),
            ("ref_free_bias", self.free_coefficients.bias),
            ("ref_free_basis", self.free_basis),
        ):
            self.register_buffer(name, value.detach().clone())

    def _state_features(self, states: Tensor) -> Tensor:
        config = self.config
        if states.shape[-3:] != (
            self.STATE_ROLES, config.layers, config.raw_width
        ):
            raise ValueError("causal residual states must end in [4,L,D]")
        values = []
        for role in range(self.STATE_ROLES):
            value = self.state_norm[role](states[..., role, :, :])
            shared = self.state_shared[role](value)
            low = torch.einsum(
                "...ld,lda->...la", value, self.state_down[role]
            )
            adapted = torch.einsum(
                "...la,law->...lw", F.silu(low), self.state_up[role]
            )
            values.append(shared + self.state_gate[role] * adapted)
        return torch.stack(values, dim=-2).mean(-2)

    def _history_features(
        self,
        logits: Tensor,
        selected_ids: Tensor,
        selected_weights: Tensor,
    ) -> Tensor:
        config = self.config
        if logits.ndim != 4 or logits.shape[-2:] != (
            config.layers, config.experts
        ):
            raise ValueError("causal route history must be [B,T,L,E]")
        batch, history, layers, _ = logits.shape
        if history != 8 or layers != config.layers:
            raise ValueError("causal route history must contain eight tokens")
        if selected_ids.shape != (batch, history, layers, config.exact_k):
            raise ValueError("history selected IDs have invalid geometry")
        if selected_weights.shape != selected_ids.shape:
            raise ValueError("history execution weights have invalid geometry")
        layer = torch.arange(layers, device=logits.device)[None, None, :, None]
        embedded = self.route_embedding[layer, selected_ids.long()]
        weights = selected_weights.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        route = self.route_projection((embedded * weights[..., None]).sum(-2))
        dense = self.history_logits(logits.float())
        per_lag = route + dense
        lag = torch.softmax(self.lag_logits.float(), dim=-1)
        return torch.einsum("ht,btlw->bhlw", lag, per_lag)

    def _output_delta(self, hidden: Tensor) -> Tensor:
        live_hidden = self.query_hidden(hidden)
        live_q = self.query_output(live_hidden)
        live_low = F.silu(torch.einsum(
            "bhlw,lwa->bhla", hidden, self.query_down
        ))
        live_q = live_q + torch.einsum(
            "bhla,lar->bhlr", live_low, self.query_up
        )

        detached = hidden.detach()
        ref_hidden = self.query_hidden(detached).detach()
        ref_q = F.linear(ref_hidden, self.ref_query_weight, self.ref_query_bias)
        ref_low = F.silu(torch.einsum(
            "bhlw,lwa->bhla", detached, self.ref_query_down
        ))
        ref_q = ref_q + torch.einsum(
            "bhla,lar->bhlr", ref_low, self.ref_query_up
        )
        delta_q = live_q.float() - ref_q.float()
        geometry = torch.einsum(
            "bhlr,ler->bhle", delta_q, self.expert_keys
        )

        live_coeff = self.free_coefficients(hidden)
        live_free = torch.einsum(
            "bhla,lea->bhle", live_coeff.float(), self.free_basis.float()
        )
        ref_coeff = F.linear(
            detached, self.ref_free_weight, self.ref_free_bias
        )
        ref_free = torch.einsum(
            "bhla,lea->bhle", ref_coeff.float(), self.ref_free_basis.float()
        )
        return geometry + live_free - ref_free + self.horizon_bias[None]

    def forward(
        self,
        *,
        context_states: Tensor,
        current_states: Tensor,
        current_queries: Tensor,
        history_logits: Tensor,
        history_selected_ids: Tensor,
        history_selected_weights: Tensor,
    ) -> CausalStateFactualOutput:
        config = self.config
        batch = current_states.shape[0]
        if context_states.shape != (
            batch, config.horizons, config.layers, config.latent_width
        ):
            raise ValueError("parent context has invalid geometry")
        if current_queries.shape != (
            batch, config.layers, config.router_rank
        ):
            raise ValueError("current router coordinates have invalid geometry")
        state = self._state_features(current_states)
        history = self._history_features(
            history_logits, history_selected_ids, history_selected_weights
        )
        token = (
            state[:, None]
            + self.current_query(current_queries.float())[:, None]
            + history
            + self.parent_context(context_states)
            + self.horizon.weight[None, :, None]
            + self.layer.weight[None, None]
        )
        token = self.input_norm(token)
        sequence = token.reshape(
            batch, config.horizons * config.layers, config.latent_width
        )
        hidden = self.axial(sequence).reshape(
            batch, config.horizons, config.layers, config.latent_width
        )
        delta = self._output_delta(hidden)
        delta = delta.clone(); delta[:, 0] = 0.0
        return CausalStateFactualOutput(delta, hidden)


__all__ = ["CausalStateFactualHead", "CausalStateFactualOutput"]
