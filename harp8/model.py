"""HARP-8T accuracy-first endpoint forecaster."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import HARPConfig


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.up = nn.Linear(width, hidden * 2)
        self.down = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(self.norm(inputs)).chunk(2, dim=-1)
        return inputs + self.dropout(self.down(F.silu(gate) * value))


class _CrossAttentionBlock(nn.Module):
    def __init__(self, config: HARPConfig) -> None:
        super().__init__()
        width = config.model_width
        self.query_norm = nn.RMSNorm(width)
        self.node_norm = nn.RMSNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.feedforward = _ResidualBlock(
            width, config.mtp_ffn_width, config.dropout
        )

    def forward(
        self,
        queries: torch.Tensor,
        nodes: torch.Tensor,
        *,
        node_padding_mask: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        update, _weights = self.attention(
            self.query_norm(queries),
            self.node_norm(nodes),
            self.node_norm(nodes),
            key_padding_mask=node_padding_mask,
            attn_mask=attention_bias,
            need_weights=False,
        )
        return self.feedforward(queries + self.dropout(update))


def _transformer_layer(
    width: int,
    heads: int,
    feedforward: int,
    dropout: float,
) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=feedforward,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class HARP8Teacher(nn.Module):
    """Direct, non-recursive router-score predictor for horizons 1 through 8.

    The forward interface contains only information available at the token-end
    decision point. Future labels are intentionally accepted by the loss module,
    never by this model.
    """

    SOURCE_ROUTE = 0
    SOURCE_TARGET_STATE = 1
    SOURCE_MTP = 2

    def __init__(self, config: HARPConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.route_norm = nn.LayerNorm(config.experts)
        self.route_cell = nn.Sequential(
            nn.Linear(config.experts + 6, config.route_width * 2),
            nn.SiLU(),
            nn.Linear(config.route_width * 2, config.route_width),
            nn.RMSNorm(config.route_width),
        )
        self.lag_embedding = nn.Embedding(config.route_history, config.route_width)
        self.route_layer_embedding = nn.Embedding(config.layers, config.route_width)
        self.temporal_blocks = nn.ModuleList(
            [
                _transformer_layer(
                    config.route_width,
                    config.attention_heads,
                    config.route_ffn_width,
                    config.dropout,
                )
                for _ in range(config.temporal_blocks)
            ]
        )
        self.layer_route_blocks = nn.ModuleList(
            [
                _transformer_layer(
                    config.route_width,
                    config.attention_heads,
                    config.route_ffn_width,
                    config.dropout,
                )
                for _ in range(config.layer_blocks)
            ]
        )
        self.route_output_norm = nn.RMSNorm(config.route_width)

        self.target_state_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.RMSNorm(config.target_state_width),
                    nn.Linear(config.target_state_width, config.state_width),
                )
                for _ in range(config.target_state_channels)
            ]
        )
        self.target_channel_embedding = nn.Embedding(
            config.target_state_channels, config.state_width
        )
        self.target_layer_embedding = nn.Embedding(config.layers, config.state_width)
        self.target_state_blocks = nn.ModuleList(
            [
                _transformer_layer(
                    config.state_width,
                    config.attention_heads,
                    config.route_ffn_width,
                    config.dropout,
                )
                for _ in range(config.state_blocks)
            ]
        )
        self.target_state_norm = nn.RMSNorm(config.state_width)

        self.mtp_state_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.RMSNorm(config.mtp_state_width),
                    nn.Linear(
                        config.mtp_state_width,
                        config.mtp_state_projection_width,
                    ),
                )
                for _ in range(config.mtp_state_channels)
            ]
        )
        self.mtp_channel_embedding = nn.Embedding(
            config.mtp_state_channels,
            config.mtp_state_projection_width,
        )
        self.mtp_router_projection = nn.Sequential(
            nn.LayerNorm(config.experts), nn.Linear(config.experts, 128), nn.SiLU()
        )
        self.mtp_metadata_projection = nn.Sequential(
            nn.LayerNorm(config.mtp_metadata_width),
            nn.Linear(config.mtp_metadata_width, 64),
            nn.SiLU(),
        )
        if config.mtp_vocab_width:
            self.mtp_vocab_projection: nn.Module | None = nn.Sequential(
                nn.RMSNorm(config.mtp_vocab_width),
                nn.Linear(config.mtp_vocab_width, 256),
                nn.SiLU(),
            )
            vocab_output = 256
        else:
            self.mtp_vocab_projection = None
            vocab_output = 0
        node_input = config.mtp_state_projection_width + 128 + 64 + vocab_output
        self.mtp_node_fusion = nn.Sequential(
            nn.Linear(node_input, config.mtp_ffn_width),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mtp_ffn_width, config.model_width),
            nn.RMSNorm(config.model_width),
        )
        self.mtp_depth_embedding = nn.Embedding(config.mtp_depths + 1, config.model_width)

        self.route_to_model = nn.Linear(config.route_width, config.model_width)
        # Preserve high-information flattened signals as additive skip paths.
        # The original HARP path compresses these through attention; these
        # projections let training retain exact local route, target-state, MTP
        # hidden/router, and request-position evidence when useful.
        self.direct_route_history = min(3, config.route_history)
        direct_route_width = self.direct_route_history * config.experts + self.direct_route_history
        self.direct_route_projection = nn.Sequential(
            nn.LayerNorm(direct_route_width),
            nn.Linear(direct_route_width, config.model_width),
            nn.SiLU(),
        )
        direct_mtp_width = config.mtp_depths * (config.mtp_state_width + config.experts + 2)
        self.direct_mtp_projection = nn.Sequential(
            nn.LayerNorm(direct_mtp_width),
            nn.Linear(direct_mtp_width, config.model_width),
            nn.SiLU(),
        )
        self.direct_target_projection = nn.Sequential(
            nn.RMSNorm(config.target_state_width),
            nn.Linear(config.target_state_width, config.model_width),
            nn.SiLU(),
        )
        self.position_projection = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, config.model_width),
            nn.SiLU(),
        )
        self.state_to_model = nn.Linear(config.state_width, config.model_width)
        self.output_layer_embedding = nn.Embedding(config.layers, config.model_width)
        self.horizon_embedding = nn.Embedding(config.horizons, config.model_width)
        self.alignment_bias = nn.Embedding(
            config.horizons * 2 + 1, config.attention_heads
        )
        self.cross_blocks = nn.ModuleList(
            [_CrossAttentionBlock(config) for _ in range(config.mtp_cross_blocks)]
        )

        self.source_embedding = nn.Parameter(torch.zeros(3, config.model_width))
        self.source_value = nn.Linear(config.model_width, config.model_width)
        self.source_score = nn.Linear(config.model_width, 1, bias=False)
        self.fusion_blocks = nn.ModuleList(
            [
                _ResidualBlock(
                    config.model_width,
                    config.fusion_ffn_width,
                    config.dropout,
                )
                for _ in range(config.fusion_blocks)
            ]
        )
        self.final_norm = nn.RMSNorm(config.model_width)

        if config.dense_output:
            self.output_weight = nn.Parameter(
                torch.empty(
                    config.horizons,
                    config.layers,
                    config.model_width,
                    config.experts,
                )
            )
            self.output_a = None
            self.output_b = None
        else:
            self.register_parameter("output_weight", None)
            self.output_a = nn.Parameter(
                torch.empty(config.horizons, config.model_width, config.output_rank)
            )
            self.output_b = nn.Parameter(
                torch.empty(
                    config.horizons,
                    config.layers,
                    config.output_rank,
                    config.experts,
                )
            )
        self.output_bias = nn.Parameter(
            torch.zeros(config.horizons, config.layers, config.experts)
        )
        self.copy_query = nn.Parameter(
            torch.zeros(config.horizons, config.layers, config.model_width)
        )
        self.copy_scalar_bias = nn.Parameter(
            torch.zeros(config.horizons, config.layers)
        )
        self.copy_expert_bias = nn.Parameter(
            torch.empty(config.horizons, config.layers, config.experts)
        )
        self.future_latent_head = (
            nn.Linear(config.model_width, config.future_latent_width)
            if config.future_latent_width
            else None
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.output_weight is not None:
            nn.init.normal_(self.output_weight, mean=0.0, std=0.02)
        if self.output_a is not None and self.output_b is not None:
            nn.init.xavier_uniform_(self.output_a)
            nn.init.zeros_(self.output_b)
        priors = torch.linspace(0.80, 0.35, self.config.horizons)
        logits = torch.logit(priors).view(-1, 1, 1)
        with torch.no_grad():
            self.copy_expert_bias.copy_(logits.expand_as(self.copy_expert_bias))
        nn.init.zeros_(self.alignment_bias.weight)
        # Start additive direct skips at zero so the enhanced model is a
        # function-preserving extension of the validated HARP endpoint. The
        # optimizer can then grow each skip only when it improves the loss.
        for module in (
            self.direct_route_projection,
            self.direct_mtp_projection,
            self.direct_target_projection,
            self.position_projection,
        ):
            nn.init.zeros_(module[1].weight)
            if module[1].bias is not None:
                nn.init.zeros_(module[1].bias)

    def _check_shape(
        self, value: torch.Tensor, trailing: tuple[int, ...], name: str
    ) -> None:
        if tuple(value.shape[1:]) != trailing:
            raise ValueError(
                f"{name} has shape {tuple(value.shape)}; expected [B,{','.join(map(str, trailing))}]"
            )

    def _route_statistics(self, centered: torch.Tensor) -> torch.Tensor:
        values = centered.float()
        probabilities = torch.softmax(values, dim=-1)
        top_values = torch.topk(values, 9, dim=-1, sorted=True).values
        top_probabilities = probabilities.gather(
            -1, torch.topk(values, 8, dim=-1, sorted=False).indices
        )
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        entropy = entropy / math.log(self.config.experts)
        return torch.stack(
            [
                entropy,
                top_values[..., 0] - top_values[..., 1],
                top_values[..., 7] - top_values[..., 8],
                top_probabilities.sum(-1),
                values.std(dim=-1, unbiased=False),
                values.abs().mean(dim=-1),
            ],
            dim=-1,
        ).to(centered.dtype)

    def _encode_route_grid(
        self, route_history: torch.Tensor, route_available: torch.Tensor
    ) -> torch.Tensor:
        config = self.config
        self._check_shape(
            route_history,
            (config.layers, config.route_history, config.experts),
            "route_history",
        )
        self._check_shape(route_available, (config.route_history,), "route_available")
        batch = route_history.shape[0]
        centered = route_history - route_history.mean(dim=-1, keepdim=True)
        statistics = self._route_statistics(centered)
        cells = self.route_cell(
            torch.cat([self.route_norm(centered), statistics], dim=-1)
        )
        lag_ids = torch.arange(config.route_history, device=cells.device)
        layer_ids = torch.arange(config.layers, device=cells.device)
        cells = (
            cells
            + self.lag_embedding(lag_ids)[None, None]
            + self.route_layer_embedding(layer_ids)[None, :, None]
        )
        cells = cells * route_available[:, None, :, None].to(cells.dtype)
        temporal = cells.reshape(batch * config.layers, config.route_history, -1)
        padding = (~route_available.bool())[:, None].expand(-1, config.layers, -1)
        padding = padding.reshape(batch * config.layers, config.route_history)
        for block in self.temporal_blocks:
            temporal = block(temporal, src_key_padding_mask=padding)
        layer_state = temporal[:, 0].reshape(batch, config.layers, -1)
        for block in self.layer_route_blocks:
            layer_state = block(layer_state)
        return self.route_output_norm(layer_state)

    def _encode_target_states(
        self,
        target_states: torch.Tensor,
        target_state_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        self._check_shape(
            target_states,
            (
                config.layers,
                config.target_state_channels,
                config.target_state_width,
            ),
            "target_states",
        )
        self._check_shape(
            target_state_available,
            (config.layers, config.target_state_channels),
            "target_state_available",
        )
        encoded = []
        for channel, projection in enumerate(self.target_state_projections):
            value = projection(target_states[:, :, channel])
            value = value + self.target_channel_embedding.weight[channel]
            encoded.append(value)
        stacked = torch.stack(encoded, dim=2)
        available = target_state_available.to(stacked.dtype)
        state = (stacked * available[..., None]).sum(dim=2)
        state = state / available.sum(dim=2, keepdim=True).clamp_min(1)
        layer_ids = torch.arange(config.layers, device=state.device)
        state = state + self.target_layer_embedding(layer_ids)[None]
        for block in self.target_state_blocks:
            state = block(state)
        layer_available = target_state_available.any(dim=-1)
        return self.target_state_norm(state), layer_available

    def _encode_mtp_nodes(
        self,
        mtp_states: torch.Tensor,
        mtp_router_logits: torch.Tensor,
        mtp_metadata: torch.Tensor,
        mtp_depth_ids: torch.Tensor,
        mtp_vocab_features: torch.Tensor | None,
    ) -> torch.Tensor:
        config = self.config
        nodes = mtp_states.shape[1]
        self._check_shape(
            mtp_states,
            (nodes, config.mtp_state_channels, config.mtp_state_width),
            "mtp_states",
        )
        self._check_shape(mtp_router_logits, (nodes, config.experts), "mtp_router_logits")
        self._check_shape(mtp_metadata, (nodes, config.mtp_metadata_width), "mtp_metadata")
        self._check_shape(mtp_depth_ids, (nodes,), "mtp_depth_ids")
        state_parts = []
        for channel, projection in enumerate(self.mtp_state_projections):
            value = projection(mtp_states[:, :, channel])
            value = value + self.mtp_channel_embedding.weight[channel]
            state_parts.append(value)
        state = torch.stack(state_parts, dim=2).mean(dim=2)
        centered_router = mtp_router_logits - mtp_router_logits.mean(
            dim=-1, keepdim=True
        )
        parts = [
            state,
            self.mtp_router_projection(centered_router),
            self.mtp_metadata_projection(mtp_metadata),
        ]
        if self.mtp_vocab_projection is not None:
            if mtp_vocab_features is None:
                raise ValueError("configured MTP vocabulary features are missing")
            self._check_shape(
                mtp_vocab_features,
                (nodes, config.mtp_vocab_width),
                "mtp_vocab_features",
            )
            parts.append(self.mtp_vocab_projection(mtp_vocab_features))
        depth = mtp_depth_ids.clamp(0, config.mtp_depths)
        return self.mtp_node_fusion(torch.cat(parts, dim=-1)) + self.mtp_depth_embedding(depth)

    def _attention_bias(
        self, mtp_depth_ids: torch.Tensor, nodes: int
    ) -> torch.Tensor:
        config = self.config
        batch = mtp_depth_ids.shape[0]
        horizon = torch.arange(1, config.horizons + 1, device=mtp_depth_ids.device)
        query_horizon = horizon[:, None].expand(-1, config.layers).reshape(-1)
        delta = mtp_depth_ids[:, None, :] - query_horizon[None, :, None]
        delta = delta.clamp(-config.horizons, config.horizons) + config.horizons
        bias = self.alignment_bias(delta)
        bias = bias.permute(0, 3, 1, 2).reshape(
            batch * config.attention_heads, config.horizons * config.layers, nodes
        )
        return bias

    def _source_availability(
        self,
        state_available: torch.Tensor,
        mtp_available: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        config = self.config
        batch = state_available.shape[0]
        route = torch.ones(
            (batch, config.horizons, config.layers),
            dtype=torch.bool,
            device=state_available.device,
        )
        target = state_available[:, None].expand(-1, config.horizons, -1)
        mtp = mtp_available.any(dim=-1)[:, None, None].expand(
            -1, config.horizons, config.layers
        )
        available = torch.stack([route, target, mtp], dim=-1)
        if self.training:
            if config.target_state_source_dropout:
                keep = torch.rand(batch, 1, 1, device=available.device)
                keep = keep >= config.target_state_source_dropout
                available[..., self.SOURCE_TARGET_STATE] &= keep
            if config.mtp_source_dropout:
                keep = torch.rand(batch, 1, 1, device=available.device)
                keep = keep >= config.mtp_source_dropout
                available[..., self.SOURCE_MTP] &= keep
        return available

    def forward(
        self,
        *,
        route_history: torch.Tensor,
        route_available: torch.Tensor,
        target_states: torch.Tensor,
        target_state_available: torch.Tensor,
        mtp_states: torch.Tensor,
        mtp_router_logits: torch.Tensor,
        mtp_metadata: torch.Tensor,
        mtp_depth_ids: torch.Tensor,
        mtp_available: torch.Tensor,
        mtp_vocab_features: torch.Tensor | None = None,
        within_request: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        config = self.config
        batch = route_history.shape[0]
        route = self._encode_route_grid(route_history, route_available)
        state, state_available = self._encode_target_states(
            target_states, target_state_available
        )
        if not config.use_target_state:
            state_available = torch.zeros_like(state_available)
        nodes = self._encode_mtp_nodes(
            mtp_states,
            mtp_router_logits,
            mtp_metadata,
            mtp_depth_ids,
            mtp_vocab_features,
        )
        node_count = nodes.shape[1]
        active_depths = config.mtp_active_depths or config.mtp_depths
        effective_mtp_available = mtp_available.bool() & (
            mtp_depth_ids <= active_depths
        )
        if not config.use_mtp:
            effective_mtp_available = torch.zeros_like(effective_mtp_available)
        safe_available = effective_mtp_available.clone()
        missing_all = ~safe_available.any(dim=-1)
        if missing_all.any():
            safe_available[missing_all, 0] = True
            nodes = nodes.clone()
            nodes[missing_all, 0] = 0

        layer_ids = torch.arange(config.layers, device=route.device)
        horizon_ids = torch.arange(config.horizons, device=route.device)

        route_direct = route_history.float()
        route_direct = route_direct - route_direct.mean(dim=-1, keepdim=True)
        route_direct = torch.cat(
            [
                route_direct[:, :, : self.direct_route_history].reshape(
                    batch, config.layers, -1
                ),
                route_available[:, : self.direct_route_history].float()[:, None].expand(
                    -1, config.layers, -1
                ),
            ],
            dim=-1,
        )
        route_direct = self.direct_route_projection(route_direct.to(route.dtype))

        # Canonicalize nodes by their explicit depth ID before flattening, so
        # direct skips do not turn CUDA batch order into a semantic feature.
        depth_order = torch.argsort(mtp_depth_ids, dim=1)
        depth_gather_state = depth_order[:, :, None, None].expand(
            -1, -1, mtp_states.shape[2], mtp_states.shape[3]
        )
        depth_gather_router = depth_order[:, :, None].expand(
            -1, -1, mtp_router_logits.shape[2]
        )
        depth_gather = depth_order
        mtp_states_direct = torch.gather(mtp_states, 1, depth_gather_state)
        mtp_router_direct = torch.gather(mtp_router_logits, 1, depth_gather_router)
        mtp_available_direct = torch.gather(effective_mtp_available, 1, depth_gather)
        mtp_depth_ids_direct = torch.gather(mtp_depth_ids, 1, depth_gather)
        mtp_direct_states = mtp_states_direct[:, :, 0].float()
        mtp_direct_states = mtp_direct_states * mtp_available_direct[:, :, None].float()
        mtp_direct_router = mtp_router_direct.float()
        mtp_direct_router = mtp_direct_router - mtp_direct_router.mean(
            dim=-1, keepdim=True
        )
        mtp_direct_router = mtp_direct_router * mtp_available_direct[:, :, None].float()
        mtp_direct = torch.cat(
            [
                mtp_direct_states.reshape(batch, -1),
                mtp_direct_router.reshape(batch, -1),
                mtp_available_direct.float(),
                mtp_depth_ids_direct.float() / max(1, config.mtp_depths),
            ],
            dim=-1,
        )
        mtp_direct = self.direct_mtp_projection(mtp_direct.to(route.dtype))
        if not config.use_mtp:
            mtp_direct = torch.zeros_like(mtp_direct)

        target_direct = self.direct_target_projection(
            target_states[:, :, 0].to(route.dtype)
        )
        if not config.use_target_state:
            target_direct = torch.zeros_like(target_direct)
        position = within_request
        if position is None:
            position = torch.zeros(batch, device=route.device, dtype=route.dtype)
        position = position.float().view(batch, 1)
        position = torch.cat([position / 33.0, (position / 33.0).square()], dim=-1)
        position_direct = self.position_projection(position.to(route.dtype))

        route_source = self.route_to_model(route)[:, None].expand(
            -1, config.horizons, -1, -1
        ) + route_direct[:, None]
        query = (
            route_source
            + target_direct[:, None]
            + mtp_direct[:, None, None]
            + position_direct[:, None, None]
            + self.output_layer_embedding(layer_ids)[None, None]
            + self.horizon_embedding(horizon_ids)[None, :, None]
        )
        mtp_source = query.reshape(batch, config.horizons * config.layers, -1)
        attention_bias = self._attention_bias(mtp_depth_ids, node_count)
        node_padding_mask = torch.zeros(
            safe_available.shape,
            dtype=attention_bias.dtype,
            device=safe_available.device,
        ).masked_fill(~safe_available, -torch.inf)
        for block in self.cross_blocks:
            mtp_source = block(
                mtp_source,
                nodes,
                node_padding_mask=node_padding_mask,
                attention_bias=attention_bias,
            )
        mtp_source = mtp_source.reshape(
            batch, config.horizons, config.layers, config.model_width
        )
        state_source = self.state_to_model(state)[:, None].expand(
            -1, config.horizons, -1, -1
        )
        sources = torch.stack([route_source, state_source, mtp_source], dim=-2)
        gate_hidden = torch.tanh(
            self.source_value(sources)
            + query.unsqueeze(-2)
            + self.source_embedding[None, None, None]
        )
        gate_logits = self.source_score(gate_hidden).squeeze(-1)
        source_available = self._source_availability(
            state_available, effective_mtp_available, dtype=gate_logits.dtype
        )
        gate_logits = gate_logits.masked_fill(~source_available, -torch.inf)
        source_weights = torch.softmax(gate_logits.float(), dim=-1).to(gate_logits.dtype)
        hidden = (sources * source_weights.unsqueeze(-1)).sum(dim=-2)
        for block in self.fusion_blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)

        if self.output_weight is not None:
            delta = torch.einsum("bhld,hlde->bhle", hidden, self.output_weight)
        else:
            assert self.output_a is not None and self.output_b is not None
            low_rank = torch.einsum("bhld,hdr->bhlr", hidden, self.output_a)
            delta = torch.einsum("bhlr,hlre->bhle", F.silu(low_rank), self.output_b)
        delta = delta + self.output_bias[None]
        dynamic_copy = torch.einsum("bhld,hld->bhl", hidden, self.copy_query)
        dynamic_copy = dynamic_copy + self.copy_scalar_bias[None]
        copy_gate = torch.sigmoid(
            dynamic_copy[..., None] + self.copy_expert_bias[None]
        )
        current = route_history[:, :, 0]
        current = current - current.mean(dim=-1, keepdim=True)
        predicted = copy_gate * current[:, None] + delta
        outputs = {
            "future_router_scores": predicted,
            "future_inclusion_probabilities": torch.sigmoid(predicted.float()).to(
                predicted.dtype
            ),
            "source_gate_weights": source_weights,
            "copy_gate": copy_gate,
            "generator_context": hidden,
        }
        if self.future_latent_head is not None:
            outputs["future_latent"] = self.future_latent_head(hidden.mean(dim=2))
        return outputs

    @torch.no_grad()
    def top_experts(
        self, outputs: dict[str, torch.Tensor], candidate_count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 1 <= candidate_count <= self.config.experts:
            raise ValueError("candidate_count lies outside the expert namespace")
        return torch.topk(
            outputs["future_router_scores"], candidate_count, dim=-1, sorted=True
        )
