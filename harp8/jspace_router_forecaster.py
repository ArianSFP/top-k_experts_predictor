"""Accuracy-first, strictly causal J-space full-router forecaster.

The model predicts a complete score vector for every expert at H1--H8.  It is
not constrained to the frozen HARP candidate namespace: HARP's expanded
full-router scores are only a causal residual baseline, and independent
horizon/layer heads may promote any of the 256 experts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .jspace_router_data import causal_router_forecaster_inputs


JSPACE_ROUTER_FORECASTER_SCHEMA = "harp8_jspace_full_router_forecaster_v1"


@dataclass(frozen=True)
class JSpaceRouterForecasterConfig:
    """Default profile is designed for BF16 training on a 24 GiB GPU."""

    experts: int = 256
    layers: int = 40
    horizons: int = 8
    history: int = 3
    j_width: int = 513
    # None preserves the exact single-stream v1 module/state-dict layout.
    secondary_j_width: int | None = None
    generator_context_width: int = 384
    mtp_nodes: int = 6
    mtp_state_channels: int = 1
    mtp_state_width: int = 2048

    model_width: int = 192
    attention_heads: int = 6
    feedforward_width: int = 768
    layer_blocks: int = 2
    mtp_blocks: int = 1
    fusion_blocks: int = 2
    output_rank: int = 64
    dropout: float = 0.05

    # Opt-in MTP-path corrections.  Disabled values deliberately reproduce
    # the original v1 modules, parameter names, and forward path.
    mtp_diagonal_from_local: bool = False
    mtp_null_only_when_all_missing: bool = False
    mtp_horizon_depth_attention_bias: bool = False

    def validate(self) -> None:
        positive = {
            name: int(getattr(self, name))
            for name in (
                "experts",
                "layers",
                "horizons",
                "history",
                "j_width",
                "generator_context_width",
                "mtp_nodes",
                "mtp_state_channels",
                "mtp_state_width",
                "model_width",
                "attention_heads",
                "feedforward_width",
                "layer_blocks",
                "mtp_blocks",
                "fusion_blocks",
                "output_rank",
            )
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive full-router dimensions required: {invalid}")
        if self.secondary_j_width is not None and self.secondary_j_width <= 0:
            raise ValueError("secondary_j_width must be positive when enabled")
        if self.horizons != 8:
            raise ValueError("full-router v1 emits direct H1-H8 predictions")
        if self.model_width % self.attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Old single-stream checkpoints predate this optional field. Omitting
        # its disabled value preserves exact config/resume compatibility.
        if result["secondary_j_width"] is None:
            result.pop("secondary_j_width")
        # Old checkpoints predate these switches.  Omit disabled values so
        # their serialized model configs remain byte-for-byte compatible.
        for name in (
            "mtp_diagonal_from_local",
            "mtp_null_only_when_all_missing",
            "mtp_horizon_depth_attention_bias",
        ):
            if result[name] is False:
                result.pop(name)
        return result


@dataclass
class JSpaceRouterForecasterOutput:
    future_router_scores: torch.Tensor
    delta: torch.Tensor
    base_router_scores: torch.Tensor
    horizon_layer_context: torch.Tensor
    source_weights: torch.Tensor
    target_stream_weights: torch.Tensor


class _ResidualSwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.input = nn.Linear(width, hidden * 2)
        self.output = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, value = self.input(self.norm(inputs)).chunk(2, dim=-1)
        update = self.output(F.silu(gate) * value)
        return inputs + self.dropout(update)


def _encoder_layer(config: JSpaceRouterForecasterConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.model_width,
        nhead=config.attention_heads,
        dim_feedforward=config.feedforward_width,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class CausalRouteJHistoryEncoder(nn.Module):
    """Fuse lagged J states and target routes, then mix across layers."""

    def __init__(self, config: JSpaceRouterForecasterConfig) -> None:
        super().__init__()
        self.config = config
        self.j_projection = nn.Sequential(
            nn.RMSNorm(config.j_width),
            nn.Linear(config.j_width, config.model_width),
        )
        self.secondary_target_projection: nn.Sequential | None = None
        self.target_stream_gate: nn.Sequential | None = None
        if config.secondary_j_width is not None:
            # Streams are normalized/projected independently. Only their
            # projected representations enter the learned per-cell gate.
            self.secondary_target_projection = nn.Sequential(
                nn.RMSNorm(config.secondary_j_width),
                nn.Linear(config.secondary_j_width, config.model_width),
            )
            self.target_stream_gate = nn.Sequential(
                nn.RMSNorm(config.model_width * 2),
                nn.Linear(config.model_width * 2, 2),
            )
            gate_linear = self.target_stream_gate[-1]
            nn.init.zeros_(gate_linear.weight)
            nn.init.zeros_(gate_linear.bias)
        self.route_projection = nn.Sequential(
            nn.LayerNorm(config.experts),
            nn.Linear(config.experts, config.model_width),
        )
        self.cell_fusion = nn.Sequential(
            nn.RMSNorm(config.model_width * 2),
            nn.Linear(config.model_width * 2, config.model_width),
            nn.SiLU(),
        )
        self.lag_embedding = nn.Embedding(config.history, config.model_width)
        self.layer_embedding = nn.Embedding(config.layers, config.model_width)
        self.temporal_bias = nn.Parameter(torch.zeros(config.layers, config.history))
        self.temporal_score = nn.Sequential(
            nn.RMSNorm(config.model_width), nn.Linear(config.model_width, 1)
        )
        self.layer_blocks = nn.ModuleList(
            [_encoder_layer(config) for _ in range(config.layer_blocks)]
        )
        self.norm = nn.RMSNorm(config.model_width)

    def forward(
        self,
        j_states: torch.Tensor,
        route_history: torch.Tensor,
        mask: torch.Tensor,
        secondary_j_states: torch.Tensor | None = None,
        secondary_mask: torch.Tensor | None = None,
        *,
        return_stream_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if j_states.ndim != 4:
            raise ValueError("j_states must have shape [B,T,L,D_J]")
        batch, history, layers, j_width = j_states.shape
        expected = (self.config.history, self.config.layers, self.config.j_width)
        if (history, layers, j_width) != expected:
            raise ValueError(
                f"J history geometry {(history, layers, j_width)} disagrees with {expected}"
            )
        if route_history.shape != (
            batch,
            history,
            layers,
            self.config.experts,
        ):
            raise ValueError("route_history must have shape [B,T,L,E]")
        if mask.shape != (batch, history, layers):
            raise ValueError("history mask must have shape [B,T,L]")
        mask = mask.bool()
        # FullRouterForecastData constructs this invariant on CPU. Retain the
        # assertion for direct CPU/synthetic callers without synchronizing
        # every production CUDA forward.
        if not mask.is_cuda and not bool(mask.any(dim=1).all()):
            raise ValueError("every layer requires at least one causal history cell")

        primary_target = self.j_projection(j_states.float())
        if self.config.secondary_j_width is None:
            if secondary_j_states is not None or secondary_mask is not None:
                raise ValueError(
                    "secondary target states require a dual-stream model config"
                )
            target_state = primary_target
            stream_weights = primary_target.new_ones(
                batch, history, layers, 1
            )
        else:
            if secondary_j_states is None or secondary_mask is None:
                raise KeyError("dual-stream model requires secondary target states/mask")
            if secondary_j_states.shape != (
                batch,
                history,
                layers,
                self.config.secondary_j_width,
            ):
                raise ValueError("secondary target-state history geometry is invalid")
            if secondary_mask.shape != mask.shape:
                raise ValueError("secondary target-state mask geometry is invalid")
            # Production masks are constructed from the same audited history
            # rows. Keep the equality assertion for CPU/synthetic callers
            # without introducing a per-forward CUDA synchronization.
            if not mask.is_cuda and not torch.equal(secondary_mask.bool(), mask):
                raise ValueError(
                    "primary and secondary target streams must share the audited mask"
                )
            if self.secondary_target_projection is None or self.target_stream_gate is None:
                raise AssertionError("dual-stream projection modules were not constructed")
            secondary_target = self.secondary_target_projection(
                secondary_j_states.float()
            )
            stream_weights = torch.softmax(
                self.target_stream_gate(
                    torch.cat([primary_target, secondary_target], dim=-1)
                ),
                dim=-1,
            )
            target_state = (
                primary_target * stream_weights[..., 0, None]
                + secondary_target * stream_weights[..., 1, None]
            )

        centered_routes = route_history.float() - route_history.float().mean(
            dim=-1, keepdim=True
        )
        cells = self.cell_fusion(
            torch.cat(
                [target_state, self.route_projection(centered_routes)],
                dim=-1,
            )
        )
        lag_ids = torch.arange(history, device=j_states.device)
        layer_ids = torch.arange(layers, device=j_states.device)
        cells = (
            cells
            + self.lag_embedding(lag_ids)[None, :, None]
            + self.layer_embedding(layer_ids)[None, None]
        )
        temporal_logits = self.temporal_score(cells).squeeze(-1)
        temporal_logits = temporal_logits + self.temporal_bias.T[None]
        temporal_logits = temporal_logits.masked_fill(~mask, -torch.inf)
        weights = torch.softmax(temporal_logits, dim=1)
        pooled = (cells * weights.unsqueeze(-1)).sum(dim=1)
        for block in self.layer_blocks:
            pooled = block(pooled)
        encoded = self.norm(pooled)
        if return_stream_weights:
            return encoded, stream_weights
        return encoded


class SeparatedNativeMTPMemory(nn.Module):
    """Keep native MTP hidden and router evidence separate until fusion."""

    def __init__(self, config: JSpaceRouterForecasterConfig) -> None:
        super().__init__()
        self.config = config
        self.state_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.RMSNorm(config.mtp_state_width),
                    nn.Linear(config.mtp_state_width, config.model_width),
                )
                for _ in range(config.mtp_state_channels)
            ]
        )
        self.channel_embedding = nn.Embedding(
            config.mtp_state_channels, config.model_width
        )
        self.state_scale = nn.Linear(config.mtp_state_channels, config.model_width)
        self.router_projection = nn.Sequential(
            nn.LayerNorm(config.experts),
            nn.Linear(config.experts, config.model_width),
        )
        self.router_scale = nn.Linear(2, config.model_width)
        self.hidden_depth_embedding = nn.Embedding(
            config.mtp_nodes, config.model_width
        )
        self.router_depth_embedding = nn.Embedding(
            config.mtp_nodes, config.model_width
        )
        self.hidden_blocks = nn.ModuleList(
            [_encoder_layer(config) for _ in range(config.mtp_blocks)]
        )
        self.router_blocks = nn.ModuleList(
            [_encoder_layer(config) for _ in range(config.mtp_blocks)]
        )
        self.hidden_missing = nn.Parameter(torch.zeros(1, 1, config.model_width))
        self.router_missing = nn.Parameter(torch.zeros(1, 1, config.model_width))
        self.hidden_norm = nn.RMSNorm(config.model_width)
        self.router_norm = nn.RMSNorm(config.model_width)

    def forward(
        self,
        states: torch.Tensor,
        router_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if states.ndim == 3 and self.config.mtp_state_channels == 1:
            states = states.unsqueeze(2)
        if states.ndim != 4:
            raise ValueError("mtp_states must have shape [B,N,S,D_M]")
        batch, nodes, channels, width = states.shape
        if (nodes, channels, width) != (
            self.config.mtp_nodes,
            self.config.mtp_state_channels,
            self.config.mtp_state_width,
        ):
            raise ValueError("MTP hidden geometry disagrees with model config")
        if router_logits.shape != (batch, nodes, self.config.experts):
            raise ValueError("MTP router logits must have shape [B,N,E]")
        if mask.shape != (batch, nodes):
            raise ValueError("mtp_mask must have shape [B,N]")
        mask = mask.bool()

        state_values = states.float()
        state_parts = [
            projection(state_values[:, :, channel])
            + self.channel_embedding.weight[channel]
            for channel, projection in enumerate(self.state_projections)
        ]
        hidden = torch.stack(state_parts, dim=0).mean(dim=0)
        state_rms = torch.log1p(
            state_values.square().mean(dim=-1).sqrt().clamp_min(0.0)
        )
        hidden = hidden + self.state_scale(state_rms)

        router_values = router_logits.float()
        centered = router_values - router_values.mean(dim=-1, keepdim=True)
        router = self.router_projection(centered)
        top2 = centered.topk(2, dim=-1).values
        router_statistics = torch.stack(
            [
                torch.log1p(centered.square().mean(dim=-1).sqrt()),
                top2[..., 0] - top2[..., 1],
            ],
            dim=-1,
        )
        router = router + self.router_scale(router_statistics)

        depth_ids = torch.arange(nodes, device=states.device)
        hidden = hidden + self.hidden_depth_embedding(depth_ids)[None]
        router = router + self.router_depth_embedding(depth_ids)[None]
        # These node-local encodings have not mixed information across draft
        # depths.  The corrected diagonal path consumes them, while the
        # all-depth attention path continues to consume contextual memories.
        local_hidden = self.hidden_norm(hidden)
        local_router = self.router_norm(router)
        null_padding = (
            mask.any(dim=1, keepdim=True)
            if self.config.mtp_null_only_when_all_missing
            else torch.zeros(batch, 1, dtype=torch.bool, device=mask.device)
        )
        padding = torch.cat(
            [~mask, null_padding],
            dim=1,
        )
        hidden = torch.cat(
            [hidden, self.hidden_missing.expand(batch, -1, -1)], dim=1
        )
        router = torch.cat(
            [router, self.router_missing.expand(batch, -1, -1)], dim=1
        )
        for block in self.hidden_blocks:
            hidden = block(hidden, src_key_padding_mask=padding)
        for block in self.router_blocks:
            router = block(router, src_key_padding_mask=padding)
        return (
            self.hidden_norm(hidden),
            self.router_norm(router),
            padding,
            local_hidden,
            local_router,
        )


class HorizonLayerResidualHead(nn.Module):
    """Independent low-rank direct-logit correction for every (horizon, layer)."""

    def __init__(self, config: JSpaceRouterForecasterConfig) -> None:
        super().__init__()
        self.config = config
        self.down = nn.Parameter(
            torch.empty(
                config.horizons,
                config.layers,
                config.model_width,
                config.output_rank,
            )
        )
        self.up = nn.Parameter(
            torch.empty(
                config.horizons,
                config.layers,
                config.output_rank,
                config.experts,
            )
        )
        self.bias = nn.Parameter(
            torch.zeros(config.horizons, config.layers, config.experts)
        )
        for horizon in range(config.horizons):
            for layer in range(config.layers):
                nn.init.xavier_uniform_(self.down[horizon, layer])
        # Exact epoch-zero frozen-HARP passthrough.  Bias and up receive
        # gradients immediately; down begins learning once up becomes nonzero.
        nn.init.zeros_(self.up)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        expected = (
            self.config.horizons,
            self.config.layers,
            self.config.model_width,
        )
        if context.ndim != 4 or tuple(context.shape[1:]) != expected:
            raise ValueError("output-head context must have shape [B,H,L,D]")
        low = F.silu(torch.einsum("bhlw,hlwr->bhlr", context, self.down))
        return (
            torch.einsum("bhlr,hlre->bhle", low, self.up)
            + self.bias[None]
        )


class JSpaceFullRouterForecaster(nn.Module):
    """Direct H1--H8 predictor over the complete expert namespace."""

    SOURCE_COUNT = 5

    def __init__(self, config: JSpaceRouterForecasterConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.history_encoder = CausalRouteJHistoryEncoder(config)
        self.generator_projection = nn.Sequential(
            nn.RMSNorm(config.generator_context_width),
            nn.Linear(config.generator_context_width, config.model_width),
        )
        self.horizon_embedding = nn.Embedding(config.horizons, config.model_width)
        self.layer_embedding = nn.Embedding(config.layers, config.model_width)
        self.position_projection = nn.Sequential(
            nn.Linear(3, config.model_width), nn.SiLU()
        )
        self.local_fusion = nn.Sequential(
            nn.RMSNorm(config.model_width * 5),
            nn.Linear(config.model_width * 5, config.model_width),
            nn.SiLU(),
        )

        self.mtp_encoder = SeparatedNativeMTPMemory(config)
        self.hidden_attention = nn.MultiheadAttention(
            config.model_width,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.router_attention = nn.MultiheadAttention(
            config.model_width,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.hidden_horizon_depth_attention_bias: nn.Parameter | None = None
        self.router_horizon_depth_attention_bias: nn.Parameter | None = None
        if config.mtp_horizon_depth_attention_bias:
            # Separate zero-initialized priors let each head learn a direct
            # horizon-to-draft-depth relation without removing content-based
            # attention.  The learned null token receives no additive prior.
            bias_shape = (
                config.attention_heads,
                config.horizons,
                config.mtp_nodes,
            )
            self.hidden_horizon_depth_attention_bias = nn.Parameter(
                torch.zeros(bias_shape)
            )
            self.router_horizon_depth_attention_bias = nn.Parameter(
                torch.zeros(bias_shape)
            )
        self.query_norm = nn.RMSNorm(config.model_width)
        self.memory_norm = nn.RMSNorm(config.model_width)
        self.read_norm = nn.RMSNorm(config.model_width)
        self.diagonal_missing = nn.Parameter(
            torch.zeros(2, config.horizons, config.model_width)
        )
        self.source_score = nn.Sequential(
            nn.RMSNorm(config.model_width),
            nn.Linear(config.model_width, config.model_width // 2),
            nn.Tanh(),
            nn.Linear(config.model_width // 2, 1),
        )
        self.source_fusion = nn.Sequential(
            nn.RMSNorm(config.model_width * (self.SOURCE_COUNT + 1)),
            nn.Linear(
                config.model_width * (self.SOURCE_COUNT + 1), config.model_width
            ),
            nn.SiLU(),
        )
        self.fusion_blocks = nn.ModuleList(
            [
                _ResidualSwiGLU(
                    config.model_width, config.feedforward_width, config.dropout
                )
                for _ in range(config.fusion_blocks)
            ]
        )
        self.context_norm = nn.RMSNorm(config.model_width)
        self.output_head = HorizonLayerResidualHead(config)

    def _mtp_attention_mask(
        self,
        horizon_depth_bias: torch.Tensor,
        padding: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Expand learned [head,horizon,depth] bias for batched HxL queries."""

        heads = self.config.attention_heads
        horizons = self.config.horizons
        layers = self.config.layers
        nodes = self.config.mtp_nodes
        if horizon_depth_bias.shape != (heads, horizons, nodes):
            raise ValueError("MTP attention-bias geometry disagrees with config")
        batch = padding.shape[0]
        if padding.shape != (batch, nodes + 1):
            raise ValueError("MTP padding geometry disagrees with config")
        # The final column addresses the learned null token.  It is governed
        # exclusively by the availability mask, not by a learned depth prior.
        with_null = torch.cat(
            [
                horizon_depth_bias.to(dtype=dtype),
                horizon_depth_bias.new_zeros(heads, horizons, 1).to(dtype=dtype),
            ],
            dim=-1,
        )
        expanded = with_null[:, :, None, :].expand(
            heads, horizons, layers, nodes + 1
        )
        expanded = expanded.reshape(heads, horizons * layers, nodes + 1)
        expanded = expanded[None].expand(batch, -1, -1, -1).clone()
        expanded = expanded.masked_fill(
            padding[:, None, None, :], -torch.inf
        )
        return expanded.reshape(
            batch * heads, horizons * layers, nodes + 1
        )

    def _diagonal_sources(
        self,
        hidden_memory: torch.Tensor,
        router_memory: torch.Tensor,
        mtp_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = hidden_memory.shape[0]
        hidden_values = []
        router_values = []
        for horizon in range(self.config.horizons):
            if horizon < self.config.mtp_nodes:
                available = mtp_mask[:, horizon].bool()[:, None]
                hidden = torch.where(
                    available,
                    hidden_memory[:, horizon],
                    self.diagonal_missing[0, horizon][None],
                )
                router = torch.where(
                    available,
                    router_memory[:, horizon],
                    self.diagonal_missing[1, horizon][None],
                )
            else:
                hidden = self.diagonal_missing[0, horizon][None].expand(
                    batch, -1
                )
                router = self.diagonal_missing[1, horizon][None].expand(
                    batch, -1
                )
            hidden_values.append(hidden)
            router_values.append(router)
        return torch.stack(hidden_values, dim=1), torch.stack(router_values, dim=1)

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> JSpaceRouterForecasterOutput:
        dual_stream = self.config.secondary_j_width is not None
        causal = causal_router_forecaster_inputs(
            batch, secondary_target_enabled=dual_stream
        )
        base = causal["base_router_scores"].float()
        expected_base = (
            self.config.horizons,
            self.config.layers,
            self.config.experts,
        )
        if base.ndim != 4 or tuple(base.shape[1:]) != expected_base:
            raise ValueError("base_router_scores must have shape [B,H,L,E]")
        batch_size = base.shape[0]
        # Production batches are validated on the CPU by FullRouterForecastData
        # before transfer. Repeating this check on CUDA would synchronize every
        # forward pass; retain it for direct CPU and synthetic callers.
        if not base.is_cuda and not bool(torch.isfinite(base).all()):
            raise ValueError("base_router_scores contain non-finite values")
        if causal["route_mask"].shape != causal["j_mask"].shape:
            raise ValueError("J and route history masks must have identical geometry")

        history, target_stream_weights = self.history_encoder(
            causal["j_states"],
            causal["route_history"],
            causal["j_mask"] & causal["route_mask"],
            causal.get("secondary_target_states"),
            causal.get("secondary_target_mask"),
            return_stream_weights=True,
        )
        generator = causal["generator_context"].float()
        if generator.shape != (
            batch_size,
            self.config.horizons,
            self.config.layers,
            self.config.generator_context_width,
        ):
            raise ValueError("generator_context must have shape [B,H,L,D_G]")
        generator = self.generator_projection(generator)
        horizon_ids = torch.arange(self.config.horizons, device=base.device)
        layer_ids = torch.arange(self.config.layers, device=base.device)
        horizon = self.horizon_embedding(horizon_ids)[None, :, None].expand(
            batch_size, -1, self.config.layers, -1
        )
        layer = self.layer_embedding(layer_ids)[None, None].expand(
            batch_size, self.config.horizons, -1, -1
        )
        expanded_history = history[:, None].expand(
            -1, self.config.horizons, -1, -1
        )
        position = causal["within_request"].float()
        if position.shape != (batch_size,):
            raise ValueError("within_request must have shape [B]")
        position_features = torch.stack(
            [position, position.square(), torch.sin(math.pi * position)], dim=-1
        )
        position_hidden = self.position_projection(position_features)
        position_hidden = position_hidden[:, None, None].expand(
            -1, self.config.horizons, self.config.layers, -1
        )
        local = self.local_fusion(
            torch.cat(
                [expanded_history, generator, horizon, layer, position_hidden], dim=-1
            )
        )

        (
            hidden_memory,
            router_memory,
            padding,
            local_hidden_memory,
            local_router_memory,
        ) = self.mtp_encoder(
            causal["mtp_states"],
            causal["mtp_router_logits"],
            causal["mtp_mask"],
        )
        query = local.reshape(
            batch_size,
            self.config.horizons * self.config.layers,
            self.config.model_width,
        )
        normalized_query = self.query_norm(query)
        hidden_attention_mask = None
        router_attention_mask = None
        attention_padding = padding
        if self.config.mtp_horizon_depth_attention_bias:
            if (
                self.hidden_horizon_depth_attention_bias is None
                or self.router_horizon_depth_attention_bias is None
            ):
                raise AssertionError("enabled MTP attention bias was not constructed")
            hidden_attention_mask = self._mtp_attention_mask(
                self.hidden_horizon_depth_attention_bias,
                padding,
                dtype=normalized_query.dtype,
            )
            router_attention_mask = self._mtp_attention_mask(
                self.router_horizon_depth_attention_bias,
                padding,
                dtype=normalized_query.dtype,
            )
            # Padding is already folded into the per-sample additive masks.
            attention_padding = None
        hidden_read, _ = self.hidden_attention(
            normalized_query,
            self.memory_norm(hidden_memory),
            self.memory_norm(hidden_memory),
            key_padding_mask=attention_padding,
            attn_mask=hidden_attention_mask,
            need_weights=False,
        )
        router_read, _ = self.router_attention(
            normalized_query,
            self.memory_norm(router_memory),
            self.memory_norm(router_memory),
            key_padding_mask=attention_padding,
            attn_mask=router_attention_mask,
            need_weights=False,
        )
        target_shape = (
            batch_size,
            self.config.horizons,
            self.config.layers,
            self.config.model_width,
        )
        hidden_read = self.read_norm(hidden_read.reshape(target_shape))
        router_read = self.read_norm(router_read.reshape(target_shape))
        diagonal_hidden, diagonal_router = self._diagonal_sources(
            (
                local_hidden_memory
                if self.config.mtp_diagonal_from_local
                else hidden_memory
            ),
            (
                local_router_memory
                if self.config.mtp_diagonal_from_local
                else router_memory
            ),
            causal["mtp_mask"],
        )
        diagonal_hidden = diagonal_hidden[:, :, None].expand(
            -1, -1, self.config.layers, -1
        )
        diagonal_router = diagonal_router[:, :, None].expand(
            -1, -1, self.config.layers, -1
        )

        sources = torch.stack(
            [local, hidden_read, router_read, diagonal_hidden, diagonal_router],
            dim=-2,
        )
        source_weights = torch.softmax(
            self.source_score(sources).squeeze(-1), dim=-1
        )
        weighted = (sources * source_weights.unsqueeze(-1)).sum(dim=-2)
        context = self.source_fusion(
            torch.cat(
                [local, hidden_read, router_read, diagonal_hidden, diagonal_router, weighted],
                dim=-1,
            )
        )
        for block in self.fusion_blocks:
            context = block(context)
        context = self.context_norm(context)
        delta = self.output_head(context)
        scores = base + delta
        return JSpaceRouterForecasterOutput(
            future_router_scores=scores,
            delta=delta,
            base_router_scores=base,
            horizon_layer_context=context,
            source_weights=source_weights,
            target_stream_weights=target_stream_weights,
        )

    def inference_dict(
        self, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        output = self(batch)
        return {
            "future_router_scores": output.future_router_scores,
            "predicted_top8": torch.argsort(
                output.future_router_scores,
                dim=-1,
                descending=True,
                stable=True,
            )[..., :8],
            "delta": output.delta,
            "source_weights": output.source_weights,
            "target_stream_weights": output.target_stream_weights,
        }


@dataclass(frozen=True)
class FullRouterLossConfig:
    """Top-8-aligned objective with auxiliary full-distribution supervision."""

    temperature: float = 1.0
    router_kl: float = 0.25
    boundary: float = 1.0
    predicted_boundary: float = 1.0
    top8_swap: float = 0.0
    full_membership: float = 0.25
    centered_score: float = 0.05
    # Opt-in trust-region terms. Disabled values are omitted from serialized
    # configs so legacy experiments retain their exact schema and arithmetic.
    base_kl: float = 0.0
    delta_l2: float = 0.0
    relative_regret: float = 0.0
    hard_negative_end_rank: int = 32
    predicted_negative_count: int = 16
    margin: float = 0.0
    horizon_weights: tuple[float, ...] = (
        1.0,
        1.0,
        1.0,
        1.0,
        0.5,
        0.5,
        0.5,
        0.5,
    )

    def validate(self, horizons: int, experts: int) -> None:
        if self.temperature <= 0:
            raise ValueError("loss temperature must be positive")
        if len(self.horizon_weights) != horizons or not any(self.horizon_weights):
            raise ValueError("loss needs one nonzero weight per configured horizon")
        if any(value < 0 for value in self.horizon_weights):
            raise ValueError("horizon weights cannot be negative")
        if not 8 < self.hard_negative_end_rank <= experts:
            raise ValueError("hard-negative end rank must lie in [9,E]")
        if not 1 <= self.predicted_negative_count <= experts - 8:
            raise ValueError("predicted negative count lies outside namespace")
        for name in (
            "router_kl",
            "boundary",
            "predicted_boundary",
            "top8_swap",
            "full_membership",
            "centered_score",
            "base_kl",
            "delta_l2",
            "relative_regret",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} loss weight must be finite and non-negative"
                )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Preserve resumability of v2 checkpoints written before this optional
        # objective existed. A nonzero experiment is still explicit in its
        # checkpoint and manifest.
        if result["top8_swap"] == 0.0:
            result.pop("top8_swap")
        for name in ("base_kl", "delta_l2", "relative_regret"):
            if result[name] == 0.0:
                result.pop(name)
        result["horizon_weights"] = list(self.horizon_weights)
        return result


@dataclass
class FullRouterLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    metric_names: tuple[str, ...]
    metric_values: torch.Tensor

    @property
    def metrics(self) -> dict[str, float]:
        """Materialize diagnostics only when explicitly requested.

        Training retains ``metric_values`` on-device until the epoch has
        finished. This compatibility property keeps diagnostic and unit-test
        callers on the historical mapping API without synchronizing inside
        the loss itself.
        """

        values = self.metric_values.detach().cpu().tolist()
        return dict(zip(self.metric_names, (float(value) for value in values)))


def _masked_horizon_mean(
    values: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    while values.ndim > 3:
        values = values.mean(dim=-1)
    mask = valid[:, :, None].expand_as(values)
    return (values * mask).sum(dim=(0, 2)) / mask.sum(dim=(0, 2)).clamp_min(1)


def _stable_masked_topk_ids(
    values: torch.Tensor,
    excluded: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Select score-descending, expert-ID-ascending non-excluded IDs."""

    if values.shape != excluded.shape or excluded.dtype is not torch.bool:
        raise ValueError("stable top-k values/mask geometry is invalid")
    if not 1 <= count <= values.shape[-1]:
        raise ValueError("stable top-k count lies outside the namespace")
    masked = values.masked_fill(excluded, -torch.inf)
    return torch.argsort(
        masked,
        dim=-1,
        descending=True,
        stable=True,
    )[..., :count]


def _top8_swap_rows(
    scores: torch.Tensor,
    target_top8: torch.Tensor,
    membership: torch.Tensor,
    margin: float,
    *,
    predicted_top8: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recall-aligned loss for the swaps needed to repair predicted top-8.

    Only authoritative positives missing from the current predicted set and
    non-target experts occupying their slots participate. Consequently the
    term is exactly zero for an exact top-8 set and is invariant to adding any
    row-wise constant to the scores. Stable sorting matches the evaluation
    tie rule: score descending, then expert ID ascending.
    """

    if predicted_top8 is None:
        predicted_top8 = torch.argsort(
            scores,
            dim=-1,
            descending=True,
            stable=True,
        )[..., :8]
    elif predicted_top8.shape != target_top8.shape:
        raise ValueError("predicted top-8 geometry disagrees with target top-8")
    predicted_membership = torch.zeros_like(membership)
    predicted_membership.scatter_(-1, predicted_top8, True)

    missing = ~predicted_membership.gather(-1, target_top8)
    intruding = ~membership.gather(-1, predicted_top8)
    missing_scores = scores.gather(-1, target_top8)
    intruding_scores = scores.gather(-1, predicted_top8)
    pair_mask = missing[..., :, None] & intruding[..., None, :]
    pair_loss = F.softplus(
        intruding_scores[..., None, :]
        - missing_scores[..., :, None]
        + float(margin)
    ).masked_fill(~pair_mask, 0.0)

    # The two set differences have equal cardinality n. Dividing by 8*n
    # makes a row's scale proportional to its slot-error fraction n/8 while
    # averaging the n^2 relevant swap comparisons.
    intruder_count = intruding.sum(dim=-1).clamp_min(1)
    return pair_loss.sum(dim=(-2, -1)) / (8.0 * intruder_count)


def full_router_forecaster_loss(
    output: JSpaceRouterForecasterOutput | torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    config: FullRouterLossConfig | None = None,
) -> FullRouterLossOutput:
    """Compute full-router KL and top-8 boundary-focused supervision."""

    structured_output = isinstance(output, JSpaceRouterForecasterOutput)
    scores = (
        output.future_router_scores
        if structured_output
        else output
    ).float()
    target_top8 = batch["target_top8"].long()
    valid = batch["valid_future"].bool()
    if scores.ndim != 4:
        raise ValueError("student scores must have [B,H,L,E] geometry")
    if target_top8.shape != scores.shape[:-1] + (8,):
        raise ValueError("target_top8 must have shape [B,H,L,8]")
    if valid.shape != scores.shape[:2]:
        raise ValueError("valid_future must have shape [B,H]")
    cfg = config or FullRouterLossConfig()
    cfg.validate(scores.shape[1], scores.shape[-1])

    # Do not even read the frozen baseline on a legacy/disabled path. This
    # preserves the historical values, gradients, and accepted batch schema.
    base: torch.Tensor | None = None
    if cfg.base_kl or cfg.relative_regret or (cfg.delta_l2 and not structured_output):
        base_value = (
            output.base_router_scores
            if structured_output
            else batch.get("base_router_scores")
        )
        if base_value is None:
            raise KeyError("enabled trust-region losses require base_router_scores")
        base = base_value.detach().float()
        if base.shape != scores.shape:
            raise ValueError("student/base must share [B,H,L,E] geometry")

    zero_h = scores.new_zeros(scores.shape[1])
    teacher_required = bool(cfg.router_kl or cfg.boundary or cfg.centered_score)
    teacher_value = batch.get("teacher_router_scores")
    if teacher_required and teacher_value is None:
        raise KeyError("enabled teacher-supervised losses require teacher_router_scores")
    teacher = teacher_value.float() if teacher_value is not None else None
    if teacher is not None and teacher.shape != scores.shape:
        raise ValueError("student/teacher must share [B,H,L,E] geometry")

    if cfg.router_kl:
        if teacher is None:
            raise AssertionError("router KL lost its required teacher")
        tau = float(cfg.temperature)
        kl_rows = F.kl_div(
            torch.log_softmax(scores / tau, dim=-1),
            torch.softmax(teacher / tau, dim=-1),
            reduction="none",
        ).sum(dim=-1) * tau**2
        kl_h = _masked_horizon_mean(kl_rows, valid)
    else:
        kl_h = zero_h

    top8_supervision = bool(
        cfg.boundary
        or cfg.predicted_boundary
        or cfg.top8_swap
        or cfg.full_membership
        or cfg.relative_regret
    )
    positives: torch.Tensor | None = None
    membership: torch.Tensor | None = None
    if top8_supervision:
        positives = scores.gather(-1, target_top8)
        # Raw BF16-expanded router logits contain exact cutoff ties. The stored
        # target_top8 is authoritative; never infer positives from raw-logit topk.
        membership = torch.zeros_like(scores, dtype=torch.bool)
        membership.scatter_(-1, target_top8, True)

    if cfg.boundary:
        if teacher is None or positives is None or membership is None:
            raise AssertionError("boundary loss lost required supervision")
        hard_negative_count = cfg.hard_negative_end_rank - 8
        hard_negative_ids = _stable_masked_topk_ids(
            teacher, membership, hard_negative_count
        )
        hard_negatives = scores.gather(-1, hard_negative_ids)
        boundary_rows = F.softplus(
            hard_negatives[..., None, :]
            - positives[..., :, None]
            + float(cfg.margin)
        )
        boundary_h = _masked_horizon_mean(boundary_rows, valid)
    else:
        boundary_h = zero_h

    predicted_top8: torch.Tensor | None = None
    if cfg.predicted_boundary and cfg.top8_swap:
        if membership is None:
            raise AssertionError("predicted ranking lost authoritative membership")
        predicted_order = torch.argsort(
            scores, dim=-1, descending=True, stable=True
        )
        predicted_top8 = predicted_order[..., :8]
        ordered_is_positive = membership.gather(-1, predicted_order)
        ordered_negatives = predicted_order.masked_select(
            ~ordered_is_positive
        ).reshape(predicted_order.shape[:-1] + (scores.shape[-1] - 8,))
        mined_negative_ids = ordered_negatives[
            ..., : cfg.predicted_negative_count
        ]
    elif cfg.predicted_boundary:
        if membership is None:
            raise AssertionError("predicted boundary lost authoritative membership")
        mined_negative_ids = _stable_masked_topk_ids(
            scores, membership, cfg.predicted_negative_count
        )

    if cfg.predicted_boundary:
        if positives is None:
            raise AssertionError("predicted boundary lost authoritative positives")
        mined_negatives = scores.gather(-1, mined_negative_ids)
        predicted_boundary_rows = F.softplus(
            mined_negatives[..., None, :]
            - positives[..., :, None]
            + float(cfg.margin)
        )
        predicted_boundary_h = _masked_horizon_mean(
            predicted_boundary_rows, valid
        )
    else:
        predicted_boundary_h = zero_h

    if cfg.top8_swap:
        if membership is None:
            raise AssertionError("top-8 swap lost authoritative membership")
        top8_swap_rows = _top8_swap_rows(
            scores,
            target_top8,
            membership,
            float(cfg.margin),
            predicted_top8=predicted_top8,
        )
        top8_swap_h = _masked_horizon_mean(top8_swap_rows, valid)
    else:
        top8_swap_h = zero_h

    if cfg.full_membership:
        if membership is None:
            raise AssertionError("membership loss lost authoritative membership")
        positive_bce = F.softplus(-scores).masked_fill(~membership, 0.0).sum(
            dim=-1
        ) / 8.0
        negative_bce = F.softplus(scores).masked_fill(membership, 0.0).sum(
            dim=-1
        ) / max(1, scores.shape[-1] - 8)
        membership_h = _masked_horizon_mean(
            0.5 * (positive_bce + negative_bce), valid
        )
    else:
        membership_h = zero_h

    if cfg.centered_score:
        if teacher is None:
            raise AssertionError("centered-score loss lost its required teacher")
        centered_scores = scores - scores.mean(dim=-1, keepdim=True)
        centered_teacher = teacher - teacher.mean(dim=-1, keepdim=True)
        centered_rows = F.smooth_l1_loss(
            centered_scores, centered_teacher, reduction="none"
        ).mean(dim=-1)
        centered_h = _masked_horizon_mean(centered_rows, valid)
    else:
        centered_h = zero_h

    optional_horizon_components: dict[str, torch.Tensor] = {}
    optional_coefficients: dict[str, float] = {}
    if cfg.base_kl:
        if base is None:
            raise AssertionError("base KL lost its frozen baseline")
        tau = float(cfg.temperature)
        base_kl_rows = F.kl_div(
            torch.log_softmax(scores / tau, dim=-1),
            torch.softmax(base / tau, dim=-1),
            reduction="none",
        ).sum(dim=-1) * tau**2
        optional_horizon_components["base_kl"] = _masked_horizon_mean(
            base_kl_rows, valid
        )
        optional_coefficients["base_kl"] = cfg.base_kl

    if cfg.delta_l2:
        if structured_output:
            delta = output.delta.float()
        else:
            if base is None:
                raise AssertionError("delta L2 lost its frozen baseline")
            delta = scores - base
        if delta.shape != scores.shape:
            raise ValueError("student delta must share [B,H,L,E] geometry")
        centered_delta = delta - delta.mean(dim=-1, keepdim=True)
        delta_l2_rows = centered_delta.square().mean(dim=-1)
        optional_horizon_components["delta_l2"] = _masked_horizon_mean(
            delta_l2_rows, valid
        )
        optional_coefficients["delta_l2"] = cfg.delta_l2

    if cfg.relative_regret:
        if base is None or membership is None:
            raise AssertionError("relative regret lost its frozen baseline/targets")
        fixed_negative_ids = _stable_masked_topk_ids(
            base, membership, cfg.predicted_negative_count
        )
        new_positive_scores = scores.gather(-1, target_top8)
        base_positive_scores = base.gather(-1, target_top8)
        new_negative_scores = scores.gather(-1, fixed_negative_ids)
        base_negative_scores = base.gather(-1, fixed_negative_ids)
        new_boundary_rows = F.softplus(
            new_negative_scores[..., None, :]
            - new_positive_scores[..., :, None]
            + float(cfg.margin)
        ).mean(dim=(-2, -1))
        base_boundary_rows = F.softplus(
            base_negative_scores[..., None, :]
            - base_positive_scores[..., :, None]
            + float(cfg.margin)
        ).mean(dim=(-2, -1))
        relative_regret_rows = F.relu(new_boundary_rows - base_boundary_rows)
        optional_horizon_components["relative_regret"] = _masked_horizon_mean(
            relative_regret_rows, valid
        )
        optional_coefficients["relative_regret"] = cfg.relative_regret

    weights = scores.new_tensor(cfg.horizon_weights)
    weights = weights / weights.sum()
    horizon_components = {
        "router_kl": kl_h,
        "boundary": boundary_h,
        "predicted_boundary": predicted_boundary_h,
        "top8_swap": top8_swap_h,
        "full_membership": membership_h,
        "centered_score": centered_h,
    }
    components = {
        name: (values * weights).sum()
        for name, values in horizon_components.items()
    }
    components.update(
        {
            name: (values * weights).sum()
            for name, values in optional_horizon_components.items()
        }
    )
    coefficients = {
        "router_kl": cfg.router_kl,
        "boundary": cfg.boundary,
        "predicted_boundary": cfg.predicted_boundary,
        "top8_swap": cfg.top8_swap,
        "full_membership": cfg.full_membership,
        "centered_score": cfg.centered_score,
    }
    coefficients.update(optional_coefficients)
    total = sum(components[name] * coefficients[name] for name in components)
    if not total.requires_grad:
        total = total + scores.sum() * 0.0
    metric_names = [*components, "loss"]
    metric_values = [*components.values(), total]
    for index in range(scores.shape[1]):
        metric_names.extend(
            (
                f"router_kl_h{index + 1}",
                f"boundary_h{index + 1}",
                f"top8_swap_h{index + 1}",
            )
        )
        metric_values.extend((kl_h[index], boundary_h[index], top8_swap_h[index]))
        for name, values in optional_horizon_components.items():
            metric_names.append(f"{name}_h{index + 1}")
            metric_values.append(values[index])
    return FullRouterLossOutput(
        total=total,
        components=components,
        metric_names=tuple(metric_names),
        metric_values=torch.stack(metric_values).detach(),
    )


__all__ = [
    "CausalRouteJHistoryEncoder",
    "FullRouterLossConfig",
    "FullRouterLossOutput",
    "HorizonLayerResidualHead",
    "JSPACE_ROUTER_FORECASTER_SCHEMA",
    "JSpaceFullRouterForecaster",
    "JSpaceRouterForecasterConfig",
    "JSpaceRouterForecasterOutput",
    "SeparatedNativeMTPMemory",
    "_stable_masked_topk_ids",
    "_top8_swap_rows",
    "full_router_forecaster_loss",
]
