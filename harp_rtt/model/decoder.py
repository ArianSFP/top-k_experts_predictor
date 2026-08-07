"""Six-repetition causal horizon/layer endpoint decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import CrossAttentionResidual, SelfAttentionResidual, SwiGLUResidual, safe_padding_mask
from .config import HARPRTTConfig
from .tree import TreeEncoding


class _AxialDecoderBlock(nn.Module):
    def __init__(self, config: HARPRTTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer_attention = SelfAttentionResidual(
            config.model_width,
            config.attention_heads,
            config.decoder_ffn_width,
            config.dropout,
        )
        self.horizon_attention = SelfAttentionResidual(
            config.model_width,
            config.attention_heads,
            config.decoder_ffn_width,
            config.dropout,
        )
        self.tree_attention = CrossAttentionResidual(
            config.model_width,
            config.tree_width,
            config.attention_heads,
            config.decoder_ffn_width,
            config.dropout,
        )
        self.feedforward = SwiGLUResidual(
            config.model_width, config.decoder_ffn_width, config.dropout
        )

    def forward(self, hidden: Tensor, tree: TreeEncoding) -> Tensor:
        batch, horizons, layers, width = hidden.shape
        layer_axis = hidden.reshape(batch * horizons, layers, width)
        layer_axis = self.layer_attention(layer_axis)
        hidden = layer_axis.reshape(batch, horizons, layers, width)

        horizon_axis = hidden.permute(0, 2, 1, 3).reshape(
            batch * layers, horizons, width
        )
        causal = torch.triu(
            torch.ones(
                horizons, horizons, dtype=torch.bool, device=hidden.device
            ),
            diagonal=1,
        )
        horizon_axis = self.horizon_attention(
            horizon_axis, attention_mask=causal
        )
        hidden = horizon_axis.reshape(batch, layers, horizons, width).permute(
            0, 2, 1, 3
        )

        # Every endpoint can attend to every valid causal tree node.  Horizon
        # embeddings and node depth embeddings provide a learned soft
        # alignment; neighboring depths are deliberately not hard-excluded.
        queries = hidden.reshape(batch, horizons * layers, width)
        nodes = tree.states.shape[1]
        allowed = tree.available[:, None, :].expand(-1, horizons * layers, -1)
        safe, padding = safe_padding_mask(tree.available)
        missing = ~allowed.any(dim=-1)
        if missing.any():
            fallback = safe[:, None, :].expand(-1, horizons * layers, -1)
            allowed = torch.where(missing[..., None], fallback, allowed)
        blocked = ~allowed
        blocked = blocked[:, None].expand(
            -1, self.config.attention_heads, -1, -1
        ).reshape(batch * self.config.attention_heads, horizons * layers, nodes)
        queries = self.tree_attention(
            queries,
            tree.states,
            memory_padding_mask=padding,
            attention_mask=blocked,
        )
        return self.feedforward(queries.reshape(batch, horizons, layers, width))


class EndpointDecoder(nn.Module):
    """Decode direct H1--H4 layer endpoints from route/state/tree sources."""

    def __init__(self, config: HARPRTTConfig) -> None:
        super().__init__()
        self.config = config
        self.route_projection = nn.Linear(config.route_width, config.model_width)
        self.target_projection = nn.Linear(config.model_width, config.model_width)
        self.token_projection = nn.Sequential(
            nn.RMSNorm(config.exact_token_width),
            nn.Linear(config.exact_token_width, config.model_width),
        )
        self.final_hidden_projection = nn.Sequential(
            nn.RMSNorm(config.final_hidden_width),
            nn.Linear(config.final_hidden_width, config.model_width),
        )
        self.token_horizon_gates = nn.Parameter(
            torch.zeros(config.active_horizons, config.model_width)
        )
        self.horizon_embedding = nn.Embedding(
            config.active_horizons, config.model_width
        )
        self.layer_embedding = nn.Embedding(config.layers, config.model_width)
        self.initial_tree_attention = CrossAttentionResidual(
            config.model_width,
            config.tree_width,
            config.attention_heads,
            config.decoder_ffn_width,
            config.dropout,
        )
        self.blocks = nn.ModuleList(
            [_AxialDecoderBlock(config) for _ in range(config.decoder_blocks)]
        )
        self.short_adapter = SwiGLUResidual(
            config.model_width, config.decoder_ffn_width, config.dropout
        )
        self.long_adapter = SwiGLUResidual(
            config.model_width, config.decoder_ffn_width, config.dropout
        )
        self.h1_adapter = SwiGLUResidual(
            config.model_width, config.decoder_ffn_width, config.dropout
        )
        self.output_norm = nn.RMSNorm(config.model_width)

    def forward(
        self,
        route: Tensor,
        target: Tensor,
        tree: TreeEncoding,
        *,
        exact_token_embedding: Tensor,
        final_hidden: Tensor,
    ) -> Tensor:
        config = self.config
        batch = route.shape[0]
        if route.shape != (batch, config.layers, config.route_width):
            raise ValueError("route encoding has the wrong shape")
        if target.shape != (batch, config.layers, config.model_width):
            raise ValueError("target encoding has the wrong shape")
        if exact_token_embedding.shape != (batch, config.exact_token_width):
            raise ValueError("exact token embedding has the wrong shape")
        if final_hidden.shape != (batch, config.final_hidden_width):
            raise ValueError("final hidden state has the wrong shape")
        horizon_ids = torch.arange(config.active_horizons, device=route.device)
        layer_ids = torch.arange(config.layers, device=route.device)
        route_source = self.route_projection(route)[:, None]
        target_source = self.target_projection(target)[:, None]
        token = self.token_projection(exact_token_embedding)
        token = token[:, None, None] * torch.sigmoid(
            self.token_horizon_gates
        )[None, :, None]
        final = self.final_hidden_projection(final_hidden)[:, None, None]
        hidden = (
            route_source
            + target_source
            + token
            + final
            + self.horizon_embedding(horizon_ids)[None, :, None]
            + self.layer_embedding(layer_ids)[None, None]
        )
        queries = hidden.reshape(
            batch, config.active_horizons * config.layers, config.model_width
        )
        nodes = tree.states.shape[1]
        allowed = tree.available[:, None, :].expand(
            -1, config.active_horizons * config.layers, -1
        )
        safe, padding = safe_padding_mask(tree.available)
        missing = ~allowed.any(dim=-1)
        if missing.any():
            fallback = safe[:, None, :].expand(
                -1, config.active_horizons * config.layers, -1
            )
            allowed = torch.where(missing[..., None], fallback, allowed)
        blocked = (~allowed)[:, None].expand(
            -1, config.attention_heads, -1, -1
        ).reshape(
            batch * config.attention_heads,
            config.active_horizons * config.layers,
            nodes,
        )
        hidden = self.initial_tree_attention(
            queries,
            tree.states,
            memory_padding_mask=padding,
            attention_mask=blocked,
        ).reshape(
            batch, config.active_horizons, config.layers, config.model_width
        )
        for block in self.blocks:
            hidden = block(hidden, tree)
        short_count = min(2, config.active_horizons)
        short = self.short_adapter(hidden[:, :short_count])
        if config.active_horizons > short_count:
            long = self.long_adapter(hidden[:, short_count:])
            hidden = torch.cat([short, long], dim=1)
        else:
            hidden = short
        h1 = self.h1_adapter(hidden[:, :1])
        hidden = torch.cat([h1, hidden[:, 1:]], dim=1)
        return self.output_norm(hidden)


__all__ = ["EndpointDecoder"]
