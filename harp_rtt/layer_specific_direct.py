"""Layer-specific raw-channel fusion for direct DeltaRoute translation."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .direct_route import DirectDeltaRouteTrajectory
from .route_dynamics import DeltaRouteConfig


class LayerSpecificChannelPool(nn.Module):
    """Delay every 2,048-dimensional channel bottleneck until target layer."""

    RAW_CHANNELS = 5
    CHANNELS = 7

    def __init__(self, config: DeltaRouteConfig, *, raw_rank: int = 32) -> None:
        super().__init__()
        config.validate()
        if raw_rank < 1:
            raise ValueError("layer-specific raw rank must be positive")
        self.config = config
        self.raw_rank = raw_rank
        self.raw_down = nn.Parameter(torch.empty(
            self.RAW_CHANNELS, config.layers, config.raw_width, raw_rank
        ))
        self.raw_up = nn.Parameter(torch.empty(
            self.RAW_CHANNELS, config.layers, raw_rank, config.latent_width
        ))
        self.router_logits = nn.Linear(config.experts, config.latent_width)
        self.metadata = nn.Linear(config.metadata_width, config.latent_width)
        self.channel_embedding = nn.Embedding(self.CHANNELS, config.latent_width)
        self.layer = nn.Embedding(config.layers, config.latent_width)
        self.horizon = nn.Embedding(config.horizons, config.latent_width)
        self.query = nn.Linear(config.latent_width, config.latent_width, bias=False)
        self.key = nn.Linear(config.latent_width, config.latent_width, bias=False)
        self.value = nn.Linear(config.latent_width, config.latent_width, bias=False)
        self.output = nn.Linear(config.latent_width, config.latent_width)
        self.path_cell = nn.GRUCell(config.latent_width, config.latent_width)
        nn.init.normal_(self.raw_down, std=0.02)
        nn.init.normal_(self.raw_up, std=0.02)

    def _tokens(
        self,
        *,
        fused: Tensor,
        post_ffn: Tensor,
        router_input: Tensor,
        vocabulary: Tensor,
        token: Tensor,
        router_logits: Tensor,
        metadata: Tensor,
    ) -> Tensor:
        raw_values = (fused, post_ffn, router_input, vocabulary, token)
        if any(value.shape != fused.shape for value in raw_values):
            raise ValueError("layer-specific raw channels must share [B,N,D]")
        batch, nodes, width = fused.shape
        config = self.config
        if width != config.raw_width:
            raise ValueError("layer-specific raw width differs from config")
        raw = torch.stack(raw_values, dim=2).float()
        low = torch.einsum(
            "bncd,clda->bncla", raw, self.raw_down.float()
        )
        raw_tokens = torch.einsum(
            "bncla,claw->blncw", torch.nn.functional.silu(low),
            self.raw_up.float(),
        )
        if router_logits.shape != (batch, nodes, config.experts):
            raise ValueError("layer-specific router logits have invalid geometry")
        if metadata.shape != (batch, nodes, config.metadata_width):
            raise ValueError("layer-specific metadata has invalid geometry")
        logits = self.router_logits(router_logits)[:, None].expand(
            batch, config.layers, nodes, config.latent_width
        )
        meta = self.metadata(metadata)[:, None].expand_as(logits)
        tokens = torch.cat(
            (raw_tokens, logits[..., None, :], meta[..., None, :]), dim=3
        )
        return tokens + self.channel_embedding.weight[None, None, None]

    def _path_states(
        self, tokens: Tensor, parents: Tensor, available: Tensor
    ) -> Tensor:
        batch, layers, nodes, _, width = tokens.shape
        if parents.shape != (batch, nodes) or available.shape != (batch, nodes):
            raise ValueError("layer-specific path topology must be [B,N]")
        source = tokens.mean(3)
        states: list[Tensor] = []
        zero = torch.zeros(
            batch, layers, width, device=tokens.device, dtype=tokens.dtype
        )
        for node in range(nodes):
            parent = parents[:, node].long()
            if bool((available[:, node] & ((parent >= node) | (parent < -1))).any()):
                raise ValueError("available layer-specific parent must precede child")
            if node == 0:
                inherited = zero
            else:
                stacked = torch.stack(states, dim=2)
                inherited = stacked.gather(
                    2,
                    parent.clamp_min(0)[:, None, None, None].expand(
                        batch, layers, 1, width
                    ),
                ).squeeze(2)
                inherited = torch.where(
                    (parent >= 0)[:, None, None], inherited, zero
                )
            state = self.path_cell(
                source[:, :, node].reshape(batch * layers, width),
                inherited.reshape(batch * layers, width),
            ).reshape(batch, layers, width)
            state = state * available[:, node, None, None].to(state.dtype)
            states.append(state)
        return torch.stack(states, dim=2)

    def forward(
        self,
        *,
        fused: Tensor,
        post_ffn: Tensor,
        router_input: Tensor,
        vocabulary: Tensor,
        token: Tensor,
        router_logits: Tensor,
        metadata: Tensor,
        parents: Tensor,
        available: Tensor,
    ) -> tuple[Tensor, Tensor]:
        tokens = self._tokens(
            fused=fused, post_ffn=post_ffn, router_input=router_input,
            vocabulary=vocabulary, token=token, router_logits=router_logits,
            metadata=metadata,
        )
        path = self._path_states(tokens, parents, available.bool())
        config = self.config
        heads = config.attention_heads
        head_width = config.latent_width // heads
        query = self.query(
            self.layer.weight[None] + self.horizon.weight[:, None]
        ).reshape(config.horizons, config.layers, heads, head_width)
        key = self.key(tokens).reshape(
            tokens.shape[0], config.layers, config.nodes,
            self.CHANNELS, heads, head_width,
        )
        value = self.value(tokens).reshape_as(key)
        attention = torch.einsum(
            "hlad,blncad->bhlnac", query, key
        ) / math.sqrt(head_width)
        attention = torch.softmax(attention.float(), dim=-1).to(value.dtype)
        pooled = torch.einsum(
            "bhlnac,blncad->bhlnad", attention, value
        )
        pooled = self.output(pooled.flatten(-2)) + path[:, None]
        pooled = pooled * available[:, None, None, :, None].to(pooled.dtype)
        return pooled, path


class LayerSpecificDirectDeltaRouteTrajectory(DirectDeltaRouteTrajectory):
    """Direct translator with a separate low-rank raw map per target layer."""

    def __init__(
        self,
        config: DeltaRouteConfig,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
        *,
        raw_rank: int = 32,
    ) -> None:
        super().__init__(config, expert_keys, centered_bias, rank_mask)
        self.channels = LayerSpecificChannelPool(config, raw_rank=raw_rank)


__all__ = [
    "LayerSpecificChannelPool", "LayerSpecificDirectDeltaRouteTrajectory"
]
