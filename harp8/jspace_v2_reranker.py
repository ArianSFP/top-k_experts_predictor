"""Layer-aware J-HARP-C64 v2 candidate reranker.

This experimental successor leaves the v1 implementation untouched and
targets three concrete information-path bottlenecks:

* it consumes the frozen HARP generator context for every active
  horizon/layer pair;
* target-layer/local-context queries read MTP hidden and router memories
  separately; and
* every target layer owns a low-rank query map into its independently oriented
  router-SVD coordinate system.

Only H1--H4 are trainable in the v2 scientific contract.  Public composite
inference preserves the frozen HARP candidate scores bit-for-bit at H5--H8.
The final residual head is initialized to exact zero, so epoch-zero scores are
identical to the frozen generator for all candidates.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .jspace_reranker import (
    JContextEncoder,
    JSpaceRerankerConfig,
    JSpaceRerankerLossConfig,
    JSpaceRerankerLossOutput,
    jspace_reranker_loss,
)
from .jspace_v2_data import assert_no_label_only_inputs, causal_v2_model_inputs


JSPACE_V2_ACTIVE_HORIZONS = 4
JSPACE_V2_POOL_HORIZONS = 8
JSPACE_V2_MODEL_SCHEMA = "harp8_jspace_v2_candidate_reranker_v1"

_HORIZON_KEYS = frozenset(
    {
        "candidate_scores",
        "candidate_ids",
        "target_membership",
        "teacher_candidate_scores",
        "valid_future",
        "candidate_mask",
        "candidate_features",
        "generator_context",
    }
)


@dataclass(frozen=True)
class JSpaceV2RerankerConfig:
    """Accuracy-first geometry for the layer-aware v2 candidate ranker."""

    experts: int = 256
    layers: int = 40
    horizons: int = JSPACE_V2_ACTIVE_HORIZONS
    pool_horizons: int = JSPACE_V2_POOL_HORIZONS
    candidate_count: int = 64
    native_k: int = 8

    j_lags: int = 3
    j_width: int = 512
    generator_context_width: int = 384
    mtp_nodes: int = 6
    mtp_state_channels: int = 1
    mtp_state_width: int = 2048
    mtp_metadata_width: int = 0

    router_key_width: int = 256
    router_query_rank: int = 64
    expert_embedding_width: int = 128
    candidate_feature_width: int = 13

    model_width: int = 384
    attention_heads: int = 8
    feedforward_width: int = 1536
    temporal_blocks: int = 2
    axial_blocks: int = 4
    mtp_hidden_blocks: int = 2
    mtp_router_blocks: int = 2
    set_blocks: int = 2
    inducing_points: int = 16
    dropout: float = 0.05

    def validate(self) -> None:
        positive = {
            "experts": self.experts,
            "layers": self.layers,
            "horizons": self.horizons,
            "pool_horizons": self.pool_horizons,
            "candidate_count": self.candidate_count,
            "native_k": self.native_k,
            "j_lags": self.j_lags,
            "j_width": self.j_width,
            "generator_context_width": self.generator_context_width,
            "mtp_nodes": self.mtp_nodes,
            "mtp_state_channels": self.mtp_state_channels,
            "mtp_state_width": self.mtp_state_width,
            "router_key_width": self.router_key_width,
            "router_query_rank": self.router_query_rank,
            "expert_embedding_width": self.expert_embedding_width,
            "model_width": self.model_width,
            "attention_heads": self.attention_heads,
            "feedforward_width": self.feedforward_width,
            "temporal_blocks": self.temporal_blocks,
            "axial_blocks": self.axial_blocks,
            "mtp_hidden_blocks": self.mtp_hidden_blocks,
            "mtp_router_blocks": self.mtp_router_blocks,
            "set_blocks": self.set_blocks,
            "inducing_points": self.inducing_points,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive J-HARP v2 dimensions required: {invalid}")
        if self.horizons != JSPACE_V2_ACTIVE_HORIZONS:
            raise ValueError("scientific J-HARP v2 reranks exactly H1-H4")
        if self.pool_horizons != JSPACE_V2_POOL_HORIZONS:
            raise ValueError("J-HARP v2 composite inference requires an H1-H8 pool")
        if self.candidate_count < self.native_k:
            raise ValueError("candidate_count must be at least native_k")
        if self.model_width % self.attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")
        if self.router_query_rank > min(self.model_width, self.router_key_width):
            raise ValueError("router_query_rank exceeds its factor dimensions")
        if self.candidate_feature_width < 0 or self.mtp_metadata_width < 0:
            raise ValueError("optional feature widths cannot be negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")

    @property
    def uses_full_router_rank(self) -> bool:
        return self.router_key_width == self.experts

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def j_encoder_config(self) -> JSpaceRerankerConfig:
        """Create the v1-compatible geometry used only by JContextEncoder."""

        return JSpaceRerankerConfig(
            experts=self.experts,
            layers=self.layers,
            horizons=self.horizons,
            candidate_count=self.candidate_count,
            native_k=self.native_k,
            j_lags=self.j_lags,
            j_width=self.j_width,
            mtp_nodes=self.mtp_nodes,
            mtp_state_channels=self.mtp_state_channels,
            mtp_state_width=self.mtp_state_width,
            mtp_metadata_width=self.mtp_metadata_width,
            router_key_width=self.router_key_width,
            expert_embedding_width=self.expert_embedding_width,
            candidate_feature_width=self.candidate_feature_width,
            model_width=self.model_width,
            attention_heads=self.attention_heads,
            feedforward_width=self.feedforward_width,
            temporal_blocks=self.temporal_blocks,
            axial_blocks=self.axial_blocks,
            mtp_blocks=max(self.mtp_hidden_blocks, self.mtp_router_blocks),
            set_blocks=self.set_blocks,
            inducing_points=self.inducing_points,
            dropout=self.dropout,
        )


@dataclass
class JSpaceV2RerankerOutput:
    scores: torch.Tensor
    delta: torch.Tensor
    router_dot_scores: torch.Tensor
    horizon_layer_context: torch.Tensor
    mtp_hidden_context: torch.Tensor
    mtp_router_context: torch.Tensor
    source_weights: torch.Tensor


class _SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(width, hidden * 2)
        self.output = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, value = self.input(inputs).chunk(2, dim=-1)
        return self.dropout(self.output(F.silu(gate) * value))


class _ResidualSwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.block = _SwiGLU(width, hidden, dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(self.norm(inputs))


def _encoder_layer(config: JSpaceV2RerankerConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.model_width,
        nhead=config.attention_heads,
        dim_feedforward=config.feedforward_width,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class _AttentionResidual(nn.Module):
    def __init__(self, config: JSpaceV2RerankerConfig) -> None:
        super().__init__()
        self.query_norm = nn.RMSNorm(config.model_width)
        self.memory_norm = nn.RMSNorm(config.model_width)
        self.attention = nn.MultiheadAttention(
            config.model_width,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.feedforward = _ResidualSwiGLU(
            config.model_width, config.feedforward_width, config.dropout
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        update, _ = self.attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        return self.feedforward(queries + self.dropout(update))


class _InducedSetAttentionBlock(nn.Module):
    """Permutation-equivariant set attention without candidate positions."""

    def __init__(self, config: JSpaceV2RerankerConfig) -> None:
        super().__init__()
        self.inducing = nn.Parameter(
            torch.empty(config.inducing_points, config.model_width)
        )
        self.inducing_reads_set = _AttentionResidual(config)
        self.set_reads_inducing = _AttentionResidual(config)
        nn.init.normal_(self.inducing, std=0.02)

    def forward(
        self,
        candidates: torch.Tensor,
        *,
        candidate_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        inducing = self.inducing.unsqueeze(0).expand(candidates.shape[0], -1, -1)
        summaries = self.inducing_reads_set(
            inducing,
            candidates,
            memory_padding_mask=candidate_padding_mask,
        )
        return self.set_reads_inducing(candidates, summaries)


class SeparatedMTPEncoder(nn.Module):
    """Produce independent hidden-state and router-logit MTP memories."""

    def __init__(self, config: JSpaceV2RerankerConfig) -> None:
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
        self.state_scale = nn.Linear(config.mtp_state_channels, config.model_width)
        self.channel_embedding = nn.Embedding(
            config.mtp_state_channels, config.model_width
        )
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
        self.metadata_hidden_projection = (
            nn.Sequential(
                nn.LayerNorm(config.mtp_metadata_width),
                nn.Linear(config.mtp_metadata_width, config.model_width),
            )
            if config.mtp_metadata_width
            else None
        )
        self.metadata_router_projection = (
            nn.Sequential(
                nn.LayerNorm(config.mtp_metadata_width),
                nn.Linear(config.mtp_metadata_width, config.model_width),
            )
            if config.mtp_metadata_width
            else None
        )
        self.hidden_blocks = nn.ModuleList(
            [_encoder_layer(config) for _ in range(config.mtp_hidden_blocks)]
        )
        self.router_blocks = nn.ModuleList(
            [_encoder_layer(config) for _ in range(config.mtp_router_blocks)]
        )
        self.hidden_missing = nn.Parameter(torch.zeros(1, 1, config.model_width))
        self.router_missing = nn.Parameter(torch.zeros(1, 1, config.model_width))
        self.hidden_norm = nn.RMSNorm(config.model_width)
        self.router_norm = nn.RMSNorm(config.model_width)

    def forward(
        self,
        states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if states.ndim == 3 and self.config.mtp_state_channels == 1:
            states = states.unsqueeze(2)
        if states.ndim != 4:
            raise ValueError("mtp_states must have shape [B,N,S,D_M]")
        batch, nodes, channels, width = states.shape
        expected = (
            self.config.mtp_nodes,
            self.config.mtp_state_channels,
            self.config.mtp_state_width,
        )
        if (nodes, channels, width) != expected:
            raise ValueError(
                f"mtp_states geometry {(nodes, channels, width)} disagrees with {expected}"
            )
        if router_logits.shape != (batch, nodes, self.config.experts):
            raise ValueError("mtp_router_logits must have shape [B,N,E]")
        if mask is None:
            mask = torch.ones(batch, nodes, dtype=torch.bool, device=states.device)
        if mask.shape != (batch, nodes):
            raise ValueError("mtp_mask must have shape [B,N]")
        mask = mask.bool()

        state_values = states.float()
        hidden_parts = []
        for channel, projection in enumerate(self.state_projections):
            hidden_parts.append(
                projection(state_values[:, :, channel])
                + self.channel_embedding.weight[channel]
            )
        hidden = torch.stack(hidden_parts, dim=0).mean(dim=0)
        state_rms = torch.log1p(
            state_values.square().mean(dim=-1).sqrt().clamp_min(0.0)
        )
        hidden = hidden + self.state_scale(state_rms)
        hidden = hidden + self.hidden_depth_embedding(
            torch.arange(nodes, device=states.device)
        )[None]

        router_values = router_logits.float()
        centered_router = router_values - router_values.mean(dim=-1, keepdim=True)
        router = self.router_projection(centered_router)
        router_statistics = torch.stack(
            [
                torch.log1p(centered_router.square().mean(dim=-1).sqrt()),
                centered_router.amax(dim=-1) - centered_router.topk(2, dim=-1).values[..., 1],
            ],
            dim=-1,
        )
        router = router + self.router_scale(router_statistics)
        router = router + self.router_depth_embedding(
            torch.arange(nodes, device=states.device)
        )[None]

        if self.config.mtp_metadata_width:
            expected_metadata = (batch, nodes, self.config.mtp_metadata_width)
            if metadata is None or metadata.shape != expected_metadata:
                raise ValueError("mtp_metadata has the configured [B,N,M] geometry")
            assert self.metadata_hidden_projection is not None
            assert self.metadata_router_projection is not None
            hidden = hidden + self.metadata_hidden_projection(metadata.float())
            router = router + self.metadata_router_projection(metadata.float())
        elif metadata is not None:
            raise ValueError("mtp_metadata was supplied but metadata width is disabled")

        padding = torch.cat(
            [~mask, torch.zeros(batch, 1, dtype=torch.bool, device=mask.device)],
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
        return self.hidden_norm(hidden), self.router_norm(router), padding


class LayerSpecificRouterQuery(nn.Module):
    """Independent low-rank maps into each layer's router-SVD coordinates."""

    def __init__(self, config: JSpaceV2RerankerConfig) -> None:
        super().__init__()
        self.config = config
        self.down = nn.Parameter(
            torch.empty(config.layers, config.model_width, config.router_query_rank)
        )
        self.up = nn.Parameter(
            torch.empty(
                config.layers, config.router_query_rank, config.router_key_width
            )
        )
        self.bias = nn.Parameter(
            torch.zeros(config.layers, config.router_key_width)
        )
        for layer in range(config.layers):
            nn.init.xavier_uniform_(self.down[layer])
            nn.init.xavier_uniform_(self.up[layer])

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 4 or context.shape[2:] != (
            self.config.layers,
            self.config.model_width,
        ):
            raise ValueError("router query context must have shape [B,H,L,D]")
        low_rank = F.silu(torch.einsum("bhlw,lwq->bhlq", context, self.down))
        return (
            torch.einsum("bhlq,lqr->bhlr", low_rank, self.up)
            + self.bias[None, None]
        )


class JSpaceV2CandidateReranker(nn.Module):
    """Layer-aware, permutation-equivariant residual candidate ranker."""

    DERIVED_SCALAR_WIDTH = 6

    def __init__(
        self,
        config: JSpaceV2RerankerConfig,
        router_keys: torch.Tensor,
    ) -> None:
        super().__init__()
        config.validate()
        expected_keys = (
            config.layers,
            config.experts,
            config.router_key_width,
        )
        if tuple(router_keys.shape) != expected_keys:
            raise ValueError(
                f"router_keys geometry {tuple(router_keys.shape)} disagrees with {expected_keys}"
            )
        if not bool(torch.isfinite(router_keys).all()):
            raise ValueError("router_keys contain non-finite values")
        self.config = config
        self.register_buffer("router_keys", router_keys.detach().float().clone())

        self.j_encoder = JContextEncoder(config.j_encoder_config())
        self.generator_projection = nn.Sequential(
            nn.RMSNorm(config.generator_context_width),
            nn.Linear(config.generator_context_width, config.model_width),
        )
        self.horizon_embedding = nn.Embedding(config.horizons, config.model_width)
        self.layer_embedding = nn.Embedding(config.layers, config.model_width)
        self.local_context = nn.Sequential(
            nn.RMSNorm(config.model_width * 4),
            nn.Linear(config.model_width * 4, config.model_width),
            nn.SiLU(),
        )

        self.mtp_encoder = SeparatedMTPEncoder(config)
        self.hidden_cross_attention = _AttentionResidual(config)
        self.router_cross_attention = _AttentionResidual(config)
        self.source_score = nn.Sequential(
            nn.RMSNorm(config.model_width),
            nn.Linear(config.model_width, config.model_width // 2),
            nn.Tanh(),
            nn.Linear(config.model_width // 2, 1),
        )
        self.source_fusion = nn.Sequential(
            nn.RMSNorm(config.model_width * 4),
            nn.Linear(config.model_width * 4, config.model_width),
            nn.SiLU(),
        )
        self.context_block = _ResidualSwiGLU(
            config.model_width, config.feedforward_width, config.dropout
        )
        self.context_norm = nn.RMSNorm(config.model_width)

        self.router_query = LayerSpecificRouterQuery(config)
        self.layer_key_projection = nn.Parameter(
            torch.empty(config.layers, config.router_key_width, config.model_width)
        )
        for layer in range(config.layers):
            nn.init.xavier_uniform_(self.layer_key_projection[layer])
        self.expert_embedding = nn.Embedding(
            config.layers * config.experts, config.expert_embedding_width
        )
        self.expert_projection = nn.Linear(
            config.expert_embedding_width, config.model_width
        )
        scalar_width = self.DERIVED_SCALAR_WIDTH + config.candidate_feature_width
        self.scalar_projection = nn.Sequential(
            nn.LayerNorm(scalar_width),
            nn.Linear(scalar_width, config.model_width),
            nn.SiLU(),
        )
        self.dot_projection = nn.Linear(1, config.model_width, bias=False)
        self.elementwise_interaction = _SwiGLU(
            config.model_width, config.feedforward_width, config.dropout
        )
        self.candidate_norm = nn.RMSNorm(config.model_width)
        self.set_blocks = nn.ModuleList(
            [_InducedSetAttentionBlock(config) for _ in range(config.set_blocks)]
        )
        self.delta_head = nn.Sequential(
            nn.RMSNorm(config.model_width),
            nn.Linear(config.model_width, config.model_width),
            nn.SiLU(),
            nn.Linear(config.model_width, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.expert_embedding.weight, std=0.02)
        # Exact frozen-generator passthrough is a hard public API invariant.
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def _candidate_geometry(
        self, base: torch.Tensor, ids: torch.Tensor
    ) -> tuple[int, int, int, int]:
        if base.ndim != 4 or ids.shape != base.shape:
            raise ValueError("candidate scores/IDs must share [B,H,L,C] geometry")
        batch, horizons, layers, candidates = base.shape
        expected = (
            self.config.horizons,
            self.config.layers,
            self.config.candidate_count,
        )
        if (horizons, layers, candidates) != expected:
            raise ValueError(
                f"candidate geometry {(horizons, layers, candidates)} disagrees with {expected}"
            )
        if ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("candidate IDs must be integer-valued")
        if not bool(((ids >= 0) & (ids < self.config.experts)).all()):
            raise ValueError("candidate IDs lie outside the expert namespace")
        return batch, horizons, layers, candidates

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> JSpaceV2RerankerOutput:
        # Strip the outer objective batch to a causal allowlist before reading
        # any value. Future membership/teacher tensors never cross this point.
        batch = causal_v2_model_inputs(batch)
        base = batch["candidate_scores"].float()
        candidate_ids = batch["candidate_ids"].long()
        batch_size, horizons, layers, candidates = self._candidate_geometry(
            base, candidate_ids
        )
        candidate_mask = batch.get("candidate_mask")
        if candidate_mask is None:
            candidate_mask = torch.ones_like(candidate_ids, dtype=torch.bool)
        if candidate_mask.shape != base.shape:
            raise ValueError("candidate_mask must have shape [B,H,L,C]")
        candidate_mask = candidate_mask.bool()
        if not bool((candidate_mask.sum(dim=-1) >= self.config.native_k).all()):
            raise ValueError("each candidate set needs at least native_k entries")

        generator_context = batch.get("generator_context")
        expected_generator = (
            batch_size,
            horizons,
            layers,
            self.config.generator_context_width,
        )
        if generator_context is None or tuple(generator_context.shape) != expected_generator:
            raise ValueError("generator_context must have configured [B,H,L,D_G] geometry")

        j_context = self.j_encoder(batch["j_states"], batch.get("j_mask"))
        generator_hidden = self.generator_projection(generator_context.float())
        horizon_ids = torch.arange(horizons, device=base.device)
        layer_ids = torch.arange(layers, device=base.device)
        horizon_hidden = self.horizon_embedding(horizon_ids)[None, :, None].expand(
            batch_size, -1, layers, -1
        )
        layer_hidden = self.layer_embedding(layer_ids)[None, None].expand(
            batch_size, horizons, -1, -1
        )
        expanded_j = j_context[:, None].expand(-1, horizons, -1, -1)
        local = self.local_context(
            torch.cat(
                [expanded_j, generator_hidden, horizon_hidden, layer_hidden], dim=-1
            )
        )

        hidden_memory, router_memory, mtp_padding = self.mtp_encoder(
            batch["mtp_states"],
            batch["mtp_router_logits"],
            mask=batch.get("mtp_mask"),
            metadata=batch.get("mtp_metadata"),
        )
        # Every query is target-layer and horizon specific and already contains
        # local J plus frozen-generator context.  No [B,H,D] MTP summary is
        # broadcast across layers.
        query_sequence = local.reshape(
            batch_size, horizons * layers, self.config.model_width
        )
        hidden_read = self.hidden_cross_attention(
            query_sequence,
            hidden_memory,
            memory_padding_mask=mtp_padding,
        ).reshape(batch_size, horizons, layers, self.config.model_width)
        router_read = self.router_cross_attention(
            query_sequence,
            router_memory,
            memory_padding_mask=mtp_padding,
        ).reshape(batch_size, horizons, layers, self.config.model_width)

        # MTP hidden and router evidence meet for the first time here.  The
        # explicit source weights remain available for attribution.
        source_stack = torch.stack([local, hidden_read, router_read], dim=-2)
        source_weights = torch.softmax(
            self.source_score(source_stack).squeeze(-1), dim=-1
        )
        weighted = (source_stack * source_weights.unsqueeze(-1)).sum(dim=-2)
        context = self.source_fusion(
            torch.cat([local, hidden_read, router_read, weighted], dim=-1)
        )
        context = self.context_norm(self.context_block(context))

        object_ids = (
            candidate_ids
            + layer_ids.view(1, 1, layers, 1) * self.config.experts
        )
        flat_keys = self.router_keys.reshape(
            self.config.layers * self.config.experts,
            self.config.router_key_width,
        )
        candidate_keys = flat_keys[object_ids]
        normalized_keys = candidate_keys / candidate_keys.square().mean(
            dim=-1, keepdim=True
        ).add(1e-8).sqrt()
        key_hidden = torch.einsum(
            "bhlcr,lrw->bhlcw", normalized_keys, self.layer_key_projection
        ) / math.sqrt(self.config.router_key_width)
        key_hidden = F.silu(key_hidden)
        expert_hidden = self.expert_projection(self.expert_embedding(object_ids))

        query = self.router_query(context)
        dot_scores = torch.einsum(
            "bhlr,bhlcr->bhlc", query, candidate_keys
        ) / math.sqrt(self.config.router_key_width)

        masked_base = base.masked_fill(
            ~candidate_mask, -torch.finfo(base.dtype).max
        )
        kth = torch.topk(
            masked_base, self.config.native_k, dim=-1, sorted=True
        ).values[..., -1:]
        minimum = masked_base.masked_fill(~candidate_mask, torch.inf).amin(
            dim=-1, keepdim=True
        )
        valid_count = candidate_mask.sum(dim=-1, keepdim=True).clamp_min(2)
        mean = base.masked_fill(~candidate_mask, 0.0).sum(
            dim=-1, keepdim=True
        ) / valid_count
        greater = (
            masked_base.unsqueeze(-2) > masked_base.unsqueeze(-1)
        ).sum(dim=-1)
        rank_score = 1.0 - greater.to(base.dtype) / (valid_count - 1)
        key_rms = candidate_keys.square().mean(dim=-1).sqrt()
        scalars = torch.stack(
            [
                base,
                base - mean,
                rank_score,
                base - kth,
                base - minimum,
                torch.log1p(key_rms),
            ],
            dim=-1,
        )
        extra = batch.get("candidate_features")
        if self.config.candidate_feature_width:
            expected_extra = base.shape + (self.config.candidate_feature_width,)
            if extra is None or tuple(extra.shape) != expected_extra:
                raise ValueError(
                    "candidate_features must have configured [B,H,L,C,F] geometry"
                )
            scalars = torch.cat([scalars, extra.float()], dim=-1)
        elif extra is not None:
            raise ValueError(
                "candidate_features supplied while candidate_feature_width is disabled"
            )
        scalar_hidden = self.scalar_projection(scalars)

        expanded_context = context.unsqueeze(-2).expand(
            -1, -1, -1, candidates, -1
        )
        hidden = self.candidate_norm(
            expanded_context
            + key_hidden
            + expert_hidden
            + scalar_hidden
            + self.dot_projection(dot_scores.unsqueeze(-1))
            + self.elementwise_interaction(key_hidden * expanded_context)
        )
        flat_hidden = hidden.reshape(
            batch_size * horizons * layers,
            candidates,
            self.config.model_width,
        )
        flat_padding = (~candidate_mask).reshape(
            batch_size * horizons * layers, candidates
        )
        for block in self.set_blocks:
            flat_hidden = block(
                flat_hidden, candidate_padding_mask=flat_padding
            )
        hidden = flat_hidden.reshape(
            batch_size,
            horizons,
            layers,
            candidates,
            self.config.model_width,
        )
        delta = self.delta_head(hidden).squeeze(-1)
        delta = delta.masked_fill(~candidate_mask, 0.0)
        return JSpaceV2RerankerOutput(
            scores=base + delta,
            delta=delta,
            router_dot_scores=dot_scores,
            horizon_layer_context=context,
            mtp_hidden_context=hidden_read,
            mtp_router_context=router_read,
            source_weights=source_weights,
        )


def slice_v2_horizon_batch(
    batch: Mapping[str, torch.Tensor], active_horizons: int = 4
) -> dict[str, torch.Tensor]:
    """Slice all declared horizon tensors while leaving J/MTP histories intact."""

    if active_horizons != JSPACE_V2_ACTIVE_HORIZONS:
        raise ValueError("v2 scientific inference requires exactly four active horizons")
    result = dict(batch)
    for name in _HORIZON_KEYS:
        value = result.get(name)
        if value is None:
            continue
        if value.ndim < 2 or value.shape[1] < active_horizons:
            raise ValueError(f"{name} cannot supply H1-H{active_horizons}")
        result[name] = value[:, :active_horizons]
    return result


def composite_v2_horizon_scores(
    active_scores: torch.Tensor, frozen_base_scores: torch.Tensor
) -> torch.Tensor:
    """Return reranked H1-H4 plus exact frozen-generator H5-H8 scores."""

    if frozen_base_scores.ndim != 4 or frozen_base_scores.shape[1] != 8:
        raise ValueError("frozen base scores must have exact [B,8,L,C] geometry")
    if active_scores.shape != frozen_base_scores[:, :4].shape:
        raise ValueError("active scores must match the base H1-H4 prefix")
    return torch.cat([active_scores, frozen_base_scores[:, 4:]], dim=1)


class CompositeJSpaceV2Inference(nn.Module):
    """Public H1-H8 inference wrapper with exact inactive-horizon passthrough."""

    def __init__(self, reranker: JSpaceV2CandidateReranker) -> None:
        super().__init__()
        if reranker.config.horizons != 4 or reranker.config.pool_horizons != 8:
            raise ValueError("composite v2 inference requires active H1-H4 and pool H1-H8")
        self.reranker = reranker

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        assert_no_label_only_inputs(batch)
        frozen = batch["candidate_scores"].float()
        active_batch = slice_v2_horizon_batch(batch)
        active = self.reranker(active_batch)
        scores = composite_v2_horizon_scores(active.scores, frozen)
        if not torch.equal(scores[:, 4:], frozen[:, 4:]):
            raise AssertionError("H5-H8 composite passthrough changed frozen scores")
        return {
            "scores": scores,
            "active_scores": active.scores,
            "active_delta": active.delta,
            "router_dot_scores": active.router_dot_scores,
            "source_weights": active.source_weights,
        }


def v2_reranker_loss(
    output: JSpaceV2RerankerOutput | torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    config: JSpaceRerankerLossConfig | None = None,
) -> JSpaceRerankerLossOutput:
    """Use the audited v1 rank objective with v2 scores and active H1-H4."""

    scores = output.scores if isinstance(output, JSpaceV2RerankerOutput) else output
    return jspace_reranker_loss(scores, batch, config)


__all__ = [
    "CompositeJSpaceV2Inference",
    "JSPACE_V2_ACTIVE_HORIZONS",
    "JSPACE_V2_MODEL_SCHEMA",
    "JSPACE_V2_POOL_HORIZONS",
    "JSpaceV2CandidateReranker",
    "JSpaceV2RerankerConfig",
    "JSpaceV2RerankerOutput",
    "LayerSpecificRouterQuery",
    "SeparatedMTPEncoder",
    "composite_v2_horizon_scores",
    "slice_v2_horizon_batch",
    "v2_reranker_loss",
]
