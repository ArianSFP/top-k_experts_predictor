"""J-Lens-conditioned candidate reranking for the HARP expert forecaster.

The target J-Lens is intentionally outside this module.  ``j_states`` must be
precomputed from target post-block residuals at the audited lens boundary.  In
particular, :class:`MTPNodeEncoder` consumes native MTP tensors directly and
has no reference to, or parameter shaped like, a target J-Lens.

Tensor contract
---------------

Required batch entries are::

    candidate_scores:         [B, H, L, C]
    candidate_ids:            [B, H, L, C]
    j_states:                 [B, T, L, D_J]
    mtp_states:               [B, N, S, D_M] or [B, N, D_M] when S == 1
    mtp_router_logits:        [B, N, E]

Optional entries are ``j_mask[B,T,L]``, ``mtp_mask[B,N]``,
``candidate_mask[B,H,L,C]``, and
``candidate_features[B,H,L,C,F]``.  ``T`` is ordered current-token first,
then older lags.  The default production geometry is H=8, L=40, C=64,
T=3, N=6, and full router rank E=256.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class JSpaceRerankerConfig:
    """Geometry for the accuracy-first J-HARP candidate reranker."""

    experts: int = 256
    layers: int = 40
    horizons: int = 8
    candidate_count: int = 64
    native_k: int = 8

    j_lags: int = 3
    j_width: int = 512
    mtp_nodes: int = 6
    mtp_state_channels: int = 1
    mtp_state_width: int = 2048
    mtp_metadata_width: int = 0

    # A full-rank router factorization has R == E.  Smaller values are allowed
    # only when explicitly requested as a representation ablation.
    router_key_width: int = 256
    expert_embedding_width: int = 128
    candidate_feature_width: int = 0

    model_width: int = 384
    attention_heads: int = 8
    feedforward_width: int = 1536
    temporal_blocks: int = 2
    axial_blocks: int = 4
    mtp_blocks: int = 2
    set_blocks: int = 2
    inducing_points: int = 16
    dropout: float = 0.05

    def validate(self) -> None:
        positive = {
            "experts": self.experts,
            "layers": self.layers,
            "horizons": self.horizons,
            "candidate_count": self.candidate_count,
            "native_k": self.native_k,
            "j_lags": self.j_lags,
            "j_width": self.j_width,
            "mtp_nodes": self.mtp_nodes,
            "mtp_state_channels": self.mtp_state_channels,
            "mtp_state_width": self.mtp_state_width,
            "router_key_width": self.router_key_width,
            "expert_embedding_width": self.expert_embedding_width,
            "model_width": self.model_width,
            "attention_heads": self.attention_heads,
            "feedforward_width": self.feedforward_width,
            "temporal_blocks": self.temporal_blocks,
            "axial_blocks": self.axial_blocks,
            "mtp_blocks": self.mtp_blocks,
            "set_blocks": self.set_blocks,
            "inducing_points": self.inducing_points,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive J-HARP dimensions required: {invalid}")
        if self.candidate_count < self.native_k:
            raise ValueError("candidate_count must be at least native_k")
        if self.model_width % self.attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")
        if self.candidate_feature_width < 0 or self.mtp_metadata_width < 0:
            raise ValueError("optional feature widths cannot be negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

    @property
    def uses_full_router_rank(self) -> bool:
        return self.router_key_width == self.experts

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JSpaceRerankerLossConfig:
    """Preregistered candidate-ranking objective and horizon emphasis."""

    boundary: float = 1.0
    balanced_bce: float = 0.5
    listwise: float = 0.5
    restricted_kl: float = 0.25
    temperature: float = 2.0
    hard_negative_count: int = 24
    horizon_weights: tuple[float, ...] = (
        1.0,
        1.0,
        1.25,
        1.5,
        0.25,
        0.25,
        0.25,
        0.25,
    )

    def validate(self, horizons: int) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.hard_negative_count <= 0:
            raise ValueError("hard_negative_count must be positive")
        coefficients = (
            self.boundary,
            self.balanced_bce,
            self.listwise,
            self.restricted_kl,
        )
        if any(value < 0 for value in coefficients):
            raise ValueError("loss coefficients must be non-negative")
        if len(self.horizon_weights) != horizons:
            raise ValueError("horizon_weights must contain one value per horizon")
        if any(value < 0 for value in self.horizon_weights):
            raise ValueError("horizon weights must be non-negative")
        if not any(value > 0 for value in self.horizon_weights):
            raise ValueError("at least one horizon weight must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JSpaceRerankerOutput:
    """Model scores plus diagnostics needed by audits and ablations."""

    scores: torch.Tensor
    delta: torch.Tensor
    router_dot_scores: torch.Tensor
    horizon_layer_context: torch.Tensor


@dataclass
class JSpaceRerankerLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    per_horizon: dict[str, torch.Tensor]
    metrics: dict[str, float]


def _encoder_layer(config: JSpaceRerankerConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.model_width,
        nhead=config.attention_heads,
        dim_feedforward=config.feedforward_width,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


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


class _AttentionResidual(nn.Module):
    """Pre-norm multihead attention followed by a SwiGLU residual."""

    def __init__(self, config: JSpaceRerankerConfig) -> None:
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
    """Permutation-equivariant Set Transformer ISAB without positions."""

    def __init__(self, config: JSpaceRerankerConfig) -> None:
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
        batch = candidates.shape[0]
        inducing = self.inducing.unsqueeze(0).expand(batch, -1, -1)
        summaries = self.inducing_reads_set(
            inducing,
            candidates,
            memory_padding_mask=candidate_padding_mask,
        )
        return self.set_reads_inducing(candidates, summaries)


class JContextEncoder(nn.Module):
    """Encode the target lag-by-layer J grid: temporal first, axial second."""

    def __init__(self, config: JSpaceRerankerConfig) -> None:
        super().__init__()
        self.config = config
        self.input = nn.Sequential(
            nn.RMSNorm(config.j_width),
            nn.Linear(config.j_width, config.model_width),
        )
        self.lag_embedding = nn.Embedding(config.j_lags, config.model_width)
        self.layer_embedding = nn.Embedding(config.layers, config.model_width)
        self.temporal = nn.ModuleList(
            [_encoder_layer(config) for _ in range(config.temporal_blocks)]
        )
        self.temporal_score = nn.Linear(config.model_width, 1, bias=False)
        self.axial = nn.ModuleList(
            [_encoder_layer(config) for _ in range(config.axial_blocks)]
        )
        self.output_norm = nn.RMSNorm(config.model_width)

    def forward(
        self, states: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if states.ndim != 4:
            raise ValueError("j_states must have shape [B,T,L,D_J]")
        batch, lags, layers, width = states.shape
        expected = (self.config.j_lags, self.config.layers, self.config.j_width)
        if (lags, layers, width) != expected:
            raise ValueError(
                f"j_states geometry {(lags, layers, width)} disagrees with {expected}"
            )
        if mask is None:
            mask = torch.ones(
                batch, lags, layers, dtype=torch.bool, device=states.device
            )
        if mask.shape != states.shape[:3]:
            raise ValueError("j_mask must have shape [B,T,L]")
        mask = mask.bool()
        # Every target layer needs at least one available lag.  Failing closed
        # prevents all-masked attention rows from silently producing NaNs.
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every target layer needs at least one available J lag")

        lag_ids = torch.arange(lags, device=states.device)
        layer_ids = torch.arange(layers, device=states.device)
        hidden = self.input(states.float())
        hidden = hidden + self.lag_embedding(lag_ids)[None, :, None, :]
        hidden = hidden + self.layer_embedding(layer_ids)[None, None, :, :]

        temporal = hidden.permute(0, 2, 1, 3).reshape(
            batch * layers, lags, self.config.model_width
        )
        temporal_padding = (~mask.permute(0, 2, 1)).reshape(batch * layers, lags)
        for block in self.temporal:
            temporal = block(temporal, src_key_padding_mask=temporal_padding)
        scores = self.temporal_score(temporal).squeeze(-1)
        scores = scores.masked_fill(temporal_padding, -torch.finfo(scores.dtype).max)
        weights = torch.softmax(scores, dim=-1)
        pooled = (temporal * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled.reshape(batch, layers, self.config.model_width)

        layer_mask = mask.any(dim=1)
        for block in self.axial:
            pooled = block(pooled, src_key_padding_mask=~layer_mask)
        return self.output_norm(pooled)


class MTPNodeEncoder(nn.Module):
    """Encode six native MTP nodes without applying the target J-Lens."""

    def __init__(self, config: JSpaceRerankerConfig) -> None:
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
        self.router_projection = nn.Sequential(
            nn.LayerNorm(config.experts),
            nn.Linear(config.experts, config.model_width),
        )
        self.metadata_projection = (
            nn.Sequential(
                nn.LayerNorm(config.mtp_metadata_width),
                nn.Linear(config.mtp_metadata_width, config.model_width),
            )
            if config.mtp_metadata_width
            else None
        )
        self.depth_embedding = nn.Embedding(config.mtp_nodes, config.model_width)
        self.missing_node = nn.Parameter(torch.zeros(1, 1, config.model_width))
        self.fusion = nn.Sequential(
            nn.RMSNorm(config.model_width),
            nn.Linear(config.model_width, config.model_width * 2),
            nn.SiLU(),
            nn.Linear(config.model_width * 2, config.model_width),
        )
        self.blocks = nn.ModuleList(
            [_encoder_layer(config) for _ in range(config.mtp_blocks)]
        )
        self.output_norm = nn.RMSNorm(config.model_width)

    def forward(
        self,
        states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        state_parts = []
        for channel, projection in enumerate(self.state_projections):
            state_parts.append(
                projection(states[:, :, channel].float())
                + self.channel_embedding.weight[channel]
            )
        state_hidden = torch.stack(state_parts, dim=0).mean(dim=0)
        hidden = state_hidden + self.router_projection(router_logits.float())
        hidden = hidden + self.depth_embedding(
            torch.arange(nodes, device=states.device)
        )[None]
        if self.metadata_projection is not None:
            if metadata is None or metadata.shape != (
                batch,
                nodes,
                self.config.mtp_metadata_width,
            ):
                raise ValueError("mtp_metadata has the configured [B,N,M] geometry")
            hidden = hidden + self.metadata_projection(metadata.float())
        elif metadata is not None:
            raise ValueError("mtp_metadata was supplied but metadata width is disabled")
        hidden = self.fusion(hidden)

        # The always-valid missing-source token makes the all-MTP-missing case
        # numerically well-defined and lets each horizon learn a fallback.
        hidden = torch.cat(
            [hidden, self.missing_node.expand(batch, -1, -1)], dim=1
        )
        padding = torch.cat(
            [~mask, torch.zeros(batch, 1, dtype=torch.bool, device=mask.device)],
            dim=1,
        )
        for block in self.blocks:
            hidden = block(hidden, src_key_padding_mask=padding)
        return self.output_norm(hidden), padding


class JSpaceCandidateReranker(nn.Module):
    """J-conditioned, permutation-equivariant residual candidate ranker."""

    DERIVED_SCALAR_WIDTH = 6

    def __init__(
        self,
        config: JSpaceRerankerConfig,
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

        self.j_encoder = JContextEncoder(config)
        self.mtp_encoder = MTPNodeEncoder(config)
        self.horizon_embedding = nn.Embedding(config.horizons, config.model_width)
        self.output_layer_embedding = nn.Embedding(config.layers, config.model_width)
        self.mtp_cross_attention = _AttentionResidual(config)
        self.context_fusion = _ResidualSwiGLU(
            config.model_width, config.feedforward_width, config.dropout
        )
        self.context_norm = nn.RMSNorm(config.model_width)

        self.router_query = nn.Linear(
            config.model_width, config.router_key_width, bias=False
        )
        self.router_key_projection = nn.Sequential(
            nn.LayerNorm(config.router_key_width),
            nn.Linear(config.router_key_width, config.model_width),
        )
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
        # The exact-zero correction is an API invariant: loading this module on
        # top of a frozen generator cannot alter candidate ranking at step zero.
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def _validate_candidate_geometry(
        self, base: torch.Tensor, candidate_ids: torch.Tensor
    ) -> tuple[int, int, int, int]:
        if base.ndim != 4:
            raise ValueError("candidate_scores must have shape [B,H,L,C]")
        if candidate_ids.shape != base.shape:
            raise ValueError("candidate_ids must match candidate_scores")
        batch, horizons, layers, candidates = base.shape
        expected = (
            self.config.horizons,
            self.config.layers,
            self.config.candidate_count,
        )
        if (horizons, layers, candidates) != expected:
            raise ValueError(
                "candidate geometry "
                f"{(horizons, layers, candidates)} disagrees with {expected}"
            )
        if candidate_ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("candidate_ids must be an integer tensor")
        if not bool(((candidate_ids >= 0) & (candidate_ids < self.config.experts)).all()):
            raise ValueError("candidate_ids lie outside the expert namespace")
        return batch, horizons, layers, candidates

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> JSpaceRerankerOutput:
        base = batch["candidate_scores"].float()
        candidate_ids = batch["candidate_ids"].long()
        batch_size, horizons, layers, candidates = self._validate_candidate_geometry(
            base, candidate_ids
        )

        candidate_mask = batch.get("candidate_mask")
        if candidate_mask is None:
            candidate_mask = torch.ones_like(candidate_ids, dtype=torch.bool)
        if candidate_mask.shape != base.shape:
            raise ValueError("candidate_mask must have shape [B,H,L,C]")
        candidate_mask = candidate_mask.bool()
        if not bool((candidate_mask.sum(dim=-1) >= self.config.native_k).all()):
            raise ValueError("every candidate set must contain at least native_k entries")

        j_context = self.j_encoder(batch["j_states"], batch.get("j_mask"))
        mtp_nodes, mtp_padding = self.mtp_encoder(
            batch["mtp_states"],
            batch["mtp_router_logits"],
            mask=batch.get("mtp_mask"),
            metadata=batch.get("mtp_metadata"),
        )
        horizon_ids = torch.arange(horizons, device=base.device)
        horizon_queries = self.horizon_embedding(horizon_ids)[None].expand(
            batch_size, -1, -1
        )
        mtp_context = self.mtp_cross_attention(
            horizon_queries,
            mtp_nodes,
            memory_padding_mask=mtp_padding,
        )
        layer_ids = torch.arange(layers, device=base.device)
        context = (
            j_context[:, None]
            + mtp_context[:, :, None]
            + self.horizon_embedding(horizon_ids)[None, :, None]
            + self.output_layer_embedding(layer_ids)[None, None]
        )
        context = self.context_norm(self.context_fusion(context))

        object_ids = (
            candidate_ids
            + layer_ids.view(1, 1, layers, 1) * self.config.experts
        )
        flat_keys = self.router_keys.reshape(
            self.config.layers * self.config.experts,
            self.config.router_key_width,
        )
        candidate_keys = flat_keys[object_ids]
        key_hidden = self.router_key_projection(candidate_keys)
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
        valid_sum = base.masked_fill(~candidate_mask, 0.0).sum(
            dim=-1, keepdim=True
        )
        mean = valid_sum / valid_count
        # Pairwise greater-than counts are tie invariant, unlike argsort ranks.
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
            if extra is None or extra.shape != expected_extra:
                raise ValueError(
                    "candidate_features must have configured [B,H,L,C,F] geometry"
                )
            scalars = torch.cat([scalars, extra.float()], dim=-1)
        elif extra is not None:
            raise ValueError(
                "candidate_features were supplied but candidate_feature_width is zero"
            )
        scalar_hidden = self.scalar_projection(scalars)

        expanded_context = context.unsqueeze(-2).expand(
            -1, -1, -1, candidates, -1
        )
        elementwise = self.elementwise_interaction(key_hidden * expanded_context)
        hidden = self.candidate_norm(
            expanded_context
            + key_hidden
            + expert_hidden
            + scalar_hidden
            + self.dot_projection(dot_scores.unsqueeze(-1))
            + elementwise
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
        return JSpaceRerankerOutput(
            scores=base + delta,
            delta=delta,
            router_dot_scores=dot_scores,
            horizon_layer_context=context,
        )


def _valid_layer_mask(
    valid_future: torch.Tensor,
    *,
    batch: int,
    horizons: int,
    layers: int,
) -> torch.Tensor:
    if valid_future.shape == (batch, horizons):
        return valid_future.bool().unsqueeze(-1).expand(-1, -1, layers)
    if valid_future.shape == (batch, horizons, layers):
        return valid_future.bool()
    raise ValueError("valid_future must have shape [B,H] or [B,H,L]")


def jspace_reranker_loss(
    output: JSpaceRerankerOutput | torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    config: JSpaceRerankerLossConfig | None = None,
) -> JSpaceRerankerLossOutput:
    """Combined boundary, balanced-BCE, listwise, and restricted-KL loss."""

    predicted = output.scores if isinstance(output, JSpaceRerankerOutput) else output
    predicted = predicted.float()
    if predicted.ndim != 4:
        raise ValueError("predicted candidate scores must have shape [B,H,L,C]")
    batch_size, horizons, layers, candidates = predicted.shape
    loss_config = config or JSpaceRerankerLossConfig()
    loss_config.validate(horizons)

    membership = batch["target_membership"].bool()
    teacher = batch["teacher_candidate_scores"].float()
    if membership.shape != predicted.shape or teacher.shape != predicted.shape:
        raise ValueError("membership and teacher scores must match predicted scores")
    valid = _valid_layer_mask(
        batch["valid_future"],
        batch=batch_size,
        horizons=horizons,
        layers=layers,
    )
    candidate_mask = batch.get("candidate_mask")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(membership)
    if candidate_mask.shape != predicted.shape:
        raise ValueError("candidate_mask must match predicted scores")
    candidate_mask = candidate_mask.bool()
    membership = membership & candidate_mask
    positive_count = membership.sum(dim=-1)
    negative_mask = candidate_mask & ~membership
    negative_count = negative_mask.sum(dim=-1)
    usable = valid & (positive_count > 0) & (negative_count > 0)

    positive_term = F.softplus(-predicted).masked_fill(~membership, 0.0)
    negative_term = F.softplus(predicted).masked_fill(~negative_mask, 0.0)
    bce_rows = 0.5 * (
        positive_term.sum(dim=-1) / positive_count.clamp_min(1)
        + negative_term.sum(dim=-1) / negative_count.clamp_min(1)
    )

    negative_teacher = teacher.masked_fill(
        ~negative_mask, -torch.finfo(teacher.dtype).max
    )
    hard_count = min(loss_config.hard_negative_count, candidates)
    hard_values, hard_ids = torch.topk(
        negative_teacher, hard_count, dim=-1, sorted=False
    )
    hard_valid = hard_values > -torch.finfo(teacher.dtype).max / 2
    hard_scores = predicted.gather(-1, hard_ids)
    pairwise = F.softplus(
        hard_scores.unsqueeze(-2) - predicted.unsqueeze(-1)
    )
    pair_mask = membership.unsqueeze(-1) & hard_valid.unsqueeze(-2)
    boundary_rows = (
        pairwise.masked_fill(~pair_mask, 0.0).sum(dim=(-1, -2))
        / pair_mask.sum(dim=(-1, -2)).clamp_min(1)
    )

    masked_predicted = predicted.masked_fill(
        ~candidate_mask, -torch.finfo(predicted.dtype).max
    )
    log_probability = torch.log_softmax(masked_predicted, dim=-1)
    positive_target = membership.to(predicted.dtype) / positive_count.clamp_min(1).unsqueeze(-1)
    listwise_rows = -(positive_target * log_probability).sum(dim=-1)

    temperature = float(loss_config.temperature)
    masked_teacher = teacher.masked_fill(
        ~candidate_mask, -torch.finfo(teacher.dtype).max
    )
    teacher_log_probability = torch.log_softmax(
        masked_teacher / temperature, dim=-1
    )
    student_log_probability = torch.log_softmax(
        masked_predicted / temperature, dim=-1
    )
    teacher_probability = torch.softmax(masked_teacher / temperature, dim=-1)
    kl_rows = (
        teacher_probability
        * (teacher_log_probability - student_log_probability)
    ).sum(dim=-1) * temperature**2

    def reduce_horizons(rows: torch.Tensor) -> torch.Tensor:
        mask = usable.to(rows.dtype)
        return (rows * mask).sum(dim=(0, 2)) / mask.sum(dim=(0, 2)).clamp_min(1)

    per_horizon = {
        "boundary": reduce_horizons(boundary_rows),
        "balanced_bce": reduce_horizons(bce_rows),
        "listwise": reduce_horizons(listwise_rows),
        "restricted_kl": reduce_horizons(kl_rows),
    }
    horizon_weights = torch.as_tensor(
        loss_config.horizon_weights,
        device=predicted.device,
        dtype=predicted.dtype,
    )
    horizon_weights = horizon_weights / horizon_weights.sum()
    components = {
        name: (values * horizon_weights).sum()
        for name, values in per_horizon.items()
    }
    total = (
        loss_config.boundary * components["boundary"]
        + loss_config.balanced_bce * components["balanced_bce"]
        + loss_config.listwise * components["listwise"]
        + loss_config.restricted_kl * components["restricted_kl"]
    )
    metrics = {name: float(value.detach()) for name, value in components.items()}
    metrics["loss"] = float(total.detach())
    for name, values in per_horizon.items():
        for horizon, value in enumerate(values, start=1):
            metrics[f"{name}_h{horizon}"] = float(value.detach())
    return JSpaceRerankerLossOutput(
        total=total,
        components=components,
        per_horizon=per_horizon,
        metrics=metrics,
    )


__all__ = [
    "JContextEncoder",
    "JSpaceCandidateReranker",
    "JSpaceRerankerConfig",
    "JSpaceRerankerLossConfig",
    "JSpaceRerankerLossOutput",
    "JSpaceRerankerOutput",
    "MTPNodeEncoder",
    "jspace_reranker_loss",
]
