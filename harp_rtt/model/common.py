"""Small attention and residual building blocks shared by HARP-RTT."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def zero_linear(module: nn.Linear) -> nn.Linear:
    """Zero a terminal residual projection and return it."""

    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


def safe_padding_mask(available: Tensor) -> tuple[Tensor, Tensor]:
    """Return a non-empty availability mask and the corresponding padding mask.

    PyTorch attention produces NaNs for an all-masked row.  The synthetic
    fallback token is always zeroed by callers, so making slot zero temporarily
    visible cannot leak information.
    """

    if available.ndim != 2:
        raise ValueError("attention availability must be [batch, sequence]")
    safe = available.bool().clone()
    missing = ~safe.any(dim=-1)
    if missing.any():
        safe[missing, 0] = True
    return safe, ~safe


class SwiGLUResidual(nn.Module):
    """Pre-norm SwiGLU residual block."""

    def __init__(
        self,
        width: int,
        hidden_width: int,
        dropout: float,
        *,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.up = nn.Linear(width, hidden_width * 2)
        self.down = nn.Linear(hidden_width, width)
        if zero_output:
            zero_linear(self.down)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, value = self.up(self.norm(inputs)).chunk(2, dim=-1)
        return inputs + self.dropout(self.down(F.silu(gate) * value))


class SelfAttentionResidual(nn.Module):
    """Pre-norm batch-first self attention followed by SwiGLU."""

    def __init__(
        self,
        width: int,
        heads: int,
        ffn_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.feedforward = SwiGLUResidual(width, ffn_width, dropout)

    def forward(
        self,
        inputs: Tensor,
        *,
        padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        normalized = self.norm(inputs)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        return self.feedforward(inputs + self.dropout(update))


class CrossAttentionResidual(nn.Module):
    """Pre-norm cross attention followed by SwiGLU."""

    def __init__(
        self,
        query_width: int,
        memory_width: int,
        heads: int,
        ffn_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.RMSNorm(query_width)
        self.memory_norm = nn.RMSNorm(memory_width)
        self.key = (
            nn.Identity()
            if query_width == memory_width
            else nn.Linear(memory_width, query_width)
        )
        self.attention = nn.MultiheadAttention(
            query_width, heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.feedforward = SwiGLUResidual(query_width, ffn_width, dropout)

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        *,
        memory_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        projected = self.key(self.memory_norm(memory))
        update, _ = self.attention(
            self.query_norm(queries),
            projected,
            projected,
            key_padding_mask=memory_padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        return self.feedforward(queries + self.dropout(update))


def gather_layer_experts(values: Tensor, expert_ids: Tensor) -> Tensor:
    """Gather ``[L,E,D]`` object features for IDs ending in ``[H,L,C]``."""

    if values.ndim != 3 or expert_ids.ndim != 4:
        raise ValueError("expected values [L,E,D] and expert_ids [B,H,L,C]")
    layers, experts, width = values.shape
    if expert_ids.shape[2] != layers:
        raise ValueError("candidate layer axis disagrees with object table")
    offsets = torch.arange(layers, device=expert_ids.device) * experts
    flattened_ids = expert_ids + offsets[None, None, :, None]
    return F.embedding(flattened_ids, values.reshape(layers * experts, width))


def gather_dense_experts(values: Tensor, expert_ids: Tensor) -> Tensor:
    """Gather a dense ``[B,H,L,E]`` tensor into ``[B,H,L,C]``."""

    if values.ndim != 4 or expert_ids.ndim != 4:
        raise ValueError("expected dense values and candidate IDs to be rank four")
    if values.shape[:3] != expert_ids.shape[:3]:
        raise ValueError("dense values and candidates disagree before expert axis")
    return values.gather(-1, expert_ids)


__all__ = [
    "CrossAttentionResidual",
    "SelfAttentionResidual",
    "SwiGLUResidual",
    "gather_dense_experts",
    "gather_layer_experts",
    "safe_padding_mask",
    "zero_linear",
]
