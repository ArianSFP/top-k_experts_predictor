"""Independent-gate target control/content encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import SelfAttentionResidual
from .config import HARPRTTConfig


class _SourceProjection(nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(input_width)
        self.projection = nn.Linear(input_width, output_width)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(self.norm(inputs))


class TargetStateEncoder(nn.Module):
    """Fuse router-visible and router-blind state without a source softmax."""

    SOURCE_NAMES = (
        "control",
        "content",
        "post_attention",
        "post_moe",
        "routed",
        "shared",
    )

    def __init__(self, config: HARPRTTConfig) -> None:
        super().__init__()
        self.config = config
        widths = {
            "control": config.target_control_width,
            "content": config.target_content_width,
            "post_attention": config.target_post_attention_width,
            "post_moe": config.target_post_moe_width,
            "routed": config.target_routed_width,
            "shared": config.target_shared_width,
        }
        self.projections = nn.ModuleDict(
            {
                name: _SourceProjection(width, config.target_width)
                for name, width in widths.items()
            }
        )
        # Each source owns an independent vector gate.  Control is also the
        # ungated anchor, ensuring there is no competition between sources.
        self.gate_logits = nn.Parameter(
            torch.zeros(len(self.SOURCE_NAMES), config.target_width)
        )
        self.layer_embedding = nn.Embedding(config.layers, config.target_width)
        self.blocks = nn.ModuleList(
            [
                SelfAttentionResidual(
                    config.target_width,
                    # target_width can differ from model width but must remain
                    # divisible by the production head count.
                    config.attention_heads,
                    config.target_ffn_width,
                    config.dropout,
                )
                for _ in range(config.target_blocks)
            ]
        )
        self.output_lift = nn.Sequential(
            nn.RMSNorm(config.target_width),
            nn.Linear(config.target_width, config.model_width),
        )

    def forward(
        self,
        *,
        control: Tensor,
        content: Tensor,
        post_attention: Tensor,
        post_moe: Tensor,
        routed: Tensor,
        shared: Tensor,
        available: Tensor | None = None,
    ) -> Tensor:
        values = {
            "control": control,
            "content": content,
            "post_attention": post_attention,
            "post_moe": post_moe,
            "routed": routed,
            "shared": shared,
        }
        batch = control.shape[0]
        expected_prefix = (batch, self.config.layers)
        for name, tensor in values.items():
            if tensor.shape[:2] != expected_prefix:
                raise ValueError(f"target {name} must begin [B,L]")
        if available is None:
            source_available = torch.ones(
                batch,
                self.config.layers,
                len(self.SOURCE_NAMES),
                dtype=torch.bool,
                device=control.device,
            )
        else:
            expected = (batch, self.config.layers, len(self.SOURCE_NAMES))
            if available.shape != expected:
                raise ValueError(f"target availability must have shape {expected}")
            source_available = available.bool()
        projected = torch.stack(
            [self.projections[name](values[name]) for name in self.SOURCE_NAMES],
            dim=2,
        )
        gates = torch.sigmoid(self.gate_logits)[None, None]
        gated = projected * gates * source_available[..., None].to(projected.dtype)
        anchor = projected[:, :, 0] * source_available[:, :, 0, None].to(projected.dtype)
        hidden = anchor + gated.sum(dim=2)
        layer_ids = torch.arange(self.config.layers, device=hidden.device)
        hidden = hidden + self.layer_embedding(layer_ids)[None]
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_lift(hidden)


__all__ = ["TargetStateEncoder"]
