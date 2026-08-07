"""Eight-token, forty-layer route-grid encoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .common import SelfAttentionResidual, safe_padding_mask
from .config import HARPRTTConfig


class RouteGridEncoder(nn.Module):
    """Encode full causal route logits with temporal and layer axial blocks."""

    _DERIVED_STATISTICS = 6

    def __init__(self, config: HARPRTTConfig, expert_keys: Tensor) -> None:
        super().__init__()
        self.config = config
        if expert_keys.shape != (
            config.layers,
            config.experts,
            config.router_rank,
        ):
            raise ValueError("expert_keys disagree with HARP-RTT geometry")
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.key_projection = nn.Linear(config.router_rank, config.route_key_width)
        cell_width = (
            config.experts
            + config.route_key_width
            + self._DERIVED_STATISTICS
            + config.route_summary_width
        )
        self.logit_norm = nn.LayerNorm(config.experts)
        self.cell = nn.Sequential(
            nn.Linear(cell_width, config.route_ffn_width),
            nn.SiLU(),
            nn.Linear(config.route_ffn_width, config.route_width),
            nn.RMSNorm(config.route_width),
        )
        self.lag_embedding = nn.Embedding(config.route_history, config.route_width)
        self.layer_embedding = nn.Embedding(config.layers, config.route_width)
        self.temporal_blocks = nn.ModuleList(
            [
                SelfAttentionResidual(
                    config.route_width,
                    config.attention_heads,
                    config.route_ffn_width,
                    config.dropout,
                )
                for _ in range(config.temporal_blocks)
            ]
        )
        self.layer_blocks = nn.ModuleList(
            [
                SelfAttentionResidual(
                    config.route_width,
                    config.attention_heads,
                    config.route_ffn_width,
                    config.dropout,
                )
                for _ in range(config.route_layer_blocks)
            ]
        )
        self.local_norm = nn.RMSNorm(config.route_width)
        self.local = nn.Conv1d(
            config.route_width,
            config.route_width,
            kernel_size=3,
            padding=1,
            groups=config.route_width,
        )
        self.local_mix = nn.Linear(config.route_width, config.route_width)
        self.output_norm = nn.RMSNorm(config.route_width)

    def _statistics(self, centered: Tensor) -> Tensor:
        values = centered.float()
        probabilities = torch.softmax(values, dim=-1)
        boundary_k = min(9, self.config.experts)
        top = torch.topk(values, boundary_k, dim=-1, sorted=True).values
        first_gap = top[..., 0] - top[..., min(1, boundary_k - 1)]
        selected = min(self.config.exact_k, self.config.experts)
        boundary_index = min(selected - 1, boundary_k - 1)
        outside_index = min(selected, boundary_k - 1)
        boundary_gap = top[..., boundary_index] - top[..., outside_index]
        top_ids = torch.topk(values, selected, dim=-1, sorted=False).indices
        selected_mass = probabilities.gather(-1, top_ids).sum(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        entropy = entropy / math.log(max(2, self.config.experts))
        return torch.stack(
            [
                entropy,
                first_gap,
                boundary_gap,
                selected_mass,
                values.std(dim=-1, unbiased=False),
                values.abs().mean(dim=-1),
            ],
            dim=-1,
        ).to(centered.dtype)

    def _selected_key_pool(
        self,
        centered: Tensor,
        selected_ids: Tensor | None,
        selected_weights: Tensor | None,
    ) -> Tensor:
        batch, layers, history, _ = centered.shape
        count = min(self.config.exact_k, self.config.experts)
        if selected_ids is None:
            values, ids = torch.topk(centered, count, dim=-1, sorted=False)
            weights = torch.softmax(values.float(), dim=-1).to(centered.dtype)
        else:
            expected = (batch, layers, history, count)
            if selected_ids.shape != expected:
                raise ValueError(
                    f"history_selected_ids must have shape {expected}, got {tuple(selected_ids.shape)}"
                )
            ids = selected_ids.long()
            if selected_weights is None:
                weights = torch.full_like(ids, 1.0 / count, dtype=centered.dtype)
            else:
                if selected_weights.shape != expected:
                    raise ValueError("history_selected_weights disagree with IDs")
                weights = selected_weights.to(centered.dtype)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if (ids < 0).any() or (ids >= self.config.experts).any():
            raise ValueError("history selected expert ID lies outside the namespace")
        projected = self.key_projection(self.expert_keys.to(centered.dtype))
        offsets = torch.arange(layers, device=ids.device) * self.config.experts
        flattened = ids + offsets[None, :, None, None]
        keys = torch.nn.functional.embedding(
            flattened, projected.reshape(layers * self.config.experts, -1)
        )
        return (keys * weights[..., None]).sum(dim=-2)

    def forward(
        self,
        history_logits: Tensor,
        history_available: Tensor,
        *,
        selected_ids: Tensor | None = None,
        selected_weights: Tensor | None = None,
        history_summary: Tensor | None = None,
    ) -> Tensor:
        config = self.config
        expected = (config.layers, config.route_history, config.experts)
        if history_logits.shape[1:] != expected:
            raise ValueError(
                f"history_logits must be [B,{expected}], got {tuple(history_logits.shape)}"
            )
        batch = history_logits.shape[0]
        if history_available.shape == (batch, config.route_history):
            available = history_available[:, None].expand(-1, config.layers, -1)
        elif history_available.shape == (batch, config.layers, config.route_history):
            available = history_available
        else:
            raise ValueError("history_available must be [B,T] or [B,L,T]")
        centered = history_logits - history_logits.mean(dim=-1, keepdim=True)
        key_pool = self._selected_key_pool(centered, selected_ids, selected_weights)
        if history_summary is None:
            summary = centered.new_zeros(
                batch,
                config.layers,
                config.route_history,
                config.route_summary_width,
            )
        else:
            expected_summary = (
                batch,
                config.layers,
                config.route_history,
                config.route_summary_width,
            )
            if history_summary.shape != expected_summary:
                raise ValueError("history_summary has the wrong shape")
            summary = history_summary.to(centered.dtype)
        cells = self.cell(
            torch.cat(
                [
                    self.logit_norm(centered),
                    key_pool,
                    self._statistics(centered),
                    summary,
                ],
                dim=-1,
            )
        )
        lag_ids = torch.arange(config.route_history, device=cells.device)
        layer_ids = torch.arange(config.layers, device=cells.device)
        cells = (
            cells
            + self.lag_embedding(lag_ids)[None, None]
            + self.layer_embedding(layer_ids)[None, :, None]
        )
        cells = cells * available[..., None].to(cells.dtype)
        temporal = cells.reshape(batch * config.layers, config.route_history, -1)
        temporal_available = available.reshape(batch * config.layers, -1)
        safe, padding = safe_padding_mask(temporal_available)
        if (~temporal_available.any(dim=-1)).any():
            temporal = temporal.clone()
            temporal[~temporal_available.any(dim=-1), 0] = 0
        for block in self.temporal_blocks:
            temporal = block(temporal, padding_mask=padding)
        # Lag zero is the newest committed token by contract.
        route = temporal[:, 0].reshape(batch, config.layers, config.route_width)
        layer_available = available.any(dim=-1)
        safe_layers, layer_padding = safe_padding_mask(layer_available)
        if (~layer_available.any(dim=-1)).any():
            route = route.clone()
            route[~layer_available.any(dim=-1), 0] = 0
        for block in self.layer_blocks:
            route = block(route, padding_mask=layer_padding)
        local = self.local(self.local_norm(route).transpose(1, 2)).transpose(1, 2)
        route = route + self.local_mix(torch.nn.functional.silu(local))
        route = route * layer_available[..., None].to(route.dtype)
        return self.output_norm(route)


__all__ = ["RouteGridEncoder"]
