"""Exactly two rounds of detached-self-conditioning route refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .common import SelfAttentionResidual, zero_linear
from .config import HARPRTTConfig
from .heads import exact_projected_marginals


@dataclass(frozen=True)
class TrajectoryOutput:
    scores: Tensor
    marginals: Tensor
    round_scores: tuple[Tensor, Tensor]
    round_marginals: tuple[Tensor, Tensor]
    corrections: tuple[Tensor, Tensor]
    cardinality_errors: tuple[Tensor, Tensor]


class _TrajectoryRound(nn.Module):
    def __init__(self, config: HARPRTTConfig) -> None:
        super().__init__()
        self.config = config
        input_width = (
            config.model_width
            + config.router_rank
            + config.route_width
            + config.model_width
        )
        self.fusion = nn.Sequential(
            nn.Linear(input_width, config.decoder_ffn_width),
            nn.SiLU(),
            nn.Linear(config.decoder_ffn_width, config.model_width),
            nn.RMSNorm(config.model_width),
        )
        self.layer_path = SelfAttentionResidual(
            config.model_width,
            config.attention_heads,
            config.decoder_ffn_width,
            config.dropout,
        )
        self.horizon_path = SelfAttentionResidual(
            config.model_width,
            config.attention_heads,
            config.decoder_ffn_width,
            config.dropout,
        )
        self.previous_layer = nn.Linear(config.router_rank, config.model_width)
        self.previous_horizon = nn.Linear(config.router_rank, config.model_width)
        self.correction = zero_linear(
            nn.Linear(config.model_width, config.experts)
        )

    def forward(
        self,
        endpoint: Tensor,
        marginals: Tensor,
        route: Tensor,
        target: Tensor,
        expert_keys: Tensor,
    ) -> Tensor:
        config = self.config
        # This expectation lives in the frozen router K coordinate system.
        # Preserve it in FP32 before feeding it to autocastable learned blocks.
        with torch.autocast(device_type=marginals.device.type, enabled=False):
            selected = torch.einsum(
                "bhle,ler->bhlr", marginals.float(), expert_keys.float()
            ) / float(config.exact_k)
        route_source = route[:, None].expand(-1, config.active_horizons, -1, -1)
        target_source = target[:, None].expand(-1, config.active_horizons, -1, -1)
        hidden = self.fusion(
            torch.cat([endpoint, selected, route_source, target_source], dim=-1)
        )
        # Explicit causal transition messages are derived only from predicted
        # marginals.  Position zero receives an exact zero message.
        zero_layer = torch.zeros_like(selected[:, :, :1])
        previous_layer = torch.cat([zero_layer, selected[:, :, :-1]], dim=2)
        zero_horizon = torch.zeros_like(selected[:, :1])
        previous_horizon = torch.cat([zero_horizon, selected[:, :-1]], dim=1)
        hidden = (
            hidden
            + self.previous_layer(previous_layer)
            + self.previous_horizon(previous_horizon)
        )
        batch, horizons, layers, width = hidden.shape
        layer_axis = hidden.reshape(batch * horizons, layers, width)
        layer_axis = self.layer_path(layer_axis)
        hidden = layer_axis.reshape(batch, horizons, layers, width)
        horizon_axis = hidden.permute(0, 2, 1, 3).reshape(
            batch * layers, horizons, width
        )
        causal = torch.triu(
            torch.ones(horizons, horizons, dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        horizon_axis = self.horizon_path(
            horizon_axis, attention_mask=causal
        )
        hidden = horizon_axis.reshape(batch, layers, horizons, width).permute(
            0, 2, 1, 3
        )
        return self.correction(hidden)


class TwoRoundTrajectoryRefiner(nn.Module):
    """Refine direct endpoint scores twice, always residual to round zero."""

    def __init__(self, config: HARPRTTConfig, expert_keys: Tensor) -> None:
        super().__init__()
        if config.trajectory_rounds != 2:
            raise ValueError("trajectory refiner requires exactly two rounds")
        self.config = config
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.rounds = nn.ModuleList([_TrajectoryRound(config) for _ in range(2)])

    def forward(
        self,
        direct_scores: Tensor,
        initial_marginals: Tensor,
        *,
        endpoint: Tensor,
        route: Tensor,
        target: Tensor,
    ) -> TrajectoryOutput:
        base = direct_scores
        conditioned = initial_marginals
        score_rounds: list[Tensor] = []
        marginal_rounds: list[Tensor] = []
        corrections: list[Tensor] = []
        errors: list[Tensor] = []
        for block in self.rounds:
            if self.config.detach_self_conditioning:
                conditioned = conditioned.detach()
            correction = block(
                endpoint,
                conditioned,
                route,
                target,
                self.expert_keys,
            )
            scores = base + correction
            _, marginals, error = exact_projected_marginals(
                scores, self.config.exact_k
            )
            score_rounds.append(scores)
            marginal_rounds.append(marginals)
            corrections.append(correction)
            errors.append(error)
            conditioned = marginals
        return TrajectoryOutput(
            scores=score_rounds[-1],
            marginals=marginal_rounds[-1],
            round_scores=(score_rounds[0], score_rounds[1]),
            round_marginals=(marginal_rounds[0], marginal_rounds[1]),
            corrections=(corrections[0], corrections[1]),
            cardinality_errors=(errors[0], errors[1]),
        )


__all__ = ["TrajectoryOutput", "TwoRoundTrajectoryRefiner"]
