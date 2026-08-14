"""Router-blind content diagnostics for HARP-DeltaRoute.

This module is deliberately teacher-only.  It asks whether the part of a
target router input that is invisible to the current router contains the
information needed to predict the next layer's route.  A positive result is
evidence for distilling a causal content state; it is not a serving model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .exact_k import stable_topk


@dataclass(frozen=True)
class ContentTransitionConfig:
    layers: int = 40
    experts: int = 256
    hidden_width: int = 2048
    router_rank: int = 255
    exact_k: int = 8
    latent_width: int = 256
    effect_width: int = 64
    transition_width: int = 512
    content_adapter_rank: int = 8
    output_adapter_rank: int = 8
    dropout: float = 0.05

    def validate(self) -> None:
        values = (
            self.layers, self.experts, self.hidden_width, self.router_rank,
            self.exact_k, self.latent_width, self.effect_width,
            self.transition_width, self.content_adapter_rank,
            self.output_adapter_rank,
        )
        if any(value < 1 for value in values):
            raise ValueError("content-transition dimensions must be positive")
        if self.layers < 2 or self.exact_k > self.experts:
            raise ValueError("content-transition layer/expert geometry is invalid")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("content-transition dropout must lie in [0,1)")

    def to_dict(self) -> dict[str, int | float]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ContentTransitionOutput:
    predicted_queries: Tensor
    predicted_scores: Tensor
    selected_ids: Tensor
    router_queries: Tensor
    router_blind_inputs: Tensor


class FullStateTransitionProbe(nn.Module):
    """Predict ``q[l+1]`` from the full factual target state at layer ``l``.

    The full state is split exactly into router-visible and router-blind
    components with the frozen SVD basis.  ``use_router_blind_content=False``
    provides a matched ablation with identical learned weights.
    """

    def __init__(
        self,
        config: ContentTransitionConfig,
        input_basis: Tensor,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
    ) -> None:
        super().__init__()
        config.validate()
        if input_basis.shape != (
            config.layers, config.hidden_width, config.router_rank
        ):
            raise ValueError("content probe input basis has invalid geometry")
        if expert_keys.shape != (
            config.layers, config.experts, config.router_rank
        ):
            raise ValueError("content probe expert keys have invalid geometry")
        if centered_bias.shape != (config.layers, config.experts):
            raise ValueError("content probe centered bias has invalid geometry")
        if rank_mask.shape != (config.layers, config.router_rank):
            raise ValueError("content probe rank mask has invalid geometry")
        self.config = config
        self.register_buffer("input_basis", input_basis.detach().float().clone())
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.register_buffer("centered_bias", centered_bias.detach().float().clone())
        self.register_buffer("rank_mask", rank_mask.detach().bool().clone())

        transitions = config.layers - 1
        self.content_input = nn.Linear(
            config.hidden_width, config.latent_width, bias=False
        )
        self.content_down = nn.Parameter(torch.empty(
            transitions, config.hidden_width, config.content_adapter_rank
        ))
        self.content_up = nn.Parameter(torch.zeros(
            transitions, config.content_adapter_rank, config.latent_width
        ))
        self.query_input = nn.Linear(config.router_rank, config.latent_width)
        self.expert_effect = nn.Parameter(torch.empty(
            transitions, config.experts, config.effect_width
        ))
        self.effect_input = nn.Linear(config.effect_width, config.latent_width)
        self.layer_embedding = nn.Embedding(transitions, config.latent_width)
        self.transition = nn.Sequential(
            nn.RMSNorm(config.latent_width),
            nn.Linear(config.latent_width, config.transition_width),
            nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(config.transition_width, config.latent_width),
            nn.SiLU(),
        )
        self.query_output = nn.Linear(config.latent_width, config.router_rank)
        self.output_down = nn.Parameter(torch.empty(
            transitions, config.latent_width, config.output_adapter_rank
        ))
        self.output_up = nn.Parameter(torch.zeros(
            transitions, config.output_adapter_rank, config.router_rank
        ))
        self.control_diagonal = nn.Parameter(torch.ones(
            transitions, config.router_rank
        ))
        self.control_bias = nn.Parameter(torch.zeros(
            transitions, config.router_rank
        ))
        nn.init.normal_(self.content_down, std=0.02)
        nn.init.normal_(self.output_down, std=0.02)
        nn.init.normal_(self.expert_effect, std=0.02)

    def decompose(self, router_inputs: Tensor) -> tuple[Tensor, Tensor]:
        config = self.config
        if router_inputs.shape[-2:] != (config.layers, config.hidden_width):
            raise ValueError("factual router inputs must end in [L,D]")
        values = router_inputs.float()
        queries = torch.einsum("...ld,ldr->...lr", values, self.input_basis)
        queries = queries * self.rank_mask.to(queries.dtype)
        visible = torch.einsum("...lr,ldr->...ld", queries, self.input_basis)
        return queries, values - visible

    def forward(
        self,
        router_inputs: Tensor,
        selected_ids: Tensor,
        selected_weights: Tensor,
        *,
        use_router_blind_content: bool = True,
    ) -> ContentTransitionOutput:
        config = self.config
        expected = router_inputs.shape[:-1] + (config.exact_k,)
        if selected_ids.shape != expected or selected_weights.shape != expected:
            raise ValueError("factual route labels disagree with router inputs")
        queries, blind = self.decompose(router_inputs)
        current_q = queries[..., :-1, :]
        current_blind = blind[..., :-1, :]
        if not use_router_blind_content:
            current_blind = torch.zeros_like(current_blind)

        content = self.content_input(current_blind)
        content_low = torch.einsum(
            "...ld,lda->...la", current_blind, self.content_down
        )
        content = content + torch.einsum(
            "...la,law->...lw", content_low, self.content_up
        )

        ids = selected_ids[..., :-1, :].long()
        weights = selected_weights[..., :-1, :].float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        transition_index = torch.arange(
            config.layers - 1, device=ids.device
        ).view(*((1,) * (ids.ndim - 2)), config.layers - 1, 1)
        effects = self.expert_effect[transition_index, ids]
        route_message = (effects * weights[..., None]).sum(-2)

        hidden = (
            content + self.query_input(current_q)
            + self.effect_input(route_message)
            + self.layer_embedding.weight
        )
        transformed = self.transition(hidden)
        nonlinear = self.query_output(transformed)
        output_low = torch.einsum(
            "...lw,lwa->...la", transformed, self.output_down
        )
        nonlinear = nonlinear + torch.einsum(
            "...la,lar->...lr", output_low, self.output_up
        )
        predicted_q = (
            current_q * self.control_diagonal + self.control_bias + nonlinear
        )
        predicted_q = predicted_q * self.rank_mask[1:].to(predicted_q.dtype)
        scores = torch.einsum(
            "...lr,ler->...le", predicted_q, self.expert_keys[1:]
        ) + self.centered_bias[1:]
        return ContentTransitionOutput(
            predicted_q, scores, stable_topk(scores, config.exact_k),
            queries, blind,
        )


__all__ = [
    "ContentTransitionConfig", "ContentTransitionOutput",
    "FullStateTransitionProbe",
]
