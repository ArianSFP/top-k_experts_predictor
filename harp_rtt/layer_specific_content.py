"""Layer-specific full-state ceiling for router-transition diagnostics."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .content_transition import ContentTransitionConfig, ContentTransitionOutput
from .exact_k import stable_topk


class LayerSpecificFullStateProbe(nn.Module):
    """Use a distinct rank-128 router-blind content map per transition."""

    def __init__(
        self,
        config: ContentTransitionConfig,
        input_basis: Tensor,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
        *,
        content_rank: int = 128,
    ) -> None:
        super().__init__()
        config.validate()
        if content_rank < 1:
            raise ValueError("layer-specific content rank must be positive")
        if input_basis.shape != (
            config.layers, config.hidden_width, config.router_rank
        ) or expert_keys.shape != (
            config.layers, config.experts, config.router_rank
        ) or centered_bias.shape != (config.layers, config.experts):
            raise ValueError("layer-specific probe frozen geometry is invalid")
        if rank_mask.shape != (config.layers, config.router_rank):
            raise ValueError("layer-specific probe rank mask is invalid")
        self.config = config
        self.content_rank = content_rank
        self.register_buffer("input_basis", input_basis.detach().float().clone())
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.register_buffer("centered_bias", centered_bias.detach().float().clone())
        self.register_buffer("rank_mask", rank_mask.detach().bool().clone())
        transitions = config.layers - 1
        self.content_down = nn.Parameter(torch.empty(
            transitions, config.hidden_width, content_rank
        ))
        self.content_up = nn.Parameter(torch.empty(
            transitions, content_rank, config.router_rank
        ))
        self.content_bias = nn.Parameter(torch.zeros(transitions, content_rank))
        self.expert_effect = nn.Parameter(torch.empty(
            transitions, config.experts, config.effect_width
        ))
        self.effect_up = nn.Parameter(torch.empty(
            transitions, config.effect_width, config.router_rank
        ))
        self.control_diagonal = nn.Parameter(torch.ones(
            transitions, config.router_rank
        ))
        self.control_down = nn.Parameter(torch.empty(
            transitions, config.router_rank, config.content_adapter_rank
        ))
        self.control_up = nn.Parameter(torch.zeros(
            transitions, config.content_adapter_rank, config.router_rank
        ))
        self.output_bias = nn.Parameter(torch.zeros(
            transitions, config.router_rank
        ))
        nn.init.normal_(self.content_down, std=0.02)
        nn.init.normal_(self.content_up, std=0.02)
        nn.init.normal_(self.expert_effect, std=0.02)
        nn.init.normal_(self.effect_up, std=0.02)
        nn.init.normal_(self.control_down, std=0.02)

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
        content = torch.einsum(
            "...ld,lda->...la", current_blind, self.content_down
        ) + self.content_bias
        content_delta = torch.einsum(
            "...la,lar->...lr", torch.nn.functional.silu(content), self.content_up
        )
        ids = selected_ids[..., :-1, :].long()
        weights = selected_weights[..., :-1, :].float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        layer = torch.arange(config.layers - 1, device=ids.device).view(
            *((1,) * (ids.ndim - 2)), config.layers - 1, 1
        )
        effects = self.expert_effect[layer, ids]
        route_message = (effects * weights[..., None]).sum(-2)
        route_delta = torch.einsum(
            "...la,lar->...lr", route_message, self.effect_up
        )
        control = current_q * self.control_diagonal
        control_low = torch.einsum(
            "...lr,lra->...la", current_q, self.control_down
        )
        control = control + torch.einsum(
            "...la,lar->...lr", control_low, self.control_up
        )
        predicted_q = control + content_delta + route_delta + self.output_bias
        predicted_q = predicted_q * self.rank_mask[1:].to(predicted_q.dtype)
        scores = torch.einsum(
            "...lr,ler->...le", predicted_q, self.expert_keys[1:]
        ) + self.centered_bias[1:]
        return ContentTransitionOutput(
            predicted_q, scores, stable_topk(scores, config.exact_k), queries, blind
        )


__all__ = ["LayerSpecificFullStateProbe"]
