"""Post-MoE target-state transition ceiling for HARP-DeltaRoute."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .content_transition import ContentTransitionConfig, ContentTransitionOutput
from .exact_k import stable_topk


class LayerSpecificPostMoeProbe(nn.Module):
    """Predict the next router query from the actual current block output."""

    def __init__(
        self,
        config: ContentTransitionConfig,
        input_basis: Tensor,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
        *,
        state_rank: int = 192,
    ) -> None:
        super().__init__()
        config.validate()
        if state_rank < 1:
            raise ValueError("post-MoE state rank must be positive")
        if input_basis.shape != (
            config.layers, config.hidden_width, config.router_rank
        ) or expert_keys.shape != (
            config.layers, config.experts, config.router_rank
        ) or centered_bias.shape != (config.layers, config.experts):
            raise ValueError("post-MoE probe router geometry is invalid")
        if rank_mask.shape != (config.layers, config.router_rank):
            raise ValueError("post-MoE probe rank mask is invalid")
        self.config = config
        self.state_rank = state_rank
        self.register_buffer("input_basis", input_basis.detach().float().clone())
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.register_buffer("centered_bias", centered_bias.detach().float().clone())
        self.register_buffer("rank_mask", rank_mask.detach().bool().clone())
        transitions = config.layers - 1
        self.state_down = nn.Parameter(torch.empty(
            transitions, config.hidden_width, state_rank
        ))
        self.state_up = nn.Parameter(torch.empty(
            transitions, state_rank, config.router_rank
        ))
        self.state_bias = nn.Parameter(torch.zeros(transitions, state_rank))
        self.expert_effect = nn.Parameter(torch.empty(
            transitions, config.experts, config.effect_width
        ))
        self.effect_up = nn.Parameter(torch.empty(
            transitions, config.effect_width, config.router_rank
        ))
        self.output_bias = nn.Parameter(torch.zeros(
            transitions, config.router_rank
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
        nn.init.normal_(self.state_down, std=0.02)
        nn.init.normal_(self.state_up, std=0.02)
        nn.init.normal_(self.expert_effect, std=0.02)
        nn.init.normal_(self.effect_up, std=0.02)
        nn.init.normal_(self.control_down, std=0.02)

    def forward(
        self,
        combined_states: Tensor,
        selected_ids: Tensor,
        selected_weights: Tensor,
        *,
        use_router_blind_content: bool = True,
    ) -> ContentTransitionOutput:
        config = self.config
        if combined_states.shape[-2:] != (
            config.layers, 2 * config.hidden_width
        ):
            raise ValueError("combined post-MoE/router state must end in [L,2D]")
        post_moe_states, router_inputs = combined_states.split(
            config.hidden_width, dim=-1
        )
        expected = post_moe_states.shape[:-1] + (config.exact_k,)
        if selected_ids.shape != expected or selected_weights.shape != expected:
            raise ValueError("post-MoE route labels disagree with states")
        states = post_moe_states[..., :-1, :].float()
        if not use_router_blind_content:
            states = torch.zeros_like(states)
        hidden = torch.einsum(
            "...ld,lda->...la", states, self.state_down
        ) + self.state_bias
        state_delta = torch.einsum(
            "...la,lar->...lr", torch.nn.functional.silu(hidden), self.state_up
        )
        ids = selected_ids[..., :-1, :].long()
        weights = selected_weights[..., :-1, :].float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        layer = torch.arange(config.layers - 1, device=ids.device).view(
            *((1,) * (ids.ndim - 2)), config.layers - 1, 1
        )
        effects = self.expert_effect[layer, ids]
        route = (effects * weights[..., None]).sum(-2)
        route_delta = torch.einsum("...la,lar->...lr", route, self.effect_up)
        current_q = torch.einsum(
            "...ld,ldr->...lr", router_inputs.float(), self.input_basis
        )[..., :-1, :]
        current_q = current_q * self.rank_mask[:-1].to(current_q.dtype)
        control = current_q * self.control_diagonal
        control_low = torch.einsum(
            "...lr,lra->...la", current_q, self.control_down
        )
        control = control + torch.einsum(
            "...la,lar->...lr", control_low, self.control_up
        )
        queries = control + state_delta + route_delta + self.output_bias
        queries = queries * self.rank_mask[1:].to(queries.dtype)
        scores = torch.einsum(
            "...lr,ler->...le", queries, self.expert_keys[1:]
        ) + self.centered_bias[1:]
        placeholder = torch.zeros(
            *post_moe_states.shape[:-1], config.router_rank,
            device=post_moe_states.device, dtype=torch.float32,
        )
        return ContentTransitionOutput(
            queries, scores, stable_topk(scores, config.exact_k),
            placeholder, post_moe_states.float(),
        )


__all__ = ["LayerSpecificPostMoeProbe"]
