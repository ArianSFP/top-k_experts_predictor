"""Sparse routed-expert substitutes for HARP-ShadowRoute v1.

The classes in this module intentionally implement the call contract used by
the official Qwen3.5-MoE routed expert module::

    forward(hidden_states, top_k_index, top_k_weights) -> routed_delta

They do not predict routes.  The exact frozen target router remains
authoritative and supplies both the selected IDs and their execution weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RoutedExperts(Protocol):
    def __call__(
        self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor
    ) -> Tensor: ...


@dataclass(frozen=True)
class ShadowExpertConfig:
    hidden_width: int = 2048
    experts: int = 256
    exact_k: int = 8
    shadow_width: int = 16
    target_intermediate_width: int = 512

    def validate(self) -> None:
        for name in (
            "hidden_width", "experts", "exact_k", "shadow_width",
            "target_intermediate_width",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.exact_k > self.experts:
            raise ValueError("exact_k exceeds the expert count")
        if self.shadow_width > self.target_intermediate_width:
            raise ValueError("shadow width exceeds the target expert width")

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return asdict(self)


def _validate_routed_inputs(
    hidden_states: Tensor,
    top_k_index: Tensor,
    top_k_weights: Tensor,
    *,
    hidden_width: int,
    experts: int,
) -> tuple[Tensor, Tensor, Tensor, tuple[int, ...]]:
    if hidden_states.ndim < 2 or hidden_states.shape[-1] != hidden_width:
        raise ValueError("hidden states must end in the configured hidden width")
    leading = tuple(hidden_states.shape[:-1])
    if top_k_index.shape[:-1] != leading or top_k_weights.shape != top_k_index.shape:
        raise ValueError("route IDs/weights do not align with hidden states")
    if top_k_index.shape[-1] < 1:
        raise ValueError("at least one routed expert is required")
    if top_k_index.dtype not in {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    }:
        raise TypeError("top_k_index must use an integer dtype")
    ids = top_k_index.reshape(-1, top_k_index.shape[-1]).long()
    weights = top_k_weights.reshape_as(ids).to(hidden_states.dtype)
    hidden = hidden_states.reshape(-1, hidden_width)
    if ids.numel() and bool(((ids < 0) | (ids >= experts)).any()):
        raise ValueError("route ID lies outside the expert namespace")
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("route weights must be finite and non-negative")
    return hidden, ids, weights, leading


def target_selected_expert_outputs(
    hidden_states: Tensor,
    selected_ids: Tensor,
    target_gate_up: Tensor,
    target_down: Tensor,
) -> Tensor:
    """Execute only the selected frozen target experts.

    This is the layer-local teacher used by S0 and S2. It intentionally
    mirrors the authoritative Qwen capture implementation: SiLU gate, element
    wise gate/up product, then the expert-specific down projection.
    """

    if target_gate_up.ndim != 3 or target_down.ndim != 3:
        raise ValueError("target expert tensors must be rank three")
    experts, twice_intermediate, hidden_width = target_gate_up.shape
    if twice_intermediate % 2:
        raise ValueError("target gate/up width must be even")
    intermediate = twice_intermediate // 2
    if target_down.shape != (experts, hidden_width, intermediate):
        raise ValueError("target gate/up and down tensors disagree")
    hidden, ids, _weights, leading = _validate_routed_inputs(
        hidden_states,
        selected_ids,
        torch.ones_like(selected_ids, dtype=hidden_states.dtype),
        hidden_width=hidden_width,
        experts=experts,
    )
    output = torch.zeros(
        hidden.shape[0], ids.shape[1], hidden_width,
        dtype=hidden.dtype, device=hidden.device,
    )
    gate_up = target_gate_up.to(device=hidden.device, dtype=hidden.dtype)
    down = target_down.to(device=hidden.device, dtype=hidden.dtype)
    for expert_id in torch.unique(ids).tolist():
        positions = (ids == int(expert_id)).nonzero(as_tuple=False)
        token_index, slot_index = positions[:, 0], positions[:, 1]
        gate, up = F.linear(hidden[token_index], gate_up[int(expert_id)]).chunk(2, -1)
        output[token_index, slot_index] = F.linear(
            F.silu(gate) * up, down[int(expert_id)]
        )
    return output.reshape(*leading, ids.shape[1], hidden_width)


def target_neuron_importance(target_gate_up: Tensor, target_down: Tensor) -> Tensor:
    """Static, deterministic importance used to initialize width-16 shadows."""

    if target_gate_up.ndim != 3 or target_down.ndim != 3:
        raise ValueError("target expert tensors must be rank three")
    experts, twice_intermediate, hidden_width = target_gate_up.shape
    if twice_intermediate % 2:
        raise ValueError("target gate/up width must be even")
    intermediate = twice_intermediate // 2
    if target_down.shape != (experts, hidden_width, intermediate):
        raise ValueError("target expert tensor shapes disagree")
    gate, up = target_gate_up.float().chunk(2, dim=1)
    return (
        torch.linalg.vector_norm(gate, dim=-1)
        * torch.linalg.vector_norm(up, dim=-1)
        * torch.linalg.vector_norm(target_down.float(), dim=1)
    )


class SwiGLUDraftExpert(nn.Module):
    """One compact dense SwiGLU expert."""

    def __init__(self, hidden_width: int, intermediate_width: int) -> None:
        super().__init__()
        if hidden_width < 1 or intermediate_width < 1:
            raise ValueError("draft expert dimensions must be positive")
        self.hidden_width = int(hidden_width)
        self.intermediate_width = int(intermediate_width)
        self.gate_up_proj = nn.Linear(
            self.hidden_width, 2 * self.intermediate_width, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_width, self.hidden_width, bias=False
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class ExactTop1PlusDraftExperts(nn.Module):
    """Privileged S0 mechanism oracle: native top-one plus learned residual."""

    def __init__(
        self,
        native_experts: RoutedExperts,
        draft_expert: SwiGLUDraftExpert,
        *,
        experts: int = 256,
    ) -> None:
        super().__init__()
        if isinstance(native_experts, nn.Module):
            self.native_experts = native_experts
            self._native_callable = None
        else:  # pragma: no cover - official implementation is an nn.Module
            self.native_experts = None
            self._native_callable = native_experts
        self.draft_expert = draft_expert
        self.experts = int(experts)

    def _native(self, hidden: Tensor, ids: Tensor, weights: Tensor) -> Tensor:
        function = self.native_experts if self.native_experts is not None else self._native_callable
        if function is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("native expert implementation is absent")
        return function(hidden, ids, weights)

    def forward(
        self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor
    ) -> Tensor:
        hidden, ids, weights, leading = _validate_routed_inputs(
            hidden_states, top_k_index, top_k_weights,
            hidden_width=self.draft_expert.hidden_width, experts=self.experts,
        )
        exact_top1 = self._native(hidden, ids[:, :1], weights[:, :1])
        residual = self.draft_expert(hidden)
        return (exact_top1 + residual).reshape(*leading, hidden.shape[-1])


class SharedResidualExperts(nn.Module):
    """S1 control: one layer-local expert predicts the entire routed sum."""

    def __init__(self, draft_expert: SwiGLUDraftExpert, *, experts: int = 256) -> None:
        super().__init__()
        self.draft_expert = draft_expert
        self.experts = int(experts)

    def forward(
        self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor
    ) -> Tensor:
        hidden, _ids, _weights, leading = _validate_routed_inputs(
            hidden_states, top_k_index, top_k_weights,
            hidden_width=self.draft_expert.hidden_width, experts=self.experts,
        )
        return self.draft_expert(hidden).reshape(*leading, hidden.shape[-1])


class IndexedShadowExperts(nn.Module):
    """S2 expert-indexed miniature pool with exact sparse weighted execution.

    An untrained expert does not silently emit a random vector.  If any slot
    for a token refers to an untrained cell, that token deterministically uses
    the layer-level S1 fallback for its complete routed residual.
    """

    def __init__(
        self,
        config: ShadowExpertConfig = ShadowExpertConfig(),
        *,
        fallback: SharedResidualExperts | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.gate_up_proj = nn.Parameter(torch.empty(
            config.experts, 2 * config.shadow_width, config.hidden_width
        ))
        self.down_proj = nn.Parameter(torch.empty(
            config.experts, config.hidden_width, config.shadow_width
        ))
        self.register_buffer(
            "trained_experts", torch.zeros(config.experts, dtype=torch.bool)
        )
        self.fallback = fallback
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.config.hidden_width)
        nn.init.uniform_(self.gate_up_proj, -bound, bound)
        nn.init.uniform_(self.down_proj, -bound, bound)

    def set_trained_counts(self, counts: Tensor, *, minimum: int = 128) -> None:
        if counts.shape != (self.config.experts,) or counts.dtype not in {
            torch.int32, torch.int64,
        }:
            raise ValueError("expert counts must be an integer [experts] tensor")
        if minimum < 1 or bool((counts < 0).any()):
            raise ValueError("training-count threshold is invalid")
        self.trained_experts.copy_(counts.to(self.trained_experts.device) >= minimum)

    @torch.no_grad()
    def initialize_from_target_neurons(
        self,
        target_gate_up: Tensor,
        target_down: Tensor,
        importance: Tensor,
    ) -> Tensor:
        """Copy the most important exact SwiGLU neurons from each target expert."""

        cfg = self.config
        expected_gate = (cfg.experts, 2 * cfg.target_intermediate_width, cfg.hidden_width)
        expected_down = (cfg.experts, cfg.hidden_width, cfg.target_intermediate_width)
        if target_gate_up.shape != expected_gate or target_down.shape != expected_down:
            raise ValueError("target expert tensors disagree with the Qwen contract")
        if importance.shape != (cfg.experts, cfg.target_intermediate_width):
            raise ValueError("importance must be [experts,target_intermediate_width]")
        if not torch.isfinite(importance).all():
            raise ValueError("importance contains NaN or Inf")
        # stable descending order makes the lower neuron ID authoritative on ties.
        selected = torch.argsort(
            importance.float(), dim=-1, descending=True, stable=True
        )[:, : cfg.shadow_width]
        gather_gate = selected[..., None].expand(-1, -1, cfg.hidden_width)
        target_gate = target_gate_up[:, : cfg.target_intermediate_width]
        target_up = target_gate_up[:, cfg.target_intermediate_width :]
        self.gate_up_proj[:, : cfg.shadow_width].copy_(
            target_gate.gather(1, gather_gate).to(self.gate_up_proj)
        )
        self.gate_up_proj[:, cfg.shadow_width :].copy_(
            target_up.gather(1, gather_gate).to(self.gate_up_proj)
        )
        gather_down = selected[:, None, :].expand(-1, cfg.hidden_width, -1)
        self.down_proj.copy_(target_down.gather(2, gather_down).to(self.down_proj))
        return selected

    def selected_unweighted(self, hidden_states: Tensor, selected_ids: Tensor) -> Tensor:
        cfg = self.config
        hidden, ids, _weights, leading = _validate_routed_inputs(
            hidden_states,
            selected_ids,
            torch.ones_like(selected_ids, dtype=hidden_states.dtype),
            hidden_width=cfg.hidden_width,
            experts=cfg.experts,
        )
        output = torch.zeros(
            hidden.shape[0], ids.shape[1], cfg.hidden_width,
            dtype=hidden.dtype, device=hidden.device,
        )
        for expert_id in torch.unique(ids).tolist():
            positions = (ids == int(expert_id)).nonzero(as_tuple=False)
            token_index, slot_index = positions[:, 0], positions[:, 1]
            current = hidden[token_index]
            projection = F.linear(current, self.gate_up_proj[int(expert_id)])
            gate, up = projection.chunk(2, dim=-1)
            value = F.linear(F.silu(gate) * up, self.down_proj[int(expert_id)])
            output[token_index, slot_index] = value
        return output.reshape(*leading, ids.shape[1], cfg.hidden_width)

    def forward(
        self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor
    ) -> Tensor:
        cfg = self.config
        hidden, ids, weights, leading = _validate_routed_inputs(
            hidden_states, top_k_index, top_k_weights,
            hidden_width=cfg.hidden_width, experts=cfg.experts,
        )
        values = self.selected_unweighted(hidden, ids).reshape(
            hidden.shape[0], ids.shape[1], cfg.hidden_width
        )
        routed = (values * weights[..., None]).sum(dim=1)
        untrained = ~self.trained_experts[ids]
        fallback_rows = untrained.any(dim=-1)
        if bool(fallback_rows.any()):
            if self.fallback is None:
                raise RuntimeError("selected route contains an untrained expert without fallback")
            fallback_value = self.fallback(
                hidden[fallback_rows], ids[fallback_rows], weights[fallback_rows]
            )
            routed = routed.clone()
            routed[fallback_rows] = fallback_value.reshape(-1, cfg.hidden_width)
        return routed.reshape(*leading, cfg.hidden_width)


def shadow_pool_parameter_count(
    *, layers: int = 40, experts: int = 256, hidden_width: int = 2048,
    shadow_width: int = 16,
) -> int:
    values = (layers, experts, hidden_width, shadow_width)
    if any(value < 1 for value in values):
        raise ValueError("shadow-pool dimensions must be positive")
    return 3 * layers * experts * hidden_width * shadow_width


__all__ = [
    "ExactTop1PlusDraftExperts",
    "IndexedShadowExperts",
    "RoutedExperts",
    "ShadowExpertConfig",
    "SharedResidualExperts",
    "SwiGLUDraftExpert",
    "shadow_pool_parameter_count",
    "target_neuron_importance",
    "target_selected_expert_outputs",
]
