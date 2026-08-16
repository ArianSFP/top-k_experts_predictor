"""Optional nonlinear expert/router adaptation for RouteMTP R3."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SwitchableExpertLoRA(nn.Module):
    """Rank-r gate/up/down adaptation under recomputed native MTP IDs."""

    def __init__(self, base: nn.Module, rank: int = 8) -> None:
        super().__init__()
        required = ("gate_up_proj", "down_proj", "num_experts", "act_fn")
        if any(not hasattr(base, name) for name in required):
            raise TypeError("unsupported packed MTP expert module")
        if rank < 1:
            raise ValueError("expert LoRA rank must be positive")
        self.base = base.requires_grad_(False)
        self.rank = int(rank)
        experts, doubled_intermediate, hidden = base.gate_up_proj.shape
        down_experts, down_hidden, intermediate = base.down_proj.shape
        if experts != down_experts or hidden != down_hidden or doubled_intermediate != 2 * intermediate:
            raise ValueError("packed MTP expert tensor geometry is invalid")
        self.gate_up_down = nn.Parameter(torch.empty(experts, rank, hidden))
        self.gate_up_up = nn.Parameter(torch.zeros(experts, doubled_intermediate, rank))
        self.down_down = nn.Parameter(torch.empty(experts, rank, intermediate))
        self.down_up = nn.Parameter(torch.zeros(experts, hidden, rank))
        nn.init.kaiming_uniform_(self.gate_up_down, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_down, a=math.sqrt(5))
        self.enabled = False

    @contextmanager
    def active(self, enabled: bool = True) -> Iterator[None]:
        previous = self.enabled; self.enabled = bool(enabled)
        try:
            yield
        finally:
            self.enabled = previous

    def forward(self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
        if not self.enabled:
            return self.base(hidden_states, top_k_index, top_k_weights)
        hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        ids = top_k_index.reshape(hidden.shape[0], -1)
        weights = top_k_weights.reshape_as(ids)
        result = torch.zeros_like(hidden)
        for expert_tensor in torch.unique(ids):
            expert = int(expert_tensor.item())
            positions = (ids == expert).nonzero(as_tuple=False)
            token_index = positions[:, 0]; slot_index = positions[:, 1]
            current = hidden[token_index]
            gate_up = F.linear(current, self.base.gate_up_proj[expert])
            low = F.linear(current, self.gate_up_down[expert])
            gate_up = gate_up + F.linear(low, self.gate_up_up[expert])
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = self.base.act_fn(gate) * up
            output = F.linear(intermediate, self.base.down_proj[expert])
            down_low = F.linear(intermediate, self.down_down[expert])
            output = output + F.linear(down_low, self.down_up[expert])
            result.index_add_(
                0,
                token_index,
                output * weights[token_index, slot_index, None],
            )
        return result.reshape_as(hidden_states)


class SwitchableMTPRouterResidual(nn.Module):
    """Route-only MTP router residual; never supervised with target expert IDs."""

    def __init__(self, base: nn.Module, hidden_width: int, experts: int, rank: int = 16, top_k: int = 8) -> None:
        super().__init__()
        if min(hidden_width, experts, rank, top_k) < 1 or top_k > experts:
            raise ValueError("invalid MTP router adapter geometry")
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(hidden_width, rank, bias=False)
        self.up = nn.Linear(rank, experts, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5)); nn.init.zeros_(self.up.weight)
        self.top_k = int(top_k)
        self.active_top_k = int(top_k)
        self.temperature = 1.0
        self.enabled = False
        self._regularization_terms: list[tuple[Tensor, Tensor]] = []

    @contextmanager
    def active(self, enabled: bool = True) -> Iterator[None]:
        previous = self.enabled; self.enabled = bool(enabled)
        if enabled:
            self._regularization_terms = []
        try:
            yield
        finally:
            self.enabled = previous

    def set_training_progress(self, progress: float) -> None:
        """Apply the declared sparse-soft-to-hard R3 router curriculum."""

        if not 0.0 <= float(progress) <= 1.0:
            raise ValueError("router training progress must lie in [0,1]")
        self.temperature = 1.0 - 0.8 * min(float(progress) / 0.6, 1.0)
        self.active_top_k = min(16, self.up.out_features) if progress < 0.6 else self.top_k

    def set_inference_mode(self) -> None:
        self.temperature = 0.2
        self.active_top_k = self.top_k

    def forward(self, hidden_states: Tensor):
        base_output = self.base(hidden_states)
        if not self.enabled:
            return base_output
        if not isinstance(base_output, (tuple, list)) or len(base_output) != 3:
            raise TypeError("native MTP router must return logits, weights, IDs")
        base_logits = base_output[0]
        logits = base_logits + self.up(self.down(hidden_states)).reshape_as(base_logits)
        base_probability = torch.softmax(base_logits.float(), -1)
        trust = F.kl_div(
            torch.log_softmax(logits.float(), -1), base_probability,
            reduction="batchmean",
        )
        mean_load = torch.softmax(logits.float(), -1).mean(
            tuple(range(logits.ndim - 1))
        )
        uniform = torch.full_like(mean_load, 1.0 / mean_load.numel())
        balance = (mean_load - uniform).square().mean()
        self._regularization_terms.append((trust, balance))
        probabilities = torch.softmax(logits.float() / float(self.temperature), dim=-1)
        selected_probabilities, selected_ids = torch.topk(
            probabilities, self.active_top_k, dim=-1
        )
        selected_weights = selected_probabilities / selected_probabilities.sum(-1, keepdim=True)
        return logits, selected_weights.to(logits.dtype), selected_ids

    def regularization(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Base-router trust KL and batch load-balance penalty."""

        with torch.no_grad():
            base_logits = self.base(hidden_states)[0].float()
        adapted = base_logits + self.up(self.down(hidden_states)).reshape_as(base_logits)
        base_probability = torch.softmax(base_logits, -1)
        trust = F.kl_div(torch.log_softmax(adapted, -1), base_probability, reduction="batchmean")
        mean_load = torch.softmax(adapted, -1).mean(tuple(range(adapted.ndim - 1)))
        uniform = torch.full_like(mean_load, 1.0 / mean_load.numel())
        balance = (mean_load - uniform).square().mean()
        return trust, balance

    def collected_regularization(self) -> tuple[Tensor, Tensor]:
        if not self._regularization_terms:
            zero = self.up.weight.sum() * 0.0
            return zero, zero
        trust = torch.stack([value[0] for value in self._regularization_terms]).mean()
        balance = torch.stack([value[1] for value in self._regularization_terms]).mean()
        return trust, balance


__all__ = ["SwitchableExpertLoRA", "SwitchableMTPRouterResidual"]
