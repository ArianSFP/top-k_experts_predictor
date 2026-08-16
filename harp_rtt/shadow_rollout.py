"""Exact-prefix, frozen-backbone ShadowRoute token/tree rollout."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .shadow_backbone import InstalledShadowBackbone, exact_prefix_experts, resolve_text_layers
from .shadow_cache import (
    ShadowNodeResult, ShadowTreeResult, ShadowTreeRunner, compact_visible_tree,
)


class ShadowRouteHooks(AbstractContextManager):
    """Capture the exact frozen router output and post-layer shadow state."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.rows: dict[int, dict[str, Tensor]] = {
            layer: {} for layer in range(40)
        }
        self.handles: list[Any] = []

    def __enter__(self):
        for layer_id, layer in enumerate(resolve_text_layers(self.model)):
            row = self.rows[layer_id]

            def gate_post(_module, _args, output, row=row):
                logits, weights, ids = output
                row["router_logits"] = logits.detach()
                row["selected_weights"] = weights.detach()
                row["selected_ids"] = ids.detach()

            def block_post(_module, _args, output, row=row):
                value = output[0] if isinstance(output, tuple) else output
                row["hidden_state"] = value.detach()

            self.handles.extend(
                [
                    layer.mlp.gate.register_forward_hook(gate_post),
                    layer.register_forward_hook(block_post),
                ]
            )
        return self

    def clear(self) -> None:
        for row in self.rows.values():
            row.clear()

    @staticmethod
    def _token_row(value: Tensor, token_index: int) -> Tensor:
        if value.ndim == 2:
            return value[token_index]
        if value.ndim == 3 and value.shape[0] == 1:
            return value[0, token_index]
        raise ValueError("ShadowRoute hook tensor has an unsupported token geometry")

    def stacked(self, *, token_index: int = -1) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        required = ("router_logits", "selected_ids", "selected_weights", "hidden_state")
        if any(any(name not in self.rows[layer] for name in required) for layer in range(40)):
            raise RuntimeError("ShadowRoute hook capture is incomplete")
        logits = torch.stack([
            self._token_row(self.rows[layer]["router_logits"], token_index)
            for layer in range(40)
        ])
        ids = torch.stack([
            self._token_row(self.rows[layer]["selected_ids"], token_index)
            for layer in range(40)
        ])
        weights = torch.stack([
            self._token_row(self.rows[layer]["selected_weights"], token_index)
            for layer in range(40)
        ])
        hidden = torch.stack([
            self._token_row(self.rows[layer]["hidden_state"], token_index)
            for layer in range(40)
        ])
        return logits, ids, weights, hidden

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


@dataclass(frozen=True)
class ShadowPrefixResult:
    cache: Any
    router_logits: Tensor
    selected_ids: Tensor
    selected_weights: Tensor
    hidden_state: Tensor


@torch.inference_mode()
def exact_prefix_state(
    installed: InstalledShadowBackbone,
    hooks: ShadowRouteHooks,
    authoritative_prefix: list[int],
) -> ShadowPrefixResult:
    """Replay the exact committed prefix and retain its final-token route."""

    if not authoritative_prefix:
        raise ValueError("authoritative prefix cannot be empty")
    parameter = next(installed.model.parameters())
    hooks.clear()
    with exact_prefix_experts(installed):
        output = installed.model(
            input_ids=torch.tensor(
                [authoritative_prefix], dtype=torch.long, device=parameter.device
            ),
            use_cache=True,
            output_hidden_states=False,
            output_router_logits=True,
            return_dict=True,
        )
    logits, ids, weights, hidden = hooks.stacked(token_index=-1)
    return ShadowPrefixResult(
        cache=output.past_key_values,
        router_logits=logits.detach(),
        selected_ids=ids.detach(),
        selected_weights=weights.detach(),
        hidden_state=hidden.detach(),
    )


@torch.inference_mode()
def exact_prefix_cache(
    installed: InstalledShadowBackbone, authoritative_prefix: list[int]
) -> Any:
    if not authoritative_prefix:
        raise ValueError("authoritative prefix cannot be empty")
    parameter = next(installed.model.parameters())
    with exact_prefix_experts(installed):
        output = installed.model(
            input_ids=torch.tensor(
                [authoritative_prefix], dtype=torch.long, device=parameter.device
            ),
            use_cache=True,
            output_hidden_states=False,
            output_router_logits=True,
            return_dict=True,
        )
    return output.past_key_values


@torch.inference_mode()
def run_shadow_tree(
    installed: InstalledShadowBackbone,
    hooks: ShadowRouteHooks,
    *,
    authoritative_prefix: list[int],
    token_ids: Tensor,
    parent_indices: Tensor,
    node_mask: Tensor,
    prefix_cache: Any | None = None,
) -> ShadowTreeResult:
    """Restore an exact target prefix, then execute future nodes with shadows."""

    if prefix_cache is None:
        prefix_cache = exact_prefix_cache(installed, authoritative_prefix)
    parameter = next(installed.model.parameters())

    def step(token_id: int, cache: Any, _depth: int) -> ShadowNodeResult:
        hooks.clear()
        output = installed.model(
            input_ids=torch.tensor(
                [[token_id]], dtype=torch.long, device=parameter.device
            ),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=False,
            output_router_logits=True,
            return_dict=True,
        )
        logits, ids, weights, hidden = hooks.stacked()
        vocabulary_logits = output.logits[0, -1].float()
        return ShadowNodeResult(
            router_logits=logits,
            selected_ids=ids,
            selected_weights=weights,
            hidden_state=hidden,
            cache=output.past_key_values,
            vocabulary_log_probabilities=(
                vocabulary_logits - torch.logsumexp(vocabulary_logits, dim=-1)
            ).detach(),
        )

    compact_tokens, compact_parents, original = compact_visible_tree(
        token_ids.to(parameter.device),
        parent_indices.to(parameter.device),
        node_mask.to(parameter.device),
    )
    compact = ShadowTreeRunner(step).run(
        root_cache=prefix_cache,
        token_ids=compact_tokens,
        parent_indices=compact_parents,
        node_mask=torch.ones_like(compact_tokens, dtype=torch.bool),
    )
    if original.numel() == token_ids.numel():
        return compact
    nodes = token_ids.numel()
    logits = compact.router_logits.new_zeros(
        nodes, *compact.router_logits.shape[1:]
    )
    ids = compact.selected_ids.new_full(
        (nodes, *compact.selected_ids.shape[1:]), -1
    )
    weights = compact.selected_weights.new_zeros(
        nodes, *compact.selected_weights.shape[1:]
    )
    hidden = compact.hidden_states.new_zeros(
        nodes, *compact.hidden_states.shape[1:]
    )
    logits[original] = compact.router_logits
    ids[original] = compact.selected_ids
    weights[original] = compact.selected_weights
    hidden[original] = compact.hidden_states
    vocabulary = None
    if compact.vocabulary_log_probabilities is not None:
        vocabulary = compact.vocabulary_log_probabilities.new_full(
            (nodes, compact.vocabulary_log_probabilities.shape[-1]), -torch.inf
        )
        vocabulary[original] = compact.vocabulary_log_probabilities
    caches: list[Any | None] = [None] * nodes
    for compact_index, original_index in enumerate(original.tolist()):
        caches[original_index] = compact.caches[compact_index]
    valid = node_mask.bool().to(logits.device)[:, None].expand(
        -1, logits.shape[1]
    )
    return ShadowTreeResult(
        logits, ids, weights, hidden, valid, tuple(caches), vocabulary
    )


__all__ = [
    "ShadowPrefixResult",
    "ShadowRouteHooks",
    "exact_prefix_cache",
    "exact_prefix_state",
    "run_shadow_tree",
]
