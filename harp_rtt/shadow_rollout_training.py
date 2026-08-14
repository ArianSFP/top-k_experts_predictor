"""Training-only factual rollout primitives for HARP-ShadowRoute.

The serving rollout API in :mod:`harp_rtt.shadow_rollout` never accepts target
labels.  This module deliberately keeps the gradient-bearing hooks and cache
materialisation used by factual distillation in a separate training-only API.
"""

from __future__ import annotations

import copy
from contextlib import AbstractContextManager
from typing import Any, Callable

import torch
from torch import Tensor

from .shadow_backbone import InstalledShadowBackbone, exact_prefix_experts, resolve_text_layers


def _map_cache_value(value: Any, function: Callable[[Tensor], Tensor]) -> Any:
    if isinstance(value, Tensor):
        return function(value)
    if isinstance(value, list):
        return [_map_cache_value(item, function) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_cache_value(item, function) for item in value)
    if isinstance(value, dict):
        return {key: _map_cache_value(item, function) for key, item in value.items()}
    return value


def map_hybrid_cache_tensors(cache: Any, function: Callable[[Tensor], Tensor]) -> Any:
    """Shallow-copy a Transformers hybrid cache and transform every state tensor.

    Qwen3.5-MoE mixes dynamic K/V layers with mutable linear-attention conv and
    recurrent state.  Copying the layer objects prevents sibling/sample state
    contamination; transforming every tensor removes inference/autograd lineage.
    """

    layers = getattr(cache, "layers", None)
    if not isinstance(layers, list) or not layers:
        raise TypeError("ShadowRoute training requires a populated hybrid cache")
    result = copy.copy(cache)
    copied_layers = []
    for layer in layers:
        copied = copy.copy(layer)
        for name, value in vars(layer).items():
            setattr(copied, name, _map_cache_value(value, function))
        copied_layers.append(copied)
    result.layers = copied_layers
    return result


def clone_detached_hybrid_cache(
    cache: Any,
    *,
    device: torch.device | str | None = None,
) -> Any:
    """Return an isolated, ordinary-tensor cache with no graph ancestry."""

    target_device = None if device is None else torch.device(device)

    def transform(value: Tensor) -> Tensor:
        copied = value.detach().clone()
        return copied if target_device is None else copied.to(target_device)

    return map_hybrid_cache_tensors(cache, transform)


def cache_to_cpu(cache: Any) -> Any:
    """Detach one exact prefix cache into pageable host memory."""

    return map_hybrid_cache_tensors(
        cache, lambda value: value.detach().to(device="cpu", copy=True)
    )


@torch.no_grad()
def exact_prefix_cache_for_training(
    installed: InstalledShadowBackbone,
    authoritative_prefix: list[int],
) -> Any:
    """Execute the authoritative prefix with native experts and normal tensors."""

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
    return clone_detached_hybrid_cache(output.past_key_values)


class ShadowTrainingHooks(AbstractContextManager):
    """Capture gradient-bearing routed effects, router logits and layer states."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.rows: dict[int, dict[str, Tensor]] = {layer: {} for layer in range(40)}
        self.handles: list[Any] = []

    def __enter__(self):
        for layer_id, layer in enumerate(resolve_text_layers(self.model)):
            row = self.rows[layer_id]

            def gate_post(_module, _args, output, row=row):
                logits, weights, ids = output
                row["router_logits"] = logits
                row["selected_weights"] = weights
                row["selected_ids"] = ids

            def experts_post(_module, _args, output, row=row):
                row["routed_delta"] = output

            def block_post(_module, _args, output, row=row):
                row["hidden_state"] = output[0] if isinstance(output, tuple) else output

            self.handles.extend(
                [
                    layer.mlp.gate.register_forward_hook(gate_post),
                    layer.mlp.experts.register_forward_hook(experts_post),
                    layer.register_forward_hook(block_post),
                ]
            )
        return self

    def clear(self) -> None:
        for row in self.rows.values():
            row.clear()

    def stacked(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        required = (
            "router_logits",
            "selected_ids",
            "selected_weights",
            "routed_delta",
            "hidden_state",
        )
        if any(any(name not in self.rows[layer] for name in required) for layer in range(40)):
            raise RuntimeError("ShadowRoute training hook capture is incomplete")
        logits = torch.stack([self.rows[layer]["router_logits"][0] for layer in range(40)])
        ids = torch.stack([self.rows[layer]["selected_ids"][0] for layer in range(40)])
        weights = torch.stack([self.rows[layer]["selected_weights"][0] for layer in range(40)])
        routed = torch.stack(
            [self.rows[layer]["routed_delta"][0, -1] for layer in range(40)]
        )
        hidden = torch.stack(
            [self.rows[layer]["hidden_state"][0, -1] for layer in range(40)]
        )
        return logits, ids, weights, routed, hidden

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


__all__ = [
    "ShadowTrainingHooks",
    "cache_to_cpu",
    "clone_detached_hybrid_cache",
    "exact_prefix_cache_for_training",
    "map_hybrid_cache_tensors",
]
