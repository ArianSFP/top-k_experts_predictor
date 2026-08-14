"""Exact-prefix, frozen-backbone ShadowRoute token/tree rollout."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import Tensor

from .shadow_backbone import InstalledShadowBackbone, exact_prefix_experts, resolve_text_layers
from .shadow_cache import ShadowNodeResult, ShadowTreeResult, ShadowTreeRunner


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

    def stacked(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        required = ("router_logits", "selected_ids", "selected_weights", "hidden_state")
        if any(any(name not in self.rows[layer] for name in required) for layer in range(40)):
            raise RuntimeError("ShadowRoute hook capture is incomplete")
        logits = torch.stack([self.rows[layer]["router_logits"][0] for layer in range(40)])
        ids = torch.stack([self.rows[layer]["selected_ids"][0] for layer in range(40)])
        weights = torch.stack([self.rows[layer]["selected_weights"][0] for layer in range(40)])
        hidden = torch.stack([
            self.rows[layer]["hidden_state"][0, -1] for layer in range(40)
        ])
        return logits, ids, weights, hidden

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


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
) -> ShadowTreeResult:
    """Restore an exact target prefix, then execute future nodes with shadows."""

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

    return ShadowTreeRunner(step).run(
        root_cache=prefix_cache,
        token_ids=token_ids.to(parameter.device),
        parent_indices=parent_indices.to(parameter.device),
        node_mask=node_mask.to(parameter.device),
    )


__all__ = [
    "ShadowRouteHooks",
    "exact_prefix_cache",
    "run_shadow_tree",
]
