"""Label-only compact target supervision contract for ShadowRoute."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .counterfactual import SEALED_SPLITS
from .shadow_cache import validate_tree_topology


SHADOW_CAPTURE_SCHEMA = "harp_shadowroute_target_companion_v1"


def assert_shadow_labels_allowed(*, split: str, training: bool, enabled: bool) -> None:
    normalized = str(split).strip().lower()
    if enabled and (normalized != "train" or normalized in SEALED_SPLITS):
        raise PermissionError("ShadowRoute teacher labels are outer-train only")
    if enabled and not training:
        raise PermissionError("inference cannot request ShadowRoute teacher labels")


@dataclass(frozen=True)
class ShadowCaptureDimensions:
    nodes: int = 16
    layers: int = 40
    hidden_width: int = 2048
    experts: int = 256
    exact_k: int = 8
    router_rank: int = 255

    def validate(self) -> None:
        for value in (
            self.nodes, self.layers, self.hidden_width, self.experts,
            self.exact_k, self.router_rank,
        ):
            if value < 1:
                raise ValueError("capture dimensions must be positive")
        if self.exact_k > self.experts or self.layers < 2:
            raise ValueError("capture route geometry is invalid")


def empty_shadow_capture(
    dimensions: ShadowCaptureDimensions = ShadowCaptureDimensions(),
) -> dict[str, Tensor]:
    dimensions.validate()
    n, l, d = dimensions.nodes, dimensions.layers, dimensions.hidden_width
    e, k, r = dimensions.experts, dimensions.exact_k, dimensions.router_rank
    return {
        "node_mask": torch.zeros(n, dtype=torch.bool),
        "parent_local_indices": torch.full((n,), -1, dtype=torch.int64),
        "post_attention_residuals": torch.zeros(n, l, d, dtype=torch.bfloat16),
        "routed_deltas": torch.zeros(n, l, d, dtype=torch.bfloat16),
        "post_layer_states": torch.zeros(n, l, d, dtype=torch.bfloat16),
        "router_logits": torch.zeros(n, l, e, dtype=torch.bfloat16),
        "selected_ids": torch.full((n, l, k), -1, dtype=torch.int32),
        "selected_weights": torch.zeros(n, l, k, dtype=torch.bfloat16),
        "expert_effect_next_router": torch.zeros(
            n, l - 1, k, r, dtype=torch.bfloat16
        ),
        "valid": torch.zeros(n, l, dtype=torch.bool),
        "raw_effect_audit_mask": torch.zeros(n, dtype=torch.bool),
    }


def validate_shadow_capture(
    tensors: Mapping[str, Tensor],
    dimensions: ShadowCaptureDimensions = ShadowCaptureDimensions(),
) -> None:
    reference = empty_shadow_capture(dimensions)
    if set(tensors) != set(reference):
        raise ValueError("ShadowRoute companion keys disagree with the v1 contract")
    for name, expected in reference.items():
        value = tensors[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(
                f"{name} has {tuple(value.shape)}/{value.dtype}, "
                f"expected {tuple(expected.shape)}/{expected.dtype}"
            )
    mask = tensors["node_mask"].bool()
    validate_tree_topology(
        tensors["parent_local_indices"], mask, max_depth=4
    )
    valid = tensors["valid"].bool()
    if not torch.equal(valid, mask[:, None].expand_as(valid)):
        raise ValueError("ShadowRoute layer validity must equal node visibility")
    ids = tensors["selected_ids"].long()
    active_ids = ids[valid]
    if active_ids.numel() and bool(
        ((active_ids < 0) | (active_ids >= dimensions.experts)).any()
    ):
        raise ValueError("active selected expert ID is invalid")
    if bool((ids[~valid] != -1).any()):
        raise ValueError("padded selected expert IDs must remain sentinel -1")
    weights = tensors["selected_weights"].float()
    if not torch.isfinite(weights[valid]).all() or bool((weights[valid] < 0).any()):
        raise ValueError("selected execution weights are invalid")
    if bool((weights[~valid] != 0).any()):
        raise ValueError("padded execution weights must remain zero")
    for name in (
        "post_attention_residuals", "routed_deltas", "post_layer_states",
        "router_logits", "expert_effect_next_router",
    ):
        if not torch.isfinite(tensors[name].float()).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if bool((tensors["raw_effect_audit_mask"] & ~mask).any()):
        raise ValueError("raw effect audit selects padded nodes")


def estimated_shadow_capture_bytes(
    source_positions: int,
    *,
    dimensions: ShadowCaptureDimensions = ShadowCaptureDimensions(),
    raw_audit_fraction: float = 0.05,
) -> int:
    dimensions.validate()
    if source_positions < 1 or not 0.0 <= raw_audit_fraction <= 1.0:
        raise ValueError("capture size inputs are invalid")
    n, l, d = dimensions.nodes, dimensions.layers, dimensions.hidden_width
    e, k, r = dimensions.experts, dimensions.exact_k, dimensions.router_rank
    # Three full BF16 state channels, routes, and projected per-expert effects.
    per_node = (
        3 * l * d * 2
        + l * e * 2
        + l * k * (4 + 2)
        + (l - 1) * k * r * 2
    )
    # Full unweighted selected-expert output vectors exist only in the audit.
    per_node += round(raw_audit_fraction * l * k * d * 2)
    return int(source_positions) * n * per_node


__all__ = [
    "SHADOW_CAPTURE_SCHEMA",
    "ShadowCaptureDimensions",
    "assert_shadow_labels_allowed",
    "empty_shadow_capture",
    "estimated_shadow_capture_bytes",
    "validate_shadow_capture",
]
