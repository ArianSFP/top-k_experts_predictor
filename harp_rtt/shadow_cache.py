"""Causal hybrid-cache traversal for ShadowRoute tree execution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class ShadowNodeResult:
    router_logits: Tensor
    selected_ids: Tensor
    selected_weights: Tensor
    hidden_state: Tensor
    cache: Any
    vocabulary_log_probabilities: Tensor | None = None


@dataclass(frozen=True)
class ShadowTreeResult:
    router_logits: Tensor
    selected_ids: Tensor
    selected_weights: Tensor
    hidden_states: Tensor
    valid: Tensor
    caches: tuple[Any | None, ...]
    vocabulary_log_probabilities: Tensor | None = None


def clone_hybrid_cache(cache: Any) -> Any:
    """Reference sibling-isolation operation.

    The official Qwen cache mixes mutable recurrent/conv state with dynamic
    K/V tensors.  A deep copy is deliberately the correctness reference; an
    optimized copy-on-write implementation must prove parity against it.
    """

    return copy.deepcopy(cache)


def validate_tree_topology(
    parent_indices: Tensor, node_mask: Tensor, *, max_depth: int = 4
) -> Tensor:
    if parent_indices.ndim != 1 or node_mask.shape != parent_indices.shape:
        raise ValueError("tree topology must be one-dimensional and aligned")
    if parent_indices.dtype not in {torch.int32, torch.int64}:
        raise TypeError("parent indices must be integer")
    mask = node_mask.bool()
    count = int(mask.sum().item())
    if count < 1 or not torch.equal(
        mask, torch.arange(mask.numel(), device=mask.device) < count
    ):
        raise ValueError("node mask must be a nonempty contiguous prefix")
    parent = parent_indices.long()
    if int(parent[0]) != -1 or bool((parent[count:] != -1).any()):
        raise ValueError("root/padded parent indices are invalid")
    depth = torch.zeros_like(parent)
    depth[0] = 1
    for index in range(1, count):
        value = int(parent[index])
        if not 0 <= value < index:
            raise ValueError("parents must precede children")
        depth[index] = depth[value] + 1
        if int(depth[index]) > max_depth:
            raise ValueError("tree exceeds the configured depth")
    return depth


class ShadowTreeRunner:
    """Execute each visible token node once with isolated parent cache state."""

    def __init__(
        self,
        step: Callable[[int, Any, int], ShadowNodeResult],
        *,
        clone_cache: Callable[[Any], Any] = clone_hybrid_cache,
        max_depth: int = 4,
    ) -> None:
        self.step = step
        self.clone_cache = clone_cache
        self.max_depth = int(max_depth)

    def run(
        self,
        *,
        root_cache: Any,
        token_ids: Tensor,
        parent_indices: Tensor,
        node_mask: Tensor,
    ) -> ShadowTreeResult:
        if token_ids.shape != parent_indices.shape or token_ids.ndim != 1:
            raise ValueError("token IDs and tree topology must align")
        if token_ids.dtype not in {torch.int32, torch.int64}:
            raise TypeError("token IDs must be integer")
        depth = validate_tree_topology(
            parent_indices, node_mask, max_depth=self.max_depth
        )
        count = int(node_mask.bool().sum().item())
        outputs: list[ShadowNodeResult] = []
        caches: list[Any | None] = [None] * token_ids.numel()
        for index in range(count):
            parent = int(parent_indices[index])
            source_cache = root_cache if parent < 0 else caches[parent]
            if source_cache is None:  # pragma: no cover - topology invariant
                raise RuntimeError("parent cache is unavailable")
            isolated = self.clone_cache(source_cache)
            result = self.step(int(token_ids[index]), isolated, int(depth[index]))
            if not isinstance(result, ShadowNodeResult):
                raise TypeError("shadow step must return ShadowNodeResult")
            outputs.append(result)
            caches[index] = result.cache
        reference = outputs[0]
        layers, experts = reference.router_logits.shape
        k = reference.selected_ids.shape[-1]
        hidden_width = reference.hidden_state.shape[-1]
        logits = reference.router_logits.new_zeros(token_ids.numel(), layers, experts)
        ids = reference.selected_ids.new_full((token_ids.numel(), layers, k), -1)
        weights = reference.selected_weights.new_zeros(token_ids.numel(), layers, k)
        hidden = reference.hidden_state.new_zeros(token_ids.numel(), layers, hidden_width)
        vocabulary = None
        if reference.vocabulary_log_probabilities is not None:
            if reference.vocabulary_log_probabilities.ndim != 1:
                raise ValueError("node vocabulary log probabilities must be one-dimensional")
            vocabulary = reference.vocabulary_log_probabilities.new_full(
                (token_ids.numel(), reference.vocabulary_log_probabilities.numel()),
                -torch.inf,
            )
        for index, result in enumerate(outputs):
            if result.router_logits.shape != (layers, experts):
                raise ValueError("node router-logit geometry changed within a tree")
            if result.selected_ids.shape != (layers, k) or result.selected_weights.shape != (layers, k):
                raise ValueError("node route geometry changed within a tree")
            if result.hidden_state.shape != (layers, hidden_width):
                raise ValueError("node hidden geometry changed within a tree")
            if (result.vocabulary_log_probabilities is None) != (vocabulary is None):
                raise ValueError("node vocabulary capture changed within a tree")
            logits[index] = result.router_logits
            ids[index] = result.selected_ids
            weights[index] = result.selected_weights
            hidden[index] = result.hidden_state
            if vocabulary is not None:
                assert result.vocabulary_log_probabilities is not None
                if result.vocabulary_log_probabilities.shape != vocabulary.shape[1:]:
                    raise ValueError("node vocabulary geometry changed within a tree")
                vocabulary[index] = result.vocabulary_log_probabilities
        valid = node_mask.bool()[:, None].expand(-1, layers)
        return ShadowTreeResult(
            logits, ids, weights, hidden, valid, tuple(caches), vocabulary
        )


__all__ = [
    "ShadowNodeResult", "ShadowTreeResult", "ShadowTreeRunner",
    "clone_hybrid_cache", "validate_tree_topology",
]
