"""Leakage-safe RouteMTP batch preparation and anytime visibility masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from .delta_batch import factual_branch_indices
from .geometry import CenteredRouterGeometry
from .routemtp import ANYTIME_BUDGETS, RouteMTPConfig


CAUSAL_INPUT_KEYS = frozenset({
    "captured_fused_state",
    "captured_router_input",
    "captured_post_moe_hidden",
    "captured_vocabulary_head_input",
    "captured_mtp_router_logits",
    "captured_mtp_selected_ids",
    "captured_mtp_selected_weights",
    "node_token_ids",
    "node_token_embeddings",
    "node_parent_ids",
    "node_depths",
    "node_child_ranks",
    "node_mask",
    "native_edge_log_probabilities",
    "native_path_log_probabilities",
    "anytime_node_masks",
    "current_post_layer",
    "current_queries",
    "current_centered_router_logits",
    "current_selected_ids",
    "current_selected_weights",
    "current_final_hidden",
})


@dataclass(frozen=True)
class RouteMTPPreparedBatch:
    model_inputs: Mapping[str, Tensor]
    teacher_node_logits: Tensor
    teacher_node_ids: Tensor
    teacher_node_queries: Tensor
    branch_valid: Tensor
    factual_ids: Tensor
    factual_valid: Tensor
    factual_branch_indices: Tensor


def assert_causal_routemtp_inputs(inputs: Mapping[str, Any]) -> None:
    unexpected = sorted(set(inputs) - CAUSAL_INPUT_KEYS)
    if unexpected:
        raise PermissionError(f"RouteMTP model inputs contain undeclared fields: {unexpected}")
    for name, value in inputs.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"RouteMTP causal input {name} must be a tensor")
    forbidden = (
        "future", "counterfactual", "acceptance", "target_path",
        "target_edge", "prefix_hash", "factual_branch",
    )
    for name in inputs:
        if any(part in name.lower() for part in forbidden):
            raise PermissionError(f"label-like RouteMTP model input rejected: {name}")


def build_anytime_node_masks(
    node_mask: Tensor,
    nested_budget_masks: Tensor,
) -> Tensor:
    """Return ancestor-closed masks for budgets 1/4/8/16.

    The immutable companion stores masks for 4/8/16.  Budget one is the exact
    H1 root only.  Cross-budget nesting is rechecked rather than assumed.
    """

    if node_mask.ndim != 2:
        raise ValueError("node mask must be [B,N]")
    batch, nodes = node_mask.shape
    if nested_budget_masks.shape != (batch, 3, nodes):
        raise ValueError("nested budget masks must be [B,3,N]")
    root = torch.zeros_like(node_mask)
    root[:, 0] = True
    result = torch.cat((root[:, None], nested_budget_masks.bool()), dim=1)
    if bool((result & ~node_mask[:, None].bool()).any()):
        raise ValueError("anytime mask selects unavailable nodes")
    for lower in range(len(ANYTIME_BUDGETS) - 1):
        if bool((result[:, lower] & ~result[:, lower + 1]).any()):
            raise ValueError("anytime masks are not nested")
    if not bool(result[:, :, 0].all()):
        raise ValueError("every anytime budget must retain H1")
    return result


def _token_lookup(token_ids: Tensor, embedding: Tensor) -> Tensor:
    if embedding.ndim != 2:
        raise ValueError("frozen token embedding must be [V,D]")
    if bool(((token_ids < 0) | (token_ids >= embedding.shape[0])).any()):
        raise ValueError("tree token ID lies outside the frozen embedding")
    return embedding[token_ids.long()].detach()


def prepare_routemtp_batch(
    batch: Mapping[str, Any],
    *,
    geometry: CenteredRouterGeometry,
    token_embedding: Tensor,
    config: RouteMTPConfig,
) -> RouteMTPPreparedBatch:
    """Split causal runtime tensors from train-only route labels."""

    config.validate(); geometry.validate()
    inputs = batch.get("inputs"); targets = batch.get("targets")
    if not isinstance(inputs, Mapping) or not isinstance(targets, Mapping):
        raise TypeError("RouteMTP batch requires inputs and targets mappings")
    if "counterfactual" in inputs:
        raise PermissionError("counterfactual labels leaked into RouteMTP inputs")
    tree = inputs.get("tree"); current = inputs.get("current")
    counterfactual = targets.get("counterfactual")
    if not all(isinstance(value, Mapping) for value in (tree, current, counterfactual)):
        raise TypeError("RouteMTP requires tree/current inputs and counterfactual targets")
    assert isinstance(tree, Mapping) and isinstance(current, Mapping)
    assert isinstance(counterfactual, Mapping)

    states = tree["states"]
    if states.ndim != 4 or states.shape[2] != 4:
        raise ValueError("captured MTP states must be [B,N,4,D]")
    batch_size, nodes, _, hidden = states.shape
    if (nodes, hidden) != (config.max_nodes, config.hidden_width):
        raise ValueError("captured MTP state geometry differs from RouteMTP config")
    node_mask = tree["mask"].bool()
    if not torch.equal(node_mask, counterfactual["node_mask"].bool()):
        raise ValueError("base tree and companion node masks differ")
    parents = tree["parent"].long()
    depths = tree["depth"].long()
    if not torch.equal(parents, counterfactual["parent_local_indices"].long()):
        raise ValueError("base tree and companion parents differ")
    if not torch.equal(depths, counterfactual["depth"].long()):
        raise ValueError("base tree and companion depths differ")

    token_ids = tree["token_ids"].long()
    safe_token_ids = torch.where(node_mask, token_ids, torch.zeros_like(token_ids))
    token_embeddings = _token_lookup(safe_token_ids, token_embedding)
    token_embeddings = token_embeddings * node_mask[..., None]
    current_router_input = current["normalized_target_router_input_a"]
    current_queries = geometry.encode_router_inputs(current_router_input)
    current_logits = current["raw_target_router_logits"].float()
    current_centered = current_logits - current_logits.mean(-1, keepdim=True)

    model_inputs: dict[str, Tensor] = {
        "captured_fused_state": states[:, :, 0],
        "captured_router_input": states[:, :, 2],
        "captured_post_moe_hidden": states[:, :, 1],
        "captured_vocabulary_head_input": states[:, :, 3],
        "captured_mtp_router_logits": tree["router_logits"],
        "captured_mtp_selected_ids": tree["selected_ids"].long(),
        "captured_mtp_selected_weights": tree["execution_weights"],
        "node_token_ids": safe_token_ids,
        "node_token_embeddings": token_embeddings,
        "node_parent_ids": parents,
        "node_depths": depths,
        "node_child_ranks": tree["child_ranks"].long(),
        "node_mask": node_mask,
        "native_edge_log_probabilities": counterfactual["source_edge_logp"].float().nan_to_num(0.0),
        "native_path_log_probabilities": counterfactual["source_path_logp"].float().nan_to_num(0.0),
        "anytime_node_masks": build_anytime_node_masks(
            node_mask, counterfactual["budget_node_masks"].bool()
        ),
        "current_post_layer": current["post_moe_residual_xplus"],
        "current_queries": current_queries,
        "current_centered_router_logits": current_centered,
        "current_selected_ids": current["selected_expert_ids"].long(),
        "current_selected_weights": current["selected_execution_weights"],
        "current_final_hidden": inputs["final_hidden"],
    }
    assert_causal_routemtp_inputs(model_inputs)

    teacher_logits = counterfactual["router_logits"].float().clone()
    teacher_ids = counterfactual["selected_ids"].long().clone()
    teacher_queries = counterfactual["query_coordinates"].float().clone()
    branch_valid = counterfactual["valid"].bool().clone()
    future_logits = targets["future_router_logits"].float()
    future_ids = targets["future_selected_ids"].long()
    future_inputs = targets["future_router_inputs"].float()
    future_valid = targets["future_available"].bool()
    if future_logits.shape != (
        batch_size, config.horizons, config.target_layers, config.experts
    ):
        raise ValueError("factual future router geometry is invalid")
    # The companion deliberately masks H1; attach its exact factual root only
    # on the label side.
    teacher_logits[:, 0] = future_logits[:, 0]
    teacher_ids[:, 0] = future_ids[:, 0]
    teacher_queries[:, 0] = geometry.encode_router_inputs(future_inputs[:, 0])
    branch_valid[:, 0] = future_valid[:, 0]

    factual = factual_branch_indices(
        exact_prefix_hashes=tree["exact_prefix_hashes"],
        node_depths=depths,
        node_available=node_mask,
        future_prefix_hashes=targets["future_prefix_hashes"],
    )
    # Sentinel N is the explicit OTHER/fallback outcome.
    factual = torch.where(factual == nodes, factual, factual.clamp_min(0))
    return RouteMTPPreparedBatch(
        model_inputs=model_inputs,
        teacher_node_logits=teacher_logits,
        teacher_node_ids=teacher_ids,
        teacher_node_queries=teacher_queries,
        branch_valid=branch_valid,
        factual_ids=future_ids,
        factual_valid=future_valid,
        factual_branch_indices=factual,
    )


__all__ = [
    "CAUSAL_INPUT_KEYS",
    "RouteMTPPreparedBatch",
    "assert_causal_routemtp_inputs",
    "build_anytime_node_masks",
    "prepare_routemtp_batch",
]
