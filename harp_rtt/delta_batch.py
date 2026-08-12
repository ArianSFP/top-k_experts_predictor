"""Audited rich-dataset to HARP-DeltaTree v3 input adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from .delta import HARPDeltaConfig
from .static_artifacts import frozen_token_embeddings


@dataclass(frozen=True)
class DeltaPreparedBatch:
    model_inputs: Mapping[str, Tensor]
    factual_branch_index: Tensor


def factual_branch_indices(
    *,
    exact_prefix_hashes: Tensor,
    node_depths: Tensor,
    node_available: Tensor,
    future_prefix_hashes: Tensor,
) -> Tensor:
    """Return matching node index or explicit OTHER=N for each horizon."""

    if exact_prefix_hashes.ndim != 3 or exact_prefix_hashes.shape[-1] != 32:
        raise ValueError("tree exact prefix hashes must be [B,N,32]")
    batch, nodes, _ = exact_prefix_hashes.shape
    if node_depths.shape != (batch, nodes) or node_available.shape != (batch, nodes):
        raise ValueError("tree prefix topology has invalid geometry")
    if future_prefix_hashes.shape != (batch, 4, 32):
        raise ValueError("future factual prefix hashes must be [B,4,32]")
    result = torch.full((batch, 4), nodes, dtype=torch.long, device=node_depths.device)
    for horizon in range(4):
        same_hash = (exact_prefix_hashes == future_prefix_hashes[:, horizon, None]).all(-1)
        matches = same_hash & node_available.bool() & (node_depths.long() == horizon + 1)
        if bool((matches.sum(-1) > 1).any()):
            raise ValueError("adaptive tree contains duplicate factual prefixes at one depth")
        exists = matches.any(-1)
        chosen = matches.long().argmax(-1)
        result[:, horizon] = torch.where(exists, chosen, result[:, horizon])
    if bool((result[:, 0] == nodes).any()):
        raise ValueError("adaptive tree lacks its exact factual H1 root")
    return result


def _formal_tree_metadata(tree: Mapping[str, Tensor], config: HARPDeltaConfig) -> Tensor:
    path = tree["path_log_probabilities"].float()
    components = (
        path,
        tree["local_probabilities"].float(),
        tree["source_ready"].float(),
        tree["depth"].float() / 4.0,
        tree["child_ranks"].float() / 64.0,
        tree["first_divergence_depths"].float() / 4.0,
        tree["cumulative_path_ranks"].float() / float(config.max_tree_nodes),
        tree["sibling_counts"].float() / float(config.max_tree_nodes),
    )
    if any(value.shape != path.shape for value in components):
        raise ValueError("formal adaptive-tree metadata fields disagree")
    return torch.stack(components, dim=-1)


def prepare_delta_batch(
    batch: Mapping[str, Any],
    *,
    anchor_scores: Tensor,
    token_embedding: Tensor,
    input_basis: Tensor,
    rank_mask: Tensor,
    config: HARPDeltaConfig,
) -> DeltaPreparedBatch:
    """Build v3 runtime tensors and label-only factual branch indices.

    Call this before moving the host batch when the frozen token table is kept
    on CPU. Counterfactual tensors are intentionally untouched under targets.
    """

    inputs = batch.get("inputs")
    targets = batch.get("targets")
    if not isinstance(inputs, Mapping) or not isinstance(targets, Mapping):
        raise TypeError("rich Delta batch requires inputs and targets mappings")
    if "counterfactual" in inputs:
        raise PermissionError("counterfactual labels appeared in Delta model inputs")
    history = inputs.get("history")
    current = inputs.get("current")
    tree = inputs.get("tree")
    if not all(isinstance(value, Mapping) for value in (history, current, tree)):
        raise TypeError("rich Delta history/current/tree inputs must be mappings")
    assert isinstance(history, Mapping) and isinstance(current, Mapping) and isinstance(tree, Mapping)

    route = history["logits"]
    if route.ndim != 4 or route.shape[2:] != (config.layers, config.experts):
        raise ValueError("rich history logits must be [B,T,L,E]")
    route = route.permute(0, 2, 1, 3).float()
    router_input = current["normalized_target_router_input_a"]
    if router_input.shape != (route.shape[0], config.layers, input_basis.shape[1]):
        raise ValueError("current normalized router input geometry is invalid")
    with torch.autocast(device_type=router_input.device.type, enabled=False):
        control = torch.einsum(
            "bld,ldr->blr", router_input.float(), input_basis.float()
        ) * rank_mask[None].float()

    exact_ids = inputs["exact_next_token_id"].long()
    exact_embedding = frozen_token_embeddings(exact_ids, token_embedding)
    meta = tree["meta"]
    if meta.ndim != 3 or meta.shape[-1] <= 7:
        raise ValueError("adaptive tree meta lacks conditioning token IDs")
    tree_tokens = frozen_token_embeddings(meta[..., 7].long(), token_embedding)
    vocab_ids = tree["vocab_top64_ids"].long()
    vocab_logp = tree["vocab_top64_log_probabilities"].float()
    if vocab_ids.shape != vocab_logp.shape or vocab_ids.shape[-1] != 64:
        raise ValueError("adaptive tree vocabulary evidence must be paired top-64")
    vocab_tokens = frozen_token_embeddings(vocab_ids, token_embedding)
    vocab_embedding = (
        vocab_logp.clamp(max=0.0).exp()[..., None] * vocab_tokens.float()
    ).sum(-2)
    states = tree["states"]
    if states.ndim != 4 or states.shape[2] < 4:
        raise ValueError("adaptive tree states must be [B,N,4,D]")
    position = inputs["within_request"].float().reshape(-1)
    if not torch.equal(position, position.round()) or bool((position < 0).any()):
        raise ValueError("within-request source positions must be non-negative integers")

    factual = factual_branch_indices(
        exact_prefix_hashes=tree["exact_prefix_hashes"],
        node_depths=tree["depth"],
        node_available=tree["mask"],
        future_prefix_hashes=targets["future_prefix_hashes"],
    )
    model_inputs = {
        "anchor_scores": anchor_scores[:, : config.horizons],
        "route_history_logits": route,
        "target_control": control,
        "exact_token_embedding": exact_embedding,
        "final_hidden": inputs["final_hidden"],
        "tree_hidden": states[:, :, 3],
        "tree_fused": states[:, :, 0],
        "tree_router_input": states[:, :, 2],
        "tree_router_logits": tree["router_logits"],
        "tree_token_embeddings": tree_tokens,
        "tree_vocab_embedding": vocab_embedding,
        "tree_vocab_statistics": tree["vocab_statistics"],
        "tree_metadata": _formal_tree_metadata(tree, config),
        "node_parent_ids": tree["parent"],
        "node_available": tree["mask"],
        "node_path_log_probabilities": tree["path_log_probabilities"],
        "node_horizon_mask": tree["horizon_mask"],
        "source_positions": position.long(),
    }
    return DeltaPreparedBatch(model_inputs=model_inputs, factual_branch_index=factual)


__all__ = ["DeltaPreparedBatch", "factual_branch_indices", "prepare_delta_batch"]
