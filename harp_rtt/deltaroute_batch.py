"""Leakage-safe rich-batch adaptation for HARP-DeltaRoute v4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from torch import Tensor

from .delta import HARPDeltaConfig
from .delta_batch import prepare_delta_batch


@dataclass(frozen=True)
class DeltaRoutePreparedBatch:
    parent_inputs: Mapping[str, Tensor]
    trajectory_inputs: Mapping[str, Tensor]
    factual_branch_index: Tensor
    history_targets: Mapping[str, Tensor]


def prepare_deltaroute_batch(
    batch: Mapping[str, Any],
    *,
    anchor_scores: Tensor,
    token_embedding: Tensor,
    input_basis: Tensor,
    rank_mask: Tensor,
    config: HARPDeltaConfig,
) -> DeltaRoutePreparedBatch:
    """Build causal v4 channels while retaining route histories as labels."""

    inputs = batch.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("DeltaRoute batch requires an inputs mapping")
    if "counterfactual" in inputs or "targets" in inputs:
        raise PermissionError("target-only tensors appeared in DeltaRoute inputs")
    parent = prepare_delta_batch(
        batch,
        anchor_scores=anchor_scores,
        token_embedding=token_embedding,
        input_basis=input_basis,
        rank_mask=rank_mask,
        config=config,
    )
    tree = inputs.get("tree")
    history = inputs.get("history")
    if not isinstance(tree, Mapping) or not isinstance(history, Mapping):
        raise TypeError("DeltaRoute requires rich tree and history mappings")
    states = tree["states"]
    if states.ndim != 4 or states.shape[2] != 4:
        raise ValueError("DeltaRoute tree state stack must be [B,N,4,D]")
    trajectory = {
        "fused": states[:, :, 0],
        "post_ffn": states[:, :, 1],
        "router_input": states[:, :, 2],
        "vocabulary": parent.model_inputs["tree_vocab_embedding"],
        "token": parent.model_inputs["tree_token_embeddings"],
        "router_logits": tree["router_logits"],
        "metadata": parent.model_inputs["tree_metadata"],
        "parents": tree["parent"],
        "available": tree["mask"],
    }
    # These are teacher-only histories and are never exposed in trajectory.
    history_targets = {
        "logits": history["logits"],
        "selected_ids": history["selected_ids"],
        "execution_weights": history["execution_weights"],
        "available": history["available"],
    }
    return DeltaRoutePreparedBatch(
        parent.model_inputs, trajectory, parent.factual_branch_index,
        history_targets,
    )


__all__ = ["DeltaRoutePreparedBatch", "prepare_deltaroute_batch"]
