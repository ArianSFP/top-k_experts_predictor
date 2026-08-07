"""Tensor contracts shared by HARP-RTT capture, model, and training code.

All public tensors are batch-major.  History is ordered newest-first along
``T``.  Primary predictions cover H1--H4; frozen HARP compatibility scores
retain H1--H8.  Label-only fields are deliberately omitted by
:meth:`HARPRTTBatch.model_inputs` and :meth:`TreeBatch.model_inputs`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import torch
from torch import Tensor

from .exact_k import stable_topk


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _shape(tensor: Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected}")


def _integer(tensor: Tensor, name: str) -> None:
    if tensor.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype")


def _boolean(tensor: Tensor, name: str) -> None:
    if tensor.dtype != torch.bool:
        raise TypeError(f"{name} must be boolean")


def _finite(tensor: Tensor, name: str) -> None:
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite and floating-point")


@dataclass(frozen=True)
class HARPRTTDimensions:
    """Default production tensor geometry, overridable by focused tests."""

    layers: int = 40
    experts: int = 256
    primary_horizons: int = 4
    legacy_horizons: int = 8
    selected_experts: int = 8
    history_tokens: int = 8
    tree_nodes: int = 32
    candidates: int = 64
    trajectory_rounds: int = 2

    def validate(self) -> None:
        for field in fields(self):
            if int(getattr(self, field.name)) <= 0:
                raise ValueError(f"{field.name} must be positive")
        if self.selected_experts > self.experts:
            raise ValueError("selected_experts cannot exceed experts")
        if self.candidates < self.selected_experts:
            raise ValueError("candidates cannot be smaller than selected_experts")
        if self.primary_horizons > self.legacy_horizons:
            raise ValueError("primary_horizons cannot exceed legacy_horizons")


@dataclass
class TreeBatch:
    """Ragged adaptive-tree data padded to a fixed node axis ``[B,N]``."""

    node_ids: Tensor
    parent_ids: Tensor
    path_ids: Tensor
    branch_ids: Tensor
    depths: Tensor
    target_positions: Tensor
    conditioning_token_ids: Tensor
    token_ids: Tensor
    path_log_probabilities: Tensor
    local_probabilities: Tensor
    source_ready: Tensor
    valid: Tensor
    token_embeddings: Tensor
    hidden_states: Tensor
    fused_states: Tensor
    router_inputs: Tensor
    router_logits: Tensor
    vocabulary_features: Tensor | None = None
    conditioning_classes: Tensor | None = None
    exact_prefix_hashes: Tensor | None = None
    # Labels are capture/training metadata and are never returned by model_inputs.
    acceptance_labels: Tensor | None = None
    acceptance_label_valid: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.node_ids.shape[0])

    @property
    def nodes(self) -> int:
        return int(self.node_ids.shape[1])

    def validate(self, *, maximum_nodes: int | None = None) -> None:
        if self.node_ids.ndim != 2:
            raise ValueError("node_ids must be [B,N]")
        shape = tuple(self.node_ids.shape)
        if maximum_nodes is not None and self.nodes > maximum_nodes:
            raise ValueError(f"tree has {self.nodes} nodes, maximum is {maximum_nodes}")
        integer_fields = {
            "node_ids": self.node_ids,
            "parent_ids": self.parent_ids,
            "path_ids": self.path_ids,
            "branch_ids": self.branch_ids,
            "depths": self.depths,
            "target_positions": self.target_positions,
            "conditioning_token_ids": self.conditioning_token_ids,
            "token_ids": self.token_ids,
        }
        for name, tensor in integer_fields.items():
            _shape(tensor, shape, name)
            _integer(tensor, name)
        _shape(self.valid, shape, "valid")
        _boolean(self.valid, "valid")
        for name, tensor in {
            "path_log_probabilities": self.path_log_probabilities,
            "local_probabilities": self.local_probabilities,
            "source_ready": self.source_ready,
        }.items():
            _shape(tensor, shape, name)
            _finite(tensor, name)
        for name, tensor in {
            "token_embeddings": self.token_embeddings,
            "hidden_states": self.hidden_states,
            "fused_states": self.fused_states,
            "router_inputs": self.router_inputs,
            "router_logits": self.router_logits,
        }.items():
            if tensor.ndim != 3 or tensor.shape[:2] != self.node_ids.shape:
                raise ValueError(f"{name} must be [B,N,D]")
            _finite(tensor, name)
        for name, tensor in {
            "vocabulary_features": self.vocabulary_features,
            "exact_prefix_hashes": self.exact_prefix_hashes,
        }.items():
            if tensor is not None and (tensor.ndim != 3 or tensor.shape[:2] != self.node_ids.shape):
                raise ValueError(f"{name} must be [B,N,D] when present")
        if self.conditioning_classes is not None:
            _shape(self.conditioning_classes, shape, "conditioning_classes")
            _integer(self.conditioning_classes, "conditioning_classes")
        if (self.acceptance_labels is None) != (self.acceptance_label_valid is None):
            raise ValueError("acceptance labels and their validity mask must appear together")
        if self.acceptance_labels is not None:
            _shape(self.acceptance_labels, shape, "acceptance_labels")
            _shape(self.acceptance_label_valid, shape, "acceptance_label_valid")
            _boolean(self.acceptance_labels, "acceptance_labels")
            _boolean(self.acceptance_label_valid, "acceptance_label_valid")
            if (self.acceptance_label_valid & ~self.valid).any():
                raise ValueError("acceptance labels cannot be valid on padded nodes")

        active_local = self.local_probabilities[self.valid]
        if ((active_local < 0) | (active_local > 1)).any():
            raise ValueError("valid-node local probabilities must lie in [0,1]")
        if (self.path_log_probabilities[self.valid] > 1e-6).any():
            raise ValueError("valid-node path log probabilities cannot exceed zero")
        if (self.depths[self.valid] < 0).any():
            raise ValueError("valid-node depths must be non-negative")

        # Nodes are stored parent-before-child.  This makes causal tree masks
        # deterministic and catches cross-request or dangling-parent joins.
        for batch in range(self.batch_size):
            valid_indices = torch.nonzero(self.valid[batch], as_tuple=False).flatten()
            ids = self.node_ids[batch, valid_indices].to(torch.int64)
            if ids.numel() != torch.unique(ids).numel():
                raise ValueError(f"tree {batch} contains duplicate node IDs")
            seen: dict[int, int] = {}
            for index in valid_indices.tolist():
                node_id = int(self.node_ids[batch, index])
                parent_id = int(self.parent_ids[batch, index])
                depth = int(self.depths[batch, index])
                if parent_id == -1:
                    if depth != 0:
                        raise ValueError("tree roots must have depth zero")
                else:
                    if parent_id not in seen:
                        raise ValueError("tree parents must precede their children")
                    if depth != seen[parent_id] + 1:
                        raise ValueError("child depth must equal parent depth plus one")
                seen[node_id] = depth

    def model_inputs(self) -> dict[str, Tensor]:
        """Return causal model inputs, excluding acceptance labels by design."""

        result = {
            "node_ids": self.node_ids,
            "parent_ids": self.parent_ids,
            "path_ids": self.path_ids,
            "branch_ids": self.branch_ids,
            "depths": self.depths,
            "target_positions": self.target_positions,
            "conditioning_token_ids": self.conditioning_token_ids,
            "token_ids": self.token_ids,
            "path_log_probabilities": self.path_log_probabilities,
            "local_probabilities": self.local_probabilities,
            "source_ready": self.source_ready,
            "valid": self.valid,
            "token_embeddings": self.token_embeddings,
            "hidden_states": self.hidden_states,
            "fused_states": self.fused_states,
            "router_inputs": self.router_inputs,
            "router_logits": self.router_logits,
        }
        for name in (
            "vocabulary_features",
            "conditioning_classes",
            "exact_prefix_hashes",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass
class CandidateBatch:
    """C64 object IDs/features plus the authoritative dense-256 score base."""

    dense_base_scores: Tensor
    candidate_ids: Tensor
    candidate_mask: Tensor
    candidate_features: Tensor | None = None

    def validate(self, dimensions: HARPRTTDimensions = HARPRTTDimensions()) -> None:
        dimensions.validate()
        if self.dense_base_scores.ndim != 4:
            raise ValueError("dense_base_scores must be [B,H,L,E]")
        batch = int(self.dense_base_scores.shape[0])
        dense_shape = (
            batch,
            dimensions.primary_horizons,
            dimensions.layers,
            dimensions.experts,
        )
        _shape(self.dense_base_scores, dense_shape, "dense_base_scores")
        candidate_shape = dense_shape[:-1] + (dimensions.candidates,)
        _shape(self.candidate_ids, candidate_shape, "candidate_ids")
        _shape(self.candidate_mask, candidate_shape, "candidate_mask")
        _integer(self.candidate_ids, "candidate_ids")
        _boolean(self.candidate_mask, "candidate_mask")
        _finite(self.dense_base_scores, "dense_base_scores")
        if self.candidate_features is not None:
            if (
                self.candidate_features.ndim != 5
                or self.candidate_features.shape[:4] != self.candidate_ids.shape
            ):
                raise ValueError("candidate_features must be [B,H,L,C,F]")
            _finite(self.candidate_features, "candidate_features")
        active_ids = self.candidate_ids[self.candidate_mask].to(torch.int64)
        if ((active_ids < 0) | (active_ids >= dimensions.experts)).any():
            raise ValueError("active candidate IDs lie outside the expert range")
        flat_ids = self.candidate_ids.reshape(-1, dimensions.candidates)
        flat_mask = self.candidate_mask.reshape(-1, dimensions.candidates)
        for ids, mask in zip(flat_ids, flat_mask, strict=True):
            selected = ids[mask]
            if selected.numel() != torch.unique(selected).numel():
                raise ValueError("active candidates must be unique within each pool")

    def gather_base_scores(self) -> Tensor:
        """Gather candidate scores without treating C64 as the universe."""

        safe_ids = torch.where(
            self.candidate_mask, self.candidate_ids, torch.zeros_like(self.candidate_ids)
        ).long()
        gathered = self.dense_base_scores.gather(-1, safe_ids)
        return gathered.masked_fill(~self.candidate_mask, 0.0)

    def apply_residuals(self, candidate_residuals: Tensor) -> Tensor:
        """Scatter C64 residuals while leaving every outside-pool score intact."""

        _shape(candidate_residuals, tuple(self.candidate_ids.shape), "candidate_residuals")
        safe_ids = torch.where(
            self.candidate_mask, self.candidate_ids, torch.zeros_like(self.candidate_ids)
        ).long()
        updates = candidate_residuals * self.candidate_mask.to(candidate_residuals.dtype)
        return self.dense_base_scores.scatter_add(-1, safe_ids, updates)


@dataclass
class HARPRTTBatch:
    """Causal token-end batch, with future tensors explicitly label-only."""

    route_history_logits: Tensor
    route_history_selected_ids: Tensor
    route_history_selected_weights: Tensor
    route_history_features: Tensor
    route_history_mask: Tensor
    current_router_coordinates: Tensor
    current_content_sketch: Tensor
    post_attention_states: Tensor
    post_moe_states: Tensor
    routed_residuals: Tensor
    shared_residuals: Tensor
    exact_next_token_ids: Tensor
    exact_next_token_embeddings: Tensor
    final_hidden_states: Tensor
    tree: TreeBatch
    harp_scores: Tensor
    candidates: CandidateBatch | None = None
    # Labels below are unavailable to model_inputs.
    target_topk: Tensor | None = None
    target_router_logits: Tensor | None = None
    future_valid: Tensor | None = None

    def validate(self, dimensions: HARPRTTDimensions = HARPRTTDimensions()) -> None:
        dimensions.validate()
        if self.route_history_logits.ndim != 4:
            raise ValueError("route_history_logits must be [B,T,L,E]")
        batch = int(self.route_history_logits.shape[0])
        history = (
            batch,
            dimensions.history_tokens,
            dimensions.layers,
        )
        _shape(
            self.route_history_logits,
            history + (dimensions.experts,),
            "route_history_logits",
        )
        _shape(
            self.route_history_selected_ids,
            history + (dimensions.selected_experts,),
            "route_history_selected_ids",
        )
        _shape(
            self.route_history_selected_weights,
            history + (dimensions.selected_experts,),
            "route_history_selected_weights",
        )
        _shape(self.route_history_mask, history, "route_history_mask")
        _integer(self.route_history_selected_ids, "route_history_selected_ids")
        _boolean(self.route_history_mask, "route_history_mask")
        if self.route_history_features.ndim != 4 or self.route_history_features.shape[:3] != history:
            raise ValueError("route_history_features must be [B,T,L,F]")
        for name, tensor in {
            "route_history_logits": self.route_history_logits,
            "route_history_selected_weights": self.route_history_selected_weights,
            "route_history_features": self.route_history_features,
        }.items():
            _finite(tensor, name)

        layer_prefix = (batch, dimensions.layers)
        for name in (
            "current_router_coordinates",
            "current_content_sketch",
            "post_attention_states",
            "post_moe_states",
            "routed_residuals",
            "shared_residuals",
        ):
            tensor = getattr(self, name)
            if tensor.ndim != 3 or tensor.shape[:2] != layer_prefix:
                raise ValueError(f"{name} must be [B,L,D]")
            _finite(tensor, name)
        _shape(self.exact_next_token_ids, (batch,), "exact_next_token_ids")
        _integer(self.exact_next_token_ids, "exact_next_token_ids")
        for name in ("exact_next_token_embeddings", "final_hidden_states"):
            tensor = getattr(self, name)
            if tensor.ndim != 2 or tensor.shape[0] != batch:
                raise ValueError(f"{name} must be [B,D]")
            _finite(tensor, name)
        _shape(
            self.harp_scores,
            (
                batch,
                dimensions.legacy_horizons,
                dimensions.layers,
                dimensions.experts,
            ),
            "harp_scores",
        )
        _finite(self.harp_scores, "harp_scores")
        if self.tree.batch_size != batch:
            raise ValueError("tree batch size disagrees with route history")
        self.tree.validate(maximum_nodes=dimensions.tree_nodes)
        if self.candidates is not None:
            self.candidates.validate(dimensions)
            if self.candidates.dense_base_scores.shape[0] != batch:
                raise ValueError("candidate batch size disagrees with route history")

        labels = (self.target_topk, self.target_router_logits, self.future_valid)
        if any(value is not None for value in labels) and not all(
            value is not None for value in labels
        ):
            raise ValueError("target_topk, target_router_logits, and future_valid appear together")
        if self.target_topk is not None:
            primary = (
                batch,
                dimensions.primary_horizons,
                dimensions.layers,
            )
            _shape(
                self.target_topk,
                primary + (dimensions.selected_experts,),
                "target_topk",
            )
            _shape(
                self.target_router_logits,
                primary + (dimensions.experts,),
                "target_router_logits",
            )
            _shape(
                self.future_valid,
                (batch, dimensions.primary_horizons),
                "future_valid",
            )
            _integer(self.target_topk, "target_topk")
            _finite(self.target_router_logits, "target_router_logits")
            _boolean(self.future_valid, "future_valid")

    def model_inputs(self) -> dict[str, object]:
        """Return causal fields only; no realized future or acceptance labels."""

        result: dict[str, object] = {
            "route_history_logits": self.route_history_logits,
            "route_history_selected_ids": self.route_history_selected_ids,
            "route_history_selected_weights": self.route_history_selected_weights,
            "route_history_features": self.route_history_features,
            "route_history_mask": self.route_history_mask,
            "current_router_coordinates": self.current_router_coordinates,
            "current_content_sketch": self.current_content_sketch,
            "post_attention_states": self.post_attention_states,
            "post_moe_states": self.post_moe_states,
            "routed_residuals": self.routed_residuals,
            "shared_residuals": self.shared_residuals,
            "exact_next_token_ids": self.exact_next_token_ids,
            "exact_next_token_embeddings": self.exact_next_token_embeddings,
            "final_hidden_states": self.final_hidden_states,
            "tree": self.tree.model_inputs(),
            "harp_scores": self.harp_scores,
        }
        if self.candidates is not None:
            result["candidates"] = self.candidates
        return result


@dataclass
class CandidateDiagnostics:
    candidate_ids: Tensor
    candidate_mask: Tensor
    base_scores: Tensor
    residual_scores: Tensor
    final_candidate_scores: Tensor

    def validate(self, expected_shape: tuple[int, ...]) -> None:
        for name in (
            "candidate_ids",
            "candidate_mask",
            "base_scores",
            "residual_scores",
            "final_candidate_scores",
        ):
            _shape(getattr(self, name), expected_shape, name)
        _integer(self.candidate_ids, "candidate_ids")
        _boolean(self.candidate_mask, "candidate_mask")
        for name in ("base_scores", "residual_scores", "final_candidate_scores"):
            _finite(getattr(self, name), name)


@dataclass
class HARPRTTOutput:
    """Dense H1--H4 exact-8 output with branch/trajectory diagnostics."""

    dense_scores: Tensor
    exact_k_marginals: Tensor
    topk_ids: Tensor
    query_coordinates: Tensor
    branch_posteriors: Tensor
    branch_marginals: Tensor
    trajectory_scores: Tensor
    component_scores: Mapping[str, Tensor]
    legacy_h5_h8_scores: Tensor
    candidate_diagnostics: CandidateDiagnostics | None = None

    def validate(
        self,
        dimensions: HARPRTTDimensions = HARPRTTDimensions(),
        *,
        cardinality_tolerance: float = 2e-5,
    ) -> None:
        dimensions.validate()
        if self.dense_scores.ndim != 4:
            raise ValueError("dense_scores must be [B,H,L,E]")
        batch = int(self.dense_scores.shape[0])
        primary = (
            batch,
            dimensions.primary_horizons,
            dimensions.layers,
            dimensions.experts,
        )
        _shape(self.dense_scores, primary, "dense_scores")
        _shape(self.exact_k_marginals, primary, "exact_k_marginals")
        _shape(
            self.topk_ids,
            primary[:-1] + (dimensions.selected_experts,),
            "topk_ids",
        )
        _integer(self.topk_ids, "topk_ids")
        _finite(self.dense_scores, "dense_scores")
        _finite(self.exact_k_marginals, "exact_k_marginals")
        if ((self.exact_k_marginals < 0) | (self.exact_k_marginals > 1 + 1e-6)).any():
            raise ValueError("exact_k_marginals must lie in [0,1]")
        mass_error = (
            self.exact_k_marginals.sum(-1) - float(dimensions.selected_experts)
        ).abs().max()
        if float(mass_error) > cardinality_tolerance:
            raise ValueError(
                f"exact_k_marginals violate cardinality by {float(mass_error):.3e}"
            )
        expected_ids = stable_topk(
            self.exact_k_marginals, dimensions.selected_experts
        )
        if not torch.equal(self.topk_ids.to(torch.int64), expected_ids):
            raise ValueError("topk_ids must be the stable ranking of exact_k_marginals")
        if self.query_coordinates.ndim != 4 or self.query_coordinates.shape[:3] != primary[:3]:
            raise ValueError("query_coordinates must be [B,H,L,R]")
        _finite(self.query_coordinates, "query_coordinates")
        if self.branch_posteriors.ndim != 3 or self.branch_posteriors.shape[:2] != primary[:2]:
            raise ValueError("branch_posteriors must be [B,H,N]")
        branches = int(self.branch_posteriors.shape[-1])
        _shape(
            self.branch_marginals,
            primary[:3] + (branches, dimensions.experts),
            "branch_marginals",
        )
        _finite(self.branch_posteriors, "branch_posteriors")
        _finite(self.branch_marginals, "branch_marginals")
        _shape(
            self.trajectory_scores,
            (batch, dimensions.trajectory_rounds) + primary[1:],
            "trajectory_scores",
        )
        _finite(self.trajectory_scores, "trajectory_scores")
        for name, tensor in self.component_scores.items():
            _shape(tensor, primary, f"component_scores[{name!r}]")
            _finite(tensor, f"component_scores[{name!r}]")
        _shape(
            self.legacy_h5_h8_scores,
            (
                batch,
                dimensions.legacy_horizons - dimensions.primary_horizons,
                dimensions.layers,
                dimensions.experts,
            ),
            "legacy_h5_h8_scores",
        )
        _finite(self.legacy_h5_h8_scores, "legacy_h5_h8_scores")
        if self.candidate_diagnostics is not None:
            self.candidate_diagnostics.validate(
                primary[:-1] + (dimensions.candidates,)
            )


__all__ = [
    "CandidateBatch",
    "CandidateDiagnostics",
    "HARPRTTBatch",
    "HARPRTTDimensions",
    "HARPRTTOutput",
    "TreeBatch",
]
