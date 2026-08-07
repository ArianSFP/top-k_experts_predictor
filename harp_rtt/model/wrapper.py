"""Function-preserving HARP-RTT wrapper around a supplied HARP anchor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from harp_rtt.exact_k import stable_topk
from harp_rtt.geometry import CenteredRouterGeometry
from harp_rtt.schema import CandidateDiagnostics, HARPRTTOutput

from .candidates import CandidateUnion
from .config import HARPRTTConfig
from .decoder import EndpointDecoder
from .heads import (
    BranchMixtureResidual,
    ExactBranchMixture,
    HybridRouterScoreHead,
    exact_projected_marginals,
)
from .reranker import AxialCandidateReranker
from .route import RouteGridEncoder
from .target import TargetStateEncoder
from .trajectory import TwoRoundTrajectoryRefiner
from .tree import AdaptiveTreeEncoder


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"none of the required fields are present: {names}")


def _optional_mapping_value(mapping: Mapping[str, Any], *names: str) -> Any | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _compact_branch_hashes(
    branch_hashes: Tensor,
    available: Tensor,
    *,
    maximum_branches: int,
) -> Tensor:
    """Map opaque branch hashes to collision-free sample-local IDs.

    Capture stores a stable signed-63-bit branch identity.  An arithmetic
    remainder can alias two real branches before the embedding lookup.  This
    tensor-only first-occurrence compaction reserves zero for padding and
    produces IDs in ``1..maximum_branches`` without a CPU/GPU synchronization.
    """

    if branch_hashes.ndim != 2 or available.shape != branch_hashes.shape:
        raise ValueError("branch hashes and availability must both be [B,N]")
    batch, nodes = branch_hashes.shape
    positions = torch.arange(nodes, device=branch_hashes.device)
    same = branch_hashes[:, :, None] == branch_hashes[:, None, :]
    # Only valid nodes define branch identities.  This also keeps arbitrary
    # bytes in a non-contiguous padded slot from consuming an embedding ID.
    same &= available.bool()[:, None, :]
    sentinel = torch.full(
        (batch, nodes, nodes), nodes, dtype=torch.long, device=branch_hashes.device
    )
    first = torch.where(
        same,
        positions[None, None, :].expand(batch, nodes, -1),
        sentinel,
    ).amin(dim=-1)
    first = torch.where(available.bool(), first, torch.zeros_like(first))
    is_first = (first == positions[None]) & available.bool()
    dense_at_position = is_first.long().cumsum(dim=-1)
    dense = dense_at_position.gather(1, first)
    dense = torch.where(available.bool(), dense, torch.zeros_like(dense))
    if (dense > int(maximum_branches)).any():
        raise ValueError(
            f"rich tree contains more than {maximum_branches} distinct branches"
        )
    return dense


class HARPRTTTeacher(nn.Module):
    """H1--H4 residual extension that leaves H5--H8 on frozen HARP.

    ``forward`` accepts either the explicit tensor contract or ``batch=`` with
    the rich dataset's ``{"metadata", "inputs", "targets"}`` structure.  Future
    targets and acceptance labels are never read by the adapter.
    """

    def __init__(
        self,
        anchor: nn.Module,
        config: HARPRTTConfig,
        geometry: CenteredRouterGeometry,
        *,
        token_embedding: nn.Module | Tensor | None = None,
        candidate_temperatures: Tensor | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        geometry.validate()
        if (geometry.layers, geometry.experts, geometry.maximum_rank) != (
            config.layers,
            config.experts,
            config.router_rank,
        ):
            raise ValueError("centered router geometry disagrees with model config")
        self.config = config
        self.anchor = anchor
        self._anchor_frozen = bool(config.freeze_anchor)
        if self._anchor_frozen:
            for parameter in self.anchor.parameters():
                parameter.requires_grad_(False)
            self.anchor.eval()

        self.register_buffer(
            "input_basis", geometry.input_basis.detach().float().clone()
        )
        self.register_buffer(
            "rank_mask", geometry.rank_mask.detach().bool().clone()
        )
        if isinstance(token_embedding, Tensor):
            if token_embedding.ndim != 2 or token_embedding.shape[1] != config.exact_token_width:
                raise ValueError("token embedding table has the wrong width")
            self.token_embedding: nn.Module | None = nn.Embedding.from_pretrained(
                token_embedding.detach(), freeze=True
            )
        else:
            self.token_embedding = token_embedding
            if self.token_embedding is not None:
                for parameter in self.token_embedding.parameters():
                    parameter.requires_grad_(False)

        keys = geometry.expert_keys
        self.route_encoder = RouteGridEncoder(config, keys)
        self.target_encoder = TargetStateEncoder(config)
        self.tree_encoder = AdaptiveTreeEncoder(config)
        self.decoder = EndpointDecoder(config)
        self.score_head = HybridRouterScoreHead(
            config,
            keys,
            geometry.rank_mask,
            geometry.centered_bias,
        )
        self.branch_mixture = ExactBranchMixture(config)
        self.branch_residual = BranchMixtureResidual(config)
        self.trajectory = TwoRoundTrajectoryRefiner(config, keys)
        self.candidate_union = CandidateUnion(config, candidate_temperatures)
        self.reranker = AxialCandidateReranker(
            config, keys, geometry.row_norms
        )

    def train(self, mode: bool = True) -> "HARPRTTTeacher":
        super().train(mode)
        if self._anchor_frozen:
            self.anchor.eval()
        if self.token_embedding is not None:
            self.token_embedding.eval()
        return self

    def _anchor_outputs(
        self,
        anchor_inputs: Mapping[str, Tensor] | None,
        anchor_outputs: Mapping[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        if anchor_outputs is not None:
            result = dict(anchor_outputs)
        else:
            if anchor_inputs is None:
                raise ValueError("anchor_inputs or precomputed anchor_outputs are required")
            if self._anchor_frozen:
                with torch.no_grad():
                    result = dict(self.anchor(**dict(anchor_inputs)))
            else:
                result = dict(self.anchor(**dict(anchor_inputs)))
        if "future_router_scores" not in result:
            raise KeyError("HARP anchor did not return future_router_scores")
        return result

    def _lookup_token(self, token_ids: Tensor) -> Tensor:
        if self.token_embedding is None:
            raise ValueError(
                "rich batch contains only exact_next_token_id; provide a frozen token_embedding"
            )
        values = self.token_embedding(token_ids.long())
        if values.shape != token_ids.shape + (self.config.exact_token_width,):
            raise ValueError("frozen token embedding returned an unexpected shape")
        return values

    def _adapt_history(self, value: Tensor, *, trailing: int | None = None) -> Tensor:
        config = self.config
        # Dataset order is [B,T,L,...]; model order is [B,L,T,...].
        if value.shape[1:3] == (config.route_history, config.layers):
            permutation = [0, 2, 1] + list(range(3, value.ndim))
            return value.permute(*permutation)
        if value.shape[1:3] == (config.layers, config.route_history):
            return value
        raise ValueError("history tensor does not expose [T,L] or [L,T] axes")

    def _adapt_rich_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        inputs = batch.get("inputs", batch)
        if not isinstance(inputs, Mapping):
            raise TypeError("rich batch inputs must be a mapping")
        if "route_history_logits" in inputs:
            return self._adapt_schema_inputs(inputs)
        history = inputs.get("history", inputs)
        if not isinstance(history, Mapping):
            raise TypeError("rich history inputs must be a mapping")
        current = _mapping_value(inputs, "current", "current_target")
        tree = _mapping_value(inputs, "tree", "mtp_tree")
        if not isinstance(current, Mapping) or not isinstance(tree, Mapping):
            raise TypeError("current and tree rich inputs must be mappings")

        history_logits = self._adapt_history(
            _mapping_value(history, "logits", "history_logits", "route_history")
        )
        history_available = self._adapt_history(
            _mapping_value(history, "available", "history_available", "route_available")
        )
        selected_ids = _optional_mapping_value(history, "selected_ids", "history_selected_ids")
        selected_weights = _optional_mapping_value(
            history, "execution_weights", "selected_weights", "history_selected_weights"
        )
        history_summary = _optional_mapping_value(history, "summary", "history_summary")
        if selected_ids is not None:
            selected_ids = self._adapt_history(selected_ids)
        if selected_weights is not None:
            selected_weights = self._adapt_history(selected_weights)
        if history_summary is not None:
            history_summary = self._adapt_history(history_summary)

        router_input = _mapping_value(
            current,
            "router_input",
            "normalized_target_router_input_a",
            "normalized_router_input",
        )
        control = _optional_mapping_value(
            current, "control", "router_control", "router_visible_coordinates"
        )
        if control is None:
            # V is frozen target-router geometry.  The surrounding training
            # forward runs under BF16 autocast, but this projection defines
            # the canonical router coordinates and must retain FP32 decision
            # equivalence with the extracted target router.
            with torch.autocast(device_type=router_input.device.type, enabled=False):
                control = torch.einsum(
                    "bld,ldr->blr",
                    router_input.float(),
                    self.input_basis.float(),
                )
                control = control * self.rank_mask[None].to(control.dtype)
        content = _optional_mapping_value(
            current, "content", "router_blind_content", "content_sketch"
        )
        if content is None:
            # Preserve the exact router-blind residual when the index has not
            # yet applied a train-fitted compact sketch.  Passing the raw
            # router input here would duplicate the visible control channel
            # and violate the control/content separation in the formal model.
            with torch.autocast(device_type=control.device.type, enabled=False):
                visible = torch.einsum(
                    "blr,ldr->bld",
                    control.float(),
                    self.input_basis.float(),
                )
                content = router_input.float() - visible
        target_available = _optional_mapping_value(inputs, "current_available")
        if target_available is None:
            target_available = _optional_mapping_value(
                current, "source_available", "current_available"
            )
        if target_available is not None:
            if target_available.ndim == 2:
                target_available = target_available[..., None].expand(-1, -1, 6)
            else:
                roles = inputs.get("current_roles")
                if roles is not None:
                    if not isinstance(roles, (tuple, list)):
                        raise ValueError("current_roles must be a tuple or list")
                    if len(roles) != target_available.shape[-1]:
                        raise ValueError(
                            "current_roles length disagrees with current_available"
                        )
                    role_columns = {name: index for index, name in enumerate(roles)}
                    wanted = (
                        "normalized_target_router_input_a",
                        "normalized_target_router_input_a",
                        "post_attention_residual_u",
                        "post_moe_residual_xplus",
                        "routed_expert_output_delta_r",
                        "shared_expert_output_delta_s",
                    )
                    missing_roles = [name for name in wanted if name not in role_columns]
                    if missing_roles:
                        raise ValueError(
                            "current_roles lacks target source roles: "
                            + ", ".join(sorted(set(missing_roles)))
                        )
                    target_available = torch.stack(
                        [target_available[..., role_columns[name]] for name in wanted],
                        dim=-1,
                    )
                elif target_available.shape[-1] != 6:
                    raise ValueError(
                        "current_available requires current_roles when it has raw-role columns"
                    )

        exact_embedding = _optional_mapping_value(
            inputs, "exact_next_token_embedding", "exact_token_embedding"
        )
        if exact_embedding is None:
            token_ids = _mapping_value(
                inputs, "exact_next_token_id", "committed_next_token_id"
            )
            exact_embedding = self._lookup_token(token_ids)

        states = _mapping_value(tree, "states", "mtp_states")
        if states.ndim != 4 or states.shape[2] < 4:
            raise ValueError("tree states must be [B,N,4,D]")
        meta_value = _optional_mapping_value(tree, "meta", "metadata")
        meta_mapping = meta_value if isinstance(meta_value, Mapping) else {}
        metadata = _optional_mapping_value(tree, "scalars", "node_scalars")
        if metadata is None and isinstance(meta_value, Tensor):
            metadata = meta_value
        if metadata is None:
            raise KeyError("tree scalar metadata are missing")
        depth = _optional_mapping_value(tree, "depth", "depth_ids")
        if depth is None and isinstance(meta_mapping, Mapping):
            depth = _optional_mapping_value(meta_mapping, "depth", "depth_ids")
        parent = _optional_mapping_value(tree, "parent", "parent_ids")
        if parent is None and isinstance(meta_mapping, Mapping):
            parent = _optional_mapping_value(meta_mapping, "parent", "parent_ids")
        branch = _optional_mapping_value(tree, "branch", "branch_ids")
        if branch is None and isinstance(meta_mapping, Mapping):
            branch = _optional_mapping_value(meta_mapping, "branch", "branch_ids")
        if depth is None and isinstance(meta_value, Tensor):
            depth = meta_value[..., 4].long()
        tree_available = _mapping_value(tree, "mask", "available")
        if branch is None and isinstance(meta_value, Tensor):
            branch = _compact_branch_hashes(
                meta_value[..., 11].long(),
                tree_available,
                maximum_branches=self.config.max_tree_branches,
            )
        elif branch is not None:
            invalid_branch = (branch < 0) | (
                branch > self.config.max_tree_branches
            )
            if (invalid_branch & tree_available.bool()).any():
                raise ValueError("valid rich-tree branch ID exceeds configured range")
        if depth is None or parent is None:
            raise KeyError("tree depth and parent IDs are required")
        if branch is None:
            branch = torch.zeros_like(depth)
        # New adaptive indices expose the formal eta-node fields by name. Use
        # those fields when available while retaining the anonymous eight-value
        # legacy scalar vector for already-built greedy-chain indices.
        named_tree_metadata = (
            "path_log_probabilities",
            "local_probabilities",
            "source_ready",
            "target_positions",
        )
        if all(name in tree for name in named_tree_metadata):
            structural_valid = _optional_mapping_value(
                tree, "structural_valid", "structural_validity"
            )
            metadata_tree: dict[str, Any] = {
                "path_log_probabilities": tree["path_log_probabilities"],
                "local_probabilities": tree["local_probabilities"],
                "source_ready": tree["source_ready"],
                "depths": depth,
                "valid": (
                    structural_valid
                    if structural_valid is not None
                    else tree_available
                ),
                "target_positions": tree["target_positions"],
                "branch_ids": branch,
            }
            conditioning = _optional_mapping_value(
                tree, "conditioning_classes", "conditioning_class"
            )
            if conditioning is not None:
                metadata_tree["conditioning_classes"] = conditioning
            formal_metadata = self._schema_tree_metadata(metadata_tree)
            adaptive_contract = _optional_mapping_value(
                tree, "adaptive_contract", "formal_metadata_valid"
            )
            if adaptive_contract is None:
                metadata = formal_metadata
            else:
                contract = adaptive_contract.bool()
                while contract.ndim < formal_metadata.ndim:
                    contract = contract.unsqueeze(-1)
                metadata = torch.where(contract, formal_metadata, metadata)
        if not isinstance(meta_value, Tensor) or meta_value.shape[-1] <= 7:
            raise ValueError(
                "tree meta must expose conditioning token IDs in column 7"
            )
        tree_token_embeddings = self._lookup_token(meta_value[..., 7].long())

        return {
            "route_history": history_logits,
            "route_available": history_available,
            "history_selected_ids": selected_ids,
            "history_selected_weights": selected_weights,
            "history_summary": history_summary,
            "target_control": control,
            "target_content": content,
            "target_post_attention": _mapping_value(
                current, "post_attention", "post_attention_residual_u"
            ),
            "target_post_moe": _mapping_value(
                current, "post_moe", "post_moe_residual_xplus"
            ),
            "target_routed": _mapping_value(
                current, "routed", "routed_expert_output_delta_r"
            ),
            "target_shared": _mapping_value(
                current, "shared", "shared_expert_output_delta_s"
            ),
            "target_available": target_available,
            "exact_token_embedding": exact_embedding,
            "final_hidden": _mapping_value(inputs, "final_hidden"),
            # The vocabulary-head input is the native MTP hidden h^M.  The
            # post-FFN state remains available in capture, but is not confused
            # with either h^M or the separately looked-up token embedding.
            "tree_hidden": states[:, :, 3],
            "tree_fused": states[:, :, 0],
            "tree_router_input": states[:, :, 2],
            "tree_token_embeddings": tree_token_embeddings,
            "tree_router_logits": _mapping_value(
                tree, "router_logits", "mtp_router_logits"
            ),
            "tree_metadata": metadata,
            "tree_depth_ids": depth,
            "tree_parent_ids": parent,
            "tree_branch_ids": branch,
            "tree_available": tree_available,
            "tree_horizon_mask": (
                _optional_mapping_value(tree, "horizon_mask").transpose(1, 2)
                if _optional_mapping_value(tree, "horizon_mask") is not None
                else None
            ),
            "tree_horizon_ids": _optional_mapping_value(tree, "horizon_ids"),
            "persistence_scores": _optional_mapping_value(
                inputs, "persistence_scores"
            ),
            "transition_scores": _optional_mapping_value(inputs, "transition_scores"),
            "extra_candidate_features": _optional_mapping_value(
                inputs, "candidate_features"
            ),
            "anchor_inputs": _optional_mapping_value(
                inputs, "anchor_inputs", "harp_inputs"
            ),
            "anchor_outputs": None,
        }

    def _schema_tree_metadata(self, tree: Mapping[str, Any]) -> Tensor:
        """Build the formal eta-node scalar vector from ``TreeBatch`` inputs."""

        depth = _mapping_value(tree, "depths", "depth")
        valid = _mapping_value(tree, "valid", "mask")
        target = _mapping_value(tree, "target_positions", "horizon_ids")
        branch = _mapping_value(tree, "branch_ids", "branch")
        components = [
            _mapping_value(tree, "path_log_probabilities").float(),
            _mapping_value(tree, "local_probabilities").float(),
            _mapping_value(tree, "source_ready").float(),
            depth.float() / max(1, self.config.max_tree_depth),
            valid.float(),
            target.float() / max(1, self.config.active_horizons),
            branch.float() / max(1, self.config.max_tree_branches),
        ]
        conditioning = _optional_mapping_value(tree, "conditioning_classes")
        if conditioning is None:
            components.append(torch.zeros_like(components[0]))
        else:
            components.append(conditioning.float() / 4.0)
        metadata = torch.stack(components, dim=-1)
        width = self.config.tree_metadata_width
        if metadata.shape[-1] < width:
            metadata = torch.nn.functional.pad(
                metadata, (0, width - metadata.shape[-1])
            )
        return metadata[..., :width]

    @staticmethod
    def _parent_node_ids_to_indices(
        node_ids: Tensor, parent_node_ids: Tensor, valid: Tensor
    ) -> Tensor:
        """Map schema parent object IDs to permutation-aware padded indices."""

        matches = parent_node_ids[..., None] == node_ids[:, None, :]
        found = matches.any(dim=-1)
        indices = matches.to(torch.int64).argmax(dim=-1)
        roots = parent_node_ids < 0
        if ((~roots & valid.bool()) & ~found).any():
            raise ValueError("valid tree node references an absent parent object ID")
        return torch.where(roots | ~valid.bool(), torch.full_like(indices, -1), indices)

    def _adapt_schema_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Adapt the public ``HARPRTTBatch.model_inputs`` mapping."""

        tree = _mapping_value(inputs, "tree")
        if not isinstance(tree, Mapping):
            raise TypeError("schema tree model inputs must be a mapping")
        node_ids = _mapping_value(tree, "node_ids")
        parent_ids = self._parent_node_ids_to_indices(
            node_ids,
            _mapping_value(tree, "parent_ids"),
            _mapping_value(tree, "valid"),
        )
        harp_scores = _mapping_value(inputs, "harp_scores")
        return {
            "route_history": self._adapt_history(
                _mapping_value(inputs, "route_history_logits")
            ),
            "route_available": self._adapt_history(
                _mapping_value(inputs, "route_history_mask")
            ),
            "history_selected_ids": self._adapt_history(
                _mapping_value(inputs, "route_history_selected_ids")
            ),
            "history_selected_weights": self._adapt_history(
                _mapping_value(inputs, "route_history_selected_weights")
            ),
            "history_summary": self._adapt_history(
                _mapping_value(inputs, "route_history_features")
            ),
            "target_control": _mapping_value(inputs, "current_router_coordinates"),
            "target_content": _mapping_value(inputs, "current_content_sketch"),
            "target_post_attention": _mapping_value(inputs, "post_attention_states"),
            "target_post_moe": _mapping_value(inputs, "post_moe_states"),
            "target_routed": _mapping_value(inputs, "routed_residuals"),
            "target_shared": _mapping_value(inputs, "shared_residuals"),
            "target_available": None,
            "exact_token_embedding": _mapping_value(inputs, "exact_next_token_embeddings"),
            "final_hidden": _mapping_value(inputs, "final_hidden_states"),
            "tree_hidden": _mapping_value(tree, "hidden_states"),
            "tree_fused": _mapping_value(tree, "fused_states"),
            "tree_router_input": _mapping_value(tree, "router_inputs"),
            "tree_router_logits": _mapping_value(tree, "router_logits"),
            "tree_token_embeddings": _mapping_value(tree, "token_embeddings"),
            "tree_metadata": self._schema_tree_metadata(tree),
            "tree_depth_ids": _mapping_value(tree, "depths"),
            "tree_parent_ids": parent_ids,
            "tree_branch_ids": _mapping_value(tree, "branch_ids"),
            "tree_available": _mapping_value(tree, "valid"),
            "tree_horizon_mask": None,
            "tree_horizon_ids": _mapping_value(tree, "target_positions"),
            "persistence_scores": None,
            "transition_scores": None,
            "extra_candidate_features": None,
            "anchor_inputs": None,
            "anchor_outputs": {
                "future_router_scores": harp_scores,
                "future_inclusion_probabilities": torch.sigmoid(harp_scores.float()).to(
                    harp_scores.dtype
                ),
            },
        }

    def forward(
        self,
        *,
        batch: Mapping[str, Any] | None = None,
        anchor_inputs: Mapping[str, Tensor] | None = None,
        anchor_outputs: Mapping[str, Tensor] | None = None,
        route_history: Tensor | None = None,
        route_available: Tensor | None = None,
        history_selected_ids: Tensor | None = None,
        history_selected_weights: Tensor | None = None,
        history_summary: Tensor | None = None,
        target_control: Tensor | None = None,
        target_content: Tensor | None = None,
        target_post_attention: Tensor | None = None,
        target_post_moe: Tensor | None = None,
        target_routed: Tensor | None = None,
        target_shared: Tensor | None = None,
        target_available: Tensor | None = None,
        exact_token_embedding: Tensor | None = None,
        final_hidden: Tensor | None = None,
        tree_hidden: Tensor | None = None,
        tree_fused: Tensor | None = None,
        tree_router_input: Tensor | None = None,
        tree_router_logits: Tensor | None = None,
        tree_token_embeddings: Tensor | None = None,
        tree_metadata: Tensor | None = None,
        tree_depth_ids: Tensor | None = None,
        tree_parent_ids: Tensor | None = None,
        tree_branch_ids: Tensor | None = None,
        tree_available: Tensor | None = None,
        tree_horizon_mask: Tensor | None = None,
        tree_horizon_ids: Tensor | None = None,
        persistence_scores: Tensor | None = None,
        transition_scores: Tensor | None = None,
        extra_candidate_features: Tensor | None = None,
    ) -> dict[str, Any]:
        if batch is not None:
            adapted = self._adapt_rich_batch(batch)
            if anchor_inputs is None:
                anchor_inputs = adapted.pop("anchor_inputs")
            else:
                adapted.pop("anchor_inputs")
            adapted_anchor_outputs = adapted.pop("anchor_outputs")
            if anchor_outputs is None:
                anchor_outputs = adapted_anchor_outputs
            explicit = {
                "route_history": route_history,
                "route_available": route_available,
                "history_selected_ids": history_selected_ids,
                "history_selected_weights": history_selected_weights,
                "history_summary": history_summary,
                "target_control": target_control,
                "target_content": target_content,
                "target_post_attention": target_post_attention,
                "target_post_moe": target_post_moe,
                "target_routed": target_routed,
                "target_shared": target_shared,
                "target_available": target_available,
                "exact_token_embedding": exact_token_embedding,
                "final_hidden": final_hidden,
                "tree_hidden": tree_hidden,
                "tree_fused": tree_fused,
                "tree_router_input": tree_router_input,
                "tree_router_logits": tree_router_logits,
                "tree_token_embeddings": tree_token_embeddings,
                "tree_metadata": tree_metadata,
                "tree_depth_ids": tree_depth_ids,
                "tree_parent_ids": tree_parent_ids,
                "tree_branch_ids": tree_branch_ids,
                "tree_available": tree_available,
                "tree_horizon_mask": tree_horizon_mask,
                "tree_horizon_ids": tree_horizon_ids,
                "persistence_scores": persistence_scores,
                "transition_scores": transition_scores,
                "extra_candidate_features": extra_candidate_features,
            }
            for name, value in explicit.items():
                if value is not None:
                    adapted[name] = value
            return self.forward(
                anchor_inputs=anchor_inputs,
                anchor_outputs=anchor_outputs,
                **adapted,
            )

        required = {
            "route_history": route_history,
            "route_available": route_available,
            "target_control": target_control,
            "target_content": target_content,
            "target_post_attention": target_post_attention,
            "target_post_moe": target_post_moe,
            "target_routed": target_routed,
            "target_shared": target_shared,
            "exact_token_embedding": exact_token_embedding,
            "final_hidden": final_hidden,
            "tree_hidden": tree_hidden,
            "tree_fused": tree_fused,
            "tree_router_input": tree_router_input,
            "tree_router_logits": tree_router_logits,
            "tree_token_embeddings": tree_token_embeddings,
            "tree_metadata": tree_metadata,
            "tree_depth_ids": tree_depth_ids,
            "tree_parent_ids": tree_parent_ids,
            "tree_branch_ids": tree_branch_ids,
            "tree_available": tree_available,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"missing HARP-RTT causal inputs: {missing}")
        # Narrow Optional types after the explicit check above.
        assert route_history is not None and route_available is not None
        assert target_control is not None and target_content is not None
        assert target_post_attention is not None and target_post_moe is not None
        assert target_routed is not None and target_shared is not None
        assert exact_token_embedding is not None and final_hidden is not None
        assert tree_hidden is not None and tree_fused is not None
        assert tree_router_input is not None and tree_router_logits is not None
        assert tree_token_embeddings is not None and tree_metadata is not None
        assert tree_depth_ids is not None and tree_parent_ids is not None
        assert tree_branch_ids is not None and tree_available is not None

        anchor = self._anchor_outputs(anchor_inputs, anchor_outputs)
        anchor_scores = anchor["future_router_scores"]
        config = self.config
        expected = (
            route_history.shape[0],
            config.anchor_horizons,
            config.layers,
            config.experts,
        )
        if anchor_scores.shape != expected:
            raise ValueError(
                f"anchor future_router_scores must have shape {expected}, got {tuple(anchor_scores.shape)}"
            )
        active_anchor = anchor_scores[:, : config.active_horizons]
        route = self.route_encoder(
            route_history,
            route_available,
            selected_ids=history_selected_ids,
            selected_weights=history_selected_weights,
            history_summary=history_summary,
        )
        target = self.target_encoder(
            control=target_control,
            content=target_content,
            post_attention=target_post_attention,
            post_moe=target_post_moe,
            routed=target_routed,
            shared=target_shared,
            available=target_available,
        )
        tree = self.tree_encoder(
            hidden=tree_hidden,
            fused=tree_fused,
            router_input=tree_router_input,
            router_logits=tree_router_logits,
            token_embeddings=tree_token_embeddings,
            metadata=tree_metadata,
            depth_ids=tree_depth_ids,
            parent_ids=tree_parent_ids,
            branch_ids=tree_branch_ids,
            available=tree_available,
            horizon_mask=tree_horizon_mask,
            horizon_ids=tree_horizon_ids,
        )
        endpoint = self.decoder(
            route,
            target,
            tree,
            exact_token_embedding=exact_token_embedding,
            final_hidden=final_hidden,
        )
        causal_support = self.reranker.build_causal_route_support(
            route_history,
            history_selected_ids,
            history_selected_weights,
            route_available,
        )
        if transition_scores is None:
            transition = causal_support.transition_scores
        else:
            transition = transition_scores
        if persistence_scores is None:
            current = route_history[:, :, 0]
            current = current - current.mean(dim=-1, keepdim=True)
            persistence = current[:, None].expand(
                -1, config.active_horizons, -1, -1
            )
        else:
            persistence = persistence_scores
        for name, value in (("transition", transition), ("persistence", persistence)):
            if value.shape != active_anchor.shape:
                raise ValueError(f"{name} scores must match active anchor scores")
        hybrid = self.score_head(
            endpoint,
            tree,
            active_anchor,
            transition_scores=transition,
        )
        mixture = self.branch_mixture(
            hybrid.branch_scores,
            hybrid.branch_logits,
            tree.horizon_mask,
        )
        direct_scores = self.branch_residual(
            active_anchor, mixture.mixture_logits
        )
        trajectory = self.trajectory(
            direct_scores,
            mixture.mixture_marginals,
            endpoint=endpoint,
            route=route,
            target=target,
        )
        # Do not re-quantize the FP32 Kq branch scores while marginalizing the
        # branch posterior: this dense geometry evidence enters candidate
        # generation and therefore can determine the final top-8 set.
        with torch.autocast(
            device_type=hybrid.geometry_scores.device.type, enabled=False
        ):
            geometry = torch.einsum(
                "bhn,bhlne->bhle",
                mixture.branch_weights.float(),
                hybrid.geometry_scores.float(),
            )
        free = torch.einsum(
            "bhn,bhlne->bhle",
            mixture.branch_weights,
            hybrid.free_scores,
        )
        candidate_union = self.candidate_union(
            [
                active_anchor,
                geometry,
                mixture.mixture_logits,
                trajectory.scores,
                persistence,
                transition,
            ]
        )
        reranked = self.reranker(
            trajectory.scores,
            candidate_union.expert_ids,
            dense_evidence=[
                active_anchor,
                geometry,
                free,
                mixture.mixture_logits,
                trajectory.scores,
                persistence,
                transition,
            ],
            history_logits=route_history,
            endpoint_context=endpoint,
            mixture_marginals=mixture.mixture_marginals,
            causal_support=causal_support,
            transition_scores=transition,
            branch_scores=hybrid.branch_scores,
            branch_marginals=mixture.branch_marginals,
            branch_weights=mixture.branch_weights,
            branch_mask=tree.horizon_mask,
            extra_candidate_features=extra_candidate_features,
        )
        _, active_marginals, final_cardinality_error = exact_projected_marginals(
            reranked.dense_scores, config.exact_k
        )
        active_top8 = stable_topk(active_marginals, config.exact_k)

        full_scores = torch.cat(
            [
                reranked.dense_scores,
                anchor_scores[:, config.active_horizons :],
            ],
            dim=1,
        )
        anchor_probabilities = anchor.get("future_inclusion_probabilities")
        if anchor_probabilities is None:
            anchor_probabilities = torch.sigmoid(anchor_scores.float()).to(
                anchor_scores.dtype
            )
        full_probabilities = torch.cat(
            [
                active_marginals.to(anchor_probabilities.dtype),
                anchor_probabilities[:, config.active_horizons :],
            ],
            dim=1,
        )
        with torch.autocast(
            device_type=hybrid.predicted_queries.device.type, enabled=False
        ):
            query_coordinates = torch.einsum(
                "bhn,bhlnr->bhlr",
                mixture.branch_weights.float(),
                hybrid.predicted_queries.float(),
            )
        candidate_active_mask = torch.ones_like(
            candidate_union.expert_ids, dtype=torch.bool
        )
        candidate_base = trajectory.scores.gather(
            -1, candidate_union.expert_ids
        )
        candidate_diagnostics = CandidateDiagnostics(
            candidate_ids=candidate_union.expert_ids,
            candidate_mask=candidate_active_mask,
            base_scores=candidate_base,
            residual_scores=reranked.candidate_corrections,
            final_candidate_scores=reranked.dense_scores.gather(
                -1, candidate_union.expert_ids
            ),
        )
        component_scores = {
            "anchor": active_anchor,
            "geometry": geometry,
            "free": free,
            "branch_mixture": mixture.mixture_logits,
            "trajectory": trajectory.scores,
            "persistence": persistence,
            "transition": transition,
        }
        structured = HARPRTTOutput(
            dense_scores=reranked.dense_scores,
            exact_k_marginals=active_marginals,
            topk_ids=active_top8,
            query_coordinates=query_coordinates,
            branch_posteriors=mixture.branch_weights,
            branch_marginals=mixture.branch_marginals,
            trajectory_scores=torch.stack(trajectory.round_scores, dim=1),
            component_scores=component_scores,
            legacy_h5_h8_scores=anchor_scores[:, config.active_horizons :],
            candidate_diagnostics=candidate_diagnostics,
        )
        outputs: dict[str, Any] = dict(anchor)
        outputs.update(
            {
                "future_router_scores": full_scores,
                "future_inclusion_probabilities": full_probabilities,
                "active_scores": reranked.dense_scores,
                "active_marginals": active_marginals,
                "top8_ids": active_top8,
                "branch_scores": hybrid.branch_scores,
                "branch_marginals": mixture.branch_marginals,
                "branch_weights": mixture.branch_weights,
                "branch_posteriors": mixture.branch_weights,
                "branch_posterior_logits": hybrid.branch_logits,
                "branch_mask": tree.horizon_mask,
                "branch_mixture_marginals": mixture.mixture_marginals,
                "branch_log_z": mixture.branch_log_z,
                "branch_cardinality_error": mixture.cardinality_error,
                "trajectory_round_scores": torch.stack(
                    trajectory.round_scores, dim=1
                ),
                "trajectory_round_marginals": torch.stack(
                    trajectory.round_marginals, dim=1
                ),
                "trajectory_corrections": torch.stack(
                    trajectory.corrections, dim=1
                ),
                "candidate_ids": candidate_union.expert_ids,
                "candidate_mask": candidate_active_mask,
                "candidate_dense_mask": candidate_union.dense_mask,
                "candidate_corrections": reranked.candidate_corrections,
                "candidate_causal_history_features": (
                    reranked.causal_history_features
                ),
                "candidate_branch_support_features": (
                    reranked.branch_support_features
                ),
                "router_queries": hybrid.predicted_queries,
                "query_coordinates": query_coordinates,
                "rtt_context": endpoint,
                "final_cardinality_error": final_cardinality_error,
                "component_scores": component_scores,
                "legacy_h5_h8_scores": anchor_scores[
                    :, config.active_horizons :
                ],
                "structured_output": structured,
            }
        )
        return outputs


__all__ = ["HARPRTTTeacher"]
