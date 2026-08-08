"""Node-indexed, label-only counterfactual target companion for B1.5."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .b15 import b15_selector_sha256
from .counterfactual import (
    EXPERTS,
    MAX_DEPTH,
    NATIVE_BF16_CENTERED_LOGIT_TOLERANCE,
    ROUTER_RANK,
    SEALED_SPLITS,
    TARGET_LAYERS,
    TOP_K,
    assert_label_only_mapping,
)
from .exact_k import stable_topk
from .geometry import CenteredRouterGeometry


NODE_COUNTERFACTUAL_SCHEMA = "harp_rtt_counterfactual_target_companion_v3_nodes"
NODE_COUNTERFACTUAL_RECORD_SCHEMA = "harp_rtt_counterfactual_target_record_v3_nodes"
NODE_ROUTER_AUDIT_SCHEMA = "harp_rtt_counterfactual_router_input_audit_v3_nodes"
MAX_TREE_NODES = 32


def assert_node_counterfactual_access_allowed(
    *, split: str, training: bool, enabled: bool
) -> None:
    normalized = str(split).strip().lower()
    if enabled and (normalized in SEALED_SPLITS or normalized != "train"):
        raise PermissionError(
            f"node counterfactual labels are outer-train only, not split={split!r}"
        )
    if enabled and not training:
        raise PermissionError(
            "ordinary inference cannot request node counterfactual labels"
        )


def empty_node_counterfactual_tensors(
    *,
    nodes: int = MAX_TREE_NODES,
    layers: int = TARGET_LAYERS,
    rank: int = ROUTER_RANK,
    experts: int = EXPERTS,
) -> dict[str, Tensor]:
    return {
        "node_mask": torch.zeros(nodes, dtype=torch.bool),
        "node_local_indices": torch.full((nodes,), -1, dtype=torch.int64),
        "parent_local_indices": torch.full((nodes,), -1, dtype=torch.int64),
        "depth": torch.zeros(nodes, dtype=torch.int64),
        "first_divergence_depth": torch.full((nodes,), -1, dtype=torch.int64),
        "budget_node_masks": torch.zeros(3, nodes, dtype=torch.bool),
        "budget_endpoint_masks": torch.zeros(3, nodes, dtype=torch.bool),
        "budget_realized": torch.zeros(3, dtype=torch.int64),
        "budget_category_counts": torch.zeros(3, 4, dtype=torch.int64),
        "source_edge_logp": torch.full((nodes,), float("nan"), dtype=torch.float32),
        "source_path_logp": torch.full((nodes,), float("nan"), dtype=torch.float32),
        "target_edge_logp": torch.full((nodes,), float("nan"), dtype=torch.float32),
        "target_path_logp": torch.full((nodes,), float("nan"), dtype=torch.float32),
        "query_coordinates": torch.zeros(nodes, layers, rank, dtype=torch.float32),
        "router_logits": torch.zeros(nodes, layers, experts, dtype=torch.bfloat16),
        "selected_ids": torch.full((nodes, layers, TOP_K), -1, dtype=torch.int32),
        "selected_weights": torch.zeros(nodes, layers, TOP_K, dtype=torch.bfloat16),
        "valid": torch.zeros(nodes, layers, dtype=torch.bool),
    }


def validate_node_counterfactual_tensors(
    tensors: Mapping[str, Tensor],
    *,
    nodes: int = MAX_TREE_NODES,
    layers: int = TARGET_LAYERS,
    rank: int = ROUTER_RANK,
    experts: int = EXPERTS,
) -> None:
    expected = empty_node_counterfactual_tensors(
        nodes=nodes, layers=layers, rank=rank, experts=experts
    )
    if set(tensors) != set(expected):
        raise ValueError("node counterfactual keys disagree with the v3 contract")
    for name, reference in expected.items():
        value = tensors[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(
                f"node counterfactual {name} has {tuple(value.shape)}/{value.dtype}, "
                f"expected {tuple(reference.shape)}/{reference.dtype}"
            )
    mask = tensors["node_mask"].bool()
    local = tensors["node_local_indices"].long()
    parent = tensors["parent_local_indices"].long()
    depth = tensors["depth"].long()
    divergence = tensors["first_divergence_depth"].long()
    budget_nodes = tensors["budget_node_masks"].bool()
    budget_endpoints = tensors["budget_endpoint_masks"].bool()
    budget_realized = tensors["budget_realized"].long()
    budget_counts = tensors["budget_category_counts"].long()
    valid = tensors["valid"].bool()
    count = int(mask.sum().item())
    if count < 2:
        raise ValueError("node companion must contain H1 plus at least one future node")
    if not torch.equal(mask, torch.arange(nodes) < count):
        raise ValueError("node mask must be a contiguous parent-before-child prefix")
    if not torch.equal(local[mask], torch.arange(count, dtype=torch.int64)):
        raise ValueError("node local indices must be contiguous")
    if (local[~mask] != -1).any() or (parent[~mask] != -1).any() or depth[~mask].any():
        raise ValueError("padded node topology is populated")
    if parent[0] != -1 or depth[0] != 1:
        raise ValueError("node zero must be the exact H1 root")
    for index in range(1, count):
        parent_id = int(parent[index].item())
        if not 0 <= parent_id < index:
            raise ValueError("node parent must precede child")
        if depth[index] != depth[parent_id] + 1 or depth[index] > MAX_DEPTH:
            raise ValueError("node depth is inconsistent with its parent")
    if valid[0].any() or valid[~mask].any():
        raise ValueError("H1 and padded counterfactual labels must remain masked")
    if not valid[1:count].all():
        raise ValueError("every captured H2-H4 node must contain every target layer")
    if (divergence[mask] < -1).any() or (divergence[mask] > MAX_DEPTH).any():
        raise ValueError("invalid first-divergence depth")
    if ((divergence[mask] == 0) | (divergence[mask] == 1)).any():
        raise ValueError("first divergence may only be absent or H2-H4")
    if (budget_nodes & ~mask[None]).any() or (budget_endpoints & ~mask[None]).any():
        raise ValueError("budget masks select padded nodes")
    if budget_nodes[:, 0].logical_not().any():
        raise ValueError("every realized path budget must retain the H1 root")
    if not torch.equal(budget_endpoints.sum(-1), budget_realized):
        raise ValueError("budget endpoint count disagrees with realized budget")
    if not torch.equal(budget_counts.sum(-1), budget_realized):
        raise ValueError("budget category counts disagree with realized budget")
    for budget_index in range(3):
        selected = budget_nodes[budget_index]
        for node_index in torch.nonzero(selected, as_tuple=False).flatten().tolist():
            parent_id = int(parent[node_index])
            if parent_id >= 0 and not bool(selected[parent_id]):
                raise ValueError("budget node visibility is not ancestor-closed")
        if (budget_endpoints[budget_index] & ~selected).any():
            raise ValueError("budget endpoint is absent from its node mask")

    source_edge = tensors["source_edge_logp"].float()
    source_path = tensors["source_path_logp"].float()
    if not torch.isfinite(source_edge[mask]).all() or not torch.isfinite(source_path[mask]).all():
        raise ValueError("source probabilities are missing for a valid node")
    if torch.isfinite(source_edge[~mask]).any() or torch.isfinite(source_path[~mask]).any():
        raise ValueError("source probabilities populate padding")
    if abs(float(source_edge[0])) > 1e-7 or abs(float(source_path[0])) > 1e-7:
        raise ValueError("source tree is not rooted at exact H1")
    if (source_edge[mask] > 1e-6).any() or (source_path[mask] > 1e-6).any():
        raise ValueError("source log probability is positive")
    for index in range(1, count):
        expected_path = source_path[parent[index]] + source_edge[index]
        if not torch.isclose(source_path[index], expected_path, rtol=0.0, atol=2e-5):
            raise ValueError("source edge/path probabilities are inconsistent")

    target_edge = tensors["target_edge_logp"].float()
    target_path = tensors["target_path_logp"].float()
    if not torch.isnan(target_edge[0]) or not torch.isnan(target_path[0]):
        raise ValueError("target H1 probability must remain masked")
    if not torch.isfinite(target_edge[1:count]).all() or not torch.isfinite(target_path[1:count]).all():
        raise ValueError("target probability is missing for a future node")
    if torch.isfinite(target_edge[~mask]).any() or torch.isfinite(target_path[~mask]).any():
        raise ValueError("target probabilities populate padding")
    if (target_edge[1:count] > 1e-6).any() or (target_path[1:count] > 1e-6).any():
        raise ValueError("target log probability is positive")
    for index in range(1, count):
        parent_id = int(parent[index])
        expected_path = (
            target_edge[index]
            if parent_id == 0
            else target_path[parent_id] + target_edge[index]
        )
        if not torch.isclose(target_path[index], expected_path, rtol=0.0, atol=2e-5):
            raise ValueError("target edge/path probabilities are inconsistent")

    active_ids = tensors["selected_ids"][valid]
    if active_ids.numel() and ((active_ids < 0) | (active_ids >= experts)).any():
        raise ValueError("selected expert ID is outside the target namespace")


def audit_node_counterfactual_geometry(
    tensors: Mapping[str, Tensor],
    geometry: CenteredRouterGeometry,
    *,
    maximum_logit_error: float = NATIVE_BF16_CENTERED_LOGIT_TOLERANCE,
    authoritative_topk_device: str | torch.device | None = None,
) -> dict[str, Any]:
    validate_node_counterfactual_tensors(
        tensors,
        nodes=tensors["node_mask"].numel(),
        layers=geometry.layers,
        rank=geometry.maximum_rank,
        experts=geometry.experts,
    )
    valid = tensors["valid"].bool()
    q = tensors["query_coordinates"].float()
    captured = tensors["router_logits"].float()
    reconstructed = geometry.score_coordinates(q)
    centered = captured - captured.mean(dim=-1, keepdim=True)
    difference = (reconstructed - centered)[valid]
    native_scores = captured
    if authoritative_topk_device is not None:
        device = torch.device(authoritative_topk_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("authoritative CUDA top-k requested without CUDA")
        native_scores = captured.to(device)
    native_top = torch.topk(
        torch.softmax(native_scores.float(), dim=-1), TOP_K, dim=-1
    ).indices.to(tensors["selected_ids"].device)
    geometry_top = stable_topk(reconstructed, TOP_K)
    stored = tensors["selected_ids"].long()
    native_rows = (
        native_top[valid].sort(-1).values == stored[valid].sort(-1).values
    ).all(-1)
    geometry_rows = (
        geometry_top[valid].sort(-1).values == stored[valid].sort(-1).values
    ).all(-1)
    maximum = float(difference.abs().max().item())
    rows = int(valid.sum().item())
    return {
        "valid_layer_rows": rows,
        "maximum_absolute_centered_logit_error": maximum,
        "native_selected_ids_exact": torch.equal(native_top[valid], stored[valid]),
        "native_selected_set_matching_rows": int(native_rows.sum().item()),
        "geometry_selected_ids_exact": torch.equal(geometry_top[valid], stored[valid]),
        "geometry_selected_set_matching_rows": int(geometry_rows.sum().item()),
        "geometry_numerical_boundary_rows": int((~geometry_rows).sum().item()),
        "geometry_selected_set_agreement": float(geometry_rows.float().mean().item()),
        "authoritative_topk_device": str(native_scores.device),
        "passed": bool(maximum <= maximum_logit_error and native_rows.all()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_node_counterfactual_companion(
    root: str | Path,
    *,
    split: str,
    training: bool,
    expected_bindings: Mapping[str, str] | None = None,
) -> tuple[dict[tuple[str, int], dict[str, Tensor]], dict[str, Any]]:
    assert_node_counterfactual_access_allowed(
        split=split, training=training, enabled=True
    )
    directory = Path(root).expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != NODE_COUNTERFACTUAL_SCHEMA:
        raise ValueError("node counterfactual companion schema mismatch")
    assert_label_only_mapping(manifest)
    if manifest.get("split") != "train" or manifest.get("sealed_test_opened") is not False:
        raise PermissionError("node counterfactual companion is not outer-train only")
    contract = manifest.get("tensor_contract")
    if not isinstance(contract, Mapping) or (
        contract.get("layout") != "unique_parent_before_child_nodes"
        or contract.get("h1_masked") is not True
        or contract.get("target_path_probability_condition") != "exact_committed_h1"
    ):
        raise ValueError("node counterfactual tensor contract is invalid")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping) or bindings.get("selector_sha256") != b15_selector_sha256():
        raise ValueError("node counterfactual selector binding mismatch")
    if expected_bindings is not None:
        for name, expected in expected_bindings.items():
            if bindings.get(name) != expected:
                raise ValueError(f"node counterfactual binding mismatch for {name}")
    audit = json.loads((directory / "COUNTERFACTUAL_AUDIT.json").read_text())
    if audit.get("passed") is not True:
        raise ValueError("node counterfactual companion has not passed its audit")
    checksums = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest
    from safetensors import safe_open
    from safetensors.torch import load_file

    labels: dict[tuple[str, int], dict[str, Tensor]] = {}
    for record in manifest.get("records", []):
        relative = str(record["record"]["path"])
        path = directory / relative
        expected = checksums.get(relative)
        if expected is None or _sha256(path) != expected or record["record"].get("sha256") != expected:
            raise ValueError(f"node counterfactual record checksum mismatch: {relative}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        if (
            metadata.get("schema") != NODE_COUNTERFACTUAL_RECORD_SCHEMA
            or metadata.get("label_only") != "true"
            or metadata.get("runtime_available") != "false"
        ):
            raise ValueError("node counterfactual record metadata is invalid")
        tensors = dict(load_file(path, device="cpu"))
        validate_node_counterfactual_tensors(tensors)
        key = (str(record["request_id"]), int(record["source_position"]))
        if key in labels:
            raise ValueError(f"duplicate node counterfactual join key {key}")
        labels[key] = tensors
    if not labels:
        raise ValueError("node counterfactual companion contains no records")
    return labels, manifest


class NodeCounterfactualDatasetAdapter(Dataset[dict[str, Any]]):
    def __init__(
        self,
        base: Dataset[dict[str, Any]],
        labels: Mapping[tuple[str, int], Mapping[str, Tensor]],
        *,
        split: str,
        training: bool,
        enabled: bool = True,
    ) -> None:
        assert_node_counterfactual_access_allowed(
            split=split, training=training, enabled=enabled
        )
        self.base = base
        self.labels = labels
        self.enabled = enabled

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        if not self.enabled:
            return item
        metadata = item.get("metadata")
        targets = item.get("targets")
        inputs = item.get("inputs")
        if not isinstance(metadata, Mapping) or not isinstance(targets, Mapping) or not isinstance(inputs, Mapping):
            raise TypeError("base item must expose metadata, inputs, and targets")
        if "counterfactual" in inputs:
            raise ValueError("node counterfactual labels leaked into model inputs")
        key = (str(metadata["request_id"]), int(metadata["position"]))
        record = self.labels.get(key)
        if record is None:
            raise KeyError(f"node companion has no label for {key}")
        validate_node_counterfactual_tensors(record)
        result = dict(item)
        result_targets = dict(targets)
        result_targets["counterfactual"] = dict(record)
        result["targets"] = result_targets
        return result


__all__ = [
    "MAX_TREE_NODES",
    "NODE_COUNTERFACTUAL_RECORD_SCHEMA",
    "NODE_COUNTERFACTUAL_SCHEMA",
    "NODE_ROUTER_AUDIT_SCHEMA",
    "NodeCounterfactualDatasetAdapter",
    "assert_node_counterfactual_access_allowed",
    "audit_node_counterfactual_geometry",
    "empty_node_counterfactual_tensors",
    "load_node_counterfactual_companion",
    "validate_node_counterfactual_tensors",
]
