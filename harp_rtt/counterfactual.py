"""Leak-safe counterfactual target companion for HARP-RTT v2.

The companion is deliberately separate from causal capture.  Selection reads
only MTP tree structure/probabilities.  Target-route tensors are label-only and
can be joined only to outer-train examples in an explicitly training dataset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .exact_k import stable_topk
from .geometry import CenteredRouterGeometry


COUNTERFACTUAL_SCHEMA = "harp_rtt_counterfactual_target_companion_v2"
COUNTERFACTUAL_RECORD_SCHEMA = "harp_rtt_counterfactual_target_record_v2"
SELECTOR_SCHEMA = "harp_rtt_divergence_depth_selector_v1"
PATH_SLOTS = 4
MAX_DEPTH = 4
TARGET_LAYERS = 40
ROUTER_RANK = 255
EXPERTS = 256
TOP_K = 8
SEALED_SPLITS = frozenset(
    {"validation", "calibration", "test", "sealed_test", "sealed-test"}
)


class TreeNodeLike(Protocol):
    local_index: int
    parent_local_index: int | None
    depth: int
    token_id: int
    token_rank_under_parent: int
    token_path_ids: tuple[int, ...]
    token_path_log_probabilities: tuple[float, ...]
    path_log_probability: float


@dataclass(frozen=True)
class SelectedCounterfactualPath:
    """One causal selector slot; ``node_local_indices`` is H1--H4 padded."""

    slot: int
    first_divergence_depth: int | None
    endpoint_local_index: int
    path_depth: int
    token_ids: tuple[int, ...]
    node_local_indices: tuple[int, int, int, int]
    source_edge_log_probabilities: tuple[float, float, float, float]


def selector_manifest() -> dict[str, Any]:
    return {
        "schema": SELECTOR_SCHEMA,
        "slots": [
            "greedy",
            "first_divergence_h2",
            "match_h2_first_divergence_h3",
            "match_h2_h3_first_divergence_h4",
        ],
        "missing_category": "masked_without_backfill",
        "ranking": [
            "maximum_available_depth",
            "descending_cumulative_mtp_log_probability",
            "lexicographic_token_path",
            "local_index",
        ],
        "uses_target_labels": False,
        "uses_acceptance": False,
        "uses_factual_continuation": False,
    }


def selector_sha256() -> str:
    encoded = json.dumps(
        selector_manifest(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_nodes(nodes: Sequence[TreeNodeLike]) -> None:
    if not nodes:
        raise ValueError("counterfactual selector received an empty tree")
    for expected, node in enumerate(nodes):
        if node.local_index != expected:
            raise ValueError("tree local indices must be contiguous")
        if not 1 <= node.depth <= MAX_DEPTH:
            raise ValueError("counterfactual selector accepts only H1--H4")
        if len(node.token_path_ids) != node.depth:
            raise ValueError("node path length disagrees with depth")
        if len(node.token_path_log_probabilities) != node.depth:
            raise ValueError("node edge-log-probability length disagrees with depth")
        if expected == 0:
            if node.depth != 1 or node.parent_local_index is not None:
                raise ValueError("tree node zero must be the exact H1 root")
        else:
            parent = node.parent_local_index
            if parent is None or not 0 <= parent < expected:
                raise ValueError("tree parent must precede child")
            if nodes[parent].token_path_ids != node.token_path_ids[:-1]:
                raise ValueError("child path does not extend its parent")


def _greedy_tokens(nodes: Sequence[TreeNodeLike]) -> tuple[int, ...]:
    tokens = [int(nodes[0].token_id)]
    parent = 0
    for depth in range(2, MAX_DEPTH + 1):
        children = [
            node
            for node in nodes
            if node.parent_local_index == parent and node.depth == depth
        ]
        greedy = [node for node in children if node.token_rank_under_parent == 0]
        if not greedy:
            break
        chosen = min(greedy, key=lambda node: (node.token_id, node.local_index))
        tokens.append(int(chosen.token_id))
        parent = chosen.local_index
    return tuple(tokens)


def _first_divergence(path: tuple[int, ...], greedy: tuple[int, ...]) -> int | None:
    for index, (actual, reference) in enumerate(zip(path, greedy), start=1):
        if actual != reference:
            return index
    return None


def _path_indices(
    nodes: Sequence[TreeNodeLike], endpoint: TreeNodeLike
) -> tuple[int, int, int, int]:
    result = [-1] * MAX_DEPTH
    current: TreeNodeLike | None = endpoint
    while current is not None:
        result[current.depth - 1] = current.local_index
        parent = current.parent_local_index
        current = None if parent is None else nodes[parent]
    return tuple(result)  # type: ignore[return-value]


def select_counterfactual_paths(
    nodes: Sequence[TreeNodeLike],
) -> tuple[SelectedCounterfactualPath | None, ...]:
    """Select greedy/H2/H3/H4 divergence slots using causal MTP values only."""

    _validate_nodes(nodes)
    greedy = _greedy_tokens(nodes)
    categories: tuple[int | None, ...] = (None, 2, 3, 4)
    selected: list[SelectedCounterfactualPath | None] = []
    for slot, category in enumerate(categories):
        candidates = []
        for node in nodes:
            path = tuple(int(value) for value in node.token_path_ids)
            divergence = _first_divergence(path, greedy)
            if category is None:
                eligible = divergence is None
            else:
                eligible = divergence == category and node.depth >= category
            if eligible:
                candidates.append(node)
        if not candidates:
            selected.append(None)
            continue
        # A deeper path gives more H2--H4 labels.  Within the deepest available
        # stratum, choose the highest MTP path probability with stable ties.
        maximum_depth = max(node.depth for node in candidates)
        endpoint = min(
            (node for node in candidates if node.depth == maximum_depth),
            key=lambda node: (
                -float(node.path_log_probability),
                tuple(int(value) for value in node.token_path_ids),
                int(node.local_index),
            ),
        )
        edge = list(float(value) for value in endpoint.token_path_log_probabilities)
        edge.extend([float("nan")] * (MAX_DEPTH - len(edge)))
        selected.append(
            SelectedCounterfactualPath(
                slot=slot,
                first_divergence_depth=category,
                endpoint_local_index=endpoint.local_index,
                path_depth=endpoint.depth,
                token_ids=tuple(int(value) for value in endpoint.token_path_ids),
                node_local_indices=_path_indices(nodes, endpoint),
                source_edge_log_probabilities=tuple(edge),  # type: ignore[arg-type]
            )
        )
    return tuple(selected)


def assert_counterfactual_access_allowed(
    *, split: str, training: bool, enabled: bool
) -> None:
    normalized = str(split).strip().lower()
    if enabled and (normalized in SEALED_SPLITS or normalized != "train"):
        raise PermissionError(
            f"counterfactual labels are outer-train only, not split={split!r}"
        )
    if enabled and not training:
        raise PermissionError("ordinary inference cannot request counterfactual labels")


def assert_label_only_mapping(value: Mapping[str, Any]) -> None:
    if value.get("label_only") is not True:
        raise ValueError("counterfactual record must be explicitly label-only")
    if value.get("runtime_available") is not False:
        raise ValueError("counterfactual record must be runtime-unavailable")
    forbidden = ("inputs", "feature", "causal_input", "serving")
    for key in value:
        lowered = str(key).lower()
        if any(token in lowered for token in forbidden):
            raise ValueError(f"counterfactual label field has input-like name {key!r}")


def empty_counterfactual_tensors(
    *, layers: int = TARGET_LAYERS, rank: int = ROUTER_RANK, experts: int = EXPERTS
) -> dict[str, Tensor]:
    shape = (PATH_SLOTS, MAX_DEPTH)
    return {
        "path_mask": torch.zeros(PATH_SLOTS, dtype=torch.bool),
        "path_depths": torch.zeros(PATH_SLOTS, dtype=torch.int64),
        "first_divergence_depths": torch.full((PATH_SLOTS,), -1, dtype=torch.int64),
        "node_local_indices": torch.full(shape, -1, dtype=torch.int64),
        "source_path_logp": torch.full(shape, float("nan"), dtype=torch.float32),
        "target_edge_logp": torch.full(shape, float("nan"), dtype=torch.float32),
        "target_path_logp": torch.full(shape, float("nan"), dtype=torch.float32),
        "query_coordinates": torch.zeros(*shape, layers, rank, dtype=torch.float32),
        "router_logits": torch.zeros(*shape, layers, experts, dtype=torch.bfloat16),
        "selected_ids": torch.full((*shape, layers, TOP_K), -1, dtype=torch.int32),
        "selected_weights": torch.zeros(*shape, layers, TOP_K, dtype=torch.bfloat16),
        "valid": torch.zeros(*shape, layers, dtype=torch.bool),
    }


def validate_counterfactual_tensors(
    tensors: Mapping[str, Tensor],
    *,
    layers: int = TARGET_LAYERS,
    rank: int = ROUTER_RANK,
    experts: int = EXPERTS,
) -> None:
    expected = empty_counterfactual_tensors(layers=layers, rank=rank, experts=experts)
    if set(tensors) != set(expected):
        raise ValueError("counterfactual tensor keys disagree with the v2 contract")
    for name, reference in expected.items():
        value = tensors[name]
        if tuple(value.shape) != tuple(reference.shape) or value.dtype != reference.dtype:
            raise ValueError(
                f"counterfactual {name} has {tuple(value.shape)}/{value.dtype}, "
                f"expected {tuple(reference.shape)}/{reference.dtype}"
            )
    path_mask = tensors["path_mask"].bool()
    valid = tensors["valid"].bool()
    if valid[:, 0].any():
        raise ValueError("counterfactual H1 labels must remain masked")
    if (valid & ~path_mask[:, None, None]).any():
        raise ValueError("masked counterfactual path contains valid labels")
    if (tensors["path_depths"] > MAX_DEPTH).any():
        raise ValueError("counterfactual path depth exceeds H4")
    active_ids = tensors["selected_ids"][valid]
    if active_ids.numel() and ((active_ids < 0) | (active_ids >= experts)).any():
        raise ValueError("counterfactual selected expert ID is out of range")


def audit_counterfactual_geometry(
    tensors: Mapping[str, Tensor],
    geometry: CenteredRouterGeometry,
    *,
    maximum_logit_error: float = 2e-2,
) -> dict[str, Any]:
    validate_counterfactual_tensors(
        tensors,
        layers=geometry.layers,
        rank=geometry.maximum_rank,
        experts=geometry.experts,
    )
    valid = tensors["valid"].bool()
    if not valid.any():
        raise ValueError("counterfactual audit observed no valid H2--H4 label")
    q = tensors["query_coordinates"].float()
    captured = tensors["router_logits"].float()
    reconstructed = geometry.score_coordinates(q)
    centered = captured - captured.mean(dim=-1, keepdim=True)
    difference = (reconstructed - centered)[valid]
    captured_top = stable_topk(captured, TOP_K)
    geometry_top = stable_topk(reconstructed, TOP_K)
    stored = tensors["selected_ids"].long()
    native_match = torch.equal(captured_top[valid], stored[valid])
    geometry_match = torch.equal(geometry_top[valid], stored[valid])
    maximum = float(difference.abs().max().item())
    report = {
        "valid_layer_rows": int(valid.sum().item()),
        "maximum_absolute_centered_logit_error": maximum,
        "native_selected_ids_exact": native_match,
        "geometry_selected_ids_exact": geometry_match,
    }
    report["passed"] = bool(
        maximum <= maximum_logit_error and native_match and geometry_match
    )
    return report
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_counterfactual_companion(
    root: str | Path,
    *,
    split: str,
    training: bool,
    expected_bindings: Mapping[str, str] | None = None,
) -> tuple[dict[tuple[str, int], dict[str, Tensor]], dict[str, Any]]:
    """Load an audited immutable companion into a train-only join index."""

    assert_counterfactual_access_allowed(split=split, training=training, enabled=True)
    directory = Path(root).expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != COUNTERFACTUAL_SCHEMA:
        raise ValueError("counterfactual companion manifest schema mismatch")
    assert_label_only_mapping(manifest)
    if manifest.get("split") != "train" or manifest.get("sealed_test_opened") is not False:
        raise PermissionError("counterfactual companion is not outer-train-only")
    audit = json.loads((directory / "COUNTERFACTUAL_AUDIT.json").read_text())
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise ValueError("counterfactual companion has not passed its blocking audit")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("counterfactual companion bindings are missing")
    if bindings.get("selector_sha256") != selector_sha256():
        raise ValueError("counterfactual selector binding mismatch")
    if expected_bindings is not None:
        for name, expected in expected_bindings.items():
            if bindings.get(name) != expected:
                raise ValueError(f"counterfactual binding mismatch for {name}")
    checksums: dict[str, str] = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest
    from safetensors import safe_open
    from safetensors.torch import load_file

    labels: dict[tuple[str, int], dict[str, Tensor]] = {}
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("counterfactual companion contains no records")
    for record in records:
        relative = str(record["record"]["path"])
        path = directory / relative
        expected = checksums.get(relative)
        if expected is None or _sha256_file(path) != expected:
            raise ValueError(f"counterfactual record checksum mismatch: {relative}")
        if record["record"].get("sha256") != expected:
            raise ValueError("record checksum disagrees with companion manifest")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        if metadata.get("label_only") != "true" or metadata.get("runtime_available") != "false":
            raise ValueError("counterfactual record crossed the label-only boundary")
        tensors = dict(load_file(path, device="cpu"))
        validate_counterfactual_tensors(tensors)
        key = (str(record["request_id"]), int(record["source_position"]))
        if key in labels:
            raise ValueError(f"duplicate counterfactual join key {key}")
        labels[key] = tensors
    return labels, manifest




class CounterfactualDatasetAdapter(Dataset[dict[str, Any]]):
    """Join a verified per-example tensor mapping under ``targets`` only."""

    def __init__(
        self,
        base: Dataset[dict[str, Any]],
        labels: Mapping[tuple[str, int], Mapping[str, Tensor]],
        *,
        split: str,
        training: bool,
        enabled: bool = True,
    ) -> None:
        assert_counterfactual_access_allowed(
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
        if not isinstance(metadata, Mapping) or not isinstance(targets, Mapping):
            raise TypeError("base item must expose metadata and targets mappings")
        if not isinstance(inputs, Mapping):
            raise TypeError("base item must expose an inputs mapping")
        if "counterfactual" in inputs:
            raise ValueError("counterfactual labels leaked into base inputs")
        key = (str(metadata["request_id"]), int(metadata["position"]))
        record = self.labels.get(key)
        if record is None:
            raise KeyError(f"counterfactual companion has no label for {key}")
        validate_counterfactual_tensors(record)
        result = dict(item)
        result_targets = dict(targets)
        result_targets["counterfactual"] = dict(record)
        result["targets"] = result_targets
        if "counterfactual" in result["inputs"]:
            raise AssertionError("counterfactual labels crossed the target boundary")
        return result


__all__ = [
    "COUNTERFACTUAL_RECORD_SCHEMA",
    "COUNTERFACTUAL_SCHEMA",
    "CounterfactualDatasetAdapter",
    "SelectedCounterfactualPath",
    "assert_counterfactual_access_allowed",
    "assert_label_only_mapping",
    "audit_counterfactual_geometry",
    "empty_counterfactual_tensors",
    "select_counterfactual_paths",
    "load_counterfactual_companion",
    "selector_manifest",
    "selector_sha256",
    "validate_counterfactual_tensors",
]
