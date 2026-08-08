"""Causal metadata contract for the HARP-RTT v2 matched tree controls.

The control trees are diagnostic views only.  Their ready records contain MTP
information available before target H1 executes; factual path occurrence is
derived later and emitted in a separate label-only resolved record.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

from adaptive_mtp_tree import AdaptiveTreeNode, build_acceptance_labels


CONTROL_SCHEMA = "harp_rtt_matched_tree_controls_v1"
CONTROL_READY_EVENT = "mtp_matched_control_tree_ready"
CONTROL_RESOLVED_EVENT = "mtp_matched_control_tree_resolved"
CONTROL_NAMES = ("greedy", "fixed16", "adaptive16", "fixed32", "adaptive32")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def structural_rows(nodes: Iterable[AdaptiveTreeNode]) -> list[dict[str, Any]]:
    """Return the stable structural/probability identity of a tree view."""

    return [
        {
            "local_index": int(node.local_index),
            "parent_local_index": (
                None
                if node.parent_local_index is None
                else int(node.parent_local_index)
            ),
            "depth": int(node.depth),
            "token_id": int(node.token_id),
            "token_rank_under_parent": int(node.token_rank_under_parent),
            "token_path_ids": [int(value) for value in node.token_path_ids],
            "token_path_log_probabilities": [
                float(value) for value in node.token_path_log_probabilities
            ],
            "local_token_log_probability": float(
                node.local_token_log_probability
            ),
            "path_log_probability": float(node.path_log_probability),
        }
        for node in nodes
    ]


def structural_hash(nodes: Iterable[AdaptiveTreeNode]) -> str:
    return _canonical_hash(structural_rows(nodes))


def assert_identical_structure(
    left: Iterable[AdaptiveTreeNode], right: Iterable[AdaptiveTreeNode]
) -> None:
    left_rows = structural_rows(left)
    right_rows = structural_rows(right)
    if left_rows != right_rows:
        raise RuntimeError("canonical adaptive-16 prefix differs from independent build")


def greedy_spine_prefix(nodes: list[AdaptiveTreeNode]) -> list[AdaptiveTreeNode]:
    """Return the canonical rank-zero prefix, shortened when it terminates.

    The adaptive builder emits its greedy path first, but deliberately stops
    expanding that path when MTP predicts EOS.  Subsequent nodes may therefore
    belong to a different branch and must never be mislabeled as greedy.
    """

    if not nodes:
        raise ValueError("cannot extract a greedy spine from an empty tree")
    root = nodes[0]
    if root.parent_local_index is not None or root.depth != 1:
        raise ValueError("adaptive tree has an invalid greedy root")
    result = [root]
    while result[-1].depth < 4:
        parent = result[-1]
        children = [
            node
            for node in nodes
            if node.parent_local_index == parent.local_index
            and node.token_rank_under_parent == 0
        ]
        if not children:
            break
        if len(children) != 1:
            raise ValueError("adaptive tree has multiple rank-zero children")
        child = children[0]
        if child.local_index != len(result) or child.depth != parent.depth + 1:
            raise ValueError("adaptive greedy spine is not the canonical tree prefix")
        result.append(child)
    return result

def serialize_nodes(nodes: Iterable[AdaptiveTreeNode]) -> list[dict[str, Any]]:
    """Serialize causal node data sufficient to re-audit beam selection."""

    result: list[dict[str, Any]] = []
    for node in nodes:
        observation = node.observation
        result.append(
            {
                **structural_rows((node,))[0],
                "path_probability": float(node.path_probability),
                "path_id": str(node.path_id),
                "branch_id": str(node.branch_id),
                "vocabulary_size": int(observation.vocabulary_size),
                "vocabulary_entropy": float(observation.vocabulary_entropy),
                "vocab_top64_token_ids": [
                    int(value) for value in observation.top_token_ids
                ],
                "vocab_top64_log_probabilities": [
                    float(value) for value in observation.top_log_probabilities
                ],
            }
        )
    return result


def resolved_labels(
    nodes: list[AdaptiveTreeNode],
    *,
    committed_token_ids: list[int],
    committed_prefix_position: int,
) -> dict[str, Any]:
    labels = build_acceptance_labels(
        nodes,
        committed_token_ids=committed_token_ids,
        committed_prefix_position=committed_prefix_position,
    )
    occurrence = {
        str(depth): any(
            label["acceptance_label_valid"] is True
            and label["branch_path_accepted"] is True
            and int(label["engine_draft_depth"]) == depth
            for label in labels
        )
        for depth in range(1, 5)
    }
    return {
        "labels": labels,
        "path_occurrence_by_horizon": occurrence,
        "all_labels_valid": all(
            label["acceptance_label_valid"] is True for label in labels
        ),
    }


def assert_ready_record(record: dict[str, Any]) -> None:
    if record.get("control_schema") != CONTROL_SCHEMA:
        raise ValueError("matched-control schema mismatch")
    if record.get("control_name") not in CONTROL_NAMES:
        raise ValueError("unknown matched-control name")
    if record.get("diagnostic_only") is not True:
        raise ValueError("matched controls must be diagnostic-only")
    if record.get("model_input") is not False:
        raise ValueError("matched controls may not enter model inputs")
    if record.get("target_labels_present") is not False:
        raise ValueError("ready control records may not contain target labels")
    nodes = record.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("matched control must contain at least one node")
    if int(record.get("node_count", -1)) != len(nodes):
        raise ValueError("matched-control node count mismatch")
    expected_indices = list(range(len(nodes)))
    if [int(node["local_index"]) for node in nodes] != expected_indices:
        raise ValueError("matched-control local indices are not contiguous")
    for index, node in enumerate(nodes):
        depth = int(node["depth"])
        parent = node["parent_local_index"]
        path = [int(value) for value in node["token_path_ids"]]
        logps = [float(value) for value in node["token_path_log_probabilities"]]
        if len(path) != depth or len(logps) != depth:
            raise ValueError("matched-control path depth is inconsistent")
        if parent is None:
            if index != 0 or depth != 1:
                raise ValueError("matched-control root is invalid")
        else:
            parent = int(parent)
            if not 0 <= parent < index:
                raise ValueError("matched-control parent is not earlier")
            parent_node = nodes[parent]
            if depth != int(parent_node["depth"]) + 1:
                raise ValueError("matched-control child depth mismatch")
            if path[:-1] != [int(value) for value in parent_node["token_path_ids"]]:
                raise ValueError("matched-control child does not extend its parent")
        if not math.isclose(
            float(node["path_probability"]),
            math.exp(float(node["path_log_probability"])),
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            raise ValueError("matched-control path probability mismatch")
    if record.get("structural_hash") != _canonical_hash(
        [
            {
                key: node[key]
                for key in (
                    "local_index",
                    "parent_local_index",
                    "depth",
                    "token_id",
                    "token_rank_under_parent",
                    "token_path_ids",
                    "token_path_log_probabilities",
                    "local_token_log_probability",
                    "path_log_probability",
                )
            }
            for node in nodes
        ]
    ):
        raise ValueError("matched-control structural hash mismatch")


def assert_resolved_record(record: dict[str, Any]) -> None:
    if record.get("control_schema") != CONTROL_SCHEMA:
        raise ValueError("matched-control resolved schema mismatch")
    if record.get("control_name") not in CONTROL_NAMES:
        raise ValueError("unknown resolved matched-control name")
    if record.get("labels_only") is not True:
        raise ValueError("resolved matched-control records must be labels-only")
    if record.get("available_at_runtime") is not False:
        raise ValueError("resolved control labels may not be available at runtime")
    labels = record.get("labels")
    if not isinstance(labels, list) or int(record.get("node_count", -1)) != len(labels):
        raise ValueError("resolved matched-control label count mismatch")
    if any(label.get("label_only") is not True for label in labels):
        raise ValueError("resolved control contains a non-label-only row")
