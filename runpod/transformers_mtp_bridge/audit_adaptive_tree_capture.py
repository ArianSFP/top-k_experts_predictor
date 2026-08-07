#!/usr/bin/env python3
"""Blocking audit for HARP-RTT adaptive native-MTP capture schema v2."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_transformers_segment import (  # noqa: E402
    AuditError,
    Store,
    _audit_route,
    audit_sequences,
    audit_target,
    check_checksums,
    check_tensor_roles,
)
from adaptive_capture_contract import (  # noqa: E402
    ANCHOR_SPINE_NODE_EVENT,
    ANCHOR_SPINE_READY_EVENT,
    ANCHOR_SPINE_REQUIRED_DEPTH,
    ANCHOR_SPINE_ROLES,
    ANCHOR_SPINE_SCHEMA,
    FORMAT_VERSION,
    MAX_CAPTURE_DEPTH,
    SCHEMA,
    assert_causal_anchor_spine_node_record,
    assert_causal_mtp_node_record,
)
from capture_transformers_segment import prefix_hash  # noqa: E402


ADAPTIVE_MTP_REQUIRED = {
    "frozen_target_token_embedding": (2048,),
    "mtp_normalized_token_embedding": (2048,),
    "mtp_normalized_previous_hidden": (2048,),
    "mtp_fused_state": (2048,),
    "mtp_router_input": (2048,),
    "mtp_hidden_state": (2048,),
    "mtp_post_ffn_hidden": (2048,),
    "mtp_vocabulary_head_input": (2048,),
    "raw_mtp_router_logits": (256,),
    "full_mtp_router_probabilities": (256,),
    "mtp_selected_expert_ids": (8,),
    "mtp_selected_execution_weights": (8,),
    "vocab_top64_token_ids": (64,),
    "vocab_top64_log_probabilities": (64,),
    "vocab_top64_probabilities": (64,),
}

ANCHOR_SPINE_REQUIRED = {
    "harp_anchor_mtp_vocabulary_head_input": (2048,),
    "harp_anchor_raw_mtp_router_logits": (256,),
    "harp_anchor_vocab_top64_token_ids": (64,),
    "harp_anchor_vocab_top64_log_probabilities": (64,),
}
ANCHOR_SPINE_PAYLOAD_FILES = {
    "harp_anchor_mtp_vocabulary_head_input": "sidecars/harp_anchor_states.bin",
    "harp_anchor_raw_mtp_router_logits": "sidecars/harp_anchor_routes.bin",
    "harp_anchor_vocab_top64_token_ids": "sidecars/harp_anchor_vocab.bin",
    "harp_anchor_vocab_top64_log_probabilities": "sidecars/harp_anchor_vocab.bin",
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _check_anchor_checksum_inventory(root: Path) -> int:
    listed = {
        line.split("  ", 1)[1]
        for line in (root / "SHA256SUMS").read_text().splitlines()
        if line.strip()
    }
    expected = set(ANCHOR_SPINE_PAYLOAD_FILES.values())
    missing = expected - listed
    if missing:
        raise AuditError(
            f"anchor sidecars absent from checksum inventory: {sorted(missing)}"
        )
    return len(expected)


def _event_map(store: Store, kind: str, key: str) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for row in store.rows:
        if row.get("event") != kind:
            continue
        value = row[key]
        if value in result:
            raise AuditError(f"duplicate {kind} key {value!r}")
        result[value] = row
    return result


def _audit_eos_terminated_depths(
    rows: list[dict[str, Any]],
    *,
    eos_token_id: int,
    tree_id: str,
) -> None:
    """Require H1--H4 coverage unless the last materialized frontier is EOS-only.

    Once an EOS-only frontier legitimately terminates a tree, later depths are
    intentionally absent. Returning at that point avoids treating the empty
    next frontier as a second, independent coverage failure.
    """

    rows_by_depth = {
        depth: [
            row
            for row in rows
            if int(row["engine_draft_depth"]) == depth
        ]
        for depth in range(1, MAX_CAPTURE_DEPTH + 1)
    }
    for depth in range(1, MAX_CAPTURE_DEPTH):
        if rows_by_depth[depth + 1]:
            continue
        frontier = rows_by_depth[depth]
        later_depths = [
            later
            for later in range(depth + 2, MAX_CAPTURE_DEPTH + 1)
            if rows_by_depth[later]
        ]
        if later_depths:
            raise AuditError(
                f"tree {tree_id} has nodes after missing H{depth + 1}"
            )
        if not frontier or any(
            int(row["node_token_id"]) != eos_token_id for row in frontier
        ):
            raise AuditError(
                f"tree {tree_id} lacks H{depth + 1} without an EOS-only frontier"
            )
        return


def _require_flag(
    row: dict[str, Any],
    key: str,
    expected: bool,
    *,
    subject: str,
) -> None:
    if row.get(key) is not expected:
        raise AuditError(f"{subject} must declare {key}={expected}")


def _audit_anchor_manifest(
    manifest: dict[str, Any],
    anchor_report: dict[str, Any],
) -> dict[str, Any]:
    if tuple(ANCHOR_SPINE_REQUIRED) != ANCHOR_SPINE_ROLES:
        raise AuditError("auditor anchor tensor roles drifted from the capture contract")
    anchor = manifest.get("legacy_harp_anchor_spine_manifest")
    if not isinstance(anchor, dict):
        raise AuditError("missing immutable legacy HARP anchor-spine manifest")
    expected = {
        "schema": ANCHOR_SPINE_SCHEMA,
        "purpose": "epoch-zero LegacyHARPAnchorBridge compatibility only",
        "root": "same exact committed target H1 token as the adaptive tree",
        "required_depth": ANCHOR_SPINE_REQUIRED_DEPTH,
        "depths": list(range(1, ANCHOR_SPINE_REQUIRED_DEPTH + 1)),
        "parent_rule": "H2-H6 consume the immediately preceding node local top-1",
        "eos_rule": "continue through all six depths even when an earlier token is EOS",
        "native_branch_execution": (
            "independent isolated full-prefix recomputation; no adaptive sibling state"
        ),
        "tensor_roles": list(ANCHOR_SPINE_ROLES),
        "adaptive_node_budget_consumed": False,
        "adaptive_model_input": False,
        "labels_emitted": False,
        "realized_future_inputs": False,
        "source_ready_before_target_h1_execution": True,
    }
    if anchor != expected:
        raise AuditError("legacy HARP anchor-spine manifest contract mismatch")
    expected_hash = _canonical_hash(anchor)
    if manifest.get("legacy_harp_anchor_spine_manifest_hash") != expected_hash:
        raise AuditError("legacy HARP anchor-spine manifest hash mismatch")

    policy = manifest.get("capture_policy", {})
    policy_expectations = {
        "legacy_anchor_spine_separate": True,
        "legacy_anchor_required_depth": ANCHOR_SPINE_REQUIRED_DEPTH,
        "legacy_anchor_continues_through_eos": True,
        "legacy_anchor_consumes_adaptive_budget": False,
        "legacy_anchor_enters_adaptive_model_inputs": False,
        "legacy_anchor_labels_present": False,
    }
    for key, expected_value in policy_expectations.items():
        if policy.get(key) != expected_value:
            raise AuditError(f"capture policy has invalid anchor declaration {key}")

    counts = manifest.get("counts", {})
    count_expectations = {
        "legacy_anchor_spines": int(anchor_report["anchor_spines"]),
        "legacy_anchor_spine_nodes": int(anchor_report["anchor_nodes"]),
        "legacy_anchor_native_mtp_branch_calls": int(anchor_report["anchor_nodes"]),
    }
    for key, expected_value in count_expectations.items():
        if int(counts.get(key, -1)) != expected_value:
            raise AuditError(f"manifest anchor count mismatch for {key}")
    return {
        "schema": ANCHOR_SPINE_SCHEMA,
        "manifest_hash": expected_hash,
        "capture_counts_match": True,
    }


def _audit_anchor_spines(
    store: Store,
    *,
    adaptive_trees: dict[str, list[dict[str, Any]]],
    sequence_end: dict[Any, dict[str, Any]],
    target_event_by_position: dict[tuple[Any, int], dict[str, Any]],
    label_by_node: dict[Any, dict[str, Any]],
) -> dict[str, Any]:
    """Audit the label-free H1--H6 legacy-anchor channel.

    The anchor is deliberately not an adaptive-tree subset: it is a separate,
    fixed-depth, parent-coherent greedy spine used only by the frozen legacy
    HARP preprocessing bridge. In particular, adaptive EOS termination must
    never shorten this channel.
    """

    nodes = [
        row for row in store.rows if row.get("event") == ANCHOR_SPINE_NODE_EVENT
    ]
    if not nodes:
        raise AuditError("adaptive capture contains no HARP anchor-spine nodes")
    node_by_event = {int(row["event_id"]): row for row in nodes}
    if len(node_by_event) != len(nodes):
        raise AuditError("duplicate HARP anchor-spine event IDs")

    spines: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in nodes:
        spines[str(row["anchor_spine_id"])].append(row)
    ready_by_spine = _event_map(
        store, ANCHOR_SPINE_READY_EVENT, "anchor_spine_id"
    )
    if set(ready_by_spine) != set(spines):
        missing = sorted(set(spines) - set(ready_by_spine))
        orphan = sorted(set(ready_by_spine) - set(spines))
        raise AuditError(
            "HARP anchor ready/spine identity mismatch "
            f"(missing={missing}, orphan={orphan})"
        )

    spines_by_tree: dict[str, list[str]] = defaultdict(list)
    for spine_id, rows in spines.items():
        tree_ids = {str(row["tree_id"]) for row in rows}
        if len(tree_ids) != 1:
            raise AuditError(f"anchor spine {spine_id} crosses adaptive trees")
        spines_by_tree[next(iter(tree_ids))].append(spine_id)
    adaptive_tree_ids = set(adaptive_trees)
    if set(spines_by_tree) != adaptive_tree_ids:
        missing = sorted(adaptive_tree_ids - set(spines_by_tree))
        orphan = sorted(set(spines_by_tree) - adaptive_tree_ids)
        raise AuditError(
            "adaptive-tree/anchor-spine coverage mismatch "
            f"(missing={missing}, orphan={orphan})"
        )
    for tree_id, spine_ids in spines_by_tree.items():
        if len(spine_ids) != 1:
            raise AuditError(
                f"adaptive tree {tree_id} has {len(spine_ids)} anchor spines, expected one"
            )

    anchor_event_ids = set(node_by_event)
    labelled_anchor_ids = anchor_event_ids.intersection(int(v) for v in label_by_node)
    if labelled_anchor_ids:
        raise AuditError(
            "HARP anchor-spine nodes entered acceptance labels: "
            f"{sorted(labelled_anchor_ids)}"
        )
    for row in store.rows:
        if row.get("event") == "mtp_acceptance_label" and int(
            row["mtp_node_event_id"]
        ) in anchor_event_ids:
            raise AuditError(
                f"HARP anchor node {row['mtp_node_event_id']} has an acceptance label"
            )
        if row.get("event") == "target_layer":
            overlap = anchor_event_ids.intersection(
                int(value) for value in row.get("ready_mtp_node_event_ids", [])
            )
            if overlap:
                raise AuditError(
                    "HARP anchor-only nodes entered adaptive model-ready inputs: "
                    f"{sorted(overlap)}"
                )

    depth_counts: Counter[int] = Counter()
    seen_sources: set[tuple[str, int]] = set()
    for tree_id in sorted(adaptive_trees):
        spine_id = spines_by_tree[tree_id][0]
        rows = spines[spine_id]
        rows.sort(key=lambda row: int(row["anchor_local_index"]))
        if len(rows) != ANCHOR_SPINE_REQUIRED_DEPTH:
            raise AuditError(
                f"anchor spine {spine_id} has {len(rows)} nodes, expected H1--H6"
            )
        ready = ready_by_spine[spine_id]
        ready_event_id = int(ready["event_id"])
        event_ids = [int(row["event_id"]) for row in rows]
        if event_ids != sorted(event_ids) or any(
            event_id >= ready_event_id for event_id in event_ids
        ):
            raise AuditError(
                f"anchor spine {spine_id} is not parent-before-child then ready"
            )
        if [int(value) for value in ready["anchor_node_event_ids"]] != event_ids:
            raise AuditError(f"anchor spine {spine_id} ready-node order mismatch")
        if (
            ready.get("anchor_spine_schema") != ANCHOR_SPINE_SCHEMA
            or int(ready["exact_h1_root_event_id"]) != event_ids[0]
        ):
            raise AuditError(f"anchor spine {spine_id} ready schema/root mismatch")
        if int(ready["node_count"]) != ANCHOR_SPINE_REQUIRED_DEPTH or int(
            ready["required_depth"]
        ) != ANCHOR_SPINE_REQUIRED_DEPTH:
            raise AuditError(f"anchor spine {spine_id} ready depth/count mismatch")
        if int(ready.get("source_ready_event_order", -1)) != ready_event_id:
            raise AuditError(f"anchor spine {spine_id} has invalid ready event order")
        for key, expected in (
            ("synchronous_before_target_h1_execution", True),
            ("local_top1_parent_coherent", True),
            ("continues_through_eos", True),
            ("eos_termination_applies", False),
            ("consumes_adaptive_node_budget", False),
            ("adaptive_model_input", False),
            ("expansion_uses_realized_future", False),
            ("expansion_uses_acceptance", False),
            ("labels_emitted", False),
        ):
            _require_flag(ready, key, expected, subject=f"anchor spine {spine_id}")

        adaptive_rows = adaptive_trees[tree_id]
        adaptive_root = min(
            adaptive_rows, key=lambda row: int(row["node_local_index"])
        )
        sequence_id = str(adaptive_root["sequence_id"])
        base = int(adaptive_root["committed_prefix_position"])
        source_key = (sequence_id, base)
        if source_key in seen_sources:
            raise AuditError(f"multiple adaptive trees share anchor source {source_key}")
        seen_sources.add(source_key)
        tokens = [
            int(value)
            for value in sequence_end[sequence_id]["full_committed_token_ids"]
        ]
        if base + 1 >= len(tokens):
            raise AuditError(f"anchor spine {spine_id} lacks committed exact H1")
        expected_source_hash = prefix_hash(tokens[: base + 1])
        if (
            str(ready["tree_id"]) != tree_id
            or str(ready["sequence_id"]) != sequence_id
            or int(ready["committed_prefix_position"]) != base
            or ready["authoritative_prefix_hash"] != expected_source_hash
        ):
            raise AuditError(f"anchor spine {spine_id} ready/source mismatch")

        h1_target = target_event_by_position.get((sequence_id, base + 1))
        if h1_target is None or ready_event_id >= int(h1_target["event_id"]):
            raise AuditError(
                f"anchor spine {spine_id} was not ready before target H1 execution"
            )

        previous: dict[str, Any] | None = None
        previous_top_ids: np.ndarray | None = None
        previous_top_logps: np.ndarray | None = None
        previous_path_logps: tuple[float, ...] | None = None
        previous_monotonic_ns: int | None = None
        seen_path_ids: set[str] = set()
        for expected_local, row in enumerate(rows):
            event_id = int(row["event_id"])
            subject = f"anchor spine {spine_id} node {expected_local}"
            try:
                assert_causal_anchor_spine_node_record(row)
            except ValueError as exc:
                raise AuditError(str(exc)) from exc
            for key in row:
                if "label" in str(key).lower() and key != "label_records_present":
                    raise AuditError(f"{subject} carries label field {key!r}")
            if not row.get("record_valid") or not row.get("structural_validity"):
                raise AuditError(f"{subject} is invalid")
            if row.get("native_branch_execution_mode") != (
                "isolated_full_prefix_recomputation_no_shared_mutable_kv"
            ):
                raise AuditError(f"{subject} has unrecognized branch isolation mode")
            for key, expected in (
                ("anchor_only", True),
                ("adaptive_model_input", False),
                ("consumes_adaptive_node_budget", False),
                ("continues_through_eos", True),
                ("eos_termination_applies", False),
                ("label_records_present", False),
            ):
                _require_flag(row, key, expected, subject=subject)
            if int(row["required_anchor_depth"]) != ANCHOR_SPINE_REQUIRED_DEPTH:
                raise AuditError(f"{subject} has wrong required anchor depth")

            local = int(row["anchor_local_index"])
            depth = int(row["engine_draft_depth"])
            depth_counts[depth] += 1
            if local != expected_local or depth != expected_local + 1:
                raise AuditError(f"anchor spine {spine_id} is not exact H1--H6")
            if (
                str(row["tree_id"]) != tree_id
                or str(row["anchor_spine_id"]) != spine_id
                or str(row["sequence_id"]) != sequence_id
                or int(row["committed_prefix_position"]) != base
            ):
                raise AuditError(f"{subject} crosses its source identity")
            if int(row.get("source_ready_event_order", -1)) != event_id:
                raise AuditError(f"{subject} has invalid source-ready event order")
            if (
                not row.get("source_ready_timestamp_utc")
                or row.get("source_clock_domain_id")
                != "host_monotonic_after_native_mtp_materialization"
            ):
                raise AuditError(f"{subject} lacks source-readiness clock provenance")
            monotonic_ns = int(row.get("source_ready_monotonic_ns", -1))
            if monotonic_ns < 0 or (
                previous_monotonic_ns is not None
                and monotonic_ns < previous_monotonic_ns
            ):
                raise AuditError(f"{subject} has noncausal readiness time")
            previous_monotonic_ns = monotonic_ns
            if row["authoritative_prefix_hash"] != expected_source_hash:
                raise AuditError(f"{subject} has wrong authoritative prefix hash")

            path_tokens = tuple(int(value) for value in row["branch_path_token_ids"])
            path_logps = tuple(
                float(value)
                for value in row["branch_path_token_log_probabilities"]
            )
            if len(path_tokens) != depth or len(path_logps) != depth:
                raise AuditError(f"{subject} path/depth mismatch")
            path_id = str(row.get("path_id", ""))
            if not path_id or path_id in seen_path_ids:
                raise AuditError(f"{subject} has missing/duplicate path identity")
            seen_path_ids.add(path_id)
            expected_conditioning_hash = prefix_hash(
                tokens[: base + 1] + list(path_tokens)
            )
            if (
                row["mtp_conditioning_prefix_hash"] != expected_conditioning_hash
                or row["exact_prefix_hash"] != expected_conditioning_hash
            ):
                raise AuditError(f"{subject} conditioning-prefix hash mismatch")
            if (
                int(row["node_token_id"]) != path_tokens[-1]
                or int(row["gcrp_target_position"]) != base + depth
            ):
                raise AuditError(f"{subject} token/path/target-position mismatch")

            check_tensor_roles(store, event_id, ANCHOR_SPINE_REQUIRED)
            tensor_roles = store.tensors[event_id]
            if set(tensor_roles) != set(ANCHOR_SPINE_REQUIRED):
                raise AuditError(f"{subject} has tensors outside the anchor-only channel")
            for role in ANCHOR_SPINE_REQUIRED:
                descriptor = tensor_roles[role]
                if descriptor.get("entity_kind") != ANCHOR_SPINE_NODE_EVENT:
                    raise AuditError(f"{subject} role {role} crossed tensor channels")
                expected_position = base + depth + int(
                    role.startswith("harp_anchor_vocab_top64_")
                )
                if (
                    int(descriptor.get("parent_event_id", -1)) != event_id
                    or str(descriptor.get("sequence_id")) != sequence_id
                    or int(descriptor.get("engine_draft_depth", -1)) != depth
                    or int(descriptor.get("absolute_position", -1))
                    != expected_position
                    or descriptor.get("payload_file")
                    != ANCHOR_SPINE_PAYLOAD_FILES[role]
                ):
                    raise AuditError(f"{subject} role {role} has misaligned provenance")
                if not event_id < int(descriptor["event_id"]) < ready_event_id:
                    raise AuditError(f"{subject} role {role} was not source-ready")
            for role in (
                "harp_anchor_mtp_vocabulary_head_input",
                "harp_anchor_raw_mtp_router_logits",
            ):
                if tensor_roles[role].get("native_dtype") != "bf16":
                    raise AuditError(f"{subject} role {role} is not native BF16")
            if tensor_roles["harp_anchor_vocab_top64_token_ids"].get(
                "native_dtype"
            ) != "int32" or tensor_roles[
                "harp_anchor_vocab_top64_log_probabilities"
            ].get(
                "native_dtype"
            ) != "float32":
                raise AuditError(f"{subject} vocabulary tensor dtype mismatch")
            top_ids = store.tensor(
                event_id, "harp_anchor_vocab_top64_token_ids"
            ).astype(np.int64)
            top_logps = store.tensor(
                event_id, "harp_anchor_vocab_top64_log_probabilities"
            ).astype(np.float32)
            if (
                len(set(top_ids.tolist())) != 64
                or np.any(top_ids < 0)
                or np.any(np.diff(top_logps) > 1e-6)
            ):
                raise AuditError(f"{subject} has invalid sorted top64 vocabulary")
            if int(row["vocabulary_top_token_id"]) != int(top_ids[0]):
                raise AuditError(f"{subject} top vocabulary metadata mismatch")
            if (
                abs(
                    float(row["top_token_probability"])
                    - math.exp(float(top_logps[0]))
                )
                > 2e-6
            ):
                raise AuditError(f"{subject} top vocabulary probability mismatch")

            parent_event = row.get("parent_anchor_node_event_id")
            parent_local = row.get("parent_anchor_local_index")
            local_logp = float(row["node_local_token_log_probability"])
            local_probability = float(row["node_local_token_probability"])
            path_logp = float(row["path_log_probability"])
            if local == 0:
                if (
                    parent_event is not None
                    or parent_local is not None
                    or row.get("token_rank_under_parent") is not None
                    or row.get("exact_committed_h1_root") is not True
                    or path_tokens != (tokens[base + 1],)
                    or abs(local_logp) > 1e-12
                    or abs(local_probability - 1.0) > 1e-12
                    or abs(path_logp) > 1e-12
                    or len(path_logps) != 1
                    or abs(path_logps[0]) > 1e-12
                ):
                    raise AuditError(f"anchor spine {spine_id} has invalid exact-H1 root")
                if int(ready["exact_h1_token_id"]) != path_tokens[0]:
                    raise AuditError(f"anchor spine {spine_id} ready H1 mismatch")
                if int(adaptive_root["node_token_id"]) != path_tokens[0]:
                    raise AuditError(f"anchor spine {spine_id} differs from adaptive H1")
            else:
                assert previous is not None
                assert previous_top_ids is not None
                assert previous_top_logps is not None
                assert previous_path_logps is not None
                if (
                    parent_local != local - 1
                    or parent_event != int(previous["event_id"])
                    or int(row["token_rank_under_parent"]) != 0
                    or path_tokens[:-1]
                    != tuple(int(value) for value in previous["branch_path_token_ids"])
                    or int(previous_top_ids[0]) != path_tokens[-1]
                ):
                    raise AuditError(f"{subject} is not the parent's local top-1 child")
                if len(previous_path_logps) + 1 != len(path_logps) or any(
                    abs(lhs - rhs) > 2e-5
                    for lhs, rhs in zip(path_logps[:-1], previous_path_logps)
                ):
                    raise AuditError(f"{subject} log-probability path is incoherent")
                if (
                    abs(local_logp - float(previous_top_logps[0])) > 2e-5
                    or abs(path_logps[-1] - local_logp) > 2e-5
                    or abs(
                        path_logp
                        - (float(previous["path_log_probability"]) + local_logp)
                    )
                    > 2e-5
                ):
                    raise AuditError(f"{subject} local/path probability mismatch")
                if abs(local_probability - math.exp(local_logp)) > 2e-6:
                    raise AuditError(f"{subject} local probability is not exp(logp)")
            if abs(float(row["path_probability"]) - math.exp(path_logp)) > 2e-6:
                raise AuditError(f"{subject} path probability is not exp(logp)")

            previous = row
            previous_top_ids = top_ids
            previous_top_logps = top_logps
            previous_path_logps = path_logps

    expected_depth_counts = {
        depth: len(adaptive_trees)
        for depth in range(1, ANCHOR_SPINE_REQUIRED_DEPTH + 1)
    }
    if dict(depth_counts) != expected_depth_counts:
        raise AuditError(
            f"HARP anchor H1--H6 coverage mismatch: {dict(depth_counts)}"
        )
    return {
        "anchor_spines": len(spines),
        "anchor_nodes": len(nodes),
        "anchor_nodes_by_depth": dict(sorted(depth_counts.items())),
        "required_anchor_depth": ANCHOR_SPINE_REQUIRED_DEPTH,
        "local_top1_parent_coherent": True,
        "continues_through_eos": True,
        "labels_emitted": False,
        "consumes_adaptive_node_budget": False,
        "adaptive_model_input": False,
    }


def audit_adaptive_mtp(
    store: Store, *, maximum_node_budget: int, exact_node_budget: bool = False
) -> dict[str, Any]:
    nodes = [row for row in store.rows if row.get("event") == "mtp_node"]
    if not nodes:
        raise AuditError("adaptive capture contains no MTP nodes")
    node_by_event = {int(row["event_id"]): row for row in nodes}
    if len(node_by_event) != len(nodes):
        raise AuditError("duplicate adaptive MTP event IDs")
    label_by_node = _event_map(store, "mtp_acceptance_label", "mtp_node_event_id")
    ready_by_tree = _event_map(store, "mtp_tree_ready", "tree_id")
    resolved_by_tree = _event_map(store, "mtp_tree_resolved", "tree_id")
    sequence_end = _event_map(store, "sequence_end", "sequence_id")
    target_event_by_position = {
        (row["sequence_id"], int(row["absolute_sequence_position"])): row
        for row in store.rows
        if row.get("event") == "target_token"
    }
    trees: dict[str, list[dict[str, Any]]] = defaultdict(list)
    probability_errors: list[float] = []
    selected_weight_errors: list[float] = []
    native_selected_weight_errors: list[float] = []
    full_vocab_audits = 0

    for row in nodes:
        event_id = int(row["event_id"])
        try:
            assert_causal_mtp_node_record(row)
        except ValueError as exc:
            raise AuditError(str(exc)) from exc
        if not row.get("record_valid") or not row.get("structural_validity"):
            raise AuditError(f"invalid adaptive node {event_id}")
        if row.get("native_branch_execution_mode") != (
            "isolated_full_prefix_recomputation_no_shared_mutable_kv"
        ):
            raise AuditError(f"unrecognized branch isolation mode at {event_id}")
        tree_id = str(row["tree_id"])
        trees[tree_id].append(row)
        if event_id not in label_by_node:
            raise AuditError(f"missing label-only record for adaptive node {event_id}")
        label = label_by_node[event_id]
        if label.get("label_only") is not True or int(label["event_id"]) <= event_id:
            raise AuditError(f"acceptance is not a later label-only event for {event_id}")
        check_tensor_roles(store, event_id, ADAPTIVE_MTP_REQUIRED)
        probability_error, weight_error, selected_failure = _audit_route(
            store,
            event_id,
            logits_role="raw_mtp_router_logits",
            probs_role="full_mtp_router_probabilities",
            ids_role="mtp_selected_expert_ids",
            weights_role="mtp_selected_execution_weights",
        )
        probability_errors.append(probability_error)
        selected_weight_errors.append(weight_error)
        if selected_failure:
            raise AuditError(f"positive-gap native MTP route mismatch at {event_id}")
        weight_descriptor = store.tensors[event_id][
            "mtp_selected_execution_weights"
        ]
        logit_descriptor = store.tensors[event_id]["raw_mtp_router_logits"]
        if weight_descriptor["native_dtype"] != logit_descriptor["native_dtype"]:
            raise AuditError(
                f"MTP execution-weight dtype differs from router dtype at {event_id}"
            )
        if weight_descriptor["native_dtype"] != "bf16":
            raise AuditError(
                f"MTP execution weights are not native BF16 at {event_id}"
            )
        probabilities = store.tensor(
            event_id, "full_mtp_router_probabilities"
        ).astype(np.float32)
        selected_ids = store.tensor(
            event_id, "mtp_selected_expert_ids"
        ).astype(np.int64)
        stored_weights = store.tensor(
            event_id, "mtp_selected_execution_weights"
        ).astype(np.float32)
        expected_native_weights = torch.from_numpy(
            probabilities[selected_ids].copy()
        )
        expected_native_weights = (
            expected_native_weights / expected_native_weights.sum()
        ).to(torch.bfloat16).float().numpy()
        native_selected_weight_errors.append(
            float(np.max(np.abs(expected_native_weights - stored_weights)))
        )
        top_ids = store.tensor(event_id, "vocab_top64_token_ids").astype(np.int64)
        top_logps = store.tensor(
            event_id, "vocab_top64_log_probabilities"
        ).astype(np.float32)
        top_probs = store.tensor(
            event_id, "vocab_top64_probabilities"
        ).astype(np.float32)
        if len(set(top_ids.tolist())) != 64 or np.any(np.diff(top_logps) > 1e-6):
            raise AuditError(f"invalid sorted top64 vocabulary at {event_id}")
        if float(np.max(np.abs(np.exp(top_logps) - top_probs))) > 2e-6:
            raise AuditError(f"top64 probability/log-probability mismatch at {event_id}")
        if int(row["vocabulary_top_token_id"]) != int(top_ids[0]):
            raise AuditError(f"top vocabulary token metadata mismatch at {event_id}")
        if "full_vocabulary_logits_audit" in store.tensors[event_id]:
            full_vocab_audits += 1
            full = store.tensor(
                event_id, "full_vocabulary_logits_audit"
            ).astype(np.float32)
            stable = full - full.max()
            log_probs = stable - math.log(float(np.exp(stable).sum()))
            cutoff = float(np.partition(log_probs, -64)[-64])
            if float(log_probs[top_ids].min()) + 1e-7 < cutoff:
                raise AuditError(f"full-vocabulary top64 mismatch at {event_id}")
            if float(np.max(np.abs(log_probs[top_ids] - top_logps))) > 2e-5:
                raise AuditError(f"full-vocabulary log-prob mismatch at {event_id}")

    depth_counts: Counter[int] = Counter()
    valid_labels = 0
    accepted_labels = 0
    for tree_id, rows in trees.items():
        rows.sort(key=lambda row: int(row["node_local_index"]))
        if len(rows) > maximum_node_budget:
            raise AuditError(f"tree {tree_id} exceeds first-teacher budget: {len(rows)}")
        if exact_node_budget and len(rows) != maximum_node_budget:
            raise AuditError(
                f"tree {tree_id} has {len(rows)} nodes, expected exact "
                f"budget {maximum_node_budget}"
            )
        if tree_id not in ready_by_tree or tree_id not in resolved_by_tree:
            raise AuditError(f"tree {tree_id} lacks ready/resolved boundary events")
        ready = ready_by_tree[tree_id]
        if int(ready["node_count"]) != len(rows) or int(
            ready["maximum_node_budget"]
        ) != maximum_node_budget:
            raise AuditError(f"tree {tree_id} budget metadata mismatch")
        if ready.get("expansion_uses_realized_future") is not False:
            raise AuditError(f"tree {tree_id} declares realized-future expansion")
        if ready.get("expansion_uses_acceptance") is not False:
            raise AuditError(f"tree {tree_id} declares acceptance-driven expansion")
        event_ids = [int(row["event_id"]) for row in rows]
        if [int(v) for v in ready["mtp_node_event_ids"]] != event_ids:
            raise AuditError(f"tree {tree_id} ready-node order mismatch")

        sequence_id = str(rows[0]["sequence_id"])
        tokens = [int(v) for v in sequence_end[sequence_id]["full_committed_token_ids"]]
        base = int(rows[0]["committed_prefix_position"])
        expected_source_hash = prefix_hash(tokens[: base + 1])
        seen_node_ids: set[str] = set()
        seen_path_ids: set[str] = set()
        by_local: dict[int, dict[str, Any]] = {}
        for expected_local, row in enumerate(rows):
            event_id = int(row["event_id"])
            local = int(row["node_local_index"])
            depth = int(row["engine_draft_depth"])
            depth_counts[depth] += 1
            if local != expected_local:
                raise AuditError(f"tree {tree_id} local node IDs are not contiguous")
            if not 1 <= depth <= MAX_CAPTURE_DEPTH:
                raise AuditError(f"tree {tree_id} node {local} outside H1--H4")
            if row["authoritative_prefix_hash"] != expected_source_hash:
                raise AuditError(f"tree {tree_id} source-prefix hash mismatch")
            node_id, path_id = str(row["tree_node_id"]), str(row["path_id"])
            if node_id in seen_node_ids or path_id in seen_path_ids:
                raise AuditError(f"tree {tree_id} duplicates node/path identity")
            seen_node_ids.add(node_id)
            seen_path_ids.add(path_id)
            by_local[local] = row
            path_tokens = tuple(int(v) for v in row["branch_path_token_ids"])
            path_logps = tuple(
                float(v) for v in row["branch_path_token_log_probabilities"]
            )
            if len(path_tokens) != depth or len(path_logps) != depth:
                raise AuditError(f"tree {tree_id} node {local} path/depth mismatch")
            expected_conditioning_hash = prefix_hash(
                tokens[: base + 1] + list(path_tokens)
            )
            if row["mtp_conditioning_prefix_hash"] != expected_conditioning_hash:
                raise AuditError(f"tree {tree_id} node {local} conditioning hash mismatch")
            if int(row["node_token_id"]) != path_tokens[-1]:
                raise AuditError(f"tree {tree_id} node {local} token/path mismatch")
            if int(row["gcrp_target_position"]) != base + depth:
                raise AuditError(f"tree {tree_id} node {local} target-position mismatch")
            target_row = target_event_by_position.get((sequence_id, base + depth))
            if target_row is not None and event_id >= int(target_row["event_id"]):
                raise AuditError(f"tree {tree_id} node {local} was not source-ready causally")

            parent_local = row.get("parent_local_index")
            parent_event = row.get("parent_node_event_id")
            if local == 0:
                if (
                    depth != 1
                    or parent_local is not None
                    or parent_event is not None
                    or row.get("exact_committed_h1_root") is not True
                    or float(row["node_local_token_probability"]) != 1.0
                    or float(row["path_log_probability"]) != 0.0
                ):
                    raise AuditError(f"tree {tree_id} has invalid exact-H1 root")
                if base + 1 >= len(tokens) or int(row["node_token_id"]) != tokens[base + 1]:
                    raise AuditError(f"tree {tree_id} root is not committed x[t+1]")
            else:
                if parent_local is None or not 0 <= int(parent_local) < local:
                    raise AuditError(f"tree {tree_id} child precedes/misses its parent")
                parent = by_local[int(parent_local)]
                if int(parent["event_id"]) != int(parent_event):
                    raise AuditError(f"tree {tree_id} parent event identity mismatch")
                if depth != int(parent["engine_draft_depth"]) + 1:
                    raise AuditError(f"tree {tree_id} child depth mismatch")
                parent_path = tuple(int(v) for v in parent["branch_path_token_ids"])
                if path_tokens[:-1] != parent_path:
                    raise AuditError(f"tree {tree_id} child path does not extend parent")
                parent_top_ids = store.tensor(
                    int(parent["event_id"]), "vocab_top64_token_ids"
                ).astype(np.int64)
                parent_top_logps = store.tensor(
                    int(parent["event_id"]), "vocab_top64_log_probabilities"
                ).astype(np.float32)
                rank = int(row["token_rank_under_parent"])
                if rank < 0 or rank >= 64 or int(parent_top_ids[rank]) != path_tokens[-1]:
                    raise AuditError(f"tree {tree_id} child token/rank mismatch")
                local_logp = float(row["node_local_token_log_probability"])
                if abs(local_logp - float(parent_top_logps[rank])) > 2e-5:
                    raise AuditError(f"tree {tree_id} child local probability mismatch")
                expected_path_logp = float(parent["path_log_probability"]) + local_logp
                if abs(float(row["path_log_probability"]) - expected_path_logp) > 2e-5:
                    raise AuditError(f"tree {tree_id} path probability is not additive")

            label = label_by_node[event_id]
            start, stop = base + 1, base + depth + 1
            realized = tuple(tokens[start:stop])
            expected_valid = len(realized) == depth
            matched = 0
            for predicted, actual in zip(path_tokens, realized):
                if predicted != actual:
                    break
                matched += 1
            expected_accepted = expected_valid and matched == depth
            if bool(label["acceptance_label_valid"]) != expected_valid:
                raise AuditError(f"tree {tree_id} label validity mismatch")
            if expected_valid:
                valid_labels += 1
                accepted_labels += int(expected_accepted)
                if (
                    int(label["accepted_prefix_length"]) != matched
                    or bool(label["branch_path_accepted"]) != expected_accepted
                    or bool(label["draft_token_accepted"]) != expected_accepted
                ):
                    raise AuditError(f"tree {tree_id} acceptance label mismatch")
            elif any(
                label.get(key) is not None
                for key in (
                    "accepted_prefix_length",
                    "accepted_prefix_label",
                    "branch_path_accepted",
                    "draft_token_accepted",
                )
            ):
                raise AuditError(f"tree {tree_id} invalid tail carries a label value")

        _audit_eos_terminated_depths(
            rows,
            eos_token_id=int(ready["eos_token_id"]),
            tree_id=tree_id,
        )

    if full_vocab_audits < 1:
        raise AuditError("adaptive capture contains no full-vocabulary audit node")
    if max(probability_errors) > 2e-6 or max(selected_weight_errors) > 0.002:
        raise AuditError("adaptive MTP router reconstruction tolerance failed")
    if max(native_selected_weight_errors) > 0.001:
        raise AuditError("adaptive MTP native BF16 execution-weight audit failed")
    anchor_spine = _audit_anchor_spines(
        store,
        adaptive_trees=trees,
        sequence_end=sequence_end,
        target_event_by_position=target_event_by_position,
        label_by_node=label_by_node,
    )
    return {
        "adaptive_trees": len(trees),
        "mtp_nodes": len(nodes),
        "mtp_nodes_by_depth": dict(sorted(depth_counts.items())),
        "acceptance_labels": len(label_by_node),
        "acceptance_valid_labels": valid_labels,
        "accepted_branch_prefix_labels": accepted_labels,
        "full_vocabulary_audit_nodes": full_vocab_audits,
        "router_probability_abs_max": max(probability_errors),
        "selected_weight_abs_max": max(selected_weight_errors),
        "native_bf16_selected_weight_abs_max": max(native_selected_weight_errors),
        "maximum_nodes_per_tree": max(len(rows) for rows in trees.values()),
        "anchor_spine": anchor_spine,
    }


def audit(root: Path) -> dict[str, Any]:
    store = Store(root)
    try:
        manifest = store.manifest
        if manifest.get("training_started") is not False:
            raise AuditError("training flag is not false")
        if manifest.get("sealed_test_opened") is not False:
            raise AuditError("sealed-test flag is not false")
        header = store.rows[0]
        if (
            header.get("event") != "trace_header"
            or header.get("trace_header") is not True
            or header.get("schema") != SCHEMA
            or int(header.get("format_version", -1)) != FORMAT_VERSION
        ):
            raise AuditError("invalid adaptive trace header")
        if len(store.by_id) != len(store.rows) or set(store.by_id) != set(
            range(len(store.rows))
        ):
            raise AuditError("adaptive event IDs are duplicate or non-contiguous")
        policy = manifest.get("adaptive_expansion_policy", {})
        maximum_node_budget = int(
            policy.get("max_nodes_including_exact_h1_root", -1)
        )
        if not 4 <= maximum_node_budget <= 32:
            raise AuditError("H1--H4 first-teacher budget must be in [4, 32]")
        if policy.get("uses_realized_future_tokens") is not False:
            raise AuditError("policy declares realized-future access")
        if policy.get("uses_acceptance_labels") is not False:
            raise AuditError("policy declares acceptance-label access")
        checksum_count = check_checksums(root)
        anchor_checksum_count = _check_anchor_checksum_inventory(root)
        target = audit_target(store)
        mtp = audit_adaptive_mtp(
            store,
            maximum_node_budget=maximum_node_budget,
            exact_node_budget=(
                manifest.get("capture_policy", {}).get("exact_tree_node_budget")
                is True
            ),
        )
        anchor_provenance = _audit_anchor_manifest(
            manifest, mtp["anchor_spine"]
        )
        anchor_provenance["sidecar_checksums_verified"] = anchor_checksum_count
        sequences = audit_sequences(store)
        return {
            "schema": "gcrp2r_transformers_adaptive_mtp_tree_audit_v2",
            "passed": True,
            "run_id": manifest["run_id"],
            "checksum_files_verified": checksum_count,
            "target": target,
            "mtp": mtp,
            "anchor_provenance": anchor_provenance,
            "sequences": sequences,
            "gates": {
                "exact_committed_h1_root": True,
                "adaptive_h2_h4_native_paths": True,
                "maximum_32_nodes": True,
                "parent_before_child": True,
                "acceptance_label_only": True,
                "no_realized_future_expansion": True,
                "no_future_target_state_expansion": True,
                "isolated_sibling_execution": True,
                "explicit_legacy_anchor_h1_h6": True,
                "anchor_local_top1_parent_coherent": True,
                "anchor_continues_through_eos": True,
                "anchor_separate_from_adaptive_budget_and_inputs": True,
                "anchor_labels_emitted": False,
                "sealed_test_opened": False,
                "training_started": False,
            },
        }
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.capture / "CAPTURE_AUDIT_ADAPTIVE.json"
    try:
        report = audit(args.capture)
    except Exception as exc:
        report = {
            "schema": "gcrp2r_transformers_adaptive_mtp_tree_audit_v2",
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "training_started": False,
            "sealed_test_opened": False,
        }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
