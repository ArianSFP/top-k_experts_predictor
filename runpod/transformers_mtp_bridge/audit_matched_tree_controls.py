#!/usr/bin/env python3
"""Blocking auditor for HARP-RTT v2 causal matched tree controls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_transformers_segment import AuditError, check_checksums  # noqa: E402
from matched_tree_controls import (  # noqa: E402
    CONTROL_NAMES,
    CONTROL_READY_EVENT,
    CONTROL_RESOLVED_EVENT,
    CONTROL_SCHEMA,
    assert_ready_record,
    assert_resolved_record,
)


EXPECTED_COUNTS = {
    "fixed16": 16,
    "adaptive16": 16,
    "fixed32": 32,
    "adaptive32": 32,
}
GREEDY_COUNT_RANGE = (1, 4)
EXPECTED_DEPTHS = {
    "greedy": [1, 1, 1, 1],
    "fixed16": [1, 5, 5, 5],
    "fixed32": [1, 11, 10, 10],
}
STRUCTURAL_KEYS = (
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


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _structure(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: node[key] for key in STRUCTURAL_KEYS} for node in nodes]


def _load_control_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = root / "events.jsonl"
    if not path.is_file():
        raise AuditError(f"missing event trace {path}")
    ready: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("event") == CONTROL_READY_EVENT:
                ready.append(row)
            elif row.get("event") == CONTROL_RESOLVED_EVENT:
                resolved.append(row)
    return ready, resolved


def _audit_fixed_beam(ready: dict[str, Any]) -> None:
    name = str(ready["control_name"])
    nodes = ready["nodes"]
    expected_depths = EXPECTED_DEPTHS[name]
    by_depth = {
        depth: [node for node in nodes if int(node["depth"]) == depth]
        for depth in range(1, 5)
    }
    if [len(by_depth[depth]) for depth in range(1, 5)] != expected_depths:
        raise AuditError(f"{name} depth widths do not match the frozen control")

    for depth in range(2, 5):
        proposals: list[tuple[float, tuple[int, ...], int, int, int]] = []
        for parent in by_depth[depth - 1]:
            parent_path = tuple(int(value) for value in parent["token_path_ids"])
            parent_logp = float(parent["path_log_probability"])
            for rank, (token, logp) in enumerate(
                zip(
                    parent["vocab_top64_token_ids"],
                    parent["vocab_top64_log_probabilities"],
                    strict=True,
                )
            ):
                token = int(token)
                proposals.append(
                    (
                        -(parent_logp + float(logp)),
                        parent_path + (token,),
                        int(parent["local_index"]),
                        rank,
                        token,
                    )
                )
        proposals.sort()
        selected = proposals[: expected_depths[depth - 1]]
        actual = [
            (
                -float(node["path_log_probability"]),
                tuple(int(value) for value in node["token_path_ids"]),
                int(node["parent_local_index"]),
                int(node["token_rank_under_parent"]),
                int(node["token_id"]),
            )
            for node in by_depth[depth]
        ]
        for expected, observed in zip(selected, actual, strict=True):
            if expected[1:] != observed[1:] or not math.isclose(
                expected[0], observed[0], rel_tol=0.0, abs_tol=2e-7
            ):
                raise AuditError(
                    f"{name} H{depth} is not the stable cumulative-probability beam"
                )


def audit(root: Path) -> dict[str, Any]:
    check_checksums(root)
    manifest = json.loads((root / "run_manifest.json").read_text())
    control_manifest = manifest.get("matched_tree_control_manifest")
    if not isinstance(control_manifest, dict) or not control_manifest.get("enabled"):
        raise AuditError("capture did not enable matched B1 controls")
    if control_manifest.get("schema") != CONTROL_SCHEMA:
        raise AuditError("matched-control manifest schema mismatch")
    if manifest.get("matched_tree_control_manifest_hash") != _canonical_hash(
        control_manifest
    ):
        raise AuditError("matched-control manifest hash mismatch")
    capture_policy = manifest.get("capture_policy", {})
    if capture_policy.get("exact_tree_node_budget") is not True:
        raise AuditError("capture does not declare an exact adaptive node budget")
    if capture_policy.get("matched_controls_emitted") is not True:
        raise AuditError("capture policy does not declare matched controls")

    ready_rows, resolved_rows = _load_control_rows(root)
    ready_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    resolved_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ready_rows:
        try:
            assert_ready_record(row)
        except ValueError as error:
            raise AuditError(str(error)) from error
        key = (str(row["canonical_adaptive_tree_id"]), str(row["control_name"]))
        if key in ready_by_key:
            raise AuditError(f"duplicate matched-control ready key {key}")
        ready_by_key[key] = row
    for row in resolved_rows:
        try:
            assert_resolved_record(row)
        except ValueError as error:
            raise AuditError(str(error)) from error
        key = (str(row["canonical_adaptive_tree_id"]), str(row["control_name"]))
        if key in resolved_by_key:
            raise AuditError(f"duplicate matched-control resolved key {key}")
        resolved_by_key[key] = row
    if set(ready_by_key) != set(resolved_by_key):
        raise AuditError("ready/resolved matched-control keys differ")

    by_tree: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    occurrence: dict[str, Counter[int]] = {
        name: Counter() for name in CONTROL_NAMES
    }
    for key, ready in ready_by_key.items():
        tree_id, name = key
        resolved = resolved_by_key[key]
        if int(resolved["control_ready_event_id"]) != int(ready["event_id"]):
            raise AuditError(f"resolved control {key} links the wrong ready event")
        if int(resolved["event_id"]) <= int(ready["event_id"]):
            raise AuditError(f"resolved control {key} precedes its causal ready record")
        if resolved["structural_hash"] != ready["structural_hash"]:
            raise AuditError(f"resolved control {key} changed tree structure")
        node_count = int(ready["node_count"])
        if name == "greedy":
            if not GREEDY_COUNT_RANGE[0] <= node_count <= GREEDY_COUNT_RANGE[1]:
                raise AuditError("greedy control has an invalid terminal length")
        elif node_count != EXPECTED_COUNTS[name]:
            raise AuditError(f"{name} did not consume its exact node budget")
        if resolved.get("all_labels_valid") is not True:
            raise AuditError(f"{name} has incomplete H1--H4 factual labels")
        if "labels" in ready or "path_occurrence_by_horizon" in ready:
            raise AuditError(f"causal ready record {key} contains target labels")
        for depth in range(1, 5):
            occurrence[name][depth] += int(
                resolved["path_occurrence_by_horizon"][str(depth)]
            )
        by_tree[tree_id][name] = ready
        if name in {"fixed16", "fixed32"}:
            _audit_fixed_beam(ready)

    expected_names = set(CONTROL_NAMES)
    for tree_id, views in by_tree.items():
        if set(views) != expected_names:
            raise AuditError(f"tree {tree_id} does not contain all matched controls")
        adaptive32 = views["adaptive32"]["nodes"]
        adaptive16 = views["adaptive16"]["nodes"]
        greedy = views["greedy"]["nodes"]
        if _structure(adaptive32[:16]) != _structure(adaptive16):
            raise AuditError(f"tree {tree_id} adaptive16 is not adaptive32[:16]")
        if _structure(adaptive32[: len(greedy)]) != _structure(greedy):
            raise AuditError(f"tree {tree_id} greedy is not an adaptive32 prefix")
        if [int(node["depth"]) for node in greedy] != list(
            range(1, len(greedy) + 1)
        ) or any(int(node["token_rank_under_parent"]) != 0 for node in greedy):
            raise AuditError(f"tree {tree_id} greedy is not a rank-zero spine")
        if views["adaptive16"].get("independent_adaptive16_exact_match") is not True:
            raise AuditError(f"tree {tree_id} lacks independent adaptive16 parity")
        if (
            views["adaptive16"].get("independent_adaptive16_structural_hash")
            != views["adaptive16"]["structural_hash"]
        ):
            raise AuditError(f"tree {tree_id} independent adaptive16 hash differs")

    counts = manifest.get("counts", {})
    expected_tree_counts = {name: len(by_tree) for name in CONTROL_NAMES}
    expected_node_counts = {
        name: len(by_tree) * count for name, count in EXPECTED_COUNTS.items()
    }
    expected_node_counts["greedy"] = sum(
        int(ready["node_count"])
        for (_tree_id, name), ready in ready_by_key.items()
        if name == "greedy"
    )
    if counts.get("matched_control_trees") != expected_tree_counts:
        raise AuditError("manifest matched-control tree counts differ")
    if counts.get("matched_control_nodes") != expected_node_counts:
        raise AuditError("manifest matched-control node counts differ")
    native_calls = counts.get("matched_control_native_mtp_calls", {})
    expected_calls = {
        "adaptive16_independent": len(by_tree) * 16,
        "fixed16": len(by_tree) * 16,
        "fixed32": len(by_tree) * 32,
    }
    if native_calls != expected_calls:
        raise AuditError("manifest matched-control native MTP call counts differ")

    tree_count = len(by_tree)
    report = {
        "schema": "harp_rtt_matched_tree_controls_audit_v1",
        "passed": True,
        "capture_root": str(root),
        "tree_count": tree_count,
        "ready_record_count": len(ready_rows),
        "resolved_record_count": len(resolved_rows),
        "node_count_contract": {
            "greedy": {"minimum": 1, "maximum": 4, "terminal_shortened": True},
            **{name: {"exact": count} for name, count in EXPECTED_COUNTS.items()},
        },
        "adaptive16_prefix_parity": True,
        "adaptive16_independent_parity": True,
        "fixed_beam_selection_reconstructed": True,
        "ready_records_label_free": True,
        "resolved_records_label_only": True,
        "path_occurrence": {
            name: {
                f"H{depth}": occurrence[name][depth] / tree_count
                for depth in range(1, 5)
            }
            for name in CONTROL_NAMES
        },
        "sealed_test_opened": False,
        "training_started": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.capture)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        if args.report.exists():
            raise FileExistsError(f"refusing to overwrite {args.report}")
        args.report.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
