#!/usr/bin/env python3
"""Audit that independent adaptive-16 equals canonical adaptive-32 prefix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FIELDS = (
    "node_local_index",
    "parent_local_index",
    "depth",
    "node_token_id",
    "token_rank_under_parent",
    "branch_path_token_ids",
    "branch_path_token_log_probabilities",
    "path_log_probability",
    "vocabulary_top_token_id",
)


def tree_rows(root: Path) -> dict[tuple[str, int], list[dict]]:
    result: dict[tuple[str, int], list[dict]] = {}
    for line in (root / "events.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "mtp_node" and row.get("record_valid") is True:
            key = (str(row["source_request_id"]) if "source_request_id" in row else str(row["request_id"]), int(row["committed_prefix_position"]))
            result.setdefault(key, []).append(row)
    for rows in result.values():
        rows.sort(key=lambda row: int(row["node_local_index"]))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive-32", type=Path, required=True)
    parser.add_argument("--adaptive-16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = tree_rows(args.adaptive_32)
    independent = tree_rows(args.adaptive_16)
    if set(canonical) != set(independent):
        raise ValueError("adaptive-16/32 source-position keys disagree")
    mismatches = []
    compared_nodes = 0
    for key in sorted(canonical):
        expected = canonical[key][:16]
        actual = independent[key]
        if len(actual) != len(expected):
            mismatches.append({"key": key, "reason": "node_count"})
            continue
        for left, right in zip(expected, actual):
            for field in FIELDS:
                if field.endswith("probability") or field.endswith("probabilities"):
                    lhs = left[field]
                    rhs = right[field]
                    if isinstance(lhs, list):
                        same = len(lhs) == len(rhs) and all(
                            math.isclose(a, b, abs_tol=1e-7, rel_tol=1e-7)
                            for a, b in zip(lhs, rhs)
                        )
                    else:
                        same = math.isclose(lhs, rhs, abs_tol=1e-7, rel_tol=1e-7)
                else:
                    same = left[field] == right[field]
                if not same:
                    mismatches.append(
                        {
                            "key": key,
                            "node": left["node_local_index"],
                            "field": field,
                        }
                    )
            compared_nodes += 1
    report = {
        "schema": "harp_rtt_adaptive_16_prefix_parity_v2",
        "passed": not mismatches,
        "source_positions": len(canonical),
        "compared_nodes": compared_nodes,
        "mismatches": mismatches[:100],
        "training_started": False,
        "sealed_test_opened": False,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite parity report {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
