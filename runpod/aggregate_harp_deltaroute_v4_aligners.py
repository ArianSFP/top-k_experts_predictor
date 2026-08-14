#!/usr/bin/env python3
"""Aggregate the two-seed M0/M1 factual-aligner experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.deltaroute_metrics import paired_h2_h4_request_bootstrap  # noqa: E402
from harp_rtt.training import sha256_file  # noqa: E402
from runpod.train_harp_deltaroute_v4 import RESULT_SCHEMA  # noqa: E402


SCHEMA = "harp_deltaroute_v4_aligner_aggregate_v1"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _average_rows(row_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for rows in row_sets:
        seen: set[tuple[str, int]] = set()
        for row in rows:
            key = (str(row["request_id"]), int(row["horizon"]))
            if key in seen:
                raise ValueError("aligner seed contains duplicate request/horizon")
            seen.add(key)
            values[key].append(float(row["slot_recall_at_8"]))
    if not values or any(len(items) != len(row_sets) for items in values.values()):
        raise ValueError("aligner seed prediction rows are incomplete or differ")
    return [
        {"request_id": request, "horizon": horizon,
         "slot_recall_at_8": sum(items) / len(items)}
        for (request, horizon), items in sorted(values.items())
    ]


def aggregate(stage_runs: Mapping[str, list[Path]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for stage in ("align_m0", "align_m1"):
        paths = stage_runs[stage]
        if len(paths) != 2:
            raise ValueError(f"{stage} requires exactly two seed runs")
        records: list[dict[str, Any]] = []
        prediction_sets: list[list[dict[str, Any]]] = []
        parent_sets: list[list[dict[str, Any]]] = []
        for path in paths:
            result_path = path / "STAGE_RESULT.json"
            value = json.loads(result_path.read_text(encoding="utf-8"))
            if value.get("schema") != RESULT_SCHEMA or value.get("stage") != stage:
                raise ValueError(f"{stage} result schema/stage mismatch")
            if value.get("seed") not in (42, 43):
                raise ValueError(f"{stage} contains an undeclared seed")
            for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
                if value.get(key) is not False:
                    raise PermissionError(f"{stage} result violates {key}")
            records.append(value)
            prediction_sets.append(_read_rows(path / "development_request_predictions.jsonl"))
            parent_sets.append(_read_rows(path / "development_parent_predictions.jsonl"))
        if {int(value["seed"]) for value in records} != {42, 43}:
            raise ValueError(f"{stage} does not contain seeds 42 and 43")
        averaged = _average_rows(prediction_sets)
        averaged_parent = _average_rows(parent_sets)
        bootstrap = paired_h2_h4_request_bootstrap(
            averaged, averaged_parent, replicates=1_000, seed=42
        )
        positive_both = all(
            value["single_seed_continuation"]["positive_h2_h4_point_gain"]
            for value in records
        )
        no_h4 = all(
            value["single_seed_continuation"]["no_h4_regression"]
            for value in records
        )
        no_mismatch = all(
            value["single_seed_continuation"]["no_h4_mismatch_regression"]
            for value in records
        )
        metrics = [value["development"] for value in records]
        results[stage] = {
            "runs": [
                {"path": str(path), "result_sha256": sha256_file(path / "STAGE_RESULT.json"),
                 "checkpoint_sha256": record["checkpoint_sha256"], "seed": record["seed"]}
                for path, record in zip(paths, records, strict=True)
            ],
            "mean_seed_h2_h4_c64": sum(float(item["candidate_coverage_h2_h4"]) for item in metrics) / 2,
            "mean_seed_h4_c64": sum(float(item["candidate_coverage_h4"]) for item in metrics) / 2,
            "mean_seed_h4_mismatch_c64": sum(float(item["candidate_coverage_h4_mismatch"]) for item in metrics) / 2,
            "paired_mean_seed_bootstrap": bootstrap,
            "passed": bool(
                positive_both and bootstrap["lower_bound_positive"]
                and no_h4 and no_mismatch
            ),
            "positive_both_seeds": positive_both,
            "no_h4_regression": no_h4,
            "no_h4_mismatch_regression": no_mismatch,
        }
    passing = [stage for stage in ("align_m0", "align_m1") if results[stage]["passed"]]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda stage: (
                results[stage]["mean_seed_h2_h4_c64"],
                results[stage]["mean_seed_h4_mismatch_c64"],
                -int(stage == "align_m1"),
            ),
        )
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stages": results,
        "selected_aligner": selected,
        "aligner_promoted": selected is not None,
        "route_dynamics_remains_authorized_if_aligner_fails": True,
        "training_started_by_aggregate": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ("m0", "m1"):
        parser.add_argument(f"--{stage}-seed42", type=Path, required=True)
        parser.add_argument(f"--{stage}-seed43", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite aligner aggregate {args.output}")
    report = aggregate({
        "align_m0": [args.m0_seed42, args.m0_seed43],
        "align_m1": [args.m1_seed42, args.m1_seed43],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
