#!/usr/bin/env python3
"""Apply the paired F1-Replay -> R2-QO RouteMTP information gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

from harp_rtt.training import sha256_file


SCHEMA = "harp_routemtp_representation_gate_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f1-run", type=Path, required=True)
    parser.add_argument("--r2-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_rows(root: Path, expected_stage: str) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    result = _json(root / "STAGE_RESULT.json")
    manifest = _json(root / "RUN_MANIFEST.json")
    if result.get("stage") != expected_stage or manifest.get("stage") != expected_stage:
        raise ValueError(f"RouteMTP comparison expected {expected_stage}")
    for key in (
        "official_tune_opened", "diagnostic_development_opened",
        "formal_validation_opened", "calibration_opened", "sealed_test_opened",
    ):
        if result.get(key) is not False or manifest.get(key) is not False:
            raise PermissionError(f"RouteMTP comparison input violates {key}")
    best_epoch = int(result["best_epoch"])
    selected: dict[str, Any] | None = None
    with (root / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if int(value["epoch"]) == best_epoch:
                selected = value
    if selected is None:
        raise ValueError("RouteMTP best epoch is absent from metrics")
    rows = selected["internal_dev"]["budgets"]["16"]["request_rows"]
    keyed = {str(row["request_id"]): row for row in rows}
    if len(keyed) != len(rows) or not keyed:
        raise ValueError("RouteMTP request metrics are empty or duplicated")
    return keyed, manifest


def paired_bootstrap(
    deltas: list[float], *, replicates: int = 1_000, seed: int = 42
) -> tuple[float, list[float]]:
    if not deltas:
        raise ValueError("paired RouteMTP bootstrap requires requests")
    point = sum(deltas) / len(deltas)
    generator = random.Random(seed); size = len(deltas)
    samples = sorted(
        sum(deltas[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(replicates)
    )
    return point, [samples[int(0.025 * replicates)], samples[int(0.975 * replicates) - 1]]


def main() -> None:
    args = parse_args()
    f1, f1_manifest = _best_rows(args.f1_run, "f1_replay")
    r2, r2_manifest = _best_rows(args.r2_run, "r2_qo")
    lineage_keys = (
        "source_commit", "hydration_source_commit", "hydration_manifest_sha256",
        "counterfactual_companion_manifest_sha256",
    )
    if any(f1_manifest.get(key) != r2_manifest.get(key) for key in lineage_keys):
        raise ValueError("RouteMTP F1/R2 lineage differs")
    if _json(args.f1_run / "INTERNAL_SPLIT.json") != _json(
        args.r2_run / "INTERNAL_SPLIT.json"
    ):
        raise ValueError("RouteMTP F1/R2 request split differs")
    if set(f1) != set(r2):
        raise ValueError("RouteMTP F1/R2 request rows differ")

    metrics: dict[str, Any] = {}
    for key in ("branch_recall_at_8", "branch_h4_recall_at_8"):
        deltas = [float(r2[request][key]) - float(f1[request][key]) for request in sorted(f1)]
        point, interval = paired_bootstrap(deltas)
        metrics[key] = {"paired_gain": point, "paired_bootstrap_95": interval}
    overall = metrics["branch_recall_at_8"]
    h4 = metrics["branch_h4_recall_at_8"]
    information_positive = (
        overall["paired_gain"] >= 0.03
        and overall["paired_bootstrap_95"][0] > 0.0
        and h4["paired_gain"] > 0.0
        and h4["paired_bootstrap_95"][0] > 0.0
    )
    result = {
        "schema": SCHEMA,
        "f1_checkpoint_sha256": _json(args.f1_run / "STAGE_RESULT.json")["checkpoint_sha256"],
        "r2_checkpoint_sha256": _json(args.r2_run / "STAGE_RESULT.json")["checkpoint_sha256"],
        "f1_run_manifest_sha256": sha256_file(args.f1_run / "RUN_MANIFEST.json"),
        "r2_run_manifest_sha256": sha256_file(args.r2_run / "RUN_MANIFEST.json"),
        "requests": len(f1),
        "metrics": metrics,
        "information_positive_gate_passed": information_positive,
        "breakthrough_gain_passed": overall["paired_gain"] >= 0.10,
        "bootstrap_replicates": 1_000,
        "bootstrap_seed": 42,
        "optimizer_constructed": False,
        "training_started": False,
        "official_tune_opened": False,
        "diagnostic_development_opened": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
