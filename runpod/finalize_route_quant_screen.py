#!/usr/bin/env python3
"""Seal a complete RouteQuant screen after post-metric interruption."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.shadow_checkpoint import sha256_file


RESULT_SCHEMA = "harp_routequant_representative_screen_result_v1"
FINALIZER_SCHEMA = "harp_routequant_screen_finalization_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def finalize(run: Path, *, source_commit: str) -> dict[str, Any]:
    if len(source_commit) != 40:
        raise ValueError("finalizer source commit must be full length")
    if (run / "STAGE_RESULT.json").exists() or (run / "SHA256SUMS").exists():
        raise FileExistsError("RouteQuant screen is already sealed")
    manifest = json.loads((run / "run_manifest.json").read_text())
    if any(
        manifest.get(name) is not False
        for name in (
            "optimizer_constructed", "training_started",
            "development_opened_for_metrics", "formal_validation_opened",
            "calibration_opened", "sealed_test_opened",
        )
    ):
        raise PermissionError("RouteQuant interrupted run crossed a forbidden state")
    rows = [
        json.loads(line)
        for line in (run / "candidate_metrics.jsonl").read_text().splitlines()
        if line
    ]
    layers = tuple(int(layer) for layer in manifest["layers"])
    candidates = tuple(
        (int(value.split(":")[0]), value.split(":")[1])
        for value in manifest["candidates"]
    )
    fractions = tuple(float(value) for value in manifest["mixed_upgrade_fractions"])
    expected = len(layers) * (len(candidates) + len(fractions))
    if len(rows) != expected:
        raise ValueError(
            f"RouteQuant interrupted run has {len(rows)} rows, expected {expected}"
        )
    utility_files = sorted(run.glob("layer_*_train_utility.json"))
    if fractions and len(utility_files) != len(layers):
        raise ValueError("RouteQuant interrupted run lacks complete train utilities")
    aggregates = []
    for bits, method in candidates:
        selected = [
            row for row in rows
            if row.get("bits") == bits and row.get("scale_method") == method
        ]
        if {int(row["layer"]) for row in selected} != set(layers):
            raise ValueError("RouteQuant uniform candidate lacks a complete layer set")
        aggregates.append({
            "bits": bits,
            "scale_method": method,
            "mean_next_router_recall_at_8": sum(
                row["metrics"]["request_macro_next_router_recall_at_8"]
                for row in selected
            ) / len(selected),
            "mean_normalized_residual_rmse": sum(
                row["metrics"]["normalized_residual_rmse"] for row in selected
            ) / len(selected),
            "maximum_layer_recall_regression_from_exact": max(
                row["exact_baseline"]["request_macro_next_router_recall_at_8"]
                - row["metrics"]["request_macro_next_router_recall_at_8"]
                for row in selected
            ),
            "uniform_40_layer_projected_bytes": selected[0][
                "uniform_40_layer_projected_bytes"
            ],
            "uniform_40_layer_projected_gib": selected[0][
                "uniform_40_layer_projected_gib"
            ],
        })
    variant = (
        f"mixed_int{manifest['mixed_base_bits']}_int"
        f"{manifest['mixed_upgrade_bits']}"
    )
    mixed = []
    for fraction in fractions:
        selected = [
            row for row in rows
            if row.get("variant") == variant
            and float(row["upgrade_fraction"]) == fraction
        ]
        if {int(row["layer"]) for row in selected} != set(layers):
            raise ValueError("RouteQuant mixed candidate lacks a complete layer set")
        mixed.append({
            "variant": variant,
            "base_bits": int(manifest["mixed_base_bits"]),
            "upgrade_bits": int(manifest["mixed_upgrade_bits"]),
            "upgrade_fraction": fraction,
            "mean_next_router_recall_at_8": sum(
                row["metrics"]["request_macro_next_router_recall_at_8"]
                for row in selected
            ) / len(selected),
            "mean_normalized_residual_rmse": sum(
                row["metrics"]["normalized_residual_rmse"] for row in selected
            ) / len(selected),
            "maximum_layer_recall_regression_from_exact": max(
                row["exact_baseline"]["request_macro_next_router_recall_at_8"]
                - row["metrics"]["request_macro_next_router_recall_at_8"]
                for row in selected
            ),
            "uniform_40_layer_projected_bytes": selected[0][
                "uniform_40_layer_projected_bytes"
            ],
            "uniform_40_layer_projected_gib": selected[0][
                "uniform_40_layer_projected_gib"
            ],
        })
    result = {
        "schema": RESULT_SCHEMA,
        "layers": list(layers),
        "aggregates": aggregates,
        "mixed_aggregates": mixed,
        "candidate_rows": len(rows),
        "measurement_source_commit": manifest["source_commit"],
        "finalizer_source_commit": source_commit,
        "recovered_after_post_metric_interruption": True,
        "optimizer_constructed": False,
        "training_started": False,
        "development_opened_for_metrics": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(run / "STAGE_RESULT.json", result)
    write_json_exclusive(run / "FINALIZATION.json", {
        "schema": FINALIZER_SCHEMA,
        "measurement_source_commit": manifest["source_commit"],
        "finalizer_source_commit": source_commit,
        "candidate_metrics_sha256": sha256_file(run / "candidate_metrics.jsonl"),
        "candidate_rows": len(rows),
        "train_utility_files": len(utility_files),
    })
    files = sorted(
        path for path in run.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    with (run / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


def main() -> None:
    args = parse_args()
    print(json.dumps(
        finalize(args.run, source_commit=args.source_commit),
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
