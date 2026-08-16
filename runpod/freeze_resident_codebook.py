#!/usr/bin/env python3
"""Aggregate, gate, and freeze forty Resident Functional Codebook shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.resident_codebook import validate_codebook_tables  # noqa: E402
from harp_rtt.shadow_checkpoint import sha256_file  # noqa: E402
from harp_rtt.shadow_expert import (  # noqa: E402
    resident_codebook_storage_bytes,
    resident_int4_storage_bytes,
)


FIT_SCHEMA = "harp_resident_functional_codebook_fit_v1"
LOCAL_SCHEMA = "harp_shadowroute_local_expert_layer_v1"
RESULT_SCHEMA = "harp_resident_functional_codebook_freeze_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-next-router-recall", type=float, default=0.978)
    parser.add_argument("--minimum-gap-recovery", type=float, default=0.80)
    parser.add_argument("--minimum-residual-reduction", type=float, default=0.60)
    parser.add_argument("--top2-minimum-gain", type=float, default=0.02)
    parser.add_argument("--maximum-shadow-gib", type=float, default=6.0)
    return parser.parse_args()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def layer_inputs(root: Path) -> list[tuple[Path, Path]]:
    result = []
    for layer in range(40):
        candidates = sorted(root.rglob(f"resident_codebook_fit_layer_{layer:02d}.pt"))
        reports = sorted(
            path for path in root.rglob("STAGE_RESULT.json")
            if path.parent in {candidate.parent for candidate in candidates}
        )
        if len(candidates) != 1 or len(reports) != 1:
            raise ValueError(
                f"expected one fit payload/report for layer {layer}, "
                f"found {len(candidates)}/{len(reports)}"
            )
        result.append((candidates[0], reports[0]))
    return result


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("resident_only", "nearest_one", "nearest_two", "exact_tail")
    output: dict[str, Any] = {}
    for name in names:
        recalls = [
            float(report["metrics"][name]["request_macro_next_router_recall"])
            for report in reports[:-1]
        ]
        residuals = [
            float(report["metrics"][name]["normalized_residual_mse"])
            for report in reports
        ]
        output[name] = {
            "next_router_recall": sum(recalls) / len(recalls),
            "normalized_residual_mse": sum(residuals) / len(residuals),
        }
    resident = output["resident_only"]["next_router_recall"]
    exact = output["exact_tail"]["next_router_recall"]
    for name in ("nearest_one", "nearest_two"):
        recall = output[name]["next_router_recall"]
        output[name]["gap_recovery"] = (
            (recall - resident) / max(exact - resident, 1e-12)
        )
        output[name]["residual_error_reduction"] = 1.0 - (
            output[name]["normalized_residual_mse"]
            / max(output["resident_only"]["normalized_residual_mse"], 1e-12)
        )
    return output


def phase_regressions(
    reports: list[dict[str, Any]], variant: str
) -> dict[str, float]:
    phases = {"early": range(0, 10), "middle_a": range(10, 20),
              "middle_b": range(20, 30), "late": range(30, 39)}
    result = {}
    for name, layers in phases.items():
        resident = sum(
            float(reports[layer]["metrics"]["resident_only"][
                "request_macro_next_router_recall"
            ])
            for layer in layers
        ) / len(layers)
        current = sum(
            float(reports[layer]["metrics"][variant][
                "request_macro_next_router_recall"
            ])
            for layer in layers
        ) / len(layers)
        result[name] = current - resident
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite codebook freeze {args.output}")
    if (
        len(args.source_commit) != 40
        or any(character not in "0123456789abcdef" for character in args.source_commit)
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    inputs = layer_inputs(args.fit_root)
    payloads = []
    reports = []
    target_sha = None
    total_residents = 0
    for layer, (payload_path, report_path) in enumerate(inputs):
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != FIT_SCHEMA
            or report.get("schema") != FIT_SCHEMA
            or int(payload.get("layer", -1)) != layer
            or int(report.get("layer", -1)) != layer
            or payload.get("source_commit") != args.source_commit
            or report.get("formal_validation_opened") is not False
            or report.get("calibration_opened") is not False
            or report.get("sealed_test_opened") is not False
        ):
            raise ValueError(f"resident codebook layer {layer} lineage changed")
        current_target = str(payload["target_checkpoint_index_sha256"])
        if target_sha is None:
            target_sha = current_target
        elif target_sha != current_target:
            raise ValueError("resident codebook layers use mixed target checkpoints")
        total_residents += int(payload["resident_count"])
        payloads.append(payload)
        reports.append(report)
    metrics = aggregate(reports)
    variant = (
        "nearest_two"
        if (
            metrics["nearest_two"]["next_router_recall"]
            - metrics["nearest_one"]["next_router_recall"]
        ) >= args.top2_minimum_gain
        else "nearest_one"
    )
    phase_gain = phase_regressions(reports, variant)
    selected = metrics[variant]
    passed = (
        selected["next_router_recall"] >= args.minimum_next_router_recall
        and selected["gap_recovery"] >= args.minimum_gap_recovery
        and selected["residual_error_reduction"]
        >= args.minimum_residual_reduction
        and min(phase_gain.values()) >= -0.01
    )
    logical_bytes = (
        resident_int4_storage_bytes(layers=1, residents=total_residents)
        + resident_codebook_storage_bytes()
    )
    if logical_bytes >= args.maximum_shadow_gib * 2**30:
        raise ValueError("resident codebook exceeds its exact logical size gate")
    args.output.mkdir(parents=True)
    manifest = {
        "schema": RESULT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "target_checkpoint_index_sha256": target_sha,
        "selected_variant": variant,
        "metrics": metrics,
        "phase_gain": phase_gain,
        "gates": {
            "minimum_next_router_recall": args.minimum_next_router_recall,
            "minimum_gap_recovery": args.minimum_gap_recovery,
            "minimum_residual_reduction": args.minimum_residual_reduction,
            "top2_minimum_gain": args.top2_minimum_gain,
        },
        "passed": passed,
        "total_residents": total_residents,
        "logical_shadow_bytes": logical_bytes,
        "logical_shadow_gib": logical_bytes / 2**30,
        "optimizer_constructed": False,
        "training_started": False,
        "target_expert_loads_permitted": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json(args.output / "run_manifest.json", manifest)
    layers_result = []
    if passed:
        for layer, payload in enumerate(payloads):
            table = payload[variant]
            validate_codebook_tables(
                payload["resident_expert_ids"],
                table["proxy_ids"],
                table["proxy_coefficients"],
                table["proxy_count"],
            )
            state = dict(payload["parent_model_state_dict"])
            state.update({
                "codebook_proxy_ids": table["proxy_ids"].to(torch.int16),
                "codebook_proxy_coefficients": table[
                    "proxy_coefficients"
                ].to(torch.bfloat16),
                "codebook_proxy_count": table["proxy_count"].to(torch.uint8),
            })
            directory = args.output / f"layer_{layer:02d}"
            directory.mkdir()
            checkpoint = (
                directory
                / f"shadow_resident_int4_codebook_layer_{layer:02d}.pt"
            )
            value = {
                "schema": LOCAL_SCHEMA,
                "source_commit": args.source_commit,
                "mode": "resident_int4_codebook",
                "layer": layer,
                "seed": 42,
                "model_state_dict": state,
                "resident_expert_ids": payload["resident_expert_ids"].long(),
                "resident_count": int(payload["resident_count"]),
                "codebook_variant": variant,
                "target_checkpoint_index_sha256": target_sha,
                "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
                "diagnostic_only": True,
                "closed_loop_authorized": False,
                "target_expert_loads_permitted": False,
                "formal_validation_opened": False,
                "calibration_opened": False,
                "sealed_test_opened": False,
            }
            with checkpoint.open("xb") as handle:
                torch.save(value, handle)
                handle.flush()
                os.fsync(handle.fileno())
            layer_manifest = {
                "schema": LOCAL_SCHEMA,
                "source_commit": args.source_commit,
                "mode": "resident_int4_codebook",
                "layer": layer,
                "checkpoint_sha256": sha256_file(checkpoint),
                "resident_count": int(payload["resident_count"]),
                "codebook_variant": variant,
                "optimizer_constructed": False,
                "training_started": False,
                "target_expert_loads_permitted": False,
                "formal_validation_opened": False,
                "calibration_opened": False,
                "sealed_test_opened": False,
            }
            write_json(directory / "run_manifest.json", layer_manifest)
            write_json(directory / "STAGE_RESULT.json", layer_manifest)
            with (directory / "SHA256SUMS").open("x", encoding="utf-8") as handle:
                for path in sorted(
                    item for item in directory.iterdir()
                    if item.is_file() and item.name != "SHA256SUMS"
                ):
                    handle.write(f"{sha256_file(path)}  {path.name}\n")
                handle.flush()
                os.fsync(handle.fileno())
            layers_result.append({
                "layer": layer,
                "checkpoint": str(checkpoint.relative_to(args.output)),
                "checkpoint_sha256": sha256_file(checkpoint),
                "resident_count": int(payload["resident_count"]),
            })
    result = dict(manifest)
    result["layers"] = layers_result
    result["closed_loop_evaluation_started"] = False
    write_json(args.output / "STAGE_RESULT.json", result)
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in sorted(
            item for item in args.output.iterdir()
            if item.is_file() and item.name != "SHA256SUMS"
        ):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
