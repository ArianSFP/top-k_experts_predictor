#!/usr/bin/env python3
"""Evaluate the frozen B3.1 route-source/posterior/candidate factorial.

The input is a label-bearing, outer-train diagnostic bundle exported by a
model forward. This script never opens a dataset and refuses bundles whose
provenance does not explicitly keep validation, calibration, and test sealed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.b31 import evaluate_factorial, evaluate_selected_factorial  # noqa: E402
from harp_rtt.training import sha256_file  # noqa: E402


BUNDLE_SCHEMA = "harp_rtt_b31_factor_bundle_v1"
REPORT_SCHEMA = "harp_rtt_b31_factorial_report_v1"


def _tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"B3.1 bundle field {name} must be a tensor")
    return value.cpu()


def load_bundle(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("B3.1 factor bundle schema mismatch")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("B3.1 factor bundle lacks provenance")
    if provenance.get("outer_split") != "train":
        raise PermissionError("B3.1 diagnostics are restricted to outer-train")
    for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if provenance.get(key) is not False:
            raise PermissionError(f"B3.1 factor bundle violates {key}")
    branch_scores = value.get("branch_scores")
    branch_selected_ids = value.get("branch_selected_ids")
    posteriors = value.get("posteriors")
    has_scores = isinstance(branch_scores, Mapping) and bool(branch_scores)
    has_ids = isinstance(branch_selected_ids, Mapping) and bool(branch_selected_ids)
    if has_scores == has_ids:
        raise ValueError("B3.1 bundle must contain exactly one route-source representation")
    if not isinstance(posteriors, Mapping) or not posteriors:
        raise ValueError("B3.1 factor bundle has no posterior conditions")
    return value


def evaluate_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    raw_scores = value.get("branch_scores")
    raw_ids = value.get("branch_selected_ids")
    branch_scores = None if raw_scores is None else {
        str(name): _tensor(tensor, f"branch_scores.{name}")
        for name, tensor in raw_scores.items()
    }
    branch_selected_ids = None if raw_ids is None else {
        str(name): _tensor(tensor, f"branch_selected_ids.{name}")
        for name, tensor in raw_ids.items()
    }
    posteriors: dict[str, tuple[Tensor, Tensor]] = {}
    for name, condition in value["posteriors"].items():
        if not isinstance(condition, Mapping):
            raise TypeError(f"posterior condition {name} must be a mapping")
        posteriors[str(name)] = (
            _tensor(condition.get("captured"), f"posteriors.{name}.captured"),
            _tensor(condition.get("other"), f"posteriors.{name}.other"),
        )
    common = {
        "anchor_scores": _tensor(value.get("anchor_scores"), "anchor_scores"),
        "anchor_marginals": _tensor(value.get("anchor_marginals"), "anchor_marginals"),
        "target_ids": _tensor(value.get("target_ids"), "target_ids"),
        "posteriors": posteriors,
        "branch_mask": _tensor(value.get("branch_mask"), "branch_mask"),
        "width": int(value.get("candidate_width", 64)),
        "strata_masks": (
            None
            if value.get("strata_masks") is None
            else {
                str(name): _tensor(mask, f"strata_masks.{name}")
                for name, mask in value["strata_masks"].items()
            }
        ),
    }
    if branch_scores is not None:
        report = evaluate_factorial(
            **common, branch_scores=branch_scores, exact_k=int(value.get("exact_k", 8))
        )
        route_sources = sorted(branch_scores)
    else:
        assert branch_selected_ids is not None
        report = evaluate_selected_factorial(
            **common, branch_selected_ids=branch_selected_ids
        )
        route_sources = sorted(branch_selected_ids)
    return {
        "schema": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": dict(value["provenance"]),
        "factorial": report,
        "conditions": len(report),
        "route_sources": route_sources,
        "posterior_conditions": sorted(posteriors),
        "training_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite B3.1 report {args.output}")
    value = load_bundle(args.bundle)
    report = evaluate_bundle(value)
    report["bundle_sha256"] = sha256_file(args.bundle)
    write_json_exclusive(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
