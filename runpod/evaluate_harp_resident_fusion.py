#!/usr/bin/env python3
"""Request-disjoint confirmation of a frozen HARP rescue checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.resident_fusion import ExpertHarpRescue, ScalarHarpRescue  # noqa: E402
from harp_rtt.shadow_checkpoint import sha256_file  # noqa: E402
from runpod.train_harp_resident_fusion import (  # noqa: E402
    FEATURE_WIDTH, FusionRows, SCHEMA as TRAINING_SCHEMA, evaluate, load_pair,
    paired_gain_bootstrap, write_json_exclusive, write_jsonl_exclusive,
)


SCHEMA = "harp_resident_harp_fusion_confirmation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--ceiling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output directory {args.output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != TRAINING_SCHEMA:
        raise ValueError("fusion confirmation checkpoint schema mismatch")
    if checkpoint.get("source_commit") != args.source_commit:
        raise ValueError("fusion confirmation source lineage changed")
    variant = str(checkpoint.get("variant"))
    if variant == "f0":
        model: torch.nn.Module = ScalarHarpRescue()
    elif variant in {"f1_cold", "f1_all"}:
        model = ExpertHarpRescue(FEATURE_WIDTH)
    else:
        raise ValueError("fusion confirmation variant is invalid")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("fusion confirmation checkpoint lacks model state")
    model.load_state_dict(state, strict=True)
    sidecar, ceiling = load_pair(args.sidecar, args.ceiling)
    if sidecar["provenance"]["resident_policy"] != checkpoint.get("resident_policy"):
        raise ValueError("fusion confirmation resident policy changed")
    data = FusionRows(sidecar, ceiling)
    prior = set(checkpoint.get("fit_request_ids", [])) | set(
        checkpoint.get("tune_request_ids", [])
    )
    overlap = prior & data.request_ids
    if overlap:
        raise PermissionError(
            f"fusion confirmation overlaps {len(overlap)} fitting/tuning requests"
        )
    device = torch.device(args.device); model.to(device)
    metrics, rows = evaluate(model, data, variant, device)
    bootstrap = paired_gain_bootstrap(rows, replicates=1_000, seed=42)
    horizon_gains = [
        metrics[f"fusion_recall_h{h}"] - metrics[f"baseline_recall_h{h}"]
        for h in (1, 2, 3, 4)
    ]
    promotion = bool(
        metrics["fusion_recall_h1_h4"] >= 0.90
        and metrics["gain_h1_h4"] >= 0.07
        and metrics["fusion_recall_h4"] >= 0.85
        and metrics["gain_h4"] >= 0.06
        and bootstrap["ci95_low"] > 0.0
        and min(horizon_gains) >= -0.005
    )
    args.output.mkdir(parents=True)
    write_json_exclusive(args.output / "run_manifest.json", {
        "schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "prediction_sidecar_sha256": sha256_file(args.sidecar),
        "ceiling_bundle_sha256": sha256_file(args.ceiling),
        "resident_policy": checkpoint["resident_policy"],
        "request_count": len(data.request_ids), "optimizer_constructed": False,
        "training_started": False, "formal_validation_opened": False,
        "calibration_opened": False, "sealed_test_opened": False,
    })
    write_jsonl_exclusive(args.output / "request_predictions.jsonl", rows)
    write_json_exclusive(args.output / "CONFIRMATION_RESULT.json", {
        "schema": SCHEMA, "variant": variant, "metrics": metrics,
        "paired_request_bootstrap": bootstrap,
        "promotion_passed": promotion,
        "requirements": {
            "recall_h1_h4": 0.90, "gain_h1_h4": 0.07,
            "recall_h4": 0.85, "gain_h4": 0.06,
            "paired_ci95_low": ">0", "maximum_horizon_regression": 0.005,
        },
    })
    paths = sorted(path for path in args.output.iterdir() if path.name != "SHA256SUMS")
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"metrics": metrics, "promotion_passed": promotion}, sort_keys=True))


if __name__ == "__main__":
    main()
