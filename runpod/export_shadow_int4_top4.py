#!/usr/bin/env python3
"""Quantize complete target experts into a deployable INT4 top-four bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.shadow_bundle import LOCAL_SCHEMA  # noqa: E402
from harp_rtt.shadow_checkpoint import (  # noqa: E402
    IndexedCheckpoint,
    load_target_layer_experts,
    sha256_file,
)
from harp_rtt.shadow_expert import (  # noqa: E402
    dequantize_groupwise_int4,
    quantize_groupwise_int4,
)


SCHEMA = "harp_shadowroute_int4_top4_export_v1"
RESULT_SCHEMA = "harp_shadowroute_int4_top4_export_result_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--group-size", type=int, choices=(32, 64), default=64)
    parser.add_argument("--active-slots", type=int, choices=(4,), default=4)
    parser.add_argument("--expert-chunk", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def quantize_expert_chunks(
    weight: Tensor,
    *,
    group_size: int,
    chunk: int,
) -> tuple[Tensor, Tensor, float]:
    if chunk < 1:
        raise ValueError("expert chunk must be positive")
    packed = torch.empty(
        *weight.shape[:-1], weight.shape[-1] // 2,
        dtype=torch.uint8,
        device="cpu",
    )
    scales = torch.empty(
        *weight.shape[:-1], weight.shape[-1] // group_size,
        dtype=torch.bfloat16,
        device="cpu",
    )
    squared_error = squared_target = 0.0
    for start in range(0, weight.shape[0], chunk):
        stop = min(start + chunk, weight.shape[0])
        current = weight[start:stop]
        current_packed, current_scales = quantize_groupwise_int4(
            current, group_size=group_size
        )
        reconstructed = dequantize_groupwise_int4(
            current_packed,
            current_scales,
            group_size=group_size,
            dtype=torch.float32,
        )
        squared_error += float((reconstructed - current.float()).square().sum())
        squared_target += float(current.float().square().sum())
        packed[start:stop].copy_(current_packed.cpu())
        scales[start:stop].copy_(current_scales.cpu())
        del current_packed, current_scales, reconstructed
    return packed, scales, math.sqrt(squared_error / max(squared_target, 1e-30))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite INT4 export {args.output}")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    args.output.mkdir(parents=True)
    checkpoint = IndexedCheckpoint(args.model)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "group_size": args.group_size,
        "active_slots": args.active_slots,
        "quantization": "symmetric_signed_int4_per_output_group",
        "optimizer_constructed": False,
        "training_started": False,
        "diagnostic_only": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)
    layer_metrics = []
    for layer in range(40):
        gate_up, down = load_target_layer_experts(
            checkpoint, layer, device=args.device, dtype=torch.bfloat16
        )
        gate_packed, gate_scales, gate_nrmse = quantize_expert_chunks(
            gate_up,
            group_size=args.group_size,
            chunk=args.expert_chunk,
        )
        down_packed, down_scales, down_nrmse = quantize_expert_chunks(
            down,
            group_size=args.group_size,
            chunk=args.expert_chunk,
        )
        directory = args.output / f"layer_{layer:02d}"
        directory.mkdir()
        path = directory / f"shadow_int4_top4_layer_{layer:02d}.pt"
        value = {
            "schema": LOCAL_SCHEMA,
            "mode": "int4_top4",
            "layer": layer,
            "source_commit": args.source_commit,
            "target_checkpoint_index_sha256": checkpoint.index_sha256,
            "model_state_dict": {
                "gate_up_packed": gate_packed,
                "gate_up_scales": gate_scales,
                "down_packed": down_packed,
                "down_scales": down_scales,
            },
            "group_size": args.group_size,
            "active_slots": args.active_slots,
            "gate_up_weight_nrmse": gate_nrmse,
            "down_weight_nrmse": down_nrmse,
            "diagnostic_only": True,
            "closed_loop_authorized": False,
            "optimizer_constructed": False,
            "training_started": False,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        }
        with path.open("xb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        layer_metrics.append(
            {
                "layer": layer,
                "gate_up_weight_nrmse": gate_nrmse,
                "down_weight_nrmse": down_nrmse,
                "checkpoint_sha256": sha256_file(path),
            }
        )
        del gate_up, down, gate_packed, gate_scales, down_packed, down_scales
        torch.cuda.empty_cache()
        print(json.dumps(layer_metrics[-1]), flush=True)
    result = {
        "schema": RESULT_SCHEMA,
        "layers": 40,
        "group_size": args.group_size,
        "active_slots": args.active_slots,
        "maximum_gate_up_weight_nrmse": max(row["gate_up_weight_nrmse"] for row in layer_metrics),
        "maximum_down_weight_nrmse": max(row["down_weight_nrmse"] for row in layer_metrics),
        "layer_metrics": layer_metrics,
        "optimizer_constructed": False,
        "training_started": False,
        "closed_loop_authorized": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    paths = sorted(
        path for path in args.output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.relative_to(args.output)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
