#!/usr/bin/env python3
"""Export full-width identity-preserving RouteQuant expert shards."""

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

from harp_rtt.route_quant import (
    PackedRouteQuantExperts,
    mixed_routequant_projected_bytes,
)
from harp_rtt.shadow_bundle import LOCAL_SCHEMA
from harp_rtt.shadow_checkpoint import (
    IndexedCheckpoint,
    load_target_layer_experts,
    sha256_file,
)


SCHEMA = "harp_routequant_export_v1"
RESULT_SCHEMA = "harp_routequant_export_result_v1"
SCHEDULE_SCHEMA = "harp_routequant_schedule_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bits", type=int, choices=(1, 2, 3, 4))
    group.add_argument("--schedule", type=Path)
    parser.add_argument("--group-size", type=int, choices=(32, 64), default=64)
    parser.add_argument(
        "--scale-storage", choices=("bf16", "log8"), default="log8"
    )
    parser.add_argument("--scale-method", choices=("amax", "mse"), default="mse")
    parser.add_argument("--expert-chunk", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def resolve_scale_storage(name: str) -> tuple[torch.dtype, int]:
    if name == "bf16":
        return torch.bfloat16, 2
    if name == "log8":
        return torch.uint8, 1
    raise ValueError("deployable RouteQuant export supports BF16 or LOG8 scales")


def load_schedule(
    path: Path | None,
    *,
    uniform_bits: int | None,
    source_commit: str,
    target_checkpoint_index_sha256: str,
) -> tuple[torch.Tensor, str | None]:
    if path is None:
        if uniform_bits not in (1, 2, 3, 4):
            raise ValueError("uniform RouteQuant export requires bits 1..4")
        return torch.full((40, 256), int(uniform_bits), dtype=torch.int8), None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema") != SCHEDULE_SCHEMA:
        raise ValueError("RouteQuant schedule schema mismatch")
    if value.get("source_commit") != source_commit:
        raise ValueError("RouteQuant schedule source lineage differs")
    if (
        value.get("target_checkpoint_index_sha256")
        != target_checkpoint_index_sha256
    ):
        raise ValueError("RouteQuant schedule target checkpoint differs")
    widths = torch.as_tensor(value.get("bit_widths"), dtype=torch.int8)
    if widths.shape != (40, 256):
        raise ValueError("RouteQuant schedule must be [40,256]")
    if any(int(item) not in (1, 2, 3, 4) for item in widths.flatten().tolist()):
        raise ValueError("deployable RouteQuant schedule cannot omit or exceed experts")
    return widths, sha256_file(path)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite RouteQuant export {args.output}")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    if args.expert_chunk < 1:
        raise ValueError("expert chunk must be positive")
    checkpoint = IndexedCheckpoint(args.model)
    schedule, schedule_sha = load_schedule(
        args.schedule,
        uniform_bits=args.bits,
        source_commit=args.source_commit,
        target_checkpoint_index_sha256=checkpoint.index_sha256,
    )
    scale_dtype, scale_bytes = resolve_scale_storage(args.scale_storage)
    args.output.mkdir(parents=True)
    projected_bytes = mixed_routequant_projected_bytes(
        schedule, group_size=args.group_size, scale_bytes=scale_bytes
    )
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "schedule_sha256": schedule_sha,
        "uniform_bits": args.bits,
        "bit_histogram": {
            str(bits): int((schedule == bits).sum()) for bits in range(1, 5)
        },
        "group_size": args.group_size,
        "scale_storage": args.scale_storage,
        "scale_method": args.scale_method,
        "projected_persistent_bytes": projected_bytes,
        "projected_persistent_gib": projected_bytes / 2**30,
        "active_slots": 8,
        "target_expert_loads_at_runtime": False,
        "native_route_ids_and_weights_unchanged": True,
        "optimizer_constructed": False,
        "training_started": False,
        "diagnostic_only": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)
    rows = []
    serialized_bytes = 0
    for layer in range(40):
        gate_up, down = load_target_layer_experts(
            checkpoint, layer, device=args.device, dtype=torch.bfloat16
        )
        module = PackedRouteQuantExperts.from_target(
            gate_up,
            down,
            gate_up_bits=schedule[layer],
            exact_k=8,
            group_size=args.group_size,
            scale_method=args.scale_method,
            item_chunk=args.expert_chunk,
            scale_dtype=scale_dtype,
        )
        directory = args.output / f"layer_{layer:02d}"
        directory.mkdir()
        path = directory / f"shadow_routequant_all8_layer_{layer:02d}.pt"
        value = {
            "schema": LOCAL_SCHEMA,
            "mode": "routequant_all8",
            "layer": layer,
            "source_commit": args.source_commit,
            "target_checkpoint_index_sha256": checkpoint.index_sha256,
            "model_state_dict": module.state_dict(),
            "bit_widths": schedule[layer],
            "group_size": args.group_size,
            "scale_storage": args.scale_storage,
            "scale_method": args.scale_method,
            "active_slots": 8,
            "persistent_bytes": module.persistent_nbytes(),
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
        size = path.stat().st_size
        serialized_bytes += size
        row = {
            "layer": layer,
            "bit_histogram": {
                str(bits): int((schedule[layer] == bits).sum())
                for bits in range(1, 5)
            },
            "persistent_bytes": module.persistent_nbytes(),
            "serialized_bytes": size,
            "checkpoint_sha256": sha256_file(path),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del gate_up, down, module, value
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "schema": RESULT_SCHEMA,
        "layers": 40,
        "projected_persistent_bytes": projected_bytes,
        "serialized_checkpoint_bytes": serialized_bytes,
        "layer_metrics": rows,
        "closed_loop_authorized": False,
        "optimizer_constructed": False,
        "training_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    files = sorted(
        path for path in args.output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in files:
            handle.write(
                f"{sha256_file(path)}  {path.relative_to(args.output)}\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
