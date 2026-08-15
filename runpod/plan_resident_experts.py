#!/usr/bin/env python3
"""Freeze a train-only, storage-optimal resident-expert allocation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.resident_policy import allocate_resident_experts  # noqa: E402
from harp_rtt.shadow_checkpoint import sha256_file  # noqa: E402
from harp_rtt.shadow_expert import resident_int4_storage_bytes  # noqa: E402
from runpod.train_harp_delta_v3 import (  # noqa: E402
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
)
from runpod.train_shadow_experts_local import load_split  # noqa: E402


SCHEMA = "harp_shadowroute_resident_allocation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--train-companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-residents", type=int, default=3680)
    parser.add_argument("--minimum-per-layer", type=int, default=64)
    parser.add_argument("--maximum-per-layer", type=int, default=128)
    return parser.parse_args()


def all_layer_counts(dataset) -> torch.Tensor:
    counts = torch.zeros(40, 256, dtype=torch.int64)
    layer_offsets = torch.arange(40, dtype=torch.int64)
    for source_index in dataset.indices:
        record = dataset.source.records[source_index]
        segment = dataset.source.segments[record.segment]
        for future_row in record.future_rows:
            ids = segment.read(
                "target",
                (int(future_row) + layer_offsets).tolist(),
                "selected_expert_ids",
            ).long()
            if ids.shape != (40, 8) or bool(((ids < 0) | (ids >= 256)).any()):
                raise ValueError("resident allocation encountered invalid target routes")
            for layer in range(40):
                counts[layer] += torch.bincount(ids[layer], minlength=256)
    return counts


def write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if len(args.source_commit) != 40 or any(
        value not in "0123456789abcdef" for value in args.source_commit
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    validate_partition(args.partition_manifest, "b2_reuse_4096")
    reuse = validate_reuse_split(args.reuse_split_manifest)
    args.data_profile = "b2_reuse_4096"
    args.layer = 0
    args.next_router_agreement = False
    dataset, groups = load_split(
        args,
        "train",
        selected_requests=set(reuse["inner_split"]["training_requests"]),
    )
    counts = all_layer_counts(dataset)
    allocation = allocate_resident_experts(
        counts,
        total_residents=args.total_residents,
        minimum_per_layer=args.minimum_per_layer,
        maximum_per_layer=args.maximum_per_layer,
    )
    packed_bytes = resident_int4_storage_bytes(layers=1, residents=1) * sum(
        allocation.resident_counts
    )
    shared_fallback_bytes = 3 * 2048 * 512 * 40 * 2
    value = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "train_companion_manifest_sha256": sha256_file(
            args.train_companion / "manifest.json"
        ),
        "training_requests": len(groups),
        "training_rows": len(dataset),
        "expert_counts": counts.tolist(),
        "resident_expert_ids_by_layer": [
            ids.tolist() for ids in allocation.resident_ids
        ],
        "resident_counts_by_layer": list(allocation.resident_counts),
        "covered_slots_by_layer": list(allocation.covered_slots),
        "total_slots_by_layer": list(allocation.total_slots),
        "coverage_by_layer": [
            covered / max(total, 1)
            for covered, total in zip(
                allocation.covered_slots, allocation.total_slots, strict=True
            )
        ],
        "mean_train_slot_coverage": allocation.mean_coverage,
        "total_residents": sum(allocation.resident_counts),
        "minimum_per_layer": args.minimum_per_layer,
        "maximum_per_layer": args.maximum_per_layer,
        "packed_int4_bytes": packed_bytes,
        "shared_fallback_bf16_bytes": shared_fallback_bytes,
        "total_shadow_bytes": packed_bytes + shared_fallback_bytes,
        "total_shadow_gib": (packed_bytes + shared_fallback_bytes) / 2**30,
        "selection_uses_train_routes_only": True,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    if value["total_shadow_gib"] >= 6.0:
        raise ValueError("resident allocation exceeds the six-GiB shadow budget")
    write_json_exclusive(args.output / "resident_allocation.json", value)
    checksum = sha256_file(args.output / "resident_allocation.json")
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        handle.write(f"{checksum}  resident_allocation.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({
        "output": str(args.output),
        "mean_train_slot_coverage": allocation.mean_coverage,
        "resident_counts_by_layer": allocation.resident_counts,
        "total_shadow_gib": value["total_shadow_gib"],
    }, indent=2))


if __name__ == "__main__":
    main()
