#!/usr/bin/env python3
"""Plan a fallback-free resident INT4 namespace from train-only route mass."""

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

from harp_rtt.resident_policy import allocate_resident_utility  # noqa: E402
from harp_rtt.shadow_checkpoint import sha256_file  # noqa: E402
from harp_rtt.shadow_expert import resident_int4_storage_bytes  # noqa: E402
from runpod.train_harp_delta_v3 import (  # noqa: E402
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
)
from runpod.train_shadow_experts_local import load_split  # noqa: E402


SCHEMA = "harp_shadowroute_resident_allocation_v2"
LAYERS = 40
EXPERTS = 256
SLOTS = 8
BUNDLE_OVERHEAD_RESERVE_BYTES = 8 * 2**20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--train-companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-residents", type=int, default=3850)
    parser.add_argument("--minimum-per-layer", type=int, default=64)
    parser.add_argument("--maximum-per-layer", type=int, default=128)
    return parser.parse_args()


def train_route_statistics(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.zeros(LAYERS, EXPERTS, dtype=torch.int64)
    weight_mass = torch.zeros(LAYERS, EXPERTS, dtype=torch.float64)
    layer_offsets = torch.arange(LAYERS, dtype=torch.int64)
    for source_index in dataset.indices:
        record = dataset.source.records[source_index]
        segment = dataset.source.segments[record.segment]
        for future_row in record.future_rows:
            rows = (int(future_row) + layer_offsets).tolist()
            ids = segment.read("target", rows, "selected_expert_ids").long()
            weights = segment.read(
                "target", rows, "selected_execution_weights"
            ).double()
            if ids.shape != (LAYERS, SLOTS) or weights.shape != ids.shape:
                raise ValueError("resident-only planner route tensors changed")
            if (
                bool(((ids < 0) | (ids >= EXPERTS)).any())
                or not torch.isfinite(weights).all()
                or bool((weights < 0).any())
            ):
                raise ValueError("resident-only planner saw an invalid route")
            for layer in range(LAYERS):
                counts[layer].scatter_add_(
                    0, ids[layer], torch.ones(SLOTS, dtype=torch.int64)
                )
                weight_mass[layer].scatter_add_(0, ids[layer], weights[layer])
    if bool((counts.sum(dim=1) == 0).any()) or bool(
        (weight_mass.sum(dim=1) <= 0).any()
    ):
        raise ValueError("resident-only planner found an empty target layer")
    return counts, weight_mass


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
    counts, weight_mass = train_route_statistics(dataset)
    allocation = allocate_resident_utility(
        weight_mass,
        total_residents=args.total_residents,
        minimum_per_layer=args.minimum_per_layer,
        maximum_per_layer=args.maximum_per_layer,
    )
    cell_bytes = resident_int4_storage_bytes(layers=1, residents=1)
    packed_bytes = cell_bytes * sum(allocation.resident_counts)
    planned_bytes = packed_bytes + BUNDLE_OVERHEAD_RESERVE_BYTES
    if planned_bytes >= 6 * 2**30:
        raise ValueError("resident-only allocation exceeds the six-GiB budget")
    resident_ids = allocation.resident_ids
    covered_slots = tuple(
        int(counts[layer, resident_ids[layer]].sum())
        for layer in range(LAYERS)
    )
    total_slots = tuple(int(counts[layer].sum()) for layer in range(LAYERS))
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
        "allocation_objective": "selected_execution_weight_mass",
        "expert_counts": counts.tolist(),
        "expert_weight_mass": weight_mass.tolist(),
        "resident_expert_ids_by_layer": [
            ids.tolist() for ids in resident_ids
        ],
        "resident_counts_by_layer": list(allocation.resident_counts),
        "covered_slots_by_layer": list(covered_slots),
        "total_slots_by_layer": list(total_slots),
        "coverage_by_layer": [
            covered / max(total, 1)
            for covered, total in zip(covered_slots, total_slots, strict=True)
        ],
        "captured_weight_mass_by_layer": list(allocation.captured_utility),
        "total_weight_mass_by_layer": list(allocation.total_utility),
        "weight_mass_coverage_by_layer": [
            captured / max(total, 1e-12)
            for captured, total in zip(
                allocation.captured_utility,
                allocation.total_utility,
                strict=True,
            )
        ],
        "mean_train_slot_coverage": sum(
            covered / max(total, 1)
            for covered, total in zip(covered_slots, total_slots, strict=True)
        ) / LAYERS,
        "mean_train_weight_mass_coverage": allocation.mean_utility_coverage,
        "total_residents": sum(allocation.resident_counts),
        "minimum_per_layer": args.minimum_per_layer,
        "maximum_per_layer": args.maximum_per_layer,
        "resident_cell_bytes": cell_bytes,
        "packed_int4_bytes": packed_bytes,
        "fallback_bytes": 0,
        "bundle_overhead_reserve_bytes": BUNDLE_OVERHEAD_RESERVE_BYTES,
        "maximum_planned_bytes": planned_bytes,
        "maximum_planned_gib": planned_bytes / 2**30,
        "selection_uses_train_routes_only": True,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "resident_allocation.json", value)
    checksum = sha256_file(args.output / "resident_allocation.json")
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        handle.write(f"{checksum}  resident_allocation.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({
        "output": str(args.output),
        "mean_train_slot_coverage": value["mean_train_slot_coverage"],
        "mean_train_weight_mass_coverage": value[
            "mean_train_weight_mass_coverage"
        ],
        "resident_counts_by_layer": allocation.resident_counts,
        "maximum_planned_gib": value["maximum_planned_gib"],
    }, indent=2))


if __name__ == "__main__":
    main()
