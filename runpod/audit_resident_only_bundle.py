#!/usr/bin/env python3
"""Seal a forty-layer fallback-free Resident-Shadow bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.shadow_bundle import LOCAL_SCHEMA, discover_layer_checkpoints  # noqa: E402
from harp_rtt.shadow_checkpoint import sha256_file  # noqa: E402
from harp_rtt.shadow_expert import resident_int4_storage_bytes  # noqa: E402


SCHEMA = "harp_shadowroute_resident_only_bundle_v2"
PLAN_SCHEMA = "harp_shadowroute_resident_allocation_v2"
RESULT_SCHEMA = "harp_shadowroute_local_expert_layer_result_v1"
EXPECTED_STATE_KEYS = {
    "resident_ids", "expert_to_resident",
    "gate_up_packed", "gate_up_scales", "down_packed", "down_scales",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--allocation-plan", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--target-checkpoint-index-sha256", required=True)
    parser.add_argument("--maximum-shadow-gib", type=float, default=6.0)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def verify_layer_checksums(directory: Path) -> None:
    sums = directory / "SHA256SUMS"
    if not sums.is_file():
        raise ValueError(f"resident-only layer lacks checksums: {directory}")
    seen = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or Path(name).name != name
            or name in seen
        ):
            raise ValueError(f"malformed resident-only checksum: {sums}")
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"resident-only checksum mismatch: {path}")
        seen.add(name)
    expected = {
        path.name for path in directory.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if seen != expected:
        raise ValueError(f"resident-only checksum inventory changed: {directory}")


def write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if not args.bundle_root.is_dir():
        raise FileNotFoundError(args.bundle_root)
    if (args.bundle_root / "BUNDLE_RESULT.json").exists():
        raise FileExistsError("refusing to reseal resident-only bundle")
    plan = read_json(args.allocation_plan)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("source_commit") != args.source_commit:
        raise ValueError("resident-only allocation lineage mismatch")
    for name, expected in {
        "fallback_bytes": 0,
        "selection_uses_train_routes_only": True,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }.items():
        if plan.get(name) != expected:
            raise PermissionError(f"resident-only plan has unsafe {name}")
    paths = discover_layer_checkpoints(args.bundle_root, "resident_int4_only")
    planned_ids = plan["resident_expert_ids_by_layer"]
    planned_counts = plan["resident_counts_by_layer"]
    layers = []
    serialized_bytes = 0
    total_residents = 0
    for layer, checkpoint_path in enumerate(paths):
        verify_layer_checksums(checkpoint_path.parent)
        manifest = read_json(checkpoint_path.parent / "run_manifest.json")
        result = read_json(checkpoint_path.parent / "STAGE_RESULT.json")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        expected_identity = {
            "schema": LOCAL_SCHEMA,
            "source_commit": args.source_commit,
            "mode": "resident_int4_only",
            "layer": layer,
            "target_checkpoint_index_sha256": args.target_checkpoint_index_sha256,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        }
        for name, expected in expected_identity.items():
            if checkpoint.get(name) != expected:
                raise ValueError(f"resident-only checkpoint {name} mismatch at layer {layer}")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict) or set(state) != EXPECTED_STATE_KEYS:
            raise ValueError(f"resident-only state changed at layer {layer}")
        ids = checkpoint.get("resident_expert_ids")
        expected_ids = torch.as_tensor(planned_ids[layer], dtype=torch.long)
        if not isinstance(ids, torch.Tensor) or not torch.equal(ids.long(), expected_ids):
            raise ValueError(f"resident-only namespace mismatch at layer {layer}")
        count = int(checkpoint.get("resident_count", -1))
        if count != int(planned_counts[layer]) or count != ids.numel():
            raise ValueError(f"resident-only count mismatch at layer {layer}")
        if manifest.get("optimizer_constructed") is not False or manifest.get(
            "training_started"
        ) is not False:
            raise PermissionError(f"resident-only export optimized layer {layer}")
        if result.get("schema") != RESULT_SCHEMA or result.get(
            "checkpoint_sha256"
        ) != sha256_file(checkpoint_path):
            raise ValueError(f"resident-only stage result mismatch at layer {layer}")
        if result.get("training_started") is not False or result.get(
            "optimizer_constructed"
        ) is not False:
            raise PermissionError(f"resident-only result optimized layer {layer}")
        size = checkpoint_path.stat().st_size
        serialized_bytes += size
        total_residents += count
        layers.append({
            "layer": layer,
            "resident_count": count,
            "checkpoint": str(checkpoint_path.relative_to(args.bundle_root)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "serialized_bytes": size,
        })
    if total_residents != int(plan["total_residents"]):
        raise ValueError("resident-only total differs from its plan")
    logical_bytes = resident_int4_storage_bytes(
        layers=1, residents=1
    ) * total_residents
    if logical_bytes != int(plan["packed_int4_bytes"]):
        raise ValueError("resident-only logical storage differs from plan")
    if serialized_bytes >= args.maximum_shadow_gib * 2**30:
        raise ValueError("resident-only serialized package exceeds size gate")
    destination = args.bundle_root / "resident_allocation.json"
    shutil.copy2(args.allocation_plan, destination)
    if sha256_file(destination) != sha256_file(args.allocation_plan):
        raise IOError("resident-only allocation copy changed")
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "target_checkpoint_index_sha256": args.target_checkpoint_index_sha256,
        "allocation_plan_sha256": sha256_file(destination),
        "layers": layers,
        "layer_count": 40,
        "total_residents": total_residents,
        "packed_int4_bytes": logical_bytes,
        "fallback_bytes": 0,
        "serialized_checkpoint_bytes": serialized_bytes,
        "serialized_checkpoint_gib": serialized_bytes / 2**30,
        "native_mtp_included_in_size": False,
        "target_expert_loads_permitted": False,
        "optimizer_constructed": False,
        "closed_loop_evaluation_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "passed": True,
    }
    write_json_exclusive(args.bundle_root / "BUNDLE_RESULT.json", result)
    files = sorted(
        path for path in args.bundle_root.rglob("*")
        if path.is_file() and path != args.bundle_root / "BUNDLE_SHA256SUMS"
    )
    with (args.bundle_root / "BUNDLE_SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(args.bundle_root)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
