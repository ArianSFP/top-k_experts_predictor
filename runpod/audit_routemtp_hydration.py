#!/usr/bin/env python3
"""Audit every record in an immutable RouteMTP prefix hydration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file

from harp_rtt.routemtp_cache import (
    ROUTEMTP_CACHE_RECORD_SCHEMA,
    ROUTEMTP_CACHE_SCHEMA,
    RouteMTPCacheGeometry,
    RouteMTPSourceOffset,
    sha256_file,
    validate_cache_tensors,
)


SCHEMA = "harp_routemtp_hydration_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hydration", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--expected-source-positions", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if args.expected_requests < 1 or args.expected_source_positions < 1:
        raise ValueError("expected RouteMTP hydration counts must be positive")
    if len(args.source_commit) != 40:
        raise ValueError("RouteMTP hydration audit requires a full source commit")
    manifest_path = args.hydration / "HYDRATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ROUTEMTP_CACHE_SCHEMA:
        raise ValueError("RouteMTP hydration schema mismatch")
    for key in (
        "formal_validation_opened", "calibration_opened", "sealed_test_opened",
        "training_started",
    ):
        if manifest.get(key) is not False:
            raise PermissionError(f"RouteMTP hydration violates {key}")
    if (
        manifest.get("outer_split") != "train"
        or manifest.get("causal_slice_required") is not True
        or manifest.get("runtime_available") is not True
    ):
        raise PermissionError("RouteMTP hydration is not causal outer-train runtime state")
    bindings = manifest.get("bindings", {})
    if bindings.get("source_commit") != args.source_commit:
        raise ValueError("RouteMTP hydration source binding differs")
    geometry = RouteMTPCacheGeometry(**manifest["geometry"])
    geometry.validate()
    records = list(manifest.get("records", []))
    offsets = list(manifest.get("source_offsets", []))
    if len(records) != args.expected_requests:
        raise ValueError("RouteMTP hydration request count differs")
    if len(offsets) != args.expected_source_positions:
        raise ValueError("RouteMTP hydration source-position count differs")

    request_lengths: dict[str, int] = {}
    total_bytes = 0
    for record in records:
        if (
            record.get("schema") != ROUTEMTP_CACHE_RECORD_SCHEMA
            or record.get("causal_slice_required") is not True
        ):
            raise ValueError("RouteMTP hydration record contract differs")
        request_id = str(record["request_id"])
        if request_id in request_lengths:
            raise ValueError("duplicate RouteMTP hydration request")
        path = args.hydration / str(record["relative_path"])
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"RouteMTP hydration checksum differs: {request_id}")
        tensors = load_file(path, device="cpu")
        length = validate_cache_tensors(tensors, geometry)
        if length != int(record["cache_length"]):
            raise ValueError("RouteMTP hydration record length differs")
        request_lengths[request_id] = length
        total_bytes += path.stat().st_size

    offset_keys: set[tuple[str, int]] = set()
    positions_by_request: dict[str, int] = {request: 0 for request in request_lengths}
    for value in offsets:
        offset = RouteMTPSourceOffset(**value)
        if offset.request_id not in request_lengths:
            raise ValueError("RouteMTP source offset lacks a request record")
        offset.validate(request_lengths[offset.request_id])
        key = (offset.request_id, offset.source_position)
        if key in offset_keys:
            raise ValueError("duplicate RouteMTP hydration source offset")
        offset_keys.add(key)
        positions_by_request[offset.request_id] += 1
    if not all(positions_by_request.values()):
        raise ValueError("RouteMTP hydration contains an unused request record")

    result = {
        "schema": SCHEMA,
        "passed": True,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "hydration_manifest_sha256": sha256_file(manifest_path),
        "requests": len(records),
        "source_positions": len(offsets),
        "record_bytes": total_bytes,
        "minimum_cache_length": min(request_lengths.values()),
        "maximum_cache_length": max(request_lengths.values()),
        "bindings": bindings,
        "all_record_checksums_passed": True,
        "all_tensor_geometry_passed": True,
        "all_offsets_causal_and_unique": True,
        "optimizer_constructed": False,
        "training_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
