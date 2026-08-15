#!/usr/bin/env python3
"""Create the minimal serving package from a sealed Resident-Shadow bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.shadow_checkpoint import sha256_file  # noqa: E402


SOURCE_SCHEMA = "harp_shadowroute_resident_bundle_v1"
SCHEMA = "harp_shadowroute_resident_deployment_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-package-gib", type=float, default=6.0)
    return parser.parse_args()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    source_result_path = args.sealed_bundle / "BUNDLE_RESULT.json"
    source_sums_path = args.sealed_bundle / "SHA256SUMS"
    if not source_result_path.is_file() or not source_sums_path.is_file():
        raise ValueError("source Resident-Shadow bundle is not sealed")
    source = json.loads(source_result_path.read_text(encoding="utf-8"))
    if source.get("schema") != SOURCE_SCHEMA or source.get("passed") is not True:
        raise ValueError("source Resident-Shadow bundle did not pass its audit")
    if source.get("target_expert_loads_permitted") is not False:
        raise PermissionError("source bundle permits target expert loading")
    if source.get("native_mtp_included_in_size") is not False:
        raise ValueError("source bundle size contract unexpectedly includes native MTP")
    layers = source.get("layers")
    if not isinstance(layers, list) or len(layers) != 40:
        raise ValueError("source bundle does not contain forty layers")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite deployment package {args.output}")

    args.output.mkdir(parents=True)
    output_layers = args.output / "layers"
    output_layers.mkdir()
    packaged_layers = []
    checkpoint_bytes = 0
    for expected_layer, item in enumerate(layers):
        if not isinstance(item, dict) or item.get("layer") != expected_layer:
            raise ValueError("source bundle layer order changed")
        source_path = args.sealed_bundle / str(item.get("checkpoint"))
        expected_sha = str(item.get("checkpoint_sha256"))
        if not source_path.is_file() or sha256_file(source_path) != expected_sha:
            raise ValueError(f"source checkpoint digest mismatch at layer {expected_layer}")
        destination = output_layers / source_path.name
        shutil.copy2(source_path, destination)
        if sha256_file(destination) != expected_sha:
            raise IOError(f"copied checkpoint digest mismatch at layer {expected_layer}")
        checkpoint_bytes += destination.stat().st_size
        packaged_layers.append({
            "layer": expected_layer,
            "path": str(destination.relative_to(args.output)),
            "sha256": expected_sha,
            "resident_count": int(item["resident_count"]),
        })

    plan_source = args.sealed_bundle / "resident_allocation.json"
    if not plan_source.is_file() or sha256_file(plan_source) != source.get(
        "allocation_plan_sha256"
    ):
        raise ValueError("source allocation plan digest mismatch")
    plan_destination = args.output / "resident_allocation.json"
    shutil.copy2(plan_source, plan_destination)
    actual_gib = checkpoint_bytes / 2**30
    if actual_gib >= args.maximum_package_gib:
        raise ValueError("serialized Resident-Shadow package exceeds its size gate")

    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_bundle_result_sha256": sha256_file(source_result_path),
        "source_bundle_sums_sha256": sha256_file(source_sums_path),
        "source_commit": source["source_commit"],
        "target_checkpoint_index_sha256": source[
            "target_checkpoint_index_sha256"
        ],
        "allocation_plan_sha256": source["allocation_plan_sha256"],
        "layers": packaged_layers,
        "layer_count": len(packaged_layers),
        "total_residents": source["total_residents"],
        "logical_shadow_bytes": source["total_shadow_bytes"],
        "serialized_checkpoint_bytes": checkpoint_bytes,
        "serialized_checkpoint_gib": actual_gib,
        "native_mtp_included_in_size": False,
        "duplicates_target_nonexpert_backbone": False,
        "duplicates_target_embedding_or_lm_head": False,
        "target_expert_references_present": False,
        "target_expert_loads_permitted": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "passed": True,
    }
    _write_json(args.output / "DEPLOYMENT_MANIFEST.json", manifest)
    paths = sorted(
        path for path in args.output.rglob("*")
        if path.is_file() and path != args.output / "SHA256SUMS"
    )
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.relative_to(args.output)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
