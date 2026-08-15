#!/usr/bin/env python3
"""Seal and checksum a forty-layer Resident-Shadow bundle.

This command is intentionally read-mostly: it validates an already completed
bundle and writes only ``BUNDLE_RESULT.json`` and the root ``SHA256SUMS``.  All
layer runs and the allocation plan must be immutable before it is invoked.
"""

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

from harp_rtt.shadow_bundle import (  # noqa: E402
    LOCAL_SCHEMA,
    discover_layer_checkpoints,
)
from harp_rtt.shadow_checkpoint import sha256_file  # noqa: E402
from harp_rtt.shadow_expert import resident_int4_storage_bytes  # noqa: E402


SCHEMA = "harp_shadowroute_resident_bundle_v1"
PLAN_SCHEMA = "harp_shadowroute_resident_allocation_v1"
RESULT_SCHEMA = "harp_shadowroute_local_expert_layer_result_v1"
EXPECTED_STATE_KEYS = {
    "resident_ids",
    "expert_to_resident",
    "gate_up_packed",
    "gate_up_scales",
    "down_packed",
    "down_scales",
    "fallback.draft_expert.gate_up_proj.weight",
    "fallback.draft_expert.down_proj.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--allocation-plan", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--target-checkpoint-index-sha256", required=True)
    parser.add_argument("--maximum-shadow-gib", type=float, default=6.0)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_sha(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _verify_run_checksums(directory: Path) -> None:
    sums = directory / "SHA256SUMS"
    if not sums.is_file():
        raise ValueError(f"layer run lacks SHA256SUMS: {directory}")
    seen: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        _validate_sha(digest, name="layer checksum")
        if separator != "  " or not relative or relative in seen:
            raise ValueError(f"malformed or duplicate checksum entry: {sums}")
        if Path(relative).name != relative:
            raise ValueError("layer checksum entry must name a direct child")
        path = directory / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"layer checksum mismatch: {path}")
        seen.add(relative)
    expected = {
        path.name for path in directory.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if seen != expected:
        raise ValueError(f"layer checksum inventory is incomplete: {directory}")


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_root_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path != root / "SHA256SUMS"
    )
    with (root / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.relative_to(root)}\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if not args.bundle_root.is_dir():
        raise FileNotFoundError(args.bundle_root)
    if (args.bundle_root / "BUNDLE_RESULT.json").exists() or (
        args.bundle_root / "SHA256SUMS"
    ).exists():
        raise FileExistsError("refusing to reseal an existing resident bundle")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    _validate_sha(
        args.target_checkpoint_index_sha256,
        name="target checkpoint index SHA-256",
    )

    plan = _json(args.allocation_plan)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("resident allocation plan schema mismatch")
    if plan.get("source_commit") != args.source_commit:
        raise ValueError("resident allocation plan source lineage mismatch")
    for flag in (
        "selection_uses_train_routes_only",
        "optimizer_constructed",
        "formal_validation_opened",
        "calibration_opened",
        "sealed_test_opened",
    ):
        expected = flag == "selection_uses_train_routes_only"
        if plan.get(flag) is not expected:
            raise PermissionError(f"resident allocation plan has unsafe {flag}")
    planned_ids = plan.get("resident_expert_ids_by_layer")
    planned_counts = plan.get("resident_counts_by_layer")
    if not isinstance(planned_ids, list) or len(planned_ids) != 40:
        raise ValueError("resident allocation plan does not contain forty layers")
    if not isinstance(planned_counts, list) or len(planned_counts) != 40:
        raise ValueError("resident allocation plan count vector is invalid")
    plan_sha256 = sha256_file(args.allocation_plan)

    paths = discover_layer_checkpoints(args.bundle_root, "resident_int4_shared")
    layers: list[dict[str, Any]] = []
    total_residents = 0
    for layer, checkpoint_path in enumerate(paths):
        run_dir = checkpoint_path.parent
        _verify_run_checksums(run_dir)
        manifest = _json(run_dir / "run_manifest.json")
        result = _json(run_dir / "STAGE_RESULT.json")
        value = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict) or value.get("schema") != LOCAL_SCHEMA:
            raise ValueError(f"resident checkpoint schema mismatch: {checkpoint_path}")
        expected_identity = {
            "source_commit": args.source_commit,
            "mode": "resident_int4_shared",
            "layer": layer,
            "target_checkpoint_index_sha256": args.target_checkpoint_index_sha256,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        }
        for name, expected in expected_identity.items():
            if value.get(name) != expected:
                raise ValueError(f"resident checkpoint {name} mismatch at layer {layer}")
        if value.get("closed_loop_authorized") is not False:
            raise ValueError("diagnostic component shard unexpectedly authorizes promotion")
        state = value.get("model_state_dict")
        if not isinstance(state, dict) or set(state) != EXPECTED_STATE_KEYS:
            raise ValueError(f"resident checkpoint state is incomplete at layer {layer}")
        ids = value.get("resident_expert_ids")
        expected_ids = torch.as_tensor(planned_ids[layer], dtype=torch.long)
        if not isinstance(ids, torch.Tensor) or not torch.equal(ids.long(), expected_ids):
            raise ValueError(f"resident expert namespace differs from plan at layer {layer}")
        count = int(value.get("resident_count", -1))
        if count != int(planned_counts[layer]) or count != ids.numel():
            raise ValueError(f"resident expert count differs from plan at layer {layer}")
        if ids.unique().numel() != ids.numel() or bool(((ids < 0) | (ids >= 256)).any()):
            raise ValueError(f"resident expert namespace is invalid at layer {layer}")
        if manifest.get("resident_allocation_plan_sha256") != plan_sha256:
            raise ValueError(f"layer {layer} was not produced from the frozen allocation plan")
        for name, expected in {
            **expected_identity,
            "optimizer_constructed": False,
            "training_started": None,
            "resident_count": count,
        }.items():
            if name == "training_started":
                continue
            if manifest.get(name) != expected:
                raise ValueError(f"resident run manifest {name} mismatch at layer {layer}")
        if manifest.get("trainable_parameters") != 0 or manifest.get("trainable_names") != []:
            raise ValueError(f"resident export constructed trainable parameters at layer {layer}")
        if result.get("schema") != RESULT_SCHEMA or result.get("layer") != layer:
            raise ValueError(f"resident stage result identity mismatch at layer {layer}")
        if result.get("training_started") is not False or result.get("optimizer_constructed") is not False:
            raise ValueError(f"resident export started optimization at layer {layer}")
        for flag in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
            if result.get(flag) is not False:
                raise PermissionError(f"resident stage result crossed {flag} at layer {layer}")
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if result.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(f"resident result checkpoint digest mismatch at layer {layer}")
        total_residents += count
        layers.append({
            "layer": layer,
            "resident_count": count,
            "resident_train_slot_coverage": value.get("resident_train_slot_coverage"),
            "tune_next_router_recall_at_8": value.get("tune", {}).get("next_router_recall_at_8"),
            "development_next_router_recall_at_8": value.get("development", {}).get("next_router_recall_at_8"),
            "checkpoint": str(checkpoint_path.relative_to(args.bundle_root)),
            "checkpoint_sha256": checkpoint_sha256,
        })

    if total_residents != int(plan.get("total_residents", -1)):
        raise ValueError("sealed bundle resident total differs from allocation plan")
    packed_bytes = resident_int4_storage_bytes(
        layers=1, residents=1
    ) * total_residents
    fallback_bytes = 3 * 2048 * 512 * 40 * 2
    shadow_bytes = packed_bytes + fallback_bytes
    if shadow_bytes != int(plan.get("total_shadow_bytes", -1)):
        raise ValueError("sealed bundle logical storage differs from allocation plan")
    shadow_gib = shadow_bytes / 2**30
    if shadow_gib >= args.maximum_shadow_gib:
        raise ValueError("sealed bundle exceeds its resident shadow storage gate")

    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "target_checkpoint_index_sha256": args.target_checkpoint_index_sha256,
        "allocation_plan": str(args.allocation_plan),
        "allocation_plan_sha256": plan_sha256,
        "layers": layers,
        "layer_count": len(layers),
        "total_residents": total_residents,
        "packed_int4_bytes": packed_bytes,
        "shared_fallback_bf16_bytes": fallback_bytes,
        "total_shadow_bytes": shadow_bytes,
        "total_shadow_gib": shadow_gib,
        "native_mtp_included_in_size": False,
        "target_expert_loads_permitted": False,
        "optimizer_constructed": False,
        "closed_loop_evaluation_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "passed": True,
    }
    _exclusive_json(args.bundle_root / "BUNDLE_RESULT.json", result)
    _write_root_checksums(args.bundle_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
