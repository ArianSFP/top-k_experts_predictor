#!/usr/bin/env python3
"""Export one optimizer-free fallback-free resident INT4 layer shard."""

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

from harp_rtt.shadow_checkpoint import IndexedCheckpoint, sha256_file  # noqa: E402
from harp_rtt.shadow_expert import (  # noqa: E402
    PackedInt4ResidentExperts,
    resident_int4_storage_bytes,
)
from runpod.train_shadow_experts_local import load_target_layer_experts  # noqa: E402


SCHEMA = "harp_shadowroute_local_expert_layer_v1"
RESULT_SCHEMA = "harp_shadowroute_local_expert_layer_result_v1"
PLAN_SCHEMA = "harp_shadowroute_resident_allocation_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--allocation-plan", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=range(40), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_checksums(output: Path) -> None:
    paths = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite resident-only shard {args.output}")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    plan = json.loads(args.allocation_plan.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("resident-only allocation plan schema changed")
    expected_plan = {
        "source_commit": args.source_commit,
        "allocation_objective": "selected_execution_weight_mass",
        "fallback_bytes": 0,
        "selection_uses_train_routes_only": True,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    for name, expected in expected_plan.items():
        if plan.get(name) != expected:
            raise ValueError(f"resident-only allocation plan {name} mismatch")
    ids_by_layer = plan.get("resident_expert_ids_by_layer")
    counts_by_layer = plan.get("resident_counts_by_layer")
    if (
        not isinstance(ids_by_layer, list)
        or len(ids_by_layer) != 40
        or not isinstance(counts_by_layer, list)
        or len(counts_by_layer) != 40
    ):
        raise ValueError("resident-only plan lacks forty layers")
    resident_ids = torch.as_tensor(
        ids_by_layer[args.layer], dtype=torch.long, device=args.device
    )
    if (
        resident_ids.ndim != 1
        or resident_ids.unique().numel() != resident_ids.numel()
        or resident_ids.numel() != int(counts_by_layer[args.layer])
        or bool(((resident_ids < 0) | (resident_ids >= 256)).any())
    ):
        raise ValueError("resident-only layer namespace is invalid")
    checkpoint = IndexedCheckpoint(args.target_model)
    gate_up, down = load_target_layer_experts(
        checkpoint, args.layer, device=torch.device(args.device), dtype=torch.bfloat16
    )
    module = PackedInt4ResidentExperts.from_target(
        resident_ids, None, gate_up, down, exact_k=8, group_size=64
    )
    module.requires_grad_(False)
    state = {name: value.detach().cpu() for name, value in module.state_dict().items()}
    expected_state = {
        "resident_ids", "expert_to_resident",
        "gate_up_packed", "gate_up_scales", "down_packed", "down_scales",
    }
    if set(state) != expected_state:
        raise ValueError("resident-only export contains unexpected state")
    args.output.mkdir(parents=True)
    plan_sha = sha256_file(args.allocation_plan)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "mode": "resident_int4_only",
        "layer": args.layer,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "resident_allocation_plan_sha256": plan_sha,
        "resident_count": resident_ids.numel(),
        "resident_expert_ids": resident_ids.cpu().tolist(),
        "logical_resident_bytes": resident_int4_storage_bytes(
            layers=1, residents=resident_ids.numel()
        ),
        "trainable_names": [],
        "trainable_parameters": 0,
        "optimizer_constructed": False,
        "training_started": False,
        "target_expert_loads_permitted": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)
    checkpoint_path = (
        args.output / f"shadow_resident_int4_only_layer_{args.layer:02d}.pt"
    )
    value = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "mode": "resident_int4_only",
        "layer": args.layer,
        "seed": 42,
        "model_state_dict": state,
        "resident_expert_ids": resident_ids.cpu(),
        "resident_count": resident_ids.numel(),
        "resident_train_slot_coverage": plan["coverage_by_layer"][args.layer],
        "resident_train_weight_mass_coverage": plan[
            "weight_mass_coverage_by_layer"
        ][args.layer],
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "partition_manifest_sha256": plan["partition_manifest_sha256"],
        "diagnostic_only": True,
        "closed_loop_authorized": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    with checkpoint_path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    result = {
        "schema": RESULT_SCHEMA,
        "mode": "resident_int4_only",
        "layer": args.layer,
        "resident_count": resident_ids.numel(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_started": False,
        "optimizer_constructed": False,
        "closed_loop_evaluation_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    write_checksums(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

