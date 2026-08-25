#!/usr/bin/env python3
"""Build a fair, frequency-core-locked single-layer resident plan.

The candidate changes only one layer of a fully compliant reference plan.  It
keeps the exact per-layer cell count, total footprint, and mandatory top-64
frequency core, then ranks the remaining cells by a sealed train-only omission
damage score.  No evaluation split is opened here.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.resident_damage import (  # noqa: E402
    _canonical_ids_sha256,
    frequency_core_ids,
    validate_core_inclusion,
)
from harp_rtt.shadow_checkpoint import sha256_file  # noqa: E402
from runpod.train_shadow_experts_local import (  # noqa: E402
    write_checksums,
    write_json_exclusive,
)


SCHEMA = "harp_resident_damage_layer_plan_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-allocation", type=Path, required=True)
    parser.add_argument("--damage-result", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=range(40), required=True)
    parser.add_argument("--metric", default="router_logit_mse")
    parser.add_argument("--expected-frequency-core-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _rank_descending(values: list[float], allowed: set[int]) -> list[int]:
    if len(values) != 256:
        raise ValueError("damage vector must cover exactly 256 experts")
    if not all(torch.isfinite(torch.tensor(values))):
        raise ValueError("damage vector contains NaN or Inf")
    return sorted(allowed, key=lambda expert: (-float(values[expert]), expert))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(args.source_commit) != 40:
        raise ValueError("source commit must be a full SHA")
    reference = _load_json(args.reference_allocation)
    damage = _load_json(args.damage_result)
    if (
        damage.get("layer") != args.layer
        or damage.get("complete_train_scan") is not True
        or damage.get("eligible_for_final_allocation") is not True
        or damage.get("rows_seen") != 3584
        or damage.get("valid_endpoints") != 14336
        or damage.get("selected_omissions") != 114688
        or damage.get("formal_validation_opened") is not False
        or damage.get("calibration_opened") is not False
        or damage.get("sealed_test_opened") is not False
    ):
        raise ValueError("damage result is not an eligible complete train-only layer scan")
    ids_raw = reference.get("resident_expert_ids_by_layer")
    counts_raw = reference.get("expert_counts")
    if not isinstance(ids_raw, list) or not isinstance(counts_raw, list):
        raise ValueError("reference allocation lacks IDs or expert counts")
    ids = [[int(value) for value in row] for row in ids_raw]
    counts = torch.as_tensor(counts_raw, dtype=torch.int64)
    if counts.shape != (40, 256) or len(ids) != 40:
        raise ValueError("reference allocation geometry changed")
    core, core_hash = frequency_core_ids(counts, core_size=64)
    if core_hash != args.expected_frequency_core_sha256:
        raise ValueError("frequency-core hash changed")
    validate_core_inclusion(ids, core)
    if sum(map(len, ids)) != 3850:
        raise ValueError("reference allocation is not the exact 3,850-cell plan")
    target_count = len(ids[args.layer])
    if target_count < 64:
        raise ValueError("target layer cannot contain its mandatory core")
    metric_sums = damage.get("metric_sums_by_expert")
    if not isinstance(metric_sums, dict) or args.metric not in metric_sums:
        raise ValueError("requested damage metric is absent")
    values = [float(value) for value in metric_sums[args.metric]]
    core_set = set(int(value) for value in core[args.layer].tolist())
    optional = _rank_descending(values, set(range(256)) - core_set)
    candidate_layer = sorted(core_set | set(optional[: target_count - 64]))
    candidate = deepcopy(ids)
    candidate[args.layer] = candidate_layer
    validate_core_inclusion(candidate, core)
    if len(candidate_layer) != target_count or sum(map(len, candidate)) != 3850:
        raise AssertionError("candidate footprint changed")
    reference_hits = sum(
        int(counts[layer, expert])
        for layer, layer_ids in enumerate(ids)
        for expert in layer_ids
    )
    candidate_hits = sum(
        int(counts[layer, expert])
        for layer, layer_ids in enumerate(candidate)
        for expert in layer_ids
    )
    hit_cap = int(reference.get("resident_hit_count_cap", 3_245_387))
    if reference_hits > hit_cap or candidate_hits > hit_cap:
        raise ValueError("reference or candidate exceeds the train resident-hit cap")
    removed = sorted(set(ids[args.layer]) - set(candidate_layer))
    added = sorted(set(candidate_layer) - set(ids[args.layer]))
    plan = deepcopy(reference)
    plan.update(
        {
            "allocation_method": "frequency_core_locked_single_layer_damage_v1",
            "allocation_objective": args.metric,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "resident_expert_ids_by_layer": candidate,
            "resident_ids_sha256": _canonical_ids_sha256(candidate),
            "resident_hit_count": candidate_hits,
            "resident_hit_count_cap_verified": True,
            "frequency_core_inclusion_verified": True,
            "selection_uses_train_routes_only": True,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        }
    )
    result = {
        "schema": SCHEMA,
        "layer": args.layer,
        "metric": args.metric,
        "reference_allocation_sha256": sha256_file(args.reference_allocation),
        "damage_result_sha256": sha256_file(args.damage_result),
        "reference_membership_sha256": _canonical_ids_sha256(ids),
        "candidate_membership_sha256": _canonical_ids_sha256(candidate),
        "frequency_core_sha256": core_hash,
        "frequency_core_inclusion_verified": True,
        "total_resident_cells": sum(map(len, candidate)),
        "layer_resident_cells": len(candidate_layer),
        "outside_layer_membership_identical": all(
            candidate[layer] == ids[layer] for layer in range(40) if layer != args.layer
        ),
        "removed_experts": removed,
        "added_experts": added,
        "membership_overlap": len(set(ids[args.layer]) & set(candidate_layer)),
        "reference_train_resident_hits": reference_hits,
        "candidate_train_resident_hits": candidate_hits,
        "resident_hit_count_cap": hit_cap,
        "resident_hit_count_cap_verified": candidate_hits <= hit_cap,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    args.output.mkdir(parents=True)
    write_json_exclusive(args.output / "resident_allocation.json", plan)
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    write_checksums(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
