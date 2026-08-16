#!/usr/bin/env python3
"""Build a globally allocated, hash-bound RouteQuant bit schedule.

The input screen must contain train-only per-expert upgrade utilities for every
router transition (layers 0..38). Layer 39 is forced to the upgrade precision
because its output drives Shadow-LM path likelihood and has no next-router
transition on which to estimate the same utility.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch


SCHEMA = "harp_routequant_schedule_v1"
INPUT_SCHEMA = "harp_routequant_representative_screen_v1"
LAYERS = 40
EXPERTS = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_global_upgrade_scores(
    screen: Path,
    *,
    base_bits: int,
    upgrade_bits: int,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, str]]:
    manifest_path = screen / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError("RouteQuant screen schema mismatch")
    if manifest.get("mixed_utility_split") != "train":
        raise ValueError("RouteQuant schedule requires train-only utility")
    if manifest.get("development_opened_for_metrics") is not False:
        raise ValueError("development metrics cannot select RouteQuant schedules")
    if manifest.get("formal_validation_opened") is not False or manifest.get(
        "sealed_test_opened"
    ) is not False:
        raise ValueError("sealed data cannot select RouteQuant schedules")
    if manifest.get("mixed_base_bits") != base_bits or manifest.get(
        "mixed_upgrade_bits"
    ) != upgrade_bits:
        raise ValueError("RouteQuant screen bit transition mismatch")
    if upgrade_bits != base_bits + 1:
        raise ValueError("RouteQuant schedules support adjacent bit upgrades only")
    if sorted(manifest.get("layers", [])) != list(range(39)):
        raise ValueError("global RouteQuant schedule requires layers 0..38")

    scores = torch.full((LAYERS, EXPERTS), -math.inf, dtype=torch.float64)
    input_hashes = {"run_manifest.json": sha256_file(manifest_path)}
    for layer in range(39):
        path = screen / f"layer_{layer:02d}_train_utility.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("layer") != layer:
            raise ValueError(f"RouteQuant utility layer mismatch at {layer}")
        utility = torch.as_tensor(value.get("score"), dtype=torch.float64)
        occurrences = torch.as_tensor(value.get("occurrences"), dtype=torch.int64)
        if utility.shape != (EXPERTS,) or occurrences.shape != (EXPERTS,):
            raise ValueError(f"RouteQuant utility shape mismatch at layer {layer}")
        observed = occurrences > 0
        if not torch.isfinite(utility[observed]).all():
            raise ValueError(f"non-finite observed utility at layer {layer}")
        scores[layer, observed] = utility[observed]
        input_hashes[path.name] = sha256_file(path)
    scores[39] = math.inf
    return manifest, scores, input_hashes


def allocate_global_schedule(
    scores: torch.Tensor,
    *,
    base_bits: int,
    upgrade_bits: int,
    upgrade_fraction: float,
) -> torch.Tensor:
    values = torch.as_tensor(scores, dtype=torch.float64).cpu()
    if values.shape != (LAYERS, EXPERTS):
        raise ValueError("RouteQuant global utility must be [40,256]")
    if upgrade_bits != base_bits + 1 or base_bits not in (1, 2, 3):
        raise ValueError("RouteQuant schedule requires adjacent widths in 1..4")
    if not 0.0 < upgrade_fraction < 1.0:
        raise ValueError("RouteQuant global upgrade fraction must lie in (0,1)")
    total = round(LAYERS * EXPERTS * upgrade_fraction)
    if total < EXPERTS:
        raise ValueError("global budget cannot cover the forced final layer")

    candidates = [
        (float(values[layer, expert]), layer, expert)
        for layer in range(39)
        for expert in range(EXPERTS)
        if math.isfinite(float(values[layer, expert]))
    ]
    remaining = total - EXPERTS
    if len(candidates) < remaining:
        raise ValueError("insufficient observed experts for requested upgrade budget")
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    schedule = torch.full((LAYERS, EXPERTS), base_bits, dtype=torch.int8)
    schedule[39] = upgrade_bits
    for _, layer, expert in candidates[:remaining]:
        schedule[layer, expert] = upgrade_bits
    if int((schedule == upgrade_bits).sum()) != total:
        raise AssertionError("RouteQuant global schedule cardinality drift")
    return schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--base-bits", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--upgrade-bits", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--upgrade-fraction", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    manifest, scores, input_hashes = load_global_upgrade_scores(
        args.screen, base_bits=args.base_bits, upgrade_bits=args.upgrade_bits
    )
    schedule = allocate_global_schedule(
        scores,
        base_bits=args.base_bits,
        upgrade_bits=args.upgrade_bits,
        upgrade_fraction=args.upgrade_fraction,
    )
    value: Mapping[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "parent_screen_source_commit": manifest["source_commit"],
        "parent_screen_path": str(args.screen),
        "target_checkpoint_index_sha256": manifest[
            "target_checkpoint_index_sha256"
        ],
        "partition_manifest_sha256": manifest["partition_manifest_sha256"],
        "reuse_split_manifest_sha256": manifest["reuse_split_manifest_sha256"],
        "utility_source": "training_only_first_order_next_router_plus_exact_residual",
        "allocation": "global_stable_marginal_utility",
        "base_bits": args.base_bits,
        "upgrade_bits": args.upgrade_bits,
        "requested_upgrade_fraction": args.upgrade_fraction,
        "actual_upgrade_count": int((schedule == args.upgrade_bits).sum()),
        "actual_upgrade_fraction": float(
            (schedule == args.upgrade_bits).sum().item() / schedule.numel()
        ),
        "layer_39_policy": "forced_upgrade_for_shadow_lm",
        "bit_widths": schedule.tolist(),
        "input_sha256": input_hashes,
        "optimizer_constructed": False,
        "training_started": False,
        "development_opened_for_metrics": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(dict(value), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
