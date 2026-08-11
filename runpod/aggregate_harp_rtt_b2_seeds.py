#!/usr/bin/env python3
"""Aggregate three frozen B2 translator seeds with request-paired bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "harp_rtt_b2_translator_gate_v1"
SEEDS = (42, 43, 44)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in SEEDS:
        parser.add_argument(f"--seed-{seed}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def request_bootstrap(
    request_ids: np.ndarray,
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    unique = np.unique(request_ids)
    per_request = np.asarray(
        [values[request_ids == request].mean() for request in unique], dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = generator.integers(0, len(unique), size=len(unique))
        draws[index] = per_request[sampled].mean()
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "requests": int(len(unique)),
        "replicates": int(replicates),
        "seed": int(seed),
        "point": float(per_request.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "lower_bound_positive": bool(lower > 0),
    }


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B2 gate output {output}")
    if args.bootstrap_replicates != 1000 or args.bootstrap_seed != 42:
        raise ValueError("formal B2 gate requires 1000 request bootstraps with seed 42")
    directories = {seed: getattr(args, f"seed_{seed}").resolve() for seed in SEEDS}
    arrays = {}
    results = {}
    for seed, directory in directories.items():
        result_path = directory / "B2_SEED_RESULT.json"
        result = json.loads(result_path.read_text())
        if result.get("schema") != "harp_rtt_b2_translator_training_v1":
            raise ValueError(f"seed {seed} result schema mismatch")
        if int(result.get("seed", -1)) != seed:
            raise ValueError("B2 seed directory/result identity mismatch")
        if (
            result.get("formal_validation_opened") is not False
            or result.get("calibration_opened") is not False
            or result.get("sealed_test_opened") is not False
            or result.get("h1_training_started") is not False
            or result.get("b3_training_started") is not False
        ):
            raise PermissionError("B2 seed result violates the promotion boundary")
        metrics_path = directory / result["probe_request_metrics"]
        if sha256_file(metrics_path) != result["probe_request_metrics_sha256"]:
            raise ValueError("B2 probe metric checksum mismatch")
        arrays[seed] = np.load(metrics_path)
        results[seed] = result

    reference = arrays[42]
    for seed in (43, 44):
        for name in ("request_ids", "source_positions", "greedy_match"):
            if not np.array_equal(reference[name], arrays[seed][name]):
                raise ValueError(f"seed {seed} probe rows differ on {name}")
        if not np.array_equal(
            reference["anchor_coverage_at_64"], arrays[seed]["anchor_coverage_at_64"]
        ) or not np.array_equal(
            reference["oracle_coverage_at_64"], arrays[seed]["oracle_coverage_at_64"]
        ):
            raise ValueError("anchor/oracle metrics changed between seeds")

    request_ids = reference["request_ids"]
    greedy = reference["greedy_match"]
    anchor = reference["anchor_coverage_at_64"]
    oracle = reference["oracle_coverage_at_64"]
    per_seed: dict[str, Any] = {}
    mean_delta_by_horizon: dict[int, np.ndarray] = {}
    for horizon in (3, 4):
        index = horizon - 1
        mismatch = ~greedy[:, index]
        denominator = float(oracle[mismatch, index].mean() - anchor[mismatch, index].mean())
        if denominator <= 0:
            raise ValueError(f"H{horizon} oracle has no positive lift over anchor")
        seed_values = {}
        deltas = []
        for seed in SEEDS:
            learned = arrays[seed]["learned_coverage_at_64"]
            delta = learned[mismatch, index] - anchor[mismatch, index]
            recovery = float(delta.mean() / denominator)
            seed_values[str(seed)] = {
                "recovery_g": recovery,
                "learned": float(learned[mismatch, index].mean()),
                "anchor": float(anchor[mismatch, index].mean()),
                "oracle": float(oracle[mismatch, index].mean()),
                "delta": float(delta.mean()),
            }
            deltas.append(delta)
        per_seed[f"H{horizon}_mismatch"] = seed_values
        mean_delta_by_horizon[horizon] = np.mean(np.stack(deltas), axis=0)

    aggregate: dict[str, Any] = {}
    gates = []
    for horizon in (3, 4):
        index = horizon - 1
        mismatch = ~greedy[:, index]
        denominator = float(oracle[mismatch, index].mean() - anchor[mismatch, index].mean())
        seed_recovery = [
            per_seed[f"H{horizon}_mismatch"][str(seed)]["recovery_g"]
            for seed in SEEDS
        ]
        mean_recovery = float(np.mean(seed_recovery))
        bootstrap = request_bootstrap(
            request_ids[mismatch],
            mean_delta_by_horizon[horizon],
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        passed = (
            mean_recovery >= 0.5
            and bootstrap["lower_bound_positive"]
            and all(value >= 0 for value in seed_recovery)
        )
        gates.append(passed)
        aggregate[f"H{horizon}_mismatch"] = {
            "mean_seed_recovery_g": mean_recovery,
            "minimum_seed_recovery_g": float(min(seed_recovery)),
            "no_seed_negative": bool(all(value >= 0 for value in seed_recovery)),
            "paired_request_bootstrap_delta": bootstrap,
            "passed": passed,
        }

    output.mkdir(parents=True)
    report = {
        "schema": SCHEMA,
        "passed": bool(all(gates)),
        "decision": (
            "promote_to_B3_generator_training"
            if all(gates)
            else "stop_before_B3_and_expand_or_revise_B2"
        ),
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "gate": {
            "mean_seed_recovery_g_minimum": 0.5,
            "paired_request_bootstrap_lower_bound_above_zero": True,
            "no_seed_negative_recovery": True,
        },
        "bindings": {
            str(seed): {
                "directory": str(directory),
                "result_sha256": sha256_file(directory / "B2_SEED_RESULT.json"),
                "metrics_sha256": results[seed]["probe_request_metrics_sha256"],
                "checkpoint_sha256": results[seed]["checkpoint_sha256"],
            }
            for seed, directory in directories.items()
        },
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "h1_training_started": False,
        "b3_training_started": False,
    }
    report_path = output / "B2_TRANSLATOR_GATE_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "SHA256SUMS").write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
