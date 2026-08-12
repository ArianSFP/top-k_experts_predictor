"""Request-grouped promotion metrics for HARP-DeltaRoute v4."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def paired_h2_h4_request_bootstrap(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired complete-request bootstrap over the branch horizons only."""

    if replicates < 1_000:
        raise ValueError("DeltaRoute promotion requires at least 1000 bootstraps")

    def table(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray[Any, Any]]:
        values: dict[str, dict[int, float]] = defaultdict(dict)
        for row in rows:
            horizon = int(row["horizon"])
            if horizon not in (2, 3, 4):
                continue
            request = str(row["request_id"])
            if horizon in values[request]:
                raise ValueError(f"duplicate request/H{horizon} DeltaRoute metric")
            values[request][horizon] = float(row["slot_recall_at_8"])
        if not values or any(set(item) != {2, 3, 4} for item in values.values()):
            raise ValueError("DeltaRoute bootstrap requires complete H2-H4 requests")
        return {
            request: np.asarray([item[h] for h in (2, 3, 4)], dtype=np.float64)
            for request, item in values.items()
        }

    candidate = table(candidate_rows)
    baseline = table(baseline_rows)
    if set(candidate) != set(baseline):
        raise ValueError("DeltaRoute paired bootstrap request sets differ")
    request_ids = sorted(candidate)
    candidate_values = np.stack([candidate[key] for key in request_ids])
    baseline_values = np.stack([baseline[key] for key in request_ids])
    per_horizon = candidate_values - baseline_values
    differences = per_horizon.mean(1)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        chosen = rng.integers(0, len(differences), size=len(differences))
        samples[index] = differences[chosen].mean()
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        "schema": "harp_deltaroute_v4_paired_h2_h4_bootstrap_v1",
        "requests": len(request_ids),
        "replicates": replicates,
        "seed": seed,
        "candidate_point_estimate": float(candidate_values.mean()),
        "baseline_point_estimate": float(baseline_values.mean()),
        "paired_gain": float(differences.mean()),
        "per_horizon_gain_h2_h3_h4": [float(value) for value in per_horizon.mean(0)],
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "lower_bound_positive": float(lower) > 0.0,
    }


__all__ = ["paired_h2_h4_request_bootstrap"]
