#!/usr/bin/env python3
"""Train DeltaRoute axial dynamics with a factual-deployment-first objective."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.deltaroute_training import DeltaRouteLoss  # noqa: E402
from runpod.train_harp_deltaroute_axial_open import build_trajectory  # noqa: E402
import runpod.train_harp_deltaroute_direct as direct  # noqa: E402


STAGE = "axial_route_factual"


def factual_first_objective(
    output: Any, trajectory: Any, *, stage: str
) -> DeltaRouteLoss:
    if stage != STAGE:
        raise ValueError("factual-first objective received another stage")
    # Reuse the fully audited loss construction, then change only the declared
    # scalar objective.  Counterfactual route supervision remains a stabilizer;
    # the two losses that train the deployed H2--H4 mixture are primary.
    original_stage = direct.STAGE
    direct.STAGE = stage
    try:
        base = direct.objective(output, trajectory, stage=stage)
    finally:
        direct.STAGE = original_stage
    components = base.components
    factual = components["factual_branch_mixture"]
    aligned = components["factual_aligned_exact_set"]
    counterfactual = components["counterfactual_trajectory"]
    pair = components["missing_true_pair"]
    total = 2.0 * factual + 4.0 * aligned + 0.25 * counterfactual + 0.2 * pair
    return DeltaRouteLoss(total, {
        **components,
        "weighted_factual_branch_mixture": 2.0 * factual,
        "weighted_factual_aligned_exact_set": 4.0 * aligned,
        "weighted_counterfactual_trajectory": 0.25 * counterfactual,
        "weighted_missing_true_pair": 0.2 * pair,
    })


if __name__ == "__main__":
    direct.STAGE = STAGE
    direct.baseline.STAGES = (*direct.baseline.STAGES, STAGE)
    direct.baseline.PREDECESSOR[STAGE] = None
    direct.baseline.build_trajectory = build_trajectory
    direct.baseline.configure_stage_parameters = direct.configure_stage_parameters
    direct.baseline.route_forward = direct.route_forward
    direct.baseline.objective = factual_first_objective
    direct.baseline.write_json_exclusive = direct.write_json_exclusive
    direct.baseline.main()
