#!/usr/bin/env python3
"""Train DeltaRoute axial dynamics against the deployed C64 boundary."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.candidate_boundary import candidate_entry_loss  # noqa: E402
from harp_rtt.deltaroute_training import DeltaRouteLoss  # noqa: E402
from runpod.train_harp_deltaroute_axial_open import build_trajectory  # noqa: E402
import runpod.train_harp_deltaroute_direct as direct  # noqa: E402


STAGE = "axial_route_candidate"


def candidate_first_objective(
    output: Any, trajectory: Any, *, stage: str
) -> DeltaRouteLoss:
    if stage != STAGE:
        raise ValueError("candidate-first objective received another stage")
    base = direct.objective(output, trajectory, stage=stage)
    labels = output.targets["future_selected_ids"].long()
    valid = output.targets["future_available"].bool()
    if valid.ndim == 2:
        valid = valid[..., None].expand(labels.shape[:-1])
    valid = valid.clone(); valid[:, 0] = False
    entry = candidate_entry_loss(
        output.aligned.scores, output.anchor_scores, labels, valid,
        anchor_quota=32, candidate_width=64, margin=0.125,
    )
    components = base.components
    total = (
        components["factual_branch_mixture"]
        + components["factual_aligned_exact_set"]
        + 0.25 * components["counterfactual_trajectory"]
        + 8.0 * entry
    )
    return DeltaRouteLoss(total, {
        **components,
        "candidate_entry": entry,
        "weighted_candidate_entry": 8.0 * entry,
    })


if __name__ == "__main__":
    direct.STAGE = STAGE
    direct.baseline.STAGES = (*direct.baseline.STAGES, STAGE)
    direct.baseline.PREDECESSOR[STAGE] = None
    direct.baseline.build_trajectory = build_trajectory
    direct.baseline.configure_stage_parameters = direct.configure_stage_parameters
    direct.baseline.route_forward = direct.route_forward
    direct.baseline.objective = candidate_first_objective
    direct.baseline.write_json_exclusive = direct.write_json_exclusive
    direct.baseline.main()
