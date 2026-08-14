#!/usr/bin/env python3
"""Train the layer-specific direct HARP-DeltaRoute translator."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.layer_specific_direct import (  # noqa: E402
    LayerSpecificDirectDeltaRouteTrajectory,
)
from harp_rtt.route_dynamics import DeltaRouteConfig  # noqa: E402
import runpod.train_harp_deltaroute_direct as direct  # noqa: E402


STAGE = "parallel_direct_layerspecific"


def build_trajectory(parent: Any, static: Any) -> LayerSpecificDirectDeltaRouteTrajectory:
    config = DeltaRouteConfig(
        experts=parent.config.experts, layers=parent.config.layers,
        horizons=parent.config.horizons, nodes=parent.config.max_tree_nodes,
        router_rank=parent.config.router_rank,
        raw_width=static.geometry.hidden_width, metadata_width=8,
        latent_width=256, effect_width=64, transition_width=512,
        layer_adapter_rank=16, free_rank=16, attention_heads=4,
        exact_k=parent.config.exact_k, dropout=0.05,
    )
    return LayerSpecificDirectDeltaRouteTrajectory(
        config, static.geometry.expert_keys, static.geometry.centered_bias,
        static.geometry.rank_mask, raw_rank=32,
    )


if __name__ == "__main__":
    direct.STAGE = STAGE
    direct.baseline.STAGES = (*direct.baseline.STAGES, STAGE)
    direct.baseline.PREDECESSOR[STAGE] = None
    direct.baseline.build_trajectory = build_trajectory
    direct.baseline.configure_stage_parameters = direct.configure_stage_parameters
    direct.baseline.route_forward = direct.route_forward
    direct.baseline.objective = direct.objective
    direct.baseline.write_json_exclusive = direct.write_json_exclusive
    direct.baseline.main()
