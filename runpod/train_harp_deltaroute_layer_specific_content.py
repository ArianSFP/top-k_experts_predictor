#!/usr/bin/env python3
"""Run the layer-specific HARP-DeltaRoute full-state ceiling."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.content_transition import ContentTransitionConfig  # noqa: E402
from harp_rtt.layer_specific_content import LayerSpecificFullStateProbe  # noqa: E402
import runpod.train_harp_deltaroute_content_probe as baseline  # noqa: E402


def build_model(static: object) -> LayerSpecificFullStateProbe:
    geometry = static.geometry  # type: ignore[attr-defined]
    config = ContentTransitionConfig(
        layers=geometry.layers, experts=geometry.experts,
        hidden_width=geometry.hidden_width,
        router_rank=geometry.maximum_rank, exact_k=8,
        latent_width=256, effect_width=64, transition_width=512,
        content_adapter_rank=8, output_adapter_rank=8, dropout=0.05,
    )
    return LayerSpecificFullStateProbe(
        config, geometry.input_basis, geometry.expert_keys,
        geometry.centered_bias, geometry.rank_mask, content_rank=128,
    )


if __name__ == "__main__":
    baseline.SCHEMA = "harp_deltaroute_v4_layer_specific_content_probe_v1"
    baseline.RESULT_SCHEMA = "harp_deltaroute_v4_layer_specific_content_probe_result_v1"
    baseline.build_model = build_model
    baseline.main()
