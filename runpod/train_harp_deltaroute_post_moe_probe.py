#!/usr/bin/env python3
"""Run the existing-data post-MoE transition ceiling."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.content_transition import ContentTransitionConfig  # noqa: E402
from harp_rtt.future_state_adapter import FutureTargetStateAdapter  # noqa: E402
from harp_rtt.post_moe_transition import LayerSpecificPostMoeProbe  # noqa: E402
from harp_rtt.training import move_to_device  # noqa: E402
import runpod.train_harp_deltaroute_content_probe as baseline  # noqa: E402


def build_model(static: object) -> LayerSpecificPostMoeProbe:
    geometry = static.geometry  # type: ignore[attr-defined]
    config = ContentTransitionConfig(
        layers=geometry.layers, experts=geometry.experts,
        hidden_width=geometry.hidden_width,
        router_rank=geometry.maximum_rank, exact_k=8,
        latent_width=256, effect_width=64, transition_width=512,
        content_adapter_rank=8, output_adapter_rank=8, dropout=0.05,
    )
    return LayerSpecificPostMoeProbe(
        config, geometry.input_basis, geometry.expert_keys, geometry.centered_bias,
        geometry.rank_mask, state_rank=192,
    )


ORIGINAL_LOAD_SPLIT = baseline.load_split


def load_split(*args: Any, **kwargs: Any) -> tuple[FutureTargetStateAdapter, set[str]]:
    dataset, groups = ORIGINAL_LOAD_SPLIT(*args, **kwargs)
    return FutureTargetStateAdapter(
        dataset, roles=("post_moe_residual_xplus",)
    ), groups


def batch_tensors(host: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    batch = move_to_device(host, device)
    targets = batch["targets"]
    states = targets.get("future_states")
    if not isinstance(states, dict):
        raise TypeError("post-MoE probe lacks target-only future states")
    return (
        torch.cat(
            (states["post_moe_residual_xplus"], targets["future_router_inputs"]),
            dim=-1,
        ),
        targets["future_router_logits"],
        targets["future_selected_ids"], targets["future_execution_weights"],
        targets["future_available"],
    )


if __name__ == "__main__":
    baseline.SCHEMA = "harp_deltaroute_v4_post_moe_transition_probe_v1"
    baseline.RESULT_SCHEMA = "harp_deltaroute_v4_post_moe_transition_probe_result_v1"
    baseline.build_model = build_model
    baseline.load_split = load_split
    baseline.batch_tensors = batch_tensors
    baseline.main()
