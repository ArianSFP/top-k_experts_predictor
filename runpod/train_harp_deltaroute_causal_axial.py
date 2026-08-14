#!/usr/bin/env python3
"""Train target-state-conditioned axial branch translation."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import Tensor

from harp_rtt.b31 import quota_candidate_union  # noqa: E402
from harp_rtt.causal_state_axial import CausalStateAxialRouteTrajectory  # noqa: E402
from harp_rtt.exact_k import stable_topk  # noqa: E402
from harp_rtt.route_dynamics import DeltaRouteConfig, gather_node_horizon  # noqa: E402
import runpod.train_harp_deltaroute_causal_state as causal  # noqa: E402
import runpod.train_harp_deltaroute_direct as direct  # noqa: E402
import runpod.train_harp_deltaroute_v4_dynamics as baseline  # noqa: E402


STAGE = "causal_state_axial"
_ZERO_AUDIT: dict[str, bool] | None = None


def build_trajectory(parent: Any, static: Any) -> CausalStateAxialRouteTrajectory:
    config = DeltaRouteConfig(
        experts=parent.config.experts, layers=parent.config.layers,
        horizons=parent.config.horizons, nodes=parent.config.max_tree_nodes,
        router_rank=parent.config.router_rank,
        raw_width=static.geometry.hidden_width, metadata_width=8,
        latent_width=256, effect_width=64, transition_width=512,
        layer_adapter_rank=16, free_rank=16, attention_heads=8,
        exact_k=parent.config.exact_k, dropout=0.05,
    )
    return CausalStateAxialRouteTrajectory(
        config, static.geometry.input_basis, static.geometry.expert_keys,
        static.geometry.centered_bias, static.geometry.rank_mask,
        raw_rank=24, visible_rank=24, route_width=64, axial_blocks=2,
    )


def route_forward(
    *,
    stage: str,
    host: Mapping[str, Any],
    parent: Any,
    anchor: Any,
    trajectory: CausalStateAxialRouteTrajectory,
    aligner: None,
    runtime_static: Any,
    token_embedding: Tensor,
    input_basis: Tensor,
    rank_mask: Tensor,
    device: torch.device,
    teacher_probability: float = 0.0,
) -> Any:
    del teacher_probability
    if stage != STAGE or aligner is not None:
        raise ValueError("causal-axial forward received another stage")
    semantic, targets, counterfactual, anchor_scores, raw = baseline._parent_forward(
        host=host, parent=parent, anchor=anchor, trajectory=trajectory,
        runtime_static=runtime_static, token_embedding=token_embedding,
        input_basis=input_basis, rank_mask=rank_mask, device=device,
    )
    current = causal._causal_inputs(
        host, input_basis=input_basis, rank_mask=rank_mask, device=device
    )
    node_mask = counterfactual["node_mask"].bool()
    depth = counterfactual["depth"].long()
    label_valid = (counterfactual["valid"].bool() & node_mask[..., None]).any(-1)
    horizon_mask = raw["available"][:, None] & host["inputs"]["tree"][
        "horizon_mask"
    ].to(device).bool()
    budget16 = baseline._budget16_mask(counterfactual)
    branch_mask = horizon_mask & label_valid[:, None] & budget16[:, None]
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        trajectory_output = trajectory(
            parent_queries=semantic.node_queries,
            parent_scores=semantic.node_scores,
            context_states=semantic.context_states,
            **raw, **current,
        )
    queries = gather_node_horizon(
        trajectory_output.queries, depth, node_mask
    )
    scores = gather_node_horizon(
        trajectory_output.scores, depth, node_mask
    )
    aligned = direct._direct_alignment(
        trajectory_output=trajectory_output, semantic=semantic,
        anchor_scores=anchor_scores, parent=parent, branch_mask=branch_mask,
    )
    output = baseline.BatchRouteOutput(
        queries=queries, scores=scores,
        selected_ids=stable_topk(scores, trajectory.config.exact_k),
        affine_queries=None, targets=targets, counterfactual=counterfactual,
        anchor_scores=anchor_scores, semantic=semantic,
        branch_mask=branch_mask, aligned=aligned,
    )
    global _ZERO_AUDIT
    if _ZERO_AUDIT is None:
        parent_queries = gather_node_horizon(
            semantic.node_queries, depth, node_mask
        ).float()
        parent_scores = gather_node_horizon(
            semantic.node_scores, depth, node_mask
        ).float()
        parent_candidates = quota_candidate_union(
            anchor_scores,
            parent.core.semantic_marginals(semantic, anchor_scores)[3],
            anchor_quota=32, width=parent.config.candidate_width,
        ).expert_ids
        _ZERO_AUDIT = {
            "causal_axial_parent_queries_exact": bool(torch.equal(
                queries, parent_queries
            )),
            "causal_axial_parent_scores_exact": bool(torch.equal(
                scores, parent_scores
            )),
            "causal_axial_parent_c64_exact": bool(torch.equal(
                aligned.candidate_ids, parent_candidates
            )),
        }
        if not all(_ZERO_AUDIT.values()):
            raise RuntimeError(
                f"causal-axial epoch-zero reproduction failed: {_ZERO_AUDIT}"
            )
    return output


_ORIGINAL_WRITE_JSON = baseline.write_json_exclusive


def write_json_exclusive(path: Any, value: Mapping[str, Any]) -> None:
    record = dict(value)
    if path.name in ("EPOCH_ZERO_AUDIT.json", "PREFLIGHT_RESULT.json"):
        if _ZERO_AUDIT is None:
            raise RuntimeError("causal-axial zero audit was not evaluated")
        record.update(_ZERO_AUDIT)
    _ORIGINAL_WRITE_JSON(path, record)


if __name__ == "__main__":
    direct.STAGE = STAGE
    baseline.STAGES = (*baseline.STAGES, STAGE)
    baseline.PREDECESSOR[STAGE] = None
    baseline.build_trajectory = build_trajectory
    baseline.configure_stage_parameters = direct.configure_stage_parameters
    baseline.route_forward = route_forward
    baseline.objective = direct.objective
    baseline.write_json_exclusive = write_json_exclusive
    baseline.main()
