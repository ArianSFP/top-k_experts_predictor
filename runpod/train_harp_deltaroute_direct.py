#!/usr/bin/env python3
"""Train the direct layer-conditioned HARP-DeltaRoute translator."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import Tensor

from harp_rtt.b31 import quota_candidate_union
from harp_rtt.direct_route import DirectDeltaRouteTrajectory
from harp_rtt.deltaroute_training import (
    DeltaRouteLoss,
    counterfactual_trajectory_loss,
    depth_balanced_node_weights,
    factual_alignment_loss,
    joint_factual_mixture_loss,
)
from harp_rtt.exact_k import cardinality_project_marginals, stable_topk
from harp_rtt.factual_branch_attention import (
    FactualAlignmentOutput,
    _trainable_exact_marginals,
    parent_branch_marginals,
)
from harp_rtt.route_dynamics import DeltaRouteConfig, gather_node_horizon
import runpod.train_harp_deltaroute_v4_dynamics as baseline


STAGE = "parallel_direct"
_ZERO_AUDIT: dict[str, bool] | None = None


def build_trajectory(parent: Any, static: Any) -> DirectDeltaRouteTrajectory:
    config = DeltaRouteConfig(
        experts=parent.config.experts,
        layers=parent.config.layers,
        horizons=parent.config.horizons,
        nodes=parent.config.max_tree_nodes,
        router_rank=parent.config.router_rank,
        raw_width=static.geometry.hidden_width,
        metadata_width=8,
        latent_width=256,
        effect_width=64,
        transition_width=512,
        layer_adapter_rank=16,
        free_rank=16,
        attention_heads=4,
        exact_k=parent.config.exact_k,
        dropout=0.05,
    )
    return DirectDeltaRouteTrajectory(
        config, static.geometry.expert_keys, static.geometry.centered_bias,
        static.geometry.rank_mask,
    )


def configure_stage_parameters(
    trajectory: DirectDeltaRouteTrajectory,
    aligner: None,
    stage: str,
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    if stage != STAGE or aligner is not None:
        raise ValueError("direct-route driver owns only parallel_direct")
    primary = [
        (f"trajectory.{name}", parameter)
        for name, parameter in trajectory.named_parameters()
    ]
    for _, parameter in primary:
        parameter.requires_grad_(True)
    return primary, []


def _direct_alignment(
    *,
    trajectory_output: Any,
    semantic: Any,
    anchor_scores: Tensor,
    parent: Any,
    branch_mask: Tensor,
) -> FactualAlignmentOutput:
    anchor_marginals, _, parent_node_marginals, parent_marginals = (
        parent.core.semantic_marginals(semantic, anchor_scores)
    )
    node_marginals = _trainable_exact_marginals(
        trajectory_output.scores, parent.config.exact_k
    )
    posterior = semantic.factual_path_posterior.detach().float()
    proposed = parent_branch_marginals(
        node_marginals, posterior, branch_mask, anchor_marginals,
        k=parent.config.exact_k,
    )
    with torch.no_grad():
        zero_path = parent_branch_marginals(
            parent_node_marginals, posterior, branch_mask, anchor_marginals,
            k=parent.config.exact_k,
        )
    # Preserve the selected parent densely at initialization while retaining
    # the derivative of the new budget-16 branch mixture.
    aligned = parent_marginals.float() + proposed - zero_path
    aligned = aligned.clone()
    aligned[:, 0] = parent_marginals[:, 0]
    aligned = cardinality_project_marginals(
        aligned, parent.config.exact_k
    )[0]
    epsilon = torch.finfo(torch.float32).eps
    scores = torch.logit(aligned.clamp(epsilon, 1.0 - epsilon))
    scores = scores - scores.mean(-1, keepdim=True)
    candidates = quota_candidate_union(
        anchor_scores, aligned, anchor_quota=32,
        width=parent.config.candidate_width,
    ).expert_ids
    return FactualAlignmentOutput(
        scores=scores,
        marginals=aligned,
        correction=aligned - parent_marginals.float(),
        expert_branch_weights=None,
        coherence_kl=scores.sum() * 0.0,
        candidate_ids=candidates,
    )


def route_forward(
    *,
    stage: str,
    host: Mapping[str, Any],
    parent: Any,
    anchor: Any,
    trajectory: DirectDeltaRouteTrajectory,
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
        raise ValueError("direct-route forward received another stage")
    semantic, targets, counterfactual, anchor_scores, raw = baseline._parent_forward(
        host=host, parent=parent, anchor=anchor, trajectory=trajectory,
        runtime_static=runtime_static, token_embedding=token_embedding,
        input_basis=input_basis, rank_mask=rank_mask, device=device,
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
            **raw,
        )
    queries = gather_node_horizon(
        trajectory_output.queries, depth, node_mask
    )
    scores = gather_node_horizon(
        trajectory_output.scores, depth, node_mask
    )
    aligned = _direct_alignment(
        trajectory_output=trajectory_output,
        semantic=semantic,
        anchor_scores=anchor_scores,
        parent=parent,
        branch_mask=branch_mask,
    )
    output = baseline.BatchRouteOutput(
        queries=queries,
        scores=scores,
        selected_ids=stable_topk(scores, trajectory.config.exact_k),
        affine_queries=None,
        targets=targets,
        counterfactual=counterfactual,
        anchor_scores=anchor_scores,
        semantic=semantic,
        branch_mask=branch_mask,
        aligned=aligned,
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
            "direct_parent_queries_exact": bool(torch.equal(queries, parent_queries)),
            "direct_parent_scores_exact": bool(torch.equal(scores, parent_scores)),
            "direct_parent_c64_exact": bool(torch.equal(
                aligned.candidate_ids, parent_candidates
            )),
        }
        if not all(_ZERO_AUDIT.values()):
            raise RuntimeError("direct-route epoch-zero parent reproduction failed")
    return output


def objective(
    output: Any,
    trajectory: DirectDeltaRouteTrajectory,
    *,
    stage: str,
) -> DeltaRouteLoss:
    if stage != STAGE:
        raise ValueError("direct-route objective received another stage")
    counterfactual = output.counterfactual
    node_mask = counterfactual["node_mask"].bool()
    valid = counterfactual["valid"].bool() & node_mask[..., None]
    weights = depth_balanced_node_weights(counterfactual["depth"], node_mask)
    route = counterfactual_trajectory_loss(
        output.queries, output.scores,
        counterfactual["query_coordinates"].float(),
        counterfactual["router_logits"].float(),
        counterfactual["selected_ids"].long(), valid,
        trajectory.expert_keys,
        supervision_weights=weights,
        exact_weight=1.0, logit_weight=1.0, boundary_weight=0.2,
    )
    aligned_loss = factual_alignment_loss(
        output.aligned, output.anchor_scores, output.targets,
        pair_weight=0.1, coherence_weight=0.0,
    )
    future_valid = output.targets["future_available"].bool()
    labels = output.targets["future_selected_ids"]
    if future_valid.ndim == 2:
        future_valid = future_valid[..., None].expand(labels.shape[:-1])
    mixture = joint_factual_mixture_loss(
        branch_scores=baseline._deployed_branch_scores(output),
        anchor_scores=output.anchor_scores,
        branch_probabilities=output.semantic.factual_path_posterior.detach(),
        branch_mask=output.branch_mask,
        targets={**output.targets, "future_available": future_valid},
        aligned=output.aligned,
        counterfactual=route,
        factual_weight=1.0,
        aligned_weight=1.0,
        counterfactual_weight=1.0,
    )
    total = mixture.total + 0.1 * aligned_loss.components["missing_true_pair"]
    return DeltaRouteLoss(
        total,
        {
            **mixture.components,
            "missing_true_pair": aligned_loss.components["missing_true_pair"],
        },
    )


_ORIGINAL_WRITE_JSON = baseline.write_json_exclusive


def write_json_exclusive(path: Any, value: Mapping[str, Any]) -> None:
    record = dict(value)
    if path.name in ("EPOCH_ZERO_AUDIT.json", "PREFLIGHT_RESULT.json"):
        if _ZERO_AUDIT is None:
            raise RuntimeError("direct-route zero audit was not evaluated")
        record.update(_ZERO_AUDIT)
    _ORIGINAL_WRITE_JSON(path, record)


if __name__ == "__main__":
    baseline.STAGES = (*baseline.STAGES, STAGE)
    baseline.PREDECESSOR[STAGE] = None
    baseline.build_trajectory = build_trajectory
    baseline.configure_stage_parameters = configure_stage_parameters
    baseline.route_forward = route_forward
    baseline.objective = objective
    baseline.write_json_exclusive = write_json_exclusive
    baseline.main()
