#!/usr/bin/env python3
"""Train the expert-conditioned tree-global factual HARP head."""

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
from harp_rtt.candidate_boundary import candidate_entry_loss  # noqa: E402
from harp_rtt.deltaroute_training import (  # noqa: E402
    DeltaRouteLoss, factual_alignment_loss,
)
from harp_rtt.exact_k import cardinality_project_marginals, stable_topk  # noqa: E402
from harp_rtt.factual_branch_attention import (  # noqa: E402
    FactualAlignmentOutput, _trainable_exact_marginals,
)
from harp_rtt.route_dynamics import DeltaRouteConfig, gather_node_horizon  # noqa: E402
from harp_rtt.tree_global_factual import TreeGlobalFactualHead  # noqa: E402
import runpod.train_harp_deltaroute_v4_dynamics as baseline  # noqa: E402


STAGE = "tree_global_factual"
_ZERO_AUDIT: dict[str, bool] | None = None


def build_trajectory(parent: Any, static: Any) -> TreeGlobalFactualHead:
    config = DeltaRouteConfig(
        experts=parent.config.experts, layers=parent.config.layers,
        horizons=parent.config.horizons, nodes=parent.config.max_tree_nodes,
        router_rank=parent.config.router_rank,
        raw_width=static.geometry.hidden_width, metadata_width=8,
        latent_width=256, effect_width=64, transition_width=512,
        layer_adapter_rank=16, free_rank=16, attention_heads=4,
        exact_k=parent.config.exact_k, dropout=0.05,
    )
    return TreeGlobalFactualHead(config, static.geometry.expert_keys)


def configure_stage_parameters(
    trajectory: TreeGlobalFactualHead,
    aligner: None,
    stage: str,
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    if stage != STAGE or aligner is not None:
        raise ValueError("tree-global driver owns only tree_global_factual")
    primary = [
        (f"trajectory.{name}", parameter)
        for name, parameter in trajectory.named_parameters()
    ]
    for _, parameter in primary:
        parameter.requires_grad_(True)
    return primary, []


def route_forward(
    *,
    stage: str,
    host: Mapping[str, Any],
    parent: Any,
    anchor: Any,
    trajectory: TreeGlobalFactualHead,
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
        raise ValueError("tree-global forward received another stage")
    semantic, targets, counterfactual, anchor_scores, raw = baseline._parent_forward(
        host=host, parent=parent, anchor=anchor, trajectory=trajectory,
        runtime_static=runtime_static, token_embedding=token_embedding,
        input_basis=input_basis, rank_mask=rank_mask, device=device,
    )
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        factual = trajectory(
            context_states=semantic.context_states,
            posterior=semantic.factual_path_posterior.detach(),
            **raw,
        )
    parent_marginals = parent.core.semantic_marginals(
        semantic, anchor_scores
    )[3].detach().float()
    epsilon = torch.finfo(torch.float32).eps
    parent_scores = torch.logit(parent_marginals.clamp(epsilon, 1.0 - epsilon))
    parent_scores = parent_scores - parent_scores.mean(-1, keepdim=True)
    scores = parent_scores + factual.score_delta.float()
    proposed_marginals = _trainable_exact_marginals(
        scores, parent.config.exact_k
    )
    with torch.no_grad():
        zero_marginals = _trainable_exact_marginals(
            parent_scores, parent.config.exact_k
        )
    proposed = cardinality_project_marginals(
        parent_marginals + proposed_marginals - zero_marginals,
        parent.config.exact_k,
    )[0]
    with torch.no_grad():
        zero = cardinality_project_marginals(
            parent_marginals, parent.config.exact_k
        )[0]
    marginals = parent_marginals + proposed - zero
    candidates = quota_candidate_union(
        anchor_scores, marginals, anchor_quota=32,
        width=parent.config.candidate_width,
    ).expert_ids
    aligned = FactualAlignmentOutput(
        scores=scores, marginals=marginals,
        correction=marginals - parent_marginals,
        expert_branch_weights=factual.expert_branch_weights,
        coherence_kl=scores.sum() * 0.0,
        candidate_ids=candidates,
    )
    depth = counterfactual["depth"].long()
    node_mask = counterfactual["node_mask"].bool()
    queries = gather_node_horizon(semantic.node_queries, depth, node_mask).float()
    node_scores = gather_node_horizon(semantic.node_scores, depth, node_mask).float()
    output = baseline.BatchRouteOutput(
        queries=queries, scores=node_scores,
        selected_ids=stable_topk(node_scores, parent.config.exact_k),
        affine_queries=None, targets=targets, counterfactual=counterfactual,
        anchor_scores=anchor_scores, semantic=semantic,
        branch_mask=torch.zeros(
            node_scores.shape[0], parent.config.horizons,
            parent.config.max_tree_nodes, dtype=torch.bool, device=device,
        ),
        aligned=aligned,
    )
    global _ZERO_AUDIT
    if _ZERO_AUDIT is None:
        parent_candidates = quota_candidate_union(
            anchor_scores, parent_marginals, anchor_quota=32,
            width=parent.config.candidate_width,
        ).expert_ids
        _ZERO_AUDIT = {
            "tree_global_parent_scores_exact": bool(torch.equal(
                scores, parent_scores
            )),
            "tree_global_parent_marginals_exact": bool(torch.equal(
                marginals, parent_marginals
            )),
            "tree_global_parent_c64_exact": bool(torch.equal(
                candidates, parent_candidates
            )),
        }
        if not all(_ZERO_AUDIT.values()):
            raise RuntimeError(
                f"tree-global epoch-zero parent reproduction failed: {_ZERO_AUDIT}"
            )
    return output


def objective(
    output: Any,
    trajectory: TreeGlobalFactualHead,
    *,
    stage: str,
) -> DeltaRouteLoss:
    del trajectory
    if stage != STAGE:
        raise ValueError("tree-global objective received another stage")
    factual = factual_alignment_loss(
        output.aligned, output.anchor_scores, output.targets,
        pair_weight=0.1, coherence_weight=0.0,
    )
    labels = output.targets["future_selected_ids"].long()
    valid = output.targets["future_available"].bool()
    if valid.ndim == 2:
        valid = valid[..., None].expand(labels.shape[:-1])
    valid = valid.clone(); valid[:, 0] = False
    entry = candidate_entry_loss(
        output.aligned.scores, output.anchor_scores, labels, valid,
        anchor_quota=32, candidate_width=64, margin=0.125,
    )
    total = factual.total + 2.0 * entry
    return DeltaRouteLoss(total, {
        **factual.components, "candidate_entry": entry,
        "weighted_candidate_entry": 2.0 * entry,
    })


_ORIGINAL_WRITE_JSON = baseline.write_json_exclusive


def write_json_exclusive(path: Any, value: Mapping[str, Any]) -> None:
    record = dict(value)
    if path.name in ("EPOCH_ZERO_AUDIT.json", "PREFLIGHT_RESULT.json"):
        if _ZERO_AUDIT is None:
            raise RuntimeError("tree-global zero audit was not evaluated")
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
