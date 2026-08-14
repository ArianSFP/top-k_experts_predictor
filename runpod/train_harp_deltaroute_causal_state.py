#!/usr/bin/env python3
"""Train the exact-current-state factual DeltaRoute head."""

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
from harp_rtt.causal_state_factual import CausalStateFactualHead  # noqa: E402
from harp_rtt.deltaroute_training import (  # noqa: E402
    DeltaRouteLoss, factual_alignment_loss,
)
from harp_rtt.exact_k import cardinality_project_marginals, stable_topk  # noqa: E402
from harp_rtt.factual_branch_attention import (  # noqa: E402
    FactualAlignmentOutput, _trainable_exact_marginals,
)
from harp_rtt.route_dynamics import DeltaRouteConfig, gather_node_horizon  # noqa: E402
import runpod.train_harp_deltaroute_v4_dynamics as baseline  # noqa: E402


STAGE = "causal_state_factual"
_ZERO_AUDIT: dict[str, bool] | None = None


def build_trajectory(parent: Any, static: Any) -> CausalStateFactualHead:
    config = DeltaRouteConfig(
        experts=parent.config.experts, layers=parent.config.layers,
        horizons=parent.config.horizons, nodes=parent.config.max_tree_nodes,
        router_rank=parent.config.router_rank,
        raw_width=static.geometry.hidden_width, metadata_width=8,
        latent_width=256, effect_width=64, transition_width=768,
        layer_adapter_rank=16, free_rank=32, attention_heads=8,
        exact_k=parent.config.exact_k, dropout=0.05,
    )
    return CausalStateFactualHead(
        config, static.geometry.expert_keys,
        state_rank=16, route_width=64, axial_blocks=2,
    )


def configure_stage_parameters(
    trajectory: CausalStateFactualHead,
    aligner: None,
    stage: str,
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    if stage != STAGE or aligner is not None:
        raise ValueError("causal-state driver owns only causal_state_factual")
    primary = [
        (f"trajectory.{name}", parameter)
        for name, parameter in trajectory.named_parameters()
    ]
    for _, parameter in primary:
        parameter.requires_grad_(True)
    return primary, []


_ORIGINAL_LOAD_SPLIT = baseline.load_split


def load_split(*args: Any, **kwargs: Any) -> tuple[Any, set[str]]:
    """Avoid loading all-node teacher payloads in the factual-only fit loop."""

    dataset, groups = _ORIGINAL_LOAD_SPLIT(*args, **kwargs)
    split = str(args[1]) if len(args) > 1 else str(kwargs.get("split"))
    if split == "train":
        if not hasattr(dataset, "base"):
            raise TypeError("counterfactual adapter lacks its causal base dataset")
        return dataset.base, groups
    return dataset, groups


def _causal_inputs(
    host: Mapping[str, Any],
    *,
    input_basis: Tensor,
    rank_mask: Tensor,
    device: torch.device,
) -> dict[str, Tensor]:
    inputs = host.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("causal-state batch lacks inputs")
    if "targets" in inputs or "counterfactual" in inputs:
        raise PermissionError("label-only data appeared under causal inputs")
    current = inputs.get("current")
    history = inputs.get("history")
    if not isinstance(current, Mapping) or not isinstance(history, Mapping):
        raise TypeError("causal-state batch lacks current/history mappings")
    roles = (
        "post_attention_residual_u",
        "post_moe_residual_xplus",
        "routed_expert_output_delta_r",
        "shared_expert_output_delta_s",
    )
    if any(role not in current for role in roles):
        raise KeyError("causal-state batch lacks one required residual role")
    states = torch.stack(
        [current[role].to(device, non_blocking=True) for role in roles], dim=1
    )
    router_input = current["normalized_target_router_input_a"].to(
        device, non_blocking=True
    )
    with torch.autocast(device_type=device.type, enabled=False):
        query = torch.einsum(
            "bld,ldr->blr", router_input.float(), input_basis.float()
        ) * rank_mask[None].float()
    return {
        "current_states": states,
        "current_queries": query,
        "history_logits": history["logits"].to(device, non_blocking=True),
        "history_selected_ids": history["selected_ids"].to(
            device, non_blocking=True
        ),
        "history_selected_weights": history["execution_weights"].to(
            device, non_blocking=True
        ),
    }


def _counterfactual_or_structural_dummy(
    host: Mapping[str, Any],
    *,
    config: Any,
) -> tuple[Mapping[str, Any], bool]:
    targets = host.get("targets")
    inputs = host.get("inputs")
    if not isinstance(targets, Mapping) or not isinstance(inputs, Mapping):
        raise TypeError("causal-state host batch lacks inputs/targets")
    if "counterfactual" in targets:
        return host, False
    tree = inputs.get("tree")
    if not isinstance(tree, Mapping):
        raise TypeError("causal-state host batch lacks tree structure")
    depth = tree["depth"].long()
    node_mask = tree["mask"].bool()
    batch, nodes = depth.shape
    dummy = {
        "depth": depth,
        "node_mask": node_mask,
        "valid": torch.zeros(batch, nodes, config.layers, dtype=torch.bool),
        "query_coordinates": torch.zeros(
            batch, nodes, config.layers, config.router_rank
        ),
        "selected_ids": torch.zeros(
            batch, nodes, config.layers, config.exact_k, dtype=torch.long
        ),
        "selected_weights": torch.zeros(
            batch, nodes, config.layers, config.exact_k
        ),
    }
    amended_targets = dict(targets)
    amended_targets["counterfactual"] = dummy
    amended = dict(host)
    amended["targets"] = amended_targets
    return amended, True


def route_forward(
    *,
    stage: str,
    host: Mapping[str, Any],
    parent: Any,
    anchor: Any,
    trajectory: CausalStateFactualHead,
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
        raise ValueError("causal-state forward received another stage")
    parent_host, structural_dummy = _counterfactual_or_structural_dummy(
        host, config=parent.config
    )
    semantic, targets, counterfactual, anchor_scores, _ = baseline._parent_forward(
        host=parent_host, parent=parent, anchor=anchor, trajectory=trajectory,
        runtime_static=runtime_static, token_embedding=token_embedding,
        input_basis=input_basis, rank_mask=rank_mask, device=device,
    )
    causal = _causal_inputs(
        host, input_basis=input_basis, rank_mask=rank_mask, device=device
    )
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        factual = trajectory(
            context_states=semantic.context_states.detach(), **causal
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
    reference_scores = parent_scores.detach().requires_grad_(True)
    zero_marginals = _trainable_exact_marginals(
        reference_scores, parent.config.exact_k
    ).detach()
    proposed = cardinality_project_marginals(
        parent_marginals + (proposed_marginals - zero_marginals),
        parent.config.exact_k,
    )[0]
    with torch.no_grad():
        zero = cardinality_project_marginals(
            parent_marginals, parent.config.exact_k
        )[0]
    marginals = parent_marginals + (proposed - zero)
    candidates = quota_candidate_union(
        anchor_scores, marginals, anchor_quota=32,
        width=parent.config.candidate_width,
    ).expert_ids
    aligned = FactualAlignmentOutput(
        scores=scores, marginals=marginals,
        correction=marginals - parent_marginals,
        expert_branch_weights=semantic.factual_path_posterior[:, :, None, None],
        coherence_kl=scores.sum() * 0.0,
        candidate_ids=candidates,
    )

    depth = counterfactual["depth"].long()
    node_mask = counterfactual["node_mask"].bool()
    node_scores = gather_node_horizon(
        semantic.node_scores, depth, node_mask
    ).float()
    node_queries = gather_node_horizon(
        semantic.node_queries, depth, node_mask
    ).float()
    if structural_dummy:
        # Training never evaluates counterfactual routes; preserve a valid
        # structural shape without fabricating a teacher label.
        counterfactual = dict(counterfactual)
        counterfactual["label_only_payload_loaded"] = False
    output = baseline.BatchRouteOutput(
        queries=node_queries, scores=node_scores,
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
            "causal_state_parent_scores_exact": bool(torch.equal(
                scores, parent_scores
            )),
            "causal_state_parent_marginals_exact": bool(torch.equal(
                marginals, parent_marginals
            )),
            "causal_state_parent_c64_exact": bool(torch.equal(
                candidates, parent_candidates
            )),
        }
        if not all(_ZERO_AUDIT.values()):
            raise RuntimeError(
                f"causal-state epoch-zero reproduction failed: {_ZERO_AUDIT}"
            )
    return output


def objective(
    output: Any,
    trajectory: CausalStateFactualHead,
    *,
    stage: str,
) -> DeltaRouteLoss:
    del trajectory
    if stage != STAGE:
        raise ValueError("causal-state objective received another stage")
    return factual_alignment_loss(
        output.aligned, output.anchor_scores, output.targets,
        pair_weight=0.1, coherence_weight=0.0,
    )


_ORIGINAL_WRITE_JSON = baseline.write_json_exclusive


def write_json_exclusive(path: Any, value: Mapping[str, Any]) -> None:
    record = dict(value)
    if path.name in ("EPOCH_ZERO_AUDIT.json", "PREFLIGHT_RESULT.json"):
        if _ZERO_AUDIT is None:
            raise RuntimeError("causal-state zero audit was not evaluated")
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
    baseline.load_split = load_split
    baseline.main()
