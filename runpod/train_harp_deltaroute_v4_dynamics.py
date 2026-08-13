#!/usr/bin/env python3
"""Train R0/R1/R2/joint stages of the HARP-DeltaRoute v4 world model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.anchor import LegacyHARPAnchorBridge  # noqa: E402
from harp_rtt.b31 import (  # noqa: E402
    anchor_spine_prefix_matches,
    quota_candidate_union,
)
from harp_rtt.delta import HARPDeltaConfig, HARPDeltaTeacher  # noqa: E402
from harp_rtt.deltaroute_batch import prepare_deltaroute_batch  # noqa: E402
from harp_rtt.deltaroute_metrics import paired_h2_h4_request_bootstrap  # noqa: E402
from harp_rtt.deltaroute_training import (  # noqa: E402
    DeltaRouteLoss,
    counterfactual_trajectory_loss,
    depth_balanced_node_weights,
    factual_alignment_loss,
    joint_factual_mixture_loss,
    sampled_teacher_force_mask,
    teacher_forcing_probability,
)
from harp_rtt.exact_k import cardinality_project_marginals, stable_topk  # noqa: E402
from harp_rtt.factual_branch_attention import (  # noqa: E402
    ExpertConditionedBranchAttention,
    _trainable_exact_marginals,
    parent_branch_marginals,
)
from harp_rtt.route_dynamics import (  # noqa: E402
    DeltaRouteConfig,
    DeltaRouteTrajectory,
    gather_node_horizon,
)
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.train import prepare_model_batch, runtime_static_artifacts  # noqa: E402
from harp_rtt.training import move_to_device, seed_everything, sha256_file  # noqa: E402
from runpod.aggregate_harp_deltaroute_v4_aligners import SCHEMA as ALIGNER_AGGREGATE_SCHEMA  # noqa: E402
from runpod.train_harp_delta_v3 import (  # noqa: E402
    SCHEMA as PARENT_SCHEMA,
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
    load_split,
    loader,
)


SCHEMA = "harp_deltaroute_v4_dynamics_training_v1"
RESULT_SCHEMA = "harp_deltaroute_v4_dynamics_result_v1"
STAGES = ("transition_r0", "rollout_r1", "rollout_r2", "factual_joint")
PREDECESSOR = {
    "transition_r0": None,
    "rollout_r1": "transition_r0",
    "rollout_r2": "rollout_r1",
    "factual_joint": "rollout_r2",
}
MICROBATCH_CHOICES = (32, 16, 8, 4, 2, 1)
EFFECTIVE_BATCH = 32
R0_PARENT_ALLNODE_RECALL = 0.565437
R0_REQUIRED_RECALL = R0_PARENT_ALLNODE_RECALL + 0.5 * (1.0 - R0_PARENT_ALLNODE_RECALL)
R2_PARENT_ROUTE_RECALL = 0.557170
PARENT_H2_H4_C64 = 0.914467
PARENT_H4_C64 = 0.898477


@dataclass(frozen=True)
class BatchRouteOutput:
    queries: Tensor
    scores: Tensor
    selected_ids: Tensor
    affine_queries: Tensor | None
    targets: Mapping[str, Tensor]
    counterfactual: Mapping[str, Tensor]
    anchor_scores: Tensor
    semantic: Any
    branch_mask: Tensor
    aligned: Any | None


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    for split in ("train", "tune", "development"):
        parser.add_argument(f"--{split}-index", type=Path, required=True)
        parser.add_argument(f"--{split}-corpus", type=Path, required=True)
        parser.add_argument(f"--{split}-companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--aligner-aggregate", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--aligner-initializer", type=Path)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--microbatch-size", type=int, choices=(0, *MICROBATCH_CHOICES), default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def validate_aligner_aggregate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != ALIGNER_AGGREGATE_SCHEMA:
        raise ValueError("DeltaRoute aligner aggregate schema mismatch")
    if value.get("route_dynamics_remains_authorized_if_aligner_fails") is not True:
        raise PermissionError("aligner result did not authorize route dynamics")
    for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if value.get(key) is not False:
            raise PermissionError(f"aligner aggregate violates {key}")
    return {
        "sha256": sha256_file(path),
        "selected_aligner": value.get("selected_aligner"),
        "aligner_promoted": bool(value.get("aligner_promoted")),
    }


def load_parent(
    path: Path,
    static: Any,
    partition_sha256: str,
) -> tuple[HARPDeltaTeacher, dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != PARENT_SCHEMA:
        raise ValueError("DeltaRoute dynamics parent schema mismatch")
    if value.get("stage") != "semantic" or value.get("counterfactual_budget") != "16":
        raise ValueError("DeltaRoute dynamics requires budget-16 semantic parent")
    if value.get("partition_manifest_sha256") != partition_sha256:
        raise ValueError("DeltaRoute dynamics parent partition mismatch")
    config = HARPDeltaConfig(**value["config"])
    model = HARPDeltaTeacher(
        config, static.geometry.expert_keys, static.geometry.centered_bias,
        raw_width=static.geometry.hidden_width,
        target_control_width=static.geometry.maximum_rank,
        metadata_width=8,
    )
    model.load_state_dict(value["model_state_dict"], strict=True)
    return model.requires_grad_(False).eval(), value


def build_trajectory(parent: HARPDeltaTeacher, static: Any) -> DeltaRouteTrajectory:
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
        layer_adapter_rank=8,
        free_rank=16,
        attention_heads=4,
        exact_k=parent.config.exact_k,
        dropout=0.05,
    )
    return DeltaRouteTrajectory(
        config, static.geometry.expert_keys, static.geometry.centered_bias,
        static.geometry.rank_mask,
    )


def load_predecessor(
    trajectory: DeltaRouteTrajectory,
    path: Path | None,
    stage: str,
    *,
    parent_sha256: str,
    partition_sha256: str,
) -> dict[str, Any] | None:
    expected = PREDECESSOR[stage]
    if (expected is None) != (path is None):
        raise ValueError("DeltaRoute dynamics predecessor presence mismatch")
    if path is None:
        return None
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("DeltaRoute dynamics predecessor schema mismatch")
    if value.get("stage") != expected:
        raise ValueError(f"{stage} requires completed {expected}")
    if value.get("parent_checkpoint_sha256") != parent_sha256:
        raise ValueError("DeltaRoute dynamics predecessor parent mismatch")
    if value.get("partition_manifest_sha256") != partition_sha256:
        raise ValueError("DeltaRoute dynamics predecessor partition mismatch")
    gate = value.get("gate")
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise PermissionError(f"{stage} predecessor did not pass its continuation gate")
    trajectory.load_state_dict(value["trajectory_state_dict"], strict=True)
    return value


def _budget16_mask(counterfactual: Mapping[str, Tensor]) -> Tensor:
    masks = counterfactual["budget_node_masks"].bool()
    if masks.ndim != 3 or masks.shape[1] != 3:
        raise ValueError("DeltaRoute dynamics requires nested 4/8/16 masks")
    selection = masks[:, 2]
    if bool((selection & ~counterfactual["node_mask"].bool()).any()):
        raise ValueError("budget-16 mask selects an unavailable node")
    return selection


def parent_residual_teacher_queries(
    parent_queries: Tensor,
    target_queries: Tensor,
    transition_queries: Tensor,
) -> Tensor:
    """Attach a one-step transition as a residual over the static parent."""

    if parent_queries.shape != target_queries.shape:
        raise ValueError("parent and target query trajectories differ")
    if transition_queries.shape != target_queries[..., 1:, :].shape:
        raise ValueError("one-step transition trajectory has invalid geometry")
    correction = (
        transition_queries.float() - target_queries[..., :-1, :].float()
    )
    return torch.cat(
        (parent_queries[..., :1, :].float(),
         parent_queries[..., 1:, :].float() + correction),
        dim=-2,
    )


def _parent_forward(
    *,
    host: Mapping[str, Any],
    parent: HARPDeltaTeacher,
    anchor: LegacyHARPAnchorBridge,
    trajectory: DeltaRouteTrajectory,
    runtime_static: Any,
    token_embedding: Tensor,
    input_basis: Tensor,
    rank_mask: Tensor,
    device: torch.device,
) -> tuple[Any, Mapping[str, Tensor], Mapping[str, Tensor], Tensor, Mapping[str, Tensor]]:
    batch = move_to_device(host, device)
    prepared = prepare_model_batch(batch, runtime_static)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        anchor_scores = anchor(batch=prepared)["future_router_scores"][:, :4].float()
    adapted = prepare_deltaroute_batch(
        prepared, anchor_scores=anchor_scores, token_embedding=token_embedding,
        input_basis=input_basis, rank_mask=rank_mask, config=parent.config,
    )
    targets = dict(prepared["targets"])
    targets["factual_branch_index"] = adapted.factual_branch_index
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        semantic = parent(**adapted.parent_inputs, semantic_only=True)
    counterfactual = targets["counterfactual"]
    return semantic, targets, counterfactual, anchor_scores, adapted.trajectory_inputs


def route_forward(
    *,
    stage: str,
    host: Mapping[str, Any],
    parent: HARPDeltaTeacher,
    anchor: LegacyHARPAnchorBridge,
    trajectory: DeltaRouteTrajectory,
    aligner: ExpertConditionedBranchAttention | None,
    runtime_static: Any,
    token_embedding: Tensor,
    input_basis: Tensor,
    rank_mask: Tensor,
    device: torch.device,
    teacher_probability: float = 0.0,
) -> BatchRouteOutput:
    semantic, targets, counterfactual, anchor_scores, raw = _parent_forward(
        host=host, parent=parent, anchor=anchor, trajectory=trajectory,
        runtime_static=runtime_static, token_embedding=token_embedding,
        input_basis=input_basis, rank_mask=rank_mask, device=device,
    )
    node_mask = counterfactual["node_mask"].bool()
    depth = counterfactual["depth"].long()
    valid = counterfactual["valid"].bool() & node_mask[..., None]
    target_queries = torch.where(
        valid[..., None], counterfactual["query_coordinates"].float(), 0.0
    )
    target_ids = torch.where(
        valid[..., None], counterfactual["selected_ids"].long(), 0
    )
    target_weights = torch.where(
        valid[..., None], counterfactual["selected_weights"].float(), 0.0
    )
    label_valid = valid.any(-1)
    horizon_mask = raw["available"][:, None] & host["inputs"]["tree"][
        "horizon_mask"
    ].to(device).bool()
    # All 32 causal runtime nodes participate in deployed factual evidence.
    # Budget 16 remains a counterfactual semantic-supervision mask only.
    branch_mask = horizon_mask & label_valid[:, None]
    affine: Tensor | None = None
    aligned = None

    if stage in ("transition_r0", "rollout_r1"):
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            context, _, _ = trajectory.causal_context(
                context_states=semantic.context_states, **raw
            )
            node_context = gather_node_horizon(context, depth, node_mask)
            parent_queries = gather_node_horizon(
                semantic.node_queries, depth, node_mask
            )
            if stage == "transition_r0":
                transition_queries = trajectory.dynamics.teacher_forced(
                    target_queries, target_ids, target_weights, node_context
                )
                queries = parent_residual_teacher_queries(
                    parent_queries, target_queries, transition_queries
                )
                affine = torch.cat((
                    target_queries[..., :1, :],
                    trajectory.dynamics.affine_control(target_queries),
                ), dim=-2)
            else:
                force = sampled_teacher_force_mask(
                    target_queries.shape[:-2], trajectory.config.layers,
                    teacher_probability, device=device,
                )
                rollout = trajectory.dynamics.rollout(
                    target_queries[..., 0, :], node_context,
                    trajectory.expert_keys, trajectory.centered_bias,
                    teacher_ids=target_ids, teacher_weights=target_weights,
                    teacher_force_mask=force,
                )
                queries = rollout.queries
    else:
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
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
    if stage in ("transition_r0", "rollout_r1"):
        scores = torch.einsum(
            "bnlr,ler->bnle", queries.float(), trajectory.expert_keys.float()
        ) + trajectory.centered_bias[None, None]
    selected_ids = stable_topk(scores, trajectory.config.exact_k)
    if stage == "factual_joint":
        node_scores = trajectory_output.scores
        node_marginals = _trainable_exact_marginals(
            node_scores, trajectory.config.exact_k
        )
        if aligner is None:
            raise ValueError("factual_joint requires expert-conditioned alignment")
        anchor_marginals, _, _, v3_parent_marginals = parent.core.semantic_marginals(
            semantic, anchor_scores
        )
        with torch.autocast(device_type=device.type, enabled=False):
            factual_parent_marginals = parent_branch_marginals(
                node_marginals.float(), semantic.factual_path_posterior.detach().float(),
                branch_mask, anchor_marginals, k=trajectory.config.exact_k,
            )
        factual_parent_marginals = factual_parent_marginals.clone()
        factual_parent_marginals[:, 0] = v3_parent_marginals[:, 0]
        aligned = aligner(
            anchor_scores=anchor_scores,
            anchor_marginals=anchor_marginals,
            node_marginals=node_marginals,
            posterior=semantic.factual_path_posterior.detach(),
            node_mask=branch_mask,
            tree_states=semantic.tree_states.detach(),
            context_states=semantic.context_states.detach(),
            parent_marginals=factual_parent_marginals,
        )
    return BatchRouteOutput(
        queries, scores, selected_ids, affine, targets, counterfactual,
        anchor_scores, semantic, branch_mask, aligned,
    )


def objective(
    output: BatchRouteOutput,
    trajectory: DeltaRouteTrajectory,
    *,
    stage: str,
) -> DeltaRouteLoss:
    counterfactual = output.counterfactual
    node_mask = (
        _budget16_mask(counterfactual)
        if stage == "factual_joint"
        else counterfactual["node_mask"].bool()
    )
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
    if stage != "factual_joint":
        return route
    labels = output.targets["future_selected_ids"].long()
    future_valid = output.targets["future_available"].bool()
    if future_valid.ndim == 2:
        future_valid = future_valid[..., None].expand(labels.shape[:-1])
    aligned_loss = factual_alignment_loss(
        output.aligned, output.anchor_scores, output.targets,
        pair_weight=0.1, coherence_weight=0.01,
    )
    mixture = joint_factual_mixture_loss(
        branch_scores=_deployed_branch_scores(output),
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
    return DeltaRouteLoss(
        mixture.total + 0.1 * aligned_loss.components["missing_true_pair"]
        + 0.01 * output.aligned.coherence_kl,
        {**mixture.components,
         "missing_true_pair": aligned_loss.components["missing_true_pair"],
         "coherence_kl": output.aligned.coherence_kl},
    )


def _deployed_branch_scores(output: BatchRouteOutput) -> Tensor:
    """Insert each node rollout at its depth into the full runtime grid."""

    parent_at_depth = gather_node_horizon(
        output.semantic.node_scores,
        output.counterfactual["depth"],
        output.counterfactual["node_mask"],
    )
    delta = (output.scores - parent_at_depth).permute(0, 2, 1, 3)
    depth_index = (
        output.counterfactual["depth"].long() - 1
    ).clamp(0, output.semantic.node_scores.shape[1] - 1)
    horizon_delta = torch.zeros_like(output.semantic.node_scores).scatter(
        1,
        depth_index[:, None, None, :, None].expand(
            -1, 1, delta.shape[1], -1, delta.shape[-1]
        ),
        delta[:, None],
    )
    return output.semantic.node_scores + horizon_delta


def _runtime_candidate_ids(
    output: BatchRouteOutput,
    parent: HARPDeltaTeacher,
    trajectory: DeltaRouteTrajectory,
) -> Tensor:
    """Build deployed C64 from all 32 runtime node exact-k marginals."""

    if output.aligned is not None:
        return output.aligned.candidate_ids
    anchor_marginals = parent.core.semantic_marginals(
        output.semantic, output.anchor_scores
    )[0]
    node_marginals = _trainable_exact_marginals(
        _deployed_branch_scores(output), trajectory.config.exact_k
    )
    posterior = output.semantic.factual_path_posterior.float()
    captured = torch.einsum(
        "bhn,bhlne->bhle",
        posterior[..., :-1] * output.branch_mask.float(),
        node_marginals.float(),
    )
    branch_marginals = captured + posterior[..., -1, None, None] * anchor_marginals
    branch_marginals = cardinality_project_marginals(
        branch_marginals, trajectory.config.exact_k
    )[0]
    return quota_candidate_union(
        output.anchor_scores, branch_marginals, anchor_quota=32,
        width=parent.config.candidate_width,
    ).expert_ids


def _request_ids(host: Mapping[str, Any]) -> list[str]:
    metadata = host.get("metadata")
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("request_id"), list):
        raise ValueError("DeltaRoute dynamics batch lacks request IDs")
    return [str(value) for value in metadata["request_id"]]


@torch.no_grad()
def evaluate(
    *,
    stage: str,
    trajectory: DeltaRouteTrajectory,
    aligner: ExpertConditionedBranchAttention | None,
    parent: HARPDeltaTeacher,
    anchor: LegacyHARPAnchorBridge,
    dataset: Dataset[Any],
    runtime_static: Any,
    token_embedding: Tensor,
    input_basis: Tensor,
    rank_mask: Tensor,
    device: torch.device,
    microbatch: int,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    trajectory.eval(); parent.eval(); anchor.eval()
    if aligner is not None:
        aligner.eval()
    route_cells: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    parent_cells: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    c64_cells: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    mismatch_cells: dict[str, list[Tensor]] = defaultdict(list)
    affine_hits = affine_slots = 0.0
    loss_total = loss_rows = 0.0
    for host in loader(
        dataset, batch=microbatch, shuffle=False, seed=0,
        workers=workers, device=device,
    ):
        output = route_forward(
            stage=stage, host=host, parent=parent, anchor=anchor,
            trajectory=trajectory, aligner=aligner,
            runtime_static=runtime_static, token_embedding=token_embedding,
            input_basis=input_basis, rank_mask=rank_mask, device=device,
        )
        loss = objective(output, trajectory, stage=stage)
        cf = output.counterfactual
        valid = cf["valid"].bool() & cf["node_mask"].bool()[..., None]
        target = cf["selected_ids"].long()
        route_hit = (
            target[..., None] == output.selected_ids[..., None, :]
        ).any(-1).float().mean(-1)
        parent_scores = gather_node_horizon(
            output.semantic.node_scores, cf["depth"], cf["node_mask"]
        )
        parent_ids = stable_topk(parent_scores, trajectory.config.exact_k)
        parent_hit = (
            target[..., None] == parent_ids[..., None, :]
        ).any(-1).float().mean(-1)
        if output.affine_queries is not None:
            affine_scores = torch.einsum(
                "bnlr,ler->bnle", output.affine_queries.float(),
                trajectory.expert_keys.float(),
            ) + trajectory.centered_bias[None, None]
            affine_ids = stable_topk(affine_scores, trajectory.config.exact_k)
            hits = (target[..., None] == affine_ids[..., None, :]).any(-1).float()
            affine_hits += float((hits * valid[..., None]).sum())
            affine_slots += float(valid.sum() * trajectory.config.exact_k)

        candidate_ids = _runtime_candidate_ids(output, parent, trajectory)
        factual = output.targets["future_selected_ids"].long()
        future_valid = output.targets["future_available"].bool()
        if future_valid.ndim == 2:
            future_valid = future_valid[..., None].expand(factual.shape[:-1])
        coverage = (
            factual[..., None] == candidate_ids[..., None, :]
        ).any(-1).float().mean(-1)
        requests = _request_ids(host)
        mismatch = ~anchor_spine_prefix_matches(
            host["anchor_inputs"]["mtp_spine"]["exact_prefix_hashes"],
            host["targets"]["future_prefix_hashes"],
        )
        depth = cf["depth"].long()
        for row, request in enumerate(requests):
            for horizon in (2, 3, 4):
                active_nodes = cf["node_mask"][row].bool() & (depth[row] == horizon)
                active = valid[row] & active_nodes[:, None]
                if not bool(active.any()):
                    raise ValueError(f"request row lacks H{horizon} node labels")
                route_cells[(request, horizon)].append(route_hit[row][active].cpu())
                parent_cells[(request, horizon)].append(parent_hit[row][active].cpu())
                active_future = future_valid[row, horizon - 1]
                c64_cells[(request, horizon)].append(
                    coverage[row, horizon - 1][active_future].cpu()
                )
            if bool(mismatch[row, 3]):
                active_future = future_valid[row, 3]
                mismatch_cells[request].append(coverage[row, 3][active_future].cpu())
        active_count = int(valid.sum())
        loss_total += float(loss.total) * active_count
        loss_rows += active_count

    def rows(cells: Mapping[tuple[str, int], list[Tensor]]) -> list[dict[str, Any]]:
        return [
            {"request_id": request, "horizon": horizon,
             "slot_recall_at_8": float(torch.cat(values).mean())}
            for (request, horizon), values in sorted(cells.items())
        ]

    route_rows = rows(route_cells)
    parent_rows = rows(parent_cells)
    c64_rows = rows(c64_cells)

    def horizon_mean(metric_rows: list[dict[str, Any]], horizon: int) -> float:
        values = [float(row["slot_recall_at_8"]) for row in metric_rows if row["horizon"] == horizon]
        return sum(values) / len(values)

    route_h = {h: horizon_mean(route_rows, h) for h in (2, 3, 4)}
    parent_h = {h: horizon_mean(parent_rows, h) for h in (2, 3, 4)}
    c64_h = {h: horizon_mean(c64_rows, h) for h in (2, 3, 4)}
    mismatch_h4 = (
        sum(float(torch.cat(values).mean()) for values in mismatch_cells.values())
        / len(mismatch_cells) if mismatch_cells else None
    )
    bootstrap = paired_h2_h4_request_bootstrap(
        route_rows, parent_rows, replicates=1_000, seed=42
    )
    metrics = {
        "loss": loss_total / max(1, loss_rows),
        "counterfactual_route_recall": sum(route_h.values()) / 3,
        "parent_counterfactual_route_recall": sum(parent_h.values()) / 3,
        "counterfactual_route_recall_h2": route_h[2],
        "counterfactual_route_recall_h3": route_h[3],
        "counterfactual_route_recall_h4": route_h[4],
        "parent_route_recall_h2": parent_h[2],
        "parent_route_recall_h3": parent_h[3],
        "parent_route_recall_h4": parent_h[4],
        "factual_candidate_c64_h2_h4": sum(c64_h.values()) / 3,
        "factual_candidate_c64_h2": c64_h[2],
        "factual_candidate_c64_h3": c64_h[3],
        "factual_candidate_c64_h4": c64_h[4],
        "factual_candidate_c64_h4_mismatch": mismatch_h4,
        "affine_control_route_recall": (
            affine_hits / affine_slots if affine_slots else None
        ),
        "paired_route_bootstrap_vs_parent": bootstrap,
    }
    return metrics, route_rows, parent_rows


def configure_stage_parameters(
    trajectory: DeltaRouteTrajectory,
    aligner: ExpertConditionedBranchAttention | None,
    stage: str,
) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    primary: list[tuple[str, nn.Parameter]] = []
    core: list[tuple[str, nn.Parameter]] = []
    for name, parameter in trajectory.named_parameters():
        if stage in ("transition_r0", "rollout_r1"):
            active = name.startswith(("channels.", "context.", "dynamics."))
        else:
            active = not (
                stage == "rollout_r2"
                and name.startswith(("free_coefficients.", "free_basis", "free_gate"))
            )
        parameter.requires_grad_(active)
        if active:
            delayed = name.startswith("dynamics.") and stage in (
                "rollout_r2", "factual_joint"
            )
            if delayed:
                parameter.requires_grad_(False)
                core.append((f"trajectory.{name}", parameter))
            else:
                primary.append((f"trajectory.{name}", parameter))
    if aligner is not None:
        for name, parameter in aligner.named_parameters():
            parameter.requires_grad_(stage == "factual_joint")
            if parameter.requires_grad:
                primary.append((f"aligner.{name}", parameter))
    if not primary and not core:
        raise ValueError("DeltaRoute dynamics stage selected no parameters")
    return primary, core


def _autotune(
    args: argparse.Namespace,
    **forward_kwargs: Any,
) -> tuple[int, list[dict[str, Any]]]:
    choices = (args.microbatch_size,) if args.microbatch_size else MICROBATCH_CHOICES
    trace: list[dict[str, Any]] = []
    dataset = forward_kwargs.pop("dataset")
    device: torch.device = forward_kwargs["device"]
    trajectory: DeltaRouteTrajectory = forward_kwargs["trajectory"]
    aligner = forward_kwargs["aligner"]
    for size in choices:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            host = next(iter(loader(
                dataset, batch=size, shuffle=False, seed=0,
                workers=args.num_workers, device=device,
            )))
            trajectory.zero_grad(set_to_none=True)
            if aligner is not None:
                aligner.zero_grad(set_to_none=True)
            output = route_forward(stage=args.stage, host=host, **forward_kwargs)
            objective(output, trajectory, stage=args.stage).total.backward()
            trajectory.zero_grad(set_to_none=True)
            if aligner is not None:
                aligner.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
            accepted = device.type != "cuda" or peak <= 21.0
            trace.append({"microbatch": size, "peak_reserved_gib": peak, "accepted": accepted})
            if accepted:
                return size, trace
        except torch.OutOfMemoryError:
            trace.append({"microbatch": size, "oom": True, "accepted": False})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("no DeltaRoute dynamics microbatch fits the 21-GiB 3090 budget")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def _write_checksums(output: Path) -> None:
    names = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for name in names:
            handle.write(f"{sha256_file(output / name)}  {name}\n")
        handle.flush(); os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    args.data_profile = "b2_reuse_4096"
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite DeltaRoute dynamics run {args.output}")
    if args.stage != "factual_joint" and args.aligner_initializer is not None:
        raise ValueError("aligner initializer applies only to factual_joint")
    partition = validate_partition(args.partition_manifest, args.data_profile)
    reuse = validate_reuse_split(args.reuse_split_manifest)
    aligner_record = validate_aligner_aggregate(args.aligner_aggregate)
    partition_sha = sha256_file(args.partition_manifest)
    parent_sha = sha256_file(args.parent_checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    seed_everything(args.seed, deterministic=args.deterministic)
    inner = reuse["inner_split"]
    selected = {
        "train": set(inner["training_requests"]),
        "tune": set(inner["tuning_requests"]),
        "development": None,
    }
    datasets: dict[str, Dataset[Any]] = {}
    groups: dict[str, set[str]] = {}
    for split in ("train", "tune", "development"):
        datasets[split], groups[split] = load_split(
            args, split, selected_requests=selected[split]
        )
    if groups["train"] & groups["tune"] or groups["train"] & groups["development"] or groups["tune"] & groups["development"]:
        raise PermissionError("DeltaRoute dynamics request groups overlap")
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    if static.token_embedding is None:
        raise RuntimeError("DeltaRoute dynamics requires token embeddings")
    parent, parent_record = load_parent(
        args.parent_checkpoint, static, partition_sha
    )
    parent = parent.to(device)
    trajectory = build_trajectory(parent, static).to(device)
    predecessor = load_predecessor(
        trajectory, args.initialize_from, args.stage,
        parent_sha256=parent_sha, partition_sha256=partition_sha,
    )
    aligner = None
    if args.stage == "factual_joint":
        aligner = ExpertConditionedBranchAttention(
            trajectory.expert_keys, horizons=4, exact_k=8,
            tree_width=parent.config.tree_width, hidden_width=64,
            candidate_width=64, anchor_quota=32,
        ).to(device)
        if args.aligner_initializer is not None:
            initializer = torch.load(args.aligner_initializer, map_location="cpu", weights_only=True)
            if initializer.get("stage") != "align_m1" or initializer.get("parent_checkpoint_sha256") != parent_sha:
                raise ValueError("factual_joint aligner initializer is incompatible")
            aligner.load_state_dict(initializer["aligner_state_dict"], strict=True)
            # Reuse learned reliability features without changing the selected
            # R2 C64 before the joint factual optimizer is constructed.
            with torch.no_grad():
                aligner.tau.zero_()
    anchor, anchor_provenance = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint, args.target_preprocessing, args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    anchor = anchor.to(device).requires_grad_(False).eval()
    runtime_static = runtime_static_artifacts(static, device)
    token_embedding = static.token_embedding.to(device)
    input_basis = static.geometry.input_basis.to(device)
    rank_mask = static.geometry.rank_mask.to(device)
    primary, delayed_core = configure_stage_parameters(
        trajectory, aligner, args.stage
    )
    if sum(parameter.numel() for _, parameter in (*primary, *delayed_core)) > 25_000_000:
        raise RuntimeError("DeltaRoute dynamics architecture exceeds 25M parameters")
    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "seed": args.seed,
        "source_commit": args.source_commit,
        "trajectory_config": trajectory.config.to_dict(),
        "partition_schema": partition["schema"],
        "partition_manifest_sha256": partition_sha,
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "aligner_aggregate": aligner_record,
        "parent_checkpoint_sha256": parent_sha,
        "parent_development": parent_record.get("development"),
        "predecessor_sha256": None if args.initialize_from is None else sha256_file(args.initialize_from),
        "aligner_initializer_sha256": None if args.aligner_initializer is None else sha256_file(args.aligner_initializer),
        "runtime_tree_nodes": 32,
        "factual_supervision_budget": 16,
        "allnode_transition_supervision": True,
        "trainable_names_initial": [name for name, _ in primary],
        "delayed_core_names": [name for name, _ in delayed_core],
        "trainable_parameters": sum(parameter.numel() for _, parameter in (*primary, *delayed_core)),
        "anchor_provenance": anchor_provenance,
        "optimizer_constructed": False,
        "counterfactual_model_input": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)
    common = dict(
        parent=parent, anchor=anchor, trajectory=trajectory, aligner=aligner,
        runtime_static=runtime_static, token_embedding=token_embedding,
        input_basis=input_basis, rank_mask=rank_mask, device=device,
    )
    microbatch, autotune = _autotune(
        args, dataset=datasets["train"], **common
    )
    write_json_exclusive(args.output / "MEMORY_AUTOTUNE.json", {
        "selected_microbatch": microbatch, "effective_batch": EFFECTIVE_BATCH,
        "trace": autotune, "optimizer_constructed": False,
    })
    host = next(iter(loader(
        datasets["train"], batch=1, shuffle=False, seed=0,
        workers=args.num_workers, device=device,
    )))
    with torch.no_grad():
        epoch_zero_output = route_forward(stage=args.stage, host=host, **common)
        parent_scores = gather_node_horizon(
            epoch_zero_output.semantic.node_scores,
            epoch_zero_output.counterfactual["depth"],
            epoch_zero_output.counterfactual["node_mask"],
        )
        epoch_zero = {
            "r2_parent_queries_exact": (
                None if args.stage != "rollout_r2" else bool(torch.equal(
                    epoch_zero_output.queries,
                    gather_node_horizon(
                        epoch_zero_output.semantic.node_queries,
                        epoch_zero_output.counterfactual["depth"],
                        epoch_zero_output.counterfactual["node_mask"],
                    ).float(),
                ))
            ),
            "r2_parent_scores_exact": (
                None if args.stage != "rollout_r2" else bool(torch.equal(
                    epoch_zero_output.scores, parent_scores.float()
                ))
            ),
            "runtime_tree_nodes": 32,
            "factual_supervision_budget": 16,
            "optimizer_constructed": False,
        }
        if args.stage == "factual_joint":
            unaligned = BatchRouteOutput(
                epoch_zero_output.queries, epoch_zero_output.scores,
                epoch_zero_output.selected_ids, epoch_zero_output.affine_queries,
                epoch_zero_output.targets, epoch_zero_output.counterfactual,
                epoch_zero_output.anchor_scores, epoch_zero_output.semantic,
                epoch_zero_output.branch_mask, None,
            )
            baseline_candidates = _runtime_candidate_ids(
                unaligned, parent, trajectory
            )
            epoch_zero["joint_aligner_reproduces_r2_c64"] = bool(torch.equal(
                epoch_zero_output.aligned.candidate_ids, baseline_candidates
            ))
    if args.stage == "rollout_r2" and not (
        epoch_zero["r2_parent_queries_exact"] and epoch_zero["r2_parent_scores_exact"]
    ):
        raise RuntimeError("DeltaRoute R2 epoch-zero parent reproduction failed")
    if args.stage == "factual_joint" and not epoch_zero[
        "joint_aligner_reproduces_r2_c64"
    ]:
        raise RuntimeError("DeltaRoute joint epoch-zero R2 C64 reproduction failed")
    write_json_exclusive(args.output / "EPOCH_ZERO_AUDIT.json", epoch_zero)
    if args.preflight_only:
        write_json_exclusive(args.output / "PREFLIGHT_RESULT.json", {
            **epoch_zero, "stage": args.stage, "training_started": False,
        })
        _write_checksums(args.output)
        return

    initial_parameters = [parameter for _, parameter in primary]
    if not initial_parameters:
        # R2/joint may deliberately begin with only initializer/adapters; this
        # should never be empty under the declared architecture.
        raise RuntimeError("DeltaRoute dynamics initial optimizer is empty")
    optimizer = torch.optim.AdamW(
        initial_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    write_json_exclusive(args.output / "OPTIMIZER_START.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage, "optimizer": "AdamW",
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "initial_trainable_names": [name for name, _ in primary],
        "delayed_core_unfreeze_epoch": 6 if delayed_core else None,
    })
    accumulation = EFFECTIVE_BATCH // microbatch
    total_steps = args.epochs * math.ceil(len(datasets["train"]) / EFFECTIVE_BATCH)
    optimizer_step = 0
    best_value = -math.inf
    best_epoch = 0
    best_trajectory: dict[str, Tensor] | None = None
    best_aligner: dict[str, Tensor] | None = None
    stale = 0
    core_opened = not delayed_core
    for epoch in range(1, args.epochs + 1):
        if delayed_core and not core_opened and epoch == 6:
            for _, parameter in delayed_core:
                parameter.requires_grad_(True)
            optimizer.add_param_group({
                "params": [parameter for _, parameter in delayed_core],
                "lr": args.learning_rate * 0.1,
                "weight_decay": args.weight_decay,
            })
            core_opened = True
            write_json_exclusive(args.output / "TRANSITION_CORE_UNFREEZE.json", {
                "epoch": epoch, "learning_rate": args.learning_rate * 0.1,
                "trainable_names": [name for name, _ in delayed_core],
            })
        trajectory.train(); parent.eval(); anchor.eval()
        if aligner is not None:
            aligner.train()
        optimizer.zero_grad(set_to_none=True)
        total = batches = 0.0
        train_loader = loader(
            datasets["train"], batch=microbatch, shuffle=True,
            seed=args.seed + epoch, workers=args.num_workers, device=device,
        )
        for step, host in enumerate(train_loader, start=1):
            probability = (
                teacher_forcing_probability(optimizer_step, total_steps)
                if args.stage == "rollout_r1" else 0.0
            )
            output = route_forward(
                stage=args.stage, host=host,
                teacher_probability=probability, **common,
            )
            loss = objective(output, trajectory, stage=args.stage)
            (loss.total / accumulation).backward()
            if step % accumulation == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(
                    [parameter for group in optimizer.param_groups for parameter in group["params"]],
                    args.gradient_clip,
                )
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            total += float(loss.total.detach()); batches += 1
        tune, _, _ = evaluate(
            stage=args.stage, trajectory=trajectory, aligner=aligner,
            parent=parent, anchor=anchor, dataset=datasets["tune"],
            runtime_static=runtime_static, token_embedding=token_embedding,
            input_basis=input_basis, rank_mask=rank_mask, device=device,
            microbatch=microbatch, workers=args.num_workers,
        )
        append_jsonl(args.output / "metrics.jsonl", {
            "epoch": epoch, "train_loss": total / max(1, batches),
            "tune": tune,
        })
        value = float(
            tune["counterfactual_route_recall"]
            if args.stage in ("transition_r0", "rollout_r1")
            else tune["factual_candidate_c64_h2_h4"]
        )
        if value > best_value:
            best_value, best_epoch, stale = value, epoch, 0
            best_trajectory = {
                name: tensor.detach().cpu().clone()
                for name, tensor in trajectory.state_dict().items()
            }
            best_aligner = (
                None if aligner is None else {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in aligner.state_dict().items()
                }
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_trajectory is None:
        raise RuntimeError("DeltaRoute dynamics produced no checkpoint")
    trajectory.load_state_dict(best_trajectory, strict=True)
    if aligner is not None and best_aligner is not None:
        aligner.load_state_dict(best_aligner, strict=True)
    development, route_rows, parent_rows = evaluate(
        stage=args.stage, trajectory=trajectory, aligner=aligner,
        parent=parent, anchor=anchor, dataset=datasets["development"],
        runtime_static=runtime_static, token_embedding=token_embedding,
        input_basis=input_basis, rank_mask=rank_mask, device=device,
        microbatch=microbatch, workers=args.num_workers,
    )
    _write_rows(args.output / "development_route_predictions.jsonl", route_rows)
    _write_rows(args.output / "development_parent_route_predictions.jsonl", parent_rows)

    route_recall = float(development["counterfactual_route_recall"])
    per_horizon_nonnegative = all(
        float(development[f"counterfactual_route_recall_h{h}"])
        >= float(development[f"parent_route_recall_h{h}"])
        for h in (2, 3, 4)
    )
    paired_positive = bool(
        development["paired_route_bootstrap_vs_parent"]["lower_bound_positive"]
    )
    gate: dict[str, Any]
    if args.stage == "transition_r0":
        recovery = (route_recall - R0_PARENT_ALLNODE_RECALL) / (1.0 - R0_PARENT_ALLNODE_RECALL)
        gate = {
            "route_recovery": recovery,
            "required_recovery": 0.5,
            "required_recall": R0_REQUIRED_RECALL,
            "paired_lower_bound_positive": paired_positive,
            "per_horizon_nonnegative": per_horizon_nonnegative,
            "passed": recovery >= 0.5 and paired_positive and per_horizon_nonnegative,
        }
    elif args.stage == "rollout_r1":
        if predecessor is None:
            raise RuntimeError("R1 lost its R0 predecessor")
        r0 = float(predecessor["development"]["counterfactual_route_recall"])
        retention = (route_recall - R0_PARENT_ALLNODE_RECALL) / max(
            1e-12, r0 - R0_PARENT_ALLNODE_RECALL
        )
        largest_loss = max(
            float(predecessor["development"][f"counterfactual_route_recall_h{h}"])
            - float(development[f"counterfactual_route_recall_h{h}"])
            for h in (2, 3, 4)
        )
        gate = {
            "r0_lift_retention": retention,
            "required_retention": 0.8,
            "largest_horizon_loss": largest_loss,
            "maximum_horizon_loss": 0.05,
            "passed": retention >= 0.8 and largest_loss <= 0.05,
        }
    elif args.stage == "rollout_r2":
        gate = {
            "route_recall_above_parent": route_recall > R2_PARENT_ROUTE_RECALL,
            "paired_lower_bound_positive": paired_positive,
            "c64_at_least_parent": float(development["factual_candidate_c64_h2_h4"]) >= PARENT_H2_H4_C64,
            "h4_at_least_parent": float(development["factual_candidate_c64_h4"]) >= PARENT_H4_C64,
        }
        gate["passed"] = all(gate.values())
    else:
        gate = {
            "h2_h4_c64": float(development["factual_candidate_c64_h2_h4"]),
            "required_h2_h4_c64": 0.98,
            "h4_c64": float(development["factual_candidate_c64_h4"]),
            "required_h4_c64": 0.97,
            "h4_mismatch_c64": development["factual_candidate_c64_h4_mismatch"],
            "required_h4_mismatch_c64": 0.93,
        }
        gate["passed"] = bool(
            gate["h2_h4_c64"] >= 0.98
            and gate["h4_c64"] >= 0.97
            and gate["h4_mismatch_c64"] is not None
            and float(gate["h4_mismatch_c64"]) >= 0.93
        )

    checkpoint = {
        "schema": SCHEMA,
        "stage": args.stage,
        "seed": args.seed,
        "source_commit": args.source_commit,
        "trajectory_config": trajectory.config.to_dict(),
        "trajectory_state_dict": best_trajectory,
        "aligner_state_dict": best_aligner,
        "best_epoch": best_epoch,
        "best_tune_value": best_value,
        "development": development,
        "gate": gate,
        "parent_checkpoint_sha256": parent_sha,
        "partition_manifest_sha256": partition_sha,
        "predecessor_sha256": None if args.initialize_from is None else sha256_file(args.initialize_from),
        "run_manifest_sha256": sha256_file(args.output / "run_manifest.json"),
        "runtime_tree_nodes": 32,
        "factual_supervision_budget": 16,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    checkpoint_path = args.output / "best_deltaroute_dynamics.pt"
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle); handle.flush(); os.fsync(handle.fileno())
    result = {
        "schema": RESULT_SCHEMA,
        "stage": args.stage,
        "best_epoch": best_epoch,
        "best_tune_value": best_value,
        "development": development,
        "gate": gate,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "ranker_authorized": bool(args.stage == "factual_joint" and gate["passed"]),
        "training_started": True,
        "optimizer_constructed": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    _write_checksums(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
