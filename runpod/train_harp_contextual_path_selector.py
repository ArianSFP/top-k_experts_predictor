#!/usr/bin/env python3
"""Train the high-headroom context-conditioned factual branch selector."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from harp_rtt.b31 import anchor_spine_prefix_matches  # noqa: E402
from harp_rtt.contextual_path_selector import (  # noqa: E402
    ContextualFactualPathSelector,
)
from harp_rtt.deltaroute_metrics import paired_h2_h4_request_bootstrap  # noqa: E402
from harp_rtt.exact_k import cardinality_project_marginals, stable_topk  # noqa: E402
from harp_rtt.route_dynamics import (  # noqa: E402
    DeltaRouteConfig, gather_node_horizon,
)
import runpod.train_harp_deltaroute_v4_dynamics as baseline  # noqa: E402


STAGE = "contextual_path"


def build_trajectory(parent: Any, static: Any) -> ContextualFactualPathSelector:
    config = DeltaRouteConfig(
        experts=parent.config.experts, layers=parent.config.layers,
        horizons=parent.config.horizons, nodes=parent.config.max_tree_nodes,
        router_rank=parent.config.router_rank,
        raw_width=static.geometry.hidden_width, metadata_width=8,
        latent_width=parent.config.tree_width, effect_width=64,
        transition_width=768, layer_adapter_rank=8, free_rank=8,
        attention_heads=8, exact_k=parent.config.exact_k, dropout=0.05,
    )
    return ContextualFactualPathSelector(config, blocks=2)


def configure_stage_parameters(
    trajectory: ContextualFactualPathSelector,
    aligner: None,
    stage: str,
) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    if stage != STAGE or aligner is not None:
        raise ValueError("contextual-path driver owns only contextual_path")
    values = [
        (f"trajectory.{name}", parameter)
        for name, parameter in trajectory.named_parameters()
    ]
    for _, parameter in values: parameter.requires_grad_(True)
    return values, []


def _budget16(counterfactual: Mapping[str, Tensor]) -> Tensor:
    masks = counterfactual["budget_node_masks"].bool()
    if masks.ndim != 3 or masks.shape[1] < 3:
        raise ValueError("contextual path requires nested 4/8/16/all masks")
    return masks[:, 2]


def _visible_and_labels(
    host: Mapping[str, Any], counterfactual: Mapping[str, Tensor],
    factual: Tensor, *, device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    tree = host["inputs"]["tree"]
    available = tree["mask"].to(device).bool()
    structural = tree["horizon_mask"].to(device).bool()
    label_valid = (
        counterfactual["valid"].bool()
        & counterfactual["node_mask"].bool()[..., None]
    ).any(-1)
    visible = structural & available[:, None] & label_valid[:, None]
    visible = visible & _budget16(counterfactual)[:, None]
    nodes = visible.shape[-1]
    safe = factual.long().clamp(0, nodes - 1)
    present = visible.gather(-1, safe[..., None]).squeeze(-1) & (factual < nodes)
    labels = torch.where(present, factual.long(), torch.full_like(factual, nodes))
    labels[:, 0] = 0
    return visible, labels, available


def _target_distribution(
    counterfactual: Mapping[str, Tensor], visible: Tensor
) -> Tensor | None:
    target = counterfactual.get("target_path_distribution")
    if target is None: return None
    target = target.float()
    if target.shape != visible.shape[:-1] + (visible.shape[-1] + 1,):
        raise ValueError("target path posterior has invalid geometry")
    captured = target[..., :-1] * visible.float()
    other = 1.0 - captured.sum(-1)
    result = torch.cat((captured, other.clamp_min(0.0)[..., None]), -1)
    return result / result.sum(-1, keepdim=True).clamp_min(1e-12)


def route_forward(
    *, stage: str, host: Mapping[str, Any], parent: Any, anchor: Any,
    trajectory: ContextualFactualPathSelector, aligner: None,
    runtime_static: Any, token_embedding: Tensor, input_basis: Tensor,
    rank_mask: Tensor, device: torch.device,
    teacher_probability: float = 0.0,
) -> Any:
    del teacher_probability
    if stage != STAGE or aligner is not None:
        raise ValueError("contextual-path forward received another stage")
    semantic, targets, counterfactual, anchor_scores, _ = baseline._parent_forward(
        host=host, parent=parent, anchor=anchor, trajectory=trajectory,
        runtime_static=runtime_static, token_embedding=token_embedding,
        input_basis=input_basis, rank_mask=rank_mask, device=device,
    )
    visible, labels, available = _visible_and_labels(
        host, counterfactual, targets["factual_branch_index"], device=device
    )
    parent_posterior = semantic.factual_path_posterior.detach().float()
    captured_parent = torch.where(
        visible, parent_posterior[..., :-1],
        torch.zeros_like(parent_posterior[..., :-1]),
    )
    parent_posterior = torch.cat((
        captured_parent,
        (1.0 - captured_parent.sum(-1)).clamp_min(0.0)[..., None],
    ), dim=-1)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        selected = trajectory(
            tree_states=semantic.tree_states,
            context_states=semantic.context_states,
            base_probabilities=parent_posterior,
            horizon_mask=visible, node_available=available,
        )
    semantic = replace(
        semantic, factual_path_logits=selected.logits,
        factual_path_posterior=selected.probabilities,
    )
    targets = dict(targets)
    targets["factual_branch_index_deployed"] = labels
    target_posterior = _target_distribution(counterfactual, visible)
    if target_posterior is not None:
        targets["target_path_distribution_deployed"] = target_posterior
    node_mask = counterfactual["node_mask"].bool()
    depth = counterfactual["depth"].long()
    node_scores = gather_node_horizon(semantic.node_scores, depth, node_mask).float()
    node_queries = gather_node_horizon(semantic.node_queries, depth, node_mask).float()
    return baseline.BatchRouteOutput(
        queries=node_queries, scores=node_scores,
        selected_ids=stable_topk(node_scores, parent.config.exact_k),
        affine_queries=None, targets=targets, counterfactual=counterfactual,
        anchor_scores=anchor_scores, semantic=semantic,
        branch_mask=visible, aligned=parent_posterior,
    )


def objective(
    output: Any, trajectory: ContextualFactualPathSelector, *, stage: str,
) -> Any:
    del trajectory
    if stage != STAGE: raise ValueError("contextual-path objective received another stage")
    labels = output.targets["factual_branch_index_deployed"].long()
    logits = output.semantic.factual_path_logits.float()
    factual = F.cross_entropy(logits[:, 1:].flatten(0, 1), labels[:, 1:].flatten())
    target = output.targets.get("target_path_distribution_deployed")
    if target is None:
        posterior = factual * 0.0
    else:
        posterior = F.kl_div(
            F.log_softmax(logits[:, 1:], -1), target[:, 1:].detach().float(),
            reduction="batchmean",
        )
    return baseline.DeltaRouteLoss(
        factual + 0.2 * posterior,
        {"factual_path_ce": factual, "target_posterior_kl": posterior},
    )


def _native_top8(
    output: Any, posterior: Tensor, parent: Any,
) -> Tensor:
    native = output.counterfactual["selected_ids"].long()
    valid = (
        output.counterfactual["valid"].bool()
        & output.counterfactual["node_mask"].bool()[..., None]
    )
    batch, nodes, layers, k = native.shape
    safe_native = torch.where(valid[..., None], native, 0)
    if bool(((safe_native < 0) | (safe_native >= parent.config.experts)).any()):
        raise ValueError("valid native expert ID lies outside the expert namespace")
    membership = torch.zeros(
        batch, nodes, layers, parent.config.experts,
        device=native.device, dtype=torch.float32,
    ).scatter_(-1, safe_native, 1.0)
    membership = membership * valid[..., None].float()
    captured = torch.einsum(
        "bhn,bnle->bhle",
        posterior[..., :-1].float() * output.branch_mask.float(), membership,
    )
    anchor = parent.core.semantic_marginals(
        output.semantic, output.anchor_scores
    )[0].detach().float()
    marginals = captured + posterior[..., -1, None, None].float() * anchor
    marginals = cardinality_project_marginals(
        marginals, parent.config.exact_k
    )[0]
    return stable_topk(marginals, parent.config.exact_k)


@torch.no_grad()
def evaluate(
    *, stage: str, trajectory: ContextualFactualPathSelector, aligner: None,
    parent: Any, anchor: Any, dataset: Any, runtime_static: Any,
    token_embedding: Tensor, input_basis: Tensor, rank_mask: Tensor,
    device: torch.device, microbatch: int, workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if stage != STAGE or aligner is not None:
        raise ValueError("contextual-path evaluator received another stage")
    trajectory.eval(); parent.eval(); anchor.eval()
    candidate_cells: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    parent_cells: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    candidate_accuracy: dict[int, list[Tensor]] = defaultdict(list)
    parent_accuracy: dict[int, list[Tensor]] = defaultdict(list)
    mismatch_cells: dict[str, list[Tensor]] = defaultdict(list)
    loss_total = rows = 0.0
    for host in baseline.loader(
        dataset, batch=microbatch, shuffle=False, seed=0,
        workers=workers, device=device,
    ):
        output = route_forward(
            stage=stage, host=host, parent=parent, anchor=anchor,
            trajectory=trajectory, aligner=None, runtime_static=runtime_static,
            token_embedding=token_embedding, input_basis=input_basis,
            rank_mask=rank_mask, device=device,
        )
        loss = objective(output, trajectory, stage=stage)
        predicted = _native_top8(
            output, output.semantic.factual_path_posterior, parent
        )
        parent_ids = _native_top8(output, output.aligned, parent)
        target = output.targets["future_selected_ids"].long()
        candidate_recall = (
            target[..., None] == predicted[..., None, :]
        ).any(-1).float().mean((-1, -2))
        parent_recall = (
            target[..., None] == parent_ids[..., None, :]
        ).any(-1).float().mean((-1, -2))
        labels = output.targets["factual_branch_index_deployed"].long()
        candidate_class = output.semantic.factual_path_logits.argmax(-1)
        parent_class = output.aligned.argmax(-1)
        mismatch = ~anchor_spine_prefix_matches(
            host["anchor_inputs"]["mtp_spine"]["exact_prefix_hashes"],
            host["targets"]["future_prefix_hashes"],
        )
        requests = baseline._request_ids(host)
        for row, request in enumerate(requests):
            for horizon in (2, 3, 4):
                candidate_cells[(request, horizon)].append(
                    candidate_recall[row, horizon - 1].reshape(1).cpu()
                )
                parent_cells[(request, horizon)].append(
                    parent_recall[row, horizon - 1].reshape(1).cpu()
                )
                candidate_accuracy[horizon].append(
                    (candidate_class[row, horizon - 1] == labels[row, horizon - 1]).float().cpu()
                )
                parent_accuracy[horizon].append(
                    (parent_class[row, horizon - 1] == labels[row, horizon - 1]).float().cpu()
                )
            if bool(mismatch[row, 3]):
                mismatch_cells[request].append(candidate_recall[row, 3].reshape(1).cpu())
        loss_total += float(loss.total) * len(requests); rows += len(requests)

    def metric_rows(cells: Mapping[tuple[str, int], list[Tensor]]) -> list[dict[str, Any]]:
        return [
            {"request_id": request, "horizon": horizon,
             "slot_recall_at_8": float(torch.cat(values).mean())}
            for (request, horizon), values in sorted(cells.items())
        ]
    candidate_rows = metric_rows(candidate_cells)
    parent_rows = metric_rows(parent_cells)
    def horizon(rows_: list[dict[str, Any]], h: int) -> float:
        values = [float(row["slot_recall_at_8"]) for row in rows_ if row["horizon"] == h]
        return sum(values) / len(values)
    candidate_h = {h: horizon(candidate_rows, h) for h in (2, 3, 4)}
    parent_h = {h: horizon(parent_rows, h) for h in (2, 3, 4)}
    bootstrap = paired_h2_h4_request_bootstrap(
        candidate_rows, parent_rows, replicates=1_000, seed=42
    )
    mismatch = (
        sum(float(torch.cat(values).mean()) for values in mismatch_cells.values())
        / len(mismatch_cells) if mismatch_cells else None
    )
    metrics: dict[str, Any] = {
        "loss": loss_total / max(1, rows),
        "counterfactual_route_recall": sum(candidate_h.values()) / 3,
        "parent_counterfactual_route_recall": sum(parent_h.values()) / 3,
        "factual_candidate_c64_h2_h4": sum(candidate_h.values()) / 3,
        "factual_candidate_c64_h4_mismatch": mismatch,
        "paired_route_bootstrap_vs_parent": bootstrap,
        "affine_control_route_recall": None,
    }
    for h in (2, 3, 4):
        metrics[f"counterfactual_route_recall_h{h}"] = candidate_h[h]
        metrics[f"parent_route_recall_h{h}"] = parent_h[h]
        metrics[f"factual_candidate_c64_h{h}"] = candidate_h[h]
        metrics[f"factual_path_accuracy_h{h}"] = float(torch.stack(candidate_accuracy[h]).mean())
        metrics[f"parent_factual_path_accuracy_h{h}"] = float(torch.stack(parent_accuracy[h]).mean())
    metrics["metric_semantics"] = "native-route posterior Top8 Recall@8; legacy c64 field names retained only for driver selection compatibility"
    return metrics, candidate_rows, parent_rows


if __name__ == "__main__":
    baseline.STAGES = (*baseline.STAGES, STAGE)
    baseline.PREDECESSOR[STAGE] = None
    baseline.build_trajectory = build_trajectory
    baseline.configure_stage_parameters = configure_stage_parameters
    baseline.route_forward = route_forward
    baseline.objective = objective
    baseline.evaluate = evaluate
    baseline.main()
