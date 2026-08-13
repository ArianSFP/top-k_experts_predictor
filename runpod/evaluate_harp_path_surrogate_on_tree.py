#!/usr/bin/env python3
"""Evaluate a factual-pretrained token-path surrogate on adaptive MTP trees."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import Tensor

from harp_rtt.anchor import LegacyHARPAnchorBridge  # noqa: E402
from harp_rtt.b31 import anchor_spine_prefix_matches, quota_candidate_union  # noqa: E402
from harp_rtt.dataset import HarpRTTDataset  # noqa: E402
from harp_rtt.deltaroute_batch import prepare_deltaroute_batch  # noqa: E402
from harp_rtt.exact_k import cardinality_project_marginals, stable_topk  # noqa: E402
from harp_rtt.factual_branch_attention import (  # noqa: E402
    _trainable_exact_marginals, parent_branch_marginals,
)
from harp_rtt.node_counterfactual import (  # noqa: E402
    NodeCounterfactualDatasetAdapter, load_node_counterfactual_companion,
)
from harp_rtt.path_route_surrogate import PathRouteSurrogateConfig  # noqa: E402
from harp_rtt.path_route_trajectory import (  # noqa: E402
    LayerwiseTokenRouteSurrogate as TokenConditionedRouteSurrogate,
)
from harp_rtt.path_route_tree import reconstruct_tree_token_prefixes  # noqa: E402
from harp_rtt.route_dynamics import gather_node_horizon  # noqa: E402
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.train import prepare_model_batch, runtime_static_artifacts  # noqa: E402
from harp_rtt.training import move_to_device, seed_everything, sha256_file  # noqa: E402
from runpod.train_harp_delta_v3 import loader  # noqa: E402
from runpod.train_harp_deltaroute_v4_dynamics import load_parent  # noqa: E402


SCHEMA = "harp_path_surrogate_tree_evaluation_v1"
ROLES = (
    "post_attention_residual_u", "post_moe_residual_xplus",
    "routed_expert_output_delta_r", "shared_expert_output_delta_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-index", type=Path, required=True)
    parser.add_argument("--development-corpus", type=Path, required=True)
    parser.add_argument("--development-companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--surrogate-checkpoint", type=Path, required=True)
    parser.add_argument("--surrogate-cache", type=Path, required=True)
    parser.add_argument("--diagnostic-request-manifest", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--node-microbatch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def request_id(dataset: HarpRTTDataset, index: int) -> str:
    record = dataset.records[index]
    return str(dataset.segments[record.segment].sequences[record.sequence]["request_id"])


def load_development(args: argparse.Namespace) -> tuple[Any, set[str], str]:
    base = HarpRTTDataset(
        args.development_index, "train",
        corpus_root=args.development_corpus, max_tree_nodes=32,
    )
    counts = Counter(request_id(base, index) for index in range(len(base)))
    if len(base) != 2_048 or len(counts) != 128 or set(counts.values()) != {16}:
        raise ValueError("path-surrogate development probe must be 128x16")
    labels, manifest = load_node_counterfactual_companion(
        args.development_companion, split="train", training=True
    )
    if len(labels) != len(base):
        raise ValueError("development companion/base row count differs")
    for flag in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if manifest.get(flag) is not False:
            raise PermissionError(f"development companion violates {flag}")
    return (
        NodeCounterfactualDatasetAdapter(base, labels, split="train", training=True),
        set(counts), sha256_file(args.development_companion / "manifest.json"),
    )


def project_current_state(
    batch: Mapping[str, Any],
    *,
    means: Tensor,
    components: Tensor,
    input_basis: Tensor,
    rank_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    current = batch["inputs"]["current"]
    states = torch.stack([current[role] for role in ROLES], dim=1).float()
    coordinates = torch.einsum(
        "bqld,ldr->bqlr", states - means[None, None], components
    )
    router_input = current["normalized_target_router_input_a"].float()
    queries = torch.einsum(
        "bld,ldr->blr", router_input, input_basis.float()
    ) * rank_mask.float()[None]
    history = batch["inputs"]["history"]
    return (
        coordinates, queries, history["selected_ids"].long(),
        history["execution_weights"].float(),
    )


def predict_nodes(
    model: TokenConditionedRouteSurrogate,
    *,
    token_embedding: Tensor,
    state_coordinates: Tensor,
    current_queries: Tensor,
    history_ids: Tensor,
    history_weights: Tensor,
    tree: Mapping[str, Tensor],
    node_microbatch: int,
) -> tuple[Tensor, Tensor]:
    if node_microbatch < 1:
        raise ValueError("node microbatch must be positive")
    prefix, _ = reconstruct_tree_token_prefixes(
        tree["token_ids"].long(), tree["parent"].long(),
        tree["depth"].long(), tree["mask"].bool(),
        horizons=model.config.horizons,
    )
    active = torch.nonzero(tree["mask"].bool(), as_tuple=False)
    batch, nodes = tree["mask"].shape
    scores = torch.zeros(
        batch, nodes, model.config.layers, model.config.experts,
        device=state_coordinates.device, dtype=torch.float32,
    )
    queries = torch.zeros(
        batch, nodes, model.config.layers, model.config.router_rank,
        device=state_coordinates.device, dtype=torch.float32,
    )
    for start in range(0, len(active), node_microbatch):
        selected = active[start : start + node_microbatch]
        rows, node_ids = selected[:, 0], selected[:, 1]
        token_ids = prefix[rows, node_ids]
        if bool((token_ids < 0).any()) or bool((token_ids >= token_embedding.shape[0]).any()):
            raise ValueError("adaptive tree token is outside frozen vocabulary")
        with torch.autocast(
            device_type=state_coordinates.device.type, dtype=torch.bfloat16,
            enabled=state_coordinates.device.type == "cuda",
        ):
            output = model(
                state_coordinates=state_coordinates[rows],
                current_queries=current_queries[rows],
                history_selected_ids=history_ids[rows],
                history_selected_weights=history_weights[rows],
                path_token_embeddings=token_embedding[token_ids],
            )
        horizons = tree["depth"][rows, node_ids].long() - 1
        local = torch.arange(len(selected), device=rows.device)
        scores[rows, node_ids] = output.scores[local, horizons].float()
        queries[rows, node_ids] = output.queries[local, horizons].float()
    return queries, scores


def deployed_grid(node_scores: Tensor, semantic: Any, depth: Tensor, mask: Tensor) -> Tensor:
    grid = semantic.node_scores.float().clone()
    active = torch.nonzero(mask.bool(), as_tuple=False)
    rows, node_ids = active[:, 0], active[:, 1]
    horizons = depth[rows, node_ids].long() - 1
    grid[rows, horizons, :, node_ids] = node_scores[rows, node_ids]
    return grid


def candidate_ids(
    *,
    node_scores: Tensor,
    semantic: Any,
    anchor_scores: Tensor,
    parent: Any,
    branch_mask: Tensor,
) -> Tensor:
    anchor_marginals = parent.core.semantic_marginals(
        semantic, anchor_scores
    )[0]
    marginals = _trainable_exact_marginals(node_scores, parent.config.exact_k)
    branch = parent_branch_marginals(
        marginals, semantic.factual_path_posterior.float(), branch_mask,
        anchor_marginals, k=parent.config.exact_k,
    )
    branch = cardinality_project_marginals(branch, parent.config.exact_k)[0]
    return quota_candidate_union(
        anchor_scores, branch, anchor_quota=32,
        width=parent.config.candidate_width,
    ).expert_ids


def summarize(
    rows: list[dict[str, Any]], *, prefix: str, include_mismatch: bool
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    horizons = {}
    for horizon in (1, 2, 3, 4):
        values = [float(row[prefix]) for row in rows if row["horizon"] == horizon]
        horizons[horizon] = sum(values) / len(values)
        result[f"{prefix}_h{horizon}"] = horizons[horizon]
    result[f"{prefix}_h2_h4"] = sum(
        horizons[horizon] for horizon in (2, 3, 4)
    ) / 3
    result[f"{prefix}_h1_h4"] = sum(horizons.values()) / 4
    if include_mismatch:
        field = f"{prefix}_mismatch"
        mismatch = [
            float(row[field]) for row in rows
            if row["horizon"] == 4 and row[field] is not None
        ]
        result[f"{prefix}_h4_mismatch"] = (
            sum(mismatch) / len(mismatch) if mismatch else None
        )
    return result


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation {args.output}")
    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=False)
    dataset, requests, companion_sha = load_development(args)
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    partition_sha = sha256_file(args.partition_manifest)
    parent, parent_record = load_parent(
        args.parent_checkpoint, static, partition_sha
    )
    parent = parent.to(device).requires_grad_(False).eval()
    anchor, anchor_provenance = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint, args.target_preprocessing, args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    anchor = anchor.to(device).requires_grad_(False).eval()
    checkpoint = torch.load(
        args.surrogate_checkpoint, map_location="cpu", weights_only=True
    )
    if checkpoint.get("schema") != "harp_path_surrogate_training_v1" or checkpoint.get("stage") != "path_surrogate_pretrain":
        raise ValueError("path surrogate checkpoint is incompatible")
    cache_manifest_path = args.surrogate_cache / "manifest.json"
    cache_audit_path = args.surrogate_cache / "CACHE_AUDIT.json"
    if (
        sha256_file(cache_manifest_path) != checkpoint.get("cache_manifest_sha256")
        or sha256_file(cache_audit_path) != checkpoint.get("cache_audit_sha256")
    ):
        raise ValueError("surrogate cache differs from checkpoint lineage")
    cache_manifest = json.loads(cache_manifest_path.read_text())
    cache_requests = set(cache_manifest["train_requests"]) | set(
        cache_manifest["tune_requests"]
    )
    source_development = {
        str(json.loads(line)["request_id"])
        for line in args.diagnostic_request_manifest.read_text().splitlines()
        if line.strip()
    }
    if (
        cache_manifest.get("source_lineage_enforced") is not True
        or source_development != set(cache_manifest.get("excluded_development_requests", []))
        or sha256_file(args.diagnostic_request_manifest)
        != cache_manifest.get("diagnostic_request_manifest_sha256")
    ):
        raise PermissionError("tree development is not bound to the excluded source lineage")
    if source_development & cache_requests:
        raise PermissionError("surrogate pretraining and tree-development sources overlap")
    config = PathRouteSurrogateConfig(**checkpoint["config"])
    surrogate = TokenConditionedRouteSurrogate(
        config, static.geometry.expert_keys, static.geometry.centered_bias,
        static.geometry.rank_mask,
    ).to(device)
    surrogate.load_state_dict(checkpoint["model_state_dict"], strict=True)
    surrogate.requires_grad_(False).eval()
    if static.token_embedding is None:
        raise RuntimeError("tree evaluation requires frozen token embeddings")
    token_embedding = static.token_embedding.to(device)
    runtime_static = runtime_static_artifacts(static, device)
    input_basis = static.geometry.input_basis.to(device)
    rank_mask = static.geometry.rank_mask.to(device)
    preprocessing = torch.load(
        args.target_preprocessing, map_location="cpu", weights_only=True
    )
    means = preprocessing["local_means"].float().to(device)
    components = preprocessing["local_components"].float().to(device)

    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "surrogate_checkpoint_sha256": sha256_file(args.surrogate_checkpoint),
        "surrogate_cache_audit_sha256": checkpoint.get("cache_audit_sha256"),
        "surrogate_pretraining_requests": len(cache_requests),
        "pretraining_development_request_disjoint": True,
        "source_lineage_development_requests": len(source_development),
        "diagnostic_request_manifest_sha256": sha256_file(args.diagnostic_request_manifest),
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "partition_manifest_sha256": partition_sha,
        "development_companion_manifest_sha256": companion_sha,
        "development_requests": len(requests),
        "development_rows": len(dataset),
        "anchor": anchor_provenance,
        "optimizer_constructed": False,
        "counterfactual_labels_used_only_for_evaluation": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)
    metric_cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    route_cells: dict[tuple[str, int], dict[str, list[Tensor]]] = defaultdict(
        lambda: {"budget16": [], "allnode": [], "parent": []}
    )
    for host in loader(
        dataset, batch=args.microbatch_size, shuffle=False, seed=0,
        workers=args.num_workers, device=device,
    ):
        batch = move_to_device(host, device)
        prepared = prepare_model_batch(batch, runtime_static)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            anchor_scores = anchor(batch=prepared)["future_router_scores"][:, :4].float()
        adapted = prepare_deltaroute_batch(
            prepared, anchor_scores=anchor_scores,
            token_embedding=token_embedding, input_basis=input_basis,
            rank_mask=rank_mask, config=parent.config,
        )
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            semantic = parent(**adapted.parent_inputs, semantic_only=True)
        state, current_q, history_ids, history_weights = project_current_state(
            batch, means=means, components=components,
            input_basis=input_basis, rank_mask=rank_mask,
        )
        tree = batch["inputs"]["tree"]
        _, node_at_depth = predict_nodes(
            surrogate, token_embedding=token_embedding,
            state_coordinates=state, current_queries=current_q,
            history_ids=history_ids, history_weights=history_weights,
            tree=tree, node_microbatch=args.node_microbatch_size,
        )
        grid = deployed_grid(
            node_at_depth, semantic, tree["depth"], tree["mask"]
        )
        counterfactual = batch["targets"]["counterfactual"]
        label_valid = (
            counterfactual["valid"].bool()
            & counterfactual["node_mask"].bool()[..., None]
        ).any(-1)
        horizon_mask = tree["horizon_mask"].bool() & label_valid[:, None]
        budget16 = counterfactual["budget_node_masks"][:, 2].bool()
        conditions = {
            "budget16": horizon_mask & budget16[:, None],
            "allnode": horizon_mask,
        }
        candidates = {
            name: candidate_ids(
                node_scores=grid, semantic=semantic, anchor_scores=anchor_scores,
                parent=parent, branch_mask=branch_mask,
            )
            for name, branch_mask in conditions.items()
        }
        parent_marginals = parent.core.semantic_marginals(
            semantic, anchor_scores
        )[3]
        candidates["parent"] = quota_candidate_union(
            anchor_scores, parent_marginals, anchor_quota=32,
            width=parent.config.candidate_width,
        ).expert_ids
        target = batch["targets"]["future_selected_ids"].long()
        coverage = {
            name: (target[..., None] == value[..., None, :]).any(-1).float().mean(-1)
            for name, value in candidates.items()
        }
        mismatch = ~anchor_spine_prefix_matches(
            batch["anchor_inputs"]["mtp_spine"]["exact_prefix_hashes"],
            batch["targets"]["future_prefix_hashes"],
        )
        predicted_ids = stable_topk(node_at_depth, config.exact_k)
        parent_ids = stable_topk(
            gather_node_horizon(
                semantic.node_scores, tree["depth"], tree["mask"]
            ),
            config.exact_k,
        )
        native = counterfactual["selected_ids"].long()
        route = (
            native[..., None] == predicted_ids[..., None, :]
        ).any(-1).float().mean(-1)
        parent_route = (
            native[..., None] == parent_ids[..., None, :]
        ).any(-1).float().mean(-1)
        request_values = [str(value) for value in host["metadata"]["request_id"]]
        for row, request in enumerate(request_values):
            for horizon in (1, 2, 3, 4):
                record = {
                    "request_id": request,
                    "horizon": horizon,
                    "prefix_mismatch": bool(mismatch[row, horizon - 1]),
                    **{
                        f"c64_{name}": float(coverage[name][row, horizon - 1].mean())
                        for name in candidates
                    },
                }
                metric_cells[(request, horizon)].append(record)
                depth_nodes = tree["mask"][row].bool() & (
                    tree["depth"][row] == horizon
                )
                for name, selection in conditions.items():
                    chosen = depth_nodes & selection[row, horizon - 1]
                    route_cells[(request, horizon)][name].append(
                        route[row, chosen].reshape(-1).cpu()
                    )
                route_cells[(request, horizon)]["parent"].append(
                    parent_route[row, depth_nodes].reshape(-1).cpu()
                )

    rows: list[dict[str, Any]] = []
    for key in sorted(metric_cells):
        request, horizon = key
        values = metric_cells[key]
        row = {
            "request_id": request, "horizon": horizon,
            "prefix_mismatch": any(value["prefix_mismatch"] for value in values),
        }
        for condition in ("budget16", "allnode", "parent"):
            row[f"c64_{condition}"] = sum(
                float(value[f"c64_{condition}"]) for value in values
            ) / len(values)
            mismatch_values = [
                float(value[f"c64_{condition}"]) for value in values
                if value["prefix_mismatch"]
            ]
            row[f"c64_{condition}_mismatch"] = (
                sum(mismatch_values) / len(mismatch_values)
                if mismatch_values else None
            )
            tensors = route_cells[key][condition]
            row[f"route_recall_{condition}"] = float(torch.cat(tensors).mean())
        rows.append(row)
    metrics: dict[str, Any] = {
        "schema": SCHEMA,
        "parent_checkpoint_development": parent_record.get("development"),
    }
    for name in ("budget16", "allnode", "parent"):
        metrics.update(summarize(
            rows, prefix=f"c64_{name}", include_mismatch=True
        ))
        metrics.update(summarize(
            rows, prefix=f"route_recall_{name}", include_mismatch=False
        ))
    metrics["candidate_gate"] = {
        "required_h1": 0.98, "required_h2_h4": 0.98, "required_h4": 0.97,
        "required_h4_mismatch": 0.93,
        "budget16_passed": bool(
            metrics["c64_budget16_h2_h4"] >= 0.98
            and metrics["c64_budget16_h4"] >= 0.97
            and metrics["c64_budget16_h4_mismatch"] is not None
            and metrics["c64_budget16_h4_mismatch"] >= 0.93
        ),
        "h1_passed": bool(metrics["c64_budget16_h1"] >= 0.98),
        "allnode_passed": bool(
            metrics["c64_allnode_h2_h4"] >= 0.98
            and metrics["c64_allnode_h4"] >= 0.97
            and metrics["c64_allnode_h4_mismatch"] is not None
            and metrics["c64_allnode_h4_mismatch"] >= 0.93
        ),
    }
    write_rows(args.output / "request_metrics.jsonl", rows)
    write_json_exclusive(args.output / "STAGE_RESULT.json", {
        **metrics,
        "training_started": False, "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False, "sealed_test_opened": False,
    })
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in sorted(args.output.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    run(parse_args())
