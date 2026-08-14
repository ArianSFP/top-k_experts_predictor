#!/usr/bin/env python3
"""Adapt the factual path-route surrogate on existing counterfactual branches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from harp_rtt.losses import boundary_loss_per_endpoint, exact_set_nll  # noqa: E402
from harp_rtt.path_route_surrogate import PathRouteSurrogateConfig  # noqa: E402
from harp_rtt.path_route_trajectory import (  # noqa: E402
    LayerwiseTokenRouteSurrogate as TokenConditionedRouteSurrogate,
)
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.training import seed_everything, sha256_file  # noqa: E402


SCHEMA = "harp_branch_surrogate_training_v1"
RESULT_SCHEMA = "harp_branch_surrogate_result_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--source-batch-size", type=int, default=8)
    parser.add_argument("--nodes-per-source", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


class BranchCacheDataset(Dataset[dict[str, Tensor]]):
    NAMES = (
        "state_coordinates", "current_queries", "history_selected_ids",
        "history_selected_weights", "path_token_ids", "node_depth",
        "node_mask", "budget16_mask", "target_queries",
        "target_selected_ids",
    )

    def __init__(self, root: Path, split: int) -> None:
        self.arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r")
            for name in self.NAMES
        }
        flags = np.load(root / "split.npy", mmap_mode="r")
        self.indices = np.flatnonzero(flags == split).astype(np.int64)
        if not len(self.indices):
            raise ValueError("branch-cache split is empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = int(self.indices[index])
        return {
            name: torch.from_numpy(np.array(array[row], copy=True))
            for name, array in self.arrays.items()
        }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def choose_nodes(
    depth: Tensor,
    mask: Tensor,
    *,
    nodes_per_source: int | None,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    """Choose depth-balanced deployed nodes; all nodes when budget is None."""

    selected_rows: list[Tensor] = []
    selected_nodes: list[Tensor] = []
    quotas = {2: 2, 3: 2, 4: 4}
    for row in range(mask.shape[0]):
        active = torch.nonzero(mask[row].bool(), as_tuple=False).flatten()
        if nodes_per_source is None or len(active) <= nodes_per_source:
            chosen = active
        else:
            parts: list[Tensor] = []
            for value in (2, 3, 4):
                candidates = active[depth[row, active].long() == value]
                count = min(quotas[value], len(candidates))
                if count:
                    order = torch.randperm(len(candidates), generator=generator)
                    parts.append(candidates[order[:count]])
            chosen = torch.cat(parts) if parts else active[:0]
            if len(chosen) < nodes_per_source:
                remaining = active[~torch.isin(active, chosen)]
                order = torch.randperm(len(remaining), generator=generator)
                chosen = torch.cat((chosen, remaining[order[: nodes_per_source - len(chosen)]]))
        selected_rows.append(torch.full((len(chosen),), row, dtype=torch.long))
        selected_nodes.append(chosen.long())
    return torch.cat(selected_rows), torch.cat(selected_nodes)


def branch_forward(
    model: TokenConditionedRouteSurrogate,
    batch: Mapping[str, Tensor],
    *,
    token_embedding: Tensor,
    device: torch.device,
    nodes_per_source: int | None,
    generator: torch.Generator | None,
) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, Tensor]:
    source_rows, node_ids = choose_nodes(
        batch["node_depth"], batch["budget16_mask"],
        nodes_per_source=nodes_per_source, generator=generator,
    )
    depth = batch["node_depth"][source_rows, node_ids].long()
    path_ids = batch["path_token_ids"][source_rows, node_ids].long()
    if bool(((path_ids < 0) | (path_ids >= token_embedding.shape[0])).any()):
        raise ValueError("branch prefix token is outside the frozen vocabulary")
    source_rows = source_rows.to(device)
    node_ids = node_ids.to(device)
    depth = depth.to(device)
    path_ids = path_ids.to(device)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model(
            state_coordinates=batch["state_coordinates"].to(device)[source_rows],
            current_queries=batch["current_queries"].to(device)[source_rows],
            history_selected_ids=batch["history_selected_ids"].to(device).long()[source_rows],
            history_selected_weights=batch["history_selected_weights"].to(device)[source_rows],
            path_token_embeddings=token_embedding[path_ids],
        )
    local = torch.arange(len(source_rows), device=device)
    horizons = depth - 1
    queries = output.queries[local, horizons].float()
    scores = output.scores[local, horizons].float()
    target_queries = batch["target_queries"].to(device)[source_rows, node_ids].float()
    target_ids = batch["target_selected_ids"].to(device).long()[source_rows, node_ids]
    return output, queries, scores, target_queries, target_ids, depth


def supervision_weights(depth: Tensor) -> Tensor:
    weights = torch.zeros_like(depth, dtype=torch.float32)
    for value in (2, 3, 4):
        active = depth == value
        weights[active] = 1.0 / max(1, int(active.sum()))
    return weights / 3.0


def objective(
    model: TokenConditionedRouteSurrogate,
    batch: Mapping[str, Tensor],
    *,
    token_embedding: Tensor,
    device: torch.device,
    nodes_per_source: int | None,
    generator: torch.Generator | None,
) -> tuple[Tensor, dict[str, float], tuple[Tensor, Tensor, Tensor]]:
    _, queries, scores, target_queries, target_ids, depth = branch_forward(
        model, batch, token_embedding=token_embedding, device=device,
        nodes_per_source=nodes_per_source, generator=generator,
    )
    node_weights = supervision_weights(depth)
    valid = node_weights[:, None].expand(scores.shape[:-1])
    exact = exact_set_nll(scores, target_ids, valid=valid, k=model.config.exact_k)
    induced = torch.einsum(
        "mlr,ler->mle", queries - target_queries, model.expert_keys.float()
    )
    dense_rows = F.huber_loss(
        induced, torch.zeros_like(induced), reduction="none", delta=1.0
    ).mean(-1)
    denominator = valid.sum().clamp_min(1.0)
    dense = (dense_rows * valid).sum() / denominator
    target_scores = torch.einsum(
        "mlr,ler->mle", target_queries, model.expert_keys.float()
    ) + model.centered_bias.float()[None]
    boundary_rows = boundary_loss_per_endpoint(
        scores, target_scores, target_ids,
        model_rank_start=6, teacher_rank_start=9, rank_end=32,
    )
    boundary = (boundary_rows * valid).sum() / denominator
    total = exact + dense + 0.2 * boundary
    return total, {
        "exact_set": float(exact.detach()),
        "induced_logit_huber": float(dense.detach()),
        "top_boundary": float(boundary.detach()), "total": float(total.detach()),
    }, (scores.detach(), target_ids.detach(), depth.detach())


@torch.no_grad()
def evaluate(
    model: TokenConditionedRouteSurrogate,
    dataset: Dataset[dict[str, Tensor]],
    *,
    token_embedding: Tensor,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, float]:
    model.eval()
    hits = {2: 0.0, 3: 0.0, 4: 0.0}
    slots = {2: 0.0, 3: 0.0, 4: 0.0}
    loss_total = rows = 0.0
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=False,
    )
    for batch in loader:
        loss, _, (scores, target_ids, depth) = objective(
            model, batch, token_embedding=token_embedding, device=device,
            nodes_per_source=None, generator=None,
        )
        predicted = scores.topk(model.config.exact_k, dim=-1).indices
        overlap = (
            target_ids[..., None] == predicted[..., None, :]
        ).any(-1).float()
        for value in (2, 3, 4):
            selected = depth == value
            hits[value] += float(overlap[selected].sum())
            slots[value] += float(selected.sum() * target_ids.shape[-2] * target_ids.shape[-1])
        loss_total += float(loss) * batch["node_mask"].shape[0]
        rows += batch["node_mask"].shape[0]
    recall = {value: hits[value] / slots[value] for value in (2, 3, 4)}
    return {
        "loss": loss_total / rows,
        "route_recall_h2": recall[2], "route_recall_h3": recall[3],
        "route_recall_h4": recall[4],
        "route_recall_h2_h4": sum(recall.values()) / 3,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite branch adaptation run {args.output}")
    manifest = json.loads((args.cache / "manifest.json").read_text())
    audit_path = args.cache / "CACHE_AUDIT.json"
    audit = json.loads(audit_path.read_text())
    if (
        manifest.get("schema") != "harp_branch_surrogate_cache_v1"
        or manifest.get("complete") is not True
        or audit.get("schema") != "harp_branch_surrogate_cache_audit_v1"
        or audit.get("complete") is not True
        or audit.get("manifest_sha256") != sha256_file(args.cache / "manifest.json")
    ):
        raise ValueError("branch cache/audit is incomplete or incompatible")
    for flag in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if manifest.get(flag) is not False or audit.get(flag) is not False:
            raise PermissionError(f"branch cache violates {flag}")
    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=False)
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    if static.token_embedding is None:
        raise RuntimeError("branch adaptation requires frozen token embeddings")
    token_embedding = static.token_embedding.to(device)
    initializer = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
    if (
        initializer.get("schema") != "harp_path_surrogate_training_v1"
        or initializer.get("stage") != "path_surrogate_pretrain"
    ):
        raise ValueError("branch adaptation initializer is incompatible")
    config = PathRouteSurrogateConfig(**initializer["config"])
    if config.state_rank != int(manifest["state_rank"]):
        raise ValueError("branch cache and initializer state ranks differ")
    model = TokenConditionedRouteSurrogate(
        config, static.geometry.expert_keys, static.geometry.centered_bias,
        static.geometry.rank_mask,
    ).to(device)
    model.load_state_dict(initializer["model_state_dict"], strict=True)
    train = BranchCacheDataset(args.cache, 0)
    tune = BranchCacheDataset(args.cache, 1)
    args.output.mkdir(parents=True)
    write_json(args.output / "run_manifest.json", {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit, "seed": args.seed,
        "architecture": "factual_pretrained_layerwise_route_trajectory_counterfactual_adaptation_v1",
        "initializer_sha256": sha256_file(args.initialize_from),
        "cache_manifest_sha256": sha256_file(args.cache / "manifest.json"),
        "cache_audit_sha256": sha256_file(audit_path),
        "train_rows": len(train), "tune_rows": len(tune),
        "train_requests": 224, "tune_requests": 32,
        "nodes_per_training_source": args.nodes_per_source,
        "training_node_depth_quota": {"h2": 2, "h3": 2, "h4": 4},
        "evaluation_mask": "budget16_all_nodes",
        "config": config.to_dict(),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "optimizer_constructed": False, "formal_validation_opened": False,
        "calibration_opened": False, "sealed_test_opened": False,
    })
    sample = next(iter(DataLoader(train, batch_size=min(2, args.source_batch_size))))
    model.train(); model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss, _, _ = objective(
        model, sample, token_embedding=token_embedding, device=device,
        nodes_per_source=args.nodes_per_source,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loss.backward(); model.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
    write_json(args.output / "PREFLIGHT_RESULT.json", {
        "loss_finite": bool(torch.isfinite(loss)), "peak_reserved_gib": peak,
        "training_started": False, "optimizer_constructed": False,
    })
    if not torch.isfinite(loss) or peak > 21.0:
        raise RuntimeError("branch adaptation failed numerical/memory preflight")
    if args.preflight_only:
        return
    epoch_zero = evaluate(
        model, tune, token_embedding=token_embedding, device=device,
        batch_size=args.source_batch_size, workers=args.num_workers,
    )
    write_json(args.output / "EPOCH_ZERO_AUDIT.json", {
        "tune": epoch_zero, "training_started": False,
        "optimizer_constructed": False, "formal_validation_opened": False,
        "calibration_opened": False, "sealed_test_opened": False,
    })
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    write_json(args.output / "OPTIMIZER_START.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer": "AdamW", "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    })
    best = -math.inf; best_epoch = 0; stale = 0
    best_state = best_metrics = None
    metrics_path = args.output / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train(); total = batches = 0.0
        loader = DataLoader(
            train, batch_size=args.source_batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + epoch),
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        node_generator = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = objective(
                model, batch, token_embedding=token_embedding, device=device,
                nodes_per_source=args.nodes_per_source,
                generator=node_generator,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); total += float(loss.detach()); batches += 1
        tune_metrics = evaluate(
            model, tune, token_embedding=token_embedding, device=device,
            batch_size=args.source_batch_size, workers=args.num_workers,
        )
        append_jsonl(metrics_path, {
            "epoch": epoch, "train_loss": total / batches, "tune": tune_metrics,
        })
        value = float(tune_metrics["route_recall_h2_h4"])
        if value > best:
            best = value; best_epoch = epoch; stale = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_metrics = tune_metrics
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("branch adaptation produced no checkpoint")
    checkpoint = args.output / "best_branch_surrogate.pt"
    with checkpoint.open("xb") as handle:
        torch.save({
            "schema": SCHEMA, "stage": "counterfactual_branch_adaptation",
            "source_commit": args.source_commit, "seed": args.seed,
            "epoch": best_epoch, "config": config.to_dict(),
            "architecture": "factual_pretrained_layerwise_route_trajectory_counterfactual_adaptation_v1",
            "initializer_sha256": sha256_file(args.initialize_from),
            "run_manifest_sha256": sha256_file(args.output / "run_manifest.json"),
            "cache_manifest_sha256": sha256_file(args.cache / "manifest.json"),
            "cache_audit_sha256": sha256_file(audit_path),
            "model_state_dict": best_state, "tune": best_metrics,
            "formal_validation_opened": False, "calibration_opened": False,
            "sealed_test_opened": False,
        }, handle)
        handle.flush(); os.fsync(handle.fileno())
    write_json(args.output / "STAGE_RESULT.json", {
        "schema": RESULT_SCHEMA, "best_epoch": best_epoch,
        "best_tune": best_metrics, "checkpoint_sha256": sha256_file(checkpoint),
        "training_started": True, "formal_validation_opened": False,
        "calibration_opened": False, "sealed_test_opened": False,
    })
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in sorted(args.output.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush(); os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
