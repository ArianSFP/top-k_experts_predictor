#!/usr/bin/env python3
"""Pretrain the token-conditioned HARP target-route surrogate."""

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

from harp_rtt.exact_k import stable_topk  # noqa: E402
from harp_rtt.losses import boundary_loss_per_endpoint, exact_set_nll  # noqa: E402
from harp_rtt.path_route_surrogate import (  # noqa: E402
    PathRouteSurrogateConfig, TokenConditionedRouteSurrogate,
)
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.training import seed_everything, sha256_file  # noqa: E402


SCHEMA = "harp_path_surrogate_training_v1"
RESULT_SCHEMA = "harp_path_surrogate_result_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--effective-batch", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


class PathCacheDataset(Dataset[dict[str, Tensor]]):
    NAMES = (
        "state_coordinates", "current_queries", "history_selected_ids",
        "history_selected_weights", "path_token_ids",
        "target_centered_logits", "target_selected_ids",
    )

    def __init__(self, root: Path, split: int) -> None:
        self.root = root
        self.arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r")
            for name in self.NAMES
        }
        flags = np.load(root / "split.npy", mmap_mode="r")
        self.indices = np.flatnonzero(flags == split).astype(np.int64)
        if not len(self.indices):
            raise ValueError("path cache split is empty")
        rows = {array.shape[0] for array in self.arrays.values()}
        if len(rows) != 1 or next(iter(rows)) != len(flags):
            raise ValueError("path cache arrays disagree on row count")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = int(self.indices[index])
        result: dict[str, Tensor] = {}
        for name, array in self.arrays.items():
            value = np.array(array[row], copy=True)
            result[name] = torch.from_numpy(value)
        return result


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def model_inputs(
    batch: Mapping[str, Tensor],
    *,
    token_embedding: Tensor,
    device: torch.device,
) -> dict[str, Tensor]:
    ids = batch["path_token_ids"].to(device, non_blocking=True).long()
    if bool(((ids < 0) | (ids >= token_embedding.shape[0])).any()):
        raise ValueError("path cache token ID lies outside frozen vocabulary")
    return {
        "state_coordinates": batch["state_coordinates"].to(
            device, non_blocking=True
        ),
        "current_queries": batch["current_queries"].to(
            device, non_blocking=True
        ),
        "history_selected_ids": batch["history_selected_ids"].to(
            device, non_blocking=True
        ).long(),
        "history_selected_weights": batch["history_selected_weights"].to(
            device, non_blocking=True
        ),
        "path_token_embeddings": token_embedding[ids],
    }


def objective(
    model: TokenConditionedRouteSurrogate,
    batch: Mapping[str, Tensor],
    *,
    token_embedding: Tensor,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    inputs = model_inputs(batch, token_embedding=token_embedding, device=device)
    labels = batch["target_selected_ids"].to(device, non_blocking=True).long()
    target_logits = batch["target_centered_logits"].to(
        device, non_blocking=True
    ).float()
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model(**inputs)
    horizon_weights = torch.tensor(
        [1.0, 1.0, 1.25, 1.5], device=device, dtype=torch.float32
    )[None, :, None].expand(labels.shape[:-1])
    exact = exact_set_nll(
        output.scores, labels, valid=horizon_weights, k=model.config.exact_k
    )
    predicted = output.scores.float()
    predicted = predicted - predicted.mean(-1, keepdim=True)
    dense_endpoint = F.huber_loss(
        predicted, target_logits, reduction="none", delta=1.0
    ).mean(-1)
    boundary_endpoint = boundary_loss_per_endpoint(
        output.scores, target_logits, labels,
        model_rank_start=6, teacher_rank_start=9, rank_end=32,
    )
    denominator = horizon_weights.sum().clamp_min(1.0)
    dense = (dense_endpoint * horizon_weights).sum() / denominator
    boundary = (boundary_endpoint * horizon_weights).sum() / denominator
    total = exact + dense + 0.2 * boundary
    return total, {
        "exact_set": float(exact.detach()),
        "centered_logit_huber": float(dense.detach()),
        "top_boundary": float(boundary.detach()),
        "total": float(total.detach()),
    }


@torch.no_grad()
def evaluate(
    model: TokenConditionedRouteSurrogate,
    dataset: Dataset[dict[str, Tensor]],
    *,
    token_embedding: Tensor,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    model.eval(); hits = torch.zeros(4); slots = torch.zeros(4)
    loss_total = rows = 0.0
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    for batch in loader:
        loss, _ = objective(
            model, batch, token_embedding=token_embedding, device=device
        )
        inputs = model_inputs(batch, token_embedding=token_embedding, device=device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(**inputs)
        labels = batch["target_selected_ids"].to(device).long()
        overlap = (
            labels[..., None] == output.selected_ids[..., None, :]
        ).any(-1).float()
        hits += overlap.sum((0, 2, 3)).cpu()
        slots += torch.tensor(
            [labels.shape[0] * labels.shape[2] * labels.shape[3]] * 4
        )
        loss_total += float(loss) * labels.shape[0]; rows += labels.shape[0]
    horizon = (hits / slots).tolist()
    return {
        "loss": loss_total / rows,
        "route_recall_h1_h4": sum(horizon) / 4,
        "route_recall_h2_h4": sum(horizon[1:]) / 3,
        **{f"route_recall_h{index + 1}": value
           for index, value in enumerate(horizon)},
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite path surrogate run {args.output}")
    if args.effective_batch % args.microbatch_size:
        raise ValueError("effective batch must divide by microbatch")
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if manifest.get("schema") != "harp_path_surrogate_cache_v1" or manifest.get("complete") is not True:
        raise ValueError("path surrogate cache is incomplete or incompatible")
    for flag in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if manifest.get(flag) is not False:
            raise PermissionError(f"path cache violates {flag}")
    audit_path = args.cache / "CACHE_AUDIT.json"
    if not audit_path.is_file():
        raise FileNotFoundError("path surrogate cache has not passed its audit")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("schema") != "harp_path_surrogate_cache_audit_v1"
        or audit.get("complete") is not True
        or audit.get("manifest_sha256") != sha256_file(args.cache / "manifest.json")
    ):
        raise ValueError("path surrogate cache audit is incompatible")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA path training requested but unavailable")
    seed_everything(args.seed, deterministic=False)
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    if static.token_embedding is None:
        raise RuntimeError("path surrogate needs frozen target token embeddings")
    token_embedding = static.token_embedding.to(device)
    config = PathRouteSurrogateConfig(state_rank=int(manifest["state_rank"]))
    model = TokenConditionedRouteSurrogate(
        config, static.geometry.expert_keys,
        static.geometry.centered_bias, static.geometry.rank_mask,
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) > 15_000_000:
        raise RuntimeError("path surrogate exceeds its 15M parameter cap")
    train = PathCacheDataset(args.cache, 0)
    tune = PathCacheDataset(args.cache, 1)
    args.output.mkdir(parents=True)
    write_json(args.output / "run_manifest.json", {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit, "seed": args.seed,
        "cache_manifest_sha256": sha256_file(args.cache / "manifest.json"),
        "cache_audit_sha256": sha256_file(audit_path),
        "cache_rows": int(manifest["rows"]),
        "train_rows": len(train), "tune_rows": len(tune),
        "config": config.to_dict(),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "factual_path_tokens_are_training_only_teacher_inputs": True,
        "serving_path_tokens_must_come_from_causal_mtp_tree": True,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    })
    sample = next(iter(DataLoader(train, batch_size=min(args.microbatch_size, 4))))
    model.train(); model.zero_grad(set_to_none=True)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    loss, _ = objective(
        model, sample, token_embedding=token_embedding, device=device
    )
    loss.backward(); model.zero_grad(set_to_none=True)
    peak = (
        torch.cuda.max_memory_reserved(device) / 2**30
        if device.type == "cuda" else 0.0
    )
    write_json(args.output / "PREFLIGHT_RESULT.json", {
        "loss_finite": bool(torch.isfinite(loss)),
        "peak_reserved_gib": peak,
        "optimizer_constructed": False,
        "training_started": False,
    })
    if not torch.isfinite(loss) or peak > 21.0:
        raise RuntimeError("path surrogate failed numerical/memory preflight")
    if args.preflight_only:
        return
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    write_json(args.output / "OPTIMIZER_START.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer": "AdamW", "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    })
    accumulation = args.effective_batch // args.microbatch_size
    best = -math.inf; best_epoch = 0; stale = 0; best_state = None; best_metrics = None
    metrics_path = args.output / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        total = batches = 0.0
        loader = DataLoader(
            train, batch_size=args.microbatch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + epoch),
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        for step, batch in enumerate(loader, start=1):
            loss, _ = objective(
                model, batch, token_embedding=token_embedding, device=device
            )
            (loss / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()); batches += 1
        tune_metrics = evaluate(
            model, tune, token_embedding=token_embedding, device=device,
            batch_size=args.microbatch_size, workers=args.num_workers,
        )
        append_jsonl(metrics_path, {
            "epoch": epoch, "train_loss": total / batches,
            "tune": tune_metrics,
        })
        value = float(tune_metrics["route_recall_h2_h4"])
        if value > best:
            best = value; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = tune_metrics
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("path surrogate produced no checkpoint")
    checkpoint = args.output / "best_path_surrogate.pt"
    torch.save({
        "schema": SCHEMA, "stage": "path_surrogate_pretrain",
        "source_commit": args.source_commit, "seed": args.seed,
        "epoch": best_epoch, "config": config.to_dict(),
        "cache_manifest_sha256": sha256_file(args.cache / "manifest.json"),
        "cache_audit_sha256": sha256_file(audit_path),
        "model_state_dict": best_state, "tune": best_metrics,
        "formal_validation_opened": False,
        "calibration_opened": False, "sealed_test_opened": False,
    }, checkpoint)
    write_json(args.output / "STAGE_RESULT.json", {
        "schema": RESULT_SCHEMA, "best_epoch": best_epoch,
        "best_tune": best_metrics, "checkpoint_sha256": sha256_file(checkpoint),
        "training_started": True, "formal_validation_opened": False,
        "calibration_opened": False, "sealed_test_opened": False,
    })


if __name__ == "__main__":
    main()
