#!/usr/bin/env python3
"""Train the raw-branch-conditioned layerwise route trajectory (one seed)."""

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

from harp_rtt.branch_conditioned_path import (  # noqa: E402
    BranchConditionedLayerwiseRouteSurrogate,
)
from harp_rtt.branch_direct_score import (  # noqa: E402
    DirectHighRankBranchRouteSurrogate,
)
from harp_rtt.losses import boundary_loss_per_endpoint, exact_set_nll  # noqa: E402
from harp_rtt.path_route_surrogate import PathRouteSurrogateConfig  # noqa: E402
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.training import seed_everything, sha256_file  # noqa: E402
from runpod.train_harp_branch_surrogate import (  # noqa: E402
    choose_nodes, supervision_weights,
)


SCHEMA = "harp_branch_conditioned_surrogate_training_v1"
RESULT_SCHEMA = "harp_branch_conditioned_surrogate_result_v1"
BASE_NAMES = (
    "state_coordinates", "current_queries", "history_selected_ids",
    "history_selected_weights", "path_token_ids", "node_depth",
    "node_mask", "budget16_mask", "target_queries", "target_selected_ids",
)
STATE_NAMES = (
    "branch_states", "branch_router_logits", "branch_selected_ids",
    "branch_selected_weights", "branch_vocab_embedding",
    "branch_vocab_statistics", "branch_scalars", "branch_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--architecture", choices=("trajectory", "direct_score"),
        default="trajectory",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--branch-learning-rate", type=float, default=3e-4)
    parser.add_argument("--base-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--source-batch-size", type=int, default=4)
    parser.add_argument("--nodes-per-source", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--large-gain-check-epoch", type=int, default=3)
    parser.add_argument("--minimum-check-gain", type=float, default=0.04)
    parser.add_argument("--required-large-gain", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


class BranchStateDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, root: Path, state_root: Path, split: int) -> None:
        self.arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r")
            for name in BASE_NAMES
        }
        self.arrays.update({
            name: np.load(state_root / f"{name}.npy", mmap_mode="r")
            for name in STATE_NAMES
        })
        flags = np.load(root / "split.npy", mmap_mode="r")
        self.indices = np.flatnonzero(flags == split).astype(np.int64)
        if not len(self.indices):
            raise ValueError("branch-state split is empty")

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


def branch_forward(
    model: BranchConditionedLayerwiseRouteSurrogate,
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
    if bool((batch["branch_mask"][source_rows, node_ids] == 0).any()):
        raise ValueError("selected supervised node lacks branch input state")
    depth = batch["node_depth"][source_rows, node_ids].long()
    path_ids = batch["path_token_ids"][source_rows, node_ids].long()
    if bool(((path_ids < 0) | (path_ids >= token_embedding.shape[0])).any()):
        raise ValueError("branch prefix token is outside the frozen vocabulary")
    source_rows = source_rows.to(device); node_ids = node_ids.to(device)
    depth = depth.to(device); path_ids = path_ids.to(device)
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
            branch_states=batch["branch_states"].to(device)[source_rows, node_ids],
            branch_router_logits=batch["branch_router_logits"].to(device)[source_rows, node_ids],
            branch_selected_ids=batch["branch_selected_ids"].to(device).long()[source_rows, node_ids],
            branch_selected_weights=batch["branch_selected_weights"].to(device)[source_rows, node_ids],
            branch_vocab_embedding=batch["branch_vocab_embedding"].to(device)[source_rows, node_ids],
            branch_vocab_statistics=batch["branch_vocab_statistics"].to(device)[source_rows, node_ids],
            branch_scalars=batch["branch_scalars"].to(device)[source_rows, node_ids],
        )
    local = torch.arange(len(source_rows), device=device)
    horizons = depth - 1
    queries = output.queries[local, horizons].float()
    scores = output.scores[local, horizons].float()
    target_queries = batch["target_queries"].to(device)[source_rows, node_ids].float()
    target_ids = batch["target_selected_ids"].to(device).long()[source_rows, node_ids]
    return output, queries, scores, target_queries, target_ids, depth


def objective(
    model: BranchConditionedLayerwiseRouteSurrogate,
    batch: Mapping[str, Tensor],
    *, token_embedding: Tensor, device: torch.device,
    nodes_per_source: int | None, generator: torch.Generator | None,
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
    total = exact + dense + 0.5 * boundary
    return total, {
        "exact_set": float(exact.detach()),
        "induced_logit_huber": float(dense.detach()),
        "top_boundary": float(boundary.detach()), "total": float(total.detach()),
    }, (scores.detach(), target_ids.detach(), depth.detach())


@torch.no_grad()
def evaluate(
    model: BranchConditionedLayerwiseRouteSurrogate,
    dataset: Dataset[dict[str, Tensor]],
    *, token_embedding: Tensor, device: torch.device,
    batch_size: int, workers: int,
) -> dict[str, float]:
    model.eval(); hits = {2: 0., 3: 0., 4: 0.}; slots = {2: 0., 3: 0., 4: 0.}
    losses = rows = 0.0
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
        overlap = (target_ids[..., None] == predicted[..., None, :]).any(-1).float()
        for value in (2, 3, 4):
            selected = depth == value
            hits[value] += float(overlap[selected].sum())
            slots[value] += float(selected.sum() * target_ids.shape[-2] * target_ids.shape[-1])
        losses += float(loss) * batch["node_mask"].shape[0]
        rows += batch["node_mask"].shape[0]
    recall = {value: hits[value] / slots[value] for value in (2, 3, 4)}
    return {
        "loss": losses / rows,
        "route_recall_h2": recall[2], "route_recall_h3": recall[3],
        "route_recall_h4": recall[4],
        "route_recall_h2_h4": sum(recall.values()) / 3,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite run {args.output}")
    base_manifest_path = args.cache / "manifest.json"
    base_audit_path = args.cache / "CACHE_AUDIT.json"
    state_manifest_path = args.state_cache / "manifest.json"
    state_audit_path = args.state_cache / "CACHE_AUDIT.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    base_audit = json.loads(base_audit_path.read_text())
    state_manifest = json.loads(state_manifest_path.read_text())
    state_audit = json.loads(state_audit_path.read_text())
    if (
        base_manifest.get("schema") != "harp_branch_surrogate_cache_v1"
        or base_audit.get("complete") is not True
        or state_manifest.get("schema") != "harp_branch_state_cache_v1"
        or state_audit.get("schema") != "harp_branch_state_cache_audit_v1"
        or state_audit.get("complete") is not True
        or state_manifest.get("base_cache_manifest_sha256") != sha256_file(base_manifest_path)
        or state_manifest.get("base_cache_audit_sha256") != sha256_file(base_audit_path)
        or state_audit.get("manifest_sha256") != sha256_file(state_manifest_path)
    ):
        raise ValueError("branch label/state caches are incomplete or unbound")
    for manifest in (base_manifest, base_audit, state_manifest, state_audit):
        for flag in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
            if manifest.get(flag) is not False:
                raise PermissionError(f"cache violates {flag}")

    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=False)
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    if static.token_embedding is None:
        raise RuntimeError("training requires frozen token embeddings")
    token_embedding = static.token_embedding.to(device)
    initializer = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
    if (initializer.get("schema"), initializer.get("stage")) not in {
        ("harp_path_surrogate_training_v1", "path_surrogate_pretrain"),
        ("harp_branch_surrogate_training_v1", "counterfactual_branch_adaptation"),
    }:
        raise ValueError("branch-conditioned initializer is incompatible")
    config = PathRouteSurrogateConfig(**initializer["config"])
    model_class = (
        BranchConditionedLayerwiseRouteSurrogate
        if args.architecture == "trajectory"
        else DirectHighRankBranchRouteSurrogate
    )
    model_kwargs = (
        {"branch_state_rank": 32, "branch_vocab_rank": 16}
        if args.architecture == "trajectory" else {"raw_rank": 64}
    )
    model = model_class(
        config, static.geometry.expert_keys, static.geometry.centered_bias,
        static.geometry.rank_mask, **model_kwargs,
    ).to(device)
    incompatible = model.load_state_dict(initializer["model_state_dict"], strict=False)
    if incompatible.unexpected_keys or any(
        not key.startswith(("branch_", "query_", "score_"))
        for key in incompatible.missing_keys
    ):
        raise ValueError(f"initializer mismatch: {incompatible}")
    train = BranchStateDataset(args.cache, args.state_cache, 0)
    tune = BranchStateDataset(args.cache, args.state_cache, 1)
    args.output.mkdir(parents=True)
    branch_parameters = model.branch_parameters()
    branch_ids = {id(value) for value in branch_parameters}
    base_parameters = [
        value for value in model.parameters() if id(value) not in branch_ids
    ]
    write_json(args.output / "run_manifest.json", {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit, "seed": args.seed,
        "architecture": f"branch_conditioned_{args.architecture}_v1",
        "initializer_sha256": sha256_file(args.initialize_from),
        "cache_manifest_sha256": sha256_file(base_manifest_path),
        "cache_audit_sha256": sha256_file(base_audit_path),
        "state_cache_manifest_sha256": sha256_file(state_manifest_path),
        "state_cache_audit_sha256": sha256_file(state_audit_path),
        "train_rows": len(train), "tune_rows": len(tune),
        "train_requests": 224, "tune_requests": 32,
        "nodes_per_training_source": args.nodes_per_source,
        "evaluation_mask": "budget16_all_nodes",
        "config": config.to_dict(),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "branch_parameters": sum(p.numel() for p in branch_parameters),
        "base_parameters": sum(p.numel() for p in base_parameters),
        "large_gain_contract": {
            "check_epoch": args.large_gain_check_epoch,
            "minimum_gain_at_check": args.minimum_check_gain,
            "required_final_gain": args.required_large_gain,
        },
        "optimizer_constructed": False, "formal_validation_opened": False,
        "calibration_opened": False, "sealed_test_opened": False,
    })
    sample = next(iter(DataLoader(train, batch_size=min(2, args.source_batch_size))))
    model.train(); model.zero_grad(set_to_none=True)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    loss, _, _ = objective(
        model, sample, token_embedding=token_embedding, device=device,
        nodes_per_source=min(2, args.nodes_per_source),
        generator=torch.Generator().manual_seed(args.seed),
    )
    loss.backward(); model.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
    write_json(args.output / "PREFLIGHT_RESULT.json", {
        "loss_finite": bool(torch.isfinite(loss)), "peak_reserved_gib": peak,
        "training_started": False, "optimizer_constructed": False,
    })
    if not torch.isfinite(loss) or peak > 21.0:
        raise RuntimeError("branch-conditioned preflight failed")
    epoch_zero = evaluate(
        model, tune, token_embedding=token_embedding, device=device,
        batch_size=args.source_batch_size, workers=args.num_workers,
    )
    write_json(args.output / "EPOCH_ZERO_AUDIT.json", {
        "tune": epoch_zero, "branch_gate_exact_zero": bool(
            torch.equal(model.branch_gate.detach(), torch.zeros_like(model.branch_gate))
        ), "training_started": False, "optimizer_constructed": False,
    })
    if args.preflight_only:
        return
    optimizer = torch.optim.AdamW([
        {"params": branch_parameters, "lr": args.branch_learning_rate},
        {"params": base_parameters, "lr": args.base_learning_rate},
    ], weight_decay=args.weight_decay)
    write_json(args.output / "OPTIMIZER_START.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer": "AdamW", "branch_learning_rate": args.branch_learning_rate,
        "base_learning_rate": args.base_learning_rate,
        "weight_decay": args.weight_decay,
    })
    baseline = float(epoch_zero["route_recall_h2_h4"])
    best = -math.inf; best_epoch = 0; stale = 0; killed = False
    best_state = best_metrics = None
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
                nodes_per_source=args.nodes_per_source, generator=node_generator,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); total += float(loss.detach()); batches += 1
        tune_metrics = evaluate(
            model, tune, token_embedding=token_embedding, device=device,
            batch_size=args.source_batch_size, workers=args.num_workers,
        )
        gain = float(tune_metrics["route_recall_h2_h4"]) - baseline
        append_jsonl(args.output / "metrics.jsonl", {
            "epoch": epoch, "train_loss": total / batches,
            "tune": tune_metrics, "absolute_gain_vs_epoch_zero": gain,
        })
        value = float(tune_metrics["route_recall_h2_h4"])
        if value > best:
            best = value; best_epoch = epoch; stale = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_metrics = tune_metrics
        else:
            stale += 1
        if epoch == args.large_gain_check_epoch and gain < args.minimum_check_gain:
            killed = True; break
        if stale >= args.patience:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("branch-conditioned training produced no checkpoint")
    checkpoint = args.output / "best_branch_conditioned_surrogate.pt"
    with checkpoint.open("xb") as handle:
        torch.save({
            "schema": SCHEMA, "stage": f"branch_conditioned_{args.architecture}",
            "source_commit": args.source_commit, "seed": args.seed,
            "epoch": best_epoch, "config": config.to_dict(),
            "architecture": f"branch_conditioned_{args.architecture}_v1",
            "initializer_sha256": sha256_file(args.initialize_from),
            "run_manifest_sha256": sha256_file(args.output / "run_manifest.json"),
            "cache_manifest_sha256": sha256_file(base_manifest_path),
            "cache_audit_sha256": sha256_file(base_audit_path),
            "state_cache_manifest_sha256": sha256_file(state_manifest_path),
            "state_cache_audit_sha256": sha256_file(state_audit_path),
            "model_state_dict": best_state, "tune": best_metrics,
            "formal_validation_opened": False, "calibration_opened": False,
            "sealed_test_opened": False,
        }, handle)
        handle.flush(); os.fsync(handle.fileno())
    gain = float(best_metrics["route_recall_h2_h4"]) - baseline
    write_json(args.output / "STAGE_RESULT.json", {
        "schema": RESULT_SCHEMA, "best_epoch": best_epoch,
        "epoch_zero_tune": epoch_zero, "best_tune": best_metrics,
        "absolute_gain_vs_epoch_zero": gain,
        "large_gain_check_killed": killed,
        "large_gain_gate_passed": bool(gain >= args.required_large_gain),
        "required_large_gain": args.required_large_gain,
        "checkpoint_sha256": sha256_file(checkpoint),
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
