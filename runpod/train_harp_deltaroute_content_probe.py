#!/usr/bin/env python3
"""Train the no-recapture router-blind full-state transition diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.content_transition import (  # noqa: E402
    ContentTransitionConfig,
    FullStateTransitionProbe,
)
from harp_rtt.deltaroute_metrics import paired_h2_h4_request_bootstrap  # noqa: E402
from harp_rtt.exact_k import exact_set_nll  # noqa: E402
from harp_rtt.losses import boundary_loss_per_endpoint  # noqa: E402
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.training import move_to_device, seed_everything, sha256_file  # noqa: E402
from runpod.train_harp_delta_v3 import (  # noqa: E402
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
    load_split,
    loader,
)


SCHEMA = "harp_deltaroute_v4_content_transition_probe_v1"
RESULT_SCHEMA = "harp_deltaroute_v4_content_transition_probe_result_v1"
EFFECTIVE_BATCH = 32
MICROBATCH_CHOICES = (32, 16, 8, 4, 2, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "tune", "development"):
        parser.add_argument(f"--{split}-index", type=Path, required=True)
        parser.add_argument(f"--{split}-corpus", type=Path, required=True)
        parser.add_argument(f"--{split}-companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--microbatch-size", type=int, choices=(0, *MICROBATCH_CHOICES), default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def write_checksums(output: Path) -> None:
    files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush(); os.fsync(handle.fileno())


def build_model(static: Any) -> FullStateTransitionProbe:
    geometry = static.geometry
    config = ContentTransitionConfig(
        layers=geometry.layers, experts=geometry.experts,
        hidden_width=geometry.hidden_width,
        router_rank=geometry.maximum_rank, exact_k=8,
        latent_width=256, effect_width=64, transition_width=512,
        content_adapter_rank=8, output_adapter_rank=8, dropout=0.05,
    )
    return FullStateTransitionProbe(
        config, geometry.input_basis, geometry.expert_keys,
        geometry.centered_bias, geometry.rank_mask,
    )


def batch_tensors(host: Mapping[str, Any], device: torch.device) -> tuple[Tensor, ...]:
    batch = move_to_device(host, device)
    targets = batch.get("targets")
    if not isinstance(targets, Mapping):
        raise TypeError("content probe batch lacks targets")
    required = (
        "future_router_inputs", "future_router_logits", "future_selected_ids",
        "future_execution_weights", "future_available",
    )
    if any(name not in targets for name in required):
        raise KeyError("content probe batch lacks a factual route target")
    return (
        targets["future_router_inputs"], targets["future_router_logits"],
        targets["future_selected_ids"], targets["future_execution_weights"],
        targets["future_available"],
    )


def objective(
    model: FullStateTransitionProbe,
    host: Mapping[str, Any],
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    inputs, target_logits, target_ids, target_weights, available = batch_tensors(host, device)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        output = model(inputs, target_ids, target_weights)
    valid = available[..., 1:].bool()
    labels = target_ids[..., 1:, :].long()
    logits = target_logits[..., 1:, :].float()
    exact = exact_set_nll(
        output.predicted_scores, labels, valid=valid, k=model.config.exact_k
    )
    predicted_centered = output.predicted_scores.float()
    predicted_centered = predicted_centered - predicted_centered.mean(-1, keepdim=True)
    target_centered = logits - logits.mean(-1, keepdim=True)
    per_row = F.huber_loss(
        predicted_centered, target_centered, reduction="none", delta=1.0
    ).mean(-1)
    dense = (per_row * valid.float()).sum() / valid.sum().clamp_min(1)
    boundary_rows = boundary_loss_per_endpoint(
        output.predicted_scores, logits, labels,
        model_rank_start=6, teacher_rank_start=9, rank_end=32,
    )
    boundary = (boundary_rows * valid.float()).sum() / valid.sum().clamp_min(1)
    total = exact + dense + 0.2 * boundary
    return total, {
        "exact_set": float(exact.detach()),
        "centered_logit_huber": float(dense.detach()),
        "top_boundary": float(boundary.detach()),
        "total": float(total.detach()),
    }


def request_ids(host: Mapping[str, Any]) -> list[str]:
    metadata = host.get("metadata")
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("request_id"), list):
        raise ValueError("content probe batch lacks request IDs")
    return [str(item) for item in metadata["request_id"]]


@torch.no_grad()
def evaluate(
    model: FullStateTransitionProbe,
    dataset: Dataset[Any],
    *,
    device: torch.device,
    microbatch: int,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    full_cells: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    ablated_cells: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    losses = rows = 0.0
    for host in loader(
        dataset, batch=microbatch, shuffle=False, seed=0,
        workers=workers, device=device,
    ):
        loss, _ = objective(model, host, device)
        inputs, _, ids, weights, available = batch_tensors(host, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            full = model(inputs, ids, weights, use_router_blind_content=True)
            ablated = model(inputs, ids, weights, use_router_blind_content=False)
        truth = ids[..., 1:, :].long()
        valid = available[..., 1:].bool()
        full_hit = (truth[..., None] == full.selected_ids[..., None, :]).any(-1).float().mean(-1)
        ablated_hit = (truth[..., None] == ablated.selected_ids[..., None, :]).any(-1).float().mean(-1)
        requests = request_ids(host)
        for row, request in enumerate(requests):
            for horizon in (2, 3, 4):
                active = valid[row, horizon - 1]
                full_cells[(request, horizon)].append(full_hit[row, horizon - 1][active].cpu())
                ablated_cells[(request, horizon)].append(ablated_hit[row, horizon - 1][active].cpu())
        losses += float(loss) * inputs.shape[0]
        rows += inputs.shape[0]

    def metric_rows(cells: Mapping[tuple[str, int], list[Tensor]]) -> list[dict[str, Any]]:
        return [
            {"request_id": request, "horizon": horizon,
             "slot_recall_at_8": float(torch.cat(values).mean())}
            for (request, horizon), values in sorted(cells.items())
        ]

    full_rows = metric_rows(full_cells)
    ablated_rows = metric_rows(ablated_cells)

    def horizon_mean(values: list[dict[str, Any]], horizon: int) -> float:
        selected = [float(row["slot_recall_at_8"]) for row in values if row["horizon"] == horizon]
        return sum(selected) / len(selected)

    full_h = {h: horizon_mean(full_rows, h) for h in (2, 3, 4)}
    ablated_h = {h: horizon_mean(ablated_rows, h) for h in (2, 3, 4)}
    bootstrap = paired_h2_h4_request_bootstrap(
        full_rows, ablated_rows, replicates=1_000, seed=42
    )
    metrics = {
        "loss": losses / max(1.0, rows),
        "full_state_route_recall_h2_h4": sum(full_h.values()) / 3,
        "router_visible_ablation_recall_h2_h4": sum(ablated_h.values()) / 3,
        **{f"full_state_route_recall_h{h}": full_h[h] for h in (2, 3, 4)},
        **{f"router_visible_ablation_recall_h{h}": ablated_h[h] for h in (2, 3, 4)},
        "paired_content_bootstrap": bootstrap,
    }
    return metrics, full_rows, ablated_rows


def autotune(
    model: FullStateTransitionProbe,
    dataset: Dataset[Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    choices = (args.microbatch_size,) if args.microbatch_size else MICROBATCH_CHOICES
    trace: list[dict[str, Any]] = []
    for size in choices:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            host = next(iter(loader(
                dataset, batch=size, shuffle=False, seed=0,
                workers=args.num_workers, device=device,
            )))
            model.zero_grad(set_to_none=True)
            objective(model, host, device)[0].backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
            accepted = device.type != "cuda" or peak <= 21.0
            trace.append({"microbatch": size, "peak_reserved_gib": peak, "accepted": accepted})
            if accepted:
                return size, trace
        except torch.OutOfMemoryError:
            trace.append({"microbatch": size, "oom": True, "accepted": False})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("content transition probe does not fit the 3090 memory gate")


def main() -> None:
    args = parse_args()
    args.data_profile = "b2_reuse_4096"
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite content probe {args.output}")
    partition = validate_partition(args.partition_manifest, args.data_profile)
    reuse = validate_reuse_split(args.reuse_split_manifest)
    partition_sha = sha256_file(args.partition_manifest)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    seed_everything(args.seed, deterministic=args.deterministic)
    selected = {
        "train": set(reuse["inner_split"]["training_requests"]),
        "tune": set(reuse["inner_split"]["tuning_requests"]),
        "development": None,
    }
    datasets: dict[str, Dataset[Any]] = {}
    groups: dict[str, set[str]] = {}
    for split in ("train", "tune", "development"):
        datasets[split], groups[split] = load_split(
            args, split, selected_requests=selected[split]
        )
    if groups["train"] & groups["tune"] or groups["train"] & groups["development"] or groups["tune"] & groups["development"]:
        raise PermissionError("content probe request groups overlap")

    static = load_static_target_artifacts(args.static_dir, device="cpu")
    model = build_model(static).to(device)
    trainable = [(name, parameter) for name, parameter in model.named_parameters()]
    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "seed": args.seed,
        "config": model.config.to_dict(),
        "partition_schema": partition["schema"],
        "partition_manifest_sha256": partition_sha,
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "static_manifest_sha256": sha256_file(args.static_dir / "manifest.json"),
        "diagnostic_only": True,
        "teacher_full_state_is_model_input": True,
        "serving_authorized": False,
        "optimizer_constructed": False,
        "trainable_names": [name for name, _ in trainable],
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)
    microbatch, memory_trace = autotune(model, datasets["train"], args, device)
    write_json_exclusive(args.output / "MEMORY_AUTOTUNE.json", {
        "selected_microbatch": microbatch, "effective_batch": EFFECTIVE_BATCH,
        "trace": memory_trace, "optimizer_constructed": False,
    })
    if args.preflight_only:
        write_json_exclusive(args.output / "PREFLIGHT_RESULT.json", {
            "training_started": False, "serving_authorized": False,
            "selected_microbatch": microbatch,
        })
        write_checksums(args.output)
        return

    optimizer = torch.optim.AdamW(
        (parameter for _, parameter in trainable),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    write_json_exclusive(args.output / "OPTIMIZER_START.json", {
        "optimizer": "AdamW", "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    })
    accumulation = EFFECTIVE_BATCH // microbatch
    best_value = -math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        total = batches = 0.0
        for step, host in enumerate(loader(
            datasets["train"], batch=microbatch, shuffle=True, seed=args.seed + epoch,
            workers=args.num_workers, device=device,
        ), start=1):
            loss, _ = objective(model, host, device)
            (loss / accumulation).backward()
            total += float(loss.detach()); batches += 1
            if step % accumulation == 0 or step * microbatch >= len(datasets["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        tune, _, _ = evaluate(
            model, datasets["tune"], device=device, microbatch=microbatch,
            workers=args.num_workers,
        )
        append_jsonl(args.output / "metrics.jsonl", {
            "epoch": epoch, "train_loss": total / max(1.0, batches), "tune": tune,
        })
        value = float(tune["full_state_route_recall_h2_h4"])
        if value > best_value:
            best_value = value; best_epoch = epoch; stale = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("content probe produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    development, full_rows, ablated_rows = evaluate(
        model, datasets["development"], device=device, microbatch=microbatch,
        workers=args.num_workers,
    )
    write_rows(args.output / "development_full_state_predictions.jsonl", full_rows)
    write_rows(args.output / "development_router_visible_ablation.jsonl", ablated_rows)
    bootstrap = development["paired_content_bootstrap"]
    per_horizon_positive = all(
        development[f"full_state_route_recall_h{h}"]
        > development[f"router_visible_ablation_recall_h{h}"]
        for h in (2, 3, 4)
    )
    gate = {
        "full_state_route_recall_h2_h4": development["full_state_route_recall_h2_h4"],
        "required_route_recall": 0.80,
        "paired_lower_bound_positive": bootstrap["lower_bound_positive"],
        "per_horizon_content_gain_positive": per_horizon_positive,
    }
    gate["passed"] = bool(
        gate["full_state_route_recall_h2_h4"] >= 0.80
        and gate["paired_lower_bound_positive"]
        and gate["per_horizon_content_gain_positive"]
    )
    checkpoint = {
        "schema": SCHEMA, "source_commit": args.source_commit,
        "seed": args.seed, "config": model.config.to_dict(),
        "model_state_dict": best_state, "best_epoch": best_epoch,
        "best_tune_value": best_value, "development": development, "gate": gate,
        "partition_manifest_sha256": partition_sha,
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "diagnostic_only": True, "serving_authorized": False,
        "formal_validation_opened": False, "calibration_opened": False,
        "sealed_test_opened": False,
    }
    checkpoint_path = args.output / "best_content_transition_probe.pt"
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle); handle.flush(); os.fsync(handle.fileno())
    result = {
        "schema": RESULT_SCHEMA, "best_epoch": best_epoch,
        "best_tune_value": best_value, "development": development, "gate": gate,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "content_distillation_authorized": gate["passed"],
        "training_started": True, "serving_authorized": False,
        "formal_validation_opened": False, "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    write_checksums(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
