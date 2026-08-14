#!/usr/bin/env python3
"""Train one layer-local HARP-ShadowRoute S0, S1, or S2 expert shard.

This stage reuses already-indexed outer-train factual states.  It loads only
one native target expert layer, never constructs the 67 GiB target, and never
opens formal validation, calibration, or sealed-test rows.  Its result is a
component checkpoint; only the later exact-backbone closed-loop evaluation can
authorize ShadowRoute.
"""

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
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.future_state_adapter import FutureTargetStateAdapter  # noqa: E402
from harp_rtt.shadow_checkpoint import (  # noqa: E402
    IndexedCheckpoint,
    load_target_layer_experts,
    sha256_file,
)
from harp_rtt.shadow_expert import (  # noqa: E402
    IndexedShadowExperts,
    ShadowExpertConfig,
    SwiGLUDraftExpert,
    target_neuron_importance,
    target_selected_expert_outputs,
)
from harp_rtt.shadow_training import selected_expert_distillation_loss  # noqa: E402
from harp_rtt.training import move_to_device, seed_everything  # noqa: E402
from runpod.train_harp_delta_v3 import (  # noqa: E402
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
    load_split as load_delta_split,
    loader,
)


SCHEMA = "harp_shadowroute_local_expert_layer_v1"
RESULT_SCHEMA = "harp_shadowroute_local_expert_layer_result_v1"
MODES = ("s0_exact_top1_plus_draft", "s1_shared", "s2_indexed")
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
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--layer", type=int, choices=range(40), required=True)
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
    parser.add_argument("--minimum-expert-count", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_checksums(output: Path) -> None:
    paths = sorted(path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_split(
    args: argparse.Namespace,
    split: str,
    *,
    selected_requests: set[str] | None,
) -> tuple[FutureTargetStateAdapter, set[str]]:
    base, requests = load_delta_split(
        args, split, selected_requests=selected_requests
    )
    return (
        FutureTargetStateAdapter(
            base,
            roles=(
                "post_attention_residual_u",
                "routed_expert_output_delta_r",
                "post_moe_residual_xplus",
            ),
        ),
        requests,
    )


def build_student(mode: str, device: torch.device) -> nn.Module:
    if mode == "s0_exact_top1_plus_draft":
        return SwiGLUDraftExpert(2048, 512).to(device=device, dtype=torch.bfloat16)
    if mode == "s1_shared":
        return SwiGLUDraftExpert(2048, 128).to(device=device, dtype=torch.bfloat16)
    return IndexedShadowExperts(
        ShadowExpertConfig(shadow_width=16), fallback=None
    ).to(device=device, dtype=torch.bfloat16)


def batch_tensors(
    host: Mapping[str, Any], *, layer: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[str]]:
    batch = move_to_device(host, device)
    targets = batch.get("targets")
    metadata = batch.get("metadata")
    if not isinstance(targets, Mapping) or not isinstance(metadata, Mapping):
        raise TypeError("ShadowRoute local batch lacks targets or metadata")
    states = targets.get("future_states")
    if not isinstance(states, Mapping):
        raise TypeError("ShadowRoute full-state labels are not target-only")
    if "future_states" in batch.get("inputs", {}):
        raise PermissionError("future target states leaked into ShadowRoute inputs")
    requests = metadata.get("request_id")
    if not isinstance(requests, list):
        raise ValueError("ShadowRoute batch lacks request IDs")
    return (
        targets["future_router_inputs"][:, :, layer],
        states["routed_expert_output_delta_r"][:, :, layer],
        targets["future_selected_ids"][:, :, layer],
        targets["future_execution_weights"][:, :, layer],
        targets["future_available"][:, :, layer].bool(),
        [str(value) for value in requests],
    )


def teacher_values(
    inputs: Tensor,
    ids: Tensor,
    weights: Tensor,
    gate_up: Tensor,
    down: Tensor,
) -> tuple[Tensor, Tensor]:
    with torch.no_grad():
        individual = target_selected_expert_outputs(inputs, ids, gate_up, down)
        aggregate = (individual * weights[..., None].to(individual)).sum(-2)
    return individual, aggregate


def objective(
    model: nn.Module,
    host: Mapping[str, Any],
    *,
    mode: str,
    layer: int,
    device: torch.device,
    gate_up: Tensor,
    down: Tensor,
) -> tuple[Tensor, dict[str, float], Tensor, Tensor, Tensor, list[str]]:
    inputs, routed_target, ids, weights, valid, requests = batch_tensors(
        host, layer=layer, device=device
    )
    individual_target, reconstructed = teacher_values(
        inputs, ids, weights, gate_up, down
    )
    reconstruction_error = (
        (reconstructed.float() - routed_target.float()).square().mean(-1)
    )
    if valid.any() and float(reconstruction_error[valid].max()) > 0.05:
        raise ValueError("native expert replay does not reconstruct captured routed residual")
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        if mode == "s0_exact_top1_plus_draft":
            exact_top1 = individual_target[..., 0, :] * weights[..., 0, None].to(inputs)
            predicted = exact_top1 + model(inputs)
            individual_loss = predicted.sum() * 0.0
        elif mode == "s1_shared":
            predicted = model(inputs)
            individual_loss = predicted.sum() * 0.0
        else:
            assert isinstance(model, IndexedShadowExperts)
            predicted_individual = model.selected_unweighted(inputs, ids)
            predicted = (predicted_individual * weights[..., None].to(inputs)).sum(-2)
            individual_loss = selected_expert_distillation_loss(
                predicted_individual, individual_target, weights, valid
            )
    rows = F.huber_loss(
        predicted.float(), routed_target.float(), reduction="none", delta=1.0
    ).mean(-1)
    aggregate = (rows * valid.float()).sum() / valid.sum().clamp_min(1)
    cosine_rows = 1.0 - F.cosine_similarity(
        predicted.float(), routed_target.float(), dim=-1, eps=1e-8
    )
    cosine = (cosine_rows * valid.float()).sum() / valid.sum().clamp_min(1)
    loss = aggregate + 0.1 * cosine + individual_loss
    return (
        loss,
        {
            "aggregate_huber": float(aggregate.detach()),
            "aggregate_cosine_distance": float(cosine.detach()),
            "individual_huber": float(individual_loss.detach()),
            "native_reconstruction_max_mse": float(
                reconstruction_error[valid].max() if valid.any() else 0.0
            ),
            "total": float(loss.detach()),
        },
        predicted.detach(),
        routed_target.detach(),
        valid.detach(),
        requests,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Dataset[Any],
    *,
    mode: str,
    layer: int,
    device: torch.device,
    gate_up: Tensor,
    down: Tensor,
    microbatch: int,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    request_cells: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
    error_sum = energy_sum = dot_sum = pred_norm = target_norm = 0.0
    elements = 0
    for host in loader(
        dataset, batch=microbatch, shuffle=False, seed=0,
        workers=workers, device=device,
    ):
        _loss, _parts, predicted, target, valid, requests = objective(
            model, host, mode=mode, layer=layer, device=device,
            gate_up=gate_up, down=down,
        )
        error = (predicted.float() - target.float()).square().mean(-1)
        energy = target.float().square().mean(-1)
        cosine = F.cosine_similarity(predicted.float(), target.float(), dim=-1, eps=1e-8)
        for row, request in enumerate(requests):
            for horizon in range(4):
                if bool(valid[row, horizon]):
                    request_cells[(request, horizon + 1)].append(
                        (
                            float(error[row, horizon]),
                            float(energy[row, horizon]),
                            float(cosine[row, horizon]),
                        )
                    )
        active_pred = predicted.float()[valid]
        active_target = target.float()[valid]
        error_sum += float((active_pred - active_target).square().sum())
        energy_sum += float(active_target.square().sum())
        dot_sum += float((active_pred * active_target).sum())
        pred_norm += float(active_pred.square().sum())
        target_norm += float(active_target.square().sum())
        elements += active_target.numel()
    if elements == 0:
        raise ValueError("ShadowRoute evaluation contains no valid states")
    normalized_rmse = math.sqrt(error_sum / max(energy_sum, 1e-12))
    cosine = dot_sum / max(math.sqrt(pred_norm * target_norm), 1e-12)
    rows = []
    for (request, horizon), values in sorted(request_cells.items()):
        rows.append(
            {
                "request_id": request,
                "horizon": horizon,
                "layer": layer,
                "mse": sum(value[0] for value in values) / len(values),
                "target_energy": sum(value[1] for value in values) / len(values),
                "cosine": sum(value[2] for value in values) / len(values),
            }
        )
    return (
        {
            "normalized_rmse": normalized_rmse,
            "cosine": cosine,
            "relative_mse_reduction_vs_zero": 1.0 - error_sum / max(energy_sum, 1e-12),
            "rows": len(rows),
        },
        rows,
    )


def expert_counts(
    dataset: Dataset[Any], *, layer: int, device: torch.device, workers: int
) -> Tensor:
    counts = torch.zeros(256, dtype=torch.int64)
    for host in loader(
        dataset, batch=32, shuffle=False, seed=0, workers=workers, device=device
    ):
        _inputs, _routed, ids, _weights, valid, _requests = batch_tensors(
            host, layer=layer, device=device
        )
        active = ids[valid].detach().cpu().flatten()
        counts += torch.bincount(active, minlength=256)
    return counts


def autotune(
    model: nn.Module,
    dataset: Dataset[Any],
    args: argparse.Namespace,
    *,
    device: torch.device,
    gate_up: Tensor,
    down: Tensor,
) -> tuple[int, list[dict[str, Any]]]:
    choices = (args.microbatch_size,) if args.microbatch_size else MICROBATCH_CHOICES
    trace: list[dict[str, Any]] = []
    for size in choices:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            host = next(iter(loader(
                dataset, batch=size, shuffle=False, seed=0,
                workers=args.num_workers, device=device,
            )))
            model.zero_grad(set_to_none=True)
            objective(
                model, host, mode=args.mode, layer=args.layer, device=device,
                gate_up=gate_up, down=down,
            )[0].backward()
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
    raise RuntimeError("ShadowRoute layer shard exceeds the 21 GiB training gate")


def main() -> None:
    args = parse_args()
    args.data_profile = "b2_reuse_4096"
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite ShadowRoute run {args.output}")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    partition = validate_partition(args.partition_manifest, args.data_profile)
    reuse = validate_reuse_split(args.reuse_split_manifest)
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
    if any(
        groups[left] & groups[right]
        for left, right in (("train", "tune"), ("train", "development"), ("tune", "development"))
    ):
        raise PermissionError("ShadowRoute request groups overlap")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(args.seed, deterministic=args.deterministic)
    checkpoint = IndexedCheckpoint(args.target_model)
    gate_up, down = load_target_layer_experts(
        checkpoint, args.layer, device=device, dtype=torch.bfloat16
    )
    model = build_student(args.mode, device)
    selected_neurons = None
    if isinstance(model, IndexedShadowExperts):
        selected_neurons = model.initialize_from_target_neurons(
            gate_up, down, target_neuron_importance(gate_up, down)
        ).cpu()
    counts = expert_counts(
        datasets["train"], layer=args.layer, device=device, workers=args.num_workers
    )
    trained_mass = float(
        counts[counts >= args.minimum_expert_count].sum() / counts.sum().clamp_min(1)
    )

    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "mode": args.mode,
        "layer": args.layer,
        "seed": args.seed,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "partition_schema": partition["schema"],
        "diagnostic_reuse": True,
        "target_state_is_label_only": True,
        "native_target_layer_loaded": True,
        "complete_target_model_loaded": False,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "trainable_names": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "minimum_expert_count": args.minimum_expert_count,
        "trained_expert_count": int((counts >= args.minimum_expert_count).sum()),
        "trained_selected_slot_mass": trained_mass,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)
    microbatch, trace = autotune(
        model, datasets["train"], args, device=device, gate_up=gate_up, down=down
    )
    write_json_exclusive(
        args.output / "MEMORY_AUTOTUNE.json",
        {
            "selected_microbatch": microbatch,
            "effective_batch": EFFECTIVE_BATCH,
            "trace": trace,
            "optimizer_constructed": False,
        },
    )
    if args.preflight_only:
        write_json_exclusive(
            args.output / "PREFLIGHT_RESULT.json",
            {
                "training_started": False,
                "closed_loop_authorized": False,
                "selected_microbatch": microbatch,
            },
        )
        write_checksums(args.output)
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    write_json_exclusive(
        args.output / "OPTIMIZER_START.json",
        {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
    )
    accumulation = math.ceil(EFFECTIVE_BATCH / microbatch)
    best_value = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = batches = 0.0
        for step, host in enumerate(
            loader(
                datasets["train"], batch=microbatch, shuffle=True,
                seed=args.seed + epoch, workers=args.num_workers, device=device,
            ),
            start=1,
        ):
            loss, _parts, *_ = objective(
                model, host, mode=args.mode, layer=args.layer, device=device,
                gate_up=gate_up, down=down,
            )
            (loss / accumulation).backward()
            total += float(loss.detach())
            batches += 1
            if step % accumulation == 0 or step * microbatch >= len(datasets["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        tune, _ = evaluate(
            model, datasets["tune"], mode=args.mode, layer=args.layer,
            device=device, gate_up=gate_up, down=down,
            microbatch=microbatch, workers=args.num_workers,
        )
        append_jsonl(
            args.output / "metrics.jsonl",
            {"epoch": epoch, "train_loss": total / max(batches, 1.0), "tune": tune},
        )
        value = float(tune["normalized_rmse"])
        if value < best_value:
            best_value = value
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("ShadowRoute local trainer produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    development, rows = evaluate(
        model, datasets["development"], mode=args.mode, layer=args.layer,
        device=device, gate_up=gate_up, down=down,
        microbatch=microbatch, workers=args.num_workers,
    )
    write_rows(args.output / "development_residual_predictions.jsonl", rows)
    component_gate = bool(
        development["relative_mse_reduction_vs_zero"] >= 0.50
        and development["cosine"] >= 0.70
    )
    checkpoint_value = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "mode": args.mode,
        "layer": args.layer,
        "seed": args.seed,
        "model_state_dict": best_state,
        "selected_neurons": selected_neurons,
        "expert_counts": counts,
        "minimum_expert_count": args.minimum_expert_count,
        "trained_selected_slot_mass": trained_mass,
        "best_epoch": best_epoch,
        "best_tune_normalized_rmse": best_value,
        "development": development,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "diagnostic_only": True,
        "closed_loop_authorized": component_gate,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    checkpoint_path = args.output / f"shadow_{args.mode}_layer_{args.layer:02d}.pt"
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint_value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    result = {
        "schema": RESULT_SCHEMA,
        "mode": args.mode,
        "layer": args.layer,
        "best_epoch": best_epoch,
        "development": development,
        "component_gate_passed": component_gate,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_started": True,
        "closed_loop_evaluation_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    write_checksums(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
