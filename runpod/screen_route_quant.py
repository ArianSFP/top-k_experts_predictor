#!/usr/bin/env python3
"""Screen full-width RouteQuant codecs on frozen representative target layers.

This is an optimizer-free, no-recapture component experiment.  It compresses
one native target layer at a time, executes all eight authoritative route IDs
with their original weights, and evaluates residual fidelity plus the frozen
teacher-forced next-router set objective on request-disjoint tuning rows.
Diagnostic-development rows are loaded only to prove lineage disjointness;
their labels are never evaluated by this stage.
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

import torch
from torch import Tensor
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.exact_k import stable_topk  # noqa: E402
from harp_rtt.route_quant import (  # noqa: E402
    PackedRouteQuantExperts,
    asymmetric_routequant_projected_bytes,
    mixed_routequant_projected_bytes,
    uniform_routequant_projected_bytes,
)
from harp_rtt.shadow_checkpoint import (  # noqa: E402
    IndexedCheckpoint,
    load_target_layer_experts,
    sha256_file,
)
from harp_rtt.training import seed_everything  # noqa: E402
from runpod.train_harp_delta_v3 import (  # noqa: E402
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
    loader,
)
from runpod.train_shadow_experts_local import (  # noqa: E402
    ShadowFactualStateDataset,
    batch_tensors,
    load_split,
    next_router_agreement_loss,
)


SCHEMA = "harp_routequant_representative_screen_v1"
RESULT_SCHEMA = "harp_routequant_representative_screen_result_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "tune", "development"):
        parser.add_argument(f"--{split}-index", type=Path, required=True)
        parser.add_argument(f"--{split}-corpus", type=Path, required=True)
        parser.add_argument(f"--{split}-companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,6,20,32")
    parser.add_argument(
        "--candidates",
        default="1:mse,2:mse,3:mse,4:amax,4:mse",
        help="comma-separated BIT:SCALE_METHOD candidates",
    )
    parser.add_argument(
        "--asymmetric-candidates",
        default="",
        help="optional comma-separated GATE_UP_BITS/DOWN_BITS:SCALE_METHOD candidates",
    )
    parser.add_argument("--group-size", type=int, choices=(32, 64), default=64)
    parser.add_argument("--scale-storage", choices=("bf16", "log8", "fp8_e4m3"), default="bf16")
    parser.add_argument(
        "--mixed-upgrade-fractions",
        default="",
        help="optional comma-separated fractions for a train-planned adjacent-bit sweep",
    )
    parser.add_argument("--mixed-base-bits", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--mixed-upgrade-bits", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--item-chunk", type=int, default=2)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8, 16, 32), default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def resolve_scale_storage(name: str) -> tuple[torch.dtype, int]:
    if name == "bf16":
        return torch.bfloat16, 2
    if name == "log8":
        return torch.uint8, 1
    if name == "fp8_e4m3":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("this PyTorch build lacks FP8 E4M3 scale storage")
        return torch.float8_e4m3fn, 1
    raise ValueError("unknown RouteQuant scale storage")


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


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part) for part in value.split(",") if part)
    if not layers or len(set(layers)) != len(layers) or any(layer not in range(39) for layer in layers):
        raise ValueError("representative layers must be unique values in [0,39)")
    return layers


def parse_candidates(value: str) -> tuple[tuple[int, str], ...]:
    result = []
    for item in value.split(","):
        bits_text, separator, method = item.partition(":")
        if not separator:
            raise ValueError("RouteQuant candidate must be BIT:SCALE_METHOD")
        bits = int(bits_text)
        if bits not in (1, 2, 3, 4) or method not in {"amax", "mse"}:
            raise ValueError("RouteQuant candidate is unsupported")
        result.append((bits, method))
    if not result or len(set(result)) != len(result):
        raise ValueError("RouteQuant candidates must be non-empty and unique")
    return tuple(result)


def parse_asymmetric_candidates(value: str) -> tuple[tuple[int, int, str], ...]:
    if not value:
        return ()
    result = []
    for item in value.split(","):
        widths, separator, method = item.partition(":")
        gate_text, slash, down_text = widths.partition("/")
        if not separator or not slash:
            raise ValueError("asymmetric candidate must be GU_BITS/DOWN_BITS:METHOD")
        gate_bits, down_bits = int(gate_text), int(down_text)
        if gate_bits not in (1, 2, 3, 4) or down_bits not in (1, 2, 3, 4):
            raise ValueError("asymmetric RouteQuant bit width is unsupported")
        if method not in {"amax", "mse"}:
            raise ValueError("asymmetric RouteQuant scale method is unsupported")
        result.append((gate_bits, down_bits, method))
    if len(set(result)) != len(result):
        raise ValueError("asymmetric RouteQuant candidates must be unique")
    return tuple(result)


def parse_upgrade_fractions(value: str) -> tuple[float, ...]:
    if not value:
        return ()
    result = tuple(float(item) for item in value.split(","))
    if (
        len(set(result)) != len(result)
        or any(not 0.0 < item < 1.0 for item in result)
        or tuple(sorted(result)) != result
    ):
        raise ValueError("mixed upgrade fractions must be unique, increasing, and in (0,1)")
    return result


def upgrade_schedule(
    scores: Tensor,
    *,
    fraction: float,
    base_bits: int = 3,
    upgrade_bits: int = 4,
) -> Tensor:
    values = torch.as_tensor(scores, dtype=torch.float64).cpu()
    if values.ndim != 1 or values.numel() < 1 or not torch.isfinite(values).all():
        raise ValueError("RouteQuant utility score must be one finite vector")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("RouteQuant upgrade fraction must lie in (0,1)")
    count = max(1, min(values.numel() - 1, round(values.numel() * fraction)))
    ordered = sorted(range(values.numel()), key=lambda expert: (-float(values[expert]), expert))
    schedule = torch.full((values.numel(),), base_bits, dtype=torch.int8)
    schedule[ordered[:count]] = upgrade_bits
    return schedule


def _request_metrics(
    request_cells: Mapping[tuple[str, int], list[float]],
) -> tuple[float, list[float], int]:
    per_horizon: list[list[float]] = [[], [], [], []]
    requests = set()
    for (request, horizon), values in request_cells.items():
        requests.add(request)
        per_horizon[horizon].append(sum(values) / len(values))
    if any(not values for values in per_horizon):
        raise ValueError("RouteQuant evaluation lacks a complete horizon")
    horizons = [sum(values) / len(values) for values in per_horizon]
    return sum(horizons) / 4.0, horizons, len(requests)


@torch.no_grad()
def evaluate_candidate(
    model: PackedRouteQuantExperts | None,
    dataset: Any,
    *,
    layer: int,
    device: torch.device,
    microbatch: int,
    workers: int,
    next_norm_weight: Tensor,
    next_router_weight: Tensor,
    exact_teacher: bool = False,
    zero_teacher: bool = False,
) -> dict[str, Any]:
    if exact_teacher and zero_teacher:
        raise ValueError("RouteQuant evaluation baseline is ambiguous")
    if model is not None:
        model.eval()
    request_cells: dict[tuple[str, int], list[float]] = defaultdict(list)
    error_sum = energy_sum = dot_sum = predicted_norm = target_norm = 0.0
    total_rows = 0
    loss_parts: dict[str, float] = defaultdict(float)
    for host in loader(
        dataset, batch=microbatch, shuffle=False, seed=0,
        workers=workers, device=device,
    ):
        inputs, routed, ids, weights, valid, requests, states = batch_tensors(
            host, layer=layer, device=device
        )
        if exact_teacher:
            predicted = routed
        elif zero_teacher:
            predicted = torch.zeros_like(routed)
        else:
            if model is None:
                raise ValueError("RouteQuant candidate model is absent")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                predicted = model(inputs, ids, weights)
        agreement, parts, logits = next_router_agreement_loss(
            predicted,
            states,
            valid,
            norm_weight=next_norm_weight,
            router_weight=next_router_weight,
        )
        teacher_ids = states["next_selected_expert_ids"].long()
        predicted_ids = stable_topk(logits, k=8)
        overlap = (
            teacher_ids[..., None] == predicted_ids[..., None, :]
        ).any(-1).float().mean(-1)
        active_rows = int(valid.sum())
        for name, value in {"next_router_total": agreement, **parts}.items():
            loss_parts[name] += float(value) * active_rows
        total_rows += active_rows
        for row, request in enumerate(requests):
            for horizon in range(4):
                if bool(valid[row, horizon]):
                    request_cells[(request, horizon)].append(float(overlap[row, horizon]))
        active_predicted = predicted.float()[valid]
        active_target = routed.float()[valid]
        difference = active_predicted - active_target
        error_sum += float(difference.square().sum())
        energy_sum += float(active_target.square().sum())
        dot_sum += float((active_predicted * active_target).sum())
        predicted_norm += float(active_predicted.square().sum())
        target_norm += float(active_target.square().sum())
    if total_rows == 0:
        raise ValueError("RouteQuant evaluation has no valid rows")
    request_macro, horizons, requests = _request_metrics(request_cells)
    return {
        "request_macro_next_router_recall_at_8": request_macro,
        "next_router_recall_at_8_by_horizon": horizons,
        "requests": requests,
        "active_rows": total_rows,
        "normalized_residual_rmse": math.sqrt(error_sum / max(energy_sum, 1e-30)),
        "residual_cosine": dot_sum / max(
            math.sqrt(predicted_norm * target_norm), 1e-30
        ),
        **{name: value / total_rows for name, value in loss_parts.items()},
    }


def collect_upgrade_utility(
    base: PackedRouteQuantExperts,
    upgraded: PackedRouteQuantExperts,
    dataset: Any,
    *,
    layer: int,
    device: torch.device,
    microbatch: int,
    workers: int,
    next_norm_weight: Tensor,
    next_router_weight: Tensor,
) -> dict[str, Any]:
    """Estimate train-only INT3-to-INT4 cell utility.

    The primary term is the first-order reduction of the authoritative
    next-router set/KL/boundary loss. A smaller exact routed-residual gain term
    stabilises cells whose route gradient is locally flat. Horizons are
    balanced and H4 receives twice H1's weight.
    """

    base.eval()
    upgraded.eval()
    experts = base.config.experts
    route_utility = torch.zeros(experts, dtype=torch.float64, device=device)
    residual_utility = torch.zeros_like(route_utility)
    occurrences = torch.zeros_like(route_utility)
    horizon_weights = torch.tensor(
        [1.0, 1.0, 1.25, 2.0], dtype=torch.float32, device=device
    )
    batches = active_rows = 0
    for host in loader(
        dataset, batch=microbatch, shuffle=False, seed=0,
        workers=workers, device=device,
    ):
        inputs, routed, ids, weights, valid, _requests, states = batch_tensors(
            host, layer=layer, device=device
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            base_values = base.selected_unweighted(inputs, ids)
            upgraded_values = upgraded.selected_unweighted(inputs, ids)
            base_predicted = (
                base_values * weights.to(base_values)[..., None]
            ).sum(-2)
            delta = (
                (upgraded_values - base_values)
                * weights.to(base_values)[..., None]
            ).float()
        predicted_for_grad = base_predicted.detach().float().requires_grad_(True)
        route_objective = predicted_for_grad.sum() * 0.0
        active_weight = 0.0
        for horizon, horizon_weight in enumerate(horizon_weights):
            horizon_valid = torch.zeros_like(valid)
            horizon_valid[:, horizon] = valid[:, horizon]
            if bool(horizon_valid.any()):
                current, _parts, _logits = next_router_agreement_loss(
                    predicted_for_grad,
                    states,
                    horizon_valid,
                    norm_weight=next_norm_weight,
                    router_weight=next_router_weight,
                )
                route_objective = route_objective + float(horizon_weight) * current
                active_weight += float(horizon_weight)
        if active_weight <= 0:
            continue
        route_objective = route_objective / active_weight
        gradient = torch.autograd.grad(route_objective, predicted_for_grad)[0]
        route_gain = -(gradient[..., None, :] * delta).sum(-1)
        error = routed.float() - base_predicted.float()
        residual_gain = (
            error.square().mean(-1, keepdim=True)
            - (error[..., None, :] - delta).square().mean(-1)
        )
        residual_gain = residual_gain * horizon_weights.view(1, 4, 1)
        active = valid[..., None].expand_as(ids)
        flat_ids = ids.long()[active]
        route_utility.index_add_(0, flat_ids, route_gain.detach()[active].double())
        residual_utility.index_add_(0, flat_ids, residual_gain.detach()[active].double())
        occurrences.index_add_(
            0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float64)
        )
        batches += 1
        active_rows += int(valid.sum())
    if batches == 0 or active_rows == 0:
        raise ValueError("RouteQuant utility planner saw no training rows")
    observed_experts = int((occurrences > 0).sum())
    if observed_experts == 0:
        raise ValueError("RouteQuant utility planner did not observe any expert")
    route_scale = route_utility.abs().sum().clamp_min(1e-30)
    residual_scale = residual_utility.abs().sum().clamp_min(1e-30)
    score = route_utility / route_scale + 0.25 * residual_utility / residual_scale
    if not torch.isfinite(score).all():
        raise ValueError("RouteQuant utility planner produced a non-finite score")
    return {
        "score": score.cpu(),
        "route_utility": route_utility.cpu(),
        "residual_utility": residual_utility.cpu(),
        "occurrences": occurrences.cpu(),
        "batches": batches,
        "active_rows": active_rows,
        "observed_experts": observed_experts,
        "unobserved_experts": experts - observed_experts,
        "horizon_weights": horizon_weights.cpu().tolist(),
        "utility_source": "training_only_first_order_next_router_plus_exact_residual",
    }


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    candidates = parse_candidates(args.candidates)
    asymmetric_candidates = parse_asymmetric_candidates(args.asymmetric_candidates)
    upgrade_fractions = parse_upgrade_fractions(args.mixed_upgrade_fractions)
    scale_dtype, scale_bytes = resolve_scale_storage(args.scale_storage)
    if upgrade_fractions and args.mixed_upgrade_bits != args.mixed_base_bits + 1:
        raise ValueError("mixed RouteQuant sweep requires adjacent bit widths")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite RouteQuant run {args.output}")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    args.data_profile = "b2_reuse_4096"
    args.layer = layers[0]
    args.next_router_agreement = True
    partition = validate_partition(args.partition_manifest, args.data_profile)
    reuse = validate_reuse_split(args.reuse_split_manifest)
    selected = {
        "train": set(reuse["inner_split"]["training_requests"]),
        "tune": set(reuse["inner_split"]["tuning_requests"]),
        "development": None,
    }
    datasets = {}
    groups = {}
    for split in ("train", "tune", "development"):
        datasets[split], groups[split] = load_split(
            args, split, selected_requests=selected[split]
        )
    if any(
        groups[left] & groups[right]
        for left, right in (
            ("train", "tune"), ("train", "development"), ("tune", "development")
        )
    ):
        raise PermissionError("RouteQuant request groups overlap")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(args.seed, deterministic=args.deterministic)
    checkpoint = IndexedCheckpoint(args.target_model)
    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "partition_schema": partition["schema"],
        "layers": list(layers),
        "candidates": [f"{bits}:{method}" for bits, method in candidates],
        "asymmetric_candidates": [
            f"{gate}/{down}:{method}"
            for gate, down, method in asymmetric_candidates
        ],
        "group_size": args.group_size,
        "scale_storage": args.scale_storage,
        "scale_storage_bytes": scale_bytes,
        "mixed_upgrade_fractions": list(upgrade_fractions),
        "mixed_base_bits": args.mixed_base_bits,
        "mixed_upgrade_bits": args.mixed_upgrade_bits,
        "mixed_utility_split": "train",
        "mixed_selection_split": "tune",
        "seed": args.seed,
        "optimizer_constructed": False,
        "training_started": False,
        "development_opened_for_metrics": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "target_expert_loads_at_runtime": False,
        "native_route_ids_and_weights_unchanged": True,
        "no_expert_substitution": True,
        "scientific_reference_dequantization": True,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)
    results: list[dict[str, Any]] = []
    for layer in layers:
        layer_datasets = {
            split: ShadowFactualStateDataset(
                dataset, layer=layer, next_router_agreement=True
            )
            for split, dataset in datasets.items()
        }
        gate_up, down = load_target_layer_experts(
            checkpoint, layer, device=device, dtype=torch.bfloat16
        )
        prefix = f"model.language_model.layers.{layer + 1}."
        names = (
            prefix + "post_attention_layernorm.weight",
            prefix + "mlp.gate.weight",
        )
        next_values = checkpoint.tensors(names)
        next_norm = next_values[names[0]].to(device=device, dtype=torch.bfloat16)
        next_router = next_values[names[1]].to(device=device, dtype=torch.bfloat16)
        baselines = {}
        for name, exact, zero in (("exact", True, False), ("zero", False, True)):
            baselines[name] = evaluate_candidate(
                None,
                layer_datasets["tune"],
                layer=layer,
                device=device,
                microbatch=args.microbatch,
                workers=args.num_workers,
                next_norm_weight=next_norm,
                next_router_weight=next_router,
                exact_teacher=exact,
                zero_teacher=zero,
            )
        if upgrade_fractions:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            base_model = PackedRouteQuantExperts.from_target(
                gate_up,
                down,
                gate_up_bits=args.mixed_base_bits,
                exact_k=8,
                group_size=args.group_size,
                scale_method="mse",
                item_chunk=args.item_chunk,
                scale_dtype=scale_dtype,
            ).to(device)
            upgraded_model = PackedRouteQuantExperts.from_target(
                gate_up,
                down,
                gate_up_bits=args.mixed_upgrade_bits,
                exact_k=8,
                group_size=args.group_size,
                scale_method="mse",
                item_chunk=args.item_chunk,
                scale_dtype=scale_dtype,
            ).to(device)
            base_model.enable_dequantized_cache(True, max_experts=256)
            upgraded_model.enable_dequantized_cache(True, max_experts=256)
            utility = collect_upgrade_utility(
                base_model,
                upgraded_model,
                layer_datasets["train"],
                layer=layer,
                device=device,
                microbatch=args.microbatch,
                workers=args.num_workers,
                next_norm_weight=next_norm,
                next_router_weight=next_router,
            )
            utility_record = {
                "layer": layer,
                "batches": utility["batches"],
                "active_rows": utility["active_rows"],
                "observed_experts": utility["observed_experts"],
                "unobserved_experts": utility["unobserved_experts"],
                "horizon_weights": utility["horizon_weights"],
                "utility_source": utility["utility_source"],
                "score": utility["score"].tolist(),
                "route_utility": utility["route_utility"].tolist(),
                "residual_utility": utility["residual_utility"].tolist(),
                "occurrences": utility["occurrences"].tolist(),
            }
            write_json_exclusive(
                args.output / f"layer_{layer:02d}_train_utility.json",
                utility_record,
            )
            del base_model, upgraded_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for fraction in upgrade_fractions:
                schedule = upgrade_schedule(
                    utility["score"],
                    fraction=fraction,
                    base_bits=args.mixed_base_bits,
                    upgrade_bits=args.mixed_upgrade_bits,
                )
                model = PackedRouteQuantExperts.from_target(
                    gate_up,
                    down,
                    gate_up_bits=schedule,
                    exact_k=8,
                    group_size=args.group_size,
                    scale_method="mse",
                    item_chunk=args.item_chunk,
                    scale_dtype=scale_dtype,
                ).to(device)
                model.enable_dequantized_cache(True, max_experts=256)
                metrics = evaluate_candidate(
                    model,
                    layer_datasets["tune"],
                    layer=layer,
                    device=device,
                    microbatch=args.microbatch,
                    workers=args.num_workers,
                    next_norm_weight=next_norm,
                    next_router_weight=next_router,
                )
                global_schedule = schedule.view(1, -1).expand(40, -1).contiguous()
                row = {
                    "layer": layer,
                    "variant": (
                        f"mixed_int{args.mixed_base_bits}_int{args.mixed_upgrade_bits}"
                    ),
                    "base_bits": args.mixed_base_bits,
                    "upgrade_bits": args.mixed_upgrade_bits,
                    "upgrade_fraction": fraction,
                    "upgrade_count": int((schedule == args.mixed_upgrade_bits).sum()),
                    "upgraded_expert_ids": (schedule == args.mixed_upgrade_bits).nonzero(
                        as_tuple=False
                    ).flatten().tolist(),
                    "layer_persistent_bytes": model.persistent_nbytes(),
                    "scale_storage": args.scale_storage,
                    "uniform_40_layer_projected_bytes": (
                        mixed_routequant_projected_bytes(
                            global_schedule,
                            group_size=args.group_size,
                            scale_bytes=scale_bytes,
                        )
                    ),
                    "uniform_40_layer_projected_gib": (
                        mixed_routequant_projected_bytes(
                            global_schedule,
                            group_size=args.group_size,
                            scale_bytes=scale_bytes,
                        ) / 2**30
                    ),
                    "metrics": metrics,
                    "exact_baseline": baselines["exact"],
                    "zero_baseline": baselines["zero"],
                    "utility_source": utility["utility_source"],
                    "peak_reserved_gib": (
                        torch.cuda.max_memory_reserved(device) / 2**30
                        if device.type == "cuda" else 0.0
                    ),
                }
                results.append(row)
                append_jsonl(args.output / "candidate_metrics.jsonl", row)
                print(json.dumps(row, sort_keys=True), flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        for gate_bits, down_bits, method in asymmetric_candidates:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = PackedRouteQuantExperts.from_target(
                gate_up,
                down,
                gate_up_bits=gate_bits,
                down_bits=down_bits,
                exact_k=8,
                group_size=args.group_size,
                scale_method=method,
                item_chunk=args.item_chunk,
                scale_dtype=scale_dtype,
            ).to(device)
            model.enable_dequantized_cache(True, max_experts=256)
            metrics = evaluate_candidate(
                model,
                layer_datasets["tune"],
                layer=layer,
                device=device,
                microbatch=args.microbatch,
                workers=args.num_workers,
                next_norm_weight=next_norm,
                next_router_weight=next_router,
            )
            gate_schedule = torch.full((40, 256), gate_bits, dtype=torch.int8)
            down_schedule = torch.full((40, 256), down_bits, dtype=torch.int8)
            projected = asymmetric_routequant_projected_bytes(
                gate_schedule,
                down_schedule,
                group_size=args.group_size,
                scale_bytes=scale_bytes,
            )
            row = {
                "layer": layer,
                "variant": "asymmetric_projection_bits",
                "gate_up_bits": gate_bits,
                "down_bits": down_bits,
                "scale_method": method,
                "layer_persistent_bytes": model.persistent_nbytes(),
                "scale_storage": args.scale_storage,
                "uniform_40_layer_projected_bytes": projected,
                "uniform_40_layer_projected_gib": projected / 2**30,
                "metrics": metrics,
                "exact_baseline": baselines["exact"],
                "zero_baseline": baselines["zero"],
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved(device) / 2**30
                    if device.type == "cuda" else 0.0
                ),
            }
            results.append(row)
            append_jsonl(args.output / "candidate_metrics.jsonl", row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for bits, method in candidates:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = PackedRouteQuantExperts.from_target(
                gate_up,
                down,
                gate_up_bits=bits,
                exact_k=8,
                group_size=args.group_size,
                scale_method=method,
                item_chunk=args.item_chunk,
                scale_dtype=scale_dtype,
            ).to(device)
            model.enable_dequantized_cache(True, max_experts=256)
            metrics = evaluate_candidate(
                model,
                layer_datasets["tune"],
                layer=layer,
                device=device,
                microbatch=args.microbatch,
                workers=args.num_workers,
                next_norm_weight=next_norm,
                next_router_weight=next_router,
            )
            row = {
                "layer": layer,
                "bits": bits,
                "scale_method": method,
                "layer_persistent_bytes": model.persistent_nbytes(),
                "scale_storage": args.scale_storage,
                "uniform_40_layer_projected_bytes": uniform_routequant_projected_bytes(
                    bits=bits,
                    group_size=args.group_size,
                    scale_bytes=scale_bytes,
                ),
                "uniform_40_layer_projected_gib": uniform_routequant_projected_bytes(
                    bits=bits,
                    group_size=args.group_size,
                    scale_bytes=scale_bytes,
                ) / 2**30,
                "metrics": metrics,
                "exact_baseline": baselines["exact"],
                "zero_baseline": baselines["zero"],
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved(device) / 2**30
                    if device.type == "cuda" else 0.0
                ),
            }
            results.append(row)
            append_jsonl(args.output / "candidate_metrics.jsonl", row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del gate_up, down, next_norm, next_router
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregates = []
    for bits, method in candidates:
        rows = [
            row for row in results
            if row.get("bits") == bits and row.get("scale_method") == method
        ]
        aggregates.append({
            "bits": bits,
            "scale_method": method,
            "mean_next_router_recall_at_8": sum(
                row["metrics"]["request_macro_next_router_recall_at_8"] for row in rows
            ) / len(rows),
            "mean_normalized_residual_rmse": sum(
                row["metrics"]["normalized_residual_rmse"] for row in rows
            ) / len(rows),
            "maximum_layer_recall_regression_from_exact": max(
                row["exact_baseline"]["request_macro_next_router_recall_at_8"]
                - row["metrics"]["request_macro_next_router_recall_at_8"]
                for row in rows
            ),
            "uniform_40_layer_projected_bytes": rows[0]["uniform_40_layer_projected_bytes"],
            "uniform_40_layer_projected_gib": rows[0]["uniform_40_layer_projected_gib"],
        })
    asymmetric_aggregates = []
    for gate_bits, down_bits, method in asymmetric_candidates:
        rows = [
            row for row in results
            if row.get("variant") == "asymmetric_projection_bits"
            and row.get("gate_up_bits") == gate_bits
            and row.get("down_bits") == down_bits
            and row.get("scale_method") == method
        ]
        asymmetric_aggregates.append({
            "variant": "asymmetric_projection_bits",
            "gate_up_bits": gate_bits,
            "down_bits": down_bits,
            "scale_method": method,
            "mean_next_router_recall_at_8": sum(
                row["metrics"]["request_macro_next_router_recall_at_8"] for row in rows
            ) / len(rows),
            "mean_normalized_residual_rmse": sum(
                row["metrics"]["normalized_residual_rmse"] for row in rows
            ) / len(rows),
            "maximum_layer_recall_regression_from_exact": max(
                row["exact_baseline"]["request_macro_next_router_recall_at_8"]
                - row["metrics"]["request_macro_next_router_recall_at_8"]
                for row in rows
            ),
            "uniform_40_layer_projected_bytes": rows[0]["uniform_40_layer_projected_bytes"],
            "uniform_40_layer_projected_gib": rows[0]["uniform_40_layer_projected_gib"],
        })
    mixed_aggregates = []
    for fraction in upgrade_fractions:
        rows = [
            row for row in results
            if row.get("variant") == (
                f"mixed_int{args.mixed_base_bits}_int{args.mixed_upgrade_bits}"
            )
            and row["upgrade_fraction"] == fraction
        ]
        mixed_aggregates.append({
            "variant": (
                f"mixed_int{args.mixed_base_bits}_int{args.mixed_upgrade_bits}"
            ),
            "base_bits": args.mixed_base_bits,
            "upgrade_bits": args.mixed_upgrade_bits,
            "upgrade_fraction": fraction,
            "mean_next_router_recall_at_8": sum(
                row["metrics"]["request_macro_next_router_recall_at_8"] for row in rows
            ) / len(rows),
            "mean_normalized_residual_rmse": sum(
                row["metrics"]["normalized_residual_rmse"] for row in rows
            ) / len(rows),
            "maximum_layer_recall_regression_from_exact": max(
                row["exact_baseline"]["request_macro_next_router_recall_at_8"]
                - row["metrics"]["request_macro_next_router_recall_at_8"]
                for row in rows
            ),
            "uniform_40_layer_projected_bytes": rows[0]["uniform_40_layer_projected_bytes"],
            "uniform_40_layer_projected_gib": rows[0]["uniform_40_layer_projected_gib"],
        })
    result = {
        "schema": RESULT_SCHEMA,
        "layers": list(layers),
        "aggregates": aggregates,
        "asymmetric_aggregates": asymmetric_aggregates,
        "mixed_aggregates": mixed_aggregates,
        "candidate_rows": len(results),
        "optimizer_constructed": False,
        "training_started": False,
        "development_opened_for_metrics": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    paths = sorted(
        path for path in args.output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
