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
    parser.add_argument("--group-size", type=int, choices=(32, 64), default=64)
    parser.add_argument("--item-chunk", type=int, default=2)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8, 16, 32), default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
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


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    candidates = parse_candidates(args.candidates)
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
        "group_size": args.group_size,
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
                "uniform_40_layer_projected_bytes": uniform_routequant_projected_bytes(
                    bits=bits, group_size=args.group_size
                ),
                "uniform_40_layer_projected_gib": uniform_routequant_projected_bytes(
                    bits=bits, group_size=args.group_size
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
            if row["bits"] == bits and row["scale_method"] == method
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
    result = {
        "schema": RESULT_SCHEMA,
        "layers": list(layers),
        "aggregates": aggregates,
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
