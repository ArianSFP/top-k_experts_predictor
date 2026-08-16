#!/usr/bin/env python3
"""Fit one deterministic Resident Functional Codebook layer."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.exact_k import stable_topk  # noqa: E402
from harp_rtt.resident_codebook import (  # noqa: E402
    fit_resident_proxy,
    validate_codebook_tables,
)
from harp_rtt.shadow_checkpoint import (  # noqa: E402
    IndexedCheckpoint,
    load_target_layer_experts,
    sha256_file,
)
from harp_rtt.shadow_expert import (  # noqa: E402
    PackedInt4ResidentExperts,
    target_selected_expert_outputs,
)
from runpod.train_harp_delta_v3 import (  # noqa: E402
    _reuse_split_manifest as validate_reuse_split,
)
from runpod.train_shadow_experts_local import (  # noqa: E402
    batch_tensors,
    build_scaled_training_dataset,
    load_split,
    next_router_agreement_loss,
)
from runpod.train_harp_delta_v3 import _manifest as validate_partition  # noqa: E402


SCHEMA = "harp_resident_functional_codebook_fit_v1"
LOCAL_SCHEMA = "harp_shadowroute_local_expert_layer_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "tune", "development"):
        parser.add_argument(f"--{split}-index", type=Path, required=True)
        parser.add_argument(f"--{split}-corpus", type=Path, required=True)
        parser.add_argument(f"--{split}-companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--scale-train-index", type=Path, required=True)
    parser.add_argument("--scale-train-corpus", type=Path, required=True)
    parser.add_argument("--outer-split-manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=range(40), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-fingerprint-rows", type=int, default=64)
    parser.add_argument("--maximum-expert-rows", type=int, default=128)
    parser.add_argument("--maximum-rows-per-request", type=int, default=4)
    parser.add_argument("--shortlist", type=int, default=16)
    parser.add_argument("--tune-batch-size", type=int, default=8)
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_checksums(output: Path) -> None:
    paths = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_parent(
    path: Path,
    *,
    layer: int,
    target_sha: str,
    device: torch.device,
) -> tuple[PackedInt4ResidentExperts, dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(value, dict)
        or value.get("schema") != LOCAL_SCHEMA
        or value.get("mode") != "resident_int4_only"
        or int(value.get("layer", -1)) != layer
        or value.get("target_checkpoint_index_sha256") != target_sha
        or value.get("formal_validation_opened") is not False
        or value.get("calibration_opened") is not False
        or value.get("sealed_test_opened") is not False
    ):
        raise ValueError("resident codebook parent checkpoint contract changed")
    ids = value.get("resident_expert_ids")
    state = value.get("model_state_dict")
    if not isinstance(ids, Tensor) or not isinstance(state, dict):
        raise ValueError("resident codebook parent lacks namespace/state")
    module = PackedInt4ResidentExperts(
        ids.to(device), None, device=device
    ).to(dtype=torch.bfloat16)
    module.load_state_dict(state, strict=True)
    module.eval()
    return module, value


def item_rows(item: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor, str]:
    targets = item["targets"]
    inputs = targets["future_router_inputs"]
    ids = targets["future_selected_ids"].long()
    weights = targets["future_execution_weights"].float()
    available = targets["future_available"].bool()
    if inputs.ndim != 2 or ids.ndim != 2 or weights.shape != ids.shape:
        raise ValueError("resident codebook item geometry changed")
    request = str(item["metadata"]["request_id"])
    return inputs[available], ids[available], weights[available], request


def collect_fit_rows(
    dataset: Dataset[Any],
    resident_ids: Tensor,
    *,
    shared_rows: int,
    maximum_expert_rows: int,
    maximum_rows_per_request: int,
) -> tuple[Tensor, dict[int, tuple[Tensor, Tensor]], dict[str, Any]]:
    residents = set(int(value) for value in resident_ids.tolist())
    shared: list[Tensor] = []
    shared_requests: set[str] = set()
    expert_inputs: dict[int, list[Tensor]] = defaultdict(list)
    expert_weights: dict[int, list[float]] = defaultdict(list)
    request_counts: dict[tuple[int, str], int] = defaultdict(int)
    scanned = 0
    for index in range(len(dataset)):
        inputs, ids, weights, request = item_rows(dataset[index])
        scanned += int(inputs.shape[0])
        if request not in shared_requests and len(shared) < shared_rows:
            shared.append(inputs[0].to(torch.bfloat16))
            shared_requests.add(request)
        for row in range(inputs.shape[0]):
            for slot in range(ids.shape[1]):
                expert = int(ids[row, slot])
                if expert in residents:
                    continue
                key = (expert, request)
                if (
                    len(expert_inputs[expert]) >= maximum_expert_rows
                    or request_counts[key] >= maximum_rows_per_request
                ):
                    continue
                expert_inputs[expert].append(inputs[row].to(torch.bfloat16))
                expert_weights[expert].append(float(weights[row, slot]))
                request_counts[key] += 1
    if not shared:
        raise ValueError("resident codebook fitting found no shared activations")
    rows = {
        expert: (
            torch.stack(values),
            torch.tensor(expert_weights[expert], dtype=torch.float32),
        )
        for expert, values in expert_inputs.items()
    }
    return torch.stack(shared), rows, {
        "scanned_rows": scanned,
        "shared_rows": len(shared),
        "missing_experts_observed": len(rows),
        "expert_rows": {str(key): len(value[0]) for key, value in rows.items()},
    }


@torch.no_grad()
def execute_residents(
    module: PackedInt4ResidentExperts,
    hidden: Tensor,
    global_ids: Tensor,
) -> Tensor:
    values = []
    for expert in global_ids.tolist():
        local = int(module.expert_to_resident[int(expert)])
        if local < 0:
            raise ValueError("functional candidate is not resident")
        gate_up, down = module._resident_weights(local, dtype=hidden.dtype)
        gate, up = F.linear(hidden, gate_up).chunk(2, dim=-1)
        values.append(F.linear(F.silu(gate) * up, down))
    return torch.stack(values, dim=1)


def composite_values(values: Tensor, router: Tensor | None) -> Tensor:
    base = values.float()
    base = base / base.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    if router is None:
        return base
    projected = F.linear(values.float(), router.float())
    projected = projected / projected.square().mean(
        -1, keepdim=True
    ).clamp_min(1e-8).sqrt()
    return torch.cat((base, projected), dim=-1)


@torch.no_grad()
def fit_tables(
    module: PackedInt4ResidentExperts,
    gate_up: Tensor,
    down: Tensor,
    shared: Tensor,
    expert_rows: dict[int, tuple[Tensor, Tensor]],
    *,
    router: Tensor | None,
    shortlist: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    resident_ids = module.resident_ids.long()
    shared_device = shared.to(device)
    all_ids = torch.arange(256, device=device).expand(shared.shape[0], -1)
    target_fingerprint = target_selected_expert_outputs(
        shared_device, all_ids, gate_up, down
    )
    resident_fingerprint = execute_residents(module, shared_device, resident_ids)
    top1_ids = torch.full((256, 2), -1, dtype=torch.int16)
    top1_coefficients = torch.zeros(256, 2, dtype=torch.bfloat16)
    top1_count = torch.ones(256, dtype=torch.uint8)
    top2_ids = top1_ids.clone()
    top2_coefficients = top1_coefficients.clone()
    top2_count = top1_count.clone()
    diagnostics: dict[str, Any] = {}
    resident_set = set(int(value) for value in resident_ids.tolist())
    for expert in range(256):
        if expert in resident_set:
            top1_ids[expert, 0] = expert
            top1_coefficients[expert, 0] = 1
            top2_ids[expert, 0] = expert
            top2_coefficients[expert, 0] = 1
            diagnostics[str(expert)] = {
                "resident": True, "rows": 0, "top1_error": 0.0, "top2_error": 0.0
            }
            continue
        target_shared = composite_values(
            target_fingerprint[:, expert], router
        )
        resident_shared = composite_values(resident_fingerprint, router)
        fingerprint_error = (
            target_shared[:, None] - resident_shared
        ).square().mean(dim=(0, 2))
        order = sorted(
            range(resident_ids.numel()),
            key=lambda index: (
                float(fingerprint_error[index]),
                int(resident_ids[index]),
            ),
        )[: min(shortlist, resident_ids.numel())]
        candidate_ids = resident_ids[order]
        if expert in expert_rows:
            hidden, weights = expert_rows[expert]
            hidden = hidden.to(device)
            target = target_selected_expert_outputs(
                hidden,
                torch.full(
                    (hidden.shape[0], 1), expert, dtype=torch.long, device=device
                ),
                gate_up,
                down,
            )[:, 0]
            candidates = execute_residents(module, hidden, candidate_ids)
        else:
            weights = torch.ones(shared.shape[0])
            target = target_fingerprint[:, expert]
            candidates = resident_fingerprint[:, order]
        target_fit = composite_values(target, router)
        candidates_fit = composite_values(candidates, router)
        importance = weights.to(device).square().clamp_min(1e-8)
        one = fit_resident_proxy(
            target_fit, candidates_fit, importance, candidate_ids, proxies=1
        )
        two = fit_resident_proxy(
            target_fit, candidates_fit, importance, candidate_ids, proxies=2
        )
        top1_ids[expert] = one.proxy_ids
        top1_coefficients[expert] = one.coefficients
        top2_ids[expert] = two.proxy_ids
        top2_coefficients[expert] = two.coefficients
        top2_count[expert] = two.proxy_count
        diagnostics[str(expert)] = {
            "resident": False,
            "rows": int(weights.numel()),
            "shortlist": [int(value) for value in candidate_ids.tolist()],
            "top1_error": one.weighted_error,
            "top2_error": two.weighted_error,
            "relative_pair_gain": (
                0.0
                if one.weighted_error <= 0
                else (one.weighted_error - two.weighted_error)
                / one.weighted_error
            ),
        }
    tables = []
    for ids, coefficients, count in (
        (top1_ids, top1_coefficients, top1_count),
        (top2_ids, top2_coefficients, top2_count),
    ):
        validate_codebook_tables(
            resident_ids.cpu(), ids, coefficients, count
        )
        tables.append({
            "proxy_ids": ids,
            "proxy_coefficients": coefficients,
            "proxy_count": count,
        })
    return tables[0], tables[1], diagnostics


def module_with_codebook(
    parent: PackedInt4ResidentExperts,
    table: Mapping[str, Tensor],
    device: torch.device,
) -> PackedInt4ResidentExperts:
    module = PackedInt4ResidentExperts(
        parent.resident_ids,
        None,
        device=device,
        codebook_proxy_ids=table["proxy_ids"],
        codebook_proxy_coefficients=table["proxy_coefficients"],
        codebook_proxy_count=table["proxy_count"],
    ).to(dtype=torch.bfloat16)
    parent_state = parent.state_dict()
    missing, unexpected = module.load_state_dict(parent_state, strict=False)
    expected_missing = {
        "codebook_proxy_ids", "codebook_proxy_coefficients", "codebook_proxy_count"
    }
    if set(missing) != expected_missing or unexpected:
        raise ValueError("resident codebook parent import changed")
    module.codebook_proxy_ids.copy_(table["proxy_ids"].to(device))
    module.codebook_proxy_coefficients.copy_(
        table["proxy_coefficients"].to(device)
    )
    module.codebook_proxy_count.copy_(table["proxy_count"].to(device))
    module.eval()
    return module


@torch.no_grad()
def evaluate(
    variants: Mapping[str, PackedInt4ResidentExperts | None],
    dataset: Dataset[Any],
    *,
    layer: int,
    device: torch.device,
    next_norm: Tensor | None,
    next_router: Tensor | None,
    batch_size: int,
) -> dict[str, Any]:
    totals = {
        name: {"hits": 0.0, "slots": 0, "sse": 0.0, "target": 0.0, "rows": 0}
        for name in variants
    }
    request_hits: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in variants
    }
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for host in loader:
        inputs, routed, ids, weights, valid, requests, states = batch_tensors(
            host, layer=layer, device=device
        )
        predictions = {
            "resident_only": variants["resident_only"](inputs, ids, weights),
            "nearest_one": variants["nearest_one"](inputs, ids, weights),
            "nearest_two": variants["nearest_two"](inputs, ids, weights),
            "exact_tail": routed,
        }
        for name, predicted in predictions.items():
            active = valid.bool()
            difference = predicted.float() - routed.float()
            totals[name]["sse"] += float(
                (difference.square().mean(-1) * active).sum()
            )
            totals[name]["target"] += float(
                (routed.float().square().mean(-1) * active).sum()
            )
            totals[name]["rows"] += int(active.sum())
            if next_router is None or next_norm is None:
                continue
            _loss, _parts, logits = next_router_agreement_loss(
                predicted,
                states,
                valid,
                norm_weight=next_norm,
                router_weight=next_router,
            )
            predicted_ids = stable_topk(logits, 8)
            teacher_ids = states["next_selected_expert_ids"].long()
            hits = (
                predicted_ids[..., None] == teacher_ids[..., None, :]
            ).any(-1).float().sum(-1)
            totals[name]["hits"] += float((hits * active).sum())
            totals[name]["slots"] += int(active.sum()) * 8
            for row, request in enumerate(requests):
                selected = active[row]
                if bool(selected.any()):
                    request_hits[name][request].append(
                        float(hits[row][selected].mean() / 8.0)
                    )
    result = {}
    for name, values in totals.items():
        request_macro = None
        if request_hits[name]:
            request_macro = sum(
                sum(rows) / len(rows) for rows in request_hits[name].values()
            ) / len(request_hits[name])
        result[name] = {
            "next_router_recall": (
                None if values["slots"] == 0 else values["hits"] / values["slots"]
            ),
            "request_macro_next_router_recall": request_macro,
            "normalized_residual_mse": values["sse"] / max(values["target"], 1e-12),
            "rows": values["rows"],
            "requests": len(request_hits[name]),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite codebook fit {args.output}")
    if (
        len(args.source_commit) != 40
        or any(character not in "0123456789abcdef" for character in args.source_commit)
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    args.data_profile = "b2_reuse_4096"
    partition = validate_partition(args.partition_manifest, args.data_profile)
    reuse = validate_reuse_split(args.reuse_split_manifest)
    selected = {
        "tune": set(reuse["inner_split"]["tuning_requests"]),
        "development": None,
    }
    tune, tune_groups = load_split(
        args, "tune", selected_requests=selected["tune"]
    )
    development, development_groups = load_split(
        args, "development", selected_requests=None
    )
    train, scale_provenance = build_scaled_training_dataset(
        args, partition=partition, tune=tune, development=development
    )
    if train.requests & tune_groups or train.requests & development_groups:
        raise PermissionError("resident codebook fitting overlaps a holdout group")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    checkpoint = IndexedCheckpoint(args.target_model)
    parent, parent_value = load_parent(
        args.parent_checkpoint,
        layer=args.layer,
        target_sha=checkpoint.index_sha256,
        device=device,
    )
    gate_up, down = load_target_layer_experts(
        checkpoint, args.layer, device=device, dtype=torch.bfloat16
    )
    router = None
    next_norm = None
    next_router = None
    if args.layer < 39:
        prefix = f"model.language_model.layers.{args.layer + 1}."
        tensors = checkpoint.tensors((
            prefix + "post_attention_layernorm.weight",
            prefix + "mlp.gate.weight",
        ))
        next_norm = tensors[
            prefix + "post_attention_layernorm.weight"
        ].to(device=device, dtype=torch.bfloat16)
        next_router = tensors[prefix + "mlp.gate.weight"].to(
            device=device, dtype=torch.bfloat16
        )
        router = next_router - next_router.mean(0, keepdim=True)
    shared, expert_rows, fit_provenance = collect_fit_rows(
        train,
        parent.resident_ids.cpu(),
        shared_rows=args.shared_fingerprint_rows,
        maximum_expert_rows=args.maximum_expert_rows,
        maximum_rows_per_request=args.maximum_rows_per_request,
    )
    top1, top2, diagnostics = fit_tables(
        parent,
        gate_up,
        down,
        shared,
        expert_rows,
        router=router,
        shortlist=args.shortlist,
        device=device,
    )
    one = module_with_codebook(parent, top1, device)
    two = module_with_codebook(parent, top2, device)
    metrics = evaluate(
        {
            "resident_only": parent,
            "nearest_one": one,
            "nearest_two": two,
            "exact_tail": None,
        },
        tune,
        layer=args.layer,
        device=device,
        next_norm=next_norm,
        next_router=next_router,
        batch_size=args.tune_batch_size,
    )
    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "layer": args.layer,
        "seed": args.seed,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "outer_split_manifest_sha256": sha256_file(args.outer_split_manifest),
        "scale_provenance": scale_provenance,
        "fit_provenance": fit_provenance,
        "shortlist": args.shortlist,
        "selection_uses_train_only": True,
        "development_opened_for_metrics": False,
        "optimizer_constructed": False,
        "training_started": False,
        "target_expert_loads_permitted": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)
    payload = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "layer": args.layer,
        "resident_expert_ids": parent.resident_ids.cpu(),
        "resident_count": parent.resident_count,
        "parent_model_state_dict": {
            name: value.detach().cpu()
            for name, value in parent.state_dict().items()
        },
        "nearest_one": top1,
        "nearest_two": top2,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    candidate_path = args.output / f"resident_codebook_fit_layer_{args.layer:02d}.pt"
    with candidate_path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    write_json_exclusive(
        args.output / "FIT_DIAGNOSTICS.json",
        {"experts": diagnostics},
    )
    result = {
        "schema": SCHEMA,
        "layer": args.layer,
        "metrics": metrics,
        "candidate_sha256": sha256_file(candidate_path),
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "optimizer_constructed": False,
        "training_started": False,
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
