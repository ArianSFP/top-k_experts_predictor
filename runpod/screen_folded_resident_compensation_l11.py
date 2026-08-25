#!/usr/bin/env python3
"""Screen zero-runtime-cost folded omission compensation at routed layer 11.

The serving artifact remains a production ``PackedInt4ResidentExperts`` shard.
Train-only maps are folded into every resident expert's down projection before
group-64 amax and MSE INT4 quantizers; no map is serialized or executed at serving.
This is a request-disjoint local architecture gate, not formal validation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.exact_k import stable_topk  # noqa: E402
from harp_rtt.resident_damage import (  # noqa: E402
    _canonical_ids_sha256,
    frequency_core_ids,
    validate_core_inclusion,
)
from harp_rtt.shadow_checkpoint import (  # noqa: E402
    IndexedCheckpoint,
    load_target_layer_experts,
    sha256_file,
)
from harp_rtt.shadow_expert import (  # noqa: E402
    PackedInt4ResidentExperts,
    quantize_groupwise_int4,
)
from runpod.train_harp_delta_v3 import (  # noqa: E402
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
    loader,
)
from runpod.train_shadow_experts_local import (  # noqa: E402
    batch_tensors,
    load_split,
    write_checksums,
    write_json_exclusive,
    write_rows,
)


SCHEMA = "harp_folded_resident_compensation_l11_v1"
LAYER = 11
GROUP_SIZE = 64
TOTAL_CELLS = 3_850
CELL_BYTES = 1_671_168
TOTAL_PACKED_BYTES = 6_433_996_800
EXPECTED_PLAN_SHA256 = "8b25bdb5a39f08089f8b175ae5f8befa0fe2c3995d5d16bffcc28d2d8cd96536"
EXPECTED_MEMBERSHIP_SHA256 = "89d6b5f49f00f4e329d049434acedf183d8516572312bdfdcd794ea29692392a"
EXPECTED_CORE_SHA256 = "c090aef6da46f85e9004f8fee53d7df26fb4e9d408ce2ae042243a546daba1d8"
EXPECTED_PARTITION_SHA256 = "6b0d3fb37cecd8e87f37a8e0850842e92f2c32c971ded0447420d1be809e6e43"
EXPECTED_REUSE_SHA256 = "ac4b08c92ae4edf0e5b0fa2607b20896b944f90c257009d29b7f9cb8cd1af777"
EXPECTED_TARGET_SHA256 = "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--resident-plan", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ridge-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-recall-lift", type=float, default=0.015)
    parser.add_argument("--maximum-horizon-regression", type=float, default=0.002)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _load_and_validate_plan(path: Path) -> tuple[dict[str, Any], list[list[int]], dict[str, Any]]:
    if sha256_file(path) != EXPECTED_PLAN_SHA256:
        raise ValueError("resident allocation file SHA changed")
    plan = json.loads(path.read_text(encoding="utf-8"))
    raw = plan.get("resident_expert_ids_by_layer")
    if not isinstance(raw, list) or len(raw) != 40:
        raise ValueError("resident allocation must contain exactly 40 layers")
    resident = [[int(value) for value in row] for row in raw]
    if sum(map(len, resident)) != TOTAL_CELLS or int(plan.get("total_residents", -1)) != TOTAL_CELLS:
        raise ValueError("resident allocation is not exactly 3,850 cells")
    if int(plan.get("resident_cell_bytes", -1)) != CELL_BYTES:
        raise ValueError("resident cell byte geometry changed")
    if int(plan.get("packed_int4_bytes", -1)) != TOTAL_PACKED_BYTES:
        raise ValueError("packed resident payload changed")
    if int(plan.get("fallback_bytes", -1)) != 0:
        raise ValueError("compliant allocation must not carry fallback bytes")
    if _canonical_ids_sha256(resident) != EXPECTED_MEMBERSHIP_SHA256:
        raise ValueError("resident membership hash changed")
    counts = torch.as_tensor(plan.get("expert_counts"), dtype=torch.int64)
    if counts.shape != (40, 256):
        raise ValueError("train frequency table geometry changed")
    core, core_hash = frequency_core_ids(counts, core_size=64)
    validate_core_inclusion(resident, core)
    if core_hash != EXPECTED_CORE_SHA256 or plan.get("frequency_core_sha256") != core_hash:
        raise ValueError("mandatory frequency-top64 core hash changed")
    hits = sum(
        int(counts[layer, expert])
        for layer, layer_ids in enumerate(resident)
        for expert in layer_ids
    )
    cap = int(plan.get("resident_hit_count_cap", -1))
    if hits != int(plan.get("resident_hit_count", -1)) or hits > cap or cap != 3_245_387:
        raise ValueError("resident train-hit contract changed")
    if len(resident[LAYER]) != 98:
        raise ValueError("compliant layer 11 must contain exactly 98 resident cells")
    contract = {
        "resident_plan_sha256": EXPECTED_PLAN_SHA256,
        "resident_membership_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "frequency_core_sha256": core_hash,
        "frequency_core_inclusion_verified": True,
        "frequency_core_size_per_layer": 64,
        "layer_resident_cells": len(resident[LAYER]),
        "total_resident_cells": TOTAL_CELLS,
        "resident_cell_bytes": CELL_BYTES,
        "packed_int4_bytes": TOTAL_PACKED_BYTES,
        "fallback_bytes": 0,
        "train_resident_hits": hits,
        "resident_hit_count_cap": cap,
    }
    return plan, resident, contract


def _router_logits(
    predicted_routed: Tensor,
    states: Mapping[str, Tensor],
    *,
    norm_weight: Tensor,
    router_weight: Tensor,
) -> Tensor:
    current_u = states["post_attention_residual_u"]
    current_xplus = states["post_moe_residual_xplus"]
    shared = states["shared_expert_output_delta_s"]
    next_u = states["next_post_attention_residual_u"]
    predicted_xplus = current_u + shared + predicted_routed
    predicted_next_u = predicted_xplus + (next_u - current_xplus)
    values = predicted_next_u.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    normalized = normalized * (1.0 + norm_weight.float())
    return F.linear(normalized, router_weight.float())


def _overlap(logits: Tensor, teacher_ids: Tensor) -> Tensor:
    predicted = stable_topk(logits, k=8)
    return (teacher_ids[..., None] == predicted[..., None, :]).any(-1).float().mean(-1)


def _request_macro(
    values: dict[tuple[str, int], list[float]],
) -> tuple[float, dict[str, float]]:
    per_request_horizon = {
        key: sum(rows) / len(rows) for key, rows in values.items() if rows
    }
    if not per_request_horizon:
        raise ValueError("evaluation contains no valid endpoints")
    by_horizon: dict[int, list[float]] = defaultdict(list)
    for (_request, horizon), value in per_request_horizon.items():
        by_horizon[horizon].append(value)
    horizons = {
        str(horizon): sum(rows) / len(rows)
        for horizon, rows in sorted(by_horizon.items())
    }
    return sum(per_request_horizon.values()) / len(per_request_horizon), horizons


@torch.no_grad()
def _bf16_resident_output(
    hidden_states: Tensor,
    selected_ids: Tensor,
    selected_weights: Tensor,
    membership: Tensor,
    gate_up: Tensor,
    down: Tensor,
) -> Tensor:
    leading = hidden_states.shape[:-1]
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    ids = selected_ids.reshape(hidden.shape[0], -1).long()
    weights = selected_weights.reshape_as(ids).to(hidden.dtype)
    resident = membership[ids]
    output = torch.zeros_like(hidden)
    for expert in torch.unique(ids[resident]).tolist():
        positions = (ids == int(expert)).nonzero(as_tuple=False)
        token_index, slot_index = positions[:, 0], positions[:, 1]
        gate, up = F.linear(hidden[token_index], gate_up[int(expert)]).chunk(2, -1)
        exact = F.linear(F.silu(gate) * up, down[int(expert)])
        output[token_index] += exact * weights[token_index, slot_index, None]
    return output.reshape(*leading, hidden.shape[-1])


@torch.no_grad()
def _fit_maps(
    dataset: Any,
    *,
    device: torch.device,
    microbatch: int,
    num_workers: int,
    resident_ids: Tensor,
    gate_up: Tensor,
    down: Tensor,
    ridge_fraction: float,
) -> tuple[dict[str, Tensor | float], dict[str, Any]]:
    membership = torch.zeros(256, dtype=torch.bool, device=device)
    membership[resident_ids] = True
    cached_x: list[Tensor] = []
    cached_y: list[Tensor] = []
    calls = 0
    endpoints = 0
    for batch_index, host in enumerate(loader(
        dataset,
        batch=microbatch,
        shuffle=False,
        seed=0,
        workers=num_workers,
        device=device,
    )):
        inputs, routed, ids, weights, valid, _requests, _states = batch_tensors(
            host, layer=LAYER, device=device
        )
        resident = _bf16_resident_output(inputs, ids, weights, membership, gate_up, down)
        cached_x.append(resident[valid].to(device="cpu", dtype=torch.bfloat16))
        cached_y.append(routed[valid].to(device="cpu", dtype=torch.bfloat16))
        calls += int((membership[ids] & valid[..., None]).sum())
        endpoints += int(valid.sum())
        if batch_index == 0:
            print(json.dumps({"event": "first_train_batch", "valid_endpoints": endpoints, "resident_calls": calls}), flush=True)
    x = torch.cat(cached_x).to(device=device, dtype=torch.float32)
    y = torch.cat(cached_y).to(device=device, dtype=torch.float32)
    del cached_x, cached_y
    scalar_den = x.square().sum()
    raw_gamma = (x * y).sum() / scalar_den.clamp_min(1e-12)
    gamma = raw_gamma.clamp(0.0, 4.0)
    diag_den = x.square().sum(0)
    diag_num = (x * y).sum(0)
    diag_lambda = ridge_fraction * diag_den.mean()
    diagonal = (diag_num + diag_lambda) / (diag_den + diag_lambda).clamp_min(1e-12)
    diagonal = diagonal.clamp(0.0, 4.0)

    sample_count = x.shape[0]
    cxx = (x.T @ x) / sample_count
    cxy = (x.T @ y) / sample_count
    ridge_lambda = ridge_fraction * torch.diagonal(cxx).mean()
    identity = torch.eye(x.shape[1], device=device, dtype=torch.float32)
    # X @ B approximates Y; the folded column-vector map is A = B.T.
    coefficient = torch.linalg.solve(cxx + ridge_lambda * identity, cxy)
    full = coefficient.T.contiguous()

    predictions = {
        "gamma": x * gamma,
        "diagonal": x * diagonal,
        "ridge_full": F.linear(x, full),
    }
    train_mse = {
        name: float((value - y).square().mean())
        for name, value in predictions.items()
    }
    baseline_mse = float((x - y).square().mean())
    result = {
        "train_requests": 224,
        "train_rows": len(dataset),
        "train_valid_endpoints": endpoints,
        "train_resident_expert_calls": calls,
        "raw_gamma": float(raw_gamma),
        "gamma": float(gamma),
        "diagonal_min": float(diagonal.min()),
        "diagonal_max": float(diagonal.max()),
        "diagonal_mean": float(diagonal.mean()),
        "diagonal_ridge_lambda": float(diag_lambda),
        "ridge_fraction": ridge_fraction,
        "ridge_lambda": float(ridge_lambda),
        "train_endpoint_mean_routed_mse": {
            "bf16_baseline": baseline_mse,
            **{f"bf16_{key}": value for key, value in train_mse.items()},
        },
    }
    del x, y, cxx, cxy, coefficient, identity, predictions
    torch.cuda.empty_cache()
    return {"gamma": float(gamma), "diagonal": diagonal, "ridge_full": full}, result


def _quantize_int4(
    weight: Tensor, *, scale_method: str
) -> tuple[Tensor, Tensor]:
    """Production-layout INT4 with amax or RouteQuant's validated MSE scales."""

    if scale_method == "amax":
        return quantize_groupwise_int4(weight, group_size=GROUP_SIZE)
    if scale_method != "mse":
        raise ValueError("scale method must be amax or mse")
    grouped = weight.float().reshape(*weight.shape[:-1], -1, GROUP_SIZE)
    scales = grouped.abs().amax(-1).div(7.0).clamp_min(1e-8)
    quantized = torch.zeros_like(grouped)
    # Same four alternating least-squares refinements as RouteQuant v1.
    for step in range(5):
        quantized = torch.round(grouped / scales[..., None]).clamp(-7, 7)
        if step < 4:
            numerator = (grouped * quantized).sum(-1)
            denominator = quantized.square().sum(-1).clamp_min(1e-12)
            scales = (numerator / denominator).clamp_min(1e-8)
    # Retain PackedInt4ResidentExperts' signed-code layout (+8, not +7).
    unsigned = (quantized.to(torch.int8) + 8).to(torch.uint8).reshape(
        *weight.shape[:-1], -1
    )
    packed = unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)
    return packed.contiguous(), scales.to(torch.bfloat16).contiguous()


@torch.no_grad()
def _build_folded_model(
    baseline: PackedInt4ResidentExperts,
    target_gate_up: Tensor,
    target_down: Tensor,
    transform_name: str,
    transform: Tensor | float,
    *,
    scale_method: str,
) -> PackedInt4ResidentExperts:
    device = target_down.device
    model = PackedInt4ResidentExperts(
        baseline.resident_ids,
        None,
        hidden_width=baseline.hidden_width,
        intermediate_width=baseline.intermediate_width,
        experts=baseline.experts,
        exact_k=baseline.exact_k,
        group_size=baseline.group_size,
        device=device,
    )
    block = 8
    selected_gate_up = target_gate_up.index_select(0, baseline.resident_ids)
    selected_down = target_down.index_select(0, baseline.resident_ids)
    for start in range(0, baseline.resident_count, block):
        stop = min(start + block, baseline.resident_count)
        gate_packed, gate_scales = _quantize_int4(
            selected_gate_up[start:stop], scale_method=scale_method
        )
        model.gate_up_packed[start:stop].copy_(gate_packed)
        model.gate_up_scales[start:stop].copy_(gate_scales)
        weights = selected_down[start:stop].float()
        if transform_name == "gamma":
            folded = weights * float(transform)
        elif transform_name == "diagonal":
            assert isinstance(transform, Tensor)
            folded = weights * transform.float()[None, :, None]
        elif transform_name == "ridge_full":
            assert isinstance(transform, Tensor)
            folded = torch.matmul(transform.float()[None], weights)
        else:
            raise ValueError(f"unknown folded transform {transform_name}")
        packed, scales = _quantize_int4(
            folded.to(torch.bfloat16), scale_method=scale_method
        )
        model.down_packed[start:stop].copy_(packed)
        model.down_scales[start:stop].copy_(scales)
    model.requires_grad_(False)
    model.eval()
    return model


def _apply_map(values: Tensor, name: str, transform: Tensor | float) -> Tensor:
    if name == "gamma":
        return values.float() * float(transform)
    if name == "diagonal":
        assert isinstance(transform, Tensor)
        return values.float() * transform.float()
    if name == "ridge_full":
        assert isinstance(transform, Tensor)
        return F.linear(values.float(), transform.float())
    raise ValueError(name)


@torch.no_grad()
def _evaluate(
    dataset: Any,
    *,
    device: torch.device,
    microbatch: int,
    num_workers: int,
    resident_ids: Tensor,
    gate_up: Tensor,
    down: Tensor,
    next_norm: Tensor,
    next_router: Tensor,
    quantized_models: dict[str, PackedInt4ResidentExperts],
    transforms: dict[str, Tensor | float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    membership = torch.zeros(256, dtype=torch.bool, device=device)
    membership[resident_ids] = True
    methods = ["teacher", "bf16_baseline"]
    methods += [f"bf16_{name}" for name in transforms]
    methods += [f"int4_{name}" for name in quantized_models]
    recalls: dict[str, dict[tuple[str, int], list[float]]] = {
        name: defaultdict(list) for name in methods
    }
    routed_mse: dict[str, list[float]] = {
        name: [] for name in methods if name != "teacher"
    }
    logit_mse: dict[str, list[float]] = {
        name: [] for name in methods if name != "teacher"
    }
    calls: dict[str, int] = {
        f"int4_{name}": 0 for name in quantized_models
    }
    valid_endpoints = 0
    rows_out: list[dict[str, Any]] = []
    for batch_index, host in enumerate(loader(
        dataset,
        batch=microbatch,
        shuffle=False,
        seed=0,
        workers=num_workers,
        device=device,
    )):
        inputs, routed, ids, weights, valid, requests, states = batch_tensors(
            host, layer=LAYER, device=device
        )
        bf16_base = _bf16_resident_output(
            inputs, ids, weights, membership, gate_up, down
        )
        predicted: dict[str, Tensor] = {
            "teacher": routed,
            "bf16_baseline": bf16_base,
        }
        for name, transform in transforms.items():
            predicted[f"bf16_{name}"] = _apply_map(
                bf16_base, name, transform
            )
        for name, model in quantized_models.items():
            components = model.forward_components(inputs, ids, weights)
            method = f"int4_{name}"
            predicted[method] = components.resident_output
            calls[method] += int(
                ((~components.missing_mask) & valid[..., None]).sum()
            )
        teacher_logits = states["next_raw_target_router_logits"].float()
        teacher_ids = states["next_selected_expert_ids"].long()
        logits = {
            name: _router_logits(
                value, states, norm_weight=next_norm, router_weight=next_router
            )
            for name, value in predicted.items()
        }
        batch_recall = {
            name: _overlap(value, teacher_ids) for name, value in logits.items()
        }
        batch_routed_mse = {
            name: (value.float() - routed.float()).square().mean(-1)
            for name, value in predicted.items() if name != "teacher"
        }
        batch_logit_mse = {
            name: (value.float() - teacher_logits).square().mean(-1)
            for name, value in logits.items() if name != "teacher"
        }
        valid_endpoints += int(valid.sum())
        if batch_index == 0:
            print(json.dumps({
                "event": "first_tune_batch",
                "methods": methods,
                "valid_endpoints": valid_endpoints,
            }), flush=True)
        for row, request in enumerate(requests):
            for horizon_index in range(valid.shape[1]):
                if not bool(valid[row, horizon_index]):
                    continue
                horizon = horizon_index + 1
                key = (request, horizon)
                payload: dict[str, Any] = {
                    "request_id": request, "horizon": horizon, "layer": LAYER
                }
                for name in methods:
                    recall_value = float(
                        batch_recall[name][row, horizon_index]
                    )
                    recalls[name][key].append(recall_value)
                    payload[f"{name}_recall_at_8"] = recall_value
                    if name != "teacher":
                        rv = float(batch_routed_mse[name][row, horizon_index])
                        lv = float(batch_logit_mse[name][row, horizon_index])
                        routed_mse[name].append(rv)
                        logit_mse[name].append(lv)
                        payload[f"{name}_routed_mse"] = rv
                        payload[f"{name}_next_router_logit_mse"] = lv
                rows_out.append(payload)
    metrics: dict[str, Any] = {}
    for name in methods:
        mean, horizons = _request_macro(recalls[name])
        metrics[name] = {
            "request_macro_recall_at_8": mean,
            "by_horizon": horizons,
        }
        if name != "teacher":
            metrics[name].update({
                "endpoint_mean_routed_mse": (
                    sum(routed_mse[name]) / len(routed_mse[name])
                ),
                "endpoint_mean_next_router_logit_mse": (
                    sum(logit_mse[name]) / len(logit_mse[name])
                ),
            })
        if name.startswith("int4_"):
            metrics[name]["resident_expert_calls"] = calls[name]
            metrics[name]["resident_call_fraction"] = (
                calls[name] / (8 * valid_endpoints)
            )
    return {
        "valid_endpoints": valid_endpoints,
        "metrics": metrics,
        "calls": calls,
    }, rows_out

def _payload_bytes(model: PackedInt4ResidentExperts) -> int:
    return sum(
        int(getattr(model, name).numel() * getattr(model, name).element_size())
        for name in ("gate_up_packed", "gate_up_scales", "down_packed", "down_scales")
    )


def _state_bytes(model: PackedInt4ResidentExperts) -> int:
    return sum(int(value.numel() * value.element_size()) for value in model.state_dict().values())


def _serialized_state_bytes(model: PackedInt4ResidentExperts) -> int:
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return stream.tell()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if len(args.source_commit) != 40:
        raise ValueError("source commit must be a full SHA")
    if not 0.0 < args.ridge_fraction <= 1.0:
        raise ValueError("ridge fraction must lie in (0,1]")
    if args.output.exists() and not args.preflight_only:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if sha256_file(args.partition_manifest) != EXPECTED_PARTITION_SHA256:
        raise ValueError("partition manifest SHA changed")
    if sha256_file(args.reuse_split_manifest) != EXPECTED_REUSE_SHA256:
        raise ValueError("reuse split manifest SHA changed")
    validate_partition(args.partition_manifest, "b2_reuse_4096")
    reuse = validate_reuse_split(args.reuse_split_manifest)
    train_requests = set(reuse["inner_split"]["training_requests"])
    tune_requests = set(reuse["inner_split"]["tuning_requests"])
    if train_requests & tune_requests or len(train_requests) != 224 or len(tune_requests) != 32:
        raise ValueError("B2 inner split is not the frozen request-disjoint 224/32 split")
    namespace = SimpleNamespace(
        data_profile="b2_reuse_4096",
        layer=LAYER,
        next_router_agreement=True,
        train_index=args.index,
        train_corpus=args.corpus,
        train_companion=args.companion,
        tune_index=args.index,
        tune_corpus=args.corpus,
        tune_companion=args.companion,
    )
    train_dataset, train_groups = load_split(namespace, "train", selected_requests=train_requests)
    tune_dataset, tune_groups = load_split(namespace, "tune", selected_requests=tune_requests)
    if train_groups != train_requests or tune_groups != tune_requests or len(train_dataset) != 3584 or len(tune_dataset) != 512:
        raise ValueError("loaded B2 inner split changed")
    _plan, resident_by_layer, contract = _load_and_validate_plan(args.resident_plan)
    checkpoint = IndexedCheckpoint(args.target_model)
    if checkpoint.index_sha256 != EXPECTED_TARGET_SHA256:
        raise ValueError("target checkpoint index SHA changed")
    preflight = {
        "schema": SCHEMA,
        **contract,
        "layer": LAYER,
        "train_requests": len(train_groups),
        "train_rows": len(train_dataset),
        "tune_requests": len(tune_groups),
        "tune_rows": len(tune_dataset),
        "request_disjoint": True,
        "partition_manifest_sha256": EXPECTED_PARTITION_SHA256,
        "reuse_split_manifest_sha256": EXPECTED_REUSE_SHA256,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "source_sha256": sha256_file(Path(__file__)),
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    device = torch.device(args.device)
    resident_ids = torch.tensor(resident_by_layer[LAYER], dtype=torch.long, device=device)
    gate_up, down = load_target_layer_experts(checkpoint, LAYER, device=device, dtype=torch.bfloat16)
    baseline = PackedInt4ResidentExperts.from_target(
        resident_ids, None, gate_up, down, exact_k=8, group_size=GROUP_SIZE
    )
    baseline.requires_grad_(False)
    baseline.eval()
    prefix = f"model.language_model.layers.{LAYER + 1}."
    names = (prefix + "post_attention_layernorm.weight", prefix + "mlp.gate.weight")
    target = checkpoint.tensors(names)
    next_norm = target[names[0]].to(device=device, dtype=torch.bfloat16)
    next_router = target[names[1]].to(device=device, dtype=torch.bfloat16)

    transforms, fitting = _fit_maps(
        train_dataset,
        device=device,
        microbatch=args.microbatch,
        num_workers=args.num_workers,
        resident_ids=resident_ids,
        gate_up=gate_up,
        down=down,
        ridge_fraction=args.ridge_fraction,
    )
    baseline_payload = _payload_bytes(baseline)
    baseline_state = _state_bytes(baseline)
    baseline_serialized = _serialized_state_bytes(baseline)
    expected_layer_payload = 98 * CELL_BYTES
    if baseline_payload != expected_layer_payload:
        raise ValueError("production L11 packed payload geometry changed")

    quantized_models: dict[str, PackedInt4ResidentExperts] = {
        "amax_baseline": baseline,
        "mse_baseline": _build_folded_model(
            baseline, gate_up, down, "gamma", 1.0, scale_method="mse"
        ),
    }
    for name, transform in transforms.items():
        for scale_method in ("amax", "mse"):
            quantized_models[f"{scale_method}_{name}"] = _build_folded_model(
                baseline,
                gate_up,
                down,
                name,
                transform,
                scale_method=scale_method,
            )

    evaluation, rows = _evaluate(
        tune_dataset,
        device=device,
        microbatch=args.microbatch,
        num_workers=args.num_workers,
        resident_ids=resident_ids,
        gate_up=gate_up,
        down=down,
        next_norm=next_norm,
        next_router=next_router,
        quantized_models=quantized_models,
        transforms=transforms,
    )
    if evaluation["metrics"]["teacher"]["request_macro_recall_at_8"] < 0.98:
        raise ValueError(
            "authoritative next-router reconstruction fell below 0.98"
        )

    baseline_method = "int4_amax_baseline"
    baseline_recall = float(
        evaluation["metrics"][baseline_method]["request_macro_recall_at_8"]
    )
    baseline_calls = int(evaluation["calls"][baseline_method])
    storage: dict[str, Any] = {}
    rung_results: dict[str, Any] = {}
    for model_name, model in quantized_models.items():
        payload = _payload_bytes(model)
        state_bytes = _state_bytes(model)
        serialized_bytes = _serialized_state_bytes(model)
        if (
            payload != baseline_payload
            or state_bytes != baseline_state
            or serialized_bytes != baseline_serialized
        ):
            raise ValueError(f"{model_name} serialized/storage parity failed")
        method = f"int4_{model_name}"
        candidate_recall = float(
            evaluation["metrics"][method]["request_macro_recall_at_8"]
        )
        horizon_lifts = {
            horizon: (
                float(evaluation["metrics"][method]["by_horizon"][horizon])
                - float(
                    evaluation["metrics"][baseline_method]["by_horizon"][
                        horizon
                    ]
                )
            )
            for horizon in evaluation["metrics"][method]["by_horizon"]
        }
        call_delta = int(evaluation["calls"][method]) - baseline_calls
        lift = candidate_recall - baseline_recall
        rung_results[model_name] = {
            "candidate_recall_lift": lift,
            "candidate_recall_lift_by_horizon": horizon_lifts,
            "candidate_routed_mse_delta": (
                evaluation["metrics"][method]["endpoint_mean_routed_mse"]
                - evaluation["metrics"][baseline_method][
                    "endpoint_mean_routed_mse"
                ]
            ),
            "candidate_next_router_logit_mse_delta": (
                evaluation["metrics"][method][
                    "endpoint_mean_next_router_logit_mse"
                ]
                - evaluation["metrics"][baseline_method][
                    "endpoint_mean_next_router_logit_mse"
                ]
            ),
            "resident_expert_call_delta": call_delta,
            "local_gate_passed": (
                lift >= args.minimum_recall_lift
                and min(horizon_lifts.values())
                >= -args.maximum_horizon_regression
                and call_delta == 0
            ),
        }
        storage[model_name] = {
            "logical_packed_weight_scale_bytes": payload,
            "state_tensor_bytes": state_bytes,
            "serialized_state_bytes": serialized_bytes,
            "serialized_byte_parity_verified": True,
        }

    selectable = [name for name in quantized_models if name != "amax_baseline"]
    best_name = max(
        selectable,
        key=lambda value: rung_results[value]["candidate_recall_lift"],
    )
    args.output.mkdir(parents=True)
    checkpoint_path = args.output / f"l11_{best_name}_packed_int4_state.pt"
    torch.save(quantized_models[best_name].state_dict(), checkpoint_path)
    # Path-based torch ZIP containers include the basename in record headers.
    # Canonical BytesIO serialization above is the exact like-for-like parity
    # check; report the export container bytes separately from serving payload.
    exported_artifact = {
        "model_name": best_name,
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "export_file_container_bytes": checkpoint_path.stat().st_size,
        "archive_container_overhead_is_not_serving_payload": True,
        **storage[best_name],
    }

    manifest = {
        **preflight,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "split_policy": "B2 train fit; request-disjoint B2 tune gate",
        "quantizers": {
            "amax": "PackedInt4ResidentExperts production reference",
            "mse": "RouteQuant v1 four-step alternating least-squares scales",
            "group_size": GROUP_SIZE,
            "scale_dtype": "bfloat16",
            "packed_layout": "production signed INT4 +8",
        },
        "optimizer_constructed": False,
        "training_started": False,
        "formal_validation_opened": False,
        "development_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "serving_runtime_module_added": False,
        "serving_calls_macs_traffic_changed": False,
    }
    result = {
        "schema": SCHEMA,
        **contract,
        "layer": LAYER,
        "split": "tune",
        "requests": len(tune_groups),
        "rows": len(tune_dataset),
        "valid_endpoints": evaluation["valid_endpoints"],
        "fitting": fitting,
        "metrics": evaluation["metrics"],
        "rungs": rung_results,
        "best_rung": best_name,
        "best_recall_lift": rung_results[best_name][
            "candidate_recall_lift"
        ],
        "minimum_recall_lift": args.minimum_recall_lift,
        "maximum_horizon_regression": args.maximum_horizon_regression,
        "any_local_gate_passed": any(
            rung_results[name]["local_gate_passed"] for name in selectable
        ),
        "efficiency_nonregression_verified": all(
            value["resident_expert_call_delta"] == 0
            for value in rung_results.values()
        ),
        "all_layer_logical_packed_bytes_unchanged": TOTAL_PACKED_BYTES,
        "serialized_baseline_l11_state_bytes": baseline_serialized,
        "storage_parity": storage,
        "exported_artifact": exported_artifact,
        "formal_validation_opened": False,
        "development_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "RUN_MANIFEST.json", manifest)
    write_rows(args.output / "tune_predictions.jsonl", rows)
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    write_checksums(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
