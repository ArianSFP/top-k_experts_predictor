#!/usr/bin/env python3
"""Leakage-safe L11 screen for exact Core64 plus learned width-128 J-tail.

This is an architecture screen, not a formal evaluation.  It consumes only the
frozen B2 fitting train/tune requests and never opens development, calibration,
or sealed-test data.  The 64 frequency-core experts are frozen exact BF16; only
the 192 compact tail experts are optimized.
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
import time
from types import SimpleNamespace
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor, nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.exact_k import stable_topk  # noqa: E402
from harp_rtt.shadow_checkpoint import IndexedCheckpoint, load_target_layer_experts, sha256_file  # noqa: E402
from harp_rtt.shadow_expert import (  # noqa: E402
    dequantize_groupwise_int4,
    quantize_groupwise_int4,
    target_selected_expert_outputs,
)
from runpod.train_harp_delta_v3 import (  # noqa: E402
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
    loader,
)
from runpod.train_shadow_experts_local import (  # noqa: E402
    batch_tensors,
    load_split,
    next_router_agreement_loss,
)


SCHEMA = "hybridcore64_learned_jtail_l11_screen_v1"
HIDDEN = 2048
NATIVE_WIDTH = 512
TAIL_WIDTH = 128
EXPERTS = 256
CORE_SIZE = 64
DEPLOY_TAIL_CELLS = 5160
TOTAL_LAYERS = 40
PACKED_CAP = 6_433_996_800
MAC_CAP_L11 = 17_510_000
FULL_CELL_BYTES = 1_671_168
TAIL_CELL_BYTES = 417_792


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--selectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--layer", type=int, default=11, choices=range(39))
    parser.add_argument("--microbatch", type=int, default=8, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--router-weight", type=float, default=0.05)
    parser.add_argument("--jspace-weight", type=float, default=0.10)
    parser.add_argument("--aggregate-weight", type=float, default=1.0)
    parser.add_argument("--individual-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.10)
    parser.add_argument("--latency-repeats", type=int, default=50)
    parser.add_argument("--allow-reference-grouped-path", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def emit(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def overlap(logits: Tensor, teacher_ids: Tensor) -> Tensor:
    predicted = stable_topk(logits, k=8)
    return (teacher_ids[..., None] == predicted[..., None, :]).any(-1).float().mean(-1)


def request_macro(values: Mapping[tuple[str, int], list[float]]) -> dict[str, Any]:
    per_request_horizon = {key: sum(rows) / len(rows) for key, rows in values.items() if rows}
    by_horizon: dict[int, list[float]] = defaultdict(list)
    for (_request, horizon), value in per_request_horizon.items():
        by_horizon[horizon].append(value)
    horizons = {str(h): sum(rows) / len(rows) for h, rows in sorted(by_horizon.items())}
    return {
        "request_macro_recall_at_8": sum(per_request_horizon.values()) / len(per_request_horizon),
        "by_horizon": horizons,
    }


def frequency_core(allocation: Mapping[str, Any], layer: int) -> tuple[list[int], list[int]]:
    counts = torch.as_tensor(allocation["expert_counts"], dtype=torch.int64)
    if counts.shape != (TOTAL_LAYERS, EXPERTS):
        raise ValueError("allocation expert-count geometry changed")
    # Stable sort makes the lower expert ID authoritative on count ties.
    core = torch.argsort(counts[layer], descending=True, stable=True)[:CORE_SIZE].tolist()
    residents = [int(value) for value in allocation["resident_expert_ids_by_layer"][layer]]
    if len(residents) != 98 or not set(core).issubset(residents):
        raise ValueError("L11 compliant residents no longer contain mandatory frequency Core64")
    return [int(value) for value in core], residents


class LearnedJTail(nn.Module):
    """Materialized expert-specific width-128 SwiGLU cells for noncore IDs."""

    def __init__(
        self,
        core_ids: list[int],
        selected_neurons: Tensor,
        target_gate_up: Tensor,
        target_down: Tensor,
        initializer_scale: float,
    ) -> None:
        super().__init__()
        core = set(core_ids)
        tail_ids = [expert for expert in range(EXPERTS) if expert not in core]
        if len(tail_ids) != EXPERTS - CORE_SIZE:
            raise ValueError("tail namespace is not exactly 192 experts")
        self.register_buffer("tail_ids", torch.tensor(tail_ids, dtype=torch.int64, device=target_gate_up.device))
        mapping = torch.full((EXPERTS,), -1, dtype=torch.int64, device=target_gate_up.device)
        mapping[self.tail_ids] = torch.arange(len(tail_ids), device=target_gate_up.device)
        self.register_buffer("tail_index", mapping)
        rows = selected_neurons.to(device=target_gate_up.device, dtype=torch.int64).index_select(0, self.tail_ids)
        if rows.shape != (192, TAIL_WIDTH):
            raise ValueError("activation_j selector geometry changed")
        gate = target_gate_up.index_select(0, self.tail_ids)[:, :NATIVE_WIDTH]
        up = target_gate_up.index_select(0, self.tail_ids)[:, NATIVE_WIDTH:]
        gather = rows[..., None].expand(-1, -1, HIDDEN)
        compact_gate = gate.gather(1, gather)
        compact_up = up.gather(1, gather)
        compact_down = target_down.index_select(0, self.tail_ids).gather(
            2, rows[:, None, :].expand(-1, HIDDEN, -1)
        )
        self.gate_up = nn.Parameter(torch.cat((compact_gate, compact_up), dim=1).to(torch.bfloat16))
        self.down = nn.Parameter((compact_down * float(initializer_scale)).to(torch.bfloat16))

    def selected(self, hidden_states: Tensor, selected_ids: Tensor) -> tuple[Tensor, Tensor]:
        leading = hidden_states.shape[:-1]
        hidden = hidden_states.reshape(-1, HIDDEN)
        ids = selected_ids.reshape(hidden.shape[0], -1).long()
        local = self.tail_index[ids]
        active = local >= 0
        output = torch.zeros(hidden.shape[0], ids.shape[1], HIDDEN, dtype=hidden.dtype, device=hidden.device)
        for local_id in torch.unique(local[active]).tolist():
            positions = (local == int(local_id)).nonzero(as_tuple=False)
            token, slot = positions[:, 0], positions[:, 1]
            gate, up = F.linear(hidden[token], self.gate_up[int(local_id)]).chunk(2, -1)
            output[token, slot] = F.linear(F.silu(gate) * up, self.down[int(local_id)])
        return output.reshape(*leading, ids.shape[1], HIDDEN), active.reshape(*leading, ids.shape[1])


def quantized_tail_weights(model: LearnedJTail) -> tuple[Tensor, Tensor]:
    values = []
    for weight in (model.gate_up, model.down):
        packed, scales = quantize_groupwise_int4(weight.detach(), group_size=64)
        values.append(dequantize_groupwise_int4(packed, scales, group_size=64, dtype=torch.bfloat16))
    return values[0], values[1]


def selected_with_weights(
    model: LearnedJTail, hidden_states: Tensor, selected_ids: Tensor, weights: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    gate_up, down = weights
    leading = hidden_states.shape[:-1]
    hidden = hidden_states.reshape(-1, HIDDEN)
    ids = selected_ids.reshape(hidden.shape[0], -1).long()
    local = model.tail_index[ids]
    active = local >= 0
    output = torch.zeros(hidden.shape[0], ids.shape[1], HIDDEN, dtype=hidden.dtype, device=hidden.device)
    for local_id in torch.unique(local[active]).tolist():
        positions = (local == int(local_id)).nonzero(as_tuple=False)
        token, slot = positions[:, 0], positions[:, 1]
        gate, up = F.linear(hidden[token], gate_up[int(local_id)]).chunk(2, -1)
        output[token, slot] = F.linear(F.silu(gate) * up, down[int(local_id)])
    return output.reshape(*leading, ids.shape[1], HIDDEN), active.reshape(*leading, ids.shape[1])


def aggregate(values: Tensor, weights: Tensor, mask: Tensor) -> Tensor:
    return (values.float() * weights[..., None].float() * mask[..., None].float()).sum(-2)


@torch.no_grad()
def evaluate(
    dataset: Any,
    model: LearnedJTail,
    gate_up: Tensor,
    down: Tensor,
    next_norm: Tensor,
    next_router: Tensor,
    residents: list[int],
    *,
    layer: int,
    microbatch: int,
    device: torch.device,
    quantized: bool = False,
) -> dict[str, Any]:
    model.eval()
    qweights = quantized_tail_weights(model) if quantized else None
    resident_table = torch.zeros(EXPERTS, dtype=torch.bool, device=device)
    resident_table[torch.tensor(residents, device=device)] = True
    recall_values = {name: defaultdict(list) for name in ("teacher", "exact", "compliant", "hybrid")}
    mse = {name: [] for name in ("compliant", "hybrid")}
    calls = {name: 0 for name in ("core", "tail", "compliant")}
    valid_endpoints = 0
    for host in loader(dataset, batch=microbatch, shuffle=False, seed=0, workers=0, device=device):
        inputs, routed, ids, route_weights, valid, requests, states = batch_tensors(host, layer=layer, device=device)
        exact_values = target_selected_expert_outputs(inputs, ids, gate_up, down)
        # Preserve the authoritative captured accumulation and subtract only
        # the exact cells being replaced. This exactly matches the frozen
        # selector screen's compliant BF16 arithmetic.
        exact_routed = routed
        compliant_mask = resident_table[ids]
        compliant = routed - aggregate(exact_values, route_weights, ~compliant_mask)
        if qweights is None:
            tail_values, tail_mask = model.selected(inputs, ids)
        else:
            tail_values, tail_mask = selected_with_weights(model, inputs, ids, qweights)
        core_mask = ~tail_mask
        hybrid = (
            routed
            - aggregate(exact_values, route_weights, tail_mask)
            + aggregate(tail_values, route_weights, tail_mask)
        )
        routed_by_name = {"teacher": routed, "exact": exact_routed, "compliant": compliant, "hybrid": hybrid}
        teacher_logits = None
        batch_recall: dict[str, Tensor] = {}
        for name, predicted in routed_by_name.items():
            _loss, _parts, logits = next_router_agreement_loss(
                predicted, states, valid, norm_weight=next_norm, router_weight=next_router
            )
            if name == "teacher":
                teacher_logits = logits
            batch_recall[name] = overlap(logits, states["next_selected_expert_ids"].long())
            if name in mse:
                mse[name].extend(((predicted.float() - routed.float()).square().mean(-1)[valid]).cpu().tolist())
        active_slots = valid[..., None]
        calls["core"] += int((core_mask & active_slots).sum())
        calls["tail"] += int((tail_mask & active_slots).sum())
        calls["compliant"] += int((compliant_mask & active_slots).sum())
        valid_endpoints += int(valid.sum())
        for row, request in enumerate(requests):
            for horizon_index in range(valid.shape[1]):
                if not bool(valid[row, horizon_index]):
                    continue
                key = (request, horizon_index + 1)
                for name in batch_recall:
                    recall_values[name][key].append(float(batch_recall[name][row, horizon_index]))
    metrics = {name: request_macro(rows) for name, rows in recall_values.items()}
    for name in mse:
        metrics[name]["endpoint_mean_routed_mse"] = sum(mse[name]) / len(mse[name])
    metrics["calls"] = {
        **calls,
        "valid_endpoints": valid_endpoints,
        "core_per_endpoint": calls["core"] / valid_endpoints,
        "tail_per_endpoint": calls["tail"] / valid_endpoints,
        "compliant_per_endpoint": calls["compliant"] / valid_endpoints,
    }
    return metrics


def execute_sparse(
    inputs: Tensor,
    ids: Tensor,
    route_weights: Tensor,
    allowed: Tensor,
    gate_up: Tensor,
    down: Tensor,
    id_map: Tensor | None = None,
) -> Tensor:
    hidden = inputs.reshape(-1, HIDDEN)
    flat_ids = ids.reshape(hidden.shape[0], -1)
    weights = route_weights.reshape_as(flat_ids).to(hidden.dtype)
    selected = allowed[flat_ids]
    output = torch.zeros_like(hidden)
    for expert in torch.unique(flat_ids[selected]).tolist():
        positions = (flat_ids == int(expert)).nonzero(as_tuple=False)
        keep = selected[positions[:, 0], positions[:, 1]]
        positions = positions[keep]
        token, slot = positions[:, 0], positions[:, 1]
        row = int(id_map[int(expert)]) if id_map is not None else int(expert)
        gate, up = F.linear(hidden[token], gate_up[row]).chunk(2, -1)
        value = F.linear(F.silu(gate) * up, down[row])
        output.index_add_(0, token, value * weights[token, slot, None])
    return output.reshape_as(inputs)


def execute_vector_tail(
    inputs: Tensor,
    ids: Tensor,
    route_weights: Tensor,
    model: LearnedJTail,
) -> Tensor:
    """Execute all selected tails with two native batched GEMM calls."""

    hidden = inputs.reshape(-1, HIDDEN)
    flat_ids = ids.reshape(hidden.shape[0], -1).long()
    weights = route_weights.reshape_as(flat_ids).to(hidden.dtype)
    local = model.tail_index[flat_ids]
    positions = (local >= 0).nonzero(as_tuple=False)
    token, slot = positions[:, 0], positions[:, 1]
    rows = local[token, slot]
    selected_gate_up = model.gate_up.index_select(0, rows)
    projected = torch.bmm(selected_gate_up, hidden[token, :, None]).squeeze(-1)
    del selected_gate_up
    gate, up = projected.chunk(2, -1)
    activation = F.silu(gate) * up
    selected_down = model.down.index_select(0, rows)
    values = torch.bmm(selected_down, activation[..., None]).squeeze(-1)
    del selected_down
    output = torch.zeros_like(hidden)
    output.index_add_(0, token, values * weights[token, slot, None])
    return output.reshape_as(inputs)


@torch.no_grad()
def benchmark(
    host: Mapping[str, Any], model: LearnedJTail, gate_up: Tensor, down: Tensor,
    residents: list[int], core: list[int], layer: int, device: torch.device, repeats: int,
) -> dict[str, Any]:
    inputs, _routed, ids, weights, _valid, _requests, _states = batch_tensors(host, layer=layer, device=device)
    resident_table = torch.zeros(EXPERTS, dtype=torch.bool, device=device)
    resident_table[torch.tensor(residents, device=device)] = True
    core_table = torch.zeros_like(resident_table)
    core_table[torch.tensor(core, device=device)] = True
    tail_table = ~core_table

    def compliant() -> Tensor:
        return execute_sparse(inputs, ids, weights, resident_table, gate_up, down)

    def hybrid() -> Tensor:
        return execute_sparse(inputs, ids, weights, core_table, gate_up, down) + execute_sparse(
            inputs, ids, weights, tail_table, model.gate_up, model.down, model.tail_index
        )

    def hybrid_vector_tail() -> Tensor:
        return execute_sparse(inputs, ids, weights, core_table, gate_up, down) + execute_vector_tail(
            inputs, ids, weights, model
        )

    for _ in range(5):
        compliant(); hybrid(); hybrid_vector_tail()
    torch.cuda.synchronize(device)

    def timed(function: Any) -> list[float]:
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record(); function(); stop.record(); stop.synchronize()
            values.append(float(start.elapsed_time(stop)))
        return sorted(values)

    baseline = timed(compliant)
    candidate = timed(hybrid)
    torch.cuda.synchronize(device)
    before = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    vector_candidate = timed(hybrid_vector_tail)
    peak_temporary = max(0, int(torch.cuda.max_memory_allocated(device)) - before)
    median = lambda rows: rows[len(rows) // 2]
    return {
        "batch_requests": len(host["metadata"]["request_id"]),
        "endpoints": int(inputs.numel() // HIDDEN),
        "repeats": repeats,
        "compliant_median_ms": median(baseline),
        "hybrid_median_ms": median(candidate),
        "latency_ratio": median(candidate) / median(baseline),
        "hybrid_vector_tail_median_ms": median(vector_candidate),
        "vector_tail_latency_ratio": median(vector_candidate) / median(baseline),
        "vector_tail_peak_temporary_bytes": peak_temporary,
        "vector_tail_selected_occurrences": int(tail_table[ids].sum()),
        "vector_tail_gather_bytes_per_occurrence_bf16": 3 * HIDDEN * TAIL_WIDTH * 2,
        "compliant_samples_ms": baseline,
        "hybrid_samples_ms": candidate,
        "hybrid_vector_tail_samples_ms": vector_candidate,
        "implementation": "route-grouped exact core plus two native torch.bmm calls over gathered tail occurrences",
    }


def loss_batch(
    model: LearnedJTail, host: Mapping[str, Any], gate_up: Tensor, down: Tensor,
    next_norm: Tensor, next_router: Tensor, args: argparse.Namespace,
) -> tuple[Tensor, dict[str, float]]:
    inputs, routed, ids, weights, valid, _requests, states = batch_tensors(host, layer=args.layer, device=gate_up.device)
    with torch.no_grad():
        exact_values = target_selected_expert_outputs(inputs, ids, gate_up, down)
    tail_values, tail_mask = model.selected(inputs, ids)
    core_mask = ~tail_mask
    predicted = (
        routed
        - aggregate(exact_values, weights, tail_mask)
        + aggregate(tail_values, weights, tail_mask)
    )
    active_tail = tail_mask & valid[..., None]
    individual_rows = F.smooth_l1_loss(tail_values.float(), exact_values.float(), reduction="none").mean(-1)
    individual = (individual_rows * active_tail.float()).sum() / active_tail.sum().clamp_min(1)
    aggregate_rows = F.smooth_l1_loss(predicted.float(), routed.float(), reduction="none").mean(-1)
    aggregate_loss = (aggregate_rows * valid.float()).sum() / valid.sum().clamp_min(1)
    cosine_rows = 1.0 - F.cosine_similarity(predicted.float(), routed.float(), dim=-1)
    cosine = (cosine_rows * valid.float()).sum() / valid.sum().clamp_min(1)
    router_loss, _parts, logits = next_router_agreement_loss(
        predicted, states, valid, norm_weight=next_norm, router_weight=next_router
    )
    teacher_logits = states["next_raw_target_router_logits"].float()
    centered_pred = logits.float() - logits.float().mean(-1, keepdim=True)
    centered_teacher = teacher_logits - teacher_logits.mean(-1, keepdim=True)
    scale = centered_teacher.square().mean(-1).clamp_min(1e-4)
    j_rows = (centered_pred - centered_teacher).square().mean(-1) / scale
    jspace = (j_rows * valid.float()).sum() / valid.sum().clamp_min(1)
    total = (
        args.individual_weight * individual
        + args.aggregate_weight * aggregate_loss
        + args.cosine_weight * cosine
        + args.router_weight * router_loss
        + args.jspace_weight * jspace
    )
    return total, {
        "loss": float(total.detach()), "individual": float(individual.detach()),
        "aggregate": float(aggregate_loss.detach()), "cosine": float(cosine.detach()),
        "router": float(router_loss.detach()), "jspace": float(jspace.detach()),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    partition = validate_partition(args.partition_manifest, "b2_reuse_4096")
    reuse = validate_reuse_split(args.reuse_split_manifest)
    train_requests = set(reuse["inner_split"]["training_requests"])
    tune_requests = set(reuse["inner_split"]["tuning_requests"])
    if train_requests & tune_requests:
        raise PermissionError("train/tune request overlap")
    namespace = SimpleNamespace(
        data_profile="b2_reuse_4096", layer=args.layer, next_router_agreement=True,
        train_index=args.index, train_corpus=args.corpus, train_companion=args.companion,
        tune_index=args.index, tune_corpus=args.corpus, tune_companion=args.companion,
    )
    train, train_groups = load_split(namespace, "train", selected_requests=train_requests)
    tune, tune_groups = load_split(namespace, "tune", selected_requests=tune_requests)
    allocation = json.loads(args.allocation.read_text(encoding="utf-8"))
    core, residents = frequency_core(allocation, args.layer)
    selectors = torch.load(args.selectors, map_location="cpu", weights_only=False)
    selected = selectors["selected_neurons"]["activation_j"]
    initializer_scale = float(selectors["scales"]["activation_j"])
    checkpoint = IndexedCheckpoint(args.target_model)
    gate_up, down = load_target_layer_experts(checkpoint, args.layer, device=device, dtype=torch.bfloat16)
    prefix = f"model.language_model.layers.{args.layer + 1}."
    names = (prefix + "post_attention_layernorm.weight", prefix + "mlp.gate.weight")
    tensors = checkpoint.tensors(names)
    next_norm = tensors[names[0]].to(device=device, dtype=torch.bfloat16)
    next_router = tensors[names[1]].to(device=device, dtype=torch.bfloat16)
    model = LearnedJTail(core, selected, gate_up, down, initializer_scale).to(device)

    packed_bytes = CORE_SIZE * TOTAL_LAYERS * FULL_CELL_BYTES + DEPLOY_TAIL_CELLS * TAIL_CELL_BYTES
    if packed_bytes != PACKED_CAP:
        raise ValueError("static packed byte identity changed")
    first_host = next(iter(loader(tune, batch=args.microbatch, shuffle=False, seed=0, workers=0, device=device)))
    latency = benchmark(first_host, model, gate_up, down, residents, core, args.layer, device, args.latency_repeats)
    epoch0 = evaluate(tune, model, gate_up, down, next_norm, next_router, residents,
                      layer=args.layer, microbatch=args.microbatch, device=device)
    calls = epoch0["calls"]
    full_mac = 3 * HIDDEN * NATIVE_WIDTH
    tail_mac = 3 * HIDDEN * TAIL_WIDTH
    hybrid_mac = calls["core_per_endpoint"] * full_mac + calls["tail_per_endpoint"] * tail_mac
    compliant_mac = calls["compliant_per_endpoint"] * full_mac
    traffic = {
        "compliant_bytes_per_endpoint": calls["compliant_per_endpoint"] * FULL_CELL_BYTES,
        "hybrid_bytes_per_endpoint": calls["core_per_endpoint"] * FULL_CELL_BYTES + calls["tail_per_endpoint"] * TAIL_CELL_BYTES,
    }
    references_ok = (
        abs(epoch0["compliant"]["request_macro_recall_at_8"] - 0.9410400390625) <= 5e-6
        and abs(epoch0["exact"]["request_macro_recall_at_8"] - 0.9827880859375) <= 5e-6
    )
    latency_ok = latency["vector_tail_latency_ratio"] <= 1.0
    grouped_path = bool(args.allow_reference_grouped_path) and hybrid_mac <= compliant_mac
    preflight = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit, "layer": args.layer,
        "train_requests": len(train_groups), "train_rows": len(train),
        "tune_requests": len(tune_groups), "tune_rows": len(tune),
        "request_disjoint": not bool(train_groups & tune_groups),
        "core_ids": core, "core_count": len(core), "compliant_resident_count": len(residents),
        "learned_tail_experts_l11": 192, "deploy_tail_cells_all_layers": DEPLOY_TAIL_CELLS,
        "packed_bytes": packed_bytes, "packed_byte_cap": PACKED_CAP,
        "selector_runtime_bytes": 0,
        "selector_note": "activation_j IDs initialize materialized weights and are not needed by serving",
        "hybrid_mac_per_endpoint": hybrid_mac, "compliant_mac_per_endpoint": compliant_mac,
        "mac_cap_l11": MAC_CAP_L11, "traffic": traffic, "latency": latency,
        "epoch0": epoch0, "reference_metrics_reproduced": references_ok,
        "static_gate": packed_bytes <= PACKED_CAP and hybrid_mac <= MAC_CAP_L11,
        "latency_gate": latency_ok, "reference_grouped_path_asserted": grouped_path,
        "training_authorized": references_ok and packed_bytes <= PACKED_CAP and hybrid_mac <= MAC_CAP_L11 and (latency_ok or grouped_path),
        "formal_validation_opened": False, "calibration_opened": False, "sealed_test_opened": False,
        "lineage": {
            "allocation_sha256": sha256_file(args.allocation), "selectors_sha256": sha256_file(args.selectors),
            "partition_manifest_sha256": sha256_file(args.partition_manifest),
            "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
            "target_checkpoint_index_sha256": checkpoint.index_sha256,
        },
    }
    emit(args.output / "PREFLIGHT.json", preflight)
    print(json.dumps({"event": "preflight", **preflight}, sort_keys=True), flush=True)
    if args.preflight_only or not preflight["training_authorized"]:
        emit(args.output / "STAGE_RESULT.json", {**preflight, "training_started": False, "stop_reason": "preflight_only" if args.preflight_only else "efficiency_or_reference_gate_failed"})
        raise SystemExit(0 if args.preflight_only else 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    accumulation = max(1, 32 // args.microbatch)
    best_recall = epoch0["hybrid"]["request_macro_recall_at_8"]
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, Any]] = [{"epoch": 0, "metrics": epoch0}]
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = defaultdict(float); steps = 0
        started = time.monotonic()
        batches = loader(train, batch=args.microbatch, shuffle=True, seed=args.seed + epoch, workers=0, device=device)
        for step, host in enumerate(batches, 1):
            loss, parts = loss_batch(model, host, gate_up, down, next_norm, next_router, args)
            (loss / accumulation).backward()
            if step % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            for name, value in parts.items(): sums[name] += value
            steps += 1
        if steps % accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(tune, model, gate_up, down, next_norm, next_router, residents,
                           layer=args.layer, microbatch=args.microbatch, device=device)
        recall = metrics["hybrid"]["request_macro_recall_at_8"]
        record = {"epoch": epoch, "seconds": time.monotonic() - started,
                  "train": {name: value / steps for name, value in sums.items()}, "metrics": metrics}
        history.append(record); emit(args.output / "PROGRESS.json", {"schema": SCHEMA, "history": history})
        print(json.dumps({"event": "epoch", **record}, sort_keys=True), flush=True)
        if recall > best_recall + 1e-6:
            best_recall = recall; stale = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict({name: value.to(device) for name, value in best_state.items()})
    bf16 = evaluate(tune, model, gate_up, down, next_norm, next_router, residents,
                    layer=args.layer, microbatch=args.microbatch, device=device)
    int4 = evaluate(tune, model, gate_up, down, next_norm, next_router, residents,
                   layer=args.layer, microbatch=args.microbatch, device=device, quantized=True)
    control = bf16["compliant"]["request_macro_recall_at_8"]
    lift = int4["hybrid"]["request_macro_recall_at_8"] - control
    horizon_lifts = {h: int4["hybrid"]["by_horizon"][h] - bf16["compliant"]["by_horizon"][h]
                     for h in bf16["compliant"]["by_horizon"]}
    passed = lift >= 0.015 and min(horizon_lifts.values()) >= -0.002
    checkpoint_out = {"schema": SCHEMA, "layer": args.layer, "core_ids": core,
                      "tail_ids": model.tail_ids.cpu(), "gate_up": model.gate_up.detach().cpu(),
                      "down": model.down.detach().cpu(), "initializer": "activation_j"}
    torch.save(checkpoint_out, args.output / "best_tail_bf16.pt")
    result = {**preflight, "training_started": True, "epochs_completed": len(history) - 1,
              "history": history, "best_bf16": bf16, "post_quant_int4": int4,
              "int4_lift_vs_compliant": lift, "int4_horizon_lifts": horizon_lifts,
              "promotion_gate_passed": passed, "minimum_lift": 0.015,
              "maximum_horizon_regression": 0.002}
    emit(args.output / "STAGE_RESULT.json", result)
    print(json.dumps({"event": "final", "promotion_gate_passed": passed,
                      "int4_lift_vs_compliant": lift, "int4_horizon_lifts": horizon_lifts}, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
