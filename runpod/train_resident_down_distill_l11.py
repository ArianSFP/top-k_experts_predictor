#!/usr/bin/env python3
"""Train a foldable L11 correction to compliant resident down projections.

The serving architecture is unchanged.  Gate/up tensors, resident IDs, native
top-8 IDs and weights are frozen.  Training-only expert-specific low-rank
corrections act on the frozen SwiGLU activations and are folded into the same
resident down tensors before group-64 INT4 export.  No adapter survives export.

This is a request-disjoint B2 representative-layer screen.  It must not open
formal validation, calibration, development, or sealed-test data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
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

from harp_rtt.exact_k import exact_set_nll, stable_topk  # noqa: E402
from harp_rtt.losses import boundary_loss_per_endpoint  # noqa: E402
from harp_rtt.shadow_checkpoint import IndexedCheckpoint, sha256_file  # noqa: E402
from harp_rtt.shadow_expert import (  # noqa: E402
    dequantize_groupwise_int4,
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
)


SCHEMA = "resident_down_distill_l11_screen_v1"
EXPECTED_ALLOCATION_SHA256 = "8b25bdb5a39f08089f8b175ae5f8befa0fe2c3995d5d16bffcc28d2d8cd96536"
EXPECTED_TARGET_INDEX_SHA256 = "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
LAYERS = 40
LAYER = 11
EXPERTS = 256
RESIDENTS = 98
TOPK = 8
HIDDEN = 2048
INTERMEDIATE = 512
GROUP_SIZE = 64
CELL_BYTES = 1_671_168
L11_RESIDENT_BYTES = RESIDENTS * CELL_BYTES
FULL_BUNDLE_RESIDENT_BYTES = 6_433_996_800
MINIMUM_RECALL_LIFT = 0.02
MAXIMUM_HORIZON_REGRESSION = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--nonexpert-model", type=Path, required=True)
    parser.add_argument("--resident-checkpoint", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--predecessor-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--rank", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--minimum-improvement", type=float, default=2e-4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--materialize-batch", type=int, default=16)
    parser.add_argument("--endpoint-batch", type=int, default=256)
    parser.add_argument("--aggregate-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.10)
    parser.add_argument("--preserve-weight", type=float, default=0.25)
    parser.add_argument("--jspace-weight", type=float, default=0.05)
    parser.add_argument("--kl-weight", type=float, default=0.02)
    parser.add_argument("--exact-set-weight", type=float, default=0.01)
    parser.add_argument("--boundary-weight", type=float, default=0.002)
    parser.add_argument("--router-warmup-epochs", type=int, default=1)
    parser.add_argument("--quant-refinement-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def emit(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_allocation(path: Path) -> tuple[dict[str, Any], list[int], str]:
    if sha256_file(path) != EXPECTED_ALLOCATION_SHA256:
        raise RuntimeError("resident allocation SHA256 changed")
    allocation = json.loads(path.read_text(encoding="utf-8"))
    residents_by_layer = allocation.get("resident_expert_ids_by_layer")
    counts = torch.as_tensor(allocation.get("expert_counts"), dtype=torch.int64)
    if (
        counts.shape != (LAYERS, EXPERTS)
        or not isinstance(residents_by_layer, list)
        or len(residents_by_layer) != LAYERS
        or sum(len(row) for row in residents_by_layer) != 3850
        or int(allocation.get("total_residents", -1)) != 3850
        or int(allocation.get("frequency_core_size_per_layer", -1)) != 64
        or allocation.get("frequency_core_inclusion_verified") is not True
        or int(allocation.get("packed_int4_bytes", -1)) != FULL_BUNDLE_RESIDENT_BYTES
    ):
        raise RuntimeError("resident allocation geometry or footprint changed")
    cores: list[list[int]] = []
    for layer in range(LAYERS):
        residents = [int(value) for value in residents_by_layer[layer]]
        if len(residents) != len(set(residents)) or any(not 0 <= value < EXPERTS for value in residents):
            raise RuntimeError(f"layer {layer} resident namespace is invalid")
        core = torch.argsort(counts[layer], descending=True, stable=True)[:64].tolist()
        if not set(core).issubset(residents):
            raise RuntimeError(f"layer {layer} omits a recomputed frequency top64 expert")
        cores.append([int(value) for value in core])
    core_sha = hashlib.sha256(json.dumps(cores, separators=(",", ":")).encode()).hexdigest()
    if core_sha != str(allocation.get("frequency_core_sha256")):
        raise RuntimeError("recomputed frequency-core SHA256 changed")
    residents_l11 = [int(value) for value in residents_by_layer[LAYER]]
    if len(residents_l11) != RESIDENTS:
        raise RuntimeError("L11 resident count is not the expected 98")
    return allocation, residents_l11, core_sha


def validate_resident_checkpoint(
    path: Path, resident_ids: list[int], device: torch.device
) -> tuple[dict[str, Any], dict[str, Tensor], Tensor, Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("resident checkpoint is not a mapping")
    if (
        checkpoint.get("schema") != "harp_shadowroute_local_expert_layer_v1"
        or checkpoint.get("mode") != "resident_int4_only"
        or int(checkpoint.get("layer", -1)) != LAYER
        or int(checkpoint.get("resident_count", -1)) != RESIDENTS
        or checkpoint.get("resident_allocation_plan_sha256") != EXPECTED_ALLOCATION_SHA256
        or checkpoint.get("target_checkpoint_index_sha256") != EXPECTED_TARGET_INDEX_SHA256
    ):
        raise RuntimeError("resident checkpoint lineage changed")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("resident checkpoint lacks model_state_dict")
    expected = {
        "resident_ids": ((RESIDENTS,), torch.int64),
        "expert_to_resident": ((EXPERTS,), torch.int64),
        "gate_up_packed": ((RESIDENTS, 2 * INTERMEDIATE, HIDDEN // 2), torch.uint8),
        "gate_up_scales": ((RESIDENTS, 2 * INTERMEDIATE, HIDDEN // GROUP_SIZE), torch.bfloat16),
        "down_packed": ((RESIDENTS, HIDDEN, INTERMEDIATE // 2), torch.uint8),
        "down_scales": ((RESIDENTS, HIDDEN, INTERMEDIATE // GROUP_SIZE), torch.bfloat16),
    }
    clean: dict[str, Tensor] = {}
    for name, (shape, dtype) in expected.items():
        value = state.get(name)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
            raise RuntimeError(f"resident tensor {name} geometry changed")
        clean[name] = value.cpu().contiguous()
    expected_ids = torch.tensor(resident_ids, dtype=torch.int64)
    mapping = torch.full((EXPERTS,), -1, dtype=torch.int64)
    mapping[expected_ids] = torch.arange(RESIDENTS, dtype=torch.int64)
    if not torch.equal(clean["resident_ids"], expected_ids) or not torch.equal(clean["expert_to_resident"], mapping):
        raise RuntimeError("resident checkpoint namespace differs from allocation")
    logical_bytes = sum(
        clean[name].numel() * clean[name].element_size()
        for name in ("gate_up_packed", "gate_up_scales", "down_packed", "down_scales")
    )
    if logical_bytes != L11_RESIDENT_BYTES:
        raise RuntimeError("L11 resident tensor footprint changed")
    gate_up = dequantize_groupwise_int4(
        clean["gate_up_packed"].to(device), clean["gate_up_scales"].to(device),
        group_size=GROUP_SIZE, dtype=torch.bfloat16,
    )
    down = dequantize_groupwise_int4(
        clean["down_packed"].to(device), clean["down_scales"].to(device),
        group_size=GROUP_SIZE, dtype=torch.bfloat16,
    )
    return dict(checkpoint), clean, gate_up, down


def load_datasets(args: argparse.Namespace) -> tuple[Any, Any, set[str], set[str]]:
    validate_partition(args.partition_manifest, "b2_reuse_4096")
    split = validate_reuse_split(args.reuse_split_manifest)
    train_requests = set(split["inner_split"]["training_requests"])
    tune_requests = set(split["inner_split"]["tuning_requests"])
    if train_requests & tune_requests or (len(train_requests), len(tune_requests)) != (224, 32):
        raise PermissionError("frozen B2 train/tune request split changed")
    namespace = SimpleNamespace(
        data_profile="b2_reuse_4096", layer=LAYER, next_router_agreement=True,
        train_index=args.index, train_corpus=args.corpus, train_companion=args.companion,
        tune_index=args.index, tune_corpus=args.corpus, tune_companion=args.companion,
    )
    train, train_groups = load_split(namespace, "train", selected_requests=train_requests)
    tune, tune_groups = load_split(namespace, "tune", selected_requests=tune_requests)
    if (
        len(train) != 3584 or len(tune) != 512
        or train_groups != train_requests or tune_groups != tune_requests
        or train_groups & tune_groups
    ):
        raise RuntimeError("frozen B2 row/request contract changed")
    return train, tune, train_groups, tune_groups


def load_next_router(
    target_model: Path, nonexpert_model: Path, device: torch.device
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    target = IndexedCheckpoint(target_model)
    nonexpert = IndexedCheckpoint(nonexpert_model)
    if target.index_sha256 != EXPECTED_TARGET_INDEX_SHA256:
        raise RuntimeError("authoritative target checkpoint index changed")
    prefix = f"model.language_model.layers.{LAYER + 1}."
    names = (prefix + "post_attention_layernorm.weight", prefix + "mlp.gate.weight")
    target_values = target.tensors(names)
    candidate_values = nonexpert.tensors(names)
    proof: dict[str, Any] = {}
    for name in names:
        left = target_values[name].cpu().contiguous()
        right = candidate_values[name].cpu().contiguous()
        equal = torch.equal(left, right)
        proof[name] = {
            "shape": list(left.shape), "dtype": str(left.dtype), "bitwise_equal": equal,
            "target_sha256": tensor_sha256(left), "nonexpert_sha256": tensor_sha256(right),
        }
        if not equal:
            raise RuntimeError(f"nonexpert next-router tensor differs from target: {name}")
    # Execute from the authoritative target tensors.  The compact nonexpert
    # checkpoint is opened only for a bitwise portability proof; it is never
    # the source of teacher arithmetic in this screen.
    return (
        target_values[names[0]].to(device=device, dtype=torch.bfloat16),
        target_values[names[1]].to(device=device, dtype=torch.bfloat16),
        {
            "target_index_sha256": target.index_sha256,
            "nonexpert_index_sha256": nonexpert.index_sha256,
            "tensor_bitwise_proof": proof,
        },
    )


def next_logits(
    routed: Tensor,
    current_base: Tensor,
    attention_delta: Tensor,
    norm_weight: Tensor,
    router_weight: Tensor,
) -> Tensor:
    # Preserve the evaluator's native-dtype addition order exactly.  In
    # particular, do not pre-sum these BF16 states into one offset.
    predicted_xplus = current_base + routed
    values = (predicted_xplus + attention_delta).float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    normalized = normalized * (1.0 + norm_weight.float())
    return F.linear(normalized, router_weight.float())


@torch.no_grad()
def materialize(
    dataset: Any,
    *,
    gate_up: Tensor,
    down: Tensor,
    expert_to_resident: Tensor,
    next_norm: Tensor,
    next_router: Tensor,
    args: argparse.Namespace,
    keep_identity: bool,
) -> dict[str, Any]:
    device = gate_up.device
    fields: dict[str, list[Tensor]] = defaultdict(list)
    requests_flat: list[str] = []
    horizons_flat: list[int] = []
    for host in loader(
        dataset, batch=args.materialize_batch, shuffle=False, seed=0, workers=0, device=device
    ):
        inputs, routed, ids, weights, valid, requests, states = batch_tensors(host, layer=LAYER, device=device)
        rows = int(inputs.shape[0] * inputs.shape[1])
        hidden = inputs.reshape(rows, HIDDEN)
        ids_flat = ids.reshape(rows, TOPK).long()
        weights_flat = weights.reshape(rows, TOPK).to(hidden.dtype)
        local = expert_to_resident[ids_flat]
        activations = torch.zeros(rows, TOPK, INTERMEDIATE, dtype=torch.bfloat16, device=device)
        base = torch.zeros(rows, HIDDEN, dtype=hidden.dtype, device=device)
        active = local >= 0
        for local_id in torch.unique(local[active]).tolist():
            positions = (local == int(local_id)).nonzero(as_tuple=False)
            token, slot = positions[:, 0], positions[:, 1]
            gate, up = F.linear(hidden[token], gate_up[int(local_id)]).chunk(2, -1)
            activation = F.silu(gate) * up
            activations[token, slot] = activation
            value = F.linear(activation, down[int(local_id)])
            base[token] += value * weights_flat[token, slot, None]
        valid_flat = valid.reshape(-1).bool()
        current_base = (
            states["post_attention_residual_u"] + states["shared_expert_output_delta_s"]
        ).reshape(rows, HIDDEN)
        attention_delta = (
            states["next_post_attention_residual_u"] - states["post_moe_residual_xplus"]
        ).reshape(rows, HIDDEN)
        teacher_logits = states["next_raw_target_router_logits"].reshape(rows, EXPERTS)
        teacher_ids = states["next_selected_expert_ids"].reshape(rows, TOPK)
        fields["activations"].append(activations[valid_flat].cpu())
        fields["local_ids"].append(local[valid_flat].to(torch.int16).cpu())
        fields["weights"].append(weights_flat[valid_flat].to(torch.bfloat16).cpu())
        fields["base_routed"].append(base[valid_flat].to(torch.bfloat16).cpu())
        fields["target_routed"].append(routed.reshape(rows, HIDDEN)[valid_flat].cpu())
        fields["current_base"].append(current_base[valid_flat].cpu())
        fields["attention_delta"].append(attention_delta[valid_flat].cpu())
        fields["teacher_logits"].append(teacher_logits[valid_flat].cpu())
        fields["teacher_ids"].append(teacher_ids[valid_flat].to(torch.int16).cpu())
        if keep_identity:
            for row, request in enumerate(requests):
                for horizon in range(valid.shape[1]):
                    if bool(valid[row, horizon]):
                        requests_flat.append(str(request))
                        horizons_flat.append(horizon + 1)
    cache: dict[str, Any] = {
        name: torch.cat(parts).to(device, non_blocking=False) for name, parts in fields.items()
    }
    cache["requests"] = requests_flat
    cache["horizons"] = torch.tensor(horizons_flat, dtype=torch.int16)
    replay_logits = next_logits(
        cache["target_routed"], cache["current_base"], cache["attention_delta"],
        next_norm, next_router,
    )
    parity_error = replay_logits.float() - cache["teacher_logits"].float()
    cache["teacher_replay_logits"] = replay_logits.to(torch.bfloat16)
    cache["parity"] = {
        "centered_logit_mse": float(
            (parity_error - parity_error.mean(-1, keepdim=True)).square().mean()
        ),
        "maximum_endpoint_abs_error": float(parity_error.abs().amax()),
        "stable_top8_recall": float(
            overlap(replay_logits, cache["teacher_ids"].long()).mean()
        ),
    }
    return cache


def overlap(logits: Tensor, teacher_ids: Tensor) -> Tensor:
    predicted = stable_topk(logits, k=TOPK)
    return (teacher_ids[..., None] == predicted[..., None, :]).any(-1).float().mean(-1)


def request_macro(values: Tensor, requests: list[str], horizons: Tensor) -> dict[str, Any]:
    if len(requests) != values.numel() or horizons.numel() != values.numel():
        raise ValueError("request-macro identity geometry changed")
    cells: dict[tuple[str, int], list[float]] = defaultdict(list)
    for request, horizon, value in zip(requests, horizons.tolist(), values.tolist()):
        cells[(request, int(horizon))].append(float(value))
    per_cell = {key: sum(rows) / len(rows) for key, rows in cells.items()}
    by_horizon = {}
    for horizon in range(1, 5):
        rows = [value for (_request, h), value in per_cell.items() if h == horizon]
        by_horizon[str(horizon)] = sum(rows) / len(rows)
    return {
        "request_macro_recall_at_8": sum(per_cell.values()) / len(per_cell),
        "by_horizon": by_horizon,
        "request_horizon_cells": len(per_cell),
    }


def metric_from_routed(
    routed: Tensor, cache: Mapping[str, Any], next_norm: Tensor, next_router: Tensor
) -> dict[str, Any]:
    logits = next_logits(
        routed, cache["current_base"], cache["attention_delta"], next_norm, next_router
    )
    hits = overlap(logits, cache["teacher_ids"].long()).detach().cpu()
    metric = request_macro(hits, cache["requests"], cache["horizons"])
    target = cache["target_routed"].float()
    metric["endpoint_routed_mse"] = float((routed.float() - target).square().mean())
    metric["endpoint_routed_cosine"] = float(F.cosine_similarity(routed.float(), target, dim=-1).mean())
    return metric


class ResidentDownLoRA(nn.Module):
    """Training-only expert-specific low-rank corrections, absent at serving."""

    def __init__(self, base_down: Tensor, *, rank: int, alpha: float, seed: int) -> None:
        super().__init__()
        if base_down.shape != (RESIDENTS, HIDDEN, INTERMEDIATE):
            raise ValueError("base resident down geometry changed")
        if not math.isfinite(float(alpha)) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.register_buffer("base_down", base_down.detach().to(torch.bfloat16))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initializer = torch.randn(RESIDENTS, rank, INTERMEDIATE, generator=generator)
        initializer = initializer / initializer.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.lora_b = nn.Parameter(initializer.to(device=base_down.device, dtype=torch.float32))
        self.lora_a = nn.Parameter(
            torch.zeros(RESIDENTS, HIDDEN, rank, device=base_down.device, dtype=torch.float32)
        )

    def correction(self, activations: Tensor, local_ids: Tensor, weights: Tensor) -> Tensor:
        rows, slots, width = activations.shape
        if width != INTERMEDIATE or local_ids.shape != (rows, slots) or weights.shape != (rows, slots):
            raise ValueError("cached resident activation geometry changed")
        active = local_ids >= 0
        positions = active.nonzero(as_tuple=False)
        if not positions.numel():
            return torch.zeros(rows, HIDDEN, device=activations.device, dtype=torch.float32)
        token, slot = positions[:, 0], positions[:, 1]
        local = local_ids[token, slot].long()
        values = activations[token, slot].float()
        low = torch.bmm(self.lora_b.index_select(0, local), values[..., None]).squeeze(-1)
        projected = torch.bmm(self.lora_a.index_select(0, local), low[..., None]).squeeze(-1)
        projected = projected * self.scale * weights[token, slot, None].float()
        output = torch.zeros(rows, HIDDEN, device=activations.device, dtype=torch.float32)
        output.index_add_(0, token, projected)
        return output

    def forward_cache(self, cache: Mapping[str, Tensor], indices: Tensor) -> tuple[Tensor, Tensor]:
        base = cache["base_routed"].index_select(0, indices).float()
        correction = self.correction(
            cache["activations"].index_select(0, indices),
            cache["local_ids"].index_select(0, indices),
            cache["weights"].index_select(0, indices),
        )
        return base + correction, correction

    @torch.no_grad()
    def folded_chunk(self, start: int, stop: int) -> Tensor:
        delta = torch.bmm(self.lora_a[start:stop], self.lora_b[start:stop]) * self.scale
        return self.base_down[start:stop].float() + delta


def quantize_groupwise_int4_mse(
    weight: Tensor, *, refinement_steps: int
) -> tuple[Tensor, Tensor]:
    """RouteQuant MSE scales with the resident decoder's signed-code offset."""

    if refinement_steps < 0:
        raise ValueError("quantization refinement steps must be non-negative")
    grouped = weight.float().reshape(*weight.shape[:-1], -1, GROUP_SIZE)
    scales = grouped.abs().amax(-1).div(7.0).clamp_min(1e-8)
    quantized = torch.zeros_like(grouped)
    for step in range(refinement_steps + 1):
        quantized = torch.round(grouped / scales[..., None]).clamp(-7, 7)
        if step < refinement_steps:
            numerator = (grouped * quantized).sum(-1)
            denominator = quantized.square().sum(-1).clamp_min(1e-12)
            scales = (numerator / denominator).clamp_min(1e-8)
    # PackedInt4ResidentExperts decodes nibbles by subtracting eight.  Keep
    # that established format while using RouteQuant's optimized scale.
    unsigned = (quantized.to(torch.int8) + 8).to(torch.uint8).reshape(*weight.shape[:-1], -1)
    packed = unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)
    return packed.contiguous(), scales.to(torch.bfloat16).contiguous()


@torch.no_grad()
def quantize_folded_down(
    model: ResidentDownLoRA, *, method: str, refinement_steps: int
) -> tuple[Tensor, Tensor]:
    if method not in {"standard_amax", "routequant_mse"}:
        raise ValueError("unknown folded-down quantization method")
    packed = torch.empty(RESIDENTS, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8)
    scales = torch.empty(RESIDENTS, HIDDEN, INTERMEDIATE // GROUP_SIZE, dtype=torch.bfloat16)
    for start in range(0, RESIDENTS, 2):
        stop = min(start + 2, RESIDENTS)
        value = model.folded_chunk(start, stop)
        if method == "standard_amax":
            current_packed, current_scales = quantize_groupwise_int4(value, group_size=GROUP_SIZE)
        else:
            current_packed, current_scales = quantize_groupwise_int4_mse(
                value, refinement_steps=refinement_steps
            )
        packed[start:stop].copy_(current_packed.cpu())
        scales[start:stop].copy_(current_scales.cpu())
    return packed, scales


@torch.no_grad()
def routed_from_quantized_down(
    cache: Mapping[str, Any], packed: Tensor, scales: Tensor, *, device: torch.device
) -> Tensor:
    down = dequantize_groupwise_int4(
        packed.to(device), scales.to(device), group_size=GROUP_SIZE, dtype=torch.bfloat16
    )
    activations = cache["activations"]
    local_ids = cache["local_ids"]
    weights = cache["weights"]
    rows = int(activations.shape[0])
    output = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device)
    active = local_ids >= 0
    for local_id in torch.unique(local_ids[active]).tolist():
        positions = (local_ids == int(local_id)).nonzero(as_tuple=False)
        token, slot = positions[:, 0], positions[:, 1]
        values = F.linear(activations[token, slot], down[int(local_id)])
        output[token] += values * weights[token, slot, None].to(values.dtype)
    return output


@torch.no_grad()
def evaluate_lora(
    model: ResidentDownLoRA, cache: Mapping[str, Any], next_norm: Tensor, next_router: Tensor,
    *, batch: int,
) -> dict[str, Any]:
    pieces = []
    for start in range(0, cache["base_routed"].shape[0], batch):
        indices = torch.arange(
            start, min(start + batch, cache["base_routed"].shape[0]),
            device=cache["base_routed"].device,
        )
        prediction, _correction = model.forward_cache(cache, indices)
        pieces.append(prediction)
    return metric_from_routed(torch.cat(pieces), cache, next_norm, next_router)


def objective(
    model: ResidentDownLoRA,
    cache: Mapping[str, Tensor],
    indices: Tensor,
    next_norm: Tensor,
    next_router: Tensor,
    args: argparse.Namespace,
    *,
    router_scale: float,
) -> tuple[Tensor, dict[str, float]]:
    predicted, correction = model.forward_cache(cache, indices)
    target = cache["target_routed"].index_select(0, indices).float()
    base = cache["base_routed"].index_select(0, indices).float()
    current_base = cache["current_base"].index_select(0, indices)
    attention_delta = cache["attention_delta"].index_select(0, indices)
    teacher_logits = cache["teacher_logits"].index_select(0, indices).float()
    teacher_ids = cache["teacher_ids"].index_select(0, indices).long()
    aggregate = F.smooth_l1_loss(predicted, target)
    cosine = (1.0 - F.cosine_similarity(predicted, target, dim=-1)).mean()
    preserve = correction.square().mean() / base.square().mean().detach().clamp_min(1e-6)
    logits = next_logits(
        predicted, current_base, attention_delta, next_norm, next_router
    )
    centered_predicted = logits - logits.mean(-1, keepdim=True)
    centered_teacher = teacher_logits - teacher_logits.mean(-1, keepdim=True)
    teacher_scale = centered_teacher.square().mean(-1).clamp_min(1e-4)
    jspace = ((centered_predicted - centered_teacher).square().mean(-1) / teacher_scale).mean()
    teacher_probability = torch.softmax(teacher_logits.detach(), dim=-1)
    kl = F.kl_div(torch.log_softmax(logits, dim=-1), teacher_probability, reduction="batchmean")
    valid = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    exact = exact_set_nll(logits, teacher_ids, valid=valid, k=TOPK)
    boundary = boundary_loss_per_endpoint(
        logits, teacher_logits.detach(), teacher_ids,
        margin=0.125, model_rank_start=9, teacher_rank_start=9, rank_end=32,
    ).mean()
    total = (
        args.aggregate_weight * aggregate
        + args.cosine_weight * cosine
        + args.preserve_weight * preserve
        + float(router_scale) * (
            args.jspace_weight * jspace
            + args.kl_weight * kl
            + args.exact_set_weight * exact
            + args.boundary_weight * boundary
        )
    )
    return total, {
        "total": float(total.detach()), "aggregate": float(aggregate.detach()),
        "cosine": float(cosine.detach()), "preserve": float(preserve.detach()),
        "jspace": float(jspace.detach()), "kl": float(kl.detach()),
        "exact_set": float(exact.detach()), "boundary": float(boundary.detach()),
        "router_scale": float(router_scale),
    }


def quantized_evaluations(
    model: ResidentDownLoRA,
    cache: Mapping[str, Any],
    next_norm: Tensor,
    next_router: Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, tuple[Tensor, Tensor]]]:
    metrics = {}
    tensors = {}
    for method in ("standard_amax", "routequant_mse"):
        packed, scales = quantize_folded_down(
            model, method=method, refinement_steps=args.quant_refinement_steps
        )
        routed = routed_from_quantized_down(cache, packed, scales, device=next_norm.device)
        metrics[method] = metric_from_routed(routed, cache, next_norm, next_router)
        tensors[method] = (packed, scales)
        del routed
    return metrics, tensors


def metric_lift(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean": float(candidate["request_macro_recall_at_8"] - baseline["request_macro_recall_at_8"]),
        "by_horizon": {
            str(horizon): float(candidate["by_horizon"][str(horizon)] - baseline["by_horizon"][str(horizon)])
            for horizon in range(1, 5)
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    allocation, resident_ids, core_sha = validate_allocation(args.allocation)
    resident_checkpoint, resident_state, gate_up, base_down = validate_resident_checkpoint(
        args.resident_checkpoint, resident_ids, device
    )
    train, tune, train_requests, tune_requests = load_datasets(args)
    next_norm, next_router, router_lineage = load_next_router(
        args.target_model, args.nonexpert_model, device
    )
    trainable_parameters = RESIDENTS * args.rank * (HIDDEN + INTERMEDIATE)
    source_path = Path(__file__).resolve()
    predecessor = None
    if args.predecessor_result is not None:
        predecessor = {
            "path": str(args.predecessor_result),
            "sha256": sha256_file(args.predecessor_result),
            "payload": json.loads(args.predecessor_result.read_text(encoding="utf-8")),
        }
    static = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "layer": LAYER,
        "allocation": {
            "sha256": sha256_file(args.allocation),
            "membership_sha256": allocation["resident_ids_sha256"],
            "frequency_core_sha256": core_sha,
            "exact_resident_cells": 3850,
            "l11_residents": len(resident_ids),
            "top64_recomputed_and_verified_every_layer": True,
        },
        "split": {
            "train_requests": len(train_requests), "train_rows": len(train),
            "tune_requests": len(tune_requests), "tune_rows": len(tune),
            "request_overlap": len(train_requests & tune_requests),
        },
        "lineage": {
            **router_lineage,
            "allocation_sha256": sha256_file(args.allocation),
            "resident_checkpoint_sha256": sha256_file(args.resident_checkpoint),
            "partition_manifest_sha256": sha256_file(args.partition_manifest),
            "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        },
        "architecture": {
            "trainable_form": "per-resident rank-r correction A_e @ B_e on frozen SwiGLU activation",
            "rank": args.rank, "lora_alpha": args.lora_alpha,
            "training_only_parameters": trainable_parameters,
            "gate_up_frozen": True, "resident_namespace_frozen": True,
            "native_top8_ids_and_weights_frozen": True,
            "fold_target": "existing resident down_packed/down_scales",
            "standard_amax_quant_control": True,
            "routequant_mse_scale_candidate": True,
        },
        "deployment_resources": {
            "resident_cells_before": 3850, "resident_cells_after": 3850,
            "l11_resident_tensor_bytes_before": L11_RESIDENT_BYTES,
            "l11_resident_tensor_bytes_after": L11_RESIDENT_BYTES,
            "bundle_resident_tensor_bytes_before": FULL_BUNDLE_RESIDENT_BYTES,
            "bundle_resident_tensor_bytes_after": FULL_BUNDLE_RESIDENT_BYTES,
            "extra_runtime_parameters": 0, "extra_runtime_bytes": 0,
            "extra_expert_calls": 0, "extra_mac": 0,
            "packed_tensor_shapes_changed": False,
            "packed_traffic_changed": False,
        },
        "predecessor": predecessor,
        "capture_used": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "development_opened": False,
        "sealed_test_opened": False,
    }
    emit(args.output / "STATIC_PREFLIGHT.json", static)
    print(json.dumps({"event": "static_preflight", **static}, sort_keys=True), flush=True)
    if args.preflight_only:
        emit(args.output / "STAGE_RESULT.json", {
            **static, "training_started": False, "stop_reason": "preflight_only",
            "wall_seconds": time.monotonic() - started,
        })
        return

    train_cache = materialize(
        train, gate_up=gate_up, down=base_down,
        expert_to_resident=resident_state["expert_to_resident"].to(device),
        next_norm=next_norm, next_router=next_router, args=args, keep_identity=False,
    )
    tune_cache = materialize(
        tune, gate_up=gate_up, down=base_down,
        expert_to_resident=resident_state["expert_to_resident"].to(device),
        next_norm=next_norm, next_router=next_router, args=args, keep_identity=True,
    )
    del gate_up
    baseline = metric_from_routed(tune_cache["base_routed"], tune_cache, next_norm, next_router)
    teacher_replay = metric_from_routed(
        tune_cache["target_routed"], tune_cache, next_norm, next_router
    )
    available_headroom = (
        teacher_replay["request_macro_recall_at_8"] - baseline["request_macro_recall_at_8"]
    )
    if tune_cache["parity"]["stable_top8_recall"] < 0.98 or available_headroom < MINIMUM_RECALL_LIFT:
        result = {
            **static, "training_started": False,
            "baseline": baseline, "teacher_replay": teacher_replay,
            "teacher_reconstruction": tune_cache["parity"],
            "available_recall_headroom": available_headroom,
            "stop_reason": "teacher_parity_or_headroom_gate_failed",
            "wall_seconds": time.monotonic() - started,
        }
        emit(args.output / "STAGE_RESULT.json", result)
        raise SystemExit(2)

    model = ResidentDownLoRA(
        base_down, rank=args.rank, alpha=args.lora_alpha, seed=args.seed
    ).to(device)
    del base_down
    epoch0 = evaluate_lora(
        model, tune_cache, next_norm, next_router, batch=args.endpoint_batch
    )
    pretrain = {
        **static,
        "baseline": baseline,
        "epoch0": epoch0,
        "teacher_replay": teacher_replay,
        "teacher_reconstruction": tune_cache["parity"],
        "available_recall_headroom": available_headroom,
        "gate_fraction_of_available_headroom": MINIMUM_RECALL_LIFT / available_headroom,
        "training_authorized": True,
    }
    emit(args.output / "PREFLIGHT.json", pretrain)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, Any]] = []
    best_score = float(baseline["request_macro_recall_at_8"])
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_method = "standard_amax"
    best_quant_metrics: dict[str, Any] | None = None
    stale = 0
    train_rows = int(train_cache["base_routed"].shape[0])
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(args.seed + epoch)
        order = torch.randperm(train_rows, generator=generator)
        sums: dict[str, float] = defaultdict(float)
        steps = 0
        epoch_started = time.monotonic()
        router_scale = min(
            1.0, max(0.0, (epoch - args.router_warmup_epochs) / max(1, args.router_warmup_epochs))
        ) if args.router_warmup_epochs else 1.0
        for start in range(0, train_rows, args.endpoint_batch):
            indices = order[start:start + args.endpoint_batch].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = objective(
                model, train_cache, indices, next_norm, next_router, args,
                router_scale=router_scale,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            for name, value in parts.items():
                sums[name] += value
            steps += 1
        model.eval()
        bf16 = evaluate_lora(
            model, tune_cache, next_norm, next_router, batch=args.endpoint_batch
        )
        quant_metrics, _quant_tensors = quantized_evaluations(
            model, tune_cache, next_norm, next_router, args
        )
        selected_method = max(
            quant_metrics,
            key=lambda name: quant_metrics[name]["request_macro_recall_at_8"],
        )
        score = float(quant_metrics[selected_method]["request_macro_recall_at_8"])
        record = {
            "epoch": epoch, "seconds": time.monotonic() - epoch_started,
            "train": {name: value / steps for name, value in sums.items()},
            "bf16": bf16, "post_int4": quant_metrics,
            "selected_quant_method": selected_method,
            "selected_post_int4_lift": metric_lift(quant_metrics[selected_method], baseline),
        }
        history.append(record)
        emit(args.output / "PROGRESS.json", {"schema": SCHEMA, "history": history})
        print(json.dumps({"event": "epoch", **record}, sort_keys=True), flush=True)
        if score > best_score + args.minimum_improvement:
            best_score = score
            best_method = selected_method
            best_quant_metrics = quant_metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    model.load_state_dict({name: value.to(device) for name, value in best_state.items()})
    model.eval()
    final_bf16 = evaluate_lora(
        model, tune_cache, next_norm, next_router, batch=args.endpoint_batch
    )
    final_quant, final_tensors = quantized_evaluations(
        model, tune_cache, next_norm, next_router, args
    )
    best_method = max(
        final_quant,
        key=lambda name: final_quant[name]["request_macro_recall_at_8"],
    )
    best_quant_metrics = final_quant
    chosen = final_quant[best_method]
    lift = metric_lift(chosen, baseline)
    promotion_passed = (
        lift["mean"] >= MINIMUM_RECALL_LIFT
        and min(lift["by_horizon"].values()) >= -MAXIMUM_HORIZON_REGRESSION
    )
    down_packed, down_scales = final_tensors[best_method]
    export_state = {
        "down_scales": down_scales.cpu().contiguous(),
        "gate_up_packed": resident_state["gate_up_packed"],
        "gate_up_scales": resident_state["gate_up_scales"],
        "down_packed": down_packed.cpu().contiguous(),
        "resident_ids": resident_state["resident_ids"],
        "expert_to_resident": resident_state["expert_to_resident"],
    }
    exported_bytes = sum(
        export_state[name].numel() * export_state[name].element_size()
        for name in ("gate_up_packed", "gate_up_scales", "down_packed", "down_scales")
    )
    if exported_bytes != L11_RESIDENT_BYTES:
        raise RuntimeError("folded export changed the resident tensor footprint")
    checkpoint_out = {
        **{key: value for key, value in resident_checkpoint.items() if key != "model_state_dict"},
        "source_commit": args.source_commit,
        "mode": "resident_int4_only",
        "diagnostic_only": True,
        "closed_loop_authorized": False,
        "training_started": True,
        "resident_down_distillation_schema": SCHEMA,
        "resident_down_distillation_source_sha256": sha256_file(source_path),
        "resident_down_distillation_quant_method": best_method,
        "model_state_dict": export_state,
    }
    checkpoint_path = args.output / "shadow_resident_int4_down_distilled_layer_11.pt"
    torch.save(checkpoint_out, checkpoint_path)
    selected_tensor_hashes = {name: tensor_sha256(value) for name, value in export_state.items()}
    gate_unchanged = all(
        torch.equal(export_state[name], resident_state[name])
        for name in ("gate_up_packed", "gate_up_scales", "resident_ids", "expert_to_resident")
    )
    result = {
        **pretrain,
        "training_started": True,
        "epochs_completed": len(history),
        "early_stopped": len(history) < args.epochs,
        "history": history,
        "best_bf16": final_bf16,
        "post_int4": final_quant,
        "selected_quant_method": best_method,
        "selected_post_int4": chosen,
        "post_int4_lift": lift,
        "minimum_required_mean_lift": MINIMUM_RECALL_LIFT,
        "maximum_per_horizon_regression": MAXIMUM_HORIZON_REGRESSION,
        "promotion_gate_passed": promotion_passed,
        "gate_up_and_namespace_bitwise_unchanged": gate_unchanged,
        "exported_resident_tensor_bytes": exported_bytes,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_tensor_sha256": selected_tensor_hashes,
        "wall_seconds": time.monotonic() - started,
    }
    emit(args.output / "STAGE_RESULT.json", result)
    manifest = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "source_sha256": sha256_file(source_path),
        "allocation_sha256": EXPECTED_ALLOCATION_SHA256,
        "resident_checkpoint_sha256": sha256_file(args.resident_checkpoint),
        "output_checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_quant_method": best_method,
        "promotion_gate_passed": promotion_passed,
        "resident_cells": 3850,
        "resident_tensor_bytes": FULL_BUNDLE_RESIDENT_BYTES,
        "runtime_parameter_delta": 0,
        "runtime_byte_delta": 0,
        "expert_call_delta": 0,
        "mac_delta": 0,
        "capture_used": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "development_opened": False,
        "sealed_test_opened": False,
    }
    emit(args.output / "run_manifest.json", manifest)
    sums = []
    for path in sorted(value for value in args.output.iterdir() if value.is_file() and value.name != "SHA256SUMS"):
        sums.append(f"{sha256_file(path)}  {path.name}")
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "final", "promotion_gate_passed": promotion_passed,
        "selected_quant_method": best_method, "post_int4_lift": lift,
        "output": str(args.output),
    }, sort_keys=True), flush=True)
    if not promotion_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
