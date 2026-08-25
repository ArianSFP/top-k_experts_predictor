#!/usr/bin/env python3
"""Convex L11 screen for exact Core64 plus activation-J width-128 tails.

The activation-J gate/up rows are frozen.  For each of the 192 non-core
experts, train-only request-balanced normal equations fit the 128-to-2048
down projection to the exact full-expert output.  A small, predeclared ridge
grid is selected on the request-disjoint tune split before one post-INT4
evaluation.  Serving geometry and the paired-v4 evaluator are unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.shadow_checkpoint import (  # noqa: E402
    IndexedCheckpoint,
    load_target_layer_experts,
    sha256_file,
)
from harp_rtt.shadow_expert import target_selected_expert_outputs  # noqa: E402
from runpod import screen_hybridcore64_learned_jtail_l11 as paired  # noqa: E402
from runpod.train_harp_delta_v3 import (  # noqa: E402
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
    loader,
)
from runpod.train_shadow_experts_local import batch_tensors, load_split  # noqa: E402


SCHEMA = "hybridcore64_convex_jtail_l11_screen_v1"
RIDGE_GRID = (1.0e-4, 1.0e-3, 1.0e-2)
MINIMUM_POST_INT4_LIFT = 0.02
MAXIMUM_HORIZON_REGRESSION = 0.0


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
    parser.add_argument("--layer", type=int, default=11, choices=(11,))
    parser.add_argument("--microbatch", type=int, default=8, choices=(8,))
    parser.add_argument("--latency-repeats", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def request_macro_no_regression(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> bool:
    return all(
        float(candidate["by_horizon"][h]) >= float(control["by_horizon"][h])
        for h in control["by_horizon"]
    )


@torch.no_grad()
def accumulate_normal_equations(
    dataset: Any,
    model: paired.LearnedJTail,
    target_gate_up: Tensor,
    target_down: Tensor,
    *,
    layer: int,
    microbatch: int,
    request_rows: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Accumulate weighted X'X and X'Y for all non-core experts.

    Each captured row receives weight 1/request_rows.  Squared native route
    weight aligns the independent regression with routed-output squared error.
    """

    tails = int(model.tail_ids.numel())
    xtx = torch.zeros(
        tails, paired.TAIL_WIDTH, paired.TAIL_WIDTH,
        dtype=torch.float32, device=device,
    )
    xty = torch.zeros(
        tails, paired.TAIL_WIDTH, paired.HIDDEN,
        dtype=torch.float32, device=device,
    )
    counts = torch.zeros(tails, dtype=torch.int64, device=device)
    effective_weight = torch.zeros(tails, dtype=torch.float64, device=device)
    valid_endpoints = 0
    tail_occurrences = 0
    batches = 0
    started = time.monotonic()

    for host in loader(
        dataset, batch=microbatch, shuffle=False, seed=0, workers=0, device=device
    ):
        inputs, _routed, ids, route_weights, valid, _requests, _states = batch_tensors(
            host, layer=layer, device=device
        )
        exact_values = target_selected_expert_outputs(
            inputs, ids, target_gate_up, target_down
        )
        hidden = inputs.reshape(-1, paired.HIDDEN)
        flat_ids = ids.reshape(hidden.shape[0], -1).long()
        flat_weights = route_weights.reshape_as(flat_ids).float()
        flat_valid = valid.reshape(-1)
        local = model.tail_index[flat_ids]
        active = (local >= 0) & flat_valid[:, None]
        positions = active.nonzero(as_tuple=False)
        if positions.numel() == 0:
            continue
        token, slot = positions[:, 0], positions[:, 1]
        rows = local[token, slot]
        selected_gate_up = model.gate_up.index_select(0, rows)
        projected = torch.bmm(
            selected_gate_up, hidden[token, :, None]
        ).squeeze(-1)
        gate, up = projected.chunk(2, dim=-1)
        # Match deployed paired-v4 arithmetic exactly: SiLU and the
        # elementwise product round in BF16 before the regression sees X.
        activation = (F.silu(gate) * up).float().contiguous()
        target = exact_values.reshape(
            hidden.shape[0], flat_ids.shape[1], paired.HIDDEN
        )[token, slot].float()
        sample_weight = (
            flat_weights[token, slot].square() / float(request_rows)
        ).clamp_min(1.0e-12)

        for expert in torch.unique(rows).tolist():
            chosen = rows == int(expert)
            x = activation[chosen]
            y = target[chosen]
            root = sample_weight[chosen].sqrt()[:, None]
            xw = x * root
            yw = y * root
            xtx[int(expert)].add_(xw.T @ xw)
            xty[int(expert)].add_(xw.T @ yw)
            counts[int(expert)] += int(chosen.sum())
            effective_weight[int(expert)] += sample_weight[chosen].double().sum()

        valid_endpoints += int(valid.sum())
        tail_occurrences += int(active.sum())
        batches += 1
        if batches % 64 == 0:
            print(json.dumps({
                "event": "normal_equations_progress",
                "batches": batches,
                "valid_endpoints": valid_endpoints,
                "tail_occurrences": tail_occurrences,
                "seconds": time.monotonic() - started,
            }, sort_keys=True), flush=True)

    if int((counts == 0).sum()) != 0:
        raise RuntimeError("one or more non-core experts lack train occurrences")
    asymmetry = float((xtx - xtx.transpose(-1, -2)).abs().max())
    xtx = 0.5 * (xtx + xtx.transpose(-1, -2))
    facts = {
        "request_balance": f"one_over_{request_rows}_rows_per_request",
        "route_weighting": "squared_native_execution_weight",
        "batches": batches,
        "valid_endpoints": valid_endpoints,
        "tail_occurrences": tail_occurrences,
        "expert_occurrence_min": int(counts.min()),
        "expert_occurrence_median": int(counts.float().median()),
        "expert_occurrence_max": int(counts.max()),
        "effective_weight_min": float(effective_weight.min()),
        "effective_weight_max": float(effective_weight.max()),
        "xtx_max_asymmetry_before_symmetrization": asymmetry,
        "seconds": time.monotonic() - started,
    }
    return xtx, xty, facts


@torch.no_grad()
def solve_down(
    model: paired.LearnedJTail,
    xtx: Tensor,
    xty: Tensor,
    ridge_relative: float,
    prior_down: Tensor,
) -> dict[str, Any]:
    """Solve ridge toward the frozen activation-J native-column prior."""

    diagonal_mean = xtx.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1.0e-12)
    ridge = diagonal_mean * float(ridge_relative)
    identity = torch.eye(paired.TAIL_WIDTH, dtype=xtx.dtype, device=xtx.device)
    system = xtx + ridge[:, None, None] * identity[None]
    prior = prior_down.float().transpose(-1, -2).contiguous()
    rhs = xty + ridge[:, None, None] * prior
    solution = torch.linalg.solve(system, rhs)
    if not bool(torch.isfinite(solution).all()):
        raise FloatingPointError("ridge solution is not finite")
    model.down.copy_(solution.transpose(-1, -2).to(model.down.dtype))
    residual = system @ solution - rhs
    return {
        "ridge_relative": float(ridge_relative),
        "ridge_absolute_min": float(ridge.min()),
        "ridge_absolute_median": float(ridge.median()),
        "ridge_absolute_max": float(ridge.max()),
        "normal_equation_relative_residual": float(
            residual.norm() / rhs.norm().clamp_min(1.0e-12)
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    validate_partition(args.partition_manifest, "b2_reuse_4096")
    reuse = validate_reuse_split(args.reuse_split_manifest)
    train_requests = set(reuse["inner_split"]["training_requests"])
    tune_requests = set(reuse["inner_split"]["tuning_requests"])
    if train_requests & tune_requests:
        raise PermissionError("train/tune request overlap")
    namespace = SimpleNamespace(
        data_profile="b2_reuse_4096", layer=args.layer,
        next_router_agreement=True,
        train_index=args.index, train_corpus=args.corpus,
        train_companion=args.companion,
        tune_index=args.index, tune_corpus=args.corpus,
        tune_companion=args.companion,
    )
    train, train_groups = load_split(
        namespace, "train", selected_requests=train_requests
    )
    tune, tune_groups = load_split(
        namespace, "tune", selected_requests=tune_requests
    )
    if len(train_groups) != 224 or len(tune_groups) != 32:
        raise ValueError("frozen request split cardinality changed")
    if len(train) != 3584 or len(tune) != 512:
        raise ValueError("frozen row cardinality changed")
    if len(train) % len(train_groups):
        raise ValueError("train rows are not request balanced")
    request_rows = len(train) // len(train_groups)
    if request_rows != 16:
        raise ValueError("expected exactly 16 train rows per request")

    allocation = json.loads(args.allocation.read_text(encoding="utf-8"))
    core, residents = paired.frequency_core(allocation, args.layer)
    allocation_sha256 = sha256_file(args.allocation)
    core_sha256 = hashlib.sha256(
        json.dumps(core, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if allocation_sha256 != paired.EXPECTED_ALLOCATION_SHA256:
        raise ValueError("compliant allocation lineage changed")
    if core_sha256 != paired.EXPECTED_CORE_SHA256:
        raise ValueError("mandatory L11 frequency Core64 changed")

    selectors = torch.load(args.selectors, map_location="cpu", weights_only=False)
    selected = selectors["selected_neurons"]["activation_j"]
    initializer_scale = float(selectors["scales"]["activation_j"])
    checkpoint = IndexedCheckpoint(args.target_model)
    gate_up, down = load_target_layer_experts(
        checkpoint, args.layer, device=device, dtype=torch.bfloat16
    )
    prefix = f"model.language_model.layers.{args.layer + 1}."
    names = (
        prefix + "post_attention_layernorm.weight",
        prefix + "mlp.gate.weight",
    )
    tensors = checkpoint.tensors(names)
    next_norm = tensors[names[0]].to(device=device, dtype=torch.bfloat16)
    next_router = tensors[names[1]].to(device=device, dtype=torch.bfloat16)
    model = paired.LearnedJTail(
        core, selected, gate_up, down, initializer_scale
    ).to(device)
    model.gate_up.requires_grad_(False)
    prior_down = model.down.detach().clone()

    packed_bytes = (
        paired.CORE_SIZE * paired.TOTAL_LAYERS * paired.FULL_CELL_BYTES
        + paired.DEPLOY_TAIL_CELLS * paired.TAIL_CELL_BYTES
    )
    if packed_bytes != paired.PACKED_CAP:
        raise ValueError("static packed-byte identity changed")

    first_host = next(iter(loader(
        tune, batch=args.microbatch, shuffle=False, seed=0,
        workers=0, device=device,
    )))
    latency = paired.benchmark(
        first_host, model, gate_up, down, residents, core,
        args.layer, device, args.latency_repeats,
    )
    epoch0 = paired.evaluate(
        tune, model, gate_up, down, next_norm, next_router, residents,
        layer=args.layer, microbatch=args.microbatch, device=device,
    )
    calls = epoch0["calls"]
    full_mac = 3 * paired.HIDDEN * paired.NATIVE_WIDTH
    tail_mac = 3 * paired.HIDDEN * paired.TAIL_WIDTH
    hybrid_mac = (
        calls["core_per_endpoint"] * full_mac
        + calls["tail_per_endpoint"] * tail_mac
    )
    compliant_mac = calls["compliant_per_endpoint"] * full_mac
    traffic = {
        "compliant_bytes_per_endpoint": (
            calls["compliant_per_endpoint"] * paired.FULL_CELL_BYTES
        ),
        "hybrid_bytes_per_endpoint": (
            calls["core_per_endpoint"] * paired.FULL_CELL_BYTES
            + calls["tail_per_endpoint"] * paired.TAIL_CELL_BYTES
        ),
    }
    references_ok = (
        abs(epoch0["compliant"]["request_macro_recall_at_8"]
            - paired.EXPECTED_CONTROL_RECALL_M8) <= 5.0e-6
        and abs(epoch0["exact"]["request_macro_recall_at_8"]
                - paired.EXPECTED_EXACT_RECALL_M8) <= 5.0e-6
    )
    efficiency_ok = (
        packed_bytes == paired.PACKED_CAP
        and hybrid_mac <= compliant_mac
        and traffic["hybrid_bytes_per_endpoint"]
        <= traffic["compliant_bytes_per_endpoint"]
        and latency["vector_tail_latency_ratio"] <= 1.0
    )
    preflight = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "layer": args.layer,
        "train_requests": len(train_groups), "train_rows": len(train),
        "tune_requests": len(tune_groups), "tune_rows": len(tune),
        "request_disjoint": not bool(train_groups & tune_groups),
        "request_rows": request_rows,
        "ridge_grid_predeclared": list(RIDGE_GRID),
        "core_ids": core, "core_ids_sha256": core_sha256,
        "core_count": len(core), "tail_count": 192,
        "compliant_resident_count": len(residents),
        "packed_bytes": packed_bytes, "packed_byte_cap": paired.PACKED_CAP,
        "hybrid_mac_per_endpoint": hybrid_mac,
        "compliant_mac_per_endpoint": compliant_mac,
        "traffic": traffic, "latency": latency,
        "epoch0": epoch0,
        "reference_metrics_reproduced": references_ok,
        "efficiency_nonregression_verified": efficiency_ok,
        "training_authorized": references_ok and efficiency_ok,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "lineage": {
            "allocation_sha256": allocation_sha256,
            "selectors_sha256": sha256_file(args.selectors),
            "partition_manifest_sha256": sha256_file(args.partition_manifest),
            "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
            "target_checkpoint_index_sha256": checkpoint.index_sha256,
            "paired_driver_sha256": sha256_file(Path(paired.__file__)),
        },
    }
    paired.emit(args.output / "PREFLIGHT.json", preflight)
    print(json.dumps({"event": "preflight", **preflight}, sort_keys=True), flush=True)
    if args.preflight_only or not preflight["training_authorized"]:
        paired.emit(args.output / "STAGE_RESULT.json", {
            **preflight,
            "fit_started": False,
            "stop_reason": (
                "preflight_only" if args.preflight_only
                else "efficiency_or_reference_gate_failed"
            ),
        })
        raise SystemExit(0 if args.preflight_only else 2)

    xtx, xty, normal_facts = accumulate_normal_equations(
        train, model, gate_up, down, layer=args.layer,
        microbatch=args.microbatch, request_rows=request_rows, device=device,
    )
    paired.emit(args.output / "NORMAL_EQUATIONS.json", {
        "schema": SCHEMA, **normal_facts,
    })
    candidates: list[dict[str, Any]] = []
    best_recall = -1.0
    best_down: Tensor | None = None
    best_ridge: float | None = None
    best_eligible = False
    for ridge in RIDGE_GRID:
        solve_facts = solve_down(model, xtx, xty, ridge, prior_down)
        bf16_metrics = paired.evaluate(
            tune, model, gate_up, down, next_norm, next_router, residents,
            layer=args.layer, microbatch=args.microbatch, device=device,
        )
        int4_metrics = paired.evaluate(
            tune, model, gate_up, down, next_norm, next_router, residents,
            layer=args.layer, microbatch=args.microbatch, device=device,
            quantized=True,
        )
        recall = float(
            int4_metrics["hybrid"]["request_macro_recall_at_8"]
        )
        no_regression = request_macro_no_regression(
            int4_metrics["hybrid"], int4_metrics["compliant"]
        )
        record = {
            **solve_facts,
            "bf16_metrics": bf16_metrics,
            "post_int4_metrics": int4_metrics,
            "bf16_lift_vs_compliant": (
                float(bf16_metrics["hybrid"]["request_macro_recall_at_8"])
                - float(bf16_metrics["compliant"]["request_macro_recall_at_8"])
            ),
            "post_int4_lift_vs_compliant": (
                recall - float(
                    int4_metrics["compliant"]["request_macro_recall_at_8"]
                )
            ),
            "post_int4_no_horizon_regression": no_regression,
        }
        candidates.append(record)
        paired.emit(args.output / "RIDGE_PROGRESS.json", {
            "schema": SCHEMA, "candidates": candidates,
        })
        print(json.dumps({"event": "ridge", **record}, sort_keys=True), flush=True)
        # Prefer a no-regression deployed candidate. If none exists, retain
        # the highest deployed recall as the quantitative rejection point.
        if (
            (no_regression and not best_eligible)
            or (no_regression == best_eligible and recall > best_recall)
        ):
            best_recall = recall
            best_ridge = float(ridge)
            best_down = model.down.detach().cpu().clone()
            best_eligible = no_regression

    if best_down is None or best_ridge is None:
        raise RuntimeError("ridge grid produced no candidate")
    with torch.no_grad():
        model.down.copy_(best_down.to(device=device, dtype=model.down.dtype))
    best_bf16 = paired.evaluate(
        tune, model, gate_up, down, next_norm, next_router, residents,
        layer=args.layer, microbatch=args.microbatch, device=device,
    )
    post_int4 = paired.evaluate(
        tune, model, gate_up, down, next_norm, next_router, residents,
        layer=args.layer, microbatch=args.microbatch, device=device,
        quantized=True,
    )
    control = post_int4["compliant"]
    candidate = post_int4["hybrid"]
    lift = (
        float(candidate["request_macro_recall_at_8"])
        - float(control["request_macro_recall_at_8"])
    )
    horizon_lifts = {
        h: float(candidate["by_horizon"][h]) - float(control["by_horizon"][h])
        for h in control["by_horizon"]
    }
    passed = (
        lift >= MINIMUM_POST_INT4_LIFT
        and min(horizon_lifts.values()) >= -MAXIMUM_HORIZON_REGRESSION
        and efficiency_ok
    )
    checkpoint_out = {
        "schema": SCHEMA, "layer": args.layer,
        "core_ids": core, "tail_ids": model.tail_ids.cpu(),
        "gate_up": model.gate_up.detach().cpu(),
        "down": model.down.detach().cpu(),
        "initializer": "activation_j",
        "fit": "request_balanced_weighted_ridge_down_projection",
        "selected_ridge_relative": best_ridge,
    }
    torch.save(checkpoint_out, args.output / "best_tail_bf16.pt")
    result = {
        **preflight, "fit_started": True,
        "normal_equations": normal_facts,
        "ridge_candidates": candidates,
        "selected_ridge_relative": best_ridge,
        "best_bf16": best_bf16,
        "post_quant_int4": post_int4,
        "post_int4_lift_vs_compliant": lift,
        "post_int4_horizon_lifts": horizon_lifts,
        "promotion_gate_passed": passed,
        "minimum_post_int4_lift": MINIMUM_POST_INT4_LIFT,
        "maximum_horizon_regression": MAXIMUM_HORIZON_REGRESSION,
    }
    paired.emit(args.output / "STAGE_RESULT.json", result)
    print(json.dumps({
        "event": "final", "selected_ridge_relative": best_ridge,
        "post_int4_lift_vs_compliant": lift,
        "post_int4_horizon_lifts": horizon_lifts,
        "promotion_gate_passed": passed,
    }, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
