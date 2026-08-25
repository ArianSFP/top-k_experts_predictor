#!/usr/bin/env python3
"""Evaluate two same-footprint resident plans on a disjoint local split.

Only one routed layer is materialized.  Both plans use the production packed
INT4 resident implementation and the same exact BF16 target tensors, while the
captured factual next-router state supplies a deterministic local Recall@8
gate.  This is a tuning-split architecture screen, not formal validation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

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
from harp_rtt.shadow_expert import PackedInt4ResidentExperts  # noqa: E402
from runpod.train_harp_delta_v3 import (  # noqa: E402
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
    loader,
)
from runpod.train_shadow_experts_local import (  # noqa: E402
    batch_tensors,
    load_split,
    next_router_agreement_loss,
    write_checksums,
    write_json_exclusive,
    write_rows,
)


SCHEMA = "harp_resident_plan_pair_local_gate_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--control-plan", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=range(39), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-recall-lift", type=float, default=0.002)
    parser.add_argument("--maximum-horizon-regression", type=float, default=0.002)
    return parser.parse_args()


def _load_plan(path: Path) -> tuple[dict[str, Any], list[list[int]]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise TypeError("resident plan must be a JSON object")
    raw = plan.get("resident_expert_ids_by_layer")
    if not isinstance(raw, list) or len(raw) != 40:
        raise ValueError("resident plan must cover exactly 40 layers")
    ids = [[int(value) for value in row] for row in raw]
    if sum(map(len, ids)) != 3850:
        raise ValueError("resident plan must contain exactly 3,850 cells")
    return plan, ids


def _validate_pair(
    control_plan: dict[str, Any],
    control: list[list[int]],
    candidate_plan: dict[str, Any],
    candidate: list[list[int]],
    *,
    layer: int,
) -> dict[str, Any]:
    raw_counts = control_plan.get("expert_counts")
    counts = torch.as_tensor(raw_counts, dtype=torch.int64)
    if counts.shape != (40, 256):
        raise ValueError("control plan expert-count geometry changed")
    if candidate_plan.get("expert_counts") != raw_counts:
        raise ValueError("candidate and control plans do not share train statistics")
    core, core_hash = frequency_core_ids(counts, core_size=64)
    validate_core_inclusion(control, core)
    validate_core_inclusion(candidate, core)
    if any(control[index] != candidate[index] for index in range(40) if index != layer):
        raise ValueError("pair differs outside the declared layer")
    if len(control[layer]) != len(candidate[layer]):
        raise ValueError("pair changes the declared layer footprint")
    cap = int(control_plan.get("resident_hit_count_cap", 3_245_387))
    hits: dict[str, int] = {}
    for name, values in (("control", control), ("candidate", candidate)):
        hits[name] = sum(
            int(counts[index, expert])
            for index, layer_ids in enumerate(values)
            for expert in layer_ids
        )
        if hits[name] > cap:
            raise ValueError(f"{name} plan exceeds train resident-hit cap")
    return {
        "frequency_core_sha256": core_hash,
        "control_membership_sha256": _canonical_ids_sha256(control),
        "candidate_membership_sha256": _canonical_ids_sha256(candidate),
        "control_train_resident_hits": hits["control"],
        "candidate_train_resident_hits": hits["candidate"],
        "resident_hit_count_cap": cap,
        "layer_resident_cells": len(control[layer]),
        "total_resident_cells": 3850,
    }


def _overlap(logits: Tensor, teacher_ids: Tensor) -> Tensor:
    predicted = stable_topk(logits, k=8)
    return (teacher_ids[..., None] == predicted[..., None, :]).any(-1).float().mean(-1)


def _request_macro(
    values: dict[tuple[str, int], list[float]],
) -> tuple[float, dict[str, float], dict[tuple[str, int], float]]:
    per_request_horizon = {
        key: sum(rows) / len(rows) for key, rows in values.items() if rows
    }
    if not per_request_horizon:
        raise ValueError("evaluation contains no valid rows")
    by_horizon: dict[int, list[float]] = defaultdict(list)
    for (_request, horizon), value in per_request_horizon.items():
        by_horizon[horizon].append(value)
    horizons = {
        str(horizon): sum(rows) / len(rows)
        for horizon, rows in sorted(by_horizon.items())
    }
    return (
        sum(per_request_horizon.values()) / len(per_request_horizon),
        horizons,
        per_request_horizon,
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(args.source_commit) != 40:
        raise ValueError("source commit must be a full SHA")
    partition = validate_partition(args.partition_manifest, "b2_reuse_4096")
    reuse = validate_reuse_split(args.reuse_split_manifest)
    selected = set(reuse["inner_split"]["tuning_requests"])
    namespace = SimpleNamespace(
        data_profile="b2_reuse_4096",
        layer=args.layer,
        next_router_agreement=True,
        tune_index=args.index,
        tune_corpus=args.corpus,
        tune_companion=args.companion,
    )
    dataset, groups = load_split(namespace, "tune", selected_requests=selected)
    control_plan, control_ids = _load_plan(args.control_plan)
    candidate_plan, candidate_ids = _load_plan(args.candidate_plan)
    pair = _validate_pair(
        control_plan, control_ids, candidate_plan, candidate_ids, layer=args.layer
    )
    device = torch.device(args.device)
    checkpoint = IndexedCheckpoint(args.target_model)
    gate_up, down = load_target_layer_experts(
        checkpoint, args.layer, device=device, dtype=torch.bfloat16
    )
    models = {
        "control": PackedInt4ResidentExperts.from_target(
            torch.tensor(control_ids[args.layer], device=device),
            None,
            gate_up,
            down,
            exact_k=8,
            group_size=64,
        ),
        "candidate": PackedInt4ResidentExperts.from_target(
            torch.tensor(candidate_ids[args.layer], device=device),
            None,
            gate_up,
            down,
            exact_k=8,
            group_size=64,
        ),
    }
    del gate_up, down
    for model in models.values():
        model.requires_grad_(False)
        model.eval()
    prefix = f"model.language_model.layers.{args.layer + 1}."
    names = (prefix + "post_attention_layernorm.weight", prefix + "mlp.gate.weight")
    target = checkpoint.tensors(names)
    next_norm = target[names[0]].to(device=device, dtype=torch.bfloat16)
    next_router = target[names[1]].to(device=device, dtype=torch.bfloat16)

    recalls: dict[str, dict[tuple[str, int], list[float]]] = {
        name: defaultdict(list) for name in ("teacher", *models)
    }
    routed_mse: dict[str, list[float]] = {name: [] for name in models}
    logit_mse: dict[str, list[float]] = {name: [] for name in models}
    resident_calls = {name: 0 for name in models}
    valid_endpoints = 0
    rows_out: list[dict[str, Any]] = []
    for host in loader(
        dataset,
        batch=args.microbatch,
        shuffle=False,
        seed=0,
        workers=args.num_workers,
        device=device,
    ):
        inputs, routed, ids, weights, valid, requests, states = batch_tensors(
            host, layer=args.layer, device=device
        )
        _loss, _parts, teacher_logits = next_router_agreement_loss(
            routed, states, valid, norm_weight=next_norm, router_weight=next_router
        )
        batch_recalls = {
            "teacher": _overlap(teacher_logits, states["next_selected_expert_ids"].long())
        }
        batch_routed_mse: dict[str, Tensor] = {}
        batch_logit_mse: dict[str, Tensor] = {}
        for name, model in models.items():
            components = model.forward_components(inputs, ids, weights)
            predicted_routed = components.resident_output
            _loss, _parts, logits = next_router_agreement_loss(
                predicted_routed,
                states,
                valid,
                norm_weight=next_norm,
                router_weight=next_router,
            )
            batch_recalls[name] = _overlap(
                logits, states["next_selected_expert_ids"].long()
            )
            batch_routed_mse[name] = (
                predicted_routed.float() - routed.float()
            ).square().mean(-1)
            batch_logit_mse[name] = (
                logits.float() - teacher_logits.float()
            ).square().mean(-1)
            resident_calls[name] += int(((~components.missing_mask) & valid[..., None]).sum())
        valid_endpoints += int(valid.sum())
        for row, request in enumerate(requests):
            for horizon_index in range(valid.shape[1]):
                if not bool(valid[row, horizon_index]):
                    continue
                horizon = horizon_index + 1
                key = (request, horizon)
                payload: dict[str, Any] = {
                    "request_id": request,
                    "horizon": horizon,
                    "layer": args.layer,
                }
                for name, values in batch_recalls.items():
                    value = float(values[row, horizon_index])
                    recalls[name][key].append(value)
                    payload[f"{name}_recall_at_8"] = value
                for name in models:
                    routed_value = float(batch_routed_mse[name][row, horizon_index])
                    logit_value = float(batch_logit_mse[name][row, horizon_index])
                    routed_mse[name].append(routed_value)
                    logit_mse[name].append(logit_value)
                    payload[f"{name}_routed_mse"] = routed_value
                    payload[f"{name}_next_router_logit_mse"] = logit_value
                rows_out.append(payload)

    metrics: dict[str, Any] = {}
    request_values: dict[str, dict[tuple[str, int], float]] = {}
    for name, values in recalls.items():
        mean, horizons, requests = _request_macro(values)
        request_values[name] = requests
        metrics[name] = {
            "request_macro_recall_at_8": mean,
            "by_horizon": horizons,
        }
        if name in models:
            metrics[name].update(
                {
                    "endpoint_mean_routed_mse": sum(routed_mse[name]) / len(routed_mse[name]),
                    "endpoint_mean_next_router_logit_mse": sum(logit_mse[name]) / len(logit_mse[name]),
                    "resident_expert_calls": resident_calls[name],
                    "resident_call_fraction": resident_calls[name] / (8 * valid_endpoints),
                }
            )
    if metrics["teacher"]["request_macro_recall_at_8"] < 0.98:
        raise ValueError("authoritative next-router reconstruction fell below 0.98")
    control_mean = float(metrics["control"]["request_macro_recall_at_8"])
    candidate_mean = float(metrics["candidate"]["request_macro_recall_at_8"])
    lift = candidate_mean - control_mean
    horizon_lifts = {
        horizon: float(metrics["candidate"]["by_horizon"][horizon])
        - float(metrics["control"]["by_horizon"][horizon])
        for horizon in metrics["control"]["by_horizon"]
    }
    efficiency_ok = (
        pair["candidate_train_resident_hits"] <= pair["control_train_resident_hits"]
        and resident_calls["candidate"] <= resident_calls["control"]
    )
    gate_passed = (
        lift >= args.minimum_recall_lift
        and min(horizon_lifts.values()) >= -args.maximum_horizon_regression
        and efficiency_ok
    )
    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "layer": args.layer,
        "split": "tune",
        "requests": len(groups),
        "rows": len(dataset),
        "control_plan_sha256": sha256_file(args.control_plan),
        "candidate_plan_sha256": sha256_file(args.candidate_plan),
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "optimizer_constructed": False,
        "training_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    result = {
        "schema": SCHEMA,
        **pair,
        "split": "tune",
        "requests": len(groups),
        "rows": len(dataset),
        "valid_endpoints": valid_endpoints,
        "metrics": metrics,
        "candidate_recall_lift": lift,
        "candidate_recall_lift_by_horizon": horizon_lifts,
        "candidate_routed_mse_delta": metrics["candidate"]["endpoint_mean_routed_mse"]
        - metrics["control"]["endpoint_mean_routed_mse"],
        "candidate_next_router_logit_mse_delta": metrics["candidate"]["endpoint_mean_next_router_logit_mse"]
        - metrics["control"]["endpoint_mean_next_router_logit_mse"],
        "candidate_resident_call_delta": resident_calls["candidate"] - resident_calls["control"],
        "efficiency_nonregression_verified": efficiency_ok,
        "minimum_recall_lift": args.minimum_recall_lift,
        "maximum_horizon_regression": args.maximum_horizon_regression,
        "local_gate_passed": gate_passed,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "RUN_MANIFEST.json", manifest)
    write_rows(args.output / "request_predictions.jsonl", rows_out)
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    write_checksums(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
