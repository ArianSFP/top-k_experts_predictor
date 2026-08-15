#!/usr/bin/env python3
"""Optimizer-free Resident-Shadow v2 tail/control headroom audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
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
from harp_rtt.shadow_checkpoint import (  # noqa: E402
    IndexedCheckpoint,
    load_target_layer_experts,
    sha256_file,
)
from harp_rtt.shadow_expert import (  # noqa: E402
    PackedInt4ResidentExperts,
    SharedResidualExperts,
    SwiGLUDraftExpert,
)
from runpod.train_harp_delta_v3 import (  # noqa: E402
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
    loader,
)
from runpod.train_shadow_experts_local import (  # noqa: E402
    SCHEMA as LOCAL_SCHEMA,
    batch_tensors,
    load_split,
    next_router_agreement_loss,
    write_checksums,
    write_json_exclusive,
    write_rows,
)


SCHEMA = "harp_resident_shadow_v2_headroom_audit_v1"
REPRESENTATIVE_LAYERS = (0, 6, 20, 32)
RANKS = (32, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=range(39), required=True)
    parser.add_argument(
        "--parent-mode",
        choices=("resident_int4_shared", "resident_int4_only"),
        default="resident_int4_shared",
    )
    parser.add_argument("--split", choices=("train", "tune"), default="tune")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8, 16, 32), default=16)
    return parser.parse_args()


def _request_macro(values: dict[tuple[str, int], list[float]]) -> tuple[float, dict[int, float]]:
    request_horizon = {
        key: sum(rows) / len(rows) for key, rows in values.items() if rows
    }
    if not request_horizon:
        raise ValueError("headroom audit contains no request rows")
    by_horizon: dict[int, list[float]] = defaultdict(list)
    for (_request, horizon), value in request_horizon.items():
        by_horizon[horizon].append(value)
    horizon = {
        key: sum(rows) / len(rows) for key, rows in sorted(by_horizon.items())
    }
    return sum(request_horizon.values()) / len(request_horizon), horizon


def _overlap(logits: Tensor, teacher_ids: Tensor) -> Tensor:
    predicted = stable_topk(logits, k=8)
    return (
        teacher_ids[..., None] == predicted[..., None, :]
    ).any(-1).float().mean(-1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit {args.output}")
    if len(args.source_commit) != 40:
        raise ValueError("source commit must be a full SHA")
    partition = validate_partition(args.partition_manifest, "b2_reuse_4096")
    reuse = validate_reuse_split(args.reuse_split_manifest)
    selected = set(reuse["inner_split"][
        "training_requests" if args.split == "train" else "tuning_requests"
    ])
    namespace = SimpleNamespace(
        data_profile="b2_reuse_4096",
        layer=args.layer,
        next_router_agreement=True,
        **{
            f"{args.split}_index": args.index,
            f"{args.split}_corpus": args.corpus,
            f"{args.split}_companion": args.companion,
        },
    )
    dataset, groups = load_split(
        namespace, args.split, selected_requests=selected
    )
    device = torch.device(args.device)
    checkpoint = IndexedCheckpoint(args.target_model)
    gate_up, down = load_target_layer_experts(
        checkpoint, args.layer, device=device, dtype=torch.bfloat16
    )
    prefix = f"model.language_model.layers.{args.layer + 1}."
    names = (
        prefix + "post_attention_layernorm.weight",
        prefix + "mlp.gate.weight",
    )
    target = checkpoint.tensors(names)
    next_norm = target[names[0]].to(device=device, dtype=torch.bfloat16)
    next_router = target[names[1]].to(device=device, dtype=torch.bfloat16)
    parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    expected = {
        "schema": LOCAL_SCHEMA,
        "mode": args.parent_mode,
        "layer": args.layer,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    for field, value in expected.items():
        if parent.get(field) != value:
            raise ValueError(f"resident parent {field} mismatch")
    resident_ids = parent.get("resident_expert_ids")
    state = parent.get("model_state_dict")
    if not isinstance(resident_ids, Tensor) or not isinstance(state, dict):
        raise ValueError("resident parent lacks namespace or state")
    fallback = (
        None
        if args.parent_mode == "resident_int4_only"
        else SharedResidualExperts(
            SwiGLUDraftExpert(2048, 512).to(device=device, dtype=torch.bfloat16)
        )
    )
    model = PackedInt4ResidentExperts.from_target(
        resident_ids.to(device), fallback, gate_up, down,
        exact_k=8, group_size=64,
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()

    centered = next_router.float() - next_router.float().mean(dim=0, keepdim=True)
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    bases = {rank: vh[:rank].contiguous() for rank in RANKS}

    names_to_values: dict[str, dict[tuple[str, int], list[float]]] = {
        name: defaultdict(list)
        for name in ("resident_only", "baseline", "exact_tail", *(f"rank_{r}" for r in RANKS))
    }
    missing_mass: list[float] = []
    rows_out: list[dict[str, Any]] = []
    teacher_recall_sum = teacher_rows = 0.0
    for host in loader(
        dataset, batch=args.microbatch, shuffle=False, seed=0,
        workers=args.num_workers, device=device,
    ):
        inputs, routed, ids, weights, valid, requests, states = batch_tensors(
            host, layer=args.layer, device=device
        )
        components = model.forward_components(inputs, ids, weights)
        candidates = {
            "resident_only": components.resident_output,
            "baseline": components.output,
            "exact_tail": routed,
        }
        error = routed.float() - components.output.float()
        for rank, basis in bases.items():
            projected = (error @ basis.T) @ basis
            candidates[f"rank_{rank}"] = (
                components.output.float() + projected
            ).to(components.output.dtype)
        teacher_loss, _teacher_parts, teacher_logits = next_router_agreement_loss(
            routed, states, valid, norm_weight=next_norm, router_weight=next_router
        )
        teacher_overlap = _overlap(
            teacher_logits, states["next_selected_expert_ids"].long()
        )
        teacher_recall_sum += float((teacher_overlap * valid.float()).sum())
        teacher_rows += float(valid.sum())
        batch_scores = {}
        for name, predicted_routed in candidates.items():
            _loss, _parts, logits = next_router_agreement_loss(
                predicted_routed, states, valid,
                norm_weight=next_norm, router_weight=next_router,
            )
            batch_scores[name] = _overlap(
                logits, states["next_selected_expert_ids"].long()
            )
        for row, request in enumerate(requests):
            for horizon in range(valid.shape[1]):
                if not bool(valid[row, horizon]):
                    continue
                key = (request, horizon + 1)
                payload = {
                    name: float(scores[row, horizon])
                    for name, scores in batch_scores.items()
                }
                for name, value in payload.items():
                    names_to_values[name][key].append(value)
                mass = float(components.missing_mass[row, horizon, 0])
                missing_mass.append(mass)
                rows_out.append({
                    "request_id": request,
                    "horizon": horizon + 1,
                    "layer": args.layer,
                    "missing_mass": mass,
                    **payload,
                })
    teacher_recall = teacher_recall_sum / max(teacher_rows, 1.0)
    if teacher_recall < 0.98:
        raise ValueError("authoritative next-router reconstruction fell below 0.98")
    metrics: dict[str, Any] = {}
    for name, values in names_to_values.items():
        mean, horizons = _request_macro(values)
        metrics[name] = {
            "request_macro_recall_at_8": mean,
            "by_horizon": {str(key): value for key, value in horizons.items()},
        }
    baseline = float(metrics["baseline"]["request_macro_recall_at_8"])
    rank128 = float(metrics["rank_128"]["request_macro_recall_at_8"])
    lift128 = rank128 - baseline
    selected_rank = None
    for rank in RANKS:
        lift = float(metrics[f"rank_{rank}"]["request_macro_recall_at_8"]) - baseline
        if lift >= 0.08 and lift >= 0.9 * max(lift128, 0.0):
            selected_rank = rank
            break
    passed = lift128 >= 0.10 and selected_rank is not None
    args.output.mkdir(parents=True)
    write_json_exclusive(args.output / "run_manifest.json", {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "layer": args.layer,
        "split": args.split,
        "requests": len(groups),
        "rows": len(dataset),
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "optimizer_constructed": False,
        "training_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    })
    write_rows(args.output / "request_predictions.jsonl", rows_out)
    result = {
        "schema": SCHEMA,
        "layer": args.layer,
        "teacher_reconstruction_recall_at_8": teacher_recall,
        "metrics": metrics,
        "rank128_lift": lift128,
        "selected_rank": selected_rank,
        "mean_missing_mass": sum(missing_mass) / max(len(missing_mass), 1),
        "headroom_gate_passed": passed,
        "minimum_rank128_lift": 0.10,
        "optimizer_constructed": False,
        "training_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    write_checksums(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
