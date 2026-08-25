#!/usr/bin/env python3
"""Leak-safe C32 joint factual/CacheSet bottleneck audit.

Only even request-order indices are ever sliced from the monolithic retained
bundle.  Training indices select simple causal heuristics; calibration indices
are request-disjoint and are evaluated once without selection.  Odd requests
are excluded from every row-level calculation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.cachetrajectory_c32 import (  # noqa: E402
    EXPERTS,
    HORIZONS,
    LAYERS,
    TOP_K,
    build_candidate_ids,
    request_macro_metrics,
    split_masks,
    target_membership,
)
from harp_rtt.shadow_checkpoint import sha256_file  # noqa: E402


SCHEMA = "cachetrajectory_c32_joint_bottleneck_audit_v1"
PUBLISHED_GLOBAL_C32_CANDIDATE_CEILING = 0.935219
FACTUAL_GOAL = 0.90
CACHE_SET_GOAL = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def request_hash(requests: list[str]) -> str:
    payload = "\n".join(requests).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_design_only(path: Path) -> dict[str, Any]:
    """Deserialize once, immediately clone even rows, then discard the source."""

    raw = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema", "condition", "request_ids", "tree_keys", "scores",
        "current_ids", "target_ids", "valid", "experts", "top_k",
    }
    if set(raw) != expected:
        raise ValueError(f"retained bundle fields changed: {sorted(raw)}")
    if raw["scores"].shape != (512, HORIZONS, LAYERS, EXPERTS):
        raise ValueError("retained score geometry changed")
    request_order = list(dict.fromkeys(str(value) for value in raw["request_ids"]))
    if len(request_order) != 32:
        raise ValueError("retained bundle no longer has 32 requests")
    order_index = {request: index for index, request in enumerate(request_order)}
    request_index = torch.tensor(
        [order_index[str(value)] for value in raw["request_ids"]], dtype=torch.int64
    )
    counts = torch.bincount(request_index, minlength=32)
    if counts.tolist() != [16] * 32:
        raise ValueError("retained requests are not exactly 16 trees each")
    masks = split_masks(request_index)
    design_rows = masks["design"].nonzero(as_tuple=False).flatten()
    # No reduction, comparison, print, or metric touches an odd row.
    safe = {
        "scores": raw["scores"].index_select(0, design_rows).clone(),
        "current_ids": raw["current_ids"].index_select(0, design_rows).clone(),
        "target_ids": raw["target_ids"].index_select(0, design_rows).clone(),
        "valid": raw["valid"].index_select(0, design_rows).clone(),
        "request_index": request_index.index_select(0, design_rows).clone(),
        "condition": str(raw["condition"]),
        "schema": str(raw["schema"]),
        "request_order": request_order,
    }
    del raw
    safe["masks"] = split_masks(safe["request_index"])
    if not bool(safe["masks"]["design"].all()) or bool(safe["masks"]["blind"].any()):
        raise PermissionError("safe C32 slice contains a blind request")
    return safe


def select_by_boost(
    normalized_candidate_scores: Tensor,
    candidates: Tensor,
    boost: float | Tensor,
) -> Tensor:
    current = torch.arange(candidates.shape[-1]).view(1, 1, 1, -1) < TOP_K
    current = current.expand_as(candidates)
    if isinstance(boost, Tensor):
        adjusted = normalized_candidate_scores + boost[..., None] * current.float()
    else:
        adjusted = normalized_candidate_scores + float(boost) * current.float()
    order = torch.argsort(adjusted, dim=-1, descending=True, stable=True)[..., :TOP_K]
    return candidates.gather(-1, order)


def select_quota(
    candidates: Tensor,
    normalized_candidate_scores: Tensor,
    quota: int,
    *,
    current_order: str,
) -> Tensor:
    if not 0 <= quota <= TOP_K:
        raise ValueError("current quota must lie in 0..8")
    current = candidates[..., :TOP_K]
    novel = candidates[..., TOP_K:]
    if current_order == "score":
        order = torch.argsort(
            normalized_candidate_scores[..., :TOP_K],
            dim=-1,
            descending=True,
            stable=True,
        )
        current = current.gather(-1, order)
    elif current_order != "router":
        raise ValueError("unknown current ordering")
    return torch.cat((current[..., :quota], novel[..., : TOP_K - quota]), dim=-1)


def select_oracle(candidates: Tensor, target_ids: Tensor) -> Tensor:
    membership = target_membership(candidates, target_ids)
    order = torch.argsort(membership.to(torch.int8), dim=-1, descending=True, stable=True)
    return candidates.gather(-1, order[..., :TOP_K])


def metric(
    selected: Tensor,
    current: Tensor,
    target: Tensor,
    valid: Tensor,
    request_index: Tensor,
    row_mask: Tensor,
) -> dict[str, Any]:
    return request_macro_metrics(
        selected, current, target, valid, request_index, row_mask
    )


def split_counts(
    *,
    selected: Tensor,
    candidates: Tensor,
    current: Tensor,
    target: Tensor,
    valid: Tensor,
    request_index: Tensor,
    row_mask: Tensor,
) -> dict[str, Any]:
    selected_hit = (
        target.unsqueeze(-1) == selected.unsqueeze(-2)
    ).any(-1)
    candidate_hit = (
        target.unsqueeze(-1) == candidates.unsqueeze(-2)
    ).any(-1)
    survivor = (
        target.unsqueeze(-1)
        == current[:, None, :, None, :]
    ).any(-1)
    active = valid & row_mask[:, None, None]
    result: dict[str, Any] = {}
    for horizon in range(HORIZONS):
        h_active = active[:, horizon]
        cells = int(h_active.sum())
        survivor_h = survivor[:, horizon] & h_active[..., None]
        novel_h = ~survivor[:, horizon] & h_active[..., None]
        selected_h = selected_hit[:, horizon] & h_active[..., None]
        candidate_h = candidate_hit[:, horizon] & h_active[..., None]
        requests = request_index[row_mask].unique(sorted=True)
        macro: dict[str, list[float]] = defaultdict(list)
        for request in requests:
            request_cells = h_active & request_index.eq(request)[:, None]
            denominator_cells = request_cells.sum().clamp_min(1)
            request_survivor = survivor[:, horizon] & request_cells[..., None]
            request_novel = ~survivor[:, horizon] & request_cells[..., None]
            request_selected = selected_hit[:, horizon] & request_cells[..., None]
            request_candidate = candidate_hit[:, horizon] & request_cells[..., None]
            macro["survivor_mean"].append(float(request_survivor.sum() / denominator_cells))
            macro["novel_mean"].append(float(request_novel.sum() / denominator_cells))
            macro["candidate_novel_recall"].append(
                float((request_candidate & request_novel).sum() / request_novel.sum().clamp_min(1))
            )
            macro["selected_survivor_recall"].append(
                float((request_selected & request_survivor).sum() / request_survivor.sum().clamp_min(1))
            )
            macro["selected_novel_recall"].append(
                float((request_selected & request_novel).sum() / request_novel.sum().clamp_min(1))
            )
        mean = lambda name: sum(macro[name]) / len(macro[name])
        result[str(horizon + 1)] = {
            "valid_token_layer_cells": cells,
            "survivor_total": int(survivor_h.sum()),
            "novel_total": int(novel_h.sum()),
            "survivor_request_macro_mean_per_cell": mean("survivor_mean"),
            "novel_request_macro_mean_per_cell": mean("novel_mean"),
            "candidate_novel_captured_total": int((candidate_h & novel_h).sum()),
            "candidate_novel_request_macro_recall": mean("candidate_novel_recall"),
            "selected_survivor_captured_total": int((selected_h & survivor_h).sum()),
            "selected_survivor_request_macro_recall": mean("selected_survivor_recall"),
            "selected_novel_captured_total": int((selected_h & novel_h).sum()),
            "selected_novel_request_macro_recall": mean("selected_novel_recall"),
            "candidate_missing_total": int((~candidate_h & h_active[..., None]).sum()),
            "candidate_present_but_unselected_total": int(
                (candidate_h & ~selected_h & h_active[..., None]).sum()
            ),
        }
    return result


def normalized_scores(scores: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    mean = scores.mean(-1, keepdim=True)
    std = scores.std(-1, keepdim=True, unbiased=False).clamp_min(1e-5)
    normalized = (scores - mean) / std
    probability = torch.softmax(scores, dim=-1)
    sorted_normalized = torch.sort(normalized, dim=-1, descending=True, stable=True).values
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1) / math.log(EXPERTS)
    margin8 = sorted_normalized[..., 7] - sorted_normalized[..., 8]
    return normalized, {"score_entropy": entropy, "score_margin_8_9": margin8}


def scalar_correlations(
    scalars: Mapping[str, Tensor],
    dense: Tensor,
    current: Tensor,
    target: Tensor,
    valid: Tensor,
    request_index: Tensor,
    split: Tensor,
) -> dict[str, Any]:
    target_survivor = (
        target.unsqueeze(-1) == current[:, None, :, None, :]
    ).any(-1)
    target_selected = (
        target.unsqueeze(-1) == dense.unsqueeze(-2)
    ).any(-1)
    missing_survivors = (target_survivor & ~target_selected).sum(-1).float()
    active = valid & split[:, None, None]
    y = missing_survivors[active]
    out: dict[str, Any] = {}
    for name, values in scalars.items():
        x = values[active].float()
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        correlation = (x_centered * y_centered).mean() / (
            x_centered.square().mean().sqrt() * y_centered.square().mean().sqrt()
        ).clamp_min(1e-12)
        out[name] = {
            "pearson_with_dense_missing_survivor_count": float(correlation),
            "mean": float(x.mean()),
            "std": float(x.std(unbiased=False)),
        }
    return out


def pareto_frontier(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    frontier = []
    for row in rows:
        point = row[split_name]
        dominated = False
        for other in rows:
            if other is row:
                continue
            candidate = other[split_name]
            if (
                candidate["factual_mean"] >= point["factual_mean"]
                and candidate["cache_set_mean"] >= point["cache_set_mean"]
                and (
                    candidate["factual_mean"] > point["factual_mean"]
                    or candidate["cache_set_mean"] > point["cache_set_mean"]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    frontier.sort(
        key=lambda row: (
            row[split_name]["cache_set_mean"],
            row[split_name]["factual_mean"],
            row["name"],
        )
    )
    return frontier


def goal_audit(policy: Mapping[str, Any], oracle: Mapping[str, Any]) -> dict[str, float | bool]:
    factual = float(policy["factual_mean"])
    cache = float(policy["cache_set_mean"])
    oracle_factual = float(oracle["factual_mean"])
    oracle_cache = float(oracle["cache_set_mean"])
    return {
        "factual_goal": FACTUAL_GOAL,
        "cache_set_goal": CACHE_SET_GOAL,
        "factual_shortfall_to_goal": max(0.0, FACTUAL_GOAL - factual),
        "cache_set_shortfall_to_goal": max(0.0, CACHE_SET_GOAL - cache),
        "factual_candidate_unavailable_error": 1.0 - oracle_factual,
        "factual_candidate_present_ranking_error": oracle_factual - factual,
        "cache_set_candidate_unavailable_error": 1.0 - oracle_cache,
        "cache_set_candidate_present_ranking_error": oracle_cache - cache,
        "joint_goal_met": factual >= FACTUAL_GOAL and cache >= CACHE_SET_GOAL,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    safe = load_design_only(args.bundle)
    scores = safe["scores"].float()
    current = safe["current_ids"].long()
    target = safe["target_ids"].long()
    valid = safe["valid"].bool()
    request_index = safe["request_index"].long()
    masks = safe["masks"]
    normalized, scalars = normalized_scores(scores)
    candidates = build_candidate_ids(scores, current, width=32)
    candidate_scores = normalized.gather(-1, candidates)
    expanded_current = current[:, None].expand(-1, HORIZONS, -1, -1)
    probabilities = torch.softmax(scores, dim=-1)
    scalars["current_score_mass"] = probabilities.gather(-1, expanded_current).sum(-1)
    dense = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :TOP_K]
    scalars["dense_current_overlap"] = (
        dense.unsqueeze(-1) == expanded_current.unsqueeze(-2)
    ).any(-1).sum(-1).float()
    oracle = select_oracle(candidates, target)

    split_names = ("training", "calibration", "design")
    baseline = {
        name: metric(dense, current, target, valid, request_index, masks[name])
        for name in split_names
    }
    oracle_metrics = {
        name: metric(oracle, current, target, valid, request_index, masks[name])
        for name in split_names
    }
    policies: list[dict[str, Any]] = []

    def add_policy(name: str, kind: str, selected: Tensor, config: Mapping[str, Any]) -> None:
        row: dict[str, Any] = {"name": name, "kind": kind, "config": dict(config)}
        for split_name in split_names:
            row[split_name] = metric(
                selected, current, target, valid, request_index, masks[split_name]
            )
        policies.append(row)

    add_policy("dense_score_top8", "dense", dense, {})
    for order in ("router", "score"):
        for quota in range(TOP_K + 1):
            add_policy(
                f"quota_{order}_current_{quota}",
                "static_quota",
                select_quota(candidates, candidate_scores, quota, current_order=order),
                {"current_quota": quota, "current_order": order},
            )
    boosts = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
    for boost in boosts:
        add_policy(
            f"current_zboost_{boost:g}",
            "static_boost",
            select_by_boost(candidate_scores, candidates, boost),
            {"current_zscore_boost": boost},
        )

    train_active = valid & masks["training"][:, None, None]
    quantiles = (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9)
    adaptive_boosts = (0.5, 1.0, 2.0, 4.0)
    directions = {
        "score_entropy": "high",
        "score_margin_8_9": "low",
        "current_score_mass": "low",
        "dense_current_overlap": "low",
    }
    for scalar_name, scalar in scalars.items():
        for quantile in quantiles:
            threshold = float(torch.quantile(scalar[train_active], quantile))
            direction = directions[scalar_name]
            condition = scalar >= threshold if direction == "high" else scalar <= threshold
            for boost in adaptive_boosts:
                adaptive = torch.where(
                    condition[..., None],
                    select_by_boost(candidate_scores, candidates, boost),
                    dense,
                )
                add_policy(
                    f"adaptive_{scalar_name}_{direction}_q{quantile:g}_boost{boost:g}",
                    "adaptive_scalar_boost",
                    adaptive,
                    {
                        "scalar": scalar_name,
                        "direction": direction,
                        "training_quantile": quantile,
                        "threshold": threshold,
                        "current_zscore_boost": boost,
                    },
                )

    training_frontier = pareto_frontier(policies, "training")
    # Select without calibration labels: minimum joint target shortfall on
    # training, then maximum factual+CacheSet, then stable name.
    selected_joint = min(
        policies,
        key=lambda row: (
            max(0.0, FACTUAL_GOAL - row["training"]["factual_mean"])
            + max(0.0, CACHE_SET_GOAL - row["training"]["cache_set_mean"]),
            -(row["training"]["factual_mean"] + row["training"]["cache_set_mean"]),
            row["name"],
        ),
    )
    adaptive_rows = [row for row in policies if row["kind"] == "adaptive_scalar_boost"]
    selected_adaptive = min(
        adaptive_rows,
        key=lambda row: (
            max(0.0, FACTUAL_GOAL - row["training"]["factual_mean"])
            + max(0.0, CACHE_SET_GOAL - row["training"]["cache_set_mean"]),
            -(row["training"]["factual_mean"] + row["training"]["cache_set_mean"]),
            row["name"],
        ),
    )

    counts = {
        split_name: split_counts(
            selected=dense,
            candidates=candidates,
            current=current,
            target=target,
            valid=valid,
            request_index=request_index,
            row_mask=masks[split_name],
        )
        for split_name in split_names
    }
    correlations = {
        split_name: scalar_correlations(
            scalars, dense, current, target, valid, request_index, masks[split_name]
        )
        for split_name in ("training", "calibration")
    }
    frontier_summary = []
    for row in training_frontier:
        frontier_summary.append(
            {
                "name": row["name"],
                "kind": row["kind"],
                "config": row["config"],
                "training": row["training"],
                "calibration": row["calibration"],
                "design": row["design"],
            }
        )

    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "condition": safe["condition"],
        "split_contract": {
            "request_order_definition": "first appearance in frozen request_ids",
            "training_request_indices": sorted(
                set(request_index[masks["training"]].tolist())
            ),
            "calibration_request_indices": sorted(
                set(request_index[masks["calibration"]].tolist())
            ),
            "blind_request_indices_excluded": list(range(1, 32, 2)),
            "training_requests": int(request_index[masks["training"]].unique().numel()),
            "calibration_requests": int(request_index[masks["calibration"]].unique().numel()),
            "design_requests": int(request_index.unique().numel()),
            "rows_per_request": 16,
            "training_request_ids_sha256": request_hash(
                [safe["request_order"][index] for index in sorted(set(request_index[masks["training"]].tolist()))]
            ),
            "calibration_request_ids_sha256": request_hash(
                [safe["request_order"][index] for index in sorted(set(request_index[masks["calibration"]].tolist()))]
            ),
            "blind_row_metrics_computed": False,
            "blind_predictions_or_labels_reported": False,
            "monolithic_bundle_deserialized_then_immediately_even_sliced": True,
        },
        "bundle_fields": [
            "schema", "condition", "request_ids", "tree_keys", "scores",
            "current_ids", "target_ids", "valid", "experts", "top_k",
        ],
        "candidate_width": 32,
        "dense_baseline": baseline,
        "c32_oracle": oracle_metrics,
        "per_horizon_survivor_novel": counts,
        "training_pareto_frontier": frontier_summary,
        "training_selected_joint_policy": selected_joint,
        "training_selected_adaptive_scalar_policy": selected_adaptive,
        "goal_decomposition": {
            split_name: {
                "dense": goal_audit(baseline[split_name], oracle_metrics[split_name]),
                "training_selected_joint": goal_audit(
                    selected_joint[split_name], oracle_metrics[split_name]
                ),
                "training_selected_adaptive": goal_audit(
                    selected_adaptive[split_name], oracle_metrics[split_name]
                ),
                "oracle_joint_goal_met": (
                    oracle_metrics[split_name]["factual_mean"] >= FACTUAL_GOAL
                    and oracle_metrics[split_name]["cache_set_mean"] >= CACHE_SET_GOAL
                ),
            }
            for split_name in split_names
        },
        "causal_scalar_audit": {
            "available_without_new_runtime_state": sorted(scalars),
            "training_and_calibration_correlations": correlations,
            "adaptive_policy_count": len(adaptive_rows),
            "j_space_or_router_agreement_field_in_bundle": False,
            "j_space_join_test": "not_joinable",
            "reason": (
                "The sole aligned retained C32 bundle has no J-space/router-agreement "
                "scalar or activation key. Score entropy, score margin, current score "
                "mass, and dense/current overlap were tested as zero-footprint proxies."
            ),
        },
        "published_blind_information_only": {
            "global_c32_candidate_ceiling": PUBLISHED_GLOBAL_C32_CANDIDATE_CEILING,
            "source": "already-published global candidate ceiling supplied by parent",
        },
        "lineage": {
            "bundle_path": str(args.bundle),
            "bundle_sha256": sha256_file(args.bundle),
            "bundle_result_path": str(args.bundle_result),
            "bundle_result_sha256": sha256_file(args.bundle_result),
            "driver_path": str(Path(__file__).resolve()),
            "driver_sha256": sha256_file(Path(__file__).resolve()),
            "cachetrajectory_module_path": str(
                REPO_ROOT / "harp_rtt" / "cachetrajectory_c32.py"
            ),
            "cachetrajectory_module_sha256": sha256_file(
                REPO_ROOT / "harp_rtt" / "cachetrajectory_c32.py"
            ),
        },
        "new_capture": False,
        "optimizer_constructed": False,
        "training_started": False,
        "gpu_used": False,
        "formal_validation_opened": False,
        "sealed_test_opened": False,
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "C32_BOTTLENECK_AUDIT.json", result)
    checksums = {
        "C32_BOTTLENECK_AUDIT.json": sha256_file(
            args.output / "C32_BOTTLENECK_AUDIT.json"
        )
    }
    write_json(args.output / "SHA256.json", checksums)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
