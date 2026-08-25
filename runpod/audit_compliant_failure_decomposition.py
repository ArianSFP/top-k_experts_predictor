#!/usr/bin/env python3
"""Audit Resident-Only factual failures without a model run or new capture.

The diagnostic uses the retained exact-512 score audit and the locked
frequency-core resident allocation.  It reports:

* an associative future-target namespace/error partition;
* current-token resident execution coverage;
* survivor-versus-novel factual errors and candidate-set oracles;
* the retained all-layer full-expert INT4 quantization proxy; and
* a deliberately favorable, label-aware optional-cell reallocation surrogate.

The namespace partition and reallocation surrogate are not causal
counterfactuals: the deployed router still scores all 256 expert IDs.  They are
used only to reject resident allocation as a sufficiently large direct lever.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.resident_damage import (  # noqa: E402
    allocate_core_locked_utility,
    frequency_core_ids,
    validate_core_inclusion,
)


LAYERS = 40
EXPERTS = 256
HORIZONS = 4
TOP_K = 8
TARGET_RECALL = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--q4-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sealed output: {path}")
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stable_dense_order(scores: Tensor) -> Tensor:
    return scores.float().argsort(dim=-1, descending=True, stable=True)


def candidate_ids(scores: Tensor, current_ids: Tensor, width: int) -> Tensor:
    order = stable_dense_order(scores)
    current = current_ids.long()[:, None].expand(-1, HORIZONS, -1, -1)
    ranked_is_current = (order.unsqueeze(-1) == current.unsqueeze(-2)).any(-1)
    novel = order[~ranked_is_current].reshape(
        scores.shape[0], HORIZONS, LAYERS, EXPERTS - TOP_K
    )
    result = torch.cat((current, novel[..., : width - TOP_K]), dim=-1)
    sorted_ids = result.sort(dim=-1).values
    if not bool((sorted_ids[..., 1:] != sorted_ids[..., :-1]).all()):
        raise AssertionError("candidate construction produced duplicate IDs")
    return result


def candidate_dense_rank(scores: Tensor, candidates: Tensor) -> Tensor:
    order = stable_dense_order(scores)
    inverse = torch.empty_like(order)
    inverse.scatter_(
        -1,
        order,
        torch.arange(EXPERTS).view(1, 1, 1, EXPERTS).expand_as(order),
    )
    return inverse.gather(-1, candidates.long())


def stable_select(
    candidates: Tensor,
    values: Tensor,
    dense_rank: Tensor,
    count: int = TOP_K,
) -> Tensor:
    """Select with authoritative full-256 dense rank as the tie-break."""
    rank_order = dense_rank.long().argsort(
        dim=-1, descending=False, stable=True
    )
    ordered_values = values.float().gather(-1, rank_order)
    value_order = ordered_values.argsort(
        dim=-1, descending=True, stable=True
    )[..., :count]
    return candidates.long().gather(
        -1, rank_order.gather(-1, value_order)
    )


def target_membership(candidates: Tensor, target_ids: Tensor) -> Tensor:
    return (
        candidates.long().unsqueeze(-1)
        == target_ids.long().unsqueeze(-2)
    ).any(-1)


def request_macro_metrics(
    selected_ids: Tensor,
    current_ids: Tensor,
    target_ids: Tensor,
    valid: Tensor,
    request_ids: list[str],
) -> dict[str, Any]:
    request_names = sorted(set(request_ids))
    request_index = torch.tensor(
        [request_names.index(value) for value in request_ids], dtype=torch.long
    )
    factual = (
        selected_ids.unsqueeze(-1) == target_ids.unsqueeze(-2)
    ).any(-1).sum(-1).float() / TOP_K
    target_in_current = (
        target_ids.unsqueeze(-1)
        == current_ids[:, None, :, None, :]
    ).any(-1)
    target_in_prediction = (
        target_ids.unsqueeze(-1) == selected_ids.unsqueeze(-2)
    ).any(-1)
    factual_h: list[float] = []
    cache_h: list[float] = []
    for horizon in range(HORIZONS):
        factual_requests: list[Tensor] = []
        cache_requests: list[Tensor] = []
        for request in range(len(request_names)):
            rows = request_index.eq(request)[:, None] & valid[:, horizon]
            factual_requests.append(factual[:, horizon][rows].mean())
            numerator = (
                target_in_current[:, horizon]
                & target_in_prediction[:, horizon]
                & rows.unsqueeze(-1)
            ).sum()
            denominator = (
                target_in_current[:, horizon] & rows.unsqueeze(-1)
            ).sum()
            cache_requests.append(
                numerator.float() / denominator.clamp_min(1)
            )
        factual_h.append(float(torch.stack(factual_requests).mean()))
        cache_h.append(float(torch.stack(cache_requests).mean()))
    return {
        "factual_h": factual_h,
        "factual_mean": sum(factual_h) / HORIZONS,
        "cache_set_h": cache_h,
        "cache_set_mean": sum(cache_h) / HORIZONS,
    }


def masked_partition(
    active: Tensor,
    correct: Tensor,
    partition: Tensor,
) -> dict[str, float | int]:
    denominator = int(active.sum())
    slots = int((active & partition).sum())
    hits = int((active & partition & correct).sum())
    misses = slots - hits
    return {
        "slots": slots,
        "slot_fraction": slots / denominator,
        "recall": hits / slots if slots else 0.0,
        "misses": misses,
        "miss_contribution": misses / denominator,
    }


def namespace_partition(
    active: Tensor,
    correct: Tensor,
    target_is_resident: Tensor,
) -> dict[str, float | int]:
    denominator = int(active.sum())
    missed_absent = active & ~correct & ~target_is_resident
    missed_present = active & ~correct & target_is_resident
    recall = float(correct[active].float().mean())
    absent_rate = int(missed_absent.sum()) / denominator
    present_rate = int(missed_present.sum()) / denominator
    return {
        "active_slots": denominator,
        "recall": recall,
        "error_rate": 1.0 - recall,
        "target_resident_coverage": float(
            target_is_resident[active].float().mean()
        ),
        "miss_target_nonresident": absent_rate,
        "miss_target_resident": present_rate,
        "nonresident_miss_fraction_of_errors": (
            absent_rate / (absent_rate + present_rate)
        ),
        "resident_miss_fraction_of_errors": (
            present_rate / (absent_rate + present_rate)
        ),
        "all_nonresident_misses_fixed_ceiling": recall + absent_rate,
    }


def load_q4_proxy(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    q4 = [row for row in rows if row.get("bits") == 4]
    if len(q4) != 39:
        raise ValueError(f"expected 39 full-expert q4 rows, found {len(q4)}")
    q4.sort(key=lambda row: int(row["layer"]))
    exact_mean = sum(
        float(row["exact_baseline"]["request_macro_next_router_recall_at_8"])
        for row in q4
    ) / len(q4)
    quant_mean = sum(
        float(row["metrics"]["request_macro_next_router_recall_at_8"])
        for row in q4
    ) / len(q4)
    exact_h = [
        sum(
            float(row["exact_baseline"]["next_router_recall_at_8_by_horizon"][h])
            for row in q4
        )
        / len(q4)
        for h in range(HORIZONS)
    ]
    quant_h = [
        sum(
            float(row["metrics"]["next_router_recall_at_8_by_horizon"][h])
            for row in q4
        )
        / len(q4)
        for h in range(HORIZONS)
    ]
    return {
        "interpretation": (
            "Retained full-expert group64 INT4-MSE/log8 local next-router "
            "proxy; it is not the exact compliant closed-loop condition."
        ),
        "layers": len(q4),
        "exact_mean": exact_mean,
        "q4_mean": quant_mean,
        "quantization_loss": exact_mean - quant_mean,
        "exact_h": exact_h,
        "q4_h": quant_h,
        "quantization_loss_h": [
            exact - quant for exact, quant in zip(exact_h, quant_h)
        ],
        "per_layer": [
            {
                "layer": int(row["layer"]),
                "exact": float(
                    row["exact_baseline"][
                        "request_macro_next_router_recall_at_8"
                    ]
                ),
                "q4": float(
                    row["metrics"]["request_macro_next_router_recall_at_8"]
                ),
                "quantization_loss": float(
                    row["exact_baseline"][
                        "request_macro_next_router_recall_at_8"
                    ]
                    - row["metrics"][
                        "request_macro_next_router_recall_at_8"
                    ]
                ),
            }
            for row in q4
        ],
    }


def build_result(
    audit_path: Path,
    allocation_path: Path,
    q4_ledger_path: Path,
) -> dict[str, Any]:
    audit = torch.load(audit_path, map_location="cpu", weights_only=False)
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    scores = audit["scores"].float()
    current = audit["current_ids"].long()
    target = audit["target_ids"].long()
    valid = audit["valid"].bool()
    request_ids = [str(value) for value in audit["request_ids"]]
    if scores.shape != (512, HORIZONS, LAYERS, EXPERTS):
        raise ValueError("score audit geometry changed")
    if current.shape != (512, LAYERS, TOP_K):
        raise ValueError("current route geometry changed")
    if target.shape != (512, HORIZONS, LAYERS, TOP_K):
        raise ValueError("target route geometry changed")
    if valid.shape != (512, HORIZONS, LAYERS):
        raise ValueError("valid-mask geometry changed")
    if len(request_ids) != 512 or len(set(request_ids)) != 32:
        raise ValueError("request population changed")

    resident_ids = [
        [int(value) for value in row]
        for row in allocation["resident_expert_ids_by_layer"]
    ]
    expert_counts = torch.as_tensor(
        allocation["expert_counts"], dtype=torch.int64
    )
    if len(resident_ids) != LAYERS or expert_counts.shape != (LAYERS, EXPERTS):
        raise ValueError("allocation geometry changed")
    if sum(map(len, resident_ids)) != 3850:
        raise ValueError("allocation is not exactly 3,850 cells")
    core, core_hash = frequency_core_ids(expert_counts, core_size=64)
    validate_core_inclusion(resident_ids, core)
    if core_hash != allocation["frequency_core_sha256"]:
        raise ValueError("frequency-core hash changed")
    if int(allocation["resident_hit_count"]) > int(
        allocation["resident_hit_count_cap"]
    ):
        raise ValueError("resident-hit cap is violated")

    resident = torch.zeros((LAYERS, EXPERTS), dtype=torch.bool)
    for layer, ids in enumerate(resident_ids):
        resident[layer, ids] = True
    prediction = stable_dense_order(scores)[..., :TOP_K]
    correct = (
        target.unsqueeze(-1) == prediction.unsqueeze(-2)
    ).any(-1)
    active = valid.unsqueeze(-1).expand_as(target)
    target_resident = resident[
        torch.arange(LAYERS).view(1, 1, LAYERS, 1), target
    ]

    global_namespace = namespace_partition(
        active, correct, target_resident
    )
    horizon_namespace = [
        {
            "horizon": horizon + 1,
            **namespace_partition(
                active[:, horizon],
                correct[:, horizon],
                target_resident[:, horizon],
            ),
        }
        for horizon in range(HORIZONS)
    ]
    layer_namespace = [
        {
            "layer": layer,
            **namespace_partition(
                active[:, :, layer],
                correct[:, :, layer],
                target_resident[:, :, layer],
            ),
        }
        for layer in range(LAYERS)
    ]

    current_resident = resident[
        torch.arange(LAYERS).view(1, LAYERS, 1), current
    ]
    omitted_count = (~current_resident).sum(-1)
    any_omission = omitted_count.gt(0)
    miss = active & ~correct
    omission_conditioned: list[dict[str, float | int]] = []
    for count in range(TOP_K + 1):
        cells = omitted_count.eq(count)
        cell_slots = cells[:, None, :, None] & active
        if bool(cell_slots.any()):
            omission_conditioned.append(
                {
                    "omitted_experts": count,
                    "layer_cells": int(cells.sum()),
                    "factual_recall": float(
                        correct[cell_slots.expand_as(correct)].float().mean()
                    ),
                }
            )
    execution = {
        "resident_selected_slot_coverage": float(
            current_resident.float().mean()
        ),
        "mean_omitted_selected_experts_per_layer_cell": float(
            omitted_count.float().mean()
        ),
        "layer_cells_with_any_omission": float(
            any_omission.float().mean()
        ),
        "miss_contribution_on_cells_with_any_local_omission": (
            int((miss & any_omission[:, None, :, None]).sum())
            / int(active.sum())
        ),
        "miss_contribution_on_locally_complete_cells": (
            int((miss & ~any_omission[:, None, :, None]).sum())
            / int(active.sum())
        ),
        "recall_by_omitted_selected_expert_count": omission_conditioned,
        "per_layer_selected_slot_coverage": [
            float(current_resident[:, layer].float().mean())
            for layer in range(LAYERS)
        ],
    }

    survivor = (
        target.unsqueeze(-1) == current[:, None, :, None, :]
    ).any(-1)
    survivor_novel = {
        "global": {
            "survivor": masked_partition(
                active, correct, survivor
            ),
            "novel": masked_partition(
                active, correct, ~survivor
            ),
        },
        "by_horizon": [
            {
                "horizon": horizon + 1,
                "survivor": masked_partition(
                    active[:, horizon],
                    correct[:, horizon],
                    survivor[:, horizon],
                ),
                "novel": masked_partition(
                    active[:, horizon],
                    correct[:, horizon],
                    ~survivor[:, horizon],
                ),
            }
            for horizon in range(HORIZONS)
        ],
        "by_layer": [
            {
                "layer": layer,
                "survivor": masked_partition(
                    active[:, :, layer],
                    correct[:, :, layer],
                    survivor[:, :, layer],
                ),
                "novel": masked_partition(
                    active[:, :, layer],
                    correct[:, :, layer],
                    ~survivor[:, :, layer],
                ),
            }
            for layer in range(LAYERS)
        ],
    }

    candidates32 = candidate_ids(scores, current, 32)
    base32 = scores.gather(-1, candidates32)
    rank32 = candidate_dense_rank(scores, candidates32)
    target_member32 = target_membership(candidates32, target)
    current_member32 = (
        candidates32.unsqueeze(-1)
        == current[:, None, :, None, :]
    ).any(-1)
    oracle_conditions = {
        "dense_baseline": stable_select(candidates32, base32, rank32),
        "survivor_only": stable_select(
            candidates32,
            base32
            + (target_member32 & current_member32).float() * 1_000_000.0,
            rank32,
        ),
        "novel_only": stable_select(
            candidates32,
            base32
            + (target_member32 & ~current_member32).float() * 1_000_000.0,
            rank32,
        ),
        "full_c32": stable_select(
            candidates32,
            base32 + target_member32.float() * 1_000_000.0,
            rank32,
        ),
    }
    oracle_metrics = {
        name: request_macro_metrics(
            selected, current, target, valid, request_ids
        )
        for name, selected in oracle_conditions.items()
    }
    width_frontier: list[dict[str, float | int]] = []
    for width in range(TOP_K, 33):
        candidates = candidate_ids(scores, current, width)
        base = scores.gather(-1, candidates)
        dense_rank = candidate_dense_rank(scores, candidates)
        member = target_membership(candidates, target)
        selected = stable_select(
            candidates,
            base + member.float() * 1_000_000.0,
            dense_rank,
        )
        metric = request_macro_metrics(
            selected, current, target, valid, request_ids
        )
        width_frontier.append(
            {
                "width": width,
                "factual_mean": metric["factual_mean"],
                "cache_set_mean": metric["cache_set_mean"],
            }
        )

    utility = torch.zeros((LAYERS, EXPERTS), dtype=torch.float64)
    missed_nonresident = active & ~correct & ~target_resident
    for layer in range(LAYERS):
        ids = target[:, :, layer][missed_nonresident[:, :, layer]]
        utility[layer].scatter_add_(
            0, ids.reshape(-1), torch.ones(len(ids), dtype=torch.float64)
        )
    optional_oracle = allocate_core_locked_utility(
        utility,
        expert_counts,
        total_residents=3850,
        core_size=64,
        maximum_per_layer=128,
        hit_count_cap=int(allocation["resident_hit_count_cap"]),
    )
    candidate_resident = torch.zeros_like(resident)
    for layer, ids in enumerate(
        optional_oracle.resident_expert_ids_by_layer
    ):
        candidate_resident[layer, list(ids)] = True
    candidate_target_resident = candidate_resident[
        torch.arange(LAYERS).view(1, 1, LAYERS, 1), target
    ]
    recovered = missed_nonresident & candidate_target_resident
    recovered_h = [
        int(recovered[:, horizon].sum()) / int(active[:, horizon].sum())
        for horizon in range(HORIZONS)
    ]
    optional_result = {
        "interpretation": (
            "Leaked-label direct namespace-recovery surrogate. It awards a "
            "correct prediction for every currently missed, nonresident "
            "future target ID admitted by the candidate plan, ignores all "
            "damage from evicted cells, and does not rerun hidden states."
        ),
        "frequency_core_size_per_layer": 64,
        "mandatory_cells": 2560,
        "optional_cells": optional_oracle.optional_residents,
        "total_cells": optional_oracle.total_residents,
        "maximum_per_layer": 128,
        "resident_hit_count": optional_oracle.resident_hit_count,
        "resident_hit_count_cap": optional_oracle.hit_count_cap,
        "lagrange_lambda": optional_oracle.lagrange_lambda,
        "membership_sha256": optional_oracle.resident_ids_sha256,
        "frequency_core_sha256": optional_oracle.frequency_core_sha256,
        "recovered_slots": int(recovered.sum()),
        "optimistic_recall_lift": int(recovered.sum()) / int(active.sum()),
        "optimistic_recall": (
            float(correct[active].float().mean())
            + int(recovered.sum()) / int(active.sum())
        ),
        "optimistic_recall_lift_h": recovered_h,
        "optimistic_recall_h": [
            float(correct[:, horizon][active[:, horizon]].float().mean())
            + recovered_h[horizon]
            for horizon in range(HORIZONS)
        ],
        "required_recall_lift": (
            TARGET_RECALL - float(correct[active].float().mean())
        ),
    }

    q4_proxy = load_q4_proxy(q4_ledger_path)
    q4_by_layer = {
        int(row["layer"]): float(row["quantization_loss"])
        for row in q4_proxy["per_layer"]
    }
    bottlenecks = []
    for row in sorted(
        layer_namespace, key=lambda value: float(value["recall"])
    )[:16]:
        layer = int(row["layer"])
        bottlenecks.append(
            {
                **row,
                "q4_proxy_quantization_loss": q4_by_layer.get(layer),
                "novel_miss_contribution": survivor_novel["by_layer"][layer][
                    "novel"
                ]["miss_contribution"],
                "survivor_miss_contribution": survivor_novel["by_layer"][layer][
                    "survivor"
                ]["miss_contribution"],
            }
        )

    source_path = Path(__file__).resolve()
    return {
        "schema": "harp_compliant_failure_decomposition_v1",
        "contracts": {
            "new_capture": False,
            "model_run": False,
            "gpu_used": False,
            "training_started": False,
            "optimizer_constructed": False,
            "resident_plan_modified": False,
            "formal_validation_opened": False,
            "sealed_test_opened": False,
        },
        "inputs": {
            "audit": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "allocation": str(allocation_path),
            "allocation_sha256": sha256_file(allocation_path),
            "q4_ledger": str(q4_ledger_path),
            "q4_ledger_sha256": sha256_file(q4_ledger_path),
            "source": str(source_path),
            "source_sha256": sha256_file(source_path),
        },
        "allocation_contract": {
            "resident_cells": sum(map(len, resident_ids)),
            "mandatory_frequency_core_cells": LAYERS * 64,
            "optional_cells": sum(map(len, resident_ids)) - LAYERS * 64,
            "packed_int4_bytes": int(allocation["packed_int4_bytes"]),
            "resident_hit_count": int(allocation["resident_hit_count"]),
            "resident_hit_count_cap": int(
                allocation["resident_hit_count_cap"]
            ),
            "resident_ids_sha256": allocation["resident_ids_sha256"],
            "frequency_core_sha256": core_hash,
            "frequency_core_inclusion_verified": True,
        },
        "future_target_namespace_partition": {
            "interpretation": (
                "Associative error partition by whether the future factual "
                "target ID is in the resident set. The 256-way router output "
                "is not namespace-restricted, so this is not causal credit."
            ),
            "global": global_namespace,
            "by_horizon": horizon_namespace,
            "by_layer": layer_namespace,
        },
        "current_execution": execution,
        "survivor_novel": survivor_novel,
        "candidate_oracles": {
            "tie_break": "authoritative full-256 dense rank",
            "c32": oracle_metrics,
            "full_oracle_width_frontier": width_frontier,
            "first_width_reaching_target": next(
                (
                    int(row["width"])
                    for row in width_frontier
                    if float(row["factual_mean"]) >= TARGET_RECALL
                ),
                None,
            ),
        },
        "q4_quantization_proxy": q4_proxy,
        "optional_reallocation_surrogate": optional_result,
        "worst_layers": bottlenecks,
        "conclusion": {
            "resident_optional_reallocation_reaches_target": (
                optional_result["optimistic_recall"] >= TARGET_RECALL
            ),
            "dominant_error_class": "novel_future_expert_routing",
            "best_bounded_next_screen": (
                "Freeze survivor repair and train/cal-only a zero-start "
                "novel hard-negative pairwise residual over C24; retain "
                "authoritative dense-rank tie-breaking and final top8."
            ),
            "next_screen_gate": (
                "CacheSet>=0.95 and >=1.5 percentage-point factual lift on "
                "request-held-out calibration; no blind reopen."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    result = build_result(args.audit, args.allocation, args.q4_ledger)
    write_json_exclusive(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
