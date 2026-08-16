#!/usr/bin/env python3
"""Measure leakage-safe HARP/Resident-Shadow complementarity.

The prediction sidecar is label free.  Authoritative target sets and diagnostic
strata are joined only inside this offline evaluator from the immutable v4
ceiling bundle.  Every oracle in this file is privileged and is never exported
as a serving feature.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.exact_k import stable_topk  # noqa: E402
from harp_rtt.resident_fusion import (  # noqa: E402
    PREDICTION_SIDECAR_SCHEMA,
    candidate_recall,
    candidate_union,
    oracle_expert_merge_recall,
    oracle_model_switch_recall,
    paired_set_overlap,
    true_expert_ranks,
)
from harp_rtt.shadow_checkpoint import sha256_file  # noqa: E402


SCHEMA = "harp_resident_harp_complementarity_v1"
CEILING_SCHEMA = "harp_deltaroute_v4_ceiling_bundle_v1"
FULL_SHADOW_REFERENCE = 0.916573


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-sidecar", type=Path, required=True)
    parser.add_argument("--ceiling-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full-shadow-reference", type=float, default=FULL_SHADOW_REFERENCE
    )
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def write_checksums(output: Path) -> None:
    paths = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush(); os.fsync(handle.fileno())


def _strict_torch_dict(path: Path, schema: str) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{path} has the wrong schema")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{path} lacks provenance")
    for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if provenance.get(key) is not False:
            raise PermissionError(f"{path} violates {key}")
    return value


def load_inputs(sidecar_path: Path, ceiling_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sidecar = _strict_torch_dict(sidecar_path, PREDICTION_SIDECAR_SCHEMA)
    ceiling = _strict_torch_dict(ceiling_path, CEILING_SCHEMA)
    provenance = sidecar["provenance"]
    if provenance.get("label_free") is not True or provenance.get("runtime_available") is not True:
        raise PermissionError("prediction sidecar is not declared label-free/runtime-safe")
    if provenance.get("ceiling_bundle_sha256") != sha256_file(ceiling_path):
        raise ValueError("prediction sidecar/ceiling checksum binding changed")
    records = sidecar.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("prediction sidecar contains no records")
    return sidecar, ceiling


def _stable_ranked(scores: torch.Tensor, width: int) -> torch.Tensor:
    if scores.shape != (40, 256):
        raise ValueError("ranked score cell must be [40,256]")
    return stable_topk(scores.float(), width)


def _slot_recall(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (target[..., :, None] == predicted[..., None, :]).any(-1).float().mean(-1)


def _weighted_node_feature(
    value: torch.Tensor,
    captured: torch.Tensor,
    branch_mask: torch.Tensor,
) -> torch.Tensor:
    """Posterior-weight node telemetry into factual horizon/layer features."""

    if value.shape[:2] != (32, 40) or captured.shape != branch_mask.shape:
        raise ValueError("node feature/posterior geometry changed")
    weights = captured.float() * branch_mask.float()
    return torch.einsum("hn,nl->hl", weights, value.float())


def _mass_bin(value: float) -> str:
    for high, name in (
        (0.01, "lt_.01"), (0.10, ".01_.10"), (0.25, ".10_.25"),
        (0.50, ".25_.50"), (float("inf"), "ge_.50"),
    ):
        if value < high:
            return name
    raise AssertionError("unreachable")


def _count_bin(value: float) -> str:
    if value < 0.5:
        return "0"
    if value < 2.5:
        return "1_2"
    if value < 4.5:
        return "3_4"
    return "ge_5"


@dataclass
class Cell:
    request_id: str
    horizon: int
    layer: int
    prefix_mismatch: bool
    prior_missing_mass: float
    prior_missing_count: float
    shadow_recall: float
    harp_recall: float
    both: float
    shadow_unique: float
    harp_unique: float
    neither: float
    union8_recall: float
    switch_recall: float
    merge1: float
    merge2: float
    merge4: float
    merge8: float
    expert_merge_k8: float
    expert_merge_k16: float
    expert_merge_k32: float
    expert_merge_k64: float
    cold_merge8: float
    cold_truth_fraction: float
    harp_unique_cold: float
    harp_unique_available: float
    shadow_missed_count: float
    harp_missed_truth_mean_rank: float
    harp_recovered_miss8: float
    harp_recovered_miss16: float
    harp_recovered_miss32: float
    harp_recovered_miss64: float

    def row(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        value["layer_quartile"] = self.layer // 10
        value["prior_missing_mass_bin"] = _mass_bin(self.prior_missing_mass)
        value["prior_missing_count_bin"] = _count_bin(self.prior_missing_count)
        return value


METRICS = (
    "shadow_recall", "harp_recall", "both", "shadow_unique", "harp_unique",
    "neither", "union8_recall", "switch_recall", "merge1", "merge2",
    "merge4", "merge8", "expert_merge_k8", "expert_merge_k16",
    "expert_merge_k32", "expert_merge_k64", "cold_merge8", "cold_truth_fraction",
    "harp_unique_cold", "harp_unique_available", "shadow_missed_count",
    "harp_missed_truth_mean_rank",
    "harp_recovered_miss8", "harp_recovered_miss16",
    "harp_recovered_miss32", "harp_recovered_miss64",
)


def build_cells(sidecar: Mapping[str, Any], ceiling: Mapping[str, Any]) -> list[Cell]:
    cells: list[Cell] = []
    request_ids = ceiling.get("request_ids")
    if not isinstance(request_ids, list):
        raise ValueError("ceiling bundle lacks request IDs")
    for record in sidecar["records"]:
        row = int(record["ceiling_row"])
        if not 0 <= row < len(request_ids):
            raise ValueError("prediction sidecar ceiling row is invalid")
        if str(record["request_id"]) != str(request_ids[row]):
            raise ValueError("prediction sidecar request join changed")
        shadow_ids = record["final_shadow_ids"].long()
        shadow_marginals = record["final_shadow_marginals"].float()
        harp_marginals = ceiling["anchor_marginals"][row].float()
        target = ceiling["target_ids"][row].long()
        valid = ceiling["future_valid"][row].bool()
        prefix_mismatch = ceiling["prefix_mismatch"][row].bool()
        if shadow_ids.shape != target.shape or shadow_marginals.shape != harp_marginals.shape:
            raise ValueError("prediction and ceiling geometry differ")
        harp_ranked = {
            width: stable_topk(harp_marginals, width) for width in (8, 16, 32, 64)
        }
        captured = record["shadow_lm_captured_probabilities"].float()
        branch_mask = record["branch_mask"].bool()
        prior_mass = _weighted_node_feature(
            record["prior_missing_mass"].float(), captured, branch_mask
        )
        prior_count = _weighted_node_feature(
            record["prior_missing_count"].float(), captured, branch_mask
        )
        availability = record["available_experts"].bool()
        for horizon in range(4):
            for layer in range(40):
                if not bool(valid[horizon, layer]):
                    continue
                s = shadow_ids[horizon, layer]
                h8 = harp_ranked[8][horizon, layer]
                y = target[horizon, layer]
                overlap = paired_set_overlap(s[None], h8[None], y[None])
                ranked = harp_ranked[64][horizon, layer]
                merges = oracle_expert_merge_recall(
                    s[None], ranked[None], y[None], swap_caps=(1, 2, 4, 8)
                )
                expert_merge_by_k = {
                    width: oracle_expert_merge_recall(
                        s[None], harp_ranked[width][horizon, layer][None],
                        y[None], swap_caps=(8,),
                    )[8]
                    for width in (8, 16, 32, 64)
                }
                cold = ~availability[layer].gather(0, y)
                cold_merge = oracle_expert_merge_recall(
                    s[None], harp_ranked[32][horizon, layer][None], y[None], swap_caps=(8,),
                    cold_expert_mask=cold[None],
                )[8]
                shadow_hit = (y[:, None] == s[None, :]).any(-1)
                missed = ~shadow_hit
                harp_hit8 = (y[:, None] == h8[None, :]).any(-1)
                missed_ranks = true_expert_ranks(
                    harp_marginals[horizon, layer][None], y[None]
                )[0][missed]
                recovered: dict[int, float] = {}
                for width in (8, 16, 32, 64):
                    hit = (y[:, None] == harp_ranked[width][horizon, layer][None, :]).any(-1)
                    recovered[width] = float((hit & missed).sum()) / 8.0
                union8 = candidate_union(s[None], h8[None])
                cells.append(Cell(
                    request_id=str(record["source_request_id"]),
                    horizon=horizon + 1,
                    layer=layer,
                    prefix_mismatch=bool(prefix_mismatch[horizon]),
                    prior_missing_mass=float(prior_mass[horizon, layer]),
                    prior_missing_count=float(prior_count[horizon, layer]),
                    shadow_recall=float(overlap.shadow_recall[0]),
                    harp_recall=float(overlap.harp_recall[0]),
                    both=float(overlap.both[0]),
                    shadow_unique=float(overlap.shadow_unique[0]),
                    harp_unique=float(overlap.harp_unique[0]),
                    neither=float(overlap.neither[0]),
                    union8_recall=float(candidate_recall(union8, y[None])[0]),
                    switch_recall=float(oracle_model_switch_recall(s[None], h8[None], y[None])[0]),
                    merge1=float(merges[1][0]), merge2=float(merges[2][0]),
                    merge4=float(merges[4][0]), merge8=float(merges[8][0]),
                    expert_merge_k8=float(expert_merge_by_k[8][0]),
                    expert_merge_k16=float(expert_merge_by_k[16][0]),
                    expert_merge_k32=float(expert_merge_by_k[32][0]),
                    expert_merge_k64=float(expert_merge_by_k[64][0]),
                    cold_merge8=float(cold_merge[0]),
                    cold_truth_fraction=float(cold.float().mean()),
                    harp_unique_cold=float((harp_hit8 & missed & cold).sum()) / 8.0,
                    harp_unique_available=float((harp_hit8 & missed & ~cold).sum()) / 8.0,
                    shadow_missed_count=float(missed.sum()),
                    harp_missed_truth_mean_rank=(
                        0.0 if missed_ranks.numel() == 0
                        else float(missed_ranks.float().mean())
                    ),
                    harp_recovered_miss8=recovered[8],
                    harp_recovered_miss16=recovered[16],
                    harp_recovered_miss32=recovered[32],
                    harp_recovered_miss64=recovered[64],
                ))
    if not cells:
        raise ValueError("paired analysis has no valid cells")
    return cells


def request_macro(cells: list[Cell], metric: str, *, horizon: int | None = None) -> float:
    by_request: dict[str, list[float]] = defaultdict(list)
    for cell in cells:
        if horizon is None or cell.horizon == horizon:
            by_request[cell.request_id].append(float(getattr(cell, metric)))
    if not by_request or any(not values for values in by_request.values()):
        raise ValueError("request macro has incomplete data")
    return sum(sum(values) / len(values) for values in by_request.values()) / len(by_request)


def summarize(cells: list[Cell]) -> dict[str, Any]:
    summary: dict[str, Any] = {"requests": len({cell.request_id for cell in cells})}
    for metric in METRICS:
        summary[f"{metric}_h1_h4"] = request_macro(cells, metric)
        for horizon in (1, 2, 3, 4):
            summary[f"{metric}_h{horizon}"] = request_macro(cells, metric, horizon=horizon)
    strata: dict[str, Any] = {}
    for name, predicate in {
        "prefix_matched": lambda c: not c.prefix_mismatch,
        "prefix_mismatched": lambda c: c.prefix_mismatch,
        "high_prior_missing_mass": lambda c: c.prior_missing_mass >= 0.25,
        "low_prior_missing_mass": lambda c: c.prior_missing_mass < 0.10,
        "late_layers": lambda c: c.layer >= 30,
    }.items():
        selected = [cell for cell in cells if predicate(cell)]
        if selected:
            strata[name] = {
                metric: request_macro(selected, metric)
                for metric in (
                    "shadow_recall", "harp_recall", "harp_unique",
                    "expert_merge_k32", "merge8",
                )
            }
    summary["strata"] = strata
    return summary


def paired_bootstrap(
    cells: list[Cell], *, metric: str, baseline: str = "shadow_recall",
    replicates: int, seed: int,
) -> dict[str, float]:
    requests = sorted({cell.request_id for cell in cells})
    per_request: dict[str, tuple[float, float]] = {}
    for request in requests:
        rows = [cell for cell in cells if cell.request_id == request]
        per_request[request] = (
            sum(float(getattr(cell, metric)) for cell in rows) / len(rows),
            sum(float(getattr(cell, baseline)) for cell in rows) / len(rows),
        )
    delta = torch.tensor([per_request[key][0] - per_request[key][1] for key in requests])
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(replicates):
        indices = torch.randint(len(requests), (len(requests),), generator=generator)
        samples.append(float(delta[indices].mean()))
    distribution = torch.tensor(samples).sort().values
    low = float(distribution[int(0.025 * (replicates - 1))])
    high = float(distribution[int(0.975 * (replicates - 1))])
    return {"point_delta": float(delta.mean()), "ci95_low": low, "ci95_high": high}


def branch_compatibility_grid(
    sidecar: Mapping[str, Any],
    ceiling: Mapping[str, Any],
    *,
    eta_values: Iterable[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
) -> dict[str, Any]:
    """Cached Shadow/HARP branch-compatibility factorial.

    Captured mass and OTHER mass remain fixed.  Only the conditional
    distribution among already captured causal branches is reweighted.
    This is deliberately diagnostic: the old HARP predictor remains a factual
    horizon forecast and is never broadcast as a counterfactual branch label.
    """

    request_ids = ceiling["request_ids"]
    results: dict[str, Any] = {}
    for eta in eta_values:
        by_request: dict[tuple[str, int], list[float]] = defaultdict(list)
        branch_rows: list[tuple[bool, float]] = []
        for record in sidecar["records"]:
            row = int(record["ceiling_row"])
            if str(record["request_id"]) != str(request_ids[row]):
                raise ValueError("branch compatibility join changed")
            source_request = str(record["source_request_id"])
            node_ids = record["selected_ids"].long()
            branch_mask = record["branch_mask"].bool()
            captured = record["shadow_lm_captured_probabilities"].float()
            other = record["shadow_lm_other_probabilities"].float()
            harp = ceiling["anchor_marginals"][row].float()
            target = ceiling["target_ids"][row].long()
            valid = ceiling["future_valid"][row].bool()
            factual = ceiling["factual_branch_indices"][row].long()
            predictions = record["final_shadow_ids"].long().clone()
            for horizon in range(1, 4):
                active = branch_mask[horizon] & (captured[horizon] > 0)
                if bool(active.any()):
                    support = (
                        node_ids[..., None] == torch.arange(256)[None, None, None, :]
                    ).any(-2).float()
                    compatibility = (
                        support * harp[horizon][None]
                    ).sum(-1).mean(-1) / 8.0
                    logits = captured[horizon].clamp_min(1e-30).log() + float(eta) * compatibility
                    conditional = torch.softmax(logits.masked_fill(~active, -torch.inf), dim=-1)
                    adjusted = conditional * captured[horizon].sum()
                    marginal = torch.einsum("n,nle->le", adjusted, support)
                    marginal = marginal + other[horizon] * harp[horizon]
                    predictions[horizon] = stable_topk(marginal, 8)
                    order = torch.argsort(adjusted, descending=True, stable=True)
                    factual_index = int(factual[horizon])
                    if 0 <= factual_index < 32 and bool(active[factual_index]):
                        rank = int((order == factual_index).nonzero(as_tuple=False)[0]) + 1
                        branch_rows.append((rank == 1, 1.0 / rank))
                    else:
                        branch_rows.append((False, 0.0))
            recall = _slot_recall(predictions, target)
            for horizon in range(4):
                active_layers = valid[horizon]
                by_request[(source_request, horizon + 1)].append(
                    float(recall[horizon][active_layers].mean())
                )
        horizon_metrics = {}
        for horizon in (1, 2, 3, 4):
            request_values = [
                sum(values) / len(values)
                for (request, h), values in by_request.items() if h == horizon
            ]
            horizon_metrics[f"recall_h{horizon}"] = sum(request_values) / len(request_values)
        horizon_metrics["recall_h1_h4"] = sum(
            horizon_metrics[f"recall_h{h}"] for h in (1, 2, 3, 4)
        ) / 4.0
        horizon_metrics["factual_branch_top1"] = (
            0.0 if not branch_rows else sum(float(v[0]) for v in branch_rows) / len(branch_rows)
        )
        horizon_metrics["factual_branch_mrr"] = (
            0.0 if not branch_rows else sum(v[1] for v in branch_rows) / len(branch_rows)
        )
        results[str(float(eta))] = horizon_metrics
    best_eta = max(
        results, key=lambda value: (results[value]["recall_h1_h4"], -float(value))
    )
    return {"conditions": results, "best_eta": float(best_eta), "diagnostic_only": True}


def information_gate(summary: Mapping[str, Any], *, reference: float) -> dict[str, Any]:
    baseline = float(summary["shadow_recall_h1_h4"])
    oracle = float(summary["expert_merge_k32_h1_h4"])
    oracle_h4 = float(summary["expert_merge_k32_h4"])
    gap = max(0.0, float(reference) - baseline)
    recovery = 1.0 if gap == 0.0 else (oracle - baseline) / gap
    high = summary.get("strata", {}).get("high_prior_missing_mass", {})
    missed = 1.0 - float(high.get("shadow_recall", 1.0))
    high_recovery = (
        0.0 if missed <= 0 else float(high.get("harp_unique", 0.0)) / missed
    )
    large_headroom = (
        (oracle - baseline >= 0.08 and oracle_h4 - float(summary["shadow_recall_h4"]) >= 0.08)
        or oracle >= 0.90
    )
    passed = bool(large_headroom and recovery >= 0.5 and high_recovery >= 0.40)
    return {
        "passed": passed,
        "large_headroom": large_headroom,
        "full_shadow_gap_recovery": recovery,
        "high_omission_missed_slot_recovery": high_recovery,
        "requirements": {
            "oracle_h1_h4_gain_and_h4_gain": 0.08,
            "or_oracle_h1_h4": 0.90,
            "full_shadow_gap_recovery": 0.50,
            "high_omission_missed_slot_recovery": 0.40,
        },
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output directory {args.output}")
    args.output.mkdir(parents=True)
    sidecar, ceiling = load_inputs(args.prediction_sidecar, args.ceiling_bundle)
    write_json_exclusive(args.output / "run_manifest.json", {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "prediction_sidecar_sha256": sha256_file(args.prediction_sidecar),
        "ceiling_bundle_sha256": sha256_file(args.ceiling_bundle),
        "resident_policy": sidecar["provenance"]["resident_policy"],
        "optimizer_constructed": False,
        "training_started": False,
        "outer_train_only": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    })
    cells = build_cells(sidecar, ceiling)
    summary = summarize(cells)
    bootstrap = {
        name: paired_bootstrap(
            cells, metric=name, replicates=args.bootstrap_replicates, seed=args.seed
        )
        for name in (
            "harp_recall", "switch_recall", "merge1", "merge2", "merge4",
            "merge8", "expert_merge_k8", "expert_merge_k16",
            "expert_merge_k32", "expert_merge_k64",
        )
    }
    gate = information_gate(summary, reference=args.full_shadow_reference)
    branch_grid = branch_compatibility_grid(sidecar, ceiling)
    write_jsonl_exclusive(args.output / "paired_cells.jsonl", (cell.row() for cell in cells))
    write_json_exclusive(args.output / "INFORMATION_GATE.json", gate)
    write_json_exclusive(args.output / "RESULT.json", {
        "schema": SCHEMA,
        "summary": summary,
        "paired_request_bootstrap": bootstrap,
        "branch_compatibility_grid": branch_grid,
        "information_gate": gate,
        "training_started": False,
    })
    write_checksums(args.output)
    print(json.dumps({"summary": summary, "information_gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
