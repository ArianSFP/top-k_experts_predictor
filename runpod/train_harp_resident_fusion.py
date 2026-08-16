#!/usr/bin/env python3
"""Fit zero-baseline HARP rescue heads after the oracle information gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.exact_k import exact_set_nll, stable_topk  # noqa: E402
from harp_rtt.resident_fusion import (  # noqa: E402
    ExpertHarpRescue,
    PREDICTION_SIDECAR_SCHEMA,
    ScalarHarpRescue,
    rescue_parameter_count,
)
from harp_rtt.shadow_checkpoint import sha256_file  # noqa: E402


SCHEMA = "harp_resident_harp_fusion_training_v1"
CEILING_SCHEMA = "harp_deltaroute_v4_ceiling_bundle_v1"
FEATURE_WIDTH = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-sidecar", type=Path, required=True)
    parser.add_argument("--fit-ceiling", type=Path, required=True)
    parser.add_argument("--tune-sidecar", type=Path)
    parser.add_argument("--tune-ceiling", type=Path)
    parser.add_argument("--information-gate", type=Path, required=True)
    parser.add_argument("--variant", choices=("f0", "f1_cold", "f1_all"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--fold-modulo", type=int, default=4)
    parser.add_argument("--tune-fold", type=int, default=3)
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


def atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {path}")
    with temporary.open("x+b") as handle:
        torch.save(dict(value), handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_pair(sidecar_path: Path, ceiling_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    ceiling = torch.load(ceiling_path, map_location="cpu", weights_only=True)
    if not isinstance(sidecar, dict) or sidecar.get("schema") != PREDICTION_SIDECAR_SCHEMA:
        raise ValueError("fusion sidecar schema mismatch")
    if not isinstance(ceiling, dict) or ceiling.get("schema") != CEILING_SCHEMA:
        raise ValueError("fusion ceiling schema mismatch")
    provenance = sidecar.get("provenance")
    ceiling_provenance = ceiling.get("provenance")
    if not isinstance(provenance, Mapping) or not isinstance(ceiling_provenance, Mapping):
        raise ValueError("fusion inputs lack provenance")
    for item in (provenance, ceiling_provenance):
        for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
            if item.get(key) is not False:
                raise PermissionError(f"fusion input violates {key}")
    if provenance.get("label_free") is not True:
        raise PermissionError("fusion predictions are not label-free")
    if provenance.get("ceiling_bundle_sha256") != sha256_file(ceiling_path):
        raise ValueError("fusion sidecar/ceiling binding mismatch")
    return sidecar, ceiling


def request_fold(request_id: str, modulo: int) -> int:
    if modulo < 2:
        raise ValueError("request fold modulus must be at least two")
    digest = hashlib.sha256(request_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def _ranks(scores: Tensor) -> Tensor:
    order = torch.argsort(scores.float(), dim=-1, descending=True, stable=True)
    return torch.argsort(order, dim=-1, stable=True).float() / 255.0


def _weighted_node(value: Tensor, captured: Tensor, branch_mask: Tensor) -> Tensor:
    return torch.einsum(
        "hn,nl->hl", captured.float() * branch_mask.float(), value.float()
    )


def causal_features(record: Mapping[str, Any], harp: Tensor) -> Tensor:
    """Build serving-safe features; target and diagnostic strata are excluded."""

    shadow = record["final_shadow_marginals"].float()
    captured_shadow = record["captured_shadow_marginals"].float()
    captured_nodes = record["shadow_lm_captured_probabilities"].float()
    branch_mask = record["branch_mask"].bool()
    captured_probability = record["captured_probability"].float()
    other = record["shadow_lm_other_probabilities"].float()
    availability = record["available_experts"].bool()
    if shadow.shape != (4, 40, 256) or harp.shape != shadow.shape:
        raise ValueError("fusion marginal geometry changed")
    prior_mass = _weighted_node(record["prior_missing_mass"], captured_nodes, branch_mask)
    prior_count = _weighted_node(record["prior_missing_count"], captured_nodes, branch_mask)
    probability = torch.cat((captured_nodes, other[:, None]), dim=-1).clamp_min(1e-12)
    branch_entropy = -(probability * probability.log()).sum(-1) / math.log(33.0)
    shadow_order = stable_topk(shadow, 9)
    harp_order = stable_topk(harp, 9)
    shadow_membership = torch.zeros_like(shadow).scatter(-1, shadow_order[..., :8], 1.0)
    harp_membership = torch.zeros_like(harp).scatter(-1, harp_order[..., :8], 1.0)
    cold = (~availability)[None].expand(4, -1, -1).float()
    scalar = lambda x: x[..., None].expand(4, 40, 256)
    values = (
        shadow, harp, harp - shadow, cold, captured_shadow,
        scalar(prior_mass), scalar(prior_count / 8.0),
        scalar(captured_probability[:, None].expand(4, 40)),
        scalar(other[:, None].expand(4, 40)),
        scalar(branch_entropy[:, None].expand(4, 40)),
        _ranks(shadow), _ranks(harp), shadow_membership, harp_membership,
        (shadow_membership != harp_membership).float(),
    )
    result = torch.stack(values, dim=-1)
    if result.shape != (4, 40, 256, FEATURE_WIDTH) or not torch.isfinite(result).all():
        raise ValueError("fusion feature mapping is invalid")
    return result


class FusionRows:
    def __init__(self, sidecar: Mapping[str, Any], ceiling: Mapping[str, Any]) -> None:
        requests = ceiling.get("request_ids")
        if not isinstance(requests, list):
            raise ValueError("fusion ceiling lacks request IDs")
        self.rows: list[dict[str, Any]] = []
        for record in sidecar.get("records", []):
            index = int(record["ceiling_row"])
            if str(record["request_id"]) != str(requests[index]):
                raise ValueError("fusion record/ceiling join changed")
            harp = ceiling["anchor_marginals"][index].float()
            self.rows.append({
                "request_id": str(record["source_request_id"]),
                "baseline": record["final_shadow_marginals"].float(),
                "captured_shadow": record["captured_shadow_marginals"].float(),
                "captured_probability": record["captured_probability"].float(),
                "harp": harp,
                "features": causal_features(record, harp),
                "cold": (~record["available_experts"].bool())[None].expand(4, -1, -1),
                "target": ceiling["target_ids"][index].long(),
                "valid": ceiling["future_valid"][index].bool(),
            })
        if not self.rows:
            raise ValueError("fusion dataset contains no rows")

    def subset(self, keep: set[str]) -> "FusionRows":
        value = object.__new__(FusionRows)
        value.rows = [row for row in self.rows if row["request_id"] in keep]
        if not value.rows:
            raise ValueError("fusion request subset is empty")
        return value

    @property
    def request_ids(self) -> set[str]:
        return {row["request_id"] for row in self.rows}


def stack(rows: list[dict[str, Any]], key: str, device: torch.device) -> Tensor:
    return torch.stack([row[key] for row in rows]).to(device, non_blocking=True)


def forward_model(model: torch.nn.Module, rows: list[dict[str, Any]], variant: str, device: torch.device) -> Tensor:
    arguments: dict[str, Tensor] = {
        "baseline_marginals": stack(rows, "baseline", device),
        "harp_marginals": stack(rows, "harp", device),
    }
    if isinstance(model, ScalarHarpRescue):
        arguments.update({
            "captured_shadow_marginals": stack(rows, "captured_shadow", device),
            "captured_probability": stack(rows, "captured_probability", device),
        })
    else:
        arguments["features"] = stack(rows, "features", device)
        if variant == "f1_cold":
            arguments["cold_mask"] = stack(rows, "cold", device)
    return model(**arguments)


def swap_boundary(scores: Tensor, target: Tensor, valid: Tensor, margin: float = 0.125) -> Tensor:
    true_mask = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, target, True)
    false = scores.masked_fill(true_mask, -torch.inf).amax(-1)
    weakest_true = scores.gather(-1, target).amin(-1)
    values = torch.nn.functional.softplus(float(margin) + false - weakest_true)
    return values[valid].mean()


def objective(fused: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    epsilon = 1e-5
    scores = torch.logit(fused.float().clamp(epsilon, 1 - epsilon))
    return exact_set_nll(scores, target, valid=valid) + 0.1 * swap_boundary(scores, target, valid)


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: FusionRows, variant: str, device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    by_request: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for offset in range(0, len(data.rows), 4):
        rows = data.rows[offset : offset + 4]
        fused = forward_model(model, rows, variant, device).cpu()
        baseline = torch.stack([row["baseline"] for row in rows])
        target = torch.stack([row["target"] for row in rows])
        valid = torch.stack([row["valid"] for row in rows])
        predicted = stable_topk(fused, 8); incumbent = stable_topk(baseline, 8)
        hit = (target[..., :, None] == predicted[..., None, :]).any(-1).float().mean(-1)
        base_hit = (target[..., :, None] == incumbent[..., None, :]).any(-1).float().mean(-1)
        for index, row in enumerate(rows):
            for horizon in range(4):
                active = valid[index, horizon]
                by_request[(row["request_id"], horizon + 1)].append((
                    float(hit[index, horizon][active].mean()),
                    float(base_hit[index, horizon][active].mean()),
                ))
    request_rows = []
    for (request, horizon), values in sorted(by_request.items()):
        request_rows.append({
            "request_id": request, "horizon": horizon,
            "fusion_recall": sum(v[0] for v in values) / len(values),
            "baseline_recall": sum(v[1] for v in values) / len(values),
        })
    metrics: dict[str, float] = {}
    for key in ("fusion_recall", "baseline_recall"):
        all_h = []
        for horizon in (1, 2, 3, 4):
            values = [row[key] for row in request_rows if row["horizon"] == horizon]
            metrics[f"{key}_h{horizon}"] = sum(values) / len(values)
            all_h.append(metrics[f"{key}_h{horizon}"])
        metrics[f"{key}_h1_h4"] = sum(all_h) / 4.0
    metrics["gain_h1_h4"] = metrics["fusion_recall_h1_h4"] - metrics["baseline_recall_h1_h4"]
    metrics["gain_h4"] = metrics["fusion_recall_h4"] - metrics["baseline_recall_h4"]
    return metrics, request_rows


def paired_gain_bootstrap(
    request_rows: list[dict[str, Any]], *, replicates: int = 1_000, seed: int = 42
) -> dict[str, float]:
    by_request: dict[str, list[float]] = defaultdict(list)
    for row in request_rows:
        by_request[str(row["request_id"])].append(
            float(row["fusion_recall"]) - float(row["baseline_recall"])
        )
    if not by_request or any(len(values) != 4 for values in by_request.values()):
        raise ValueError("fusion bootstrap requires complete H1-H4 requests")
    delta = torch.tensor([
        sum(by_request[request]) / 4.0 for request in sorted(by_request)
    ])
    generator = torch.Generator().manual_seed(seed)
    samples = torch.empty(replicates)
    for index in range(replicates):
        draw = torch.randint(delta.numel(), (delta.numel(),), generator=generator)
        samples[index] = delta[draw].mean()
    ordered = samples.sort().values
    return {
        "point": float(delta.mean()),
        "ci95_low": float(ordered[int(0.025 * (replicates - 1))]),
        "ci95_high": float(ordered[int(0.975 * (replicates - 1))]),
        "requests": int(delta.numel()),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output directory {args.output}")
    if args.epochs < 1 or args.patience < 1 or args.microbatch < 1:
        raise ValueError("fusion training schedule must be positive")
    if args.effective_batch < args.microbatch or args.effective_batch % args.microbatch:
        raise ValueError("effective batch must be a multiple of microbatch")
    gate = json.loads(args.information_gate.read_text())
    if gate.get("passed") is not True:
        raise PermissionError("fusion optimizer is forbidden until the information gate passes")
    args.output.mkdir(parents=True)
    fit_sidecar, fit_ceiling = load_pair(args.fit_sidecar, args.fit_ceiling)
    fit_all = FusionRows(fit_sidecar, fit_ceiling)
    if (args.tune_sidecar is None) != (args.tune_ceiling is None):
        raise ValueError("tune sidecar and ceiling must be supplied together")
    if args.tune_sidecar is None:
        tune_ids = {
            request for request in fit_all.request_ids
            if request_fold(request, args.fold_modulo) == args.tune_fold
        }
        fit_ids = fit_all.request_ids - tune_ids
        fit = fit_all.subset(fit_ids); tune = fit_all.subset(tune_ids)
    else:
        tune_sidecar, tune_ceiling = load_pair(args.tune_sidecar, args.tune_ceiling)
        fit = fit_all; tune = FusionRows(tune_sidecar, tune_ceiling)
        if fit.request_ids & tune.request_ids:
            raise PermissionError("fusion fit/tune request groups overlap")
    if fit_sidecar["provenance"]["resident_policy"] != (
        fit_sidecar if args.tune_sidecar is None else tune_sidecar
    )["provenance"]["resident_policy"]:
        raise ValueError("fusion resident policy changed between fit and tune")
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model: torch.nn.Module
    if args.variant == "f0":
        model = ScalarHarpRescue()
    else:
        model = ExpertHarpRescue(FEATURE_WIDTH)
    model.to(device)
    if rescue_parameter_count(model) >= 1_000_000:
        raise RuntimeError("fusion rescue head exceeds its one-million-parameter cap")
    epoch_zero, _ = evaluate(model, tune, args.variant, device)
    if epoch_zero["gain_h1_h4"] != 0.0 or epoch_zero["gain_h4"] != 0.0:
        raise RuntimeError("fusion epoch zero does not reproduce Resident-Shadow")
    write_json_exclusive(args.output / "run_manifest.json", {
        "schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit, "variant": args.variant,
        "seed": args.seed, "fit_sidecar_sha256": sha256_file(args.fit_sidecar),
        "fit_ceiling_sha256": sha256_file(args.fit_ceiling),
        "tune_sidecar_sha256": None if args.tune_sidecar is None else sha256_file(args.tune_sidecar),
        "tune_ceiling_sha256": None if args.tune_ceiling is None else sha256_file(args.tune_ceiling),
        "information_gate_sha256": sha256_file(args.information_gate),
        "parameters": rescue_parameter_count(model), "optimizer_constructed": False,
        "training_started": False, "formal_validation_opened": False,
        "calibration_opened": False, "sealed_test_opened": False,
    })
    write_json_exclusive(args.output / "EPOCH_ZERO_AUDIT.json", epoch_zero)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    write_json_exclusive(args.output / "OPTIMIZER_START.json", {
        "optimizer": "AdamW", "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "trainable_names": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
    })
    accumulation = args.effective_batch // args.microbatch
    best = -float("inf"); best_epoch = -1; stale = 0; history = []
    best_state: dict[str, Tensor] | None = None
    for epoch in range(args.epochs):
        model.train(); order = list(range(len(fit.rows)))
        random.Random(args.seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True); losses = []
        for step, start in enumerate(range(0, len(order), args.microbatch)):
            rows = [fit.rows[index] for index in order[start : start + args.microbatch]]
            fused = forward_model(model, rows, args.variant, device)
            target = stack(rows, "target", device); valid = stack(rows, "valid", device)
            loss = objective(fused, target, valid) / accumulation
            loss.backward(); losses.append(float(loss.detach()) * accumulation)
            if (step + 1) % accumulation == 0 or start + args.microbatch >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        tune_metrics, _ = evaluate(model, tune, args.variant, device)
        row = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), **tune_metrics}
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        score = tune_metrics["fusion_recall_h1_h4"]
        if score > best + 1e-12:
            best = score; best_epoch = epoch + 1; stale = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("fusion training produced no checkpoint")
    model.load_state_dict(best_state); final_metrics, request_rows = evaluate(model, tune, args.variant, device)
    bootstrap = paired_gain_bootstrap(request_rows, replicates=1_000, seed=42)
    horizon_gains = [
        final_metrics[f"fusion_recall_h{h}"] - final_metrics[f"baseline_recall_h{h}"]
        for h in (1, 2, 3, 4)
    ]
    ready_for_confirmation = bool(
        bootstrap["ci95_low"] > 0.0
        and final_metrics["gain_h4"] >= 0.0
        and min(horizon_gains) >= -0.005
    )
    atomic_torch_save(args.output / "best.pt", {
        "schema": SCHEMA, "variant": args.variant, "source_commit": args.source_commit,
        "epoch": best_epoch, "model": best_state,
        "resident_policy": fit_sidecar["provenance"]["resident_policy"],
        "fit_request_ids": sorted(fit.request_ids),
        "tune_request_ids": sorted(tune.request_ids),
    })
    write_jsonl_exclusive(args.output / "metrics.jsonl", history)
    write_jsonl_exclusive(args.output / "request_predictions.jsonl", request_rows)
    write_json_exclusive(args.output / "STAGE_RESULT.json", {
        "schema": SCHEMA, "best_epoch": best_epoch, "metrics": final_metrics,
        "paired_request_bootstrap": bootstrap,
        "positive_gain": final_metrics["gain_h1_h4"] > 0,
        "no_h4_regression": final_metrics["gain_h4"] >= 0,
        "no_horizon_regression_over_0p005": min(horizon_gains) >= -0.005,
        "ready_for_confirmation": ready_for_confirmation,
        "confirmation_opened": False,
    })
    paths = sorted(path for path in args.output.iterdir() if path.name != "SHA256SUMS")
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush(); os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
