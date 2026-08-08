"""Validation-only runner for the unchanged HARP8 C64 reranker baseline.

This versioned entry point imports the original model, objective, optimizer,
and evaluation implementation from :mod:`harp8.reranker`.  Its only semantic
correction is metric reporting: full candidate-pool coverage and actual
coverage@16 are measured and named separately.  Wide pools are always trained
with the original diagnostic ``force`` path because their true fixed-16 gate
does not pass.

There is deliberately no test-pool argument or test-split escape hatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import harp8.reranker as legacy
from harp8.candidates import POOL_SCHEMA
from harp8.train import sha256_file


REPORT_SCHEMA = "harp8_c64_baseline_metric_report_v1"
RUNNER_SCHEMA = "harp8_c64_baseline_metricfix_v1"


def _checked_pool(root: Path, expected_split: str) -> legacy.CandidatePool:
    """Open only an explicitly declared train or validation pool."""

    if expected_split not in {"train", "validation"}:
        raise ValueError("this runner accepts only train and validation splits")
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != POOL_SCHEMA:
        raise ValueError(f"{manifest_path} has an incompatible schema")
    if manifest.get("split") != expected_split:
        raise ValueError(
            f"{manifest_path} declares {manifest.get('split')!r}, not "
            f"{expected_split!r}"
        )
    if bool(manifest.get("allow_test", False)) or manifest.get("split") == "test":
        raise ValueError("sealed test data is forbidden in this baseline runner")
    pool = legacy.CandidatePool(root)
    if pool.candidate_count != 64:
        raise ValueError(
            f"the C64 baseline requires exactly 64 candidates, got "
            f"{pool.candidate_count}"
        )
    return pool


def _legacy_micro_coverage(pool: legacy.CandidatePool, width: int) -> list[float]:
    """Reproduce the historical row/layer aggregation at an explicit width."""

    width = min(int(width), pool.candidate_count)
    result: list[float] = []
    for horizon in range(min(4, pool.horizons)):
        membership = np.asarray(
            pool.target_membership[:, horizon, :, :width], dtype=np.uint8
        )
        row_layer = membership.sum(axis=-1, dtype=np.int16) / 8.0
        valid = np.asarray(pool.valid_future[:, horizon], dtype=np.uint8).astype(bool)
        selected = row_layer[valid]
        result.append(float(selected.mean()) if selected.size else 0.0)
    return result


def corrected_candidate_gate(pool: legacy.CandidatePool) -> dict[str, Any]:
    """Return separately named true fixed-16 and full-pool coverage gates."""

    fixed16 = _legacy_micro_coverage(pool, 16)
    pool_width = _legacy_micro_coverage(pool, pool.candidate_count)
    fixed16_mean = float(np.mean(fixed16)) if fixed16 else 0.0
    fixed16_h4 = fixed16[3] if len(fixed16) >= 4 else 0.0
    pool_mean = float(np.mean(pool_width)) if pool_width else 0.0
    pool_h4 = pool_width[3] if len(pool_width) >= 4 else 0.0
    fixed16_passes = bool(fixed16_mean >= 0.95 and fixed16_h4 >= 0.93)
    pool_passes = bool(pool_mean >= 0.95 and pool_h4 >= 0.93)
    return {
        "aggregation": "valid_position_layer_micro",
        "pool_candidate_count": int(pool.candidate_count),
        "mean_h1_h4_pool_coverage": pool_mean,
        "h4_pool_coverage": pool_h4,
        "pool_gate_passes": pool_passes,
        "mean_h1_h4_coverage_at_16": fixed16_mean,
        "h4_coverage_at_16": fixed16_h4,
        "fixed16_gate_passes": fixed16_passes,
        # Preserve the legacy caller's interface, now with its literal meaning.
        "passes": fixed16_passes,
    }


def _request_macro_widths(
    pool: legacy.CandidatePool, widths: Sequence[int]
) -> dict[int, list[float]]:
    """Compute request-macro target coverage for each explicit prefix width."""

    request_ids = np.asarray(pool.request_ids, dtype=np.int64)
    _, inverse = np.unique(request_ids, return_inverse=True)
    request_count = int(inverse.max()) + 1
    result: dict[int, list[float]] = {int(width): [] for width in widths}
    for horizon in range(pool.horizons):
        valid = np.asarray(pool.valid_future[:, horizon], dtype=np.uint8).astype(bool)
        selected_inverse = inverse[valid]
        row_counts = np.bincount(selected_inverse, minlength=request_count)
        for raw_width in widths:
            width = min(int(raw_width), pool.candidate_count)
            membership = np.asarray(
                pool.target_membership[:, horizon, :, :width], dtype=np.uint8
            )
            # Average all 40 layers within a position before grouping positions
            # into complete requests.
            position = membership.sum(axis=-1, dtype=np.int16).mean(axis=-1) / 8.0
            request_sums = np.bincount(
                selected_inverse,
                weights=position[valid].astype(np.float64, copy=False),
                minlength=request_count,
            )
            present = row_counts > 0
            request_values = request_sums[present] / row_counts[present]
            result[int(raw_width)].append(
                float(request_values.mean()) if request_values.size else 0.0
            )
    return result


def coverage_report(pool: legacy.CandidatePool) -> dict[str, Any]:
    """Report generator top-8, literal top-16, and full C64 oracle coverage."""

    widths = (8, 16, pool.candidate_count)
    values = _request_macro_widths(pool, widths)
    metrics: dict[str, Any] = {}
    for width in widths:
        horizons = values[int(width)]
        metrics[f"coverage_at_{int(width)}"] = {
            "per_horizon": horizons,
            "mean_h1_h4": float(np.mean(horizons[:4])),
            "h4": float(horizons[3]),
        }
    return {
        "aggregation": "complete_request_macro_after_layer_and_position_mean",
        "candidate_count": int(pool.candidate_count),
        "metrics": metrics,
        "legacy_gate_aggregation": corrected_candidate_gate(pool),
    }


def _paired_interval(
    candidate: np.ndarray,
    anchor: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        anchor, dtype=np.float64
    )
    if delta.ndim != 1 or delta.size == 0:
        raise ValueError("paired bootstrap requires a non-empty vector")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, delta.size, size=(replicates, delta.size))
    means = delta[draws].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def detailed_model_report(
    model: legacy.Fixed16SetReranker,
    pool: legacy.CandidatePool,
    *,
    batch_size: int,
    device: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate generator, reranker, and candidate ceilings by request."""

    unique_requests, inverse = np.unique(
        np.asarray(pool.request_ids, dtype=np.int64), return_inverse=True
    )
    requests = len(unique_requests)
    sums = {
        name: np.zeros((requests, pool.horizons), dtype=np.float64)
        for name in ("generator", "reranker", "top16", "pool_oracle")
    }
    counts = np.zeros((requests, pool.horizons), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, pool.rows, batch_size):
            rows = np.arange(start, min(pool.rows, start + batch_size), dtype=np.int64)
            batch = pool.batch(rows, device)
            scores = model(batch)
            membership = batch["target_membership"].bool()
            order = torch.topk(scores, 8, dim=-1).indices
            position_values = {
                "generator": membership[..., :8].sum(dim=-1).float().mean(dim=-1)
                / 8.0,
                "reranker": membership.gather(-1, order)
                .sum(dim=-1)
                .float()
                .mean(dim=-1)
                / 8.0,
                "top16": membership[..., :16].sum(dim=-1).float().mean(dim=-1)
                / 8.0,
                "pool_oracle": membership.sum(dim=-1).float().mean(dim=-1) / 8.0,
            }
            valid = batch["valid_future"].bool().cpu().numpy()
            cpu_values = {
                name: value.cpu().numpy() for name, value in position_values.items()
            }
            for local, global_row in enumerate(rows.tolist()):
                request = int(inverse[global_row])
                for horizon in range(pool.horizons):
                    if not valid[local, horizon]:
                        continue
                    counts[request, horizon] += 1
                    for name in sums:
                        sums[name][request, horizon] += float(
                            cpu_values[name][local, horizon]
                        )

    values: dict[str, np.ndarray] = {}
    present = counts > 0
    for name, total in sums.items():
        output = np.full(total.shape, np.nan, dtype=np.float64)
        output[present] = total[present] / counts[present]
        values[name] = output

    horizons: list[dict[str, Any]] = []
    for horizon in range(pool.horizons):
        mask = present[:, horizon]
        generator = values["generator"][mask, horizon]
        reranker = values["reranker"][mask, horizon]
        oracle = values["pool_oracle"][mask, horizon]
        generator_mean = float(generator.mean())
        reranker_mean = float(reranker.mean())
        oracle_mean = float(oracle.mean())
        available_gap = oracle_mean - generator_mean
        horizons.append(
            {
                "horizon": horizon + 1,
                "request_count": int(mask.sum()),
                "generator_recall_at_8": generator_mean,
                "reranker_recall_at_8": reranker_mean,
                "coverage_at_16": float(values["top16"][mask, horizon].mean()),
                "pool_coverage_at_64": oracle_mean,
                "recall_over_pool_oracle": (
                    reranker_mean / oracle_mean if oracle_mean else 0.0
                ),
                "incremental_oracle_gap_recovery": (
                    (reranker_mean - generator_mean) / available_gap
                    if available_gap > 0.0
                    else 0.0
                ),
                "paired_bootstrap_vs_generator": _paired_interval(
                    reranker,
                    generator,
                    replicates=bootstrap_replicates,
                    seed=seed + horizon,
                ),
            }
        )

    common = present[:, :4].all(axis=1)
    mean_values = {
        name: values[name][common, :4].mean(axis=1) for name in values
    }
    generator_mean = float(mean_values["generator"].mean())
    reranker_mean = float(mean_values["reranker"].mean())
    oracle_mean = float(mean_values["pool_oracle"].mean())
    return {
        "schema": REPORT_SCHEMA,
        "aggregation": "complete_request_macro_after_layer_and_position_mean",
        "bootstrap_replicates": int(bootstrap_replicates),
        "horizon_metrics": horizons,
        "mean_h1_h4": {
            "request_count": int(common.sum()),
            "generator_recall_at_8": generator_mean,
            "reranker_recall_at_8": reranker_mean,
            "coverage_at_16": float(mean_values["top16"].mean()),
            "pool_coverage_at_64": oracle_mean,
            "recall_over_pool_oracle": (
                reranker_mean / oracle_mean if oracle_mean else 0.0
            ),
            "incremental_oracle_gap_recovery": (
                (reranker_mean - generator_mean) / (oracle_mean - generator_mean)
                if oracle_mean > generator_mean
                else 0.0
            ),
            "paired_bootstrap_vs_generator": _paired_interval(
                mean_values["reranker"],
                mean_values["generator"],
                replicates=bootstrap_replicates,
                seed=seed + 100,
            ),
        },
    }


def _load_best_model(
    checkpoint: Path, pool: legacy.CandidatePool, device: str
) -> tuple[legacy.Fixed16SetReranker, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = legacy.Fixed16SetReranker(**payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    if model.candidate_count != pool.candidate_count:
        raise ValueError("checkpoint and validation pool candidate widths disagree")
    return model, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pool", type=Path)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--report-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_pool = _checked_pool(args.validation_pool, "validation")
    initial_report = {
        "schema": REPORT_SCHEMA,
        "runner_schema": RUNNER_SCHEMA,
        "validation_manifest": {
            "path": str(args.validation_pool / "manifest.json"),
            "sha256": sha256_file(args.validation_pool / "manifest.json"),
        },
        "candidate_coverage": coverage_report(validation_pool),
    }
    if args.report_only:
        print(json.dumps(initial_report, indent=2, sort_keys=True))
        return 0
    if args.train_pool is None or args.output_dir is None:
        raise ValueError("training requires --train-pool and --output-dir")
    train_pool = _checked_pool(args.train_pool, "train")
    if train_pool.candidate_count != validation_pool.candidate_count:
        raise ValueError("train and validation candidate widths disagree")

    # The original trainer resolves candidate_gate from its module globals.
    # Replacing only that reporting function keeps its model, loss, optimizer,
    # update order, evaluation, checkpoint selection, and RNG behavior intact.
    legacy.candidate_gate = corrected_candidate_gate
    manifest = legacy.train_reranker(
        train_pool,
        validation_pool,
        args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        force=True,
    )
    model, checkpoint = _load_best_model(
        args.output_dir / "best.pt", validation_pool, args.device
    )
    final_report = detailed_model_report(
        model,
        validation_pool,
        batch_size=args.batch_size,
        device=args.device,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    final_report.update(
        {
            "runner_schema": RUNNER_SCHEMA,
            "best_epoch": int(checkpoint["epoch"]),
            "training_manifest": manifest,
            "train_manifest_sha256": sha256_file(args.train_pool / "manifest.json"),
            "validation_manifest_sha256": sha256_file(
                args.validation_pool / "manifest.json"
            ),
        }
    )
    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
