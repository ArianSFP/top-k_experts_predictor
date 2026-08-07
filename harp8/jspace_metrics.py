"""Candidate-pool metrics for J-HARP with request-grouped uncertainty."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np
import torch

from .jspace_data import AlignedJCandidateData


def _scores(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "scores"):
        value = getattr(output, "scores")
        if isinstance(value, torch.Tensor):
            return value
    for name in ("scores", "candidate_scores", "final_scores"):
        if name in output:
            return output[name]
    raise KeyError("model output contains no candidate scores")


def _position_band(within: int, rows_per_request: int) -> str:
    fraction = within / max(1, rows_per_request - 1)
    if fraction < 0.25:
        return "q1"
    if fraction < 0.50:
        return "q2"
    if fraction < 0.75:
        return "q3"
    return "q4"


def evaluate_jspace_ranker(
    model: torch.nn.Module,
    data: AlignedJCandidateData,
    *,
    batch_size: int,
    device: str,
    autocast: bool = True,
) -> dict[str, Any]:
    """Evaluate with metric names and denominators derived from pool geometry."""
    native_k = int(data.pool.manifest.get("native_k", 8))
    candidate_count = int(data.pool.candidate_count)
    if not 1 <= native_k <= candidate_count:
        raise ValueError("native_k must lie within the candidate pool")
    coverage_name = f"coverage_at_{candidate_count}"
    request_coverage_name = f"request_macro_coverage_at_{candidate_count}"
    recall_name = f"recall_at_{native_k}"
    base_recall_name = f"base_recall_at_{native_k}"
    exact_name = f"exact_set_at_{native_k}"
    request_recall_name = f"request_macro_recall_at_{native_k}"
    request_base_name = f"request_macro_base_recall_at_{native_k}"
    totals: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    requests: dict[tuple[int, int], defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    layers: dict[tuple[int, int], defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    domains: dict[tuple[str, int], defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    positions: dict[tuple[str, int], defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    model.eval()
    with torch.inference_mode():
        for rows in data.sequential_batches(batch_size):
            batch = data.batch(rows, device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=autocast and str(device).startswith("cuda"),
            ):
                predicted = _scores(model(batch)).float()
            base = batch["candidate_scores"].float()
            membership = batch["target_membership"].bool()
            valid = batch["valid_future"].bool()
            if predicted.shape != membership.shape:
                raise ValueError("predicted score and candidate membership shapes differ")
            candidate_mask = batch.get("candidate_mask")
            if candidate_mask is None:
                candidate_mask = torch.ones_like(membership)
            if candidate_mask.shape != predicted.shape:
                raise ValueError("candidate_mask must match predicted scores")
            candidate_mask = candidate_mask.bool()
            if not bool((candidate_mask.sum(dim=-1) >= native_k).all()):
                raise ValueError("every candidate set must contain at least native_k entries")
            membership = membership & candidate_mask
            minimum = -torch.finfo(predicted.dtype).max
            predicted_top = torch.topk(
                predicted.masked_fill(~candidate_mask, minimum), native_k, dim=-1
            ).indices
            base_top = torch.topk(
                base.masked_fill(~candidate_mask, minimum), native_k, dim=-1
            ).indices
            recall = (
                membership.gather(-1, predicted_top).sum(dim=-1).float() / native_k
            )
            base_recall = (
                membership.gather(-1, base_top).sum(dim=-1).float() / native_k
            )
            coverage = membership.sum(dim=-1).float() / native_k
            exact = recall == 1.0
            recall_np = recall.cpu().numpy()
            base_np = base_recall.cpu().numpy()
            coverage_np = coverage.cpu().numpy()
            exact_np = exact.cpu().numpy()
            valid_np = valid.cpu().numpy()
            for local, pool_row in enumerate(rows.tolist()):
                request_id = int(data.pool.request_ids[pool_row])
                domain = str(data.pool.domains[pool_row])
                within = int(data.pool.within[pool_row])
                band = _position_band(within, data.rows_per_request)
                for column in range(data.pool.horizons):
                    if not valid_np[local, column]:
                        continue
                    horizon = column + 1
                    row_recall = float(recall_np[local, column].mean())
                    row_base = float(base_np[local, column].mean())
                    row_coverage = float(coverage_np[local, column].mean())
                    row_exact = float(exact_np[local, column].mean())
                    totals[horizon]["rows"] += 1
                    totals[horizon]["recall"] += row_recall
                    totals[horizon]["base_recall"] += row_base
                    totals[horizon]["coverage"] += row_coverage
                    totals[horizon]["exact"] += row_exact
                    for collection in (
                        requests[(request_id, horizon)],
                        domains[(domain, horizon)],
                        positions[(band, horizon)],
                    ):
                        collection["rows"] += 1
                        collection["recall"] += row_recall
                        collection["base_recall"] += row_base
                        collection["coverage"] += row_coverage
                        collection["exact"] += row_exact
                    for layer in range(data.pool.layers):
                        collection = layers[(layer, horizon)]
                        collection["rows"] += 1
                        collection["recall"] += float(recall_np[local, column, layer])
                        collection["base_recall"] += float(base_np[local, column, layer])
                        collection["coverage"] += float(coverage_np[local, column, layer])
                        collection["exact"] += float(exact_np[local, column, layer])

    def row(values: defaultdict[str, float]) -> dict[str, float | int]:
        count = max(1.0, values["rows"])
        recall_value = values["recall"] / count
        coverage_value = values["coverage"] / count
        return {
            "rows": int(values["rows"]),
            recall_name: recall_value,
            base_recall_name: values["base_recall"] / count,
            coverage_name: coverage_value,
            "conditional_recovery": recall_value / max(coverage_value, 1e-12),
            exact_name: values["exact"] / count,
        }

    request_rows = [
        {"request_id": request_id, "horizon": horizon, **row(values)}
        for (request_id, horizon), values in sorted(requests.items())
    ]
    horizon_rows: list[dict[str, Any]] = []
    for horizon in range(1, data.pool.horizons + 1):
        request_subset = [value for value in request_rows if value["horizon"] == horizon]
        macro_recall = float(np.mean([value[recall_name] for value in request_subset])) if request_subset else 0.0
        macro_base = float(np.mean([value[base_recall_name] for value in request_subset])) if request_subset else 0.0
        macro_coverage = float(
            np.mean([value[coverage_name] for value in request_subset])
        ) if request_subset else 0.0
        horizon_rows.append({
            "horizon": horizon,
            **row(totals[horizon]),
            request_recall_name: macro_recall,
            request_base_name: macro_base,
            request_coverage_name: macro_coverage,
            "request_macro_conditional_recovery": macro_recall / max(macro_coverage, 1e-12),
        })
    return {
        "horizon_metrics": horizon_rows,
        "request_metrics": request_rows,
        "layer_metrics": [
            {"layer": layer, "horizon": horizon, **row(values)}
            for (layer, horizon), values in sorted(layers.items())
        ],
        "domain_metrics": [
            {"domain": domain, "horizon": horizon, **row(values)}
            for (domain, horizon), values in sorted(domains.items())
        ],
        "position_metrics": [
            {"position_band": band, "horizon": horizon, **row(values)}
            for (band, horizon), values in sorted(positions.items())
        ],
        f"mean_h1_h4_recall_at_{native_k}": float(np.mean([
            value[request_recall_name] for value in horizon_rows[:4]
        ])),
        f"mean_h1_h4_base_recall_at_{native_k}": float(np.mean([
            value[request_base_name] for value in horizon_rows[:4]
        ])),
        f"mean_h1_h4_coverage_at_{candidate_count}": float(np.mean([
            value[request_coverage_name] for value in horizon_rows[:4]
        ])),
        "native_k": native_k,
        "candidate_count": candidate_count,
    }


def paired_request_bootstrap(
    request_metrics: list[dict[str, Any]],
    *,
    replicates: int = 2000,
    seed: int = 42,
    native_k: int = 8,
) -> dict[str, float | int]:
    recall_name = f"recall_at_{native_k}"
    base_recall_name = f"base_recall_at_{native_k}"
    by_request: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for value in request_metrics:
        if 1 <= int(value["horizon"]) <= 4:
            by_request[int(value["request_id"])].append(
                (float(value[recall_name]), float(value[base_recall_name]))
            )
    if not by_request or any(len(values) != 4 for values in by_request.values()):
        raise ValueError("paired H1-H4 bootstrap requires all four horizons per request")
    pairs = np.asarray([
        np.mean(by_request[request_id], axis=0) for request_id in sorted(by_request)
    ], dtype=np.float64)
    differences = pairs[:, 0] - pairs[:, 1]
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        chosen = rng.integers(0, len(differences), len(differences))
        samples[index] = differences[chosen].mean()
    return {
        "requests": len(differences),
        "replicates": replicates,
        "model_point_estimate": float(pairs[:, 0].mean()),
        "base_point_estimate": float(pairs[:, 1].mean()),
        "paired_gain": float(differences.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
    }
