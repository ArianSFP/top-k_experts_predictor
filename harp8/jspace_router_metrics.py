"""Validation metrics for complete-namespace J-space router forecasts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .jspace_router_data import FullRouterForecastData
from .jspace_router_forecaster import JSpaceFullRouterForecaster


_METRIC_NAMES = (
    "recall_at_8",
    "recall_at_16",
    "base_recall_at_8",
    "exact_set_at_8",
    "router_kl",
)


def slot_recall_at_k(
    scores: torch.Tensor,
    target_top8: torch.Tensor,
    k: int = 8,
) -> torch.Tensor:
    """Return per-row/layer overlap divided by the native denominator eight."""

    if scores.ndim < 2 or target_top8.shape != scores.shape[:-1] + (8,):
        raise ValueError("scores and target_top8 have incompatible geometry")
    if not 1 <= k <= scores.shape[-1]:
        raise ValueError("k lies outside the expert namespace")
    # Native BF16 router traces contain exact cutoff ties.  Stable descending
    # sort preserves lower expert IDs because the last axis is expert ordered.
    predicted = torch.argsort(
        scores, dim=-1, descending=True, stable=True
    )[..., :k]
    overlap = (
        (predicted.unsqueeze(-1) == target_top8.unsqueeze(-2))
        .any(dim=-2)
        .sum(dim=-1)
    )
    return overlap.float() / 8.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _sum_by_code(
    values: np.ndarray,
    codes: np.ndarray,
    groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate float32 event values into float64 groups in event order."""

    values = np.asarray(values)
    codes = np.asarray(codes, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(_METRIC_NAMES):
        raise ValueError("metric events must have shape [N, 5]")
    if codes.shape != (len(values),):
        raise ValueError("metric group codes must have shape [N]")
    if groups < 0 or (
        len(codes) and ((codes < 0).any() or (codes >= groups).any())
    ):
        raise ValueError("metric group code lies outside the requested range")
    sums = np.zeros((groups, len(_METRIC_NAMES)), dtype=np.float64)
    counts = np.zeros(groups, dtype=np.int64)
    # add.at is deliberately used instead of a parallel/grouped reduction. It
    # visits repeated indices in input order, matching the legacy Python-float
    # accumulation while doing the grouping in NumPy rather than Python loops.
    np.add.at(sums, codes, values.astype(np.float64, copy=False))
    np.add.at(counts, codes, 1)
    return sums, counts


def _values_dict(
    sums: np.ndarray,
    count: int,
) -> defaultdict[str, float]:
    result: defaultdict[str, float] = defaultdict(float)
    result["rows"] = float(count)
    for index, name in enumerate(_METRIC_NAMES):
        result[name] = float(sums[index])
    return result


def _aggregate_metric_events(
    row_values: np.ndarray,
    layer_values: np.ndarray,
    event_horizons: np.ndarray,
    event_request_ids: np.ndarray,
    event_domains: np.ndarray,
    *,
    horizons: int,
    layers: int,
) -> tuple[
    dict[int, defaultdict[str, float]],
    dict[tuple[int, int], defaultdict[str, float]],
    dict[tuple[int, int], defaultdict[str, float]],
    dict[tuple[str, int], defaultdict[str, float]],
]:
    """Vectorize the legacy row/horizon/layer metric aggregation."""

    row_values = np.asarray(row_values)
    layer_values = np.asarray(layer_values)
    event_horizons = np.asarray(event_horizons, dtype=np.int64)
    event_request_ids = np.asarray(event_request_ids, dtype=np.int64)
    event_domains = np.asarray(event_domains, dtype=str)
    events = len(row_values)
    if row_values.shape != (events, len(_METRIC_NAMES)):
        raise ValueError("row metric values must have shape [N, 5]")
    if layer_values.shape != (events, layers, len(_METRIC_NAMES)):
        raise ValueError("layer metric values must have shape [N, L, 5]")
    for name, values in (
        ("horizon", event_horizons),
        ("request ID", event_request_ids),
        ("domain", event_domains),
    ):
        if values.shape != (events,):
            raise ValueError(f"event {name} values must have shape [N]")
    if events and (
        (event_horizons < 1).any() or (event_horizons > horizons).any()
    ):
        raise ValueError("event horizon lies outside the configured range")

    totals: dict[int, defaultdict[str, float]] = {
        horizon: defaultdict(float) for horizon in range(1, horizons + 1)
    }
    requests: dict[tuple[int, int], defaultdict[str, float]] = {}
    layer_totals: dict[tuple[int, int], defaultdict[str, float]] = {}
    domains: dict[tuple[str, int], defaultdict[str, float]] = {}
    if not events:
        return totals, requests, layer_totals, domains

    horizon_codes = event_horizons - 1
    sums, counts = _sum_by_code(row_values, horizon_codes, horizons)
    for horizon_index in range(horizons):
        if counts[horizon_index]:
            totals[horizon_index + 1] = _values_dict(
                sums[horizon_index], int(counts[horizon_index])
            )

    unique_requests, request_inverse = np.unique(
        event_request_ids, return_inverse=True
    )
    request_codes = request_inverse * horizons + horizon_codes
    sums, counts = _sum_by_code(
        row_values, request_codes, len(unique_requests) * horizons
    )
    for request_index, request_id in enumerate(unique_requests.tolist()):
        for horizon_index in range(horizons):
            code = request_index * horizons + horizon_index
            if counts[code]:
                requests[(int(request_id), horizon_index + 1)] = _values_dict(
                    sums[code], int(counts[code])
                )

    unique_domains, domain_inverse = np.unique(event_domains, return_inverse=True)
    domain_codes = domain_inverse * horizons + horizon_codes
    sums, counts = _sum_by_code(
        row_values, domain_codes, len(unique_domains) * horizons
    )
    for domain_index, domain in enumerate(unique_domains.tolist()):
        for horizon_index in range(horizons):
            code = domain_index * horizons + horizon_index
            if counts[code]:
                domains[(str(domain), horizon_index + 1)] = _values_dict(
                    sums[code], int(counts[code])
                )

    layer_ids = np.arange(layers, dtype=np.int64)
    layer_codes = (
        np.repeat(horizon_codes, layers) * layers
        + np.tile(layer_ids, events)
    )
    sums, counts = _sum_by_code(
        layer_values.reshape(events * layers, len(_METRIC_NAMES)),
        layer_codes,
        horizons * layers,
    )
    for layer in range(layers):
        for horizon_index in range(horizons):
            code = horizon_index * layers + layer
            if counts[code]:
                layer_totals[(layer, horizon_index + 1)] = _values_dict(
                    sums[code], int(counts[code])
                )
    return totals, requests, layer_totals, domains


def evaluate_full_router_forecaster(
    model: JSpaceFullRouterForecaster,
    data: FullRouterForecastData,
    *,
    batch_size: int,
    device: str,
    autocast: bool = True,
) -> dict[str, Any]:
    """Evaluate only the supplied train/validation view, grouped by request."""

    if data.split not in {"train", "validation"}:
        raise PermissionError("full-router evaluator cannot access a sealed test split")
    row_chunks: list[np.ndarray] = []
    layer_chunks: list[np.ndarray] = []
    horizon_chunks: list[np.ndarray] = []
    request_chunks: list[np.ndarray] = []
    domain_chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for rows in data.sequential_batches(batch_size):
            batch = data.batch(rows, device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=autocast and str(device).startswith("cuda"),
            ):
                output = model(batch)
            scores = output.future_router_scores.float()
            base = output.base_router_scores.float()
            target = batch["target_top8"].long()
            valid = batch["valid_future"].bool()
            recall8 = slot_recall_at_k(scores, target, 8)
            recall16 = slot_recall_at_k(scores, target, min(16, data.experts))
            base8 = slot_recall_at_k(base, target, 8)
            exact = recall8 == 1.0
            teacher = batch["teacher_router_scores"].float()
            kl = F.kl_div(
                torch.log_softmax(scores, dim=-1),
                torch.softmax(teacher, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            # Compute the legacy per-row means on-device, pack them beside the
            # per-layer values, and transfer the entire evaluation payload once.
            # This replaces O(B * H * L) scalar synchronizations with one batch
            # synchronization without changing any metric definitions.
            layer_metric_tensor = torch.stack(
                (recall8, recall16, base8, exact.float(), kl), dim=-1
            )
            # Keep each reduction on its original contiguous tensor. Besides
            # matching the former implementation exactly, this avoids changing
            # CUDA reduction order merely because the five metrics are packed.
            row_metric_tensor = torch.stack(
                (
                    recall8.mean(dim=2),
                    recall16.mean(dim=2),
                    base8.mean(dim=2),
                    exact.float().mean(dim=2),
                    kl.mean(dim=2),
                ),
                dim=-1,
            ).unsqueeze(2)
            packed_metrics = torch.cat(
                (layer_metric_tensor, row_metric_tensor), dim=2
            )
            packed_valid = valid[:, :, None, None].to(packed_metrics.dtype).expand(
                -1, -1, data.layers + 1, 1
            )
            host_payload = torch.cat(
                (packed_metrics, packed_valid), dim=-1
            ).cpu().numpy()
            valid_np = host_payload[:, :, data.layers, -1].astype(bool)
            local_indices, columns = np.nonzero(valid_np)
            if not len(local_indices):
                continue
            request_ids, domain_names, _within = data.metadata(rows)
            row_chunks.append(
                host_payload[
                    local_indices, columns, data.layers, : len(_METRIC_NAMES)
                ]
            )
            layer_chunks.append(
                host_payload[
                    local_indices, columns, : data.layers, : len(_METRIC_NAMES)
                ]
            )
            horizon_chunks.append(columns.astype(np.int64, copy=False) + 1)
            request_chunks.append(
                np.asarray(request_ids, dtype=np.int64)[local_indices]
            )
            domain_chunks.append(np.asarray(domain_names, dtype=str)[local_indices])

    if row_chunks:
        event_rows = np.concatenate(row_chunks, axis=0)
        event_layers = np.concatenate(layer_chunks, axis=0)
        event_horizons = np.concatenate(horizon_chunks, axis=0)
        event_requests = np.concatenate(request_chunks, axis=0)
        event_domains = np.concatenate(domain_chunks, axis=0)
    else:
        event_rows = np.empty((0, len(_METRIC_NAMES)), dtype=np.float32)
        event_layers = np.empty(
            (0, data.layers, len(_METRIC_NAMES)), dtype=np.float32
        )
        event_horizons = np.empty(0, dtype=np.int64)
        event_requests = np.empty(0, dtype=np.int64)
        event_domains = np.empty(0, dtype=str)
    totals, requests, layers, domains = _aggregate_metric_events(
        event_rows,
        event_layers,
        event_horizons,
        event_requests,
        event_domains,
        horizons=data.horizons,
        layers=data.layers,
    )

    def summarize(values: defaultdict[str, float]) -> dict[str, float | int]:
        count = max(1.0, values["rows"])
        return {
            "rows": int(values["rows"]),
            "recall_at_8": values["recall_at_8"] / count,
            "recall_at_16": values["recall_at_16"] / count,
            "base_recall_at_8": values["base_recall_at_8"] / count,
            "gain_at_8": (
                values["recall_at_8"] - values["base_recall_at_8"]
            )
            / count,
            "exact_set_at_8": values["exact_set_at_8"] / count,
            "router_kl": values["router_kl"] / count,
        }

    request_rows = [
        {"request_id": request, "horizon": horizon, **summarize(values)}
        for (request, horizon), values in sorted(requests.items())
    ]
    horizon_rows: list[dict[str, Any]] = []
    for horizon in range(1, data.horizons + 1):
        matching = [row for row in request_rows if row["horizon"] == horizon]
        row = {
            "split": data.split,
            "horizon": horizon,
            **summarize(totals[horizon]),
            "requests": len(matching),
            "request_macro_recall_at_8": _mean(
                [float(value["recall_at_8"]) for value in matching]
            ),
            "request_macro_recall_at_16": _mean(
                [float(value["recall_at_16"]) for value in matching]
            ),
            "request_macro_base_recall_at_8": _mean(
                [float(value["base_recall_at_8"]) for value in matching]
            ),
        }
        row["request_macro_gain_at_8"] = (
            row["request_macro_recall_at_8"]
            - row["request_macro_base_recall_at_8"]
        )
        horizon_rows.append(row)

    first_four = horizon_rows[:4]
    return {
        "schema": "harp8_jspace_full_router_validation_metrics_v1",
        "split": data.split,
        "horizon_metrics": horizon_rows,
        "request_metrics": request_rows,
        "layer_metrics": [
            {"layer": layer, "horizon": horizon, **summarize(values)}
            for (layer, horizon), values in sorted(layers.items())
        ],
        "domain_metrics": [
            {"domain": domain, "horizon": horizon, **summarize(values)}
            for (domain, horizon), values in sorted(domains.items())
        ],
        "mean_h1_h4_recall_at_8": _mean(
            [float(row["request_macro_recall_at_8"]) for row in first_four]
        ),
        "mean_h1_h4_base_recall_at_8": _mean(
            [float(row["request_macro_base_recall_at_8"]) for row in first_four]
        ),
        "mean_h1_h8_recall_at_8": _mean(
            [float(row["request_macro_recall_at_8"]) for row in horizon_rows]
        ),
        "h2_request_macro_recall_at_8": float(
            horizon_rows[1]["request_macro_recall_at_8"]
        ),
        "sealed_test_accessed": False,
    }


__all__ = ["evaluate_full_router_forecaster", "slot_recall_at_k"]
