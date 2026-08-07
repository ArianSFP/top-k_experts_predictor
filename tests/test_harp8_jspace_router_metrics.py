from __future__ import annotations

from collections import defaultdict
from inspect import getsource
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from harp8.jspace_router_metrics import (
    _METRIC_NAMES,
    _aggregate_metric_events,
    evaluate_full_router_forecaster,
)


def _legacy_aggregate(
    row_values: np.ndarray,
    layer_values: np.ndarray,
    horizons: np.ndarray,
    request_ids: np.ndarray,
    domains: np.ndarray,
    *,
    horizon_count: int,
    layer_count: int,
):
    totals = {
        horizon: defaultdict(float) for horizon in range(1, horizon_count + 1)
    }
    requests = defaultdict(lambda: defaultdict(float))
    layers = defaultdict(lambda: defaultdict(float))
    domain_totals = defaultdict(lambda: defaultdict(float))
    for event in range(len(row_values)):
        horizon = int(horizons[event])
        for collection in (
            totals[horizon],
            requests[(int(request_ids[event]), horizon)],
            domain_totals[(str(domains[event]), horizon)],
        ):
            collection["rows"] += 1
            for metric, value in zip(_METRIC_NAMES, row_values[event], strict=True):
                collection[metric] += float(value)
        for layer in range(layer_count):
            collection = layers[(layer, horizon)]
            collection["rows"] += 1
            for metric, value in zip(
                _METRIC_NAMES, layer_values[event, layer], strict=True
            ):
                collection[metric] += float(value)
    return totals, requests, layers, domain_totals


def _plain(collection):
    return {key: dict(values) for key, values in collection.items()}


def test_vectorized_aggregation_exactly_matches_legacy_reference() -> None:
    rng = np.random.default_rng(711)
    layer_values = rng.normal(size=(9, 3, 5)).astype(np.float32)
    row_values = layer_values.mean(axis=1, dtype=np.float32)
    horizons = np.asarray([1, 2, 1, 3, 2, 1, 3, 2, 1], dtype=np.int64)
    request_ids = np.asarray([7, 7, 3, 3, 7, 9, 9, 3, 7], dtype=np.int64)
    domains = np.asarray(
        ["wiki", "wiki", "code", "code", "wiki", "math", "math", "code", "wiki"]
    )
    reference = _legacy_aggregate(
        row_values,
        layer_values,
        horizons,
        request_ids,
        domains,
        horizon_count=3,
        layer_count=3,
    )
    actual = _aggregate_metric_events(
        row_values,
        layer_values,
        horizons,
        request_ids,
        domains,
        horizons=3,
        layers=3,
    )
    for expected_collection, actual_collection in zip(reference, actual, strict=True):
        assert _plain(actual_collection) == _plain(expected_collection)


def _scores_for_overlap(overlap: int, experts: int = 16) -> torch.Tensor:
    selected = list(range(overlap)) + list(range(8, 8 + (8 - overlap)))
    remaining = [expert for expert in range(experts) if expert not in selected]
    order = selected + remaining
    scores = torch.empty(experts, dtype=torch.float32)
    for rank, expert in enumerate(order):
        scores[expert] = float(experts - rank)
    return scores


class _MetricData:
    split = "validation"
    rows = 3
    horizons = 2
    layers = 2
    experts = 16

    def __init__(self) -> None:
        overlaps = np.asarray(
            [
                [[8, 7], [6, 5]],
                [[4, 6], [8, 8]],
                [[7, 8], [6, 7]],
            ]
        )
        self.scores = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack(
                            [_scores_for_overlap(int(value)) for value in horizon]
                        )
                        for horizon in row
                    ]
                )
                for row in overlaps
            ]
        )
        self.base = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack([_scores_for_overlap(6) for _ in horizon])
                        for horizon in row
                    ]
                )
                for row in overlaps
            ]
        )
        teacher = torch.arange(16, 0, -1, dtype=torch.float32)
        self.teacher = teacher.expand(3, 2, 2, 16).clone()
        self.target = torch.arange(8).expand(3, 2, 2, 8).clone()
        self.valid = torch.tensor([[True, True], [True, False], [True, True]])
        self.request_ids = np.asarray([20, 10, 20])
        self.domains = np.asarray(["zeta", "alpha", "zeta"])

    def sequential_batches(self, batch_size: int):
        for start in range(0, self.rows, batch_size):
            yield np.arange(start, min(self.rows, start + batch_size))

    def batch(self, rows: np.ndarray, device: str):
        return {
            "model_scores": self.scores[rows].to(device),
            "base_router_scores": self.base[rows].to(device),
            "teacher_router_scores": self.teacher[rows].to(device),
            "target_top8": self.target[rows].to(device),
            "valid_future": self.valid[rows].to(device),
        }

    def metadata(self, rows: np.ndarray):
        return (
            self.request_ids[rows],
            self.domains[rows],
            np.asarray(rows),
        )


class _MetricModel(torch.nn.Module):
    def forward(self, batch):
        return SimpleNamespace(
            future_router_scores=batch["model_scores"],
            base_router_scores=batch["base_router_scores"],
        )


def test_evaluator_preserves_request_layer_domain_metrics_and_order() -> None:
    metrics = evaluate_full_router_forecaster(
        _MetricModel(), _MetricData(), batch_size=2, device="cpu", autocast=False
    )
    assert metrics["schema"] == "harp8_jspace_full_router_validation_metrics_v1"
    assert [
        (row["request_id"], row["horizon"])
        for row in metrics["request_metrics"]
    ] == [(10, 1), (20, 1), (20, 2)]
    assert [
        (row["layer"], row["horizon"]) for row in metrics["layer_metrics"]
    ] == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert [
        (row["domain"], row["horizon"]) for row in metrics["domain_metrics"]
    ] == [("alpha", 1), ("zeta", 1), ("zeta", 2)]

    request = {
        (row["request_id"], row["horizon"]): row
        for row in metrics["request_metrics"]
    }
    assert request[(10, 1)]["recall_at_8"] == pytest.approx(0.625)
    assert request[(20, 1)]["recall_at_8"] == pytest.approx(0.9375)
    assert request[(20, 2)]["recall_at_8"] == pytest.approx(0.75)
    layers = {
        (row["layer"], row["horizon"]): row for row in metrics["layer_metrics"]
    }
    assert layers[(0, 1)]["recall_at_8"] == pytest.approx((1 + 0.5 + 0.875) / 3)
    assert layers[(1, 1)]["recall_at_8"] == pytest.approx((0.875 + 0.75 + 1) / 3)


def test_evaluator_source_has_exactly_one_cpu_transfer_site() -> None:
    source = getsource(evaluate_full_router_forecaster)
    assert source.count(".cpu()") == 1
    assert ".cpu())" not in source
