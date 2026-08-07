"""Endpoint metrics with request-grouped aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .data import CompactHARPData


CANDIDATE_COUNTS = (8, 12, 16, 24, 32)


def model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    names = (
        "route_history",
        "route_available",
        "target_states",
        "target_state_available",
        "mtp_states",
        "mtp_router_logits",
        "mtp_metadata",
        "mtp_depth_ids",
        "mtp_available",
        "mtp_vocab_features",
        "within_request",
    )
    return {name: batch[name] for name in names if name in batch}


def slot_overlap(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        (predicted.unsqueeze(-1) == target.unsqueeze(-2))
        .any(dim=-1)
        .sum(dim=-1)
    )


@dataclass
class EvaluationResult:
    horizon_metrics: list[dict[str, Any]]
    request_metrics: list[dict[str, Any]]
    layer_metrics: list[dict[str, Any]]
    domain_metrics: list[dict[str, Any]]

    @property
    def h2_request_macro_recall(self) -> float:
        row = next(
            value for value in self.horizon_metrics if value["horizon"] == 2
        )
        return float(row["request_macro_slot_recall_at_8"])

    def mean_h1_h4(self, candidate_count: int = 8) -> float:
        rows = [
            row for row in self.horizon_metrics if 1 <= int(row["horizon"]) <= 4
        ]
        return float(
            np.mean(
                [
                    row[f"request_macro_slot_recall_at_{candidate_count}"]
                    for row in rows
                ]
            )
        )


def evaluate_split(
    model: torch.nn.Module,
    data: CompactHARPData,
    split: str,
    *,
    batch_size: int,
    device: str,
    allow_test: bool = False,
) -> EvaluationResult:
    horizons = data.config.horizons
    candidate_counts = tuple(
        count for count in CANDIDATE_COUNTS if count <= data.config.experts
    )
    if 8 not in candidate_counts:
        raise ValueError("evaluation requires an expert namespace with native top-8")
    totals: dict[int, defaultdict[str, float]] = {
        horizon: defaultdict(float) for horizon in range(1, horizons + 1)
    }
    requests: dict[tuple[int, int], defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    layers: dict[tuple[int, int], defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    domains: dict[tuple[str, int], defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    model.eval()
    with torch.inference_mode():
        for indices in data.sequential_batches(
            split, batch_size, allow_test=allow_test
        ):
            batch = data.batch(indices, device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                outputs = model(**model_inputs(batch))
            predicted_scores = outputs["future_router_scores"].float()
            top32 = torch.topk(
                predicted_scores,
                max(candidate_counts),
                dim=-1,
                sorted=True,
            ).indices
            teacher = batch["teacher_router_scores"].float()
            teacher_probability = torch.softmax(teacher / 2.0, dim=-1)
            kl = F.kl_div(
                torch.log_softmax(predicted_scores / 2.0, dim=-1),
                teacher_probability,
                reduction="none",
            ).sum(dim=-1) * 4.0
            request_ids, request_domains, _within = data.metadata(indices)
            for column in range(horizons):
                horizon = column + 1
                valid = batch["valid_future"][:, column]
                if not valid.any():
                    continue
                target = batch["target_top8"][:, column]
                overlap_by_count = {
                    count: slot_overlap(top32[:, column, :, :count], target).float()
                    for count in candidate_counts
                }
                membership = torch.zeros_like(
                    predicted_scores[:, column], dtype=torch.bool
                )
                membership.scatter_(-1, target, True)
                pool_count = min(16, max(candidate_counts))
                negative_boundary_rank = pool_count - target.shape[-1] + 1
                candidate_negative = torch.topk(
                    predicted_scores[:, column].masked_fill(membership, -torch.inf),
                    negative_boundary_rank,
                    dim=-1,
                    sorted=True,
                ).values[..., -1]
                weakest_positive = predicted_scores[:, column].gather(
                    -1, target
                ).min(dim=-1).values
                candidate_margin16 = weakest_positive - candidate_negative
                contained16 = overlap_by_count[pool_count] == 8
                missing16 = 8.0 - overlap_by_count[pool_count]
                exact = overlap_by_count[8] == 8
                valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
                selected_kl = kl[valid, column]
                selected_exact = exact[valid].float()
                valid_count = int(valid.sum())
                totals[horizon]["source_rows"] += valid_count
                totals[horizon]["router_kl"] += float(
                    selected_kl.mean(dim=-1).sum()
                )
                totals[horizon]["exact_set_at_8"] += float(
                    selected_exact.mean(dim=-1).sum()
                )
                totals[horizon]["all_eight_contained_at_16"] += float(
                    contained16[valid].float().mean(dim=-1).sum()
                )
                totals[horizon]["mean_missing_at_16"] += float(
                    missing16[valid].mean(dim=-1).sum()
                )
                totals[horizon]["candidate_margin_at_16"] += float(
                    candidate_margin16[valid].mean(dim=-1).sum()
                )
                selected_overlap = {
                    count: overlap[valid]
                    for count, overlap in overlap_by_count.items()
                }
                for count, overlap in selected_overlap.items():
                    totals[horizon][f"slot_recall_at_{count}"] += float(
                        (overlap.mean(dim=-1) / 8.0).sum()
                    )
                source_kl_values = (
                    kl[:, column].mean(dim=-1).detach().cpu().numpy()
                )
                source_exact_values = (
                    exact.float().mean(dim=-1).detach().cpu().numpy()
                )
                source_contained16_values = (
                    contained16.float().mean(dim=-1).detach().cpu().numpy()
                )
                source_missing16_values = (
                    missing16.mean(dim=-1).detach().cpu().numpy()
                )
                source_margin16_values = (
                    candidate_margin16.mean(dim=-1).detach().cpu().numpy()
                )
                source_recall_values = {
                    count: (overlap.mean(dim=-1) / 8.0).detach().cpu().numpy()
                    for count, overlap in overlap_by_count.items()
                }
                layer_recall_sums = {
                    count: overlap.sum(dim=0) / 8.0
                    for count, overlap in selected_overlap.items()
                }
                layer_contained_sum = contained16[valid].float().sum(dim=0)
                layer_missing_sum = missing16[valid].sum(dim=0)
                layer_margin_sum = candidate_margin16[valid].sum(dim=0)
                layer_kl_sum = selected_kl.sum(dim=0)
                for layer in range(data.config.layers):
                    layer_key = (horizon, layer)
                    layers[layer_key]["source_rows"] += valid_count
                    for count, values in layer_recall_sums.items():
                        layers[layer_key][f"slot_recall_at_{count}"] += float(
                            values[layer]
                        )
                    layers[layer_key]["all_eight_contained_at_16"] += float(
                        layer_contained_sum[layer]
                    )
                    layers[layer_key]["mean_missing_at_16"] += float(
                        layer_missing_sum[layer]
                    )
                    layers[layer_key]["candidate_margin_at_16"] += float(
                        layer_margin_sum[layer]
                    )
                    layers[layer_key]["router_kl"] += float(layer_kl_sum[layer])
                for local in valid_indices:
                    request_id = int(request_ids[local])
                    domain = str(request_domains[local])
                    row_key = (request_id, horizon)
                    domain_key = (domain, horizon)
                    requests[row_key]["source_rows"] += 1
                    domains[domain_key]["source_rows"] += 1
                    source_kl = float(source_kl_values[local])
                    source_exact = float(source_exact_values[local])
                    for collection, key in (
                        (requests[row_key], None),
                        (domains[domain_key], None),
                    ):
                        collection["router_kl"] += source_kl
                        collection["exact_set_at_8"] += source_exact
                        collection["all_eight_contained_at_16"] += float(
                            source_contained16_values[local]
                        )
                        collection["mean_missing_at_16"] += float(
                            source_missing16_values[local]
                        )
                        collection["candidate_margin_at_16"] += float(
                            source_margin16_values[local]
                        )
                    for count, overlap in overlap_by_count.items():
                        source_recall = float(source_recall_values[count][local])
                        metric = f"slot_recall_at_{count}"
                        requests[row_key][metric] += source_recall
                        domains[domain_key][metric] += source_recall

    request_rows: list[dict[str, Any]] = []
    for (request_id, horizon), values in sorted(requests.items()):
        count = max(1.0, values["source_rows"])
        request_number = int(np.flatnonzero(data.request_ids == request_id)[0])
        row: dict[str, Any] = {
            "split": split,
            "request_id": request_id,
            "domain": str(data.request_domains[request_number]),
            "horizon": horizon,
            "source_rows": int(values["source_rows"]),
            "router_kl": values["router_kl"] / count,
            "exact_set_at_8": values["exact_set_at_8"] / count,
            "all_eight_contained_at_16": values[
                "all_eight_contained_at_16"
            ] / count,
            "mean_missing_at_16": values["mean_missing_at_16"] / count,
            "candidate_margin_at_16": values[
                "candidate_margin_at_16"
            ] / count,
        }
        for candidate_count in candidate_counts:
            metric = f"slot_recall_at_{candidate_count}"
            row[metric] = values[metric] / count
        request_rows.append(row)

    horizon_rows: list[dict[str, Any]] = []
    for horizon in range(1, horizons + 1):
        values = totals[horizon]
        count = max(1.0, values["source_rows"])
        matching_requests = [
            row for row in request_rows if int(row["horizon"]) == horizon
        ]
        row = {
            "split": split,
            "horizon": horizon,
            "source_rows": int(values["source_rows"]),
            "requests": len(matching_requests),
            "router_kl": values["router_kl"] / count,
            "exact_set_at_8": values["exact_set_at_8"] / count,
            "all_eight_contained_at_16": values[
                "all_eight_contained_at_16"
            ] / count,
            "mean_missing_at_16": values["mean_missing_at_16"] / count,
            "candidate_margin_at_16": values[
                "candidate_margin_at_16"
            ] / count,
        }
        for candidate_count in candidate_counts:
            metric = f"slot_recall_at_{candidate_count}"
            row[f"micro_{metric}"] = values[metric] / count
            row[f"request_macro_{metric}"] = float(
                np.mean([value[metric] for value in matching_requests])
            ) if matching_requests else 0.0
        horizon_rows.append(row)

    layer_rows = []
    for (horizon, layer), values in sorted(layers.items()):
        count = max(1.0, values["source_rows"])
        row = {
            "split": split,
            "horizon": horizon,
            "layer": layer,
            "source_rows": int(values["source_rows"]),
            "all_eight_contained_at_16": values["all_eight_contained_at_16"] / count,
            "mean_missing_at_16": values["mean_missing_at_16"] / count,
            "candidate_margin_at_16": values["candidate_margin_at_16"] / count,
            "router_kl": values["router_kl"] / count,
        }
        for candidate_count in candidate_counts:
            row[f"slot_recall_at_{candidate_count}"] = values[
                f"slot_recall_at_{candidate_count}"
            ] / count
        layer_rows.append(row)

    domain_rows = []
    for (domain, horizon), values in sorted(domains.items()):
        count = max(1.0, values["source_rows"])
        row = {
            "split": split,
            "domain": domain,
            "horizon": horizon,
            "source_rows": int(values["source_rows"]),
            "slot_recall_at_8": values["slot_recall_at_8"] / count,
            "router_kl": values["router_kl"] / count,
            "exact_set_at_8": values["exact_set_at_8"] / count,
            "all_eight_contained_at_16": values[
                "all_eight_contained_at_16"
            ] / count,
            "mean_missing_at_16": values["mean_missing_at_16"] / count,
            "candidate_margin_at_16": values[
                "candidate_margin_at_16"
            ] / count,
        }
        for candidate_count in candidate_counts[1:]:
            metric = f"slot_recall_at_{candidate_count}"
            row[metric] = values[metric] / count
        domain_rows.append(row)
    return EvaluationResult(
        horizon_metrics=horizon_rows,
        request_metrics=request_rows,
        layer_metrics=layer_rows,
        domain_metrics=domain_rows,
    )


def request_bootstrap_h2(
    request_metrics: list[dict[str, Any]],
    *,
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    values = np.asarray(
        [
            float(row["slot_recall_at_8"])
            for row in request_metrics
            if int(row["horizon"]) == 2
        ],
        dtype=np.float64,
    )
    if not len(values):
        raise ValueError("no horizon-2 request rows available for bootstrap")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sample = rng.integers(0, len(values), len(values))
        estimates[replicate] = values[sample].mean()
    return {
        "requests": int(len(values)),
        "replicates": int(replicates),
        "point_estimate": float(values.mean()),
        "ci95_lower": float(np.quantile(estimates, 0.025)),
        "ci95_upper": float(np.quantile(estimates, 0.975)),
    }


def request_bootstrap_mean_h1_h4(
    request_metrics: list[dict[str, Any]],
    *,
    candidate_count: int,
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    metric = f"slot_recall_at_{candidate_count}"
    by_request: dict[int, list[float]] = defaultdict(list)
    for row in request_metrics:
        if 1 <= int(row["horizon"]) <= 4:
            by_request[int(row["request_id"])].append(float(row[metric]))
    if not by_request or any(len(values) != 4 for values in by_request.values()):
        raise ValueError("H1-H4 bootstrap requires four horizons per request")
    values = np.asarray(
        [np.mean(by_request[request_id]) for request_id in sorted(by_request)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sample = rng.integers(0, len(values), len(values))
        estimates[replicate] = values[sample].mean()
    return {
        "requests": int(len(values)),
        "replicates": int(replicates),
        "candidate_count": int(candidate_count),
        "point_estimate": float(values.mean()),
        "ci95_lower": float(np.quantile(estimates, 0.025)),
        "ci95_upper": float(np.quantile(estimates, 0.975)),
    }
