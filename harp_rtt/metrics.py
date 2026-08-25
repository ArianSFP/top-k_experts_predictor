"""Leak-safe HARP-RTT evaluation and model-selection metrics.

The primary score is request-macro SlotRecall@8 over H1--H4.  Rankings are
always recomputed from the dense 256-expert scores with the repository's
stable tie convention (lower expert ID wins an exact tie).  Candidate-pool
coverage is reported independently and is only a gate: it never replaces the
dense-score ranking metric.

The evaluator consumes the rich dataset's nested ``metadata/inputs/targets``
batch and :class:`harp_rtt.model.HARPRTTTeacher` output directly.  Loading or
evaluating the sealed outer test split requires an explicit ``allow_test``
override.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .exact_k import (
    exact_k_logz_marginals_fast,
    stable_topk,
    validate_exact_set_labels,
)
from .schema import HARPRTTOutput


METRICS_SCHEMA = "harp_rtt_metrics_v1"
PRIMARY_HORIZONS = 4
DEFAULT_K = 8
DEFAULT_CANDIDATES = 64


def assert_split_allowed(split: str | None, *, allow_test: bool = False) -> None:
    """Reject the sealed outer test split unless explicitly authorized."""

    if split is not None and str(split).lower() == "test" and not allow_test:
        raise PermissionError(
            "the outer test split is sealed; pass allow_test=True only after the "
            "architecture, seed policy, calibration, and candidate width are frozen"
        )


def _endpoint_valid(valid: Tensor | None, scores: Tensor) -> Tensor:
    expected = scores.shape[:-1]
    if valid is None:
        return torch.ones(expected, dtype=torch.bool, device=scores.device)
    if not isinstance(valid, Tensor):
        raise TypeError("valid must be a torch.Tensor or None")
    if valid.device != scores.device:
        raise ValueError("valid and scores must occupy the same device")
    if valid.shape == expected[:2]:
        valid = valid[..., None].expand(expected)
    if valid.shape != expected:
        raise ValueError(
            f"valid must have shape {tuple(expected[:2])} or {tuple(expected)}, "
            f"got {tuple(valid.shape)}"
        )
    return valid.bool()


def slot_recall_at_k(
    scores: Tensor,
    target_ids: Tensor,
    valid: Tensor | None = None,
    *,
    k: int = DEFAULT_K,
) -> Tensor:
    """Return per-endpoint SlotRecall@``k`` using stable dense-score ranking.

    The returned shape is ``scores.shape[:-1]``.  Invalid rows are zeroed; the
    caller must retain ``valid`` when reducing them.
    """

    safe_ids, weights = validate_exact_set_labels(
        scores, target_ids, valid=_endpoint_valid(valid, scores), k=k
    )
    predicted_ids = stable_topk(scores.float(), k)
    membership = torch.zeros_like(scores, dtype=torch.bool)
    membership.scatter_(-1, safe_ids, True)
    recall = membership.gather(-1, predicted_ids).sum(-1).float() / float(k)
    return recall * (weights > 0).to(recall.dtype)


def cache_set_counts(
    predicted_ids: Tensor,
    current_ids: Tensor,
    target_ids: Tensor,
    valid: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-request/horizon correct, true, and valid-cell CacheSet counts.

    For current expert set S_t, future truth S_(t+h), and future prediction
    P_(t+h), CacheSet recall is:

        |S_t intersect S_(t+h) intersect P_(t+h)|
        ------------------------------------------------
                 |S_t intersect S_(t+h)|

    The first two returned tensors contain the numerator and denominator. The
    third counts valid token-layer cells, allowing the true intersection size
    to be reported independently as a mean cardinality per cell.
    """

    if predicted_ids.shape != target_ids.shape or predicted_ids.ndim != 4:
        raise ValueError(
            "CacheSet predictions and targets must share [B,H,L,K] geometry"
        )
    if current_ids.shape != (
        predicted_ids.shape[0],
        predicted_ids.shape[2],
        predicted_ids.shape[3],
    ):
        raise ValueError("CacheSet current IDs must have [B,L,K] geometry")
    if valid.shape != predicted_ids.shape[:-1]:
        raise ValueError("CacheSet validity must have [B,H,L] geometry")

    target_in_current = (
        target_ids.long().unsqueeze(-1)
        == current_ids.long()[:, None, :, None, :]
    ).any(-1)
    target_in_prediction = (
        target_ids.long().unsqueeze(-1)
        == predicted_ids.long().unsqueeze(-2)
    ).any(-1)
    active = valid.bool().unsqueeze(-1)
    true_counts = (target_in_current & active).sum(dim=(2, 3))
    correct_counts = (
        target_in_current & target_in_prediction & active
    ).sum(dim=(2, 3))
    valid_cells = valid.bool().sum(dim=2)
    return correct_counts, true_counts, valid_cells


def exact_set_nll_per_endpoint(
    scores: Tensor,
    target_ids: Tensor,
    valid: Tensor | None = None,
    *,
    k: int = DEFAULT_K,
) -> Tensor:
    """Return the exact-cardinality set NLL before any endpoint reduction."""

    endpoint_valid = _endpoint_valid(valid, scores)
    safe_ids, weights = validate_exact_set_labels(
        scores, target_ids, valid=endpoint_valid, k=k
    )
    with torch.no_grad():
        log_z, _ = exact_k_logz_marginals_fast(scores.float(), k)
        true_sum = scores.float().gather(-1, safe_ids).sum(-1)
    return (log_z - true_sum) * (weights > 0).to(log_z.dtype)


def router_kl_per_endpoint(
    scores: Tensor,
    target_router_logits: Tensor,
    valid: Tensor | None = None,
) -> Tensor:
    """Return ``KL(target router || predicted router)`` per endpoint in FP32."""

    if scores.shape != target_router_logits.shape:
        raise ValueError("scores and target_router_logits must have identical shapes")
    if not scores.is_floating_point() or not target_router_logits.is_floating_point():
        raise TypeError("router scores must be floating-point")
    if not torch.isfinite(scores).all() or not torch.isfinite(target_router_logits).all():
        raise ValueError("router scores contain NaN or Inf")
    endpoint_valid = _endpoint_valid(valid, scores)
    target_probability = torch.softmax(target_router_logits.detach().float(), dim=-1)
    result = F.kl_div(
        torch.log_softmax(scores.float(), dim=-1),
        target_probability,
        reduction="none",
    ).sum(-1)
    return result * endpoint_valid.to(result.dtype)


def candidate_coverage_at_k(
    candidate_ids: Tensor,
    target_ids: Tensor,
    candidate_mask: Tensor | None = None,
    valid: Tensor | None = None,
    *,
    experts: int | None = None,
    k: int = DEFAULT_K,
) -> Tensor:
    """Return the fraction of true slots contained in each candidate pool.

    Candidate IDs are treated as a set, so accidental duplicate IDs cannot
    inflate coverage.  Invalid/padded candidates must be marked false or use a
    negative sentinel.
    """

    if candidate_ids.ndim < 2:
        raise ValueError("candidate_ids must expose endpoint and candidate axes")
    if candidate_ids.shape[:-1] != target_ids.shape[:-1]:
        raise ValueError("candidate and target leading dimensions differ")
    if target_ids.shape[-1] != int(k):
        raise ValueError(f"target_ids must contain exactly {k} experts")
    if candidate_ids.device != target_ids.device:
        raise ValueError("candidate_ids and target_ids must occupy the same device")
    if candidate_mask is None:
        candidate_mask = candidate_ids >= 0
    if candidate_mask.shape != candidate_ids.shape:
        raise ValueError("candidate_mask must match candidate_ids")
    candidate_mask = candidate_mask.bool()
    endpoint_shape = candidate_ids.shape[:-1]
    if valid is None:
        endpoint_valid = torch.ones(
            endpoint_shape, dtype=torch.bool, device=candidate_ids.device
        )
    else:
        # _endpoint_valid only uses the leading score geometry; a dummy expert
        # axis keeps validity handling identical to the dense metrics.
        dummy = torch.empty(
            *endpoint_shape, 1, dtype=torch.float32, device=candidate_ids.device
        )
        endpoint_valid = _endpoint_valid(valid, dummy)
    active_targets = target_ids[endpoint_valid]
    if active_targets.numel():
        ordered = active_targets.to(torch.int64).sort(dim=-1).values
        if (ordered[..., 1:] == ordered[..., :-1]).any():
            raise ValueError("active target sets must contain distinct expert IDs")
        if (active_targets < 0).any():
            raise ValueError("active target IDs cannot be negative")
    maximum = -1
    if bool(candidate_mask.any()):
        maximum = max(maximum, int(candidate_ids[candidate_mask].max()))
    if active_targets.numel():
        maximum = max(maximum, int(active_targets.max()))
    namespace = int(experts) if experts is not None else maximum + 1
    if namespace < 1:
        raise ValueError("cannot infer a non-empty expert namespace")
    if bool(candidate_mask.any()):
        active_candidates = candidate_ids[candidate_mask]
        if ((active_candidates < 0) | (active_candidates >= namespace)).any():
            raise ValueError("active candidate IDs lie outside the expert namespace")
    if active_targets.numel() and (active_targets >= namespace).any():
        raise ValueError("active target IDs lie outside the expert namespace")

    safe_candidates = torch.where(
        candidate_mask, candidate_ids, torch.zeros_like(candidate_ids)
    ).long()
    # Boolean scatter assignment is not a set union when duplicate indices
    # contain both an active candidate and a padded sentinel.  In particular,
    # a later false sentinel mapped to expert zero could erase a real expert
    # zero.  Accumulate integer membership counts, then threshold.
    candidate_counts = torch.zeros(
        *endpoint_shape, namespace, dtype=torch.int16, device=candidate_ids.device
    )
    candidate_counts.scatter_add_(
        -1, safe_candidates, candidate_mask.to(torch.int16)
    )
    dense = candidate_counts > 0
    safe_targets = torch.where(
        endpoint_valid[..., None], target_ids, torch.zeros_like(target_ids)
    ).long()
    coverage = dense.gather(-1, safe_targets).sum(-1).float() / float(k)
    return coverage * endpoint_valid.to(coverage.dtype)


def _validate_metric_geometry(values: Tensor, valid: Tensor) -> tuple[int, int, int]:
    if values.ndim != 3:
        raise ValueError("endpoint metrics must be [B,H,L]")
    if valid.shape != values.shape:
        raise ValueError("metric values and validity mask must have identical shapes")
    batch, horizons, layers = map(int, values.shape)
    if horizons < PRIMARY_HORIZONS:
        raise ValueError("HARP-RTT metrics require at least H1--H4")
    return batch, horizons, layers


class RequestMetricAccumulator:
    """Accumulate endpoint metrics before request-macro reduction."""

    def __init__(self, *, k: int = DEFAULT_K) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.k = int(k)
        self._rows: dict[tuple[str, int], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._candidate_width: int | None = None
        self._has_candidates: bool | None = None

    def update(
        self,
        scores: Tensor,
        target_ids: Tensor,
        request_ids: Sequence[str | int],
        *,
        target_router_logits: Tensor,
        valid: Tensor | None = None,
        candidate_ids: Tensor | None = None,
        candidate_mask: Tensor | None = None,
    ) -> None:
        endpoint_valid = _endpoint_valid(valid, scores)
        recall = slot_recall_at_k(scores, target_ids, endpoint_valid, k=self.k)
        nll = exact_set_nll_per_endpoint(scores, target_ids, endpoint_valid, k=self.k)
        router_kl = router_kl_per_endpoint(scores, target_router_logits, endpoint_valid)
        batch, horizons, _ = _validate_metric_geometry(recall, endpoint_valid)
        if len(request_ids) != batch:
            raise ValueError("request_ids length must equal the batch size")
        has_candidates = candidate_ids is not None
        if self._has_candidates is None:
            self._has_candidates = has_candidates
        elif self._has_candidates != has_candidates:
            raise ValueError("candidate pools must be present for every batch or none")
        coverage: Tensor | None = None
        if candidate_ids is not None:
            width = int(candidate_ids.shape[-1])
            if self._candidate_width is None:
                self._candidate_width = width
            elif self._candidate_width != width:
                raise ValueError("candidate width changed during evaluation")
            coverage = candidate_coverage_at_k(
                candidate_ids,
                target_ids,
                candidate_mask,
                endpoint_valid,
                experts=int(scores.shape[-1]),
                k=self.k,
            )
        elif candidate_mask is not None:
            raise ValueError("candidate_mask was provided without candidate_ids")

        valid_cpu = endpoint_valid.detach().cpu()
        recall_cpu = recall.detach().float().cpu()
        nll_cpu = nll.detach().float().cpu()
        kl_cpu = router_kl.detach().float().cpu()
        coverage_cpu = coverage.detach().float().cpu() if coverage is not None else None
        for batch_index, request_id in enumerate(request_ids):
            rid = str(request_id)
            for horizon_index in range(min(horizons, PRIMARY_HORIZONS)):
                active = valid_cpu[batch_index, horizon_index]
                count = int(active.sum())
                if count == 0:
                    continue
                row = self._rows[(rid, horizon_index + 1)]
                row["endpoints"] += count
                row["recall"] += float(
                    recall_cpu[batch_index, horizon_index][active].sum()
                )
                row["nll"] += float(nll_cpu[batch_index, horizon_index][active].sum())
                row["router_kl"] += float(
                    kl_cpu[batch_index, horizon_index][active].sum()
                )
                if coverage_cpu is not None:
                    row["coverage"] += float(
                        coverage_cpu[batch_index, horizon_index][active].sum()
                    )

    def finalize(
        self,
        *,
        split: str | None = "validation",
        allow_test: bool = False,
    ) -> dict[str, Any]:
        assert_split_allowed(split, allow_test=allow_test)
        if not self._rows:
            raise ValueError("cannot finalize empty HARP-RTT metrics")
        recall_name = f"slot_recall_at_{self.k}"
        coverage_name = (
            f"candidate_coverage_at_{self._candidate_width}"
            if self._candidate_width is not None
            else None
        )
        request_rows: list[dict[str, Any]] = []
        for (request_id, horizon), totals in sorted(self._rows.items()):
            denominator = totals["endpoints"]
            row: dict[str, Any] = {
                "request_id": request_id,
                "horizon": horizon,
                "endpoints": int(denominator),
                recall_name: totals["recall"] / denominator,
                "exact_set_nll": totals["nll"] / denominator,
                "router_kl": totals["router_kl"] / denominator,
            }
            if coverage_name is not None:
                row[coverage_name] = totals["coverage"] / denominator
            request_rows.append(row)

        horizon_rows: list[dict[str, Any]] = []
        for horizon in range(1, PRIMARY_HORIZONS + 1):
            selected = [row for row in request_rows if row["horizon"] == horizon]
            if not selected:
                raise ValueError(f"no valid endpoints were observed for H{horizon}")
            row = {
                "horizon": horizon,
                "requests": len(selected),
                "endpoints": int(sum(value["endpoints"] for value in selected)),
                f"request_macro_{recall_name}": float(
                    np.mean([value[recall_name] for value in selected])
                ),
                "request_macro_exact_set_nll": float(
                    np.mean([value["exact_set_nll"] for value in selected])
                ),
                "request_macro_router_kl": float(
                    np.mean([value["router_kl"] for value in selected])
                ),
            }
            if coverage_name is not None:
                row[f"request_macro_{coverage_name}"] = float(
                    np.mean([value[coverage_name] for value in selected])
                )
            horizon_rows.append(row)

        recall_values = [row[f"request_macro_{recall_name}"] for row in horizon_rows]
        nll_values = [row["request_macro_exact_set_nll"] for row in horizon_rows]
        kl_values = [row["request_macro_router_kl"] for row in horizon_rows]
        horizon_sets: dict[str, set[int]] = defaultdict(set)
        for row in request_rows:
            horizon_sets[str(row["request_id"])].add(int(row["horizon"]))
        complete = sum(
            horizons == set(range(1, PRIMARY_HORIZONS + 1))
            for horizons in horizon_sets.values()
        )
        report: dict[str, Any] = {
            "schema": METRICS_SCHEMA,
            "split": split,
            "native_k": self.k,
            "candidate_width": self._candidate_width,
            "requests": len(horizon_sets),
            "complete_requests": int(complete),
            "horizon_metrics": horizon_rows,
            "request_metrics": request_rows,
            f"mean_h1_h4_request_macro_{recall_name}": float(np.mean(recall_values)),
            f"min_h1_h4_request_macro_{recall_name}": float(np.min(recall_values)),
            f"h4_request_macro_{recall_name}": float(recall_values[3]),
            "mean_h1_h4_request_macro_exact_set_nll": float(np.mean(nll_values)),
            "mean_h1_h4_request_macro_router_kl": float(np.mean(kl_values)),
        }
        if coverage_name is not None:
            coverage_values = [
                row[f"request_macro_{coverage_name}"] for row in horizon_rows
            ]
            report[f"mean_h1_h4_request_macro_{coverage_name}"] = float(
                np.mean(coverage_values)
            )
            report[f"h4_request_macro_{coverage_name}"] = float(coverage_values[3])
            if self._candidate_width == DEFAULT_CANDIDATES:
                report["candidate_gate"] = candidate_coverage_gate(report)
        return report


def summarize_harp_rtt_predictions(
    scores: Tensor,
    target_ids: Tensor,
    request_ids: Sequence[str | int],
    *,
    target_router_logits: Tensor,
    valid: Tensor | None = None,
    candidate_ids: Tensor | None = None,
    candidate_mask: Tensor | None = None,
    split: str | None = "validation",
    allow_test: bool = False,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Summarize one or more request-grouped dense prediction tensors."""

    accumulator = RequestMetricAccumulator(k=k)
    accumulator.update(
        scores,
        target_ids,
        request_ids,
        target_router_logits=target_router_logits,
        valid=valid,
        candidate_ids=candidate_ids,
        candidate_mask=candidate_mask,
    )
    return accumulator.finalize(split=split, allow_test=allow_test)


def _nested_to(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _nested_to(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_nested_to(item, device) for item in value)
    if isinstance(value, list):
        return [_nested_to(item, device) for item in value]
    return value


def _dataset_splits(value: Any, seen: set[int] | None = None) -> set[str]:
    if seen is None:
        seen = set()
    if value is None or id(value) in seen:
        return set()
    seen.add(id(value))
    result: set[str] = set()
    split = getattr(value, "split", None)
    if split is not None:
        result.add(str(split))
    child = getattr(value, "dataset", None)
    if child is not None:
        result.update(_dataset_splits(child, seen))
    children = getattr(value, "datasets", None)
    if children is not None:
        for item in children:
            result.update(_dataset_splits(item, seen))
    return result


def _output_value(output: Any, *names: str) -> Any | None:
    if isinstance(output, Mapping):
        for name in names:
            if name in output:
                return output[name]
        structured = output.get("structured_output")
        if structured is not None:
            return _output_value(structured, *names)
    if isinstance(output, HARPRTTOutput):
        for name in names:
            if hasattr(output, name):
                return getattr(output, name)
    else:
        for name in names:
            if hasattr(output, name):
                return getattr(output, name)
    return None


def _request_ids(batch: Mapping[str, Any], batch_size: int) -> list[str]:
    metadata = batch.get("metadata")
    if not isinstance(metadata, Mapping) or "request_id" not in metadata:
        raise KeyError("rich batch metadata.request_id is required for request-macro metrics")
    values = metadata["request_id"]
    if isinstance(values, Tensor):
        result = [str(value) for value in values.detach().cpu().tolist()]
    elif isinstance(values, (list, tuple)):
        result = [str(value) for value in values]
    else:
        if batch_size != 1:
            raise ValueError("scalar request_id is only valid for batch size one")
        result = [str(values)]
    if len(result) != batch_size:
        raise ValueError("metadata.request_id length differs from the batch size")
    return result


def evaluate_harp_rtt(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    split: str | None = None,
    allow_test: bool = False,
    autocast: bool = True,
    k: int = DEFAULT_K,
    model_call: Callable[[nn.Module, Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the actual nested rich-batch/model-output contract."""

    inferred = _dataset_splits(getattr(batches, "dataset", None))
    if "test" in {value.lower() for value in inferred}:
        assert_split_allowed("test", allow_test=allow_test)
    if split is not None and inferred and split not in inferred:
        raise ValueError(f"declared split {split!r} disagrees with dataset splits {sorted(inferred)}")
    resolved_split = split or (next(iter(inferred)) if len(inferred) == 1 else None)
    assert_split_allowed(resolved_split, allow_test=allow_test)

    accumulator = RequestMetricAccumulator(k=k)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for host_batch in batches:
                if not isinstance(host_batch, Mapping):
                    raise TypeError("HARP-RTT evaluation batches must be mappings")
                batch = _nested_to(host_batch, device)
                targets = batch.get("targets")
                if not isinstance(targets, Mapping):
                    raise KeyError("rich batch targets mapping is required")
                target_ids = targets.get("future_selected_ids")
                teacher_logits = targets.get("future_router_logits")
                valid = targets.get("future_available")
                if not isinstance(target_ids, Tensor) or not isinstance(teacher_logits, Tensor):
                    raise KeyError(
                        "targets.future_selected_ids and future_router_logits are required"
                    )
                enabled = bool(autocast and str(device).startswith("cuda"))
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=enabled
                ):
                    output = (
                        model_call(model, batch)
                        if model_call is not None
                        else model(batch=batch)
                    )
                scores = _output_value(output, "active_scores", "dense_scores")
                if not isinstance(scores, Tensor):
                    raise KeyError("model output contains no active_scores/dense_scores")
                candidate_ids = _output_value(output, "candidate_ids")
                candidate_mask = _output_value(output, "candidate_mask")
                if candidate_ids is not None and not isinstance(candidate_ids, Tensor):
                    raise TypeError("model candidate_ids must be a tensor")
                if candidate_mask is not None and not isinstance(candidate_mask, Tensor):
                    raise TypeError("model candidate_mask must be a tensor")
                accumulator.update(
                    scores.float(),
                    target_ids,
                    _request_ids(host_batch, int(scores.shape[0])),
                    target_router_logits=teacher_logits,
                    valid=valid,
                    candidate_ids=candidate_ids,
                    candidate_mask=candidate_mask,
                )
    finally:
        model.train(was_training)
    return accumulator.finalize(split=resolved_split, allow_test=allow_test)


def candidate_coverage_gate(
    report: Mapping[str, Any],
    *,
    width: int = DEFAULT_CANDIDATES,
    mean_threshold: float = 0.985,
    h4_threshold: float = 0.970,
) -> dict[str, Any]:
    """Apply the formal C64 generator gate (mean >= .985 and H4 >= .97)."""

    if int(report.get("candidate_width") or -1) != int(width):
        raise ValueError(f"candidate coverage gate requires a C{width} report")
    mean_name = f"mean_h1_h4_request_macro_candidate_coverage_at_{width}"
    h4_name = f"h4_request_macro_candidate_coverage_at_{width}"
    if mean_name not in report or h4_name not in report:
        raise KeyError("report does not contain candidate coverage summaries")
    mean_value = float(report[mean_name])
    h4_value = float(report[h4_name])
    if not all(math.isfinite(value) for value in (mean_value, h4_value)):
        raise ValueError("candidate coverage values must be finite")
    return {
        "candidate_width": int(width),
        "mean_h1_h4": mean_value,
        "h4": h4_value,
        "mean_threshold": float(mean_threshold),
        "h4_threshold": float(h4_threshold),
        "mean_margin": mean_value - float(mean_threshold),
        "h4_margin": h4_value - float(h4_threshold),
        "passed": mean_value >= float(mean_threshold)
        and h4_value >= float(h4_threshold),
    }


def model_selection_tuple(
    report: Mapping[str, Any],
    *,
    k: int = DEFAULT_K,
    candidate_width: int = DEFAULT_CANDIDATES,
) -> tuple[float, float, float, float, float, float]:
    """Return the formal lexicographically-maximized checkpoint tuple."""

    names = (
        f"mean_h1_h4_request_macro_slot_recall_at_{k}",
        f"min_h1_h4_request_macro_slot_recall_at_{k}",
        f"h4_request_macro_slot_recall_at_{k}",
        "mean_h1_h4_request_macro_exact_set_nll",
        f"mean_h1_h4_request_macro_candidate_coverage_at_{candidate_width}",
        "mean_h1_h4_request_macro_router_kl",
    )
    missing = [name for name in names if name not in report]
    if missing:
        raise KeyError(f"report is missing model-selection metrics: {missing}")
    values = tuple(float(report[name]) for name in names)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("model-selection metrics must be finite")
    mean_recall, minimum, h4, nll, coverage, router_kl = values
    return mean_recall, minimum, h4, -nll, coverage, -router_kl


def _request_recall_table(
    report_or_rows: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    k: int,
) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    rows: Sequence[Mapping[str, Any]]
    if isinstance(report_or_rows, Mapping):
        raw = report_or_rows.get("request_metrics")
        if not isinstance(raw, Sequence):
            raise KeyError("metric report contains no request_metrics sequence")
        rows = raw
    else:
        rows = report_or_rows
    name = f"slot_recall_at_{k}"
    by_request: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        horizon = int(row["horizon"])
        if not 1 <= horizon <= PRIMARY_HORIZONS:
            continue
        request_id = str(row["request_id"])
        if horizon in by_request[request_id]:
            raise ValueError(f"duplicate request/horizon metric for {request_id!r}, H{horizon}")
        if name not in row:
            raise KeyError(f"request metric contains no {name}")
        by_request[request_id][horizon] = float(row[name])
    if not by_request:
        raise ValueError("paired bootstrap received no H1--H4 request metrics")
    required = set(range(1, PRIMARY_HORIZONS + 1))
    incomplete = [request for request, values in by_request.items() if set(values) != required]
    if incomplete:
        raise ValueError(
            "paired bootstrap requires complete H1--H4 metrics for every request; "
            f"incomplete={incomplete[:4]}"
        )
    return {
        request: np.asarray([values[h] for h in range(1, 5)], dtype=np.float64)
        for request, values in by_request.items()
    }


def paired_complete_request_bootstrap(
    candidate: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    replicates: int = 2_000,
    seed: int = 42,
    confidence: float = 0.95,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Paired bootstrap of H1--H4 request means with at least 1,000 draws."""

    if not isinstance(replicates, int) or replicates < 1_000:
        raise ValueError("paired complete-request bootstrap requires at least 1,000 replicates")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    candidate_table = _request_recall_table(candidate, k=k)
    baseline_table = _request_recall_table(baseline, k=k)
    if set(candidate_table) != set(baseline_table):
        missing_candidate = sorted(set(baseline_table) - set(candidate_table))
        missing_baseline = sorted(set(candidate_table) - set(baseline_table))
        raise ValueError(
            "paired reports must contain identical complete requests; "
            f"missing_candidate={missing_candidate[:4]}, "
            f"missing_baseline={missing_baseline[:4]}"
        )
    request_ids = sorted(candidate_table)
    candidate_values = np.stack([candidate_table[key] for key in request_ids])
    baseline_values = np.stack([baseline_table[key] for key in request_ids])
    per_horizon_delta = candidate_values - baseline_values
    differences = per_horizon_delta.mean(axis=1)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        chosen = rng.integers(0, len(differences), size=len(differences))
        samples[index] = differences[chosen].mean()
    alpha = (1.0 - float(confidence)) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return {
        "schema": "harp_rtt_paired_request_bootstrap_v1",
        "requests": len(request_ids),
        "replicates": replicates,
        "seed": int(seed),
        "confidence": float(confidence),
        "candidate_point_estimate": float(candidate_values.mean()),
        "baseline_point_estimate": float(baseline_values.mean()),
        "paired_gain": float(differences.mean()),
        "per_horizon_gain": [float(value) for value in per_horizon_delta.mean(axis=0)],
        "lower_bound": lower,
        "upper_bound": upper,
        "ci95_lower": lower if confidence == 0.95 else None,
        "ci95_upper": upper if confidence == 0.95 else None,
        "lower_bound_positive": lower > 0.0,
    }


__all__ = [
    "DEFAULT_CANDIDATES",
    "DEFAULT_K",
    "METRICS_SCHEMA",
    "PRIMARY_HORIZONS",
    "RequestMetricAccumulator",
    "assert_split_allowed",
    "cache_set_counts",
    "candidate_coverage_at_k",
    "candidate_coverage_gate",
    "evaluate_harp_rtt",
    "exact_set_nll_per_endpoint",
    "model_selection_tuple",
    "paired_complete_request_bootstrap",
    "router_kl_per_endpoint",
    "slot_recall_at_k",
    "summarize_harp_rtt_predictions",
]
