"""Phase-0 diagnostics for the HARP-RTT experiment ladder.

These probes are deliberately training-framework agnostic.  They operate on
the rich nested batch contract, dense generator score families, or completed
metric reports, making them usable both in focused local audits and from the
RunPod training driver.  None opens the sealed outer test split by default.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from statistics import NormalDist
import struct
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .exact_k import stable_topk
from .geometry import CenteredRouterGeometry
from .metrics import (
    DEFAULT_K,
    PRIMARY_HORIZONS,
    assert_split_allowed,
    candidate_coverage_at_k,
    candidate_coverage_gate,
    model_selection_tuple,
    paired_complete_request_bootstrap,
)


PROBES_SCHEMA = "harp_rtt_phase0_probes_v1"
_PREFIX_DOMAIN = b"HARP_RTT_CAUSAL_PREFIX_V1"


def _valid_endpoints(valid: Tensor | None, leading: tuple[int, int, int], device: torch.device) -> Tensor:
    if valid is None:
        return torch.ones(leading, dtype=torch.bool, device=device)
    if valid.device != device:
        raise ValueError("validity mask and probe tensors must occupy the same device")
    if valid.shape == leading[:2]:
        valid = valid[..., None].expand(leading)
    if tuple(valid.shape) != leading:
        raise ValueError(
            f"validity mask must have shape {leading[:2]} or {leading}, got {tuple(valid.shape)}"
        )
    return valid.bool()


def _request_ids(values: Sequence[str | int], batch: int) -> list[str]:
    if len(values) != batch:
        raise ValueError("request_ids length must equal the batch size")
    return [str(value) for value in values]


def candidate_union_ids(
    score_sources: Tensor | Sequence[Tensor],
    width: int,
    *,
    temperatures: Tensor | Sequence[float] | None = None,
) -> Tensor:
    """Build the production-style stable, balanced union for an arbitrary C.

    With one score tensor this is simply stable dense top-C.  With multiple
    sources it mirrors :class:`harp_rtt.model.candidates.CandidateUnion`:
    temperature-normalize and center every source, add round-robin source
    rankings, then fill remaining unique slots from the aggregate ranking.
    """

    sources = [score_sources] if isinstance(score_sources, Tensor) else list(score_sources)
    if not sources:
        raise ValueError("at least one dense score source is required")
    first = sources[0]
    if first.ndim != 4:
        raise ValueError("candidate score sources must be [B,H,L,E]")
    if not first.is_floating_point() or not torch.isfinite(first).all():
        raise ValueError("candidate score sources must be finite and floating-point")
    if any(source.shape != first.shape for source in sources):
        raise ValueError("all candidate score sources must have identical shapes")
    if any(source.device != first.device for source in sources):
        raise ValueError("all candidate score sources must occupy the same device")
    experts = int(first.shape[-1])
    if not 1 <= int(width) <= experts:
        raise ValueError(f"candidate width must lie in 1..{experts}")
    if len(sources) == 1 and temperatures is None:
        return stable_topk(first, int(width))

    if temperatures is None:
        temperature_values = torch.ones(
            len(sources), dtype=torch.float32, device=first.device
        )
    else:
        temperature_values = torch.as_tensor(
            temperatures, dtype=torch.float32, device=first.device
        )
    if temperature_values.shape != (len(sources),):
        raise ValueError("temperatures must contain one scalar per score source")
    if not torch.isfinite(temperature_values).all() or (temperature_values <= 0).any():
        raise ValueError("candidate temperatures must be finite and positive")

    stacked = torch.stack([source.float() for source in sources], dim=-2)
    normalized = stacked / temperature_values[None, None, None, :, None]
    normalized = normalized - normalized.mean(dim=-1, keepdim=True)
    aggregate = normalized.mean(dim=-2)
    leading = first.shape[:-1]
    rows = int(np.prod(leading))
    source_count = len(sources)
    ranked_sources = torch.argsort(
        normalized.reshape(rows, source_count, experts),
        dim=-1,
        descending=True,
        stable=True,
    )
    ranked_aggregate = torch.argsort(
        aggregate.reshape(rows, experts), dim=-1, descending=True, stable=True
    )
    selected = torch.zeros(rows, experts, dtype=torch.bool, device=first.device)
    result = torch.full(
        (rows, int(width)), -1, dtype=torch.int64, device=first.device
    )
    counts = torch.zeros(rows, dtype=torch.int64, device=first.device)
    row_ids = torch.arange(rows, device=first.device)

    def add(ids: Tensor) -> None:
        is_new = ~selected.gather(1, ids[:, None]).squeeze(1)
        keep = is_new & (counts < int(width))
        if bool(keep.any()):
            active_rows = row_ids[keep]
            active_ids = ids[keep]
            result[active_rows, counts[keep]] = active_ids
            selected[active_rows, active_ids] = True
            counts[keep] += 1

    quota = (int(width) + source_count - 1) // source_count
    for rank in range(min(quota, experts)):
        for source in range(source_count):
            add(ranked_sources[:, source, rank])
    for rank in range(experts):
        if bool((counts >= int(width)).all()):
            break
        add(ranked_aggregate[:, rank])
    if (result < 0).any():
        raise RuntimeError("candidate union did not fill every requested slot")
    return result.reshape(*leading, int(width))


def _coverage_summary(
    coverage: Tensor,
    valid: Tensor,
    request_ids: Sequence[str],
    *,
    width: int,
) -> dict[str, Any]:
    if coverage.ndim != 3 or coverage.shape != valid.shape:
        raise ValueError("coverage and valid must be matching [B,H,L] tensors")
    totals: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    coverage_cpu = coverage.detach().float().cpu()
    valid_cpu = valid.detach().cpu()
    for batch_index, request_id in enumerate(request_ids):
        for horizon in range(PRIMARY_HORIZONS):
            active = valid_cpu[batch_index, horizon]
            count = int(active.sum())
            if count:
                row = totals[(request_id, horizon + 1)]
                row[0] += float(coverage_cpu[batch_index, horizon][active].sum())
                row[1] += count
    request_rows = [
        {
            "request_id": request_id,
            "horizon": horizon,
            "endpoints": int(values[1]),
            f"candidate_coverage_at_{width}": values[0] / values[1],
        }
        for (request_id, horizon), values in sorted(totals.items())
    ]
    horizon_rows: list[dict[str, Any]] = []
    for horizon in range(1, PRIMARY_HORIZONS + 1):
        selected = [row for row in request_rows if row["horizon"] == horizon]
        if not selected:
            raise ValueError(f"candidate probe observed no valid H{horizon} endpoints")
        horizon_rows.append(
            {
                "horizon": horizon,
                "requests": len(selected),
                f"request_macro_candidate_coverage_at_{width}": float(
                    np.mean(
                        [row[f"candidate_coverage_at_{width}"] for row in selected]
                    )
                ),
            }
        )
    values = [
        row[f"request_macro_candidate_coverage_at_{width}"] for row in horizon_rows
    ]
    return {
        "candidate_width": int(width),
        "requests": len({row["request_id"] for row in request_rows}),
        "request_metrics": request_rows,
        "horizon_metrics": horizon_rows,
        f"mean_h1_h4_request_macro_candidate_coverage_at_{width}": float(
            np.mean(values)
        ),
        f"h4_request_macro_candidate_coverage_at_{width}": float(values[3]),
    }


def generator_candidate_width_probe(
    score_sources: Tensor | Sequence[Tensor],
    target_ids: Tensor,
    request_ids: Sequence[str | int],
    *,
    valid: Tensor | None = None,
    widths: Sequence[int] = (32, 64, 128),
    temperatures: Tensor | Sequence[float] | None = None,
    split: str = "validation",
    allow_test: bool = False,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Measure request-macro generator union coverage at C32/C64/C128."""

    assert_split_allowed(split, allow_test=allow_test)
    sources = [score_sources] if isinstance(score_sources, Tensor) else list(score_sources)
    if not sources or sources[0].ndim != 4:
        raise ValueError("score_sources must contain [B,H,L,E] tensors")
    first = sources[0]
    if first.shape[:-1] != target_ids.shape[:-1]:
        raise ValueError("score and target leading dimensions differ")
    batch, horizons, layers = map(int, first.shape[:3])
    if horizons < PRIMARY_HORIZONS:
        raise ValueError("candidate probe requires H1--H4")
    resolved_ids = _request_ids(request_ids, batch)
    endpoint_valid = _valid_endpoints(valid, (batch, horizons, layers), first.device)
    requested_widths = tuple(int(width) for width in widths)
    if len(set(requested_widths)) != len(requested_widths):
        raise ValueError("candidate probe widths must be unique")
    reports: dict[str, Any] = {}
    for width in requested_widths:
        ids = candidate_union_ids(sources, width, temperatures=temperatures)
        coverage = candidate_coverage_at_k(
            ids,
            target_ids,
            valid=endpoint_valid,
            experts=int(first.shape[-1]),
            k=k,
        )
        report = _coverage_summary(
            coverage[:, :PRIMARY_HORIZONS],
            endpoint_valid[:, :PRIMARY_HORIZONS],
            resolved_ids,
            width=width,
        )
        if width == 64:
            report["candidate_gate"] = candidate_coverage_gate(report)
        reports[f"C{width}"] = report
    return {
        "schema": PROBES_SCHEMA,
        "probe": "generator_candidate_width",
        "split": split,
        "native_k": int(k),
        "widths": reports,
    }


def exact_token_paired_probe(
    token_present_report: Mapping[str, Any],
    token_masked_report: Mapping[str, Any],
    *,
    replicates: int = 2_000,
    seed: int = 42,
    allow_test: bool = False,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Report the paired request-level gain from the exact committed H1 token."""

    present_split = token_present_report.get("split")
    masked_split = token_masked_report.get("split")
    assert_split_allowed(None if present_split is None else str(present_split), allow_test=allow_test)
    assert_split_allowed(None if masked_split is None else str(masked_split), allow_test=allow_test)
    if present_split != masked_split:
        raise ValueError("exact-token conditions must use the same frozen split")
    bootstrap = paired_complete_request_bootstrap(
        token_present_report,
        token_masked_report,
        replicates=replicates,
        seed=seed,
        k=k,
    )
    return {
        "schema": PROBES_SCHEMA,
        "probe": "exact_token_present_vs_masked",
        "split": present_split,
        "token_present_mean": float(
            token_present_report[f"mean_h1_h4_request_macro_slot_recall_at_{k}"]
        ),
        "token_masked_mean": float(
            token_masked_report[f"mean_h1_h4_request_macro_slot_recall_at_{k}"]
        ),
        "paired": bootstrap,
        "passed": bool(bootstrap["lower_bound_positive"]),
    }


def future_router_reconstruction_probe(
    geometry: CenteredRouterGeometry,
    future_router_inputs: Tensor,
    captured_router_logits: Tensor,
    *,
    selected_ids: Tensor | None = None,
    valid: Tensor | None = None,
    k: int = DEFAULT_K,
    maximum_absolute_logit_error: float = 2e-2,
    minimum_exact_set_agreement: float = 1.0,
    split: str = "validation",
    allow_test: bool = False,
) -> dict[str, Any]:
    """Audit true future router-input reconstruction through centered geometry."""

    assert_split_allowed(split, allow_test=allow_test)
    geometry.validate()
    if future_router_inputs.ndim != 4:
        raise ValueError("future_router_inputs must be [B,H,L,D]")
    if captured_router_logits.ndim != 4:
        raise ValueError("captured_router_logits must be [B,H,L,E]")
    if future_router_inputs.shape[:3] != captured_router_logits.shape[:3]:
        raise ValueError("router input and captured logit endpoint axes differ")
    if tuple(future_router_inputs.shape[-2:]) != (
        geometry.layers,
        geometry.hidden_width,
    ):
        raise ValueError("future router inputs disagree with centered geometry")
    if tuple(captured_router_logits.shape[-2:]) != (
        geometry.layers,
        geometry.experts,
    ):
        raise ValueError("captured router logits disagree with centered geometry")
    if not math.isfinite(float(maximum_absolute_logit_error)) or maximum_absolute_logit_error < 0:
        raise ValueError("maximum_absolute_logit_error must be finite and non-negative")
    if not 0 <= float(minimum_exact_set_agreement) <= 1:
        raise ValueError("minimum_exact_set_agreement must lie in [0,1]")

    device = geometry.expert_keys.device
    inputs = future_router_inputs.to(device=device, dtype=geometry.input_basis.dtype)
    captured = captured_router_logits.to(device=device).float()
    reconstructed = geometry.centered_logits(inputs).float()
    captured_centered = captured - captured.mean(dim=-1, keepdim=True)
    endpoint_valid = _valid_endpoints(
        None if valid is None else valid.to(device),
        tuple(map(int, captured.shape[:3])),
        device,
    )
    difference = reconstructed - captured_centered
    selected_difference = difference[endpoint_valid]
    if selected_difference.numel() == 0:
        raise ValueError("reconstruction probe has no valid endpoints")
    reconstructed_ids = stable_topk(reconstructed, k)
    captured_ids = stable_topk(captured_centered, k)
    captured_membership = torch.zeros_like(captured_centered, dtype=torch.bool)
    captured_membership.scatter_(-1, captured_ids, True)
    set_recall = (
        captured_membership.gather(-1, reconstructed_ids).sum(-1).float()
        / float(k)
    )
    exact_set = set_recall == 1.0
    active_recall = set_recall[endpoint_valid]
    active_exact = exact_set[endpoint_valid]
    max_error = float(selected_difference.abs().max())
    report: dict[str, Any] = {
        "schema": PROBES_SCHEMA,
        "probe": "true_future_router_input_reconstruction",
        "split": split,
        "valid_endpoints": int(endpoint_valid.sum()),
        "maximum_absolute_logit_error": max_error,
        "root_mean_square_logit_error": float(selected_difference.square().mean().sqrt()),
        f"mean_slot_recall_at_{k}": float(active_recall.mean()),
        "exact_topk_set_agreement": float(active_exact.float().mean()),
        "maximum_absolute_logit_error_threshold": float(maximum_absolute_logit_error),
        "minimum_exact_set_agreement": float(minimum_exact_set_agreement),
    }
    if selected_ids is not None:
        labels = selected_ids.to(device)
        if labels.shape != captured_ids.shape:
            raise ValueError("selected_ids must match the reconstructed top-k shape")
        label_membership = torch.zeros_like(captured_centered, dtype=torch.bool)
        label_membership.scatter_(-1, labels.long(), True)
        label_recall = (
            label_membership.gather(-1, reconstructed_ids).sum(-1).float()
            / float(k)
        )
        report[f"selected_label_slot_recall_at_{k}"] = float(
            label_recall[endpoint_valid].mean()
        )
    report["passed"] = (
        max_error <= float(maximum_absolute_logit_error)
        and report["exact_topk_set_agreement"] >= float(minimum_exact_set_agreement)
    )
    return report


def _validate_probe_selected_ids(
    selected_ids: Tensor,
    valid: Tensor,
    *,
    experts: int,
    k: int,
) -> None:
    if selected_ids.ndim != 4 or selected_ids.shape[-1] != k:
        raise ValueError(f"selected IDs must be [B,4,L,{k}]")
    if selected_ids.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("selected IDs must use an integer dtype")
    active = selected_ids[valid]
    if active.numel() == 0:
        raise ValueError("probe contains no valid selected-ID labels")
    if ((active < 0) | (active >= experts)).any():
        raise ValueError("selected expert ID lies outside the expert universe")
    if any(torch.unique(row).numel() != k for row in active.reshape(-1, k)):
        raise ValueError("every valid selected-ID label must be an exact set")


def _request_set_retrieval_summary(
    predicted_ids: Tensor,
    target_ids: Tensor,
    valid: Tensor,
    request_ids: Sequence[str],
    *,
    k: int,
) -> dict[str, Any]:
    """Request-macro exact-set and SlotRecall metrics for H1--H4."""

    if predicted_ids.shape != target_ids.shape:
        raise ValueError("predicted and target selected IDs must have equal shapes")
    if tuple(valid.shape) != tuple(target_ids.shape[:-1]):
        raise ValueError("selected IDs and validity mask disagree")
    matches = (
        predicted_ids[..., :, None] == target_ids[..., None, :]
    ).any(dim=-1)
    recall = matches.sum(dim=-1).float() / float(k)
    exact = recall == 1.0
    recall_cpu = recall.detach().cpu()
    exact_cpu = exact.detach().cpu()
    valid_cpu = valid.detach().cpu()
    totals: dict[tuple[str, int], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0]
    )
    for batch_index, request_id in enumerate(request_ids):
        for horizon in range(PRIMARY_HORIZONS):
            active = valid_cpu[batch_index, horizon]
            count = int(active.sum())
            if count:
                row = totals[(request_id, horizon + 1)]
                row[0] += float(recall_cpu[batch_index, horizon][active].sum())
                row[1] += float(exact_cpu[batch_index, horizon][active].sum())
                row[2] += count
    request_rows = [
        {
            "request_id": request_id,
            "horizon": horizon,
            "endpoints": int(values[2]),
            f"slot_recall_at_{k}": values[0] / values[2],
            "exact_set_agreement": values[1] / values[2],
        }
        for (request_id, horizon), values in sorted(totals.items())
    ]
    horizon_rows: list[dict[str, Any]] = []
    for horizon in range(1, PRIMARY_HORIZONS + 1):
        selected = [row for row in request_rows if row["horizon"] == horizon]
        if not selected:
            raise ValueError(f"two-layer probe observed no valid H{horizon} endpoints")
        horizon_rows.append(
            {
                "horizon": horizon,
                "requests": len(selected),
                "endpoints": sum(int(row["endpoints"]) for row in selected),
                f"request_macro_slot_recall_at_{k}": float(
                    np.mean([row[f"slot_recall_at_{k}"] for row in selected])
                ),
                "request_macro_exact_set_agreement": float(
                    np.mean([row["exact_set_agreement"] for row in selected])
                ),
            }
        )
    recalls = [row[f"request_macro_slot_recall_at_{k}"] for row in horizon_rows]
    exact_sets = [row["request_macro_exact_set_agreement"] for row in horizon_rows]
    return {
        "request_metrics": request_rows,
        "horizon_metrics": horizon_rows,
        f"mean_h1_h4_request_macro_slot_recall_at_{k}": float(np.mean(recalls)),
        f"min_h1_h4_request_macro_slot_recall_at_{k}": float(np.min(recalls)),
        f"h4_request_macro_slot_recall_at_{k}": float(recalls[3]),
        "mean_h1_h4_request_macro_exact_set_agreement": float(
            np.mean(exact_sets)
        ),
    }


class _LayerwiseTwoLayerCoordinatePredictor(nn.Module):
    """Two affine layers with GELU, independently parameterized per target layer."""

    def __init__(
        self,
        *,
        layers: int,
        input_width: int,
        hidden_width: int,
        output_width: int,
    ) -> None:
        super().__init__()
        self.first_weight = nn.Parameter(
            torch.empty(layers, input_width, hidden_width)
        )
        self.first_bias = nn.Parameter(torch.zeros(layers, hidden_width))
        self.second_weight = nn.Parameter(
            torch.empty(layers, hidden_width, output_width)
        )
        self.second_bias = nn.Parameter(torch.zeros(layers, output_width))
        for layer in range(layers):
            nn.init.xavier_uniform_(self.first_weight[layer])
            nn.init.xavier_uniform_(self.second_weight[layer])

    def forward(self, values: Tensor) -> Tensor:
        hidden = F.gelu(
            torch.einsum("bhld,ldm->bhlm", values, self.first_weight)
            + self.first_bias[None, None]
        )
        return (
            torch.einsum("bhlm,lmr->bhlr", hidden, self.second_weight)
            + self.second_bias[None, None]
        )


def _masked_layer_normalization(
    values: Tensor, valid: Tensor
) -> tuple[Tensor, Tensor]:
    weights = valid.to(values.dtype).unsqueeze(-1)
    counts = weights.sum(dim=(0, 1))
    if (counts == 0).any():
        raise ValueError("probe fit has no valid examples for one or more layers")
    mean = (values * weights).sum(dim=(0, 1)) / counts
    variance = ((values - mean[None, None]).square() * weights).sum(
        dim=(0, 1)
    ) / counts
    scale = variance.sqrt().clamp_min(1e-5)
    return mean, scale


def two_layer_future_state_ceiling_probe(
    geometry: CenteredRouterGeometry,
    *,
    train_pre_attention_states: Tensor,
    train_router_inputs: Tensor,
    train_request_ids: Sequence[str | int],
    validation_pre_attention_states: Tensor,
    validation_router_inputs: Tensor,
    validation_selected_ids: Tensor,
    validation_request_ids: Sequence[str | int],
    train_valid: Tensor | None = None,
    validation_valid: Tensor | None = None,
    hidden_width: int = 128,
    optimization_steps: int = 250,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    k: int = DEFAULT_K,
    train_split: str = "train",
    validation_split: str = "validation",
) -> dict[str, Any]:
    """Fit a request-safe two-layer future-state router ceiling probe.

    True future pre-attention states are inputs. True future router inputs are
    train-only regression labels, projected to the exact centered-router-visible
    coordinates. Validation router inputs/selected IDs are held out until the
    fitted predictor is frozen and are used only for diagnostics and retrieval
    scoring. The function intentionally exposes no sealed-test override.
    """

    assert_split_allowed(train_split, allow_test=False)
    assert_split_allowed(validation_split, allow_test=False)
    if str(train_split).lower() != "train":
        raise ValueError("two-layer probe fitting split must be exactly 'train'")
    if str(validation_split).lower() != "validation":
        raise ValueError(
            "two-layer probe evaluation split must be exactly 'validation'"
        )
    geometry.validate()
    if (
        isinstance(hidden_width, bool)
        or not isinstance(hidden_width, int)
        or isinstance(optimization_steps, bool)
        or not isinstance(optimization_steps, int)
        or hidden_width < 1
        or optimization_steps < 1
    ):
        raise ValueError("hidden_width and optimization_steps must be positive")
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(weight_decay)) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if not 1 <= int(k) <= geometry.experts:
        raise ValueError("k must lie inside the expert universe")
    if (geometry.ranks <= 0).any():
        raise ValueError("two-layer probe requires nonzero router rank at every layer")

    tensors = {
        "train_pre_attention_states": train_pre_attention_states,
        "train_router_inputs": train_router_inputs,
        "validation_pre_attention_states": validation_pre_attention_states,
        "validation_router_inputs": validation_router_inputs,
    }
    for name, tensor in tensors.items():
        if tensor.ndim != 4 or not tensor.is_floating_point():
            raise ValueError(f"{name} must be floating-point [B,4,L,D]")
        if int(tensor.shape[1]) != PRIMARY_HORIZONS:
            raise ValueError(f"{name} must contain exactly H1--H4")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
    train_batch, _, layers, input_width = map(
        int, train_pre_attention_states.shape
    )
    validation_batch = int(validation_pre_attention_states.shape[0])
    if tuple(validation_pre_attention_states.shape[1:]) != (
        PRIMARY_HORIZONS,
        layers,
        input_width,
    ):
        raise ValueError("train/validation pre-attention state shapes disagree")
    expected_train_router = (
        train_batch,
        PRIMARY_HORIZONS,
        geometry.layers,
        geometry.hidden_width,
    )
    expected_validation_router = (
        validation_batch,
        PRIMARY_HORIZONS,
        geometry.layers,
        geometry.hidden_width,
    )
    if layers != geometry.layers:
        raise ValueError("pre-attention layers disagree with router geometry")
    if tuple(train_router_inputs.shape) != expected_train_router:
        raise ValueError("train router-input shape disagrees with router geometry")
    if tuple(validation_router_inputs.shape) != expected_validation_router:
        raise ValueError(
            "validation router-input shape disagrees with router geometry"
        )
    train_requests = _request_ids(train_request_ids, train_batch)
    validation_requests = _request_ids(validation_request_ids, validation_batch)
    overlap = sorted(set(train_requests) & set(validation_requests))
    if overlap:
        raise ValueError(
            "train/validation requests overlap; request-safe fitting failed: "
            f"{overlap[:4]}"
        )

    device = geometry.expert_keys.device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("two-layer probe supports CPU or CUDA geometry")
    train_mask = _valid_endpoints(
        None if train_valid is None else train_valid.to(device),
        (train_batch, PRIMARY_HORIZONS, layers),
        device,
    )
    validation_mask = _valid_endpoints(
        None if validation_valid is None else validation_valid.to(device),
        (validation_batch, PRIMARY_HORIZONS, layers),
        device,
    )
    labels = validation_selected_ids.to(device)
    if tuple(labels.shape) != (
        validation_batch,
        PRIMARY_HORIZONS,
        layers,
        int(k),
    ):
        raise ValueError(f"validation_selected_ids must be [B,4,L,{k}]")
    _validate_probe_selected_ids(
        labels, validation_mask, experts=geometry.experts, k=int(k)
    )

    train_x = train_pre_attention_states.to(device=device, dtype=torch.float32)
    validation_x = validation_pre_attention_states.to(
        device=device, dtype=torch.float32
    )
    train_router = train_router_inputs.to(device=device, dtype=torch.float32)
    validation_router = validation_router_inputs.to(
        device=device, dtype=torch.float32
    )
    with torch.no_grad():
        train_coordinates = geometry.encode_router_inputs(train_router).float()
        validation_coordinates = geometry.encode_router_inputs(
            validation_router
        ).float()
        input_mean, input_scale = _masked_layer_normalization(train_x, train_mask)
        coordinate_mean, coordinate_scale = _masked_layer_normalization(
            train_coordinates, train_mask
        )
        normalized_train_x = (
            train_x - input_mean[None, None]
        ) / input_scale[None, None]
        normalized_validation_x = (
            validation_x - input_mean[None, None]
        ) / input_scale[None, None]
        normalized_train_coordinates = (
            train_coordinates - coordinate_mean[None, None]
        ) / coordinate_scale[None, None]

    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [
            torch.cuda.current_device() if device.index is None else device.index
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        predictor = _LayerwiseTwoLayerCoordinatePredictor(
            layers=layers,
            input_width=input_width,
            hidden_width=int(hidden_width),
            output_width=geometry.maximum_rank,
        ).to(device)
        optimizer = torch.optim.AdamW(
            predictor.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        coordinate_mask = (
            train_mask[..., None] & geometry.rank_mask[None, None]
        )
        initial_loss = 0.0
        final_loss = 0.0
        predictor.train()
        for step in range(int(optimization_steps)):
            optimizer.zero_grad(set_to_none=True)
            predicted_normalized = predictor(normalized_train_x)
            loss = (
                predicted_normalized - normalized_train_coordinates
            ).square()[coordinate_mask].mean()
            if not torch.isfinite(loss):
                raise RuntimeError("two-layer probe fit produced non-finite loss")
            if step == 0:
                initial_loss = float(loss.detach())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
        predictor.eval()
        with torch.no_grad():
            predicted_coordinates = (
                predictor(normalized_validation_x)
                * coordinate_scale[None, None]
                + coordinate_mean[None, None]
            ) * geometry.rank_mask[None, None].to(torch.float32)

    with torch.no_grad():
        predicted_logits = geometry.score_coordinates(predicted_coordinates)
        predicted_ids = stable_topk(predicted_logits, int(k))
        coordinate_error = predicted_coordinates - validation_coordinates
        active_coordinate_mask = (
            validation_mask[..., None] & geometry.rank_mask[None, None]
        )
        active_logit_error = (
            predicted_logits - geometry.centered_logits(validation_router)
        )[validation_mask]
    retrieval = _request_set_retrieval_summary(
        predicted_ids,
        labels,
        validation_mask,
        validation_requests,
        k=int(k),
    )
    report = {
        "schema": PROBES_SCHEMA,
        "probe": "two_layer_true_future_pre_attention_router_ceiling",
        "fit_split": train_split,
        "evaluation_split": validation_split,
        "native_k": int(k),
        "architecture": {
            "layer_specific": True,
            "affine_layers": 2,
            "activation": "gelu",
            "input": "true_future_pre_attention_state_label_only_probe_channel",
            "regression_target": (
                "true_future_router_input_projected_to_exact_centered_router_coordinates"
            ),
            "input_width": input_width,
            "hidden_width": int(hidden_width),
            "output_width": geometry.maximum_rank,
            "target_layers": layers,
        },
        "fitting": {
            "optimizer": "AdamW",
            "optimization_steps": int(optimization_steps),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "seed": int(seed),
            "full_batch": True,
            "train_normalization_only": True,
            "validation_labels_used_for_fit": False,
            "initial_normalized_coordinate_mse": initial_loss,
            "final_normalized_coordinate_mse": final_loss,
        },
        "request_separation": {
            "train_requests": len(set(train_requests)),
            "validation_requests": len(set(validation_requests)),
            "overlap_requests": 0,
            "request_disjoint": True,
        },
        "train_valid_endpoints": int(train_mask.sum()),
        "validation_valid_endpoints": int(validation_mask.sum()),
        "validation_router_coordinate_rmse": float(
            coordinate_error[active_coordinate_mask].square().mean().sqrt()
        ),
        "validation_centered_logit_rmse": float(
            active_logit_error.square().mean().sqrt()
        ),
        **retrieval,
    }
    return report


def _causal_prefix_identity(
    value: Sequence[int] | Tensor,
) -> tuple[tuple[int, ...], str]:
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise ValueError("each causal prefix tensor must be one-dimensional")
        raw = value.detach().cpu().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raise TypeError("causal prefixes must be token-ID sequences")
    if not raw:
        raise ValueError("causal prefix identity cannot be empty")
    if any(isinstance(token, bool) or not isinstance(token, (int, np.integer)) for token in raw):
        raise TypeError("causal prefix tokens must be integers")
    tokens = tuple(int(token) for token in raw)
    if any(token < 0 or token > 2**63 - 1 for token in tokens):
        raise ValueError("causal prefix tokens must fit non-negative signed int64")
    digest = hashlib.sha256(_PREFIX_DOMAIN)
    for token in tokens:
        digest.update(struct.pack("<q", token))
    return tokens, digest.hexdigest()


def prefix_conditional_bayes_ceiling(
    causal_prefix_token_ids: Sequence[Sequence[int] | Tensor],
    selected_ids: Tensor,
    *,
    valid: Tensor | None = None,
    decoding_mode: str,
    experts: int = 256,
    k: int = DEFAULT_K,
    minimum_stochastic_repeats: int = 2,
    confidence: float = 0.95,
    split: str = "validation",
) -> dict[str, Any]:
    """Estimate the prefix-conditional Bayes-optimal expected SlotRecall@k.

    Groups are formed from exact caller-supplied causal prefix token sequences;
    no request ID, realized future token, branch, or acceptance value can enter
    the grouping key. For each prefix/horizon/layer, the Bayes action is stable
    top-k over empirical marginal expert-inclusion probabilities. Stochastic
    singletons are excluded from the primary estimate and exposed as coverage.
    The function intentionally exposes no sealed-test override.
    """

    assert_split_allowed(split, allow_test=False)
    if str(split).lower() != "validation":
        raise ValueError("prefix-conditional Bayes ceiling requires validation")
    mode = str(decoding_mode).lower()
    if mode not in {"deterministic", "stochastic"}:
        raise ValueError("decoding_mode must be deterministic or stochastic")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if isinstance(experts, bool) or not isinstance(experts, int):
        raise TypeError("experts must be an integer")
    if not 1 <= int(k) < int(experts):
        raise ValueError("k must be positive and smaller than the expert universe")
    if (
        isinstance(minimum_stochastic_repeats, bool)
        or not isinstance(minimum_stochastic_repeats, int)
        or minimum_stochastic_repeats < 2
    ):
        raise ValueError("minimum_stochastic_repeats must be at least two")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if selected_ids.ndim != 4 or int(selected_ids.shape[1]) != PRIMARY_HORIZONS:
        raise ValueError(f"selected_ids must be [N,4,L,{k}]")
    observations, _, layers, selected_k = map(int, selected_ids.shape)
    if selected_k != int(k):
        raise ValueError(f"selected_ids must contain exactly {k} experts")
    if len(causal_prefix_token_ids) != observations:
        raise ValueError("one causal prefix identity is required per observation")
    endpoint_valid = _valid_endpoints(
        valid,
        (observations, PRIMARY_HORIZONS, layers),
        selected_ids.device,
    )
    _validate_probe_selected_ids(
        selected_ids, endpoint_valid, experts=int(experts), k=int(k)
    )

    prefix_rows = [_causal_prefix_identity(value) for value in causal_prefix_token_ids]
    groups: dict[tuple[tuple[int, ...], int, int], list[tuple[int, ...]]] = defaultdict(list)
    labels = selected_ids.detach().to(torch.int64).cpu()
    valid_cpu = endpoint_valid.detach().cpu()
    for observation, (prefix, _digest) in enumerate(prefix_rows):
        for horizon in range(PRIMARY_HORIZONS):
            for layer in range(layers):
                if bool(valid_cpu[observation, horizon, layer]):
                    groups[(prefix, horizon + 1, layer)].append(
                        tuple(int(value) for value in labels[observation, horizon, layer])
                    )
    if not groups:
        raise ValueError("Bayes ceiling received no valid prefix endpoints")

    required_repeats = 1 if mode == "deterministic" else int(minimum_stochastic_repeats)
    z_value = NormalDist().inv_cdf(0.5 + float(confidence) / 2.0)
    group_rows: list[dict[str, Any]] = []
    total_observation_endpoints = 0
    eligible_observation_endpoints = 0
    all_prefixes = {prefix for prefix, _horizon, _layer in groups}
    eligible_prefixes: set[tuple[int, ...]] = set()
    for (prefix, horizon, layer), target_sets in sorted(groups.items()):
        sample_count = len(target_sets)
        total_observation_endpoints += sample_count
        canonical_sets = [tuple(sorted(values)) for values in target_sets]
        if mode == "deterministic" and len(set(canonical_sets)) != 1:
            raise ValueError(
                "deterministic decoding produced multiple target sets for one "
                f"causal prefix at H{horizon}, layer {layer}"
            )
        if sample_count < required_repeats:
            continue
        eligible_observation_endpoints += sample_count
        eligible_prefixes.add(prefix)
        counts = torch.zeros(int(experts), dtype=torch.float64)
        for values in target_sets:
            counts[list(values)] += 1.0
        marginals = counts / float(sample_count)
        optimal_ids = stable_topk(marginals[None], int(k))[0]
        optimal_mass = float(marginals[optimal_ids].sum())
        expected_recall = optimal_mass / float(k)
        observed_recall = torch.tensor(
            [
                sum(expert in values for expert in optimal_ids.tolist()) / float(k)
                for values in target_sets
            ],
            dtype=torch.float64,
        )
        standard_error = (
            float(observed_recall.std(unbiased=True) / math.sqrt(sample_count))
            if sample_count > 1
            else 0.0
        )
        lower = max(0.0, expected_recall - z_value * standard_error)
        upper = min(1.0, expected_recall + z_value * standard_error)
        ranked_probabilities = torch.sort(
            marginals, descending=True, stable=True
        ).values
        boundary_gap = float(
            ranked_probabilities[int(k) - 1] - ranked_probabilities[int(k)]
        )
        entropy_terms = torch.zeros_like(marginals)
        uncertain = (marginals > 0) & (marginals < 1)
        entropy_terms[uncertain] = -(
            marginals[uncertain] * marginals[uncertain].log()
            + (1.0 - marginals[uncertain])
            * (1.0 - marginals[uncertain]).log()
        )
        maximum_marginal_standard_error = float(
            (marginals * (1.0 - marginals) / float(sample_count)).sqrt().max()
        )
        leave_one_out: float | None = None
        if sample_count > 1:
            loo_values = []
            for values in target_sets:
                held_out_counts = counts.clone()
                held_out_counts[list(values)] -= 1.0
                held_out_ids = stable_topk(
                    (held_out_counts / float(sample_count - 1))[None], int(k)
                )[0]
                loo_values.append(
                    sum(expert in values for expert in held_out_ids.tolist())
                    / float(k)
                )
            leave_one_out = float(np.mean(loo_values))
        _tokens, digest = _causal_prefix_identity(prefix)
        group_rows.append(
            {
                "prefix_sha256": digest,
                "horizon": horizon,
                "layer": layer,
                "samples": sample_count,
                "distinct_target_sets": len(set(canonical_sets)),
                f"expected_marginal_optimal_slot_recall_at_{k}": expected_recall,
                f"leave_one_out_slot_recall_at_{k}": leave_one_out,
                "optimal_topk_boundary_gap": boundary_gap,
                "inclusion_binary_entropy_nats": float(entropy_terms.sum()),
                "maximum_marginal_standard_error": maximum_marginal_standard_error,
                "conditional_recall_standard_error": standard_error,
                "conditional_recall_confidence_interval": [lower, upper],
            }
        )
    if not group_rows:
        raise ValueError(
            "no prefix endpoint has enough repeats for the deployed decoding mode"
        )

    metric_name = f"expected_marginal_optimal_slot_recall_at_{k}"
    loo_name = f"leave_one_out_slot_recall_at_{k}"
    horizon_rows: list[dict[str, Any]] = []
    prefix_horizon_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    prefix_horizon_loo: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in group_rows:
        key = (str(row["prefix_sha256"]), int(row["horizon"]))
        prefix_horizon_values[key].append(float(row[metric_name]))
        if row[loo_name] is not None:
            prefix_horizon_loo[key].append(float(row[loo_name]))
    total_groups_by_horizon = {
        horizon: sum(key[1] == horizon for key in groups)
        for horizon in range(1, PRIMARY_HORIZONS + 1)
    }
    for horizon in range(1, PRIMARY_HORIZONS + 1):
        prefix_values = [
            float(np.mean(values))
            for (prefix_hash, current_horizon), values in prefix_horizon_values.items()
            if current_horizon == horizon
        ]
        if not prefix_values:
            raise ValueError(f"Bayes ceiling has no repeat-qualified H{horizon} groups")
        prefix_loo = [
            float(np.mean(values))
            for (_prefix_hash, current_horizon), values in prefix_horizon_loo.items()
            if current_horizon == horizon and values
        ]
        selected_groups = [row for row in group_rows if row["horizon"] == horizon]
        horizon_rows.append(
            {
                "horizon": horizon,
                "eligible_prefixes": len(prefix_values),
                "eligible_prefix_layer_groups": len(selected_groups),
                "total_prefix_layer_groups": total_groups_by_horizon[horizon],
                "prefix_layer_group_coverage": len(selected_groups)
                / total_groups_by_horizon[horizon],
                f"prefix_macro_{metric_name}": float(np.mean(prefix_values)),
                f"observation_weighted_{metric_name}": float(
                    np.average(
                        [row[metric_name] for row in selected_groups],
                        weights=[row["samples"] for row in selected_groups],
                    )
                ),
                f"prefix_macro_{loo_name}": (
                    float(np.mean(prefix_loo)) if prefix_loo else None
                ),
            }
        )
    horizon_values = [row[f"prefix_macro_{metric_name}"] for row in horizon_rows]
    eligible_group_samples = [int(row["samples"]) for row in group_rows]
    group_standard_errors = [
        float(row["conditional_recall_standard_error"]) for row in group_rows
    ]
    boundary_gaps = [float(row["optimal_topk_boundary_gap"]) for row in group_rows]
    entropies = [float(row["inclusion_binary_entropy_nats"]) for row in group_rows]
    loo_values = [float(row[loo_name]) for row in group_rows if row[loo_name] is not None]
    return {
        "schema": PROBES_SCHEMA,
        "probe": "prefix_conditional_bayes_ceiling",
        "split": split,
        "decoding_mode": mode,
        "native_k": int(k),
        "experts": int(experts),
        "estimator": "empirical_prefix_conditional_inclusion_marginal_plugin",
        "bayes_action": "stable_topk_conditional_expert_inclusion_probabilities",
        "grouping_contract": {
            "identity": "SHA-256 of exact causal prefix token IDs through source t",
            "endpoint_axes": ["horizon", "target_layer"],
            "additional_conditioning_fields": [],
            "realized_future_tokens_used_in_grouping": False,
            "request_id_used_in_grouping": False,
            "branch_or_acceptance_used_in_grouping": False,
        },
        "minimum_required_repeats": required_repeats,
        f"mean_h1_h4_prefix_macro_{metric_name}": float(np.mean(horizon_values)),
        f"min_h1_h4_prefix_macro_{metric_name}": float(np.min(horizon_values)),
        f"h4_prefix_macro_{metric_name}": float(horizon_values[3]),
        f"mean_group_{loo_name}": float(np.mean(loo_values)) if loo_values else None,
        "horizon_metrics": horizon_rows,
        "coverage": {
            "observations": observations,
            "causal_prefixes": len(all_prefixes),
            "eligible_causal_prefixes": len(eligible_prefixes),
            "eligible_prefix_fraction": len(eligible_prefixes) / len(all_prefixes),
            "prefix_layer_groups": len(groups),
            "eligible_prefix_layer_groups": len(group_rows),
            "eligible_prefix_layer_group_fraction": len(group_rows) / len(groups),
            "under_repeated_prefix_layer_groups": len(groups) - len(group_rows),
            "singleton_prefix_layer_groups": sum(
                len(values) == 1 for values in groups.values()
            ),
            "singleton_prefix_layer_group_fraction": sum(
                len(values) == 1 for values in groups.values()
            )
            / len(groups),
            "observation_endpoints": total_observation_endpoints,
            "eligible_observation_endpoints": eligible_observation_endpoints,
            "eligible_observation_endpoint_fraction": (
                eligible_observation_endpoints / total_observation_endpoints
            ),
            "minimum_group_samples": min(eligible_group_samples),
            "mean_group_samples": float(np.mean(eligible_group_samples)),
            "maximum_group_samples": max(eligible_group_samples),
        },
        "uncertainty": {
            "confidence": float(confidence),
            "interval_method": "normal_fixed_empirical_bayes_action",
            "plugin_estimate_is_finite_sample_optimistic": mode == "stochastic",
            "mean_conditional_recall_standard_error": float(
                np.mean(group_standard_errors)
            ),
            "maximum_conditional_recall_standard_error": float(
                np.max(group_standard_errors)
            ),
            "mean_optimal_topk_boundary_gap": float(np.mean(boundary_gaps)),
            "minimum_optimal_topk_boundary_gap": float(np.min(boundary_gaps)),
            "mean_inclusion_binary_entropy_nats": float(np.mean(entropies)),
            "mean_distinct_target_sets": float(
                np.mean([row["distinct_target_sets"] for row in group_rows])
            ),
            f"mean_leave_one_out_slot_recall_at_{k}": (
                float(np.mean(loo_values)) if loo_values else None
            ),
        },
        "group_metrics": group_rows,
    }


def _as_tree_batches(value: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return (value,)
    return value


def _tree_request_ids(batch: Mapping[str, Any], batch_size: int) -> list[str]:
    metadata = batch.get("metadata")
    if not isinstance(metadata, Mapping):
        return [f"batch-item-{index}" for index in range(batch_size)]
    values = metadata.get("request_id")
    if isinstance(values, Tensor):
        result = values.detach().cpu().tolist()
    elif isinstance(values, (list, tuple)):
        result = list(values)
    elif values is not None:
        result = [values]
    else:
        result = [f"batch-item-{index}" for index in range(batch_size)]
    if len(result) != batch_size:
        raise ValueError("tree batch request IDs disagree with batch size")
    return [str(value) for value in result]


def tree_availability_probe(
    batches: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    split: str = "validation",
    allow_test: bool = False,
) -> dict[str, Any]:
    """Report adaptive-tree availability and realized full-path acceptance."""

    assert_split_allowed(split, allow_test=allow_test)
    per_request: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    samples = 0
    nodes = 0
    branching_samples = 0
    tensor_complete = 0
    exact_root_samples = 0
    acceptance_nodes = 0
    accepted_nodes = 0
    for batch in _as_tree_batches(batches):
        inputs = batch.get("inputs")
        targets = batch.get("targets")
        if not isinstance(inputs, Mapping) or not isinstance(targets, Mapping):
            raise KeyError("tree probe requires nested inputs and targets mappings")
        tree = inputs.get("tree")
        if not isinstance(tree, Mapping):
            raise KeyError("tree probe requires inputs.tree")
        mask = tree.get("mask")
        parents = tree.get("parent")
        horizon_mask = tree.get("horizon_mask")
        if not all(isinstance(value, Tensor) for value in (mask, parents, horizon_mask)):
            raise KeyError("tree mask, parent, and horizon_mask tensors are required")
        assert isinstance(mask, Tensor) and isinstance(parents, Tensor)
        assert isinstance(horizon_mask, Tensor)
        acceptance = targets.get("tree_acceptance")
        acceptance_valid = targets.get("tree_acceptance_valid")
        if not isinstance(acceptance, Tensor) or not isinstance(acceptance_valid, Tensor):
            raise KeyError("tree acceptance labels and validity are required")
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
            parents = parents.unsqueeze(0)
            horizon_mask = horizon_mask.unsqueeze(0)
            acceptance = acceptance.unsqueeze(0)
            acceptance_valid = acceptance_valid.unsqueeze(0)
            tree = {
                key: value.unsqueeze(0) if isinstance(value, Tensor) and value.ndim >= 1 else value
                for key, value in tree.items()
            }
        batch_size, maximum_nodes = map(int, mask.shape)
        if parents.shape != mask.shape or acceptance.shape != mask.shape:
            raise ValueError("tree masks, parents, and acceptance must be [B,N]")
        if acceptance_valid.shape != mask.shape:
            raise ValueError("tree acceptance validity must be [B,N]")
        if horizon_mask.shape != (batch_size, PRIMARY_HORIZONS, maximum_nodes):
            raise ValueError("tree horizon_mask must be [B,4,N]")
        request_ids = _tree_request_ids(batch, batch_size)
        exact_root = tree.get("exact_root_match")
        tensor_available = tree.get("tensor_available")
        for batch_index, request_id in enumerate(request_ids):
            samples += 1
            active = mask[batch_index].bool()
            active_indices = torch.nonzero(active, as_tuple=False).flatten().tolist()
            nodes += len(active_indices)
            child_counts: dict[int, int] = defaultdict(int)
            roots = 0
            for index in active_indices:
                parent = int(parents[batch_index, index])
                if parent < 0:
                    roots += 1
                else:
                    child_counts[parent] += 1
            branching = roots > 1 or any(value > 1 for value in child_counts.values())
            branching_samples += int(branching)
            if isinstance(exact_root, Tensor):
                exact_root_samples += int(bool((exact_root[batch_index].bool() & active).any()))
            if isinstance(tensor_available, Tensor):
                complete = tensor_available[batch_index].bool().all(dim=-1) & active
                tensor_complete += int(complete.sum())

            path_accepted: dict[int, bool] = {}
            for index in active_indices:
                label_known = bool(acceptance_valid[batch_index, index])
                label = bool(acceptance[batch_index, index]) if label_known else False
                parent = int(parents[batch_index, index])
                parent_ok = True if parent < 0 else path_accepted.get(parent, False)
                path_accepted[index] = label_known and label and parent_ok
                acceptance_nodes += int(label_known)
                accepted_nodes += int(label_known and label)

            for horizon in range(PRIMARY_HORIZONS):
                horizon_nodes = horizon_mask[batch_index, horizon].bool() & active
                indices = torch.nonzero(horizon_nodes, as_tuple=False).flatten().tolist()
                row = per_request[(request_id, horizon + 1)]
                row["samples"] += 1
                row["available"] += int(bool(indices))
                row["nodes"] += len(indices)
                row["accepted_path"] += int(any(path_accepted[index] for index in indices))
                known = sum(bool(acceptance_valid[batch_index, index]) for index in indices)
                accepted = sum(
                    bool(acceptance_valid[batch_index, index])
                    and bool(acceptance[batch_index, index])
                    for index in indices
                )
                row["acceptance_known"] += known
                row["accepted_nodes"] += accepted

    if samples == 0:
        raise ValueError("tree probe received no samples")
    request_rows: list[dict[str, Any]] = []
    for (request_id, horizon), totals in sorted(per_request.items()):
        count = totals["samples"]
        known = totals["acceptance_known"]
        request_rows.append(
            {
                "request_id": request_id,
                "horizon": horizon,
                "samples": int(count),
                "availability": totals["available"] / count,
                "mean_nodes": totals["nodes"] / count,
                "path_acceptance": totals["accepted_path"] / count,
                "node_acceptance": totals["accepted_nodes"] / max(known, 1.0),
                "acceptance_labels": int(known),
            }
        )
    horizon_rows: list[dict[str, Any]] = []
    for horizon in range(1, PRIMARY_HORIZONS + 1):
        selected = [row for row in request_rows if row["horizon"] == horizon]
        horizon_rows.append(
            {
                "horizon": horizon,
                "requests": len(selected),
                "request_macro_availability": float(
                    np.mean([row["availability"] for row in selected])
                ),
                "request_macro_mean_nodes": float(
                    np.mean([row["mean_nodes"] for row in selected])
                ),
                "request_macro_path_acceptance": float(
                    np.mean([row["path_acceptance"] for row in selected])
                ),
                "request_macro_node_acceptance": float(
                    np.mean([row["node_acceptance"] for row in selected])
                ),
            }
        )
    return {
        "schema": PROBES_SCHEMA,
        "probe": "tree_availability_and_path_acceptance",
        "split": split,
        "samples": samples,
        "requests": len({key[0] for key in per_request}),
        "nodes": nodes,
        "mean_nodes_per_sample": nodes / samples,
        "branching_sample_fraction": branching_samples / samples,
        "exact_h1_root_sample_fraction": exact_root_samples / samples,
        "complete_tensor_node_fraction": tensor_complete / max(nodes, 1),
        "node_acceptance": accepted_nodes / max(acceptance_nodes, 1),
        "acceptance_labels": acceptance_nodes,
        "request_metrics": request_rows,
        "horizon_metrics": horizon_rows,
    }


def compare_tree_probes(
    adaptive: Mapping[str, Any], single_chain: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare H3/H4 availability and path acceptance on matched probe reports."""

    if adaptive.get("split") != single_chain.get("split"):
        raise ValueError("tree probe reports must use the same frozen split")
    adaptive_rows = {int(row["horizon"]): row for row in adaptive["horizon_metrics"]}
    chain_rows = {int(row["horizon"]): row for row in single_chain["horizon_metrics"]}
    deltas = []
    for horizon in (3, 4):
        if horizon not in adaptive_rows or horizon not in chain_rows:
            raise ValueError("tree comparison requires H3 and H4")
        deltas.append(
            {
                "horizon": horizon,
                "availability_delta": float(
                    adaptive_rows[horizon]["request_macro_availability"]
                    - chain_rows[horizon]["request_macro_availability"]
                ),
                "path_acceptance_delta": float(
                    adaptive_rows[horizon]["request_macro_path_acceptance"]
                    - chain_rows[horizon]["request_macro_path_acceptance"]
                ),
            }
        )
    return {
        "schema": PROBES_SCHEMA,
        "probe": "adaptive_tree_vs_single_chain",
        "split": adaptive.get("split"),
        "horizon_deltas": deltas,
        "mean_h3_h4_availability_delta": float(
            np.mean([row["availability_delta"] for row in deltas])
        ),
        "mean_h3_h4_path_acceptance_delta": float(
            np.mean([row["path_acceptance_delta"] for row in deltas])
        ),
    }


@dataclass
class C64OverfitHook:
    """Small-corpus memorization hook for the 128-request C64 Phase-0 probe."""

    expected_requests: int = 128
    maximum_gap_to_coverage: float = 0.01
    split: str = "train"
    allow_test: bool = False
    history: list[dict[str, Any]] = field(default_factory=list, init=False)
    _best_report: Mapping[str, Any] | None = field(default=None, init=False, repr=False)
    _best_step: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        assert_split_allowed(self.split, allow_test=self.allow_test)
        if self.expected_requests < 1 or self.expected_requests > 128:
            raise ValueError("C64 overfit probe must use between 1 and 128 requests")
        if not math.isfinite(float(self.maximum_gap_to_coverage)) or self.maximum_gap_to_coverage < 0:
            raise ValueError("maximum_gap_to_coverage must be finite and non-negative")

    def update(self, step: int, report: Mapping[str, Any]) -> dict[str, Any]:
        """Record one evaluation and return its compact overfit snapshot."""

        if not isinstance(step, int) or step < 0:
            raise ValueError("overfit probe step must be a non-negative integer")
        report_split = report.get("split")
        assert_split_allowed(
            None if report_split is None else str(report_split), allow_test=self.allow_test
        )
        if report_split is not None and str(report_split) != self.split:
            raise ValueError("overfit report split differs from hook split")
        complete = int(report.get("complete_requests", -1))
        if complete != self.expected_requests:
            raise ValueError(
                f"C64 overfit probe requires {self.expected_requests} complete requests, got {complete}"
            )
        if int(report.get("candidate_width") or -1) != 64:
            raise ValueError("overfit hook requires a C64 metric report")
        recall = float(report["mean_h1_h4_request_macro_slot_recall_at_8"])
        coverage = float(
            report["mean_h1_h4_request_macro_candidate_coverage_at_64"]
        )
        gap = coverage - recall
        snapshot = {
            "step": step,
            "mean_recall": recall,
            "minimum_horizon_recall": float(
                report["min_h1_h4_request_macro_slot_recall_at_8"]
            ),
            "h4_recall": float(report["h4_request_macro_slot_recall_at_8"]),
            "candidate_coverage": coverage,
            "gap_to_candidate_ceiling": gap,
            "conditional_recovery": recall / max(coverage, 1e-12),
            "selection_tuple": list(model_selection_tuple(report)),
        }
        self.history.append(snapshot)
        if self._best_report is None or model_selection_tuple(report) > model_selection_tuple(
            self._best_report
        ):
            self._best_report = report
            self._best_step = step
        return snapshot

    __call__ = update

    def report(self) -> dict[str, Any]:
        """Return best-step memorization evidence and the independent C64 gate."""

        if self._best_report is None or self._best_step is None:
            raise ValueError("overfit hook has no recorded evaluations")
        best = next(row for row in self.history if row["step"] == self._best_step)
        gate = candidate_coverage_gate(self._best_report)
        return {
            "schema": PROBES_SCHEMA,
            "probe": "c64_overfit_128_requests",
            "split": self.split,
            "expected_requests": self.expected_requests,
            "evaluations": len(self.history),
            "best_step": self._best_step,
            "best": best,
            "maximum_gap_to_coverage": float(self.maximum_gap_to_coverage),
            "overfit_passed": best["gap_to_candidate_ceiling"]
            <= float(self.maximum_gap_to_coverage),
            "candidate_gate": gate,
            "history": list(self.history),
        }


__all__ = [
    "C64OverfitHook",
    "PROBES_SCHEMA",
    "candidate_union_ids",
    "compare_tree_probes",
    "exact_token_paired_probe",
    "future_router_reconstruction_probe",
    "generator_candidate_width_probe",
    "prefix_conditional_bayes_ceiling",
    "tree_availability_probe",
    "two_layer_future_state_ceiling_probe",
]
