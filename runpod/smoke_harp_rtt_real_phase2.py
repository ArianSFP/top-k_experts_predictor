#!/usr/bin/env python3
"""Run one read-only-data production HARP-RTT Phase 2 smoke batch.

The utility deliberately exposes only the frozen ``train`` and ``validation``
splits.  It verifies the complete rich-index inventory and the pinned legacy
anchor SHA-256, loads the immutable static target artifacts, then executes the
same model construction, Phase 2 ownership, input preparation, objective, and
backward path as the production trainer.  It never creates an optimizer,
writes a checkpoint, or performs a promotion decision.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
import platform
import re
import time
from typing import Any

import torch
from torch import Tensor, nn

from harp_rtt.anchor import LegacyHARPAnchorBridge
from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt
from harp_rtt.losses import HARPRTTLossConfig, HARPRTTLossOutput, HARPRTTObjective
from harp_rtt.model import HARPRTTTeacher
from harp_rtt.schema import HARPRTTOutput
from harp_rtt.static_artifacts import MANIFEST_FILENAME, load_static_target_artifacts
from harp_rtt.train import (
    loss_dimensions,
    prepare_model_batch,
    production_config,
    runtime_static_artifacts,
    verify_index_inventory,
    verify_router_numerics_audit,
)
from harp_rtt.training import (
    AuxiliaryGradientController,
    DEFAULT_PHASES,
    autocast_context,
    configure_training_phase,
    move_to_device,
    seed_everything,
    sha256_file,
)


SCHEMA = "harp_rtt_real_phase2_smoke_v1"
ALLOWED_SPLITS = ("train", "validation")
OBJECTIVE_STEP = 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _sha256_argument(value: str) -> str:
    normalized = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise argparse.ArgumentTypeError(
            "expected a 64-character hexadecimal SHA-256"
        )
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--router-numerics-audit", type=Path, required=True)
    parser.add_argument(
        "--router-numerics-audit-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--split", choices=ALLOWED_SPLITS, default="validation")
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    return parser


def _validate_input_paths(args: argparse.Namespace) -> None:
    for name in ("index_root", "corpus_root", "static_dir"):
        path = Path(getattr(args, name)).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(
                f"--{name.replace('_', '-')} is not a directory: {path}"
            )
    for name in (
        "anchor_checkpoint",
        "router_numerics_audit",
        "target_preprocessing",
        "mtp_preprocessing",
    ):
        path = Path(getattr(args, name)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"--{name.replace('_', '-')} is not a file: {path}"
            )
    if str(args.split) not in ALLOWED_SPLITS:
        raise PermissionError("the real smoke utility never opens the sealed test split")


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed(start: float) -> float:
    value = time.perf_counter() - start
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("invalid monotonic stage timing")
    return value


def first_batch(
    dataset: HarpRTTDataset, batch_size: int
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Collate the stable leading rows of an explicitly non-test dataset."""

    if dataset.split not in ALLOWED_SPLITS:
        raise PermissionError("the real smoke utility never opens the sealed test split")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    count = min(int(batch_size), len(dataset))
    if count < 1:
        raise ValueError(f"{dataset.split} dataset contains no usable token rows")
    indices = tuple(range(count))
    return collate_harp_rtt([dataset[index] for index in indices]), indices


def tensor_inventory(value: Any) -> dict[str, dict[str, Any]]:
    """Flatten tensor shapes/dtypes/devices in mappings and dataclasses."""

    result: dict[str, dict[str, Any]] = {}
    active: set[int] = set()

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Tensor):
            result[path] = {
                "shape": list(item.shape),
                "dtype": str(item.dtype).removeprefix("torch."),
                "device": str(item.device),
                "requires_grad": bool(item.requires_grad),
            }
            return
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                return
            active.add(identity)
            for key in sorted(item, key=str):
                visit(item[key], f"{path}.{key}")
            active.remove(identity)
            return
        if is_dataclass(item) and not isinstance(item, type):
            identity = id(item)
            if identity in active:
                return
            active.add(identity)
            for field in fields(item):
                visit(getattr(item, field.name), f"{path}.{field.name}")
            active.remove(identity)
            return
        if isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "$")
    return dict(sorted(result.items()))


def verify_epoch_zero_function_preservation(
    outputs: Mapping[str, Any],
    *,
    active_horizons: int = 4,
) -> dict[str, Any]:
    """Fail closed unless Phase-2 H1--H4 scores exactly preserve the anchor."""

    if active_horizons != 4:
        raise ValueError("the real Phase 2 preservation gate requires H1--H4")
    future_scores = outputs.get("future_router_scores")
    components = outputs.get("component_scores")
    if not isinstance(future_scores, Tensor):
        raise TypeError("model output has no future_router_scores tensor")
    if not isinstance(components, Mapping):
        raise TypeError("model output has no component_scores mapping")
    anchor_scores = components.get("anchor")
    if not isinstance(anchor_scores, Tensor):
        raise TypeError("model output has no component_scores['anchor'] tensor")
    if future_scores.ndim != 4 or int(future_scores.shape[1]) < active_horizons:
        raise ValueError(
            "future_router_scores must be [B,H>=4,L,E] for the preservation gate"
        )
    if anchor_scores.ndim != 4:
        raise ValueError("component_scores['anchor'] must be [B,4,L,E]")

    active_scores = future_scores[:, :active_horizons]
    shape_equal = tuple(active_scores.shape) == tuple(anchor_scores.shape)
    dtype_equal = active_scores.dtype == anchor_scores.dtype
    device_equal = active_scores.device == anchor_scores.device
    exact_equal = bool(shape_equal and torch.equal(active_scores, anchor_scores))
    max_abs_error: float | None = None
    if shape_equal:
        difference = active_scores.detach().float() - anchor_scores.detach().float()
        max_abs_error = float(difference.abs().max().cpu())

    report = {
        "scope": "active H1-H4 future_router_scores versus component_scores['anchor']",
        "horizons": [1, 2, 3, 4],
        "torch_equal": exact_equal,
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "device_equal": device_equal,
        "max_abs_error": max_abs_error,
        "required_max_abs_error": 0.0,
        "active_future_router_scores": {
            "shape": list(active_scores.shape),
            "dtype": str(active_scores.dtype).removeprefix("torch."),
            "device": str(active_scores.device),
        },
        "anchor_component_scores": {
            "shape": list(anchor_scores.shape),
            "dtype": str(anchor_scores.dtype).removeprefix("torch."),
            "device": str(anchor_scores.device),
        },
    }
    passed = bool(
        exact_equal
        and shape_equal
        and dtype_equal
        and device_equal
        and max_abs_error == 0.0
    )
    report["passed"] = passed
    if not passed:
        raise RuntimeError(
            "Phase 2 epoch-zero function preservation failed: "
            + json.dumps(report, sort_keys=True, allow_nan=False)
        )
    return report


def adaptive_contract_report(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Label adaptive versus legacy input without rejecting compatibility data."""

    inputs = batch.get("inputs")
    tree = inputs.get("tree") if isinstance(inputs, Mapping) else None
    marker = tree.get("adaptive_contract") if isinstance(tree, Mapping) else None
    if not isinstance(marker, Tensor):
        raise KeyError("prepared batch lacks inputs.tree.adaptive_contract")
    if marker.dtype != torch.bool:
        raise TypeError("inputs.tree.adaptive_contract must be boolean")
    values = marker.detach().reshape(-1).cpu()
    if values.numel() < 1:
        raise ValueError("adaptive-contract marker is empty")
    adaptive_samples = int(values.sum())
    sample_count = int(values.numel())
    legacy_samples = sample_count - adaptive_samples
    if adaptive_samples == sample_count:
        classification = "adaptive_contract"
    elif adaptive_samples == 0:
        classification = "legacy_compatibility"
    else:
        classification = "mixed_adaptive_and_legacy_compatibility"
    return {
        "field": "inputs.tree.adaptive_contract",
        "classification": classification,
        "uses_adaptive_contract": adaptive_samples == sample_count,
        "any_adaptive_contract": adaptive_samples > 0,
        "all_adaptive_contract": adaptive_samples == sample_count,
        "sample_count": sample_count,
        "adaptive_contract_samples": adaptive_samples,
        "legacy_compatibility_samples": legacy_samples,
        "legacy_rejected": False,
        "compatibility_only_smoke": True,
    }


def _finite_scalar(value: Tensor | float, *, name: str) -> float:
    result = (
        float(value.detach().float().item())
        if isinstance(value, Tensor)
        else float(value)
    )
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite scalar in smoke report: {name}")
    return result


def _finite_vector(value: Tensor, *, name: str) -> list[float]:
    flattened = value.detach().float().cpu().flatten().tolist()
    result = [float(item) for item in flattened]
    if any(not math.isfinite(item) for item in result):
        raise RuntimeError(f"non-finite vector in smoke report: {name}")
    return result


def loss_report(losses: HARPRTTLossOutput, effective: Tensor) -> dict[str, Any]:
    return {
        "objective_step": OBJECTIVE_STEP,
        "effective": _finite_scalar(effective, name="effective"),
        "total": _finite_scalar(losses.total, name="total"),
        "primary": _finite_scalar(losses.primary, name="primary"),
        "auxiliary": _finite_scalar(losses.auxiliary, name="auxiliary"),
        "components": {
            name: _finite_scalar(value, name=f"components.{name}")
            for name, value in sorted(losses.components.items())
        },
        "weighted_components": {
            name: _finite_scalar(value, name=f"weighted_components.{name}")
            for name, value in sorted(losses.weighted_components.items())
        },
        "horizon_components_h1_h4": {
            name: _finite_vector(value, name=f"horizon_components.{name}")
            for name, value in sorted(losses.horizon_components.items())
        },
        "temperature": _finite_scalar(losses.temperature, name="temperature"),
        "recall_progress": _finite_scalar(
            losses.recall_progress, name="recall_progress"
        ),
        "scheduled_weights": {
            name: _finite_scalar(value, name=f"scheduled_weights.{name}")
            for name, value in sorted(losses.scheduled_weights.items())
        },
    }


def gradient_coverage(model: nn.Module) -> dict[str, Any]:
    """Report exact Phase-owned gradient coverage without retaining gradients."""

    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("gradient coverage requires trainable parameters")

    records: list[dict[str, Any]] = []
    reductions: list[Tensor] = []
    for name, parameter in parameters:
        gradient = parameter.grad
        record: dict[str, Any] = {
            "name": name,
            "module": name.split(".", 1)[0],
            "elements": int(parameter.numel()),
            "has_gradient": gradient is not None,
        }
        if gradient is not None:
            detached = gradient.detach()
            finite = torch.isfinite(detached)
            reductions.append(
                torch.stack(
                    (
                        finite.sum(dtype=torch.int64),
                        torch.count_nonzero(detached),
                    )
                )
            )
            finite_values = torch.where(finite, detached, torch.zeros_like(detached))
            record["square_sum"] = finite_values.float().square().sum()
            record["maximum_absolute"] = finite_values.float().abs().max()
        records.append(record)

    if reductions:
        counts = torch.stack(reductions).detach().cpu().tolist()
    else:
        counts = []
    count_index = 0
    square_sums: list[Tensor] = []
    maximums: list[Tensor] = []
    for record in records:
        if record["has_gradient"]:
            finite_elements, nonzero_elements = counts[count_index]
            count_index += 1
            record["finite_elements"] = int(finite_elements)
            record["nonzero_elements"] = int(nonzero_elements)
            square_sums.append(record.pop("square_sum"))
            maximums.append(record.pop("maximum_absolute"))
        else:
            record["finite_elements"] = 0
            record["nonzero_elements"] = 0

    if square_sums:
        global_l2 = float(torch.stack(square_sums).sum().sqrt().detach().cpu())
        maximum_absolute = float(torch.stack(maximums).max().detach().cpu())
    else:
        global_l2 = 0.0
        maximum_absolute = 0.0
    if not math.isfinite(global_l2) or not math.isfinite(maximum_absolute):
        raise RuntimeError("finite-gradient reductions produced a non-finite result")

    def aggregate(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
        tensors = len(values)
        elements = sum(int(value["elements"]) for value in values)
        with_gradient = sum(bool(value["has_gradient"]) for value in values)
        finite_elements = sum(int(value["finite_elements"]) for value in values)
        nonzero_elements = sum(int(value["nonzero_elements"]) for value in values)
        fully_finite = sum(
            bool(value["has_gradient"])
            and int(value["finite_elements"]) == int(value["elements"])
            for value in values
        )
        nonzero_tensors = sum(int(value["nonzero_elements"]) > 0 for value in values)
        return {
            "parameter_tensors": tensors,
            "parameter_elements": elements,
            "with_gradient_tensors": with_gradient,
            "fully_finite_gradient_tensors": fully_finite,
            "nonzero_gradient_tensors": nonzero_tensors,
            "finite_gradient_elements": finite_elements,
            "nonzero_gradient_elements": nonzero_elements,
            "tensor_coverage_fraction": with_gradient / tensors if tensors else None,
            "finite_element_coverage_fraction": (
                finite_elements / elements if elements else None
            ),
            "nonzero_element_fraction": nonzero_elements / elements if elements else None,
        }

    by_module_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_module_records[str(record["module"])].append(record)
    total_elements = sum(int(record["elements"]) for record in records)
    nonfinite_names = [
        str(record["name"])
        for record in records
        if bool(record["has_gradient"])
        and int(record["finite_elements"]) != int(record["elements"])
    ]
    return {
        **aggregate(records),
        "global_finite_gradient_l2_norm": global_l2,
        "maximum_finite_absolute_gradient": maximum_absolute,
        "all_gradient_elements_finite": not nonfinite_names,
        "missing_gradient_parameter_names": [
            str(record["name"])
            for record in records
            if not bool(record["has_gradient"])
        ],
        "zero_gradient_parameter_names": [
            str(record["name"])
            for record in records
            if bool(record["has_gradient"])
            and int(record["nonzero_elements"]) == 0
        ],
        "nonfinite_gradient_parameter_names": nonfinite_names,
        "by_top_level_module": {
            name: aggregate(values)
            for name, values in sorted(by_module_records.items())
        },
        "coverage_denominator": "all Phase 2 trainable parameters",
        "trainable_elements_crosscheck": total_elements,
    }


def _phase2_spec():
    matches = [spec for spec in DEFAULT_PHASES if spec.name == "phase2"]
    if len(matches) != 1:
        raise RuntimeError("production phase registry must contain exactly one Phase 2")
    return matches[0]


def _runtime_report(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_device_capability": list(
                    torch.cuda.get_device_capability(device)
                ),
            }
        )
    return result


def _cuda_memory_start(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "baseline_allocated_bytes": None,
            "baseline_reserved_bytes": None,
        }
    _synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "baseline_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "baseline_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _cuda_memory_finish(
    device: torch.device, baseline: Mapping[str, int | None]
) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            **dict(baseline),
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "incremental_peak_allocated_bytes": None,
            "incremental_peak_reserved_bytes": None,
            "measured": False,
        }
    _synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    baseline_allocated = int(baseline["baseline_allocated_bytes"] or 0)
    baseline_reserved = int(baseline["baseline_reserved_bytes"] or 0)
    return {
        **dict(baseline),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": max(
            0, peak_allocated - baseline_allocated
        ),
        "incremental_peak_reserved_bytes": max(0, peak_reserved - baseline_reserved),
        "measured": True,
    }


def _sample_identity(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = batch.get("metadata")
    if not isinstance(metadata, Mapping):
        raise KeyError("collated rich batch contains no metadata mapping")
    required = ("segment", "sequence_id", "request_id", "position")
    values: dict[str, list[Any]] = {}
    for name in required:
        value = metadata.get(name)
        if not isinstance(value, list):
            raise TypeError(f"metadata.{name} must be a collated list")
        values[name] = value
    count = len(values["request_id"])
    if any(len(value) != count for value in values.values()):
        raise ValueError("collated sample identity columns disagree in length")
    return [
        {name: values[name][index] for name in required}
        for index in range(count)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    total_start = time.perf_counter()
    _validate_input_paths(args)
    device = _resolve_device(str(args.device))
    seed_everything(int(args.seed), deterministic=bool(args.deterministic))
    timings: dict[str, float] = {}

    stage = time.perf_counter()
    index_inventory = verify_index_inventory(args.index_root)
    timings["verify_index_inventory_seconds"] = _elapsed(stage)

    stage = time.perf_counter()
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    timings["load_static_artifacts_seconds"] = _elapsed(stage)

    stage = time.perf_counter()
    router_numerics = verify_router_numerics_audit(
        args.router_numerics_audit,
        args.router_numerics_audit_sha256,
        static.manifest,
    )
    timings["verify_router_numerics_audit_seconds"] = _elapsed(stage)

    stage = time.perf_counter()
    bridge, anchor_provenance = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint,
        args.target_preprocessing,
        args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    timings["load_anchor_and_preprocessing_seconds"] = _elapsed(stage)

    stage = time.perf_counter()
    config = production_config(static, bridge)
    dimensions = loss_dimensions(config)
    model = HARPRTTTeacher(
        bridge,
        config,
        static.geometry,
        token_embedding=static.token_embedding,
    ).to(device)
    runtime_static = runtime_static_artifacts(static, device)
    objective = HARPRTTObjective(
        HARPRTTLossConfig(dimensions=dimensions),
        router_input_basis=static.geometry.input_basis,
        router_rank_mask=static.geometry.rank_mask,
    ).to(device)
    phase = _phase2_spec()
    _, ownership = configure_training_phase(model, phase)
    if ownership.legacy_parameter_count != 0:
        raise RuntimeError("Phase 2 unexpectedly owns legacy-anchor parameters")
    controller = AuxiliaryGradientController()
    del static
    _synchronize(device)
    timings["construct_model_objective_phase2_seconds"] = _elapsed(stage)

    stage = time.perf_counter()
    dataset = HarpRTTDataset(
        args.index_root,
        str(args.split),
        corpus_root=args.corpus_root,
        max_tree_nodes=config.max_tree_nodes,
        allow_test=False,
    )
    host_batch, row_indices = first_batch(dataset, int(args.batch_size))
    timings["open_split_and_collate_first_batch_seconds"] = _elapsed(stage)

    stage = time.perf_counter()
    batch = move_to_device(host_batch, device)
    _synchronize(device)
    timings["host_to_device_seconds"] = _elapsed(stage)

    model.train()
    model.zero_grad(set_to_none=True)
    memory_start = _cuda_memory_start(device)

    stage = time.perf_counter()
    with autocast_context(device, enabled=True):
        prepared = prepare_model_batch(batch, runtime_static)
    _synchronize(device)
    timings["prepare_batch_seconds"] = _elapsed(stage)
    capture_contract = adaptive_contract_report(prepared)

    stage = time.perf_counter()
    with autocast_context(device, enabled=True):
        outputs = model(batch=prepared, anchor_inputs={"batch": prepared})
    _synchronize(device)
    timings["forward_seconds"] = _elapsed(stage)

    structured = outputs.get("structured_output")
    if not isinstance(structured, HARPRTTOutput):
        raise TypeError("production forward returned no HARPRTTOutput diagnostics")
    stage = time.perf_counter()
    structured.validate(dimensions)
    _synchronize(device)
    timings["validate_structured_output_seconds"] = _elapsed(stage)

    stage = time.perf_counter()
    function_preservation = verify_epoch_zero_function_preservation(
        outputs, active_horizons=dimensions.primary_horizons
    )
    _synchronize(device)
    timings["epoch_zero_function_preservation_seconds"] = _elapsed(stage)

    stage = time.perf_counter()
    with autocast_context(device, enabled=True):
        losses = objective(outputs, prepared, step=OBJECTIVE_STEP)
        effective = controller.effective_loss(losses)
    _synchronize(device)
    timings["objective_seconds"] = _elapsed(stage)
    losses_json = loss_report(losses, effective)

    stage = time.perf_counter()
    effective.backward()
    _synchronize(device)
    timings["backward_seconds"] = _elapsed(stage)

    stage = time.perf_counter()
    gradients = gradient_coverage(model)
    if int(gradients["parameter_elements"]) != ownership.trainable_parameter_count:
        raise RuntimeError("gradient denominator differs from Phase 2 ownership")
    _synchronize(device)
    timings["gradient_coverage_seconds"] = _elapsed(stage)
    cuda_memory = _cuda_memory_finish(device, memory_start)

    input_shapes = tensor_inventory(prepared)
    output_shapes = tensor_inventory(outputs)
    tree_mask = prepared["inputs"]["tree"]["mask"]
    future_valid = prepared["targets"]["future_available"]
    if not isinstance(tree_mask, Tensor) or not isinstance(future_valid, Tensor):
        raise TypeError("prepared batch lacks tree/future validity tensors")

    result = {
        "schema": SCHEMA,
        "status": "completed",
        "phase": "phase2",
        "model_mode": "train",
        "split": str(args.split),
        "seed": int(args.seed),
        "deterministic_algorithms": bool(args.deterministic),
        "batch": {
            "requested_size": int(args.batch_size),
            "effective_size": len(row_indices),
            "row_indices": list(row_indices),
            "selection": "stable_leading_dataset_rows_no_shuffle",
            "dataset_rows": len(dataset),
            "samples": _sample_identity(host_batch),
            "valid_future_endpoints": int(future_valid.sum().detach().cpu()),
            "valid_tree_nodes": int(tree_mask.sum().detach().cpu()),
            "capture_contract": capture_contract,
        },
        "artifacts": {
            "paths": {
                "index_root": str(Path(args.index_root).expanduser().resolve()),
                "corpus_root": str(Path(args.corpus_root).expanduser().resolve()),
                "static_dir": str(Path(args.static_dir).expanduser().resolve()),
                "router_numerics_audit": str(
                    Path(args.router_numerics_audit).expanduser().resolve()
                ),
                "anchor_checkpoint": str(
                    Path(args.anchor_checkpoint).expanduser().resolve()
                ),
                "target_preprocessing": str(
                    Path(args.target_preprocessing).expanduser().resolve()
                ),
                "mtp_preprocessing": str(
                    Path(args.mtp_preprocessing).expanduser().resolve()
                ),
            },
            "index_inventory": {
                "schema": index_inventory["schema"],
                "summary_sha256": index_inventory["summary_sha256"],
                "split_manifest_sha256": index_inventory[
                    "split_manifest_sha256"
                ],
                "segments": index_inventory["segments"],
                "inventory_sha256": index_inventory["inventory_sha256"],
                "verified": True,
            },
            "static": {
                "manifest_sha256": sha256_file(
                    Path(args.static_dir) / MANIFEST_FILENAME
                ),
                "schema": runtime_static.manifest.get("schema"),
                "immutable": runtime_static.manifest.get("immutable"),
                "verified": True,
            },
            "router_numerics_audit": router_numerics,
            "anchor": {**anchor_provenance, "verified": True},
        },
        "model_config": config.to_dict(),
        "phase2_ownership": {
            "new_parameter_tensors": len(ownership.new_names),
            "legacy_parameter_tensors": len(ownership.legacy_names),
            "frozen_parameter_tensors": len(ownership.frozen_names),
            "new_parameter_elements": ownership.new_parameter_count,
            "legacy_parameter_elements": ownership.legacy_parameter_count,
            "frozen_parameter_elements": ownership.frozen_parameter_count,
            "trainable_parameter_elements": ownership.trainable_parameter_count,
            "anchor_frozen": bool(getattr(model, "_anchor_frozen", False)),
        },
        "epoch_zero_function_preservation": function_preservation,
        "tensor_inventory": {
            "prepared_batch": input_shapes,
            "outputs": output_shapes,
        },
        "losses": losses_json,
        "gradient_coverage": gradients,
        "timings": timings,
        "cuda_memory": cuda_memory,
        "side_effects": {
            "source_data_writes": False,
            "optimizer_constructed": False,
            "optimizer_step": False,
            "checkpoint_written": False,
            "promotion_evaluated": False,
        },
        "sealed_test": {
            "opened": False,
            "authorization_option_exposed": False,
        },
        "runtime": _runtime_report(device),
    }
    timings["total_seconds"] = _elapsed(total_start)
    # Validate the complete success payload before it reaches stdout.  The
    # strict encoder rejects NaN and Infinity, which Python's default encoder
    # would otherwise emit as non-standard JSON tokens.
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def strict_json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(strict_json_dumps(run(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_SPLITS",
    "OBJECTIVE_STEP",
    "SCHEMA",
    "adaptive_contract_report",
    "build_parser",
    "first_batch",
    "gradient_coverage",
    "loss_report",
    "main",
    "run",
    "strict_json_dumps",
    "tensor_inventory",
    "verify_epoch_zero_function_preservation",
]
