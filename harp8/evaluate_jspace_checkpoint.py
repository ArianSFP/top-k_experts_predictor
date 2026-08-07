"""Bounded evaluation of exact saved J-HARP v1/v2 checkpoints.

This command never constructs an optimizer and exposes no test-split switch.
It evaluates the exact ``model_state`` stored in a best, last, paused, or
interrupted checkpoint.  Input paths are recovered from immutable checkpoint
provenance by default and may be explicitly relocated to another machine.
Ordinary evaluation remains validation-only; a v2 residual-scale diagnostic
may additionally use leakage-safe level-2 meta-training rows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .jspace_data import AlignedJCandidateData
from .jspace_metrics import evaluate_jspace_ranker, paired_request_bootstrap
from .jspace_v2_data import AlignedJContextCandidateData
from .train import canonical_json, sha256_file, write_csv
from .train_jspace_reranker import (
    JSPACE_TRAINING_SCHEMA,
    _assert_matching_data,
    _hash_file,
    _input_provenance,
    _read_pool_manifest,
    _validate_scientific_lineage,
    load_composite_jspace_checkpoint,
)
from .train_jspace_v2_reranker import (
    JSPACE_V2_TRAINING_SCHEMA,
    _assert_v2_matching_data,
    load_composite_jspace_v2_checkpoint,
)


JSPACE_CHECKPOINT_EVALUATION_SCHEMA = "harp8_jspace_checkpoint_evaluation_v1"
JSPACE_CANDIDATE_RESIDUAL_SCALE_DIAGNOSTIC_SCHEMA = (
    "harp8_jspace_v2_candidate_residual_scale_diagnostic_v1"
)
DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_REPLICATES = 5_000
DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_SEED = 20_260_807
_ALLOWED_SCALE_SPLITS = frozenset({"train", "validation"})
_SUPPORTED_SCHEMAS = frozenset(
    {JSPACE_TRAINING_SCHEMA, JSPACE_V2_TRAINING_SCHEMA}
)


def _validated_residual_scales(values: Sequence[float]) -> tuple[float, ...]:
    scales = tuple(float(value) for value in values)
    if not scales:
        raise ValueError("residual scale grid must contain at least one value")
    if any(not np.isfinite(value) or value < 0.0 for value in scales):
        raise ValueError("residual scales must be finite and nonnegative")
    if len(set(scales)) != len(scales):
        raise ValueError("residual scale grid must not contain duplicates")
    return scales


def parse_residual_scale_grid(value: str) -> tuple[float, ...]:
    """Parse a comma-separated post-hoc candidate-residual scale grid."""

    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "residual scale grid must be a comma-separated list of numbers"
        )
    try:
        return _validated_residual_scales(tuple(float(part) for part in parts))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _validate_scale_bootstrap_configuration(
    replicates: int, seed: int
) -> tuple[int, int]:
    replicates = int(replicates)
    seed = int(seed)
    if replicates <= 0:
        raise ValueError("scale bootstrap replicates must be positive")
    if seed < 0:
        raise ValueError("scale bootstrap seed must be nonnegative")
    return replicates, seed


@dataclass(frozen=True)
class JSpaceEvaluationPaths:
    train_pool: Path
    validation_pool: Path
    capture_dir: Path
    mtp_dir: Path
    target_features: Path
    target_feature_rms: Path | None
    precision_audit: Path | None


class _RowLimitView:
    """Read-only split prefix matching the saved training contract."""

    def __init__(self, base: Any, maximum: int | None) -> None:
        self.base = base
        self.rows = base.rows if maximum is None else min(base.rows, int(maximum))
        if self.rows <= 0:
            raise ValueError("evaluation view must contain at least one row")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def batch(self, rows: np.ndarray, device: str | torch.device, **kwargs: Any):
        values = np.asarray(rows, dtype=np.int64)
        if values.ndim != 1 or (values < 0).any() or (values >= self.rows).any():
            raise IndexError("evaluation view rows are out of range")
        return self.base.batch(values, device, **kwargs)

    def sequential_batches(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("evaluation batch size must be positive")
        for start in range(0, self.rows, batch_size):
            yield np.arange(start, min(self.rows, start + batch_size), dtype=np.int64)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _file_record(value: Any, label: str) -> Mapping[str, Any]:
    record = _mapping(value, label)
    if not record.get("path") or not record.get("sha256"):
        raise ValueError(f"{label} lacks path/hash provenance")
    return record


def _original_path(value: Any, label: str) -> Path:
    return Path(str(_file_record(value, label)["path"]))


def _find_aligned_record(
    aligned: Mapping[str, Any], *, suffixes: tuple[str, ...], required: bool = True
) -> Mapping[str, Any] | None:
    matches = [
        _file_record(value, f"aligned input {name}")
        for name, value in aligned.items()
        if any(str(name).endswith(suffix) for suffix in suffixes)
    ]
    if len(matches) > 1:
        raise ValueError(f"ambiguous aligned input for {suffixes}")
    if not matches:
        if required:
            raise ValueError(f"checkpoint provenance omits aligned input {suffixes}")
        return None
    return matches[0]


def _resolve_paths(
    payload: Mapping[str, Any],
    *,
    train_pool: Path | None,
    validation_pool: Path | None,
    capture_dir: Path | None,
    mtp_dir: Path | None,
    target_features: Path | None,
    target_feature_rms: Path | None,
    precision_audit: Path | None,
) -> JSpaceEvaluationPaths:
    provenance = _mapping(payload.get("input_provenance"), "input provenance")
    pools = _mapping(provenance.get("pools"), "pool provenance")
    train_record = _mapping(pools.get("train"), "train pool provenance")
    validation_record = _mapping(
        pools.get("validation"), "validation pool provenance"
    )
    train_root = Path(train_pool) if train_pool is not None else _original_path(
        train_record.get("manifest"), "train pool manifest"
    ).parent
    validation_root = (
        Path(validation_pool)
        if validation_pool is not None
        else _original_path(
            validation_record.get("manifest"), "validation pool manifest"
        ).parent
    )
    aligned = _mapping(provenance.get("aligned_inputs"), "aligned inputs")
    requests = _find_aligned_record(
        aligned, suffixes=("capture/requests.jsonl", "capture_requests")
    )
    mtp_hidden = _find_aligned_record(
        aligned, suffixes=("mtp/mtp_hidden_depths.npy", "mtp_hidden")
    )
    features = _find_aligned_record(
        aligned,
        suffixes=("features/features_normalized.npy", "target_features"),
    )
    rms = _find_aligned_record(
        aligned,
        suffixes=("features/log_rms.npy", "target_feature_rms"),
        required=False,
    )
    capture_root = (
        Path(capture_dir)
        if capture_dir is not None
        else _original_path(requests, "capture requests").parent
    )
    mtp_root = (
        Path(mtp_dir)
        if mtp_dir is not None
        else _original_path(mtp_hidden, "MTP hidden states").parent
    )
    feature_path = (
        Path(target_features)
        if target_features is not None
        else _original_path(features, "target features")
    )
    inference = _mapping(payload.get("inference_contract", {}), "inference contract")
    requires_rms = bool(inference.get("include_feature_rms", rms is not None))
    rms_path: Path | None
    if target_feature_rms is not None:
        rms_path = Path(target_feature_rms)
    elif rms is not None:
        rms_path = _original_path(rms, "target feature RMS")
    else:
        rms_path = None
    if requires_rms and rms_path is None:
        raise ValueError("checkpoint requires target feature RMS but none is resolvable")

    semantic = _mapping(provenance.get("semantic_lineage", {}), "semantic lineage")
    precision = _mapping(
        semantic.get("precision_attribution", {}), "precision attribution"
    )
    audit_record = precision.get("audit")
    audit_path = Path(precision_audit) if precision_audit is not None else None
    if audit_path is None and isinstance(audit_record, Mapping) and audit_record.get("path"):
        audit_path = Path(str(audit_record["path"]))
    return JSpaceEvaluationPaths(
        train_pool=train_root,
        validation_pool=validation_root,
        capture_dir=capture_root,
        mtp_dir=mtp_root,
        target_features=feature_path,
        target_feature_rms=rms_path,
        precision_audit=audit_path,
    )


def _verify_record(path: Path, record: Any, label: str) -> None:
    value = _file_record(record, label)
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    if path.stat().st_size != int(value.get("bytes", -1)):
        raise ValueError(f"{label} byte count differs from checkpoint provenance")
    if sha256_file(path) != str(value.get("sha256")):
        raise ValueError(f"{label} hash differs from checkpoint provenance")


def _verify_pool(
    root: Path,
    split: str,
    record: Any,
    *,
    require_context: bool,
) -> None:
    # Split rejection happens before CandidatePool memory-maps any tensor.
    _read_pool_manifest(
        root,
        split,
        allow_legacy_level2_train=True,
        require_context=require_context,
    )
    value = _mapping(record, f"{split} pool provenance")
    _verify_record(root / "manifest.json", value.get("manifest"), f"{split} manifest")
    _verify_record(root / "metadata.json", value.get("metadata"), f"{split} metadata")
    arrays = value.get("arrays")
    if not isinstance(arrays, list) or not arrays:
        raise ValueError(f"{split} pool provenance has no arrays")
    for index, array in enumerate(arrays):
        array_record = _file_record(array, f"{split} array {index}")
        filename = Path(str(array_record["path"])).name
        _verify_record(root / filename, array_record, f"{split} array {filename}")


def _actual_aligned_path(
    key: str, record: Mapping[str, Any], paths: JSpaceEvaluationPaths
) -> Path:
    filename = Path(str(record["path"])).name
    if key.startswith("capture/") or key in {
        "capture_requests", "target_router_logits", "target_top8"
    }:
        return paths.capture_dir / filename
    if key.startswith("mtp/") or key in {"mtp_hidden", "mtp_router_logits"}:
        return paths.mtp_dir / filename
    if (
        key.endswith("features_normalized.npy")
        or key.endswith("target_features")
    ):
        return paths.target_features
    if filename == "log_rms.npy" or key.endswith("target_feature_rms"):
        if paths.target_feature_rms is None:
            raise ValueError("checkpoint records target RMS but no RMS path is supplied")
        return paths.target_feature_rms
    raise ValueError(f"cannot relocate aligned input {key!r}")


def _verify_nonstrict_provenance(
    payload: Mapping[str, Any], paths: JSpaceEvaluationPaths
) -> None:
    provenance = _mapping(payload.get("input_provenance"), "input provenance")
    pools = _mapping(provenance.get("pools"), "pool provenance")
    require_context = payload.get("schema") == JSPACE_V2_TRAINING_SCHEMA
    _verify_pool(
        paths.train_pool,
        "train",
        pools.get("train"),
        require_context=require_context,
    )
    _verify_pool(
        paths.validation_pool,
        "validation",
        pools.get("validation"),
        require_context=require_context,
    )
    aligned = _mapping(provenance.get("aligned_inputs"), "aligned inputs")
    for key, record_value in aligned.items():
        record = _file_record(record_value, f"aligned input {key}")
        _verify_record(
            _actual_aligned_path(str(key), record, paths),
            record,
            f"aligned input {key}",
        )


def _without_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_paths(item)
            for key, item in value.items()
            if key != "path"
        }
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    return value


def _verify_strict_lineage(
    payload: Mapping[str, Any],
    train_data: AlignedJCandidateData,
    validation_data: AlignedJCandidateData,
    paths: JSpaceEvaluationPaths,
) -> None:
    recorded = _mapping(payload.get("input_provenance"), "input provenance")
    semantic = _mapping(recorded.get("semantic_lineage"), "semantic lineage")
    if semantic.get("strict") is not True:
        raise ValueError("checkpoint lacks strict scientific lineage")
    precision = _mapping(
        semantic.get("precision_attribution", {}), "precision attribution"
    )
    provisional = precision.get("attribution") == "provisional"
    current_semantic = _validate_scientific_lineage(
        train_data,
        validation_data,
        target_features=paths.target_features,
        target_feature_rms=paths.target_feature_rms,
        precision_audit=paths.precision_audit,
        allow_provisional_j_without_probe=provisional,
    )
    for name in ("decision_profile", "mtp_timing_causal", "acceptance_labels_used"):
        if name in semantic:
            current_semantic[name] = semantic[name]
    router_provenance = _mapping(
        recorded.get("router_keys", {}), "router-key provenance"
    )
    current = _input_provenance(
        train_data,
        validation_data,
        target_features=paths.target_features,
        target_feature_rms=paths.target_feature_rms,
        router_provenance=router_provenance,
        semantic_lineage=current_semantic,
    )
    for section in ("pools", "aligned_inputs", "semantic_lineage"):
        if canonical_json(_without_paths(current.get(section))) != canonical_json(
            _without_paths(recorded.get(section))
        ):
            raise ValueError(f"current {section} differs from checkpoint provenance")


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") not in _SUPPORTED_SCHEMAS:
        raise ValueError("checkpoint is not a supported J-HARP v1/v2 training state")
    if not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("checkpoint contains no model state")
    if int(payload.get("completed_epoch", -1)) < 0:
        raise ValueError("checkpoint has an invalid completed epoch")
    return payload


def _paired_checkpoint_bootstrap(
    primary: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    *,
    native_k: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    metric = f"recall_at_{native_k}"
    left = {
        (int(row["request_id"]), int(row["horizon"])): float(row[metric])
        for row in primary if 1 <= int(row["horizon"]) <= 4
    }
    right = {
        (int(row["request_id"]), int(row["horizon"])): float(row[metric])
        for row in comparison if 1 <= int(row["horizon"]) <= 4
    }
    if left.keys() != right.keys() or not left:
        raise ValueError("checkpoint comparison requires identical H1-H4 requests")
    request_ids = sorted({key[0] for key in left})
    if any((request_id, horizon) not in left for request_id in request_ids for horizon in range(1, 5)):
        raise ValueError("checkpoint comparison requires all H1-H4 request metrics")
    pairs = np.asarray([
        [
            np.mean([left[(request_id, horizon)] for horizon in range(1, 5)]),
            np.mean([right[(request_id, horizon)] for horizon in range(1, 5)]),
        ]
        for request_id in request_ids
    ], dtype=np.float64)
    differences = pairs[:, 0] - pairs[:, 1]
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        differences[rng.integers(0, len(differences), len(differences))].mean()
        for _ in range(replicates)
    ])
    return {
        "requests": len(request_ids),
        "replicates": replicates,
        "primary_point_estimate": float(pairs[:, 0].mean()),
        "comparison_point_estimate": float(pairs[:, 1].mean()),
        "paired_gain": float(differences.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
    }


def _candidate_position_band(within: int, rows_per_request: int) -> str:
    fraction = within / max(1, rows_per_request - 1)
    if fraction < 0.25:
        return "q1"
    if fraction < 0.50:
        return "q2"
    if fraction < 0.75:
        return "q3"
    return "q4"


def _new_candidate_metric_accumulator() -> dict[str, Any]:
    return {
        "totals": defaultdict(lambda: defaultdict(float)),
        "requests": defaultdict(lambda: defaultdict(float)),
        "layers": defaultdict(lambda: defaultdict(float)),
        "domains": defaultdict(lambda: defaultdict(float)),
        "positions": defaultdict(lambda: defaultdict(float)),
    }


def _accumulate_candidate_metrics(
    accumulator: Mapping[str, Any],
    data: _RowLimitView,
    rows: np.ndarray,
    batch: Mapping[str, torch.Tensor],
    scores: torch.Tensor,
    base: torch.Tensor,
) -> None:
    """Accumulate the exact public candidate-ranker metrics for one score tensor."""

    native_k = int(data.pool.manifest.get("native_k", 8))
    candidate_count = int(data.pool.candidate_count)
    if not 1 <= native_k <= candidate_count:
        raise ValueError("native_k must lie within the candidate pool")
    membership = batch["target_membership"].bool()
    valid = batch["valid_future"].bool()
    if scores.shape != membership.shape or base.shape != membership.shape:
        raise ValueError("scaled/base scores and candidate membership must match")
    candidate_mask = batch.get("candidate_mask")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(membership)
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask must match scaled candidate scores")
    candidate_mask = candidate_mask.bool()
    if not bool((candidate_mask.sum(dim=-1) >= native_k).all()):
        raise ValueError("every candidate set must contain at least native_k entries")
    if not bool(torch.isfinite(scores[candidate_mask]).all()):
        raise FloatingPointError("scaled checkpoint emitted non-finite candidate scores")
    membership = membership & candidate_mask
    minimum = -torch.finfo(scores.dtype).max
    predicted_top = torch.topk(
        scores.masked_fill(~candidate_mask, minimum), native_k, dim=-1
    ).indices
    base_top = torch.topk(
        base.masked_fill(~candidate_mask, minimum), native_k, dim=-1
    ).indices
    recall = membership.gather(-1, predicted_top).sum(dim=-1).float() / native_k
    base_recall = membership.gather(-1, base_top).sum(dim=-1).float() / native_k
    coverage = membership.sum(dim=-1).float() / native_k
    exact = recall == 1.0
    recall_np = recall.cpu().numpy()
    base_np = base_recall.cpu().numpy()
    coverage_np = coverage.cpu().numpy()
    exact_np = exact.cpu().numpy()
    valid_np = valid.cpu().numpy()
    totals = accumulator["totals"]
    requests = accumulator["requests"]
    layers = accumulator["layers"]
    domains = accumulator["domains"]
    positions = accumulator["positions"]
    for local, pool_row in enumerate(rows.tolist()):
        request_id = int(data.pool.request_ids[pool_row])
        domain = str(data.pool.domains[pool_row])
        within = int(data.pool.within[pool_row])
        band = _candidate_position_band(within, data.rows_per_request)
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


def _finalize_candidate_metrics(
    data: _RowLimitView, accumulator: Mapping[str, Any]
) -> dict[str, Any]:
    """Finalize a scale using the same schema/formulas as evaluate_jspace_ranker."""

    native_k = int(data.pool.manifest.get("native_k", 8))
    candidate_count = int(data.pool.candidate_count)
    coverage_name = f"coverage_at_{candidate_count}"
    request_coverage_name = f"request_macro_coverage_at_{candidate_count}"
    recall_name = f"recall_at_{native_k}"
    base_recall_name = f"base_recall_at_{native_k}"
    exact_name = f"exact_set_at_{native_k}"
    request_recall_name = f"request_macro_recall_at_{native_k}"
    request_base_name = f"request_macro_base_recall_at_{native_k}"

    def row(values: Mapping[str, float]) -> dict[str, float | int]:
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

    totals = accumulator["totals"]
    requests = accumulator["requests"]
    layers = accumulator["layers"]
    domains = accumulator["domains"]
    positions = accumulator["positions"]
    request_rows = [
        {"request_id": request_id, "horizon": horizon, **row(values)}
        for (request_id, horizon), values in sorted(requests.items())
    ]
    horizon_rows: list[dict[str, Any]] = []
    for horizon in range(1, data.pool.horizons + 1):
        request_subset = [value for value in request_rows if value["horizon"] == horizon]
        macro_recall = float(
            np.mean([value[recall_name] for value in request_subset])
        ) if request_subset else 0.0
        macro_base = float(
            np.mean([value[base_recall_name] for value in request_subset])
        ) if request_subset else 0.0
        macro_coverage = float(
            np.mean([value[coverage_name] for value in request_subset])
        ) if request_subset else 0.0
        horizon_rows.append({
            "horizon": horizon,
            **row(totals[horizon]),
            request_recall_name: macro_recall,
            request_base_name: macro_base,
            request_coverage_name: macro_coverage,
            "request_macro_conditional_recovery": macro_recall
            / max(macro_coverage, 1e-12),
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


def _evaluate_v2_residual_scales_once(
    model: torch.nn.Module,
    data: _RowLimitView,
    scales: Sequence[float],
    *,
    batch_size: int,
    device: str,
    autocast: bool,
) -> dict[float, dict[str, Any]]:
    """Evaluate every candidate-residual scale with one model forward per batch."""

    checked = _validated_residual_scales(scales)
    accumulators = {scale: _new_candidate_metric_accumulator() for scale in checked}
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
            if not isinstance(output, Mapping):
                raise TypeError("v2 composite checkpoint must return a mapping")
            frozen = batch["candidate_scores"].float()
            active_delta = output.get("active_delta")
            exact_scores = output.get("scores")
            if not isinstance(active_delta, torch.Tensor) or not isinstance(
                exact_scores, torch.Tensor
            ):
                raise ValueError("v2 composite output omits active_delta/scores")
            active_delta = active_delta.float()
            exact_scores = exact_scores.float()
            if active_delta.shape != frozen[:, :4].shape:
                raise ValueError("active_delta must match frozen candidate H1-H4")
            if exact_scores.shape != frozen.shape:
                raise ValueError("composite scores must match frozen H1-H8 geometry")
            if not torch.equal(exact_scores[:, 4:], frozen[:, 4:]):
                raise AssertionError("saved v2 checkpoint changed frozen H5-H8 scores")
            for scale in checked:
                if scale == 0.0:
                    scores = frozen
                elif scale == 1.0:
                    scores = exact_scores
                else:
                    scores = torch.cat(
                        [frozen[:, :4] + active_delta * scale, frozen[:, 4:]],
                        dim=1,
                    )
                if not torch.equal(scores[:, 4:], frozen[:, 4:]):
                    raise AssertionError("residual scaling changed frozen H5-H8 scores")
                _accumulate_candidate_metrics(
                    accumulators[scale], data, rows, batch, scores, frozen
                )
    return {
        scale: _finalize_candidate_metrics(data, accumulators[scale])
        for scale in checked
    }


def _paired_scale_bootstrap(
    paired_request_gains: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    replicates, seed = _validate_scale_bootstrap_configuration(replicates, seed)
    values = np.asarray(paired_request_gains, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired scale bootstrap requires a nonempty gain vector")
    if not np.isfinite(values).all():
        raise ValueError("paired scale gains must be finite")
    generator = np.random.default_rng(seed)
    statistics = np.empty(replicates, dtype=np.float64)
    chunk_size = min(512, replicates)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        indices = generator.integers(
            0, len(values), size=(stop - start, len(values)), endpoint=False
        )
        statistics[start:stop] = values[indices].mean(axis=1, dtype=np.float64)
    low, high = np.quantile(statistics, (0.025, 0.975))
    return {
        "requests": int(len(values)),
        "paired_gain_at_8": float(values.mean(dtype=np.float64)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def _paired_candidate_gain_for_horizons(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    horizons: Sequence[int],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    requested = frozenset(int(horizon) for horizon in horizons)
    if not requested:
        raise ValueError("paired scale gain requires at least one horizon")

    def values_by_key(metrics: Mapping[str, Any]) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for row in metrics["request_metrics"]:
            horizon = int(row["horizon"])
            if horizon not in requested:
                continue
            key = (int(row["request_id"]), horizon)
            if key in result:
                raise ValueError("duplicate request/horizon scale metric")
            value = float(row["recall_at_8"])
            if not np.isfinite(value):
                raise ValueError("non-finite request recall in scale diagnostic")
            result[key] = value
        return result

    candidate_values = values_by_key(candidate)
    baseline_values = values_by_key(baseline)
    if candidate_values.keys() != baseline_values.keys():
        raise ValueError("scale and alpha=0 request/horizon pairs differ")
    if not candidate_values:
        raise ValueError("no paired request/horizon values for scale diagnostic")
    by_request: dict[int, list[float]] = {}
    for (request_id, horizon), value in sorted(candidate_values.items()):
        by_request.setdefault(request_id, []).append(
            value - baseline_values[(request_id, horizon)]
        )
    if any(len(values) != len(requested) for values in by_request.values()):
        raise ValueError("paired scale bootstrap requires every requested horizon")
    result = _paired_scale_bootstrap(
        [
            float(np.mean(values, dtype=np.float64))
            for _request_id, values in sorted(by_request.items())
        ],
        replicates=replicates,
        seed=seed,
    )
    result["request_horizon_pairs"] = len(candidate_values)
    return result


def _candidate_residual_scale_diagnostic(
    scales: Sequence[float],
    metrics_by_scale: Mapping[float, Mapping[str, Any]],
    *,
    split: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if split == "train":
        designation = "level2_meta_train_scale_selection"
        eligible_for_model_selection = True
    elif split == "validation":
        designation = "post_hoc_validation_only_oracle_diagnostic"
        eligible_for_model_selection = False
    else:
        raise PermissionError("residual-scale diagnostics permit only train/validation")
    checked = _validated_residual_scales(scales)
    bootstrap_replicates, bootstrap_seed = _validate_scale_bootstrap_configuration(
        bootstrap_replicates, bootstrap_seed
    )
    if 0.0 not in metrics_by_scale or 1.0 not in metrics_by_scale:
        raise ValueError("residual scale diagnostic requires alpha=0 and alpha=1 anchors")
    baseline = metrics_by_scale[0.0]
    summaries = []
    for scale in checked:
        metrics = metrics_by_scale[scale]
        native_k = int(metrics["native_k"])
        request_recall_name = f"request_macro_recall_at_{native_k}"
        horizon_metrics = list(metrics["horizon_metrics"])
        mean_h1_h8 = float(np.mean([
            float(row[request_recall_name]) for row in horizon_metrics
        ]))
        h2_recall = float(horizon_metrics[1][request_recall_name])
        paired_h1_h4 = _paired_candidate_gain_for_horizons(
            metrics,
            baseline,
            (1, 2, 3, 4),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        paired_horizons = []
        for horizon in range(1, 5):
            paired_horizons.append({
                "horizon": horizon,
                **_paired_candidate_gain_for_horizons(
                    metrics,
                    baseline,
                    (horizon,),
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed,
                ),
            })
        summaries.append({
            "residual_scale": scale,
            "mean_h1_h4_recall_at_8": float(metrics["mean_h1_h4_recall_at_8"]),
            "mean_h1_h8_recall_at_8": mean_h1_h8,
            "h2_request_macro_recall_at_8": h2_recall,
            "paired_request_h1_h4_gain_vs_alpha0": paired_h1_h4,
            "paired_request_horizon_gains_vs_alpha0": paired_horizons,
            "horizon_metrics": list(metrics["horizon_metrics"]),
        })
    best_index = max(
        range(len(summaries)),
        key=lambda index: (
            summaries[index]["mean_h1_h4_recall_at_8"],
            summaries[index]["h2_request_macro_recall_at_8"],
        ),
    )
    best = summaries[best_index]
    return {
        "schema": JSPACE_CANDIDATE_RESIDUAL_SCALE_DIAGNOSTIC_SCHEMA,
        "split": split,
        "designation": designation,
        "uses_same_split_labels_for_selection": True,
        "uses_validation_labels_for_selection": split == "validation",
        "eligible_for_model_selection": eligible_for_model_selection,
        "creates_trained_checkpoint": False,
        "checkpoint_forward_passes_per_batch": 1,
        "candidate_score_formula_h1_h4": (
            "frozen_candidate_scores + residual_scale * active_delta"
        ),
        "h5_h8_exact_frozen_passthrough": True,
        "paired_reference_residual_scale": 0.0,
        "exact_checkpoint_residual_scale": 1.0,
        "bootstrap": {
            "unit": "request",
            "pairing": "same_request_and_horizon",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "interval": "percentile",
        },
        "selection_metric_order": [
            "mean_h1_h4_recall_at_8",
            "h2_request_macro_recall_at_8",
        ],
        "tie_breaking_rule": "first_scale_in_explicit_grid",
        "requested_scales": list(checked),
        "best_residual_scale": best["residual_scale"],
        "best_mean_h1_h4_recall_at_8": best["mean_h1_h4_recall_at_8"],
        "best_h2_request_macro_recall_at_8": best[
            "h2_request_macro_recall_at_8"
        ],
        "scale_metrics": summaries,
        "sealed_test_accessed": False,
    }


def _write_candidate_residual_scale_diagnostic(
    output_dir: Path, diagnostic: Mapping[str, Any]
) -> None:
    (output_dir / "residual_scale_diagnostic.json").write_text(
        json.dumps(dict(diagnostic), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_rows = []
    horizon_rows = []
    for result in diagnostic["scale_metrics"]:
        paired_summary = result["paired_request_h1_h4_gain_vs_alpha0"]
        summary_rows.append({
            "residual_scale": result["residual_scale"],
            "mean_h1_h4_recall_at_8": result["mean_h1_h4_recall_at_8"],
            "mean_h1_h8_recall_at_8": result["mean_h1_h8_recall_at_8"],
            "h2_request_macro_recall_at_8": result[
                "h2_request_macro_recall_at_8"
            ],
            "paired_h1_h4_gain_vs_alpha0": paired_summary["paired_gain_at_8"],
            "paired_h1_h4_ci95_low": paired_summary["ci95_low"],
            "paired_h1_h4_ci95_high": paired_summary["ci95_high"],
            "paired_h1_h4_requests": paired_summary["requests"],
            "paired_h1_h4_request_horizon_pairs": paired_summary[
                "request_horizon_pairs"
            ],
            "bootstrap_replicates": paired_summary["bootstrap_replicates"],
            "bootstrap_seed": paired_summary["bootstrap_seed"],
        })
        paired_by_horizon = {
            int(row["horizon"]): row
            for row in result["paired_request_horizon_gains_vs_alpha0"]
        }
        for row in result["horizon_metrics"]:
            horizon = int(row["horizon"])
            output_row = {"residual_scale": result["residual_scale"], **row}
            if horizon <= 4:
                paired = paired_by_horizon[horizon]
                output_row.update({
                    "paired_gain_vs_alpha0": paired["paired_gain_at_8"],
                    "paired_gain_ci95_low": paired["ci95_low"],
                    "paired_gain_ci95_high": paired["ci95_high"],
                    "paired_requests": paired["requests"],
                    "bootstrap_replicates": paired["bootstrap_replicates"],
                    "bootstrap_seed": paired["bootstrap_seed"],
                })
            horizon_rows.append(output_row)
    write_csv(output_dir / "residual_scale_summary.csv", summary_rows)
    write_csv(output_dir / "residual_scale_horizon_metrics.csv", horizon_rows)


def _write_metrics(output_dir: Path, prefix: str, metrics: Mapping[str, Any]) -> None:
    (output_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(dict(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for suffix, key in (
        ("horizon", "horizon_metrics"),
        ("request", "request_metrics"),
        ("layer", "layer_metrics"),
        ("domain", "domain_metrics"),
        ("position", "position_metrics"),
    ):
        write_csv(output_dir / f"{prefix}_{suffix}_metrics.csv", list(metrics[key]))


def evaluate_saved_jspace_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    *,
    train_pool: Path | None = None,
    validation_pool: Path | None = None,
    capture_dir: Path | None = None,
    mtp_dir: Path | None = None,
    target_features: Path | None = None,
    target_feature_rms: Path | None = None,
    precision_audit: Path | None = None,
    comparison_checkpoint: Path | None = None,
    paired_against_base: bool = False,
    split: str = "validation",
    residual_scale_grid: Sequence[float] | None = None,
    batch_size: int | None = None,
    device: str = "cuda:0",
    bootstrap_replicates: int = 2000,
    seed: int = 42,
    scale_bootstrap_replicates: int = DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_REPLICATES,
    scale_bootstrap_seed: int = DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_SEED,
    strict_lineage: bool = True,
) -> dict[str, Any]:
    """Evaluate one exact checkpoint state without entering a training path."""

    if split not in _ALLOWED_SCALE_SPLITS:
        # Refuse sealed/test-like paths before loading any checkpoint or pool.
        raise PermissionError("J-HARP evaluation permits only train/validation")
    checked_scales = (
        None
        if residual_scale_grid is None
        else _validated_residual_scales(residual_scale_grid)
    )
    if split != "validation" and checked_scales is None:
        raise ValueError("train evaluation is exposed only for residual-scale selection")
    if checked_scales is not None:
        if 0.0 not in checked_scales or 1.0 not in checked_scales:
            raise ValueError(
                "residual scale grid must explicitly contain alpha=0 and alpha=1"
            )
        scale_bootstrap_replicates, scale_bootstrap_seed = (
            _validate_scale_bootstrap_configuration(
                scale_bootstrap_replicates, scale_bootstrap_seed
            )
        )
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse evaluation directory {output_dir}")
    primary_payload = _load_payload(checkpoint)
    if (
        checked_scales is not None
        and primary_payload["schema"] != JSPACE_V2_TRAINING_SCHEMA
    ):
        raise ValueError("candidate residual scaling is supported only for v2 checkpoints")
    comparison_payload = (
        _load_payload(Path(comparison_checkpoint))
        if comparison_checkpoint is not None else None
    )
    if strict_lineage:
        recorded = _mapping(
            primary_payload.get("input_provenance"), "input provenance"
        )
        semantic = _mapping(
            recorded.get("semantic_lineage", {}), "semantic lineage"
        )
        if semantic.get("strict") is not True:
            raise ValueError("checkpoint lacks strict scientific lineage")
    paths = _resolve_paths(
        primary_payload,
        train_pool=train_pool,
        validation_pool=validation_pool,
        capture_dir=capture_dir,
        mtp_dir=mtp_dir,
        target_features=target_features,
        target_feature_rms=target_feature_rms,
        precision_audit=precision_audit,
    )
    use_v2_data = primary_payload["schema"] == JSPACE_V2_TRAINING_SCHEMA or (
        comparison_payload is not None
        and comparison_payload["schema"] == JSPACE_V2_TRAINING_SCHEMA
    )
    # Fail closed before opening pool tensors. Non-strict evaluation exists
    # only for historical/debug fixtures; scientific evaluation requires the
    # same v2 OOF proof as training.
    _read_pool_manifest(
        paths.train_pool,
        "train",
        allow_legacy_level2_train=not strict_lineage,
        require_context=use_v2_data,
    )
    _read_pool_manifest(
        paths.validation_pool,
        "validation",
        require_context=use_v2_data,
    )
    config = _mapping(primary_payload.get("model_config"), "model config")
    history = int(config.get("j_lags", 0))
    if history <= 0:
        raise ValueError("checkpoint has invalid J-history geometry")
    capture_manifest_path = paths.capture_dir / "manifest.json"
    rows_per_request: int
    if capture_manifest_path.is_file():
        capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
        rows_per_request = int(capture_manifest["routed_positions_per_request"])
    else:
        request_count = sum(
            1
            for line in (paths.capture_dir / "requests.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
        router_rows = int(np.load(
            paths.capture_dir / "raw_router_logits.npy",
            mmap_mode="r",
            allow_pickle=False,
        ).shape[0])
        if request_count <= 0 or router_rows % request_count:
            raise ValueError("cannot infer routed positions per request")
        rows_per_request = router_rows // request_count

    data_class = AlignedJContextCandidateData if use_v2_data else AlignedJCandidateData
    options = {
        "capture_dir": paths.capture_dir,
        "mtp_dir": paths.mtp_dir,
        "target_features": paths.target_features,
        "target_feature_rms": paths.target_feature_rms,
        "rows_per_request": rows_per_request,
        "history": history,
    }
    train_data = data_class(paths.train_pool, **options)
    validation_data = data_class(paths.validation_pool, **options)
    if use_v2_data:
        _assert_v2_matching_data(train_data, validation_data)  # type: ignore[arg-type]
    else:
        _assert_matching_data(train_data, validation_data)

    if strict_lineage:
        _verify_strict_lineage(primary_payload, train_data, validation_data, paths)
    else:
        _verify_nonstrict_provenance(primary_payload, paths)
    if comparison_payload is not None:
        for section in ("pools", "aligned_inputs"):
            primary_section = _mapping(
                primary_payload["input_provenance"], "primary provenance"
            ).get(section)
            comparison_section = _mapping(
                comparison_payload["input_provenance"], "comparison provenance"
            ).get(section)
            if canonical_json(_without_paths(primary_section)) != canonical_json(
                _without_paths(comparison_section)
            ):
                raise ValueError(f"comparison checkpoint {section} differs")
        if strict_lineage:
            comparison_semantic = _mapping(
                comparison_payload["input_provenance"], "comparison provenance"
            ).get("semantic_lineage")
            if _mapping(comparison_semantic, "comparison lineage").get("strict") is not True:
                raise ValueError("comparison checkpoint lacks strict lineage")

    saved_training = _mapping(primary_payload.get("training_config"), "training config")
    selected_data = train_data if split == "train" else validation_data
    maximum = saved_training.get(
        "max_train_rows" if split == "train" else "max_validation_rows"
    )
    selected_view = _RowLimitView(
        selected_data, None if maximum is None else int(maximum)
    )
    selected_batch_size = int(
        batch_size
        if batch_size is not None
        else saved_training.get("evaluation_batch_size", 1)
    )
    if selected_batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    if primary_payload["schema"] == JSPACE_V2_TRAINING_SCHEMA:
        primary_model = load_composite_jspace_v2_checkpoint(checkpoint, device=device)
    else:
        primary_model = load_composite_jspace_checkpoint(checkpoint, device=device)
    scale_diagnostic: dict[str, Any] | None = None
    if checked_scales is None:
        primary_metrics = evaluate_jspace_ranker(
            primary_model,
            selected_view,
            batch_size=selected_batch_size,
            device=device,
            autocast=str(device).startswith("cuda"),
        )
    else:
        scale_metrics = _evaluate_v2_residual_scales_once(
            primary_model,
            selected_view,
            checked_scales,
            batch_size=selected_batch_size,
            device=device,
            autocast=str(device).startswith("cuda"),
        )
        # Alpha=1 remains the exact saved-checkpoint artifact.
        primary_metrics = scale_metrics[1.0]
        scale_diagnostic = _candidate_residual_scale_diagnostic(
            checked_scales,
            scale_metrics,
            split=split,
            bootstrap_replicates=scale_bootstrap_replicates,
            bootstrap_seed=scale_bootstrap_seed,
        )
    comparison_metrics: dict[str, Any] | None = None
    comparison_bootstrap: dict[str, Any] | None = None
    if comparison_checkpoint is not None and comparison_payload is not None:
        if comparison_payload["schema"] == JSPACE_V2_TRAINING_SCHEMA:
            comparison_model = load_composite_jspace_v2_checkpoint(
                Path(comparison_checkpoint), device=device
            )
        else:
            comparison_model = load_composite_jspace_checkpoint(
                Path(comparison_checkpoint), device=device
            )
        comparison_metrics = evaluate_jspace_ranker(
            comparison_model,
            selected_view,
            batch_size=selected_batch_size,
            device=device,
            autocast=str(device).startswith("cuda"),
        )
        comparison_bootstrap = _paired_checkpoint_bootstrap(
            primary_metrics["request_metrics"],
            comparison_metrics["request_metrics"],
            native_k=int(primary_metrics["native_k"]),
            replicates=bootstrap_replicates,
            seed=seed,
        )
    base_bootstrap = (
        paired_request_bootstrap(
            primary_metrics["request_metrics"],
            replicates=bootstrap_replicates,
            seed=seed,
            native_k=int(primary_metrics["native_k"]),
        )
        if paired_against_base else None
    )

    output_dir.mkdir(parents=True)
    _write_metrics(output_dir, split, primary_metrics)
    if comparison_metrics is not None:
        _write_metrics(output_dir, f"comparison_{split}", comparison_metrics)
    if scale_diagnostic is not None:
        _write_candidate_residual_scale_diagnostic(output_dir, scale_diagnostic)
    if base_bootstrap is not None:
        (output_dir / "paired_against_base.json").write_text(
            json.dumps(base_bootstrap, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if comparison_bootstrap is not None:
        (output_dir / "paired_against_checkpoint.json").write_text(
            json.dumps(comparison_bootstrap, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema": JSPACE_CHECKPOINT_EVALUATION_SCHEMA,
        "checkpoint": _hash_file(checkpoint),
        "checkpoint_schema": primary_payload["schema"],
        "completed_epoch": int(primary_payload["completed_epoch"]),
        "global_step": int(primary_payload.get("global_step", 0)),
        "validation_rows": selected_view.rows,
        "evaluation_batch_size": selected_batch_size,
        "strict_lineage_verified": strict_lineage,
        "split": split,
        "sealed_test_accessed": False,
        "optimizer_constructed": False,
        "training_resume_invoked": False,
        "paired_against_base": base_bootstrap,
        "paired_against_checkpoint": comparison_bootstrap,
        "comparison_checkpoint": (
            _hash_file(Path(comparison_checkpoint))
            if comparison_checkpoint is not None else None
        ),
        **(
            {}
            if scale_diagnostic is None
            else {
                "residual_scale_diagnostic": {
                    "designation": scale_diagnostic["designation"],
                    "uses_same_split_labels_for_selection": scale_diagnostic[
                        "uses_same_split_labels_for_selection"
                    ],
                    "eligible_for_model_selection": scale_diagnostic[
                        "eligible_for_model_selection"
                    ],
                    "requested_scales": scale_diagnostic["requested_scales"],
                    "best_residual_scale": scale_diagnostic["best_residual_scale"],
                    "paired_reference_residual_scale": 0.0,
                    "exact_checkpoint_residual_scale": 1.0,
                    "h5_h8_exact_frozen_passthrough": True,
                    "bootstrap": scale_diagnostic["bootstrap"],
                    "creates_trained_checkpoint": False,
                    "path": "residual_scale_diagnostic.json",
                }
            }
        ),
        "outputs": {
            path.name: _hash_file(path) for path in sorted(output_dir.iterdir())
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-pool", type=Path)
    parser.add_argument("--validation-pool", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--mtp-dir", type=Path)
    parser.add_argument("--target-features", type=Path)
    parser.add_argument("--target-feature-rms", type=Path)
    parser.add_argument("--precision-audit", type=Path)
    parser.add_argument("--comparison-checkpoint", type=Path)
    parser.add_argument("--paired-against-base", action="store_true")
    parser.add_argument(
        "--split", choices=("train", "validation"), default="validation"
    )
    parser.add_argument(
        "--residual-scale-grid",
        type=parse_residual_scale_grid,
        help=(
            "explicit comma-separated v2 scale grid; must contain 0 and 1; "
            "validation use is post-hoc diagnostic only"
        ),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scale-bootstrap-replicates",
        type=int,
        default=DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--scale-bootstrap-seed",
        type=int,
        default=DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_SEED,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate_saved_jspace_checkpoint(
        args.checkpoint,
        args.output_dir,
        train_pool=args.train_pool,
        validation_pool=args.validation_pool,
        capture_dir=args.capture_dir,
        mtp_dir=args.mtp_dir,
        target_features=args.target_features,
        target_feature_rms=args.target_feature_rms,
        precision_audit=args.precision_audit,
        comparison_checkpoint=args.comparison_checkpoint,
        paired_against_base=args.paired_against_base,
        split=args.split,
        residual_scale_grid=args.residual_scale_grid,
        batch_size=args.batch_size,
        device=args.device,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        scale_bootstrap_replicates=args.scale_bootstrap_replicates,
        scale_bootstrap_seed=args.scale_bootstrap_seed,
        strict_lineage=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_REPLICATES",
    "DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_SEED",
    "JSPACE_CANDIDATE_RESIDUAL_SCALE_DIAGNOSTIC_SCHEMA",
    "JSPACE_CHECKPOINT_EVALUATION_SCHEMA",
    "JSpaceEvaluationPaths",
    "build_parser",
    "evaluate_saved_jspace_checkpoint",
    "parse_residual_scale_grid",
]
