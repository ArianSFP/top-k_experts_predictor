"""Reproduce the validation-only dense-J/raw ridge complementarity audit.

This is the durable implementation of the diagnostic reported in
``J_HARP_JLENS_COMPLEMENTARITY_AUDIT_20260807.md``.  It fits independent
per-layer ridge heads on deterministic *training* source rows and evaluates
all H1--H4-eligible rows from complete *validation* requests.  Test tensors
are never indexed: only the split labels and request IDs in ``requests.jsonl``
are inspected to count test requests.

The probe is intentionally narrow.  It consumes only the current dense
transported J-Lens PCA feature or raw-residual PCA feature and predicts four
future full-router logit vectors.  It does not consume route history, MTP, or
the nonlinear Full256 model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .jspace_features import FEATURE_SCHEMA, sha256_file
from .prepare_jspace_features import PREPARATION_SCHEMA


AUDIT_SCHEMA = "harp8_jlens_raw_complementarity_ridge_audit_v1"
HISTORICAL_AUDIT_SCHEMA = "harp8_jlens_raw_complementarity_audit_v1"
DEFAULT_HORIZONS = (1, 2, 3, 4)
DEFAULT_SEED = 20260807
DEFAULT_BOOTSTRAP_SEED = 20260810


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("values must be non-empty and unique")
    return parsed


def stable_topk(scores: np.ndarray, k: int = 8) -> np.ndarray:
    """Score-descending top-k with ascending expert ID at exact ties."""

    values = np.asarray(scores)
    if values.ndim < 2 or not 1 <= k <= values.shape[-1]:
        raise ValueError("stable top-k expects [..., experts] and a valid k")
    # The input axis is ordered by expert ID.  A stable score sort therefore
    # retains ascending ID at the cutoff when scores are exactly equal.
    return np.argsort(-values, axis=-1, kind="stable")[..., :k]


def slot_recall_rows(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Per-row slot recall with the authoritative native-k denominator."""

    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    if predicted.ndim != 2 or actual.ndim != 2 or len(predicted) != len(actual):
        raise ValueError("predicted and actual expert sets must be aligned matrices")
    if actual.shape[1] == 0:
        raise ValueError("native expert set may not be empty")
    return (predicted[:, :, None] == actual[:, None, :]).any(axis=1).sum(
        axis=1, dtype=np.int64
    ).astype(np.float64) / float(actual.shape[1])


def _read_requests(path: Path) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                split = str(record["offline_split"])
                request_id = int(record["request_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid request metadata at {path}:{line_number}"
                ) from exc
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"unsupported offline split {split!r}")
            requests.append(
                {**record, "offline_split": split, "request_id": request_id}
            )
    request_ids = [int(record["request_id"]) for record in requests]
    if not requests or len(request_ids) != len(set(request_ids)):
        raise ValueError("request metadata must contain unique request IDs")
    return requests


def request_source_rows(
    requests: Sequence[Mapping[str, Any]],
    split: str,
    *,
    rows_per_request: int,
    maximum_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return request-major source rows without ever forming test tensor rows."""

    if split not in {"train", "validation"}:
        raise PermissionError(
            "the complementarity audit supports train/validation only"
        )
    if rows_per_request <= maximum_horizon or maximum_horizon <= 0:
        raise ValueError("rows_per_request must exceed the maximum horizon")
    source_rows: list[int] = []
    request_ids: list[int] = []
    within_positions: list[int] = []
    eligible = rows_per_request - maximum_horizon
    for request_number, record in enumerate(requests):
        if str(record["offline_split"]) != split:
            continue
        base = request_number * rows_per_request
        source_rows.extend(range(base, base + eligible))
        request_ids.extend([int(record["request_id"])] * eligible)
        within_positions.extend(range(eligible))
    return (
        np.asarray(source_rows, dtype=np.int64),
        np.asarray(request_ids, dtype=np.int64),
        np.asarray(within_positions, dtype=np.int64),
    )


def deterministic_sample(rows: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
        raise ValueError("sampling population must contain unique rows")
    if maximum <= 0:
        raise ValueError("training sample size must be positive")
    if len(rows) <= maximum:
        return np.sort(rows.copy())
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=maximum, replace=False))


def standardize_train_validation(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Standardize from train statistics only, matching the historical audit."""

    train = np.asarray(train, dtype=np.float32)
    validation = np.asarray(validation, dtype=np.float32)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("ridge features must be aligned matrices")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("selected train/validation features contain non-finite values")
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    safe = np.maximum(std, np.float32(epsilon))
    return (
        np.asarray((train - mean) / safe, dtype=np.float32),
        np.asarray((validation - mean) / safe, dtype=np.float32),
        {
            "minimum_training_std": float(std.min()),
            "median_training_std": float(np.median(std)),
            "maximum_training_std": float(std.max()),
        },
    )


def fit_ridge_predict(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    train_y: np.ndarray,
    *,
    ridge_per_example: float,
    device: str | torch.device,
) -> np.ndarray:
    """Fit the historical centered-target primal ridge and predict validation."""

    if ridge_per_example <= 0:
        raise ValueError("ridge_per_example must be positive")
    train_x = np.ascontiguousarray(train_x, dtype=np.float32)
    validation_x = np.ascontiguousarray(validation_x, dtype=np.float32)
    train_y = np.ascontiguousarray(train_y, dtype=np.float32)
    if (
        train_x.ndim != 2
        or validation_x.ndim != 2
        or train_y.ndim != 2
        or len(train_x) != len(train_y)
        or train_x.shape[1] != validation_x.shape[1]
    ):
        raise ValueError(
            "ridge train, validation, and target matrices are incompatible"
        )
    target = torch.device(device)
    x = torch.from_numpy(train_x).to(device=target, dtype=torch.float32)
    xv = torch.from_numpy(validation_x).to(device=target, dtype=torch.float32)
    y = torch.from_numpy(train_y).to(device=target, dtype=torch.float32)
    y_mean = y.mean(dim=0, keepdim=True)
    gram = x.T @ x
    gram.diagonal().add_(float(ridge_per_example) * x.shape[0])
    weights = torch.linalg.solve(gram, x.T @ (y - y_mean))
    prediction = xv @ weights + y_mean
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("ridge prediction contains a non-finite value")
    return prediction.cpu().numpy()


def _group_request_mean(
    values: np.ndarray,
    request_codes: np.ndarray,
    request_count: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    request_codes = np.asarray(request_codes, dtype=np.int64)
    if values.ndim != 1 or request_codes.shape != values.shape:
        raise ValueError("request aggregation inputs are incompatible")
    if (
        request_count <= 0
        or (request_codes < 0).any()
        or (request_codes >= request_count).any()
    ):
        raise ValueError("request aggregation code lies outside its namespace")
    counts = np.bincount(request_codes, minlength=request_count)
    if (counts == 0).any():
        raise ValueError("a validation request has no eligible rows")
    sums = np.bincount(request_codes, weights=values, minlength=request_count)
    return sums / counts


def _bootstrap_delta(
    delta: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or not len(delta):
        raise ValueError("bootstrap requires one value per complete request")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    sampled = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = rng.integers(0, len(delta), size=len(delta))
        sampled[replicate] = delta[indices].mean()
    return {
        "gain": float(delta.mean()),
        "ci95": [float(value) for value in np.quantile(sampled, (0.025, 0.975))],
    }


def paired_request_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    horizons: Sequence[int],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired confidence intervals over complete requests, by H and H1--H4."""

    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if (
        candidate.ndim != 2
        or baseline.ndim != 2
        or candidate.shape != baseline.shape
        or candidate.shape[1] != len(horizons)
    ):
        raise ValueError("paired request matrices are incompatible")
    delta = candidate - baseline
    result = {
        "mean_h1_h4": _bootstrap_delta(
            delta.mean(axis=1), replicates=replicates, seed=seed
        )
    }
    for index, horizon in enumerate(horizons):
        result[f"h{int(horizon)}"] = _bootstrap_delta(
            delta[:, index], replicates=replicates, seed=seed
        )
    result.update(
        {
            "bootstrap_replicates": int(replicates),
            "bootstrap_seed": int(seed),
            "resampling_unit": "complete validation request",
        }
    )
    return result


def _find_artifact_record(value: Any, filename: str) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        path = value.get("path")
        if isinstance(path, str) and Path(path).name == filename and "sha256" in value:
            return dict(value)
        for child in value.values():
            found = _find_artifact_record(child, filename)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _find_artifact_record(child, filename)
            if found is not None:
                return found
    return None


def _validate_feature_contract(
    feature_root: Path,
    *,
    rank: int,
    rows_per_request: int | None,
) -> tuple[dict[str, Any], int, Path, Path]:
    root_manifest_path = Path(feature_root) / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    if root_manifest.get("schema") != PREPARATION_SCHEMA:
        raise ValueError("feature root has an incompatible preparation schema")
    if root_manifest.get("fit", {}).get("split") != "train":
        raise PermissionError("J/raw PCA must be fitted on the train split only")
    manifest_rows = int(root_manifest.get("geometry", {}).get("rows_per_request", -1))
    if manifest_rows <= 0:
        raise ValueError("feature manifest lacks rows-per-request geometry")
    if rows_per_request is not None and int(rows_per_request) != manifest_rows:
        raise ValueError("requested rows_per_request disagrees with feature provenance")
    pair_hash = str(root_manifest.get("fit", {}).get("pairs_sha256", ""))
    if len(pair_hash) != 64:
        raise ValueError("feature manifest lacks the train PCA fit-pair hash")

    feature_paths: dict[str, Path] = {}
    rank_manifests: dict[str, dict[str, Any]] = {}
    for representation in ("j_lens", "raw_residual"):
        try:
            record = root_manifest["representations"][representation]["ranks"][
                str(rank)
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"feature manifest lacks {representation} rank {rank}"
            ) from exc
        store = record.get("feature_store", {})
        if (
            store.get("schema") != FEATURE_SCHEMA
            or store.get("representation") != representation
        ):
            raise ValueError(
                f"invalid {representation} rank-{rank} feature-store contract"
            )
        if int(store.get("feature_width", -1)) != rank:
            raise ValueError(
                f"{representation} feature width does not equal requested rank"
            )
        if store.get("pca", {}).get("fit_pairs_sha256") != pair_hash:
            raise ValueError("J and raw stores do not share the frozen train fit pairs")
        path = (
            Path(feature_root)
            / representation
            / f"rank{rank}"
            / "features_normalized.npy"
        )
        output = store.get("outputs", {}).get("features_normalized.npy", {})
        if len(str(output.get("sha256", ""))) != 64:
            raise ValueError(f"{representation} store lacks an immutable feature hash")
        if int(output.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"{representation} feature byte count changed")
        feature_paths[representation] = path
        rank_manifests[representation] = store
    return (
        {
            "root_manifest": root_manifest,
            "root_manifest_path": root_manifest_path,
            "rank_manifests": rank_manifests,
        },
        manifest_rows,
        feature_paths["j_lens"],
        feature_paths["raw_residual"],
    )


def _validate_arrays(
    *,
    requests: Sequence[Mapping[str, Any]],
    rows_per_request: int,
    j_features: np.ndarray,
    raw_features: np.ndarray,
    router_logits: np.ndarray,
    top8: np.ndarray,
    layers: Sequence[int],
    horizons: Sequence[int],
) -> tuple[int, int, int]:
    expected_rows = len(requests) * rows_per_request
    if j_features.ndim != 3 or raw_features.ndim != 3:
        raise ValueError("J/raw features must have shape [rows,layers,width]")
    if j_features.shape[:2] != raw_features.shape[:2]:
        raise ValueError("J/raw feature rows and layers are not aligned")
    if j_features.shape[0] != expected_rows:
        raise ValueError("feature rows disagree with requests and rows_per_request")
    layer_count = int(j_features.shape[1])
    if router_logits.ndim != 3 or router_logits.shape[:2] != (
        expected_rows,
        layer_count,
    ):
        raise ValueError("router logits disagree with feature geometry")
    experts = int(router_logits.shape[2])
    if top8.ndim != 3 or top8.shape[:2] != (expected_rows, layer_count):
        raise ValueError("authoritative top-k disagrees with feature geometry")
    native_k = int(top8.shape[2])
    if native_k != 8:
        raise ValueError(
            "the historical complementarity contract requires native top-8"
        )
    if experts < native_k:
        raise ValueError("expert namespace is smaller than native top-k")
    if not layers or min(layers) < 0 or max(layers) >= layer_count:
        raise ValueError("requested layer lies outside captured geometry")
    if tuple(horizons) != tuple(sorted(horizons)) or min(horizons) <= 0:
        raise ValueError("horizons must be positive and increasing")
    if max(horizons) >= rows_per_request:
        raise ValueError("horizon crosses the fixed request geometry")
    return layer_count, experts, native_k


def _request_order(row_request_ids: np.ndarray) -> np.ndarray:
    order: list[int] = []
    seen: set[int] = set()
    for value in row_request_ids.tolist():
        request_id = int(value)
        if request_id not in seen:
            seen.add(request_id)
            order.append(request_id)
    return np.asarray(order, dtype=np.int64)


def _request_group_codes(
    row_request_ids: np.ndarray, ordered_request_ids: np.ndarray
) -> np.ndarray:
    """Map each row to its stable request-order index in linear time."""

    mapping = {
        int(request_id): index
        for index, request_id in enumerate(ordered_request_ids.tolist())
    }
    try:
        return np.fromiter(
            (mapping[int(request_id)] for request_id in row_request_ids.tolist()),
            dtype=np.int64,
            count=len(row_request_ids),
        )
    except KeyError as error:
        raise ValueError(f"unknown validation request ID {error.args[0]}") from error


def _recall_summary(values: np.ndarray, horizons: Sequence[int]) -> dict[str, float]:
    result = {
        f"h{int(h)}": float(values[:, index].mean()) for index, h in enumerate(horizons)
    }
    result["mean_h1_h4"] = float(values.mean())
    return result


def run_complementarity_audit(
    *,
    feature_root: Path,
    capture_dir: Path,
    output_dir: Path,
    rank: int = 512,
    rows_per_request: int | None = None,
    train_sample: int = 4096,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    layers: Sequence[int] | None = None,
    ridge_per_example: float = 0.01,
    seed: int = DEFAULT_SEED,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    device: str = "cpu",
    export_predictions: bool = False,
    command_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit and evaluate the immutable train/validation complementarity probe."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse audit output directory {output_dir}")
    if rank <= 0 or train_sample <= 0 or ridge_per_example <= 0:
        raise ValueError("rank, train_sample, and ridge_per_example must be positive")
    horizons = tuple(int(value) for value in horizons)
    feature_contract, manifest_rpr, j_path, raw_path = _validate_feature_contract(
        Path(feature_root), rank=rank, rows_per_request=rows_per_request
    )
    rows_per_request = manifest_rpr
    capture_dir = Path(capture_dir)
    requests_path = capture_dir / "requests.jsonl"
    router_path = capture_dir / "raw_router_logits.npy"
    top8_path = capture_dir / "top8_expert_ids.npy"
    capture_manifest_path = capture_dir / "manifest.json"
    requests = _read_requests(requests_path)

    # Opening an NPY memmap reads only its header.  Every later advanced index
    # is built solely from train/validation request metadata; no test tensor
    # row is ever materialized.
    j_features = np.load(j_path, mmap_mode="r", allow_pickle=False)
    raw_features = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    router_logits = np.load(router_path, mmap_mode="r", allow_pickle=False)
    top8 = np.load(top8_path, mmap_mode="r", allow_pickle=False)
    layer_values = (
        tuple(range(int(j_features.shape[1])))
        if layers is None
        else tuple(int(value) for value in layers)
    )
    layer_count, experts, native_k = _validate_arrays(
        requests=requests,
        rows_per_request=rows_per_request,
        j_features=j_features,
        raw_features=raw_features,
        router_logits=router_logits,
        top8=top8,
        layers=layer_values,
        horizons=horizons,
    )

    train_eligible, _train_request_ids, _train_within = request_source_rows(
        requests,
        "train",
        rows_per_request=rows_per_request,
        maximum_horizon=max(horizons),
    )
    validation_rows, validation_row_request_ids, validation_within = (
        request_source_rows(
            requests,
            "validation",
            rows_per_request=rows_per_request,
            maximum_horizon=max(horizons),
        )
    )
    train_rows = deterministic_sample(train_eligible, train_sample, seed)
    if not len(validation_rows):
        raise ValueError("no H1-H4-eligible validation source rows")
    validation_request_ids = _request_order(validation_row_request_ids)
    validation_request_codes = _request_group_codes(
        validation_row_request_ids, validation_request_ids
    )
    conditions = (f"j{rank}", f"raw{rank}", f"dual{rank}")
    j_name, raw_name, dual_name = conditions
    request_recall = {
        condition: np.zeros(
            (len(validation_request_ids), len(horizons)), dtype=np.float64
        )
        for condition in conditions
    }
    request_union_coverage = np.zeros_like(request_recall[j_name])
    request_jaccard = np.zeros_like(request_recall[j_name])
    request_union_size = np.zeros_like(request_recall[j_name])
    prediction_store = (
        {
            condition: np.empty(
                (len(validation_rows), len(horizons), len(layer_values), native_k),
                dtype=np.uint16,
            )
            for condition in conditions
        }
        if export_predictions
        else None
    )
    per_layer_metrics: list[dict[str, Any]] = []
    feature_statistics: dict[str, Any] = {}
    started = time.monotonic()

    target_device = torch.device(device)
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA audit requested but CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    for layer_index, layer in enumerate(layer_values):
        train_j, validation_j, j_stats = standardize_train_validation(
            np.asarray(j_features[train_rows, layer], dtype=np.float32),
            np.asarray(j_features[validation_rows, layer], dtype=np.float32),
        )
        train_raw, validation_raw, raw_stats = standardize_train_validation(
            np.asarray(raw_features[train_rows, layer], dtype=np.float32),
            np.asarray(raw_features[validation_rows, layer], dtype=np.float32),
        )
        feature_statistics[str(layer)] = {j_name: j_stats, raw_name: raw_stats}
        train_targets = np.concatenate(
            [
                np.asarray(router_logits[train_rows + horizon, layer], dtype=np.float32)
                for horizon in horizons
            ],
            axis=1,
        )
        if not np.isfinite(train_targets).all():
            raise ValueError("selected train router targets contain non-finite values")
        feature_pairs = {
            j_name: (train_j, validation_j),
            raw_name: (train_raw, validation_raw),
            dual_name: (
                np.concatenate((train_j, train_raw), axis=1),
                np.concatenate((validation_j, validation_raw), axis=1),
            ),
        }
        predictions = {
            condition: fit_ridge_predict(
                train_x,
                validation_x,
                train_targets,
                ridge_per_example=ridge_per_example,
                device=target_device,
            )
            for condition, (train_x, validation_x) in feature_pairs.items()
        }
        layer_record: dict[str, Any] = {"layer": int(layer)}
        for horizon_index, horizon in enumerate(horizons):
            start = horizon_index * experts
            stop = start + experts
            actual = np.asarray(top8[validation_rows + horizon, layer], dtype=np.int64)
            if (
                (actual < 0).any()
                or (actual >= experts).any()
                or (np.diff(np.sort(actual, axis=1), axis=1) == 0).any()
            ):
                raise ValueError("selected authoritative top-8 IDs are invalid")
            ids = {
                condition: stable_topk(prediction[:, start:stop], native_k)
                for condition, prediction in predictions.items()
            }
            for condition in conditions:
                recalls = slot_recall_rows(ids[condition], actual)
                request_recall[condition][:, horizon_index] += _group_request_mean(
                    recalls,
                    validation_request_codes,
                    len(validation_request_ids),
                )
                layer_record[f"{condition}_h{horizon}_recall_at_8"] = float(
                    recalls.mean()
                )
                if prediction_store is not None:
                    prediction_store[condition][:, horizon_index, layer_index] = ids[
                        condition
                    ].astype(np.uint16, copy=False)

            intersections = (
                (ids[j_name][:, :, None] == ids[raw_name][:, None, :])
                .any(axis=2)
                .sum(axis=1)
            )
            union_coverage = (
                (actual[:, :, None] == ids[j_name][:, None, :]).any(axis=2)
                | (actual[:, :, None] == ids[raw_name][:, None, :]).any(axis=2)
            ).sum(axis=1).astype(np.float64) / native_k
            union_size = (2 * native_k - intersections).astype(np.float64)
            jaccard = intersections.astype(np.float64) / union_size
            request_union_coverage[:, horizon_index] += _group_request_mean(
                union_coverage,
                validation_request_codes,
                len(validation_request_ids),
            )
            request_jaccard[:, horizon_index] += _group_request_mean(
                jaccard,
                validation_request_codes,
                len(validation_request_ids),
            )
            request_union_size[:, horizon_index] += _group_request_mean(
                union_size,
                validation_request_codes,
                len(validation_request_ids),
            )
        per_layer_metrics.append(layer_record)

    divisor = float(len(layer_values))
    for values in request_recall.values():
        values /= divisor
    request_union_coverage /= divisor
    request_jaccard /= divisor
    request_union_size /= divisor
    paired = paired_request_bootstrap(
        request_recall[dual_name],
        request_recall[raw_name],
        horizons=horizons,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    request_rows_json: list[dict[str, Any]] = []
    for request_index, request_id in enumerate(validation_request_ids.tolist()):
        for horizon_index, horizon in enumerate(horizons):
            request_rows_json.append(
                {
                    "request_id": int(request_id),
                    "horizon": int(horizon),
                    f"{j_name}_recall_at_8": float(
                        request_recall[j_name][request_index, horizon_index]
                    ),
                    f"{raw_name}_recall_at_8": float(
                        request_recall[raw_name][request_index, horizon_index]
                    ),
                    f"{dual_name}_recall_at_8": float(
                        request_recall[dual_name][request_index, horizon_index]
                    ),
                    "j_raw_union_target_coverage": float(
                        request_union_coverage[request_index, horizon_index]
                    ),
                    "j_raw_top8_jaccard": float(
                        request_jaccard[request_index, horizon_index]
                    ),
                    "j_raw_union_size": float(
                        request_union_size[request_index, horizon_index]
                    ),
                }
            )

    split_counts = {
        split: sum(str(record["offline_split"]) == split for record in requests)
        for split in ("train", "validation", "test")
    }
    root_manifest = feature_contract["root_manifest"]
    capture_manifest = (
        json.loads(capture_manifest_path.read_text(encoding="utf-8"))
        if capture_manifest_path.is_file()
        else None
    )
    feature_shas = {
        representation: str(
            feature_contract["rank_manifests"][representation]["outputs"][
                "features_normalized.npy"
            ]["sha256"]
        )
        for representation in ("j_lens", "raw_residual")
    }
    router_record = (
        _find_artifact_record(capture_manifest, router_path.name)
        if capture_manifest is not None
        else None
    )
    top8_record = (
        _find_artifact_record(capture_manifest, top8_path.name)
        if capture_manifest is not None
        else None
    )
    invocation = list(command_argv) if command_argv is not None else list(sys.argv)
    summary: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "historical_contract_schema": HISTORICAL_AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_test_accessed": False,
        "fit_split": "train",
        "evaluation_split": "validation",
        "command": shlex.join(invocation),
        "configuration": {
            "rank": int(rank),
            "layers": [int(value) for value in layer_values],
            "horizons": [int(value) for value in horizons],
            "rows_per_request": int(rows_per_request),
            "train_sample": int(train_sample),
            "ridge_per_example": float(ridge_per_example),
            "seed": int(seed),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
            "device": str(target_device),
            "torch_threads": int(torch.get_num_threads()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "stable_tie_break": "score_descending_expert_id_ascending",
        },
        "data": {
            "request_counts": {
                "train": int(split_counts["train"]),
                "validation": int(split_counts["validation"]),
                "test_metadata_only": int(split_counts["test"]),
            },
            "train_eligible_source_rows": int(len(train_eligible)),
            "train_source_rows": int(len(train_rows)),
            "validation_source_rows": int(len(validation_rows)),
            "validation_requests": int(len(validation_request_ids)),
            "layers": int(layer_count),
            "experts": int(experts),
            "native_k": int(native_k),
            "train_source_rows_sha256": _array_sha256(train_rows.astype("<i8")),
            "validation_source_rows_sha256": _array_sha256(
                validation_rows.astype("<i8")
            ),
        },
        "provenance": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "feature_root_manifest": {
                "path": str(feature_contract["root_manifest_path"]),
                "sha256": sha256_file(feature_contract["root_manifest_path"]),
                "schema": root_manifest["schema"],
                "pca_fit_pairs_sha256": root_manifest["fit"]["pairs_sha256"],
            },
            "j_lens_features": {"path": str(j_path), "sha256": feature_shas["j_lens"]},
            "raw_residual_features": {
                "path": str(raw_path),
                "sha256": feature_shas["raw_residual"],
            },
            "requests_jsonl": {
                "path": str(requests_path),
                "sha256": sha256_file(requests_path),
            },
            "capture_manifest": (
                None
                if capture_manifest is None
                else {
                    "path": str(capture_manifest_path),
                    "sha256": sha256_file(capture_manifest_path),
                }
            ),
            # These hashes are trusted immutable-capture manifest values.  The
            # audit deliberately does not stream/hash all rows, which would
            # read sealed test tensor bytes merely for provenance.
            "router_logits": {
                "path": str(router_path),
                "manifest_sha256": (
                    None if router_record is None else router_record["sha256"]
                ),
                "full_array_rehashed": False,
            },
            "native_top8": {
                "path": str(top8_path),
                "manifest_sha256": (
                    None if top8_record is None else top8_record["sha256"]
                ),
                "full_array_rehashed": False,
            },
        },
        "future_router_linear_probe": {
            "layers": int(len(layer_values)),
            "horizons": [int(value) for value in horizons],
            "train_source_rows": int(len(train_rows)),
            "validation_source_rows": int(len(validation_rows)),
            "validation_requests": int(len(validation_request_ids)),
            "ridge_per_example": float(ridge_per_example),
            f"{j_name}_recall_at_8": _recall_summary(request_recall[j_name], horizons),
            f"{raw_name}_recall_at_8": _recall_summary(
                request_recall[raw_name], horizons
            ),
            f"{dual_name}_recall_at_8": _recall_summary(
                request_recall[dual_name], horizons
            ),
            f"paired_{dual_name}_minus_{raw_name}": paired,
            "j_raw_top8_union": {
                "mean_h1_h4_target_coverage": float(request_union_coverage.mean()),
                "mean_union_size": float(request_union_size.mean()),
                "mean_jaccard": float(request_jaccard.mean()),
            },
            "per_layer_metrics": per_layer_metrics,
            "feature_standardization": feature_statistics,
        },
        "limitations": [
            "The probe is linear and omits route history, MTP, and nonlinear Full256 fusion.",
            "Only request metadata for the sealed test split was counted; no test tensor row was indexed.",
            "Feature-array hashes are trusted from immutable preparation manifests to avoid reading test tensor bytes.",
        ],
        "elapsed_seconds": float(time.monotonic() - started),
    }

    output_dir.mkdir(parents=True)
    request_metrics_path = output_dir / "validation_request_metrics.jsonl"
    request_metrics_path.write_text(
        "".join(_canonical_json(row) + "\n" for row in request_rows_json),
        encoding="utf-8",
    )
    summary["outputs"] = {
        "validation_request_metrics": {
            "path": str(request_metrics_path),
            "sha256": sha256_file(request_metrics_path),
            "rows": len(request_rows_json),
        }
    }
    if prediction_store is not None:
        predictions_path = output_dir / "validation_predicted_top8.npz"
        np.savez_compressed(
            predictions_path,
            source_row=validation_rows.astype(np.int64),
            request_id=validation_row_request_ids.astype(np.int64),
            within_request=validation_within.astype(np.int16),
            horizons=np.asarray(horizons, dtype=np.int16),
            layers=np.asarray(layer_values, dtype=np.int16),
            **{name: values for name, values in prediction_store.items()},
            contains_target_top8=np.asarray(False),
            contains_router_targets=np.asarray(False),
            sealed_test_accessed=np.asarray(False),
        )
        summary["outputs"]["validation_predictions"] = {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
            "contains_labels": False,
        }
    summary_path = output_dir / "audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--rows-per-request", type=int)
    parser.add_argument("--train-sample", type=int, default=4096)
    parser.add_argument("--horizons", type=_parse_int_csv, default=DEFAULT_HORIZONS)
    parser.add_argument(
        "--layers",
        type=_parse_int_csv,
        help="comma-separated diagnostic override; default is every captured layer",
    )
    parser.add_argument("--ridge-per-example", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=32)
    parser.add_argument("--export-predictions", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.torch_threads <= 0:
        raise SystemExit("--torch-threads must be positive")
    torch.set_num_threads(min(int(args.torch_threads), max(1, torch.get_num_threads())))
    invocation = [sys.executable, "-m", "harp8.audit_jlens_complementarity"] + list(
        sys.argv[1:] if argv is None else argv
    )
    summary = run_complementarity_audit(
        feature_root=args.feature_root,
        capture_dir=args.capture_dir,
        output_dir=args.output_dir,
        rank=args.rank,
        rows_per_request=args.rows_per_request,
        train_sample=args.train_sample,
        horizons=args.horizons,
        layers=args.layers,
        ridge_per_example=args.ridge_per_example,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        device=args.device,
        export_predictions=args.export_predictions,
        command_argv=invocation,
    )
    print(json.dumps(summary["future_router_linear_probe"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_SCHEMA",
    "build_parser",
    "deterministic_sample",
    "fit_ridge_predict",
    "paired_request_bootstrap",
    "request_source_rows",
    "run_complementarity_audit",
    "slot_recall_rows",
    "stable_topk",
    "standardize_train_validation",
]
