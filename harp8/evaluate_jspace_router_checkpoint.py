"""Bounded train/validation-only evaluation of saved Full256 checkpoints.

The command loads an exact ``best.pt`` or ``last.pt`` model state without
constructing an optimizer or entering the trainer.  All capture, pool, target
stream, and baseline-expansion inputs are supplied explicitly and checked
against checkpoint provenance before inference.  Test-like splits are refused
before any candidate-pool tensor is opened.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .jspace_router_data import (
    FullRouterForecastData,
    _read_pool_manifest,
    causal_router_forecaster_inputs,
)
from .jspace_router_metrics import (
    _METRIC_NAMES,
    _aggregate_metric_events,
    _mean,
    evaluate_full_router_forecaster,
    slot_recall_at_k,
)
from .train import canonical_json, sha256_file, write_csv
from .train_jspace_reranker import (
    _input_provenance,
    _validate_scientific_lineage,
)
from .train_jspace_router_forecaster import (
    JSPACE_ROUTER_ARCHITECTURE_NAME,
    JSPACE_ROUTER_TRAINING_SCHEMA,
    _assert_matching,
    _baseline_expansion_contract,
    _feature_representation,
    _simple_provenance,
    _target_state_stream_contract,
    _validate_secondary_feature_lineage,
    load_jspace_router_forecaster_checkpoint,
)


FULL_ROUTER_CHECKPOINT_EVALUATION_SCHEMA = (
    "harp8_jspace_full_router_checkpoint_evaluation_v1"
)
FULL_ROUTER_RESIDUAL_SCALE_DIAGNOSTIC_SCHEMA = (
    "harp8_jspace_full_router_residual_scale_diagnostic_v1"
)
DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_REPLICATES = 5_000
DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_SEED = 20_260_807
_ALLOWED_SPLITS = frozenset({"train", "validation"})


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
    """Parse the explicit comma-separated post-hoc residual scale grid."""

    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "residual scale grid must be a comma-separated list of numbers"
        )
    try:
        return _validated_residual_scales(tuple(float(part) for part in parts))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_residual_scale(value: str) -> float:
    """Parse one finite nonnegative residual scale for prediction export."""

    try:
        return _validated_residual_scales((float(value),))[0]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _validate_bootstrap_configuration(replicates: int, seed: int) -> tuple[int, int]:
    replicates = int(replicates)
    seed = int(seed)
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if seed < 0:
        raise ValueError("bootstrap seed must be nonnegative")
    return replicates, seed


def _paired_request_bootstrap(
    paired_gains: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Deterministic percentile bootstrap over paired request-level gains."""

    replicates, seed = _validate_bootstrap_configuration(replicates, seed)
    values = np.asarray(paired_gains, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired request bootstrap requires a nonempty vector")
    if not np.isfinite(values).all():
        raise ValueError("paired request gains must be finite")
    generator = np.random.default_rng(seed)
    statistics = np.empty(replicates, dtype=np.float64)
    # Bound peak memory for larger validation corpora while preserving the
    # exact RNG stream and result for a given seed/replicate count.
    chunk_size = min(512, replicates)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        indices = generator.integers(
            0,
            len(values),
            size=(stop - start, len(values)),
            endpoint=False,
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


@dataclass(frozen=True)
class FullRouterEvaluationPaths:
    train_pool: Path
    validation_pool: Path
    capture_dir: Path
    mtp_dir: Path
    target_features: Path
    target_feature_rms: Path | None = None
    secondary_target_features: Path | None = None
    secondary_target_feature_rms: Path | None = None
    precision_audit: Path | None = None


class _RowLimitView:
    """Read-only prefix view that preserves request metadata and split gates."""

    def __init__(self, base: FullRouterForecastData, maximum: int | None) -> None:
        self.base = base
        self.rows = base.rows if maximum is None else min(base.rows, int(maximum))
        if self.rows <= 0:
            raise ValueError("evaluation view must contain at least one row")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def batch(
        self, rows: np.ndarray, device: str | torch.device
    ) -> dict[str, torch.Tensor]:
        values = np.asarray(rows, dtype=np.int64)
        if values.ndim != 1 or (values < 0).any() or (values >= self.rows).any():
            raise IndexError("evaluation rows lie outside the bounded view")
        return self.base.batch(values, device)

    def sequential_batches(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("evaluation batch size must be positive")
        for start in range(0, self.rows, batch_size):
            yield np.arange(start, min(self.rows, start + batch_size), dtype=np.int64)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


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


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != (
        JSPACE_ROUTER_TRAINING_SCHEMA
    ):
        raise ValueError("checkpoint is not a supported Full256 training state")
    if payload.get("architecture") not in (None, JSPACE_ROUTER_ARCHITECTURE_NAME):
        raise ValueError("checkpoint architecture is not Full256 v1")
    if not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("checkpoint contains no model state")
    if int(payload.get("completed_epoch", -1)) < 0:
        raise ValueError("checkpoint has an invalid completed epoch")
    if not isinstance(payload.get("input_provenance"), Mapping):
        raise ValueError("checkpoint lacks input provenance")
    return payload


def _infer_rows_per_request(capture_dir: Path) -> int:
    manifest_path = Path(capture_dir) / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = int(manifest.get("routed_positions_per_request", 0))
        if value > 0:
            return value
    requests_path = Path(capture_dir) / "requests.jsonl"
    request_count = sum(
        bool(line.strip())
        for line in requests_path.read_text(encoding="utf-8").splitlines()
    )
    router_rows = int(
        np.load(
            Path(capture_dir) / "raw_router_logits.npy",
            mmap_mode="r",
            allow_pickle=False,
        ).shape[0]
    )
    quotient, remainder = divmod(router_rows, request_count)
    if request_count <= 0 or remainder or quotient <= 0:
        raise ValueError("cannot infer a constant routed-position count")
    return quotient


def _assert_debug_provenance(
    payload: Mapping[str, Any],
    train: FullRouterForecastData,
    validation: FullRouterForecastData,
    paths: FullRouterEvaluationPaths,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    recorded = _mapping(payload.get("input_provenance"), "input provenance")
    if bool(recorded.get("strict", False)):
        raise ValueError("debug evaluation cannot bypass strict checkpoint lineage")
    current = _simple_provenance(
        train,
        validation,
        paths.target_features,
        paths.target_feature_rms,
        paths.secondary_target_features,
        paths.secondary_target_feature_rms,
    )
    # Training appends execution-only sections after building this core debug
    # provenance. Compare every core field while allowing those extra recorded
    # sections to remain checkpoint-owned.
    for section, value in current.items():
        if canonical_json(_without_paths(value)) != canonical_json(
            _without_paths(recorded.get(section))
        ):
            raise ValueError(
                "supplied paths differ from checkpoint input_provenance " f"{section}"
            )
    # Debug primary streams predate mandatory content hashes. Fail closed on
    # their exact recorded locations in addition to shape/dtype comparison.
    for name, supplied in (
        ("target_features", paths.target_features),
        ("target_feature_rms", paths.target_feature_rms),
    ):
        expected = recorded.get(name)
        if expected is None and supplied is None:
            continue
        if not isinstance(expected, Mapping) or supplied is None:
            raise ValueError(f"checkpoint and supplied {name} availability differs")
        if Path(str(expected.get("path", ""))).resolve() != Path(supplied).resolve():
            raise ValueError(f"supplied {name} path differs from debug checkpoint")
    primary = {
        "strict": False,
        "representation": _feature_representation(paths.target_features),
    }
    secondary = (
        {
            "strict": False,
            "representation": _feature_representation(paths.secondary_target_features),
        }
        if paths.secondary_target_features is not None
        else None
    )
    return primary, secondary


def _assert_strict_provenance(
    payload: Mapping[str, Any],
    train: FullRouterForecastData,
    validation: FullRouterForecastData,
    paths: FullRouterEvaluationPaths,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    recorded = _mapping(payload.get("input_provenance"), "input provenance")
    recorded_semantic = _mapping(recorded.get("semantic_lineage"), "semantic lineage")
    if recorded_semantic.get("strict") is not True:
        raise ValueError("checkpoint lacks strict scientific lineage")
    precision = _mapping(
        recorded_semantic.get("precision_attribution", {}),
        "precision attribution",
    )
    provisional = precision.get("attribution") == "provisional"
    primary = _validate_scientific_lineage(
        train.aligned,
        validation.aligned,
        target_features=paths.target_features,
        target_feature_rms=paths.target_feature_rms,
        precision_audit=paths.precision_audit,
        allow_provisional_j_without_probe=provisional,
    )
    primary.update(
        {
            "architecture": JSPACE_ROUTER_ARCHITECTURE_NAME,
            "full_router_supervision": True,
            "full_expert_namespace": True,
            "acceptance_labels_used": False,
            "decision_profile": recorded_semantic.get("decision_profile"),
            "mtp_timing_causal": False,
        }
    )
    secondary = None
    if train.secondary_target_enabled:
        if paths.secondary_target_features is None:
            raise ValueError("dual-stream checkpoint requires secondary features")
        secondary = _validate_secondary_feature_lineage(
            train,
            primary_semantic=primary,
            target_features=paths.secondary_target_features,
            target_feature_rms=paths.secondary_target_feature_rms,
        )
    router_provenance = _mapping(recorded.get("router_keys", {}), "router provenance")
    current = _input_provenance(
        train.aligned,
        validation.aligned,
        target_features=paths.target_features,
        target_feature_rms=paths.target_feature_rms,
        router_provenance=router_provenance,
        semantic_lineage=primary,
    )
    if secondary is not None:
        current["secondary_target_semantic_lineage"] = secondary
    for section in (
        "pools",
        "aligned_inputs",
        "router_keys",
        "semantic_lineage",
        "secondary_target_semantic_lineage",
    ):
        if section not in current and section not in recorded:
            continue
        if canonical_json(_without_paths(current.get(section))) != canonical_json(
            _without_paths(recorded.get(section))
        ):
            raise ValueError(
                f"supplied paths differ from checkpoint input_provenance {section}"
            )
    return primary, secondary


def _assert_checkpoint_contracts(
    payload: Mapping[str, Any],
    train: FullRouterForecastData,
    validation: FullRouterForecastData,
    paths: FullRouterEvaluationPaths,
    *,
    strict_lineage: bool,
) -> None:
    _assert_matching(train, validation)
    recorded_baseline = _mapping(
        payload.get("baseline_expansion_contract"),
        "baseline expansion contract",
    )
    current_baseline = _baseline_expansion_contract(train)
    if canonical_json(current_baseline) != canonical_json(recorded_baseline):
        raise ValueError(
            "supplied pools violate checkpoint baseline_expansion contract"
        )

    if strict_lineage:
        primary, secondary = _assert_strict_provenance(
            payload, train, validation, paths
        )
    else:
        primary, secondary = _assert_debug_provenance(payload, train, validation, paths)
    current_streams = _target_state_stream_contract(train, primary, secondary)
    if canonical_json(_without_paths(current_streams)) != canonical_json(
        _without_paths(payload.get("target_state_stream_contract"))
    ):
        raise ValueError(
            "supplied paths violate checkpoint target_state_stream_contract"
        )

    model_config = _mapping(payload.get("model_config"), "model config")
    expected_geometry = (
        int(model_config.get("experts", -1)),
        int(model_config.get("layers", -1)),
        int(model_config.get("horizons", -1)),
        int(model_config.get("history", -1)),
        int(model_config.get("j_width", -1)),
        model_config.get("secondary_j_width"),
    )
    actual_geometry = (
        train.experts,
        train.layers,
        train.horizons,
        train.history,
        train.target_width,
        train.secondary_target_width,
    )
    if expected_geometry != actual_geometry:
        raise ValueError("checkpoint model and supplied input geometry differ")


def _scaled_metric_payload(
    scores: torch.Tensor,
    base: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    layers: int,
    experts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute the exact Full256 metric payload for one residual scale."""

    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("scaled checkpoint emitted non-finite router scores")
    target = batch["target_top8"].long()
    valid = batch["valid_future"].bool()
    recall8 = slot_recall_at_k(scores, target, 8)
    recall16 = slot_recall_at_k(scores, target, min(16, experts))
    base8 = slot_recall_at_k(base, target, 8)
    exact = recall8 == 1.0
    teacher = batch["teacher_router_scores"].float()
    kl = F.kl_div(
        torch.log_softmax(scores, dim=-1),
        torch.softmax(teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    layer_metrics = torch.stack((recall8, recall16, base8, exact.float(), kl), dim=-1)
    row_metrics = torch.stack(
        (
            recall8.mean(dim=2),
            recall16.mean(dim=2),
            base8.mean(dim=2),
            exact.float().mean(dim=2),
            kl.mean(dim=2),
        ),
        dim=-1,
    ).unsqueeze(2)
    packed = torch.cat((layer_metrics, row_metrics), dim=2)
    packed_valid = (
        valid[:, :, None, None].to(packed.dtype).expand(-1, -1, layers + 1, 1)
    )
    host = torch.cat((packed, packed_valid), dim=-1).cpu().numpy()
    valid_host = host[:, :, layers, -1].astype(bool)
    local_indices, horizons = np.nonzero(valid_host)
    if not len(local_indices):
        return None
    return host, local_indices, horizons


def _finalize_scaled_metrics(
    data: _RowLimitView,
    chunks: Mapping[str, list[np.ndarray]],
) -> dict[str, Any]:
    """Summarize cached per-event arrays with the production metric contract."""

    if chunks["rows"]:
        event_rows = np.concatenate(chunks["rows"], axis=0)
        event_layers = np.concatenate(chunks["layers"], axis=0)
        event_horizons = np.concatenate(chunks["horizons"], axis=0)
        event_requests = np.concatenate(chunks["requests"], axis=0)
        event_domains = np.concatenate(chunks["domains"], axis=0)
    else:
        event_rows = np.empty((0, len(_METRIC_NAMES)), dtype=np.float32)
        event_layers = np.empty((0, data.layers, len(_METRIC_NAMES)), dtype=np.float32)
        event_horizons = np.empty(0, dtype=np.int64)
        event_requests = np.empty(0, dtype=np.int64)
        event_domains = np.empty(0, dtype=str)
    totals, requests, layer_totals, domains = _aggregate_metric_events(
        event_rows,
        event_layers,
        event_horizons,
        event_requests,
        event_domains,
        horizons=data.horizons,
        layers=data.layers,
    )

    def summarize(values: Mapping[str, float]) -> dict[str, float | int]:
        count = max(1.0, values["rows"])
        return {
            "rows": int(values["rows"]),
            "recall_at_8": values["recall_at_8"] / count,
            "recall_at_16": values["recall_at_16"] / count,
            "base_recall_at_8": values["base_recall_at_8"] / count,
            "gain_at_8": (values["recall_at_8"] - values["base_recall_at_8"]) / count,
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
            row["request_macro_recall_at_8"] - row["request_macro_base_recall_at_8"]
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
            for (layer, horizon), values in sorted(layer_totals.items())
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


def _evaluate_residual_scales_once(
    model: torch.nn.Module,
    data: _RowLimitView,
    scales: Sequence[float],
    *,
    batch_size: int,
    device: str,
    autocast: bool,
) -> dict[float, dict[str, Any]]:
    """Evaluate every scale using one checkpoint forward per input batch."""

    checked = _validated_residual_scales(scales)
    chunks = {
        scale: {
            "rows": [],
            "layers": [],
            "horizons": [],
            "requests": [],
            "domains": [],
        }
        for scale in checked
    }
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
            base = output.base_router_scores.float()
            delta = output.delta.float()
            request_ids, domain_names, _within = data.metadata(rows)
            request_ids = np.asarray(request_ids, dtype=np.int64)
            domain_names = np.asarray(domain_names, dtype=str)
            for scale in checked:
                # Preserve the two scientific anchors exactly: alpha=0 is the
                # frozen HARP baseline and alpha=1 is the saved checkpoint.
                if scale == 0.0:
                    scores = base
                elif scale == 1.0:
                    scores = output.future_router_scores.float()
                else:
                    scores = base + delta * scale
                payload = _scaled_metric_payload(
                    scores,
                    base,
                    batch,
                    layers=data.layers,
                    experts=data.experts,
                )
                if payload is None:
                    continue
                host, local_indices, horizons = payload
                current = chunks[scale]
                current["rows"].append(
                    host[
                        local_indices,
                        horizons,
                        data.layers,
                        : len(_METRIC_NAMES),
                    ]
                )
                current["layers"].append(
                    host[
                        local_indices,
                        horizons,
                        : data.layers,
                        : len(_METRIC_NAMES),
                    ]
                )
                current["horizons"].append(horizons.astype(np.int64, copy=False) + 1)
                current["requests"].append(request_ids[local_indices])
                current["domains"].append(domain_names[local_indices])
    return {scale: _finalize_scaled_metrics(data, chunks[scale]) for scale in checked}


def _paired_gain_for_horizons(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    horizons: Sequence[int],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Pair candidate/base request macros, then cluster-bootstrap requests."""

    requested = frozenset(int(horizon) for horizon in horizons)
    if not requested:
        raise ValueError("paired gain requires at least one horizon")

    def recall_by_key(metrics: Mapping[str, Any]) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for row in metrics["request_metrics"]:
            horizon = int(row["horizon"])
            if horizon not in requested:
                continue
            key = (int(row["request_id"]), horizon)
            if key in result:
                raise ValueError("duplicate request/horizon metric in scale diagnostic")
            value = float(row["recall_at_8"])
            if not np.isfinite(value):
                raise ValueError("non-finite request recall in scale diagnostic")
            result[key] = value
        return result

    candidate_values = recall_by_key(candidate)
    baseline_values = recall_by_key(baseline)
    if candidate_values.keys() != baseline_values.keys():
        raise ValueError("scale and alpha=0 request/horizon pairs differ")
    if not candidate_values:
        raise ValueError("no paired request/horizon values for scale diagnostic")
    by_request: dict[int, list[float]] = {}
    for (request_id, horizon), value in sorted(candidate_values.items()):
        by_request.setdefault(request_id, []).append(
            value - baseline_values[(request_id, horizon)]
        )
    paired_request_gains = [
        float(np.mean(values, dtype=np.float64))
        for _request_id, values in sorted(by_request.items())
    ]
    result = _paired_request_bootstrap(
        paired_request_gains,
        replicates=replicates,
        seed=seed,
    )
    result["request_horizon_pairs"] = len(candidate_values)
    return result


def _residual_scale_diagnostic(
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
    bootstrap_replicates, bootstrap_seed = _validate_bootstrap_configuration(
        bootstrap_replicates, bootstrap_seed
    )
    if 0.0 not in metrics_by_scale:
        raise ValueError("residual scale diagnostic requires an alpha=0 reference")
    baseline = metrics_by_scale[0.0]
    summaries = []
    for scale in checked:
        metrics = metrics_by_scale[scale]
        paired_h1_h4 = _paired_gain_for_horizons(
            metrics,
            baseline,
            (1, 2, 3, 4),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        paired_horizons = []
        for horizon in range(1, 9):
            horizon_gain = _paired_gain_for_horizons(
                metrics,
                baseline,
                (horizon,),
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            paired_horizons.append({"horizon": horizon, **horizon_gain})
        summaries.append(
            {
                "residual_scale": scale,
                "mean_h1_h4_recall_at_8": float(metrics["mean_h1_h4_recall_at_8"]),
                "mean_h1_h8_recall_at_8": float(metrics["mean_h1_h8_recall_at_8"]),
                "h2_request_macro_recall_at_8": float(
                    metrics["h2_request_macro_recall_at_8"]
                ),
                "paired_request_h1_h4_gain_vs_alpha0": paired_h1_h4,
                "paired_request_horizon_gains_vs_alpha0": paired_horizons,
                "horizon_metrics": list(metrics["horizon_metrics"]),
            }
        )
    best_index = max(
        range(len(summaries)),
        key=lambda index: (
            summaries[index]["mean_h1_h4_recall_at_8"],
            summaries[index]["h2_request_macro_recall_at_8"],
        ),
    )
    best = summaries[best_index]
    return {
        "schema": FULL_ROUTER_RESIDUAL_SCALE_DIAGNOSTIC_SCHEMA,
        "split": split,
        "designation": designation,
        "uses_same_split_labels_for_selection": True,
        "uses_validation_labels_for_selection": split == "validation",
        "eligible_for_model_selection": eligible_for_model_selection,
        "creates_trained_checkpoint": False,
        "paired_reference_residual_scale": 0.0,
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
        "best_h2_request_macro_recall_at_8": best["h2_request_macro_recall_at_8"],
        "scale_metrics": summaries,
        "sealed_test_accessed": False,
    }


def _write_residual_scale_diagnostic(
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
        summary_rows.append(
            {
                "residual_scale": result["residual_scale"],
                "mean_h1_h4_recall_at_8": result["mean_h1_h4_recall_at_8"],
                "mean_h1_h8_recall_at_8": result["mean_h1_h8_recall_at_8"],
                "h2_request_macro_recall_at_8": result["h2_request_macro_recall_at_8"],
                "paired_h1_h4_gain_vs_alpha0": paired_summary["paired_gain_at_8"],
                "paired_h1_h4_ci95_low": paired_summary["ci95_low"],
                "paired_h1_h4_ci95_high": paired_summary["ci95_high"],
                "paired_h1_h4_requests": paired_summary["requests"],
                "paired_h1_h4_request_horizon_pairs": paired_summary[
                    "request_horizon_pairs"
                ],
                "bootstrap_replicates": paired_summary["bootstrap_replicates"],
                "bootstrap_seed": paired_summary["bootstrap_seed"],
            }
        )
        paired_by_horizon = {
            int(row["horizon"]): row
            for row in result["paired_request_horizon_gains_vs_alpha0"]
        }
        for row in result["horizon_metrics"]:
            paired = paired_by_horizon[int(row["horizon"])]
            horizon_rows.append(
                {
                    "residual_scale": result["residual_scale"],
                    **row,
                    "paired_gain_vs_alpha0": paired["paired_gain_at_8"],
                    "paired_gain_ci95_low": paired["ci95_low"],
                    "paired_gain_ci95_high": paired["ci95_high"],
                    "paired_requests": paired["requests"],
                    "bootstrap_replicates": paired["bootstrap_replicates"],
                    "bootstrap_seed": paired["bootstrap_seed"],
                }
            )
    write_csv(output_dir / "residual_scale_summary.csv", summary_rows)
    write_csv(output_dir / "residual_scale_horizon_metrics.csv", horizon_rows)


def _write_metrics(output_dir: Path, split: str, metrics: Mapping[str, Any]) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(dict(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for suffix, key in (
        ("horizon", "horizon_metrics"),
        ("request", "request_metrics"),
        ("layer", "layer_metrics"),
        ("domain", "domain_metrics"),
    ):
        rows = list(metrics.get(key, []))
        if rows:
            write_csv(output_dir / f"{split}_{suffix}_metrics.csv", rows)


def _export_predicted_top8(
    model: torch.nn.Module,
    data: _RowLimitView,
    path: Path,
    *,
    batch_size: int,
    device: str,
    autocast: bool,
    residual_scale: float = 1.0,
) -> dict[str, Any]:
    """Export prediction-only rows; future teacher tensors are never saved."""

    residual_scale = _validated_residual_scales((residual_scale,))[0]
    row_chunks: list[np.ndarray] = []
    request_chunks: list[np.ndarray] = []
    within_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []
    top8_chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for rows in data.sequential_batches(batch_size):
            batch = data.batch(rows, device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=autocast and str(device).startswith("cuda"),
            ):
                # Make the causal boundary explicit in the prediction export:
                # future teacher scores/top-8 never reach the model call.
                output = model(
                    causal_router_forecaster_inputs(
                        batch,
                        secondary_target_enabled=data.secondary_target_enabled,
                    )
                )
            if residual_scale == 0.0:
                scores = output.base_router_scores
            elif residual_scale == 1.0:
                scores = output.future_router_scores
            else:
                scores = (
                    output.base_router_scores.float()
                    + output.delta.float() * residual_scale
                )
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("checkpoint emitted non-finite router scores")
            predicted = torch.argsort(scores, dim=-1, descending=True, stable=True)[
                ..., :8
            ]
            packed = (
                torch.cat(
                    (
                        predicted.reshape(predicted.shape[0], -1).to(torch.int32),
                        batch["valid_future"].to(torch.int32),
                    ),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            predicted_values = packed[:, : predicted[0].numel()].reshape(
                len(rows), data.horizons, data.layers, 8
            )
            valid_values = packed[:, predicted[0].numel() :].astype(bool)
            request_ids, _domains, within = data.metadata(rows)
            row_chunks.append(np.asarray(rows, dtype=np.int64))
            request_chunks.append(np.asarray(request_ids, dtype=np.int64))
            within_chunks.append(np.asarray(within, dtype=np.int64))
            valid_chunks.append(valid_values)
            top8_chunks.append(predicted_values.astype(np.uint16))
    with Path(path).open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray("harp8_full_router_predicted_top8_rows_v1"),
            split=np.asarray(data.split),
            pool_row_index=np.concatenate(row_chunks),
            request_id=np.concatenate(request_chunks),
            within_request=np.concatenate(within_chunks),
            valid_future=np.concatenate(valid_chunks),
            predicted_top8=np.concatenate(top8_chunks),
            residual_scale=np.asarray(residual_scale, dtype=np.float64),
            contains_teacher_router_scores=np.asarray(False),
            contains_target_top8=np.asarray(False),
        )
    return {
        **_file_record(path),
        "rows": data.rows,
        "shape": [data.rows, data.horizons, data.layers, 8],
        "dtype": "uint16",
        "residual_scale": residual_scale,
        "contains_teacher_router_scores": False,
        "contains_target_top8": False,
    }


def evaluate_saved_full_router_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    *,
    split: str,
    train_pool: Path,
    validation_pool: Path,
    capture_dir: Path,
    mtp_dir: Path,
    target_features: Path,
    target_feature_rms: Path | None = None,
    secondary_target_features: Path | None = None,
    secondary_target_feature_rms: Path | None = None,
    precision_audit: Path | None = None,
    rows_per_request: int | None = None,
    maximum_rows: int | None = None,
    batch_size: int | None = None,
    device: str = "cuda:0",
    export_row_top8: bool = False,
    residual_scale_grid: Sequence[float] | None = None,
    export_residual_scale: float | None = None,
    bootstrap_replicates: int = DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_SEED,
    strict_lineage: bool = True,
) -> dict[str, Any]:
    """Evaluate one exact checkpoint on an explicitly named dev split."""

    if split not in _ALLOWED_SPLITS:
        raise PermissionError("Full256 evaluation permits only train/validation")
    checked_scales = (
        None
        if residual_scale_grid is None
        else _validated_residual_scales(residual_scale_grid)
    )
    bootstrap_replicates, bootstrap_seed = _validate_bootstrap_configuration(
        bootstrap_replicates, bootstrap_seed
    )
    if export_residual_scale is not None:
        export_residual_scale = _validated_residual_scales((export_residual_scale,))[0]
        if not export_row_top8:
            raise ValueError("export_residual_scale requires --export-row-top8")
    if checked_scales is None:
        if export_residual_scale not in (None, 1.0):
            raise ValueError("a non-unit export residual scale requires a scale grid")
        selected_export_scale = 1.0
    else:
        if export_row_top8 and export_residual_scale is None:
            raise ValueError(
                "scale-grid prediction export requires an explicit selected scale"
            )
        if (
            export_residual_scale is not None
            and export_residual_scale not in checked_scales
        ):
            raise ValueError("export residual scale must be present in the scale grid")
        selected_export_scale = export_residual_scale
    if maximum_rows is not None and maximum_rows <= 0:
        raise ValueError("maximum_rows must be positive")
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse evaluation directory {output_dir}")
    payload = _load_payload(checkpoint)
    paths = FullRouterEvaluationPaths(
        train_pool=Path(train_pool),
        validation_pool=Path(validation_pool),
        capture_dir=Path(capture_dir),
        mtp_dir=Path(mtp_dir),
        target_features=Path(target_features),
        target_feature_rms=(Path(target_feature_rms) if target_feature_rms else None),
        secondary_target_features=(
            Path(secondary_target_features) if secondary_target_features else None
        ),
        secondary_target_feature_rms=(
            Path(secondary_target_feature_rms) if secondary_target_feature_rms else None
        ),
        precision_audit=(Path(precision_audit) if precision_audit else None),
    )
    # Split manifests are rejected before FullRouterForecastData opens arrays.
    _read_pool_manifest(paths.train_pool, "train")
    _read_pool_manifest(paths.validation_pool, "validation")
    baseline = _mapping(
        payload.get("baseline_expansion_contract"),
        "baseline expansion contract",
    )
    margin = float(baseline.get("base_floor_margin", float("nan")))
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("checkpoint baseline expansion margin is invalid")
    model_config = _mapping(payload.get("model_config"), "model config")
    history = int(model_config.get("history", 0))
    if history <= 0:
        raise ValueError("checkpoint history geometry is invalid")
    routed_positions = (
        _infer_rows_per_request(paths.capture_dir)
        if rows_per_request is None
        else int(rows_per_request)
    )
    common = {
        "capture_dir": paths.capture_dir,
        "mtp_dir": paths.mtp_dir,
        "target_features": paths.target_features,
        "target_feature_rms": paths.target_feature_rms,
        "secondary_target_features": paths.secondary_target_features,
        "secondary_target_feature_rms": paths.secondary_target_feature_rms,
        "rows_per_request": routed_positions,
        "history": history,
        "base_floor_margin": margin,
    }
    train = FullRouterForecastData(paths.train_pool, expected_split="train", **common)
    validation = FullRouterForecastData(
        paths.validation_pool, expected_split="validation", **common
    )
    _assert_checkpoint_contracts(
        payload,
        train,
        validation,
        paths,
        strict_lineage=strict_lineage,
    )
    selected = train if split == "train" else validation
    saved_training = _mapping(payload.get("training_config"), "training config")
    saved_limit = saved_training.get(
        "max_train_rows" if split == "train" else "max_validation_rows"
    )
    effective_limit = maximum_rows
    if effective_limit is None and saved_limit is not None:
        effective_limit = int(saved_limit)
    view = _RowLimitView(selected, effective_limit)
    selected_batch = int(
        batch_size
        if batch_size is not None
        else saved_training.get("evaluation_batch_size", 1)
    )
    if selected_batch <= 0:
        raise ValueError("evaluation batch size must be positive")
    cuda = str(device).startswith("cuda")
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    model = load_jspace_router_forecaster_checkpoint(checkpoint, device=device)
    diagnostic = None
    if checked_scales is None:
        metrics = evaluate_full_router_forecaster(
            model,
            view,  # type: ignore[arg-type]
            batch_size=selected_batch,
            device=device,
            autocast=cuda,
        )
    else:
        # Alpha=1 remains the ordinary exact-checkpoint metrics.json even when
        # it was not requested in the diagnostic grid. All scales share the
        # same model forward for each batch.
        evaluation_scales = checked_scales
        for anchor in (0.0, 1.0):
            if anchor not in evaluation_scales:
                evaluation_scales = (*evaluation_scales, anchor)
        metrics_by_scale = _evaluate_residual_scales_once(
            model,
            view,
            evaluation_scales,
            batch_size=selected_batch,
            device=device,
            autocast=cuda,
        )
        metrics = metrics_by_scale[1.0]
        diagnostic = _residual_scale_diagnostic(
            checked_scales,
            metrics_by_scale,
            split=split,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )

    output_dir.mkdir(parents=True)
    _write_metrics(output_dir, split, metrics)
    if diagnostic is not None:
        _write_residual_scale_diagnostic(output_dir, diagnostic)
    prediction_record = None
    if export_row_top8:
        if selected_export_scale is None:
            raise AssertionError("selected export scale was not resolved")
        prediction_record = _export_predicted_top8(
            model,
            view,
            output_dir / "predicted_top8_rows.npz",
            batch_size=selected_batch,
            device=device,
            autocast=cuda,
            residual_scale=selected_export_scale,
        )
    manifest = {
        "schema": FULL_ROUTER_CHECKPOINT_EVALUATION_SCHEMA,
        "checkpoint": _file_record(checkpoint),
        "checkpoint_schema": payload["schema"],
        "completed_epoch": int(payload["completed_epoch"]),
        "global_step": int(payload.get("global_step", 0)),
        "split": split,
        "rows": view.rows,
        "full_split_rows": selected.rows,
        "maximum_rows": effective_limit,
        "evaluation_batch_size": selected_batch,
        "strict_lineage_verified": strict_lineage,
        "input_provenance_verified": True,
        "baseline_expansion_contract_verified": True,
        "target_state_stream_contract_verified": True,
        "sealed_test_accessed": False,
        "optimizer_constructed": False,
        "trainer_invoked": False,
        "exact_checkpoint_residual_scale": 1.0,
        "residual_scale_diagnostic": (
            None
            if diagnostic is None
            else {
                "designation": diagnostic["designation"],
                "uses_same_split_labels_for_selection": diagnostic[
                    "uses_same_split_labels_for_selection"
                ],
                "eligible_for_model_selection": diagnostic[
                    "eligible_for_model_selection"
                ],
                "requested_scales": diagnostic["requested_scales"],
                "best_residual_scale": diagnostic["best_residual_scale"],
                "paired_reference_residual_scale": 0.0,
                "bootstrap": diagnostic["bootstrap"],
                "creates_trained_checkpoint": False,
                "path": "residual_scale_diagnostic.json",
            }
        ),
        "row_top8_export": prediction_record,
        "outputs": {
            path.name: _file_record(path) for path in sorted(output_dir.iterdir())
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=tuple(sorted(_ALLOWED_SPLITS)), required=True
    )
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mtp-dir", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--target-feature-rms", type=Path)
    parser.add_argument("--secondary-target-features", type=Path)
    parser.add_argument("--secondary-target-feature-rms", type=Path)
    parser.add_argument("--precision-audit", type=Path)
    parser.add_argument("--rows-per-request", type=int)
    parser.add_argument("--maximum-rows", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--export-row-top8", action="store_true")
    parser.add_argument(
        "--residual-scale-grid",
        type=parse_residual_scale_grid,
        help=(
            "explicit comma-separated finite nonnegative alpha values for the "
            "split-labelled scores=base+alpha*delta diagnostic"
        ),
    )
    parser.add_argument(
        "--export-residual-scale",
        type=parse_residual_scale,
        help=(
            "explicit scale to use with --export-row-top8 when a scale grid "
            "is enabled; it must be present in that grid"
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_REPLICATES,
        help="paired request bootstrap replicates (default: 5000)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_SEED,
        help="paired request bootstrap seed (default: 20260807)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = evaluate_saved_full_router_checkpoint(
        args.checkpoint,
        args.output_dir,
        split=args.split,
        train_pool=args.train_pool,
        validation_pool=args.validation_pool,
        capture_dir=args.capture_dir,
        mtp_dir=args.mtp_dir,
        target_features=args.target_features,
        target_feature_rms=args.target_feature_rms,
        secondary_target_features=args.secondary_target_features,
        secondary_target_feature_rms=args.secondary_target_feature_rms,
        precision_audit=args.precision_audit,
        rows_per_request=args.rows_per_request,
        maximum_rows=args.maximum_rows,
        batch_size=args.batch_size,
        device=args.device,
        export_row_top8=args.export_row_top8,
        residual_scale_grid=args.residual_scale_grid,
        export_residual_scale=args.export_residual_scale,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        strict_lineage=True,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FULL_ROUTER_CHECKPOINT_EVALUATION_SCHEMA",
    "FULL_ROUTER_RESIDUAL_SCALE_DIAGNOSTIC_SCHEMA",
    "DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_REPLICATES",
    "DEFAULT_RESIDUAL_SCALE_BOOTSTRAP_SEED",
    "FullRouterEvaluationPaths",
    "build_parser",
    "evaluate_saved_full_router_checkpoint",
    "main",
    "parse_residual_scale",
    "parse_residual_scale_grid",
]
