"""Validation-only trainer for the J-space full-router H1--H8 forecaster."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .jspace_data import (
    move_tensor_batch,
    ordered_prefetched_batches,
    slice_tensor_batch,
)
from .jspace_router_data import (
    assert_disjoint_level2_request_sets,
    FullRouterForecastData,
    TARGET_STATE_STREAM_CONTRACT_SCHEMA,
)
from .jspace_router_forecaster import (
    FullRouterLossConfig,
    JSpaceFullRouterForecaster,
    JSpaceRouterForecasterConfig,
    full_router_forecaster_loss,
)
from .jspace_router_metrics import evaluate_full_router_forecaster
from .jspace_features import FEATURE_SCHEMA
from .prepare_jspace_features import PREPARATION_SCHEMA
from .train import canonical_json, sha256_file, write_csv
from .train_jspace_reranker import (
    DECISION_PROFILE,
    _assert_matching_data,
    _atomic_torch_save,
    _input_provenance,
    _mapping,
    _optimizer_groups,
    _restore_rng,
    _rng_state,
    _scheduler,
    _validate_scientific_lineage,
    _verify_file_record,
)


JSPACE_ROUTER_TRAINING_SCHEMA = "harp8_jspace_full_router_training_v2"
JSPACE_ROUTER_TRAINING_MANIFEST_SCHEMA = (
    "harp8_jspace_full_router_training_manifest_v2"
)
JSPACE_ROUTER_ARCHITECTURE_NAME = "J-HARP-Full256-v1"
BASELINE_EXPANSION_SCHEMA = "harp8_full_router_baseline_expansion_v1"
OUTPUT_HEAD_TRAINING_CONTRACT_SCHEMA = (
    "harp8_full_router_output_head_training_contract_v1"
)


@dataclass(frozen=True)
class FullRouterTrainingConfig:
    """Accuracy-first defaults that fit a 24 GiB RTX 3090 with BF16."""

    epochs: int = 20
    minimum_epochs: int = 5
    patience: int = 4
    microbatch_size: int = 4
    evaluation_batch_size: int = 4
    gradient_accumulation: int = 8
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    warmup_fraction: float = 0.03
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    gradient_clip: float = 1.0
    seed: int = 42
    max_train_rows: int | None = None
    max_validation_rows: int | None = None
    diagnostic_train_rows: int | None = None
    base_floor_margin: float = 1.0
    ordered_group_prefetch: bool = True
    freeze_output_bias: bool = False
    decision_profile: str = DECISION_PROFILE
    mtp_timing_causal: bool = False
    acceptance_labels_used: bool = False

    def validate(self) -> None:
        for name in (
            "epochs",
            "minimum_epochs",
            "patience",
            "microbatch_size",
            "evaluation_batch_size",
            "gradient_accumulation",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs cannot exceed epochs")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must lie in [0,1)")
        if not 0 < self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("minimum learning rate must lie in (0, learning rate]")
        if self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("weight decay and gradient clip are invalid")
        for name in (
            "max_train_rows",
            "max_validation_rows",
            "diagnostic_train_rows",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when supplied")
        if not math.isfinite(self.base_floor_margin) or self.base_floor_margin <= 0:
            raise ValueError(
                "base_floor_margin must be finite and positive"
            )
        if self.decision_profile != DECISION_PROFILE:
            raise ValueError("full-router v1 requires the audited token-end profile")
        if self.mtp_timing_causal:
            raise ValueError("the current MTP capture is not timing-causal")
        if self.acceptance_labels_used:
            raise ValueError("acceptance labels are forbidden as model inputs")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Disabled is the historical behavior and is omitted so old
        # checkpoints remain exactly resumable.
        if result["freeze_output_bias"] is False:
            result.pop("freeze_output_bias")
        return result


def _output_head_training_contract(
    model_config: JSpaceRouterForecasterConfig,
    training_config: FullRouterTrainingConfig,
) -> dict[str, Any] | None:
    if not training_config.freeze_output_bias:
        return None
    return {
        "schema": OUTPUT_HEAD_TRAINING_CONTRACT_SCHEMA,
        "parameter": "output_head.bias",
        "trainable": False,
        "frozen_before_optimizer_creation": True,
        "state_dict_parameter_retained": True,
        "initial_value": "zeros",
        "frozen_parameter_count": int(
            model_config.horizons * model_config.layers * model_config.experts
        ),
    }


class _PrefixView:
    def __init__(self, base: FullRouterForecastData, maximum: int | None) -> None:
        self.base = base
        self.rows = base.rows if maximum is None else min(base.rows, int(maximum))
        if self.rows <= 0:
            raise ValueError("training/evaluation view cannot be empty")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def batch(
        self,
        rows: np.ndarray,
        device: str | torch.device,
        *,
        active_horizons: int | None = None,
        include_context: bool = True,
        compact: bool = True,
    ):
        values = np.asarray(rows, dtype=np.int64)
        if values.ndim != 1 or (values < 0).any() or (values >= self.rows).any():
            raise IndexError("prefix-view rows are out of range")
        if active_horizons not in (None, self.base.horizons):
            raise ValueError("full-router prefetch cannot truncate horizons")
        if not include_context or not compact:
            raise ValueError(
                "full-router prefetch requires context and compact model batches"
            )
        return self.base.batch(values, device)

    def sequential_batches(self, batch_size: int):
        for start in range(0, self.rows, batch_size):
            yield np.arange(start, min(self.rows, start + batch_size), dtype=np.int64)


def derive_router_forecaster_config(
    data: FullRouterForecastData,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> JSpaceRouterForecasterConfig:
    values: dict[str, Any] = {
        "experts": data.experts,
        "layers": data.layers,
        "horizons": data.horizons,
        "history": data.history,
        "j_width": data.target_width,
        "secondary_j_width": data.secondary_target_width,
        "generator_context_width": data.generator_context_width,
        "mtp_nodes": data.mtp_depths,
        "mtp_state_channels": 1,
        "mtp_state_width": data.mtp_width,
    }
    if overrides:
        values.update(dict(overrides))
    config = JSpaceRouterForecasterConfig(**values)
    config.validate()
    return config


def _autocast(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _baseline_expansion_contract(
    data: FullRouterForecastData,
) -> dict[str, Any]:
    """Return the complete inference-time cK-to-full-router expansion rule."""

    return {
        "schema": BASELINE_EXPANSION_SCHEMA,
        "experts": int(data.experts),
        "candidate_count": int(data.pool.manifest["candidate_count"]),
        "native_k": int(data.pool.manifest["native_k"]),
        "absent_score_rule": "per_row_candidate_min_minus_margin",
        "base_floor_margin": float(data.base_floor_margin),
        "topk_tie_rule": "score_descending_then_expert_id_ascending_stable",
        "candidate_id_rule": "unique_layer_specific_ids",
    }



def _simple_provenance(
    train: FullRouterForecastData,
    validation: FullRouterForecastData,
    target_features: Path,
    target_feature_rms: Path | None,
    secondary_target_features: Path | None = None,
    secondary_target_feature_rms: Path | None = None,
) -> dict[str, Any]:
    def file_record(path: Path) -> dict[str, Any]:
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    result = {
        "strict": False,
        "reason": "explicit_debug_or_synthetic_run",
        "train_pool_manifest": file_record(train.pool.root / "manifest.json"),
        "validation_pool_manifest": file_record(
            validation.pool.root / "manifest.json"
        ),
        "requests": file_record(train.aligned.capture_dir / "requests.jsonl"),
        "target_features": {
            "path": str(target_features),
            "shape": list(train.aligned.target_features.shape),
            "dtype": str(train.aligned.target_features.dtype),
        },
        "target_feature_rms": (
            {"path": str(target_feature_rms)} if target_feature_rms else None
        ),
    }
    # Preserve byte-for-byte single-stream debug provenance for old resume
    # checkpoints; dual-only keys exist only under the explicit contract.
    if secondary_target_features is not None:
        result["secondary_target_features"] = file_record(
            Path(secondary_target_features)
        )
        result["secondary_target_feature_rms"] = (
            file_record(Path(secondary_target_feature_rms))
            if secondary_target_feature_rms is not None
            else None
        )
    return result


def _feature_representation(path: Path) -> str:
    manifest_path = Path(path).parent / "manifest.json"
    if not manifest_path.is_file():
        return "debug_unspecified"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(manifest.get("representation", "debug_unspecified"))


def _validate_secondary_feature_lineage(
    data: FullRouterForecastData,
    *,
    primary_semantic: Mapping[str, Any],
    target_features: Path,
    target_feature_rms: Path | None,
) -> dict[str, Any]:
    """Validate the second feature store without rehashing shared captures.

    The primary strict validator already authenticates the common capture,
    requests, MTP tensors, split, and preparation manifest. The secondary
    stream must belong to that exact preparation manifest; only its distinct
    feature/PCA artifacts need additional hashing.
    """

    feature_path = Path(target_features)
    if feature_path.resolve() != data.secondary_target_features_path.resolve():
        raise ValueError(
            "secondary target-features argument differs from the opened stream"
        )
    supplied_rms = Path(target_feature_rms).resolve() if target_feature_rms else None
    opened_rms = (
        data.secondary_target_feature_rms_path.resolve()
        if data.secondary_target_feature_rms_path is not None
        else None
    )
    if supplied_rms != opened_rms:
        raise ValueError(
            "secondary target-feature RMS differs from the opened stream"
        )
    if feature_path.name != "features_normalized.npy":
        raise ValueError("scientific secondary features require the canonical filename")

    store = feature_path.parent
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != FEATURE_SCHEMA:
        raise ValueError("secondary target feature store has an incompatible schema")
    outputs = _mapping(manifest.get("outputs"), "secondary feature outputs")
    verified = {
        "features/features_normalized.npy": _verify_file_record(
            feature_path,
            outputs.get("features_normalized.npy"),
            "secondary target feature array",
        )
    }
    if target_feature_rms is not None:
        if Path(target_feature_rms).parent.resolve() != store.resolve() or Path(
            target_feature_rms
        ).name != "log_rms.npy":
            raise ValueError("secondary RMS must be the matching feature side channel")
        verified["features/log_rms.npy"] = _verify_file_record(
            Path(target_feature_rms),
            outputs.get("log_rms.npy"),
            "secondary target feature RMS",
        )
    if (
        int(manifest.get("rows", -1))
        != int(data.secondary_target_features.shape[0])
        or int(manifest.get("layers", -1)) != data.layers
        or int(manifest.get("feature_width", -1))
        != int(data.secondary_target_features.shape[-1])
    ):
        raise ValueError("secondary feature manifest geometry disagrees with data")

    preparation_path = store.parent.parent / "manifest.json"
    primary_preparation = _mapping(
        primary_semantic.get("feature_preparation_manifest"),
        "primary feature preparation provenance",
    )
    if (
        preparation_path.resolve()
        != Path(str(primary_preparation.get("path", ""))).resolve()
        or sha256_file(preparation_path)
        != str(primary_preparation.get("sha256", ""))
    ):
        raise ValueError(
            "primary and secondary streams require the same audited preparation manifest"
        )
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation.get("schema") != PREPARATION_SCHEMA:
        raise ValueError("secondary feature preparation schema is incompatible")
    representation = str(manifest.get("representation", ""))
    rank = str(int(manifest.get("feature_width", -1)))
    representations = _mapping(
        preparation.get("representations"), "prepared representations"
    )
    representation_record = _mapping(
        representations.get(representation),
        f"prepared representation {representation}",
    )
    ranks = _mapping(representation_record.get("ranks"), "prepared ranks")
    rank_record = _mapping(ranks.get(rank), f"prepared rank {rank}")
    prepared_store = _mapping(rank_record.get("feature_store"), "prepared store")
    if prepared_store.get("schema") != FEATURE_SCHEMA:
        raise ValueError("preparation embeds an invalid secondary feature store")
    prepared_outputs = _mapping(
        prepared_store.get("outputs"), "prepared secondary outputs"
    )
    for filename, actual_value in outputs.items():
        actual = _mapping(actual_value, f"secondary output {filename}")
        expected = _mapping(
            prepared_outputs.get(filename), f"prepared secondary output {filename}"
        )
        if (
            actual.get("sha256") != expected.get("sha256")
            or int(actual.get("bytes", -1)) != int(expected.get("bytes", -2))
        ):
            raise ValueError("secondary feature store disagrees with preparation")
    feature_source = _mapping(manifest.get("source"), "secondary feature source")
    preparation_source = _mapping(
        preparation.get("source"), "feature preparation source"
    )
    prepared_residual = _mapping(
        preparation_source.get("residual_npy"), "prepared residual"
    )
    if feature_source.get("sha256") != prepared_residual.get("sha256"):
        raise ValueError("secondary feature source residual lineage disagrees")
    pca_record = _mapping(rank_record.get("pca_artifact"), "secondary PCA artifact")
    pca_path = store.parent / f"shared_pca_rank{rank}.npz"
    if sha256_file(pca_path) != str(pca_record.get("file_sha256", "")):
        raise ValueError("secondary PCA artifact hash disagrees with preparation")
    return {
        "strict": True,
        "feature_store_manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "feature_preparation_manifest": dict(primary_preparation),
        "representation": representation,
        "pca_fit_split": "train",
        "shared_capture_lineage_reused_from_primary": True,
        "precision_attribution": {
            "required": False,
            "attribution": "not_applicable",
        },
        "verified_inputs": verified,
    }


def _verified_stream_file(
    path: Path | None,
    semantic_lineage: Mapping[str, Any],
    key: str,
) -> dict[str, Any] | None:
    if path is None:
        return None
    verified = semantic_lineage.get("verified_inputs", {})
    record = verified.get(key) if isinstance(verified, Mapping) else None
    if (
        isinstance(record, Mapping)
        and Path(str(record.get("path", ""))).resolve() == Path(path).resolve()
    ):
        return dict(record)
    return {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "sha256": sha256_file(Path(path)),
    }


def _target_state_stream_contract(
    data: FullRouterForecastData,
    primary_semantic: Mapping[str, Any],
    secondary_semantic: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not data.secondary_target_enabled:
        return None
    if secondary_semantic is None:
        raise AssertionError("dual-stream data lacks secondary semantic lineage")
    streams = [
        {
            "slot": "primary",
            "batch_key": "j_states",
            "mask_key": "j_mask",
            "representation_role": str(
                primary_semantic.get(
                    "representation", _feature_representation(data.aligned.target_features_path)
                )
            ),
            "projected_width": int(data.target_width),
            "features": _verified_stream_file(
                data.aligned.target_features_path,
                primary_semantic,
                "features/features_normalized.npy",
            ),
            "rms": _verified_stream_file(
                data.aligned.target_feature_rms_path,
                primary_semantic,
                "features/log_rms.npy",
            ),
        },
        {
            "slot": "secondary",
            "batch_key": "secondary_target_states",
            "mask_key": "secondary_target_mask",
            "representation_role": str(
                secondary_semantic.get(
                    "representation",
                    _feature_representation(data.secondary_target_features_path),
                )
            ),
            "projected_width": int(data.secondary_target_width),
            "features": _verified_stream_file(
                data.secondary_target_features_path,
                secondary_semantic,
                "features/features_normalized.npy",
            ),
            "rms": _verified_stream_file(
                data.secondary_target_feature_rms_path,
                secondary_semantic,
                "features/log_rms.npy",
            ),
        },
    ]
    return {
        "schema": TARGET_STATE_STREAM_CONTRACT_SCHEMA,
        "enabled": True,
        "history": int(data.history),
        "alignment": "identical_capture_rows_lag_indices_and_history_masks",
        "streams": streams,
        "fusion": {
            "independent_rmsnorm_linear_projection": True,
            "input_concatenation_before_projection": False,
            "gate": "learned_per_cell_two_stream_softmax",
            "gate_inputs": "post_projection_streams",
        },
    }


def _checkpoint_payload(
    *,
    model: JSpaceFullRouterForecaster,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    rng: np.random.Generator | None,
    model_config: JSpaceRouterForecasterConfig,
    loss_config: FullRouterLossConfig,
    training_config: FullRouterTrainingConfig,
    input_provenance: Mapping[str, Any],
    baseline_expansion_contract: Mapping[str, Any],
    completed_epoch: int,
    global_step: int,
    best_selection: tuple[float, float],
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
    target_state_stream_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": JSPACE_ROUTER_TRAINING_SCHEMA,
        "architecture": JSPACE_ROUTER_ARCHITECTURE_NAME,
        "model_config": model_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "training_config": training_config.to_dict(),
        "input_provenance": dict(input_provenance),
        "baseline_expansion_contract": dict(baseline_expansion_contract),
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "completed_epoch": int(completed_epoch),
        "next_epoch": int(completed_epoch) + 1,
        "global_step": int(global_step),
        "best_selection": tuple(float(value) for value in best_selection),
        "best_epoch": int(best_epoch),
        "stale_epochs": int(stale_epochs),
        "history": list(history),
        "inference_contract": {
            "output_shape": "[B,8,40,256] for production geometry",
            "direct_horizons": list(range(1, 9)),
            "full_expert_namespace": True,
            "frozen_harp_residual_baseline": True,
            "future_labels_cross_model_boundary": False,
            "acceptance_labels_used": False,
            "decision_profile": DECISION_PROFILE,
            "mtp_timing_causal": False,
            "baseline_expansion_contract": dict(baseline_expansion_contract),
            "sealed_test_accessed": False,
        },
    }
    output_head_contract = _output_head_training_contract(
        model_config, training_config
    )
    if output_head_contract is not None:
        if model.output_head.bias.requires_grad:
            raise AssertionError("frozen output bias unexpectedly requires gradients")
        payload["output_head_training_contract"] = output_head_contract
    if target_state_stream_contract is not None:
        payload["target_state_stream_contract"] = dict(
            target_state_stream_contract
        )
        payload["inference_contract"]["target_state_stream_contract"] = dict(
            target_state_stream_contract
        )
    if optimizer is not None and scheduler is not None and rng is not None:
        payload.update(
            {
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "rng_state": _rng_state(rng),
            }
        )
    return payload


def load_jspace_router_forecaster_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> JSpaceFullRouterForecaster:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != JSPACE_ROUTER_TRAINING_SCHEMA:
        raise ValueError("checkpoint has an incompatible full-router schema")
    contract = payload.get("baseline_expansion_contract")
    if not isinstance(contract, Mapping) or contract.get("schema") != (
        BASELINE_EXPANSION_SCHEMA
    ):
        raise ValueError("checkpoint lacks a valid baseline-expansion contract")
    inference_contract = payload.get("inference_contract", {})
    if canonical_json(inference_contract.get("baseline_expansion_contract")) != (
        canonical_json(contract)
    ):
        raise ValueError("checkpoint inference expansion contract is inconsistent")
    config = JSpaceRouterForecasterConfig(**payload["model_config"])
    config.validate()
    stream_contract = payload.get("target_state_stream_contract")
    if config.secondary_j_width is not None:
        if (
            not isinstance(stream_contract, Mapping)
            or stream_contract.get("schema") != TARGET_STATE_STREAM_CONTRACT_SCHEMA
            or stream_contract.get("enabled") is not True
        ):
            raise ValueError("dual-stream checkpoint lacks a valid stream contract")
        inference_stream_contract = inference_contract.get(
            "target_state_stream_contract"
        )
        if canonical_json(inference_stream_contract) != canonical_json(stream_contract):
            raise ValueError("checkpoint inference stream contract is inconsistent")
        streams = stream_contract.get("streams")
        if (
            not isinstance(streams, list)
            or len(streams) != 2
            or not all(isinstance(record, Mapping) for record in streams)
        ):
            raise ValueError("dual-stream checkpoint must declare exactly two streams")
        widths = [int(record.get("projected_width", -1)) for record in streams]
        if widths != [config.j_width, config.secondary_j_width]:
            raise ValueError("dual-stream checkpoint widths disagree with model config")
    elif stream_contract is not None:
        raise ValueError("single-stream checkpoint cannot declare a dual-stream contract")
    model = JSpaceFullRouterForecaster(config)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


def _write_metrics(output_dir: Path, metrics: Mapping[str, Any]) -> None:
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in ("horizon_metrics", "request_metrics", "layer_metrics", "domain_metrics"):
        rows = list(metrics.get(name, []))
        if rows:
            write_csv(output_dir / f"validation_{name}.csv", rows)

def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write evolving epoch schemas without making epoch zero lose fields."""

    columns = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    normalized = [
        {column: row.get(column) for column in columns} for row in rows
    ]
    write_csv(path, normalized)


def _aggregate_deferred_training_metrics(
    metric_names: tuple[str, ...] | None,
    metric_rows: Sequence[torch.Tensor],
) -> dict[str, float]:
    """Transfer packed diagnostics once and preserve legacy summation order."""

    if not metric_rows:
        return {}
    if metric_names is None:
        raise AssertionError("deferred metric rows require their names")
    for row in metric_rows:
        if row.requires_grad:
            raise AssertionError("deferred training metrics must be detached")
        if row.ndim != 1 or row.shape[0] != len(metric_names):
            raise ValueError("deferred training metric geometry changed within epoch")
    host_rows = torch.stack(tuple(metric_rows)).detach().cpu().tolist()
    totals = {name: 0.0 for name in metric_names}
    for values in host_rows:
        for name, value in zip(metric_names, values, strict=True):
            totals[name] += float(value)
    return totals


def _validation_selection(
    metrics: Mapping[str, Any],
) -> tuple[float, float]:
    names = (
        "mean_h1_h4_recall_at_8",
        "h2_request_macro_recall_at_8",
        "mean_h1_h8_recall_at_8",
    )
    values = tuple(float(metrics[name]) for name in names)
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("validation selection metrics are non-finite")
    return values[0], values[1]



def _assert_matching(
    train: FullRouterForecastData,
    validation: FullRouterForecastData,
) -> None:
    _assert_matching_data(train.aligned, validation.aligned)
    if train.split != "train" or validation.split != "validation":
        raise ValueError("trainer requires explicit train and validation pools")
    assert_disjoint_level2_request_sets(train, validation)
    if train.generator_context_width != validation.generator_context_width:
        raise ValueError("generator-context widths differ")
    if train.secondary_target_enabled != validation.secondary_target_enabled:
        raise ValueError("secondary target-stream availability differs")
    if train.secondary_target_width != validation.secondary_target_width:
        raise ValueError("secondary target-stream widths differ")
    for label, left, right in (
        (
            "secondary target features",
            train.secondary_target_features_path,
            validation.secondary_target_features_path,
        ),
        (
            "secondary target RMS",
            train.secondary_target_feature_rms_path,
            validation.secondary_target_feature_rms_path,
        ),
    ):
        if left is None or right is None:
            if left is right:
                continue
            raise ValueError(f"train/validation {label} availability differs")
        if Path(left).resolve() != Path(right).resolve():
            raise ValueError(f"train/validation {label} sources differ")
    if canonical_json(_baseline_expansion_contract(train)) != canonical_json(
        _baseline_expansion_contract(validation)
    ):
        raise ValueError("train/validation baseline-expansion contracts differ")


def train_jspace_router_forecaster(
    train_data: FullRouterForecastData,
    validation_data: FullRouterForecastData,
    output_dir: Path,
    model_config: JSpaceRouterForecasterConfig,
    loss_config: FullRouterLossConfig,
    training_config: FullRouterTrainingConfig,
    *,
    target_features: Path,
    target_feature_rms: Path | None,
    secondary_target_features: Path | None = None,
    secondary_target_feature_rms: Path | None = None,
    device: str = "cuda:0",
    resume: Path | None = None,
    stop_after_epoch: int | None = None,
    strict_lineage: bool = True,
    precision_audit: Path | None = None,
    allow_provisional_j_without_probe: bool = False,
) -> dict[str, Any]:
    """Train with validation-only selection and fully resumable checkpoints."""

    model_config.validate()
    loss_config.validate(model_config.horizons, model_config.experts)
    training_config.validate()
    output_head_contract = _output_head_training_contract(
        model_config, training_config
    )
    _assert_matching(train_data, validation_data)
    baseline_contract = _baseline_expansion_contract(train_data)
    if training_config.base_floor_margin != float(
        baseline_contract["base_floor_margin"]
    ):
        raise ValueError(
            "training/data base_floor_margin values differ"
        )
    if (model_config.experts, model_config.layers, model_config.horizons) != (
        train_data.experts,
        train_data.layers,
        train_data.horizons,
    ):
        raise ValueError("model and data geometry disagree")
    if model_config.secondary_j_width != train_data.secondary_target_width:
        raise ValueError("model and secondary target-stream geometry disagree")
    supplied_secondary = (
        Path(secondary_target_features).resolve()
        if secondary_target_features is not None
        else None
    )
    opened_secondary = (
        train_data.secondary_target_features_path.resolve()
        if train_data.secondary_target_features_path is not None
        else None
    )
    if supplied_secondary != opened_secondary:
        raise ValueError(
            "secondary-target-features argument differs from the opened stream"
        )
    supplied_secondary_rms = (
        Path(secondary_target_feature_rms).resolve()
        if secondary_target_feature_rms is not None
        else None
    )
    opened_secondary_rms = (
        train_data.secondary_target_feature_rms_path.resolve()
        if train_data.secondary_target_feature_rms_path is not None
        else None
    )
    if supplied_secondary_rms != opened_secondary_rms:
        raise ValueError(
            "secondary-target-feature-rms argument differs from the opened stream"
        )

    primary_semantic: dict[str, Any]
    secondary_semantic: dict[str, Any] | None = None
    if strict_lineage:
        primary_role = _feature_representation(Path(target_features))
        secondary_role = (
            _feature_representation(Path(secondary_target_features))
            if secondary_target_features is not None
            else None
        )
        if allow_provisional_j_without_probe and "j_lens" not in {
            primary_role,
            secondary_role,
        }:
            raise ValueError(
                "provisional J precision override requires a J-Lens stream"
            )
        if train_data.secondary_target_enabled and (
            primary_role != "j_lens" or secondary_role != "raw_residual"
        ):
            raise ValueError(
                "scientific dual-stream runs require primary J-Lens and secondary raw residual"
            )
        primary_semantic = _validate_scientific_lineage(
            train_data.aligned,
            validation_data.aligned,
            target_features=Path(target_features),
            target_feature_rms=target_feature_rms,
            precision_audit=precision_audit,
            allow_provisional_j_without_probe=(
                allow_provisional_j_without_probe and primary_role == "j_lens"
            ),
        )
        if train_data.secondary_target_enabled:
            secondary_semantic = _validate_secondary_feature_lineage(
                train_data,
                primary_semantic=primary_semantic,
                target_features=Path(secondary_target_features),
                target_feature_rms=secondary_target_feature_rms,
            )
        primary_semantic.update(
            {
                "architecture": JSPACE_ROUTER_ARCHITECTURE_NAME,
                "full_router_supervision": True,
                "full_expert_namespace": True,
                "acceptance_labels_used": False,
                "decision_profile": DECISION_PROFILE,
                "mtp_timing_causal": False,
            }
        )
        provenance = _input_provenance(
            train_data.aligned,
            validation_data.aligned,
            target_features=Path(target_features),
            target_feature_rms=target_feature_rms,
            router_provenance={
                "mode": "direct_layer_horizon_residual_heads",
                "router_keys_required": False,
            },
            semantic_lineage=primary_semantic,
        )
        if secondary_semantic is not None:
            provenance["secondary_target_semantic_lineage"] = secondary_semantic
    else:
        if precision_audit is not None or allow_provisional_j_without_probe:
            raise ValueError("precision claims are unavailable in debug mode")
        primary_semantic = {
            "strict": False,
            "representation": _feature_representation(Path(target_features)),
        }
        secondary_semantic = (
            {
                "strict": False,
                "representation": _feature_representation(
                    Path(secondary_target_features)
                ),
            }
            if secondary_target_features is not None
            else None
        )
        provenance = _simple_provenance(
            train_data,
            validation_data,
            Path(target_features),
            target_feature_rms,
            secondary_target_features,
            secondary_target_feature_rms,
        )
    target_stream_contract = _target_state_stream_contract(
        train_data, primary_semantic, secondary_semantic
    )
    provenance["execution_contract"] = {
        "full_router_outputs": True,
        "direct_horizons": list(range(1, 9)),
        "causal_model_allowlist": True,
        "generator_context_required": True,
        "native_mtp_hidden_router_separated": True,
        "future_labels_cross_model_boundary": False,
        "baseline_expansion_contract": baseline_contract,
        "sealed_test_accessed": False,
    }
    if output_head_contract is not None:
        provenance["execution_contract"]["output_head_training_contract"] = (
            output_head_contract
        )
    if target_stream_contract is not None:
        provenance["execution_contract"]["target_state_stream_contract"] = (
            target_stream_contract
        )
    provenance["execution_contract"]["io_pipeline"] = {
        "schema": "harp8_ordered_group_prefetch_v1",
        "enabled": training_config.ordered_group_prefetch,
        "rows_per_materialization": (
            training_config.microbatch_size
            * training_config.gradient_accumulation
        ),
        "cpu_read_ahead_groups": 1,
        "pinned_nonblocking_h2d_on_cuda": True,
    }
    provenance["baseline_expansion_contract"] = baseline_contract

    output_dir = Path(output_dir)
    if resume is None:
        if output_dir.exists():
            raise FileExistsError(f"refusing to reuse output directory {output_dir}")
        output_dir.mkdir(parents=True)
    elif not output_dir.is_dir():
        raise FileNotFoundError("resume output directory does not exist")

    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    cuda = str(device).startswith("cuda")
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats()


    train_view = _PrefixView(train_data, training_config.max_train_rows)
    validation_view = _PrefixView(
        validation_data, training_config.max_validation_rows
    )
    diagnostic_train_view = (
        _PrefixView(
            train_data,
            min(training_config.diagnostic_train_rows, train_view.rows),
        )
        if training_config.diagnostic_train_rows is not None
        else None
    )
    model = JSpaceFullRouterForecaster(model_config).to(device)
    # Freeze before parameter grouping and optimizer construction. The
    # parameter remains in the state dict, so inference/checkpoint layout is
    # unchanged.
    if training_config.freeze_output_bias:
        model.output_head.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, training_config.weight_decay),
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
        eps=training_config.epsilon,
        fused=cuda,
    )
    updates_per_epoch = math.ceil(
        math.ceil(train_view.rows / training_config.microbatch_size)
        / training_config.gradient_accumulation
    )
    scheduler = _scheduler(
        optimizer,
        updates_per_epoch * training_config.epochs,
        training_config.warmup_fraction,
        training_config.minimum_learning_rate / training_config.learning_rate,
    )

    assertion_rows = np.arange(
        min(validation_view.rows, training_config.evaluation_batch_size),
        dtype=np.int64,
    )
    assertion_batch = validation_view.batch(assertion_rows, device)
    model.eval()
    with torch.inference_mode(), _autocast(device):
        epoch0 = model(assertion_batch)
    if not torch.equal(
        epoch0.future_router_scores.float(),
        assertion_batch["base_router_scores"].float(),
    ):
        raise AssertionError("epoch-zero full-router output differs from HARP baseline")
    if int(torch.count_nonzero(epoch0.delta)):
        raise AssertionError("epoch-zero full-router residual is nonzero")

    history: list[dict[str, Any]] = []
    completed_epoch = 0
    global_step = 0
    stale_epochs = 0
    best_selection = (-math.inf, -math.inf)
    best_epoch = 0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    if resume is None:
        epoch0_metrics = evaluate_full_router_forecaster(
            model,
            validation_view,  # type: ignore[arg-type]
            batch_size=training_config.evaluation_batch_size,
            device=device,
            autocast=cuda,
        )
        best_selection = _validation_selection(epoch0_metrics)
        history = [
            {
                "epoch": 0,
                "global_step": 0,
                "learning_rate": scheduler.get_last_lr()[0],
                "validation_mean_h1_h4_recall_at_8": best_selection[0],
                "validation_h2_recall_at_8": best_selection[1],
                "validation_mean_h1_h8_recall_at_8": epoch0_metrics[
                    "mean_h1_h8_recall_at_8"
                ],
                "epoch0_baseline_candidate": True,
            }
        ]
        epoch0_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng=rng,
            model_config=model_config,
            loss_config=loss_config,
            training_config=training_config,
            input_provenance=provenance,
            baseline_expansion_contract=baseline_contract,
            completed_epoch=0,
            global_step=0,
            best_selection=best_selection,
            best_epoch=0,
            stale_epochs=0,
            history=history,
            target_state_stream_contract=target_stream_contract,
        )
        _atomic_torch_save(epoch0_payload, best_path)
        _atomic_torch_save(epoch0_payload, last_path)
        _write_metrics(output_dir, epoch0_metrics)
        _write_history(output_dir / "training_history.csv", history)
        print(canonical_json({"event": "epoch_complete", **history[-1]}), flush=True)
    else:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state.get("schema") != JSPACE_ROUTER_TRAINING_SCHEMA:
            raise ValueError("resume checkpoint has an incompatible schema")
        expected = {
            "model_config": model_config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "training_config": training_config.to_dict(),
            "input_provenance": provenance,
            "baseline_expansion_contract": baseline_contract,
        }
        if target_stream_contract is not None:
            expected["target_state_stream_contract"] = target_stream_contract
        if output_head_contract is not None:
            expected["output_head_training_contract"] = output_head_contract
        for name, value in expected.items():
            if canonical_json(state.get(name)) != canonical_json(value):
                raise ValueError(f"resume checkpoint {name} differs from this run")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        _restore_rng(state["rng_state"], rng)
        completed_epoch = int(state["completed_epoch"])
        global_step = int(state["global_step"])
        stale_epochs = int(state["stale_epochs"])
        best_selection = tuple(float(value) for value in state["best_selection"])
        best_epoch = int(state["best_epoch"])
        history = list(state["history"])

    for epoch in range(completed_epoch + 1, training_config.epochs + 1):
        model.train()
        order = rng.permutation(train_view.rows)
        deferred_metric_names: tuple[str, ...] | None = None
        deferred_metric_rows: list[torch.Tensor] = []
        microbatches = 0
        group_rows = (
            training_config.microbatch_size
            * training_config.gradient_accumulation
        )
        row_groups = (
            order[start : start + group_rows]
            for start in range(0, train_view.rows, group_rows)
        )
        cuda_io = bool(
            training_config.ordered_group_prefetch
            and cuda
            and torch.cuda.is_available()
        )
        if training_config.ordered_group_prefetch:
            materialized = ordered_prefetched_batches(
                train_view,
                row_groups,
                active_horizons=train_view.horizons,
                include_context=True,
                compact=True,
                prefetch=True,
                pin_memory=cuda_io,
            )
        else:
            materialized = ((group, None) for group in row_groups)
        for group, host_batch in materialized:
            group_batch = (
                move_tensor_batch(host_batch, device, non_blocking=cuda_io)
                if host_batch is not None
                else None
            )
            micro_count = math.ceil(
                len(group) / training_config.microbatch_size
            )
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(
                0, len(group), training_config.microbatch_size
            ):
                micro_stop = min(
                    len(group), micro_start + training_config.microbatch_size
                )
                batch = (
                    train_view.batch(group[micro_start:micro_stop], device)
                    if group_batch is None
                    else slice_tensor_batch(group_batch, micro_start, micro_stop)
                )
                with _autocast(device):
                    output = model(batch)
                    loss = full_router_forecaster_loss(output, batch, loss_config)
                    finite_loss = torch.isfinite(loss.total.detach())
                    if loss.total.is_cuda:
                        torch._assert_async(
                            finite_loss, "training loss is non-finite"
                        )
                    elif not bool(finite_loss):
                        raise FloatingPointError("training loss is non-finite")
                (loss.total / micro_count).backward()
                microbatches += 1
                if deferred_metric_names is None:
                    deferred_metric_names = loss.metric_names
                elif deferred_metric_names != loss.metric_names:
                    raise RuntimeError("training metric schema changed within epoch")
                deferred_metric_rows.append(loss.metric_values)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training_config.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1

        running = _aggregate_deferred_training_metrics(
            deferred_metric_names, deferred_metric_rows
        )
        metrics = evaluate_full_router_forecaster(
            model,
            validation_view,  # type: ignore[arg-type]
            batch_size=training_config.evaluation_batch_size,
            device=device,
            autocast=cuda,
        )
        diagnostic_train_metrics = None
        if diagnostic_train_view is not None:
            diagnostic_train_metrics = evaluate_full_router_forecaster(
                model,
                diagnostic_train_view,  # type: ignore[arg-type]
                batch_size=training_config.evaluation_batch_size,
                device=device,
                autocast=cuda,
            )
            (output_dir / f"diagnostic_train_epoch_{epoch:02d}.json").write_text(
                json.dumps(
                    diagnostic_train_metrics, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
        selection = _validation_selection(metrics)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": scheduler.get_last_lr()[0],
            **{
                f"train_{name}": value / max(1, microbatches)
                for name, value in running.items()
            },
            "validation_mean_h1_h4_recall_at_8": selection[0],
            "validation_h2_recall_at_8": selection[1],
            "validation_mean_h1_h8_recall_at_8": metrics[
                "mean_h1_h8_recall_at_8"
            ],
        }
        if diagnostic_train_metrics is not None:
            diagnostic_selection = _validation_selection(
                diagnostic_train_metrics
            )
            record["diagnostic_train_mean_h1_h4_recall_at_8"] = (
                diagnostic_selection[0]
            )
            record["diagnostic_train_h2_recall_at_8"] = diagnostic_selection[1]
            record["diagnostic_train_mean_h1_h8_recall_at_8"] = (
                diagnostic_train_metrics["mean_h1_h8_recall_at_8"]
            )
            for horizon_row in diagnostic_train_metrics["horizon_metrics"]:
                horizon = int(horizon_row["horizon"])
                record[f"diagnostic_train_h{horizon}_recall_at_8"] = horizon_row[
                    "request_macro_recall_at_8"
                ]
        history.append(record)
        improved = selection > best_selection
        if improved:
            best_selection = selection
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng=rng,
            model_config=model_config,
            loss_config=loss_config,
            training_config=training_config,
            input_provenance=provenance,
            baseline_expansion_contract=baseline_contract,
            completed_epoch=epoch,
            global_step=global_step,
            best_selection=best_selection,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            history=history,
            target_state_stream_contract=target_stream_contract,
        )
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, best_path)
            _write_metrics(output_dir, metrics)
        _write_history(output_dir / "training_history.csv", history)
        completed_epoch = epoch
        print(canonical_json({"event": "epoch_complete", **record}), flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
        if (
            epoch >= training_config.minimum_epochs
            and stale_epochs >= training_config.patience
        ):
            break

    best_metrics_path = output_dir / "validation_metrics.json"
    if not best_metrics_path.is_file():
        raise FileNotFoundError("persisted best validation metrics are missing")
    final_metrics = json.loads(best_metrics_path.read_text(encoding="utf-8"))
    if not isinstance(final_metrics, Mapping):
        raise ValueError("persisted best validation metrics must be a mapping")
    if _validation_selection(final_metrics) != best_selection:
        raise RuntimeError(
            "persisted validation metrics disagree with best checkpoint selection"
        )
    cuda_peak_memory = (
        {
            "allocated": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved()),
        }
        if cuda
        else None
    )
    manifest = {
        "schema": JSPACE_ROUTER_TRAINING_MANIFEST_SCHEMA,
        "architecture": JSPACE_ROUTER_ARCHITECTURE_NAME,
        "model_config": model_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "training_config": training_config.to_dict(),
        "baseline_expansion_contract": baseline_contract,
        "target_state_stream_contract": target_stream_contract,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "best_epoch": best_epoch,
        "completed_epoch": completed_epoch,
        "best_selection": list(best_selection),
        "input_provenance": provenance,
        "epoch0_baseline_assertion": {
            "exact_score_equality": True,
            "zero_residual": True,
        },
        "validation_metrics": final_metrics,
        "validation_metrics_source": "persisted_best_epoch_evaluation",
        "cuda_peak_memory_bytes": cuda_peak_memory,
        "sealed_test_accessed": False,
        "artifacts": {
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "training_history": str(output_dir / "training_history.csv"),
            "validation_metrics": str(output_dir / "validation_metrics.json"),
        },
    }
    if output_head_contract is not None:
        manifest["output_head_training_contract"] = output_head_contract
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-experimental-full-router", action="store_true")
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mtp-dir", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--target-feature-rms", type=Path)
    parser.add_argument("--secondary-target-features", type=Path)
    parser.add_argument("--secondary-target-feature-rms", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision-audit", type=Path)
    parser.add_argument("--allow-provisional-j-without-probe", action="store_true")
    parser.add_argument("--no-strict-lineage", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows-per-request", type=int, default=34)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--base-floor-margin", type=float, default=1.0)

    parser.add_argument("--model-width", type=int, default=192)
    parser.add_argument("--attention-heads", type=int, default=6)
    parser.add_argument("--feedforward-width", type=int, default=768)
    parser.add_argument("--layer-blocks", type=int, default=2)
    parser.add_argument("--mtp-blocks", type=int, default=1)
    parser.add_argument("--fusion-blocks", type=int, default=2)
    parser.add_argument("--output-rank", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--mtp-diagonal-from-local",
        action="store_true",
        help=(
            "feed horizon-matched diagonal sources from node-local MTP "
            "encodings before cross-depth contextualization"
        ),
    )
    parser.add_argument(
        "--mtp-null-only-when-all-missing",
        action="store_true",
        help=(
            "mask the learned MTP null token whenever at least one real "
            "draft node is available"
        ),
    )
    parser.add_argument(
        "--mtp-horizon-depth-attention-bias",
        action="store_true",
        help=(
            "learn separate per-head horizon-by-depth additive biases for "
            "hidden and router MTP cross-attention"
        ),
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument("--diagnostic-train-rows", type=int)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--router-kl-weight", type=float, default=0.25)
    parser.add_argument("--base-kl-weight", type=float, default=0.0)
    parser.add_argument("--delta-l2-weight", type=float, default=0.0)
    parser.add_argument("--relative-regret-weight", type=float, default=0.0)
    parser.add_argument(
        "--ordered-group-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="group memmap reads and use one pinned CPU read-ahead group",
    )
    parser.add_argument(
        "--freeze-output-bias",
        action="store_true",
        help="freeze the zero-initialized [H,L,E] residual output bias",
    )
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--predicted-boundary-weight", type=float, default=1.0)
    parser.add_argument(
        "--top8-swap-weight",
        type=float,
        default=0.0,
        help=(
            "offset-invariant loss on missing target experts versus false "
            "experts occupying predicted top-8 slots"
        ),
    )
    parser.add_argument("--full-membership-weight", type=float, default=0.25)
    parser.add_argument("--centered-score-weight", type=float, default=0.05)
    parser.add_argument("--hard-negative-end-rank", type=int, default=32)
    parser.add_argument("--predicted-negative-count", type=int, default=16)
    parser.add_argument(
        "--horizon-weights",
        default="1,1,1,1,0.5,0.5,0.5,0.5",
        help="eight comma-separated non-negative values",
    )
    return parser


def _float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(result) != 8:
        raise ValueError("--horizon-weights must contain exactly eight values")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.enable_experimental_full_router:
        raise SystemExit(
            "refusing to run without --enable-experimental-full-router"
        )
    common = {
        "capture_dir": args.capture_dir,
        "mtp_dir": args.mtp_dir,
        "target_features": args.target_features,
        "target_feature_rms": args.target_feature_rms,
        "secondary_target_features": args.secondary_target_features,
        "secondary_target_feature_rms": args.secondary_target_feature_rms,
        "rows_per_request": args.rows_per_request,
        "history": args.history,
        "base_floor_margin": args.base_floor_margin,
    }
    train = FullRouterForecastData(
        args.train_pool, expected_split="train", **common
    )
    validation = FullRouterForecastData(
        args.validation_pool, expected_split="validation", **common
    )
    model_config = derive_router_forecaster_config(
        train,
        overrides={
            "model_width": args.model_width,
            "attention_heads": args.attention_heads,
            "feedforward_width": args.feedforward_width,
            "layer_blocks": args.layer_blocks,
            "mtp_blocks": args.mtp_blocks,
            "fusion_blocks": args.fusion_blocks,
            "output_rank": args.output_rank,
            "dropout": args.dropout,
            "mtp_diagonal_from_local": args.mtp_diagonal_from_local,
            "mtp_null_only_when_all_missing": (
                args.mtp_null_only_when_all_missing
            ),
            "mtp_horizon_depth_attention_bias": (
                args.mtp_horizon_depth_attention_bias
            ),
        },
    )
    loss_config = FullRouterLossConfig(
        temperature=args.temperature,
        router_kl=args.router_kl_weight,
        base_kl=args.base_kl_weight,
        delta_l2=args.delta_l2_weight,
        relative_regret=args.relative_regret_weight,
        boundary=args.boundary_weight,
        predicted_boundary=args.predicted_boundary_weight,
        top8_swap=args.top8_swap_weight,
        full_membership=args.full_membership_weight,
        centered_score=args.centered_score_weight,
        hard_negative_end_rank=args.hard_negative_end_rank,
        predicted_negative_count=args.predicted_negative_count,
        horizon_weights=_float_tuple(args.horizon_weights),
    )
    training_config = FullRouterTrainingConfig(
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        patience=args.patience,
        microbatch_size=args.microbatch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        warmup_fraction=args.warmup_fraction,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        diagnostic_train_rows=args.diagnostic_train_rows,
        base_floor_margin=args.base_floor_margin,
        ordered_group_prefetch=args.ordered_group_prefetch,
        freeze_output_bias=args.freeze_output_bias,
    )
    manifest = train_jspace_router_forecaster(
        train,
        validation,
        args.output,
        model_config,
        loss_config,
        training_config,
        target_features=args.target_features,
        target_feature_rms=args.target_feature_rms,
        secondary_target_features=args.secondary_target_features,
        secondary_target_feature_rms=args.secondary_target_feature_rms,
        device=args.device,
        resume=args.resume,
        stop_after_epoch=args.stop_after_epoch,
        strict_lineage=not args.no_strict_lineage,
        precision_audit=args.precision_audit,
        allow_provisional_j_without_probe=args.allow_provisional_j_without_probe,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FullRouterTrainingConfig",
    "JSPACE_ROUTER_ARCHITECTURE_NAME",
    "JSPACE_ROUTER_TRAINING_MANIFEST_SCHEMA",
    "JSPACE_ROUTER_TRAINING_SCHEMA",
    "build_parser",
    "derive_router_forecaster_config",
    "load_jspace_router_forecaster_checkpoint",
    "main",
    "train_jspace_router_forecaster",
]
