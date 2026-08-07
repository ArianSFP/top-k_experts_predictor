"""Train and evaluate the request-aligned J-HARP candidate reranker.

This entry point is deliberately validation-only.  It accepts separate
``train`` and ``validation`` candidate pools, verifies their immutable input
artifacts, and has no interface for opening a sealed test pool.  The dense
target features are joined to candidate rows by :class:`AlignedJCandidateData`.

The production defaults are sized for a 24 GiB RTX 3090: one complete token
record per microbatch and 32-way gradient accumulation.  CUDA execution uses
BF16 autocast, FP32 loss reductions, TF32 matrix multiplication, and fused
AdamW when PyTorch exposes it.  By default only H1-H4 enter the trainable
ranker; the deployed composite returns the frozen HARP scores unchanged for
H5-H8.  ``--active-horizons 8`` enables the full formal reranker when memory
allows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .candidates import POOL_SCHEMA, POOL_SCHEMA_V2
from .jspace_data import (
    AlignedJCandidateData,
    move_tensor_batch,
    ordered_prefetched_batches,
    slice_tensor_batch,
)
from .jspace_features import FEATURE_SCHEMA, PRECISION_AUDIT_SCHEMA
from .jspace_loss_profiles import (
    JSPACE_LOSS_PROFILE_NAMES,
    PREREGISTERED_V1,
    assert_resume_loss_profile,
    provenance_with_loss_profile,
    resolve_jspace_loss_profile,
    validate_jspace_loss_profile,
)
from .jspace_metrics import evaluate_jspace_ranker, paired_request_bootstrap
from .jspace_router_data import (
    _read_pool_manifest,
    validate_level2_pool_request_contract,
)
from .jspace_reranker import (
    JSpaceCandidateReranker,
    JSpaceRerankerConfig,
    JSpaceRerankerLossConfig,
    jspace_reranker_loss,
)
from .router_geometry import ROUTER_KEY_SCHEMA
from .prepare_jspace_features import AUDIT_SCHEMA, PREPARATION_SCHEMA
from .train import canonical_json, sha256_file, write_csv


JSPACE_TRAINING_SCHEMA = "harp8_jspace_reranker_training_v1"
JSPACE_TRAINING_MANIFEST_SCHEMA = "harp8_jspace_reranker_manifest_v1"
COMPACT_CAPTURE_SCHEMA = "jroute0_compact_bf16_capture_v1"
MTP_DEPTH_CAPTURE_SCHEMA = "jroute0_native_bf16_mtp_depth_capture_v1"
DECISION_PROFILE = "token_end_informational_upper_bound"
_HORIZON_BATCH_KEYS = frozenset(
    {
        "candidate_scores",
        "candidate_ids",
        "target_membership",
        "teacher_candidate_scores",
        "valid_future",
        "current_scores",
        "current_rank",
        "source_gates",
        "copy_gates",
        "context",
        "generator_context",
        "candidate_route_scores",
        "candidate_history_membership",
        "candidate_features",
        "candidate_mask",
    }
)


@dataclass(frozen=True)
class JSpaceTrainingConfig:
    """Optimizer, checkpoint, and validation policy for J-HARP-C64."""

    epochs: int = 30
    minimum_epochs: int = 10
    patience: int = 5
    microbatch_size: int = 1
    evaluation_batch_size: int = 1
    gradient_accumulation: int = 32
    learning_rate: float = 2e-4
    minimum_learning_rate: float = 2e-5
    warmup_fraction: float = 0.03
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    gradient_clip: float = 1.0
    seed: int = 42
    bootstrap_replicates: int = 2000
    max_train_rows: int | None = None
    max_validation_rows: int | None = None
    active_horizons: int = 4
    decision_profile: str = DECISION_PROFILE
    mtp_timing_causal: bool = False
    acceptance_labels_used: bool = False

    def validate(self) -> None:
        integer_positive = {
            "epochs": self.epochs,
            "minimum_epochs": self.minimum_epochs,
            "patience": self.patience,
            "microbatch_size": self.microbatch_size,
            "evaluation_batch_size": self.evaluation_batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "bootstrap_replicates": self.bootstrap_replicates,
            "active_horizons": self.active_horizons,
        }
        invalid = [name for name, value in integer_positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive training dimensions required: {invalid}")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs cannot exceed epochs")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must lie in [0,1)")
        if not 0.0 < self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("minimum learning rate must lie in (0, learning rate]")
        if self.weight_decay < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("weight decay and gradient clipping are invalid")
        for name in ("max_train_rows", "max_validation_rows"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when supplied")
        if self.decision_profile != DECISION_PROFILE:
            raise ValueError(f"decision_profile must be {DECISION_PROFILE!r}")
        if self.mtp_timing_causal is not False:
            raise ValueError(
                "this corpus does not support timing-causal MTP attribution"
            )
        if self.acceptance_labels_used is not False:
            raise ValueError("MTP acceptance labels must remain excluded from inputs")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _RowPrefixView:
    """Read-only prefix view used only for reproducible tiny pilots."""

    def __init__(self, base: AlignedJCandidateData, maximum: int | None) -> None:
        self.base = base
        self.rows = base.rows if maximum is None else min(base.rows, int(maximum))
        if self.rows <= 0:
            raise ValueError("a data view must contain at least one row")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def batch(
        self,
        rows: np.ndarray,
        device: str | torch.device,
        *,
        active_horizons: int | None = None,
        include_context: bool = False,
        compact: bool = False,
    ) -> dict[str, torch.Tensor]:
        values = np.asarray(rows, dtype=np.int64)
        if values.ndim != 1 or (values < 0).any() or (values >= self.rows).any():
            raise IndexError("view batch rows are out of range")
        return self.base.batch(
            values,
            device,
            active_horizons=active_horizons,
            include_context=include_context,
            compact=compact,
        )

    def sequential_batches(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.rows, batch_size):
            yield np.arange(start, min(self.rows, start + batch_size), dtype=np.int64)


def _load_router_keys(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(path)
    manifest_path: Path | None = None
    if path.is_dir():
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != ROUTER_KEY_SCHEMA:
            raise ValueError("router-key directory has an incompatible schema")
        key_path = path / str(manifest["keys"]["path"])
        expected_hash = str(manifest["keys"]["sha256"])
        actual_hash = sha256_file(key_path)
        if actual_hash != expected_hash:
            raise ValueError("router-key file disagrees with its immutable manifest")
    else:
        key_path = path
        actual_hash = sha256_file(key_path)
        sibling = key_path.parent / "manifest.json"
        if sibling.exists():
            candidate = json.loads(sibling.read_text(encoding="utf-8"))
            if candidate.get("schema") == ROUTER_KEY_SCHEMA:
                manifest_path = sibling
                expected_hash = str(candidate["keys"]["sha256"])
                if actual_hash != expected_hash:
                    raise ValueError(
                        "router-key file disagrees with its immutable manifest"
                    )
    values = np.load(key_path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("router keys must be a finite [layers,experts,rank] array")
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).copy())
    return tensor, {
        "path": str(key_path),
        "sha256": actual_hash,
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "manifest": (
            {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            }
            if manifest_path is not None
            else None
        ),
    }


def candidate_feature_width(history: int) -> int:
    """Width of the deterministic per-candidate evidence assembled below."""

    if history <= 0:
        raise ValueError("history must be positive")
    # The aligned-data contract fixes three lag slots (zero padded when the
    # configured history is shorter): current score/rank/copy (3), source
    # gates (3), route scores (3), memberships (3), and request position (1).
    return 13


def slice_horizon_batch(
    batch: Mapping[str, torch.Tensor], active_horizons: int
) -> dict[str, torch.Tensor]:
    """Slice every declared horizon-axis tensor before ranker execution."""

    if active_horizons <= 0:
        raise ValueError("active_horizons must be positive")
    result = dict(batch)
    for name in _HORIZON_BATCH_KEYS:
        value = result.get(name)
        if value is None:
            continue
        if value.ndim < 2 or value.shape[1] < active_horizons:
            raise ValueError(f"{name} cannot supply {active_horizons} active horizons")
        result[name] = value[:, :active_horizons]
    return result


def prepare_model_batch(
    batch: Mapping[str, torch.Tensor],
    *,
    include_feature_rms: bool,
    include_candidate_features: bool,
    active_horizons: int | None = None,
) -> dict[str, torch.Tensor]:
    """Translate aligned storage names to the ranker's causal tensor contract."""

    result = (
        slice_horizon_batch(batch, active_horizons)
        if active_horizons is not None
        else dict(batch)
    )
    batch = result
    history_available = batch.get("history_available")
    if "j_states" in batch:
        # AlignedJCandidateData owns the authoritative optional-RMS contract.
        # Reusing it here prevents appending the side channel twice.
        result["j_states"] = batch["j_states"].float()
    else:
        if history_available is None:
            raise KeyError("target history requires an availability tensor")
        j_states = batch["target_history"].float()
        if include_feature_rms:
            j_states = torch.cat(
                [j_states, batch["target_rms_history"].float()], dim=-1
            )
        result["j_states"] = j_states
    if "j_mask" in batch:
        result["j_mask"] = batch["j_mask"].bool()
    else:
        if history_available is None:
            raise KeyError("J states require a causal availability mask")
        result["j_mask"] = history_available.bool()[:, :, None].expand(
            -1, -1, result["j_states"].shape[2]
        )
    if "mtp_mask" in batch:
        result["mtp_mask"] = batch["mtp_mask"].bool()
    else:
        result["mtp_mask"] = batch["mtp_available"].bool()

    if include_candidate_features:
        if "candidate_features" in batch:
            result["candidate_features"] = batch["candidate_features"].float()
            return result
        base = batch["candidate_scores"]
        b, h, layers, candidates = base.shape
        source_gates = (
            batch["source_gates"]
            .float()
            .unsqueeze(-2)
            .expand(-1, -1, -1, candidates, -1)
        )
        within = (
            batch["within_request"]
            .float()
            .view(b, 1, 1, 1, 1)
            .expand(-1, h, layers, candidates, -1)
        )
        route_scores = batch["candidate_route_scores"][..., :3].float()
        route_membership = batch["candidate_history_membership"][..., :3].float()
        if route_scores.shape[-1] < 3:
            padding = route_scores.new_zeros(
                route_scores.shape[:-1] + (3 - route_scores.shape[-1],)
            )
            route_scores = torch.cat([route_scores, padding], dim=-1)
            route_membership = torch.cat([route_membership, padding], dim=-1)
        result["candidate_features"] = torch.cat(
            [
                batch["current_scores"].float().unsqueeze(-1),
                batch["current_rank"].float().unsqueeze(-1),
                batch["copy_gates"].float().unsqueeze(-1),
                source_gates,
                route_scores,
                route_membership,
                within,
            ],
            dim=-1,
        )
    return result


def derive_model_config(
    data: AlignedJCandidateData,
    router_keys: torch.Tensor,
    *,
    include_feature_rms: bool,
    include_candidate_features: bool = True,
    active_horizons: int = 4,
    allow_router_rank_ablation: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> JSpaceRerankerConfig:
    """Derive all input/output geometry from aligned data and frozen keys."""

    layers, experts, key_width = map(int, router_keys.shape)
    pool = data.pool
    if (layers, experts) != (pool.layers, pool.experts):
        raise ValueError("router keys disagree with candidate-pool geometry")
    if key_width != experts and not allow_router_rank_ablation:
        raise ValueError(
            "scientific J-HARP runs require full router rank R=E; pass the "
            "explicit router-rank ablation override only for an ablation"
        )
    if active_horizons not in (4, pool.horizons):
        raise ValueError("active_horizons must be 4 or the complete pool horizon count")
    if active_horizons > pool.horizons:
        raise ValueError("active_horizons exceeds the candidate-pool horizon count")
    values: dict[str, Any] = {
        "experts": experts,
        "layers": layers,
        "horizons": active_horizons,
        "candidate_count": pool.candidate_count,
        "native_k": int(pool.manifest.get("native_k", 8)),
        "j_lags": data.history,
        "j_width": data.target_width,
        "mtp_nodes": data.mtp_depths,
        "mtp_state_channels": 1,
        "mtp_state_width": data.mtp_width,
        "router_key_width": key_width,
        "candidate_feature_width": (
            int(
                data.batch(
                    np.asarray([0], dtype=np.int64),
                    "cpu",
                    active_horizons=active_horizons,
                    include_context=False,
                )["candidate_features"].shape[-1]
            )
            if include_candidate_features
            else 0
        ),
    }
    if overrides:
        values.update(dict(overrides))
    config = JSpaceRerankerConfig(**values)
    config.validate()
    return config


def _assert_matching_data(
    train: AlignedJCandidateData,
    validation: AlignedJCandidateData,
) -> None:
    names = (
        "horizons",
        "layers",
        "experts",
        "candidate_count",
    )
    mismatches = [
        name
        for name in names
        if getattr(train.pool, name) != getattr(validation.pool, name)
    ]
    for name in ("history", "target_width", "mtp_width", "mtp_depths"):
        if getattr(train, name) != getattr(validation, name):
            mismatches.append(name)
    if mismatches:
        raise ValueError(f"train and validation data disagree: {mismatches}")
    train_groups = set(map(int, train.pool.request_ids.tolist()))
    validation_groups = set(map(int, validation.pool.request_ids.tolist()))
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(
            f"train/validation request leakage detected for {len(overlap)} requests"
        )


def _coverage_gate(data: AlignedJCandidateData) -> dict[str, Any]:
    """Compute request-macro oracle coverage without loading dense features."""

    pool = data.pool
    native_k = int(pool.manifest.get("native_k", 8))
    if not 1 <= native_k <= pool.candidate_count:
        raise ValueError("native_k must lie within the candidate pool")
    by_request: dict[tuple[int, int], list[float]] = defaultdict(list)
    for start in range(0, pool.rows, 256):
        stop = min(pool.rows, start + 256)
        membership = np.asarray(pool.target_membership[start:stop], dtype=np.uint8)
        valid = np.asarray(pool.valid_future[start:stop], dtype=np.bool_)
        coverage = membership.sum(axis=-1).mean(axis=-1) / native_k
        for local, row in enumerate(range(start, stop)):
            request_id = int(pool.request_ids[row])
            for column in range(min(4, pool.horizons)):
                if valid[local, column]:
                    by_request[(request_id, column + 1)].append(
                        float(coverage[local, column])
                    )
    horizon_values: list[float] = []
    for horizon in range(1, 5):
        request_values = [
            float(np.mean(rows))
            for (request_id, value_horizon), rows in by_request.items()
            if value_horizon == horizon
        ]
        if not request_values:
            raise ValueError(f"validation pool has no valid horizon-{horizon} rows")
        horizon_values.append(float(np.mean(request_values)))
    mean = float(np.mean(horizon_values))
    h4 = horizon_values[3]
    return {
        "denominator": native_k,
        "candidate_count": pool.candidate_count,
        "request_macro_coverage_by_horizon": horizon_values,
        "mean_h1_h4_coverage": mean,
        "h4_coverage": h4,
        "required_mean_h1_h4": 0.97,
        "required_h4": 0.95,
        "passes": bool(mean >= 0.97 and h4 >= 0.95),
    }


def _optimizer_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if (
            parameter.ndim == 1
            or lowered.endswith("bias")
            or "embedding" in lowered
            or "norm" in lowered
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, int(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup:
            return max(1e-8, float(step + 1) / warmup)
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _rng_state(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_generator": rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_rng(state: Mapping[str, Any], rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    rng.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def _hash_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _verify_file_record(path: Path, record: Any, label: str) -> dict[str, Any]:
    """Verify one hash-bearing file record and return canonical provenance."""

    value = _mapping(record, label)
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != int(value.get("bytes", -1)):
        raise ValueError(f"{label} byte count disagrees with its manifest")
    actual_hash = sha256_file(path)
    if actual_hash != str(value.get("sha256", "")):
        raise ValueError(f"{label} hash disagrees with its manifest")
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_hash}


def _read_request_splits(path: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    records: list[dict[str, Any]] = []
    splits: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                request_id = int(record["request_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid request record at {path}:{line_number}"
                ) from exc
            if request_id in splits:
                raise ValueError("capture requests contain duplicate request IDs")
            offline = record.get("offline_split")
            generic = record.get("split")
            if (
                offline is not None
                and generic is not None
                and str(offline) != str(generic)
            ):
                raise ValueError(f"request {request_id} has conflicting split fields")
            split = offline if offline is not None else generic
            if split is None:
                raise ValueError(f"request {request_id} lacks an offline split")
            records.append(record)
            splits[request_id] = str(split)
    if not records:
        raise ValueError("capture requests are empty")
    return records, splits


def _manifest_identity(
    left: Any, right: Any, label: str, *, allow_absent: bool = False
) -> str | None:
    if left is None or right is None:
        if allow_absent and left is None and right is None:
            return None
        raise ValueError(f"train/validation {label} identity is missing")
    left_value = _mapping(left, f"train {label}")
    right_value = _mapping(right, f"validation {label}")
    left_hash = str(left_value.get("sha256", ""))
    right_hash = str(right_value.get("sha256", ""))
    if not left_hash or left_hash != right_hash:
        raise ValueError(f"train/validation {label} identities disagree")
    return left_hash


def _optional_manifest_hash(value: Any, label: str) -> str | None:
    if value is None:
        return None
    record = _mapping(value, label)
    digest = str(record.get("sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} lacks a valid SHA-256 identity")
    return digest


def _candidate_pool_split_lineage(
    train_manifest: Mapping[str, Any],
    validation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate either legacy shared-split or explicit level-2 OOF lineage."""

    train_schema = train_manifest.get("schema")
    validation_schema = validation_manifest.get("schema")
    if train_schema != validation_schema:
        raise ValueError("mixed candidate-pool schemas are not scientific lineage")
    if train_schema == POOL_SCHEMA:
        shared = _manifest_identity(
            train_manifest.get("split_manifest"),
            validation_manifest.get("split_manifest"),
            "split manifest",
            allow_absent=True,
        )
        return {
            "mode": "shared_source_split_v1",
            "split_manifest_sha256": shared,
        }
    if train_schema != POOL_SCHEMA_V2:
        raise ValueError("candidate pools have an unsupported scientific schema")

    required_roles = (
        (train_manifest, "train", "train"),
        (validation_manifest, "validation", "validation"),
    )
    for manifest, level2_split, offline_split in required_roles:
        if manifest.get("level2_split") != level2_split:
            raise ValueError(
                f"OOF {level2_split} pool has the wrong level-2 role"
            )
        if manifest.get("source_offline_split") != offline_split:
            raise ValueError(
                f"OOF {level2_split} pool has the wrong source offline split"
            )
        if manifest.get("base_fit_excluded") is not True:
            raise ValueError(
                f"OOF {level2_split} pool lacks base-fit exclusion"
            )
        if int(manifest.get("base_fit_overlap_count", -1)) != 0:
            raise ValueError(
                f"OOF {level2_split} pool reports base-fit overlap"
            )

    train_fold = str(train_manifest.get("base_fold_id", ""))
    validation_fold = str(validation_manifest.get("base_fold_id", ""))
    if not train_fold or train_fold != validation_fold:
        raise ValueError("OOF train/validation base-fold identities disagree")
    train_base = str(train_manifest.get("base_train_request_ids_sha256", ""))
    validation_base = str(
        validation_manifest.get("base_train_request_ids_sha256", "")
    )
    if (
        len(train_base) != 64
        or any(character not in "0123456789abcdef" for character in train_base)
        or train_base != validation_base
    ):
        raise ValueError("OOF train/validation base-fit request identities disagree")
    base_fit_manifest_hash = _manifest_identity(
        train_manifest.get("base_fit_split_manifest"),
        validation_manifest.get("base_fit_split_manifest"),
        "base-fit split manifest",
    )
    return {
        "mode": "level2_oof_v2",
        "split_manifest_sha256": None,
        "base_fold_id": train_fold,
        "base_train_request_ids_sha256": train_base,
        "base_fit_split_manifest_sha256": base_fit_manifest_hash,
        "source_row_split_manifest_sha256": {
            "train": _optional_manifest_hash(
                train_manifest.get("source_row_split_manifest"),
                "train source-row split manifest",
            ),
            "validation": _optional_manifest_hash(
                validation_manifest.get("source_row_split_manifest"),
                "validation source-row split manifest",
            ),
        },
    }


def _validate_scientific_lineage(
    train: AlignedJCandidateData,
    validation: AlignedJCandidateData,
    *,
    target_features: Path,
    target_feature_rms: Path | None,
    precision_audit: Path | None,
    allow_provisional_j_without_probe: bool,
) -> dict[str, Any]:
    """Fail closed over every semantic lineage edge used by a scientific run."""

    shared_paths = {
        "capture": (train.capture_dir, validation.capture_dir),
        "mtp": (train.mtp_dir, validation.mtp_dir),
        "target features": (
            train.target_features_path,
            validation.target_features_path,
        ),
        "target feature RMS": (
            train.target_feature_rms_path,
            validation.target_feature_rms_path,
        ),
    }
    for label, (left, right) in shared_paths.items():
        if left is None or right is None:
            if left is right:
                continue
            raise ValueError(f"train/validation {label} availability differs")
        if Path(left).resolve() != Path(right).resolve():
            raise ValueError(f"train/validation {label} sources differ")
    if Path(target_features).resolve() != train.target_features_path.resolve():
        raise ValueError(
            "target-features argument differs from the opened feature store"
        )
    supplied_rms = Path(target_feature_rms).resolve() if target_feature_rms else None
    opened_rms = (
        train.target_feature_rms_path.resolve()
        if train.target_feature_rms_path is not None
        else None
    )
    if supplied_rms != opened_rms:
        raise ValueError(
            "target-feature-rms argument differs from the opened feature store"
        )

    capture_manifest_path = train.capture_dir / "manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    if capture_manifest.get("schema") != COMPACT_CAPTURE_SCHEMA:
        raise ValueError("target capture has an incompatible scientific schema")
    capture_files = _mapping(capture_manifest.get("files"), "capture files")
    verified_inputs: dict[str, Any] = {}
    for filename, record in capture_files.items():
        verified_inputs[f"capture/{filename}"] = _verify_file_record(
            train.capture_dir / str(filename), record, f"capture file {filename}"
        )
    required_capture_files = {
        "requests.jsonl",
        "raw_router_logits.npy",
        "top8_expert_ids.npy",
    }
    if not required_capture_files.issubset(capture_files):
        raise ValueError("capture manifest omits required router/request files")
    capture_arrays = _mapping(capture_manifest.get("arrays"), "capture arrays")
    expected_capture_arrays = {
        "raw_router_logits": (
            "raw_router_logits.npy",
            list(train.router_logits.shape),
            str(train.router_logits.dtype),
        ),
        "top8_expert_ids": (
            "top8_expert_ids.npy",
            list(train.top8.shape),
            str(train.top8.dtype),
        ),
    }
    for label, (filename, shape, dtype) in expected_capture_arrays.items():
        record = _mapping(capture_arrays.get(label), f"capture array {label}")
        if (
            Path(str(record.get("path", ""))).name != filename
            or list(record.get("shape", [])) != shape
            or str(record.get("dtype")) != dtype
        ):
            raise ValueError(f"capture array declaration disagrees for {label}")
    request_path = train.capture_dir / "requests.jsonl"
    requests, request_splits = _read_request_splits(request_path)
    if int(capture_manifest.get("requests", -1)) != len(requests):
        raise ValueError("capture request count disagrees with requests JSONL")
    if int(capture_manifest.get("rows", -1)) != int(train.router_logits.shape[0]):
        raise ValueError("capture row count disagrees with router tensors")
    if (
        int(capture_manifest.get("routed_positions_per_request", -1))
        != train.rows_per_request
    ):
        raise ValueError("capture rows-per-request disagrees with aligned data")

    mtp_manifest_path = train.mtp_dir / "manifest.json"
    mtp_manifest = json.loads(mtp_manifest_path.read_text(encoding="utf-8"))
    if mtp_manifest.get("schema") != MTP_DEPTH_CAPTURE_SCHEMA:
        raise ValueError("MTP capture has an incompatible scientific schema")
    if mtp_manifest.get("source_capture_manifest_sha256") != sha256_file(
        capture_manifest_path
    ):
        raise ValueError("MTP source-capture manifest identity disagrees")
    alignment = _mapping(mtp_manifest.get("alignment"), "MTP alignment")
    if alignment.get("future_committed_tokens_used_as_features") is not False:
        raise ValueError("MTP capture does not attest label-free causal features")
    if (
        int(mtp_manifest.get("rows", -1)) != int(train.mtp_states.shape[0])
        or int(mtp_manifest.get("requests", -1)) != len(requests)
        or int(mtp_manifest.get("draft_depths", -1)) != train.mtp_depths
    ):
        raise ValueError("MTP manifest geometry disagrees with aligned data")
    target_model = capture_manifest.get("model_provenance")
    mtp_model = mtp_manifest.get("model_provenance")
    if target_model is not None and mtp_model is not None and target_model != mtp_model:
        raise ValueError("target and MTP model provenance disagree")
    mtp_files = _mapping(mtp_manifest.get("files"), "MTP files")
    for label, record_value in mtp_files.items():
        record = _mapping(record_value, f"MTP file {label}")
        filename = str(record.get("filename", ""))
        if not filename or Path(filename).name != filename:
            raise ValueError(f"MTP file {label} has an invalid filename")
        verified_inputs[f"mtp/{filename}"] = _verify_file_record(
            train.mtp_dir / filename, record, f"MTP file {label}"
        )
    for label, filename in {
        "hidden_depths": "mtp_hidden_depths.npy",
        "router_logits_depths": "mtp_router_logits_depths.npy",
    }.items():
        record = _mapping(mtp_files.get(label), f"MTP file {label}")
        if record.get("filename") != filename:
            raise ValueError(f"MTP manifest omits canonical {label}")

    feature_path = train.target_features_path
    feature_store_dir = feature_path.parent
    if feature_path.name != "features_normalized.npy":
        raise ValueError("scientific target features must use the canonical filename")
    feature_manifest_path = feature_store_dir / "manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("schema") != FEATURE_SCHEMA:
        raise ValueError("target feature store has an incompatible schema")
    feature_outputs = _mapping(feature_manifest.get("outputs"), "feature outputs")
    feature_record = feature_outputs.get("features_normalized.npy")
    verified_inputs["features/features_normalized.npy"] = _verify_file_record(
        feature_path, feature_record, "target feature array"
    )
    if target_feature_rms is not None:
        if (
            Path(target_feature_rms).parent.resolve() != feature_store_dir.resolve()
            or Path(target_feature_rms).name != "log_rms.npy"
        ):
            raise ValueError(
                "target RMS must be the matching feature-store side channel"
            )
        verified_inputs["features/log_rms.npy"] = _verify_file_record(
            Path(target_feature_rms),
            feature_outputs.get("log_rms.npy"),
            "target feature RMS",
        )
    if (
        int(feature_manifest.get("rows", -1)) != int(train.target_features.shape[0])
        or int(feature_manifest.get("layers", -1)) != train.pool.layers
        or int(feature_manifest.get("feature_width", -1))
        != train.projected_target_width
    ):
        raise ValueError("target feature manifest geometry disagrees with aligned data")

    preparation_manifest_path = feature_store_dir.parent.parent / "manifest.json"
    preparation = json.loads(preparation_manifest_path.read_text(encoding="utf-8"))
    if preparation.get("schema") != PREPARATION_SCHEMA:
        raise ValueError("feature preparation manifest has an incompatible schema")
    source = _mapping(preparation.get("source"), "feature preparation source")
    source_requests = _mapping(source.get("requests_jsonl"), "feature request source")
    request_hash = sha256_file(request_path)
    if source_requests.get("sha256") != request_hash:
        raise ValueError("feature requests do not match capture requests")
    fit = _mapping(preparation.get("fit"), "feature PCA fit")
    if fit.get("split") != "train":
        raise ValueError("scientific PCA must be fitted on split exactly 'train'")
    fit_ids = sorted(
        str(record["request_id"])
        for record in requests
        if request_splits[int(record["request_id"])] == "train"
    )
    expected_fit_hash = hashlib.sha256(
        canonical_json(fit_ids).encode("utf-8")
    ).hexdigest()
    if fit.get("request_ids_sha256") != expected_fit_hash:
        raise ValueError("feature PCA fit request identity disagrees with train split")
    representation = str(feature_manifest.get("representation", ""))
    rank = str(int(feature_manifest.get("feature_width", -1)))
    representations = _mapping(preparation.get("representations"), "representations")
    representation_record = _mapping(
        representations.get(representation), f"representation {representation}"
    )
    ranks = _mapping(representation_record.get("ranks"), "representation ranks")
    rank_record = _mapping(ranks.get(rank), f"representation rank {rank}")
    prepared_store = _mapping(
        rank_record.get("feature_store"), "prepared feature store"
    )
    if prepared_store.get("schema") != FEATURE_SCHEMA:
        raise ValueError("preparation manifest embeds an invalid feature store")
    prepared_outputs = _mapping(prepared_store.get("outputs"), "prepared outputs")
    for filename, actual in feature_outputs.items():
        expected = _mapping(prepared_outputs.get(filename), f"prepared {filename}")
        actual_value = _mapping(actual, f"feature {filename}")
        if actual_value.get("sha256") != expected.get("sha256") or int(
            actual_value.get("bytes", -1)
        ) != int(expected.get("bytes", -2)):
            raise ValueError("feature store disagrees with preparation manifest")
    feature_source = _mapping(feature_manifest.get("source"), "feature source")
    prepared_residual = _mapping(source.get("residual_npy"), "prepared residual")
    if feature_source.get("sha256") != prepared_residual.get("sha256"):
        raise ValueError("feature source residual lineage disagrees")
    pca_record = _mapping(rank_record.get("pca_artifact"), "PCA artifact")
    pca_path = feature_store_dir.parent / f"shared_pca_rank{rank}.npz"
    if sha256_file(pca_path) != str(pca_record.get("file_sha256", "")):
        raise ValueError("PCA artifact hash disagrees with preparation manifest")

    train_manifest = train.pool.manifest
    validation_manifest = validation.pool.manifest
    checkpoint_hash = _manifest_identity(
        train_manifest.get("checkpoint"),
        validation_manifest.get("checkpoint"),
        "candidate-generator checkpoint",
    )
    pool_split_lineage = _candidate_pool_split_lineage(
        train_manifest, validation_manifest
    )
    for split, data in (("train", train), ("validation", validation)):
        validate_level2_pool_request_contract(
            data.pool.manifest,
            data.pool.request_ids,
            expected_split=split,
        )
        wrong = sorted(
            request_id
            for request_id in set(map(int, data.pool.request_ids.tolist()))
            if request_splits.get(request_id) != split
        )
        if wrong:
            raise ValueError(
                f"{split} pool contains {len(wrong)} requests outside its offline split"
            )

    precision: dict[str, Any] = {
        "required": representation == "j_lens",
        "attribution": "not_applicable",
    }
    if representation == "j_lens":
        if precision_audit is None:
            raise ValueError("J-Lens scientific runs require --precision-audit")
        audit_path = Path(precision_audit)
        audit_manifest = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_manifest.get("schema") != AUDIT_SCHEMA:
            raise ValueError("precision audit has an incompatible command schema")
        audit = _mapping(audit_manifest.get("audit"), "precision audit payload")
        if (
            audit.get("schema") != PRECISION_AUDIT_SCHEMA
            or audit.get("accepted") is not True
        ):
            raise ValueError("precision audit is incompatible or did not pass")
        probe_delta = audit.get("probe_recall_delta")
        if probe_delta is None and not allow_provisional_j_without_probe:
            raise ValueError(
                "J-Lens precision attribution lacks a probe; use the explicit "
                "provisional override only for the representative audit"
            )
        if probe_delta is not None and allow_provisional_j_without_probe:
            raise ValueError(
                "provisional precision override is invalid when a probe exists"
            )
        audit_lens = _mapping(audit_manifest.get("lens_checkpoint"), "audit lens")
        feature_lens = _mapping(feature_manifest.get("lens"), "feature lens")
        if (
            audit_lens.get("sha256")
            and feature_lens.get("sha256")
            and audit_lens.get("sha256") != feature_lens.get("sha256")
        ):
            raise ValueError("precision audit and feature store use different J lenses")
        precision = {
            "required": True,
            "attribution": "provisional" if probe_delta is None else "validated",
            "probe_recall_delta": probe_delta,
            "audit": _hash_file(audit_path),
        }
    elif allow_provisional_j_without_probe:
        raise ValueError("provisional J precision override is valid only for J-Lens")

    level2_train_ids = set(map(int, train.pool.request_ids.tolist()))
    preprocessing_fit_ids = set(map(int, fit_ids))
    preprocessing_overlap = len(level2_train_ids.intersection(preprocessing_fit_ids))
    preprocessing_profile = (
        "transductive_preprocessing_oof_proxy"
        if pool_split_lineage["mode"] == "level2_oof_v2"
        and preprocessing_overlap > 0
        else "request_excluded_preprocessing"
    )

    return {
        "strict": True,
        "capture_manifest": _hash_file(capture_manifest_path),
        "mtp_manifest": _hash_file(mtp_manifest_path),
        "feature_store_manifest": _hash_file(feature_manifest_path),
        "feature_preparation_manifest": _hash_file(preparation_manifest_path),
        "representation": representation,
        "pca_fit_split": "train",
        "candidate_generator_checkpoint_sha256": checkpoint_hash,
        "split_manifest_sha256": pool_split_lineage["split_manifest_sha256"],
        "candidate_pool_split_lineage": pool_split_lineage,
        "preprocessing_profile": preprocessing_profile,
        "level2_train_requests_seen_by_preprocessing": preprocessing_overlap,
        "pool_request_offline_splits_verified": True,
        "precision_attribution": precision,
        "verified_inputs": verified_inputs,
    }


def _input_provenance(
    train: AlignedJCandidateData,
    validation: AlignedJCandidateData,
    *,
    target_features: Path,
    target_feature_rms: Path | None,
    router_provenance: Mapping[str, Any],
    semantic_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash every independently supplied input and verify pool array hashes."""

    pools: dict[str, Any] = {}
    for split, data in (("train", train), ("validation", validation)):
        root = data.pool.root
        declared_arrays: list[dict[str, Any]] = []
        for value in data.pool.manifest.get("arrays", []):
            array_path = root / str(value["path"])
            if array_path.stat().st_size != int(value.get("bytes", -1)):
                raise ValueError(f"candidate array byte mismatch: {array_path}")
            actual = sha256_file(array_path)
            if actual != str(value["sha256"]):
                raise ValueError(f"candidate array hash mismatch: {array_path}")
            declared_arrays.append(
                {
                    "path": str(array_path),
                    "bytes": array_path.stat().st_size,
                    "sha256": actual,
                }
            )
        pools[split] = {
            "manifest": _hash_file(root / "manifest.json"),
            "metadata": _hash_file(root / "metadata.json"),
            "arrays": declared_arrays,
        }
    verified = semantic_lineage.get("verified_inputs")
    if semantic_lineage.get("strict") is True and isinstance(verified, Mapping):
        explicit = dict(verified)
    else:
        explicit = {
            "capture_requests": _hash_file(train.capture_dir / "requests.jsonl"),
            "target_router_logits": _hash_file(
                train.capture_dir / "raw_router_logits.npy"
            ),
            "target_top8": _hash_file(train.capture_dir / "top8_expert_ids.npy"),
            "mtp_hidden": _hash_file(train.mtp_dir / "mtp_hidden_depths.npy"),
            "mtp_router_logits": _hash_file(
                train.mtp_dir / "mtp_router_logits_depths.npy"
            ),
            "target_features": _hash_file(Path(target_features)),
        }
        if target_feature_rms is not None:
            explicit["target_feature_rms"] = _hash_file(Path(target_feature_rms))
    return {
        "pools": pools,
        "aligned_inputs": explicit,
        "router_keys": dict(router_provenance),
        "semantic_lineage": dict(semantic_lineage),
    }


def _autocast(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def composite_horizon_scores(
    active_scores: torch.Tensor, base_scores: torch.Tensor
) -> torch.Tensor:
    """Return all horizons, replacing only the trainable active prefix."""

    if active_scores.ndim != 4 or base_scores.ndim != 4:
        raise ValueError("candidate scores must be [B,H,L,C]")
    if (
        active_scores.shape[0] != base_scores.shape[0]
        or active_scores.shape[2:] != base_scores.shape[2:]
        or active_scores.shape[1] > base_scores.shape[1]
    ):
        raise ValueError("active and base candidate-score geometries disagree")
    if active_scores.shape == base_scores.shape:
        return active_scores
    composite = base_scores.clone()
    composite[:, : active_scores.shape[1]] = active_scores
    return composite


class CompositeJSpaceInference(nn.Module):
    """Deployable H1--H4 reranker with exact frozen-base H5--H8 passthrough."""

    def __init__(
        self,
        reranker: JSpaceCandidateReranker,
        *,
        pool_horizons: int,
        include_feature_rms: bool,
        include_candidate_features: bool,
    ) -> None:
        super().__init__()
        if pool_horizons < reranker.config.horizons:
            raise ValueError("pool_horizons cannot be smaller than active horizons")
        self.reranker = reranker
        self.pool_horizons = int(pool_horizons)
        self.include_feature_rms = bool(include_feature_rms)
        self.include_candidate_features = bool(include_candidate_features)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base = batch["candidate_scores"].float()
        if base.ndim != 4 or base.shape[1] != self.pool_horizons:
            raise ValueError("inference batch does not match checkpoint pool horizons")
        prepared = prepare_model_batch(
            batch,
            include_feature_rms=self.include_feature_rms,
            include_candidate_features=self.include_candidate_features,
            active_horizons=self.reranker.config.horizons,
        )
        active = self.reranker(prepared)
        return {
            "scores": composite_horizon_scores(active.scores, base),
            "active_scores": active.scores,
            "active_delta": active.delta,
        }


def load_composite_jspace_checkpoint(
    checkpoint_path: Path,
    *,
    device: str | torch.device = "cpu",
    pool_horizons: int | None = None,
) -> CompositeJSpaceInference:
    """Load a self-contained J-HARP checkpoint for composite H1--H8 inference."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != JSPACE_TRAINING_SCHEMA:
        raise ValueError("checkpoint has an incompatible J-HARP training schema")
    config = JSpaceRerankerConfig(**payload["model_config"])
    config.validate()
    state = _mapping(payload.get("model_state"), "checkpoint model state")
    router_keys = state.get("router_keys")
    if not isinstance(router_keys, torch.Tensor):
        raise ValueError("checkpoint does not contain frozen router keys")
    model = JSpaceCandidateReranker(config, router_keys.float())
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    contract = payload.get("inference_contract", {})
    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint inference contract must be a mapping")
    recorded_horizons = int(contract.get("pool_horizons", max(8, config.horizons)))
    if pool_horizons is not None and int(pool_horizons) != recorded_horizons:
        raise ValueError("requested pool horizons disagree with checkpoint contract")
    aligned = payload.get("input_provenance", {}).get("aligned_inputs", {})
    include_rms = bool(
        contract.get(
            "include_feature_rms",
            "target_feature_rms" in aligned or "features/log_rms.npy" in aligned,
        )
    )
    include_candidate = bool(
        contract.get("include_candidate_features", config.candidate_feature_width > 0)
    )
    return CompositeJSpaceInference(
        model,
        pool_horizons=recorded_horizons,
        include_feature_rms=include_rms,
        include_candidate_features=include_candidate,
    )


class _EvaluationAdapter(nn.Module):
    def __init__(
        self,
        model: JSpaceCandidateReranker,
        *,
        include_feature_rms: bool,
        include_candidate_features: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.include_feature_rms = include_feature_rms
        self.include_candidate_features = include_candidate_features

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        active_horizons = self.model.config.horizons
        prepared = prepare_model_batch(
            batch,
            include_feature_rms=self.include_feature_rms,
            include_candidate_features=self.include_candidate_features,
            active_horizons=active_horizons,
        )
        active_scores = self.model(prepared).scores
        base = batch["candidate_scores"].float()
        return {"scores": composite_horizon_scores(active_scores, base)}


def _evaluate(
    model: JSpaceCandidateReranker,
    data: _RowPrefixView,
    *,
    batch_size: int,
    device: str,
    include_feature_rms: bool,
    include_candidate_features: bool,
) -> dict[str, Any]:
    adapter = _EvaluationAdapter(
        model,
        include_feature_rms=include_feature_rms,
        include_candidate_features=include_candidate_features,
    )
    return evaluate_jspace_ranker(
        adapter,
        data,  # type: ignore[arg-type]
        batch_size=batch_size,
        device=device,
        autocast=str(device).startswith("cuda"),
    )


def _selection(evaluation: Mapping[str, Any]) -> tuple[float, float]:
    rows = evaluation["horizon_metrics"][:4]
    native_k = int(evaluation.get("native_k", 8))
    metric = f"request_macro_recall_at_{native_k}"
    values = [float(row[metric]) for row in rows]
    if len(values) != 4:
        raise ValueError("checkpoint selection requires H1-H4 metrics")
    return float(np.mean(values)), float(np.min(values))


def _checkpoint_payload(
    *,
    model: JSpaceCandidateReranker,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    rng: np.random.Generator | None,
    model_config: JSpaceRerankerConfig,
    loss_config: JSpaceRerankerLossConfig,
    loss_profile: str,
    training_config: JSpaceTrainingConfig,
    input_provenance: Mapping[str, Any],
    completed_epoch: int,
    global_step: int,
    best_selection: tuple[float, float],
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
    validation_selection: tuple[float, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": JSPACE_TRAINING_SCHEMA,
        "model_config": model_config.to_dict(),
        "loss_profile": loss_profile,
        "loss_config": loss_config.to_dict(),
        "training_config": training_config.to_dict(),
        "input_provenance": dict(input_provenance),
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "global_step": global_step,
        "best_selection": best_selection,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
    }
    execution = input_provenance.get("execution_contract", {})
    payload["inference_contract"] = {
        "pool_horizons": int(execution.get("pool_horizons", 8)),
        "active_horizons": model_config.horizons,
        "include_feature_rms": bool(execution.get("include_feature_rms", False)),
        "include_candidate_features": bool(
            execution.get(
                "include_candidate_features", model_config.candidate_feature_width > 0
            )
        ),
        "inactive_horizons_are_exact_base_passthrough": True,
    }
    payload["decision_profile"] = execution.get("decision_profile", DECISION_PROFILE)
    payload["mtp_timing_causal"] = bool(execution.get("mtp_timing_causal", False))
    payload["acceptance_labels_used"] = False
    if validation_selection is not None:
        payload["validation_selection"] = validation_selection
    if optimizer is not None and scheduler is not None and rng is not None:
        payload.update(
            {
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "rng_state": _rng_state(rng),
            }
        )
    return payload


def _write_evaluation(
    output_dir: Path,
    evaluation: Mapping[str, Any],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    bootstrap = paired_request_bootstrap(
        list(evaluation["request_metrics"]),
        replicates=bootstrap_replicates,
        seed=seed,
        native_k=int(evaluation.get("native_k", 8)),
    )
    mappings = {
        "validation_horizon_metrics.csv": "horizon_metrics",
        "validation_request_metrics.csv": "request_metrics",
        "validation_layer_metrics.csv": "layer_metrics",
        "validation_domain_metrics.csv": "domain_metrics",
        "validation_position_metrics.csv": "position_metrics",
    }
    for filename, key in mappings.items():
        write_csv(output_dir / filename, list(evaluation[key]))
    bootstrap_path = output_dir / "validation_h1_h4_paired_bootstrap.json"
    bootstrap_path.write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    payload = {**dict(evaluation), "paired_h1_h4_bootstrap": bootstrap}
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bootstrap


def train_jspace_reranker(
    train_data: AlignedJCandidateData,
    validation_data: AlignedJCandidateData,
    output_dir: Path,
    router_keys: torch.Tensor,
    model_config: JSpaceRerankerConfig,
    loss_config: JSpaceRerankerLossConfig,
    training_config: JSpaceTrainingConfig,
    *,
    target_features: Path,
    target_feature_rms: Path | None,
    router_provenance: Mapping[str, Any],
    device: str = "cuda:0",
    resume: Path | None = None,
    stop_after_epoch: int | None = None,
    strict_lineage: bool = True,
    precision_audit: Path | None = None,
    allow_provisional_j_without_probe: bool = False,
    allow_router_rank_ablation: bool = False,
    decision_profile: str = DECISION_PROFILE,
    mtp_timing_causal: bool = False,
    ordered_group_prefetch: bool = False,
    loss_profile: str | None = None,
) -> dict[str, Any]:
    """Run validation-selected training, never touching a sealed test split."""

    training_config.validate()
    model_config.validate()
    loss_config.validate(model_config.horizons)
    resolved_loss_profile = validate_jspace_loss_profile(
        loss_config,
        model_config.horizons,
        loss_profile,
    )
    if decision_profile != DECISION_PROFILE:
        raise ValueError(f"decision_profile must be {DECISION_PROFILE!r}")
    if mtp_timing_causal is not False:
        raise ValueError("this corpus does not support timing-causal MTP attribution")
    if training_config.decision_profile != decision_profile:
        raise ValueError("training config and requested decision profile differ")
    if training_config.mtp_timing_causal is not False:
        raise ValueError("training config incorrectly marks MTP timing causal")
    if training_config.acceptance_labels_used is not False:
        raise ValueError("training config attempts to use acceptance labels")
    if (
        model_config.router_key_width != model_config.experts
        and not allow_router_rank_ablation
    ):
        raise ValueError(
            "scientific J-HARP training requires full router rank R=E unless the "
            "explicit ablation override is supplied"
        )
    if training_config.active_horizons != model_config.horizons:
        raise ValueError("training and model active horizon counts differ")
    if model_config.horizons not in (4, train_data.pool.horizons):
        raise ValueError("ranker must train H1-H4 or every pool horizon")
    pool_native_k = int(train_data.pool.manifest.get("native_k", 8))
    validation_native_k = int(validation_data.pool.manifest.get("native_k", 8))
    if pool_native_k != validation_native_k or model_config.native_k != pool_native_k:
        raise ValueError("model and train/validation pools disagree on native_k")
    if strict_lineage and pool_native_k != 8:
        raise ValueError("the current scientific target trace contract is native top-8")
    _assert_matching_data(train_data, validation_data)
    _read_pool_manifest(
        train_data.pool.root,
        "train",
        allow_legacy_level2_train=not strict_lineage,
        require_context=False,
    )
    _read_pool_manifest(
        validation_data.pool.root, "validation", require_context=False
    )
    if tuple(router_keys.shape) != (
        model_config.layers,
        model_config.experts,
        model_config.router_key_width,
    ):
        raise ValueError("router keys disagree with model configuration")
    if target_feature_rms is not None and train_data.target_feature_rms is None:
        raise ValueError("target feature RMS path was not opened by aligned data")
    include_feature_rms = target_feature_rms is not None
    include_candidate_features = model_config.candidate_feature_width > 0

    if strict_lineage:
        semantic_lineage = _validate_scientific_lineage(
            train_data,
            validation_data,
            target_features=target_features,
            target_feature_rms=target_feature_rms,
            precision_audit=precision_audit,
            allow_provisional_j_without_probe=allow_provisional_j_without_probe,
        )
    else:
        if precision_audit is not None or allow_provisional_j_without_probe:
            raise ValueError(
                "precision attribution cannot be asserted in non-strict mode"
            )
        semantic_lineage = {
            "strict": False,
            "reason": "explicit_non_scientific_fixture_or_debug_mode",
            "precision_attribution": {"required": False, "attribution": "unverified"},
        }
    semantic_lineage.update(
        {
            "decision_profile": decision_profile,
            "mtp_timing_causal": False,
            "acceptance_labels_used": False,
        }
    )

    coverage_gate = _coverage_gate(validation_data)
    if not coverage_gate["passes"]:
        raise RuntimeError(
            "candidate coverage gate failed: require mean H1-H4 >= 0.97 "
            "and H4 >= 0.95"
        )
    output_dir = Path(output_dir)
    if resume is None:
        if output_dir.exists():
            raise FileExistsError(f"refusing to reuse output directory {output_dir}")
        output_dir.mkdir(parents=True)
    else:
        if not output_dir.is_dir():
            raise FileNotFoundError("resume output directory does not exist")

    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    provenance = _input_provenance(
        train_data,
        validation_data,
        target_features=target_features,
        target_feature_rms=target_feature_rms,
        router_provenance=router_provenance,
        semantic_lineage=semantic_lineage,
    )
    provenance["execution_contract"] = {
        "pool_horizons": train_data.pool.horizons,
        "active_horizons": model_config.horizons,
        "include_feature_rms": include_feature_rms,
        "include_candidate_features": include_candidate_features,
        "router_rank_ablation": bool(
            model_config.router_key_width != model_config.experts
        ),
        "decision_profile": decision_profile,
        "mtp_timing_causal": False,
        "acceptance_labels_used": False,
    }
    if ordered_group_prefetch:
        provenance["execution_contract"]["io_pipeline"] = {
            "schema": "harp8_ordered_group_prefetch_v1",
            "opt_in": True,
            "rows_per_materialization": (
                training_config.microbatch_size * training_config.gradient_accumulation
            ),
            "cpu_read_ahead_groups": 1,
            "compact_model_only_transfer": True,
            "pinned_nonblocking_h2d_on_cuda": True,
        }
    provenance = provenance_with_loss_profile(
        provenance,
        loss_profile=resolved_loss_profile,
        loss_config=loss_config,
    )
    train_view = _RowPrefixView(train_data, training_config.max_train_rows)
    validation_view = _RowPrefixView(
        validation_data, training_config.max_validation_rows
    )
    model = JSpaceCandidateReranker(model_config, router_keys).to(device)
    fused = str(device).startswith("cuda")
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, training_config.weight_decay),
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
        eps=training_config.epsilon,
        fused=fused,
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

    # Function-preserving construction is checked on real aligned data before
    # optimizer state or an epoch-zero incumbent is admitted.
    assertion_rows = np.arange(
        min(validation_view.rows, training_config.evaluation_batch_size),
        dtype=np.int64,
    )
    assertion_batch = prepare_model_batch(
        validation_view.batch(
            assertion_rows,
            device,
            active_horizons=model_config.horizons,
            include_context=False,
        ),
        include_feature_rms=include_feature_rms,
        include_candidate_features=include_candidate_features,
        active_horizons=model_config.horizons,
    )
    model.eval()
    with torch.inference_mode(), _autocast(device):
        assertion_output = model(assertion_batch)
    if (
        not torch.equal(
            assertion_output.scores.float(), assertion_batch["candidate_scores"].float()
        )
        or int(torch.count_nonzero(assertion_output.delta)) != 0
    ):
        raise AssertionError("epoch-zero J-HARP scores are not exactly the base scores")
    epoch0_assertion = {
        "rows": len(assertion_rows),
        "exact_score_equality": True,
        "nonzero_delta_values": 0,
    }

    history: list[dict[str, Any]] = []
    completed_epoch = 0
    global_step = 0
    stale_epochs = 0
    epoch0 = _evaluate(
        model,
        validation_view,
        batch_size=training_config.evaluation_batch_size,
        device=device,
        include_feature_rms=include_feature_rms,
        include_candidate_features=include_candidate_features,
    )
    best_selection = _selection(epoch0)
    best_epoch = 0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    if resume is None:
        _atomic_torch_save(
            _checkpoint_payload(
                model=model,
                optimizer=None,
                scheduler=None,
                rng=None,
                model_config=model_config,
                loss_config=loss_config,
                loss_profile=resolved_loss_profile,
                training_config=training_config,
                input_provenance=provenance,
                completed_epoch=0,
                global_step=0,
                best_selection=best_selection,
                best_epoch=0,
                stale_epochs=0,
                history=[],
                validation_selection=best_selection,
            ),
            best_path,
        )
    else:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state.get("schema") != JSPACE_TRAINING_SCHEMA:
            raise ValueError("resume checkpoint has an incompatible schema")
        assert_resume_loss_profile(
            state,
            expected_profile=resolved_loss_profile,
            expected_config=loss_config,
            horizons=model_config.horizons,
        )
        saved_provenance = provenance_with_loss_profile(
            state.get("input_provenance", {}),
            loss_profile=resolved_loss_profile,
            loss_config=loss_config,
        )
        expected = {
            "model_config": model_config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "training_config": training_config.to_dict(),
            "input_provenance": provenance,
        }
        for name, value in expected.items():
            actual = saved_provenance if name == "input_provenance" else state.get(name)
            if actual != value:
                raise ValueError(f"resume checkpoint {name} differs")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        _restore_rng(state["rng_state"], rng)
        completed_epoch = int(state["completed_epoch"])
        global_step = int(state["global_step"])
        stale_epochs = int(state["stale_epochs"])
        best_selection = tuple(float(v) for v in state["best_selection"])
        best_epoch = int(state["best_epoch"])
        history = list(state["history"])

    for epoch in range(completed_epoch + 1, training_config.epochs + 1):
        model.train()
        order = rng.permutation(train_view.rows)
        running: defaultdict[str, float] = defaultdict(float)
        microbatches = 0
        optimizer_steps = 0
        group_rows = (
            training_config.microbatch_size * training_config.gradient_accumulation
        )
        row_groups = (
            order[group_start : group_start + group_rows]
            for group_start in range(0, train_view.rows, group_rows)
        )
        cuda_io = (
            ordered_group_prefetch
            and str(device).startswith("cuda")
            and torch.cuda.is_available()
        )
        if ordered_group_prefetch:
            materialized = ordered_prefetched_batches(
                train_view,
                row_groups,
                active_horizons=model_config.horizons,
                include_context=False,
                compact=True,
                prefetch=cuda_io,
                pin_memory=cuda_io,
            )
        else:
            materialized = ((group, None) for group in row_groups)
        for group, host_group_batch in materialized:
            group_batch = (
                move_tensor_batch(host_group_batch, device, non_blocking=cuda_io)
                if host_group_batch is not None
                else None
            )
            micro_starts = range(0, len(group), training_config.microbatch_size)
            microbatch_count = math.ceil(len(group) / training_config.microbatch_size)
            optimizer.zero_grad(set_to_none=True)
            for micro_start in micro_starts:
                micro_stop = min(
                    len(group), micro_start + training_config.microbatch_size
                )
                if group_batch is None:
                    raw_batch = train_view.batch(
                        group[micro_start:micro_stop],
                        device,
                        active_horizons=model_config.horizons,
                        include_context=False,
                    )
                else:
                    raw_batch = slice_tensor_batch(group_batch, micro_start, micro_stop)
                batch = prepare_model_batch(
                    raw_batch,
                    include_feature_rms=include_feature_rms,
                    include_candidate_features=include_candidate_features,
                    active_horizons=model_config.horizons,
                )
                with _autocast(device):
                    output = model(batch)
                    loss = jspace_reranker_loss(output, batch, loss_config)
                (loss.total / microbatch_count).backward()
                microbatches += 1
                for name, value in loss.metrics.items():
                    running[name] += value
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            optimizer_steps += 1
            running["gradient_norm"] += float(gradient_norm)

        validation = _evaluate(
            model,
            validation_view,
            batch_size=training_config.evaluation_batch_size,
            device=device,
            include_feature_rms=include_feature_rms,
            include_candidate_features=include_candidate_features,
        )
        selection = _selection(validation)
        record: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_gradient_norm": running["gradient_norm"] / max(1, optimizer_steps),
            "validation_mean_h1_h4_request_macro_recall_at_8": selection[0],
            "validation_min_h1_h4_request_macro_recall_at_8": selection[1],
            "validation_mean_h1_h4_base_recall_at_8": float(
                validation["mean_h1_h4_base_recall_at_8"]
            ),
            "validation_mean_h1_h4_coverage": float(
                validation[f"mean_h1_h4_coverage_at_{validation['candidate_count']}"]
            ),
            **{
                f"train_{name}": value / max(1, microbatches)
                for name, value in running.items()
                if name != "gradient_norm"
            },
        }
        validation_recall_name = (
            f"request_macro_recall_at_{int(validation.get('native_k', 8))}"
        )
        for row in validation["horizon_metrics"]:
            record[f"validation_recall_h{int(row['horizon'])}"] = float(
                row[validation_recall_name]
            )
        history.append(record)
        print(
            canonical_json({"event": "harp8_jspace_epoch", **record}),
            flush=True,
        )
        if selection > best_selection:
            best_selection = selection
            best_epoch = epoch
            stale_epochs = 0
            _atomic_torch_save(
                _checkpoint_payload(
                    model=model,
                    optimizer=None,
                    scheduler=None,
                    rng=None,
                    model_config=model_config,
                    loss_config=loss_config,
                    loss_profile=resolved_loss_profile,
                    training_config=training_config,
                    input_provenance=provenance,
                    completed_epoch=epoch,
                    global_step=global_step,
                    best_selection=best_selection,
                    best_epoch=best_epoch,
                    stale_epochs=stale_epochs,
                    history=history,
                    validation_selection=selection,
                ),
                best_path,
            )
        else:
            stale_epochs += 1
        completed_epoch = epoch
        _atomic_torch_save(
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                rng=rng,
                model_config=model_config,
                loss_config=loss_config,
                loss_profile=resolved_loss_profile,
                training_config=training_config,
                input_provenance=provenance,
                completed_epoch=completed_epoch,
                global_step=global_step,
                best_selection=best_selection,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                history=history,
            ),
            last_path,
        )
        write_csv(output_dir / "training_history.csv", history)
        (output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            status = {
                "schema": JSPACE_TRAINING_MANIFEST_SCHEMA,
                "status": "paused",
                "completed_epoch": completed_epoch,
                "next_epoch": completed_epoch + 1,
                "resume": str(last_path),
                "best_epoch": best_epoch,
                "best_selection": list(best_selection),
                "sealed_test_accessed": False,
                "loss_profile": resolved_loss_profile,
                "resolved_loss_config": loss_config.to_dict(),
                "decision_profile": decision_profile,
                "mtp_timing_causal": False,
                "acceptance_labels_used": False,
            }
            (output_dir / "run_status.json").write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return status
        if (
            epoch >= training_config.minimum_epochs
            and stale_epochs >= training_config.patience
        ):
            break

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    final_validation = _evaluate(
        model,
        validation_view,
        batch_size=training_config.evaluation_batch_size,
        device=device,
        include_feature_rms=include_feature_rms,
        include_candidate_features=include_candidate_features,
    )
    bootstrap = _write_evaluation(
        output_dir,
        final_validation,
        bootstrap_replicates=training_config.bootstrap_replicates,
        seed=training_config.seed,
    )
    history_path = output_dir / "training_history.csv"
    output_paths = [
        best_path,
        last_path,
        history_path,
        output_dir / "training_history.json",
        output_dir / "validation_horizon_metrics.csv",
        output_dir / "validation_request_metrics.csv",
        output_dir / "validation_layer_metrics.csv",
        output_dir / "validation_domain_metrics.csv",
        output_dir / "validation_position_metrics.csv",
        output_dir / "validation_metrics.json",
        output_dir / "validation_h1_h4_paired_bootstrap.json",
    ]
    manifest = {
        "schema": JSPACE_TRAINING_MANIFEST_SCHEMA,
        "model_name": "J-HARP-C64",
        "model_config": model_config.to_dict(),
        "loss_profile": resolved_loss_profile,
        "loss_config": loss_config.to_dict(),
        "training_config": training_config.to_dict(),
        "parameters_trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "router_keys_trainable": False,
        "feature_rms_included": include_feature_rms,
        "candidate_features_included": include_candidate_features,
        "candidate_coverage_gate": coverage_gate,
        "epoch0_base_score_assertion": epoch0_assertion,
        "best_epoch": best_epoch,
        "best_selection_order": [
            "validation request-macro mean H1-H4 Recall@8",
            "validation minimum individual H1-H4 Recall@8",
        ],
        "best_validation_mean_h1_h4_recall_at_8": float(
            final_validation["mean_h1_h4_recall_at_8"]
        ),
        "best_validation_min_h1_h4_recall_at_8": _selection(final_validation)[1],
        "paired_request_bootstrap": bootstrap,
        "input_provenance": provenance,
        "decision_profile": decision_profile,
        "mtp_timing_causal": False,
        "acceptance_labels_used": False,
        "precision_attribution": semantic_lineage["precision_attribution"],
        "router_rank_ablation": bool(
            model_config.router_key_width != model_config.experts
        ),
        "test_interface_exposed": False,
        "sealed_test_accessed": False,
        "target_mean_h1_h4_recall_at_8": 0.90,
        "composite_output_contract": {
            "pool_horizons": train_data.pool.horizons,
            "reranked_horizons": list(range(1, model_config.horizons + 1)),
            "frozen_harp_passthrough_horizons": list(
                range(model_config.horizons + 1, train_data.pool.horizons + 1)
            ),
            "passthrough_is_exact_base_score": True,
        },
        "outputs": {
            path.name: _hash_file(path) for path in output_paths if path.exists()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mtp-dir", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--target-feature-rms", type=Path)
    parser.add_argument(
        "--precision-audit",
        type=Path,
        help="accepted precision_audit.json; mandatory for strict J-Lens runs",
    )
    parser.add_argument(
        "--allow-provisional-j-without-probe",
        action="store_true",
        help="explicitly mark J attribution provisional when the representative audit has no recall probe",
    )
    parser.add_argument("--router-keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows-per-request", type=int, default=34)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument(
        "--active-horizons",
        type=int,
        default=4,
        help="rerank H1-H4 by default; use the pool horizon count for a full formal run",
    )
    parser.add_argument("--no-candidate-features", action="store_true")
    parser.add_argument(
        "--allow-router-rank-ablation",
        action="store_true",
        help="permit R<E router keys only for an explicitly labelled rank ablation",
    )
    parser.add_argument(
        "--decision-profile",
        choices=(DECISION_PROFILE,),
        default=DECISION_PROFILE,
        help="fixed informational upper-bound profile for the current MTP corpus",
    )
    parser.add_argument(
        "--loss-profile",
        choices=JSPACE_LOSS_PROFILE_NAMES,
        default=PREREGISTERED_V1,
        help="named immutable training objective",
    )

    parser.add_argument(
        "--ordered-group-prefetch",
        action="store_true",
        help="opt in to deterministic grouped memmap reads and one-group pinned read-ahead",
    )

    # Geometry is data-derived.  These flags alter only trainable capacity and
    # permit cheap CPU/remote pilots without weakening shape validation.
    parser.add_argument("--model-width", type=int, default=384)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--feedforward-width", type=int, default=1536)
    parser.add_argument("--expert-embedding-width", type=int, default=128)
    parser.add_argument("--temporal-blocks", type=int, default=2)
    parser.add_argument("--axial-blocks", type=int, default=4)
    parser.add_argument("--mtp-blocks", type=int, default=2)
    parser.add_argument("--set-blocks", type=int, default=2)
    parser.add_argument("--inducing-points", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Read split declarations before CandidatePool memory-maps any tensor.
    _read_pool_manifest(args.train_pool, "train", require_context=False)
    _read_pool_manifest(
        args.validation_pool, "validation", require_context=False
    )
    router_keys, router_provenance = _load_router_keys(args.router_keys)
    data_kwargs = {
        "capture_dir": args.capture_dir,
        "mtp_dir": args.mtp_dir,
        "target_features": args.target_features,
        "target_feature_rms": args.target_feature_rms,
        "rows_per_request": args.rows_per_request,
        "history": args.history,
    }
    train_data = AlignedJCandidateData(args.train_pool, **data_kwargs)
    validation_data = AlignedJCandidateData(args.validation_pool, **data_kwargs)
    include_candidate_features = not args.no_candidate_features
    model_config = derive_model_config(
        train_data,
        router_keys,
        include_feature_rms=args.target_feature_rms is not None,
        include_candidate_features=include_candidate_features,
        active_horizons=args.active_horizons,
        allow_router_rank_ablation=args.allow_router_rank_ablation,
        overrides={
            "model_width": args.model_width,
            "attention_heads": args.attention_heads,
            "feedforward_width": args.feedforward_width,
            "expert_embedding_width": args.expert_embedding_width,
            "temporal_blocks": args.temporal_blocks,
            "axial_blocks": args.axial_blocks,
            "mtp_blocks": args.mtp_blocks,
            "set_blocks": args.set_blocks,
            "inducing_points": args.inducing_points,
            "dropout": args.dropout,
        },
    )
    loss_defaults = resolve_jspace_loss_profile(
        args.loss_profile,
        model_config.horizons,
    )
    training_config = JSpaceTrainingConfig(
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
        bootstrap_replicates=args.bootstrap_replicates,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        active_horizons=args.active_horizons,
    )
    train_jspace_reranker(
        train_data,
        validation_data,
        args.output_dir,
        router_keys,
        model_config,
        loss_defaults,
        training_config,
        target_features=args.target_features,
        target_feature_rms=args.target_feature_rms,
        router_provenance=router_provenance,
        device=args.device,
        resume=args.resume,
        stop_after_epoch=args.stop_after_epoch,
        strict_lineage=True,
        precision_audit=args.precision_audit,
        allow_provisional_j_without_probe=args.allow_provisional_j_without_probe,
        allow_router_rank_ablation=args.allow_router_rank_ablation,
        decision_profile=args.decision_profile,
        mtp_timing_causal=False,
        ordered_group_prefetch=args.ordered_group_prefetch,
        loss_profile=args.loss_profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CompositeJSpaceInference",
    "DECISION_PROFILE",
    "JSPACE_TRAINING_MANIFEST_SCHEMA",
    "JSPACE_TRAINING_SCHEMA",
    "JSpaceTrainingConfig",
    "candidate_feature_width",
    "composite_horizon_scores",
    "derive_model_config",
    "load_composite_jspace_checkpoint",
    "prepare_model_batch",
    "slice_horizon_batch",
    "train_jspace_reranker",
]
