"""Leakage-safe aligned data for the full-router J-space forecaster.

The forecaster is trained from the same immutable capture and request split as
J-HARP-C64, but its target is the complete 256-expert router vector rather
than membership inside a fixed candidate set.  Frozen HARP candidate scores
are expanded into a causal full-namespace residual baseline; target routes at
future positions remain outside the model allowlist and are used only by the
loss and evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch

from .candidates import (
    POOL_SCHEMA,
    POOL_SCHEMA_V2,
    REQUEST_IDS_HASH_ENCODING,
    request_ids_sha256,
)
from .jspace_v2_data import AlignedJContextCandidateData


JSPACE_ROUTER_DATA_SCHEMA = "harp8_jspace_full_router_data_v1"
TARGET_STATE_STREAM_CONTRACT_SCHEMA = "harp8_full_router_target_state_streams_v1"

# The only tensors that may cross the model boundary.  In particular, neither
# future router labels nor committed future token identities are present.
JSPACE_ROUTER_CAUSAL_MODEL_KEYS = frozenset(
    {
        "base_router_scores",
        "route_history",
        "route_mask",
        "j_states",
        "j_mask",
        "generator_context",
        "mtp_states",
        "mtp_router_logits",
        "mtp_mask",
        "within_request",
    }
)

JSPACE_ROUTER_SECONDARY_CAUSAL_MODEL_KEYS = frozenset(
    {"secondary_target_states", "secondary_target_mask"}
)

# These trace fields are useful as supervision/audit metadata, but are never
# admissible even in the outer training batch for this v1 forecaster.  The
# exact sanctioned labels are teacher_router_scores, target_top8, and
# valid_future, all constructed by this loader.
FORBIDDEN_TRACE_INPUT_FIELDS = frozenset(
    {
        "accepted_at_depth",
        "accepted_through_depth",
        "acceptance_label",
        "actual_acceptance",
        "committed_future_token_ids",
        "first_rejection_depth",
        "future_cache_outcome",
        "future_committed_token_ids",
        "future_transfer_stall",
        "mtp_draft_target_logprobs",
        "mtp_draft_target_token_ids",
        "prefix_matches_committed",
        "target_acceptance",
    }
)

JSPACE_ROUTER_SUPERVISION_KEYS = frozenset(
    {"teacher_router_scores", "target_top8", "valid_future"}
)


def causal_router_forecaster_inputs(
    batch: Mapping[str, torch.Tensor],
    *,
    secondary_target_enabled: bool = False,
) -> dict[str, torch.Tensor]:
    """Return an exact causal allowlist and reject known leakage fields."""

    forbidden = sorted(FORBIDDEN_TRACE_INPUT_FIELDS.intersection(batch))
    if forbidden:
        raise ValueError(f"label-only trace fields cannot be inputs: {forbidden}")
    required = JSPACE_ROUTER_CAUSAL_MODEL_KEYS | (
        JSPACE_ROUTER_SECONDARY_CAUSAL_MODEL_KEYS
        if secondary_target_enabled
        else frozenset()
    )
    if not secondary_target_enabled:
        unexpected = sorted(
            JSPACE_ROUTER_SECONDARY_CAUSAL_MODEL_KEYS.intersection(batch)
        )
        if unexpected:
            raise ValueError(
                "secondary target-state inputs require the explicit "
                f"dual-stream contract: {unexpected}"
            )
    missing = required - batch.keys()
    if missing:
        raise KeyError(f"full-router causal inputs are missing: {sorted(missing)}")
    result = {name: batch[name] for name in required}
    if set(result) != required:
        raise AssertionError("full-router model allowlist is not exact")
    return result


def _valid_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _read_pool_manifest(
    root: Path,
    expected_split: str,
    *,
    allow_legacy_level2_train: bool = False,
    require_context: bool = True,
) -> dict[str, object]:
    """Reject sealed/test or unverifiable training pools before tensor access."""

    path = Path(root) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema not in {POOL_SCHEMA, POOL_SCHEMA_V2}:
        raise ValueError("candidate pool has an incompatible schema")
    if expected_split not in {"train", "validation"}:
        raise ValueError("full-router development data supports train/validation only")
    role = (
        manifest.get("level2_split")
        if schema == POOL_SCHEMA_V2
        else manifest.get("split")
    )
    if role != expected_split:
        raise ValueError(
            f"expected {expected_split!r} pool, found {role!r}"
        )
    if bool(manifest.get("allow_test", False)):
        raise PermissionError("sealed test pools are forbidden during model selection")
    if require_context and not bool(manifest.get("store_context", False)):
        raise ValueError("this model requires stored generator context")
    if schema == POOL_SCHEMA:
        if expected_split == "train" and not allow_legacy_level2_train:
            raise ValueError(
                "v1 candidate pools cannot be used for level-2 training without "
                "allow_legacy_level2_train=True; they do not prove OOF exclusion"
            )
        return manifest

    if manifest.get("split") != role or manifest.get("split_field_semantics") != (
        "deprecated_alias_of_level2_split"
    ):
        raise ValueError("v2 candidate pool has an inconsistent compatibility split")
    source_data_split = manifest.get("source_data_split")
    source_offline_split = manifest.get("source_offline_split")
    if source_data_split not in {"train", "validation"}:
        raise PermissionError("sealed/test source rows are forbidden during model selection")
    if source_offline_split not in {"train", "validation"}:
        raise PermissionError("sealed/test offline rows are forbidden during model selection")
    if not str(manifest.get("base_fold_id", "")).strip():
        raise ValueError("v2 candidate pool lacks a stable base_fold_id")
    if manifest.get("request_ids_hash_encoding") != REQUEST_IDS_HASH_ENCODING:
        raise ValueError("v2 candidate pool uses an unknown request-ID hash encoding")
    if not _valid_sha256(manifest.get("exported_request_ids_sha256")):
        raise ValueError("v2 candidate pool lacks an exported request-ID hash")
    if type(manifest.get("base_fit_excluded")) is not bool:
        raise ValueError("v2 candidate pool base_fit_excluded must be Boolean")
    base_hash = manifest.get("base_train_request_ids_sha256")
    if base_hash is not None and not _valid_sha256(base_hash):
        raise ValueError("v2 candidate pool has an invalid base-train request-ID hash")
    base_ids = manifest.get("base_train_request_ids")
    if base_ids is not None and not isinstance(base_ids, list):
        raise ValueError("v2 candidate pool base_train_request_ids must be a list")
    if expected_split == "train":
        if manifest.get("base_fit_excluded") is not True:
            raise ValueError("level-2 training requires verified base_fit_excluded=true")
        if base_hash is None or base_ids is None:
            raise ValueError(
                "level-2 training requires verifiable base-train request IDs"
            )
        if int(manifest.get("base_fit_overlap_count", -1)) != 0:
            raise ValueError("level-2 training pool reports base-fit request overlap")
    elif base_hash is None and base_ids is not None:
        raise ValueError("base-train request IDs cannot exist without their hash")
    return manifest


def validate_level2_pool_request_contract(
    manifest: Mapping[str, object],
    request_ids: np.ndarray | list[int],
    *,
    expected_split: str,
) -> None:
    """Recompute the v2 request-set proof from loaded pool metadata.

    Manifest-only preflight can validate the encoding and declared hashes, but
    it cannot prove that ``metadata.json`` contains the declared request set.
    Both the fixed-candidate rankers and the full-router forecaster use this
    second-stage check after memory-mapping the pool.
    """

    if manifest.get("schema") != POOL_SCHEMA_V2:
        return
    if expected_split not in {"train", "validation"}:
        raise ValueError("level-2 request validation supports train/validation only")
    actual_ids = [int(value) for value in np.asarray(request_ids).reshape(-1)]
    if request_ids_sha256(actual_ids) != manifest.get("exported_request_ids_sha256"):
        raise ValueError("candidate metadata disagrees with exported request-ID hash")

    base_ids_raw = manifest.get("base_train_request_ids")
    if base_ids_raw is None:
        if expected_split == "train":
            raise ValueError("level-2 training lacks base-train request IDs")
        return
    if not isinstance(base_ids_raw, list):
        raise ValueError("base-train request IDs must be a list")
    try:
        base_ids = tuple(int(value) for value in base_ids_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("base-train request IDs must be integers") from exc
    if len(base_ids) != len(set(base_ids)) or tuple(sorted(base_ids)) != base_ids:
        raise ValueError("base-train request IDs must be sorted and unique")
    if request_ids_sha256(base_ids) != manifest.get(
        "base_train_request_ids_sha256"
    ):
        raise ValueError("base-train request IDs disagree with their hash")
    overlap = sorted(set(base_ids).intersection(actual_ids))
    if manifest.get("base_fit_excluded") is True and overlap:
        raise ValueError(
            "candidate metadata overlaps the declared base-fit request set: "
            f"{overlap[:8]}"
        )


def assert_disjoint_level2_request_sets(
    train: "FullRouterForecastData",
    validation: "FullRouterForecastData",
) -> None:
    """Require request-level separation between downstream train/validation."""

    train_ids = set(int(value) for value in np.unique(train.pool.request_ids))
    validation_ids = set(
        int(value) for value in np.unique(validation.pool.request_ids)
    )
    overlap = sorted(train_ids.intersection(validation_ids))
    if overlap:
        raise ValueError(
            "level-2 train/validation request sets overlap: "
            f"{overlap[:8]}"
        )


class FullRouterForecastData:
    """Request-aligned causal inputs plus full future-router supervision.

    The candidate pool remains useful for its frozen HARP scores and latent
    generator context.  Candidate scores are expanded to all experts by
    assigning every absent expert a score below the weakest retained
    candidate.  This preserves the frozen HARP top-8 exactly at epoch zero
    while the learned full-router residual can promote any expert later.
    """

    def __init__(
        self,
        pool_root: Path,
        *,
        expected_split: str,
        capture_dir: Path,
        mtp_dir: Path,
        target_features: Path,
        target_feature_rms: Path | None = None,
        secondary_target_features: Path | None = None,
        secondary_target_feature_rms: Path | None = None,
        rows_per_request: int = 34,
        history: int = 3,
        base_floor_margin: float = 1.0,
        allow_legacy_level2_train: bool = False,
    ) -> None:
        self.split = str(expected_split)
        self.pool_manifest = _read_pool_manifest(
            Path(pool_root),
            self.split,
            allow_legacy_level2_train=allow_legacy_level2_train,
        )
        self.aligned = AlignedJContextCandidateData(
            Path(pool_root),
            capture_dir=Path(capture_dir),
            mtp_dir=Path(mtp_dir),
            target_features=Path(target_features),
            target_feature_rms=(
                Path(target_feature_rms) if target_feature_rms is not None else None
            ),
            rows_per_request=rows_per_request,
            history=history,
        )
        self.pool = self.aligned.pool
        if secondary_target_features is None and secondary_target_feature_rms is not None:
            raise ValueError("secondary target RMS requires secondary target features")
        self.secondary_target_features_path = (
            Path(secondary_target_features)
            if secondary_target_features is not None
            else None
        )
        self.secondary_target_feature_rms_path = (
            Path(secondary_target_feature_rms)
            if secondary_target_feature_rms is not None
            else None
        )
        self.secondary_target_features = (
            np.load(
                self.secondary_target_features_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            if self.secondary_target_features_path is not None
            else None
        )
        self.secondary_target_feature_rms = (
            np.load(
                self.secondary_target_feature_rms_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            if self.secondary_target_feature_rms_path is not None
            else None
        )
        self.rows_per_request = int(rows_per_request)
        self.history = int(history)
        self.base_floor_margin = float(base_floor_margin)
        if not np.isfinite(self.base_floor_margin) or self.base_floor_margin <= 0:
            raise ValueError("base_floor_margin must be finite and positive")
        if self.pool.horizons != 8:
            raise ValueError("full-router forecaster requires an H1-H8 pool")
        if int(self.pool.manifest.get("native_k", 8)) != 8:
            raise ValueError("full-router v1 requires native top-8")
        if self.pool.context is None:
            raise ValueError("candidate pool did not materialize generator context")
        self._validate_oof_request_contract()
        self._validate_secondary_target_stream()

    def _validate_oof_request_contract(self) -> None:
        validate_level2_pool_request_contract(
            self.pool_manifest,
            self.pool.request_ids,
            expected_split=self.split,
        )

    def _validate_secondary_target_stream(self) -> None:
        if self.secondary_target_features is None:
            return
        if (
            self.secondary_target_features_path.resolve()
            == self.aligned.target_features_path.resolve()
        ):
            raise ValueError("primary and secondary target feature paths must differ")
        expected_rows = int(self.aligned.target_features.shape[0])
        expected = (expected_rows, self.layers)
        if (
            self.secondary_target_features.ndim != 3
            or tuple(self.secondary_target_features.shape[:2]) != expected
        ):
            raise ValueError(
                "secondary target features must be [capture_rows,layers,width] "
                "and align with the primary stream"
            )
        if self.secondary_target_feature_rms is not None and tuple(
            self.secondary_target_feature_rms.shape
        ) not in (expected, expected + (1,)):
            raise ValueError(
                "secondary target feature RMS must align with capture rows/layers"
            )

    @property
    def rows(self) -> int:
        return int(self.pool.rows)

    @property
    def layers(self) -> int:
        return int(self.pool.layers)

    @property
    def experts(self) -> int:
        return int(self.pool.experts)

    @property
    def horizons(self) -> int:
        return int(self.pool.horizons)

    @property
    def target_width(self) -> int:
        return int(self.aligned.target_width)

    @property
    def secondary_target_enabled(self) -> bool:
        return self.secondary_target_features is not None

    @property
    def secondary_target_width(self) -> int | None:
        if self.secondary_target_features is None:
            return None
        return int(self.secondary_target_features.shape[-1]) + int(
            self.secondary_target_feature_rms is not None
        )

    @property
    def generator_context_width(self) -> int:
        return int(self.pool.model_width)

    @property
    def mtp_width(self) -> int:
        return int(self.aligned.mtp_width)

    @property
    def mtp_depths(self) -> int:
        return int(self.aligned.mtp_depths)

    def metadata(
        self, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.asarray(rows, dtype=np.int64)
        return (
            self.pool.request_ids[rows].copy(),
            self.pool.domains[rows].copy(),
            self.pool.within[rows].copy(),
        )

    @staticmethod
    def _require_finite(name: str, values: np.ndarray) -> None:
        """Reject corrupt numeric inputs before any device transfer."""

        if not np.isfinite(values).all():
            raise ValueError(f"{name} contain non-finite values")

    @staticmethod
    def _stable_candidate_topk(
        scores: np.ndarray, ids: np.ndarray, k: int
    ) -> np.ndarray:
        """Candidate score-descending, expert-ID-ascending stable top-k."""

        if scores.shape != ids.shape or not 1 <= k <= scores.shape[-1]:
            raise ValueError("candidate top-k geometry is invalid")
        by_id = np.argsort(ids, axis=-1, kind="stable")
        scores_by_id = np.take_along_axis(scores, by_id, axis=-1)
        ids_by_id = np.take_along_axis(ids, by_id, axis=-1)
        by_score = np.argsort(-scores_by_id, axis=-1, kind="stable")[..., :k]
        return np.take_along_axis(ids_by_id, by_score, axis=-1)

    def _full_harp_baseline(
        self, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expand one candidate read and prove its stable top-8 invariant."""

        if not np.isfinite(self.base_floor_margin) or self.base_floor_margin <= 0:
            raise ValueError("base_floor_margin must be finite and positive")
        scores = np.asarray(self.pool.candidate_scores[rows], dtype=np.float32)
        ids = np.asarray(self.pool.candidate_ids[rows], dtype=np.int64)
        self._require_finite("frozen HARP candidate scores", scores)
        if (ids < 0).any() or (ids >= self.experts).any():
            raise ValueError("frozen HARP candidate IDs lie outside expert namespace")
        ordered = np.sort(ids, axis=-1)
        if (np.diff(ordered, axis=-1) == 0).any():
            raise ValueError("frozen HARP candidate IDs must be unique per row")
        floor = scores.min(axis=-1, keepdims=True) - self.base_floor_margin
        if not np.isfinite(floor).all() or not np.all(floor < scores):
            raise ValueError("expanded HARP floor must be finite and below every candidate")
        full = np.broadcast_to(
            floor, scores.shape[:-1] + (self.experts,)
        ).copy()
        np.put_along_axis(full, ids, scores, axis=-1)
        # Because every absent score is strictly below every candidate, the
        # full-namespace top-8 is exactly this candidate-only stable top-8.
        expected_top8 = self._stable_candidate_topk(scores, ids, 8)
        return full, expected_top8

    def batch(
        self,
        rows: np.ndarray,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or (rows < 0).any() or (rows >= self.rows).any():
            raise IndexError("full-router batch rows are out of range")
        capture_rows = self.aligned.capture_rows[rows]
        within = self.pool.within[rows].astype(np.int64)
        batch_size = len(rows)

        j_history = np.zeros(
            (batch_size, self.history, self.layers, self.target_width),
            dtype=np.float32,
        )
        route_history = np.zeros(
            (batch_size, self.history, self.layers, self.experts),
            dtype=np.float32,
        )
        history_mask = np.zeros(
            (batch_size, self.history, self.layers), dtype=np.bool_
        )
        secondary_history = (
            np.zeros(
                (
                    batch_size,
                    self.history,
                    self.layers,
                    int(self.secondary_target_width),
                ),
                dtype=np.float32,
            )
            if self.secondary_target_enabled
            else None
        )
        for lag in range(self.history):
            valid = within >= lag
            if not valid.any():
                continue
            source = capture_rows[valid] - lag
            features = np.asarray(
                self.aligned.target_features[source], dtype=np.float32
            )
            if self.aligned.target_feature_rms is not None:
                rms = np.asarray(
                    self.aligned.target_feature_rms[source], dtype=np.float32
                ).reshape(len(source), self.layers, 1)
                features = np.concatenate([features, rms], axis=-1)
            self._require_finite("causal J-state history", features)
            j_history[valid, lag] = features
            if secondary_history is not None:
                secondary = np.asarray(
                    self.secondary_target_features[source], dtype=np.float32
                )
                if self.secondary_target_feature_rms is not None:
                    secondary_rms = np.asarray(
                        self.secondary_target_feature_rms[source], dtype=np.float32
                    ).reshape(len(source), self.layers, 1)
                    secondary = np.concatenate([secondary, secondary_rms], axis=-1)
                self._require_finite(
                    "causal secondary target-state history", secondary
                )
                secondary_history[valid, lag] = secondary
            logits = np.asarray(self.aligned.router_logits[source], dtype=np.float32)
            logits -= logits.mean(axis=-1, keepdims=True)
            self._require_finite("causal target-router history", logits)
            route_history[valid, lag] = logits
            history_mask[valid, lag] = True

        offsets = np.arange(1, self.horizons + 1, dtype=np.int64)[None]
        derived_valid = offsets <= (self.rows_per_request - 1 - within[:, None])
        raw_pool_valid = np.asarray(self.pool.valid_future[rows], dtype=np.uint8)
        if not np.isin(raw_pool_valid, (0, 1)).all():
            raise ValueError("candidate pool future mask must contain only zero or one")
        pool_valid = raw_pool_valid.astype(np.bool_)
        if np.any(pool_valid & ~derived_valid):
            raise ValueError("candidate pool future mask crosses a request boundary")
        valid_future = pool_valid & derived_valid
        future = capture_rows[:, None] + offsets
        request_last = (
            (capture_rows // self.rows_per_request)[:, None] * self.rows_per_request
            + self.rows_per_request
            - 1
        )
        safe_future = np.minimum(future, request_last)
        teacher = np.asarray(
            self.aligned.router_logits[safe_future], dtype=np.float32
        )
        target_top8 = np.asarray(
            self.aligned.top8[safe_future], dtype=np.int64
        )
        self._require_finite("future router supervision", teacher)
        if (target_top8 < 0).any() or (target_top8 >= self.experts).any():
            raise ValueError("authoritative target top-8 IDs lie outside expert namespace")
        ordered_target_top8 = np.sort(target_top8, axis=-1)
        if (np.diff(ordered_target_top8, axis=-1) == 0).any():
            raise ValueError("authoritative target top-8 IDs must be unique per row")

        mtp_states = np.asarray(
            self.aligned.mtp_states[capture_rows], dtype=np.float32
        )
        mtp_router = np.asarray(
            self.aligned.mtp_router_logits[capture_rows], dtype=np.float32
        )
        mtp_mask = np.ones((batch_size, self.mtp_depths), dtype=np.bool_)
        context = np.asarray(self.pool.context[rows], dtype=np.float32)
        # J/secondary/route slices were checked immediately after their only
        # read/transformation above; zero-filled unavailable cells are finite.
        self._require_finite("causal MTP hidden states", mtp_states)
        self._require_finite("causal MTP router logits", mtp_router)
        self._require_finite("causal frozen generator context", context)
        base, _expected_base_top8 = self._full_harp_baseline(rows)

        tensor = torch.as_tensor
        result = {
            "base_router_scores": tensor(base, device=device),
            "route_history": tensor(route_history, device=device),
            "route_mask": tensor(history_mask, device=device),
            "j_states": tensor(j_history, device=device),
            "j_mask": tensor(history_mask, device=device),
            "generator_context": tensor(context, device=device),
            "mtp_states": tensor(mtp_states, device=device),
            "mtp_router_logits": tensor(mtp_router, device=device),
            "mtp_mask": tensor(mtp_mask, device=device),
            "within_request": tensor(
                within.astype(np.float32) / max(1, self.rows_per_request - 1),
                device=device,
            ),
            "teacher_router_scores": tensor(teacher, device=device),
            "target_top8": tensor(target_top8, device=device),
            "valid_future": tensor(valid_future, device=device),
        }
        if secondary_history is not None:
            result["secondary_target_states"] = tensor(
                secondary_history, device=device
            )
            result["secondary_target_mask"] = tensor(
                history_mask.copy(), device=device
            )
        # Exercise the exact allowlist at materialization time as well as in
        # model.forward.  Supervision remains in the outer result only.
        causal_router_forecaster_inputs(
            result, secondary_target_enabled=self.secondary_target_enabled
        )
        return result

    def sequential_batches(self, batch_size: int) -> Iterator[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.rows, batch_size):
            yield np.arange(start, min(self.rows, start + batch_size), dtype=np.int64)


__all__ = [
    "FORBIDDEN_TRACE_INPUT_FIELDS",
    "FullRouterForecastData",
    "JSPACE_ROUTER_CAUSAL_MODEL_KEYS",
    "JSPACE_ROUTER_SECONDARY_CAUSAL_MODEL_KEYS",
    "JSPACE_ROUTER_DATA_SCHEMA",
    "JSPACE_ROUTER_SUPERVISION_KEYS",
    "TARGET_STATE_STREAM_CONTRACT_SCHEMA",
    "causal_router_forecaster_inputs",
]
