"""Request-aligned candidate-pool inputs for J-HARP reranking.

Candidate pools contain only the rows belonging to one split.  Dense residual,
J, route-history, and MTP arrays retain the original request-major capture
order.  This adapter joins them by immutable ``request_id`` and within-request
position instead of assuming that pool row number equals capture row number.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import torch

from .reranker import CandidatePool


JSPACE_ALIGNED_DATA_SCHEMA = "harp8_jspace_aligned_candidate_data_v1"

# Exact tensors consumed by JSpaceCandidateReranker and its loss.  The aligned
# loader creates richer intermediates to assemble candidate features, but they
# do not need to occupy pinned memory or cross PCIe during training.
JSPACE_MODEL_BATCH_KEYS = frozenset(
    {
        "candidate_scores",
        "candidate_ids",
        "target_membership",
        "teacher_candidate_scores",
        "valid_future",
        "candidate_mask",
        "j_states",
        "j_mask",
        "mtp_states",
        "mtp_router_logits",
        "mtp_mask",
        "mtp_metadata",
        "candidate_features",
    }
)


def compact_model_batch(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop batch-construction intermediates after their features are built."""

    missing = {
        "candidate_scores",
        "candidate_ids",
        "target_membership",
        "teacher_candidate_scores",
        "valid_future",
        "j_states",
        "j_mask",
        "mtp_states",
        "mtp_router_logits",
        "mtp_mask",
    } - batch.keys()
    if missing:
        raise KeyError(f"model batch is missing required tensors: {sorted(missing)}")
    return {
        name: value for name, value in batch.items()
        if name in JSPACE_MODEL_BATCH_KEYS
    }


def pin_tensor_batch(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Copy a CPU tensor batch into page-locked storage without value changes."""

    result: dict[str, torch.Tensor] = {}
    for name, value in batch.items():
        if value.device.type != "cpu":
            raise ValueError(f"cannot pin non-CPU batch tensor {name!r}")
        # ``expand`` produces zero-stride views whose elements overlap.  CUDA's
        # pinned-memory copy rejects those views as writable destinations, so
        # materialize only non-contiguous inputs before pinning.  Contiguous
        # batch tensors retain their original storage and incur no extra copy.
        source = value if value.is_contiguous() else value.contiguous()
        result[name] = source if source.is_pinned() else source.pin_memory()
    return result


def move_tensor_batch(
    batch: Mapping[str, torch.Tensor],
    device: str | torch.device,
    *,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    """Move a materialized batch while preserving every dtype and value."""

    target = torch.device(device)
    return {
        name: value.to(device=target, non_blocking=non_blocking)
        for name, value in batch.items()
    }


def slice_tensor_batch(
    batch: Mapping[str, torch.Tensor],
    start: int,
    stop: int,
) -> dict[str, torch.Tensor]:
    """Take an ordered row view from a uniformly batched tensor mapping."""

    if start < 0 or stop <= start:
        raise ValueError("batch slice must be a non-empty forward interval")
    first_dimensions = {int(value.shape[0]) for value in batch.values()}
    if len(first_dimensions) != 1:
        raise ValueError("every tensor must share the same leading batch dimension")
    rows = first_dimensions.pop()
    if stop > rows:
        raise IndexError("batch slice exceeds materialized rows")
    return {name: value[start:stop] for name, value in batch.items()}


def ordered_prefetched_batches(
    data: Any,
    row_batches: Iterable[np.ndarray],
    *,
    active_horizons: int | None,
    include_context: bool,
    compact: bool,
    prefetch: bool,
    pin_memory: bool,
) -> Iterator[tuple[np.ndarray, dict[str, torch.Tensor]]]:
    """Materialize batches in exact input order with one-batch read-ahead.

    The worker performs only deterministic, read-only CPU batch construction.
    It never touches a random generator.  Consequently prefetch changes when
    bytes are read, but not shuffled row order, tensor contents, microbatch
    boundaries, or the sequence of model/RNG operations in the training loop.
    """

    def materialize(rows: np.ndarray) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
        stable_rows = np.asarray(rows, dtype=np.int64).copy()
        values = data.batch(
            stable_rows,
            "cpu",
            active_horizons=active_horizons,
            include_context=include_context,
            compact=compact,
        )
        if pin_memory:
            values = pin_tensor_batch(values)
        return stable_rows, values

    iterator = iter(row_batches)
    try:
        first = next(iterator)
    except StopIteration:
        return
    if not prefetch:
        yield materialize(first)
        for rows in iterator:
            yield materialize(rows)
        return

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="jspace-io") as pool:
        pending: Future[tuple[np.ndarray, dict[str, torch.Tensor]]] = pool.submit(
            materialize, first
        )
        for rows in iterator:
            current = pending.result()
            pending = pool.submit(materialize, rows)
            yield current
        yield pending.result()


def _read_request_ids(path: Path) -> np.ndarray:
    values: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                values.append(int(record["request_id"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid request record at {path}:{line_number}") from exc
    if not values or len(values) != len(set(values)):
        raise ValueError("capture requests must have unique request IDs")
    return np.asarray(values, dtype=np.int64)


class AlignedJCandidateData:
    """Join a fixed candidate pool to causal target/J/MTP evidence."""

    def __init__(
        self,
        pool_root: Path,
        *,
        capture_dir: Path,
        mtp_dir: Path,
        target_features: Path,
        target_feature_rms: Path | None = None,
        rows_per_request: int = 34,
        history: int = 3,
    ) -> None:
        self.pool = CandidatePool(Path(pool_root))
        self.capture_dir = Path(capture_dir)
        self.mtp_dir = Path(mtp_dir)
        self.target_features_path = Path(target_features)
        self.target_feature_rms_path = (
            Path(target_feature_rms) if target_feature_rms is not None else None
        )
        self.rows_per_request = int(rows_per_request)
        self.history = int(history)
        if self.rows_per_request <= 1 or self.history <= 0:
            raise ValueError("rows_per_request and history must be positive")

        self.capture_request_ids = _read_request_ids(self.capture_dir / "requests.jsonl")
        id_to_number = {
            int(request_id): number
            for number, request_id in enumerate(self.capture_request_ids.tolist())
        }
        try:
            request_numbers = np.asarray(
                [id_to_number[int(request_id)] for request_id in self.pool.request_ids],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(f"candidate pool request {exc.args[0]} is absent from capture") from exc
        if (self.pool.within < 0).any() or (self.pool.within >= self.rows_per_request).any():
            raise ValueError("candidate pool has an out-of-range within-request position")
        self.capture_rows = request_numbers * self.rows_per_request + self.pool.within.astype(np.int64)

        self.target_features = np.load(
            self.target_features_path, mmap_mode="r", allow_pickle=False
        )
        self.target_feature_rms = (
            np.load(self.target_feature_rms_path, mmap_mode="r", allow_pickle=False)
            if self.target_feature_rms_path is not None
            else None
        )
        self.router_logits = np.load(
            self.capture_dir / "raw_router_logits.npy", mmap_mode="r", allow_pickle=False
        )
        self.top8 = np.load(
            self.capture_dir / "top8_expert_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.mtp_states = np.load(
            self.mtp_dir / "mtp_hidden_depths.npy", mmap_mode="r", allow_pickle=False
        )
        self.mtp_router_logits = np.load(
            self.mtp_dir / "mtp_router_logits_depths.npy", mmap_mode="r", allow_pickle=False
        )
        self._validate()

    @property
    def rows(self) -> int:
        return self.pool.rows

    @property
    def target_width(self) -> int:
        """Width presented to the J-context encoder, including optional RMS."""
        return self.projected_target_width + int(self.target_feature_rms is not None)

    @property
    def projected_target_width(self) -> int:
        """Width of transported/projected state before the RMS side channel."""
        return int(self.target_features.shape[-1])

    @property
    def mtp_width(self) -> int:
        return int(self.mtp_states.shape[-1])

    @property
    def mtp_depths(self) -> int:
        return int(self.mtp_states.shape[1])

    def _validate(self) -> None:
        expected_rows = len(self.capture_request_ids) * self.rows_per_request
        layers, experts = self.pool.layers, self.pool.experts
        if self.target_features.ndim != 3 or self.target_features.shape[:2] != (expected_rows, layers):
            raise ValueError("target features must be [capture_rows,layers,width]")
        if self.target_feature_rms is not None and self.target_feature_rms.shape not in (
            (expected_rows, layers), (expected_rows, layers, 1)
        ):
            raise ValueError("target feature RMS must be [capture_rows,layers] or [...,1]")
        if self.router_logits.shape != (expected_rows, layers, experts):
            raise ValueError("router logits disagree with candidate-pool geometry")
        if self.top8.shape != (expected_rows, layers, 8):
            raise ValueError("target top-8 IDs disagree with candidate-pool geometry")
        if self.mtp_states.ndim != 3 or self.mtp_states.shape[0] != expected_rows:
            raise ValueError("MTP states must be [capture_rows,depth,width]")
        if self.mtp_router_logits.shape != (
            expected_rows, self.mtp_states.shape[1], experts
        ):
            raise ValueError("MTP router logits disagree with MTP states or expert count")
        if not np.array_equal(
            self.capture_request_ids[self.capture_rows // self.rows_per_request],
            self.pool.request_ids,
        ):
            raise ValueError("candidate/capture request alignment failed")

    @staticmethod
    def _to_device(values: dict[str, np.ndarray], device: str | torch.device) -> dict[str, torch.Tensor]:
        return {name: torch.as_tensor(value, device=device) for name, value in values.items()}

    def batch(
        self,
        rows: np.ndarray,
        device: str | torch.device,
        *,
        active_horizons: int | None = None,
        include_context: bool = True,
        compact: bool = False,
    ) -> dict[str, torch.Tensor]:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or (rows < 0).any() or (rows >= self.rows).any():
            raise IndexError("candidate batch rows are out of range")
        capture_rows = self.capture_rows[rows]
        within = self.pool.within[rows]
        batch_size = len(rows)
        h = self.pool.horizons if active_horizons is None else int(active_horizons)
        if not 1 <= h <= self.pool.horizons:
            raise ValueError("active_horizons must lie within the candidate pool")
        layers, candidates = (
            self.pool.layers,
            self.pool.candidate_count,
        )

        target_history = np.zeros(
            (batch_size, self.history, layers, self.projected_target_width), dtype=np.float32
        )
        target_rms_history = np.zeros(
            (batch_size, self.history, layers, 1), dtype=np.float32
        )
        route_history = np.zeros(
            (batch_size, self.history, layers, self.pool.experts), dtype=np.float32
        )
        top8_history = np.zeros(
            (batch_size, self.history, layers, 8), dtype=np.int64
        )
        history_available = np.zeros((batch_size, self.history), dtype=np.bool_)
        for lag in range(self.history):
            valid = within >= lag
            if not valid.any():
                continue
            source_rows = capture_rows[valid] - lag
            target_history[valid, lag] = np.asarray(
                self.target_features[source_rows], dtype=np.float32
            )
            if self.target_feature_rms is not None:
                rms = np.asarray(self.target_feature_rms[source_rows], dtype=np.float32)
                target_rms_history[valid, lag] = rms.reshape(len(source_rows), layers, 1)
            logits = np.asarray(self.router_logits[source_rows], dtype=np.float32)
            logits -= logits.mean(axis=-1, keepdims=True)
            route_history[valid, lag] = logits
            top8_history[valid, lag] = np.asarray(self.top8[source_rows], dtype=np.int64)
            history_available[valid, lag] = True

        candidate_ids = np.asarray(
            self.pool.candidate_ids[rows, :h], dtype=np.int64
        )
        candidate_route_scores = np.empty(
            (batch_size, h, layers, candidates, self.history), dtype=np.float32
        )
        candidate_history_membership = np.empty(
            (batch_size, h, layers, candidates, self.history), dtype=np.float32
        )
        for lag in range(self.history):
            source = np.broadcast_to(
                route_history[:, lag, None, :, :],
                (batch_size, h, layers, self.pool.experts),
            )
            candidate_route_scores[..., lag] = np.take_along_axis(
                source, candidate_ids, axis=-1
            )
            selected = top8_history[:, lag, None, :, None, :]
            candidate_history_membership[..., lag] = (
                candidate_ids[..., None] == selected
            ).any(axis=-1)

        values: dict[str, np.ndarray] = {
            "target_history": target_history,
            "target_rms_history": target_rms_history,
            "history_available": history_available,
            "route_history": route_history,
            "candidate_route_scores": candidate_route_scores,
            "candidate_history_membership": candidate_history_membership,
            "mtp_states": np.asarray(self.mtp_states[capture_rows], dtype=np.float32),
            "mtp_router_logits": np.asarray(
                self.mtp_router_logits[capture_rows], dtype=np.float32
            ),
            "mtp_available": np.ones((batch_size, self.mtp_depths), dtype=np.bool_),
            "within_request": (within.astype(np.float32) / max(1, self.rows_per_request - 1)),
            "capture_row": capture_rows.astype(np.int64),
        }
        result = self.pool.batch(
            rows,
            str(device),
            active_horizons=h,
            include_context=include_context,
        )
        result.update(self._to_device(values, device))
        result["j_states"] = result["target_history"]
        if self.target_feature_rms is not None:
            result["j_states"] = torch.cat(
                [result["j_states"], result["target_rms_history"]], dim=-1
            )
        result["j_mask"] = result["history_available"].unsqueeze(-1).expand(
            -1, -1, layers
        )
        result["mtp_mask"] = result["mtp_available"]

        route_scores = result["candidate_route_scores"][..., :3]
        route_membership = result["candidate_history_membership"][..., :3]
        if self.history < 3:
            padding_shape = route_scores.shape[:-1] + (3 - self.history,)
            route_scores = torch.cat(
                [route_scores, route_scores.new_zeros(padding_shape)], dim=-1
            )
            route_membership = torch.cat(
                [route_membership, route_membership.new_zeros(padding_shape)], dim=-1
            )
        source_gates = result["source_gates"].unsqueeze(-2).expand(
            -1, -1, -1, candidates, -1
        )
        within_feature = result["within_request"].view(
            batch_size, 1, 1, 1, 1
        ).expand(-1, h, layers, candidates, -1)
        result["candidate_features"] = torch.cat(
            [
                result["current_scores"].unsqueeze(-1),
                result["current_rank"].unsqueeze(-1),
                result["copy_gates"].unsqueeze(-1),
                source_gates,
                route_scores,
                route_membership,
                within_feature,
            ],
            dim=-1,
        )
        if result["candidate_features"].shape[-1] != 13:
            raise RuntimeError("J-HARP candidate feature contract must be width 13")
        return compact_model_batch(result) if compact else result

    def sequential_batches(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.rows, batch_size):
            yield np.arange(start, min(self.rows, start + batch_size), dtype=np.int64)
