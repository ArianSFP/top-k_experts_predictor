"""Fail-closed adapters from audited traces to HARP event examples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from .config import HARPConfig


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


@dataclass(frozen=True)
class SplitSummary:
    requests: int
    source_rows: int


class CompactHARPData:
    """Compatibility adapter for the 2,048-request BF16 development trace.

    It never exposes the frozen test split unless ``allow_test=True`` is passed
    at the exact access call. This keeps development commands from opening test
    accidentally.
    """

    def __init__(
        self,
        capture_dir: Path,
        mtp_dir: Path,
        target_state_features: Path | np.ndarray,
        mtp_state_features: Path | np.ndarray,
        config: HARPConfig,
        *,
        rows_per_request: int = 34,
        split_manifest: Path | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.capture_dir = Path(capture_dir)
        self.mtp_dir = Path(mtp_dir)
        self.rows_per_request = int(rows_per_request)
        if self.rows_per_request <= config.horizons:
            raise ValueError("each request must contain more rows than forecast horizons")
        self.router_scores = np.load(
            self.capture_dir / "raw_router_logits.npy", mmap_mode="r"
        )
        self.top8 = np.load(
            self.capture_dir / "top8_expert_ids.npy", mmap_mode="r"
        )
        self.mtp_router_scores = np.load(
            self.mtp_dir / "mtp_router_logits_depths.npy", mmap_mode="r"
        )
        self.target_states = (
            np.load(target_state_features, mmap_mode="r")
            if isinstance(target_state_features, (str, Path))
            else target_state_features
        )
        self.mtp_states = (
            np.load(mtp_state_features, mmap_mode="r")
            if isinstance(mtp_state_features, (str, Path))
            else mtp_state_features
        )
        self.requests = _read_jsonl(self.capture_dir / "requests.jsonl")
        self._validate_geometry()
        self.request_ids = np.asarray(
            [int(record["request_id"]) for record in self.requests], dtype=np.int64
        )
        offline_splits = np.asarray(
            [str(record["offline_split"]) for record in self.requests]
        )
        self.split_manifest_path = (
            Path(split_manifest) if split_manifest is not None else None
        )
        self.request_splits = self._resolve_splits(offline_splits)
        self.request_domains = np.asarray(
            [str(record.get("domain", "unknown")) for record in self.requests],
            dtype=object,
        )
        self._split_indices: dict[str, np.ndarray] = {}
        for split in sorted(set(self.request_splits) - {"excluded"}):
            request_numbers = np.flatnonzero(self.request_splits == split)
            # The final row has no t+1 label and is never a source example.
            indices = [
                request * self.rows_per_request + within
                for request in request_numbers
                for within in range(self.rows_per_request - 1)
            ]
            self._split_indices[split] = np.asarray(indices, dtype=np.int64)

    def _resolve_splits(self, offline_splits: np.ndarray) -> np.ndarray:
        if self.split_manifest_path is None:
            return offline_splits
        manifest = json.loads(
            self.split_manifest_path.read_text(encoding="utf-8")
        )
        if manifest.get("schema") != "harp8_inner_split_v1":
            raise ValueError("inner split manifest has an incompatible schema")
        assignments = manifest.get("assignments")
        if not isinstance(assignments, dict):
            raise ValueError("inner split assignments must be a mapping")
        resolved = np.full(len(self.requests), "excluded", dtype="<U10")
        seen: set[int] = set()
        id_to_row = {
            int(request_id): row for row, request_id in enumerate(self.request_ids)
        }
        for raw_request_id, raw_split in assignments.items():
            request_id = int(raw_request_id)
            split = str(raw_split)
            if split not in ("train", "validation"):
                raise ValueError("inner assignment must be train or validation")
            if request_id not in id_to_row or request_id in seen:
                raise ValueError("inner split contains an unknown or duplicate request")
            row = id_to_row[request_id]
            if str(offline_splits[row]) != "train":
                raise ValueError("inner split may contain only original training requests")
            resolved[row] = split
            seen.add(request_id)
        expected = set(self.request_ids[offline_splits == "train"].tolist())
        if seen != expected:
            raise ValueError("inner split does not partition every training request")
        return resolved

    def _validate_geometry(self) -> None:
        config = self.config
        expected_rows = len(self.requests) * self.rows_per_request
        if self.router_scores.shape != (
            expected_rows,
            config.layers,
            config.experts,
        ):
            raise ValueError(
                f"router trace geometry {self.router_scores.shape} is incompatible"
            )
        if self.top8.shape != (expected_rows, config.layers, 8):
            raise ValueError("top-8 trace geometry is incompatible")
        if self.target_states.shape != (
            expected_rows,
            config.layers,
            config.target_state_width,
        ):
            raise ValueError(
                f"target-state feature geometry {self.target_states.shape} is incompatible"
            )
        if self.mtp_states.ndim != 3 or self.mtp_states.shape[0] != expected_rows:
            raise ValueError("MTP state features must be [rows,depth,width]")
        if self.mtp_states.shape[2] < config.mtp_state_width:
            raise ValueError("MTP state feature width is narrower than the model input")
        if self.mtp_router_scores.shape[:2] != self.mtp_states.shape[:2]:
            raise ValueError("MTP state/router node alignment differs")
        if self.mtp_router_scores.shape[2] != config.experts:
            raise ValueError("MTP router expert namespace disagrees")
        if self.mtp_states.shape[1] > config.mtp_depths:
            raise ValueError("capture contains more MTP depths than configured")
        if config.target_state_channels != 1 or config.mtp_state_channels != 1:
            raise ValueError(
                "compact compatibility adapter requires one target and MTP state channel"
            )

    @property
    def captured_mtp_depths(self) -> int:
        return int(self.mtp_states.shape[1])

    def split_summary(self) -> dict[str, SplitSummary]:
        return {
            split: SplitSummary(
                requests=int(np.count_nonzero(self.request_splits == split)),
                source_rows=int(len(indices)),
            )
            for split, indices in self._split_indices.items()
        }

    def indices(self, split: str, *, allow_test: bool = False) -> np.ndarray:
        if split == "test" and not allow_test:
            raise PermissionError(
                "sealed test access requires allow_test=True after model selection"
            )
        if split not in self._split_indices:
            raise KeyError(f"unknown split {split!r}")
        return self._split_indices[split].copy()

    def metadata(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        request_number = indices // self.rows_per_request
        within = indices % self.rows_per_request
        return (
            self.request_ids[request_number],
            self.request_domains[request_number],
            within,
        )

    def batch(self, indices: np.ndarray, device: str | torch.device) -> dict[str, torch.Tensor]:
        config = self.config
        indices = np.asarray(indices, dtype=np.int64)
        within = indices % self.rows_per_request
        batch_size = len(indices)

        route_history = np.zeros(
            (
                batch_size,
                config.layers,
                config.route_history,
                config.experts,
            ),
            dtype=np.float32,
        )
        route_available = np.zeros(
            (batch_size, config.route_history), dtype=np.bool_
        )
        for lag in range(config.route_history):
            valid = within >= lag
            if valid.any():
                value = np.asarray(
                    self.router_scores[indices[valid] - lag], dtype=np.float32
                )
                value -= value.mean(axis=-1, keepdims=True)
                route_history[valid, :, lag] = value
                route_available[valid, lag] = True

        target_states = np.asarray(self.target_states[indices], dtype=np.float32)
        target_states = target_states[:, :, None, :]
        target_state_available = np.ones(
            (batch_size, config.layers, 1), dtype=np.bool_
        )

        nodes = config.mtp_depths
        mtp_states = np.zeros(
            (batch_size, nodes, 1, config.mtp_state_width), dtype=np.float32
        )
        mtp_router = np.zeros(
            (batch_size, nodes, config.experts), dtype=np.float32
        )
        captured = self.captured_mtp_depths
        mtp_states[:, :captured, 0] = np.asarray(
            self.mtp_states[indices, :, : config.mtp_state_width],
            dtype=np.float32,
        )
        mtp_router[:, :captured] = np.asarray(
            self.mtp_router_scores[indices], dtype=np.float32
        )
        mtp_available = np.zeros((batch_size, nodes), dtype=np.bool_)
        mtp_available[:, :captured] = True
        mtp_depth_ids = np.broadcast_to(
            np.arange(1, nodes + 1, dtype=np.int64), (batch_size, nodes)
        ).copy()
        depth_fraction = mtp_depth_ids.astype(np.float32) / float(nodes)
        mtp_metadata = np.stack(
            [
                depth_fraction,
                np.square(depth_fraction),
                np.ones_like(depth_fraction),
                mtp_available.astype(np.float32),
            ],
            axis=-1,
        )

        future = np.arange(1, config.horizons + 1, dtype=np.int64)[None]
        valid_future = future <= (
            self.rows_per_request - 1 - within[:, None]
        )
        future_indices = indices[:, None] + future
        last = (indices // self.rows_per_request)[:, None] * self.rows_per_request
        last = last + self.rows_per_request - 1
        safe_future_indices = np.minimum(future_indices, last)
        teacher = np.asarray(
            self.router_scores[safe_future_indices], dtype=np.float32
        )
        target_top8 = np.asarray(self.top8[safe_future_indices], dtype=np.int64)
        latent_target = np.asarray(
            self.target_states[safe_future_indices, config.layers - 1],
            dtype=np.float32,
        )

        tensor = torch.as_tensor
        return {
            "route_history": tensor(route_history, device=device),
            "route_available": tensor(route_available, device=device),
            "target_states": tensor(target_states, device=device),
            "target_state_available": tensor(
                target_state_available, device=device
            ),
            "mtp_states": tensor(mtp_states, device=device),
            "mtp_router_logits": tensor(mtp_router, device=device),
            "mtp_metadata": tensor(mtp_metadata, device=device),
            "mtp_depth_ids": tensor(mtp_depth_ids, device=device),
            "mtp_available": tensor(mtp_available, device=device),
            "teacher_router_scores": tensor(teacher, device=device),
            "target_top8": tensor(target_top8, device=device),
            "future_latent_target": tensor(latent_target, device=device),
            "valid_future": tensor(valid_future, device=device),
            "within_request": tensor(within, dtype=torch.float32, device=device),
        }

    def shuffled_batches(
        self,
        split: str,
        batch_size: int,
        rng: np.random.Generator,
        *,
        allow_test: bool = False,
    ) -> Iterator[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self.indices(split, allow_test=allow_test)
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield indices[start : start + batch_size]

    def sequential_batches(
        self,
        split: str,
        batch_size: int,
        *,
        allow_test: bool = False,
    ) -> Iterable[np.ndarray]:
        indices = self.indices(split, allow_test=allow_test)
        for start in range(0, len(indices), batch_size):
            yield indices[start : start + batch_size]
