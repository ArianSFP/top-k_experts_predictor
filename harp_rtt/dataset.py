"""Leak-safe memory-mapped dataset for HARP-RTT rich captures.

The loader consumes indices produced by :mod:`harp_rtt.index`.  Tensor bytes
remain in the immutable capture sidecars, while one item represents one
committed source position with eight causal route-history tokens, four future
target labels, the exact already-selected H1 token, and a fixed-size native
MTP tree.  Future target tensors and branch acceptance are returned only in
the ``targets`` mapping; they are never mixed into the causal ``inputs``
mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .index import (
    ANCHOR_SPINE_META_COLUMNS,
    ANCHOR_SPINE_REQUIRED_DEPTH,
    ANCHOR_SPINE_ROLES,
    ANCHOR_SPINE_SCALAR_COLUMNS,
    MTP_META_COLUMNS,
    MTP_SCALAR_COLUMNS,
)


NP_DTYPES: Mapping[str, np.dtype[Any]] = {
    "bf16": np.dtype("<u2"),
    "bfloat16": np.dtype("<u2"),
    "float16": np.dtype("<f2"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
    "int16": np.dtype("<i2"),
    "int32": np.dtype("<i4"),
    "int64": np.dtype("<i8"),
    "uint8": np.dtype("u1"),
    "bool": np.dtype("u1"),
}

HISTORY_LENGTH = 8
HORIZONS = 4
TARGET_LAYERS = 40
EXPERTS = 256
TOP_K = 8
VOCAB_CANDIDATES = 64

CURRENT_ROLES = (
    "post_attention_residual_u",
    "normalized_target_router_input_a",
    "post_moe_residual_xplus",
    "routed_expert_output_delta_r",
    "shared_expert_output_delta_s",
    "raw_target_router_logits",
    "selected_expert_ids",
    "selected_execution_weights",
)
OPTIONAL_CURRENT_ROLES = (
    "block_input_residual_x",
    "complete_moe_delta",
    "full_target_router_probabilities",
)
FUTURE_ROLES = (
    "raw_target_router_logits",
    "normalized_target_router_input_a",
    "selected_expert_ids",
    "selected_execution_weights",
)
TREE_STATE_ROLES = (
    "mtp_fused_state",
    "mtp_post_ffn_hidden",
    "mtp_router_input",
    "mtp_vocabulary_head_input",
)
TREE_ROLES = TREE_STATE_ROLES + (
    "raw_mtp_router_logits",
    "mtp_selected_expert_ids",
    "mtp_selected_execution_weights",
    "vocab_top64_token_ids",
    "vocab_top64_log_probabilities",
)


def numpy_to_torch(values: np.ndarray[Any, Any], dtype_name: str) -> Tensor:
    """Copy a NumPy payload into a writable tensor, preserving BF16 bits."""

    contiguous = np.ascontiguousarray(values)
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.from_numpy(contiguous.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(contiguous)


def _zeros(shape: Sequence[int], dtype_name: str) -> Tensor:
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.zeros(tuple(shape), dtype=torch.bfloat16)
    return torch.zeros(tuple(shape), dtype=torch.from_numpy(np.empty((), NP_DTYPES[dtype_name])).dtype)


def _structural_first_divergence_depths(
    horizons: Tensor,
    child_ranks: Tensor,
    parents: Tensor,
) -> Tensor:
    """Derive branch divergence below the observed exact-H1 root.

    Adaptive capture fixes H1 to the target's already committed token.  Its
    rank under an earlier MTP distribution is descriptive evidence only; it
    is not a branch choice in the H2--H4 causal tree.
    """

    if any(value.ndim != 1 for value in (horizons, child_ranks, parents)):
        raise ValueError("tree divergence inputs must be vectors")
    if not (horizons.shape == child_ranks.shape == parents.shape):
        raise ValueError("tree divergence inputs must share geometry")
    divergence = torch.zeros_like(horizons, dtype=torch.int64)
    for local in range(int(horizons.numel())):
        current = local
        first = 0
        while current >= 0:
            horizon = int(horizons[current])
            if horizon > 1 and int(child_ranks[current]) > 0:
                first = horizon
            current = int(parents[current])
        divergence[local] = first
    return divergence


@dataclass(frozen=True)
class RichTokenRecord:
    """Resolved source, history, future, and tree rows for one sample."""

    segment: int
    sequence: int
    position: int
    current_row: int
    history_kinds: tuple[int, ...]
    history_rows: tuple[int, ...]
    future_rows: tuple[int, ...]
    tree_rows: tuple[int, ...]
    anchor_spine_rows: tuple[int, ...]


class RichSegmentIndex:
    """One memory-mapped rich index segment and its capture sidecars."""

    def __init__(self, root: Path, corpus_root: Path | None = None) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "index_manifest.json").read_text())
        if self.manifest.get("schema") != "harp_rtt_rich_event_index_v1":
            raise ValueError(f"unsupported HARP-RTT index schema in {self.root}")
        if corpus_root is None:
            self.segment_root = Path(self.manifest["source_segment"])
        else:
            candidate = Path(corpus_root) / "segments" / str(self.manifest["segment"])
            self.segment_root = candidate if candidate.is_dir() else Path(corpus_root) / str(
                self.manifest["segment"]
            )
        self.sequences: list[dict[str, Any]] = self.manifest["sequences"]
        self.descriptors: dict[str, dict[str, Any]] = self.manifest["tensor_descriptors"]
        self.adaptive_mtp_tree = bool(self.manifest.get("adaptive_mtp_tree", False))
        self.anchor_spine_contract = bool(
            self.manifest.get("anchor_spine_contract", False)
        )
        if self.adaptive_mtp_tree and not self.anchor_spine_contract:
            raise ValueError(
                f"adaptive index lacks the required anchor-spine contract in {self.root}"
            )
        self.roles = {
            "target": tuple(self.manifest["target_roles"]),
            "prompt": tuple(self.manifest["prompt_roles"]),
            "mtp": tuple(self.manifest["mtp_roles"]),
        }
        if self.anchor_spine_contract:
            if (
                self.manifest.get("anchor_spine_schema")
                != "harp_rtt_legacy_anchor_spine_v1"
                or int(self.manifest.get("anchor_spine_required_depth", -1))
                != ANCHOR_SPINE_REQUIRED_DEPTH
                or tuple(self.manifest.get("anchor_spine_roles", ()))
                != ANCHOR_SPINE_ROLES
            ):
                raise ValueError("adaptive anchor-spine manifest contract is incoherent")
            self.roles["anchor"] = tuple(self.manifest["anchor_spine_roles"])
        self.role_columns = {
            kind: {role: column for column, role in enumerate(roles)}
            for kind, roles in self.roles.items()
        }
        self.target_meta = np.load(self.root / "target_meta.npy", mmap_mode="r")
        self.target_offsets = np.load(self.root / "target_offsets.npy", mmap_mode="r")
        self.target_available = np.load(self.root / "target_available.npy", mmap_mode="r")
        self.target_scalars = np.load(self.root / "target_scalars.npy", mmap_mode="r")
        target_prefix_path = self.root / "target_prefix_sha256.npy"
        if not target_prefix_path.is_file():
            raise ValueError(f"rich index lacks target prefix hashes in {self.root}")
        self.target_prefix = np.load(target_prefix_path, mmap_mode="r")
        self.prompt_meta = np.load(self.root / "prompt_meta.npy", mmap_mode="r")
        self.prompt_offsets = np.load(self.root / "prompt_offsets.npy", mmap_mode="r")
        self.mtp_meta = np.load(self.root / "mtp_meta.npy", mmap_mode="r")
        self.mtp_offsets = np.load(self.root / "mtp_offsets.npy", mmap_mode="r")
        self.mtp_available = np.load(self.root / "mtp_available.npy", mmap_mode="r")
        self.mtp_scalars = np.load(self.root / "mtp_scalars.npy", mmap_mode="r")
        self.mtp_acceptance = np.load(
            self.root / "mtp_acceptance_labels.npy", mmap_mode="r"
        )
        self.mtp_prefix = np.load(self.root / "mtp_prefix_sha256.npy", mmap_mode="r")
        exact_prefix_path = self.root / "mtp_exact_prefix_sha256.npy"
        self.mtp_exact_prefix = np.load(
            exact_prefix_path if exact_prefix_path.is_file() else self.root / "mtp_prefix_sha256.npy",
            mmap_mode="r",
        )
        ready_timestamp_path = self.root / "mtp_source_ready_timestamp_utc.npy"
        self.mtp_source_ready_timestamp_utc = (
            np.load(ready_timestamp_path, mmap_mode="r")
            if ready_timestamp_path.is_file()
            else np.asarray([""] * len(self.mtp_meta), dtype="<U1")
        )
        self.mtp_meta_columns = tuple(
            self.manifest.get("mtp_meta_columns", MTP_META_COLUMNS[:13])
        )
        self.mtp_scalar_columns = tuple(
            self.manifest.get("mtp_scalar_columns", MTP_SCALAR_COLUMNS[:8])
        )
        if self.mtp_meta.ndim == 2 and self.mtp_meta.shape[1] != len(
            self.mtp_meta_columns
        ):
            raise ValueError("MTP metadata columns disagree with the stored array")
        if self.mtp_scalars.ndim == 2 and self.mtp_scalars.shape[1] != len(
            self.mtp_scalar_columns
        ):
            raise ValueError("MTP scalar columns disagree with the stored array")
        if self.anchor_spine_contract:
            required_anchor_files = (
                "anchor_spine_meta.npy",
                "anchor_spine_prefix_sha256.npy",
                "anchor_spine_exact_prefix_sha256.npy",
                "anchor_spine_source_ready_timestamp_utc.npy",
                "anchor_spine_offsets.npy",
                "anchor_spine_available.npy",
                "anchor_spine_scalars.npy",
            )
            missing = [name for name in required_anchor_files if not (self.root / name).is_file()]
            if missing:
                raise ValueError(f"adaptive index lacks anchor-spine arrays: {missing}")
            self.anchor_meta = np.load(
                self.root / "anchor_spine_meta.npy", mmap_mode="r"
            )
            self.anchor_offsets = np.load(
                self.root / "anchor_spine_offsets.npy", mmap_mode="r"
            )
            self.anchor_available = np.load(
                self.root / "anchor_spine_available.npy", mmap_mode="r"
            )
            self.anchor_scalars = np.load(
                self.root / "anchor_spine_scalars.npy", mmap_mode="r"
            )
            self.anchor_prefix = np.load(
                self.root / "anchor_spine_prefix_sha256.npy", mmap_mode="r"
            )
            self.anchor_exact_prefix = np.load(
                self.root / "anchor_spine_exact_prefix_sha256.npy", mmap_mode="r"
            )
            self.anchor_source_ready_timestamp_utc = np.load(
                self.root / "anchor_spine_source_ready_timestamp_utc.npy",
                mmap_mode="r",
            )
            self.anchor_meta_columns = tuple(
                self.manifest.get("anchor_spine_meta_columns", ())
            )
            self.anchor_scalar_columns = tuple(
                self.manifest.get("anchor_spine_scalar_columns", ())
            )
            if (
                self.anchor_meta_columns != ANCHOR_SPINE_META_COLUMNS
                or self.anchor_scalar_columns != ANCHOR_SPINE_SCALAR_COLUMNS
                or self.anchor_meta.shape != (
                    int(self.manifest.get("anchor_spine_nodes", -1)),
                    len(ANCHOR_SPINE_META_COLUMNS),
                )
                or self.anchor_scalars.shape
                != (len(self.anchor_meta), len(ANCHOR_SPINE_SCALAR_COLUMNS))
            ):
                raise ValueError("anchor-spine arrays disagree with their manifest")
        else:
            self.anchor_meta = np.empty((0, len(ANCHOR_SPINE_META_COLUMNS)), dtype=np.int64)
            self.anchor_offsets = np.empty((0, len(ANCHOR_SPINE_ROLES)), dtype=np.int64)
            self.anchor_available = np.empty((0, len(ANCHOR_SPINE_ROLES)), dtype=np.bool_)
            self.anchor_scalars = np.empty(
                (0, len(ANCHOR_SPINE_SCALAR_COLUMNS)), dtype=np.float32
            )
            self.anchor_prefix = np.empty((0,), dtype="S32")
            self.anchor_exact_prefix = np.empty((0,), dtype="S32")
            self.anchor_source_ready_timestamp_utc = np.empty((0,), dtype="<U1")
            self.anchor_meta_columns = ANCHOR_SPINE_META_COLUMNS
            self.anchor_scalar_columns = ANCHOR_SPINE_SCALAR_COLUMNS
        self._payloads: dict[str, np.memmap[Any, Any]] = {}

        self.target_by_position = self._complete_layer_blocks(
            self.target_meta, "target", require_valid=True
        )
        self.prompt_by_position = self._complete_layer_blocks(
            self.prompt_meta, "prompt", require_valid=False
        )
        overlap = self.target_by_position.keys() & self.prompt_by_position.keys()
        if overlap:
            raise ValueError(f"target/prompt position overlap in {self.root}: {next(iter(overlap))}")
        self.route_by_position: dict[tuple[int, int], tuple[int, int]] = {
            key: (0, row) for key, row in self.target_by_position.items()
        }
        self.route_by_position.update(
            {key: (1, row) for key, row in self.prompt_by_position.items()}
        )
        self.tree_by_root: dict[tuple[int, int], list[int]] = {}
        for row, metadata in enumerate(self.mtp_meta):
            sequence, root_position = int(metadata[0]), int(metadata[1])
            if not bool(metadata[12]):
                continue
            self.tree_by_root.setdefault((sequence, root_position), []).append(row)
        for rows in self.tree_by_root.values():
            # Source-ready event order is a causal and parent-before-child order
            # in the native capture. Event ID gives a deterministic tie-break.
            rows.sort(key=lambda row: (int(self.mtp_meta[row, 9]), int(self.mtp_meta[row, 5])))
        self.anchor_spine_by_root: dict[tuple[int, int], list[int]] = {}
        for row, metadata in enumerate(self.anchor_meta):
            if not bool(metadata[8]):
                continue
            key = (int(metadata[0]), int(metadata[1]))
            self.anchor_spine_by_root.setdefault(key, []).append(row)
        for key, rows in self.anchor_spine_by_root.items():
            rows.sort(key=lambda row: int(self.anchor_meta[row, 10]))
            if (
                len(rows) != ANCHOR_SPINE_REQUIRED_DEPTH
                or [int(self.anchor_meta[row, 2]) for row in rows]
                != list(range(1, ANCHOR_SPINE_REQUIRED_DEPTH + 1))
            ):
                raise ValueError(f"source {key} lacks exact H1-H6 anchor-spine rows")

    def mtp_meta_column(self, name: str) -> int | None:
        try:
            return self.mtp_meta_columns.index(name)
        except ValueError:
            return None

    def mtp_scalar_column(self, name: str) -> int | None:
        try:
            return self.mtp_scalar_columns.index(name)
        except ValueError:
            return None

    def _complete_layer_blocks(
        self,
        metadata: np.ndarray[Any, Any],
        name: str,
        *,
        require_valid: bool,
    ) -> dict[tuple[int, int], int]:
        if len(metadata) % TARGET_LAYERS:
            raise ValueError(f"{name} rows are not complete 40-layer tokens in {self.root}")
        result: dict[tuple[int, int], int] = {}
        for row in range(0, len(metadata), TARGET_LAYERS):
            block = np.asarray(metadata[row : row + TARGET_LAYERS])
            if not np.array_equal(block[:, 2], np.arange(TARGET_LAYERS)):
                raise ValueError(f"{name} layer discontinuity at row {row} in {self.root}")
            if not (block[:, 0] == block[0, 0]).all() or not (
                block[:, 1] == block[0, 1]
            ).all():
                raise ValueError(f"{name} token discontinuity at row {row} in {self.root}")
            if require_valid and block.shape[1] > 5 and not bool(block[:, 5].all()):
                continue
            key = (int(block[0, 0]), int(block[0, 1]))
            if key in result:
                raise ValueError(f"duplicate {name} token {key} in {self.root}")
            result[key] = row
        return result

    def _payload(self, role: str) -> np.memmap[Any, Any]:
        descriptor = self.descriptors[role]
        relative = str(descriptor["payload_file"])
        mapping = self._payloads.get(relative)
        if mapping is None:
            path = self.segment_root / relative
            mapping = np.memmap(path, mode="r", dtype=np.uint8)
            self._payloads[relative] = mapping
        return mapping

    def availability(self, kind: str, rows: Iterable[int], role: str) -> Tensor:
        row_list = [int(row) for row in rows]
        column = self.role_columns[kind][role]
        if kind == "target":
            values = self.target_available[row_list, column]
        elif kind == "mtp":
            values = self.mtp_available[row_list, column]
        elif kind == "anchor":
            values = self.anchor_available[row_list, column]
        else:
            values = self.prompt_offsets[row_list, column] >= 0
        return torch.from_numpy(np.asarray(values, dtype=np.bool_).copy())

    def read(
        self,
        kind: str,
        rows: Iterable[int],
        role: str,
        *,
        allow_missing: bool = False,
    ) -> Tensor:
        row_list = [int(row) for row in rows]
        if role not in self.role_columns[kind]:
            if not allow_missing:
                raise KeyError(f"{kind} role {role!r} is absent from {self.root}")
            descriptor = self.descriptors.get(role)
            if descriptor is None:
                raise KeyError(f"no tensor descriptor is available for optional role {role!r}")
            return _zeros((len(row_list), *descriptor["shape"]), str(descriptor["dtype"]))
        descriptor = self.descriptors[role]
        dtype_name = str(descriptor["dtype"])
        try:
            dtype = NP_DTYPES[dtype_name]
        except KeyError as exc:
            raise ValueError(f"unsupported capture dtype {dtype_name!r}") from exc
        shape = tuple(int(value) for value in descriptor["shape"])
        count = int(np.prod(shape))
        offsets = {
            "target": self.target_offsets,
            "prompt": self.prompt_offsets,
            "mtp": self.mtp_offsets,
            "anchor": self.anchor_offsets,
        }[kind]
        column = self.role_columns[kind][role]
        result = np.zeros((len(row_list), *shape), dtype=dtype)
        mapping: np.memmap[Any, Any] | None = None
        for output_row, source_row in enumerate(row_list):
            offset = int(offsets[source_row, column])
            if offset < 0:
                if allow_missing:
                    continue
                raise ValueError(
                    f"missing required {kind} tensor {role} at row {source_row}"
                )
            if mapping is None:
                mapping = self._payload(role)
            result[output_row] = np.frombuffer(
                mapping, dtype=dtype, count=count, offset=offset
            ).reshape(shape)
        return numpy_to_torch(result, dtype_name)

    def read_layer_token(
        self,
        kind: str,
        row: int,
        roles: Iterable[str],
        *,
        allow_missing: bool = False,
    ) -> dict[str, Tensor]:
        rows = range(int(row), int(row) + TARGET_LAYERS)
        return {
            role: self.read(kind, rows, role, allow_missing=allow_missing)
            for role in roles
        }


class HarpRTTDataset(Dataset[dict[str, Any]]):
    """Token-end H1--H4 HARP-RTT samples from frozen request splits."""

    def __init__(
        self,
        index_root: Path,
        split: str,
        *,
        corpus_root: Path | None = None,
        max_tree_nodes: int = 32,
        allow_test: bool = False,
        include_optional_current: bool = True,
    ) -> None:
        if split not in {"train", "validation", "calibration", "test"}:
            raise ValueError(f"unknown frozen split {split!r}")
        if split == "test" and not allow_test:
            raise PermissionError(
                "the outer test split is sealed; pass allow_test=True only after the "
                "architecture, seed policy, calibration, and candidate width are frozen"
            )
        if max_tree_nodes < 1:
            raise ValueError("max_tree_nodes must be positive")
        self.index_root = Path(index_root)
        self.split = split
        self.max_tree_nodes = int(max_tree_nodes)
        self.include_optional_current = bool(include_optional_current)
        self.segments = [
            RichSegmentIndex(path, corpus_root)
            for path in sorted(self.index_root.iterdir())
            if (path / "index_manifest.json").is_file()
        ]
        if not self.segments:
            raise ValueError(f"no HARP-RTT segment indices found in {self.index_root}")
        self.records: list[RichTokenRecord] = []
        for segment_id, segment in enumerate(self.segments):
            for (sequence, position), current_row in segment.target_by_position.items():
                if str(segment.sequences[sequence]["split"]) != split:
                    continue
                history = [
                    segment.route_by_position.get((sequence, position - lag))
                    for lag in range(HISTORY_LENGTH)
                ]
                future = [
                    segment.target_by_position.get((sequence, position + horizon))
                    for horizon in range(1, HORIZONS + 1)
                ]
                if any(value is None for value in history + future):
                    continue
                resolved_history = [value for value in history if value is not None]
                resolved_future = [int(value) for value in future if value is not None]
                tree_rows = tuple(
                    segment.tree_by_root.get((sequence, position), [])[: self.max_tree_nodes]
                )
                anchor_spine_rows = tuple(
                    segment.anchor_spine_by_root.get((sequence, position), [])
                )
                if segment.adaptive_mtp_tree:
                    # Adaptive capture may intentionally cover only a subset of
                    # generated positions. Positions outside that source set are
                    # not formal adaptive samples and must not enter training.
                    if not tree_rows and not anchor_spine_rows:
                        continue
                    if not tree_rows or len(anchor_spine_rows) != ANCHOR_SPINE_REQUIRED_DEPTH:
                        raise ValueError(
                            "adaptive sample lacks its complete separate H1-H6 anchor spine"
                        )
                self.records.append(
                    RichTokenRecord(
                        segment=segment_id,
                        sequence=sequence,
                        position=position,
                        current_row=current_row,
                        history_kinds=tuple(int(value[0]) for value in resolved_history),
                        history_rows=tuple(int(value[1]) for value in resolved_history),
                        future_rows=tuple(resolved_future),
                        tree_rows=tree_rows,
                        anchor_spine_rows=anchor_spine_rows,
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _route_summary(centered_logits: Tensor) -> Tensor:
        probabilities = torch.softmax(centered_logits.float(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
        ordered = torch.sort(centered_logits.float(), dim=-1, descending=True).values
        gap_8_9 = ordered[..., 7] - ordered[..., 8]
        gap_1_2 = ordered[..., 0] - ordered[..., 1]
        top8_mass = probabilities.topk(TOP_K, dim=-1).values.sum(-1)
        return torch.stack((entropy, gap_8_9, gap_1_2, top8_mass), dim=-1)

    def _history(self, segment: RichSegmentIndex, record: RichTokenRecord) -> dict[str, Tensor]:
        logits: list[Tensor] = []
        selected_ids: list[Tensor] = []
        execution_weights: list[Tensor] = []
        for kind_code, row in zip(record.history_kinds, record.history_rows, strict=True):
            kind = "target" if kind_code == 0 else "prompt"
            values = segment.read_layer_token(
                kind,
                row,
                (
                    "raw_target_router_logits",
                    "selected_expert_ids",
                    "selected_execution_weights",
                ),
            )
            logits.append(values["raw_target_router_logits"].float())
            selected_ids.append(values["selected_expert_ids"].to(torch.int64))
            execution_weights.append(values["selected_execution_weights"].float())
        raw_logits = torch.stack(logits)
        centered_logits = raw_logits - raw_logits.mean(dim=-1, keepdim=True)
        return {
            "logits": centered_logits,
            "selected_ids": torch.stack(selected_ids),
            "execution_weights": torch.stack(execution_weights),
            "summary": self._route_summary(centered_logits),
            "available": torch.ones(
                HISTORY_LENGTH, TARGET_LAYERS, dtype=torch.bool
            ),
        }

    def _tree(
        self,
        segment: RichSegmentIndex,
        record: RichTokenRecord,
        exact_next_token_id: int,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        count = len(record.tree_rows)
        safe_rows = list(record.tree_rows) if count else [0]
        if len(segment.mtp_meta) == 0:
            safe_rows = []
        pad = self.max_tree_nodes

        def padded_role(role: str) -> Tensor:
            descriptor = segment.descriptors[role]
            shape = tuple(int(value) for value in descriptor["shape"])
            dtype_name = str(descriptor["dtype"])
            output = _zeros((pad, *shape), dtype_name)
            if count:
                output[:count] = segment.read("mtp", record.tree_rows, role)
            return output

        states = torch.stack([padded_role(role) for role in TREE_STATE_ROLES], dim=1)
        meta_width = (
            int(segment.mtp_meta.shape[1])
            if segment.mtp_meta.ndim == 2
            else len(MTP_META_COLUMNS[:13])
        )
        scalar_width = (
            int(segment.mtp_scalars.shape[1])
            if segment.mtp_scalars.ndim == 2
            else len(MTP_SCALAR_COLUMNS[:8])
        )
        tree = {
            # Batch-level contract marker lets the adapter preserve the exact
            # anonymous-scalar behavior of historical greedy-chain indices.
            "adaptive_contract": torch.tensor(
                segment.adaptive_mtp_tree, dtype=torch.bool
            ),
            "states": states,
            "router_logits": padded_role("raw_mtp_router_logits").float(),
            "selected_ids": padded_role("mtp_selected_expert_ids").to(torch.int64),
            "execution_weights": padded_role("mtp_selected_execution_weights").float(),
            "vocab_top64_ids": padded_role("vocab_top64_token_ids").to(torch.int64),
            "vocab_top64_log_probabilities": padded_role(
                "vocab_top64_log_probabilities"
            ).float(),
            "vocab_statistics": torch.zeros((pad, 6), dtype=torch.float32),
            "meta": torch.zeros((pad, meta_width), dtype=torch.int64),
            # ``scalars`` retains the frozen eight-channel legacy contract for
            # the anchor bridge. ``capture_scalars`` preserves every indexed
            # adaptive value and the named tensors below feed the rich model.
            "scalars": torch.zeros((pad, 8), dtype=torch.float32),
            "capture_scalars": torch.zeros((pad, scalar_width), dtype=torch.float32),
            "parent": torch.full((pad,), -1, dtype=torch.int64),
            # Dense, sample-local identity for the opaque branch-path hash in
            # meta column 11.  Zero is reserved for padding; valid branches
            # are numbered by first causal occurrence.  This avoids hash
            # collisions from reducing a 63-bit identity modulo an embedding
            # table size in the model adapter.
            "branch": torch.zeros((pad,), dtype=torch.int64),
            "mask": torch.zeros((pad,), dtype=torch.bool),
            "horizon_mask": torch.zeros((HORIZONS, pad), dtype=torch.bool),
            "exact_root_match": torch.zeros((pad,), dtype=torch.bool),
            "node_ids": torch.full((pad,), -1, dtype=torch.int64),
            "parent_event_ids": torch.full((pad,), -1, dtype=torch.int64),
            "path_ids": torch.zeros((pad,), dtype=torch.int64),
            "depth": torch.zeros((pad,), dtype=torch.int64),
            "target_positions": torch.zeros((pad,), dtype=torch.int64),
            "absolute_target_positions": torch.zeros((pad,), dtype=torch.int64),
            "vocabulary_prediction_positions": torch.zeros(
                (pad,), dtype=torch.int64
            ),
            "conditioning_token_ids": torch.zeros((pad,), dtype=torch.int64),
            "token_ids": torch.zeros((pad,), dtype=torch.int64),
            "path_log_probabilities": torch.zeros((pad,), dtype=torch.float32),
            "path_probabilities": torch.zeros((pad,), dtype=torch.float32),
            "local_probabilities": torch.zeros((pad,), dtype=torch.float32),
            "source_ready": torch.zeros((pad,), dtype=torch.float32),
            "child_ranks": torch.zeros((pad,), dtype=torch.int64),
            "first_divergence_depths": torch.zeros((pad,), dtype=torch.int64),
            "cumulative_path_ranks": torch.zeros((pad,), dtype=torch.int64),
            "sibling_counts": torch.zeros((pad,), dtype=torch.int64),
            "source_ready_event_order": torch.zeros((pad,), dtype=torch.int64),
            "source_ready_monotonic_ns": torch.zeros((pad,), dtype=torch.int64),
            "structural_valid": torch.zeros((pad,), dtype=torch.bool),
            "feature_available": torch.zeros((pad,), dtype=torch.bool),
            "conditioning_classes": torch.zeros((pad,), dtype=torch.int64),
            "exact_committed_h1_root": torch.zeros((pad,), dtype=torch.bool),
            "exact_prefix_hashes": torch.zeros((pad, 32), dtype=torch.uint8),
            "tensor_available": torch.zeros(
                (pad, len(segment.roles["mtp"])), dtype=torch.bool
            ),
        }
        labels = {
            "acceptance": torch.zeros((pad,), dtype=torch.bool),
            "acceptance_valid": torch.zeros((pad,), dtype=torch.bool),
        }
        if not count:
            return tree, labels

        metadata = torch.from_numpy(
            np.asarray(segment.mtp_meta[list(record.tree_rows)]).copy()
        ).to(torch.int64)
        scalars = torch.from_numpy(
            np.asarray(segment.mtp_scalars[list(record.tree_rows)]).copy()
        ).float()
        availability = torch.from_numpy(
            np.asarray(segment.mtp_available[list(record.tree_rows)]).copy()
        ).bool()
        acceptance = torch.from_numpy(
            np.asarray(segment.mtp_acceptance[list(record.tree_rows)]).copy()
        )
        exact_prefix_bytes = np.asarray(
            segment.mtp_exact_prefix[list(record.tree_rows)]
        ).copy().view(np.uint8).reshape(count, 32)
        event_to_local = {
            int(metadata[local, 5]): local for local in range(count)
        }
        branch_to_local: dict[int, int] = {}
        dense_branches: list[int] = []
        for branch_hash in metadata[:, 11].tolist():
            key = int(branch_hash)
            if key not in branch_to_local:
                # Branch zero is the padding sentinel.
                branch_to_local[key] = len(branch_to_local) + 1
            dense_branches.append(branch_to_local[key])
        local_parents: list[int] = []
        for local in range(count):
            parent_event = int(metadata[local, 6])
            if parent_event < 0:
                local_parents.append(-1)
                continue
            if parent_event not in event_to_local:
                raise ValueError(
                    "selected rich-tree node references a parent outside the node budget"
                )
            parent_local = event_to_local[parent_event]
            if parent_local >= local:
                raise ValueError("rich-tree parent must precede its child in causal order")
            local_parents.append(parent_local)
        parents = torch.tensor(local_parents, dtype=torch.int64)
        horizons = metadata[:, 3]
        node_tokens = metadata[:, 7]
        rank_column = segment.mtp_meta_column("token_rank_under_parent")
        child_ranks = (
            metadata[:, rank_column]
            if rank_column is not None
            else torch.zeros(count, dtype=torch.int64)
        )
        child_ranks = child_ranks.clamp_min(0)
        sibling_counts = torch.tensor(
            [sum(value == parent for value in local_parents) for parent in local_parents],
            dtype=torch.int64,
        )
        path_ranks = torch.zeros(count, dtype=torch.int64)
        for horizon in range(1, HORIZONS + 1):
            rows = torch.nonzero(horizons == horizon, as_tuple=False).flatten()
            if rows.numel():
                order = torch.argsort(scalars[rows, 1], descending=True, stable=True)
                path_ranks[rows[order]] = torch.arange(1, rows.numel() + 1)
        divergence = _structural_first_divergence_depths(
            horizons, child_ranks, parents
        )
        tree["child_ranks"][:count] = child_ranks
        tree["first_divergence_depths"][:count] = divergence
        tree["cumulative_path_ranks"][:count] = path_ranks
        tree["sibling_counts"][:count] = sibling_counts
        tree["meta"][:count] = metadata
        tree["capture_scalars"][:count] = scalars
        tree["scalars"][:count] = scalars[:, :8]
        tree["parent"][:count] = parents
        tree["branch"][:count] = torch.tensor(dense_branches, dtype=torch.int64)
        tree["mask"][:count] = True
        tree["horizon_mask"][:, :count] = torch.stack(
            [horizons == horizon for horizon in range(1, HORIZONS + 1)]
        )
        tree["exact_root_match"][:count] = (horizons == 1) & (
            node_tokens == int(exact_next_token_id)
        )
        tree["node_ids"][:count] = metadata[:, 5]
        tree["parent_event_ids"][:count] = metadata[:, 6]
        path_column = segment.mtp_meta_column("path_identity_hash_i63")
        tree["path_ids"][:count] = (
            metadata[:, path_column] if path_column is not None else metadata[:, 11]
        )
        tree["depth"][:count] = metadata[:, 4]
        tree["target_positions"][:count] = horizons
        tree["absolute_target_positions"][:count] = metadata[:, 2]
        vocabulary_position_column = segment.mtp_meta_column(
            "vocabulary_prediction_position"
        )
        tree["vocabulary_prediction_positions"][:count] = (
            metadata[:, vocabulary_position_column]
            if vocabulary_position_column is not None
            else metadata[:, 2] + 1
        )
        tree["conditioning_token_ids"][:count] = node_tokens
        tree["token_ids"][:count] = node_tokens
        tree["path_log_probabilities"][:count] = scalars[:, 1]
        tree["path_probabilities"][:count] = scalars[:, 2]
        vocab_logp = tree["vocab_top64_log_probabilities"][:count].float()
        vocab_probability = vocab_logp.clamp(max=0.0).exp()
        retained_mass = vocab_probability.sum(dim=-1)
        entropy_column = segment.mtp_scalar_column("vocabulary_entropy")
        vocabulary_entropy = (
            scalars[:, entropy_column]
            if entropy_column is not None
            else -(vocab_probability * vocab_logp).sum(dim=-1)
        )
        margin_column = segment.mtp_scalar_column(
            "next_top1_top2_logprob_margin"
        )
        ordered_logp = torch.sort(
            vocab_logp, dim=-1, descending=True, stable=True
        ).values
        top1_top2_margin = (
            scalars[:, margin_column]
            if margin_column is not None
            else ordered_logp[:, 0] - ordered_logp[:, 1]
        )
        top8_mass_column = segment.mtp_scalar_column("vocabulary_top8_mass")
        top8_mass = (
            scalars[:, top8_mass_column]
            if top8_mass_column is not None
            else ordered_logp[:, :8].exp().sum(dim=-1)
        )
        tree["vocab_statistics"][:count] = torch.stack(
            [
                retained_mass,
                (1.0 - retained_mass).clamp_min(0.0),
                vocabulary_entropy,
                ordered_logp[:, 0].exp(),
                top1_top2_margin,
                top8_mass,
            ],
            dim=-1,
        )
        local_probability_column = segment.mtp_scalar_column(
            "local_token_probability"
        )
        tree["local_probabilities"][:count] = (
            scalars[:, local_probability_column]
            if local_probability_column is not None
            else scalars[:, 0].exp()
        )
        ready_order = metadata[:, 9]
        ready_delta = (ready_order - ready_order.min()).float()
        tree["source_ready"][:count] = ready_delta / ready_delta.max().clamp_min(1.0)
        tree["source_ready_event_order"][:count] = ready_order
        ready_ns_column = segment.mtp_meta_column("source_ready_monotonic_ns")
        if ready_ns_column is not None:
            tree["source_ready_monotonic_ns"][:count] = metadata[:, ready_ns_column]
        structural_column = segment.mtp_meta_column("structural_validity")
        feature_column = segment.mtp_meta_column("feature_available")
        exact_root_column = segment.mtp_meta_column("exact_committed_h1_root")
        tree["structural_valid"][:count] = (
            metadata[:, structural_column].bool()
            if structural_column is not None
            else metadata[:, 12].bool()
        )
        tree["feature_available"][:count] = (
            metadata[:, feature_column].bool()
            if feature_column is not None
            else metadata[:, 12].bool()
        )
        tree["conditioning_classes"][:count] = metadata[:, 10]
        tree["exact_committed_h1_root"][:count] = (
            metadata[:, exact_root_column].bool()
            if exact_root_column is not None
            else horizons == 1
        )
        tree["exact_prefix_hashes"][:count] = torch.from_numpy(
            exact_prefix_bytes.copy()
        )
        tree["tensor_available"][:count] = availability
        labels["acceptance"][:count] = acceptance[:, 0].bool()
        labels["acceptance_valid"][:count] = acceptance[:, 1].bool()
        if segment.adaptive_mtp_tree:
            roots = parents == -1
            exact_roots = roots & tree["exact_committed_h1_root"][:count]
            if exact_roots.sum().item() != 1:
                raise ValueError("adaptive tree must contain exactly one marked H1 root")
            root_index = int(torch.nonzero(exact_roots, as_tuple=False)[0, 0])
            if (
                int(horizons[root_index]) != 1
                or int(node_tokens[root_index]) != int(exact_next_token_id)
            ):
                raise ValueError("adaptive tree root is not the exact committed H1 token")
        return tree, labels

    def _anchor_spine(
        self,
        segment: RichSegmentIndex,
        record: RichTokenRecord,
        exact_next_token_id: int,
    ) -> dict[str, Tensor]:
        """Build the anchor-only channel; never merge it into model inputs."""

        rows = record.anchor_spine_rows
        if not segment.anchor_spine_contract or len(rows) != ANCHOR_SPINE_REQUIRED_DEPTH:
            raise ValueError("adaptive sample has no complete H1-H6 anchor-spine contract")
        metadata = torch.from_numpy(
            np.asarray(segment.anchor_meta[list(rows)]).copy()
        ).to(torch.int64)
        availability = torch.from_numpy(
            np.asarray(segment.anchor_available[list(rows)]).copy()
        ).bool()
        if not availability.all():
            raise ValueError("anchor spine lacks a required compatibility tensor")
        depths = metadata[:, 2]
        event_to_local = {int(metadata[index, 3]): index for index in range(len(rows))}
        parents: list[int] = []
        for local, parent_event in enumerate(metadata[:, 4].tolist()):
            if int(parent_event) < 0:
                parents.append(-1)
            elif int(parent_event) not in event_to_local:
                raise ValueError("anchor spine references a parent outside its channel")
            else:
                parents.append(event_to_local[int(parent_event)])
            if parents[-1] != (-1 if local == 0 else local - 1):
                raise ValueError("anchor spine is not a contiguous parent chain")
        if (
            depths.tolist() != list(range(1, ANCHOR_SPINE_REQUIRED_DEPTH + 1))
            or int(metadata[0, 5]) != int(exact_next_token_id)
            or metadata[:, 10].tolist()
            != list(range(ANCHOR_SPINE_REQUIRED_DEPTH))
            or metadata[:, 16].tolist() != [-1] + [0] * (ANCHOR_SPINE_REQUIRED_DEPTH - 1)
        ):
            raise ValueError("anchor spine violates exact-root/local-top1 depth semantics")
        exact_hashes = np.asarray(segment.anchor_exact_prefix[list(rows)]).copy()
        return {
            "contract": torch.tensor(True, dtype=torch.bool),
            "hidden_states": segment.read(
                "anchor", rows, ANCHOR_SPINE_ROLES[0]
            ),
            "router_logits": segment.read(
                "anchor", rows, ANCHOR_SPINE_ROLES[1]
            ).float(),
            "depth": depths,
            "parent": torch.tensor(parents, dtype=torch.int64),
            "token_ids": metadata[:, 5],
            "mask": torch.ones(ANCHOR_SPINE_REQUIRED_DEPTH, dtype=torch.bool),
            "node_event_ids": metadata[:, 3],
            "source_ready_event_order": metadata[:, 7],
            "exact_prefix_hashes": torch.from_numpy(
                exact_hashes.view(np.uint8).reshape(ANCHOR_SPINE_REQUIRED_DEPTH, 32)
            ),
            "local_top1_parent_coherent": torch.tensor(True, dtype=torch.bool),
            "continues_through_eos": torch.tensor(True, dtype=torch.bool),
            "labels_present": torch.tensor(False, dtype=torch.bool),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        segment = self.segments[record.segment]
        roles = list(CURRENT_ROLES)
        if self.include_optional_current:
            roles.extend(
                role for role in OPTIONAL_CURRENT_ROLES if role in segment.descriptors
            )
        current = segment.read_layer_token(
            "target", record.current_row, roles, allow_missing=True
        )
        current_available = torch.stack(
            [
                segment.availability(
                    "target",
                    range(record.current_row, record.current_row + TARGET_LAYERS),
                    role,
                )
                if role in segment.role_columns["target"]
                else torch.zeros(TARGET_LAYERS, dtype=torch.bool)
                for role in roles
            ],
            dim=-1,
        )
        future = [
            segment.read_layer_token("target", row, FUTURE_ROLES)
            for row in record.future_rows
        ]
        future_meta = np.asarray(
            [segment.target_meta[row, :6] for row in record.future_rows], dtype=np.int64
        )
        exact_next_token_id = int(future_meta[0, 3])
        tree, tree_labels = self._tree(segment, record, exact_next_token_id)
        target_meta = torch.from_numpy(
            np.asarray(
                segment.target_meta[
                    record.current_row : record.current_row + TARGET_LAYERS
                ]
            ).copy()
        )
        future_logits = torch.stack(
            [item["raw_target_router_logits"].float() for item in future]
        )
        item: dict[str, Any] = {
            "metadata": {
                "segment": str(segment.manifest["segment"]),
                "sequence_id": str(segment.sequences[record.sequence]["sequence_id"]),
                "request_id": str(segment.sequences[record.sequence]["request_id"]),
                "dataset_source": segment.sequences[record.sequence].get("dataset_source"),
                "position": record.position,
                "target_meta": target_meta,
            },
            "inputs": {
                "history": self._history(segment, record),
                "current": current,
                "current_roles": tuple(roles),
                "current_available": current_available,
                "current_scalars": torch.from_numpy(
                    np.asarray(
                        segment.target_scalars[
                            record.current_row : record.current_row + TARGET_LAYERS
                        ]
                    ).copy()
                ).float(),
                "exact_next_token_id": torch.tensor(exact_next_token_id, dtype=torch.int64),
                "within_request": torch.tensor(
                    record.position
                    - int(segment.sequences[record.sequence]["prompt_length"]),
                    dtype=torch.float32,
                ),
                "final_hidden": current["post_moe_residual_xplus"][-1],
                "tree": tree,
            },
            "targets": {
                "future_router_logits": future_logits,
                "future_centered_router_logits": future_logits
                - future_logits.mean(dim=-1, keepdim=True),
                "future_router_inputs": torch.stack(
                    [item["normalized_target_router_input_a"] for item in future]
                ),
                "future_selected_ids": torch.stack(
                    [item["selected_expert_ids"].to(torch.int64) for item in future]
                ),
                "future_execution_weights": torch.stack(
                    [item["selected_execution_weights"].float() for item in future]
                ),
                "future_available": torch.ones(
                    (HORIZONS, TARGET_LAYERS), dtype=torch.bool
                ),
                "future_meta": torch.from_numpy(future_meta.copy()),
                # Label-only: used to identify the factual adaptive-tree node.
                # This hash is never copied into model inputs.
                "future_prefix_hashes": torch.from_numpy(
                    np.asarray(segment.target_prefix[list(record.future_rows)]).copy().view(
                        np.uint8
                    ).reshape(HORIZONS, 32)
                ),
                "tree_acceptance": tree_labels["acceptance"],
                "tree_acceptance_valid": tree_labels["acceptance_valid"],
            },
        }
        if segment.adaptive_mtp_tree:
            # Top-level by design: the adaptive model adapter consumes only
            # item["inputs"], while LegacyHARPAnchorBridge receives the whole
            # batch and reads this compatibility-only channel explicitly.
            item["anchor_inputs"] = {
                "mtp_spine": self._anchor_spine(
                    segment, record, exact_next_token_id
                )
            }
        return item


def collate_harp_rtt(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate fixed-shape tensors while keeping audit metadata as Python lists."""

    if not items:
        raise ValueError("cannot collate an empty HARP-RTT batch")

    def collate(values: Sequence[Any], path: tuple[str, ...]) -> Any:
        first = values[0]
        if isinstance(first, Tensor):
            return torch.stack(list(values))
        if isinstance(first, Mapping):
            if any(set(value) != set(first) for value in values):
                raise ValueError(f"inconsistent mapping keys at {'.'.join(path)}")
            return {
                key: collate([value[key] for value in values], (*path, str(key)))
                for key in first
            }
        if path and path[0] == "metadata":
            return list(values)
        if isinstance(first, tuple) and all(value == first for value in values):
            return first
        if isinstance(first, (int, float, bool)):
            return torch.as_tensor(values)
        return list(values)

    return collate(items, ())


__all__ = [
    "CURRENT_ROLES",
    "EXPERTS",
    "HISTORY_LENGTH",
    "HORIZONS",
    "HarpRTTDataset",
    "RichSegmentIndex",
    "RichTokenRecord",
    "TARGET_LAYERS",
    "TOP_K",
    "TREE_ROLES",
    "TREE_STATE_ROLES",
    "VOCAB_CANDIDATES",
    "collate_harp_rtt",
    "numpy_to_torch",
]
