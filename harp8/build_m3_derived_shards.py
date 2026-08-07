"""Build immutable, leakage-safe token-path shards for the M3 input stream.

The native recurrent-MTP capture stores one six-node draft chain for every
authoritative target-router row.  Candidate pools are split-specific and may
reorder requests, so this builder joins them by ``(request_id,
within_request_index)`` and emits only the split-pool rows in pool order.

Two token streams are causal model inputs:

* ``state_input_token_ids``: the token that conditioned each MTP state,
  ``[authoritative x[t+1], draft_1, ..., draft_5]``;
* ``draft_output_token_ids``: the six greedy MTP outputs.

Future committed tokens are read only long enough to derive label-only token
agreement and prefix-reliability bits.  Their IDs and log-probabilities are
never copied into the derived shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .candidates import POOL_SCHEMA


# Immutable source-format identifiers are duplicated here deliberately.  The
# shard builder validates an external J-Route capture, but the HARP package
# must not require the capture implementation itself at runtime.
COMPACT_REPRESENTATION = "authoritative_bf16_greedy_decode_compact_v1"
COMPACT_SCHEMA = "jroute0_compact_bf16_capture_v1"
MTP_DEPTH_SCHEMA = "jroute0_native_bf16_mtp_depth_capture_v1"
M3_DERIVED_SCHEMA = "harp8_m3_token_path_shards_v1"
M3_DEPTHS = 6
M3_POOL_HORIZONS = 8

CAUSAL_ARRAYS = frozenset(
    {
        "state_input_token_ids",
        "state_input_valid",
        "draft_output_token_ids",
        "draft_output_valid",
    }
)
ALIGNMENT_ARRAYS = frozenset({"capture_rows"})
LABEL_ONLY_ARRAYS = frozenset(
    {"path_token_match", "path_prefix_reliability", "reliability_valid"}
)


def _sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object")
                records.append(value)
    except OSError as exc:
        raise ValueError(f"cannot read {path}") from exc
    return records


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer, not boolean")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an exact integer")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _token_list(value: Any, label: str, vocab_size: int) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a token-ID list")
    tokens = [
        _integer(item, f"{label}[{index}]", minimum=0)
        for index, item in enumerate(value)
    ]
    if any(token >= vocab_size for token in tokens):
        raise ValueError(f"{label} contains an out-of-vocabulary token ID")
    return tokens


def _safe_child(root: Path, filename: str) -> Path:
    raw = Path(filename)
    if raw.is_absolute() or raw.name != filename or filename in {"", ".", ".."}:
        raise ValueError(f"unsafe artifact filename {filename!r}")
    return root / raw


def _verified_file(
    root: Path,
    record: Mapping[str, Any],
    *,
    expected_filename: str,
) -> Path:
    # Some source manifests key file records by filename and therefore omit a
    # redundant ``filename`` member.  The caller has already selected that
    # exact key, so the expected basename is the safe default.
    filename = str(record.get("filename", record.get("path", expected_filename)))
    if filename != expected_filename:
        raise ValueError(f"manifest filename mismatch for {expected_filename}")
    path = _safe_child(root, filename)
    if not path.is_file():
        raise ValueError(f"declared artifact is absent: {path}")
    expected_bytes = _integer(record.get("bytes"), f"{filename} bytes", minimum=0)
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"file-size mismatch for {filename}")
    expected_sha = str(record.get("sha256", ""))
    if len(expected_sha) != 64 or _sha256_file(path) != expected_sha:
        raise ValueError(f"SHA-256 mismatch for {filename}")
    return path


def _pool_array_record(manifest: Mapping[str, Any], filename: str) -> Mapping[str, Any]:
    arrays = manifest.get("arrays")
    if not isinstance(arrays, list):
        raise ValueError("candidate-pool manifest must declare its arrays")
    matches = [
        record
        for record in arrays
        if isinstance(record, Mapping) and record.get("path") == filename
    ]
    if len(matches) != 1:
        raise ValueError(
            f"candidate-pool manifest must declare {filename} exactly once"
        )
    return matches[0]


def _manifest_file_record(
    manifest: Mapping[str, Any], filename: str
) -> Mapping[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not isinstance(files.get(filename), Mapping):
        raise ValueError(f"source manifest does not declare {filename}")
    return files[filename]


def _mtp_file_record(manifest: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not isinstance(files.get(label), Mapping):
        raise ValueError(f"MTP manifest does not declare {label}")
    return files[label]


def _resolve_split_manifest(
    pool_root: Path,
    pool_manifest: Mapping[str, Any],
    override: Path | None,
) -> tuple[Path, dict[str, Any]]:
    record = pool_manifest.get("split_manifest")
    if not isinstance(record, Mapping):
        raise ValueError("M3 derivation requires a hashed inner split manifest")
    declared_hash = str(record.get("sha256", ""))
    if len(declared_hash) != 64:
        raise ValueError("candidate pool has no valid split-manifest SHA-256")
    if override is not None:
        path = Path(override)
    else:
        raw = Path(str(record.get("path", "")))
        path = raw if raw.is_absolute() else pool_root / raw
    if not path.is_file() or _sha256_file(path) != declared_hash:
        raise ValueError("candidate pool split-manifest provenance mismatch")
    manifest = _json_object(path)
    if manifest.get("schema") != "harp8_inner_split_v1":
        raise ValueError("inner split manifest has an incompatible schema")
    return path, manifest


def _read_pool(
    pool_root: Path,
    *,
    expected_split: str,
    split_manifest_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any], np.ndarray]:
    manifest_path = pool_root / "manifest.json"
    manifest = _json_object(manifest_path)
    # This gate intentionally precedes every pool metadata/array read.
    if expected_split not in {"train", "validation"}:
        raise PermissionError("M3 development shards support train/validation only")
    if manifest.get("schema") != POOL_SCHEMA:
        raise ValueError("candidate pool has an incompatible schema")
    if manifest.get("split") not in {"train", "validation"}:
        raise PermissionError("sealed/test candidate pools are forbidden")
    if manifest.get("split") != expected_split:
        raise ValueError(
            f"expected {expected_split!r} pool, found {manifest.get('split')!r}"
        )
    if bool(manifest.get("allow_test", False)):
        raise PermissionError("candidate pool was exported with test access enabled")

    rows = _integer(manifest.get("rows"), "candidate-pool rows", minimum=1)
    horizons = _integer(manifest.get("horizons"), "candidate-pool horizons", minimum=1)
    if horizons != M3_POOL_HORIZONS:
        raise ValueError("M3 derivation requires an H1-H8 candidate pool")
    metadata_path = pool_root / "metadata.json"
    metadata = _json_object(metadata_path)
    for name in ("request_ids", "within", "domains"):
        if not isinstance(metadata.get(name), list) or len(metadata[name]) != rows:
            raise ValueError(f"candidate metadata {name!r} has the wrong row count")
    request_ids = np.asarray(
        [
            _integer(value, f"pool request_ids[{index}]")
            for index, value in enumerate(metadata["request_ids"])
        ],
        dtype=np.int64,
    )
    within = np.asarray(
        [
            _integer(value, f"pool within[{index}]", minimum=0)
            for index, value in enumerate(metadata["within"])
        ],
        dtype=np.int64,
    )
    metadata["request_ids_array"] = request_ids
    metadata["within_array"] = within

    valid_record = _pool_array_record(manifest, "valid_future.u1")
    valid_path = _verified_file(
        pool_root, valid_record, expected_filename="valid_future.u1"
    )
    if valid_path.stat().st_size != rows * horizons:
        raise ValueError("valid_future.u1 has incompatible geometry/dtype")
    valid_future = np.memmap(valid_path, mode="r", dtype="u1", shape=(rows, horizons))
    if np.any((valid_future != 0) & (valid_future != 1)):
        raise ValueError("valid_future.u1 must be binary")

    split_path, split_manifest = _resolve_split_manifest(
        pool_root, manifest, split_manifest_path
    )
    return manifest, metadata, split_path, split_manifest, np.asarray(valid_future)


def _read_capture(
    capture_dir: Path,
    *,
    vocab_size: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[int, int], int],
]:
    manifest_path = capture_dir / "manifest.json"
    manifest = _json_object(manifest_path)
    if (
        manifest.get("schema") != COMPACT_SCHEMA
        or manifest.get("representation") != COMPACT_REPRESENTATION
    ):
        raise ValueError("M3 requires the authoritative compact BF16 capture")
    manifest_rows = _integer(manifest.get("rows"), "capture rows", minimum=1)
    manifest_requests = _integer(
        manifest.get("requests"), "capture requests", minimum=1
    )

    request_path = _verified_file(
        capture_dir,
        _manifest_file_record(manifest, "requests.jsonl"),
        expected_filename="requests.jsonl",
    )
    rows_path = _verified_file(
        capture_dir,
        _manifest_file_record(manifest, "rows.jsonl"),
        expected_filename="rows.jsonl",
    )
    requests = _jsonl_objects(request_path)
    rows = _jsonl_objects(rows_path)
    if len(requests) != manifest_requests or len(rows) != manifest_rows:
        raise ValueError("capture JSONL counts disagree with its manifest")

    by_id: dict[int, dict[str, Any]] = {}
    request_order: list[int] = []
    for number, request in enumerate(requests):
        request_id = _integer(request.get("request_id"), f"request {number} ID")
        if request_id in by_id:
            raise ValueError("capture request IDs must be unique")
        prompt = _token_list(
            request.get("prompt_token_ids"),
            f"request {request_id} prompt_token_ids",
            vocab_size,
        )
        generated = _token_list(
            request.get("generated_token_ids"),
            f"request {request_id} generated_token_ids",
            vocab_size,
        )
        routed = _integer(
            request.get("routed_positions"),
            f"request {request_id} routed_positions",
            minimum=1,
        )
        if routed > len(generated):
            raise ValueError("routed positions exceed the committed generation")
        request["_prompt"] = prompt
        request["_generated"] = generated
        request["_routed"] = routed
        by_id[request_id] = request
        request_order.append(request_id)

    lookup: dict[tuple[int, int], int] = {}
    observed: dict[int, list[int]] = {request_id: [] for request_id in request_order}
    for capture_row, row in enumerate(rows):
        request_id = _integer(
            row.get("request_id"), f"capture row {capture_row} request ID"
        )
        within = _integer(
            row.get("within_request_index"),
            f"capture row {capture_row} within-request index",
            minimum=0,
        )
        if request_id not in by_id:
            raise ValueError("capture row refers to an unknown request")
        request = by_id[request_id]
        if within >= int(request["_routed"]):
            raise ValueError("capture row crosses its request boundary")
        key = (request_id, within)
        if key in lookup:
            raise ValueError("capture contains a duplicate request/position row")
        state_token = _integer(
            row.get("state_token_id"), "capture state token", minimum=0
        )
        if state_token != request["_generated"][within]:
            raise ValueError("capture state-token alignment mismatch")
        expected_position = len(request["_prompt"]) + within
        if (
            _integer(
                row.get("router_state_position"), "router state position", minimum=0
            )
            != expected_position
        ):
            raise ValueError("capture absolute-position alignment mismatch")
        lookup[key] = capture_row
        observed[request_id].append(within)

    cursor = 0
    for request_id in request_order:
        routed = int(by_id[request_id]["_routed"])
        if observed[request_id] != list(range(routed)):
            raise ValueError("capture rows are not canonical within each request")
        actual_rows = [lookup[(request_id, within)] for within in range(routed)]
        if actual_rows != list(range(cursor, cursor + routed)):
            raise ValueError(
                "capture request rows are not contiguous and request-major"
            )
        cursor += routed
    return manifest, requests, rows, lookup


def _validate_pool_alignment(
    *,
    expected_split: str,
    pool_metadata: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    requests: list[dict[str, Any]],
    capture_lookup: Mapping[tuple[int, int], int],
) -> np.ndarray:
    request_ids = np.asarray(pool_metadata["request_ids_array"], dtype=np.int64)
    within = np.asarray(pool_metadata["within_array"], dtype=np.int64)
    assignments_raw = split_manifest.get("assignments")
    if not isinstance(assignments_raw, Mapping):
        raise ValueError("inner split assignments must be a mapping")
    assignments: dict[int, str] = {}
    for raw_id, raw_split in assignments_raw.items():
        request_id = _integer(raw_id, "split request ID")
        split = str(raw_split)
        if split not in {"train", "validation"} or request_id in assignments:
            raise ValueError("inner split has an invalid or duplicate assignment")
        assignments[request_id] = split

    request_by_id = {int(request["request_id"]): request for request in requests}
    selected_ids = set(request_ids.tolist())
    expected_ids = {
        request_id
        for request_id, split in assignments.items()
        if split == expected_split
    }
    if selected_ids != expected_ids:
        raise ValueError("candidate pool request set disagrees with its split manifest")
    for request_id in selected_ids:
        request = request_by_id.get(request_id)
        if request is None:
            raise ValueError("candidate pool request is absent from capture")
        if str(request.get("offline_split")) != "train":
            raise PermissionError(
                "M3 development pools may use only original-training requests"
            )

    seen_keys: set[tuple[int, int]] = set()
    last_request: int | None = None
    closed_requests: set[int] = set()
    positions_by_request: dict[int, list[int]] = {}
    capture_rows = np.empty(len(request_ids), dtype="<i8")
    for row_number, (request_id_raw, within_raw) in enumerate(
        zip(request_ids, within, strict=True)
    ):
        request_id = int(request_id_raw)
        local = int(within_raw)
        if request_id != last_request:
            if request_id in closed_requests:
                raise ValueError("candidate pool request rows are not contiguous")
            if last_request is not None:
                closed_requests.add(last_request)
            last_request = request_id
        key = (request_id, local)
        if key in seen_keys:
            raise ValueError("candidate pool repeats a request/position row")
        if key not in capture_lookup:
            raise ValueError("candidate pool row crosses a capture request boundary")
        seen_keys.add(key)
        positions_by_request.setdefault(request_id, []).append(local)
        capture_rows[row_number] = capture_lookup[key]

    for request_id, positions in positions_by_request.items():
        routed = int(request_by_id[request_id]["_routed"])
        expected_positions = list(range(max(0, routed - 1)))
        if positions != expected_positions:
            raise ValueError(
                "candidate pool must contain every causal source row in order"
            )
    return capture_rows


def _read_mtp_drafts(
    mtp_dir: Path,
    *,
    capture_manifest_path: Path,
    capture_rows: int,
    capture_requests: int,
    vocab_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    manifest_path = mtp_dir / "manifest.json"
    manifest = _json_object(manifest_path)
    if manifest.get("schema") != MTP_DEPTH_SCHEMA:
        raise ValueError("M3 requires a native recurrent-MTP depth capture")
    if manifest.get("source_capture_manifest_sha256") != _sha256_file(
        capture_manifest_path
    ):
        raise ValueError("MTP and target-capture provenance disagree")
    if (
        manifest.get("alignment", {}).get("future_committed_tokens_used_as_features")
        is not False
    ):
        raise ValueError("MTP manifest does not attest leakage-safe recurrent inputs")
    if _integer(manifest.get("rows"), "MTP rows", minimum=1) != capture_rows:
        raise ValueError("MTP row count disagrees with target capture")
    if (
        _integer(manifest.get("requests"), "MTP requests", minimum=1)
        != capture_requests
    ):
        raise ValueError("MTP request count disagrees with target capture")
    depths = _integer(
        manifest.get("draft_depths"), "MTP draft depths", minimum=M3_DEPTHS
    )
    path = _verified_file(
        mtp_dir,
        _mtp_file_record(manifest, "draft_token_ids"),
        expected_filename="mtp_draft_token_ids.npy",
    )
    drafts = np.load(path, mmap_mode="r", allow_pickle=False)
    if drafts.shape != (capture_rows, depths) or drafts.dtype != np.dtype("<i4"):
        raise ValueError("MTP draft-token array has incompatible shape/dtype")
    selected = np.asarray(drafts[:, :M3_DEPTHS], dtype=np.int32)
    if (selected < 0).any() or (selected >= vocab_size).any():
        raise ValueError("MTP draft-token array contains an invalid vocabulary ID")
    return manifest, selected


def _write_array(output_dir: Path, filename: str, values: np.ndarray) -> dict[str, Any]:
    path = output_dir / filename
    contiguous = np.ascontiguousarray(values)
    contiguous.tofile(path)
    return {
        "filename": filename,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.name,
    }


def build_m3_derived_shards(
    pool_root: Path,
    capture_dir: Path,
    mtp_dir: Path,
    output_dir: Path,
    *,
    expected_split: str,
    vocab_size: int,
    split_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Derive one immutable M3 shard aligned exactly to a split candidate pool."""

    pool_root = Path(pool_root)
    capture_dir = Path(capture_dir)
    mtp_dir = Path(mtp_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite M3 shard {output_dir}")
    vocab_size = _integer(vocab_size, "vocabulary size", minimum=2)

    pool_manifest, pool_metadata, split_path, split_manifest, _valid_future = (
        _read_pool(
            pool_root,
            expected_split=expected_split,
            split_manifest_path=split_manifest_path,
        )
    )
    capture_manifest, requests, capture_records, capture_lookup = _read_capture(
        capture_dir, vocab_size=vocab_size
    )
    capture_rows = _validate_pool_alignment(
        expected_split=expected_split,
        pool_metadata=pool_metadata,
        split_manifest=split_manifest,
        requests=requests,
        capture_lookup=capture_lookup,
    )
    mtp_manifest, all_drafts = _read_mtp_drafts(
        mtp_dir,
        capture_manifest_path=capture_dir / "manifest.json",
        capture_rows=len(capture_records),
        capture_requests=len(requests),
        vocab_size=vocab_size,
    )

    request_by_id = {int(request["request_id"]): request for request in requests}
    request_ids = np.asarray(pool_metadata["request_ids_array"], dtype=np.int64)
    within = np.asarray(pool_metadata["within_array"], dtype=np.int64)
    draft_output_token_ids = np.asarray(all_drafts[capture_rows], dtype="<i4")
    draft_output_valid = np.ones(draft_output_token_ids.shape, dtype="u1")
    state_input_token_ids = np.full(draft_output_token_ids.shape, -1, dtype="<i4")
    state_input_valid = np.zeros(draft_output_token_ids.shape, dtype="u1")
    state_input_token_ids[:, 1:] = draft_output_token_ids[:, :-1]
    state_input_valid[:, 1:] = draft_output_valid[:, :-1]

    path_token_match = np.zeros(draft_output_token_ids.shape, dtype="u1")
    reliability_valid = np.zeros(draft_output_token_ids.shape, dtype="u1")
    for pool_row, (request_id_raw, local_raw) in enumerate(
        zip(request_ids, within, strict=True)
    ):
        request = request_by_id[int(request_id_raw)]
        generated = request["_generated"]
        local = int(local_raw)
        if local + 1 < len(generated):
            state_input_token_ids[pool_row, 0] = int(generated[local + 1])
            state_input_valid[pool_row, 0] = 1
        target_available = np.zeros(M3_DEPTHS, dtype=np.bool_)
        for depth in range(M3_DEPTHS):
            target_position = local + depth + 2
            if target_position < len(generated):
                target_available[depth] = True
                path_token_match[pool_row, depth] = np.uint8(
                    draft_output_token_ids[pool_row, depth]
                    == generated[target_position]
                )
        reliability_valid[pool_row] = np.logical_and.accumulate(
            target_available
        ).astype(np.uint8)
        path_token_match[pool_row, ~target_available] = 0

    # A prefix is reliable only while every preceding output matches and every
    # required authoritative comparison remains uncensored.
    path_prefix_reliability = np.logical_and.accumulate(
        path_token_match.astype(np.bool_), axis=1
    ).astype("u1")
    path_prefix_reliability[reliability_valid == 0] = 0

    # The current pools contain only source-eligible rows, so the first MTP
    # state input must be present.  Keeping the mask explicit makes the format
    # safe for future censored datasets without inventing a token ID.
    if np.any((state_input_valid == 0) & (state_input_token_ids != -1)):
        raise AssertionError("invalid state input must use the -1 sentinel")
    if np.any(
        (state_input_valid == 1)
        & ((state_input_token_ids < 0) | (state_input_token_ids >= vocab_size))
    ):
        raise AssertionError("valid state input lies outside the vocabulary")

    output_dir.mkdir(parents=True)
    arrays: dict[str, dict[str, Any]] = {}
    values = {
        "state_input_token_ids": ("state_input_token_ids.i4", state_input_token_ids),
        "state_input_valid": ("state_input_valid.u1", state_input_valid),
        "draft_output_token_ids": ("draft_output_token_ids.i4", draft_output_token_ids),
        "draft_output_valid": ("draft_output_valid.u1", draft_output_valid),
        "path_token_match": ("path_token_match.u1", path_token_match),
        "path_prefix_reliability": (
            "path_prefix_reliability.u1",
            path_prefix_reliability,
        ),
        "reliability_valid": ("reliability_valid.u1", reliability_valid),
        "capture_rows": ("capture_rows.i8", capture_rows.astype("<i8", copy=False)),
    }
    for name, (filename, value) in values.items():
        record = _write_array(output_dir, filename, value)
        if name in CAUSAL_ARRAYS:
            record["role"] = "causal_input"
        elif name in LABEL_ONLY_ARRAYS:
            record["role"] = "label_only"
        elif name in ALIGNMENT_ARRAYS:
            record["role"] = "alignment_metadata"
        else:
            raise AssertionError(f"unclassified M3 output array {name}")
        arrays[name] = record

    source_files = {
        "pool_manifest": pool_root / "manifest.json",
        "pool_metadata": pool_root / "metadata.json",
        "pool_valid_future": pool_root / "valid_future.u1",
        "split_manifest": split_path,
        "capture_manifest": capture_dir / "manifest.json",
        "capture_requests": capture_dir / "requests.jsonl",
        "capture_rows": capture_dir / "rows.jsonl",
        "mtp_manifest": mtp_dir / "manifest.json",
        "mtp_draft_token_ids": mtp_dir / "mtp_draft_token_ids.npy",
    }
    manifest: dict[str, Any] = {
        "schema": M3_DERIVED_SCHEMA,
        "split": expected_split,
        "rows": int(len(capture_rows)),
        "depths": M3_DEPTHS,
        "vocab_size": vocab_size,
        "pool_schema": pool_manifest["schema"],
        "capture_schema": capture_manifest["schema"],
        "mtp_schema": mtp_manifest["schema"],
        "alignment": {
            "join_key": ["request_id", "within_request_index"],
            "row_order": "identical to split candidate-pool row order",
            "state_input_depth_1": "authoritative committed x[t+1] used by the captured MTP execution",
            "state_input_depth_2_to_6": "preceding causal greedy MTP draft outputs 1..5",
            "draft_output_depth_1_to_6": "greedy MTP outputs predicting committed x[t+2]..x[t+7]",
            "path_prefix_reliability": "cumulative AND of output-token agreement through each depth",
            "reliability_valid": "cumulative authoritative-target availability; false after first censored target",
        },
        "leakage_contract": {
            "causal_arrays": sorted(CAUSAL_ARRAYS),
            "alignment_arrays": sorted(ALIGNMENT_ARRAYS),
            "label_only_arrays": sorted(LABEL_ONLY_ARRAYS),
            "future_committed_token_ids_exported": False,
            "future_target_logprobs_read": False,
            "future_target_logprobs_exported": False,
            "sealed_test_opened": False,
        },
        "counts": {
            "requests": int(len(set(request_ids.tolist()))),
            "state_input_valid_by_depth": state_input_valid.sum(
                axis=0, dtype=np.int64
            ).tolist(),
            "draft_output_valid_by_depth": draft_output_valid.sum(
                axis=0, dtype=np.int64
            ).tolist(),
            "reliability_valid_by_depth": reliability_valid.sum(
                axis=0, dtype=np.int64
            ).tolist(),
            "path_token_match_by_depth": path_token_match.sum(
                axis=0, dtype=np.int64
            ).tolist(),
            "path_prefix_reliable_by_depth": path_prefix_reliability.sum(
                axis=0, dtype=np.int64
            ).tolist(),
        },
        "sources": {
            label: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for label, path in source_files.items()
        },
        "arrays": arrays,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mtp-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--split-manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_m3_derived_shards(
        args.pool_root,
        args.capture_dir,
        args.mtp_dir,
        args.output_dir,
        expected_split=args.split,
        vocab_size=args.vocab_size,
        split_manifest_path=args.split_manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAUSAL_ARRAYS",
    "ALIGNMENT_ARRAYS",
    "LABEL_ONLY_ARRAYS",
    "M3_DEPTHS",
    "M3_DERIVED_SCHEMA",
    "build_m3_derived_shards",
]
