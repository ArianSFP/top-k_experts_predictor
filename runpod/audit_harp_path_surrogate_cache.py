#!/usr/bin/env python3
"""Fail-closed audit for a completed HARP token-path surrogate cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from harp_rtt.training import sha256_file  # noqa: E402


SCHEMA = "harp_path_surrogate_cache_audit_v1"
ARRAY_SPECS = {
    "state_coordinates": ((None, 4, 40, None), np.dtype("float16")),
    "current_queries": ((None, 40, 255), np.dtype("float16")),
    "history_selected_ids": ((None, 8, 40, 8), np.dtype("int16")),
    "history_selected_weights": ((None, 8, 40, 8), np.dtype("float16")),
    "path_token_ids": ((None, 4), np.dtype("int32")),
    "target_centered_logits": ((None, 4, 40, 256), np.dtype("float16")),
    "target_selected_ids": ((None, 4, 40, 8), np.dtype("int16")),
    "request_index": ((None,), np.dtype("int16")),
    "source_position": ((None,), np.dtype("int32")),
    "split": ((None,), np.dtype("uint8")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=128)
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def expected_shape(template: tuple[int | None, ...], rows: int, rank: int) -> tuple[int, ...]:
    result = []
    for axis, value in enumerate(template):
        if value is None:
            result.append(rows if axis == 0 else rank)
        else:
            result.append(value)
    return tuple(result)


def main() -> None:
    args = parse_args()
    if args.chunk_rows < 1:
        raise ValueError("chunk rows must be positive")
    audit_path = args.cache / "CACHE_AUDIT.json"
    sums_path = args.cache / "SHA256SUMS"
    if audit_path.exists() or sums_path.exists():
        raise FileExistsError("refusing to overwrite cache audit artifacts")
    manifest_path = args.cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "harp_path_surrogate_cache_v1" or manifest.get("complete") is not True:
        raise ValueError("cache manifest is incomplete or incompatible")
    for flag in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if manifest.get(flag) is not False:
            raise PermissionError(f"cache violates {flag}")
    if manifest.get("outer_split") != "train" or manifest.get("counterfactual_labels_read") is not False:
        raise PermissionError("cache is not factual outer-train-only data")

    rows = int(manifest["rows"]); rank = int(manifest["state_rank"])
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    hashes: dict[str, str] = {}
    for name, (template, dtype) in ARRAY_SPECS.items():
        path = args.cache / f"{name}.npy"
        value = np.load(path, mmap_mode="r")
        shape = expected_shape(template, rows, rank)
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"{name} has {value.shape}/{value.dtype}, expected {shape}/{dtype}"
            )
        arrays[name] = value
        hashes[path.name] = sha256_file(path)

    finite_names = (
        "state_coordinates", "current_queries", "history_selected_weights",
        "target_centered_logits",
    )
    native_hits = native_slots = 0
    for start in range(0, rows, args.chunk_rows):
        stop = min(rows, start + args.chunk_rows)
        for name in finite_names:
            if not np.isfinite(arrays[name][start:stop]).all():
                raise ValueError(f"{name} contains NaN or Inf")
        history_ids = arrays["history_selected_ids"][start:stop]
        target_ids = arrays["target_selected_ids"][start:stop]
        if (history_ids < 0).any() or (history_ids >= 256).any():
            raise ValueError("history expert IDs lie outside [0,255]")
        if (target_ids < 0).any() or (target_ids >= 256).any():
            raise ValueError("target expert IDs lie outside [0,255]")
        if (arrays["path_token_ids"][start:stop] < 0).any():
            raise ValueError("path token IDs contain a negative sentinel")
        logits = arrays["target_centered_logits"][start:stop]
        predicted = np.argsort(-logits, axis=-1, kind="stable")[..., :8]
        native_hits += int((
            target_ids[..., None] == predicted[..., None, :]
        ).any(-1).sum())
        native_slots += int(target_ids.size)

    split = np.asarray(arrays["split"])
    if not np.isin(split, (0, 1)).all():
        raise ValueError("cache split contains an undeclared value")
    if int((split == 0).sum()) != int(manifest["train_rows"]) or int((split == 1).sum()) != int(manifest["tune_rows"]):
        raise ValueError("cache split counts differ from manifest")
    train_requests = list(manifest["train_requests"])
    tune_requests = list(manifest["tune_requests"])
    if set(train_requests) & set(tune_requests):
        raise PermissionError("cache train and tuning request sets overlap")
    request_index = np.asarray(arrays["request_index"])
    if (request_index < 0).any() or (request_index >= len(train_requests) + len(tune_requests)).any():
        raise ValueError("cache request index is outside the manifest namespace")
    expected_split = (request_index >= len(train_requests)).astype(np.uint8)
    if not np.array_equal(split, expected_split):
        raise PermissionError("row split does not agree with request grouping")
    pairs = np.stack((request_index.astype(np.int64), np.asarray(arrays["source_position"])), axis=1)
    if len(np.unique(pairs, axis=0)) != rows:
        raise ValueError("cache contains duplicate request/source-position rows")

    hashes[manifest_path.name] = sha256_file(manifest_path)
    audit = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "builder_sha256": sha256_file(args.builder),
        "manifest_sha256": hashes[manifest_path.name],
        "rows": rows,
        "train_requests": len(train_requests),
        "tune_requests": len(tune_requests),
        "request_group_disjoint": True,
        "unique_request_position_rows": True,
        "native_top8_agreement_from_fp16_logits": native_hits / native_slots,
        "array_sha256": hashes,
        "complete": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(audit_path, audit)
    hashes[audit_path.name] = sha256_file(audit_path)
    with sums_path.open("x", encoding="utf-8") as handle:
        for name, digest in sorted(hashes.items()):
            handle.write(f"{digest}  {name}\n")
        handle.flush(); os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
