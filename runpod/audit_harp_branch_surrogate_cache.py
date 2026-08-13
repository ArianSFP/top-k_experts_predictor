#!/usr/bin/env python3
"""Fail-closed audit for the compact counterfactual branch cache."""

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


SCHEMA = "harp_branch_surrogate_cache_audit_v1"
SPECS = {
    "state_coordinates": ((4096, 4, 40, None), np.dtype("float16")),
    "current_queries": ((4096, 40, 255), np.dtype("float16")),
    "history_selected_ids": ((4096, 8, 40, 8), np.dtype("int16")),
    "history_selected_weights": ((4096, 8, 40, 8), np.dtype("float16")),
    "path_token_ids": ((4096, 32, 4), np.dtype("int32")),
    "node_depth": ((4096, 32), np.dtype("int8")),
    "node_mask": ((4096, 32), np.dtype("uint8")),
    "budget16_mask": ((4096, 32), np.dtype("uint8")),
    "target_queries": ((4096, 32, 40, 255), np.dtype("float16")),
    "target_selected_ids": ((4096, 32, 40, 8), np.dtype("int16")),
    "request_index": ((4096,), np.dtype("int16")),
    "source_position": ((4096,), np.dtype("int32")),
    "split": ((4096,), np.dtype("uint8")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=32)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    audit_path = args.cache / "CACHE_AUDIT.json"
    sums_path = args.cache / "SHA256SUMS"
    if audit_path.exists() or sums_path.exists():
        raise FileExistsError("refusing to overwrite branch-cache audit")
    manifest_path = args.cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "harp_branch_surrogate_cache_v1" or manifest.get("complete") is not True:
        raise ValueError("branch cache manifest is incomplete or incompatible")
    for flag in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if manifest.get(flag) is not False:
            raise PermissionError(f"branch cache violates {flag}")
    if (
        manifest.get("outer_split") != "train"
        or manifest.get("counterfactual_labels_are_training_only") is not True
        or manifest.get("source_lineage_enforced") is not True
    ):
        raise PermissionError("branch cache provenance is not train-only/source-disjoint")
    rank = int(manifest["state_rank"])
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    hashes: dict[str, str] = {}
    for name, (template, dtype) in SPECS.items():
        shape = tuple(rank if value is None else value for value in template)
        path = args.cache / f"{name}.npy"
        array = np.load(path, mmap_mode="r")
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"{name} has {array.shape}/{array.dtype}, expected {shape}/{dtype}")
        arrays[name] = array
        hashes[path.name] = sha256_file(path)

    active = total_budget = 0
    for start in range(0, 4096, args.chunk_rows):
        stop = min(4096, start + args.chunk_rows)
        for name in (
            "state_coordinates", "current_queries", "history_selected_weights",
            "target_queries",
        ):
            if not np.isfinite(arrays[name][start:stop]).all():
                raise ValueError(f"{name} contains NaN or Inf")
        history = arrays["history_selected_ids"][start:stop]
        target = arrays["target_selected_ids"][start:stop]
        if (history < 0).any() or (history >= 256).any():
            raise ValueError("history expert IDs lie outside [0,255]")
        mask = arrays["node_mask"][start:stop].astype(bool)
        budget = arrays["budget16_mask"][start:stop].astype(bool)
        depth = arrays["node_depth"][start:stop]
        if (budget & ~mask).any() or ((depth < 2) & mask).any() or ((depth > 4) & mask).any():
            raise ValueError("branch node masks/depths are invalid")
        if (arrays["path_token_ids"][start:stop][mask] < 0).any():
            raise ValueError("active branch prefixes contain negative tokens")
        target_active = target[mask]
        if (target_active < 0).any() or (target_active >= 256).any():
            raise ValueError("target expert IDs lie outside [0,255]")
        ordered = np.sort(target_active.reshape(-1, 8), axis=-1)
        if (np.diff(ordered, axis=-1) == 0).any():
            raise ValueError("target selected sets contain duplicate experts")
        active += int(mask.sum()); total_budget += int(budget.sum())
    if active != int(manifest["counterfactual_nodes"]) or total_budget != int(manifest["budget16_nodes"]):
        raise ValueError("branch node counts differ from manifest")

    split = np.asarray(arrays["split"])
    if int((split == 0).sum()) != 3584 or int((split == 1).sum()) != 512:
        raise ValueError("branch cache split is not 3584/512")
    train, tune = list(manifest["train_requests"]), list(manifest["tune_requests"])
    fitting_sources = set(manifest["fitting_source_requests"])
    excluded_sources = set(manifest["excluded_development_source_requests"])
    if len(train) != 224 or len(tune) != 32 or set(train) & set(tune):
        raise PermissionError("branch cache request split is invalid")
    if len(fitting_sources) != 256 or len(excluded_sources) != 128 or fitting_sources & excluded_sources:
        raise PermissionError("branch cache source lineages overlap")
    request_index = np.asarray(arrays["request_index"])
    if (request_index < 0).any() or (request_index >= 256).any():
        raise ValueError("branch request index is invalid")
    if not np.array_equal(split, (request_index >= 224).astype(np.uint8)):
        raise ValueError("branch row split differs from request grouping")
    pairs = np.stack((request_index.astype(np.int64), arrays["source_position"]), axis=1)
    if len(np.unique(pairs, axis=0)) != 4096:
        raise ValueError("branch cache contains duplicate request/position rows")

    hashes[manifest_path.name] = sha256_file(manifest_path)
    audit = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "builder_sha256": sha256_file(args.builder),
        "manifest_sha256": hashes[manifest_path.name],
        "rows": 4096, "train_rows": 3584, "tune_rows": 512,
        "train_requests": 224, "tune_requests": 32,
        "excluded_development_source_requests": 128,
        "counterfactual_nodes": active, "budget16_nodes": total_budget,
        "request_group_disjoint": True, "source_lineage_disjoint": True,
        "unique_request_position_rows": True, "complete": True,
        "formal_validation_opened": False, "calibration_opened": False,
        "sealed_test_opened": False, "array_sha256": hashes,
    }
    write_json(audit_path, audit)
    hashes[audit_path.name] = sha256_file(audit_path)
    with sums_path.open("x", encoding="utf-8") as handle:
        for name, digest in sorted(hashes.items()):
            handle.write(f"{digest}  {name}\n")
        handle.flush(); os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
