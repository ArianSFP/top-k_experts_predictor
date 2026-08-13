#!/usr/bin/env python3
"""Materialize causal adaptive-tree endpoint channels beside the label cache."""

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
import torch
from torch.utils.data import DataLoader

from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt  # noqa: E402
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.training import sha256_file  # noqa: E402


SCHEMA = "harp_branch_state_cache_v1"
AUDIT_SCHEMA = "harp_branch_state_cache_audit_v1"
SPECS = {
    "branch_states": ((4096, 32, 4, 2048), np.float16),
    "branch_router_logits": ((4096, 32, 256), np.float16),
    "branch_selected_ids": ((4096, 32, 8), np.int16),
    "branch_selected_weights": ((4096, 32, 8), np.float16),
    "branch_vocab_embedding": ((4096, 32, 2048), np.float16),
    "branch_vocab_statistics": ((4096, 32, 6), np.float16),
    "branch_scalars": ((4096, 32, 8), np.float16),
    "branch_mask": ((4096, 32), np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite branch-state cache {args.output}")
    base_manifest_path = args.base_cache / "manifest.json"
    base_audit_path = args.base_cache / "CACHE_AUDIT.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    base_audit = json.loads(base_audit_path.read_text())
    if (
        base_manifest.get("schema") != "harp_branch_surrogate_cache_v1"
        or base_manifest.get("complete") is not True
        or base_audit.get("schema") != "harp_branch_surrogate_cache_audit_v1"
        or base_audit.get("complete") is not True
        or base_audit.get("manifest_sha256") != sha256_file(base_manifest_path)
    ):
        raise ValueError("base counterfactual cache is incomplete or incompatible")
    for flag in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if base_manifest.get(flag) is not False or base_audit.get(flag) is not False:
            raise PermissionError(f"base counterfactual cache violates {flag}")

    dataset = HarpRTTDataset(
        args.index, "train", corpus_root=args.corpus, max_tree_nodes=32
    )
    if len(dataset) != 4096:
        raise ValueError("branch-state cache requires exactly 4,096 source rows")
    device = torch.device(args.device)
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    if static.token_embedding is None:
        raise RuntimeError("branch-state cache requires frozen token embeddings")
    token_embedding = static.token_embedding.float().to(device)
    request_order = list(base_manifest["train_requests"]) + list(
        base_manifest["tune_requests"]
    )
    request_to_index = {value: index for index, value in enumerate(request_order)}
    base_request_index = np.load(args.base_cache / "request_index.npy", mmap_mode="r")
    base_position = np.load(args.base_cache / "source_position.npy", mmap_mode="r")
    base_node_mask = np.load(args.base_cache / "node_mask.npy", mmap_mode="r")

    args.output.mkdir(parents=True)
    arrays = {
        name: np.lib.format.open_memmap(
            args.output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (shape, dtype) in SPECS.items()
    }
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda", persistent_workers=False,
        collate_fn=collate_harp_rtt,
    )
    cursor = 0
    with torch.no_grad():
        for batch_index, host in enumerate(loader):
            tree = host["inputs"]["tree"]
            size = tree["states"].shape[0]
            target_slice = slice(cursor, cursor + size)
            requests = [str(value) for value in host["metadata"]["request_id"]]
            observed_request_index = np.asarray(
                [request_to_index[value] for value in requests], dtype=np.int16
            )
            observed_position = np.asarray(host["metadata"]["position"], dtype=np.int32)
            if not np.array_equal(observed_request_index, base_request_index[target_slice]):
                raise ValueError("branch-state/base-cache request order differs")
            if not np.array_equal(observed_position, base_position[target_slice]):
                raise ValueError("branch-state/base-cache source positions differ")
            mask = tree["mask"].bool()
            if not np.array_equal(mask.numpy().astype(np.uint8), base_node_mask[target_slice]):
                raise ValueError("branch-state/base-cache node masks differ")
            vocab_ids = tree["vocab_top64_ids"].to(device, non_blocking=True).long()
            vocab_logp = tree["vocab_top64_log_probabilities"].to(
                device, non_blocking=True
            ).float()
            if bool(((vocab_ids < 0) | (vocab_ids >= token_embedding.shape[0]))[mask.to(device)].any()):
                raise ValueError("active branch vocabulary ID is out of range")
            probability = torch.softmax(vocab_logp, dim=-1)
            vocab = (
                token_embedding[vocab_ids] * probability[..., None]
            ).sum(-2)
            vocab = vocab * mask.to(device)[..., None]
            arrays["branch_states"][target_slice] = tree["states"].numpy().astype(np.float16)
            arrays["branch_router_logits"][target_slice] = tree["router_logits"].numpy().astype(np.float16)
            arrays["branch_selected_ids"][target_slice] = tree["selected_ids"].numpy().astype(np.int16)
            arrays["branch_selected_weights"][target_slice] = tree["execution_weights"].numpy().astype(np.float16)
            arrays["branch_vocab_embedding"][target_slice] = vocab.cpu().numpy().astype(np.float16)
            arrays["branch_vocab_statistics"][target_slice] = tree["vocab_statistics"].numpy().astype(np.float16)
            arrays["branch_scalars"][target_slice] = tree["scalars"].numpy().astype(np.float16)
            arrays["branch_mask"][target_slice] = mask.numpy().astype(np.uint8)
            cursor += size
            if batch_index % 32 == 0:
                write_json(args.output / "PROGRESS.json", {
                    "schema": SCHEMA, "rows_written": cursor,
                    "rows_total": len(dataset), "complete": False,
                }, exclusive=False)
    if cursor != len(dataset):
        raise RuntimeError("branch-state cache writer ended at the wrong row")
    for array in arrays.values():
        array.flush()
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "rows": len(dataset), "node_slots": 32,
        "base_cache_manifest_sha256": sha256_file(base_manifest_path),
        "base_cache_audit_sha256": sha256_file(base_audit_path),
        "source_index_summary_sha256": sha256_file(args.index / "INDEX_SUMMARY.json"),
        "static_manifest_sha256": sha256_file(args.static_dir / "manifest.json"),
        "vocabulary_pool": "normalized_top64_probability_weighted_frozen_embedding",
        "causal_inputs_only": True, "counterfactual_labels_read": False,
        "formal_validation_opened": False, "calibration_opened": False,
        "sealed_test_opened": False, "complete": True,
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "PROGRESS.json", {
        "schema": SCHEMA, "rows_written": cursor,
        "rows_total": len(dataset), "complete": True,
    }, exclusive=False)

    hashes: dict[str, str] = {}
    for name, (shape, dtype) in SPECS.items():
        path = args.output / f"{name}.npy"
        array = np.load(path, mmap_mode="r")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"{name} failed shape/dtype audit")
        hashes[path.name] = sha256_file(path)
    if not np.array_equal(np.asarray(arrays["branch_mask"]), np.asarray(base_node_mask)):
        raise ValueError("final branch-state node mask differs from label cache")
    for name in (
        "branch_states", "branch_router_logits", "branch_selected_weights",
        "branch_vocab_embedding", "branch_vocab_statistics", "branch_scalars",
    ):
        if not np.isfinite(np.asarray(arrays[name])).all():
            raise ValueError(f"{name} contains NaN/Inf")
    selected = np.asarray(arrays["branch_selected_ids"])[np.asarray(base_node_mask).astype(bool)]
    if (selected < 0).any() or (selected >= 256).any():
        raise ValueError("active branch expert ID is outside [0,255]")
    hashes["manifest.json"] = sha256_file(args.output / "manifest.json")
    audit = {
        "schema": AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit, "rows": len(dataset),
        "manifest_sha256": hashes["manifest.json"],
        "base_cache_masks_exact": True, "finite": True,
        "counterfactual_labels_read": False, "complete": True,
        "formal_validation_opened": False, "calibration_opened": False,
        "sealed_test_opened": False, "array_sha256": hashes,
    }
    write_json(args.output / "CACHE_AUDIT.json", audit)
    hashes["CACHE_AUDIT.json"] = sha256_file(args.output / "CACHE_AUDIT.json")
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for name, digest in sorted(hashes.items()):
            handle.write(f"{digest}  {name}\n")
        handle.flush(); os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
