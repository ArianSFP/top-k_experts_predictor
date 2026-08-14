#!/usr/bin/env python3
"""Build a compact outer-train cache for token-conditioned route pretraining."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from torch.utils.data import DataLoader, Subset

from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt  # noqa: E402
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.training import sha256_file  # noqa: E402


SCHEMA = "harp_path_surrogate_cache_v1"
ROLES = (
    "post_attention_residual_u", "post_moe_residual_xplus",
    "routed_expert_output_delta_r", "shared_expert_output_delta_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--diagnostic-request-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--adaptive-fitting-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions-per-request", type=int, default=64)
    parser.add_argument("--tune-requests", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def request_id(dataset: HarpRTTDataset, index: int) -> str:
    record = dataset.records[index]
    return str(dataset.segments[record.segment].sequences[record.sequence]["request_id"])


def source_lineage_partitions(
    diagnostic_requests: Path,
    reuse_split_manifest: Path,
    adaptive_fitting_events: Path,
) -> tuple[set[str], set[str]]:
    development = {
        str(json.loads(line)["request_id"])
        for line in diagnostic_requests.read_text().splitlines() if line.strip()
    }
    reuse = json.loads(reuse_split_manifest.read_text())
    tuning_adaptive = set(reuse["inner_split"]["tuning_requests"])
    mapping: dict[str, str] = {}
    for line in adaptive_fitting_events.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "sequence_start":
            mapping[str(event["request_id"])] = str(event["source_request_id"])
    missing = tuning_adaptive - set(mapping)
    if missing:
        raise ValueError("adaptive tuning requests lack source-lineage mappings")
    tuning = {mapping[value] for value in tuning_adaptive}
    if len(development) != 128 or len(tuning) != 32:
        raise ValueError("source-lineage partitions must contain 128 development and 32 tuning requests")
    if development & tuning:
        raise PermissionError("source-lineage development and tuning requests overlap")
    return development, tuning


def deterministic_rows(
    dataset: HarpRTTDataset,
    *,
    positions_per_request: int,
    tune_request_ids: set[str],
    excluded_request_ids: set[str],
) -> tuple[list[int], list[int], list[str], list[str]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        grouped[request_id(dataset, index)].append(index)
    unavailable = (tune_request_ids | excluded_request_ids) - set(grouped)
    if unavailable:
        raise ValueError("source-lineage partition names unavailable corpus requests")
    if tune_request_ids & excluded_request_ids:
        raise PermissionError("tuning and excluded source requests overlap")
    tune = sorted(tune_request_ids)
    train = sorted(set(grouped) - tune_request_ids - excluded_request_ids)
    if not train or not tune:
        raise ValueError("source-lineage cache split is empty")

    def select(requests: list[str]) -> list[int]:
        result: list[int] = []
        for request in requests:
            values = grouped[request]
            if len(values) <= positions_per_request:
                chosen = values
            else:
                offsets = np.linspace(
                    0, len(values) - 1, positions_per_request,
                    dtype=np.int64,
                ).tolist()
                chosen = [values[int(offset)] for offset in offsets]
            if len(set(chosen)) != len(chosen):
                raise ValueError("uniform request selection produced duplicates")
            result.extend(chosen)
        return result

    return select(train), select(tune), train, tune


def create_arrays(root: Path, rows: int, *, state_rank: int) -> dict[str, np.memmap[Any, Any]]:
    specs = {
        "state_coordinates": ((rows, 4, 40, state_rank), np.float16),
        "current_queries": ((rows, 40, 255), np.float16),
        "history_selected_ids": ((rows, 8, 40, 8), np.int16),
        "history_selected_weights": ((rows, 8, 40, 8), np.float16),
        "path_token_ids": ((rows, 4), np.int32),
        "target_centered_logits": ((rows, 4, 40, 256), np.float16),
        "target_selected_ids": ((rows, 4, 40, 8), np.int16),
        "request_index": ((rows,), np.int16),
        "source_position": ((rows,), np.int32),
        "split": ((rows,), np.uint8),
    }
    return {
        name: np.lib.format.open_memmap(
            root / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (shape, dtype) in specs.items()
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite cache {args.output}")
    if args.positions_per_request < 1 or args.tune_requests < 2:
        raise ValueError("path-cache selection dimensions must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA cache projection requested but unavailable")
    dataset = HarpRTTDataset(
        args.index, "train", corpus_root=args.corpus, max_tree_nodes=1
    )
    excluded_requests, forced_tune_requests = source_lineage_partitions(
        args.diagnostic_request_manifest, args.reuse_split_manifest,
        args.adaptive_fitting_events,
    )
    if len(forced_tune_requests) != args.tune_requests:
        raise ValueError("declared tuning count differs from source-lineage partition")
    train_rows, tune_rows, train_requests, tune_requests = deterministic_rows(
        dataset, positions_per_request=args.positions_per_request,
        tune_request_ids=forced_tune_requests,
        excluded_request_ids=excluded_requests,
    )
    selected = train_rows + tune_rows
    split_values = [0] * len(train_rows) + [1] * len(tune_rows)
    if args.max_rows is not None:
        selected = selected[: args.max_rows]
        split_values = split_values[: len(selected)]
    if not selected:
        raise ValueError("path cache selection is empty")

    preprocessing = torch.load(
        args.target_preprocessing, map_location="cpu", weights_only=True
    )
    means = preprocessing["local_means"].float().to(device)
    components = preprocessing["local_components"].float().to(device)
    if means.shape != (40, 2048) or components.shape[:2] != (40, 2048):
        raise ValueError("target preprocessing geometry is incompatible")
    state_rank = int(components.shape[-1])
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    input_basis = static.geometry.input_basis.float().to(device)
    rank_mask = static.geometry.rank_mask.float().to(device)

    args.output.mkdir(parents=True)
    arrays = create_arrays(args.output, len(selected), state_rank=state_rank)
    request_order = train_requests + tune_requests
    request_to_index = {request: index for index, request in enumerate(request_order)}
    loader = DataLoader(
        Subset(dataset, selected), batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
        persistent_workers=False, collate_fn=collate_harp_rtt,
    )
    cursor = 0
    with torch.no_grad():
        for batch_index, host in enumerate(loader):
            inputs = host["inputs"]; targets = host["targets"]
            current = inputs["current"]
            states = torch.stack([
                current[role].to(device, non_blocking=True) for role in ROLES
            ], dim=1).float()
            state_coordinates = torch.einsum(
                "bqld,ldr->bqlr", states - means[None, None], components
            )
            router_input = current["normalized_target_router_input_a"].to(
                device, non_blocking=True
            ).float()
            current_queries = torch.einsum(
                "bld,ldr->blr", router_input, input_basis
            ) * rank_mask[None]
            centered = targets["future_centered_router_logits"].float()
            size = states.shape[0]; target_slice = slice(cursor, cursor + size)
            arrays["state_coordinates"][target_slice] = state_coordinates.cpu().numpy().astype(np.float16)
            arrays["current_queries"][target_slice] = current_queries.cpu().numpy().astype(np.float16)
            arrays["history_selected_ids"][target_slice] = inputs["history"]["selected_ids"].numpy().astype(np.int16)
            arrays["history_selected_weights"][target_slice] = inputs["history"]["execution_weights"].numpy().astype(np.float16)
            arrays["path_token_ids"][target_slice] = targets["future_meta"][..., 3].numpy().astype(np.int32)
            arrays["target_centered_logits"][target_slice] = centered.numpy().astype(np.float16)
            arrays["target_selected_ids"][target_slice] = targets["future_selected_ids"].numpy().astype(np.int16)
            requests = [str(value) for value in host["metadata"]["request_id"]]
            arrays["request_index"][target_slice] = np.asarray(
                [request_to_index[value] for value in requests], dtype=np.int16
            )
            arrays["source_position"][target_slice] = np.asarray(
                host["metadata"]["position"], dtype=np.int32
            )
            arrays["split"][target_slice] = np.asarray(
                split_values[cursor : cursor + size], dtype=np.uint8
            )
            cursor += size
            if batch_index % 64 == 0:
                progress = {
                    "schema": SCHEMA, "rows_written": cursor,
                    "rows_total": len(selected), "complete": False,
                }
                progress_path = args.output / "PROGRESS.json"
                with progress_path.open("w", encoding="utf-8") as handle:
                    json.dump(progress, handle, sort_keys=True)
                    handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    if cursor != len(selected):
        raise RuntimeError("path cache writer ended at the wrong row")
    for array in arrays.values():
        array.flush()
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outer_split": "train", "seed": args.seed,
        "positions_per_request": args.positions_per_request,
        "rows": len(selected), "train_rows": split_values.count(0),
        "tune_rows": split_values.count(1),
        "train_requests": train_requests, "tune_requests": tune_requests,
        "excluded_development_requests": sorted(excluded_requests),
        "source_lineage_enforced": True,
        "diagnostic_request_manifest_sha256": sha256_file(args.diagnostic_request_manifest),
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "adaptive_fitting_events_sha256": sha256_file(args.adaptive_fitting_events),
        "state_roles": list(ROLES), "state_rank": state_rank,
        "source_index_summary_sha256": sha256_file(args.index / "INDEX_SUMMARY.json"),
        "source_corpus_audit_sha256": sha256_file(args.corpus.parent / "CORPUS_AUDIT.json"),
        "target_preprocessing_sha256": sha256_file(args.target_preprocessing),
        "static_manifest_sha256": sha256_file(args.static_dir / "manifest.json"),
        "factual_path_tokens_are_teacher_inputs": True,
        "counterfactual_labels_read": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "complete": True,
    }
    write_json(args.output / "manifest.json", manifest)
    progress_path = args.output / "PROGRESS.json"
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump({"schema": SCHEMA, "rows_written": cursor,
                   "rows_total": len(selected), "complete": True}, handle)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
