#!/usr/bin/env python3
"""Build a compact cache of existing counterfactual branch trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
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
from harp_rtt.node_counterfactual import (  # noqa: E402
    NodeCounterfactualDatasetAdapter,
    load_node_counterfactual_companion,
)
from harp_rtt.path_route_tree import reconstruct_tree_token_prefixes  # noqa: E402
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.training import sha256_file  # noqa: E402


SCHEMA = "harp_branch_surrogate_cache_v1"
ROLES = (
    "post_attention_residual_u", "post_moe_residual_xplus",
    "routed_expert_output_delta_r", "shared_expert_output_delta_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-request-manifest", type=Path, required=True)
    parser.add_argument("--adaptive-fitting-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def request_id(dataset: HarpRTTDataset, index: int) -> str:
    record = dataset.records[index]
    return str(dataset.segments[record.segment].sequences[record.sequence]["request_id"])


def source_lineage_map(events: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in events.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "sequence_start":
            result[str(event["request_id"])] = str(event["source_request_id"])
    return result


def create_arrays(root: Path, rows: int, state_rank: int) -> dict[str, np.memmap[Any, Any]]:
    specs = {
        "state_coordinates": ((rows, 4, 40, state_rank), np.float16),
        "current_queries": ((rows, 40, 255), np.float16),
        "history_selected_ids": ((rows, 8, 40, 8), np.int16),
        "history_selected_weights": ((rows, 8, 40, 8), np.float16),
        "path_token_ids": ((rows, 32, 4), np.int32),
        "node_depth": ((rows, 32), np.int8),
        "node_mask": ((rows, 32), np.uint8),
        "budget16_mask": ((rows, 32), np.uint8),
        "target_queries": ((rows, 32, 40, 255), np.float16),
        "target_selected_ids": ((rows, 32, 40, 8), np.int16),
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


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite branch cache {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA branch-cache projection requested but unavailable")
    base = HarpRTTDataset(
        args.index, "train", corpus_root=args.corpus, max_tree_nodes=32
    )
    labels, companion_manifest = load_node_counterfactual_companion(
        args.companion, split="train", training=True
    )
    dataset = NodeCounterfactualDatasetAdapter(
        base, labels, split="train", training=True
    )
    if len(dataset) != 4096:
        raise ValueError("branch fitting cache requires exactly 4,096 source rows")

    reuse = json.loads(args.reuse_split_manifest.read_text())
    train_requests = list(reuse["inner_split"]["training_requests"])
    tune_requests = list(reuse["inner_split"]["tuning_requests"])
    train_set, tune_set = set(train_requests), set(tune_requests)
    if len(train_set) != 224 or len(tune_set) != 32 or train_set & tune_set:
        raise PermissionError("branch fitting request split is not 224/32 disjoint")
    observed = [request_id(base, index) for index in range(len(base))]
    counts = Counter(observed)
    if set(counts) != train_set | tune_set or set(counts.values()) != {16}:
        raise ValueError("branch fitting rows do not match the frozen 256x16 split")
    lineage = source_lineage_map(args.adaptive_fitting_events)
    if set(lineage) != set(counts):
        raise ValueError("branch fitting source-lineage map is incomplete")
    diagnostic_sources = {
        str(json.loads(line)["request_id"])
        for line in args.diagnostic_request_manifest.read_text().splitlines()
        if line.strip()
    }
    fitting_sources = {lineage[request] for request in counts}
    if len(diagnostic_sources) != 128 or fitting_sources & diagnostic_sources:
        raise PermissionError("branch fitting and diagnostic source lineages overlap")

    preprocessing = torch.load(
        args.target_preprocessing, map_location="cpu", weights_only=True
    )
    means = preprocessing["local_means"].float().to(device)
    components = preprocessing["local_components"].float().to(device)
    state_rank = int(components.shape[-1])
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    input_basis = static.geometry.input_basis.float().to(device)
    rank_mask = static.geometry.rank_mask.float().to(device)

    args.output.mkdir(parents=True)
    arrays = create_arrays(args.output, len(dataset), state_rank)
    request_order = train_requests + tune_requests
    request_to_index = {request: index for index, request in enumerate(request_order)}
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda", persistent_workers=False,
        collate_fn=collate_harp_rtt,
    )
    cursor = 0
    with torch.no_grad():
        for batch_index, host in enumerate(loader):
            inputs, targets = host["inputs"], host["targets"]
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
            tree = inputs["tree"]
            prefixes, prefix_mask = reconstruct_tree_token_prefixes(
                tree["token_ids"].to(device), tree["parent"].to(device),
                tree["depth"].to(device), tree["mask"].to(device), horizons=4,
            )
            counterfactual = targets["counterfactual"]
            node_mask = counterfactual["node_mask"].bool()
            valid = counterfactual["valid"].bool()
            if bool((node_mask[..., None] != valid).any()):
                raise ValueError("counterfactual layer validity is not node-constant")
            if bool((node_mask & (counterfactual["depth"] < 2)).any()):
                raise ValueError("counterfactual cache unexpectedly exposes H1 labels")
            budget16 = counterfactual["budget_node_masks"][:, 2].bool()
            if bool((budget16 & ~node_mask).any()):
                raise ValueError("budget-16 selects an unavailable node")
            if bool((prefix_mask.sum(-1) != tree["depth"].to(device)).any()):
                raise ValueError("tree prefix reconstruction disagrees with depth")

            size = states.shape[0]
            target_slice = slice(cursor, cursor + size)
            arrays["state_coordinates"][target_slice] = state_coordinates.cpu().numpy().astype(np.float16)
            arrays["current_queries"][target_slice] = current_queries.cpu().numpy().astype(np.float16)
            arrays["history_selected_ids"][target_slice] = inputs["history"]["selected_ids"].numpy().astype(np.int16)
            arrays["history_selected_weights"][target_slice] = inputs["history"]["execution_weights"].numpy().astype(np.float16)
            arrays["path_token_ids"][target_slice] = prefixes.cpu().numpy().astype(np.int32)
            arrays["node_depth"][target_slice] = counterfactual["depth"].numpy().astype(np.int8)
            arrays["node_mask"][target_slice] = node_mask.numpy().astype(np.uint8)
            arrays["budget16_mask"][target_slice] = budget16.numpy().astype(np.uint8)
            arrays["target_queries"][target_slice] = counterfactual["query_coordinates"].numpy().astype(np.float16)
            arrays["target_selected_ids"][target_slice] = counterfactual["selected_ids"].numpy().astype(np.int16)
            requests = [str(value) for value in host["metadata"]["request_id"]]
            arrays["request_index"][target_slice] = np.asarray(
                [request_to_index[value] for value in requests], dtype=np.int16
            )
            arrays["source_position"][target_slice] = np.asarray(
                host["metadata"]["position"], dtype=np.int32
            )
            arrays["split"][target_slice] = np.asarray(
                [0 if value in train_set else 1 for value in requests], dtype=np.uint8
            )
            cursor += size
            if batch_index % 32 == 0:
                write_json(args.output / "PROGRESS.json", {
                    "schema": SCHEMA, "rows_written": cursor,
                    "rows_total": len(dataset), "complete": False,
                }, exclusive=False)
    if cursor != len(dataset):
        raise RuntimeError("branch cache writer ended at the wrong row")
    for array in arrays.values():
        array.flush()
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outer_split": "train", "rows": len(dataset),
        "train_rows": int((arrays["split"] == 0).sum()),
        "tune_rows": int((arrays["split"] == 1).sum()),
        "train_requests": train_requests, "tune_requests": tune_requests,
        "fitting_source_requests": sorted(fitting_sources),
        "excluded_development_source_requests": sorted(diagnostic_sources),
        "source_lineage_enforced": True,
        "state_roles": list(ROLES), "state_rank": state_rank,
        "node_slots": 32, "h1_masked": True,
        "counterfactual_nodes": int(arrays["node_mask"].sum()),
        "budget16_nodes": int(arrays["budget16_mask"].sum()),
        "source_index_summary_sha256": sha256_file(args.index / "INDEX_SUMMARY.json"),
        "companion_manifest_sha256": sha256_file(args.companion / "manifest.json"),
        "companion_audit_sha256": sha256_file(args.companion / "COUNTERFACTUAL_AUDIT.json"),
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "diagnostic_request_manifest_sha256": sha256_file(args.diagnostic_request_manifest),
        "adaptive_fitting_events_sha256": sha256_file(args.adaptive_fitting_events),
        "target_preprocessing_sha256": sha256_file(args.target_preprocessing),
        "static_manifest_sha256": sha256_file(args.static_dir / "manifest.json"),
        "counterfactual_labels_are_training_only": True,
        "formal_validation_opened": False, "calibration_opened": False,
        "sealed_test_opened": False, "complete": True,
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "PROGRESS.json", {
        "schema": SCHEMA, "rows_written": cursor,
        "rows_total": len(dataset), "complete": True,
    }, exclusive=False)


if __name__ == "__main__":
    main()
