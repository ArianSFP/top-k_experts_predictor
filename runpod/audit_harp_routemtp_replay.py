#!/usr/bin/env python3
"""Blocking 32-position adapter-off RouteMTP replay parity audit."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "runpod" / "transformers_mtp_bridge"
for value in (REPO_ROOT, BRIDGE):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from harp_rtt.dataset import HarpRTTDataset  # noqa: E402
from harp_rtt.node_counterfactual import (  # noqa: E402
    NodeCounterfactualDatasetAdapter,
    load_node_counterfactual_companion,
)
from harp_rtt.routemtp import install_cache_coherent_adapters  # noqa: E402
from harp_rtt.routemtp_cache import (  # noqa: E402
    ROUTEMTP_CACHE_SCHEMA,
    RouteMTPCacheGeometry,
    RouteMTPSourceOffset,
    load_causal_cache_slice,
    restore_transformers_dynamic_cache,
    sha256_file,
)
from harp_rtt.routemtp_replay import RouteMTPTreeRunner  # noqa: E402
SCHEMA = "harp_routemtp_replay_parity_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--hydration", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--maximum-state-error", type=float, default=0.02)
    parser.add_argument("--maximum-logit-error", type=float, default=0.02)
    parser.add_argument("--maximum-selected-weight-error", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _geometry(value: Mapping[str, Any]) -> RouteMTPCacheGeometry:
    names = {field.name for field in fields(RouteMTPCacheGeometry)}
    if set(value) != names:
        raise ValueError("RouteMTP hydration cache geometry keys differ")
    result = RouteMTPCacheGeometry(**dict(value)); result.validate(); return result


def _load_manifest(path: Path) -> tuple[dict[str, Any], RouteMTPCacheGeometry]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != ROUTEMTP_CACHE_SCHEMA:
        raise ValueError("RouteMTP hydration manifest schema mismatch")
    for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if value.get(key) is not False:
            raise PermissionError(f"RouteMTP hydration violates {key}")
    if value.get("outer_split") != "train" or value.get("causal_slice_required") is not True:
        raise PermissionError("RouteMTP hydration is not causal outer-train data")
    return value, _geometry(value["geometry"])


def main() -> None:
    args = parse_args()
    from qwen35_mtp import load_checkpoint_mtp
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite RouteMTP parity report {args.output}")
    if args.positions != 32 and not args.diagnostic_only:
        raise ValueError("blocking RouteMTP Stage A is exactly 32 positions")
    if args.positions < 1:
        raise ValueError("RouteMTP parity requires at least one position")
    manifest_path = args.hydration / "HYDRATION_MANIFEST.json"
    manifest, geometry = _load_manifest(manifest_path)
    records = {str(value["request_id"]): value for value in manifest["records"]}
    offsets = {
        (str(value["request_id"]), int(value["source_position"])):
        RouteMTPSourceOffset(**value)
        for value in manifest["source_offsets"]
    }
    base = HarpRTTDataset(args.index, "train", corpus_root=args.corpus, max_tree_nodes=32)
    labels, companion_manifest = load_node_counterfactual_companion(
        args.companion, split="train", training=True
    )
    dataset = NodeCounterfactualDatasetAdapter(
        base, labels, split="train", training=True
    )
    if len(dataset) < args.positions:
        raise ValueError("RouteMTP parity dataset has fewer than 32 positions")

    mtp = load_checkpoint_mtp(args.model, device=args.device)
    adapters = install_cache_coherent_adapters(mtp, rank=32)
    runner = RouteMTPTreeRunner(mtp, adapters).to(args.device)
    maximum_state_error = 0.0; maximum_logit_error = 0.0
    selected_id_mismatches = 0; selected_weight_error = 0.0
    channel_state_errors = {
        "fused_state": 0.0,
        "post_moe_hidden": 0.0,
        "router_input": 0.0,
        "vocabulary_head_input": 0.0,
    }
    by_depth = {
        str(depth): {
            "nodes": 0,
            "selected_id_mismatches": 0,
            "maximum_state_error": 0.0,
            "maximum_logit_error": 0.0,
            "maximum_selected_weight_error": 0.0,
        }
        for depth in range(1, 5)
    }
    nodes_audited = 0; calls = 0
    for index in range(args.positions):
        item = dataset[index]
        metadata = item["metadata"]; inputs = item["inputs"]; tree = inputs["tree"]
        request_id = str(metadata["request_id"]); source_position = int(metadata["position"])
        key = (request_id, source_position)
        if key not in offsets or request_id not in records:
            raise KeyError(f"RouteMTP hydration lacks parity row {key}")
        offset = offsets[key]; record = records[request_id]
        tensors = load_causal_cache_slice(
            args.hydration / str(record["relative_path"]),
            geometry,
            offset,
            expected_sha256=str(record["sha256"]),
        )

        def cache_factory(_batch_index: int) -> Any:
            return restore_transformers_dynamic_cache(
                tensors, geometry, config=mtp.decoder.config, device=args.device
            )

        node_mask = tree["mask"][None].to(args.device)
        output = runner(
            node_token_ids=tree["token_ids"][None].long().to(args.device),
            parent_ids=tree["parent"][None].long().to(args.device),
            node_mask=node_mask,
            current_target_hidden=tensors["target_final_hidden"][-1:, :].to(args.device),
            base_cache_factory=cache_factory,
            base_cache_lengths=torch.tensor([offset.prefix_length], device=args.device),
            base_prefix_token_ids=[tensors["shifted_token_ids"]],
            base_target_hidden_history=[tensors["target_final_hidden"]],
        )
        active = node_mask[0]
        captured = tree["states"].to(args.device)
        pairs = (
            ("fused_state", output.fused_state[0], captured[:, 0]),
            ("post_moe_hidden", output.post_moe_hidden[0], captured[:, 1]),
            ("router_input", output.router_input[0], captured[:, 2]),
            ("vocabulary_head_input", output.vocabulary_head_input[0], captured[:, 3]),
        )
        state_errors = []
        for name, predicted, expected in pairs:
            error = (predicted[active].float() - expected[active].float()).abs().max()
            maximum_state_error = max(maximum_state_error, float(error.item()))
            channel_state_errors[name] = max(channel_state_errors[name], float(error.item()))
            state_errors.append((predicted[active].float() - expected[active].float()).abs())
        logits = tree["router_logits"].to(args.device)
        logit_errors = (output.router_logits[0, active].float() - logits[active].float()).abs()
        maximum_logit_error = max(maximum_logit_error, float(logit_errors.max().item()))
        expected_ids = tree["selected_ids"].long().to(args.device)[active]
        id_mismatch = (output.selected_ids[0, active] != expected_ids).any(-1)
        selected_id_mismatches += int(id_mismatch.sum())
        expected_weights = tree["execution_weights"].to(args.device)[active]
        weight_errors = (
            output.selected_weights[0, active].float() - expected_weights.float()
        ).abs()
        selected_weight_error = max(
            selected_weight_error,
            float(weight_errors.max().item()),
        )
        depths = tree["depth"].to(args.device)[active]
        for depth in range(1, 5):
            mask = depths == depth
            if not bool(mask.any()):
                continue
            bucket = by_depth[str(depth)]
            bucket["nodes"] += int(mask.sum())
            bucket["selected_id_mismatches"] += int(id_mismatch[mask].sum())
            bucket["maximum_state_error"] = max(
                bucket["maximum_state_error"],
                max(float(errors[mask].max().item()) for errors in state_errors),
            )
            bucket["maximum_logit_error"] = max(
                bucket["maximum_logit_error"], float(logit_errors[mask].max().item())
            )
            bucket["maximum_selected_weight_error"] = max(
                bucket["maximum_selected_weight_error"],
                float(weight_errors[mask].max().item()),
            )
        nodes_audited += int(active.sum().item()); calls += output.call_count

    passed = (
        selected_id_mismatches == 0
        and maximum_state_error <= args.maximum_state_error
        and maximum_logit_error <= args.maximum_logit_error
        and selected_weight_error <= args.maximum_selected_weight_error
    )
    result = {
        "schema": SCHEMA,
        "positions": args.positions,
        "nodes_audited": nodes_audited,
        "route_mtp_calls": calls,
        "maximum_state_error": maximum_state_error,
        "maximum_logit_error": maximum_logit_error,
        "maximum_selected_weight_error": selected_weight_error,
        "selected_id_mismatches": selected_id_mismatches,
        "channel_state_errors": channel_state_errors,
        "by_depth": by_depth,
        "state_tolerance": args.maximum_state_error,
        "logit_tolerance": args.maximum_logit_error,
        "selected_weight_tolerance": args.maximum_selected_weight_error,
        "passed": passed,
        "adapter_off": True,
        "kv_adapters_present": False,
        "training_started": False,
        "diagnostic_only": bool(args.diagnostic_only),
        "optimizer_constructed": False,
        "hydration_manifest_sha256": sha256_file(manifest_path),
        "source_commit": manifest.get("bindings", {}).get("source_commit"),
        "mtp_checkpoint_sha256": manifest.get("bindings", {}).get("mtp_checkpoint_sha256"),
        "counterfactual_companion_manifest_sha256": sha256_file(args.companion / "manifest.json")
        if (args.companion / "manifest.json").is_file() else None,
        "companion_sealed_test_opened": companion_manifest.get("sealed_test_opened"),
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed and not args.diagnostic_only:
        raise SystemExit("RouteMTP adapter-off replay parity failed")


if __name__ == "__main__":
    main()
