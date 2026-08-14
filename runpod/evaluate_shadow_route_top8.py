#!/usr/bin/env python3
"""Evaluate ShadowRoute on immutable adaptive trees with an exact target prefix."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "runpod" / "transformers_mtp_bridge"
for candidate in (str(REPO_ROOT), str(BRIDGE)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from harp_rtt.node_counterfactual import load_node_counterfactual_companion  # noqa: E402
from harp_rtt.shadow_backbone import exact_prefix_experts, install_shadow_experts  # noqa: E402
from harp_rtt.shadow_bundle import load_shadow_bundle  # noqa: E402
from harp_rtt.shadow_checkpoint import IndexedCheckpoint, sha256_file  # noqa: E402
from harp_rtt.shadow_rollout import ShadowRouteHooks, run_shadow_tree  # noqa: E402


SCHEMA = "harp_shadowroute_closed_loop_evaluation_v1"
RESULT_SCHEMA = "harp_shadowroute_closed_loop_result_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-capture", type=Path, required=True)
    parser.add_argument("--native-companion", type=Path, required=True)
    parser.add_argument("--layer-checkpoint-root", type=Path)
    parser.add_argument("--s1-fallback-root", type=Path)
    parser.add_argument(
        "--mode",
        choices=("exact_top1_plus_draft", "shared_width128", "indexed_width16"),
        required=True,
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--native-parity", action="store_true")
    parser.add_argument(
        "--native-parity-only",
        action="store_true",
        help=(
            "audit exact prefix/cache replay against authoritative native routes "
            "without loading a learned ShadowRoute bundle"
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_checksums(output: Path) -> None:
    paths = sorted(path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())


def route_overlap(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape or predicted.shape[-1] != 8:
        raise ValueError("route overlap requires aligned native top-eight sets")
    return (
        target[..., None] == predicted[..., None, :]
    ).any(-1).float().mean(-1)


def summarize(
    cells: Mapping[tuple[str, int], list[torch.Tensor]]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    for (request, horizon), values in sorted(cells.items()):
        rows.append(
            {
                "request_id": request,
                "horizon": horizon,
                "route_recall_at_8": float(torch.cat(values).mean()),
            }
        )
    metrics = {}
    for horizon in (2, 3, 4):
        values = [
            float(row["route_recall_at_8"])
            for row in rows if row["horizon"] == horizon
        ]
        if not values:
            raise ValueError(f"closed-loop evaluation has no H{horizon} rows")
        metrics[f"route_recall_h{horizon}"] = sum(values) / len(values)
    metrics["route_recall_h2_h4"] = sum(
        metrics[f"route_recall_h{horizon}"] for horizon in (2, 3, 4)
    ) / 3.0
    return metrics, rows


def main() -> None:
    args = parse_args()
    try:
        from capture_counterfactual_companion import frozen_nodes, load_base_capture
        from capture_transformers_segment import load_target, prefix_hash
    except ImportError as exc:
        raise RuntimeError(
            "closed-loop evaluation requires the pinned Qwen3.5-MoE Transformers build"
        ) from exc
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation {args.output}")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    if args.limit is not None and args.limit < 1:
        raise ValueError("evaluation limit must be positive")
    if args.native_parity_only:
        args.native_parity = True
    elif args.layer_checkpoint_root is None:
        raise ValueError("learned evaluation requires --layer-checkpoint-root")
    if args.mode == "indexed_width16" and args.s1_fallback_root is None:
        raise ValueError("S2 evaluation requires the frozen S1 fallback bundle")

    base_manifest, trees, sequence_tokens = load_base_capture(args.base_capture)
    labels, companion_manifest = load_node_counterfactual_companion(
        args.native_companion, split="train", training=True
    )
    if args.limit is not None:
        trees = trees[: args.limit]
    if not trees:
        raise ValueError("closed-loop evaluation contains no trees")
    for tree in trees:
        key = (str(tree["request_id"]), int(tree["source_position"]))
        if key not in labels:
            raise ValueError(f"native companion lacks tree join {key}")
        if tree.get("assigned_split") != "train" or tree.get("external_evaluation") is not False:
            raise PermissionError("closed-loop evaluation is restricted to outer-train")

    checkpoint = IndexedCheckpoint(args.model)
    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "mode": args.mode,
        "trees": len(trees),
        "native_parity_requested": args.native_parity,
        "native_parity_only": args.native_parity_only,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "base_capture_manifest_sha256": sha256_file(args.base_capture / "run_manifest.json"),
        "native_companion_manifest_sha256": sha256_file(args.native_companion / "manifest.json"),
        "label_only_native_routes": True,
        "native_routes_used_as_model_inputs": False,
        "optimizer_constructed": False,
        "training_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "model_revision": base_manifest.get("model_revision"),
        "companion_schema": companion_manifest.get("schema"),
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)

    target, _config = load_target(args.model, device=args.device)
    installed = install_shadow_experts(
        target, args.mode, retain_native=True, shadow_width=16
    )
    if not args.native_parity_only:
        assert args.layer_checkpoint_root is not None
        layer_paths = load_shadow_bundle(
            installed,
            args.layer_checkpoint_root,
            source_commit=args.source_commit,
            target_checkpoint_index_sha256=checkpoint.index_sha256,
            s1_fallback_root=args.s1_fallback_root,
        )
        write_json_exclusive(
            args.output / "BUNDLE_AUDIT.json",
            {
                "layers": len(layer_paths),
                "checkpoint_sha256": [sha256_file(path) for path in layer_paths],
                "complete": len(layer_paths) == 40,
            },
        )

    cells: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    native_mismatches = 0
    peak_gib = 0.0
    with ShadowRouteHooks(target) as hooks:
        for ordinal, tree in enumerate(trees):
            tokens = sequence_tokens.get(tree["sequence_id"])
            if tokens is None:
                raise ValueError("base capture sequence has no committed ending")
            position = int(tree["source_position"])
            authoritative_prefix = tokens[: position + 1]
            if prefix_hash(authoritative_prefix) != tree["authoritative_prefix_hash"]:
                raise ValueError("authoritative prefix hash changed")
            nodes = frozen_nodes(tree["rows"])
            token_ids = torch.tensor([node.token_id for node in nodes], dtype=torch.int64)
            parents = torch.tensor([
                -1 if node.parent_local_index is None else node.parent_local_index
                for node in nodes
            ], dtype=torch.int64)
            node_mask = torch.ones(len(nodes), dtype=torch.bool)
            native = labels[(str(tree["request_id"]), position)]
            count = int(native["node_mask"].sum())
            if count != len(nodes):
                raise ValueError("native companion/base tree node count differs")

            if args.native_parity:
                with exact_prefix_experts(installed):
                    reference = run_shadow_tree(
                        installed, hooks, authoritative_prefix=authoritative_prefix,
                        token_ids=token_ids, parent_indices=parents, node_mask=node_mask,
                    )
                active = native["valid"][:count]
                native_mismatches += int(
                    (
                        reference.selected_ids[:count].cpu()[active]
                        != native["selected_ids"][:count].long()[active]
                    ).sum()
                )
                if native_mismatches:
                    raise RuntimeError("exact-prefix native cache parity failed")
                del reference

            if not args.native_parity_only:
                result = run_shadow_tree(
                    installed, hooks, authoritative_prefix=authoritative_prefix,
                    token_ids=token_ids, parent_indices=parents, node_mask=node_mask,
                )
                predicted = result.selected_ids[:count].cpu()
                truth = native["selected_ids"][:count].long()
                depth = native["depth"][:count].long()
                valid = native["valid"][:count]
                overlap = route_overlap(predicted, truth)
                request = str(tree["source_request_id"] or tree["request_id"])
                for horizon in (2, 3, 4):
                    active = valid & (depth[:, None] == horizon)
                    if active.any():
                        cells[(request, horizon)].append(overlap[active])
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                peak_gib = max(peak_gib, torch.cuda.max_memory_reserved() / 2**30)
            print(json.dumps({"tree": ordinal + 1, "total": len(trees)}), flush=True)

    if args.native_parity_only:
        metrics: dict[str, float] = {}
        rows: list[dict[str, Any]] = []
    else:
        metrics, rows = summarize(cells)
    write_rows(args.output / "request_route_predictions.jsonl", rows)
    large_gain = args.native_parity_only or metrics["route_recall_h2_h4"] >= 0.85
    h4_gate = args.native_parity_only or metrics["route_recall_h4"] >= 0.80
    result = {
        "schema": RESULT_SCHEMA,
        "mode": args.mode,
        "native_parity_only": args.native_parity_only,
        "metrics": metrics,
        "native_parity_mismatches": native_mismatches,
        "peak_reserved_gib": peak_gib,
        "gate": {
            "route_recall_h2_h4_required": 0.85,
            "route_recall_h4_required": 0.80,
            "passed": bool(large_gain and h4_gate),
        },
        "training_started": False,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    write_checksums(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
