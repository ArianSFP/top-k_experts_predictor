#!/usr/bin/env python3
"""Stream a leak-safe real-corpus HARP-RTT geometry/tree audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt
from harp_rtt.exact_k import stable_topk
from harp_rtt.probes import tree_availability_probe
from harp_rtt.static_artifacts import load_static_target_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--static-artifacts", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "calibration"), default="validation"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-root",
        type=Path,
        help="optional pinned HF checkpoint for a native-BF16 router comparison",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.max_batches < 1:
        raise ValueError("batch size and max batches must be positive")
    dataset = HarpRTTDataset(
        args.index_root,
        args.split,
        corpus_root=args.corpus_root,
        allow_test=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_harp_rtt,
    )
    static = load_static_target_artifacts(
        args.static_artifacts,
        device=args.device,
        load_embedding=False,
    )
    geometry = static.geometry
    native_weights = None
    if args.model_root is not None:
        from safetensors import safe_open

        index = json.loads(
            (args.model_root / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        weight_map = index["weight_map"]
        keys = [
            f"model.language_model.layers.{layer}.mlp.gate.weight"
            for layer in range(geometry.layers)
        ]
        by_shard: dict[str, list[str]] = {}
        for key in keys:
            by_shard.setdefault(weight_map[key], []).append(key)
        loaded: dict[str, torch.Tensor] = {}
        for shard, shard_keys in by_shard.items():
            with safe_open(
                str(args.model_root / shard), framework="pt", device="cpu"
            ) as handle:
                for key in shard_keys:
                    loaded[key] = handle.get_tensor(key)
        native_weights = torch.stack([loaded[key] for key in keys]).to(args.device)
    totals: dict[str, float | int] = {
        "batches": 0,
        "samples": 0,
        "valid_endpoints": 0,
        "logit_values": 0,
        "squared_error": 0.0,
        "maximum_absolute_logit_error": 0.0,
        "slot_recall_sum": 0.0,
        "exact_set_sum": 0.0,
        "selected_label_recall_sum": 0.0,
        "native_squared_error": 0.0,
        "native_maximum_absolute_logit_error": 0.0,
        "native_slot_recall_sum": 0.0,
        "native_exact_set_sum": 0.0,
    }

    def measured_batches():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            targets = batch["targets"]
            router_inputs = targets["future_router_inputs"].to(
                args.device, dtype=geometry.input_basis.dtype
            )
            captured = targets["future_router_logits"].to(args.device).float()
            labels = targets["future_selected_ids"].to(args.device).long()
            valid = targets["future_available"].to(args.device).bool()
            with torch.no_grad():
                reconstructed = geometry.centered_logits(router_inputs).float()
                centered = captured - captured.mean(dim=-1, keepdim=True)
                difference = reconstructed - centered
                active_difference = difference[valid]
                predicted = stable_topk(reconstructed, 8)
                captured_ids = stable_topk(centered, 8)
                captured_membership = torch.zeros_like(centered, dtype=torch.bool)
                captured_membership.scatter_(-1, captured_ids, True)
                captured_recall = (
                    captured_membership.gather(-1, predicted).sum(-1).float() / 8.0
                )
                label_membership = torch.zeros_like(centered, dtype=torch.bool)
                label_membership.scatter_(-1, labels, True)
                label_recall = (
                    label_membership.gather(-1, predicted).sum(-1).float() / 8.0
                )
                endpoints = int(valid.sum())
                totals["batches"] = int(totals["batches"]) + 1
                totals["samples"] = int(totals["samples"]) + int(labels.shape[0])
                totals["valid_endpoints"] = int(totals["valid_endpoints"]) + endpoints
                totals["logit_values"] = int(totals["logit_values"]) + int(
                    active_difference.numel()
                )
                totals["squared_error"] = float(totals["squared_error"]) + float(
                    active_difference.square().sum()
                )
                totals["maximum_absolute_logit_error"] = max(
                    float(totals["maximum_absolute_logit_error"]),
                    float(active_difference.abs().max()),
                )
                totals["slot_recall_sum"] = float(totals["slot_recall_sum"]) + float(
                    captured_recall[valid].sum()
                )
                totals["exact_set_sum"] = float(totals["exact_set_sum"]) + float(
                    (captured_recall[valid] == 1.0).sum()
                )
                totals["selected_label_recall_sum"] = float(
                    totals["selected_label_recall_sum"]
                ) + float(label_recall[valid].sum())
                if native_weights is not None:
                    native_layers = [
                        F.linear(
                            router_inputs[:, :, layer].to(native_weights.dtype),
                            native_weights[layer],
                        )
                        for layer in range(geometry.layers)
                    ]
                    native = torch.stack(native_layers, dim=2).float()
                    native_centered = native - native.mean(dim=-1, keepdim=True)
                    native_difference = native_centered - centered
                    active_native_difference = native_difference[valid]
                    native_ids = stable_topk(native_centered, 8)
                    native_recall = (
                        captured_membership.gather(-1, native_ids).sum(-1).float()
                        / 8.0
                    )
                    totals["native_squared_error"] = float(
                        totals["native_squared_error"]
                    ) + float(active_native_difference.square().sum())
                    totals["native_maximum_absolute_logit_error"] = max(
                        float(totals["native_maximum_absolute_logit_error"]),
                        float(active_native_difference.abs().max()),
                    )
                    totals["native_slot_recall_sum"] = float(
                        totals["native_slot_recall_sum"]
                    ) + float(native_recall[valid].sum())
                    totals["native_exact_set_sum"] = float(
                        totals["native_exact_set_sum"]
                    ) + float((native_recall[valid] == 1.0).sum())
            yield batch

    tree = tree_availability_probe(measured_batches(), split=args.split)
    endpoints = int(totals["valid_endpoints"])
    values = int(totals["logit_values"])
    if endpoints == 0 or values == 0:
        raise RuntimeError("real geometry audit observed no valid endpoints")
    geometry_report: dict[str, Any] = {
        "probe": "real_future_router_input_reconstruction",
        "split": args.split,
        "batches": int(totals["batches"]),
        "samples": int(totals["samples"]),
        "valid_endpoints": endpoints,
        "maximum_absolute_logit_error": float(
            totals["maximum_absolute_logit_error"]
        ),
        "root_mean_square_logit_error": math.sqrt(
            float(totals["squared_error"]) / values
        ),
        "mean_slot_recall_at_8": float(totals["slot_recall_sum"]) / endpoints,
        "exact_top8_set_agreement": float(totals["exact_set_sum"]) / endpoints,
        "selected_label_slot_recall_at_8": float(
            totals["selected_label_recall_sum"]
        )
        / endpoints,
    }
    geometry_report["passed"] = (
        geometry_report["maximum_absolute_logit_error"] <= 2e-2
        and geometry_report["exact_top8_set_agreement"] == 1.0
    )
    if native_weights is not None:
        geometry_report["native_bf16_checkpoint_router"] = {
            "maximum_absolute_logit_error": float(
                totals["native_maximum_absolute_logit_error"]
            ),
            "root_mean_square_logit_error": math.sqrt(
                float(totals["native_squared_error"]) / values
            ),
            "mean_slot_recall_at_8": float(totals["native_slot_recall_sum"])
            / endpoints,
            "exact_top8_set_agreement": float(totals["native_exact_set_sum"])
            / endpoints,
        }
    report = {
        "schema": "harp_rtt_real_data_audit_v1",
        "split": args.split,
        "dataset_records": len(dataset),
        "test_accessed": False,
        "geometry": geometry_report,
        "tree": tree,
        "static_manifest": static.manifest,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
