#!/usr/bin/env python3
"""Run one optimizer-free real-data B2 semantic forward/backward smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.anchor import LegacyHARPAnchorBridge
from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt
from harp_rtt.model import HARPRTTTeacher
from harp_rtt.node_counterfactual import (
    NodeCounterfactualDatasetAdapter,
    load_node_counterfactual_companion,
)
from harp_rtt.static_artifacts import load_static_target_artifacts
from harp_rtt.train import (
    prepare_model_batch,
    production_config,
    runtime_static_artifacts,
    verify_router_numerics_audit,
)
from harp_rtt.training import move_to_device, seed_everything
from harp_rtt.v2_losses import (
    node_counterfactual_posterior_loss,
    node_counterfactual_semantic_loss,
)

from train_harp_rtt_b2_translator import configure_b2_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--router-numerics-audit", type=Path, required=True)
    parser.add_argument("--router-numerics-audit-sha256", required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite B2 smoke {args.output}")
    device = torch.device(args.device)
    seed_everything(42)
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    router = verify_router_numerics_audit(
        args.router_numerics_audit,
        args.router_numerics_audit_sha256,
        static.manifest,
    )
    bridge, anchor = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint,
        args.target_preprocessing,
        args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    config = production_config(static, bridge)
    model = HARPRTTTeacher(
        bridge,
        config,
        static.geometry,
        token_embedding=static.token_embedding,
    ).to(device)
    runtime_static = runtime_static_artifacts(static, device)
    del static
    ownership = configure_b2_parameters(model)

    labels, companion = load_node_counterfactual_companion(
        args.companion, split="train", training=True
    )
    base = HarpRTTDataset(
        args.index, "train", corpus_root=args.corpus, max_tree_nodes=32
    )
    joined = NodeCounterfactualDatasetAdapter(
        base, labels, split="train", training=True
    )
    count = min(args.batch_size, len(joined))
    host = collate_harp_rtt([joined[index] for index in range(count)])
    batch = move_to_device(host, device)
    prepared = prepare_model_batch(batch, runtime_static)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        outputs = model(
            batch=prepared,
            anchor_inputs={"batch": prepared},
            branch_semantic_only=True,
            random_anytime_truncation=False,
        )
        counterfactual = prepared["targets"]["counterfactual"]
        selection = counterfactual["budget_node_masks"][:, 2].bool()
        semantic = node_counterfactual_semantic_loss(
            outputs, counterfactual, selection_mask=selection
        )
        posterior = node_counterfactual_posterior_loss(
            outputs["branch_posterior_logits"],
            counterfactual,
            branch_mask=outputs["branch_mask"],
            selection_mask=selection,
        )
        total = semantic.total + 0.1 * posterior
    total.backward()
    trainable_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    missing_trainable_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    forbidden_gradients = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if not trainable_gradients or missing_trainable_gradients or forbidden_gradients:
        raise RuntimeError("B2 smoke gradient ownership failed")
    report: dict[str, Any] = {
        "schema": "harp_rtt_b2_real_semantic_smoke_v1",
        "passed": True,
        "batch_size": count,
        "loss": {
            "total": float(total.detach()),
            "exact_set": float(semantic.exact_set.detach()),
            "query": float(semantic.query.detach()),
            "router_kl": float(semantic.router_kl.detach()),
            "posterior": float(posterior.detach()),
            "active_cells": semantic.active_path_depth_cells,
        },
        "output_shapes": {
            name: list(outputs[name].shape)
            for name in (
                "branch_semantic_scores",
                "router_queries",
                "branch_posterior_logits",
                "branch_mask",
            )
        },
        "ownership": ownership,
        "trainable_parameters_with_gradients": len(trainable_gradients),
        "missing_trainable_gradients": missing_trainable_gradients,
        "forbidden_gradients": forbidden_gradients,
        "peak_cuda_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "optimizer_created": False,
        "training_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "router_numerics": router,
        "anchor": anchor,
        "companion_schema": companion["schema"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
