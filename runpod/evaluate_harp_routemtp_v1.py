#!/usr/bin/env python3
"""One-shot official-tune and diagnostic evaluation of frozen RouteMTP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "runpod" / "transformers_mtp_bridge"
for value in (REPO_ROOT, BRIDGE):
    if str(value) not in sys.path: sys.path.insert(0, str(value))

from harp_rtt.anchor import LegacyHARPAnchorBridge  # noqa: E402
from harp_rtt.dataset import HarpRTTDataset  # noqa: E402
from harp_rtt.node_counterfactual import NodeCounterfactualDatasetAdapter, load_node_counterfactual_companion  # noqa: E402
from harp_rtt.routemtp import RecurrentRouteResidual, RouteMTPConfig, RouteMTPPredictor, install_cache_coherent_adapters  # noqa: E402
from harp_rtt.routemtp_adapters import SwitchableExpertLoRA, SwitchableMTPRouterResidual  # noqa: E402
from harp_rtt.routemtp_loss import RouteMTPObjective  # noqa: E402
from harp_rtt.routemtp_replay import RouteMTPTreeRunner  # noqa: E402
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.train import runtime_static_artifacts  # noqa: E402
from harp_rtt.training import sha256_file  # noqa: E402
from runpod.train_harp_routemtp_v1 import (  # noqa: E402
    HydrationStore, IndexedSubset, SCHEMA as CHECKPOINT_SCHEMA, evaluate,
    load_compact_state,
)


SCHEMA = "harp_routemtp_frozen_evaluation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("tune", "development"):
        parser.add_argument(f"--{split}-index", type=Path, required=True)
        parser.add_argument(f"--{split}-corpus", type=Path, required=True)
        parser.add_argument(f"--{split}-companion", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-split-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--tune-hydration", type=Path, required=True)
    parser.add_argument("--development-hydration", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_dataset(
    index: Path,
    corpus: Path,
    companion: Path,
    *,
    allowed_requests: set[str] | None = None,
) -> Dataset[Any]:
    base = HarpRTTDataset(index, "train", corpus_root=corpus, max_tree_nodes=32)
    labels, manifest = load_node_counterfactual_companion(companion, split="train", training=True)
    if manifest.get("sealed_test_opened") is not False:
        raise PermissionError("RouteMTP evaluation companion crossed sealed test")
    joined = NodeCounterfactualDatasetAdapter(base, labels, split="train", training=True)
    if allowed_requests is None:
        return joined
    indices: list[int] = []
    observed: set[str] = set()
    for index_value, record in enumerate(base.records):
        request = str(base.segments[record.segment].sequences[record.sequence]["request_id"])
        if request in allowed_requests:
            indices.append(index_value); observed.add(request)
    if observed != allowed_requests:
        raise KeyError(
            f"RouteMTP official tune selection misses {len(allowed_requests - observed)} requests"
        )
    return IndexedSubset(joined, indices)


def request_ids(dataset: Dataset[Any]) -> set[str]:
    return {str(dataset[index]["metadata"]["request_id"]) for index in range(len(dataset))}


def main() -> None:
    args = parse_args()
    from qwen35_mtp import load_checkpoint_mtp

    if args.output.exists(): raise FileExistsError(f"refusing to overwrite {args.output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("RouteMTP checkpoint schema mismatch")
    if len(args.source_commit) != 40 or checkpoint.get("source_commit") != args.source_commit:
        raise ValueError("RouteMTP evaluation source commit differs from checkpoint")
    if checkpoint.get("internal_split_sha256") != sha256_file(args.training_split_manifest):
        raise ValueError("RouteMTP checkpoint and protected request split differ")
    protected_split = json.loads(args.training_split_manifest.read_text(encoding="utf-8"))
    if protected_split.get("schema") != "harp_routemtp_protected_request_split_v2":
        raise ValueError("RouteMTP protected request split schema mismatch")
    official_tune = set(protected_split.get("official_tuning_requests", []))
    if (
        len(official_tune) != 32
        or int(protected_split.get("official_tuning_rows", -1)) != 512
        or protected_split.get("official_tune_opened") is not False
    ):
        raise PermissionError("RouteMTP official tune partition is not sealed 32/512")
    stage = str(checkpoint["stage"]); device = torch.device(args.device)
    tune = load_dataset(
        args.tune_index, args.tune_corpus, args.tune_companion,
        allowed_requests=official_tune,
    )
    development = load_dataset(args.development_index, args.development_corpus, args.development_companion)
    if len(tune) != 512 or request_ids(tune) != official_tune:
        raise PermissionError("RouteMTP evaluator did not isolate the official tune partition")
    if request_ids(tune) & request_ids(development):
        raise PermissionError("RouteMTP tune/development requests overlap")
    tune_hydration = HydrationStore(args.tune_hydration)
    development_hydration = HydrationStore(args.development_hydration)
    tune_hydration_sha256 = sha256_file(
        args.tune_hydration / "HYDRATION_MANIFEST.json"
    )
    development_hydration_sha256 = sha256_file(
        args.development_hydration / "HYDRATION_MANIFEST.json"
    )
    if checkpoint.get("hydration_manifest_sha256") != tune_hydration_sha256:
        raise ValueError("RouteMTP checkpoint was trained from a different hydration")
    hydration_source_commit = checkpoint.get(
        "hydration_source_commit", checkpoint.get("source_commit")
    )
    for name, store, companion in (
        ("tune", tune_hydration, args.tune_companion),
        ("development", development_hydration, args.development_companion),
    ):
        bindings = store.value.get("bindings", {})
        if hydration_source_commit != bindings.get("source_commit"):
            raise ValueError(f"RouteMTP checkpoint source differs from {name} hydration")
        companion_sha256 = sha256_file(companion / "manifest.json")
        if bindings.get("counterfactual_companion_sha256") != companion_sha256:
            raise ValueError(f"RouteMTP {name} hydration and companion differ")
    for name, dataset, store in (
        ("tune", tune, tune_hydration),
        ("development", development, development_hydration),
    ):
        missing = [
            (str(dataset[index]["metadata"]["request_id"]), int(dataset[index]["metadata"]["position"]))
            for index in range(len(dataset))
            if (
                str(dataset[index]["metadata"]["request_id"]),
                int(dataset[index]["metadata"]["position"]),
            ) not in store.offsets
        ]
        if missing:
            raise KeyError(
                f"RouteMTP {name} hydration misses {len(missing)} evaluation rows"
            )

    static = load_static_target_artifacts(args.static_dir, device="cpu")
    config = RouteMTPConfig(**checkpoint["config"]); config.validate()
    if (
        config.target_layers != static.geometry.layers
        or config.experts != static.geometry.experts
        or config.router_rank != static.geometry.maximum_rank
    ):
        raise ValueError("RouteMTP checkpoint/static geometry mismatch")
    geometry = static.geometry.to(device)
    predictor = RouteMTPPredictor(config, geometry.expert_keys, geometry.centered_bias, geometry.rank_mask).to(device)
    mtp = load_checkpoint_mtp(args.model, device=device)
    adapters = install_cache_coherent_adapters(mtp, rank=config.lora_rank)
    residual = RecurrentRouteResidual(config.hidden_width, config.residual_width).to(device) if stage.startswith("r3_") else None
    expert = None; router = None
    if stage in {"r3_expert", "r3_router"}:
        expert = SwitchableExpertLoRA(mtp.decoder.layers[0].mlp.experts, rank=8).to(device)
        mtp.decoder.layers[0].mlp.experts = expert
    if stage == "r3_router":
        router = SwitchableMTPRouterResidual(mtp.decoder.layers[0].mlp.gate, config.hidden_width, config.experts).to(device)
        mtp.decoder.layers[0].mlp.gate = router
    load_compact_state(
        args.checkpoint, expected_stage=stage, predictor=predictor,
        adapters=adapters, residual=residual, expert=expert, router=router,
    )
    runner = RouteMTPTreeRunner(
        mtp, adapters, recurrent_residual=residual,
        expert_adapter=expert, router_adapter=router,
    ).to(device).eval()
    predictor.eval()
    anchor, _ = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint, args.target_preprocessing, args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    anchor = anchor.to(device).requires_grad_(False).eval()
    token_embedding = static.token_embedding.to(device)
    runtime_static = runtime_static_artifacts(static, device)
    objective = RouteMTPObjective(geometry.expert_keys, geometry.centered_bias, exact_k=config.exact_k).to(device)
    tune_metrics = evaluate(
        stage=stage, predictor=predictor, runner=runner, objective=objective,
        anchor=anchor, runtime_static=runtime_static, geometry=geometry,
        token_embedding=token_embedding, hydration=tune_hydration, dataset=tune,
        device=device, microbatch=args.microbatch_size, workers=0,
    )
    # Architecture, checkpoint, budget policy, and temperature are frozen
    # before this one-shot diagnostic call.
    development_metrics = evaluate(
        stage=stage, predictor=predictor, runner=runner, objective=objective,
        anchor=anchor, runtime_static=runtime_static, geometry=geometry,
        token_embedding=token_embedding, hydration=development_hydration,
        dataset=development,
        device=device, microbatch=args.microbatch_size, workers=0,
    )
    selected = development_metrics["budgets"]["16"]
    useful = (
        selected["branch_recall_at_8"] >= 0.85
        and selected["factual_h2_h4_recall_at_8"] >= 0.80
        and selected["per_horizon_recall_at_8"][3] >= 0.75
        and selected["factual_h2_h4_c64_coverage"] >= 0.97
    )
    final_candidate = (
        selected["branch_recall_at_8"] >= 0.935
        and selected["factual_h1_h4_recall_at_8"] >= 0.90
        and selected["per_horizon_recall_at_8"][3] >= 0.87
    )
    result = {
        "schema": SCHEMA, "stage": stage,
        "source_commit": args.source_commit,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "tune_hydration_manifest_sha256": tune_hydration_sha256,
        "development_hydration_manifest_sha256": development_hydration_sha256,
        "tune": tune_metrics, "development": development_metrics,
        "standalone_useful_gate_passed": useful,
        "final_accuracy_candidate_gate_passed": final_candidate,
        "temperature": 1.0, "budget_policy_frozen_before_development": True,
        "optimizer_constructed": False, "training_started": False,
        "official_tune_opened": True, "diagnostic_development_opened": True,
        "formal_validation_opened": False, "calibration_opened": False,
        "sealed_test_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
