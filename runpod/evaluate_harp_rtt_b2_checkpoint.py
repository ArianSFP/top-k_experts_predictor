#!/usr/bin/env python3
"""Complete B2 probe evaluation from an already sealed translator checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.anchor import LegacyHARPAnchorBridge
from harp_rtt.dataset import HarpRTTDataset
from harp_rtt.model import HARPRTTTeacher
from harp_rtt.node_counterfactual import (
    NodeCounterfactualDatasetAdapter,
    load_node_counterfactual_companion,
)
from harp_rtt.static_artifacts import load_static_target_artifacts
from harp_rtt.train import (
    production_config,
    runtime_static_artifacts,
    verify_router_numerics_audit,
)
from harp_rtt.training import seed_everything, sha256_file
from train_harp_rtt_b2_translator import (
    EXPECTED_PROBE_ROWS,
    SCHEMA,
    configure_b2_parameters,
    evaluate_probe,
    verify_probe_corpus_binding,
)


RECOVERY_SCHEMA = "harp_rtt_b2_probe_evaluation_recovery_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-output", type=Path, required=True)
    parser.add_argument("--failed-log", type=Path, required=True)
    parser.add_argument("--probe-index", type=Path, required=True)
    parser.add_argument("--probe-corpus", type=Path, required=True)
    parser.add_argument("--probe-companion", type=Path, required=True)
    parser.add_argument("--probe-oracle-metrics", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--router-numerics-audit", type=Path, required=True)
    parser.add_argument("--router-numerics-audit-sha256", required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--training-source-commit", required=True)
    parser.add_argument("--evaluation-source-commit", required=True)
    parser.add_argument("--microbatch-size", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("B2 translator checkpoint schema mismatch")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.seed_output.expanduser().resolve()
    checkpoint_path = output / "best_translator.pt"
    manifest_path = output / "run_manifest.json"
    metrics_path = output / "metrics.jsonl"
    required = (checkpoint_path, manifest_path, metrics_path, args.failed_log)
    if not output.is_dir() or any(not path.is_file() for path in required):
        raise FileNotFoundError("evaluation recovery requires the completed failed seed output")
    for name in (
        "probe_request_metrics.npz",
        "B2_SEED_RESULT.json",
        "EVALUATION_RECOVERY.json",
        "SHA256SUMS",
    ):
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite recovered seed artifact {name}")

    checkpoint = load_checkpoint(checkpoint_path)
    training_manifest = json.loads(manifest_path.read_text())
    seed = int(checkpoint.get("seed", -1))
    if seed not in (42, 43, 44) or int(training_manifest.get("seed", -2)) != seed:
        raise ValueError("seed identity differs between checkpoint and run manifest")
    if checkpoint.get("source_commit") != args.training_source_commit:
        raise ValueError("checkpoint training-source commit mismatch")
    if checkpoint.get("run_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("checkpoint is not bound to this training manifest")
    if training_manifest.get("request_disjoint_probe") is not True:
        raise PermissionError("original training manifest did not seal a disjoint probe")
    for key in (
        "formal_validation_opened",
        "calibration_opened",
        "sealed_test_opened",
        "h1_training_started",
        "b3_training_started",
    ):
        if training_manifest.get(key) is not False:
            raise PermissionError(f"original training manifest violates {key}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    seed_everything(seed, deterministic=args.deterministic)
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    router_audit = verify_router_numerics_audit(
        args.router_numerics_audit,
        args.router_numerics_audit_sha256,
        static.manifest,
    )
    bridge, anchor_provenance = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint,
        args.target_preprocessing,
        args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    model = HARPRTTTeacher(
        bridge,
        production_config(static, bridge),
        static.geometry,
        token_embedding=static.token_embedding,
    ).to(device)
    runtime_static = runtime_static_artifacts(static, device)
    del static
    ownership = configure_b2_parameters(model)
    if ownership != checkpoint.get("ownership"):
        raise ValueError("evaluation model parameter ownership differs from training")
    incompatible = model.load_state_dict(checkpoint["state_dict_non_anchor"], strict=False)
    if incompatible.unexpected_keys or any(
        not (name.startswith("anchor.") or name == "token_embedding.weight")
        for name in incompatible.missing_keys
    ):
        raise ValueError("checkpoint state does not exactly cover the non-anchor model")

    labels, companion_manifest = load_node_counterfactual_companion(
        args.probe_companion, split="train", training=True
    )
    if sha256_file(args.probe_companion / "manifest.json") != training_manifest.get(
        "probe_companion_manifest_sha256"
    ):
        raise ValueError("probe companion differs from the training manifest")
    if sha256_file(args.probe_oracle_metrics) != training_manifest.get(
        "probe_oracle_metrics_sha256"
    ):
        raise ValueError("probe oracle metrics differ from the training manifest")
    corpus_binding = verify_probe_corpus_binding(args.probe_corpus, companion_manifest)
    base = HarpRTTDataset(
        args.probe_index,
        "train",
        corpus_root=args.probe_corpus,
        max_tree_nodes=32,
    )
    probe = NodeCounterfactualDatasetAdapter(base, labels, split="train", training=True)
    if len(probe) != len(labels) or len(probe) != EXPECTED_PROBE_ROWS:
        raise ValueError("recovered probe source/label join is invalid")

    probe_path = output / "probe_request_metrics.npz"
    probe_metrics = evaluate_probe(
        model,
        probe,
        oracle_metrics_path=args.probe_oracle_metrics,
        batch_size=args.microbatch_size,
        device=device,
        static=runtime_static,
        workers=args.num_workers,
        output=probe_path,
    )
    epoch_events = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    if not epoch_events or any(event.get("event") != "epoch" for event in epoch_events):
        raise ValueError("training metric history is incomplete or malformed")

    recovery = {
        "schema": RECOVERY_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "training_source_commit": args.training_source_commit,
        "evaluation_source_commit": args.evaluation_source_commit,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_manifest_sha256": sha256_file(manifest_path),
        "failed_log": str(args.failed_log.resolve()),
        "failed_log_sha256": sha256_file(args.failed_log),
        "failure": "broken ephemeral probe corpus segment symlink before scoring",
        "weights_changed": False,
        "optimizer_constructed": False,
        "probe_corpus_binding": corpus_binding,
        "router_numerics": router_audit,
        "anchor": anchor_provenance,
    }
    recovery_path = output / "EVALUATION_RECOVERY.json"
    write_json_exclusive(recovery_path, recovery)
    result = {
        "schema": SCHEMA,
        "seed": seed,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_tuning_total": float(checkpoint["best_tuning_total"]),
        "epochs_completed": int(epoch_events[-1]["epoch"]),
        "stopped_for_patience": False,
        "probe_metrics": probe_metrics,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "probe_request_metrics": probe_path.name,
        "probe_request_metrics_sha256": sha256_file(probe_path),
        "evaluation_recovered": True,
        "evaluation_recovery_manifest": recovery_path.name,
        "evaluation_recovery_manifest_sha256": sha256_file(recovery_path),
        "training_started": True,
        "optimizer_started": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "h1_training_started": False,
        "b3_training_started": False,
    }
    write_json_exclusive(output / "B2_SEED_RESULT.json", result)
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in sorted(item for item in output.iterdir() if item.is_file()):
            if path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run(parse_args())
