#!/usr/bin/env python3
"""Train the outer-train-only B3 generator or frozen-C64 rich reranker.

This is deliberately separate from :mod:`harp_rtt.train`.  The production
phase driver opens the formal validation split and has no B2 checkpoint
lineage.  B3 development instead reuses the frozen B2 224/32 request split and
the already-open 128-request diagnostic probe.  It exposes no validation,
calibration, or sealed-test switch.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.anchor import LegacyHARPAnchorBridge
from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt
from harp_rtt.exact_k import stable_topk
from harp_rtt.losses import HARPRTTLossConfig, HARPRTTObjective
from harp_rtt.metrics import (
    candidate_coverage_at_k,
    evaluate_harp_rtt,
    model_selection_tuple,
)
from harp_rtt.model import HARPRTTTeacher
from harp_rtt.node_counterfactual import (
    NodeCounterfactualDatasetAdapter,
    load_node_counterfactual_companion,
)
from harp_rtt.static_artifacts import load_static_target_artifacts
from harp_rtt.train import (
    loss_dimensions,
    prepare_model_batch,
    production_config,
    runtime_static_artifacts,
    verify_router_numerics_audit,
)
from harp_rtt.training import (
    PhaseSpec,
    configure_training_phase,
    move_to_device,
    seed_everything,
    sha256_file,
)
from harp_rtt.v2_losses import (
    node_counterfactual_posterior_loss,
    node_counterfactual_semantic_loss,
    swap_loss,
)
from runpod.train_harp_rtt_b2_translator import (
    BUDGET_INDEX,
    EXPECTED_FITTING_ROWS,
    EXPECTED_PROBE_ROWS,
    fitting_subsets,
    request_groups,
    verify_probe_corpus_binding,
)


GENERATOR_SCHEMA = "harp_rtt_b3_generator_development_v1"
RANKER_SCHEMA = "harp_rtt_b3_ranker_development_v1"
B2_SCHEMA = "harp_rtt_b2_translator_training_v1"
EFFECTIVE_BATCH = 32
ACTIVE_CANDIDATE_SOURCES = (True, True, True, True, False, False)
FROZEN_ANCHOR_QUOTA = 40
GENERATOR_THRESHOLDS = {
    "mean_h2_h4_c64": 0.985,
    "h4_c64": 0.970,
    "h4_prefix_mismatch_c64": 0.930,
}
RANKER_RECALL_GATE = 0.85


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generator", "ranker"), required=True)
    parser.add_argument("--fitting-index", type=Path, required=True)
    parser.add_argument("--fitting-corpus", type=Path, required=True)
    parser.add_argument("--fitting-companion", type=Path, required=True)
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
    parser.add_argument("--b2-override-record", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--capture-source-commit", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
        help="formal fixed candidates; effective batch remains exactly 32",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--counterfactual-retention-weight", type=float, default=0.1)
    parser.add_argument("--swap-weight", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run epoch-zero/binding checks and construct no optimizer",
    )
    return parser.parse_args()


def verify_b2_override(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "harp_rtt_b2_user_operational_override_v1",
        "user_authorized": True,
        "operational_pass": True,
        "move_to_b3_authorized": True,
        "thresholds_changed": False,
        "preregistered_three_seed_gate_completed": False,
        "preregistered_three_seed_gate_passed": None,
        "single_seed_statistical_claim": False,
    }
    for name, expected in required.items():
        if value.get(name) != expected:
            raise PermissionError(f"B2 operational override disagrees on {name}")
    completed = value.get("completed_seed")
    if not isinstance(completed, Mapping) or int(completed.get("seed", -1)) != 42:
        raise ValueError("B2 operational override lacks the completed seed-42 result")
    return value


def _initializer_manifest(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("initializer checkpoint must contain a mapping")
    manifest_path = checkpoint_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("initializer checkpoint has no adjacent run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = checkpoint.get("run_manifest_sha256")
    if expected != sha256_file(manifest_path):
        raise ValueError("initializer checkpoint/run manifest hash mismatch")
    return dict(checkpoint), manifest


def verify_initializer(
    checkpoint_path: Path,
    *,
    stage: str,
    override: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint, manifest = _initializer_manifest(checkpoint_path)
    if stage == "generator":
        if checkpoint.get("schema") != B2_SCHEMA or manifest.get("schema") != B2_SCHEMA:
            raise ValueError("B3 generator requires a B2 translator checkpoint")
        if int(checkpoint.get("seed", -1)) != 42:
            raise ValueError("the operational B2 promotion names seed 42 only")
        expected = override["completed_seed"]["checkpoint_sha256"]
        if sha256_file(checkpoint_path) != expected:
            raise ValueError("B2 checkpoint differs from the user-authorized seed-42 hash")
        result_path = checkpoint_path.parent / "B2_SEED_RESULT.json"
        if not result_path.is_file():
            raise FileNotFoundError("B2 checkpoint has no sealed seed result")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("formal_validation_opened") or result.get("sealed_test_opened"):
            raise PermissionError("B2 initializer crossed a sealed split boundary")
    else:
        if checkpoint.get("schema") != GENERATOR_SCHEMA:
            raise ValueError("B3 ranker requires a B3 generator checkpoint")
        result_path = checkpoint_path.parent / "B3_GENERATOR_RESULT.json"
        if not result_path.is_file():
            raise FileNotFoundError("B3 generator checkpoint has no gate result")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("candidate_gate", {}).get("passed") is not True:
            raise PermissionError("B3 ranker cannot start before the learned C64 gate passes")
        if result.get("formal_validation_opened") or result.get("sealed_test_opened"):
            raise PermissionError("B3 generator initializer crossed a sealed split boundary")
    return checkpoint, manifest, result


def load_non_anchor_state(model: nn.Module, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    state = checkpoint.get("state_dict_non_anchor")
    if not isinstance(state, Mapping):
        raise KeyError("initializer contains no non-anchor state")
    incompatible = model.load_state_dict(state, strict=False)
    forbidden_missing = [
        name
        for name in incompatible.missing_keys
        if not (name.startswith("anchor.") or name == "token_embedding.weight")
    ]
    if incompatible.unexpected_keys or forbidden_missing:
        raise ValueError(
            "initializer does not exactly cover the non-anchor model: "
            f"missing={forbidden_missing[:8]}, unexpected={incompatible.unexpected_keys[:8]}"
        )
    return {
        "tensors": len(state),
        "missing_anchor_tensors": len(incompatible.missing_keys),
        "unexpected": [],
    }


def non_anchor_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("anchor.") and name != "token_embedding.weight"
    }


def make_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_harp_rtt,
        drop_last=False,
    )


def model_forward(
    model: nn.Module,
    batch: Mapping[str, Any],
    static: Any,
    *,
    progress: float,
    random_anytime: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    prepared = prepare_model_batch(batch, static)
    output = model(
        batch=prepared,
        anchor_inputs={"batch": prepared},
        candidate_training_progress=float(progress),
        candidate_active_sources=ACTIVE_CANDIDATE_SOURCES,
        random_anytime_truncation=random_anytime,
    )
    if not isinstance(output, Mapping):
        raise TypeError("B3 model output must be a mapping")
    if int(output["candidate_anchor_quota"]) != (
        64 if progress <= 0.10 else FROZEN_ANCHOR_QUOTA if progress >= 0.50 else int(output["candidate_anchor_quota"])
    ):
        raise AssertionError("candidate curriculum returned an unexpected quota")
    return output, prepared


def configure_b3_parameters(
    model: nn.Module,
    *,
    stage: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = PhaseSpec(
        "phase2" if stage == "generator" else "phase4",
        epochs=epochs,
        new_learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    groups, report = configure_training_phase(model, spec)
    trainable = set(report.new_names) | set(report.legacy_names)
    if stage == "generator" and any(name.startswith("reranker.") for name in trainable):
        raise PermissionError("B3 generator must keep the rich reranker frozen")
    if stage == "ranker" and any(not name.startswith("reranker.") for name in trainable):
        raise PermissionError("B3 ranker ownership escaped the reranker")
    if any(name.startswith("anchor.") or name == "token_embedding.weight" for name in trainable):
        raise PermissionError("B3 cannot train the HARP anchor or token embedding")
    return groups, report.to_dict()


def _ranker_training_mode(model: HARPRTTTeacher) -> None:
    model.eval()
    model.reranker.train()


@torch.no_grad()
def epoch_zero_audit(
    model: HARPRTTTeacher,
    dataset: Dataset[Any],
    *,
    stage: str,
    device: torch.device,
    static: Any,
    workers: int,
) -> dict[str, Any]:
    model.eval()
    host = next(
        iter(
            make_loader(
                dataset,
                batch_size=1,
                shuffle=False,
                seed=0,
                workers=workers,
                device=device,
            )
        )
    )
    batch = move_to_device(host, device)
    prepared = prepare_model_batch(batch, static)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        anchor = model.anchor(batch=prepared)
        output = model(
            batch=prepared,
            anchor_inputs={"batch": prepared},
            candidate_training_progress=0.0 if stage == "generator" else 1.0,
            candidate_active_sources=ACTIVE_CANDIDATE_SOURCES,
        )
    active = output["active_scores"]
    anchor_active = anchor["future_router_scores"][:, :4]
    if stage == "generator":
        score_equal = torch.equal(active, anchor_active)
        top8_equal = torch.equal(stable_topk(active.float(), 8), stable_topk(anchor_active.float(), 8))
        expected_candidates = stable_topk(anchor_active.float(), 64)
        candidate_equal = torch.equal(output["candidate_ids"], expected_candidates)
        if not (score_equal and top8_equal and candidate_equal):
            raise RuntimeError("B3 generator epoch-zero HARP invariance failed")
        residual_reference = "frozen_harp_anchor"
    else:
        generator = output["trajectory_round_scores"][:, -1]
        score_equal = torch.equal(active, generator)
        top8_equal = torch.equal(stable_topk(active.float(), 8), stable_topk(generator.float(), 8))
        candidate_equal = int(output["candidate_anchor_quota"]) == FROZEN_ANCHOR_QUOTA
        if not (score_equal and top8_equal and candidate_equal):
            raise RuntimeError("B3 ranker epoch-zero generator invariance failed")
        residual_reference = "frozen_b3_generator"
    return {
        "schema": "harp_rtt_b3_epoch_zero_audit_v1",
        "stage": stage,
        "reference": residual_reference,
        "score_bitwise_equal": score_equal,
        "top8_equal": top8_equal,
        "candidate_policy_equal": candidate_equal,
        "candidate_anchor_quota": int(output["candidate_anchor_quota"]),
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "optimizer_constructed": False,
    }


def factual_and_stage_loss(
    *,
    stage: str,
    outputs: Mapping[str, Any],
    prepared: Mapping[str, Any],
    objective: HARPRTTObjective,
    step: int,
    counterfactual_retention_weight: float,
    swap_weight: float,
) -> tuple[Tensor, dict[str, float | int]]:
    factual = objective(outputs, prepared, step=step)
    total = factual.total
    metrics: dict[str, float | int] = {
        name: float(value.detach()) for name, value in factual.as_dict().items()
    }
    if stage == "generator":
        counterfactual = prepared["targets"]["counterfactual"]
        selection = counterfactual["budget_node_masks"][:, BUDGET_INDEX].bool()
        semantic = node_counterfactual_semantic_loss(
            outputs,
            counterfactual,
            selection_mask=selection,
            exact_k=8,
            query_weight=0.2,
            router_kl_weight=0.1,
        )
        posterior = node_counterfactual_posterior_loss(
            outputs["branch_posterior_logits"],
            counterfactual,
            branch_mask=outputs["branch_mask"],
            selection_mask=selection,
        )
        retention = semantic.total + 0.1 * posterior
        total = total + float(counterfactual_retention_weight) * retention
        metrics.update(
            {
                "counterfactual_total": float(semantic.total.detach()),
                "counterfactual_exact_set": float(semantic.exact_set.detach()),
                "counterfactual_query": float(semantic.query.detach()),
                "counterfactual_router_kl": float(semantic.router_kl.detach()),
                "counterfactual_posterior": float(posterior.detach()),
            }
        )
    else:
        scores = outputs["active_scores"]
        base = outputs["trajectory_round_scores"][:, -1].detach()
        labels = prepared["targets"]["future_selected_ids"]
        candidate_mask = outputs["candidate_dense_mask"]
        valid = prepared["targets"]["future_available"].bool()
        if valid.shape == scores.shape[:2]:
            valid = valid[..., None].expand(scores.shape[:-1])
        elif valid.shape != scores.shape[:-1]:
            raise ValueError("ranker future validity disagrees with score geometry")
        active = valid.reshape(-1)
        swap = swap_loss(
            scores.reshape(-1, scores.shape[-1])[active],
            base.reshape(-1, base.shape[-1])[active],
            labels.reshape(-1, labels.shape[-1])[active],
            exact_k=8,
            margin=0.125,
            candidate_mask=candidate_mask.reshape(-1, candidate_mask.shape[-1])[active],
        )
        total = total + float(swap_weight) * swap.loss
        metrics.update(
            {
                "swap": float(swap.loss.detach()),
                "swap_pairs": swap.swap_pairs,
                "outside_candidate_misses": swap.outside_candidate_misses,
            }
        )
    metrics["optimized_total"] = float(total.detach())
    return total, metrics


def train_epoch(
    model: HARPRTTTeacher,
    dataset: Dataset[Any],
    optimizer: torch.optim.Optimizer,
    objective: HARPRTTObjective,
    *,
    stage: str,
    epoch: int,
    epochs: int,
    global_step: int,
    microbatch: int,
    device: torch.device,
    static: Any,
    workers: int,
    seed: int,
    gradient_clip: float,
    counterfactual_retention_weight: float,
    swap_weight: float,
) -> tuple[dict[str, float], int]:
    if stage == "generator":
        model.train()
    else:
        _ranker_training_mode(model)
    accumulation = EFFECTIVE_BATCH // microbatch
    if EFFECTIVE_BATCH % microbatch:
        raise ValueError("microbatch must divide effective batch 32")
    batches = make_loader(
        dataset,
        batch_size=microbatch,
        shuffle=True,
        seed=seed * 1000 + epoch,
        workers=workers,
        device=device,
    )
    total_optimizer_steps = max(1, math.ceil(len(batches) / accumulation) * epochs)
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = defaultdict(float)
    examples = 0
    window_examples = 0
    for index, host in enumerate(batches):
        batch = move_to_device(host, device)
        count = int(batch["targets"]["future_selected_ids"].shape[0])
        examples += count
        window_examples += count
        progress = min(1.0, global_step / max(1, total_optimizer_steps - 1))
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs, prepared = model_forward(
                model,
                batch,
                static,
                progress=progress if stage == "generator" else 1.0,
                random_anytime=stage == "generator",
            )
            loss, values = factual_and_stage_loss(
                stage=stage,
                outputs=outputs,
                prepared=prepared,
                objective=objective,
                step=global_step + 1,
                counterfactual_retention_weight=counterfactual_retention_weight,
                swap_weight=swap_weight,
            )
        (loss * (count / EFFECTIVE_BATCH)).backward()
        for name, value in values.items():
            totals[name] += float(value) * count
        step_now = (index + 1) % accumulation == 0 or index + 1 == len(batches)
        if not step_now:
            continue
        if window_examples < EFFECTIVE_BATCH:
            correction = EFFECTIVE_BATCH / window_examples
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            gradient_clip,
        )
        if not torch.isfinite(norm):
            raise FloatingPointError("B3 gradient norm became non-finite")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        window_examples = 0
        global_step += 1
    if not examples:
        raise ValueError("B3 training loader is empty")
    return {name: value / examples for name, value in totals.items()}, global_step


def evaluation_report(
    model: HARPRTTTeacher,
    dataset: Dataset[Any],
    *,
    batch_size: int,
    device: torch.device,
    static: Any,
    workers: int,
) -> dict[str, Any]:
    batches = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        workers=workers,
        device=device,
    )
    return evaluate_harp_rtt(
        model,
        batches,
        device=device,
        split="train",
        autocast=True,
        model_call=lambda module, batch: model_forward(
            module,
            batch,
            static,
            progress=1.0,
            random_anytime=False,
        )[0],
    )


def generator_selection(report: Mapping[str, Any]) -> tuple[float, ...]:
    rows = report["horizon_metrics"]
    coverage = [float(row["request_macro_candidate_coverage_at_64"]) for row in rows]
    mean_h2_h4 = float(np.mean(coverage[1:4]))
    return (
        mean_h2_h4,
        coverage[3],
        float(report["mean_h1_h4_request_macro_slot_recall_at_8"]),
        -float(report["mean_h1_h4_request_macro_exact_set_nll"]),
    )


@torch.no_grad()
def position_level_generator_gate(
    model: HARPRTTTeacher,
    dataset: Dataset[Any],
    *,
    oracle_metrics: Path,
    batch_size: int,
    device: torch.device,
    static: Any,
    workers: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    model.eval()
    requests: list[str] = []
    positions: list[int] = []
    coverage_rows: list[np.ndarray[Any, Any]] = []
    recall_rows: list[np.ndarray[Any, Any]] = []
    for host in make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        workers=workers,
        device=device,
    ):
        metadata = host["metadata"]
        requests.extend(str(value) for value in metadata["request_id"])
        positions.extend(int(value) for value in metadata["position"])
        batch = move_to_device(host, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs, prepared = model_forward(
                model, batch, static, progress=1.0, random_anytime=False
            )
        valid = prepared["targets"]["future_available"].bool()
        labels = prepared["targets"]["future_selected_ids"].long()
        candidates = outputs["candidate_ids"].long()
        coverage = candidate_coverage_at_k(
            candidates,
            labels,
            outputs["candidate_mask"],
            valid,
            experts=int(outputs["active_scores"].shape[-1]),
            k=8,
        )
        predicted = stable_topk(outputs["active_scores"].float(), 8)
        safe_labels = torch.where(valid[..., None], labels, torch.zeros_like(labels))
        membership = torch.zeros_like(outputs["active_scores"], dtype=torch.bool)
        membership.scatter_(-1, safe_labels, True)
        recall = membership.gather(-1, predicted).float().mean(-1)
        recall = recall * valid.float()
        weights = valid.float()
        row_coverage = (coverage * weights).sum(-1) / weights.sum(-1).clamp_min(1)
        row_recall = (recall * weights).sum(-1) / weights.sum(-1).clamp_min(1)
        coverage_rows.append(row_coverage.cpu().numpy())
        recall_rows.append(row_recall.cpu().numpy())
    coverage_np = np.concatenate(coverage_rows)
    recall_np = np.concatenate(recall_rows)
    source = np.load(oracle_metrics)
    lookup = {
        (str(request), int(position)): index
        for index, (request, position) in enumerate(
            zip(source["request_ids"], source["source_positions"], strict=True)
        )
    }
    order = np.asarray(
        [lookup[(request, position)] for request, position in zip(requests, positions, strict=True)]
    )
    greedy = source["greedy_match"][order]
    if coverage_np.shape != (EXPECTED_PROBE_ROWS, 4) or greedy.shape != coverage_np.shape:
        raise ValueError("B3 probe/gate geometry mismatch")
    mismatch = ~greedy[:, 3]
    metrics = {
        "mean_h2_h4_c64": float(coverage_np[:, 1:4].mean()),
        "h1_c64": float(coverage_np[:, 0].mean()),
        "h4_c64": float(coverage_np[:, 3].mean()),
        "h4_prefix_mismatch_c64": float(coverage_np[mismatch, 3].mean()),
        "h4_prefix_mismatch_rows": int(mismatch.sum()),
        "mean_h1_h4_recall_at_8": float(recall_np.mean()),
        "per_horizon_c64": [float(coverage_np[:, h].mean()) for h in range(4)],
        "per_horizon_recall_at_8": [float(recall_np[:, h].mean()) for h in range(4)],
        "thresholds": dict(GENERATOR_THRESHOLDS),
    }
    metrics["passed"] = all(
        float(metrics[name]) >= threshold
        for name, threshold in GENERATOR_THRESHOLDS.items()
    )
    arrays = {
        "request_ids": np.asarray(requests),
        "source_positions": np.asarray(positions, dtype=np.int64),
        "greedy_match": greedy,
        "candidate_coverage_at_64": coverage_np,
        "recall_at_8": recall_np,
    }
    return metrics, arrays


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != 42:
        raise ValueError("the user-authorized B3 development lineage is seed 42")
    epochs = args.epochs or (12 if args.stage == "generator" else 12)
    learning_rate = args.learning_rate or (
        2.0e-4 if args.stage == "generator" else 2.0e-4
    )
    if epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if EFFECTIVE_BATCH % args.microbatch_size:
        raise ValueError("microbatch must divide effective batch 32")
    for name in (
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "counterfactual_retention_weight",
        "swap_weight",
    ):
        value = learning_rate if name == "learning_rate" else float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    override = verify_b2_override(args.b2_override_record)
    initializer, initializer_manifest, initializer_result = verify_initializer(
        args.initialize_from, stage=args.stage, override=override
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B3 output {output}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(args.seed, deterministic=args.deterministic)

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
    config = production_config(static, bridge)
    model = HARPRTTTeacher(
        bridge,
        config,
        static.geometry,
        token_embedding=static.token_embedding,
    ).to(device)
    runtime_static = runtime_static_artifacts(static, device)
    objective = HARPRTTObjective(
        HARPRTTLossConfig(dimensions=loss_dimensions(config)),
        router_input_basis=static.geometry.input_basis,
        router_rank_mask=static.geometry.rank_mask,
    ).to(device)
    del static
    loaded = load_non_anchor_state(model, initializer)

    fitting_labels, fitting_companion_manifest = load_node_counterfactual_companion(
        args.fitting_companion,
        split="train",
        training=True,
        expected_bindings={"source_commit": args.capture_source_commit},
    )
    fitting_base = HarpRTTDataset(
        args.fitting_index,
        "train",
        corpus_root=args.fitting_corpus,
        max_tree_nodes=32,
    )
    fitting = NodeCounterfactualDatasetAdapter(
        fitting_base, fitting_labels, split="train", training=True
    )
    if len(fitting) != EXPECTED_FITTING_ROWS or len(fitting) != len(fitting_labels):
        raise ValueError("B3 fitting data must contain 4096 one-to-one rows")
    train_data, tune_data, inner_split = fitting_subsets(fitting, fitting_base)

    probe_labels, probe_companion_manifest = load_node_counterfactual_companion(
        args.probe_companion, split="train", training=True
    )
    probe_binding = verify_probe_corpus_binding(
        args.probe_corpus, probe_companion_manifest
    )
    probe_base = HarpRTTDataset(
        args.probe_index,
        "train",
        corpus_root=args.probe_corpus,
        max_tree_nodes=32,
    )
    probe = NodeCounterfactualDatasetAdapter(
        probe_base, probe_labels, split="train", training=True
    )
    if len(probe) != EXPECTED_PROBE_ROWS or len(probe) != len(probe_labels):
        raise ValueError("B3 diagnostic probe must contain 2048 one-to-one rows")
    if set(request_groups(fitting_base)) & set(request_groups(probe_base)):
        raise PermissionError("B3 fitting and diagnostic-probe requests overlap")

    preflight = epoch_zero_audit(
        model,
        tune_data,
        stage=args.stage,
        device=device,
        static=runtime_static,
        workers=args.num_workers,
    )
    write_json_exclusive(output / "EPOCH_ZERO_AUDIT.json", preflight)
    if args.preflight_only:
        write_json_exclusive(
            output / "PREFLIGHT_RESULT.json",
            {
                "schema": "harp_rtt_b3_preflight_v1",
                "stage": args.stage,
                "passed": True,
                "optimizer_constructed": False,
                "formal_validation_opened": False,
                "calibration_opened": False,
                "sealed_test_opened": False,
            },
        )
        return preflight

    groups, ownership = configure_b3_parameters(
        model,
        stage=args.stage,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    schema = GENERATOR_SCHEMA if args.stage == "generator" else RANKER_SCHEMA
    run_manifest = {
        "schema": schema,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "seed": args.seed,
        "source_commit": args.source_commit,
        "capture_source_commit": args.capture_source_commit,
        "initializer": {
            "path": str(args.initialize_from.resolve()),
            "sha256": sha256_file(args.initialize_from),
            "schema": initializer.get("schema"),
            "run_manifest_sha256": initializer.get("run_manifest_sha256"),
            "result_schema": initializer_result.get("schema"),
            "load": loaded,
        },
        "b2_operational_override": {
            "path": str(args.b2_override_record.resolve()),
            "sha256": sha256_file(args.b2_override_record),
            "three_seed_gate_completed": False,
            "user_authorized_b3": True,
        },
        "data": {
            "inner_split": inner_split,
            "fitting_companion_manifest_sha256": sha256_file(
                args.fitting_companion / "manifest.json"
            ),
            "probe_companion_manifest_sha256": sha256_file(
                args.probe_companion / "manifest.json"
            ),
            "probe_oracle_metrics_sha256": sha256_file(args.probe_oracle_metrics),
            "probe_binding": probe_binding,
            "request_disjoint": True,
            "split": "outer_train_only",
        },
        "candidate_policy": {
            "width": 64,
            "anchor_quota": FROZEN_ANCHOR_QUOTA,
            "active_sources": list(ACTIVE_CANDIDATE_SOURCES),
            "source_temperatures": [
                float(value) for value in model.candidate_union.temperatures.cpu()
            ],
            "curriculum": "anchor64_through_10pct_to_anchor40_by_50pct",
        },
        "training": {
            "epochs": epochs,
            "patience": args.patience,
            "effective_batch": EFFECTIVE_BATCH,
            "microbatch": args.microbatch_size,
            "learning_rate": learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "counterfactual_retention_weight": (
                args.counterfactual_retention_weight if args.stage == "generator" else 0.0
            ),
            "swap_weight": args.swap_weight if args.stage == "ranker" else 0.0,
            "random_anytime": args.stage == "generator",
        },
        "parameter_ownership": ownership,
        "anchor": anchor_provenance,
        "router_numerics": router_audit,
        "epoch_zero_audit_sha256": sha256_file(output / "EPOCH_ZERO_AUDIT.json"),
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(output / "run_manifest.json", run_manifest)
    metrics_path = output / "metrics.jsonl"
    best_state: dict[str, Tensor] | None = None
    best_selection: tuple[float, ...] | None = None
    best_epoch = 0
    stale = 0
    global_step = 0
    for epoch in range(1, epochs + 1):
        training, global_step = train_epoch(
            model,
            train_data,
            optimizer,
            objective,
            stage=args.stage,
            epoch=epoch,
            epochs=epochs,
            global_step=global_step,
            microbatch=args.microbatch_size,
            device=device,
            static=runtime_static,
            workers=args.num_workers,
            seed=args.seed,
            gradient_clip=args.gradient_clip,
            counterfactual_retention_weight=args.counterfactual_retention_weight,
            swap_weight=args.swap_weight,
        )
        tuning = evaluation_report(
            model,
            tune_data,
            batch_size=args.microbatch_size,
            device=device,
            static=runtime_static,
            workers=args.num_workers,
        )
        selection = (
            generator_selection(tuning)
            if args.stage == "generator"
            else model_selection_tuple(tuning)
        )
        improved = best_selection is None or selection > best_selection
        if improved:
            best_selection = selection
            best_epoch = epoch
            best_state = non_anchor_state(model)
            stale = 0
        else:
            stale += 1
        append_jsonl(
            metrics_path,
            {
                "schema": schema,
                "event": "epoch",
                "epoch": epoch,
                "global_step": global_step,
                "training": training,
                "tuning": {
                    key: value for key, value in tuning.items() if key != "request_metrics"
                },
                "selection": list(selection),
                "best_epoch": best_epoch,
                "improved": improved,
                "stale_epochs": stale,
            },
        )
        if stale >= args.patience:
            break
    if best_state is None or best_selection is None:
        raise RuntimeError("B3 training produced no checkpoint")
    model.load_state_dict(best_state, strict=False)
    probe_report = evaluation_report(
        model,
        probe,
        batch_size=args.microbatch_size,
        device=device,
        static=runtime_static,
        workers=args.num_workers,
    )
    write_json_exclusive(output / "PROBE_REPORT.json", probe_report)

    if args.stage == "generator":
        gate, arrays = position_level_generator_gate(
            model,
            probe,
            oracle_metrics=args.probe_oracle_metrics,
            batch_size=args.microbatch_size,
            device=device,
            static=runtime_static,
            workers=args.num_workers,
        )
        np.savez_compressed(output / "probe_position_metrics.npz", **arrays)
        result_name = "B3_GENERATOR_RESULT.json"
        checkpoint_name = "best_generator.pt"
        result = {
            "schema": GENERATOR_SCHEMA,
            "stage": "generator",
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "best_selection": list(best_selection),
            "candidate_gate": gate,
            "ranker_training_authorized": bool(gate["passed"]),
        }
    else:
        mean_recall = float(probe_report["mean_h1_h4_request_macro_slot_recall_at_8"])
        gate = {
            "mean_h1_h4_recall_at_8": mean_recall,
            "threshold": RANKER_RECALL_GATE,
            "passed": mean_recall >= RANKER_RECALL_GATE,
        }
        result_name = "B3_RANKER_RESULT.json"
        checkpoint_name = "best_ranker.pt"
        result = {
            "schema": RANKER_SCHEMA,
            "stage": "ranker",
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "best_selection": list(best_selection),
            "ranker_gate": gate,
        }
    checkpoint_path = output / checkpoint_name
    torch.save(
        {
            "schema": schema,
            "stage": args.stage,
            "seed": args.seed,
            "source_commit": args.source_commit,
            "best_epoch": best_epoch,
            "best_selection": best_selection,
            "state_dict_non_anchor": best_state,
            "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
        },
        checkpoint_path,
    )
    result.update(
        {
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "probe_report_sha256": sha256_file(output / "PROBE_REPORT.json"),
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        }
    )
    if args.stage == "generator":
        result["probe_position_metrics_sha256"] = sha256_file(
            output / "probe_position_metrics.npz"
        )
    write_json_exclusive(output / result_name, result)
    checksum_names = [
        "EPOCH_ZERO_AUDIT.json",
        "run_manifest.json",
        "metrics.jsonl",
        "PROBE_REPORT.json",
        checkpoint_name,
        result_name,
    ]
    if args.stage == "generator":
        checksum_names.append("probe_position_metrics.npz")
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for name in checksum_names:
            handle.write(f"{sha256_file(output / name)}  {name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


if __name__ == "__main__":
    run(parse_args())
