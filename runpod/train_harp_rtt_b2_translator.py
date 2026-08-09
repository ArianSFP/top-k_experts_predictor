#!/usr/bin/env python3
"""Train one B2 branch-semantic translator seed and evaluate the frozen probe."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.anchor import LegacyHARPAnchorBridge
from harp_rtt.b15 import candidate_union
from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt
from harp_rtt.exact_k import stable_topk
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
from harp_rtt.training import move_to_device, seed_everything, sha256_file
from harp_rtt.v2_losses import (
    node_counterfactual_posterior_loss,
    node_counterfactual_semantic_loss,
)


SCHEMA = "harp_rtt_b2_translator_training_v1"
EFFECTIVE_BATCH = 32
BUDGET_INDEX = 2
ANCHOR_QUOTA = 40
POSTERIOR_WEIGHT = 0.1
TUNING_REQUESTS = 32
TRAIN_REQUESTS = 224
EXPECTED_FITTING_ROWS = 4096
EXPECTED_PROBE_ROWS = 2048
B2_PREFIXES = ("tree_encoder.", "score_head.")
B2_EXCLUDED_PARAMETERS = {
    "tree_encoder.acceptance.weight",
    "tree_encoder.acceptance.bias",
    "score_head.geometry_gate",
    "score_head.free_gate",
    "score_head.transition_gate",
}


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
    parser.add_argument("--fitting-index", type=Path, required=True)
    parser.add_argument("--fitting-corpus", type=Path, required=True)
    parser.add_argument("--fitting-companion", type=Path, required=True)
    parser.add_argument("--probe-index", type=Path, required=True)
    parser.add_argument("--probe-corpus", type=Path)
    parser.add_argument("--probe-companion", type=Path, required=True)
    parser.add_argument("--probe-oracle-metrics", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--router-numerics-audit", type=Path, required=True)
    parser.add_argument("--router-numerics-audit-sha256", required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--b15-override-record", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--microbatch-size", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def verify_operational_override(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    required = {
        "schema": "harp_rtt_b15_user_operational_override_v1",
        "user_authorized": True,
        "operational_pass": True,
        "measured_threshold_pass": False,
        "thresholds_changed": False,
        "fallback_policy_selected": False,
    }
    for name, expected in required.items():
        if value.get(name) != expected:
            raise PermissionError(f"B1.5 operational override disagrees on {name}")
    measured = value.get("measured_confirmation")
    if not isinstance(measured, Mapping):
        raise ValueError("B1.5 override lacks measured confirmation")
    if abs(float(measured.get("mean_h2_h4_c64", -1)) - 0.9828948974609375) > 1e-12:
        raise ValueError("B1.5 override changed the measured mean confirmation result")
    return value


def verify_probe_corpus_binding(
    corpus_root: Path,
    companion_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when a relocated probe corpus is missing or misbound."""

    segment_link = corpus_root.expanduser().resolve() / "segments" / "stage_a"
    try:
        segment = segment_link.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"probe corpus segment is missing or has a broken relocation link: {segment_link}"
        ) from error
    if not segment.is_dir():
        raise NotADirectoryError(f"probe corpus segment is not a directory: {segment}")

    bindings = companion_manifest.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("probe companion lacks immutable base-capture bindings")
    bound_files = {
        "base_capture_manifest_sha256": segment / "run_manifest.json",
        "base_capture_checksums_sha256": segment / "SHA256SUMS",
        "base_capture_audit_sha256": segment / "CAPTURE_AUDIT_ADAPTIVE.json",
    }
    verified: dict[str, str] = {}
    for binding, path in bound_files.items():
        expected = bindings.get(binding)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"probe companion lacks valid {binding}")
        if not path.is_file():
            raise FileNotFoundError(f"bound probe artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"probe corpus disagrees with companion {binding}")
        verified[binding] = actual

    ledger = bound_files["base_capture_checksums_sha256"]
    listed = 0
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"malformed probe checksum ledger line: {line!r}")
        relative = Path(parts[1].lstrip("*"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("probe checksum ledger contains an unsafe relative path")
        if not (segment / relative).is_file():
            raise FileNotFoundError(
                f"probe checksum ledger references a missing artifact: {relative}"
            )
        listed += 1
    if listed == 0:
        raise ValueError("probe checksum ledger is empty")
    return {
        "corpus_root": str(corpus_root.expanduser().resolve()),
        "segment_link": str(segment_link),
        "resolved_segment": str(segment),
        "listed_artifacts_present": listed,
        "bindings": verified,
    }


def configure_b2_parameters(model: nn.Module) -> dict[str, Any]:
    trainable, frozen = [], []
    for name, parameter in model.named_parameters():
        active = (
            any(name.startswith(prefix) for prefix in B2_PREFIXES)
            and name not in B2_EXCLUDED_PARAMETERS
        )
        parameter.requires_grad_(active)
        (trainable if active else frozen).append(name)
    if not trainable:
        raise ValueError("B2 ownership selected no parameters")
    if any(name.startswith("anchor.") for name in trainable):
        raise PermissionError("B2 must keep the HARP anchor frozen")
    if any(name.startswith("reranker.") for name in trainable):
        raise PermissionError("B2 must keep the rich reranker frozen")
    if hasattr(model, "_anchor_frozen"):
        setattr(model, "_anchor_frozen", True)
    anchor = getattr(model, "anchor", None)
    if isinstance(anchor, nn.Module):
        anchor.eval()
    return {
        "trainable_names": trainable,
        "frozen_names": frozen,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "prefixes": list(B2_PREFIXES),
        "excluded_unsupervised_parameters": sorted(B2_EXCLUDED_PARAMETERS),
        "anchor_frozen": True,
        "reranker_frozen": True,
    }


def request_groups(base: HarpRTTDataset) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(base.records):
        request = str(base.segments[record.segment].sequences[record.sequence]["request_id"])
        groups[request].append(index)
    return dict(groups)


def fitting_subsets(
    dataset: NodeCounterfactualDatasetAdapter,
    base: HarpRTTDataset,
) -> tuple[Subset[Any], Subset[Any], dict[str, Any]]:
    groups = request_groups(base)
    if len(groups) != TRAIN_REQUESTS + TUNING_REQUESTS or len(base) != EXPECTED_FITTING_ROWS:
        raise ValueError("B2 fitting inventory must be 256 requests and 4096 rows")
    if any(len(indices) != 16 for indices in groups.values()):
        raise ValueError("every B2 fitting request must contribute exactly 16 rows")
    ranked = sorted(
        groups,
        key=lambda request: (
            hashlib.sha256(f"harp-rtt-b2-tune\0{request}".encode()).digest(),
            request,
        ),
    )
    tuning_requests = set(ranked[:TUNING_REQUESTS])
    training_requests = set(ranked[TUNING_REQUESTS:])
    train_indices = sorted(
        index for request in training_requests for index in groups[request]
    )
    tune_indices = sorted(index for request in tuning_requests for index in groups[request])
    if len(train_indices) != TRAIN_REQUESTS * 16 or len(tune_indices) != TUNING_REQUESTS * 16:
        raise AssertionError("request-grouped B2 inner split has the wrong size")
    manifest = {
        "selection": "sha256(harp-rtt-b2-tune\\0request_id)",
        "training_requests": sorted(training_requests),
        "tuning_requests": sorted(tuning_requests),
        "training_rows": len(train_indices),
        "tuning_rows": len(tune_indices),
        "request_disjoint": not bool(training_requests & tuning_requests),
    }
    return Subset(dataset, train_indices), Subset(dataset, tune_indices), manifest


def loader(
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


def semantic_forward(
    model: nn.Module,
    batch: Mapping[str, Any],
    static: Any,
    *,
    random_anytime: bool,
) -> Mapping[str, Tensor]:
    prepared = prepare_model_batch(batch, static)
    output = model(
        batch=prepared,
        anchor_inputs={"batch": prepared},
        random_anytime_truncation=random_anytime,
        branch_semantic_only=True,
    )
    if not isinstance(output, Mapping):
        raise TypeError("B2 semantic-only model output must be a mapping")
    return output


def b2_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
) -> tuple[Tensor, dict[str, float]]:
    counterfactual = batch["targets"]["counterfactual"]
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
    total = semantic.total + POSTERIOR_WEIGHT * posterior
    return total, {
        "total": float(total.detach()),
        "exact_set": float(semantic.exact_set.detach()),
        "query": float(semantic.query.detach()),
        "router_kl": float(semantic.router_kl.detach()),
        "posterior": float(posterior.detach()),
        "active_depth_divergence_cells": semantic.active_path_depth_cells,
    }


def evaluate_loss(
    model: nn.Module,
    dataset: Dataset[Any],
    *,
    batch_size: int,
    device: torch.device,
    static: Any,
    workers: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    examples = 0
    with torch.inference_mode():
        for host in loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            workers=workers,
            device=device,
        ):
            batch = move_to_device(host, device)
            examples_in_batch = int(batch["targets"]["counterfactual"]["node_mask"].shape[0])
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = semantic_forward(model, batch, static, random_anytime=False)
                _, metrics = b2_loss(outputs, batch)
            for name, value in metrics.items():
                totals[name] += float(value) * examples_in_batch
            examples += examples_in_batch
    if not examples:
        raise ValueError("B2 tuning dataset is empty")
    return {name: value / examples for name, value in totals.items()}


def train_epoch(
    model: nn.Module,
    dataset: Dataset[Any],
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    microbatch: int,
    device: torch.device,
    static: Any,
    workers: int,
    seed: int,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    accumulation = EFFECTIVE_BATCH // microbatch
    if EFFECTIVE_BATCH % microbatch:
        raise ValueError("microbatch must divide the effective batch")
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = defaultdict(float)
    examples = 0
    window_examples = 0
    batches = loader(
        dataset,
        batch_size=microbatch,
        shuffle=True,
        seed=seed * 1000 + epoch,
        workers=workers,
        device=device,
    )
    for index, host in enumerate(batches):
        batch = move_to_device(host, device)
        batch_examples = int(batch["targets"]["counterfactual"]["node_mask"].shape[0])
        window_examples += batch_examples
        examples += batch_examples
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = semantic_forward(model, batch, static, random_anytime=True)
            loss, metrics = b2_loss(outputs, batch)
        (loss * (batch_examples / EFFECTIVE_BATCH)).backward()
        for name, value in metrics.items():
            totals[name] += float(value) * batch_examples
        step = (index + 1) % accumulation == 0 or index + 1 == len(batches)
        if not step:
            continue
        if window_examples < EFFECTIVE_BATCH:
            correction = EFFECTIVE_BATCH / window_examples
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=gradient_clip,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        window_examples = 0
    return {name: value / examples for name, value in totals.items()}


def coverage(candidate_ids: Tensor, target_ids: Tensor) -> Tensor:
    return (
        (candidate_ids[..., :, None] == target_ids[..., None, :])
        .any(dim=-2)
        .float()
        .mean(dim=-1)
    )


def learned_branch_mass(
    outputs: Mapping[str, Tensor],
    counterfactual: Mapping[str, Tensor],
) -> Tensor:
    scores = outputs["branch_semantic_scores"].float()
    posterior = outputs["branch_posterior_logits"].float()
    visible = outputs["branch_mask"].bool()
    nodes = int(counterfactual["node_mask"].shape[1])
    selection = counterfactual["budget_node_masks"][:, BUDGET_INDEX].bool()
    depth = counterfactual["depth"].long()
    predicted = stable_topk(scores[..., :nodes, :], 8)
    mass = scores.new_zeros(scores.shape[:3] + (scores.shape[-1],))
    for horizon in range(1, 4):
        selected = selection & (depth == horizon + 1) & visible[:, horizon, :nodes]
        allowed = torch.cat(
            [selected, torch.ones_like(selected[:, :1])], dim=-1
        )
        weights = torch.softmax(
            posterior[:, horizon].masked_fill(~allowed, -torch.inf), dim=-1
        )[:, :nodes]
        ids = predicted[:, horizon]
        source = weights[:, None, :, None].expand_as(ids).reshape(
            ids.shape[0], ids.shape[1], -1
        )
        mass[:, horizon].scatter_add_(
            -1,
            ids.reshape(ids.shape[0], ids.shape[1], -1),
            source,
        )
    return mass


def evaluate_probe(
    model: nn.Module,
    dataset: NodeCounterfactualDatasetAdapter,
    *,
    oracle_metrics_path: Path,
    batch_size: int,
    device: torch.device,
    static: Any,
    workers: int,
    output: Path,
) -> dict[str, Any]:
    if len(dataset) != EXPECTED_PROBE_ROWS:
        raise ValueError("B2 probe must contain exactly 2048 untouched rows")
    model.eval()
    requests: list[str] = []
    positions: list[int] = []
    learned_rows, anchor_rows = [], []
    with torch.inference_mode():
        for host in loader(
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
                outputs = semantic_forward(model, batch, static, random_anytime=False)
            anchor_scores = outputs["future_router_scores"][:, :4].float()
            branch_mass = learned_branch_mass(
                outputs, batch["targets"]["counterfactual"]
            )
            candidates = candidate_union(
                anchor_scores,
                branch_mass,
                anchor_quota=ANCHOR_QUOTA,
                width=64,
            )
            anchor_candidates = stable_topk(anchor_scores, 64)
            target_ids = batch["targets"]["future_selected_ids"].long()
            learned_rows.append(coverage(candidates, target_ids).cpu().numpy())
            anchor_rows.append(coverage(anchor_candidates, target_ids).cpu().numpy())
    learned = np.concatenate(learned_rows)
    anchor = np.concatenate(anchor_rows)
    source = np.load(oracle_metrics_path)
    lookup = {
        (str(request), int(position)): index
        for index, (request, position) in enumerate(
            zip(source["request_ids"], source["source_positions"], strict=True)
        )
    }
    order = np.asarray(
        [lookup[(request, position)] for request, position in zip(requests, positions, strict=True)]
    )
    if len(set(zip(requests, positions, strict=True))) != EXPECTED_PROBE_ROWS:
        raise ValueError("B2 probe join keys are not unique")
    oracle = source["target_quota_40_24_coverage_at_64"][order]
    greedy_match = source["greedy_match"][order]
    if oracle.shape != learned.shape or anchor.shape != learned.shape:
        raise ValueError("B2 learned/anchor/oracle probe geometry disagrees")
    metrics: dict[str, Any] = {}
    for horizon in (2, 3, 4):
        index = horizon - 1
        mismatch = ~greedy_match[:, index]
        values = {
            "rows": int(mismatch.sum()),
            "anchor": float(anchor[mismatch, index].mean()),
            "learned": float(learned[mismatch, index].mean()),
            "oracle": float(oracle[mismatch, index].mean()),
        }
        denominator = values["oracle"] - values["anchor"]
        values["recovery_g"] = (
            (values["learned"] - values["anchor"]) / denominator
            if denominator > 0
            else float("nan")
        )
        metrics[f"H{horizon}_mismatch"] = values
    metrics["mean_h2_h4"] = {
        "anchor": float(anchor[:, 1:4].mean()),
        "learned": float(learned[:, 1:4].mean()),
        "oracle": float(oracle[:, 1:4].mean()),
    }
    np.savez_compressed(
        output,
        request_ids=np.asarray(requests),
        source_positions=np.asarray(positions, dtype=np.int64),
        greedy_match=greedy_match,
        learned_coverage_at_64=learned,
        anchor_coverage_at_64=anchor,
        oracle_coverage_at_64=oracle,
    )
    return metrics


def non_anchor_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("anchor.") and name != "token_embedding.weight"
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if args.microbatch_size > EFFECTIVE_BATCH or EFFECTIVE_BATCH % args.microbatch_size:
        raise ValueError("microbatch size must divide effective batch 32")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B2 seed output {output}")
    override = verify_operational_override(args.b15_override_record)
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
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
    del static
    ownership = configure_b2_parameters(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    fitting_labels, fitting_companion_manifest = load_node_counterfactual_companion(
        args.fitting_companion,
        split="train",
        training=True,
        expected_bindings={"source_commit": args.source_commit},
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
    if len(fitting) != len(fitting_labels):
        raise ValueError("B2 fitting source/label join is not one-to-one")
    train_data, tune_data, inner_split = fitting_subsets(fitting, fitting_base)

    probe_labels, probe_companion_manifest = load_node_counterfactual_companion(
        args.probe_companion, split="train", training=True
    )
    probe_corpus_binding = verify_probe_corpus_binding(
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
    if len(probe) != len(probe_labels) or len(probe) != EXPECTED_PROBE_ROWS:
        raise ValueError("B2 held-out probe source/label join is invalid")
    fitting_requests = set(request_groups(fitting_base))
    probe_requests = set(request_groups(probe_base))
    if fitting_requests & probe_requests:
        raise PermissionError("B2 fitting and diagnostic-probe requests overlap")

    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source_commit": args.source_commit,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "operational_gate_override": {
            "path": str(args.b15_override_record.resolve()),
            "sha256": sha256_file(args.b15_override_record),
            "measured_threshold_pass": False,
            "user_authorized_operational_pass": True,
            "thresholds_changed": False,
            "measured_confirmation": override["measured_confirmation"],
        },
        "training_contract": {
            "effective_batch": EFFECTIVE_BATCH,
            "microbatch": args.microbatch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "budget": 16,
            "runtime_tree_nodes": 32,
            "posterior_weight": POSTERIOR_WEIGHT,
            "query_weight": 0.2,
            "router_kl_weight": 0.1,
            "random_anytime_training": [1, 4, 8, 16, 32],
        },
        "ownership": ownership,
        "inner_split": inner_split,
        "request_disjoint_probe": True,
        "fitting_companion_manifest_sha256": sha256_file(
            args.fitting_companion / "manifest.json"
        ),
        "probe_companion_manifest_sha256": sha256_file(
            args.probe_companion / "manifest.json"
        ),
        "probe_corpus_binding": probe_corpus_binding,
        "probe_oracle_metrics_sha256": sha256_file(args.probe_oracle_metrics),
        "anchor": anchor_provenance,
        "router_numerics": router_audit,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "h1_training_started": False,
        "b3_training_started": False,
    }
    write_json_exclusive(output / "run_manifest.json", manifest)
    metrics_path = output / "metrics.jsonl"

    best_value = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(
            model,
            train_data,
            optimizer,
            epoch=epoch,
            microbatch=args.microbatch_size,
            device=device,
            static=runtime_static,
            workers=args.num_workers,
            seed=args.seed,
            gradient_clip=args.gradient_clip,
        )
        tuning = evaluate_loss(
            model,
            tune_data,
            batch_size=args.microbatch_size,
            device=device,
            static=runtime_static,
            workers=args.num_workers,
        )
        improved = float(tuning["total"]) < best_value - 1e-8
        if improved:
            best_value = float(tuning["total"])
            best_epoch = epoch
            best_state = non_anchor_state(model)
            stale = 0
        else:
            stale += 1
        event = {
            "schema": SCHEMA,
            "event": "epoch",
            "epoch": epoch,
            "training": training,
            "tuning": tuning,
            "improved": improved,
            "best_epoch": best_epoch,
            "stale_epochs": stale,
        }
        append_jsonl(metrics_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("B2 training produced no finite best checkpoint")
    model.load_state_dict(best_state, strict=False)
    checkpoint = output / "best_translator.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_tuning_total": best_value,
            "source_commit": args.source_commit,
            "state_dict_non_anchor": best_state,
            "ownership": ownership,
            "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
        },
        checkpoint,
    )
    probe_npz = output / "probe_request_metrics.npz"
    probe_metrics = evaluate_probe(
        model,
        probe,
        oracle_metrics_path=args.probe_oracle_metrics,
        batch_size=args.microbatch_size,
        device=device,
        static=runtime_static,
        workers=args.num_workers,
        output=probe_npz,
    )
    result = {
        "schema": SCHEMA,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_tuning_total": best_value,
        "epochs_completed": epoch,
        "stopped_for_patience": stale >= args.patience,
        "probe_metrics": probe_metrics,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "probe_request_metrics": probe_npz.name,
        "probe_request_metrics_sha256": sha256_file(probe_npz),
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
