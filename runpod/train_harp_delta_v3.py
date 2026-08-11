#!/usr/bin/env python3
"""Train one strictly owned stage of the outer-train HARP-DeltaTree v3 pilot."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.anchor import LegacyHARPAnchorBridge  # noqa: E402
from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt  # noqa: E402
from harp_rtt.delta import HARPDeltaConfig, HARPDeltaTeacher  # noqa: E402
from harp_rtt.delta_batch import prepare_delta_batch  # noqa: E402
from harp_rtt.delta_training import (  # noqa: E402
    DeltaLoss,
    DeltaStage,
    candidate_loss,
    configure_delta_stage,
    ranker_loss,
    semantic_loss,
    set_delta_stage_mode,
)
from harp_rtt.exact_k import stable_topk  # noqa: E402
from harp_rtt.node_counterfactual import (  # noqa: E402
    NodeCounterfactualDatasetAdapter,
    load_node_counterfactual_companion,
)
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.train import prepare_model_batch, runtime_static_artifacts  # noqa: E402
from harp_rtt.training import move_to_device, seed_everything, sha256_file  # noqa: E402
from harp_rtt.v2_losses import node_target_branch_distribution  # noqa: E402


SCHEMA = "harp_delta_v3_stage_training_v1"
PARTITION_SCHEMA = "harp_delta_v3_20k_partition_v1"
EXPECTED_ROWS = {"train": 16_000, "tune": 2_000, "development": 2_000}
PREDECESSOR: dict[str, str | None] = {
    "semantic": None,
    "candidate": "semantic",
    "ranker": "candidate",
    "calibration": "ranker",
}
DEFAULT_EPOCHS = {"semantic": 30, "candidate": 15, "ranker": 15, "calibration": 5}
DEFAULT_LR = {"semantic": 3e-4, "candidate": 1e-3, "ranker": 3e-4, "calibration": 5e-5}
EFFECTIVE_BATCH = 32


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(PREDECESSOR), required=True)
    for split in ("train", "tune", "development"):
        parser.add_argument(f"--{split}-index", type=Path, required=True)
        parser.add_argument(f"--{split}-corpus", type=Path, required=True)
        parser.add_argument(f"--{split}-companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--microbatch-size", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != PARTITION_SCHEMA:
        raise ValueError("Delta v3 partition schema mismatch")
    if value.get("outer_split") != "train" or value.get("positions") != 20_000:
        raise PermissionError("Delta v3 stage requires the frozen 20k outer-train pilot")
    for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if value.get(key) is not False:
            raise PermissionError(f"Delta partition violates {key}")
    if value.get("counterfactual_labels_are_targets_only") is not True:
        raise PermissionError("Delta partition does not seal counterfactual labels")
    return value


def _request_counts(dataset: HarpRTTDataset) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in dataset.records:
        segment = dataset.segments[record.segment]
        counts[str(segment.sequences[record.sequence]["request_id"])] += 1
    return counts


def load_split(args: argparse.Namespace, split: str) -> Dataset[Any]:
    index = getattr(args, f"{split}_index")
    corpus = getattr(args, f"{split}_corpus")
    companion = getattr(args, f"{split}_companion")
    base = HarpRTTDataset(index, "train", corpus_root=corpus, max_tree_nodes=32)
    counts = _request_counts(base)
    expected_requests = EXPECTED_ROWS[split] // 16
    if len(base) != EXPECTED_ROWS[split] or len(counts) != expected_requests:
        raise ValueError(f"Delta {split} split has invalid row/request counts")
    if set(counts.values()) != {16}:
        raise ValueError(f"Delta {split} requests do not each contain 16 positions")
    labels, manifest = load_node_counterfactual_companion(
        companion, split="train", training=True
    )
    if manifest.get("sealed_test_opened") is not False:
        raise PermissionError(f"Delta {split} companion crossed sealed test")
    joined = NodeCounterfactualDatasetAdapter(base, labels, split="train", training=True)
    if len(joined) != len(base):
        raise ValueError(f"Delta {split} companion join is incomplete")
    return joined


def loader(
    dataset: Dataset[Any], *, batch: int, shuffle: bool, seed: int,
    workers: int, device: torch.device,
) -> DataLoader[Any]:
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        collate_fn=collate_harp_rtt, drop_last=False,
    )


def load_initializer(model: HARPDeltaTeacher, path: Path, stage: str) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Delta initializer schema mismatch")
    expected = PREDECESSOR[stage]
    if value.get("stage") != expected:
        raise ValueError(f"Delta {stage} requires a completed {expected} checkpoint")
    if value.get("config") != model.config.to_dict():
        raise ValueError("Delta initializer architecture differs from this run")
    model.load_state_dict(value["model_state_dict"], strict=True)
    return value


def _counterfactual_with_posterior(
    counterfactual: Mapping[str, Tensor], nodes: int
) -> dict[str, Tensor]:
    result = dict(counterfactual)
    masks = result["budget_node_masks"].bool()
    if masks.ndim != 3 or masks.shape[1] < 3:
        raise ValueError("Delta semantic stage requires nested 4/8/16/all masks")
    posterior, _ = node_target_branch_distribution(
        result, captured_nodes=nodes, selection_mask=masks[:, 2]
    )
    result["target_path_distribution"] = posterior
    return result


def forward_batch(
    *, model: HARPDeltaTeacher, anchor: LegacyHARPAnchorBridge,
    host: Mapping[str, Any], runtime_static: Any, token_embedding: Tensor,
    input_basis: Tensor, rank_mask: Tensor, device: torch.device,
) -> tuple[Any, dict[str, Any], dict[str, Tensor], Tensor]:
    batch = move_to_device(host, device)
    prepared = prepare_model_batch(batch, runtime_static)
    with torch.no_grad():
        anchor_output = anchor(batch=prepared)
        anchor_scores = anchor_output["future_router_scores"][:, :4].float()
    delta = prepare_delta_batch(
        prepared, anchor_scores=anchor_scores, token_embedding=token_embedding,
        input_basis=input_basis, rank_mask=rank_mask, config=model.config,
    )
    targets = dict(prepared["targets"])
    targets["factual_branch_index"] = delta.factual_branch_index
    future_router_inputs = targets["future_router_inputs"]
    with torch.autocast(device_type=device.type, enabled=False):
        targets["future_query_coordinates"] = (
            torch.einsum(
                "bhld,ldr->bhlr", future_router_inputs.float(), input_basis.float()
            ) * rank_mask[None, None].float()
        )[:, 0]
    counterfactual = _counterfactual_with_posterior(
        targets["counterfactual"], model.config.max_tree_nodes
    )
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        output = model(**delta.model_inputs)
    return output, targets, counterfactual, anchor_scores


def objective(
    stage: DeltaStage, model: HARPDeltaTeacher, output: Any,
    anchor_scores: Tensor, targets: Mapping[str, Tensor],
    counterfactual: Mapping[str, Tensor],
) -> DeltaLoss:
    if stage == "semantic":
        return semantic_loss(output, targets, counterfactual, budget_index=2)
    if stage == "candidate":
        return candidate_loss(model.core, output, anchor_scores, targets)
    if stage == "ranker":
        return ranker_loss(output, anchor_scores, targets)
    candidate = candidate_loss(model.core, output, anchor_scores, targets)
    ranked = ranker_loss(output, anchor_scores, targets)
    return DeltaLoss(
        candidate.total + ranked.total,
        {**{f"candidate_{k}": v for k, v in candidate.components.items()},
         **{f"ranker_{k}": v for k, v in ranked.components.items()}},
        ranked.outside_candidate_slots,
    )


@torch.no_grad()
def evaluate(
    *, stage: DeltaStage, model: HARPDeltaTeacher,
    anchor: LegacyHARPAnchorBridge, dataset: Dataset[Any], runtime_static: Any,
    token_embedding: Tensor, input_basis: Tensor, rank_mask: Tensor,
    device: torch.device, microbatch: int, workers: int,
) -> dict[str, float | int]:
    model.eval(); anchor.eval()
    totals = {"loss": 0.0, "recall": 0.0, "coverage": 0.0}
    rows = outside = 0
    for host in loader(
        dataset, batch=microbatch, shuffle=False, seed=0,
        workers=workers, device=device,
    ):
        output, targets, counterfactual, anchor_logits = forward_batch(
            model=model, anchor=anchor, host=host, runtime_static=runtime_static,
            token_embedding=token_embedding, input_basis=input_basis,
            rank_mask=rank_mask, device=device,
        )
        loss = objective(stage, model, output, anchor_logits, targets, counterfactual)
        labels = targets["future_selected_ids"].long()
        valid = targets["future_available"].bool()
        recall = (labels[..., :, None] == output.final_ids[..., None, :]).any(-1).float()
        coverage = (labels[..., :, None] == output.candidate_ids[..., None, :]).any(-1).float()
        active = int(valid.sum().item())
        totals["loss"] += float(loss.total) * active
        totals["recall"] += float((recall * valid[..., None]).sum())
        totals["coverage"] += float((coverage * valid[..., None]).sum())
        rows += active
        outside += loss.outside_candidate_slots
    slots = rows * model.config.exact_k
    return {
        "loss": totals["loss"] / max(1, rows),
        "recall_at_8": totals["recall"] / max(1, slots),
        "candidate_coverage_at_64": totals["coverage"] / max(1, slots),
        "valid_layer_horizons": rows,
        "outside_candidate_slots": outside,
    }


def selection_value(stage: DeltaStage, metrics: Mapping[str, float | int]) -> float:
    if stage == "semantic":
        return -float(metrics["loss"])
    if stage == "candidate":
        return float(metrics["candidate_coverage_at_64"])
    return float(metrics["recall_at_8"])


def main() -> None:
    args = parse_args()
    stage: DeltaStage = args.stage
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Delta run {args.output}")
    if (PREDECESSOR[stage] is None) != (args.initialize_from is None):
        raise ValueError("Delta initializer presence disagrees with stage lineage")
    if args.microbatch_size > EFFECTIVE_BATCH or EFFECTIVE_BATCH % args.microbatch_size:
        raise ValueError("microbatch must divide effective batch 32")
    partition = _manifest(args.partition_manifest)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    seed_everything(args.seed, deterministic=args.deterministic)
    datasets = {name: load_split(args, name) for name in EXPECTED_ROWS}

    static = load_static_target_artifacts(args.static_dir, device="cpu")
    if static.token_embedding is None:
        raise RuntimeError("Delta training requires frozen token embeddings")
    config = HARPDeltaConfig(
        experts=static.geometry.experts,
        layers=static.geometry.layers,
        router_rank=static.geometry.maximum_rank,
    )
    model = HARPDeltaTeacher(
        config, static.geometry.expert_keys, static.geometry.centered_bias,
        raw_width=static.geometry.hidden_width,
        target_control_width=static.geometry.maximum_rank,
        metadata_width=8,
    ).to(device)
    anchor, anchor_provenance = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint, args.target_preprocessing, args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    anchor = anchor.to(device).requires_grad_(False).eval()
    initializer = None
    if args.initialize_from is not None:
        initializer = load_initializer(model, args.initialize_from, stage)
    ownership = configure_delta_stage(model, stage)
    if sum(parameter.numel() for parameter in model.parameters()) > 25_000_000:
        raise RuntimeError("Delta trainable architecture exceeds the 25M cap")

    token_embedding = static.token_embedding.to(device)
    input_basis = static.geometry.input_basis.to(device)
    rank_mask = static.geometry.rank_mask.to(device)
    runtime_static = runtime_static_artifacts(static, device)
    del static
    args.output.mkdir(parents=True)
    run_manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "seed": args.seed,
        "source_commit": args.source_commit,
        "config": config.to_dict(),
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "partition_schema": partition["schema"],
        "trainable_parameters": ownership.trainable_parameters,
        "trainable_names": list(ownership.trainable_names),
        "anchor_provenance": anchor_provenance,
        "initializer_sha256": None if args.initialize_from is None else sha256_file(args.initialize_from),
        "effective_batch": EFFECTIVE_BATCH,
        "microbatch": args.microbatch_size,
        "counterfactual_budget": 16,
        "counterfactual_model_input": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "optimizer_constructed": False,
    }
    write_json_exclusive(args.output / "run_manifest.json", run_manifest)

    host = next(iter(loader(
        datasets["train"], batch=1, shuffle=False, seed=0,
        workers=args.num_workers, device=device,
    )))
    model.eval()
    output, _, _, anchor_scores = forward_batch(
        model=model, anchor=anchor, host=host, runtime_static=runtime_static,
        token_embedding=token_embedding, input_basis=input_basis,
        rank_mask=rank_mask, device=device,
    )
    epoch_zero = {
        "final_top8_equals_anchor": bool(torch.equal(output.final_ids, stable_topk(anchor_scores, 8))),
        "candidate_top64_equals_anchor": bool(torch.equal(output.candidate_ids, stable_topk(anchor_scores, 64))),
        "anchor_quota_all_64": bool((output.anchor_quotas == 64).all()),
        "optimizer_constructed": False,
    }
    if stage == "semantic" and not all(
        epoch_zero[key] for key in (
            "final_top8_equals_anchor", "candidate_top64_equals_anchor", "anchor_quota_all_64"
        )
    ):
        raise RuntimeError("Delta epoch-zero anchor-protection audit failed")
    write_json_exclusive(args.output / "EPOCH_ZERO_AUDIT.json", epoch_zero)
    if args.preflight_only:
        write_json_exclusive(
            args.output / "PREFLIGHT_RESULT.json",
            {**epoch_zero, "stage": stage, "training_started": False},
        )
        return

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate or DEFAULT_LR[stage],
        weight_decay=args.weight_decay,
    )
    run_manifest["optimizer_constructed"] = True
    epochs = args.epochs or DEFAULT_EPOCHS[stage]
    accumulation = EFFECTIVE_BATCH // args.microbatch_size
    best_value = -math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    metrics_path = args.output / "metrics.jsonl"
    for epoch in range(1, epochs + 1):
        set_delta_stage_mode(model, stage)
        anchor.eval(); optimizer.zero_grad(set_to_none=True)
        total = batches = 0.0
        train_loader = loader(
            datasets["train"], batch=args.microbatch_size, shuffle=True,
            seed=args.seed + epoch, workers=args.num_workers, device=device,
        )
        for step, host in enumerate(train_loader, start=1):
            output, targets, counterfactual, anchor_scores = forward_batch(
                model=model, anchor=anchor, host=host, runtime_static=runtime_static,
                token_embedding=token_embedding, input_basis=input_basis,
                rank_mask=rank_mask, device=device,
            )
            loss = objective(stage, model, output, anchor_scores, targets, counterfactual)
            (loss.total / accumulation).backward()
            if step % accumulation == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.total.detach()); batches += 1
        tune = evaluate(
            stage=stage, model=model, anchor=anchor, dataset=datasets["tune"],
            runtime_static=runtime_static, token_embedding=token_embedding,
            input_basis=input_basis, rank_mask=rank_mask, device=device,
            microbatch=args.microbatch_size, workers=args.num_workers,
        )
        record = {"epoch": epoch, "train_loss": total / max(1, batches), "tune": tune}
        append_jsonl(metrics_path, record)
        value = selection_value(stage, tune)
        if value > best_value:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Delta training produced no selectable checkpoint")
    model.load_state_dict(best_state, strict=True)
    development = evaluate(
        stage=stage, model=model, anchor=anchor, dataset=datasets["development"],
        runtime_static=runtime_static, token_embedding=token_embedding,
        input_basis=input_basis, rank_mask=rank_mask, device=device,
        microbatch=args.microbatch_size, workers=args.num_workers,
    )
    checkpoint = {
        "schema": SCHEMA,
        "stage": stage,
        "seed": args.seed,
        "source_commit": args.source_commit,
        "config": config.to_dict(),
        "best_epoch": best_epoch,
        "best_tune_value": best_value,
        "development": development,
        "model_state_dict": best_state,
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "run_manifest_sha256": sha256_file(args.output / "run_manifest.json"),
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    checkpoint_path = args.output / "best_delta_stage.pt"
    torch.save(checkpoint, checkpoint_path)
    result = {
        "schema": "harp_delta_v3_stage_result_v1",
        "stage": stage,
        "best_epoch": best_epoch,
        "best_tune_value": best_value,
        "development": development,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_started": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
