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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.anchor import LegacyHARPAnchorBridge  # noqa: E402
from harp_rtt.b31 import quota_candidate_union  # noqa: E402
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
REUSE_PARTITION_SCHEMA = "harp_rtt_b2_translator_fitting_partition_v1"
REUSE_SPLIT_SCHEMA = "harp_rtt_b2_translator_training_v1"
B31_REPORT_SCHEMA = "harp_rtt_b31_factorial_report_v1"
DATA_PROFILES: dict[str, dict[str, Any]] = {
    "delta20k": {
        "rows": {"train": 16_000, "tune": 2_000, "development": 2_000},
        "requests": {"train": 1_000, "tune": 125, "development": 125},
        "partition_schema": PARTITION_SCHEMA,
        "diagnostic_reuse": False,
    },
    "b2_reuse_4096": {
        "rows": {"train": 3_584, "tune": 512, "development": 2_048},
        "requests": {"train": 224, "tune": 32, "development": 128},
        "partition_schema": REUSE_PARTITION_SCHEMA,
        "diagnostic_reuse": True,
    },
}
PREDECESSOR: dict[str, str | None] = {
    "semantic": None,
    "candidate": "semantic",
    "ranker": "candidate",
    "calibration": "ranker",
}
DEFAULT_EPOCHS = {"semantic": 30, "candidate": 15, "ranker": 15, "calibration": 5}
DEFAULT_LR = {"semantic": 3e-4, "candidate": 1e-3, "ranker": 3e-4, "calibration": 5e-5}
EFFECTIVE_BATCH = 32
MICROBATCH_CHOICES = (1, 2, 4, 8, 16, 32)


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
    parser.add_argument(
        "--data-profile", choices=tuple(DATA_PROFILES), default="delta20k"
    )
    parser.add_argument("--reuse-split-manifest", type=Path)
    parser.add_argument("--b31-report", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        choices=MICROBATCH_CHOICES,
        default=2,
    )
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


def _manifest(path: Path, profile: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    contract = DATA_PROFILES[profile]
    if value.get("schema") != contract["partition_schema"]:
        raise ValueError("Delta v3 partition schema mismatch")
    expected_positions = 20_000 if profile == "delta20k" else 4_096
    if value.get("outer_split") != "train" or value.get("positions") != expected_positions:
        raise PermissionError("Delta v3 stage partition/profile contract mismatch")
    for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if value.get(key) is not False:
            raise PermissionError(f"Delta partition violates {key}")
    if profile == "delta20k" and value.get("counterfactual_labels_are_targets_only") is not True:
        raise PermissionError("Delta partition does not seal counterfactual labels")
    return value


def _reuse_split_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("b2_reuse_4096 requires --reuse-split-manifest")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != REUSE_SPLIT_SCHEMA:
        raise ValueError("Delta reuse split manifest schema mismatch")
    for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if value.get(key) is not False:
            raise PermissionError(f"Delta reuse split manifest violates {key}")
    split = value.get("inner_split")
    if not isinstance(split, Mapping) or split.get("request_disjoint") is not True:
        raise ValueError("Delta reuse split is not request-disjoint")
    training = split.get("training_requests")
    tuning = split.get("tuning_requests")
    if not isinstance(training, list) or not isinstance(tuning, list):
        raise ValueError("Delta reuse split lacks frozen request lists")
    if (
        len(training) != 224
        or len(tuning) != 32
        or set(training) & set(tuning)
        or int(split.get("training_rows", -1)) != 3_584
        or int(split.get("tuning_rows", -1)) != 512
    ):
        raise ValueError("Delta reuse split counts or disjointness changed")
    return value


def _b31_gate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != B31_REPORT_SCHEMA or value.get("conditions") != 56:
        raise ValueError("Delta v3 requires the complete B3.1 factorial report")
    for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if value.get(key) is not False:
            raise PermissionError(f"B3.1 report violates {key}")
    if value.get("training_started") is not False:
        raise PermissionError("B3.1 diagnostic unexpectedly trained")
    factorial = value.get("factorial")
    if not isinstance(factorial, Mapping):
        raise ValueError("B3.1 report has no factorial results")
    native = factorial.get("oracle_native__learned__quota_32_32")
    if not isinstance(native, Mapping):
        raise ValueError("B3.1 report lacks the native-route learned-posterior gate")
    mean_h2_h4 = sum(float(native[f"coverage_h{h}"]) for h in (2, 3, 4)) / 3.0
    if mean_h2_h4 < 0.985 or float(native["coverage_h4"]) < 0.97:
        raise PermissionError("B3.1 did not establish sufficient route information")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "native_learned_quota32_mean_h2_h4": mean_h2_h4,
        "native_learned_quota32_h4": float(native["coverage_h4"]),
        "bundle_sha256": value.get("bundle_sha256"),
    }


def _record_request_id(dataset: HarpRTTDataset, index: int) -> str:
    record = dataset.records[index]
    segment = dataset.segments[record.segment]
    return str(segment.sequences[record.sequence]["request_id"])


def _request_counts(dataset: HarpRTTDataset) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in dataset.records:
        segment = dataset.segments[record.segment]
        counts[str(segment.sequences[record.sequence]["request_id"])] += 1
    return counts


class RequestSubset(Dataset[Any]):
    """Stable request-filtered view over one immutable rich dataset."""

    def __init__(self, base: HarpRTTDataset, requests: set[str]) -> None:
        self.base = base
        self.requests = frozenset(requests)
        self.indices = tuple(
            index
            for index in range(len(base))
            if _record_request_id(base, index) in self.requests
        )
        observed = {_record_request_id(base, index) for index in self.indices}
        if observed != self.requests:
            raise ValueError("Delta request subset does not match its frozen request list")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.base[self.indices[index]]


def load_split(
    args: argparse.Namespace,
    split: str,
    *,
    selected_requests: set[str] | None,
) -> tuple[Dataset[Any], set[str]]:
    index = getattr(args, f"{split}_index")
    corpus = getattr(args, f"{split}_corpus")
    companion = getattr(args, f"{split}_companion")
    base = HarpRTTDataset(index, "train", corpus_root=corpus, max_tree_nodes=32)
    labels, manifest = load_node_counterfactual_companion(
        companion, split="train", training=True
    )
    if len(labels) != len(base):
        raise ValueError(f"Delta {split} companion/base row count differs")
    filtered: Dataset[Any]
    if selected_requests is None:
        filtered = base
        counts = _request_counts(base)
    else:
        filtered = RequestSubset(base, selected_requests)
        counts = Counter(
            _record_request_id(base, index)
            for index in filtered.indices  # type: ignore[attr-defined]
        )
    contract = DATA_PROFILES[args.data_profile]
    expected_rows = int(contract["rows"][split])
    expected_requests = int(contract["requests"][split])
    if len(filtered) != expected_rows or len(counts) != expected_requests:
        raise ValueError(f"Delta {split} split has invalid row/request counts")
    if set(counts.values()) != {16}:
        raise ValueError(f"Delta {split} requests do not each contain 16 positions")
    if manifest.get("sealed_test_opened") is not False:
        raise PermissionError(f"Delta {split} companion crossed sealed test")
    joined = NodeCounterfactualDatasetAdapter(filtered, labels, split="train", training=True)
    if len(joined) != len(filtered):
        raise ValueError(f"Delta {split} companion join is incomplete")
    return joined, set(counts)


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


def load_initializer(
    model: HARPDeltaTeacher,
    path: Path,
    stage: str,
    *,
    data_profile: str,
    partition_manifest_sha256: str,
) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Delta initializer schema mismatch")
    expected = PREDECESSOR[stage]
    if value.get("stage") != expected:
        raise ValueError(f"Delta {stage} requires a completed {expected} checkpoint")
    if value.get("config") != model.config.to_dict():
        raise ValueError("Delta initializer architecture differs from this run")
    if value.get("data_profile") != data_profile:
        raise ValueError("Delta initializer data profile differs from this run")
    if value.get("partition_manifest_sha256") != partition_manifest_sha256:
        raise ValueError("Delta initializer partition differs from this run")
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
    semantic_only: bool = False,
) -> tuple[Any, dict[str, Any], dict[str, Tensor], Tensor]:
    batch = move_to_device(host, device)
    prepared = prepare_model_batch(batch, runtime_static)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
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
        output = model(**delta.model_inputs, semantic_only=semantic_only)
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
    semantic_hits = [0.0, 0.0, 0.0, 0.0]
    semantic_slots = [0, 0, 0, 0]
    path_correct = path_rows = 0
    semantic_anchor: list[Tensor] = []
    semantic_branch: list[Tensor] = []
    semantic_labels: list[Tensor] = []
    semantic_valid: list[Tensor] = []
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
        root_ids = stable_topk(output.root_scores.float(), model.config.exact_k)
        root_hits = (
            labels[:, 0, ..., :, None] == root_ids[..., None, :]
        ).any(-1)
        root_valid = valid[:, 0]
        semantic_hits[0] += float(
            (root_hits.float() * root_valid[..., None]).sum()
        )
        semantic_slots[0] += int(root_valid.sum()) * model.config.exact_k

        node_ids = counterfactual["selected_ids"].long()
        node_valid = counterfactual["valid"].bool()
        node_depth = counterfactual["depth"].long()
        budget = counterfactual["budget_node_masks"][:, 2].bool()
        for horizon in range(1, 4):
            predicted = stable_topk(
                output.node_scores[:, horizon].float(), model.config.exact_k
            ).permute(0, 2, 1, 3)
            active_nodes = (
                node_valid
                & (budget & (node_depth == horizon + 1))[..., None]
            )
            hits = (node_ids[..., :, None] == predicted[..., None, :]).any(-1)
            semantic_hits[horizon] += float(
                (hits.float() * active_nodes[..., None]).sum()
            )
            semantic_slots[horizon] += (
                int(active_nodes.sum()) * model.config.exact_k
            )

        factual = targets["factual_branch_index"].long()
        path_valid = valid.any(-1)
        path_correct += int(
            ((output.factual_path_logits.argmax(-1) == factual) & path_valid).sum()
        )
        path_rows += int(path_valid.sum())
        semantic_anchor.append(anchor_logits.detach().cpu().to(torch.bfloat16))
        semantic_branch.append(
            output.branch_marginals.detach().cpu().to(torch.bfloat16)
        )
        semantic_labels.append(labels.detach().cpu().to(torch.int16))
        semantic_valid.append(valid.detach().cpu())
    slots = rows * model.config.exact_k
    anchor_all = torch.cat(semantic_anchor).float()
    branch_all = torch.cat(semantic_branch).float()
    labels_all = torch.cat(semantic_labels).long()
    valid_all = torch.cat(semantic_valid).bool()
    semantic_candidates = quota_candidate_union(
        anchor_all, branch_all, anchor_quota=32,
        width=model.config.candidate_width,
    ).expert_ids
    semantic_membership = (
        labels_all[..., :, None] == semantic_candidates[..., None, :]
    ).any(-1)
    semantic_coverage_h: list[float] = []
    for horizon in range(4):
        active_h = valid_all[:, horizon]
        denominator = int(active_h.sum()) * model.config.exact_k
        semantic_coverage_h.append(
            float(
                (
                    semantic_membership[:, horizon].float()
                    * active_h[..., None]
                ).sum()
            )
            / max(1, denominator)
        )
    result: dict[str, float | int] = {
        "loss": totals["loss"] / max(1, rows),
        "recall_at_8": totals["recall"] / max(1, slots),
        "candidate_coverage_at_64": totals["coverage"] / max(1, slots),
        "semantic_candidate_coverage_at_64_quota32": sum(
            semantic_coverage_h
        ) / 4.0,
        "semantic_candidate_coverage_h2_h4_quota32": sum(
            semantic_coverage_h[1:]
        ) / 3.0,
        "counterfactual_route_recall_at_8": sum(semantic_hits[1:])
        / max(1, sum(semantic_slots[1:])),
        "semantic_selected_set_recall_at_8": sum(semantic_hits)
        / max(1, sum(semantic_slots)),
        "factual_path_accuracy": path_correct / max(1, path_rows),
        "valid_layer_horizons": rows,
        "outside_candidate_slots": outside,
    }
    for horizon, value in enumerate(semantic_coverage_h, start=1):
        result[f"semantic_candidate_coverage_h{horizon}_quota32"] = value
    for horizon in range(4):
        result[f"semantic_route_recall_h{horizon + 1}"] = (
            semantic_hits[horizon] / max(1, semantic_slots[horizon])
        )
    return result


def selection_value(stage: DeltaStage, metrics: Mapping[str, float | int]) -> float:
    if stage == "semantic":
        return float(metrics["semantic_candidate_coverage_at_64_quota32"])
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
    partition = _manifest(args.partition_manifest, args.data_profile)
    b31 = _b31_gate(args.b31_report)
    reuse = (
        _reuse_split_manifest(args.reuse_split_manifest)
        if args.data_profile == "b2_reuse_4096"
        else None
    )
    if args.data_profile == "delta20k" and args.reuse_split_manifest is not None:
        raise ValueError("delta20k must not receive a B2 reuse split manifest")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    seed_everything(args.seed, deterministic=args.deterministic)
    selected: dict[str, set[str] | None] = {
        "train": None,
        "tune": None,
        "development": None,
    }
    if reuse is not None:
        inner = reuse["inner_split"]
        selected["train"] = set(inner["training_requests"])
        selected["tune"] = set(inner["tuning_requests"])
    datasets: dict[str, Dataset[Any]] = {}
    request_groups: dict[str, set[str]] = {}
    for name in ("train", "tune", "development"):
        datasets[name], request_groups[name] = load_split(
            args, name, selected_requests=selected[name]
        )
    if (
        request_groups["train"] & request_groups["tune"]
        or request_groups["train"] & request_groups["development"]
        or request_groups["tune"] & request_groups["development"]
    ):
        raise PermissionError("Delta train/tune/development requests overlap")

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
        initializer = load_initializer(
            model,
            args.initialize_from,
            stage,
            data_profile=args.data_profile,
            partition_manifest_sha256=sha256_file(args.partition_manifest),
        )
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
        "data_profile": args.data_profile,
        "diagnostic_reuse_only": bool(
            DATA_PROFILES[args.data_profile]["diagnostic_reuse"]
        ),
        "config": config.to_dict(),
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "partition_schema": partition["schema"],
        "reuse_split_manifest_sha256": (
            None
            if args.reuse_split_manifest is None
            else sha256_file(args.reuse_split_manifest)
        ),
        "b31_gate": b31,
        "split_rows": {
            name: int(DATA_PROFILES[args.data_profile]["rows"][name])
            for name in ("train", "tune", "development")
        },
        "split_requests": {
            name: len(request_groups[name])
            for name in ("train", "tune", "development")
        },
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
                semantic_only=stage == "semantic",
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
        "data_profile": args.data_profile,
        "config": config.to_dict(),
        "best_epoch": best_epoch,
        "best_tune_value": best_value,
        "development": development,
        "model_state_dict": best_state,
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "reuse_split_manifest_sha256": (
            None
            if args.reuse_split_manifest is None
            else sha256_file(args.reuse_split_manifest)
        ),
        "b31_report_sha256": sha256_file(args.b31_report),
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
