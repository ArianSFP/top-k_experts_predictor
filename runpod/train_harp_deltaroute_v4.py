#!/usr/bin/env python3
"""Train one immutable HARP-DeltaRoute v4 stage on outer-train data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.anchor import LegacyHARPAnchorBridge  # noqa: E402
from harp_rtt.b31 import anchor_spine_prefix_matches, quota_candidate_union  # noqa: E402
from harp_rtt.delta import HARPDeltaConfig, HARPDeltaTeacher  # noqa: E402
from harp_rtt.deltaroute_training import factual_alignment_loss  # noqa: E402
from harp_rtt.factual_branch_attention import (  # noqa: E402
    ExpertConditionedBranchAttention,
    FactualAlignmentOutput,
    SummaryFactualAligner,
)
from harp_rtt.deltaroute_metrics import paired_h2_h4_request_bootstrap  # noqa: E402
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.train import runtime_static_artifacts  # noqa: E402
from harp_rtt.training import seed_everything, sha256_file  # noqa: E402
from runpod.evaluate_harp_deltaroute_v4_ceiling import REPORT_SCHEMA as CEILING_SCHEMA  # noqa: E402
from runpod.train_harp_delta_v3 import (  # noqa: E402
    SCHEMA as PARENT_SCHEMA,
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
    forward_batch,
    load_split,
    loader,
)


SCHEMA = "harp_deltaroute_v4_stage_training_v1"
RESULT_SCHEMA = "harp_deltaroute_v4_stage_result_v1"
STAGES = ("align_m0", "align_m1")
EFFECTIVE_BATCH = 32
MICROBATCH_CHOICES = (32, 16, 8, 4, 2, 1)
PARENT_BUDGET = "16"
PARENT_BUDGET_INDEX = 2


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    for split in ("train", "tune", "development"):
        parser.add_argument(f"--{split}-index", type=Path, required=True)
        parser.add_argument(f"--{split}-corpus", type=Path, required=True)
        parser.add_argument(f"--{split}-companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--ceiling-result", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--microbatch-size", type=int, choices=(0, *MICROBATCH_CHOICES), default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def validate_ceiling(path: Path, parent_sha256: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != CEILING_SCHEMA:
        raise ValueError("DeltaRoute ceiling result schema mismatch")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("DeltaRoute ceiling result lacks provenance")
    if provenance.get("parent_checkpoint_sha256") != parent_sha256:
        raise ValueError("DeltaRoute ceiling and parent checkpoint differ")
    gate = value.get("native_factual_h2_h4_gate")
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise PermissionError("native factual Top8 ceiling did not pass")
    if value.get("training_started") is not False or value.get("optimizer_constructed") is not False:
        raise PermissionError("ceiling result unexpectedly constructed an optimizer")
    for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if value.get(key) is not False:
            raise PermissionError(f"ceiling result violates {key}")
    return {
        "sha256": sha256_file(path),
        "native_factual_h2_h4": float(gate["value"]),
        "threshold": float(gate["threshold"]),
    }


def load_parent(
    path: Path,
    static: Any,
    *,
    partition_sha256: str,
) -> tuple[HARPDeltaTeacher, dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != PARENT_SCHEMA:
        raise ValueError("DeltaRoute parent schema mismatch")
    if value.get("stage") != "semantic" or value.get("counterfactual_budget") != PARENT_BUDGET:
        raise ValueError("DeltaRoute requires the selected budget-16 semantic parent")
    if value.get("partition_manifest_sha256") != partition_sha256:
        raise ValueError("DeltaRoute parent and data partition differ")
    config = HARPDeltaConfig(**value["config"])
    model = HARPDeltaTeacher(
        config, static.geometry.expert_keys, static.geometry.centered_bias,
        raw_width=static.geometry.hidden_width,
        target_control_width=static.geometry.maximum_rank,
        metadata_width=8,
    )
    model.load_state_dict(value["model_state_dict"], strict=True)
    model.requires_grad_(False).eval()
    return model, value


def build_aligner(stage: str, model: HARPDeltaTeacher) -> nn.Module:
    config = model.config
    common = dict(
        horizons=config.horizons, exact_k=config.exact_k,
        candidate_width=config.candidate_width, anchor_quota=32,
    )
    if stage == "align_m0":
        return SummaryFactualAligner(
            layers=config.layers, experts=config.experts,
            hidden_width=64, **common,
        )
    if stage == "align_m1":
        return ExpertConditionedBranchAttention(
            model.core.set_head.expert_keys,
            tree_width=config.tree_width, hidden_width=64, **common,
        )
    raise ValueError(f"unknown aligner stage {stage!r}")


def _aligner_forward(
    stage: str,
    aligner: nn.Module,
    semantic: Any,
    anchor_scores: Tensor,
    anchor_marginals: Tensor,
    node_marginals: Tensor,
    parent_marginals: Tensor,
    host: Mapping[str, Any],
    device: torch.device,
) -> Any:
    posterior = semantic.factual_path_posterior.detach()
    node_mask = posterior[..., :-1] > 0
    tree = host["inputs"]["tree"]
    if stage == "align_m0":
        mtp = tree["path_log_probabilities"].to(device).float().exp()[:, None]
        mtp = mtp.expand_as(node_mask) * node_mask.float()
        return aligner(
            anchor_scores=anchor_scores,
            anchor_marginals=anchor_marginals,
            node_marginals=node_marginals,
            posterior=posterior,
            mtp_probabilities=mtp,
            node_mask=node_mask,
            first_divergence_depth=tree["first_divergence_depths"].to(device).long(),
            parent_marginals=parent_marginals,
        )
    return aligner(
        anchor_scores=anchor_scores,
        anchor_marginals=anchor_marginals,
        node_marginals=node_marginals,
        posterior=posterior,
        node_mask=node_mask,
        tree_states=semantic.tree_states.detach(),
        context_states=semantic.context_states.detach(),
        parent_marginals=parent_marginals,
    )


def _forward(
    *,
    stage: str,
    aligner: nn.Module,
    parent: HARPDeltaTeacher,
    anchor: LegacyHARPAnchorBridge,
    host: Mapping[str, Any],
    runtime_static: Any,
    token_embedding: Tensor,
    input_basis: Tensor,
    rank_mask: Tensor,
    device: torch.device,
) -> tuple[Any, Any, Tensor, Tensor, Tensor]:
    with torch.no_grad():
        semantic, targets, _, anchor_scores = forward_batch(
            model=parent, anchor=anchor, host=host,
            runtime_static=runtime_static, token_embedding=token_embedding,
            input_basis=input_basis, rank_mask=rank_mask, device=device,
            semantic_only=True,
            counterfactual_budget_index=PARENT_BUDGET_INDEX,
        )
        anchor_marginals, _, node_marginals, parent_marginals = parent.core.semantic_marginals(
            semantic, anchor_scores
        )
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        aligned = _aligner_forward(
            stage, aligner, semantic, anchor_scores, anchor_marginals,
            node_marginals, parent_marginals, host, device,
        )
        # M0/M1 are H2--H4 aligners.  The v3 parent replaces its H1 branch
        # mixture with the dedicated exact-root prediction, while the generic
        # aligner inputs contain only node marginals and therefore cannot
        # reconstruct that overwrite.  Preserve H1 explicitly so the deployed
        # dense evidence and C64 contract remain parent-identical at epoch zero.
        marginals = aligned.marginals.clone()
        marginals[:, 0] = parent_marginals[:, 0]
        aligned = FactualAlignmentOutput(
            scores=aligned.scores,
            marginals=marginals,
            correction=aligned.correction,
            expert_branch_weights=aligned.expert_branch_weights,
            coherence_kl=aligned.coherence_kl,
            candidate_ids=quota_candidate_union(
                anchor_scores, marginals, anchor_quota=32,
                width=parent.config.candidate_width,
            ).expert_ids,
        )
    return aligned, targets, anchor_scores, parent_marginals, semantic


def _request_ids(host: Mapping[str, Any]) -> list[str]:
    metadata = host.get("metadata")
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("request_id"), list):
        raise ValueError("DeltaRoute batch lacks request IDs")
    return [str(value) for value in metadata["request_id"]]


@torch.no_grad()
def evaluate(
    *,
    stage: str,
    aligner: nn.Module,
    parent: HARPDeltaTeacher,
    anchor: LegacyHARPAnchorBridge,
    dataset: Dataset[Any],
    runtime_static: Any,
    token_embedding: Tensor,
    input_basis: Tensor,
    rank_mask: Tensor,
    device: torch.device,
    microbatch: int,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    aligner.eval(); parent.eval(); anchor.eval()
    candidate_cells: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    parent_cells: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    mismatch_candidate: dict[str, list[Tensor]] = defaultdict(list)
    mismatch_parent: dict[str, list[Tensor]] = defaultdict(list)
    loss_total = rows = 0.0
    for host in loader(
        dataset, batch=microbatch, shuffle=False, seed=0,
        workers=workers, device=device,
    ):
        aligned, targets, anchor_scores, parent_marginals, _ = _forward(
            stage=stage, aligner=aligner, parent=parent, anchor=anchor,
            host=host, runtime_static=runtime_static,
            token_embedding=token_embedding, input_basis=input_basis,
            rank_mask=rank_mask, device=device,
        )
        loss = factual_alignment_loss(
            aligned, anchor_scores, targets,
            pair_weight=0.1,
            coherence_weight=0.01 if stage == "align_m1" else 0.0,
        )
        target = targets["future_selected_ids"].long()
        valid = targets["future_available"].bool()
        if valid.ndim == 2:
            valid = valid[..., None].expand(target.shape[:-1])
        parent_ids = quota_candidate_union(
            anchor_scores, parent_marginals, anchor_quota=32,
            width=parent.config.candidate_width,
        ).expert_ids
        candidate_hit = (
            target[..., None] == aligned.candidate_ids[..., None, :]
        ).any(-1).float().mean(-1)
        parent_hit = (
            target[..., None] == parent_ids[..., None, :]
        ).any(-1).float().mean(-1)
        requests = _request_ids(host)
        mismatch = ~anchor_spine_prefix_matches(
            host["anchor_inputs"]["mtp_spine"]["exact_prefix_hashes"],
            host["targets"]["future_prefix_hashes"],
        )
        for batch_row, request in enumerate(requests):
            for horizon in range(4):
                active = valid[batch_row, horizon]
                candidate_cells[(request, horizon)].append(candidate_hit[batch_row, horizon][active].cpu())
                parent_cells[(request, horizon)].append(parent_hit[batch_row, horizon][active].cpu())
            if bool(mismatch[batch_row, 3]):
                active = valid[batch_row, 3]
                mismatch_candidate[request].append(candidate_hit[batch_row, 3][active].cpu())
                mismatch_parent[request].append(parent_hit[batch_row, 3][active].cpu())
        active_rows = int(valid[:, 1:].sum())
        loss_total += float(loss.total) * active_rows
        rows += active_rows

    def metric_rows(cells: Mapping[tuple[str, int], list[Tensor]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for (request, horizon), values in sorted(cells.items()):
            result.append({
                "request_id": request,
                "horizon": horizon + 1,
                "slot_recall_at_8": float(torch.cat(values).mean()),
            })
        return result

    candidate_rows = metric_rows(candidate_cells)
    parent_rows = metric_rows(parent_cells)
    horizons = {
        horizon: sum(
            row["slot_recall_at_8"] for row in candidate_rows
            if row["horizon"] == horizon
        ) / sum(row["horizon"] == horizon for row in candidate_rows)
        for horizon in range(1, 5)
    }
    parent_horizons = {
        horizon: sum(
            row["slot_recall_at_8"] for row in parent_rows
            if row["horizon"] == horizon
        ) / sum(row["horizon"] == horizon for row in parent_rows)
        for horizon in range(1, 5)
    }

    def mismatch_macro(values: Mapping[str, list[Tensor]]) -> float | None:
        if not values:
            return None
        return sum(float(torch.cat(rows).mean()) for rows in values.values()) / len(values)

    metrics: dict[str, Any] = {
        "loss": loss_total / max(1.0, rows),
        "candidate_coverage_h2_h4": sum(horizons[h] for h in (2, 3, 4)) / 3.0,
        "candidate_coverage_h4": horizons[4],
        "candidate_coverage_h4_mismatch": mismatch_macro(mismatch_candidate),
        "parent_candidate_coverage_h2_h4": sum(parent_horizons[h] for h in (2, 3, 4)) / 3.0,
        "parent_candidate_coverage_h4": parent_horizons[4],
        "parent_candidate_coverage_h4_mismatch": mismatch_macro(mismatch_parent),
        **{f"candidate_coverage_h{horizon}": value for horizon, value in horizons.items()},
    }
    return metrics, candidate_rows, parent_rows


def _autotune(
    args: argparse.Namespace,
    *,
    aligner: nn.Module,
    parent: HARPDeltaTeacher,
    anchor: LegacyHARPAnchorBridge,
    dataset: Dataset[Any],
    runtime_static: Any,
    token_embedding: Tensor,
    input_basis: Tensor,
    rank_mask: Tensor,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    choices = (args.microbatch_size,) if args.microbatch_size else MICROBATCH_CHOICES
    trace: list[dict[str, Any]] = []
    for size in choices:
        if EFFECTIVE_BATCH % size:
            continue
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            host = next(iter(loader(
                dataset, batch=size, shuffle=False, seed=0,
                workers=args.num_workers, device=device,
            )))
            aligner.zero_grad(set_to_none=True)
            aligned, targets, anchor_scores, _, _ = _forward(
                stage=args.stage, aligner=aligner, parent=parent, anchor=anchor,
                host=host, runtime_static=runtime_static,
                token_embedding=token_embedding, input_basis=input_basis,
                rank_mask=rank_mask, device=device,
            )
            loss = factual_alignment_loss(
                aligned, anchor_scores, targets,
                coherence_weight=0.01 if args.stage == "align_m1" else 0.0,
            )
            loss.total.backward(); aligner.zero_grad(set_to_none=True)
            peak = (
                torch.cuda.max_memory_reserved(device) / 2**30
                if device.type == "cuda" else 0.0
            )
            accepted = device.type != "cuda" or peak <= 21.0
            trace.append({"microbatch": size, "peak_reserved_gib": peak, "accepted": accepted})
            if accepted:
                return size, trace
        except torch.OutOfMemoryError:
            trace.append({"microbatch": size, "oom": True, "accepted": False})
            aligner.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("no v4 aligner microbatch fits the 21-GiB 3090 budget")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def _write_checksums(output: Path) -> None:
    names = sorted(
        path.name for path in output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for name in names:
            handle.write(f"{sha256_file(output / name)}  {name}\n")
        handle.flush(); os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    args.data_profile = "b2_reuse_4096"
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite DeltaRoute run {args.output}")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    partition = validate_partition(args.partition_manifest, "b2_reuse_4096")
    reuse = validate_reuse_split(args.reuse_split_manifest)
    partition_sha = sha256_file(args.partition_manifest)
    parent_sha = sha256_file(args.parent_checkpoint)
    ceiling = validate_ceiling(args.ceiling_result, parent_sha)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    seed_everything(args.seed, deterministic=args.deterministic)
    inner = reuse["inner_split"]
    selected = {
        "train": set(inner["training_requests"]),
        "tune": set(inner["tuning_requests"]),
        "development": None,
    }
    datasets: dict[str, Dataset[Any]] = {}
    request_groups: dict[str, set[str]] = {}
    for split in ("train", "tune", "development"):
        datasets[split], request_groups[split] = load_split(
            args, split, selected_requests=selected[split]
        )
    if any(
        request_groups[a] & request_groups[b]
        for a, b in (("train", "tune"), ("train", "development"), ("tune", "development"))
    ):
        raise PermissionError("DeltaRoute train/tune/development requests overlap")
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    if static.token_embedding is None:
        raise RuntimeError("DeltaRoute requires frozen token embeddings")
    parent, parent_record = load_parent(
        args.parent_checkpoint, static, partition_sha256=partition_sha
    )
    parent = parent.to(device)
    aligner = build_aligner(args.stage, parent).to(device)
    anchor, anchor_provenance = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint, args.target_preprocessing, args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    anchor = anchor.to(device).requires_grad_(False).eval()
    runtime_static = runtime_static_artifacts(static, device)
    token_embedding = static.token_embedding.to(device)
    input_basis = static.geometry.input_basis.to(device)
    rank_mask = static.geometry.rank_mask.to(device)
    trainable = [parameter for parameter in aligner.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in aligner.named_parameters() if parameter.requires_grad]
    if not trainable or sum(parameter.numel() for parameter in trainable) > 25_000_000:
        raise RuntimeError("DeltaRoute aligner parameter contract is invalid")

    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "seed": args.seed,
        "source_commit": args.source_commit,
        "config": parent.config.to_dict(),
        "partition_schema": partition["schema"],
        "partition_manifest_sha256": partition_sha,
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "ceiling_gate": ceiling,
        "parent_checkpoint_sha256": parent_sha,
        "parent_development": parent_record.get("development"),
        "runtime_tree_nodes": 32,
        "counterfactual_supervision_budget": 16,
        "trainable_names": trainable_names,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "anchor_provenance": anchor_provenance,
        "effective_batch": EFFECTIVE_BATCH,
        "optimizer_constructed": False,
        "counterfactual_model_input": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)
    microbatch, autotune = _autotune(
        args, aligner=aligner, parent=parent, anchor=anchor,
        dataset=datasets["train"], runtime_static=runtime_static,
        token_embedding=token_embedding, input_basis=input_basis,
        rank_mask=rank_mask, device=device,
    )
    write_json_exclusive(args.output / "MEMORY_AUTOTUNE.json", {
        "selected_microbatch": microbatch,
        "effective_batch": EFFECTIVE_BATCH,
        "trace": autotune,
        "optimizer_constructed": False,
    })

    host = next(iter(loader(
        datasets["train"], batch=1, shuffle=False, seed=0,
        workers=args.num_workers, device=device,
    )))
    aligner.eval()
    with torch.no_grad():
        aligned, _, anchor_scores, parent_marginals, _ = _forward(
            stage=args.stage, aligner=aligner, parent=parent, anchor=anchor,
            host=host, runtime_static=runtime_static,
            token_embedding=token_embedding, input_basis=input_basis,
            rank_mask=rank_mask, device=device,
        )
        parent_ids = quota_candidate_union(
            anchor_scores, parent_marginals, anchor_quota=32,
            width=parent.config.candidate_width,
        ).expert_ids
        marginal_delta = (aligned.marginals.float() - parent_marginals.float()).abs()
        epoch_zero = {
            "dense_parent_marginals_bitwise_equal": bool(torch.equal(aligned.marginals, parent_marginals)),
            "dense_parent_marginals_max_abs_error": float(marginal_delta.max()),
            "dense_parent_marginal_mismatch_elements": int((marginal_delta > 0).sum()),
            "dense_parent_marginals_equal_by_horizon": [
                bool(torch.equal(aligned.marginals[:, horizon], parent_marginals[:, horizon]))
                for horizon in range(parent.config.horizons)
            ],
            "parent_c64_bitwise_equal": bool(torch.equal(aligned.candidate_ids, parent_ids)),
            "parent_c64_mismatch_elements": int((aligned.candidate_ids != parent_ids).sum()),
            "runtime_tree_nodes": 32,
            "supervision_budget": 16,
            "optimizer_constructed": False,
        }
    write_json_exclusive(args.output / "EPOCH_ZERO_AUDIT.json", epoch_zero)
    if not epoch_zero["dense_parent_marginals_bitwise_equal"] or not epoch_zero["parent_c64_bitwise_equal"]:
        raise RuntimeError("DeltaRoute epoch-zero parent reproduction failed")
    if args.preflight_only:
        write_json_exclusive(args.output / "PREFLIGHT_RESULT.json", {
            **epoch_zero, "stage": args.stage, "training_started": False,
        })
        _write_checksums(args.output)
        return

    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    write_json_exclusive(args.output / "OPTIMIZER_START.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    })
    accumulation = EFFECTIVE_BATCH // microbatch
    best_value = -math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        aligner.train(); parent.eval(); anchor.eval(); optimizer.zero_grad(set_to_none=True)
        total = batches = 0.0
        train_loader = loader(
            datasets["train"], batch=microbatch, shuffle=True,
            seed=args.seed + epoch, workers=args.num_workers, device=device,
        )
        for step, host in enumerate(train_loader, start=1):
            aligned, targets, anchor_scores, _, _ = _forward(
                stage=args.stage, aligner=aligner, parent=parent, anchor=anchor,
                host=host, runtime_static=runtime_static,
                token_embedding=token_embedding, input_basis=input_basis,
                rank_mask=rank_mask, device=device,
            )
            loss = factual_alignment_loss(
                aligned, anchor_scores, targets, pair_weight=0.1,
                coherence_weight=0.01 if args.stage == "align_m1" else 0.0,
            )
            (loss.total / accumulation).backward()
            if step % accumulation == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.total.detach()); batches += 1
        tune, _, _ = evaluate(
            stage=args.stage, aligner=aligner, parent=parent, anchor=anchor,
            dataset=datasets["tune"], runtime_static=runtime_static,
            token_embedding=token_embedding, input_basis=input_basis,
            rank_mask=rank_mask, device=device, microbatch=microbatch,
            workers=args.num_workers,
        )
        append_jsonl(args.output / "metrics.jsonl", {
            "epoch": epoch,
            "train_loss": total / max(1.0, batches),
            "tune": tune,
        })
        value = float(tune["candidate_coverage_h2_h4"])
        if value > best_value:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in aligner.state_dict().items()
            }
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("DeltaRoute aligner produced no selectable checkpoint")
    aligner.load_state_dict(best_state, strict=True)
    development, prediction_rows, parent_rows = evaluate(
        stage=args.stage, aligner=aligner, parent=parent, anchor=anchor,
        dataset=datasets["development"], runtime_static=runtime_static,
        token_embedding=token_embedding, input_basis=input_basis,
        rank_mask=rank_mask, device=device, microbatch=microbatch,
        workers=args.num_workers,
    )
    bootstrap = paired_h2_h4_request_bootstrap(
        prediction_rows, parent_rows, replicates=1_000, seed=42
    )
    _write_rows(args.output / "development_request_predictions.jsonl", prediction_rows)
    _write_rows(args.output / "development_parent_predictions.jsonl", parent_rows)
    checkpoint = {
        "schema": SCHEMA,
        "stage": args.stage,
        "seed": args.seed,
        "source_commit": args.source_commit,
        "best_epoch": best_epoch,
        "best_tune_h2_h4_c64": best_value,
        "development": development,
        "paired_request_bootstrap": bootstrap,
        "aligner_state_dict": best_state,
        "parent_checkpoint_sha256": parent_sha,
        "partition_manifest_sha256": partition_sha,
        "ceiling_result_sha256": sha256_file(args.ceiling_result),
        "run_manifest_sha256": sha256_file(args.output / "run_manifest.json"),
        "runtime_tree_nodes": 32,
        "counterfactual_supervision_budget": 16,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    checkpoint_path = args.output / "best_deltaroute_stage.pt"
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle); handle.flush(); os.fsync(handle.fileno())
    h4 = float(development["candidate_coverage_h4"])
    h4_parent = float(development["parent_candidate_coverage_h4"])
    mismatch_value = development["candidate_coverage_h4_mismatch"]
    mismatch_parent = development["parent_candidate_coverage_h4_mismatch"]
    no_h4_regression = h4 >= h4_parent
    no_mismatch_regression = (
        mismatch_value is not None and mismatch_parent is not None
        and float(mismatch_value) >= float(mismatch_parent)
    )
    result = {
        "schema": RESULT_SCHEMA,
        "stage": args.stage,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_tune_h2_h4_c64": best_value,
        "development": development,
        "paired_request_bootstrap": bootstrap,
        "single_seed_continuation": {
            "positive_h2_h4_point_gain": (
                float(development["candidate_coverage_h2_h4"])
                > float(development["parent_candidate_coverage_h2_h4"])
            ),
            "paired_lower_bound_above_zero": bool(bootstrap["lower_bound_positive"]),
            "no_h4_regression": no_h4_regression,
            "no_h4_mismatch_regression": no_mismatch_regression,
        },
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_started": True,
        "optimizer_constructed": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    _write_checksums(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
