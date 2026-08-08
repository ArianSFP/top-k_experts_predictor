"""Staged production training driver for HARP-RTT-90.

One invocation trains exactly one of Phase 2--5.  The driver only constructs
the frozen ``train`` and ``validation`` rich-corpus splits; there is
intentionally no command-line switch that can authorize the sealed outer test
split.  A later phase is initialized from a verified earlier checkpoint,
while ``--resume`` restores optimizer/audit state for the same phase into a
new, non-overwriting continuation lineage.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from .anchor import LegacyHARPAnchorBridge
from .dataset import HarpRTTDataset, collate_harp_rtt
from .losses import GradientAudit, HARPRTTLossConfig, HARPRTTObjective
from .metrics import (
    candidate_coverage_gate,
    evaluate_harp_rtt,
    model_selection_tuple,
)
from .model import HARPRTTConfig, HARPRTTTeacher
from .schema import HARPRTTDimensions
from .static_artifacts import StaticTargetArtifacts, load_static_target_artifacts
from .training import (
    AuxiliaryGradientController,
    DEFAULT_PHASES,
    OptimizerStepState,
    PhaseSpec,
    append_jsonl,
    autocast_context,
    configure_training_phase,
    cosine_warmup_multiplier,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    seed_everything,
    sha256_file,
)


DRIVER_SCHEMA = "harp_rtt_training_driver_v1"
EFFECTIVE_BATCH_SIZE = 32
MICROBATCH_CANDIDATES = (8, 4, 2, 1)
MAX_PROCESS_PEAK_BYTES = 20 * 1024**3
GRADIENT_AUDIT_INTERVAL = 50
GRADIENT_AUDIT_PASSES = 10
RANKER_GATE_RECALL = 0.85


def _sha256_argument(value: str) -> str:
    normalized = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise argparse.ArgumentTypeError("expected a 64-character hexadecimal SHA-256")
    return normalized


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--router-numerics-audit", type=Path, required=True)
    parser.add_argument(
        "--router-numerics-audit-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("phase2", "phase3", "phase4", "phase5"), required=True
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--initialize-from",
        type=Path,
        help="verified prior-phase checkpoint manifest (model state only)",
    )
    initialization.add_argument(
        "--resume",
        type=Path,
        help=(
            "same-phase checkpoint manifest; OUTPUT_DIR must be a new "
            "continuation lineage"
        ),
    )
    parser.add_argument("--epochs", type=_positive_int)
    parser.add_argument("--new-learning-rate", type=float)
    parser.add_argument("--legacy-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-fraction", type=float)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=_nonnegative_int, default=4)
    parser.add_argument("--max-train-requests", type=_positive_int)
    parser.add_argument("--max-validation-requests", type=_positive_int)
    parser.add_argument("--require-promotion-gate", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser


def resolve_phase_spec(args: argparse.Namespace) -> PhaseSpec:
    defaults = {spec.name: spec for spec in DEFAULT_PHASES}
    base = defaults[str(args.phase)]
    spec = replace(
        base,
        epochs=int(args.epochs) if args.epochs is not None else base.epochs,
        new_learning_rate=(
            float(args.new_learning_rate)
            if args.new_learning_rate is not None
            else base.new_learning_rate
        ),
        legacy_learning_rate=(
            float(args.legacy_learning_rate)
            if args.legacy_learning_rate is not None
            else base.legacy_learning_rate
        ),
        weight_decay=(
            float(args.weight_decay)
            if args.weight_decay is not None
            else base.weight_decay
        ),
        warmup_fraction=(
            float(args.warmup_fraction)
            if args.warmup_fraction is not None
            else base.warmup_fraction
        ),
    )
    spec.validate()
    if not math.isfinite(float(args.gradient_clip)) or args.gradient_clip <= 0:
        raise ValueError("gradient clipping threshold must be finite and positive")
    if spec.name == "phase2" and args.initialize_from is not None:
        raise ValueError("Phase 2 is the function-preserving initialization and takes no parent")
    if spec.name != "phase2" and args.initialize_from is None and args.resume is None:
        raise ValueError(f"{spec.name} requires --initialize-from or --resume")
    return spec


def _expected_parent_phase(phase_name: str) -> str | None:
    predecessors = {"phase3": "phase2", "phase4": "phase3", "phase5": "phase4"}
    if phase_name == "phase2":
        return None
    try:
        return predecessors[phase_name]
    except KeyError as exc:
        raise ValueError(f"unknown staged phase {phase_name!r}") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _request_digest(request_ids: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json(list(request_ids)).encode("utf-8")).hexdigest()


def _epoch_seed(seed: int, phase_name: str, epoch: int) -> int:
    """Derive a restart-stable RNG seed for one completed-epoch boundary."""

    if seed < 0 or epoch < 0:
        raise ValueError("base seed and epoch must be non-negative")
    digest = hashlib.sha256(
        f"harp-rtt-epoch\0{seed}\0{phase_name}\0{epoch}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _dataset_request_groups(dataset: HarpRTTDataset) -> dict[str, list[int]]:
    if dataset.split == "test":
        raise PermissionError("the staged driver never accepts the sealed test split")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        segment = dataset.segments[record.segment]
        request_id = str(segment.sequences[record.sequence]["request_id"])
        groups[request_id].append(index)
    if not groups:
        raise ValueError(f"{dataset.split} dataset contains no usable requests")
    return dict(groups)


def request_complete_subset(
    dataset: HarpRTTDataset,
    maximum_requests: int | None,
    *,
    seed: int,
) -> tuple[Dataset[dict[str, Any]], dict[str, Any]]:
    """Select whole requests by a stable seeded hash, never individual rows."""

    groups = _dataset_request_groups(dataset)
    request_ids = sorted(groups)
    if maximum_requests is not None:
        if maximum_requests < 1:
            raise ValueError("maximum_requests must be positive")
        ranked = sorted(
            request_ids,
            key=lambda request_id: (
                hashlib.sha256(
                    f"harp-rtt\0{seed}\0{dataset.split}\0{request_id}".encode("utf-8")
                ).digest(),
                request_id,
            ),
        )
        selected = sorted(ranked[: min(maximum_requests, len(ranked))])
    else:
        selected = request_ids
    indices = sorted(index for request_id in selected for index in groups[request_id])
    if not indices:
        raise ValueError("request-complete subset contains no token rows")
    subset: Dataset[dict[str, Any]] = (
        dataset if len(selected) == len(request_ids) else Subset(dataset, indices)
    )
    manifest = {
        "split": dataset.split,
        "available_requests": len(request_ids),
        "selected_requests": len(selected),
        "selected_request_ids": selected,
        "selected_request_ids_sha256": _request_digest(selected),
        "available_rows": len(dataset),
        "selected_rows": len(indices),
        "selection": "seeded_sha256_request_complete",
        "seed": int(seed),
    }
    return subset, manifest


def production_config(
    static: StaticTargetArtifacts,
    anchor: LegacyHARPAnchorBridge,
) -> HARPRTTConfig:
    """Derive the production input geometry from verified immutable tensors."""

    geometry = static.geometry
    geometry.validate()
    if static.token_embedding is None:
        raise ValueError("production training requires the frozen token embedding")
    hidden_width = int(geometry.hidden_width)
    if tuple(static.token_embedding.shape[1:]) != (hidden_width,):
        raise ValueError("token embedding and router hidden widths disagree")
    if static.final_rmsnorm_weight.shape != (hidden_width,):
        raise ValueError("final RMSNorm and router hidden widths disagree")
    contract = static.manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("static artifact manifest contains no target contract")
    exact_k = int(contract.get("experts_per_token", -1))
    if exact_k != 8:
        raise ValueError(f"HARP-RTT-90 requires native top-8 routing, found {exact_k}")
    legacy = anchor.config
    if (legacy.layers, legacy.experts) != (geometry.layers, geometry.experts):
        raise ValueError("legacy HARP and frozen target router geometry disagree")
    if legacy.horizons != 8 or legacy.route_history != 8:
        raise ValueError("production anchor must expose eight horizons and eight history tokens")
    if geometry.experts < 64:
        raise ValueError("production C64 reranking requires at least 64 experts")
    config = HARPRTTConfig(
        experts=geometry.experts,
        layers=geometry.layers,
        anchor_horizons=legacy.horizons,
        active_horizons=4,
        route_history=legacy.route_history,
        exact_k=exact_k,
        candidate_width=64,
        max_tree_nodes=32,
        max_tree_depth=4,
        max_tree_branches=64,
        router_rank=geometry.maximum_rank,
        target_control_width=geometry.maximum_rank,
        target_content_width=hidden_width,
        target_post_attention_width=hidden_width,
        target_post_moe_width=hidden_width,
        target_routed_width=hidden_width,
        target_shared_width=hidden_width,
        exact_token_width=hidden_width,
        final_hidden_width=hidden_width,
        tree_hidden_width=hidden_width,
        tree_fused_width=hidden_width,
        tree_router_input_width=hidden_width,
        tree_token_width=hidden_width,
    )
    config.validate()
    return config


def loss_dimensions(config: HARPRTTConfig) -> HARPRTTDimensions:
    return HARPRTTDimensions(
        layers=config.layers,
        experts=config.experts,
        primary_horizons=config.active_horizons,
        legacy_horizons=config.anchor_horizons,
        selected_experts=config.exact_k,
        history_tokens=config.route_history,
        tree_nodes=config.max_tree_nodes,
        candidates=config.candidate_width,
        trajectory_rounds=config.trajectory_rounds,
    )


def runtime_static_artifacts(
    static: StaticTargetArtifacts, device: torch.device | str
) -> StaticTargetArtifacts:
    """Keep only the final norm on the accelerator after model construction."""

    return StaticTargetArtifacts(
        geometry=static.geometry,
        token_embedding=None,
        final_rmsnorm_weight=static.final_rmsnorm_weight.to(device),
        final_rmsnorm_epsilon=static.final_rmsnorm_epsilon,
        manifest=static.manifest,
    )


def prepare_model_batch(
    batch: Mapping[str, Any], static: StaticTargetArtifacts
) -> dict[str, Any]:
    """Shallow-copy a rich batch and reconstruct deployed BF16 final RMSNorm."""

    inputs = batch.get("inputs")
    if not isinstance(inputs, Mapping):
        raise KeyError("rich batch contains no inputs mapping")
    final_hidden = inputs.get("final_hidden")
    if not isinstance(final_hidden, Tensor):
        raise KeyError("rich batch contains no inputs.final_hidden tensor")
    prepared = dict(batch)
    prepared_inputs = dict(inputs)
    prepared_inputs["final_hidden"] = static.reconstruct_final_hidden(
        final_hidden,
        activation_dtype=torch.bfloat16,
    )
    prepared["inputs"] = prepared_inputs
    return prepared


def forward_model_batch(
    model: nn.Module,
    batch: Mapping[str, Any],
    static: StaticTargetArtifacts,
) -> Mapping[str, Any]:
    """Execute the rich model with its explicit frozen-anchor causal input."""

    prepared = prepare_model_batch(batch, static)
    output = model(batch=prepared, anchor_inputs={"batch": prepared})
    if not isinstance(output, Mapping):
        raise TypeError("HARP-RTT model output must be a mapping")
    return output


def autotune_microbatch_size(
    probe: Callable[[int], int],
    *,
    candidates: Sequence[int] = MICROBATCH_CANDIDATES,
    peak_limit_bytes: int = MAX_PROCESS_PEAK_BYTES,
) -> tuple[int, list[dict[str, Any]]]:
    """Choose the first fixed candidate whose measured process peak is <=20 GiB."""

    if peak_limit_bytes < 1:
        raise ValueError("peak memory limit must be positive")
    if tuple(candidates) != MICROBATCH_CANDIDATES:
        raise ValueError("formal microbatch candidates must be exactly 8,4,2,1")
    trials: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            peak = int(probe(int(candidate)))
            if peak < 0:
                raise ValueError("microbatch probe returned a negative peak")
            safe = peak <= int(peak_limit_bytes)
            trials.append(
                {
                    "microbatch_size": int(candidate),
                    "peak_bytes": peak,
                    "peak_gib": peak / 1024**3,
                    "within_limit": safe,
                }
            )
            if safe:
                return int(candidate), trials
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(exc).lower():
                raise
            trials.append(
                {
                    "microbatch_size": int(candidate),
                    "oom": True,
                    "error": str(exc),
                    "within_limit": False,
                }
            )
    raise RuntimeError("no microbatch in 8,4,2,1 fits the 20 GiB process-peak limit")


def _loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_harp_rtt,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def _cuda_autotune(
    *,
    dataset: Dataset[dict[str, Any]],
    model: nn.Module,
    objective: HARPRTTObjective,
    controller: AuxiliaryGradientController,
    optimizer: torch.optim.Optimizer,
    runtime_static: StaticTargetArtifacts,
    device: torch.device,
    seed: int,
    num_workers: int,
    step: int,
) -> tuple[int, list[dict[str, Any]]]:
    if device.type != "cuda":
        # CPU is supported for parser/smoke work.  Production defaults to CUDA,
        # where the exact process peak is mandatory and recorded.
        return 8, [
            {
                "microbatch_size": 8,
                "peak_bytes": 0,
                "peak_gib": 0.0,
                "within_limit": True,
                "device": "cpu_diagnostic",
            }
        ]

    optimizer_reservation = _conservative_adamw_reservation_bytes(optimizer)

    def probe(candidate: int) -> int:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        loader = None
        host_batch = None
        batch = None
        try:
            loader = _loader(
                dataset,
                batch_size=candidate,
                shuffle=False,
                seed=seed,
                num_workers=num_workers,
                device=device,
            )
            host_batch = next(iter(loader))
            batch = move_to_device(host_batch, device)
            with autocast_context(device, enabled=True):
                prepared = prepare_model_batch(batch, runtime_static)
                outputs = model(
                    batch=prepared, anchor_inputs={"batch": prepared}
                )
                losses = objective(outputs, prepared, step=max(0, int(step)))
                # The 50-step audit is the highest-memory training path.  Use a
                # disposable auditor so autotuning cannot mutate run state.
                probe_audit = GradientAudit(
                    interval=GRADIENT_AUDIT_INTERVAL,
                    maximum_auxiliary_ratio=0.5,
                    required_consecutive_passes=GRADIENT_AUDIT_PASSES,
                )
                probe_controller = AuxiliaryGradientController(scale=controller.scale)
                probe_controller.audit(
                    probe_audit,
                    step=GRADIENT_AUDIT_INTERVAL,
                    losses=losses,
                    parameters=model.parameters(),
                )
                loss = controller.effective_loss(losses)
            loss.backward()
            torch.cuda.synchronize(device)
            measured = int(torch.cuda.max_memory_reserved(device))
            return measured + optimizer_reservation
        finally:
            model.zero_grad(set_to_none=True)
            del batch, host_batch, loader
            torch.cuda.empty_cache()

    return autotune_microbatch_size(probe)


def _conservative_adamw_reservation_bytes(
    optimizer: torch.optim.Optimizer,
) -> int:
    """Reserve two FP32 moments plus one FP32 update workspace per parameter."""

    parameters = {
        id(parameter): parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if isinstance(parameter, Tensor) and parameter.requires_grad
    }
    return sum(parameter.numel() * 12 for parameter in parameters.values())


def _optimizer_steps_per_epoch(rows: int, microbatch: int) -> tuple[int, int]:
    microbatches = math.ceil(rows / microbatch)
    accumulation = EFFECTIVE_BATCH_SIZE // microbatch
    return math.ceil(microbatches / accumulation), accumulation


def _set_learning_rates(
    optimizer: torch.optim.Optimizer,
    base_learning_rates: Sequence[float],
    multiplier: float,
) -> None:
    if len(optimizer.param_groups) != len(base_learning_rates):
        raise ValueError("optimizer group count changed")
    for group, base in zip(optimizer.param_groups, base_learning_rates, strict=True):
        group["lr"] = float(base) * float(multiplier)


def _train_epoch(
    *,
    epoch: int,
    model: nn.Module,
    objective: HARPRTTObjective,
    optimizer: torch.optim.Optimizer,
    controller: AuxiliaryGradientController,
    gradient_audit: GradientAudit,
    state: OptimizerStepState,
    dataset: Dataset[dict[str, Any]],
    microbatch_size: int,
    accumulation_steps: int,
    device: torch.device,
    runtime_static: StaticTargetArtifacts,
    num_workers: int,
    seed: int,
    total_steps: int,
    warmup_steps: int,
    base_learning_rates: Sequence[float],
    gradient_clip: float,
    metrics_path: Path,
) -> dict[str, Any]:
    loader = _loader(
        dataset,
        batch_size=microbatch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        device=device,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    window = 0
    window_examples = 0
    totals: dict[str, float] = defaultdict(float)
    examples = 0
    for batch_index, host_batch in enumerate(loader):
        batch = move_to_device(host_batch, device)
        batch_examples = int(batch["targets"]["future_selected_ids"].shape[0])
        examples += batch_examples
        window += 1
        window_examples += batch_examples
        last_batch = batch_index + 1 == len(loader)
        will_step = window == accumulation_steps or last_batch
        next_step = state.global_step + 1
        with autocast_context(device, enabled=True):
            prepared = prepare_model_batch(batch, runtime_static)
            outputs = model(batch=prepared, anchor_inputs={"batch": prepared})
            losses = objective(outputs, prepared, step=next_step)
        effective_auxiliary_scale = float(controller.scale)
        audit_result = None
        if will_step:
            audit_result = controller.audit(
                gradient_audit,
                step=next_step,
                losses=losses,
                parameters=model.parameters(),
            )
        # A recommendation produced by this audit starts with the next
        # effective batch; all examples accumulated into this update use one
        # identical auxiliary scale.
        effective = (
            losses.primary + effective_auxiliary_scale * losses.auxiliary
        )
        (effective * (float(batch_examples) / EFFECTIVE_BATCH_SIZE)).backward()
        state.micro_step += 1
        for name, value in losses.as_dict().items():
            totals[name] += float(value.detach()) * batch_examples
        if not will_step:
            continue

        if window_examples < EFFECTIVE_BATCH_SIZE:
            correction = float(EFFECTIVE_BATCH_SIZE) / float(window_examples)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        multiplier = cosine_warmup_multiplier(
            min(state.global_step, total_steps), total_steps, warmup_steps
        )
        _set_learning_rates(optimizer, base_learning_rates, multiplier)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=float(gradient_clip),
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state.global_step += 1
        state.auxiliary_scale = float(controller.scale)
        event: dict[str, Any] = {
            "schema": DRIVER_SCHEMA,
            "event": "optimizer_step",
            "mode": "train",
            "epoch": epoch,
            "global_step": state.global_step,
            "micro_step": state.micro_step,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "gradient_norm_before_clip": float(gradient_norm),
            "auxiliary_scale_used": effective_auxiliary_scale,
            "auxiliary_scale_next": float(controller.scale),
            "loss": float(effective.detach()),
        }
        if audit_result is not None:
            event.update(audit_result.as_dict())
        append_jsonl(metrics_path, event)
        window = 0
        window_examples = 0
    if examples == 0:
        raise ValueError("training loader produced no examples")
    return {name: value / examples for name, value in totals.items()}


def _metric_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"request_metrics"}
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validation(
    *,
    epoch: int,
    model: nn.Module,
    dataset: Dataset[dict[str, Any]],
    microbatch_size: int,
    device: torch.device,
    runtime_static: StaticTargetArtifacts,
    num_workers: int,
    seed: int,
    output_directory: Path,
    metrics_path: Path,
) -> tuple[dict[str, Any], tuple[float, ...], Path]:
    loader = _loader(
        dataset,
        batch_size=microbatch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        device=device,
    )
    report = evaluate_harp_rtt(
        model,
        loader,
        device=device,
        split="validation",
        autocast=True,
        model_call=lambda module, batch: forward_model_batch(
            module, batch, runtime_static
        ),
    )
    selection = model_selection_tuple(report)
    report_path = output_directory / f"validation_epoch_{epoch:03d}.json"
    _write_json_exclusive(report_path, report)
    append_jsonl(
        metrics_path,
        {
            "schema": DRIVER_SCHEMA,
            "event": "validation",
            "epoch": epoch,
            "selection_tuple": list(selection),
            "report": _metric_summary(report),
            "report_path": report_path.name,
            "report_sha256": sha256_file(report_path),
        },
    )
    return report, selection, report_path


def _runtime_provenance(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return result


def verify_index_inventory(index_root: Path) -> dict[str, Any]:
    """Verify every compact-index file against its immutable checksum ledger."""

    root = Path(index_root).expanduser().resolve()
    summary_path = root / "INDEX_SUMMARY.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"rich index has no INDEX_SUMMARY.json: {root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "harp_rtt_rich_event_index_collection_v1":
        raise ValueError("rich index collection schema mismatch")
    segment_paths = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "index_manifest.json").is_file()
    )
    if len(segment_paths) != int(summary.get("segments", -1)):
        raise ValueError("rich index segment count differs from INDEX_SUMMARY.json")
    records: list[dict[str, Any]] = []
    for segment in segment_paths:
        ledger_path = segment / "INDEX_SHA256SUMS.json"
        if not ledger_path.is_file():
            raise FileNotFoundError(
                f"rich index segment has no checksum ledger: {segment}"
            )
        hashes = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(hashes, dict) or "index_manifest.json" not in hashes:
            raise ValueError(f"invalid rich-index checksum ledger: {ledger_path}")
        actual_names = {
            path.name
            for path in segment.iterdir()
            if path.is_file() and path.name != ledger_path.name
        }
        if set(hashes) != actual_names:
            raise ValueError(f"rich-index ledger/file inventory mismatch: {segment}")
        for name, expected in sorted(hashes.items()):
            if (
                Path(name).name != name
                or re.fullmatch(r"[0-9a-f]{64}", str(expected)) is None
            ):
                raise ValueError(
                    f"unsafe or malformed rich-index ledger entry: {name!r}"
                )
            if sha256_file(segment / name) != expected:
                raise ValueError(f"rich-index file hash mismatch: {segment / name}")
        segment_manifest = json.loads(
            (segment / "index_manifest.json").read_text(encoding="utf-8")
        )
        if segment_manifest.get("schema") != "harp_rtt_rich_event_index_v1":
            raise ValueError(f"rich index segment schema mismatch: {segment}")
        if str(segment_manifest.get("segment")) != segment.name:
            raise ValueError(f"rich index segment identity mismatch: {segment}")
        mtp_nodes = int(segment_manifest.get("mtp_nodes", -1))
        adaptive_mtp_nodes = int(segment_manifest.get("adaptive_mtp_nodes", -1))
        legacy_mtp_nodes = int(segment_manifest.get("legacy_mtp_nodes", -1))
        anchor_spines = int(segment_manifest.get("anchor_spines", -1))
        anchor_spine_nodes = int(segment_manifest.get("anchor_spine_nodes", -1))
        anchor_complete = (
            segment_manifest.get("anchor_spine_contract") is True
            and segment_manifest.get("anchor_spine_schema")
            == "harp_rtt_legacy_anchor_spine_v1"
            and int(segment_manifest.get("anchor_spine_required_depth", -1)) == 6
            and anchor_spines > 0
            and anchor_spine_nodes == 6 * anchor_spines
            and int(segment_manifest.get("anchor_spine_labels", -1)) == 0
            and segment_manifest.get("anchor_spine_consumes_adaptive_node_budget")
            is False
            and segment_manifest.get("anchor_spine_enters_adaptive_model_inputs")
            is False
            and segment_manifest.get("anchor_spine_continues_through_eos") is True
        )
        fully_adaptive = (
            segment_manifest.get("adaptive_mtp_tree") is True
            and segment_manifest.get("decoding_profile")
            == "exact_h1_native_mtp_adaptive_h2_h4"
            and segment_manifest.get("acceptance_is_label_only") is True
            and mtp_nodes > 0
            and adaptive_mtp_nodes == mtp_nodes
            and legacy_mtp_nodes == 0
            and anchor_complete
        )
        records.append(
            {
                "segment": segment.name,
                "ledger_sha256": sha256_file(ledger_path),
                "files": dict(sorted(hashes.items())),
                "mtp_nodes": mtp_nodes,
                "adaptive_mtp_nodes": adaptive_mtp_nodes,
                "legacy_mtp_nodes": legacy_mtp_nodes,
                "anchor_spines": anchor_spines,
                "anchor_spine_nodes": anchor_spine_nodes,
                "anchor_complete": anchor_complete,
                "decoding_profile": segment_manifest.get("decoding_profile"),
                "acceptance_is_label_only": segment_manifest.get(
                    "acceptance_is_label_only"
                ),
                "fully_adaptive": fully_adaptive,
            }
        )
    adaptive_segments = sum(bool(record["fully_adaptive"]) for record in records)
    complete_anchor_segments = sum(bool(record["anchor_complete"]) for record in records)
    return {
        "schema": "harp_rtt_verified_index_inventory_v1",
        "summary_sha256": sha256_file(summary_path),
        "split_manifest_sha256": summary.get("split_manifest_sha256"),
        "segments": len(records),
        "adaptive_contract": {
            "required_decoding_profile": (
                "exact_h1_native_mtp_adaptive_h2_h4"
            ),
            "required_anchor_spine_schema": "harp_rtt_legacy_anchor_spine_v1",
            "required_anchor_spine_depth": 6,
            "fully_adaptive_segments": adaptive_segments,
            "complete_anchor_spine_segments": complete_anchor_segments,
            "legacy_or_incomplete_segments": len(records) - adaptive_segments,
            "all_segments_fully_adaptive": adaptive_segments == len(records),
            "all_segments_complete_anchor_spines": (
                complete_anchor_segments == len(records)
            ),
        },
        "inventory_sha256": hashlib.sha256(
            _canonical_json(records).encode("utf-8")
        ).hexdigest(),
        "records": records,
    }


def require_formal_adaptive_index_inventory(
    inventory: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Reject Phase 2--5 training on legacy or partially adaptive captures."""

    contract = inventory.get("adaptive_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("verified index inventory has no adaptive-tree contract")
    segment_count = int(inventory.get("segments", 0))
    fully_adaptive = int(contract.get("fully_adaptive_segments", -1))
    incomplete = int(contract.get("legacy_or_incomplete_segments", -1))
    if (
        segment_count < 1
        or contract.get("all_segments_fully_adaptive") is not True
        or fully_adaptive != segment_count
        or incomplete != 0
        or contract.get("required_anchor_spine_schema")
        != "harp_rtt_legacy_anchor_spine_v1"
        or int(contract.get("required_anchor_spine_depth", -1)) != 6
        or int(contract.get("complete_anchor_spine_segments", -1)) != segment_count
        or contract.get("all_segments_complete_anchor_spines") is not True
    ):
        raise ValueError(
            "formal Phase 2-5 training requires every rich-index segment to "
            "contain only exact-H1 native-MTP adaptive H2-H4 trees with "
            "separate label-only acceptance records and an exact causal H1-H6 "
            "legacy-anchor spine"
        )
    return contract


def verify_router_numerics_audit(
    audit_path: Path,
    expected_sha256: str,
    static_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind training to the non-test, hardware-aware real-router gate."""

    path = Path(audit_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"router numerics audit is not a file: {path}")
    expected = str(expected_sha256).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("router numerics audit SHA-256 is malformed")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"router numerics audit SHA-256 mismatch: expected {expected}, got {actual}"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("router numerics audit root must be a JSON object")
    if report.get("schema") != "harp_rtt_router_numerics_diagnostic_v2":
        raise ValueError("router numerics audit schema mismatch")
    if report.get("split") != "validation" or report.get("test_accessed") is not False:
        raise PermissionError(
            "router numerics audit must be validation-only with test_accessed=false"
        )
    if int(report.get("valid_endpoints", 0)) < 1:
        raise ValueError("router numerics audit contains no valid endpoints")
    gate = report.get("gate_assessment")
    if not isinstance(gate, Mapping):
        raise ValueError("router numerics audit has no gate_assessment")
    for name in (
        "canonical_factorization_passed",
        "captured_bf16_strict_boundary_passed",
        "training_supported_by_static_v1",
    ):
        if gate.get(name) is not True:
            raise ValueError(f"router numerics audit gate did not pass: {name}")
    if gate.get("cross_architecture_bf16_exact_replay_required") is not False:
        raise ValueError("router numerics audit has the wrong cross-hardware policy")
    observed = gate.get("canonical_factorization_observed")
    if not isinstance(observed, Mapping):
        raise ValueError("router numerics audit lacks canonical observations")
    if (
        float(observed.get("exact_set", -1.0)) != 1.0
        or float(observed.get("slot_recall", -1.0)) != 1.0
        or float(observed.get("max_abs_error", math.inf)) > 2.0e-4
    ):
        raise ValueError("router numerics canonical FP32 thresholds failed")
    strict = gate.get("captured_bf16_strict_boundary_observed")
    if not isinstance(strict, Mapping) or (
        float(strict.get("exact_set", -1.0)) != 1.0
        or float(strict.get("slot_recall", -1.0)) != 1.0
    ):
        raise ValueError("router numerics strict-boundary BF16 gate failed")

    source = report.get("source_identity")
    if not isinstance(source, Mapping):
        raise ValueError("router numerics audit lacks source identity")
    files = static_manifest.get("files")
    model = static_manifest.get("model")
    if not isinstance(files, Mapping) or not isinstance(model, Mapping):
        raise ValueError("static artifact manifest lacks files/model provenance")
    geometry_file = files.get("router_geometry.safetensors")
    if not isinstance(geometry_file, Mapping):
        raise ValueError("static artifact manifest lacks router geometry hash")
    if source.get("static_geometry_sha256") != geometry_file.get("sha256"):
        raise ValueError("router audit/static geometry SHA-256 mismatch")
    if source.get("checkpoint_revision") != model.get("repository_revision"):
        raise ValueError("router audit/static checkpoint revision mismatch")
    if source.get("checkpoint_index_sha256") != model.get("index_sha256"):
        raise ValueError("router audit/static checkpoint index SHA-256 mismatch")
    return {
        "schema": "harp_rtt_verified_router_numerics_audit_v1",
        "path": str(path),
        "sha256": actual,
        "split": "validation",
        "test_accessed": False,
        "valid_endpoints": int(report["valid_endpoints"]),
        "canonical_factorization_observed": dict(observed),
        "captured_bf16_strict_boundary_observed": dict(strict),
        "captured_tied_boundary_endpoints": int(
            report.get("captured_tied_boundary_endpoints", 0)
        ),
        "source_identity": dict(source),
        "capture_provenance": report.get("capture_provenance"),
        "environment": report.get("environment"),
    }


def _resolved_paths(args: argparse.Namespace) -> dict[str, str | None]:
    names = (
        "index_root",
        "corpus_root",
        "static_dir",
        "router_numerics_audit",
        "anchor_checkpoint",
        "target_preprocessing",
        "mtp_preprocessing",
        "output_dir",
        "initialize_from",
        "resume",
    )
    return {
        name: str(Path(getattr(args, name)).expanduser().resolve())
        if getattr(args, name, None) is not None
        else None
        for name in names
    }


def _validate_paths(args: argparse.Namespace) -> None:
    for name in (
        "index_root",
        "corpus_root",
        "static_dir",
    ):
        path = Path(getattr(args, name)).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"--{name.replace('_', '-')} is not a directory: {path}")
    for name in (
        "anchor_checkpoint",
        "router_numerics_audit",
        "target_preprocessing",
        "mtp_preprocessing",
    ):
        path = Path(getattr(args, name)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} is not a file: {path}")
    for name in ("initialize_from", "resume"):
        value = getattr(args, name)
        if value is not None and not Path(value).expanduser().is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} is not a file: {value}")


def _fresh_or_resume_output(args: argparse.Namespace) -> tuple[Path, dict[str, Any] | None]:
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory {output}")
    if args.resume is None:
        return output, None
    resume = Path(args.resume).expanduser().resolve()
    if resume.parent.name != "checkpoints":
        raise ValueError("--resume must name a checkpoint in a run checkpoints directory")
    source_run_manifest = resume.parent.parent / "run_manifest.json"
    if not source_run_manifest.is_file():
        raise ValueError("resume source contains no run_manifest.json")
    manifest = json.loads(source_run_manifest.read_text(encoding="utf-8"))
    checkpoint = json.loads(resume.read_text(encoding="utf-8"))
    if manifest.get("schema") != DRIVER_SCHEMA:
        raise ValueError("resume run manifest schema mismatch")
    if checkpoint.get("schema") != "harp_rtt_checkpoint_manifest_v1":
        raise ValueError("resume checkpoint manifest schema mismatch")
    if manifest.get("phase", {}).get("name") != args.phase:
        raise ValueError("resume run phase differs from --phase")
    provenance = checkpoint.get("provenance", {})
    if provenance.get("driver_schema") != DRIVER_SCHEMA:
        raise ValueError("resume checkpoint driver provenance mismatch")
    if provenance.get("run_manifest") != source_run_manifest.name:
        raise ValueError("resume checkpoint names a different source run manifest")
    if provenance.get("run_manifest_sha256") != sha256_file(source_run_manifest):
        raise ValueError("resume source run manifest hash mismatch")
    return output, manifest


def _validate_resume_contract(
    source: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    exact = (
        "phase",
        "static_artifact_manifest",
        "router_numerics_audit",
        "model_config",
        "loss_dimensions",
        "parameter_ownership",
        "train_subset",
        "validation_subset",
        "microbatch_size",
        "gradient_accumulation",
        "effective_batch_size",
        "memory_peak_limit_bytes",
        "memory_autotune_policy",
        "optimizer",
        "runtime",
        "seed",
        "deterministic",
        "require_promotion_gate",
        "gradient_audit_policy",
        "index_inventory",
    )
    for name in exact:
        if _canonical_json(source.get(name)) != _canonical_json(current.get(name)):
            raise ValueError(f"resume contract differs on {name}")
    hash_names = (
        "checkpoint_sha256",
        "target_preprocessing_sha256",
        "mtp_preprocessing_sha256",
    )
    for name in hash_names:
        if source.get("anchor", {}).get(name) != current.get("anchor", {}).get(name):
            raise ValueError(f"resume anchor artifact differs on {name}")


def _bound_source_run_manifest(
    checkpoint_path: Path, checkpoint_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Load the run manifest cryptographically named by a checkpoint."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("phase parent must be stored in a run checkpoints directory")
    source_path = checkpoint.parent.parent / "run_manifest.json"
    if not source_path.is_file():
        raise ValueError("phase parent source contains no run_manifest.json")
    provenance = checkpoint_manifest.get("provenance", {})
    if provenance.get("driver_schema") != DRIVER_SCHEMA:
        raise ValueError("phase parent checkpoint driver provenance mismatch")
    if provenance.get("run_manifest") != source_path.name:
        raise ValueError("phase parent checkpoint names a different run manifest")
    if provenance.get("run_manifest_sha256") != sha256_file(source_path):
        raise ValueError("phase parent run manifest hash mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema") != DRIVER_SCHEMA:
        raise ValueError("phase parent run manifest schema mismatch")
    source_phase = source.get("phase", {}).get("name")
    checkpoint_phase = checkpoint_manifest.get("phase", {}).get("name")
    if source_phase != checkpoint_phase:
        raise ValueError("phase parent checkpoint/run phase mismatch")
    return source


def _validate_phase_transition_contract(
    source: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Require a new phase to preserve every data/static/anchor contract."""

    exact = (
        "static_artifact_manifest",
        "router_numerics_audit",
        "model_config",
        "loss_dimensions",
        "train_subset",
        "validation_subset",
        "seed",
        "deterministic",
        "index_inventory",
    )
    for name in exact:
        if _canonical_json(source.get(name)) != _canonical_json(current.get(name)):
            raise ValueError(f"phase transition contract differs on {name}")
    for name in (
        "checkpoint_sha256",
        "target_preprocessing_sha256",
        "mtp_preprocessing_sha256",
    ):
        if source.get("anchor", {}).get(name) != current.get("anchor", {}).get(name):
            raise ValueError(f"phase transition anchor artifact differs on {name}")


def _checkpoint_provenance(
    *,
    run_manifest_path: Path,
    epoch: int,
    validation: Mapping[str, Any],
    parent_checkpoint: str | None,
) -> dict[str, Any]:
    return {
        "driver_schema": DRIVER_SCHEMA,
        "run_manifest": run_manifest_path.name,
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "epoch": int(epoch),
        "validation": _metric_summary(validation),
        "parent_checkpoint": parent_checkpoint,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    phase = resolve_phase_spec(args)
    _validate_paths(args)
    index_inventory = verify_index_inventory(args.index_root)
    require_formal_adaptive_index_inventory(index_inventory)
    output_directory, resume_manifest = _fresh_or_resume_output(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(args.seed, deterministic=args.deterministic)

    static = load_static_target_artifacts(args.static_dir, device="cpu")
    router_numerics = verify_router_numerics_audit(
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

    train_full = HarpRTTDataset(
        args.index_root,
        "train",
        corpus_root=args.corpus_root,
        max_tree_nodes=config.max_tree_nodes,
    )
    validation_full = HarpRTTDataset(
        args.index_root,
        "validation",
        corpus_root=args.corpus_root,
        max_tree_nodes=config.max_tree_nodes,
    )
    train_data, train_subset = request_complete_subset(
        train_full, args.max_train_requests, seed=args.seed
    )
    validation_data, validation_subset = request_complete_subset(
        validation_full, args.max_validation_requests, seed=args.seed
    )

    groups, ownership = configure_training_phase(model, phase)
    optimizer = torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    base_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    gradient_audit = GradientAudit(
        interval=GRADIENT_AUDIT_INTERVAL,
        maximum_auxiliary_ratio=0.5,
        required_consecutive_passes=GRADIENT_AUDIT_PASSES,
    )
    controller = AuxiliaryGradientController()
    state = OptimizerStepState()
    parent_checkpoint = None
    initialize_manifest: dict[str, Any] | None = None
    start_epoch = 1
    if args.initialize_from is not None:
        _, parent_manifest = load_checkpoint(
            args.initialize_from, model=model, optimizer=None, gradient_audit=None
        )
        expected_parent = _expected_parent_phase(phase.name)
        actual_parent = str(parent_manifest["phase"]["name"])
        if actual_parent != expected_parent:
            raise ValueError(
                f"{phase.name} requires a {expected_parent} checkpoint, "
                f"found {actual_parent}"
            )
        parent_checkpoint = str(Path(args.initialize_from).expanduser().resolve())
        initialize_manifest = _bound_source_run_manifest(
            args.initialize_from, parent_manifest
        )
    elif args.resume is not None:
        state, resumed = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            gradient_audit=gradient_audit,
        )
        if resumed["phase"]["name"] != phase.name:
            raise ValueError("resume checkpoint phase differs from --phase")
        controller.scale = float(state.auxiliary_scale)
        start_epoch = int(resumed.get("provenance", {}).get("epoch", -1)) + 1
        if start_epoch < 1:
            raise ValueError("resume checkpoint has no valid completed epoch")
        parent_checkpoint = str(Path(args.resume).expanduser().resolve())

    if resume_manifest is None:
        microbatch_size, memory_trials = _cuda_autotune(
            dataset=train_data,
            model=model,
            objective=objective,
            controller=controller,
            optimizer=optimizer,
            runtime_static=runtime_static,
            device=device,
            seed=args.seed,
            num_workers=args.num_workers,
            step=state.global_step + 1,
        )
    else:
        if resume_manifest["train_subset"]["selected_request_ids_sha256"] != train_subset[
            "selected_request_ids_sha256"
        ]:
            raise ValueError("resume train request subset differs from the original run")
        if resume_manifest["validation_subset"]["selected_request_ids_sha256"] != validation_subset[
            "selected_request_ids_sha256"
        ]:
            raise ValueError("resume validation request subset differs from the original run")
        microbatch_size = int(resume_manifest["microbatch_size"])
        memory_trials = list(resume_manifest["memory_autotune"])
    if microbatch_size not in MICROBATCH_CANDIDATES:
        raise ValueError("recorded microbatch size is outside 8,4,2,1")
    steps_per_epoch, accumulation_steps = _optimizer_steps_per_epoch(
        len(train_data), microbatch_size
    )
    total_steps = steps_per_epoch * phase.epochs
    warmup_steps = min(
        int(total_steps * phase.warmup_fraction), max(0, total_steps - 1)
    )
    if state.global_step > total_steps or start_epoch > phase.epochs + 1:
        raise ValueError("resume state lies beyond the configured phase schedule")

    metrics_path = output_directory / "metrics.jsonl"
    run_manifest_path = output_directory / "run_manifest.json"
    checkpoints = output_directory / "checkpoints"
    run_manifest = {
        "schema": DRIVER_SCHEMA,
        "phase": asdict(phase),
        "paths": _resolved_paths(args),
        "lineage": {
            "mode": (
                "resume"
                if args.resume is not None
                else "initialize"
                if args.initialize_from is not None
                else "fresh"
            ),
            "parent_checkpoint": parent_checkpoint,
        },
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "anchor": anchor_provenance,
        "static_artifact_manifest": runtime_static.manifest,
        "router_numerics_audit": router_numerics,
        "index_inventory": index_inventory,
        "model_config": config.to_dict(),
        "loss_dimensions": asdict(loss_dimensions(config)),
        "parameter_ownership": ownership.to_dict(),
        "train_subset": train_subset,
        "validation_subset": validation_subset,
        "microbatch_size": microbatch_size,
        "gradient_accumulation": accumulation_steps,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "memory_peak_limit_bytes": MAX_PROCESS_PEAK_BYTES,
        "memory_autotune": memory_trials,
        "memory_autotune_policy": (
            "audit_path_peak_reserved_plus_12_bytes_per_trainable_parameter_"
            "for_adamw_moments_and_workspace"
        ),
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "base_learning_rates": base_learning_rates,
            "gradient_clip": float(args.gradient_clip),
            "scheduler": "linear_warmup_then_cosine",
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
        },
        "gradient_audit": gradient_audit.state_dict(),
        "gradient_audit_policy": {
            "interval": GRADIENT_AUDIT_INTERVAL,
            "maximum_auxiliary_ratio": 0.5,
            "required_consecutive_passes": GRADIENT_AUDIT_PASSES,
            "parameter_scope": "all phase-owned trainable shared-path parameters",
            "batch_scope": (
                "final deterministic microbatch of each audited effective batch"
            ),
        },
        "require_promotion_gate": bool(args.require_promotion_gate),
        "runtime": _runtime_provenance(device),
        "sealed_test": {
            "opened": False,
            "authorization_option_exposed": False,
        },
    }
    if initialize_manifest is not None:
        _validate_phase_transition_contract(initialize_manifest, run_manifest)
    if resume_manifest is not None:
        _validate_resume_contract(resume_manifest, run_manifest)
        expected_global_step = steps_per_epoch * (start_epoch - 1)
        if state.global_step != expected_global_step:
            raise ValueError(
                "resume checkpoint is not an exact completed-epoch boundary: "
                f"global_step={state.global_step}, expected={expected_global_step}"
            )
        if state.best_selection_tuple is None:
            raise ValueError("resume checkpoint contains no best validation selection")

    output_directory.mkdir(parents=True, exist_ok=False)
    checkpoints.mkdir()
    _write_json_exclusive(run_manifest_path, run_manifest)
    append_jsonl(
        metrics_path,
        {
            "schema": DRIVER_SCHEMA,
            "event": "run_start",
            "phase": phase.name,
            "lineage_mode": run_manifest["lineage"]["mode"],
        },
    )

    baseline_epoch = start_epoch - 1 if args.resume is not None else 0
    baseline_validation, baseline_selection, _ = _validation(
        epoch=baseline_epoch,
        model=model,
        dataset=validation_data,
        microbatch_size=microbatch_size,
        device=device,
        runtime_static=runtime_static,
        num_workers=args.num_workers,
        seed=args.seed,
        output_directory=output_directory,
        metrics_path=metrics_path,
    )
    if args.resume is None:
        state.best_selection_tuple = baseline_selection
        state.best_checkpoint = "epoch_000_incumbent.manifest.json"
        save_checkpoint(
            checkpoints,
            tag="epoch_000_incumbent",
            model=model,
            optimizer=optimizer,
            state=state,
            phase=phase,
            gradient_audit=gradient_audit,
            provenance=_checkpoint_provenance(
                run_manifest_path=run_manifest_path,
                epoch=0,
                validation=baseline_validation,
                parent_checkpoint=parent_checkpoint,
            ),
        )
    else:
        # Keep the immutable source checkpoint as the incumbent until this
        # continuation produces a strictly better validation tuple.
        state.best_checkpoint = parent_checkpoint

    if args.resume is None and phase.name == "phase4":
        gate = candidate_coverage_gate(baseline_validation)
        append_jsonl(
            metrics_path,
            {"schema": DRIVER_SCHEMA, "event": "candidate_gate", **gate},
        )
        if not gate["passed"]:
            raise RuntimeError(
                "Phase 4 candidate gate failed: require C64 mean H1-H4 >= .985 "
                "and H4 >= .970"
            )
    if args.resume is None and phase.name == "phase5":
        recall = float(
            baseline_validation["mean_h1_h4_request_macro_slot_recall_at_8"]
        )
        append_jsonl(
            metrics_path,
            {
                "schema": DRIVER_SCHEMA,
                "event": "ranker_gate",
                "mean_h1_h4_recall": recall,
                "threshold": RANKER_GATE_RECALL,
                "passed": recall >= RANKER_GATE_RECALL,
            },
        )
        if recall < RANKER_GATE_RECALL:
            raise RuntimeError(
                "Phase 5 ranker gate failed: validation mean H1-H4 Recall@8 < .85"
            )

    for epoch in range(start_epoch, phase.epochs + 1):
        epoch_seed = _epoch_seed(args.seed, phase.name, epoch)
        seed_everything(epoch_seed, deterministic=args.deterministic)
        training_metrics = _train_epoch(
            epoch=epoch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            controller=controller,
            gradient_audit=gradient_audit,
            state=state,
            dataset=train_data,
            microbatch_size=microbatch_size,
            accumulation_steps=accumulation_steps,
            device=device,
            runtime_static=runtime_static,
            num_workers=args.num_workers,
            seed=epoch_seed,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_learning_rates=base_learning_rates,
            gradient_clip=args.gradient_clip,
            metrics_path=metrics_path,
        )
        append_jsonl(
            metrics_path,
            {
                "schema": DRIVER_SCHEMA,
                "event": "train_epoch",
                "epoch": epoch,
                "metrics": training_metrics,
            },
        )
        validation, selection, _ = _validation(
            epoch=epoch,
            model=model,
            dataset=validation_data,
            microbatch_size=microbatch_size,
            device=device,
            runtime_static=runtime_static,
            num_workers=args.num_workers,
            seed=args.seed,
            output_directory=output_directory,
            metrics_path=metrics_path,
        )
        if state.best_selection_tuple is None or selection > state.best_selection_tuple:
            tag = f"epoch_{epoch:03d}_best"
            state.best_selection_tuple = selection
            state.best_checkpoint = f"{tag}.manifest.json"
            save_checkpoint(
                checkpoints,
                tag=tag,
                model=model,
                optimizer=optimizer,
                state=state,
                phase=phase,
                gradient_audit=gradient_audit,
                provenance=_checkpoint_provenance(
                    run_manifest_path=run_manifest_path,
                    epoch=epoch,
                    validation=validation,
                    parent_checkpoint=parent_checkpoint,
                ),
            )
            append_jsonl(
                metrics_path,
                {
                    "schema": DRIVER_SCHEMA,
                    "event": "checkpoint",
                    "epoch": epoch,
                    "tag": tag,
                    "selection_tuple": list(selection),
                },
            )

    promotion_passed = (
        gradient_audit.promotion_ready
        and gradient_audit.last_audit_step is not None
        and gradient_audit.last_audit_step >= 500
    )
    result = {
        "schema": DRIVER_SCHEMA,
        "phase": phase.name,
        "global_step": state.global_step,
        "best_checkpoint": state.best_checkpoint,
        "best_selection_tuple": list(state.best_selection_tuple)
        if state.best_selection_tuple is not None
        else None,
        "auxiliary_scale": float(controller.scale),
        "gradient_audit": gradient_audit.state_dict(),
        "promotion_gate_requested": bool(args.require_promotion_gate),
        "promotion_gate_passed": promotion_passed,
        "sealed_test_opened": False,
    }
    append_jsonl(metrics_path, {"event": "run_end", **result})
    if args.require_promotion_gate and not promotion_passed:
        raise RuntimeError(
            "phase promotion blocked: require ten consecutive passing 50-step "
            "gradient audits spanning at least 500 optimizer steps"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DRIVER_SCHEMA",
    "EFFECTIVE_BATCH_SIZE",
    "MAX_PROCESS_PEAK_BYTES",
    "MICROBATCH_CANDIDATES",
    "autotune_microbatch_size",
    "build_parser",
    "forward_model_batch",
    "loss_dimensions",
    "main",
    "prepare_model_batch",
    "production_config",
    "request_complete_subset",
    "resolve_phase_spec",
    "run",
    "runtime_static_artifacts",
    "verify_index_inventory",
    "require_formal_adaptive_index_inventory",
    "verify_router_numerics_audit",
]
