"""Train HARP-8T on audited request-aligned traces without opening test."""

from __future__ import annotations

from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any, Sequence

import numpy as np
import torch

from .config import HARPConfig, LossConfig, TrainingConfig
from .data import CompactHARPData
from .losses import (
    calibrated_coefficients,
    component_gradient_norms,
    endpoint_loss,
)
from .metrics import (
    evaluate_split,
    model_inputs,
    request_bootstrap_h2,
    request_bootstrap_mean_h1_h4,
)
from .model import HARP8Teacher


TRAINING_SCHEMA = "harp8t_training_v1"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _optimizer_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if (
            parameter.ndim == 1
            or lowered.endswith("bias")
            or "embedding" in lowered
            or "norm" in lowered
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, int(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup:
            return max(1e-8, float(step + 1) / warmup)
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _rng_state(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_generator": rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict[str, Any], rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    rng.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _mirror(paths: list[Path], mirror_dir: Path | None) -> None:
    if mirror_dir is None:
        return
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            temporary = mirror_dir / f".{path.name}.partial"
            shutil.copy2(path, temporary)
            temporary.replace(mirror_dir / path.name)


def _save_state(
    path: Path,
    *,
    model: HARP8Teacher,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rng: np.random.Generator,
    model_config: HARPConfig,
    loss_config: LossConfig,
    training_config: TrainingConfig,
    coefficients: dict[str, float],
    epoch: int,
    next_batch_start: int,
    epoch_indices: np.ndarray,
    global_step: int,
    best_selection: tuple[float, float, float],
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
) -> None:
    payload = {
        "schema": TRAINING_SCHEMA,
        "model_config": model_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "training_config": training_config.to_dict(),
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": _rng_state(rng),
        "coefficients": coefficients,
        "epoch": epoch,
        "next_batch_start": next_batch_start,
        "epoch_indices": epoch_indices,
        "global_step": global_step,
        "best_selection": best_selection,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
    }
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def _shared_calibration_parameters(model: HARP8Teacher) -> list[torch.nn.Parameter]:
    modules = [
        model.route_cell,
        model.temporal_blocks,
        model.layer_route_blocks,
        model.mtp_node_fusion,
        model.cross_blocks,
        model.fusion_blocks,
    ]
    values: list[torch.nn.Parameter] = []
    for module in modules:
        values.extend(module.parameters())
    return values


def calibrate_gradients(
    model: HARP8Teacher,
    data: CompactHARPData,
    loss_config: LossConfig,
    training_config: TrainingConfig,
    *,
    steps: int,
    device: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    if steps <= 0:
        return {
            "router_kl": loss_config.router_kl,
            "boundary": loss_config.boundary,
            "inclusion": loss_config.inclusion,
            "centered_score": loss_config.centered_score,
            "future_latent": loss_config.future_latent,
        }, {"steps": 0, "mode": "declared_initial_coefficients"}
    rng = np.random.default_rng(training_config.seed + 991)
    observations: defaultdict[str, list[float]] = defaultdict(list)
    parameters = _shared_calibration_parameters(model)
    model.train()
    batches = data.shuffled_batches(
        "train", training_config.batch_size, rng
    )
    for step, indices in enumerate(batches):
        if step >= steps:
            break
        batch = data.batch(indices, device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.startswith("cuda"),
        ):
            outputs = model(**model_inputs(batch))
            loss = endpoint_loss(
                outputs,
                batch,
                loss_config,
                active_horizons=(2,) if training_config.h2_only else training_config.active_horizons,
            )
        norms = component_gradient_norms(loss.components, parameters)
        for name, value in norms.items():
            observations[name].append(value)
        model.zero_grad(set_to_none=True)
    medians = {
        name: float(np.median(values)) for name, values in observations.items()
    }
    coefficients = calibrated_coefficients(medians)
    for name in (
        "router_kl",
        "boundary",
        "inclusion",
        "centered_score",
        "future_latent",
        "full_membership",
        "candidate16",
    ):
        if float(getattr(loss_config, name)) == 0.0:
            coefficients[name] = 0.0
    return coefficients, {
        "steps": min(steps, len(next(iter(observations.values()), []))),
        "unweighted_median_shared_gradient_norm": medians,
        "frozen_coefficients": coefficients,
        "target_ratios": {
            "router_kl": 1.0,
            "boundary": 0.5,
            "inclusion": 0.25,
            "centered_score": 0.1,
            "future_latent": 0.1,
        },
    }


def _selection_tuple(
    metrics: list[dict[str, Any]],
    profile: str,
) -> tuple[float, ...]:
    h2 = next(row for row in metrics if int(row["horizon"]) == 2)
    h2_recall = float(h2["request_macro_slot_recall_at_8"])
    mean_recall = float(
        np.mean([row["request_macro_slot_recall_at_8"] for row in metrics])
    )
    mean_kl = float(np.mean([row["router_kl"] for row in metrics]))
    if profile == "h2":
        return h2_recall, mean_recall, -mean_kl
    if profile == "h1_h4_candidate16":
        focus = [row for row in metrics if 1 <= int(row["horizon"]) <= 4]
        coverage = [
            float(row["request_macro_slot_recall_at_16"]) for row in focus
        ]
        recall = [
            float(row["request_macro_slot_recall_at_8"]) for row in focus
        ]
        return (
            float(np.mean(coverage)),
            float(np.min(coverage)),
            float(np.mean(recall)),
            -float(np.mean([row["router_kl"] for row in focus])),
        )
    raise ValueError(f"unknown selection profile {profile!r}")


def train(
    data: CompactHARPData,
    output_dir: Path,
    model_config: HARPConfig,
    loss_config: LossConfig,
    training_config: TrainingConfig,
    *,
    device: str = "cuda:0",
    resume: Path | None = None,
    initialize_from: Path | None = None,
    gradient_calibration_steps: int = 32,
    checkpoint_interval: int = 500,
    mirror_dir: Path | None = None,
    freeze_legacy: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if resume is not None and initialize_from is not None:
        raise ValueError("resume and initialize_from are mutually exclusive")
    if output_dir.exists() and resume is None:
        raise FileExistsError(f"refusing to reuse output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    model = HARP8Teacher(model_config).to(device)
    if freeze_legacy and initialize_from is None:
        raise ValueError("freeze_legacy requires initialize_from")
    if freeze_legacy and gradient_calibration_steps > 0:
        raise ValueError("freeze_legacy requires gradient-calibration-steps=0")
    if initialize_from is not None:
        initial = torch.load(
            initialize_from, map_location="cpu", weights_only=False
        )
        if initial.get("schema") != TRAINING_SCHEMA:
            raise ValueError("initial checkpoint has an incompatible schema")
        initial_config = HARPConfig(**initial["model_config"]).to_dict()
        if initial_config != model_config.to_dict():
            raise ValueError("initial checkpoint model configuration differs")
        initial_state = initial["model_state"]
        try:
            model.load_state_dict(initial_state, strict=True)
        except RuntimeError:
            # The accuracy-expansion model adds function-preserving direct
            # skip projections. Permit a validated pre-expansion endpoint to
            # warm-start those trials: the new projection tails are explicitly
            # zero-initialized in HARP8Teacher._reset_parameters, so missing
            # keys do not perturb the endpoint at step zero. Reject any
            # mismatch outside the declared extension to avoid silently
            # loading incompatible checkpoints.
            missing, unexpected = model.load_state_dict(
                initial_state, strict=False
            )
            allowed_prefixes = (
                "direct_route_projection.",
                "direct_mtp_projection.",
                "direct_target_projection.",
                "position_projection.",
            )
            bad_missing = [
                key for key in missing
                if not key.startswith(allowed_prefixes)
            ]
            if unexpected or bad_missing:
                raise ValueError(
                    "initial checkpoint differs outside the zero-initialized "
                    f"direct extension; missing={bad_missing}, "
                    f"unexpected={unexpected}"
                )
    if freeze_legacy:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in (
            model.direct_route_projection,
            model.direct_mtp_projection,
            model.direct_target_projection,
            model.position_projection,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    fused = device.startswith("cuda")
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, training_config.weight_decay),
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
        eps=training_config.epsilon,
        fused=fused,
    )
    train_count = len(data.indices("train"))
    updates_per_epoch = math.ceil(
        math.ceil(train_count / training_config.batch_size)
        / training_config.gradient_accumulation
    )
    total_updates = updates_per_epoch * training_config.epochs
    scheduler = _scheduler(
        optimizer,
        total_updates,
        training_config.warmup_fraction,
        training_config.minimum_learning_rate / training_config.learning_rate,
    )
    coefficients, calibration = calibrate_gradients(
        model,
        data,
        loss_config,
        training_config,
        steps=gradient_calibration_steps,
        device=device,
    )
    history: list[dict[str, Any]] = []
    epoch = 1
    next_batch_start = 0
    epoch_indices = data.indices("train")
    rng.shuffle(epoch_indices)
    global_step = 0
    selection_width = (
        4 if training_config.selection_profile == "h1_h4_candidate16" else 3
    )
    if initialize_from is None:
        best_selection = tuple([-math.inf] * selection_width)
    best_epoch = 0
    stale_epochs = 0
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state.get("schema") != TRAINING_SCHEMA:
            raise ValueError("resume checkpoint has an incompatible schema")
        resumed_model_config = HARPConfig(**state["model_config"]).to_dict()
        if resumed_model_config != model_config.to_dict():
            raise ValueError("resume model configuration differs")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        coefficients = {name: float(value) for name, value in state["coefficients"].items()}
        epoch = int(state["epoch"])
        next_batch_start = int(state["next_batch_start"])
        epoch_indices = np.asarray(state["epoch_indices"], dtype=np.int64)
        global_step = int(state["global_step"])
        best_selection = tuple(float(value) for value in state["best_selection"])
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        history = list(state["history"])
        _restore_rng(state["rng_state"], rng)

    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    if initialize_from is not None:
        # Preserve the known endpoint as the incumbent. This makes a
        # warm-start experiment fail-safe: if every fine-tuning epoch regresses
        # on validation, best.pt still contains the validated initialization.
        model.eval()
        initial_validation = evaluate_split(
            model,
            data,
            "validation",
            batch_size=training_config.evaluation_batch_size,
            device=device,
        )
        best_selection = _selection_tuple(
            initial_validation.horizon_metrics,
            training_config.selection_profile,
        )
        torch.save(
            {
                "schema": TRAINING_SCHEMA,
                "model_config": model_config.to_dict(),
                "loss_config": loss_config.to_dict(),
                "training_config": training_config.to_dict(),
                "coefficients": coefficients,
                "model_state": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "epoch": 0,
                "validation_selection": best_selection,
                "seed": training_config.seed,
                "initialized_endpoint": str(initialize_from),
            },
            best_path,
        )
    else:
        best_selection = tuple([-math.inf] * (
            4 if training_config.selection_profile == "h1_h4_candidate16" else 3
        ))
    active_horizons = (
        (2,) if training_config.h2_only else training_config.active_horizons
    )
    while epoch <= training_config.epochs:
        if freeze_legacy:
            # Keep the validated backbone deterministic while the new direct
            # adapters learn; the adapters contain no dropout.
            model.eval()
        else:
            model.train()
        optimizer.zero_grad(set_to_none=True)
        running: defaultdict[str, float] = defaultdict(float)
        micro_batches = 0
        optimizer_steps = 0
        batch_starts = list(
            range(next_batch_start, len(epoch_indices), training_config.batch_size)
        )
        for ordinal, start in enumerate(batch_starts):
            indices = epoch_indices[start : start + training_config.batch_size]
            batch = data.batch(indices, device)
            last_micro = ordinal + 1 == len(batch_starts)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                outputs = model(**model_inputs(batch))
                loss = endpoint_loss(
                    outputs,
                    batch,
                    loss_config,
                    active_horizons=active_horizons,
                    coefficients=coefficients,
                )
                scaled_loss = loss.total / training_config.gradient_accumulation
            scaled_loss.backward()
            micro_batches += 1
            for name, value in loss.metrics.items():
                running[name] += value
            should_step = (
                micro_batches % training_config.gradient_accumulation == 0
                or last_micro
            )
            if should_step:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.gradient_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                optimizer_steps += 1
                running["gradient_norm"] += float(gradient_norm)
                next_batch_start = min(
                    len(epoch_indices), start + training_config.batch_size
                )
                if checkpoint_interval and global_step % checkpoint_interval == 0:
                    _save_state(
                        last_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        rng=rng,
                        model_config=model_config,
                        loss_config=loss_config,
                        training_config=training_config,
                        coefficients=coefficients,
                        epoch=epoch,
                        next_batch_start=next_batch_start,
                        epoch_indices=epoch_indices,
                        global_step=global_step,
                        best_selection=best_selection,
                        best_epoch=best_epoch,
                        stale_epochs=stale_epochs,
                        history=history,
                    )
                    _mirror([last_path], mirror_dir)

        validation = evaluate_split(
            model,
            data,
            "validation",
            batch_size=training_config.evaluation_batch_size,
            device=device,
        )
        selection = _selection_tuple(
            validation.horizon_metrics,
            training_config.selection_profile,
        )
        focus_rows = [
            row
            for row in validation.horizon_metrics
            if 1 <= int(row["horizon"]) <= 4
        ]
        h2_row = next(
            row for row in validation.horizon_metrics if int(row["horizon"]) == 2
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_h2_request_macro_recall_at_8": float(
                h2_row["request_macro_slot_recall_at_8"]
            ),
            "validation_mean_h1_h8_request_macro_recall_at_8": float(
                np.mean(
                    [
                        row["request_macro_slot_recall_at_8"]
                        for row in validation.horizon_metrics
                    ]
                )
            ),
            "validation_mean_router_kl": float(
                np.mean([row["router_kl"] for row in validation.horizon_metrics])
            ),
            "validation_mean_h1_h4_candidate_coverage_at_16": float(
                np.mean(
                    [
                        row["request_macro_slot_recall_at_16"]
                        for row in focus_rows
                    ]
                )
            ),
            "validation_min_h1_h4_candidate_coverage_at_16": float(
                np.min(
                    [
                        row["request_macro_slot_recall_at_16"]
                        for row in focus_rows
                    ]
                )
            ),
            "validation_mean_h1_h4_request_macro_recall_at_8": float(
                np.mean(
                    [
                        row["request_macro_slot_recall_at_8"]
                        for row in focus_rows
                    ]
                )
            ),
            **{
                f"train_{name}": value / max(1, micro_batches)
                for name, value in running.items()
                if name != "gradient_norm"
            },
            "train_gradient_norm": running["gradient_norm"]
            / max(1, optimizer_steps),
        }
        for row in validation.horizon_metrics:
            record[
                f"validation_recall_h{int(row['horizon'])}"
            ] = row["request_macro_slot_recall_at_8"]
        history.append(record)
        print(canonical_json({"event": "harp8_epoch", **record}), flush=True)
        if selection > best_selection:
            best_selection = selection
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "schema": TRAINING_SCHEMA,
                    "model_config": model_config.to_dict(),
                    "loss_config": loss_config.to_dict(),
                    "training_config": training_config.to_dict(),
                    "coefficients": coefficients,
                    "model_state": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "validation_selection": selection,
                    "seed": training_config.seed,
                },
                best_path,
            )
        else:
            stale_epochs += 1

        epoch += 1
        next_batch_start = 0
        epoch_indices = data.indices("train")
        rng.shuffle(epoch_indices)
        _save_state(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng=rng,
            model_config=model_config,
            loss_config=loss_config,
            training_config=training_config,
            coefficients=coefficients,
            epoch=epoch,
            next_batch_start=next_batch_start,
            epoch_indices=epoch_indices,
            global_step=global_step,
            best_selection=best_selection,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            history=history,
        )
        write_csv(output_dir / "training_history.csv", history)
        write_csv(output_dir / "validation_metrics.latest.csv", validation.horizon_metrics)
        write_csv(output_dir / "validation_request_metrics.latest.csv", validation.request_metrics)
        write_csv(output_dir / "validation_layer_metrics.latest.csv", validation.layer_metrics)
        write_csv(output_dir / "validation_domain_metrics.latest.csv", validation.domain_metrics)
        _mirror(
            [
                last_path,
                best_path,
                output_dir / "training_history.csv",
                output_dir / "validation_metrics.latest.csv",
            ],
            mirror_dir,
        )
        if (
            epoch - 1 >= training_config.minimum_epochs
            and stale_epochs >= training_config.patience
        ):
            break

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    validation = evaluate_split(
        model,
        data,
        "validation",
        batch_size=training_config.evaluation_batch_size,
        device=device,
    )
    write_csv(output_dir / "validation_metrics.csv", validation.horizon_metrics)
    write_csv(output_dir / "validation_request_metrics.csv", validation.request_metrics)
    write_csv(output_dir / "validation_layer_metrics.csv", validation.layer_metrics)
    write_csv(output_dir / "validation_domain_metrics.csv", validation.domain_metrics)
    bootstrap = request_bootstrap_h2(
        validation.request_metrics, seed=training_config.seed
    )
    (output_dir / "validation_h2_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    h1_h4_bootstrap = {
        str(candidate_count): request_bootstrap_mean_h1_h4(
            validation.request_metrics,
            candidate_count=candidate_count,
            seed=training_config.seed,
        )
        for candidate_count in (8, 16)
    }
    h1_h4_bootstrap_path = output_dir / "validation_h1_h4_bootstrap.json"
    h1_h4_bootstrap_path.write_text(
        json.dumps(h1_h4_bootstrap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selection_order = (
        [
            "validation mean H1-H4 request-macro candidate coverage@16",
            "validation minimum H1-H4 request-macro candidate coverage@16",
            "validation mean H1-H4 request-macro SlotRecall@8",
            "negative validation mean H1-H4 router KL",
        ]
        if training_config.selection_profile == "h1_h4_candidate16"
        else [
            "validation h2 request-macro SlotRecall@8",
            "validation mean h1-h8 request-macro SlotRecall@8",
            "negative validation mean router KL",
        ]
    )
    manifest = {
        "schema": TRAINING_SCHEMA,
        "model_name": "HARP-8T",
        "model_config": model_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "training_config": training_config.to_dict(),
        "frozen_loss_coefficients": coefficients,
        "gradient_calibration": calibration,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split_summary": {
            split: summary.__dict__
            for split, summary in data.split_summary().items()
        },
        "captured_mtp_depths": data.captured_mtp_depths,
        "configured_mtp_depths": model_config.mtp_depths,
        "missing_mtp_depths_zero_masked": list(
            range(data.captured_mtp_depths + 1, model_config.mtp_depths + 1)
        ),
        "best_epoch": best_epoch,
        "best_validation_h2_request_macro_recall_at_8": validation.h2_request_macro_recall,
        "best_validation_mean_h1_h4_request_macro_recall_at_8": validation.mean_h1_h4(8),
        "best_validation_mean_h1_h4_candidate_coverage_at_16": validation.mean_h1_h4(16),
        "validation_h2_bootstrap": bootstrap,
        "validation_h1_h4_bootstrap": h1_h4_bootstrap,
        "selection_order": selection_order,
        "split_manifest": (
            {
                "path": str(data.split_manifest_path),
                "sha256": sha256_file(data.split_manifest_path),
            }
            if data.split_manifest_path is not None
            else None
        ),
        "sealed_test_evaluated": False,
        "sealed_test_accessed_by_trainer": False,
        "single_model_validation_target_mean_h1_h4_recall_at_8": 0.90,
        "experimental_freeze_legacy": bool(freeze_legacy),
        "direct_feature_skip_paths": {
            "enabled": True,
            "route_history_lags": 3,
            "mtp_hidden_and_router_flattened": True,
            "target_state_skip": True,
            "within_request_position": True,
            "canonicalized_mtp_depth_order": True,
        },
        "candidate_generator_gate_mean_h1_h4_coverage_at_16": 0.95,
        "candidate_generator_gate_h4_coverage_at_16": 0.93,
        "initialized_from": (
            {
                "path": str(initialize_from),
                "sha256": sha256_file(initialize_from),
            }
            if initialize_from is not None
            else None
        ),
        "compatibility_profile_limitations": [
            "six captured MTP hidden/router nodes; depths 7-8 masked",
            "one PCA target-state channel rather than raw a and MoE delta",
            "one PCA MTP hidden channel; no fused/head-input/top64 vocabulary fields",
            "development corpus only; not the planned fresh 10,000-request corpus",
        ],
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (
                best_path,
                output_dir / "validation_metrics.csv",
                output_dir / "validation_request_metrics.csv",
                output_dir / "validation_layer_metrics.csv",
                output_dir / "validation_domain_metrics.csv",
                output_dir / "validation_h2_bootstrap.json",
                h1_h4_bootstrap_path,
            )
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _mirror([best_path, last_path, manifest_path], mirror_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mtp-dir", type=Path, required=True)
    parser.add_argument("--target-state-features", type=Path, required=True)
    parser.add_argument("--mtp-state-features", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--h2-only", action="store_true")
    parser.add_argument("--mtp-state-width", type=int, default=128)
    parser.add_argument(
        "--capacity-profile",
        choices=("baseline", "wide", "extra_wide"),
        default="baseline",
    )
    parser.add_argument(
        "--loss-profile",
        choices=("legacy", "membership", "candidate16", "combined"),
        default="legacy",
    )
    parser.add_argument(
        "--selection-profile",
        choices=("h2", "h1_h4_candidate16"),
        default="h2",
    )
    parser.add_argument(
        "--horizon-weights",
        help="comma-separated weights for H1 through H8",
    )
    parser.add_argument("--candidate-margin", type=float, default=0.0)
    parser.add_argument("--candidate-weight", type=float)
    parser.add_argument("--membership-weight", type=float)
    parser.add_argument("--hard-negative-end-rank", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--gradient-calibration-steps", type=int, default=32)
    parser.add_argument(
        "--freeze-legacy",
        action="store_true",
        help="freeze the validated pre-expansion backbone and train only zero-init direct skips; requires --initialize-from and calibration steps 0",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--reduced-pilot", action="store_true")
    parser.add_argument(
        "--condition",
        choices=("primary", "route_only", "mtp1", "residual_only", "rank128"),
        default="primary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_config = HARPConfig.compact_compatibility()
    model_config = HARPConfig(
        **{**model_config.to_dict(), "mtp_state_width": args.mtp_state_width}
    )
    capacity_overrides: dict[str, object] = {}
    if args.capacity_profile == "wide":
        capacity_overrides = {
            "mtp_state_projection_width": 512,
            "mtp_width": 512,
            "model_width": 512,
            "route_ffn_width": 1024,
            "mtp_ffn_width": 1536,
            "fusion_ffn_width": 1024,
            "attention_heads": 8,
        }
    elif args.capacity_profile == "extra_wide":
        capacity_overrides = {
            "route_width": 384,
            "state_width": 384,
            "mtp_state_projection_width": 768,
            "mtp_width": 768,
            "model_width": 768,
            "route_ffn_width": 1536,
            "mtp_ffn_width": 2048,
            "fusion_ffn_width": 1536,
            "attention_heads": 12,
        }
    if capacity_overrides:
        model_config = HARPConfig(
            **{**model_config.to_dict(), **capacity_overrides}
        )
    if args.reduced_pilot:
        model_config = HARPConfig(
            **{
                **model_config.to_dict(),
                "route_width": 128,
                "state_width": 128,
                "mtp_width": 192,
                "model_width": 192,
                "route_ffn_width": 384,
                "mtp_ffn_width": 512,
                "fusion_ffn_width": 384,
                "attention_heads": 8,
                "temporal_blocks": 1,
                "layer_blocks": 1,
                "state_blocks": 1,
                "mtp_cross_blocks": 1,
                "fusion_blocks": 1,
                "future_latent_width": 128,
            }
        )
    condition_overrides: dict[str, object] = {}
    if args.condition == "route_only":
        condition_overrides = {"use_target_state": False, "use_mtp": False}
    elif args.condition == "mtp1":
        condition_overrides = {
            "use_target_state": False,
            "use_mtp": True,
            "mtp_active_depths": 1,
        }
    elif args.condition == "residual_only":
        condition_overrides = {"use_target_state": True, "use_mtp": False}
    elif args.condition == "rank128":
        condition_overrides = {"dense_output": False, "output_rank": 128}
    if condition_overrides:
        model_config = HARPConfig(
            **{**model_config.to_dict(), **condition_overrides}
        )
    training_config = TrainingConfig(
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        seed=args.seed,
        h2_only=args.h2_only,
        selection_profile=args.selection_profile,
    )
    horizon_weights: tuple[float, ...] | None = None
    if args.horizon_weights:
        horizon_weights = tuple(
            float(value) for value in args.horizon_weights.split(",")
        )
        if len(horizon_weights) != model_config.horizons:
            raise ValueError("horizon weights must contain exactly eight values")
    loss_overrides: dict[str, object] = {
        "horizon_weights": horizon_weights,
        "temperature": args.temperature,
        "candidate_margin": args.candidate_margin,
        "hard_negative_end_rank": args.hard_negative_end_rank,
    }
    if args.loss_profile in ("membership", "combined"):
        loss_overrides["full_membership"] = 0.5
    if args.loss_profile in ("candidate16", "combined"):
        loss_overrides["candidate16"] = 1.0
    if args.candidate_weight is not None:
        loss_overrides["candidate16"] = args.candidate_weight
    if args.membership_weight is not None:
        loss_overrides["full_membership"] = args.membership_weight
    loss_config = LossConfig(**{**LossConfig().to_dict(), **loss_overrides})
    data = CompactHARPData(
        args.capture_dir,
        args.mtp_dir,
        args.target_state_features,
        args.mtp_state_features,
        model_config,
        split_manifest=args.split_manifest,
    )
    train(
        data,
        args.output_dir,
        model_config,
        loss_config,
        training_config,
        device=args.device,
        resume=args.resume,
        initialize_from=args.initialize_from,
        gradient_calibration_steps=args.gradient_calibration_steps,
        checkpoint_interval=args.checkpoint_interval,
        mirror_dir=args.mirror_dir,
        freeze_legacy=args.freeze_legacy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
