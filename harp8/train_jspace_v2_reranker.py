"""Train the experimental layer-aware J-HARP-C64 v2 ranker.

The entry point intentionally reuses the v1 immutable-artifact, split,
precision-audit, coverage, and validation infrastructure.  It changes only
the aligned model inputs and ranker architecture.  Scientific execution is
validation-selected, cannot open a test pool, requires context-enabled frozen
HARP pools, trains H1--H4, and exposes H5--H8 only as exact frozen-score
passthrough.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .jspace_data import (
    move_tensor_batch,
    ordered_prefetched_batches,
    slice_tensor_batch,
)
from .jspace_loss_profiles import (
    JSPACE_LOSS_PROFILE_NAMES,
    PREREGISTERED_V1,
    assert_resume_loss_profile,
    provenance_with_loss_profile,
    resolve_jspace_loss_profile,
    validate_jspace_loss_profile,
)
from .jspace_metrics import evaluate_jspace_ranker
from .jspace_reranker import JSpaceRerankerLossConfig
from .jspace_v2_data import (
    AlignedJContextCandidateData,
    assert_no_label_only_inputs,
)
from .jspace_v2_reranker import (
    CompositeJSpaceV2Inference,
    JSpaceV2CandidateReranker,
    JSpaceV2RerankerConfig,
    slice_v2_horizon_batch,
    v2_reranker_loss,
)
from .train import canonical_json, write_csv
from .train_jspace_reranker import (
    DECISION_PROFILE,
    JSpaceTrainingConfig,
    _assert_matching_data,
    _atomic_torch_save,
    _autocast,
    _coverage_gate,
    _hash_file,
    _input_provenance,
    _load_router_keys,
    _optimizer_groups,
    _read_pool_manifest,
    _restore_rng,
    _rng_state,
    _scheduler,
    _selection,
    _validate_scientific_lineage,
    _write_evaluation,
    candidate_feature_width,
)


JSPACE_V2_TRAINING_SCHEMA = "harp8_jspace_v2_reranker_training_v1"
JSPACE_V2_TRAINING_MANIFEST_SCHEMA = "harp8_jspace_v2_reranker_manifest_v1"
JSPACE_V2_ARCHITECTURE_NAME = "J-HARP-C64-v2-layer-aware"


class _RowPrefixView:
    def __init__(self, base: AlignedJContextCandidateData, maximum: int | None) -> None:
        self.base = base
        self.rows = base.rows if maximum is None else min(base.rows, int(maximum))
        if self.rows <= 0:
            raise ValueError("a v2 data view must contain at least one row")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def batch(
        self,
        rows: np.ndarray,
        device: str | torch.device,
        *,
        active_horizons: int | None = None,
        include_context: bool = True,
        compact: bool = False,
    ) -> dict[str, torch.Tensor]:
        values = np.asarray(rows, dtype=np.int64)
        if values.ndim != 1 or (values < 0).any() or (values >= self.rows).any():
            raise IndexError("v2 view rows are out of range")
        return self.base.batch(
            values,
            device,
            active_horizons=active_horizons,
            include_context=include_context,
            compact=compact,
        )

    def sequential_batches(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.rows, batch_size):
            yield np.arange(start, min(self.rows, start + batch_size), dtype=np.int64)


def prepare_v2_model_batch(
    batch: Mapping[str, torch.Tensor],
    *,
    include_candidate_features: bool,
    active_horizons: int = 4,
) -> dict[str, torch.Tensor]:
    """Create the label-safe v2 model view at the active horizon prefix."""

    assert_no_label_only_inputs(batch)
    result = slice_v2_horizon_batch(batch, active_horizons)
    required = {
        "candidate_scores",
        "candidate_ids",
        "j_states",
        "j_mask",
        "mtp_states",
        "mtp_router_logits",
        "mtp_mask",
        "generator_context",
    }
    missing = required - result.keys()
    if missing:
        raise KeyError(f"v2 model view is missing: {sorted(missing)}")
    if include_candidate_features:
        if "candidate_features" not in result:
            raise KeyError("v2 candidate features were enabled but are absent")
        result["candidate_features"] = result["candidate_features"].float()
    else:
        result.pop("candidate_features", None)
    result["j_states"] = result["j_states"].float()
    result["j_mask"] = result["j_mask"].bool()
    result["mtp_mask"] = result["mtp_mask"].bool()
    result["generator_context"] = result["generator_context"].float()
    return result


def derive_v2_model_config(
    data: AlignedJContextCandidateData,
    router_keys: torch.Tensor,
    *,
    include_candidate_features: bool = True,
    allow_router_rank_ablation: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> JSpaceV2RerankerConfig:
    """Derive every input geometry and enforce the v2 scientific gates."""

    layers, experts, key_width = map(int, router_keys.shape)
    if (layers, experts) != (data.pool.layers, data.pool.experts):
        raise ValueError("router keys disagree with candidate-pool geometry")
    if key_width != experts and not allow_router_rank_ablation:
        raise ValueError(
            "scientific v2 runs require full router rank R=E; use the explicit "
            "ablation override only for a labelled representation ablation"
        )
    if data.pool.horizons != 8:
        raise ValueError("v2 requires an H1-H8 frozen candidate pool")
    values: dict[str, Any] = {
        "experts": experts,
        "layers": layers,
        "horizons": 4,
        "pool_horizons": data.pool.horizons,
        "candidate_count": data.pool.candidate_count,
        "native_k": int(data.pool.manifest.get("native_k", 8)),
        "j_lags": data.history,
        "j_width": data.target_width,
        "generator_context_width": data.generator_context_width,
        "mtp_nodes": data.mtp_depths,
        "mtp_state_channels": 1,
        "mtp_state_width": data.mtp_width,
        "router_key_width": key_width,
        "candidate_feature_width": (
            candidate_feature_width(data.history) if include_candidate_features else 0
        ),
    }
    if overrides:
        values.update(dict(overrides))
    config = JSpaceV2RerankerConfig(**values)
    config.validate()
    return config


def _assert_v2_matching_data(
    train: AlignedJContextCandidateData,
    validation: AlignedJContextCandidateData,
) -> None:
    _assert_matching_data(train, validation)
    if train.generator_context_width != validation.generator_context_width:
        raise ValueError("train/validation generator-context widths disagree")
    if not bool(train.pool.manifest.get("store_context")) or not bool(
        validation.pool.manifest.get("store_context")
    ):
        raise ValueError("both v2 pools must store frozen generator context")


def _checkpoint_payload(
    *,
    model: JSpaceV2CandidateReranker,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    rng: np.random.Generator | None,
    model_config: JSpaceV2RerankerConfig,
    loss_config: JSpaceRerankerLossConfig,
    loss_profile: str,
    training_config: JSpaceTrainingConfig,
    input_provenance: Mapping[str, Any],
    completed_epoch: int,
    global_step: int,
    best_selection: tuple[float, float],
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": JSPACE_V2_TRAINING_SCHEMA,
        "architecture": JSPACE_V2_ARCHITECTURE_NAME,
        "model_config": model_config.to_dict(),
        "loss_profile": loss_profile,
        "loss_config": loss_config.to_dict(),
        "training_config": training_config.to_dict(),
        "input_provenance": dict(input_provenance),
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "global_step": global_step,
        "best_selection": best_selection,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "inference_contract": {
            "pool_horizons": 8,
            "active_horizons": 4,
            "generator_context_required": True,
            "layer_aware_mtp_queries": True,
            "mtp_hidden_router_separated_until_fusion": True,
            "layer_specific_router_coordinate_heads": True,
            "inactive_horizons_are_exact_base_passthrough": True,
            "acceptance_labels_used": False,
            "decision_profile": DECISION_PROFILE,
            "mtp_timing_causal": False,
        },
    }
    if optimizer is not None and scheduler is not None and rng is not None:
        payload.update(
            {
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "rng_state": _rng_state(rng),
            }
        )
    return payload


def load_composite_jspace_v2_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> CompositeJSpaceV2Inference:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != JSPACE_V2_TRAINING_SCHEMA:
        raise ValueError("checkpoint has an incompatible J-HARP v2 schema")
    config = JSpaceV2RerankerConfig(**payload["model_config"])
    config.validate()
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or not isinstance(
        state.get("router_keys"), torch.Tensor
    ):
        raise ValueError("v2 checkpoint does not contain frozen router keys")
    model = JSpaceV2CandidateReranker(config, state["router_keys"].float())
    model.load_state_dict(state, strict=True)
    return CompositeJSpaceV2Inference(model.to(device).eval())


class _EvaluationAdapter(nn.Module):
    def __init__(self, model: JSpaceV2CandidateReranker) -> None:
        super().__init__()
        self.composite = CompositeJSpaceV2Inference(model)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prepared = dict(batch)
        assert_no_label_only_inputs(prepared)
        return self.composite(prepared)


def _evaluate(
    model: JSpaceV2CandidateReranker,
    data: _RowPrefixView,
    *,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    return evaluate_jspace_ranker(
        _EvaluationAdapter(model),
        data,  # type: ignore[arg-type]
        batch_size=batch_size,
        device=device,
        autocast=str(device).startswith("cuda"),
    )


def _resume_state_matches(
    payload: Mapping[str, Any],
    *,
    model_config: JSpaceV2RerankerConfig,
    loss_config: JSpaceRerankerLossConfig,
    loss_profile: str,
    training_config: JSpaceTrainingConfig,
    provenance: Mapping[str, Any],
) -> None:
    if payload.get("schema") != JSPACE_V2_TRAINING_SCHEMA:
        raise ValueError("resume checkpoint has an incompatible v2 schema")
    assert_resume_loss_profile(
        payload,
        expected_profile=loss_profile,
        expected_config=loss_config,
        horizons=model_config.horizons,
    )
    saved_provenance = provenance_with_loss_profile(
        payload.get("input_provenance", {}),
        loss_profile=loss_profile,
        loss_config=loss_config,
    )
    expected = {
        "model_config": model_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "training_config": training_config.to_dict(),
        "input_provenance": dict(provenance),
    }
    for name, value in expected.items():
        actual = saved_provenance if name == "input_provenance" else payload.get(name)
        if canonical_json(actual) != canonical_json(value):
            raise ValueError(f"resume checkpoint {name} differs from this run")


def train_jspace_v2_reranker(
    train_data: AlignedJContextCandidateData,
    validation_data: AlignedJContextCandidateData,
    output_dir: Path,
    router_keys: torch.Tensor,
    model_config: JSpaceV2RerankerConfig,
    loss_config: JSpaceRerankerLossConfig,
    training_config: JSpaceTrainingConfig,
    *,
    target_features: Path,
    target_feature_rms: Path | None,
    router_provenance: Mapping[str, Any],
    device: str = "cuda:0",
    resume: Path | None = None,
    stop_after_epoch: int | None = None,
    strict_lineage: bool = True,
    precision_audit: Path | None = None,
    allow_provisional_j_without_probe: bool = False,
    allow_router_rank_ablation: bool = False,
    ordered_group_prefetch: bool = False,
    loss_profile: str | None = None,
) -> dict[str, Any]:
    """Train v2 with validation-only selection and full resumable state."""

    training_config.validate()
    model_config.validate()
    loss_config.validate(model_config.horizons)
    resolved_loss_profile = validate_jspace_loss_profile(
        loss_config,
        model_config.horizons,
        loss_profile,
    )
    if training_config.active_horizons != 4 or model_config.horizons != 4:
        raise ValueError("v2 training is gated to H1-H4")
    if training_config.decision_profile != DECISION_PROFILE:
        raise ValueError("v2 requires the audited informational decision profile")
    if training_config.mtp_timing_causal or training_config.acceptance_labels_used:
        raise ValueError(
            "v2 cannot claim timing-causal MTP or consume acceptance labels"
        )
    if (
        model_config.router_key_width != model_config.experts
        and not allow_router_rank_ablation
    ):
        raise ValueError("v2 scientific training requires full router rank R=E")
    _assert_v2_matching_data(train_data, validation_data)
    _read_pool_manifest(
        train_data.pool.root,
        "train",
        allow_legacy_level2_train=not strict_lineage,
    )
    _read_pool_manifest(validation_data.pool.root, "validation")
    train_native_k = int(train_data.pool.manifest.get("native_k", 8))
    validation_native_k = int(validation_data.pool.manifest.get("native_k", 8))
    if train_native_k != validation_native_k or train_native_k != model_config.native_k:
        raise ValueError("v2 model and pools disagree on native_k")
    if strict_lineage and train_native_k != 8:
        raise ValueError("the current v2 scientific trace contract is native top-8")
    if tuple(router_keys.shape) != (
        model_config.layers,
        model_config.experts,
        model_config.router_key_width,
    ):
        raise ValueError("router keys disagree with v2 model geometry")

    if strict_lineage:
        semantic = _validate_scientific_lineage(
            train_data,
            validation_data,
            target_features=target_features,
            target_feature_rms=target_feature_rms,
            precision_audit=precision_audit,
            allow_provisional_j_without_probe=allow_provisional_j_without_probe,
        )
    else:
        if precision_audit is not None or allow_provisional_j_without_probe:
            raise ValueError(
                "precision claims are unavailable in non-strict debug mode"
            )
        semantic = {
            "strict": False,
            "reason": "explicit_non_scientific_fixture_or_debug_mode",
            "precision_attribution": {"required": False, "attribution": "unverified"},
        }
    semantic.update(
        {
            "architecture": JSPACE_V2_ARCHITECTURE_NAME,
            "generator_context_required": True,
            "layer_aware_mtp_queries": True,
            "mtp_sources_separated": True,
            "layer_specific_router_coordinate_heads": True,
            "decision_profile": DECISION_PROFILE,
            "mtp_timing_causal": False,
            "acceptance_labels_used": False,
        }
    )
    coverage = _coverage_gate(validation_data)
    if not coverage["passes"]:
        raise RuntimeError("v2 candidate coverage gate failed")

    output_dir = Path(output_dir)
    if resume is None:
        if output_dir.exists():
            raise FileExistsError(f"refusing to reuse v2 output directory {output_dir}")
        output_dir.mkdir(parents=True)
    elif not output_dir.is_dir():
        raise FileNotFoundError("v2 resume output directory does not exist")

    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA v2 training requested but unavailable")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    provenance = _input_provenance(
        train_data,
        validation_data,
        target_features=target_features,
        target_feature_rms=target_feature_rms,
        router_provenance=router_provenance,
        semantic_lineage=semantic,
    )
    provenance["execution_contract"] = {
        "architecture": JSPACE_V2_ARCHITECTURE_NAME,
        "pool_horizons": 8,
        "active_horizons": 4,
        "generator_context_required": True,
        "generator_context_train_manifest_declared": True,
        "generator_context_validation_manifest_declared": True,
        "layer_aware_mtp_queries": True,
        "mtp_hidden_router_separated_until_fusion": True,
        "layer_specific_router_coordinate_heads": True,
        "inactive_horizons_are_exact_base_passthrough": True,
        "router_rank_ablation": model_config.router_key_width != model_config.experts,
        "decision_profile": DECISION_PROFILE,
        "mtp_timing_causal": False,
        "acceptance_labels_used": False,
    }
    if ordered_group_prefetch:
        provenance["execution_contract"]["io_pipeline"] = {
            "schema": "harp8_ordered_group_prefetch_v1",
            "opt_in": True,
            "include_generator_context": True,
            "rows_per_materialization": (
                training_config.microbatch_size * training_config.gradient_accumulation
            ),
            "cpu_read_ahead_groups": 1,
            "compact_model_only_transfer": True,
            "pinned_nonblocking_h2d_on_cuda": True,
        }
    provenance = provenance_with_loss_profile(
        provenance,
        loss_profile=resolved_loss_profile,
        loss_config=loss_config,
    )

    train_view = _RowPrefixView(train_data, training_config.max_train_rows)
    validation_view = _RowPrefixView(
        validation_data, training_config.max_validation_rows
    )
    model = JSpaceV2CandidateReranker(model_config, router_keys).to(device)
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, training_config.weight_decay),
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
        eps=training_config.epsilon,
        fused=str(device).startswith("cuda"),
    )
    updates_per_epoch = math.ceil(
        math.ceil(train_view.rows / training_config.microbatch_size)
        / training_config.gradient_accumulation
    )
    scheduler = _scheduler(
        optimizer,
        updates_per_epoch * training_config.epochs,
        training_config.warmup_fraction,
        training_config.minimum_learning_rate / training_config.learning_rate,
    )

    assertion_rows = np.arange(
        min(validation_view.rows, training_config.evaluation_batch_size),
        dtype=np.int64,
    )
    assertion_batch = prepare_v2_model_batch(
        validation_view.batch(
            assertion_rows,
            device,
            active_horizons=4,
            include_context=True,
        ),
        include_candidate_features=model_config.candidate_feature_width > 0,
    )
    model.eval()
    with torch.inference_mode(), _autocast(device):
        epoch0 = model(assertion_batch)
    if not torch.equal(
        epoch0.scores.float(), assertion_batch["candidate_scores"].float()
    ):
        raise AssertionError("epoch-zero v2 scores differ from frozen base scores")
    if int(torch.count_nonzero(epoch0.delta)):
        raise AssertionError("epoch-zero v2 residual is nonzero")

    history: list[dict[str, Any]] = []
    completed_epoch = 0
    global_step = 0
    stale_epochs = 0
    best_selection = (-math.inf, -math.inf)
    best_epoch = 0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        _resume_state_matches(
            state,
            model_config=model_config,
            loss_config=loss_config,
            loss_profile=resolved_loss_profile,
            training_config=training_config,
            provenance=provenance,
        )
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        _restore_rng(state["rng_state"], rng)
        completed_epoch = int(state["completed_epoch"])
        global_step = int(state["global_step"])
        stale_epochs = int(state["stale_epochs"])
        best_selection = tuple(float(v) for v in state["best_selection"])
        best_epoch = int(state["best_epoch"])
        history = list(state["history"])

    for epoch in range(completed_epoch + 1, training_config.epochs + 1):
        model.train()
        order = rng.permutation(train_view.rows)
        running: defaultdict[str, float] = defaultdict(float)
        microbatches = 0
        optimizer_steps = 0
        group_rows = (
            training_config.microbatch_size * training_config.gradient_accumulation
        )
        row_groups = (
            order[start : start + group_rows]
            for start in range(0, train_view.rows, group_rows)
        )
        cuda_io = bool(
            ordered_group_prefetch
            and str(device).startswith("cuda")
            and torch.cuda.is_available()
        )
        if ordered_group_prefetch:
            materialized = ordered_prefetched_batches(
                train_view,
                row_groups,
                active_horizons=4,
                include_context=True,
                compact=True,
                prefetch=cuda_io,
                pin_memory=cuda_io,
            )
        else:
            materialized = ((group, None) for group in row_groups)
        for group, host_batch in materialized:
            group_batch = (
                move_tensor_batch(host_batch, device, non_blocking=cuda_io)
                if host_batch is not None
                else None
            )
            micro_count = math.ceil(len(group) / training_config.microbatch_size)
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(group), training_config.microbatch_size):
                stop = min(len(group), start + training_config.microbatch_size)
                raw = (
                    train_view.batch(
                        group[start:stop],
                        device,
                        active_horizons=4,
                        include_context=True,
                    )
                    if group_batch is None
                    else slice_tensor_batch(group_batch, start, stop)
                )
                batch = prepare_v2_model_batch(
                    raw,
                    include_candidate_features=model_config.candidate_feature_width > 0,
                )
                with _autocast(device):
                    output = model(batch)
                    loss = v2_reranker_loss(output, batch, loss_config)
                (loss.total / micro_count).backward()
                microbatches += 1
                for name, value in loss.metrics.items():
                    running[name] += value
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            optimizer_steps += 1
            running["gradient_norm"] += float(gradient_norm)

        validation = _evaluate(
            model,
            validation_view,
            batch_size=training_config.evaluation_batch_size,
            device=device,
        )
        selection = _selection(validation)
        record: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_gradient_norm": running["gradient_norm"] / max(1, optimizer_steps),
            "validation_mean_h1_h4_request_macro_recall_at_8": selection[0],
            "validation_min_h1_h4_request_macro_recall_at_8": selection[1],
            **{
                f"train_{name}": value / max(1, microbatches)
                for name, value in running.items()
                if name != "gradient_norm"
            },
        }
        for row in validation["horizon_metrics"]:
            record[f"validation_recall_h{int(row['horizon'])}"] = float(
                row["request_macro_recall_at_8"]
            )
        history.append(record)
        print(canonical_json({"event": "harp8_jspace_v2_epoch", **record}), flush=True)

        if selection > best_selection:
            best_selection = selection
            best_epoch = epoch
            stale_epochs = 0
            _atomic_torch_save(
                _checkpoint_payload(
                    model=model,
                    optimizer=None,
                    scheduler=None,
                    rng=None,
                    model_config=model_config,
                    loss_config=loss_config,
                    loss_profile=resolved_loss_profile,
                    training_config=training_config,
                    input_provenance=provenance,
                    completed_epoch=epoch,
                    global_step=global_step,
                    best_selection=best_selection,
                    best_epoch=best_epoch,
                    stale_epochs=stale_epochs,
                    history=history,
                ),
                best_path,
            )
        else:
            stale_epochs += 1
        completed_epoch = epoch
        _atomic_torch_save(
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                rng=rng,
                model_config=model_config,
                loss_config=loss_config,
                loss_profile=resolved_loss_profile,
                training_config=training_config,
                input_provenance=provenance,
                completed_epoch=completed_epoch,
                global_step=global_step,
                best_selection=best_selection,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                history=history,
            ),
            last_path,
        )
        write_csv(output_dir / "training_history.csv", history)
        (output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            status = {
                "schema": JSPACE_V2_TRAINING_MANIFEST_SCHEMA,
                "status": "paused",
                "completed_epoch": completed_epoch,
                "next_epoch": completed_epoch + 1,
                "resume": str(last_path),
                "best_epoch": best_epoch,
                "best_selection": list(best_selection),
                "sealed_test_accessed": False,
                "loss_profile": resolved_loss_profile,
                "resolved_loss_config": loss_config.to_dict(),
            }
            (output_dir / "run_status.json").write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return status
        if (
            epoch >= training_config.minimum_epochs
            and stale_epochs >= training_config.patience
        ):
            break

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    model.to(device)
    validation = _evaluate(
        model,
        validation_view,
        batch_size=training_config.evaluation_batch_size,
        device=device,
    )
    bootstrap = _write_evaluation(
        output_dir,
        validation,
        bootstrap_replicates=training_config.bootstrap_replicates,
        seed=training_config.seed,
    )
    output_paths = [
        best_path,
        last_path,
        output_dir / "training_history.csv",
        output_dir / "training_history.json",
        output_dir / "validation_horizon_metrics.csv",
        output_dir / "validation_request_metrics.csv",
        output_dir / "validation_layer_metrics.csv",
        output_dir / "validation_domain_metrics.csv",
        output_dir / "validation_position_metrics.csv",
        output_dir / "validation_metrics.json",
        output_dir / "validation_h1_h4_paired_bootstrap.json",
    ]
    manifest = {
        "schema": JSPACE_V2_TRAINING_MANIFEST_SCHEMA,
        "model_name": JSPACE_V2_ARCHITECTURE_NAME,
        "model_config": model_config.to_dict(),
        "loss_profile": resolved_loss_profile,
        "loss_config": loss_config.to_dict(),
        "training_config": training_config.to_dict(),
        "parameters_trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "router_keys_trainable": False,
        "candidate_coverage_gate": coverage,
        "epoch0_base_score_assertion": {
            "rows": len(assertion_rows),
            "exact_score_equality": True,
            "nonzero_delta_values": 0,
        },
        "best_epoch": best_epoch,
        "best_validation_mean_h1_h4_recall_at_8": float(
            validation["mean_h1_h4_recall_at_8"]
        ),
        "paired_request_bootstrap": bootstrap,
        "input_provenance": provenance,
        "sealed_test_accessed": False,
        "target_mean_h1_h4_recall_at_8": 0.90,
        "composite_output_contract": {
            "pool_horizons": 8,
            "reranked_horizons": [1, 2, 3, 4],
            "frozen_harp_passthrough_horizons": [5, 6, 7, 8],
            "passthrough_is_exact_base_score": True,
        },
        "outputs": {
            path.name: _hash_file(path) for path in output_paths if path.exists()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enable-experimental-v2",
        action="store_true",
        help="required acknowledgement that this is the unmatched v2 architecture",
    )
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mtp-dir", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--target-feature-rms", type=Path)
    parser.add_argument("--precision-audit", type=Path)
    parser.add_argument("--allow-provisional-j-without-probe", action="store_true")
    parser.add_argument("--router-keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows-per-request", type=int, default=34)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--ordered-group-prefetch", action="store_true")
    parser.add_argument("--no-candidate-features", action="store_true")
    parser.add_argument("--allow-router-rank-ablation", action="store_true")
    parser.add_argument("--model-width", type=int, default=384)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--feedforward-width", type=int, default=1536)
    parser.add_argument("--expert-embedding-width", type=int, default=128)
    parser.add_argument("--router-query-rank", type=int, default=64)
    parser.add_argument("--temporal-blocks", type=int, default=2)
    parser.add_argument("--axial-blocks", type=int, default=4)
    parser.add_argument("--mtp-hidden-blocks", type=int, default=2)
    parser.add_argument("--mtp-router-blocks", type=int, default=2)
    parser.add_argument("--set-blocks", type=int, default=2)
    parser.add_argument("--inducing-points", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--decision-profile",
        choices=(DECISION_PROFILE,),
        default=DECISION_PROFILE,
    )
    parser.add_argument(
        "--loss-profile",
        choices=JSPACE_LOSS_PROFILE_NAMES,
        default=PREREGISTERED_V1,
        help="named immutable training objective",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.enable_experimental_v2:
        raise SystemExit(
            "pass --enable-experimental-v2 to run the unmatched v2 architecture"
        )
    _read_pool_manifest(args.train_pool, "train")
    _read_pool_manifest(args.validation_pool, "validation")
    router_keys, router_provenance = _load_router_keys(args.router_keys)
    data_args = {
        "capture_dir": args.capture_dir,
        "mtp_dir": args.mtp_dir,
        "target_features": args.target_features,
        "target_feature_rms": args.target_feature_rms,
        "rows_per_request": args.rows_per_request,
        "history": args.history,
    }
    train = AlignedJContextCandidateData(args.train_pool, **data_args)
    validation = AlignedJContextCandidateData(args.validation_pool, **data_args)
    model_config = derive_v2_model_config(
        train,
        router_keys,
        include_candidate_features=not args.no_candidate_features,
        allow_router_rank_ablation=args.allow_router_rank_ablation,
        overrides={
            "model_width": args.model_width,
            "attention_heads": args.attention_heads,
            "feedforward_width": args.feedforward_width,
            "expert_embedding_width": args.expert_embedding_width,
            "router_query_rank": args.router_query_rank,
            "temporal_blocks": args.temporal_blocks,
            "axial_blocks": args.axial_blocks,
            "mtp_hidden_blocks": args.mtp_hidden_blocks,
            "mtp_router_blocks": args.mtp_router_blocks,
            "set_blocks": args.set_blocks,
            "inducing_points": args.inducing_points,
            "dropout": args.dropout,
        },
    )
    loss_config = resolve_jspace_loss_profile(args.loss_profile, 4)
    training_config = JSpaceTrainingConfig(
        epochs=args.epochs,
        minimum_epochs=args.minimum_epochs,
        patience=args.patience,
        microbatch_size=args.microbatch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        warmup_fraction=args.warmup_fraction,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        active_horizons=4,
        decision_profile=args.decision_profile,
        mtp_timing_causal=False,
        acceptance_labels_used=False,
    )
    train_jspace_v2_reranker(
        train,
        validation,
        args.output_dir,
        router_keys,
        model_config,
        loss_config,
        training_config,
        target_features=args.target_features,
        target_feature_rms=args.target_feature_rms,
        router_provenance=router_provenance,
        device=args.device,
        resume=args.resume,
        stop_after_epoch=args.stop_after_epoch,
        strict_lineage=True,
        precision_audit=args.precision_audit,
        allow_provisional_j_without_probe=args.allow_provisional_j_without_probe,
        allow_router_rank_ablation=args.allow_router_rank_ablation,
        ordered_group_prefetch=args.ordered_group_prefetch,
        loss_profile=args.loss_profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JSPACE_V2_ARCHITECTURE_NAME",
    "JSPACE_V2_TRAINING_MANIFEST_SCHEMA",
    "JSPACE_V2_TRAINING_SCHEMA",
    "build_parser",
    "derive_v2_model_config",
    "load_composite_jspace_v2_checkpoint",
    "prepare_v2_model_batch",
    "train_jspace_v2_reranker",
]
