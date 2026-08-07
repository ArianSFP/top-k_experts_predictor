"""Staged, promotion-gated training primitives for HARP-RTT.

The module deliberately separates phase policy from the command-line driver.
It never opens a dataset split and it never infers that the sealed test split
may be used.  Its responsibilities are parameter ownership, optimizer groups,
gradient-accumulation state, the formal auxiliary-gradient gate, and safe
checkpoint persistence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from .losses import GradientAudit, GradientAuditResult, HARPRTTLossOutput


TRAINING_STATE_SCHEMA = "harp_rtt_training_state_v1"
PhaseName = Literal["phase2", "phase3", "phase4", "phase5"]


@dataclass(frozen=True)
class PhaseSpec:
    """Optimization policy for one formal HARP-RTT training phase."""

    name: PhaseName
    epochs: int
    new_learning_rate: float
    legacy_learning_rate: float = 0.0
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("phase epochs must be positive")
        for name in ("new_learning_rate", "legacy_learning_rate", "weight_decay"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.new_learning_rate <= 0:
            raise ValueError("new_learning_rate must be positive")
        if self.name == "phase3":
            if self.legacy_learning_rate <= 0:
                raise ValueError("Phase 3 requires a positive legacy learning rate")
            if self.legacy_learning_rate > self.new_learning_rate / 5.0:
                raise ValueError(
                    "Phase 3 legacy learning rate must be at least 5x below the new-module rate"
                )
        elif self.legacy_learning_rate != 0:
            raise ValueError("legacy_learning_rate is only valid for Phase 3")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must lie in [0,1)")


DEFAULT_PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec("phase2", epochs=12, new_learning_rate=2.0e-4),
    PhaseSpec(
        "phase3",
        epochs=8,
        new_learning_rate=1.0e-4,
        legacy_learning_rate=1.0e-5,
    ),
    PhaseSpec("phase4", epochs=12, new_learning_rate=2.0e-4),
    PhaseSpec("phase5", epochs=6, new_learning_rate=5.0e-5),
)


# Existing HARP components that the formal architecture permits Phase 3 to
# update.  Matching is component-boundary aware and applies below any bridge
# wrapper, so names such as ``anchor.anchor.layer_route_blocks.0...`` work.
LEGACY_UPPER_COMPONENTS: tuple[str, ...] = (
    "layer_route_blocks",
    "route_output_norm",
    "target_state_blocks",
    "target_state_norm",
    "route_to_model",
    "state_to_model",
    "cross_blocks",
    "source_value",
    "source_score",
    "fusion_blocks",
    "final_norm",
    "output_weight",
    "output_a",
    "output_b",
    "output_bias",
    "copy_query",
    "copy_scalar_bias",
    "copy_expert_bias",
    "future_latent_head",
)


# Phase 5 is intentionally narrower than Phase 3: candidate selection remains
# stop-gradient while only final generator query/score adapters and the ranker
# can move.
FINAL_GENERATOR_PREFIXES: tuple[str, ...] = (
    "score_head.",
    "branch_residual.",
    "decoder.short",
    "decoder.long",
)


@dataclass(frozen=True)
class ParameterGroupReport:
    phase: PhaseName
    new_names: tuple[str, ...]
    legacy_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    new_parameter_count: int
    legacy_parameter_count: int
    frozen_parameter_count: int

    @property
    def trainable_parameter_count(self) -> int:
        return self.new_parameter_count + self.legacy_parameter_count

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["trainable_parameter_count"] = self.trainable_parameter_count
        return result


def _is_anchor_parameter(name: str) -> bool:
    return name == "anchor" or name.startswith("anchor.")


def _is_token_embedding_parameter(name: str) -> bool:
    return name == "token_embedding" or name.startswith("token_embedding.")


def _has_component(name: str, component: str) -> bool:
    return component in name.split(".")


def _is_legacy_upper(name: str) -> bool:
    return _is_anchor_parameter(name) and any(
        _has_component(name, component) for component in LEGACY_UPPER_COMPONENTS
    )


def _is_final_generator_adapter(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in FINAL_GENERATOR_PREFIXES) or (
        name.startswith("trajectory.") and ".correction." in name
    )


def configure_training_phase(
    model: nn.Module,
    spec: PhaseSpec,
) -> tuple[list[dict[str, Any]], ParameterGroupReport]:
    """Set exact trainable ownership and return AdamW parameter groups.

    The loaded HARP anchor and frozen target embedding are never included in a
    new-module group.  Phase 3 exposes only the named upper legacy components.
    The wrapper's ``_anchor_frozen`` flag is kept consistent with ownership so
    its no-grad fast path cannot silently suppress Phase 3 gradients.
    """

    spec.validate()
    new: list[tuple[str, nn.Parameter]] = []
    legacy: list[tuple[str, nn.Parameter]] = []
    frozen: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        is_anchor = _is_anchor_parameter(name)
        is_embedding = _is_token_embedding_parameter(name)
        is_reranker = name == "reranker" or name.startswith("reranker.")
        train_new = False
        train_legacy = False
        if spec.name == "phase2":
            train_new = not is_anchor and not is_embedding and not is_reranker
        elif spec.name == "phase3":
            train_new = not is_anchor and not is_embedding and not is_reranker
            train_legacy = _is_legacy_upper(name)
        elif spec.name == "phase4":
            train_new = is_reranker
        elif spec.name == "phase5":
            train_new = is_reranker or _is_final_generator_adapter(name)
        else:  # pragma: no cover - Literal plus validate keeps this defensive.
            raise ValueError(f"unknown HARP-RTT phase {spec.name!r}")
        parameter.requires_grad_(train_new or train_legacy)
        if train_new:
            new.append((name, parameter))
        elif train_legacy:
            legacy.append((name, parameter))
        else:
            frozen.append((name, parameter))

    if not new:
        raise ValueError(f"{spec.name} selected no new-module parameters")
    if spec.name == "phase3" and not legacy:
        raise ValueError("Phase 3 selected no legacy upper parameters")
    if hasattr(model, "_anchor_frozen"):
        setattr(model, "_anchor_frozen", not bool(legacy))
    anchor = getattr(model, "anchor", None)
    if isinstance(anchor, nn.Module):
        anchor.eval()

    groups: list[dict[str, Any]] = [
        {
            "params": [parameter for _, parameter in new],
            "lr": spec.new_learning_rate,
            "weight_decay": spec.weight_decay,
            "group_name": "new",
        }
    ]
    if legacy:
        groups.append(
            {
                "params": [parameter for _, parameter in legacy],
                "lr": spec.legacy_learning_rate,
                "weight_decay": spec.weight_decay,
                "group_name": "legacy_upper",
            }
        )
    count = lambda values: sum(parameter.numel() for _, parameter in values)
    report = ParameterGroupReport(
        phase=spec.name,
        new_names=tuple(name for name, _ in new),
        legacy_names=tuple(name for name, _ in legacy),
        frozen_names=tuple(name for name, _ in frozen),
        new_parameter_count=count(new),
        legacy_parameter_count=count(legacy),
        frozen_parameter_count=count(frozen),
    )
    return groups, report


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed process-local generators without mutating split membership."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Frozen router geometry uses explicit FP32 islands inside the otherwise
    # BF16-autocast training graph.  TF32 would silently reduce those islands'
    # mantissa and can change a close top-8 boundary, so FP32 means IEEE FP32
    # throughout this accuracy-first experiment.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if deterministic:
        torch.use_deterministic_algorithms(True)


def move_to_device(value: Any, device: torch.device | str) -> Any:
    """Move nested tensors while preserving audit strings and role tuples."""

    if isinstance(value, Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        # Metadata lists contain request IDs and must stay on the host.  Nested
        # tensor lists are uncommon but supported without coercing strings.
        return [move_to_device(item, device) for item in value]
    return value


def autocast_context(device: torch.device | str, enabled: bool = True):
    resolved = torch.device(device)
    if enabled and resolved.type in {"cpu", "cuda"}:
        # CPU is a diagnostic/smoke backend, but it must exercise the same
        # learned-network dtype boundary as production CUDA.  Rich captures
        # legitimately contain BF16 activations while newly initialized HARP-
        # RTT parameters are stored in FP32; disabling CPU autocast makes the
        # first Linear see a BF16 input and FP32 weight instead of mirroring
        # the production BF16 policy.  Canonical router geometry and legacy
        # reconstruction remain FP32 through their explicit disabled islands.
        return torch.autocast(device_type=resolved.type, dtype=torch.bfloat16)
    return nullcontext()


@dataclass
class AuxiliaryGradientController:
    """Apply the formal <=0.5 shared-gradient constraint at audit steps."""

    scale: float = 1.0
    minimum_scale: float = 1.0e-4

    def effective_loss(self, losses: HARPRTTLossOutput) -> Tensor:
        return losses.primary + float(self.scale) * losses.auxiliary

    def audit(
        self,
        auditor: GradientAudit,
        *,
        step: int,
        losses: HARPRTTLossOutput,
        parameters: Iterable[Tensor],
    ) -> GradientAuditResult | None:
        auxiliaries = {
            name: value * float(self.scale)
            for name, value in losses.weighted_components.items()
            if name != "set"
        }
        result = auditor.audit(
            step=step,
            primary_loss=losses.weighted_components["set"],
            auxiliary_losses=auxiliaries,
            parameters=parameters,
        )
        if result is not None and not result.passed:
            self.scale = max(
                float(self.minimum_scale),
                float(self.scale) * float(result.recommended_auxiliary_scale),
            )
        return result


def cosine_warmup_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup followed by a non-negative cosine decay."""

    if total_steps < 1 or not 0 <= step <= total_steps:
        raise ValueError("scheduler step lies outside total_steps")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must lie in [0,total_steps)")
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class OptimizerStepState:
    global_step: int = 0
    micro_step: int = 0
    auxiliary_scale: float = 1.0
    best_selection_tuple: tuple[float, ...] | None = None
    best_checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TRAINING_STATE_SCHEMA,
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "auxiliary_scale": self.auxiliary_scale,
            "best_selection_tuple": list(self.best_selection_tuple)
            if self.best_selection_tuple is not None
            else None,
            "best_checkpoint": self.best_checkpoint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OptimizerStepState":
        if payload.get("schema") != TRAINING_STATE_SCHEMA:
            raise ValueError("training-state schema mismatch")
        selection = payload.get("best_selection_tuple")
        return cls(
            global_step=int(payload["global_step"]),
            micro_step=int(payload["micro_step"]),
            auxiliary_scale=float(payload.get("auxiliary_scale", 1.0)),
            best_selection_tuple=tuple(float(value) for value in selection)
            if selection is not None
            else None,
            best_checkpoint=payload.get("best_checkpoint"),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def save_checkpoint(
    output_directory: Path,
    *,
    tag: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: OptimizerStepState,
    phase: PhaseSpec,
    gradient_audit: GradientAudit,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically save tensor-only model/optimizer payloads plus hash manifest."""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - declared dependency.
        raise RuntimeError("safetensors is required for HARP-RTT checkpoints") from exc
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    base = output_directory / tag
    model_path = base.with_suffix(".model.safetensors")
    optimizer_path = base.with_suffix(".optimizer.pt")
    state_path = base.with_suffix(".state.json")
    manifest_path = base.with_suffix(".manifest.json")
    for path in (model_path, optimizer_path, state_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint artifact {path}")

    model_temporary = model_path.with_name(f".{model_path.name}.tmp")
    optimizer_temporary = optimizer_path.with_name(f".{optimizer_path.name}.tmp")
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if isinstance(value, Tensor)
    }
    save_file(tensors, str(model_temporary))
    os.replace(model_temporary, model_path)
    # Optimizer state is a tensor/scalar container and can be restored using
    # torch.load(weights_only=True); arbitrary pickle globals are not needed.
    torch.save(optimizer.state_dict(), optimizer_temporary)
    os.replace(optimizer_temporary, optimizer_path)
    state.auxiliary_scale = float(state.auxiliary_scale)
    state_payload = state.to_dict()
    state_payload.update(
        {
            "phase": asdict(phase),
            "gradient_audit": gradient_audit.state_dict(),
        }
    )
    _atomic_json(state_path, state_payload)
    manifest = {
        "schema": "harp_rtt_checkpoint_manifest_v1",
        "tag": tag,
        "phase": asdict(phase),
        "model": {"path": model_path.name, "sha256": sha256_file(model_path)},
        "optimizer": {
            "path": optimizer_path.name,
            "sha256": sha256_file(optimizer_path),
            "safe_load": "torch.load(weights_only=True)",
        },
        "state": {"path": state_path.name, "sha256": sha256_file(state_path)},
        "provenance": dict(provenance),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def load_checkpoint(
    manifest_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_audit: GradientAudit | None = None,
) -> tuple[OptimizerStepState, dict[str, Any]]:
    """Verify and restore a HARP-RTT checkpoint without unrestricted pickle."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for HARP-RTT checkpoints") from exc
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "harp_rtt_checkpoint_manifest_v1":
        raise ValueError("checkpoint manifest schema mismatch")
    root = manifest_path.parent
    resolved: dict[str, Path] = {}
    for key in ("model", "optimizer", "state"):
        entry = manifest[key]
        path = root / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{key} checkpoint hash mismatch")
        resolved[key] = path
    model.load_state_dict(load_file(str(resolved["model"])), strict=True)
    if optimizer is not None:
        payload = torch.load(resolved["optimizer"], map_location="cpu", weights_only=True)
        optimizer.load_state_dict(payload)
    state_payload = json.loads(resolved["state"].read_text(encoding="utf-8"))
    state = OptimizerStepState.from_dict(state_payload)
    if gradient_audit is not None:
        gradient_audit.load_state_dict(state_payload["gradient_audit"])
    return state, manifest


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one flushed metric event; partial final lines are detectable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "AuxiliaryGradientController",
    "DEFAULT_PHASES",
    "FINAL_GENERATOR_PREFIXES",
    "LEGACY_UPPER_COMPONENTS",
    "OptimizerStepState",
    "ParameterGroupReport",
    "PhaseName",
    "PhaseSpec",
    "TRAINING_STATE_SCHEMA",
    "append_jsonl",
    "autocast_context",
    "configure_training_phase",
    "cosine_warmup_multiplier",
    "load_checkpoint",
    "move_to_device",
    "save_checkpoint",
    "seed_everything",
    "sha256_file",
]
