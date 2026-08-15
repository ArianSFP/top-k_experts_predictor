"""Official-Qwen integration seam for HARP-ShadowRoute."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .shadow_checkpoint import (
    IndexedCheckpoint,
    SelectiveLoadReport,
    load_exact_nonexpert_parameters,
    validate_shadow_target_config,
)
from .shadow_expert import (
    BasisDraftConfig,
    ExactTop1PlusDraftExperts,
    IndexedShadowExperts,
    PackedInt4TopKExperts,
    RouteConditionedBasisExperts,
    ShadowExpertConfig,
    SharedResidualExperts,
    SwiGLUDraftExpert,
)


SHADOW_MODES = (
    "basisdraft_all8",
    "exact_top1_plus_draft",
    "shared_width128",
    "shared_width512",
    "indexed_width16",
    "int4_top4",
)


def resolve_text_model(model: nn.Module) -> nn.Module:
    for path in (("model", "language_model"), ("language_model",), ("model",)):
        current: Any = model
        try:
            for part in path:
                current = getattr(current, part)
        except AttributeError:
            continue
        if hasattr(current, "layers"):
            return current
    if hasattr(model, "layers"):
        return model
    raise ValueError("could not locate Qwen text model")


def resolve_text_layers(model: nn.Module) -> Sequence[nn.Module]:
    layers = getattr(resolve_text_model(model), "layers")
    if len(layers) != 40:
        raise ValueError("ShadowRoute requires exactly 40 target layers")
    return layers


@dataclass(frozen=True)
class InstalledShadowBackbone:
    model: nn.Module
    mode: str
    native_experts: tuple[nn.Module | None, ...]
    shadow_experts: tuple[nn.Module, ...]


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


def install_shadow_experts(
    model: nn.Module,
    mode: str,
    *,
    retain_native: bool = False,
    shadow_width: int = 16,
    exact_slots: int = 1,
    draft_scale: float = 1.0,
    indexed_active_slots: int = 8,
    basis_count: int = 16,
    basis_width: int = 32,
) -> InstalledShadowBackbone:
    """Replace only routed experts in an already-materialized official model."""

    if mode not in SHADOW_MODES:
        raise ValueError(f"unknown ShadowRoute mode {mode!r}")
    validate_shadow_target_config(resolve_text_model(model).config)
    native: list[nn.Module | None] = []
    installed: list[nn.Module] = []
    for layer in resolve_text_layers(model):
        original = layer.mlp.experts
        device, dtype = _module_device_dtype(original)
        if mode == "basisdraft_all8":
            replacement = RouteConditionedBasisExperts(BasisDraftConfig(
                basis_count=basis_count,
                basis_width=basis_width,
            )).to(device=device, dtype=dtype)
            native.append(original if retain_native else None)
        elif mode == "exact_top1_plus_draft":
            draft = SwiGLUDraftExpert(2048, 512).to(device=device, dtype=dtype)
            replacement = ExactTop1PlusDraftExperts(
                original,
                draft,
                experts=256,
                exact_slots=exact_slots,
                draft_scale=draft_scale,
            )
            native.append(original)
        elif mode in {"shared_width128", "shared_width512"}:
            width = 128 if mode == "shared_width128" else 512
            draft = SwiGLUDraftExpert(2048, width).to(device=device, dtype=dtype)
            replacement = SharedResidualExperts(draft, experts=256)
            native.append(original if retain_native else None)
        elif mode == "indexed_width16":
            fallback = SharedResidualExperts(
                SwiGLUDraftExpert(2048, 128), experts=256
            )
            replacement = IndexedShadowExperts(
                ShadowExpertConfig(
                    shadow_width=shadow_width,
                    active_slots=indexed_active_slots,
                ),
                fallback=fallback,
            ).to(device=device, dtype=dtype)
            native.append(original if retain_native else None)
        else:
            replacement = PackedInt4TopKExperts(
                active_slots=indexed_active_slots,
                device=device,
            )
            native.append(original if retain_native else None)
        layer.mlp.experts = replacement
        installed.append(replacement)
    for parameter in resolve_text_model(model).parameters():
        parameter.requires_grad_(False)
    for module in installed:
        if isinstance(module, ExactTop1PlusDraftExperts):
            module.draft_expert.requires_grad_(True)
            if module.native_experts is not None:
                module.native_experts.requires_grad_(False)
        elif isinstance(module, (
            SharedResidualExperts, IndexedShadowExperts,
            RouteConditionedBasisExperts,
        )):
            module.requires_grad_(True)
    return InstalledShadowBackbone(
        model=model,
        mode=mode,
        native_experts=tuple(native),
        shadow_experts=tuple(installed),
    )


def build_selective_shadow_text_model(
    model_path: str,
    *,
    mode: str = "shared_width128",
    device: str | torch.device = "cuda",
    shadow_width: int = 16,
    basis_count: int = 16,
    basis_width: int = 32,
) -> tuple[nn.Module, SelectiveLoadReport]:
    """Build the exact non-expert text stack without native expert allocation."""

    if mode not in {
        "basisdraft_all8", "shared_width128", "shared_width512",
        "indexed_width16",
    }:
        raise ValueError("selective deployment supports only resident shadow modes")
    try:
        from transformers import AutoConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTextModel,
        )
    except ImportError as exc:
        raise RuntimeError("ShadowRoute target loading requires transformers>=5.14.1") from exc
    full = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config = full.text_config
    validate_shadow_target_config(config)
    with torch.device("meta"):
        text_model = Qwen3_5MoeTextModel(config)
    for layer in text_model.layers:
        if mode == "basisdraft_all8":
            replacement = RouteConditionedBasisExperts(BasisDraftConfig(
                basis_count=basis_count,
                basis_width=basis_width,
            ))
        elif mode in {"shared_width128", "shared_width512"}:
            width = 128 if mode == "shared_width128" else 512
            replacement: nn.Module = SharedResidualExperts(
                SwiGLUDraftExpert(2048, width), experts=256
            )
        else:
            replacement = IndexedShadowExperts(
                ShadowExpertConfig(shadow_width=shadow_width),
                fallback=SharedResidualExperts(
                    SwiGLUDraftExpert(2048, 128), experts=256
                ),
            )
        layer.mlp.experts = replacement.to(device=device, dtype=torch.bfloat16)
    checkpoint = IndexedCheckpoint(model_path)
    report = load_exact_nonexpert_parameters(text_model, checkpoint, device=device)
    return text_model, report


def freeze_except_shadow(model: nn.Module) -> tuple[str, ...]:
    model.requires_grad_(False)
    for layer in resolve_text_layers(model):
        experts = layer.mlp.experts
        if isinstance(experts, ExactTop1PlusDraftExperts):
            experts.draft_expert.requires_grad_(True)
        elif isinstance(experts, (
            SharedResidualExperts, IndexedShadowExperts,
            RouteConditionedBasisExperts,
        )):
            experts.requires_grad_(True)
        else:
            raise TypeError("target layer does not contain a ShadowRoute expert module")
    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("no shadow expert parameters were selected")
    return tuple(trainable)


@contextmanager
def exact_prefix_experts(installed: InstalledShadowBackbone):
    """Temporarily restore native routed experts for committed-prefix replay."""

    layers = resolve_text_layers(installed.model)
    if any(module is None for module in installed.native_experts):
        raise RuntimeError("exact prefix replay requires retained native experts")
    shadow = [layer.mlp.experts for layer in layers]
    try:
        for layer, native in zip(layers, installed.native_experts, strict=True):
            assert native is not None
            layer.mlp.experts = native
        yield installed.model
    finally:
        for layer, module in zip(layers, shadow, strict=True):
            layer.mlp.experts = module


__all__ = [
    "InstalledShadowBackbone",
    "SHADOW_MODES",
    "build_selective_shadow_text_model",
    "exact_prefix_experts",
    "freeze_except_shadow",
    "install_shadow_experts",
    "resolve_text_layers",
    "resolve_text_model",
]
