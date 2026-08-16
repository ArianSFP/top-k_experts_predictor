"""Strict loading of forty immutable ShadowRoute layer shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .route_quant import PackedRouteQuantExperts
from .shadow_backbone import InstalledShadowBackbone
from .shadow_expert import (
    ExactTop1PlusDraftExperts,
    IndexedShadowExperts,
    PackedInt4ResidentExperts,
    PackedInt4TopKExperts,
    RouteConditionedBasisExperts,
    SharedResidualExperts,
)


LOCAL_SCHEMA = "harp_shadowroute_local_expert_layer_v1"
CHECKPOINT_MODE = {
    "basisdraft_all8": "basisdraft_all8",
    "exact_top1_plus_draft": "s0_exact_top1_plus_draft",
    "shared_width128": "s1_shared",
    "shared_width512": "s1_shared_width512",
    "indexed_width16": "s2_indexed",
    "int4_top4": "int4_top4",
    "resident_int4_only": "resident_int4_only",
    "resident_int4_shared": "resident_int4_shared",
    "resident_int4_tail_control": "resident_tail_control_v2",
    "resident_int4_codebook": "resident_int4_codebook",
    "routequant_all8": "routequant_all8",
}


def discover_layer_checkpoints(root: str | Path, mode: str) -> tuple[Path, ...]:
    expected = CHECKPOINT_MODE.get(mode)
    if expected is None:
        raise ValueError("unknown installed ShadowRoute mode")
    root_path = Path(root)
    result = []
    for layer in range(40):
        matches = sorted(root_path.rglob(f"shadow_{expected}_layer_{layer:02d}.pt"))
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one {expected} checkpoint for layer {layer}, "
                f"found {len(matches)}"
            )
        result.append(matches[0])
    return tuple(result)


def _load_value(
    path: Path,
    *,
    expected_mode: str,
    expected_layer: int,
    source_commit: str,
    target_checkpoint_index_sha256: str,
    allow_unpromoted_diagnostic: bool = False,
) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != LOCAL_SCHEMA:
        raise ValueError(f"ShadowRoute layer checkpoint schema mismatch: {path}")
    if value.get("mode") != expected_mode or int(value.get("layer", -1)) != expected_layer:
        raise ValueError(f"ShadowRoute layer checkpoint identity mismatch: {path}")
    if value.get("source_commit") != source_commit:
        raise ValueError("ShadowRoute source lineage differs across layer shards")
    if value.get("target_checkpoint_index_sha256") != target_checkpoint_index_sha256:
        raise ValueError("ShadowRoute target checkpoint differs across layer shards")
    if value.get("formal_validation_opened") is not False:
        raise PermissionError("ShadowRoute layer checkpoint opened formal validation")
    if value.get("calibration_opened") is not False or value.get("sealed_test_opened") is not False:
        raise PermissionError("ShadowRoute layer checkpoint crossed a sealed split")
    if (
        value.get("closed_loop_authorized") is not True
        and not allow_unpromoted_diagnostic
    ):
        raise PermissionError("ShadowRoute layer shard did not pass its component gate")
    state = value.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("ShadowRoute layer checkpoint lacks model state")
    return value


def resident_ids_from_bundle(
    root: str | Path,
    *,
    source_commit: str,
    target_checkpoint_index_sha256: str,
    allow_unpromoted_diagnostic: bool = False,
    mode: str = "resident_int4_shared",
) -> tuple[torch.Tensor, ...]:
    """Read the frozen per-layer resident namespace before model construction."""

    if mode not in {
        "resident_int4_only", "resident_int4_shared", "resident_int4_codebook"
    }:
        raise ValueError("resident namespace mode is not deployable")
    paths = discover_layer_checkpoints(root, mode)
    result = []
    for layer, path in enumerate(paths):
        value = _load_value(
            path,
            expected_mode=CHECKPOINT_MODE[mode],
            expected_layer=layer,
            source_commit=source_commit,
            target_checkpoint_index_sha256=target_checkpoint_index_sha256,
            allow_unpromoted_diagnostic=allow_unpromoted_diagnostic,
        )
        ids = value.get("resident_expert_ids")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1:
            raise ValueError("resident hybrid shard lacks its expert namespace")
        if int(value.get("resident_count", -1)) != ids.numel():
            raise ValueError("resident hybrid count differs from its expert namespace")
        result.append(ids.long())
    return tuple(result)


def resident_codebooks_from_bundle(
    root: str | Path,
    *,
    source_commit: str,
    target_checkpoint_index_sha256: str,
    allow_unpromoted_diagnostic: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    """Read and validate the forty immutable functional-codebook tables."""

    paths = discover_layer_checkpoints(root, "resident_int4_codebook")
    result = []
    for layer, path in enumerate(paths):
        value = _load_value(
            path,
            expected_mode=CHECKPOINT_MODE["resident_int4_codebook"],
            expected_layer=layer,
            source_commit=source_commit,
            target_checkpoint_index_sha256=target_checkpoint_index_sha256,
            allow_unpromoted_diagnostic=allow_unpromoted_diagnostic,
        )
        state = value["model_state_dict"]
        proxy_ids = state.get("codebook_proxy_ids")
        proxy_coefficients = state.get("codebook_proxy_coefficients")
        proxy_count = state.get("codebook_proxy_count")
        if (
            not isinstance(proxy_ids, torch.Tensor)
            or proxy_ids.shape != (256, 2)
            or not isinstance(proxy_coefficients, torch.Tensor)
            or proxy_coefficients.shape != (256, 2)
            or not isinstance(proxy_count, torch.Tensor)
            or proxy_count.shape != (256,)
        ):
            raise ValueError("resident codebook shard lacks valid mapping tensors")
        result.append((
            proxy_ids.to(torch.int16),
            proxy_coefficients.to(torch.bfloat16),
            proxy_count.to(torch.uint8),
        ))
    return tuple(result)


def routequant_schedules_from_bundle(
    root: str | Path,
    *,
    source_commit: str,
    target_checkpoint_index_sha256: str,
    allow_unpromoted_diagnostic: bool = False,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], str]:
    """Read strict per-layer gate/up and down schedules before construction."""

    paths = discover_layer_checkpoints(root, "routequant_all8")
    gate_schedules = []
    down_schedules = []
    storage: str | None = None
    for layer, path in enumerate(paths):
        value = _load_value(
            path,
            expected_mode="routequant_all8",
            expected_layer=layer,
            source_commit=source_commit,
            target_checkpoint_index_sha256=target_checkpoint_index_sha256,
            allow_unpromoted_diagnostic=allow_unpromoted_diagnostic,
        )
        legacy_widths = value.get("bit_widths")
        gate_widths = value.get("gate_up_bit_widths", legacy_widths)
        down_widths = value.get("down_bit_widths", legacy_widths)
        current_storage = value.get("scale_storage")
        state = value["model_state_dict"]
        for name, widths in (("gate/up", gate_widths), ("down", down_widths)):
            if (
                not isinstance(widths, torch.Tensor)
                or widths.shape != (256,)
                or any(int(item) not in (1, 2, 3, 4) for item in widths.tolist())
            ):
                raise ValueError(f"RouteQuant shard has an invalid {name} schedule")
        if current_storage not in {"bf16", "log8"}:
            raise ValueError("RouteQuant shard has invalid scale storage")
        if storage is None:
            storage = str(current_storage)
        elif storage != current_storage:
            raise ValueError("RouteQuant scale storage differs across layers")
        state_gate = state.get("gate_up.bit_widths")
        state_down = state.get("down.bit_widths")
        if (
            not isinstance(state_gate, torch.Tensor)
            or not isinstance(state_down, torch.Tensor)
            or not torch.equal(state_gate, gate_widths)
            or not torch.equal(state_down, down_widths)
        ):
            raise ValueError("RouteQuant shard state disagrees with its schedules")
        gate_schedules.append(gate_widths.to(torch.int8))
        down_schedules.append(down_widths.to(torch.int8))
    assert storage is not None
    return tuple(gate_schedules), tuple(down_schedules), storage


def load_shadow_bundle(
    installed: InstalledShadowBackbone,
    root: str | Path,
    *,
    source_commit: str,
    target_checkpoint_index_sha256: str,
    s1_fallback_root: str | Path | None = None,
    allow_unpromoted_diagnostic: bool = False,
) -> tuple[Path, ...]:
    """Load all layer shards, rejecting mixed lineage or partial bundles."""

    expected_mode = CHECKPOINT_MODE[installed.mode]
    paths = discover_layer_checkpoints(root, installed.mode)
    fallback_paths = (
        discover_layer_checkpoints(s1_fallback_root, "shared_width128")
        if s1_fallback_root is not None else None
    )
    for layer, (module, path) in enumerate(
        zip(installed.shadow_experts, paths, strict=True)
    ):
        value = _load_value(
            path,
            expected_mode=expected_mode,
            expected_layer=layer,
            source_commit=source_commit,
            target_checkpoint_index_sha256=target_checkpoint_index_sha256,
            allow_unpromoted_diagnostic=allow_unpromoted_diagnostic,
        )
        state = value["model_state_dict"]
        if isinstance(module, ExactTop1PlusDraftExperts):
            module.draft_expert.load_state_dict(state, strict=True)
        elif isinstance(module, SharedResidualExperts):
            module.draft_expert.load_state_dict(state, strict=True)
        elif isinstance(module, RouteConditionedBasisExperts):
            expected = {
                "gate_up_proj", "down_proj", "expert_coefficients",
                "expert_gate_up_proj", "expert_down_proj",
            }
            if set(state) != expected:
                raise ValueError("BasisDraft layer shard state is incomplete")
            if (
                int(value.get("basis_count", -1)) != module.config.basis_count
                or int(value.get("basis_width", -1)) != module.config.basis_width
                or int(value.get("basis_expert_residual_width", -1))
                != module.config.expert_residual_width
            ):
                raise ValueError("BasisDraft runtime configuration differs from shard")
            module.load_state_dict(state, strict=True)
        elif isinstance(module, PackedRouteQuantExperts):
            expected = set(module.state_dict())
            if set(state) != expected:
                raise ValueError("RouteQuant layer shard state is incomplete")
            legacy_widths = value.get("bit_widths")
            gate_widths = value.get("gate_up_bit_widths", legacy_widths)
            down_widths = value.get("down_bit_widths", legacy_widths)
            if (
                not isinstance(gate_widths, torch.Tensor)
                or not isinstance(down_widths, torch.Tensor)
                or not torch.equal(
                    gate_widths.to(torch.int8), module.gate_up.bit_widths.cpu()
                )
                or not torch.equal(
                    down_widths.to(torch.int8), module.down.bit_widths.cpu()
                )
            ):
                raise ValueError("RouteQuant runtime projection schedules differ from shard")
            module.load_state_dict(state, strict=True)
        elif isinstance(module, PackedInt4ResidentExperts):
            expected = {
                "resident_ids", "expert_to_resident",
                "gate_up_packed", "gate_up_scales", "down_packed", "down_scales",
            }
            if installed.mode not in {
                "resident_int4_only", "resident_int4_codebook"
            }:
                expected |= {
                    "fallback.draft_expert.gate_up_proj.weight",
                    "fallback.draft_expert.down_proj.weight",
                }
            if installed.mode == "resident_int4_codebook":
                expected |= {
                    "codebook_proxy_ids",
                    "codebook_proxy_coefficients",
                    "codebook_proxy_count",
                }
            if installed.mode == "resident_int4_tail_control" and layer < 39:
                expected |= {
                    "router_control.expert_codes.weight",
                    "router_control.hidden_projection.weight",
                    "router_control.router_delta_projection.weight",
                }
            if set(state) != expected:
                raise ValueError("resident INT4 hybrid shard state is incomplete")
            ids = value.get("resident_expert_ids")
            if not isinstance(ids, torch.Tensor) or not torch.equal(
                ids.long(), module.resident_ids.cpu()
            ):
                raise ValueError("resident INT4 runtime namespace differs from shard")
            module.load_state_dict(state, strict=True)
        elif isinstance(module, IndexedShadowExperts):
            expected = {"gate_up_proj", "down_proj", "trained_experts"}
            if set(state) != expected:
                raise ValueError("indexed shadow shard state is incomplete")
            declared_width = int(value.get("shadow_width", state["down_proj"].shape[-1]))
            declared_slots = int(value.get("indexed_active_slots", 8))
            if (
                declared_width != module.config.shadow_width
                or declared_slots != module.config.routed_slots
            ):
                raise ValueError("indexed shadow runtime configuration differs from shard")
            with torch.no_grad():
                module.gate_up_proj.copy_(state["gate_up_proj"].to(module.gate_up_proj))
                module.down_proj.copy_(state["down_proj"].to(module.down_proj))
            counts = value.get("expert_counts")
            minimum = int(value.get("minimum_expert_count", -1))
            if not isinstance(counts, torch.Tensor) or minimum < 1:
                raise ValueError("indexed shadow shard lacks expert coverage contract")
            module.set_trained_counts(counts, minimum=minimum)
            if fallback_paths is None:
                if not bool(module.trained_experts.all()):
                    raise ValueError(
                        "indexed shadow bundle has untrained experts without S1 fallback"
                    )
            else:
                if module.fallback is None:
                    raise ValueError("indexed shadow module lacks its declared S1 fallback")
                fallback = _load_value(
                    fallback_paths[layer],
                    expected_mode="s1_shared",
                    expected_layer=layer,
                    source_commit=source_commit,
                    target_checkpoint_index_sha256=target_checkpoint_index_sha256,
                    allow_unpromoted_diagnostic=allow_unpromoted_diagnostic,
                )
                module.fallback.draft_expert.load_state_dict(
                    fallback["model_state_dict"], strict=True
                )
        elif isinstance(module, PackedInt4TopKExperts):
            expected = {
                "gate_up_packed",
                "gate_up_scales",
                "down_packed",
                "down_scales",
            }
            if set(state) != expected:
                raise ValueError("INT4 shadow shard state is incomplete")
            if (
                int(value.get("group_size", -1)) != module.group_size
                or int(value.get("active_slots", -1)) != module.active_slots
            ):
                raise ValueError("INT4 shadow runtime configuration differs from shard")
            module.load_state_dict(state, strict=True)
        else:  # pragma: no cover - installed bundle invariant
            raise TypeError("installed layer is not a ShadowRoute expert")
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return paths


__all__ = [
    "CHECKPOINT_MODE",
    "LOCAL_SCHEMA",
    "discover_layer_checkpoints",
    "load_shadow_bundle",
    "resident_codebooks_from_bundle",
    "resident_ids_from_bundle",
    "routequant_schedules_from_bundle",
]
