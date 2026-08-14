"""Strict loading of forty immutable ShadowRoute layer shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .shadow_backbone import InstalledShadowBackbone
from .shadow_expert import (
    dequantize_groupwise_int4,
    ExactTop1PlusDraftExperts,
    IndexedShadowExperts,
    PackedInt4TopKExperts,
    SharedResidualExperts,
)


LOCAL_SCHEMA = "harp_shadowroute_local_expert_layer_v1"
CHECKPOINT_MODE = {
    "exact_top1_plus_draft": "s0_exact_top1_plus_draft",
    "shared_width128": "s1_shared",
    "indexed_width16": "s2_indexed",
    "int4_top4": "int4_top4",
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


def load_int4_bundle_into_native(
    installed: InstalledShadowBackbone,
    root: str | Path,
    *,
    source_commit: str,
    target_checkpoint_index_sha256: str,
    active_slots: int,
    allow_unpromoted_diagnostic: bool = False,
    expert_chunk: int = 2,
) -> tuple[Path, ...]:
    """Materialize the identical INT4 weights in the official BF16 kernel.

    This is an evaluation accelerator, not a different checkpoint.  Deployment
    retains packed INT4 weights and a fused sparse kernel; broad offline audits
    can dequantize once and avoid repeating that operation at every tree node.
    """

    if installed.mode != "exact_top1_plus_draft":
        raise ValueError("INT4 native materialization requires the exact wrapper")
    if not 1 <= active_slots <= 8 or expert_chunk < 1:
        raise ValueError("INT4 native materialization configuration is invalid")
    paths = discover_layer_checkpoints(root, "int4_top4")
    for layer, (module, path) in enumerate(
        zip(installed.shadow_experts, paths, strict=True)
    ):
        if not isinstance(module, ExactTop1PlusDraftExperts):
            raise TypeError("INT4 native materialization wrapper changed")
        if module.native_experts is None:
            raise ValueError("INT4 native materialization lacks official experts")
        if module.exact_slots != active_slots or module.draft_scale != 0.0:
            raise ValueError("INT4 native materialization execution contract changed")
        value = _load_value(
            path,
            expected_mode="int4_top4",
            expected_layer=layer,
            source_commit=source_commit,
            target_checkpoint_index_sha256=target_checkpoint_index_sha256,
            allow_unpromoted_diagnostic=allow_unpromoted_diagnostic,
        )
        state = value["model_state_dict"]
        expected = {
            "gate_up_packed", "gate_up_scales", "down_packed", "down_scales"
        }
        if set(state) != expected:
            raise ValueError("INT4 native materialization shard is incomplete")
        group_size = int(value.get("group_size", -1))
        if group_size not in {32, 64} or int(value.get("active_slots", -1)) != active_slots:
            raise ValueError("INT4 native materialization runtime differs from shard")
        native = module.native_experts
        gate_target = getattr(native, "gate_up_proj", None)
        down_target = getattr(native, "down_proj", None)
        if not isinstance(gate_target, torch.Tensor) or not isinstance(
            down_target, torch.Tensor
        ):
            raise TypeError("official expert tensor layout changed")
        experts = gate_target.shape[0]
        with torch.no_grad():
            for start in range(0, experts, expert_chunk):
                stop = min(start + expert_chunk, experts)
                gate = dequantize_groupwise_int4(
                    state["gate_up_packed"][start:stop].to(gate_target.device),
                    state["gate_up_scales"][start:stop].to(gate_target.device),
                    group_size=group_size,
                    dtype=gate_target.dtype,
                )
                down = dequantize_groupwise_int4(
                    state["down_packed"][start:stop].to(down_target.device),
                    state["down_scales"][start:stop].to(down_target.device),
                    group_size=group_size,
                    dtype=down_target.dtype,
                )
                gate_target[start:stop].copy_(gate)
                down_target[start:stop].copy_(down)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return paths


__all__ = [
    "CHECKPOINT_MODE",
    "LOCAL_SCHEMA",
    "discover_layer_checkpoints",
    "load_int4_bundle_into_native",
    "load_shadow_bundle",
]
