"""Strict loading of forty immutable ShadowRoute layer shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .shadow_backbone import InstalledShadowBackbone
from .shadow_expert import (
    ExactTop1PlusDraftExperts,
    IndexedShadowExperts,
    SharedResidualExperts,
)


LOCAL_SCHEMA = "harp_shadowroute_local_expert_layer_v1"
CHECKPOINT_MODE = {
    "exact_top1_plus_draft": "s0_exact_top1_plus_draft",
    "shared_width128": "s1_shared",
    "indexed_width16": "s2_indexed",
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
    if value.get("closed_loop_authorized") is not True:
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
            with torch.no_grad():
                module.gate_up_proj.copy_(state["gate_up_proj"].to(module.gate_up_proj))
                module.down_proj.copy_(state["down_proj"].to(module.down_proj))
            counts = value.get("expert_counts")
            minimum = int(value.get("minimum_expert_count", -1))
            if not isinstance(counts, torch.Tensor) or minimum < 1:
                raise ValueError("indexed shadow shard lacks expert coverage contract")
            module.set_trained_counts(counts, minimum=minimum)
            if fallback_paths is None or module.fallback is None:
                raise ValueError("indexed shadow bundle requires a complete S1 fallback")
            fallback = _load_value(
                fallback_paths[layer],
                expected_mode="s1_shared",
                expected_layer=layer,
                source_commit=source_commit,
                target_checkpoint_index_sha256=target_checkpoint_index_sha256,
            )
            module.fallback.draft_expert.load_state_dict(
                fallback["model_state_dict"], strict=True
            )
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
]
