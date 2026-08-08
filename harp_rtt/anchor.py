"""Exact compatibility bridge from rich captures to the validated HARP anchor.

The temperature-1 HARP checkpoint was trained on two rank-128 PCA channels:
per-layer post-MoE target residuals and depth-indexed native-MTP head inputs.
Both original training-only preprocessing bases were retained.  This module
applies those same affine transforms to the richer event capture, constructs
the legacy HARP input contract, and keeps the anchor frozen in evaluation
mode.  No realized future or acceptance label is read.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from harp8.config import HARPConfig
from harp8.model import HARP8Teacher


HARP_ANCHOR_BRIDGE_SCHEMA = "harp_rtt_legacy_harp_anchor_bridge_v2"
HARP_LEGACY_MAX_SOURCE_POSITION = 32.0
_CHECKPOINT_NUMPY_GLOBALS = {
    "numpy.dtype",
    "numpy.ndarray",
    "numpy._core.multiarray._reconstruct",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_checkpoint(path: Path) -> dict[str, Any]:
    """Load the historical training checkpoint without unrestricted pickle."""

    declared = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    unexpected = declared - _CHECKPOINT_NUMPY_GLOBALS
    if unexpected:
        raise ValueError(
            "HARP checkpoint declares unexpected pickle globals: "
            + ", ".join(sorted(unexpected))
        )
    # Historical checkpoints contain NumPy RNG state in addition to tensors.
    # Only the exact NumPy container/dtype classes needed for that inert state
    # are admitted; weights_only remains enabled throughout.
    safe_types = [
        np.dtype,
        np.ndarray,
        np._core.multiarray._reconstruct,
        np.dtypes.Int64DType,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
        np.dtypes.BoolDType,
    ]
    with torch.serialization.safe_globals(safe_types):
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, dict):
        raise TypeError("HARP checkpoint must be a mapping")
    return value


def load_harp_anchor_checkpoint(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[HARP8Teacher, dict[str, Any]]:
    """Restore the frozen HARP model with strict config/state validation."""

    path = Path(path)
    actual_hash = sha256_file(path)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise ValueError(
            f"HARP checkpoint SHA-256 mismatch: {actual_hash} != {expected_sha256}"
        )
    checkpoint = _safe_checkpoint(path)
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise KeyError("HARP checkpoint lacks model_config/model_state")
    config_payload = checkpoint["model_config"]
    state = checkpoint["model_state"]
    if not isinstance(config_payload, Mapping) or not isinstance(state, Mapping):
        raise TypeError("HARP model config and state must be mappings")
    config = HARPConfig(**dict(config_payload))
    config.validate()
    anchor = HARP8Teacher(config)
    anchor.load_state_dict(dict(state), strict=True)
    anchor.requires_grad_(False).eval()
    provenance = {
        "schema": HARP_ANCHOR_BRIDGE_SCHEMA,
        "checkpoint": str(path),
        "checkpoint_sha256": actual_hash,
        "checkpoint_schema": checkpoint.get("schema"),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "model_config": config.to_dict(),
    }
    return anchor, provenance


def _tensor_preprocessing(path: Path) -> dict[str, Tensor]:
    value = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"preprocessing artifact is not a mapping: {path}")
    return {
        name: tensor for name, tensor in value.items() if isinstance(tensor, Tensor)
    }


class LegacyHARPAnchorBridge(nn.Module):
    """Frozen HARP anchor that accepts a collated rich-dataset batch."""

    def __init__(
        self,
        anchor: HARP8Teacher,
        *,
        target_means: Tensor,
        target_components: Tensor,
        mtp_means: Tensor,
        mtp_components: Tensor,
    ) -> None:
        super().__init__()
        anchor.config.validate()
        self.anchor = anchor.requires_grad_(False).eval()
        self.config = anchor.config
        target_means = target_means.float()
        target_components = target_components.float()
        mtp_means = mtp_means.float()
        mtp_components = mtp_components.float()
        expected_target = (
            self.config.layers,
            target_components.shape[1],
        )
        if target_means.shape != expected_target:
            raise ValueError(
                f"target means have shape {tuple(target_means.shape)}, expected {expected_target}"
            )
        if target_components.shape != (
            self.config.layers,
            target_components.shape[1],
            self.config.target_state_width,
        ):
            raise ValueError("target PCA components disagree with HARP config")
        if mtp_means.ndim != 2 or mtp_components.ndim != 3:
            raise ValueError(
                "MTP preprocessing must be [D,H] means and [D,H,R] components"
            )
        if mtp_means.shape != mtp_components.shape[:2]:
            raise ValueError("MTP mean/component hidden widths disagree")
        if mtp_components.shape[-1] != self.config.mtp_state_width:
            raise ValueError("MTP PCA components disagree with HARP config")
        if mtp_means.shape[0] > self.config.mtp_depths:
            raise ValueError("MTP preprocessing has more depths than the HARP anchor")
        self.register_buffer("target_means", target_means.contiguous())
        self.register_buffer("target_components", target_components.contiguous())
        self.register_buffer("mtp_means", mtp_means.contiguous())
        self.register_buffer("mtp_components", mtp_components.contiguous())

    @classmethod
    def from_artifacts(
        cls,
        checkpoint: Path,
        target_preprocessing: Path,
        mtp_preprocessing: Path,
        *,
        expected_checkpoint_sha256: str | None = None,
    ) -> tuple["LegacyHARPAnchorBridge", dict[str, Any]]:
        anchor, provenance = load_harp_anchor_checkpoint(
            checkpoint, expected_sha256=expected_checkpoint_sha256
        )
        target = _tensor_preprocessing(target_preprocessing)
        mtp = _tensor_preprocessing(mtp_preprocessing)
        required_target = {"local_means", "local_components"}
        required_mtp = {"mtp_mean", "mtp_components"}
        if not required_target <= target.keys():
            raise KeyError(
                f"target preprocessing lacks {sorted(required_target - target.keys())}"
            )
        if not required_mtp <= mtp.keys():
            raise KeyError(
                f"MTP preprocessing lacks {sorted(required_mtp - mtp.keys())}"
            )
        bridge = cls(
            anchor,
            target_means=target["local_means"],
            target_components=target["local_components"],
            mtp_means=mtp["mtp_mean"],
            mtp_components=mtp["mtp_components"],
        )
        provenance.update(
            {
                "target_preprocessing": str(target_preprocessing),
                "target_preprocessing_sha256": sha256_file(target_preprocessing),
                "mtp_preprocessing": str(mtp_preprocessing),
                "mtp_preprocessing_sha256": sha256_file(mtp_preprocessing),
                "target_source_role": "post_moe_residual_xplus",
                "mtp_source_role": "mtp_vocabulary_head_input",
                "within_request_policy": (
                    "causal_clamp_to_legacy_training_support_[0,32]"
                ),
                "labels_read": False,
            }
        )
        return bridge, provenance

    def train(self, mode: bool = True) -> "LegacyHARPAnchorBridge":
        super().train(mode)
        self.anchor.eval()
        return self

    @staticmethod
    def _rich_inputs(batch: Mapping[str, Any]) -> Mapping[str, Any]:
        inputs = batch.get("inputs", batch)
        if not isinstance(inputs, Mapping):
            raise TypeError("rich HARP batch inputs must be a mapping")
        return inputs

    def legacy_inputs(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        """Construct only the causal tensor arguments of :class:`HARP8Teacher`."""

        inputs = self._rich_inputs(batch)
        history = inputs["history"]
        current = inputs["current"]
        tree = inputs["tree"]
        if not all(isinstance(value, Mapping) for value in (history, current, tree)):
            raise TypeError("history/current/tree must be mappings")
        route = history["logits"].float()
        available_grid = history["available"].bool()
        if route.ndim != 4 or route.shape[1:3] != (
            self.config.route_history,
            self.config.layers,
        ):
            raise ValueError("rich history logits must be [B,T,L,E]")
        if available_grid.shape != route.shape[:3]:
            raise ValueError("rich history availability must be [B,T,L]")
        route_history = route.permute(0, 2, 1, 3).contiguous()
        route_available = available_grid.all(dim=-1)

        target_raw = current["post_moe_residual_xplus"].float()
        if target_raw.shape[1:] != self.target_means.shape:
            raise ValueError("current post-MoE residual geometry disagrees with PCA")
        # These frozen PCA coordinates are part of the incumbent HARP input
        # contract.  The bridge is called inside the HARP-RTT BF16 autocast
        # scope, but recreating legacy inputs must remain bitwise FP32.
        with torch.autocast(device_type=target_raw.device.type, enabled=False):
            target_features = torch.einsum(
                "blh,lhr->blr",
                target_raw.float() - self.target_means.float()[None],
                self.target_components.float(),
            )
        target_states = target_features.unsqueeze(2)
        target_available = torch.ones(
            target_states.shape[:3], dtype=torch.bool, device=target_states.device
        )

        captured_depths = int(self.mtp_means.shape[0])
        adaptive_value = tree.get("adaptive_contract")
        adaptive_contract = (
            adaptive_value.bool().reshape(-1)
            if isinstance(adaptive_value, Tensor)
            else torch.zeros(route.shape[0], dtype=torch.bool, device=route.device)
        )
        if adaptive_contract.numel() == 1 and route.shape[0] != 1:
            adaptive_contract = adaptive_contract.expand(route.shape[0])
        if adaptive_contract.shape != (route.shape[0],):
            raise ValueError(
                "adaptive tree contract marker must be one value per batch row"
            )
        if adaptive_contract.any() and not adaptive_contract.all():
            raise ValueError("mixed adaptive/legacy anchor batches are unsupported")

        if adaptive_contract.all():
            anchor_inputs = batch.get("anchor_inputs")
            if not isinstance(anchor_inputs, Mapping):
                raise KeyError(
                    "adaptive batch lacks the separate anchor_inputs channel"
                )
            spine = anchor_inputs.get("mtp_spine")
            if not isinstance(spine, Mapping):
                raise KeyError("adaptive batch lacks its explicit legacy anchor spine")
            forbidden = [
                str(key)
                for key in spine
                if "accept" in str(key).lower()
                or "label" in str(key).lower()
                and str(key) != "labels_present"
                or "future" in str(key).lower()
            ]
            if forbidden:
                raise ValueError(
                    f"anchor spine carries forbidden label/future fields: {forbidden}"
                )
            contract = spine.get("contract")
            raw_hidden = spine.get("hidden_states")
            raw_router = spine.get("router_logits")
            depth = spine.get("depth")
            parent = spine.get("parent")
            mask = spine.get("mask")
            if not all(
                isinstance(value, Tensor)
                for value in (contract, raw_hidden, raw_router, depth, parent, mask)
            ):
                raise TypeError("anchor spine tensors/contract are incomplete")
            batch_size = route.shape[0]
            expected_grid = (batch_size, captured_depths)
            if (
                raw_hidden.shape != (*expected_grid, self.mtp_means.shape[1])
                or raw_router.shape != (*expected_grid, self.config.experts)
                or depth.shape != expected_grid
                or parent.shape != expected_grid
                or mask.shape != expected_grid
                or contract.bool().reshape(-1).shape != (batch_size,)
            ):
                raise ValueError(
                    "anchor spine geometry disagrees with pinned preprocessing"
                )
            if not contract.bool().all():
                raise ValueError(
                    "anchor spine contract is not valid for every batch row"
                )
            if not mask.bool().all():
                raise ValueError("anchor spine mask must contain every pinned depth")
            expected_depth = torch.arange(
                1, captured_depths + 1, dtype=torch.int64, device=depth.device
            )[None].expand(batch_size, -1)
            expected_parent = torch.arange(
                -1, captured_depths - 1, dtype=torch.int64, device=parent.device
            )[None].expand(batch_size, -1)
            if not torch.equal(depth.long(), expected_depth):
                raise ValueError(
                    "anchor spine depths are not exact H1 through pinned depth"
                )
            if not torch.equal(parent.long(), expected_parent):
                raise ValueError("anchor spine parent chain is incoherent")
            labels_present = spine.get("labels_present")
            if isinstance(labels_present, Tensor) and labels_present.bool().any():
                raise ValueError("anchor spine must not carry labels")
            token_ids = spine.get("token_ids")
            exact_token = inputs.get("exact_next_token_id")
            if isinstance(token_ids, Tensor) and isinstance(exact_token, Tensor):
                if not torch.equal(
                    token_ids[:, 0].long(), exact_token.reshape(-1).long()
                ):
                    raise ValueError(
                        "anchor spine root is not the exact committed H1 token"
                    )
            raw_mtp = raw_hidden.float()
            selected_router_tensor = raw_router.float()
            captured_available = mask.bool()
        else:
            # Historical greedy-chain path: preserve its exact anonymous-scalar
            # selection behavior for already-built legacy indices.
            states = tree["states"]
            router = tree["router_logits"].float()
            meta = tree["meta"].long()
            mask = tree["mask"].bool()
            scalars = tree["scalars"].float()
            if states.ndim != 4 or states.shape[2] < 4:
                raise ValueError("collated rich tree states must be [B,N,4,H]")
            batch_size, nodes = mask.shape
            if meta.shape[:2] != (batch_size, nodes) or meta.shape[-1] < 5:
                raise ValueError("rich tree metadata geometry is invalid")
            if router.shape[:2] != (batch_size, nodes):
                raise ValueError("rich tree router geometry is invalid")
            depth = meta[..., 4]
            path_probability = scalars[..., 2]
            selected_hidden: list[Tensor] = []
            selected_router: list[Tensor] = []
            selected_available: list[Tensor] = []
            for depth_id in range(1, captured_depths + 1):
                candidates = mask & (depth == depth_id)
                priority = path_probability.masked_fill(~candidates, -torch.inf)
                chosen = priority.argmax(dim=-1)
                exists = candidates.any(dim=-1)
                gather_state = chosen[:, None, None, None].expand(
                    -1, 1, states.shape[2], states.shape[3]
                )
                chosen_states = states.gather(1, gather_state).squeeze(1)
                gather_router = chosen[:, None, None].expand(-1, 1, router.shape[-1])
                chosen_router = router.gather(1, gather_router).squeeze(1)
                selected_hidden.append(chosen_states[:, 3] * exists[:, None])
                selected_router.append(chosen_router * exists[:, None])
                selected_available.append(exists)
            raw_mtp = torch.stack(selected_hidden, dim=1).float()
            captured_available = torch.stack(selected_available, dim=1)
            selected_router_tensor = torch.stack(selected_router, dim=1)
        with torch.autocast(device_type=raw_mtp.device.type, enabled=False):
            mtp_features = torch.einsum(
                "bdh,dhq->bdq",
                raw_mtp.float() - self.mtp_means.float()[None],
                self.mtp_components.float(),
            )
        mtp_features = mtp_features * captured_available[..., None]

        configured = self.config.mtp_depths
        mtp_states = torch.zeros(
            (batch_size, configured, 1, self.config.mtp_state_width),
            dtype=mtp_features.dtype,
            device=mtp_features.device,
        )
        mtp_router = torch.zeros(
            (batch_size, configured, self.config.experts),
            dtype=selected_router_tensor.dtype,
            device=selected_router_tensor.device,
        )
        mtp_available = torch.zeros(
            (batch_size, configured), dtype=torch.bool, device=captured_available.device
        )
        mtp_states[:, :captured_depths, 0] = mtp_features
        mtp_router[:, :captured_depths] = selected_router_tensor
        mtp_available[:, :captured_depths] = captured_available
        depth_ids = torch.arange(
            1, configured + 1, dtype=torch.int64, device=captured_available.device
        )[None].expand(batch_size, -1)
        fraction = depth_ids.float() / float(configured)
        metadata = torch.stack(
            (
                fraction,
                fraction.square(),
                torch.ones_like(fraction),
                mtp_available.float(),
            ),
            dim=-1,
        )
        within = inputs["within_request"].float().reshape(batch_size)
        if not torch.isfinite(within).all() or (within < 0).any():
            raise ValueError(
                "within-request source positions must be finite and non-negative"
            )
        # The frozen HARP anchor was trained on 34 rows/request, of which only
        # source rows 0--32 have t+1 labels. Its learned position projection
        # divides this scalar by 33 and includes a squared term; passing later
        # adaptive-capture positions through unchanged is severe, unsupported
        # extrapolation. Clamp only the compatibility channel while the v2
        # model retains the raw causal position in its own inputs.
        within = within.clamp_max(HARP_LEGACY_MAX_SOURCE_POSITION)
        return {
            "route_history": route_history,
            "route_available": route_available,
            "target_states": target_states,
            "target_state_available": target_available,
            "mtp_states": mtp_states,
            "mtp_router_logits": mtp_router,
            "mtp_metadata": metadata,
            "mtp_depth_ids": depth_ids,
            "mtp_available": mtp_available,
            "within_request": within,
        }

    def forward(self, *, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        # The enclosing HARPRTTTeacher owns the freeze/no-grad policy.  The
        # bridge starts with every legacy parameter frozen, but Phase 3 may
        # selectively re-enable upper HARP parameters at a lower learning
        # rate.  Keeping a second no-grad boundary here would silently make
        # that unfreezing ineffective.
        return self.anchor(**self.legacy_inputs(batch))


__all__ = [
    "HARP_ANCHOR_BRIDGE_SCHEMA",
    "HARP_LEGACY_MAX_SOURCE_POSITION",
    "LegacyHARPAnchorBridge",
    "load_harp_anchor_checkpoint",
    "sha256_file",
]
