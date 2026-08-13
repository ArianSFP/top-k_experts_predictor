"""Training-only access to already-indexed factual future target states."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import HarpRTTDataset


FUTURE_STATE_ROLES = (
    "post_attention_residual_u",
    "post_moe_residual_xplus",
    "routed_expert_output_delta_r",
    "shared_expert_output_delta_s",
)


class FutureTargetStateAdapter(Dataset[dict[str, Any]]):
    """Expose existing full states only under ``targets.future_states``."""

    def __init__(self, base: Dataset[dict[str, Any]]) -> None:
        self.base = base
        current: Any = base
        while not isinstance(current, HarpRTTDataset):
            current = getattr(current, "base", None)
            if current is None:
                raise TypeError("future-state adapter cannot locate HarpRTTDataset")
        self.source: HarpRTTDataset = current
        self.records: dict[tuple[str, int], Any] = {}
        for record in self.source.records:
            segment = self.source.segments[record.segment]
            request = str(segment.sequences[record.sequence]["request_id"])
            key = (request, int(record.position))
            if key in self.records:
                raise ValueError("duplicate future-state dataset join key")
            self.records[key] = record

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        metadata = item.get("metadata")
        targets = item.get("targets")
        inputs = item.get("inputs")
        if not isinstance(metadata, Mapping) or not isinstance(targets, Mapping) or not isinstance(inputs, Mapping):
            raise TypeError("future-state base item lacks metadata/inputs/targets")
        if "future_states" in inputs:
            raise PermissionError("future target states leaked into model inputs")
        key = (str(metadata["request_id"]), int(metadata["position"]))
        record = self.records.get(key)
        if record is None:
            raise KeyError(f"future-state source lacks record {key}")
        segment = self.source.segments[record.segment]
        future = [
            segment.read_layer_token("target", row, FUTURE_STATE_ROLES)
            for row in record.future_rows
        ]
        state_targets: dict[str, Tensor] = {
            role: torch.stack([row[role] for row in future])
            for role in FUTURE_STATE_ROLES
        }
        result = dict(item)
        result_targets = dict(targets)
        result_targets["future_states"] = state_targets
        result["targets"] = result_targets
        return result


__all__ = ["FUTURE_STATE_ROLES", "FutureTargetStateAdapter"]
