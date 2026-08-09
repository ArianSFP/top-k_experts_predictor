"""Deterministic, layer-specific C64 multi-source candidate union."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .config import HARPRTTConfig


@dataclass(frozen=True)
class CandidateUnionOutput:
    expert_ids: Tensor
    dense_mask: Tensor
    aggregate_scores: Tensor
    normalized_sources: Tensor
    anchor_quota: int


class CandidateUnion(nn.Module):
    """Build a fixed-width union while preserving all 256 dense scores."""

    SOURCE_COUNT = 6

    def __init__(
        self,
        config: HARPRTTConfig,
        temperatures: Sequence[float] | Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if temperatures is None:
            value = torch.ones(self.SOURCE_COUNT, dtype=torch.float32)
        else:
            value = torch.as_tensor(temperatures, dtype=torch.float32)
        if value.shape != (self.SOURCE_COUNT,):
            raise ValueError(f"candidate temperatures must contain {self.SOURCE_COUNT} values")
        if (value <= 0).any() or not torch.isfinite(value).all():
            raise ValueError("candidate temperatures must be finite and positive")
        self.register_buffer("temperatures", value.clone())

    @staticmethod
    def scheduled_anchor_quota(width: int, progress: float) -> int:
        """Anchor top-width through 10%, annealing to frozen top-40 by 50%.

        B1.5 selected the request-disjoint 40/24 contract before B2 began.
        The earlier top-48 target remains historical documentation; B3 must
        train and evaluate the exact policy accepted by the user rather than
        silently reverting to the development default.
        """
        if not 0.0 <= float(progress) <= 1.0:
            raise ValueError("candidate curriculum progress must lie in [0,1]")
        target = min(width, 40)
        if progress <= 0.10:
            return width
        if progress >= 0.50:
            return target
        fraction = (float(progress) - 0.10) / 0.40
        return max(target, int(round(width + fraction * (target - width))))

    def forward(
        self, sources: Sequence[Tensor], *, training_progress: float = 0.0,
        active_sources: Sequence[bool] | Tensor | None = None,
        anchor_quota_override: int | None = None,
    ) -> CandidateUnionOutput:
        if len(sources) != self.SOURCE_COUNT:
            raise ValueError(f"candidate union requires {self.SOURCE_COUNT} dense sources")
        first = sources[0]
        if first.ndim != 4 or first.shape[-1] != self.config.experts:
            raise ValueError("candidate sources must be [B,H,L,E]")
        if any(source.shape != first.shape for source in sources):
            raise ValueError("candidate sources must share one dense shape")
        if active_sources is None:
            active = torch.tensor(
                [True] + [False] * (self.SOURCE_COUNT - 1),
                dtype=torch.bool, device=first.device,
            )
        else:
            active = torch.as_tensor(
                active_sources, dtype=torch.bool, device=first.device
            )
        if active.shape != (self.SOURCE_COUNT,) or not bool(active[0]):
            raise ValueError("candidate source zero (anchor) must remain active")
        stacked = torch.stack([source.float() for source in sources], dim=-2)
        normalized = stacked / self.temperatures[None, None, None, :, None]
        normalized = normalized - normalized.mean(dim=-1, keepdim=True)
        aggregate = normalized.masked_fill(
            ~active[None, None, None, :, None], -torch.inf
        ).amax(dim=-2)
        leading = first.shape[:-1]
        rows = int(torch.tensor(leading).prod().item())
        experts = self.config.experts
        width = self.config.candidate_width
        flat = normalized.reshape(rows, self.SOURCE_COUNT, experts)
        source_order = torch.argsort(
            flat, dim=-1, descending=True, stable=True
        )
        aggregate_order = torch.argsort(
            aggregate.reshape(rows, experts),
            dim=-1,
            descending=True,
            stable=True,
        )
        selected = torch.zeros(rows, experts, dtype=torch.bool, device=first.device)
        candidate_ids = torch.full(
            (rows, width), -1, dtype=torch.long, device=first.device
        )
        counts = torch.zeros(rows, dtype=torch.long, device=first.device)
        row_ids = torch.arange(rows, device=first.device)

        def add(ids: Tensor) -> None:
            is_new = ~selected.gather(1, ids[:, None]).squeeze(1)
            room = counts < width
            keep = is_new & room
            if keep.any():
                active_rows = row_ids[keep]
                active_ids = ids[keep]
                slots = counts[keep]
                candidate_ids[active_rows, slots] = active_ids
                selected[active_rows, active_ids] = True
                counts[keep] += 1

        quota = self.scheduled_anchor_quota(width, training_progress)
        if anchor_quota_override is not None:
            if not self.training:
                raise ValueError("anchor quota override is a training-only ablation")
            if not self.config.exact_k <= anchor_quota_override <= width:
                raise ValueError("anchor quota override lies outside [exact_k,width]")
            quota = int(anchor_quota_override)
        for rank in range(min(quota, experts)):
            add(source_order[:, 0, rank])
        for rank in range(experts):
            if bool((counts >= width).all()):
                break
            add(aggregate_order[:, rank])
        if (candidate_ids < 0).any():
            raise RuntimeError("candidate union failed to fill its configured width")
        dense_mask = selected.reshape(*leading, experts)
        return CandidateUnionOutput(
            expert_ids=candidate_ids.reshape(*leading, width),
            dense_mask=dense_mask,
            aggregate_scores=aggregate.to(first.dtype),
            normalized_sources=normalized.to(first.dtype),
            anchor_quota=quota,
        )


__all__ = ["CandidateUnion", "CandidateUnionOutput"]
