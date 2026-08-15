"""Train-only allocation of a fixed resident-expert storage budget."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ResidentAllocation:
    resident_ids: tuple[Tensor, ...]
    resident_counts: tuple[int, ...]
    covered_slots: tuple[int, ...]
    total_slots: tuple[int, ...]

    @property
    def mean_coverage(self) -> float:
        values = [covered / max(total, 1) for covered, total in zip(
            self.covered_slots, self.total_slots, strict=True
        )]
        return sum(values) / len(values)




@dataclass(frozen=True)
class ResidentUtilityAllocation:
    resident_ids: tuple[Tensor, ...]
    resident_counts: tuple[int, ...]
    captured_utility: tuple[float, ...]
    total_utility: tuple[float, ...]

    @property
    def mean_utility_coverage(self) -> float:
        values = [
            captured / max(total, 1e-12)
            for captured, total in zip(
                self.captured_utility, self.total_utility, strict=True
            )
        ]
        return sum(values) / len(values)


def allocate_resident_utility(
    expert_utility: Tensor,
    *,
    total_residents: int,
    minimum_per_layer: int = 64,
    maximum_per_layer: int = 128,
) -> ResidentUtilityAllocation:
    """Spend equal-cost cells on stable, non-negative measured utility."""

    if expert_utility.ndim != 2 or expert_utility.shape[0] < 1:
        raise ValueError("expert utility must be [layers,experts]")
    if not expert_utility.is_floating_point():
        raise TypeError("expert utility must use floating point")
    if not torch.isfinite(expert_utility).all() or bool((expert_utility < 0).any()):
        raise ValueError("expert utility must be finite and non-negative")
    layers, experts = expert_utility.shape
    if not 1 <= minimum_per_layer <= maximum_per_layer <= experts:
        raise ValueError("resident per-layer bounds are invalid")
    if not layers * minimum_per_layer <= total_residents <= layers * maximum_per_layer:
        raise ValueError("resident total lies outside the per-layer bounds")
    ranking = torch.argsort(
        expert_utility, dim=-1, descending=True, stable=True
    )
    per_layer = [int(minimum_per_layer)] * layers
    remaining = int(total_residents - layers * minimum_per_layer)
    while remaining:
        candidates = [
            (
                float(expert_utility[layer, ranking[layer, per_layer[layer]]]),
                -layer,
                layer,
            )
            for layer in range(layers)
            if per_layer[layer] < maximum_per_layer
        ]
        if not candidates:
            raise RuntimeError("resident utility allocation exhausted capacity")
        # Stable ties prefer the lower layer index.
        _gain, _tie, best_layer = max(candidates)
        per_layer[best_layer] += 1
        remaining -= 1
    ids = tuple(
        ranking[layer, :per_layer[layer]].clone() for layer in range(layers)
    )
    captured = tuple(
        float(expert_utility[layer, ids[layer]].sum()) for layer in range(layers)
    )
    totals = tuple(float(expert_utility[layer].sum()) for layer in range(layers))
    return ResidentUtilityAllocation(
        ids, tuple(per_layer), captured, totals
    )


def allocate_resident_experts(
    expert_counts: Tensor,
    *,
    total_residents: int,
    minimum_per_layer: int = 64,
    maximum_per_layer: int = 128,
) -> ResidentAllocation:
    """Maximise captured train slots under one global storage budget.

    Every layer receives the declared minimum. Remaining resident cells are
    assigned greedily by their exact marginal train-slot coverage. Because
    resident experts all have identical storage cost, this is the globally
    optimal allocation for the measured slot-coverage objective.
    """

    if expert_counts.ndim != 2 or expert_counts.shape[0] < 1:
        raise ValueError("expert counts must be [layers,experts]")
    if expert_counts.dtype not in {torch.int32, torch.int64}:
        raise TypeError("expert counts must use an integer dtype")
    if bool((expert_counts < 0).any()):
        raise ValueError("expert counts cannot be negative")
    layers, experts = expert_counts.shape
    if not 1 <= minimum_per_layer <= maximum_per_layer <= experts:
        raise ValueError("resident per-layer bounds are invalid")
    if not layers * minimum_per_layer <= total_residents <= layers * maximum_per_layer:
        raise ValueError("resident total lies outside the per-layer bounds")
    ranking = torch.argsort(
        expert_counts, dim=-1, descending=True, stable=True
    )
    per_layer = [int(minimum_per_layer)] * layers
    remaining = int(total_residents - layers * minimum_per_layer)
    while remaining:
        best_layer = None
        best_gain = -1
        for layer in range(layers):
            offset = per_layer[layer]
            if offset >= maximum_per_layer:
                continue
            expert = int(ranking[layer, offset])
            gain = int(expert_counts[layer, expert])
            if gain > best_gain:
                best_gain = gain
                best_layer = layer
        if best_layer is None:  # pragma: no cover - validated capacity invariant
            raise RuntimeError("resident allocation exhausted its layer capacity")
        per_layer[best_layer] += 1
        remaining -= 1
    ids = tuple(
        ranking[layer, : per_layer[layer]].clone()
        for layer in range(layers)
    )
    covered = tuple(
        int(expert_counts[layer, ids[layer]].sum()) for layer in range(layers)
    )
    totals = tuple(int(expert_counts[layer].sum()) for layer in range(layers))
    return ResidentAllocation(ids, tuple(per_layer), covered, totals)


__all__ = [
    "ResidentAllocation",
    "ResidentUtilityAllocation",
    "allocate_resident_experts",
    "allocate_resident_utility",
]
