"""Train-only omission damage and frequency-core resident allocation helpers.

The omission score is deliberately layer local.  It removes one exact routed
expert contribution from the captured full routed output, then evaluates the
result with the frozen next-layer RMSNorm/router geometry.  No model rollout or
new capture is required.

The allocator hard-locks the most frequently executed experts in every layer.
All resident cells have equal storage cost; a Lagrangian count penalty keeps
the selected plan under an explicit resident-hit (expert-call) ceiling.  This
is a deterministic constrained heuristic, not an exact knapsack claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .losses import boundary_loss_per_endpoint
from .metrics import exact_set_nll_per_endpoint, slot_recall_at_k


def _canonical_ids_sha256(ids_by_layer: Sequence[Sequence[int]]) -> str:
    payload = json.dumps(
        [[int(expert) for expert in layer] for layer in ids_by_layer],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frequency_core_ids(
    expert_counts: Tensor,
    *,
    core_size: int = 64,
) -> tuple[Tensor, str]:
    """Return stable count-descending per-layer cores and their canonical hash.

    Exact count ties preserve the original expert-ID order, so the lower ID is
    selected first.  The returned tensor is sorted by frequency, not expert ID;
    the hash therefore commits to both membership and deterministic ordering.
    """

    if not isinstance(expert_counts, Tensor) or expert_counts.ndim != 2:
        raise TypeError("expert_counts must be a rank-two torch.Tensor")
    layers, experts = expert_counts.shape
    if not 1 <= int(core_size) <= experts:
        raise ValueError("core_size must lie in the expert namespace")
    if expert_counts.is_floating_point():
        if not torch.isfinite(expert_counts).all():
            raise ValueError("expert_counts contains NaN or Inf")
    if bool((expert_counts < 0).any()):
        raise ValueError("expert_counts must be non-negative")
    order = torch.argsort(expert_counts, dim=-1, descending=True, stable=True)
    core = order[:, : int(core_size)].to(dtype=torch.int64, device="cpu")
    values = tuple(tuple(int(value) for value in row) for row in core.tolist())
    if len(values) != layers:
        raise AssertionError("frequency core lost a layer")
    return core, _canonical_ids_sha256(values)


def validate_core_inclusion(
    resident_ids_by_layer: Sequence[Sequence[int]],
    core_ids: Tensor | Sequence[Sequence[int]],
) -> None:
    """Fail if any mandatory frequency-core cell is absent from a plan."""

    if isinstance(core_ids, Tensor):
        if core_ids.ndim != 2:
            raise ValueError("core_ids must be rank two")
        required = core_ids.to(dtype=torch.int64, device="cpu").tolist()
    else:
        required = [[int(value) for value in row] for row in core_ids]
    if len(resident_ids_by_layer) != len(required):
        raise ValueError("resident plan and frequency core have different layers")
    missing: list[tuple[int, int]] = []
    for layer, (resident, core) in enumerate(zip(resident_ids_by_layer, required)):
        resident_values = [int(value) for value in resident]
        if len(resident_values) != len(set(resident_values)):
            raise ValueError(f"resident layer {layer} contains duplicate experts")
        resident_set = set(resident_values)
        missing.extend(
            (layer, int(expert)) for expert in core if int(expert) not in resident_set
        )
    if missing:
        preview = ", ".join(f"L{layer}:E{expert}" for layer, expert in missing[:8])
        raise ValueError(f"resident plan omits mandatory frequency-core cells: {preview}")


def router_logits_for_routed_variants(
    predicted_routed: Tensor,
    states: Mapping[str, Tensor],
    *,
    norm_weight: Tensor,
    router_weight: Tensor,
) -> Tensor:
    """Project routed-output variants through the frozen next router.

    ``predicted_routed`` has ``[..., variants, hidden]`` geometry.  Each state
    tensor has the corresponding ``[..., hidden]`` geometry.  This matches the
    audited ShadowRoute next-router objective: only the routed expert output is
    varied while the captured attention delta remains frozen.
    """

    if predicted_routed.ndim < 3 or not predicted_routed.is_floating_point():
        raise ValueError("predicted_routed must be floating [...,variants,hidden]")
    hidden = predicted_routed.shape[-1]
    leading = predicted_routed.shape[:-2]
    required = (
        "post_attention_residual_u",
        "post_moe_residual_xplus",
        "shared_expert_output_delta_s",
        "next_post_attention_residual_u",
    )
    if any(name not in states for name in required):
        raise ValueError("next-router factual states are incomplete")
    values: dict[str, Tensor] = {}
    for name in required:
        value = states[name]
        if not isinstance(value, Tensor) or value.shape != (*leading, hidden):
            raise ValueError(f"next-router state {name!r} has incompatible geometry")
        if value.device != predicted_routed.device:
            raise ValueError("routed variants and factual states must share a device")
        values[name] = value
    if norm_weight.shape != (hidden,):
        raise ValueError("next RMSNorm weight has incompatible geometry")
    if router_weight.ndim != 2 or router_weight.shape[1] != hidden:
        raise ValueError("next router weight has incompatible geometry")
    if norm_weight.device != predicted_routed.device or router_weight.device != predicted_routed.device:
        raise ValueError("next-router weights and routed variants must share a device")

    current_u = values["post_attention_residual_u"].unsqueeze(-2)
    current_xplus = values["post_moe_residual_xplus"].unsqueeze(-2)
    shared = values["shared_expert_output_delta_s"].unsqueeze(-2)
    next_u = values["next_post_attention_residual_u"].unsqueeze(-2)
    predicted_xplus = current_u + shared + predicted_routed
    frozen_attention_delta = next_u - current_xplus
    predicted_next_u = predicted_xplus + frozen_attention_delta
    normalized = predicted_next_u.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(-1, keepdim=True) + 1e-6
    )
    # Qwen3.5/3.6 stores the RMSNorm delta rather than the multiplicative
    # coefficient itself.
    normalized = normalized * (1.0 + norm_weight.float())
    return F.linear(normalized, router_weight.float())


def _valid_slot_mask(valid: Tensor, slots: int, dtype: torch.dtype) -> Tensor:
    return valid.bool().unsqueeze(-1).expand(*valid.shape, slots).to(dtype=dtype)


@torch.no_grad()
def omission_damage_per_slot(
    routed: Tensor,
    individual_outputs: Tensor,
    selected_weights: Tensor,
    valid: Tensor,
    states: Mapping[str, Tensor],
    *,
    norm_weight: Tensor,
    router_weight: Tensor,
    include_exact_objective: bool = True,
) -> dict[str, Tensor]:
    """Return the next-router damage caused by omitting each selected expert.

    Every returned value has ``[..., K]`` geometry and is zero on invalid
    endpoints.  Signed deltas preserve cases where omission improves the local
    objective; ``*_positive_damage`` variants clamp those deltas at zero for a
    conservative resident-utility score.
    """

    if routed.ndim < 2 or not routed.is_floating_point():
        raise ValueError("routed must be floating [...,hidden]")
    if individual_outputs.ndim != routed.ndim + 1:
        raise ValueError("individual_outputs must be [...,K,hidden]")
    leading, hidden = routed.shape[:-1], routed.shape[-1]
    if individual_outputs.shape[:-2] != leading or individual_outputs.shape[-1] != hidden:
        raise ValueError("individual expert outputs do not align with routed output")
    slots = individual_outputs.shape[-2]
    if selected_weights.shape != (*leading, slots):
        raise ValueError("selected weights do not align with expert outputs")
    if valid.shape != leading:
        raise ValueError("validity does not align with routed output")
    if any(value.device != routed.device for value in (individual_outputs, selected_weights, valid)):
        raise ValueError("omission tensors must share a device")
    if not torch.isfinite(routed).all() or not torch.isfinite(individual_outputs).all():
        raise ValueError("omission inputs contain NaN or Inf")
    if not torch.isfinite(selected_weights).all() or bool((selected_weights < 0).any()):
        raise ValueError("selected weights must be finite and non-negative")
    teacher_logits = states.get("next_raw_target_router_logits")
    teacher_ids = states.get("next_selected_expert_ids")
    if not isinstance(teacher_logits, Tensor) or not isinstance(teacher_ids, Tensor):
        raise ValueError("next-router teacher labels are incomplete")
    if teacher_logits.shape[:-1] != leading or teacher_ids.shape != (*leading, slots):
        raise ValueError("next-router teacher labels have incompatible geometry")
    if teacher_logits.device != routed.device or teacher_ids.device != routed.device:
        raise ValueError("teacher labels and omission tensors must share a device")

    contribution = individual_outputs * selected_weights.to(individual_outputs)[..., None]
    full = routed.unsqueeze(-2)
    variants = torch.cat((full, full - contribution), dim=-2)
    logits = router_logits_for_routed_variants(
        variants,
        states,
        norm_weight=norm_weight,
        router_weight=router_weight,
    )
    full_logits = logits[..., 0, :]
    omitted_logits = logits[..., 1:, :]
    active = valid.bool()
    slot_active = _valid_slot_mask(valid, slots, torch.float32)

    logit_difference = omitted_logits - full_logits.unsqueeze(-2)
    logit_mse = logit_difference.float().square().mean(-1)
    full_logit_energy = full_logits.float().square().mean(-1).unsqueeze(-1)
    logit_relative_energy = logit_mse / full_logit_energy.clamp_min(1e-12)
    contribution_energy = contribution.float().square().mean(-1)
    routed_energy = routed.float().square().mean(-1).unsqueeze(-1)
    contribution_relative_energy = contribution_energy / routed_energy.clamp_min(1e-12)

    target_probability = torch.softmax(teacher_logits.detach().float(), dim=-1)
    full_kl = F.kl_div(
        torch.log_softmax(full_logits.float(), dim=-1),
        target_probability,
        reduction="none",
    ).sum(-1)
    omitted_kl = F.kl_div(
        torch.log_softmax(omitted_logits.float(), dim=-1),
        target_probability.unsqueeze(-2).expand_as(omitted_logits),
        reduction="none",
    ).sum(-1)
    kl_delta = omitted_kl - full_kl.unsqueeze(-1)

    safe_ids = torch.where(active.unsqueeze(-1), teacher_ids.long(), 0)
    full_recall = (
        slot_recall_at_k(full_logits, safe_ids, None, k=slots) * active.float()
    )
    expanded_ids = safe_ids.unsqueeze(-2).expand(*leading, slots, slots)
    expanded_valid = active.unsqueeze(-1).expand(*leading, slots)
    omitted_recall = slot_recall_at_k(
        omitted_logits.reshape(-1, omitted_logits.shape[-1]),
        expanded_ids.reshape(-1, slots),
        None,
        k=slots,
    ).reshape(*leading, slots) * slot_active
    recall_damage = full_recall.unsqueeze(-1) - omitted_recall

    result = {
        "router_logit_mse": logit_mse * slot_active,
        "router_logit_relative_energy": logit_relative_energy * slot_active,
        "router_kl_delta": kl_delta * slot_active,
        "router_kl_positive_damage": kl_delta.clamp_min(0.0) * slot_active,
        "slot_recall_damage": recall_damage * slot_active,
        "slot_recall_positive_damage": recall_damage.clamp_min(0.0) * slot_active,
        "contribution_energy": contribution_energy * slot_active,
        "contribution_relative_energy": contribution_relative_energy * slot_active,
    }
    if not include_exact_objective:
        return result

    full_set = exact_set_nll_per_endpoint(
        full_logits, safe_ids, None, k=slots
    ) * active.float()
    omitted_set = exact_set_nll_per_endpoint(
        omitted_logits.reshape(-1, omitted_logits.shape[-1]),
        expanded_ids.reshape(-1, slots),
        None,
        k=slots,
    ).reshape(*leading, slots) * slot_active
    full_boundary = boundary_loss_per_endpoint(
        full_logits,
        teacher_logits.detach(),
        safe_ids,
        margin=0.125,
        model_rank_start=9,
        teacher_rank_start=9,
        rank_end=32,
    ) * active.float()
    omitted_boundary = boundary_loss_per_endpoint(
        omitted_logits.reshape(-1, omitted_logits.shape[-1]),
        teacher_logits.detach().unsqueeze(-2).expand_as(omitted_logits).reshape(
            -1, omitted_logits.shape[-1]
        ),
        expanded_ids.reshape(-1, slots),
        margin=0.125,
        model_rank_start=9,
        teacher_rank_start=9,
        rank_end=32,
    ).reshape(*leading, slots) * slot_active
    set_delta = omitted_set - full_set.unsqueeze(-1)
    boundary_delta = omitted_boundary - full_boundary.unsqueeze(-1)
    objective_delta = kl_delta + set_delta + 0.2 * boundary_delta
    result.update(
        {
            "exact_set_nll_delta": set_delta * slot_active,
            "exact_set_nll_positive_damage": set_delta.clamp_min(0.0) * slot_active,
            "boundary_delta": boundary_delta * slot_active,
            "boundary_positive_damage": boundary_delta.clamp_min(0.0) * slot_active,
            "objective_delta": objective_delta * slot_active,
            "objective_positive_damage": objective_delta.clamp_min(0.0) * slot_active,
        }
    )
    return result


def reduce_expert_damage(
    selected_ids: Tensor,
    selected_weights: Tensor,
    valid: Tensor,
    per_slot: Mapping[str, Tensor],
    *,
    experts: int = 256,
) -> dict[str, Tensor]:
    """Scatter per-slot damage, occurrence counts, and route mass by expert."""

    if selected_ids.ndim < 2 or selected_weights.shape != selected_ids.shape:
        raise ValueError("selected IDs and weights must share [...,K] geometry")
    if valid.shape != selected_ids.shape[:-1]:
        raise ValueError("validity must match selected ID leading geometry")
    if selected_ids.device != selected_weights.device or selected_ids.device != valid.device:
        raise ValueError("route tensors must share a device")
    if selected_ids.numel() and bool(((selected_ids < 0) | (selected_ids >= experts)).any()):
        raise ValueError("selected expert ID lies outside the namespace")
    active = valid.bool().unsqueeze(-1).expand_as(selected_ids)
    ids = selected_ids.long().reshape(-1)
    active_flat = active.reshape(-1)
    occurrence = torch.zeros(experts, dtype=torch.float64, device=selected_ids.device)
    occurrence.scatter_add_(0, ids, active_flat.to(torch.float64))
    mass = torch.zeros_like(occurrence)
    mass.scatter_add_(
        0,
        ids,
        (selected_weights.float() * active.to(selected_weights.dtype)).reshape(-1).to(torch.float64),
    )
    result = {"occurrences": occurrence, "selected_weight_mass": mass}
    for name, values in per_slot.items():
        if values.shape != selected_ids.shape:
            raise ValueError(f"per-slot metric {name!r} has incompatible geometry")
        if values.device != selected_ids.device or not values.is_floating_point():
            raise ValueError(f"per-slot metric {name!r} must be floating on the route device")
        if not torch.isfinite(values).all():
            raise ValueError(f"per-slot metric {name!r} contains NaN or Inf")
        reduced = torch.zeros_like(occurrence)
        reduced.scatter_add_(
            0,
            ids,
            (values.float() * active.to(values.dtype)).reshape(-1).to(torch.float64),
        )
        result[name] = reduced
    return result


@dataclass(frozen=True)
class CoreLockedAllocation:
    resident_expert_ids_by_layer: tuple[tuple[int, ...], ...]
    resident_counts_by_layer: tuple[int, ...]
    total_residents: int
    optional_residents: int
    resident_hit_count: int
    hit_count_cap: int
    total_utility: float
    lagrange_lambda: float
    frequency_core_size: int
    frequency_core_sha256: str
    resident_ids_sha256: str
    method: str = "frequency_core_locked_lagrangian_hit_cap_v1"


def _select_optional_lagrangian(
    utility: Tensor,
    counts: Tensor,
    core: Tensor,
    *,
    optional_residents: int,
    maximum_per_layer: int,
    lagrange_lambda: float,
) -> tuple[tuple[tuple[int, ...], ...], int, float]:
    layers, experts = utility.shape
    core_mask = torch.zeros((layers, experts), dtype=torch.bool)
    core_mask.scatter_(1, core, True)
    candidates: list[tuple[float, int, int]] = []
    for layer in range(layers):
        for expert in range(experts):
            if not bool(core_mask[layer, expert]):
                score = float(utility[layer, expert]) - float(lagrange_lambda) * float(
                    counts[layer, expert]
                )
                candidates.append((score, layer, expert))
    # Stable, explicit tie convention: score descending, then layer/expert ID.
    candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    selected = [set(int(value) for value in core[layer].tolist()) for layer in range(layers)]
    per_layer_limit = int(maximum_per_layer)
    chosen = 0
    for _score, layer, expert in candidates:
        if len(selected[layer]) >= per_layer_limit:
            continue
        selected[layer].add(expert)
        chosen += 1
        if chosen == optional_residents:
            break
    if chosen != optional_residents:
        raise ValueError("layer maxima leave too few cells for the requested resident total")
    ids = tuple(tuple(sorted(layer_ids)) for layer_ids in selected)
    hit_count = sum(int(counts[layer, expert]) for layer, row in enumerate(ids) for expert in row)
    total_utility = sum(
        float(utility[layer, expert]) for layer, row in enumerate(ids) for expert in row
    )
    return ids, hit_count, total_utility


def allocate_core_locked_utility(
    expert_utility: Tensor,
    expert_counts: Tensor,
    *,
    total_residents: int = 3850,
    core_size: int = 64,
    maximum_per_layer: int = 128,
    hit_count_cap: int = 3_245_387,
    binary_search_steps: int = 80,
) -> CoreLockedAllocation:
    """Allocate equal-size resident cells with a mandatory frequency core.

    The Lagrangian multiplier is the smallest searched count penalty whose
    deterministic exact-cardinality selection meets ``hit_count_cap``.  This
    keeps storage and resident execution no worse than the supplied ceilings;
    it does not assert global optimality for the discrete two-constraint
    problem.
    """

    if not isinstance(expert_utility, Tensor) or not isinstance(expert_counts, Tensor):
        raise TypeError("expert utility and counts must be torch tensors")
    if expert_utility.ndim != 2 or expert_counts.shape != expert_utility.shape:
        raise ValueError("expert utility and counts must share [layers,experts] geometry")
    if not expert_utility.is_floating_point() or not torch.isfinite(expert_utility).all():
        raise ValueError("expert utility must be finite floating-point")
    if expert_counts.is_floating_point() and not torch.all(expert_counts == expert_counts.round()):
        raise ValueError("expert counts must be integral")
    if bool((expert_counts < 0).any()):
        raise ValueError("expert counts must be non-negative")
    utility = expert_utility.detach().to(device="cpu", dtype=torch.float64)
    counts = expert_counts.detach().to(device="cpu", dtype=torch.int64)
    layers, experts = utility.shape
    if not 1 <= core_size <= maximum_per_layer <= experts:
        raise ValueError("resident per-layer bounds are invalid")
    mandatory = layers * int(core_size)
    optional = int(total_residents) - mandatory
    capacity = layers * (int(maximum_per_layer) - int(core_size))
    if optional < 0 or optional > capacity:
        raise ValueError("resident total is incompatible with core and layer maximum")
    if not isinstance(hit_count_cap, int) or hit_count_cap < 0:
        raise ValueError("hit_count_cap must be a non-negative integer")
    if binary_search_steps < 1:
        raise ValueError("binary_search_steps must be positive")

    core, core_hash = frequency_core_ids(counts, core_size=core_size)
    zero_ids, zero_hits, zero_utility = _select_optional_lagrangian(
        utility,
        counts,
        core,
        optional_residents=optional,
        maximum_per_layer=maximum_per_layer,
        lagrange_lambda=0.0,
    )
    if zero_hits <= hit_count_cap:
        chosen_ids, chosen_hits, chosen_utility, chosen_lambda = (
            zero_ids,
            zero_hits,
            zero_utility,
            0.0,
        )
    else:
        low = 0.0
        high = 1.0
        high_ids, high_hits, high_utility = _select_optional_lagrangian(
            utility,
            counts,
            core,
            optional_residents=optional,
            maximum_per_layer=maximum_per_layer,
            lagrange_lambda=high,
        )
        while high_hits > hit_count_cap and high < 2.0**60:
            low = high
            high *= 2.0
            high_ids, high_hits, high_utility = _select_optional_lagrangian(
                utility,
                counts,
                core,
                optional_residents=optional,
                maximum_per_layer=maximum_per_layer,
                lagrange_lambda=high,
            )
        if high_hits > hit_count_cap:
            raise ValueError("resident hit-count cap is infeasible under the locked core")
        chosen_ids, chosen_hits, chosen_utility, chosen_lambda = (
            high_ids,
            high_hits,
            high_utility,
            high,
        )
        for _ in range(int(binary_search_steps)):
            midpoint = (low + high) / 2.0
            ids, hits, total_utility = _select_optional_lagrangian(
                utility,
                counts,
                core,
                optional_residents=optional,
                maximum_per_layer=maximum_per_layer,
                lagrange_lambda=midpoint,
            )
            if hits <= hit_count_cap:
                high = midpoint
                chosen_ids, chosen_hits, chosen_utility, chosen_lambda = (
                    ids,
                    hits,
                    total_utility,
                    midpoint,
                )
            else:
                low = midpoint

    validate_core_inclusion(chosen_ids, core)
    resident_counts = tuple(len(row) for row in chosen_ids)
    if sum(resident_counts) != total_residents or max(resident_counts) > maximum_per_layer:
        raise AssertionError("allocator violated its resident-cell constraints")
    if chosen_hits > hit_count_cap:
        raise AssertionError("allocator violated its resident-hit ceiling")
    if not math.isfinite(chosen_utility):
        raise AssertionError("allocator produced non-finite utility")
    return CoreLockedAllocation(
        resident_expert_ids_by_layer=chosen_ids,
        resident_counts_by_layer=resident_counts,
        total_residents=int(total_residents),
        optional_residents=int(optional),
        resident_hit_count=int(chosen_hits),
        hit_count_cap=int(hit_count_cap),
        total_utility=float(chosen_utility),
        lagrange_lambda=float(chosen_lambda),
        frequency_core_size=int(core_size),
        frequency_core_sha256=core_hash,
        resident_ids_sha256=_canonical_ids_sha256(chosen_ids),
    )
