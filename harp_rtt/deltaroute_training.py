"""Stage ownership and deployed objectives for HARP-DeltaRoute v4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .exact_k import exact_set_nll, stable_topk
from .factual_branch_attention import FactualAlignmentOutput
from .factual_mixture import exact_factual_mixture_nll
from .losses import boundary_loss_per_endpoint


DeltaRouteStage = Literal[
    "align_m0", "align_m1", "transition_r0", "rollout_r1", "rollout_r2",
    "factual_joint",
]


@dataclass(frozen=True)
class DeltaRouteLoss:
    total: Tensor
    components: Mapping[str, Tensor]

    def detached(self) -> dict[str, float]:
        return {
            **{name: float(value.detach()) for name, value in self.components.items()},
            "total": float(self.total.detach()),
        }


@dataclass(frozen=True)
class DeltaRouteOwnership:
    stage: DeltaRouteStage
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    trainable_parameters: int


def teacher_forcing_probability(
    step: int,
    total_steps: int,
    *,
    decay_fraction: float = 0.6,
) -> float:
    """Linearly decay route forcing to zero, then train fully closed loop."""

    if total_steps < 1 or not 0 <= step <= total_steps:
        raise ValueError("teacher-forcing step must lie in 0..total_steps")
    if not 0.0 < decay_fraction <= 1.0:
        raise ValueError("teacher-forcing decay fraction must lie in (0,1]")
    end = max(1, round(total_steps * decay_fraction))
    return max(0.0, 1.0 - step / end)


def sampled_teacher_force_mask(
    leading_shape: tuple[int, ...],
    layers: int,
    probability: float,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    if layers < 2 or not 0.0 <= probability <= 1.0:
        raise ValueError("teacher-force mask configuration is invalid")
    if probability == 0.0:
        return torch.zeros(*leading_shape, layers - 1, dtype=torch.bool, device=device)
    if probability == 1.0:
        return torch.ones(*leading_shape, layers - 1, dtype=torch.bool, device=device)
    return torch.rand(
        *leading_shape, layers - 1, device=device, generator=generator
    ) < probability


def configure_deltaroute_stage(
    parent: nn.Module,
    aligner: nn.Module | None,
    trajectory: nn.Module | None,
    stage: DeltaRouteStage,
) -> DeltaRouteOwnership:
    """Freeze the v3 parent and expose only the declared v4 subsystem."""

    valid = {
        "align_m0", "align_m1", "transition_r0", "rollout_r1", "rollout_r2",
        "factual_joint",
    }
    if stage not in valid:
        raise ValueError(f"unknown DeltaRoute stage {stage!r}")
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    trainable: list[str] = []
    frozen: list[str] = [f"parent.{name}" for name, _ in parent.named_parameters()]

    def set_module(prefix: str, module: nn.Module | None, active: bool) -> None:
        if module is None:
            if active:
                raise ValueError(f"stage {stage} requires {prefix}")
            return
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(active)
            (trainable if active else frozen).append(f"{prefix}.{name}")

    alignment_stage = stage in {"align_m0", "align_m1", "factual_joint"}
    route_stage = stage in {
        "transition_r0", "rollout_r1", "rollout_r2", "factual_joint"
    }
    set_module("aligner", aligner, alignment_stage)
    set_module("trajectory", trajectory, route_stage)
    count = sum(
        parameter.numel()
        for module in (aligner, trajectory)
        if module is not None
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    if not trainable:
        raise ValueError(f"DeltaRoute stage {stage} selected no parameters")
    return DeltaRouteOwnership(stage, tuple(trainable), tuple(frozen), count)


def _valid_future(targets: Mapping[str, Tensor], scores: Tensor) -> Tensor:
    valid = targets.get("future_available")
    if valid is None:
        return torch.ones(scores.shape[:-1], dtype=torch.bool, device=scores.device)
    valid = valid.bool()
    if valid.shape == scores.shape[:2]:
        return valid[..., None].expand(scores.shape[:-1])
    if valid.shape != scores.shape[:-1]:
        raise ValueError("future availability disagrees with route scores")
    return valid


def missing_true_pair_loss(
    scores: Tensor,
    anchor_scores: Tensor,
    true_ids: Tensor,
    valid: Tensor,
    *,
    k: int = 8,
    margin: float = 0.125,
) -> Tensor:
    """Promote missing true experts over false anchor incumbents."""

    if scores.shape != anchor_scores.shape or true_ids.shape != scores.shape[:-1] + (k,):
        raise ValueError("factual pair-loss geometry is invalid")
    anchor_ids = stable_topk(anchor_scores.detach(), k)
    anchor_mask = torch.zeros_like(anchor_scores, dtype=torch.bool)
    anchor_mask.scatter_(-1, anchor_ids, True)
    true_mask = torch.zeros_like(anchor_mask)
    true_mask.scatter_(-1, true_ids.long(), True)
    missing = true_mask & ~anchor_mask
    intruding = anchor_mask & ~true_mask
    pair_mask = missing.unsqueeze(-1) & intruding.unsqueeze(-2)
    pair_mask &= valid[..., None, None]
    if not bool(pair_mask.any()):
        return scores.sum() * 0.0
    false_minus_true = scores.unsqueeze(-2) - scores.unsqueeze(-1)
    return F.softplus(float(margin) + false_minus_true).masked_select(pair_mask).mean()


def factual_alignment_loss(
    output: FactualAlignmentOutput,
    anchor_scores: Tensor,
    targets: Mapping[str, Tensor],
    *,
    pair_weight: float = 0.1,
    coherence_weight: float = 0.0,
) -> DeltaRouteLoss:
    labels = targets["future_selected_ids"].long()
    valid = _valid_future(targets, output.scores)
    # H1 has an independent exact-root gate; branch alignment starts at H2.
    branch_valid = valid.clone()
    branch_valid[:, 0] = False
    exact = exact_set_nll(
        output.scores, labels, valid=branch_valid, k=labels.shape[-1]
    )
    pair = missing_true_pair_loss(
        output.scores, anchor_scores, labels, branch_valid, k=labels.shape[-1]
    )
    total = exact + float(pair_weight) * pair + float(coherence_weight) * output.coherence_kl
    return DeltaRouteLoss(
        total,
        {"factual_exact_set": exact, "missing_true_pair": pair,
         "coherence_kl": output.coherence_kl},
    )


def counterfactual_trajectory_loss(
    predicted_queries: Tensor,
    predicted_scores: Tensor,
    target_queries: Tensor,
    target_logits: Tensor,
    target_ids: Tensor,
    valid: Tensor,
    expert_keys: Tensor,
    *,
    supervision_weights: Tensor | None = None,
    exact_weight: float = 1.0,
    logit_weight: float = 1.0,
    boundary_weight: float = 0.2,
) -> DeltaRouteLoss:
    """Dense route-trajectory supervision with router-induced query error."""

    if predicted_queries.shape != target_queries.shape:
        raise ValueError("counterfactual query geometry differs")
    if predicted_scores.shape != target_logits.shape:
        raise ValueError("counterfactual score/logit geometry differs")
    if predicted_queries.shape[:-1] != predicted_scores.shape[:-1]:
        raise ValueError("counterfactual query and score leading axes differ")
    if target_ids.shape != predicted_scores.shape[:-1] + (target_ids.shape[-1],):
        raise ValueError("counterfactual selected-set geometry differs")
    if valid.shape != predicted_scores.shape[:-1]:
        raise ValueError("counterfactual validity geometry differs")
    layers = predicted_queries.shape[-2]
    if expert_keys.shape[:2] != (layers, predicted_scores.shape[-1]):
        raise ValueError("router geometry differs from trajectory")
    active = valid.bool()
    weights = active.float()
    if supervision_weights is not None:
        if supervision_weights.shape == valid.shape[:-1]:
            supervision_weights = supervision_weights[..., None].expand_as(valid)
        if supervision_weights.shape != valid.shape:
            raise ValueError("trajectory supervision weights have invalid geometry")
        if not torch.isfinite(supervision_weights).all() or bool((supervision_weights < 0).any()):
            raise ValueError("trajectory supervision weights must be finite and non-negative")
        weights = weights * supervision_weights.float()
    exact = exact_set_nll(
        predicted_scores, target_ids.long(), valid=weights, k=target_ids.shape[-1]
    )
    difference = predicted_queries.float() - target_queries.detach().float()
    induced = torch.einsum("...lr,ler->...le", difference, expert_keys.float())
    per_logit = F.huber_loss(
        induced, torch.zeros_like(induced), reduction="none", delta=1.0
    ).mean(-1)
    logit = (per_logit * weights).sum() / weights.sum().clamp_min(1.0)
    boundary_rows = boundary_loss_per_endpoint(
        predicted_scores, target_logits.detach(), target_ids.long(),
        model_rank_start=6, teacher_rank_start=9, rank_end=32,
    )
    boundary = (boundary_rows * weights).sum() / weights.sum().clamp_min(1.0)
    total = exact_weight * exact + logit_weight * logit + boundary_weight * boundary
    return DeltaRouteLoss(
        total, {"counterfactual_exact_set": exact, "induced_logit_huber": logit,
                "top_boundary": boundary},
    )


def depth_balanced_node_weights(
    node_depth: Tensor,
    node_mask: Tensor,
    *,
    depths: tuple[int, ...] = (2, 3, 4),
) -> Tensor:
    """Give every counterfactual depth equal total weight per source row."""

    if node_depth.shape != node_mask.shape or node_depth.ndim != 2:
        raise ValueError("depth-balanced node tensors must be [B,N]")
    result = torch.zeros_like(node_depth, dtype=torch.float32)
    active_depths = torch.zeros(node_depth.shape[0], device=node_depth.device)
    for depth in depths:
        active = node_mask.bool() & (node_depth.long() == depth)
        count = active.sum(-1).float()
        present = count > 0
        active_depths += present.float()
        result += active.float() / count[:, None].clamp_min(1.0)
    result = result / active_depths[:, None].clamp_min(1.0)
    return result


def joint_factual_mixture_loss(
    *,
    branch_scores: Tensor,
    anchor_scores: Tensor,
    branch_probabilities: Tensor,
    branch_mask: Tensor,
    targets: Mapping[str, Tensor],
    aligned: FactualAlignmentOutput | None = None,
    counterfactual: DeltaRouteLoss | None = None,
    factual_weight: float = 1.0,
    aligned_weight: float = 1.0,
    counterfactual_weight: float = 1.0,
) -> DeltaRouteLoss:
    labels = targets["future_selected_ids"].long()
    valid = _valid_future(targets, anchor_scores)
    branch_valid = valid.clone()
    branch_valid[:, 0] = False
    mixture = exact_factual_mixture_nll(
        branch_scores, anchor_scores, branch_probabilities, branch_mask,
        labels, valid=branch_valid, k=labels.shape[-1],
    )
    zero = branch_scores.sum() * 0.0
    aligned_exact = zero
    if aligned is not None:
        aligned_exact = exact_set_nll(
            aligned.scores, labels, valid=branch_valid, k=labels.shape[-1]
        )
    cf = zero if counterfactual is None else counterfactual.total
    total = factual_weight * mixture.loss + aligned_weight * aligned_exact + counterfactual_weight * cf
    components: dict[str, Tensor] = {
        "factual_branch_mixture": mixture.loss,
        "factual_aligned_exact_set": aligned_exact,
        "counterfactual_trajectory": cf,
    }
    return DeltaRouteLoss(total, components)


__all__ = [
    "DeltaRouteLoss", "DeltaRouteOwnership", "DeltaRouteStage",
    "configure_deltaroute_stage", "counterfactual_trajectory_loss",
    "depth_balanced_node_weights",
    "factual_alignment_loss", "joint_factual_mixture_loss",
    "missing_true_pair_loss", "sampled_teacher_force_mask",
    "teacher_forcing_probability",
]
