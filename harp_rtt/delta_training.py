"""Strict staged losses and ownership for HARP-DeltaTree v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .delta import HARPDeltaOutput, HARPDeltaTree
from .exact_k import exact_set_nll, stable_topk
from .v2_losses import swap_loss


DeltaStage = Literal["semantic", "candidate", "ranker", "calibration"]


@dataclass(frozen=True)
class DeltaOwnership:
    stage: DeltaStage
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    trainable_parameters: int


@dataclass(frozen=True)
class DeltaLoss:
    total: Tensor
    components: Mapping[str, Tensor]
    outside_candidate_slots: int = 0

    def detached(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            name: float(value.detach()) for name, value in self.components.items()
        }
        result["total"] = float(self.total.detach())
        result["outside_candidate_slots"] = self.outside_candidate_slots
        return result


def configure_delta_stage(model: nn.Module, stage: DeltaStage) -> DeltaOwnership:
    """Expose exactly one v3 subsystem; the incumbent anchor is external."""

    if stage not in ("semantic", "candidate", "ranker", "calibration"):
        raise ValueError(f"unknown HARP-Delta stage {stage!r}")
    trainable: list[str] = []
    frozen: list[str] = []
    count = 0
    for name, parameter in model.named_parameters():
        core_name = name.removeprefix("core.")
        is_adapter = name.startswith("adapter.")
        if stage == "semantic":
            active = is_adapter or not (
                core_name.startswith("candidates.")
                or core_name.startswith("ranker.")
            )
        elif stage == "candidate":
            active = core_name.startswith("candidates.")
        elif stage == "ranker":
            active = core_name.startswith("ranker.")
        else:
            active = (
                core_name.startswith("candidates.gain")
                or core_name.startswith("candidates.quota")
                or core_name.startswith("ranker.swap")
            )
        parameter.requires_grad_(active)
        if active:
            trainable.append(name)
            count += parameter.numel()
        else:
            frozen.append(name)
    if not trainable:
        raise ValueError(f"Delta stage {stage} selected no parameters")
    return DeltaOwnership(stage, tuple(trainable), tuple(frozen), count)


def set_delta_stage_mode(model: nn.Module, stage: DeltaStage) -> None:
    """Enable dropout only inside the subsystem owned by this stage."""

    if stage not in ("semantic", "candidate", "ranker", "calibration"):
        raise ValueError(f"unknown HARP-Delta stage {stage!r}")
    model.eval()
    prefix = getattr(model, "core", model)
    if stage == "semantic":
        adapter = getattr(model, "adapter", None)
        if isinstance(adapter, nn.Module):
            adapter.train()
        for name in ("tree", "context", "root", "position", "set_head", "path"):
            getattr(prefix, name).train()
    elif stage == "candidate":
        prefix.candidates.train()
    elif stage == "ranker":
        prefix.ranker.train()
    else:
        prefix.candidates.train()
        prefix.ranker.swap.train()


def _valid_mask(targets: Mapping[str, Tensor], scores: Tensor) -> Tensor:
    valid = targets.get("future_available")
    if valid is None:
        return torch.ones(scores.shape[:-1], dtype=torch.bool, device=scores.device)
    valid = valid.bool()
    if valid.shape == scores.shape[:2]:
        return valid[..., None].expand(scores.shape[:-1])
    if valid.shape != scores.shape[:-1]:
        raise ValueError("future availability disagrees with score geometry")
    return valid


def _active_exact_set_nll(
    scores: Tensor,
    labels: Tensor,
    active: Tensor,
    *,
    k: int,
) -> Tensor:
    """Evaluate exact-k only on supervised rows, preserving the masked loss."""

    if active.shape != scores.shape[:-1] or labels.shape != scores.shape[:-1] + (k,):
        raise ValueError("active exact-set rows disagree with score/label geometry")
    selected_scores = scores[active.bool()]
    selected_labels = labels[active.bool()]
    if selected_scores.numel() == 0:
        return scores.sum() * 0.0
    return exact_set_nll(selected_scores, selected_labels, k=k)


def semantic_loss(
    output: HARPDeltaOutput,
    targets: Mapping[str, Tensor],
    counterfactual: Mapping[str, Tensor],
    *,
    budget_index: int = 2,
    query_weight: float = 0.05,
    posterior_weight: float = 0.05,
) -> DeltaLoss:
    """Direct factual H1 plus balanced counterfactual branch-set learning."""

    labels = targets["future_selected_ids"].long()
    valid = _valid_mask(targets, output.candidate_scores)
    root = exact_set_nll(
        output.root_scores,
        labels[:, 0],
        valid=valid[:, 0],
        k=labels.shape[-1],
    )
    root_query_target = targets.get("future_query_coordinates")
    if root_query_target is None:
        root_query = output.root_queries.sum() * 0.0
    else:
        if root_query_target.shape != output.root_queries.shape[:1] + (
            output.root_queries.shape[1], output.root_queries.shape[2]
        ):
            raise ValueError("factual H1 query target has invalid geometry")
        root_difference = F.smooth_l1_loss(
            output.root_queries.float(), root_query_target.detach().float(),
            reduction="none",
        ).mean(-1)
        root_mask = valid[:, 0].float()
        root_query = (root_difference * root_mask).sum() / root_mask.sum().clamp_min(1)
    node_ids = counterfactual["selected_ids"].long()
    node_valid = counterfactual["valid"].bool()
    node_depth = counterfactual["depth"].long()
    selection = counterfactual["budget_node_masks"].bool()
    if selection.ndim == 3:
        if not 0 <= budget_index < selection.shape[1]:
            raise ValueError("counterfactual budget index is unavailable")
        selection = selection[:, budget_index]
    elif selection.ndim != 2:
        raise ValueError("counterfactual budget masks must be [B,N] or [B,P,N]")
    node_losses: list[Tensor] = []
    query_losses: list[Tensor] = []
    query_targets = counterfactual["query_coordinates"].float()
    for horizon in range(1, output.node_scores.shape[1]):
        structural = selection & (node_depth == horizon + 1)
        active = node_valid & structural[..., None]
        if not bool(active.any()):
            continue
        predicted = output.node_scores[:, horizon].permute(0, 2, 1, 3)
        queries = output.node_queries[:, horizon].permute(0, 2, 1, 3)
        node_losses.append(
            _active_exact_set_nll(
                predicted, node_ids, active, k=node_ids.shape[-1]
            )
        )
        difference = F.smooth_l1_loss(
            queries.float(), query_targets, reduction="none"
        ).mean(-1)
        mask = active.float()
        query_losses.append((difference * mask).sum() / mask.sum().clamp_min(1))
    zero = output.node_scores.sum() * 0.0
    branch_set = torch.stack(node_losses).mean() if node_losses else zero
    query = torch.stack(query_losses).mean() if query_losses else zero

    factual_branch = targets["factual_branch_index"].long()
    if factual_branch.shape != output.factual_path_logits.shape[:2]:
        raise ValueError("factual branch labels must be [B,H]")
    path = F.cross_entropy(
        output.factual_path_logits.flatten(0, 1), factual_branch.flatten()
    )
    target_posterior = counterfactual.get("target_path_distribution")
    if target_posterior is None:
        posterior = zero
    else:
        if target_posterior.shape != output.factual_path_logits.shape:
            raise ValueError("target path distribution has invalid geometry")
        posterior = F.kl_div(
            F.log_softmax(output.factual_path_logits.float(), dim=-1),
            target_posterior.detach().float(),
            reduction="batchmean",
        )
    components = {
        "root_exact_set": root,
        "root_query": root_query,
        "branch_exact_set": branch_set,
        "factual_path": path,
        "query": query,
        "target_posterior": posterior,
    }
    total = (
        root + branch_set + path
        + query_weight * (root_query + query)
        + posterior_weight * posterior
    )
    return DeltaLoss(total, components)


def quota_targets(
    model: HARPDeltaTree,
    output: HARPDeltaOutput,
    anchor_scores: Tensor,
    target_ids: Tensor,
) -> Tensor:
    """Choose maximum-coverage quota, breaking ties toward more anchor."""

    values: list[Tensor] = []
    with torch.no_grad():
        for quota in model.candidates.QUOTAS:
            selected = model.candidates(
                anchor_scores,
                output.anchor_marginals,
                output.branch_marginals,
                output.factual_path_posterior,
                forced_anchor_quota=quota,
            ).expert_ids
            membership = (
                target_ids[..., None] == selected[..., None, :]
            ).any(-1).float().sum(-1)
            values.append(membership)
    coverage = torch.stack(values, dim=-1)
    # argmax returns the first maximum; QUOTAS is ordered 64 -> 32.
    return torch.argmax(coverage, dim=-1)


def candidate_loss(
    model: HARPDeltaTree,
    output: HARPDeltaOutput,
    anchor_scores: Tensor,
    targets: Mapping[str, Tensor],
    *,
    quota_weight: float = 0.1,
) -> DeltaLoss:
    labels = targets["future_selected_ids"].long()
    valid = _valid_mask(targets, output.candidate_scores)
    direct = exact_set_nll(
        output.candidate_scores, labels, valid=valid, k=labels.shape[-1]
    )
    quota = quota_targets(model, output, anchor_scores, labels)
    quota_ce = F.cross_entropy(
        output.quota_logits.reshape(-1, output.quota_logits.shape[-1]),
        quota.reshape(-1),
    )
    anchor_top64 = stable_topk(anchor_scores.detach(), 64)
    anchor_mask = torch.zeros_like(anchor_scores, dtype=torch.bool)
    anchor_mask.scatter_(-1, anchor_top64, True)
    true_mask = torch.zeros_like(anchor_mask)
    true_mask.scatter_(-1, labels, True)
    missing = true_mask & ~anchor_mask
    false_tail = anchor_mask & ~true_mask
    # [missing, false] penalizes a false incumbent that outranks a missing
    # true expert. Spell the axes out so a transpose cannot silently reverse
    # the intended pair.
    false_minus_missing = (
        output.candidate_scores.unsqueeze(-2)
        - output.candidate_scores.unsqueeze(-1)
    )
    pair = F.softplus(0.125 + false_minus_missing)
    pair_mask = missing.unsqueeze(-1) & false_tail.unsqueeze(-2)
    pairwise = pair.masked_select(pair_mask).mean() if bool(pair_mask.any()) else direct * 0
    components = {"candidate_exact_set": direct, "quota": quota_ce, "candidate_pair": pairwise}
    return DeltaLoss(direct + quota_weight * quota_ce + 0.1 * pairwise, components)


def ranker_loss(
    output: HARPDeltaOutput,
    anchor_scores: Tensor,
    targets: Mapping[str, Tensor],
    *,
    swap_weight: float = 0.1,
) -> DeltaLoss:
    labels = targets["future_selected_ids"].long()
    valid = _valid_mask(targets, output.candidate_scores)
    # Decode can only retain anchor incumbents or select a member of C64.
    # All incumbents are guaranteed by the minimum anchor quota, so masking
    # non-candidates makes the differentiable exact-set objective match the
    # deployed selective-swap namespace.
    dense = torch.full_like(anchor_scores, -torch.inf)
    dense.scatter_(-1, output.candidate_ids, output.ranked_candidate_scores)
    contained = output.candidate_mask.gather(-1, labels).all(-1)
    conditioned_valid = valid & contained
    exact = exact_set_nll(dense, labels, valid=conditioned_valid, k=labels.shape[-1])
    active = conditioned_valid.reshape(-1)
    swap = swap_loss(
        dense.reshape(-1, dense.shape[-1])[active],
        anchor_scores.detach().reshape(-1, dense.shape[-1])[active],
        labels.reshape(-1, labels.shape[-1])[active],
        exact_k=labels.shape[-1],
        candidate_mask=output.candidate_mask.reshape(-1, dense.shape[-1])[active],
    )
    anchor_top = stable_topk(anchor_scores.detach(), labels.shape[-1])
    incumbent = torch.zeros_like(output.candidate_mask)
    incumbent.scatter_(-1, anchor_top, True)
    truth = torch.zeros_like(output.candidate_mask)
    truth.scatter_(-1, labels, True)
    outsider = ~incumbent.gather(-1, output.candidate_ids)
    swap_target = truth.gather(-1, output.candidate_ids).float()
    bce = F.binary_cross_entropy_with_logits(
        output.swap_logits[outsider], swap_target[outsider]
    ) if bool(outsider.any()) else exact * 0
    membership = (
        labels[..., :, None] == output.candidate_ids[..., None, :]
    ).any(-1)
    outside = int((~membership & valid[..., None]).sum().item())
    components = {"ranker_exact_set": exact, "swap_pair": swap.loss, "swap_confidence": bce}
    return DeltaLoss(exact + swap_weight * swap.loss + 0.1 * bce, components, outside)


__all__ = [
    "DeltaLoss",
    "DeltaOwnership",
    "DeltaStage",
    "candidate_loss",
    "configure_delta_stage",
    "quota_targets",
    "ranker_loss",
    "set_delta_stage_mode",
    "semantic_loss",
]
