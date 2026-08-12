"""Parent-preserving factual branch alignment for HARP-DeltaRoute v4."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .b31 import quota_candidate_union
from .exact_k import cardinality_project_marginals, soft_cardinality_topk
from .model.heads import exact_projected_marginals
from .route_ceiling import stable_ranks


@dataclass(frozen=True)
class FactualAlignmentOutput:
    scores: Tensor
    marginals: Tensor
    correction: Tensor
    expert_branch_weights: Tensor | None
    coherence_kl: Tensor
    candidate_ids: Tensor


def _logit(values: Tensor) -> Tensor:
    epsilon = torch.finfo(torch.float32).eps
    return torch.logit(values.float().clamp(epsilon, 1.0 - epsilon))


def _trainable_exact_marginals(scores: Tensor, k: int) -> Tensor:
    _, exact, _ = exact_projected_marginals(scores.float(), k)
    if not torch.is_grad_enabled() or not scores.requires_grad:
        return exact
    surrogate = soft_cardinality_topk(
        scores, k, temperature=1.0, bisection_steps=32, standardize=False
    )
    return exact + (surrogate - surrogate.detach()).to(exact.dtype)


def _validated_inputs(
    anchor_scores: Tensor,
    anchor_marginals: Tensor,
    node_marginals: Tensor,
    posterior: Tensor,
    node_mask: Tensor,
) -> tuple[int, int, int, int, int]:
    if anchor_scores.ndim != 4 or anchor_marginals.shape != anchor_scores.shape:
        raise ValueError("anchor tensors must share [B,H,L,E] geometry")
    if node_marginals.ndim != 5 or node_marginals.shape[:3] != anchor_scores.shape[:3]:
        raise ValueError("node marginals must be [B,H,L,N,E]")
    batch, horizons, layers, nodes, experts = node_marginals.shape
    if node_marginals.shape[-1] != anchor_scores.shape[-1]:
        raise ValueError("node and anchor expert namespaces differ")
    if posterior.shape != (batch, horizons, nodes + 1):
        raise ValueError("posterior must include captured nodes plus OTHER")
    if node_mask.shape != (batch, horizons, nodes):
        raise ValueError("node mask geometry is invalid")
    if not torch.isfinite(posterior).all() or bool((posterior < 0).any()):
        raise ValueError("posterior must be finite and non-negative")
    return batch, horizons, layers, nodes, experts


def parent_branch_marginals(
    node_marginals: Tensor,
    posterior: Tensor,
    node_mask: Tensor,
    anchor_marginals: Tensor,
    *,
    k: int,
) -> Tensor:
    captured = torch.einsum(
        "bhn,bhlne->bhle",
        posterior[..., :-1].float() * node_mask.float(),
        node_marginals.float(),
    )
    result = captured + posterior[..., -1, None, None].float() * anchor_marginals.float()
    return cardinality_project_marginals(result, k)[0]


def _candidate_ids(anchor_scores: Tensor, evidence: Tensor, *, quota: int, width: int) -> Tensor:
    return quota_candidate_union(
        anchor_scores, evidence, anchor_quota=quota, width=width
    ).expert_ids


class SummaryFactualAligner(nn.Module):
    """M0: expert-specific correction from branch summary statistics."""

    FEATURE_WIDTH = 15

    def __init__(
        self,
        *,
        horizons: int = 4,
        layers: int = 40,
        experts: int = 256,
        exact_k: int = 8,
        hidden_width: int = 64,
        candidate_width: int = 64,
        anchor_quota: int = 32,
    ) -> None:
        super().__init__()
        self.horizons = horizons
        self.layers = layers
        self.experts = experts
        self.exact_k = exact_k
        self.candidate_width = candidate_width
        self.anchor_quota = anchor_quota
        self.features = nn.Sequential(
            nn.Linear(self.FEATURE_WIDTH, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        self.horizon = nn.Embedding(horizons, hidden_width)
        self.layer = nn.Embedding(layers, hidden_width)
        self.output = nn.Linear(hidden_width, 1)
        self.gate = nn.Parameter(torch.zeros(horizons, layers, 1))

    def _feature_tensor(
        self,
        anchor_scores: Tensor,
        anchor_marginals: Tensor,
        node_marginals: Tensor,
        posterior: Tensor,
        mtp_probabilities: Tensor,
        node_mask: Tensor,
        first_divergence_depth: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, horizons, layers, nodes, experts = node_marginals.shape
        parent = parent_branch_marginals(
            node_marginals, posterior, node_mask, anchor_marginals, k=self.exact_k
        )
        mtp = mtp_probabilities.float() * node_mask.float()
        mtp_mass = mtp.sum(-1).clamp_max(1.0)
        mtp_mix = torch.einsum("bhn,bhlne->bhle", mtp, node_marginals.float())
        mtp_mix = mtp_mix + (1.0 - mtp_mass)[..., None, None] * anchor_marginals.float()
        mtp_mix = cardinality_project_marginals(mtp_mix, self.exact_k)[0]
        active = node_mask[:, :, None, :, None]
        evidence = node_marginals.float().masked_fill(~active, 0.0)
        ordered = torch.sort(evidence, dim=-2, descending=True, stable=True).values
        maximum = ordered[..., 0, :]
        second = ordered[..., min(1, nodes - 1), :]
        count = node_mask.sum(-1).clamp_min(1)[:, :, None, None].float()
        mean = evidence.sum(-2) / count
        variance = ((evidence - mean[..., None, :]).square() * active.float()).sum(-2) / count
        support_count = ((evidence > anchor_marginals[..., None, :]) & active).sum(-2).float()
        support_count = support_count / count
        divergence_features: list[Tensor] = []
        for depth in (2, 3, 4):
            depth_mask = (
                node_mask & (first_divergence_depth[:, None, :] == depth)
            )[:, :, None, :, None]
            divergence_features.append(
                node_marginals.float().masked_fill(~depth_mask, 0.0).amax(-2)
            )
        probabilities = posterior.float().clamp_min(torch.finfo(torch.float32).tiny)
        entropy = -(posterior.float() * probabilities.log()).sum(-1)
        entropy = entropy / math.log(max(2, nodes + 1))
        captured_mass = posterior[..., :-1].sum(-1)
        scalar = (
            entropy[..., None, None].expand(batch, horizons, layers, experts),
            captured_mass[..., None, None].expand(batch, horizons, layers, experts),
            posterior[..., -1, None, None].expand(batch, horizons, layers, experts),
        )
        rank = stable_ranks(anchor_scores).float() / max(1, experts - 1)
        features = torch.stack(
            (
                anchor_scores.float(), anchor_marginals.float(), rank,
                parent, mtp_mix, maximum, second, variance, support_count,
                *divergence_features, *scalar,
            ),
            dim=-1,
        )
        return features, parent

    def forward(
        self,
        *,
        anchor_scores: Tensor,
        anchor_marginals: Tensor,
        node_marginals: Tensor,
        posterior: Tensor,
        mtp_probabilities: Tensor,
        node_mask: Tensor,
        first_divergence_depth: Tensor,
    ) -> FactualAlignmentOutput:
        batch, horizons, layers, _, experts = node_marginals.shape
        if (horizons, layers, experts) != (self.horizons, self.layers, self.experts):
            raise ValueError("M0 input geometry differs from its configuration")
        _validated_inputs(
            anchor_scores, anchor_marginals, node_marginals, posterior, node_mask
        )
        if mtp_probabilities.shape != node_mask.shape:
            raise ValueError("MTP branch probabilities have invalid geometry")
        if first_divergence_depth.shape != (batch, node_marginals.shape[3]):
            raise ValueError("first-divergence depth must be [B,N]")
        features, parent = self._feature_tensor(
            anchor_scores, anchor_marginals, node_marginals, posterior,
            mtp_probabilities, node_mask, first_divergence_depth,
        )
        hidden = self.features(features)
        hidden = hidden + self.horizon.weight[None, :, None, None]
        hidden = hidden + self.layer.weight[None, None, :, None]
        correction = self.output(hidden).squeeze(-1)
        scale = self.gate[None]
        scores = _logit(parent) + scale * correction
        proposed = _trainable_exact_marginals(
            _logit(parent) + correction, self.exact_k
        )
        # Both operands have exact mass k.  The zero residual gate therefore
        # preserves the selected v3 parent densely and bit-for-bit while still
        # giving the gate a first-step factual gradient.
        marginals = parent + scale * (proposed - parent)
        candidates = _candidate_ids(
            anchor_scores, marginals, quota=self.anchor_quota,
            width=self.candidate_width,
        )
        return FactualAlignmentOutput(
            scores, marginals, correction, None, scores.sum() * 0.0, candidates
        )


class ExpertConditionedBranchAttention(nn.Module):
    """M1: branch-identity-preserving reliability for every expert."""

    def __init__(
        self,
        expert_keys: Tensor,
        *,
        horizons: int = 4,
        exact_k: int = 8,
        tree_width: int = 256,
        hidden_width: int = 64,
        candidate_width: int = 64,
        anchor_quota: int = 32,
    ) -> None:
        super().__init__()
        if expert_keys.ndim != 3:
            raise ValueError("expert keys must be [L,E,R]")
        layers, experts, router_rank = expert_keys.shape
        self.horizons = horizons
        self.layers = layers
        self.experts = experts
        self.exact_k = exact_k
        self.candidate_width = candidate_width
        self.anchor_quota = anchor_quota
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.branch = nn.Linear(tree_width, hidden_width, bias=False)
        self.context = nn.Linear(tree_width, hidden_width, bias=False)
        self.expert = nn.Linear(router_rank, hidden_width, bias=False)
        self.scalar = nn.Sequential(
            nn.Linear(4, hidden_width), nn.SiLU(), nn.Linear(hidden_width, 1)
        )
        self.tau = nn.Parameter(torch.zeros(horizons, layers, 1))

    def forward(
        self,
        *,
        anchor_scores: Tensor,
        anchor_marginals: Tensor,
        node_marginals: Tensor,
        posterior: Tensor,
        node_mask: Tensor,
        tree_states: Tensor,
        context_states: Tensor,
        query_uncertainty: Tensor | None = None,
    ) -> FactualAlignmentOutput:
        batch, horizons, layers, nodes, experts = node_marginals.shape
        _validated_inputs(
            anchor_scores, anchor_marginals, node_marginals, posterior, node_mask
        )
        if (horizons, layers, experts) != (self.horizons, self.layers, self.experts):
            raise ValueError("M1 input geometry differs from its configuration")
        if tree_states.shape[:2] != (batch, nodes):
            raise ValueError("tree states must be [B,N,W]")
        if context_states.shape[:3] != (batch, horizons, layers):
            raise ValueError("context states must be [B,H,L,W]")
        if query_uncertainty is None:
            query_uncertainty = torch.zeros(
                batch, horizons, layers, nodes,
                device=node_marginals.device, dtype=torch.float32,
            )
        if query_uncertainty.shape != (batch, horizons, layers, nodes):
            raise ValueError("query uncertainty must be [B,H,L,N]")

        branch = self.branch(tree_states)[:, None, None]
        context = self.context(context_states)[:, :, :, None]
        expert = self.expert(self.expert_keys)
        dot = torch.einsum(
            "bhlnw,lew->bhlne", torch.tanh(branch + context), expert
        ) / math.sqrt(expert.shape[-1])
        path = posterior[..., :-1][:, :, None, :, None].expand(
            batch, horizons, layers, nodes, experts
        )
        scalar = torch.stack(
            (
                node_marginals.float(),
                anchor_marginals[..., None, :].expand_as(node_marginals).float(),
                path,
                query_uncertainty[..., None].expand_as(node_marginals).float(),
            ),
            dim=-1,
        )
        reliability = dot + self.scalar(scalar).squeeze(-1)
        other_reliability = torch.zeros(
            batch, horizons, layers, 1, experts,
            device=reliability.device, dtype=reliability.dtype,
        )
        reliability = torch.cat((reliability, other_reliability), dim=3)
        prior = posterior[:, :, None, :, None].expand(
            batch, horizons, layers, nodes + 1, experts
        ).float()
        mask = torch.cat(
            (node_mask, torch.ones_like(node_mask[..., :1])), dim=-1
        )[:, :, None, :, None]
        logits = prior.clamp_min(torch.finfo(torch.float32).tiny).log()
        logits = logits + self.tau[None, :, :, None] * reliability
        weights = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=3)
        values = torch.cat(
            (
                node_marginals.float(),
                anchor_marginals[..., None, :].float(),
            ),
            dim=3,
        )
        proposed = (weights * values).sum(3)
        proposed = cardinality_project_marginals(proposed, self.exact_k)[0]
        zero_weights = torch.softmax(
            prior.clamp_min(torch.finfo(torch.float32).tiny).log().masked_fill(
                ~mask, -torch.inf
            ),
            dim=3,
        )
        zero_aligned = cardinality_project_marginals(
            (zero_weights * values).sum(3), self.exact_k
        )[0]
        parent = parent_branch_marginals(
            node_marginals, posterior, node_mask, anchor_marginals,
            k=self.exact_k,
        )
        # Subtract the numerically identical tau=0 path.  This makes parent
        # reproduction exact without cancelling the first derivative of the
        # learned expert-specific reliability correction.
        aligned = parent + (proposed - zero_aligned)
        scores = _logit(aligned)
        scores = scores - scores.mean(-1, keepdim=True)
        candidates = _candidate_ids(
            anchor_scores, aligned, quota=self.anchor_quota,
            width=self.candidate_width,
        )
        epsilon = torch.finfo(torch.float32).tiny
        coherence = (
            weights
            * (weights.clamp_min(epsilon).log() - prior.clamp_min(epsilon).log())
        ).sum(3)
        coherence = coherence.mean()
        return FactualAlignmentOutput(
            scores, aligned, reliability, weights, coherence, candidates
        )


__all__ = [
    "ExpertConditionedBranchAttention", "FactualAlignmentOutput",
    "SummaryFactualAligner", "parent_branch_marginals",
]
