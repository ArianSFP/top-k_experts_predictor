"""Permutation-equivariant local/axial C64 candidate reranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .common import (
    SelfAttentionResidual,
    SwiGLUResidual,
    gather_dense_experts,
    gather_layer_experts,
    safe_padding_mask,
    zero_linear,
)
from .config import HARPRTTConfig


@dataclass(frozen=True)
class RerankerOutput:
    dense_scores: Tensor
    candidate_corrections: Tensor
    candidate_tokens: Tensor
    route_summaries: Tensor
    causal_history_features: Tensor
    branch_support_features: Tensor


@dataclass(frozen=True)
class CausalRouteSupport:
    """Dense label-free route support derived from committed history only."""

    membership: Tensor
    execution_weights: Tensor
    previous_layer_execution: Tensor
    transition_scores: Tensor


class _AxialSummaryBlock(nn.Module):
    def __init__(self, config: HARPRTTConfig) -> None:
        super().__init__()
        self.layer = SelfAttentionResidual(
            config.reranker_width,
            config.attention_heads,
            config.reranker_ffn_width,
            config.dropout,
        )
        self.horizon = SelfAttentionResidual(
            config.reranker_width,
            config.attention_heads,
            config.reranker_ffn_width,
            config.dropout,
        )

    def forward(self, summaries: Tensor) -> Tensor:
        batch, horizons, layers, width = summaries.shape
        layer_axis = summaries.reshape(batch * horizons, layers, width)
        layer_axis = self.layer(layer_axis)
        summaries = layer_axis.reshape(batch, horizons, layers, width)
        horizon_axis = summaries.permute(0, 2, 1, 3).reshape(
            batch * layers, horizons, width
        )
        horizon_axis = self.horizon(horizon_axis)
        return horizon_axis.reshape(batch, layers, horizons, width).permute(
            0, 2, 1, 3
        )


class _SameObjectHorizonBlock(nn.Module):
    """Uniform same-object attention implemented with differentiable scatter."""

    def __init__(self, config: HARPRTTConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = nn.RMSNorm(config.reranker_width)
        self.update = nn.Linear(config.reranker_width, config.reranker_width)
        self.feedforward = SwiGLUResidual(
            config.reranker_width, config.reranker_ffn_width, config.dropout
        )

    def forward(
        self,
        tokens: Tensor,
        candidate_ids: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        batch, horizons, layers, candidates, width = tokens.shape
        # Group by the layer-specific expert object; numerical IDs are never
        # shared between different layer rows.
        normalized = self.norm(tokens).permute(0, 2, 1, 3, 4).reshape(
            batch * layers, horizons * candidates, width
        )
        ids = candidate_ids.permute(0, 2, 1, 3).reshape(
            batch * layers, horizons * candidates
        )
        mask = candidate_mask.permute(0, 2, 1, 3).reshape(
            batch * layers, horizons * candidates
        )
        dense = normalized.new_zeros(batch * layers, self.config.experts, width)
        dense.scatter_add_(
            1,
            ids[..., None].expand(-1, -1, width),
            normalized * mask[..., None].to(normalized.dtype),
        )
        counts = normalized.new_zeros(batch * layers, self.config.experts)
        counts.scatter_add_(1, ids, mask.to(normalized.dtype))
        means = dense / counts.clamp_min(1.0)[..., None]
        gathered = means.gather(1, ids[..., None].expand(-1, -1, width))
        gathered = gathered.reshape(batch, layers, horizons, candidates, width)
        gathered = gathered.permute(0, 2, 1, 3, 4)
        result = tokens + self.update(gathered) * candidate_mask[..., None].to(
            tokens.dtype
        )
        return self.feedforward(result)


class AxialCandidateReranker(nn.Module):
    """Rich set/layer/horizon scorer with a zero-initialized final residual."""

    DENSE_EVIDENCE_COUNT = 7
    BRANCH_SUPPORT_WIDTH = 6

    def __init__(
        self,
        config: HARPRTTConfig,
        expert_keys: Tensor,
        row_norms: Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        if expert_keys.shape != (
            config.layers,
            config.experts,
            config.router_rank,
        ):
            raise ValueError("reranker expert keys have the wrong shape")
        if row_norms.shape != (config.layers, config.experts):
            raise ValueError("reranker row norms have the wrong shape")
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.register_buffer("log_row_norms", row_norms.detach().float().clamp_min(1e-12).log())
        self.object_residual = nn.Parameter(
            torch.zeros(
                config.layers,
                config.experts,
                config.object_residual_width,
            )
        )
        # Dense evidence + HARP rank/margins + branch-mixture entropy.
        evidence_width = self.DENSE_EVIDENCE_COUNT + 5 + 1
        # Per lag: normalized rank, selected-set membership, and execution
        # weight.  Four summaries add streak, count, same-layer predicted
        # transition support, and previous-layer execution support.
        causal_history_width = config.route_history * 3 + 4
        token_feature_width = (
            config.router_rank
            + 1
            + config.object_residual_width
            + evidence_width
            + config.route_history
            + causal_history_width
            + self.BRANCH_SUPPORT_WIDTH
            + config.candidate_feature_width
        )
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(token_feature_width),
            nn.Linear(token_feature_width, config.reranker_width),
            nn.SiLU(),
        )
        self.context_projection = nn.Linear(config.model_width, config.reranker_width)
        self.horizon_embedding = nn.Embedding(
            config.active_horizons, config.reranker_width
        )
        self.layer_embedding = nn.Embedding(config.layers, config.reranker_width)
        self.set_blocks = nn.ModuleList(
            [
                SelfAttentionResidual(
                    config.reranker_width,
                    config.attention_heads,
                    config.reranker_ffn_width,
                    config.dropout,
                )
                for _ in range(config.reranker_set_blocks)
            ]
        )
        self.summary_blocks = nn.ModuleList(
            [_AxialSummaryBlock(config) for _ in range(config.reranker_summary_blocks)]
        )
        self.summary_broadcast = nn.Linear(config.reranker_width, config.reranker_width)
        self.same_object = _SameObjectHorizonBlock(config)
        self.residual_head = zero_linear(nn.Linear(config.reranker_width, 1))

    def _rank_features(self, scores: Tensor, candidate_ids: Tensor) -> Tensor:
        experts = scores.shape[-1]
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ordinal = torch.arange(experts, device=scores.device).expand_as(order)
        ranks.scatter_(-1, order, ordinal)
        candidate_rank = ranks.gather(-1, candidate_ids).float() / max(1, experts - 1)
        candidate_score = scores.gather(-1, candidate_ids)
        margins = []
        for boundary in (8, 16, 32, 64):
            index = min(boundary, experts) - 1
            boundary_score = torch.gather(scores, -1, order[..., index : index + 1])
            margins.append(candidate_score - boundary_score)
        return torch.stack([candidate_rank, *margins], dim=-1)

    def _history_features(
        self, history_logits: Tensor, candidate_ids: Tensor
    ) -> Tensor:
        # history [B,L,T,E] -> candidate-specific [B,H,L,C,T]
        batch, horizons, layers, candidates = candidate_ids.shape
        history = history_logits[:, None].expand(-1, horizons, -1, -1, -1)
        ids = candidate_ids[..., None, :].expand(
            -1, -1, -1, self.config.route_history, -1
        )
        gathered = history.gather(-1, ids)
        return gathered.permute(0, 1, 2, 4, 3)

    def build_causal_route_support(
        self,
        history_logits: Tensor,
        history_selected_ids: Tensor | None = None,
        history_selected_weights: Tensor | None = None,
        history_available: Tensor | None = None,
    ) -> CausalRouteSupport:
        """Build dense transition evidence without future or acceptance data."""

        config = self.config
        if history_logits.ndim != 4 or history_logits.shape[1:] != (
            config.layers,
            config.route_history,
            config.experts,
        ):
            raise ValueError("history_logits must be [B,L,T,E]")
        batch = history_logits.shape[0]
        if history_available is None:
            available = torch.ones(
                batch,
                config.layers,
                config.route_history,
                dtype=torch.bool,
                device=history_logits.device,
            )
        elif history_available.shape == (batch, config.route_history):
            available = history_available[:, None].expand(-1, config.layers, -1).bool()
        elif history_available.shape == (
            batch,
            config.layers,
            config.route_history,
        ):
            available = history_available.bool()
        else:
            raise ValueError("history_available must be [B,T] or [B,L,T]")

        count = config.exact_k
        if history_selected_ids is None:
            selected_values, selected_ids = torch.topk(
                history_logits, count, dim=-1, sorted=False
            )
            selected_weights = torch.softmax(selected_values.float(), dim=-1)
        else:
            expected = (
                batch,
                config.layers,
                config.route_history,
                count,
            )
            if history_selected_ids.shape != expected:
                raise ValueError(f"history_selected_ids must have shape {expected}")
            selected_ids = history_selected_ids.long()
            if history_selected_weights is None:
                selected_weights = torch.full(
                    expected,
                    1.0 / float(count),
                    dtype=torch.float32,
                    device=history_logits.device,
                )
            else:
                if history_selected_weights.shape != expected:
                    raise ValueError(
                        "history_selected_weights disagree with selected IDs"
                    )
                selected_weights = history_selected_weights.float()
                if not torch.isfinite(selected_weights).all() or (
                    selected_weights < 0
                ).any():
                    raise ValueError(
                        "history selected execution weights must be finite and non-negative"
                    )
                selected_weights = selected_weights / selected_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
        if (selected_ids < 0).any() or (selected_ids >= config.experts).any():
            raise ValueError("history selected expert ID lies outside the namespace")
        valid = available[..., None].to(torch.float32)
        membership = torch.zeros(
            batch,
            config.layers,
            config.route_history,
            config.experts,
            dtype=torch.float32,
            device=history_logits.device,
        )
        membership.scatter_add_(
            -1, selected_ids, torch.ones_like(selected_weights) * valid
        )
        membership.clamp_(0.0, 1.0)
        execution = torch.zeros_like(membership)
        execution.scatter_add_(-1, selected_ids, selected_weights * valid)

        current = execution[:, :, 0]
        previous_token = (
            execution[:, :, 1]
            if config.route_history > 1
            else torch.zeros_like(current)
        )
        previous_layer = torch.cat(
            [torch.zeros_like(current[:, :1]), current[:, :-1]], dim=1
        )
        lag = torch.arange(
            config.route_history, device=history_logits.device, dtype=torch.float32
        )
        decay = torch.exp(-lag / 2.0)[None, None, :, None]
        decay = decay * available[..., None].to(decay.dtype)
        recent = (execution * decay).sum(dim=2) / decay.sum(dim=2).clamp_min(1e-8)
        trend = current - previous_token
        horizons = torch.arange(
            1,
            config.active_horizons + 1,
            device=history_logits.device,
            dtype=torch.float32,
        )[None, :, None, None]
        transition = (
            recent[:, None]
            + (horizons / (horizons + 1.0)) * trend[:, None]
            + (0.5 / horizons.sqrt()) * previous_layer[:, None]
        )
        transition = transition - transition.mean(dim=-1, keepdim=True)
        return CausalRouteSupport(
            membership=membership,
            execution_weights=execution,
            previous_layer_execution=previous_layer,
            transition_scores=transition.to(history_logits.dtype),
        )

    def _causal_history_features(
        self,
        history_logits: Tensor,
        candidate_ids: Tensor,
        support: CausalRouteSupport,
        transition_scores: Tensor,
    ) -> Tensor:
        batch, horizons, layers, candidates = candidate_ids.shape
        order = torch.argsort(history_logits, dim=-1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ordinal = torch.arange(self.config.experts, device=order.device).expand_as(order)
        ranks.scatter_(-1, order, ordinal)

        def gather_history(values: Tensor) -> Tensor:
            expanded = values[:, None].expand(-1, horizons, -1, -1, -1)
            ids = candidate_ids[..., None, :].expand(
                -1, -1, -1, self.config.route_history, -1
            )
            return expanded.gather(-1, ids).permute(0, 1, 2, 4, 3)

        rank_history = gather_history(ranks.float()) / max(
            1, self.config.experts - 1
        )
        membership = gather_history(support.membership)
        execution = gather_history(support.execution_weights)
        streak = membership.cumprod(dim=-1).sum(dim=-1, keepdim=True) / float(
            self.config.route_history
        )
        recent_count = membership.mean(dim=-1, keepdim=True)
        transition = transition_scores.gather(-1, candidate_ids)[..., None]
        previous_layer = support.previous_layer_execution[:, None].expand(
            -1, horizons, -1, -1
        ).gather(-1, candidate_ids)[..., None]
        return torch.cat(
            [
                rank_history,
                membership,
                execution,
                streak,
                recent_count,
                transition,
                previous_layer,
            ],
            dim=-1,
        )

    def _branch_support_features(
        self,
        candidate_ids: Tensor,
        *,
        branch_scores: Tensor | None,
        branch_marginals: Tensor | None,
        branch_weights: Tensor | None,
        branch_mask: Tensor | None,
    ) -> Tensor:
        output_shape = candidate_ids.shape + (self.BRANCH_SUPPORT_WIDTH,)
        if branch_scores is None:
            return torch.zeros(
                output_shape, dtype=torch.float32, device=candidate_ids.device
            )
        batch, horizons, layers, candidates = candidate_ids.shape
        if branch_scores.ndim != 5 or branch_scores.shape[:3] != (
            batch,
            horizons,
            layers,
        ) or branch_scores.shape[-1] != self.config.experts:
            raise ValueError("branch_scores must be [B,H,L,N,E]")
        nodes = branch_scores.shape[3]
        expected_branch = (batch, horizons, nodes)
        if branch_marginals is None or branch_marginals.shape != branch_scores.shape:
            raise ValueError("branch_marginals must match branch_scores")
        if branch_weights is None or branch_weights.shape != expected_branch:
            raise ValueError("branch_weights must be [B,H,N]")
        if branch_mask is None or branch_mask.shape != expected_branch:
            raise ValueError("branch_mask must be [B,H,N]")
        ids = candidate_ids[:, :, :, None, :].expand(
            -1, -1, -1, nodes, -1
        )
        centered = branch_scores.float() - branch_scores.float().mean(
            dim=-1, keepdim=True
        )
        values = centered.gather(-1, ids)
        valid = branch_mask[:, :, None, :, None].bool()
        count = valid.sum(dim=3).clamp_min(1).to(values.dtype)
        mean = (values * valid).sum(dim=3) / count
        variance = (
            (values - mean[:, :, :, None]).square() * valid
        ).sum(dim=3) / count
        missing = ~branch_mask.any(dim=-1)
        safe_branch_mask = branch_mask.bool().clone()
        if missing.any():
            safe_branch_mask[..., 0] |= missing
        safe_valid = safe_branch_mask[:, :, None, :, None]
        maximum = values.masked_fill(~safe_valid, -torch.inf).amax(dim=3)
        log_mean_exp = torch.logsumexp(
            values.masked_fill(~safe_valid, -torch.inf), dim=3
        ) - count.log()
        maximum = torch.where(missing[:, :, None, None], torch.zeros_like(maximum), maximum)
        log_mean_exp = torch.where(
            missing[:, :, None, None], torch.zeros_like(log_mean_exp), log_mean_exp
        )
        weights = branch_weights.float() * branch_mask.to(torch.float32)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weighted = (values * weights[:, :, None, :, None]).sum(dim=3)
        top_ids = torch.argsort(
            branch_marginals, dim=-1, descending=True, stable=True
        )[..., : self.config.exact_k]
        supported = (
            top_ids[..., None, :] == candidate_ids[:, :, :, None, :, None]
        ).any(dim=-1)
        supporting_mass = (
            supported.to(weights.dtype) * weights[:, :, None, :, None]
        ).sum(dim=3)
        return torch.stack(
            [maximum, log_mean_exp, mean, variance, weighted, supporting_mass],
            dim=-1,
        )

    def forward(
        self,
        base_scores: Tensor,
        candidate_ids: Tensor,
        *,
        dense_evidence: Sequence[Tensor],
        history_logits: Tensor,
        endpoint_context: Tensor,
        mixture_marginals: Tensor,
        history_selected_ids: Tensor | None = None,
        history_selected_weights: Tensor | None = None,
        history_available: Tensor | None = None,
        causal_support: CausalRouteSupport | None = None,
        transition_scores: Tensor | None = None,
        branch_scores: Tensor | None = None,
        branch_marginals: Tensor | None = None,
        branch_weights: Tensor | None = None,
        branch_mask: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        extra_candidate_features: Tensor | None = None,
    ) -> RerankerOutput:
        config = self.config
        expected_ids = base_scores.shape[:-1] + (config.candidate_width,)
        if candidate_ids.shape != expected_ids:
            raise ValueError(f"candidate_ids must have shape {expected_ids}")
        if len(dense_evidence) != self.DENSE_EVIDENCE_COUNT:
            raise ValueError(
                f"reranker requires {self.DENSE_EVIDENCE_COUNT} dense evidence tensors"
            )
        if any(value.shape != base_scores.shape for value in dense_evidence):
            raise ValueError("dense reranker evidence must match base scores")
        if candidate_mask is None:
            mask = torch.ones_like(candidate_ids, dtype=torch.bool)
        else:
            if candidate_mask.shape != candidate_ids.shape:
                raise ValueError("candidate_mask disagrees with candidate IDs")
            mask = candidate_mask.bool()
        keys = gather_layer_experts(self.expert_keys, candidate_ids)
        object_residual = gather_layer_experts(self.object_residual, candidate_ids)
        row_norm = gather_layer_experts(
            self.log_row_norms[..., None], candidate_ids
        )
        evidence = torch.stack(
            [gather_dense_experts(value, candidate_ids) for value in dense_evidence],
            dim=-1,
        )
        rank_features = self._rank_features(dense_evidence[0], candidate_ids)
        probabilities = mixture_marginals / float(config.exact_k)
        entropy = -(
            probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1, keepdim=True)
        entropy = entropy / torch.log(
            torch.tensor(float(config.experts), device=entropy.device)
        )
        entropy = entropy[..., None, :].expand(
            *candidate_ids.shape, 1
        )
        history = self._history_features(history_logits, candidate_ids)
        if causal_support is None:
            causal_support = self.build_causal_route_support(
                history_logits,
                history_selected_ids,
                history_selected_weights,
                history_available,
            )
        if transition_scores is None:
            transition_scores = causal_support.transition_scores
        if transition_scores.shape != base_scores.shape:
            raise ValueError("transition_scores must match base_scores")
        causal_history = self._causal_history_features(
            history_logits,
            candidate_ids,
            causal_support,
            transition_scores,
        )
        branch_support = self._branch_support_features(
            candidate_ids,
            branch_scores=branch_scores,
            branch_marginals=branch_marginals,
            branch_weights=branch_weights,
            branch_mask=branch_mask,
        )
        parts = [
            keys,
            row_norm,
            object_residual,
            evidence,
            rank_features,
            entropy,
            history,
            causal_history,
            branch_support,
        ]
        if config.candidate_feature_width:
            expected = base_scores.shape + (config.candidate_feature_width,)
            if extra_candidate_features is None or extra_candidate_features.shape != expected:
                raise ValueError(
                    "configured dense extra_candidate_features are missing or malformed"
                )
            index = candidate_ids[..., None].expand(
                *candidate_ids.shape, config.candidate_feature_width
            )
            parts.append(extra_candidate_features.gather(-2, index))
        elif extra_candidate_features is not None:
            raise ValueError("extra candidate features were not configured")
        token_features = torch.cat(parts, dim=-1)
        tokens = self.feature_projection(token_features)
        tokens = tokens + self.context_projection(endpoint_context)[..., None, :]
        horizon_ids = torch.arange(config.active_horizons, device=tokens.device)
        layer_ids = torch.arange(config.layers, device=tokens.device)
        tokens = (
            tokens
            + self.horizon_embedding(horizon_ids)[None, :, None, None]
            + self.layer_embedding(layer_ids)[None, None, :, None]
        )
        batch, horizons, layers, candidates, width = tokens.shape
        flat = tokens.reshape(batch * horizons * layers, candidates, width)
        flat_mask = mask.reshape(batch * horizons * layers, candidates)
        safe, padding = safe_padding_mask(flat_mask)
        for block in self.set_blocks:
            flat = block(flat, padding_mask=padding)
        tokens = flat.reshape(batch, horizons, layers, candidates, width)
        mask_float = mask[..., None].to(tokens.dtype)
        summaries = (tokens * mask_float).sum(dim=3) / mask_float.sum(dim=3).clamp_min(1)
        for block in self.summary_blocks:
            summaries = block(summaries)
        tokens = tokens + self.summary_broadcast(summaries)[..., None, :]
        tokens = self.same_object(tokens, candidate_ids, mask)
        corrections = self.residual_head(tokens).squeeze(-1)
        # The exact-cardinality/geometry score path is deliberately FP32,
        # while the learned residual head runs under BF16 autocast.  Scatter
        # requires an exact dtype match between destination and source on both
        # CPU and CUDA, so cross the learned-to-exact boundary explicitly.
        corrections = (
            corrections * mask.to(corrections.dtype)
        ).to(base_scores.dtype)
        dense_corrections = torch.zeros_like(base_scores).scatter(
            -1, candidate_ids, corrections
        )
        return RerankerOutput(
            dense_scores=base_scores + dense_corrections,
            candidate_corrections=corrections,
            candidate_tokens=tokens,
            route_summaries=summaries,
            causal_history_features=causal_history,
            branch_support_features=branch_support,
        )


__all__ = ["AxialCandidateReranker", "CausalRouteSupport", "RerankerOutput"]
