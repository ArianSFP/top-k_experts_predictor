"""Anchor-protected HARP-DeltaTree v3 model.

The model predicts calibrated exact-set evidence and selective swaps. It does
not replace the incumbent HARP score vector with an unconstrained trajectory.
At initialization the quota policy exposes anchor top-64 and the swap policy
returns anchor top-8 bit-for-bit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .exact_k import cardinality_project_marginals, soft_cardinality_topk, stable_topk
from .model.heads import exact_projected_marginals


@dataclass(frozen=True)
class HARPDeltaConfig:
    experts: int = 256
    layers: int = 40
    horizons: int = 4
    exact_k: int = 8
    candidate_width: int = 64
    max_tree_nodes: int = 32
    router_rank: int = 255
    context_input_width: int = 256
    root_input_width: int = 256
    node_input_width: int = 256
    tree_width: int = 256
    set_width: int = 256
    ranker_width: int = 192
    tree_ffn_width: int = 512
    ranker_ffn_width: int = 384
    attention_heads: int = 8
    tree_blocks: int = 3
    ranker_blocks: int = 2
    free_rank: int = 32
    position_frequencies: int = 8
    maximum_swaps: int = 4
    swap_confidence_threshold: float = 0.65
    swap_margin: float = 0.0
    dropout: float = 0.05

    def validate(self) -> None:
        names = (
            "experts", "layers", "horizons", "exact_k", "candidate_width",
            "max_tree_nodes", "router_rank", "context_input_width",
            "root_input_width", "node_input_width", "tree_width", "set_width",
            "ranker_width", "tree_ffn_width", "ranker_ffn_width",
            "attention_heads", "tree_blocks", "ranker_blocks", "free_rank",
            "position_frequencies", "maximum_swaps",
        )
        if any(int(getattr(self, name)) < 1 for name in names):
            raise ValueError("HARP-Delta dimensions must be positive")
        if not self.exact_k <= self.candidate_width <= self.experts:
            raise ValueError("candidate width must lie in [exact_k, experts]")
        if self.maximum_swaps > self.exact_k:
            raise ValueError("maximum swaps cannot exceed exact_k")
        if self.tree_width % self.attention_heads or self.ranker_width % self.attention_heads:
            raise ValueError("attention widths must divide attention_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        if not 0.5 < self.swap_confidence_threshold < 1:
            raise ValueError("swap confidence threshold must lie in (0.5,1)")
        if not math.isfinite(self.swap_margin):
            raise ValueError("swap margin must be finite")

    def to_dict(self) -> dict[str, int | float]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class DeltaPathOutput:
    logits: Tensor
    probabilities: Tensor
    mask: Tensor


@dataclass(frozen=True)
class DeltaCandidateOutput:
    expert_ids: Tensor
    dense_mask: Tensor
    fused_scores: Tensor
    quota_logits: Tensor
    anchor_quotas: Tensor
    reliability_features: Tensor


@dataclass(frozen=True)
class DeltaRankerOutput:
    candidate_scores: Tensor
    corrections: Tensor
    swap_logits: Tensor
    swap_probabilities: Tensor
    final_ids: Tensor


@dataclass(frozen=True)
class HARPDeltaOutput:
    anchor_marginals: Tensor
    root_scores: Tensor
    root_marginals: Tensor
    root_queries: Tensor
    node_scores: Tensor
    node_marginals: Tensor
    node_queries: Tensor
    factual_path_logits: Tensor
    factual_path_posterior: Tensor
    branch_marginals: Tensor
    candidate_ids: Tensor
    candidate_mask: Tensor
    candidate_scores: Tensor
    quota_logits: Tensor
    anchor_quotas: Tensor
    swap_confidence: Tensor
    swap_logits: Tensor
    ranked_candidate_scores: Tensor
    final_ids: Tensor
    ranker_corrections: Tensor
    tree_states: Tensor
    context_states: Tensor


@dataclass(frozen=True)
class HARPDeltaSemanticOutput:
    """Only tensors consumed by the semantic-stage objective."""

    root_scores: Tensor
    root_queries: Tensor
    node_scores: Tensor
    node_queries: Tensor
    factual_path_logits: Tensor
    factual_path_posterior: Tensor
    candidate_scores: Tensor
    tree_states: Tensor
    context_states: Tensor


@dataclass(frozen=True)
class DeltaEncodedInputs:
    context_features: Tensor
    root_features: Tensor
    node_features: Tensor


def causal_position_features(positions: Tensor, frequencies: int = 8) -> Tensor:
    """Bounded absolute-position features using only the current position."""

    if positions.ndim != 1 or positions.is_floating_point():
        raise ValueError("source positions must be a one-dimensional integer tensor")
    if bool((positions < 0).any()) or frequencies < 1:
        raise ValueError("positions must be non-negative and frequencies positive")
    value = positions.float()
    maximum_log = math.log1p(1_000_000.0)
    log_scale = torch.log1p(value).clamp_max(maximum_log) / maximum_log
    bucket = torch.floor(torch.log2(value + 1.0)).clamp_max(20.0) / 20.0
    scales = torch.pow(
        torch.tensor(2.0, device=value.device),
        torch.arange(frequencies, device=value.device, dtype=torch.float32),
    )
    phase = value[:, None] / scales[None]
    return torch.cat(
        [log_scale[:, None], bucket[:, None], torch.sin(phase), torch.cos(phase)],
        dim=-1,
    )


def _ancestor_attention_mask(parent_ids: Tensor, available: Tensor, heads: int) -> Tensor:
    if parent_ids.ndim != 2 or available.shape != parent_ids.shape:
        raise ValueError("tree parent IDs and availability must be [B,N]")
    batch, nodes = parent_ids.shape
    valid = available.bool()
    positions = torch.arange(nodes, device=parent_ids.device)
    bad = valid & ((parent_ids >= positions[None]) | (parent_ids < -1))
    if bool(bad.any()):
        raise ValueError("valid tree parents must be -1 or precede their child")
    ancestors = torch.zeros(batch, nodes, nodes, dtype=torch.bool, device=parent_ids.device)
    ancestors[:, positions, positions] = valid
    for node in range(nodes):
        active = valid[:, node] & (parent_ids[:, node] >= 0)
        if not bool(active.any()):
            continue
        rows = active.nonzero(as_tuple=False).flatten()
        parents = parent_ids[rows, node]
        ancestors[rows, node] |= ancestors[rows, parents]
        ancestors[rows, node, parents] = True
    allowed = ancestors & valid[:, None, :]
    allowed[..., 0] |= ~valid
    return (~allowed)[:, None].expand(batch, heads, nodes, nodes).reshape(
        batch * heads, nodes, nodes
    )


class _DeltaTreeBlock(nn.Module):
    def __init__(self, config: HARPDeltaConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.tree_width)
        self.attention = nn.MultiheadAttention(
            config.tree_width, config.attention_heads, dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.RMSNorm(config.tree_width)
        self.ffn = nn.Sequential(
            nn.Linear(config.tree_width, config.tree_ffn_width), nn.SiLU(),
            nn.Linear(config.tree_ffn_width, config.tree_width),
        )

    def forward(self, states: Tensor, mask: Tensor, available: Tensor) -> Tensor:
        normalized = self.norm1(states)
        update, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        states = states + update
        states = states + self.ffn(self.norm2(states))
        return states * available[..., None].to(states.dtype)


class DeltaTreeEncoder(nn.Module):
    def __init__(self, config: HARPDeltaConfig) -> None:
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.node_input_width, config.tree_width)
        self.blocks = nn.ModuleList([_DeltaTreeBlock(config) for _ in range(config.tree_blocks)])
        self.norm = nn.RMSNorm(config.tree_width)

    def forward(self, features: Tensor, parent_ids: Tensor, available: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[:2] != parent_ids.shape:
            raise ValueError("node features must be [B,N,D]")
        if features.shape[-1] != self.config.node_input_width:
            raise ValueError("node feature width disagrees with DeltaTree config")
        if features.shape[1] > self.config.max_tree_nodes:
            raise ValueError("node count exceeds the adaptive-32 contract")
        mask = _ancestor_attention_mask(parent_ids.long(), available.bool(), self.config.attention_heads)
        states = self.input(features) * available[..., None].to(features.dtype)
        for block in self.blocks:
            states = block(states, mask, available.bool())
        return self.norm(states) * available[..., None].to(states.dtype)


def _straight_through_exact_marginals(scores: Tensor, k: int) -> Tensor:
    _, exact, _ = exact_projected_marginals(scores, k)
    if not torch.is_grad_enabled() or not scores.requires_grad:
        return exact
    surrogate = soft_cardinality_topk(
        scores, k, temperature=1.0, bisection_steps=32, standardize=False
    )
    return exact + (surrogate - surrogate.detach()).to(exact.dtype)


class DirectSelectedSetHead(nn.Module):
    """Layer-conditioned direct exact-set scorer with frozen router geometry."""

    def __init__(self, config: HARPDeltaConfig, expert_keys: Tensor, centered_bias: Tensor) -> None:
        super().__init__()
        if expert_keys.shape != (config.layers, config.experts, config.router_rank):
            raise ValueError("expert keys disagree with DeltaTree config")
        if centered_bias.shape != (config.layers, config.experts):
            raise ValueError("centered router bias disagrees with DeltaTree config")
        self.config = config
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.register_buffer("centered_bias", centered_bias.detach().float().clone())
        self.context = nn.Linear(config.set_width, config.set_width)
        self.branch = nn.Linear(config.tree_width, config.set_width)
        self.interaction = nn.Linear(config.set_width, config.set_width)
        self.norm = nn.RMSNorm(config.set_width)
        self.query = nn.Linear(config.set_width, config.router_rank)
        self.free_coefficients = nn.Linear(config.set_width, config.free_rank)
        self.free_basis = nn.Parameter(torch.zeros(config.layers, config.experts, config.free_rank))

    def forward(self, context: Tensor, branch_states: Tensor) -> tuple[Tensor, Tensor]:
        if context.ndim != 4 or branch_states.ndim != 3:
            raise ValueError("direct-set context/branches have invalid rank")
        u = self.context(context)[:, :, :, None]
        b = self.branch(branch_states)[:, None, None]
        joint = self.norm(u + b + self.interaction(u * b))
        query = self.query(joint)
        coefficients = self.free_coefficients(joint)
        with torch.autocast(device_type=query.device.type, enabled=False):
            geometry = torch.einsum("bhlnr,ler->bhlne", query.float(), self.expert_keys.float())
            free = torch.einsum("bhlna,lea->bhlne", coefficients.float(), self.free_basis.float())
            scores = geometry + free + self.centered_bias[None, None, :, None]
        return scores, query


class FactualPathSelector(nn.Module):
    """MTP-prior factual branch/OTHER posterior with learned corrections."""

    def __init__(self, config: HARPDeltaConfig) -> None:
        super().__init__()
        self.config = config
        self.horizon = nn.Embedding(config.horizons, config.tree_width)
        self.correction = nn.Linear(config.tree_width, 1, bias=False)
        nn.init.zeros_(self.correction.weight)
        self.other_correction = nn.Parameter(torch.zeros(config.horizons))

    def forward(self, states: Tensor, path_log_probabilities: Tensor, horizon_mask: Tensor) -> DeltaPathOutput:
        batch, nodes, width = states.shape
        expected = (batch, self.config.horizons, nodes)
        if path_log_probabilities.shape != (batch, nodes) or horizon_mask.shape != expected:
            raise ValueError("path selector input geometry is invalid")
        if width != self.config.tree_width:
            raise ValueError("path selector tree width is invalid")
        hidden = states[:, None] + self.horizon.weight[None, :, None]
        corrections = self.correction(hidden).squeeze(-1)
        visible = horizon_mask.bool()
        node_prior = path_log_probabilities[:, None].float().expand(expected)
        captured_mass = torch.where(visible, node_prior.exp(), torch.zeros_like(node_prior)).sum(-1)
        other_prior = (1.0 - captured_mass).clamp_min(1e-8).log()
        logits = torch.cat(
            [node_prior + corrections, (other_prior + self.other_correction[None])[..., None]],
            dim=-1,
        )
        mask = torch.cat(
            [visible, torch.ones(batch, self.config.horizons, 1, dtype=torch.bool, device=states.device)],
            dim=-1,
        )
        probabilities = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
        return DeltaPathOutput(logits=logits, probabilities=probabilities, mask=mask)


class AdaptiveAnchorCandidateSelector(nn.Module):
    """Calibrated marginal lift with a conservative adaptive anchor quota."""

    QUOTAS = (64, 48, 40, 32)

    def __init__(self, config: HARPDeltaConfig) -> None:
        super().__init__()
        self.config = config
        if config.candidate_width != 64:
            raise ValueError("DeltaTree v3 formal candidate width is exactly 64")
        self.gain = nn.Linear(5, 1)
        self.quota = nn.Linear(5, len(self.QUOTAS))
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)
        nn.init.zeros_(self.quota.weight)
        nn.init.zeros_(self.quota.bias)
        with torch.no_grad():
            self.quota.bias[0] = 1.0

    def _features(
        self, anchor_scores: Tensor, anchor_marginals: Tensor,
        branch_marginals: Tensor, path_posterior: Tensor,
    ) -> Tensor:
        experts = anchor_scores.shape[-1]
        order = torch.argsort(anchor_scores.float(), dim=-1, descending=True, stable=True)
        sorted_scores = anchor_scores.float().gather(-1, order)
        boundary = sorted_scores[..., 63] - sorted_scores[..., min(64, experts - 1)]
        epsilon = torch.finfo(torch.float32).eps
        anchor_logit = torch.logit(anchor_marginals.float().clamp(epsilon, 1 - epsilon))
        branch_logit = torch.logit(branch_marginals.float().clamp(epsilon, 1 - epsilon))
        lift = (branch_logit - anchor_logit).clamp_min(0)
        posterior = path_posterior.float().clamp_min(epsilon)
        entropy = -(posterior * posterior.log()).sum(-1) / math.log(max(2, posterior.shape[-1]))
        maximum = posterior.amax(-1)
        return torch.stack(
            [maximum[:, :, None].expand_as(boundary), entropy[:, :, None].expand_as(boundary),
             torch.tanh(boundary), lift.amax(-1).tanh(), lift.mean(-1).tanh()], dim=-1,
        )

    def forward(
        self, anchor_scores: Tensor, anchor_marginals: Tensor,
        branch_marginals: Tensor, path_posterior: Tensor,
        *, forced_anchor_quota: int | None = None,
    ) -> DeltaCandidateOutput:
        if anchor_scores.shape != anchor_marginals.shape or anchor_scores.shape != branch_marginals.shape:
            raise ValueError("candidate dense inputs must share geometry")
        features = self._features(anchor_scores, anchor_marginals, branch_marginals, path_posterior)
        gain = F.softplus(self.gain(features))
        epsilon = torch.finfo(torch.float32).eps
        anchor_logit = torch.logit(anchor_marginals.float().clamp(epsilon, 1 - epsilon))
        branch_logit = torch.logit(branch_marginals.float().clamp(epsilon, 1 - epsilon))
        fused = anchor_logit + gain * (branch_logit - anchor_logit).clamp_min(0)
        quota_logits = self.quota(features)
        if forced_anchor_quota is None:
            quota_values = torch.tensor(self.QUOTAS, device=anchor_scores.device)
            quotas = quota_values[torch.argmax(quota_logits, dim=-1)]
        else:
            if forced_anchor_quota not in self.QUOTAS:
                raise ValueError("forced quota is outside the declared quota policy")
            quotas = torch.full(anchor_scores.shape[:-1], forced_anchor_quota, dtype=torch.long, device=anchor_scores.device)

        experts = anchor_scores.shape[-1]
        rows = anchor_scores.numel() // experts
        anchor_order = torch.argsort(anchor_scores.float().reshape(rows, experts), dim=-1, descending=True, stable=True)
        fused_order = torch.argsort(fused.reshape(rows, experts), dim=-1, descending=True, stable=True)
        lift_flat = (branch_logit - anchor_logit).clamp_min(0).reshape(rows, experts)
        quota_flat = quotas.reshape(rows)
        ids = torch.full((rows, self.config.candidate_width), -1, dtype=torch.long, device=anchor_scores.device)
        dense = torch.zeros(rows, experts, dtype=torch.bool, device=anchor_scores.device)
        for row in range(rows):
            chosen: list[int] = []
            chosen_set: set[int] = set()
            for expert in anchor_order[row, : int(quota_flat[row])].tolist():
                chosen.append(int(expert)); chosen_set.add(int(expert))
            for expert in fused_order[row].tolist():
                expert = int(expert)
                if len(chosen) == self.config.candidate_width:
                    break
                if expert not in chosen_set and float(lift_flat[row, expert]) > 1e-7:
                    chosen.append(expert); chosen_set.add(expert)
            for expert in anchor_order[row].tolist():
                expert = int(expert)
                if len(chosen) == self.config.candidate_width:
                    break
                if expert not in chosen_set:
                    chosen.append(expert); chosen_set.add(expert)
            if len(chosen) != self.config.candidate_width:
                raise RuntimeError("adaptive candidate selector failed to fill C64")
            ids[row] = torch.tensor(chosen, device=ids.device)
            dense[row, ids[row]] = True
        leading = anchor_scores.shape[:-1]
        return DeltaCandidateOutput(
            expert_ids=ids.reshape(*leading, self.config.candidate_width),
            dense_mask=dense.reshape(*leading, experts), fused_scores=fused,
            quota_logits=quota_logits, anchor_quotas=quotas,
            reliability_features=features,
        )


def selective_swap_decode(
    anchor_scores: Tensor, candidate_ids: Tensor, candidate_scores: Tensor,
    swap_probabilities: Tensor, *, exact_k: int = 8, maximum_swaps: int = 4,
    confidence_threshold: float = 0.65, margin: float = 0.0,
) -> Tensor:
    """Return anchor top-k plus deterministic high-confidence beneficial swaps."""

    if candidate_ids.shape != candidate_scores.shape or candidate_ids.shape != swap_probabilities.shape:
        raise ValueError("candidate swap tensors must share geometry")
    if anchor_scores.shape[:-1] != candidate_ids.shape[:-1]:
        raise ValueError("anchor and candidate leading geometry differs")
    experts, width = anchor_scores.shape[-1], candidate_ids.shape[-1]
    rows = anchor_scores.numel() // experts
    anchor_flat = anchor_scores.detach().float().reshape(rows, experts)
    ids_flat = candidate_ids.long().reshape(rows, width)
    scores_flat = candidate_scores.detach().float().reshape(rows, width)
    probability_flat = swap_probabilities.detach().float().reshape(rows, width)
    anchor_ids = stable_topk(anchor_flat, exact_k)
    result = anchor_ids.clone()
    for row in range(rows):
        incumbents = [int(value) for value in anchor_ids[row].tolist()]
        incumbent_set = set(incumbents)
        score_by_id = {int(ids_flat[row, index]): float(scores_flat[row, index]) for index in range(width)}
        outsiders = [
            index for index in range(width)
            if int(ids_flat[row, index]) not in incumbent_set
            and float(probability_flat[row, index]) >= confidence_threshold
        ]
        outsiders.sort(key=lambda index: (-float(scores_flat[row, index]), int(ids_flat[row, index])))
        insiders = sorted(incumbents, key=lambda expert: (score_by_id.get(expert, float(anchor_flat[row, expert])), expert))
        selected = set(incumbent_set)
        swaps = 0
        for outsider_row in outsiders:
            if swaps >= maximum_swaps or swaps >= len(insiders):
                break
            outsider, insider = int(ids_flat[row, outsider_row]), insiders[swaps]
            outside_score = float(scores_flat[row, outsider_row])
            inside_score = score_by_id.get(insider, float(anchor_flat[row, insider]))
            if outside_score - inside_score < margin:
                continue
            selected.remove(insider); selected.add(outsider); swaps += 1
        ordered = sorted(selected, key=lambda expert: (-score_by_id.get(expert, float(anchor_flat[row, expert])), expert))
        result[row] = torch.tensor(ordered, dtype=torch.long, device=result.device)
    return result.reshape(*anchor_scores.shape[:-1], exact_k)


class SelectiveCandidateRanker(nn.Module):
    def __init__(self, config: HARPDeltaConfig, expert_keys: Tensor) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.scalar = nn.Linear(5, config.ranker_width)
        self.geometry = nn.Linear(config.router_rank, config.ranker_width, bias=False)
        self.context = nn.Linear(config.set_width, config.ranker_width)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                config.ranker_width, config.attention_heads, config.ranker_ffn_width,
                dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(config.ranker_blocks)
        ])
        self.axial = nn.TransformerEncoderLayer(
            config.ranker_width, config.attention_heads, config.ranker_ffn_width,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.correction = nn.Linear(config.ranker_width, 1)
        self.swap = nn.Linear(config.ranker_width, 1)
        nn.init.zeros_(self.correction.weight); nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.swap.weight); nn.init.zeros_(self.swap.bias)

    def forward(
        self, anchor_scores: Tensor, anchor_marginals: Tensor,
        branch_marginals: Tensor, fused_scores: Tensor,
        candidate_ids: Tensor, context: Tensor,
    ) -> DeltaRankerOutput:
        ids = candidate_ids.long()
        anchor_candidate = anchor_scores.gather(-1, ids).float()
        anchor_probability = anchor_marginals.gather(-1, ids).float()
        branch_probability = branch_marginals.gather(-1, ids).float()
        fused_candidate = fused_scores.gather(-1, ids).float()
        rank = torch.argsort(
            torch.argsort(anchor_scores.float(), dim=-1, descending=True, stable=True),
            dim=-1, stable=True,
        ).gather(-1, ids).float() / max(1, self.config.experts - 1)
        scalar = torch.stack([anchor_candidate, anchor_probability, branch_probability, fused_candidate, rank], dim=-1)
        batch, horizons, layers, candidates = ids.shape
        key_grid = self.expert_keys[None, None].expand(batch, horizons, -1, -1, -1)
        keys = key_grid.gather(
            3, ids[..., None].expand(batch, horizons, layers, candidates, self.config.router_rank)
        )
        tokens = self.scalar(scalar) + self.geometry(keys) + self.context(context)[..., None, :]
        width = tokens.shape[-1]
        flat = tokens.reshape(batch * horizons * layers, candidates, width)
        for block in self.blocks:
            flat = block(flat)
        tokens = flat.reshape(batch, horizons, layers, candidates, width)
        summary = self.axial(tokens.mean(-2).reshape(batch, horizons * layers, width))
        tokens = tokens + summary.reshape(batch, horizons, layers, width)[..., None, :]
        corrections = self.correction(tokens).squeeze(-1).float()
        swap_logits = self.swap(tokens).squeeze(-1).float()
        scores = anchor_candidate + corrections
        probabilities = torch.sigmoid(swap_logits)
        final_ids = selective_swap_decode(
            anchor_scores, ids, scores, probabilities, exact_k=self.config.exact_k,
            maximum_swaps=self.config.maximum_swaps,
            confidence_threshold=self.config.swap_confidence_threshold,
            margin=self.config.swap_margin,
        )
        return DeltaRankerOutput(scores, corrections, swap_logits, probabilities, final_ids)


class HARPDeltaInputAdapter(nn.Module):
    """Compact learned projections of the audited rich-capture channels."""

    def __init__(
        self,
        config: HARPDeltaConfig,
        *,
        raw_width: int = 2048,
        target_control_width: int = 255,
        metadata_width: int = 8,
    ) -> None:
        super().__init__()
        self.config = config
        self.raw_width = int(raw_width)
        self.target_control_width = int(target_control_width)
        self.metadata_width = int(metadata_width)
        if min(self.raw_width, self.target_control_width, self.metadata_width) < 1:
            raise ValueError("raw Delta input widths must be positive")

        self.history_current = nn.Linear(config.experts, 32)
        self.history_mean = nn.Linear(config.experts, 32)
        self.history_trend = nn.Linear(config.experts, 32)
        self.context_control = nn.Linear(self.target_control_width, 64)
        self.context_anchor = nn.Linear(config.experts, 64)
        self.context_token = nn.Linear(self.raw_width, 32)
        self.context_merge = nn.Linear(256, config.context_input_width)
        self.context_horizon = nn.Embedding(config.horizons, config.context_input_width)

        self.root_token = nn.Linear(self.raw_width, 64)
        self.root_final = nn.Linear(self.raw_width, 64)
        self.root_control = nn.Linear(self.target_control_width, 64)
        self.root_history = nn.Linear(config.experts, 64)
        self.root_merge = nn.Linear(256, config.root_input_width)

        self.node_hidden = nn.Linear(self.raw_width, 48)
        self.node_fused = nn.Linear(self.raw_width, 40)
        self.node_router_input = nn.Linear(self.raw_width, 40)
        self.node_token = nn.Linear(self.raw_width, 32)
        self.node_router_logits = nn.Linear(config.experts, 32)
        self.node_metadata = nn.Linear(self.metadata_width, 24)
        self.node_vocab = nn.Linear(self.raw_width, 24)
        self.node_vocab_statistics = nn.Linear(6, 16)
        self.node_merge = nn.Linear(256, config.node_input_width)

    def forward(
        self,
        *,
        anchor_scores: Tensor,
        route_history_logits: Tensor,
        target_control: Tensor,
        exact_token_embedding: Tensor,
        final_hidden: Tensor,
        tree_hidden: Tensor,
        tree_fused: Tensor,
        tree_router_input: Tensor,
        tree_router_logits: Tensor,
        tree_token_embeddings: Tensor,
        tree_vocab_embedding: Tensor,
        tree_vocab_statistics: Tensor,
        tree_metadata: Tensor,
    ) -> DeltaEncodedInputs:
        config = self.config
        batch = anchor_scores.shape[0]
        if anchor_scores.shape != (batch, config.horizons, config.layers, config.experts):
            raise ValueError("raw adapter anchor geometry is invalid")
        if route_history_logits.ndim != 4 or route_history_logits.shape[:2] != (
            batch, config.layers
        ) or route_history_logits.shape[-1] != config.experts:
            raise ValueError("route history must be [B,L,T,E]")
        if target_control.shape != (batch, config.layers, self.target_control_width):
            raise ValueError("target control geometry is invalid")
        if exact_token_embedding.shape != (batch, self.raw_width) or final_hidden.shape != (
            batch, self.raw_width
        ):
            raise ValueError("exact-token/final-hidden geometry is invalid")
        nodes = tree_hidden.shape[1]
        raw_node_shape = (batch, nodes, self.raw_width)
        for name, value in (
            ("tree_hidden", tree_hidden),
            ("tree_fused", tree_fused),
            ("tree_router_input", tree_router_input),
            ("tree_token_embeddings", tree_token_embeddings),
            ("tree_vocab_embedding", tree_vocab_embedding),
        ):
            if value.shape != raw_node_shape:
                raise ValueError(f"{name} geometry is invalid")
        if tree_router_logits.shape != (batch, nodes, config.experts):
            raise ValueError("tree router-logit geometry is invalid")
        if tree_metadata.shape != (batch, nodes, self.metadata_width):
            raise ValueError("tree metadata geometry is invalid")
        if tree_vocab_statistics.shape != (batch, nodes, 6):
            raise ValueError("tree vocabulary statistics geometry is invalid")

        current = route_history_logits[:, :, 0]
        mean = route_history_logits.float().mean(2)
        trend = route_history_logits[:, :, 0] - route_history_logits[:, :, -1]
        token = exact_token_embedding[:, None, None].expand(
            batch, config.horizons, config.layers, self.raw_width
        )
        context = torch.cat(
            [
                self.history_current(current)[:, None].expand(-1, config.horizons, -1, -1),
                self.history_mean(mean)[:, None].expand(-1, config.horizons, -1, -1),
                self.history_trend(trend)[:, None].expand(-1, config.horizons, -1, -1),
                self.context_control(target_control)[:, None].expand(-1, config.horizons, -1, -1),
                self.context_anchor(anchor_scores),
                self.context_token(token),
            ],
            dim=-1,
        )
        context = self.context_merge(context) + self.context_horizon.weight[None, :, None]

        root_token = self.root_token(exact_token_embedding)[:, None].expand(-1, config.layers, -1)
        root_final = self.root_final(final_hidden)[:, None].expand(-1, config.layers, -1)
        root = self.root_merge(
            torch.cat(
                [root_token, root_final, self.root_control(target_control), self.root_history(current)],
                dim=-1,
            )
        )
        node = self.node_merge(
            torch.cat(
                [
                    self.node_hidden(tree_hidden),
                    self.node_fused(tree_fused),
                    self.node_router_input(tree_router_input),
                    self.node_token(tree_token_embeddings),
                    self.node_router_logits(tree_router_logits),
                    self.node_metadata(tree_metadata),
                    self.node_vocab(tree_vocab_embedding),
                    self.node_vocab_statistics(tree_vocab_statistics),
                ],
                dim=-1,
            )
        )
        return DeltaEncodedInputs(context, root, node)


class HARPDeltaTree(nn.Module):
    """Direct-set branch proposer plus anchor-protected selective ranker."""

    def __init__(self, config: HARPDeltaConfig, expert_keys: Tensor, centered_bias: Tensor) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.tree = DeltaTreeEncoder(config)
        self.context = nn.Linear(config.context_input_width, config.set_width)
        self.root = nn.Linear(config.root_input_width, config.tree_width)
        self.position = nn.Linear(2 + 2 * config.position_frequencies, config.set_width)
        self.set_head = DirectSelectedSetHead(config, expert_keys, centered_bias)
        self.path = FactualPathSelector(config)
        self.candidates = AdaptiveAnchorCandidateSelector(config)
        self.ranker = SelectiveCandidateRanker(config, expert_keys)

    def forward_semantic(
        self, *, anchor_scores: Tensor, context_features: Tensor,
        root_features: Tensor, node_features: Tensor, node_parent_ids: Tensor,
        node_available: Tensor, node_path_log_probabilities: Tensor,
        node_horizon_mask: Tensor, source_positions: Tensor,
    ) -> HARPDeltaSemanticOutput:
        config = self.config
        batch = context_features.shape[0]
        expected_anchor = (batch, config.horizons, config.layers, config.experts)
        if anchor_scores.shape != expected_anchor:
            raise ValueError("anchor scores disagree with DeltaTree production geometry")
        if context_features.shape != expected_anchor[:-1] + (config.context_input_width,):
            raise ValueError("context features have invalid geometry")
        if root_features.shape != (batch, config.layers, config.root_input_width):
            raise ValueError("H1 root features have invalid geometry")
        if node_features.shape[:2] != node_parent_ids.shape or node_available.shape != node_parent_ids.shape:
            raise ValueError("node features and topology disagree")
        if node_horizon_mask.shape != (batch, config.horizons, node_features.shape[1]):
            raise ValueError("node horizon mask has invalid geometry")
        if source_positions.shape != (batch,):
            raise ValueError("source positions must be [B]")

        positions = causal_position_features(source_positions, config.position_frequencies)
        context = self.context(context_features) + self.position(positions)[:, None, None]
        nodes = self.tree(node_features, node_parent_ids, node_available)
        root = self.root(root_features)
        node_scores, node_queries = self.set_head(context, nodes)
        root_grid, root_query_grid = self.set_head(context[:, :1], root)
        layer_ids = torch.arange(config.layers, device=anchor_scores.device)
        root_scores = root_grid[:, 0, layer_ids, layer_ids]
        root_queries = root_query_grid[:, 0, layer_ids, layer_ids]
        path = self.path(nodes, node_path_log_probabilities, node_horizon_mask)
        return HARPDeltaSemanticOutput(
            root_scores=root_scores,
            root_queries=root_queries,
            node_scores=node_scores,
            node_queries=node_queries,
            factual_path_logits=path.logits,
            factual_path_posterior=path.probabilities,
            # The semantic loss needs only the future leading geometry here.
            candidate_scores=anchor_scores[..., :1],
            tree_states=nodes,
            context_states=context,
        )

    def forward(
        self, *, anchor_scores: Tensor, context_features: Tensor,
        root_features: Tensor, node_features: Tensor, node_parent_ids: Tensor,
        node_available: Tensor, node_path_log_probabilities: Tensor,
        node_horizon_mask: Tensor, source_positions: Tensor,
        anchor_marginals: Tensor | None = None,
        forced_anchor_quota: int | None = None,
        semantic_only: bool = False,
    ) -> HARPDeltaOutput | HARPDeltaSemanticOutput:
        config = self.config
        semantic = self.forward_semantic(
            anchor_scores=anchor_scores,
            context_features=context_features,
            root_features=root_features,
            node_features=node_features,
            node_parent_ids=node_parent_ids,
            node_available=node_available,
            node_path_log_probabilities=node_path_log_probabilities,
            node_horizon_mask=node_horizon_mask,
            source_positions=source_positions,
        )
        if semantic_only:
            return semantic
        if anchor_marginals is None:
            with torch.no_grad():
                _, anchor_marginals, _ = exact_projected_marginals(
                    anchor_scores.float(), config.exact_k
                )
        elif anchor_marginals.shape != anchor_scores.shape:
            raise ValueError("anchor marginal geometry differs from scores")
        anchor_marginals = anchor_marginals.detach().float()
        root_scores = semantic.root_scores
        root_queries = semantic.root_queries
        node_scores = semantic.node_scores
        node_queries = semantic.node_queries
        path_logits = semantic.factual_path_logits
        path_probabilities = semantic.factual_path_posterior
        nodes = semantic.tree_states
        context = semantic.context_states
        node_marginals = _straight_through_exact_marginals(
            node_scores, config.exact_k
        )
        root_marginals = _straight_through_exact_marginals(
            root_scores, config.exact_k
        )
        with torch.autocast(device_type=node_scores.device.type, enabled=False):
            captured = torch.einsum(
                "bhn,bhlne->bhle",
                path_probabilities[..., :-1].float(),
                node_marginals.float(),
            )
        branch = (
            captured
            + path_probabilities[..., -1][:, :, None, None] * anchor_marginals
        )
        branch = branch.clone(); branch[:, 0] = root_marginals
        branch, _ = cardinality_project_marginals(branch, config.exact_k)
        candidate = self.candidates(
            anchor_scores, anchor_marginals, branch, path_probabilities,
            forced_anchor_quota=forced_anchor_quota,
        )
        ranked = self.ranker(
            anchor_scores, anchor_marginals, branch, candidate.fused_scores,
            candidate.expert_ids, context,
        )
        return HARPDeltaOutput(
            anchor_marginals=anchor_marginals, root_scores=root_scores,
            root_marginals=root_marginals, root_queries=root_queries,
            node_scores=node_scores, node_marginals=node_marginals,
            node_queries=node_queries, factual_path_logits=path_logits,
            factual_path_posterior=path_probabilities, branch_marginals=branch,
            candidate_ids=candidate.expert_ids, candidate_mask=candidate.dense_mask,
            candidate_scores=candidate.fused_scores, quota_logits=candidate.quota_logits,
            anchor_quotas=candidate.anchor_quotas,
            swap_confidence=ranked.swap_probabilities,
            swap_logits=ranked.swap_logits,
            ranked_candidate_scores=ranked.candidate_scores,
            final_ids=ranked.final_ids,
            ranker_corrections=ranked.corrections, tree_states=nodes,
            context_states=context,
        )


class HARPDeltaTeacher(nn.Module):
    """Production raw-capture adapter followed by the protected DeltaTree core."""

    def __init__(
        self,
        config: HARPDeltaConfig,
        expert_keys: Tensor,
        centered_bias: Tensor,
        *,
        raw_width: int = 2048,
        target_control_width: int = 255,
        metadata_width: int = 8,
    ) -> None:
        super().__init__()
        self.config = config
        self.adapter = HARPDeltaInputAdapter(
            config,
            raw_width=raw_width,
            target_control_width=target_control_width,
            metadata_width=metadata_width,
        )
        self.core = HARPDeltaTree(config, expert_keys, centered_bias)

    def forward(
        self,
        *,
        anchor_scores: Tensor,
        route_history_logits: Tensor,
        target_control: Tensor,
        exact_token_embedding: Tensor,
        final_hidden: Tensor,
        tree_hidden: Tensor,
        tree_fused: Tensor,
        tree_router_input: Tensor,
        tree_router_logits: Tensor,
        tree_token_embeddings: Tensor,
        tree_vocab_embedding: Tensor,
        tree_vocab_statistics: Tensor,
        tree_metadata: Tensor,
        node_parent_ids: Tensor,
        node_available: Tensor,
        node_path_log_probabilities: Tensor,
        node_horizon_mask: Tensor,
        source_positions: Tensor,
        anchor_marginals: Tensor | None = None,
        forced_anchor_quota: int | None = None,
        semantic_only: bool = False,
    ) -> HARPDeltaOutput | HARPDeltaSemanticOutput:
        encoded = self.adapter(
            anchor_scores=anchor_scores,
            route_history_logits=route_history_logits,
            target_control=target_control,
            exact_token_embedding=exact_token_embedding,
            final_hidden=final_hidden,
            tree_hidden=tree_hidden,
            tree_fused=tree_fused,
            tree_router_input=tree_router_input,
            tree_router_logits=tree_router_logits,
            tree_token_embeddings=tree_token_embeddings,
            tree_vocab_embedding=tree_vocab_embedding,
            tree_vocab_statistics=tree_vocab_statistics,
            tree_metadata=tree_metadata,
        )
        return self.core(
            anchor_scores=anchor_scores,
            context_features=encoded.context_features,
            root_features=encoded.root_features,
            node_features=encoded.node_features,
            node_parent_ids=node_parent_ids,
            node_available=node_available,
            node_path_log_probabilities=node_path_log_probabilities,
            node_horizon_mask=node_horizon_mask,
            source_positions=source_positions,
            anchor_marginals=anchor_marginals,
            forced_anchor_quota=forced_anchor_quota,
            semantic_only=semantic_only,
        )


__all__ = [
    "AdaptiveAnchorCandidateSelector", "DeltaCandidateOutput", "DeltaEncodedInputs", "DeltaPathOutput",
    "DeltaRankerOutput", "DeltaTreeEncoder", "DirectSelectedSetHead",
    "FactualPathSelector", "HARPDeltaConfig", "HARPDeltaInputAdapter",
    "HARPDeltaOutput", "HARPDeltaSemanticOutput", "HARPDeltaTeacher", "HARPDeltaTree",
    "SelectiveCandidateRanker", "causal_position_features", "selective_swap_decode",
]
