"""RouteMTP v1: recurrent MTP adaptation for target-route prediction.

The native MTP remains the token/tree generator.  This module consumes a
separately replayed RouteMTP trajectory over that immutable tree and predicts
one forty-layer target-router path per node.  It also implements the causal
pre/post branch posterior and the raw captured-mass plus OTHER contract.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterator, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .exact_k import (
    cardinality_project_marginals,
    exact_k_logz_with_marginals,
    stable_topk,
)


ROUTEMTP_SCHEMA = "harp_routemtp_v1"
ANYTIME_BUDGETS = (1, 4, 8, 16)


@dataclass(frozen=True)
class RouteMTPConfig:
    hidden_width: int = 2048
    target_layers: int = 40
    experts: int = 256
    router_rank: int = 255
    exact_k: int = 8
    horizons: int = 4
    max_nodes: int = 32
    width: int = 256
    route_width: int = 64
    direct_rank: int = 16
    layer_mixture_rank: int = 8
    lora_rank: int = 32
    residual_width: int = 256
    dropout: float = 0.0

    def validate(self) -> None:
        positive = (
            "hidden_width", "target_layers", "experts", "router_rank",
            "exact_k", "horizons", "max_nodes", "width", "route_width",
            "direct_rank", "layer_mixture_rank", "lora_rank",
            "residual_width",
        )
        for name in positive:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.exact_k > self.experts:
            raise ValueError("exact_k exceeds the expert namespace")
        if self.horizons != 4:
            raise ValueError("RouteMTP v1 is frozen to H1--H4")
        if self.max_nodes < self.horizons:
            raise ValueError("max_nodes cannot represent all horizons")
        if not math.isfinite(float(self.dropout)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


class SwitchableLoRALinear(nn.Module):
    """A frozen linear map with an explicitly switchable low-rank residual.

    Native MTP calls leave the adapter disabled.  RouteMTP replay enables it
    in a scoped context, so both executions share immutable base parameters
    without sharing recurrent state or cache semantics.
    """

    def __init__(self, base: nn.Linear, rank: int, *, alpha: float | None = None) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA base must be nn.Linear")
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha if alpha is not None else rank) / float(rank)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.enabled = False
        self.base.requires_grad_(False)

    def forward(self, values: Tensor) -> Tensor:
        result = self.base(values)
        if self.enabled:
            result = result + self.up(self.down(values)) * self.scale
        return result

    @contextmanager
    def active(self, enabled: bool = True) -> Iterator[None]:
        previous = self.enabled
        self.enabled = bool(enabled)
        try:
            yield
        finally:
            self.enabled = previous


@dataclass(frozen=True)
class InstalledRouteMTPAdapters:
    fusion: SwitchableLoRALinear
    query: SwitchableLoRALinear
    output: SwitchableLoRALinear

    @contextmanager
    def active(self, enabled: bool = True) -> Iterator[None]:
        with self.fusion.active(enabled), self.query.active(enabled), self.output.active(enabled):
            yield

    def parameters(self) -> Iterator[nn.Parameter]:
        for module in (self.fusion, self.query, self.output):
            yield from module.down.parameters()
            yield from module.up.parameters()


def install_cache_coherent_adapters(mtp: nn.Module, rank: int = 32) -> InstalledRouteMTPAdapters:
    """Install fusion/Q/O LoRA while deliberately leaving K/V frozen."""

    if isinstance(mtp.fc, SwitchableLoRALinear):
        raise ValueError("RouteMTP adapters are already installed")
    attention = mtp.decoder.layers[0].self_attn
    for name in ("q_proj", "o_proj"):
        if not isinstance(getattr(attention, name, None), nn.Linear):
            raise TypeError(f"MTP attention lacks an nn.Linear {name}")
    fusion = SwitchableLoRALinear(mtp.fc, rank)
    query = SwitchableLoRALinear(attention.q_proj, rank)
    output = SwitchableLoRALinear(attention.o_proj, rank)
    mtp.fc = fusion
    attention.q_proj = query
    attention.o_proj = output
    return InstalledRouteMTPAdapters(fusion, query, output)


@dataclass(frozen=True)
class RouteMTPRouteOutput:
    scores: Tensor
    queries: Tensor
    marginals: Tensor
    selected_ids: Tensor
    hidden: Tensor
    node_summary: Tensor
    target_context: Tensor


@dataclass(frozen=True)
class RouteMTPPathOutput:
    probabilities: Tensor
    logits: Tensor
    mask: Tensor
    node_path_probabilities: Tensor
    pre_edge_correction: Tensor
    post_edge_correction: Tensor
    local_other_probabilities: Tensor


@dataclass(frozen=True)
class RouteMTPFactualOutput:
    marginals: Tensor
    projected_marginals: Tensor
    selected_ids: Tensor


@dataclass(frozen=True)
class RouteMTPOutput:
    route: RouteMTPRouteOutput
    path: RouteMTPPathOutput
    factual: RouteMTPFactualOutput


class LayerSpecificTargetBank(nn.Module):
    """DFlare-style static target-layer fusion with low-rank horizon bias."""

    def __init__(self, config: RouteMTPConfig) -> None:
        super().__init__(); config.validate(); self.config = config
        width = config.width
        self.content = nn.Linear(config.hidden_width, width)
        self.query = nn.Linear(config.router_rank, width)
        self.router = nn.Linear(config.experts, width)
        self.route_embedding = nn.Parameter(torch.empty(config.experts, config.route_width))
        self.route = nn.Linear(config.route_width, width)
        self.norms = nn.ModuleList(nn.RMSNorm(width) for _ in range(4))
        self.output = nn.Sequential(nn.RMSNorm(width), nn.Linear(width, width), nn.SiLU())
        self.layer_logits = nn.Parameter(torch.zeros(config.target_layers, config.target_layers))
        self.horizon_query = nn.Parameter(torch.empty(config.horizons, config.layer_mixture_rank))
        self.layer_horizon_key = nn.Parameter(torch.empty(
            config.target_layers, config.target_layers, config.layer_mixture_rank
        ))
        nn.init.normal_(self.route_embedding, std=0.02)
        nn.init.normal_(self.horizon_query, std=0.02)
        nn.init.zeros_(self.layer_horizon_key)

    def forward(
        self,
        *,
        current_post_layer: Tensor,
        current_queries: Tensor,
        current_centered_router_logits: Tensor,
        current_selected_ids: Tensor,
        current_selected_weights: Tensor,
    ) -> Tensor:
        config = self.config
        batch = current_post_layer.shape[0]
        if current_post_layer.shape != (batch, config.target_layers, config.hidden_width):
            raise ValueError("current post-layer bank must be [B,L,D]")
        if current_queries.shape != (batch, config.target_layers, config.router_rank):
            raise ValueError("current query bank must be [B,L,R]")
        if current_centered_router_logits.shape != (batch, config.target_layers, config.experts):
            raise ValueError("current router bank must be [B,L,E]")
        expected_route = (batch, config.target_layers, config.exact_k)
        if current_selected_ids.shape != expected_route or current_selected_weights.shape != expected_route:
            raise ValueError("current selected route must be [B,L,K]")
        weights = current_selected_weights.float()
        route = (
            self.route_embedding[current_selected_ids.long()] * weights[..., None]
        ).sum(-2)
        channels = (
            self.norms[0](self.content(current_post_layer.float())),
            self.norms[1](self.query(current_queries.float())),
            self.norms[2](self.router(current_centered_router_logits.float())),
            self.norms[3](self.route(route.float())),
        )
        projected = self.output(sum(channels))
        horizon_bias = torch.einsum(
            "ha,lja->hlj", self.horizon_query, self.layer_horizon_key
        )
        mixture = torch.softmax(self.layer_logits[None] + horizon_bias, dim=-1)
        return torch.einsum("hlj,bjw->bhlw", mixture.float(), projected.float())


class RouteMTPRouteHead(nn.Module):
    """Predict one target-layer route trajectory for each node's true depth."""

    CHANNELS = 5

    def __init__(
        self,
        config: RouteMTPConfig,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
    ) -> None:
        super().__init__(); config.validate(); self.config = config
        if expert_keys.shape != (config.target_layers, config.experts, config.router_rank):
            raise ValueError("expert-key geometry disagrees with RouteMTP config")
        if centered_bias.shape != (config.target_layers, config.experts):
            raise ValueError("centered-bias geometry disagrees with RouteMTP config")
        if rank_mask.shape != (config.target_layers, config.router_rank):
            raise ValueError("rank-mask geometry disagrees with RouteMTP config")
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.register_buffer("centered_bias", centered_bias.detach().float().clone())
        self.register_buffer("rank_mask", rank_mask.detach().bool().clone())
        self.target_bank = LayerSpecificTargetBank(config)
        self.state_projection = nn.ModuleList(
            nn.Linear(config.hidden_width, config.width) for _ in range(4)
        )
        self.router_projection = nn.Linear(config.experts, config.width)
        self.mtp_route_embedding = nn.Parameter(torch.empty(config.experts, config.route_width))
        self.mtp_route_projection = nn.Linear(config.route_width, config.width)
        self.channel_norm = nn.ModuleList(nn.RMSNorm(config.width) for _ in range(self.CHANNELS))
        self.channel_logits = nn.Parameter(torch.zeros(config.target_layers, self.CHANNELS))
        self.channel_horizon_query = nn.Parameter(torch.empty(config.horizons, config.layer_mixture_rank))
        self.channel_horizon_key = nn.Parameter(torch.empty(
            config.target_layers, self.CHANNELS, config.layer_mixture_rank
        ))
        self.layer_embedding = nn.Embedding(config.target_layers, config.width)
        self.horizon_embedding = nn.Embedding(config.horizons, config.width)
        self.hidden = nn.Sequential(
            nn.RMSNorm(config.width),
            nn.Linear(config.width, 2 * config.width),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.width, config.width),
        )
        self.query = nn.Linear(config.width, config.router_rank)
        self.direct_coefficients = nn.Linear(config.width, config.direct_rank)
        self.direct_basis = nn.Parameter(torch.empty(
            config.target_layers, config.experts, config.direct_rank
        ))
        nn.init.normal_(self.mtp_route_embedding, std=0.02)
        nn.init.normal_(self.channel_horizon_query, std=0.02)
        nn.init.zeros_(self.channel_horizon_key)
        nn.init.normal_(self.direct_basis, std=0.002)

    def forward(
        self,
        *,
        fused_state: Tensor,
        router_input: Tensor,
        post_moe_hidden: Tensor,
        vocabulary_head_input: Tensor,
        mtp_router_logits: Tensor,
        mtp_selected_ids: Tensor,
        mtp_selected_weights: Tensor,
        node_depths: Tensor,
        node_mask: Tensor,
        current_post_layer: Tensor,
        current_queries: Tensor,
        current_centered_router_logits: Tensor,
        current_selected_ids: Tensor,
        current_selected_weights: Tensor,
        use_target_bank: bool = True,
    ) -> RouteMTPRouteOutput:
        config = self.config
        batch, nodes = node_depths.shape
        state_shape = (batch, nodes, config.hidden_width)
        states = (fused_state, router_input, post_moe_hidden, vocabulary_head_input)
        if any(value.shape != state_shape for value in states):
            raise ValueError("RouteMTP state channels must be [B,N,D]")
        if mtp_router_logits.shape != (batch, nodes, config.experts):
            raise ValueError("MTP router logits must be [B,N,E]")
        if mtp_selected_ids.shape != (batch, nodes, config.exact_k):
            raise ValueError("MTP selected IDs must be [B,N,K]")
        if mtp_selected_weights.shape != mtp_selected_ids.shape:
            raise ValueError("MTP selected weights disagree with selected IDs")
        if node_mask.shape != (batch, nodes):
            raise ValueError("node mask geometry is invalid")
        if bool(((node_depths[node_mask] < 1) | (node_depths[node_mask] > config.horizons)).any()):
            raise ValueError("active node depth lies outside H1--H4")

        state_tokens = [
            norm(projection(value.float()))
            for norm, projection, value in zip(
                self.channel_norm[:4], self.state_projection, states
            )
        ]
        route = (
            self.mtp_route_embedding[mtp_selected_ids.long()]
            * mtp_selected_weights.float()[..., None]
        ).sum(-2)
        evidence = self.router_projection(mtp_router_logits.float()) + self.mtp_route_projection(route)
        state_tokens.append(self.channel_norm[4](evidence))
        tokens = torch.stack(state_tokens, dim=-2)

        channel_bias = torch.einsum(
            "ha,lca->hlc", self.channel_horizon_query, self.channel_horizon_key
        )
        channel_weight = torch.softmax(self.channel_logits[None] + channel_bias, dim=-1)
        pooled_all = torch.einsum("hlc,bncw->bnhlw", channel_weight.float(), tokens.float())
        depth_index = node_depths.clamp(1, config.horizons).long() - 1
        gather = depth_index[..., None, None, None].expand(
            batch, nodes, 1, config.target_layers, config.width
        )
        pooled = pooled_all.gather(2, gather).squeeze(2)

        target_all = self.target_bank(
            current_post_layer=current_post_layer,
            current_queries=current_queries,
            current_centered_router_logits=current_centered_router_logits,
            current_selected_ids=current_selected_ids,
            current_selected_weights=current_selected_weights,
        )
        if not use_target_bank:
            target_all = target_all * 0.0
        target = target_all[
            torch.arange(batch, device=depth_index.device)[:, None], depth_index
        ]
        layers = self.layer_embedding.weight[None, None]
        horizons = self.horizon_embedding(depth_index)[..., None, :]
        hidden = pooled + target + layers + horizons
        hidden = hidden + self.hidden(hidden)
        query = self.query(hidden) * self.rank_mask[None, None].to(hidden.dtype)
        coefficients = self.direct_coefficients(hidden)
        with torch.autocast(device_type=query.device.type, enabled=False):
            geometry = torch.einsum(
                "bnlr,ler->bnle", query.float(), self.expert_keys.float()
            ) + self.centered_bias[None, None].float()
            direct = torch.einsum(
                "bnla,lea->bnle", coefficients.float(), self.direct_basis.float()
            )
            scores = geometry + direct
            _, marginals = exact_k_logz_with_marginals(scores, config.exact_k)
        active = node_mask[..., None, None]
        scores = scores.masked_fill(~active, 0.0)
        query = query.masked_fill(~active, 0.0)
        marginals = marginals.masked_fill(~active, 0.0)
        hidden = hidden.masked_fill(~active, 0.0)
        return RouteMTPRouteOutput(
            scores=scores,
            queries=query,
            marginals=marginals,
            selected_ids=stable_topk(scores, config.exact_k),
            hidden=hidden,
            node_summary=hidden.mean(-2),
            target_context=target_all,
        )


class AnytimePathPosterior(nn.Module):
    """Causal pre-execution edge correction plus post-execution evidence."""

    def __init__(self, config: RouteMTPConfig) -> None:
        super().__init__(); config.validate(); self.config = config
        width = config.width
        self.parent = nn.Linear(config.hidden_width, width)
        self.child_token = nn.Linear(config.hidden_width, width)
        self.child = nn.Linear(config.hidden_width, width)
        self.context = nn.Linear(width, width)
        self.metadata = nn.Linear(4, width)
        self.pre = nn.Sequential(nn.RMSNorm(width), nn.SiLU(), nn.Linear(width, 1))
        self.post_statistics = nn.Linear(3, width)
        self.post = nn.Sequential(nn.RMSNorm(width), nn.SiLU(), nn.Linear(width, 1))
        self.other = nn.Parameter(torch.zeros(config.horizons, config.max_nodes))
        nn.init.zeros_(self.pre[-1].weight); nn.init.zeros_(self.pre[-1].bias)
        nn.init.zeros_(self.post[-1].weight); nn.init.zeros_(self.post[-1].bias)

    def corrections(
        self,
        *,
        recurrent_state: Tensor,
        token_embedding: Tensor,
        route_scores: Tensor,
        target_context: Tensor,
        parent_ids: Tensor,
        node_depths: Tensor,
        child_ranks: Tensor,
        node_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        config = self.config; batch, nodes, hidden = recurrent_state.shape
        if hidden != config.hidden_width or token_embedding.shape != recurrent_state.shape:
            raise ValueError("path recurrent/token states must be [B,N,D]")
        if route_scores.shape != (batch, nodes, config.target_layers, config.experts):
            raise ValueError("path route scores have invalid geometry")
        safe_parent = parent_ids.clamp_min(0).long()
        parent_state = recurrent_state.gather(
            1, safe_parent[..., None].expand(batch, nodes, hidden)
        )
        depth_index = node_depths.clamp(1, config.horizons).long() - 1
        context = target_context[
            torch.arange(batch, device=depth_index.device)[:, None], depth_index
        ].mean(-2)
        metadata = torch.stack((
            node_depths.float() / config.horizons,
            child_ranks.float() / 64.0,
            (parent_ids >= 0).float(),
            node_mask.float(),
        ), dim=-1)
        pre_hidden = (
            self.parent(parent_state.float())
            + self.child_token(token_embedding.float())
            + self.context(context.float())
            + self.metadata(metadata)
        )
        pre = self.pre(pre_hidden).squeeze(-1)
        probabilities = torch.softmax(route_scores.float(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        entropy = entropy.mean(-1) / math.log(config.experts)
        top = torch.topk(route_scores.float(), 2, dim=-1).values
        margin = (top[..., 0] - top[..., 1]).mean(-1).tanh()
        dispersion = route_scores.float().std(-1).mean(-1).tanh()
        statistics = torch.stack((entropy, margin, dispersion), dim=-1)
        post_hidden = self.child(recurrent_state.float()) + self.post_statistics(statistics)
        post = self.post(post_hidden).squeeze(-1)
        active = node_mask.bool()
        return pre.masked_fill(~active, 0.0), post.masked_fill(~active, 0.0)

    def forward(
        self,
        *,
        recurrent_state: Tensor,
        token_embedding: Tensor,
        route_scores: Tensor,
        target_context: Tensor,
        native_edge_log_probabilities: Tensor,
        parent_ids: Tensor,
        node_depths: Tensor,
        child_ranks: Tensor,
        node_mask: Tensor,
        visible_mask: Tensor,
        include_post_execution: bool = True,
    ) -> RouteMTPPathOutput:
        config = self.config; batch, nodes = node_depths.shape
        expected = (batch, nodes)
        if any(value.shape != expected for value in (
            native_edge_log_probabilities, parent_ids, child_ranks,
            node_mask, visible_mask,
        )):
            raise ValueError("path topology tensors disagree")
        if nodes > config.max_nodes:
            raise ValueError("tree exceeds RouteMTP maximum nodes")
        visible = visible_mask.bool() & node_mask.bool()
        if not bool(visible[:, 0].all()):
            raise ValueError("every anytime view must retain the exact H1 root")
        if not bool((parent_ids[:, 0] == -1).all()) or not bool((node_depths[:, 0] == 1).all()):
            raise ValueError("node zero must be the exact H1 root")
        for index in range(1, nodes):
            active = node_mask[:, index]
            if bool((parent_ids[active, index] >= index).any()):
                raise ValueError("tree is not parent-before-child")
            parent_visible = visible.gather(1, parent_ids[:, index:index + 1].clamp_min(0)).squeeze(1)
            if bool((visible[:, index] & ~parent_visible).any()):
                raise ValueError("anytime node visibility is not ancestor-closed")

        pre, post = self.corrections(
            recurrent_state=recurrent_state,
            token_embedding=token_embedding,
            route_scores=route_scores,
            target_context=target_context,
            parent_ids=parent_ids,
            node_depths=node_depths,
            child_ranks=child_ranks,
            node_mask=node_mask,
        )
        correction = pre + (post if include_post_execution else 0.0)
        edge_logits = native_edge_log_probabilities.float() + correction.float()
        conditional = edge_logits.new_zeros(batch, nodes)
        local_other = edge_logits.new_ones(batch, nodes)
        for parent_index in range(nodes):
            children = (parent_ids == parent_index) & visible
            if not bool(children.any()):
                continue
            native_mass = torch.where(
                children, native_edge_log_probabilities.float().exp(),
                torch.zeros_like(edge_logits),
            ).sum(-1)
            other_prior = (1.0 - native_mass).clamp_min(1e-12)
            depth = node_depths[:, parent_index].clamp(1, config.horizons).long() - 1
            other_correction = self.other[depth, parent_index]
            other_logit = other_prior.log() + other_correction
            child_logsum = torch.logsumexp(edge_logits.masked_fill(~children, -torch.inf), dim=-1)
            denominator = torch.logaddexp(other_logit, child_logsum)
            probabilities = torch.exp(edge_logits - denominator[:, None])
            conditional = torch.where(children, probabilities, conditional)
            local_other[:, parent_index] = torch.exp(other_logit - denominator)

        masses: list[Tensor] = [torch.ones(batch, device=edge_logits.device)]
        for index in range(1, nodes):
            previous = torch.stack(masses, dim=-1)
            parent_mass = previous.gather(
                1, parent_ids[:, index:index + 1].clamp_min(0)
            ).squeeze(1)
            masses.append(parent_mass * conditional[:, index])
        node_mass = torch.stack(masses, dim=-1) * visible.float()
        probabilities = edge_logits.new_zeros(batch, config.horizons, nodes + 1)
        masks = torch.zeros(
            batch, config.horizons, nodes + 1,
            dtype=torch.bool, device=edge_logits.device,
        )
        for horizon in range(config.horizons):
            same_depth = visible & (node_depths == horizon + 1)
            probabilities[:, horizon, :nodes] = node_mass * same_depth.float()
            probabilities[:, horizon, -1] = (
                1.0 - probabilities[:, horizon, :nodes].sum(-1)
            ).clamp_min(0.0)
            masks[:, horizon, :nodes] = same_depth
            masks[:, horizon, -1] = True
        probabilities[:, 0].zero_()
        probabilities[:, 0, 0] = 1.0
        masks[:, 0].zero_(); masks[:, 0, 0] = True; masks[:, 0, -1] = True
        logits = probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        return RouteMTPPathOutput(
            probabilities=probabilities,
            logits=logits,
            mask=masks,
            node_path_probabilities=node_mass,
            pre_edge_correction=pre,
            post_edge_correction=post,
            local_other_probabilities=local_other,
        )


def factual_marginals(
    node_marginals: Tensor,
    anchor_marginals: Tensor,
    posterior: RouteMTPPathOutput,
    node_depths: Tensor,
    visible_mask: Tensor,
    *,
    k: int = 8,
) -> RouteMTPFactualOutput:
    """Aggregate exact-k branch marginals without renormalizing captured mass."""

    batch, nodes, layers, experts = node_marginals.shape
    horizons = anchor_marginals.shape[1]
    if anchor_marginals.shape != (batch, horizons, layers, experts):
        raise ValueError("anchor marginals disagree with node marginals")
    if posterior.probabilities.shape != (batch, horizons, nodes + 1):
        raise ValueError("posterior geometry disagrees with node marginals")
    result = anchor_marginals.new_zeros(batch, horizons, layers, experts)
    for horizon in range(horizons):
        mask = visible_mask.bool() & (node_depths == horizon + 1)
        weights = posterior.probabilities[:, horizon, :nodes] * mask.float()
        result[:, horizon] = torch.einsum(
            "bn,bnle->ble", weights.float(), node_marginals.float()
        ) + posterior.probabilities[:, horizon, -1, None, None] * anchor_marginals[:, horizon].float()
    projected, _ = cardinality_project_marginals(result.float(), k)
    return RouteMTPFactualOutput(
        marginals=result,
        projected_marginals=projected,
        selected_ids=stable_topk(projected, k),
    )


class RecurrentRouteResidual(nn.Module):
    """Zero-gated route-only SwiGLU residual for the R3-Residual stage."""

    def __init__(self, hidden_width: int = 2048, intermediate_width: int = 256) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_width, intermediate_width, bias=False)
        self.up = nn.Linear(hidden_width, intermediate_width, bias=False)
        self.down = nn.Linear(intermediate_width, hidden_width, bias=False)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: Tensor) -> Tensor:
        residual = self.down(F.silu(self.gate(hidden)) * self.up(hidden))
        return hidden + torch.tanh(self.scale) * residual


class RouteMTPPredictor(nn.Module):
    """Integrated direct-route, anytime-posterior, and factual forecaster."""

    def __init__(
        self,
        config: RouteMTPConfig,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
    ) -> None:
        super().__init__(); config.validate(); self.config = config
        self.route_head = RouteMTPRouteHead(
            config, expert_keys, centered_bias, rank_mask
        )
        self.path_head = AnytimePathPosterior(config)

    def forward(
        self,
        *,
        fused_state: Tensor,
        router_input: Tensor,
        post_moe_hidden: Tensor,
        vocabulary_head_input: Tensor,
        mtp_router_logits: Tensor,
        mtp_selected_ids: Tensor,
        mtp_selected_weights: Tensor,
        recurrent_state: Tensor,
        node_token_embeddings: Tensor,
        node_parent_ids: Tensor,
        node_depths: Tensor,
        node_child_ranks: Tensor,
        node_mask: Tensor,
        visible_mask: Tensor,
        native_edge_log_probabilities: Tensor,
        current_post_layer: Tensor,
        current_queries: Tensor,
        current_centered_router_logits: Tensor,
        current_selected_ids: Tensor,
        current_selected_weights: Tensor,
        anchor_marginals: Tensor,
        include_post_execution: bool = True,
        use_target_bank: bool = True,
        path_gradient_scale: float = 1.0,
    ) -> RouteMTPOutput:
        if not 0.0 <= float(path_gradient_scale) <= 1.0:
            raise ValueError("path gradient scale must lie in [0,1]")
        route = self.route_head(
            fused_state=fused_state,
            router_input=router_input,
            post_moe_hidden=post_moe_hidden,
            vocabulary_head_input=vocabulary_head_input,
            mtp_router_logits=mtp_router_logits,
            mtp_selected_ids=mtp_selected_ids,
            mtp_selected_weights=mtp_selected_weights,
            node_depths=node_depths,
            node_mask=node_mask,
            current_post_layer=current_post_layer,
            current_queries=current_queries,
            current_centered_router_logits=current_centered_router_logits,
            current_selected_ids=current_selected_ids,
            current_selected_weights=current_selected_weights,
            use_target_bank=use_target_bank,
        )
        scale = float(path_gradient_scale)
        path_state = recurrent_state.detach() + scale * (
            recurrent_state - recurrent_state.detach()
        )
        path_scores = route.scores.detach() + scale * (
            route.scores - route.scores.detach()
        )
        path = self.path_head(
            recurrent_state=path_state,
            token_embedding=node_token_embeddings,
            route_scores=path_scores,
            target_context=route.target_context,
            native_edge_log_probabilities=native_edge_log_probabilities,
            parent_ids=node_parent_ids,
            node_depths=node_depths,
            child_ranks=node_child_ranks,
            node_mask=node_mask,
            visible_mask=visible_mask,
            include_post_execution=include_post_execution,
        )
        factual = factual_marginals(
            route.marginals,
            anchor_marginals,
            path,
            node_depths,
            visible_mask,
            k=self.config.exact_k,
        )
        return RouteMTPOutput(route=route, path=path, factual=factual)


__all__ = [
    "ANYTIME_BUDGETS",
    "ROUTEMTP_SCHEMA",
    "AnytimePathPosterior",
    "InstalledRouteMTPAdapters",
    "LayerSpecificTargetBank",
    "RecurrentRouteResidual",
    "RouteMTPConfig",
    "RouteMTPFactualOutput",
    "RouteMTPPathOutput",
    "RouteMTPOutput",
    "RouteMTPPredictor",
    "RouteMTPRouteHead",
    "RouteMTPRouteOutput",
    "SwitchableLoRALinear",
    "factual_marginals",
    "install_cache_coherent_adapters",
]
