"""Layerwise router-state trajectory model for HARP-DeltaRoute v4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .exact_k import soft_cardinality_topk, stable_topk


@dataclass(frozen=True)
class DeltaRouteConfig:
    experts: int = 256
    layers: int = 40
    horizons: int = 4
    nodes: int = 32
    router_rank: int = 255
    raw_width: int = 2048
    metadata_width: int = 8
    latent_width: int = 256
    effect_width: int = 64
    transition_width: int = 512
    layer_adapter_rank: int = 8
    free_rank: int = 16
    attention_heads: int = 4
    exact_k: int = 8
    dropout: float = 0.05

    def validate(self) -> None:
        dimensions = (
            self.experts, self.layers, self.horizons, self.nodes,
            self.router_rank, self.raw_width, self.metadata_width,
            self.latent_width, self.effect_width, self.transition_width,
            self.layer_adapter_rank, self.free_rank, self.attention_heads,
            self.exact_k,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("DeltaRoute dimensions must be positive")
        if self.latent_width % self.attention_heads:
            raise ValueError("latent width must divide attention heads")
        if not self.exact_k <= self.experts:
            raise ValueError("exact_k exceeds expert namespace")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")

    def to_dict(self) -> dict[str, int | float]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class RouteDistribution:
    selected_ids: Tensor
    selected_weights: Tensor
    dense_straight_through_weights: Tensor


@dataclass(frozen=True)
class RouteRolloutOutput:
    queries: Tensor
    scores: Tensor
    selected_ids: Tensor
    selected_weights: Tensor
    dense_route_weights: Tensor


@dataclass(frozen=True)
class DeltaRouteTrajectoryOutput:
    queries: Tensor
    scores: Tensor
    selected_ids: Tensor
    selected_weights: Tensor
    pooled_channels: Tensor
    path_states: Tensor


def predicted_route_distribution(scores: Tensor, *, k: int = 8) -> RouteDistribution:
    """Native TopK execution in forward, smooth exact-cardinality VJP backward."""

    ids = stable_topk(scores.float(), k)
    probabilities = torch.softmax(scores.float(), dim=-1)
    selected = probabilities.gather(-1, ids)
    weights = selected / selected.sum(-1, keepdim=True).clamp_min(1e-12)
    hard = torch.zeros_like(scores, dtype=torch.float32)
    hard.scatter_(-1, ids, weights)
    if torch.is_grad_enabled() and scores.requires_grad:
        membership = soft_cardinality_topk(
            scores, k, temperature=1.0, bisection_steps=32, standardize=False
        )
        soft = probabilities * membership
        soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-12)
        dense = hard + soft - soft.detach()
    else:
        dense = hard
    return RouteDistribution(ids, weights, dense)


class LayerConditionedChannelPool(nn.Module):
    """Keep seven causal source channels separate until layer conditioning."""

    CHANNELS = 7

    def __init__(self, config: DeltaRouteConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.raw = nn.ModuleList(
            nn.Linear(config.raw_width, config.latent_width) for _ in range(5)
        )
        self.router_logits = nn.Linear(config.experts, config.latent_width)
        self.metadata = nn.Linear(config.metadata_width, config.latent_width)
        self.channel_embedding = nn.Embedding(self.CHANNELS, config.latent_width)
        self.layer = nn.Embedding(config.layers, config.latent_width)
        self.horizon = nn.Embedding(config.horizons, config.latent_width)
        self.query = nn.Linear(config.latent_width, config.latent_width, bias=False)
        self.key = nn.Linear(config.latent_width, config.latent_width, bias=False)
        self.value = nn.Linear(config.latent_width, config.latent_width, bias=False)
        self.output = nn.Linear(config.latent_width, config.latent_width)
        self.path_cell = nn.GRUCell(config.latent_width, config.latent_width)

    def _tokens(
        self,
        *,
        fused: Tensor,
        post_ffn: Tensor,
        router_input: Tensor,
        vocabulary: Tensor,
        token: Tensor,
        router_logits: Tensor,
        metadata: Tensor,
    ) -> Tensor:
        raw_values = (fused, post_ffn, router_input, vocabulary, token)
        if any(value.shape != fused.shape for value in raw_values):
            raise ValueError("raw route channels must share [B,N,D] geometry")
        batch, nodes, width = fused.shape
        if width != self.config.raw_width:
            raise ValueError("raw route channel width differs from configuration")
        if router_logits.shape != (batch, nodes, self.config.experts):
            raise ValueError("MTP router logits have invalid geometry")
        if metadata.shape != (batch, nodes, self.config.metadata_width):
            raise ValueError("route metadata has invalid geometry")
        tokens = [projection(value) for projection, value in zip(self.raw, raw_values, strict=True)]
        tokens.extend((self.router_logits(router_logits), self.metadata(metadata)))
        result = torch.stack(tokens, dim=2)
        return result + self.channel_embedding.weight[None, None]

    def _path_states(self, tokens: Tensor, parents: Tensor, available: Tensor) -> Tensor:
        batch, nodes, _, width = tokens.shape
        if parents.shape != (batch, nodes) or available.shape != (batch, nodes):
            raise ValueError("path topology must be [B,N]")
        source = tokens.mean(2)
        states: list[Tensor] = []
        zero = torch.zeros(batch, width, device=tokens.device, dtype=tokens.dtype)
        for node in range(nodes):
            parent = parents[:, node].long()
            if bool((available[:, node] & ((parent >= node) | (parent < -1))).any()):
                raise ValueError("available route parent must precede child")
            if node == 0:
                inherited = zero
            else:
                stacked = torch.stack(states, dim=1)
                inherited = stacked.gather(
                    1, parent.clamp_min(0)[:, None, None].expand(batch, 1, width)
                ).squeeze(1)
                inherited = torch.where((parent >= 0)[:, None], inherited, zero)
            state = self.path_cell(source[:, node], inherited)
            state = state * available[:, node, None].to(state.dtype)
            states.append(state)
        return torch.stack(states, dim=1)

    def forward(
        self,
        *,
        fused: Tensor,
        post_ffn: Tensor,
        router_input: Tensor,
        vocabulary: Tensor,
        token: Tensor,
        router_logits: Tensor,
        metadata: Tensor,
        parents: Tensor,
        available: Tensor,
    ) -> tuple[Tensor, Tensor]:
        tokens = self._tokens(
            fused=fused, post_ffn=post_ffn, router_input=router_input,
            vocabulary=vocabulary, token=token, router_logits=router_logits,
            metadata=metadata,
        )
        path = self._path_states(tokens, parents, available.bool())
        config = self.config
        query = self.layer.weight[None] + self.horizon.weight[:, None]
        query = self.query(query).reshape(
            config.horizons, config.layers, config.attention_heads,
            config.latent_width // config.attention_heads,
        )
        key = self.key(tokens).reshape(
            tokens.shape[0], tokens.shape[1], self.CHANNELS,
            config.attention_heads, config.latent_width // config.attention_heads,
        )
        value = self.value(tokens).reshape_as(key)
        attention = torch.einsum("hlad,bncad->bhlnac", query, key)
        attention = attention / math.sqrt(config.latent_width // config.attention_heads)
        attention = torch.softmax(attention.float(), dim=-1).to(value.dtype)
        pooled = torch.einsum("bhlnac,bncad->bhlnad", attention, value)
        pooled = self.output(pooled.flatten(-2))
        pooled = pooled + path[:, None, None]
        pooled = pooled * available[:, None, None, :, None].to(pooled.dtype)
        return pooled, path


class RouteDynamicsCore(nn.Module):
    """Shared adjacent-layer transition with execution-weight feedback."""

    def __init__(self, config: DeltaRouteConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        transitions = config.layers - 1
        self.control_diagonal = nn.Parameter(
            torch.ones(transitions, config.router_rank)
        )
        self.control_down = nn.Parameter(
            torch.zeros(transitions, config.router_rank, config.layer_adapter_rank)
        )
        self.control_up = nn.Parameter(
            torch.zeros(transitions, config.layer_adapter_rank, config.router_rank)
        )
        self.control_bias = nn.Parameter(torch.zeros(transitions, config.router_rank))
        self.expert_base = nn.Parameter(
            torch.zeros(transitions, config.experts, config.effect_width)
        )
        self.expert_modulation = nn.Parameter(
            torch.zeros(transitions, config.experts, config.effect_width)
        )
        self.query_effect = nn.Linear(config.router_rank, config.effect_width)
        self.query_input = nn.Linear(config.router_rank, config.latent_width)
        self.effect_input = nn.Linear(config.effect_width, config.latent_width)
        self.context_input = nn.Linear(config.latent_width, config.latent_width)
        self.layer = nn.Embedding(transitions, config.latent_width)
        self.transition = nn.Sequential(
            nn.RMSNorm(config.latent_width),
            nn.Linear(config.latent_width, config.transition_width),
            nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(config.transition_width, config.latent_width),
        )
        self.query_output = nn.Linear(config.latent_width, config.router_rank)
        self.residual_gate = nn.Parameter(torch.zeros(transitions, 1))

    def affine_control(self, queries: Tensor) -> Tensor:
        """Layer-conditioned query-only control used as the R0 ablation."""

        if queries.shape[-2:] != (self.config.layers, self.config.router_rank):
            raise ValueError("affine-control queries must end in [L,R]")
        current = queries[..., :-1, :].float()
        control = current * self.control_diagonal
        low = torch.einsum("...lr,lra->...la", current, self.control_down)
        control = control + torch.einsum("...la,lar->...lr", low, self.control_up)
        return control + self.control_bias

    def step(
        self,
        current_query: Tensor,
        selected_ids: Tensor,
        selected_weights: Tensor,
        causal_context: Tensor,
        *,
        layer: int,
        dense_route_weights: Tensor | None = None,
    ) -> Tensor:
        config = self.config
        if not 0 <= layer < config.layers - 1:
            raise ValueError("transition layer lies outside 0..L-2")
        if current_query.shape[-1] != config.router_rank:
            raise ValueError("current query has invalid rank")
        if selected_ids.shape != selected_weights.shape or selected_ids.shape[:-1] != current_query.shape[:-1]:
            raise ValueError("selected routes disagree with current query")
        if selected_ids.shape[-1] != config.exact_k:
            raise ValueError("selected route does not contain exact_k experts")
        if causal_context.shape != current_query.shape[:-1] + (config.latent_width,):
            raise ValueError("causal transition context has invalid geometry")
        if bool(((selected_ids < 0) | (selected_ids >= config.experts)).any()):
            raise ValueError("selected expert ID lies outside namespace")
        state_effect = self.query_effect(current_query.float())
        if dense_route_weights is None:
            normalized_weights = selected_weights.float()
            normalized_weights = normalized_weights / normalized_weights.sum(
                -1, keepdim=True
            ).clamp_min(1e-12)
            base = self.expert_base[layer][selected_ids.long()]
            modulation = self.expert_modulation[layer][selected_ids.long()]
            message = (
                (base + modulation * state_effect[..., None, :])
                * normalized_weights[..., None]
            ).sum(-2)
        else:
            if dense_route_weights.shape != current_query.shape[:-1] + (config.experts,):
                raise ValueError("dense route weights disagree with current query")
            dense = dense_route_weights.float()
            if not torch.isfinite(dense).all():
                raise ValueError("dense route weights contain NaN or Inf")
            dense = dense / dense.sum(-1, keepdim=True).clamp_min(1e-12)
            effects = self.expert_base[layer] + (
                self.expert_modulation[layer]
                * state_effect[..., None, :]
            )
            message = torch.einsum("...e,...ed->...d", dense, effects)
        hidden = (
            self.query_input(current_query.float())
            + self.effect_input(message)
            + self.context_input(causal_context)
            + self.layer.weight[layer]
        )
        nonlinear = self.query_output(self.transition(hidden))
        control = current_query.float() * self.control_diagonal[layer]
        low = torch.einsum("...r,ra->...a", current_query.float(), self.control_down[layer])
        control = control + torch.einsum("...a,ar->...r", low, self.control_up[layer])
        control = control + self.control_bias[layer]
        return control + self.residual_gate[layer] * nonlinear

    def teacher_forced(
        self,
        queries: Tensor,
        selected_ids: Tensor,
        selected_weights: Tensor,
        causal_context: Tensor,
    ) -> Tensor:
        """Training-only one-step predictions from true layer states/routes."""

        if queries.ndim < 3 or queries.shape[-2:] != (
            self.config.layers, self.config.router_rank
        ):
            raise ValueError("teacher queries must end in [L,R]")
        if selected_ids.shape != queries.shape[:-1] + (self.config.exact_k,):
            raise ValueError("teacher selected IDs have invalid geometry")
        if selected_weights.shape != selected_ids.shape:
            raise ValueError("teacher selected weights have invalid geometry")
        if causal_context.shape != queries.shape[:-1] + (self.config.latent_width,):
            raise ValueError("teacher causal context has invalid geometry")
        predictions = [
            self.step(
                queries[..., layer, :], selected_ids[..., layer, :],
                selected_weights[..., layer, :], causal_context[..., layer, :],
                layer=layer,
            )
            for layer in range(self.config.layers - 1)
        ]
        return torch.stack(predictions, dim=-2)

    def rollout(
        self,
        seed_query: Tensor,
        causal_context: Tensor,
        expert_keys: Tensor,
        centered_bias: Tensor,
        *,
        teacher_ids: Tensor | None = None,
        teacher_weights: Tensor | None = None,
        teacher_force_mask: Tensor | None = None,
    ) -> RouteRolloutOutput:
        """Closed-loop layer scan; targets are optional training controls only."""

        config = self.config
        leading = seed_query.shape[:-1]
        if seed_query.shape[-1] != config.router_rank:
            raise ValueError("rollout seed query has invalid rank")
        if causal_context.shape != leading + (config.layers, config.latent_width):
            raise ValueError("rollout causal context must end in [L,W]")
        if expert_keys.shape != (config.layers, config.experts, config.router_rank):
            raise ValueError("rollout expert keys have invalid geometry")
        if centered_bias.shape != (config.layers, config.experts):
            raise ValueError("rollout centered bias has invalid geometry")
        controls = (teacher_ids, teacher_weights, teacher_force_mask)
        if any(value is not None for value in controls) and not all(
            value is not None for value in controls
        ):
            raise ValueError("teacher route controls must be provided together")
        if teacher_ids is not None:
            expected = leading + (config.layers, config.exact_k)
            if teacher_ids.shape != expected or teacher_weights.shape != expected:
                raise ValueError("teacher rollout routes have invalid geometry")
            if teacher_force_mask.shape != leading + (config.layers - 1,):
                raise ValueError("teacher-force mask has invalid geometry")

        query = seed_query.float()
        queries: list[Tensor] = []
        scores: list[Tensor] = []
        ids: list[Tensor] = []
        weights: list[Tensor] = []
        dense_weights: list[Tensor] = []
        for layer in range(config.layers):
            score = torch.einsum(
                "...r,er->...e", query, expert_keys[layer].float()
            ) + centered_bias[layer].float()
            route = predicted_route_distribution(score, k=config.exact_k)
            queries.append(query); scores.append(score)
            ids.append(route.selected_ids); weights.append(route.selected_weights)
            dense_weights.append(route.dense_straight_through_weights)
            if layer == config.layers - 1:
                continue
            chosen_ids, chosen_weights = route.selected_ids, route.selected_weights
            chosen_dense = route.dense_straight_through_weights
            if teacher_ids is not None:
                force = teacher_force_mask[..., layer, None].bool()
                chosen_ids = torch.where(force, teacher_ids[..., layer, :].long(), chosen_ids)
                chosen_weights = torch.where(
                    force, teacher_weights[..., layer, :].float(), chosen_weights
                )
                teacher_dense = torch.zeros_like(chosen_dense)
                teacher_dense.scatter_(
                    -1,
                    teacher_ids[..., layer, :].long(),
                    teacher_weights[..., layer, :].float(),
                )
                chosen_dense = torch.where(force, teacher_dense, chosen_dense)
            query = self.step(
                query, chosen_ids, chosen_weights,
                causal_context[..., layer, :], layer=layer,
                dense_route_weights=chosen_dense,
            )
        return RouteRolloutOutput(
            torch.stack(queries, dim=-2), torch.stack(scores, dim=-2),
            torch.stack(ids, dim=-2), torch.stack(weights, dim=-2),
            torch.stack(dense_weights, dim=-2),
        )


class DeltaRouteTrajectory(nn.Module):
    """Causal MTP initializer plus parent-residualized router rollout."""

    def __init__(
        self,
        config: DeltaRouteConfig,
        expert_keys: Tensor,
        centered_bias: Tensor,
        rank_mask: Tensor,
    ) -> None:
        super().__init__()
        config.validate()
        if expert_keys.shape != (config.layers, config.experts, config.router_rank):
            raise ValueError("expert keys disagree with DeltaRoute config")
        if centered_bias.shape != (config.layers, config.experts):
            raise ValueError("centered bias disagrees with DeltaRoute config")
        if rank_mask.shape != (config.layers, config.router_rank):
            raise ValueError("rank mask disagrees with DeltaRoute config")
        self.config = config
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.register_buffer("centered_bias", centered_bias.detach().float().clone())
        self.register_buffer("rank_mask", rank_mask.detach().bool().clone())
        self.channels = LayerConditionedChannelPool(config)
        self.context = nn.Linear(config.latent_width, config.latent_width)
        self.seed = nn.Sequential(
            nn.RMSNorm(config.latent_width),
            nn.Linear(config.latent_width, config.transition_width), nn.SiLU(),
            nn.Linear(config.transition_width, config.router_rank),
        )
        self.seed_gate = nn.Parameter(torch.zeros(config.horizons, 1, 1))
        self.dynamics = RouteDynamicsCore(config)
        self.rollout_gate = nn.Parameter(
            torch.zeros(config.horizons, config.layers, 1, 1)
        )
        self.free_coefficients = nn.Linear(config.latent_width, config.free_rank)
        self.free_basis = nn.Parameter(
            torch.zeros(config.layers, config.experts, config.free_rank)
        )
        self.free_gate = nn.Parameter(
            torch.zeros(config.horizons, config.layers, 1, 1)
        )

    def causal_context(
        self,
        *,
        context_states: Tensor,
        fused: Tensor,
        post_ffn: Tensor,
        router_input: Tensor,
        vocabulary: Tensor,
        token: Tensor,
        router_logits: Tensor,
        metadata: Tensor,
        parents: Tensor,
        available: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return layer-conditioned causal context before any target labels."""

        config = self.config
        batch = fused.shape[0]
        if context_states.shape != (
            batch, config.horizons, config.layers, config.latent_width
        ):
            raise ValueError("trajectory context states have invalid geometry")
        pooled, path = self.channels(
            fused=fused, post_ffn=post_ffn, router_input=router_input,
            vocabulary=vocabulary, token=token, router_logits=router_logits,
            metadata=metadata, parents=parents, available=available,
        )
        context = pooled + self.context(context_states)[:, :, :, None]
        return context, pooled, path

    def forward(
        self,
        *,
        parent_queries: Tensor,
        parent_scores: Tensor,
        context_states: Tensor,
        fused: Tensor,
        post_ffn: Tensor,
        router_input: Tensor,
        vocabulary: Tensor,
        token: Tensor,
        router_logits: Tensor,
        metadata: Tensor,
        parents: Tensor,
        available: Tensor,
    ) -> DeltaRouteTrajectoryOutput:
        config = self.config
        batch, horizons, layers, nodes, rank = parent_queries.shape
        if (horizons, layers, nodes, rank) != (
            config.horizons, config.layers, config.nodes, config.router_rank
        ):
            raise ValueError("parent query geometry differs from DeltaRoute config")
        if parent_scores.shape != (batch, horizons, layers, nodes, config.experts):
            raise ValueError("parent score geometry differs from parent queries")
        context, pooled, path = self.causal_context(
            context_states=context_states, fused=fused, post_ffn=post_ffn,
            router_input=router_input, vocabulary=vocabulary, token=token,
            router_logits=router_logits, metadata=metadata, parents=parents,
            available=available,
        )
        seed_delta = self.seed(context[:, :, 0])
        seed = parent_queries[:, :, 0].float() + self.seed_gate[None] * seed_delta
        flat_seed = seed.permute(0, 1, 2, 3).reshape(batch, horizons, nodes, rank)
        rollout = self.dynamics.rollout(
            flat_seed,
            context.permute(0, 1, 3, 2, 4),
            self.expert_keys, self.centered_bias,
        )
        proposed = rollout.queries.permute(0, 1, 3, 2, 4)
        queries = parent_queries.float() + self.rollout_gate[None] * (
            proposed - parent_queries.float()
        )
        queries = queries * self.rank_mask[None, None, :, None]
        geometry = torch.einsum(
            "bhlnr,ler->bhlne", queries - parent_queries.float(),
            self.expert_keys.float(),
        )
        coefficients = self.free_coefficients(context)
        free = torch.einsum(
            "bhlna,lea->bhlne", coefficients.float(), self.free_basis.float()
        )
        scores = parent_scores.float() + geometry + self.free_gate[None] * free
        route = predicted_route_distribution(scores, k=config.exact_k)
        return DeltaRouteTrajectoryOutput(
            queries, scores, route.selected_ids, route.selected_weights,
            pooled, path,
        )


def centered_logit_huber(
    predicted_queries: Tensor,
    target_queries: Tensor,
    expert_keys: Tensor,
    *,
    delta: float = 1.0,
) -> Tensor:
    """Query loss in the router-induced metric rather than raw coordinates."""

    if predicted_queries.shape != target_queries.shape:
        raise ValueError("predicted and target query geometry differs")
    if predicted_queries.shape[-2:] != (expert_keys.shape[0], expert_keys.shape[-1]):
        raise ValueError("query layer/rank axes disagree with expert keys")
    difference = predicted_queries.float() - target_queries.float()
    induced = torch.einsum("...lr,ler->...le", difference, expert_keys.float())
    return F.huber_loss(induced, torch.zeros_like(induced), delta=delta)


def gather_node_horizon(
    values: Tensor,
    node_depth: Tensor,
    node_available: Tensor | None = None,
) -> Tensor:
    """Select each node's matching H1--H4 slice from ``[B,H,...,N,...]``."""

    if values.ndim < 4 or node_depth.ndim != 2:
        raise ValueError("node-horizon gather requires values and [B,N] depths")
    batch, horizons = values.shape[:2]
    if node_depth.shape[0] != batch:
        raise ValueError("node depth batch differs from values")
    nodes = node_depth.shape[1]
    # v4 tensors place the node axis immediately before the final feature
    # axis: [B,H,L,N,D].  This explicit contract avoids an ambiguous gather.
    if values.ndim != 5 or values.shape[3] != nodes:
        raise ValueError("node-horizon values must be [B,H,L,N,D]")
    available = (
        torch.ones_like(node_depth, dtype=torch.bool)
        if node_available is None else node_available.bool()
    )
    if available.shape != node_depth.shape:
        raise ValueError("node availability differs from depth")
    indices = node_depth.long() - 1
    if bool((((indices < 0) | (indices >= horizons)) & available).any()):
        raise ValueError("node depths lie outside the H1-H4 namespace")
    indices = indices.clamp(0, horizons - 1)
    selected = values.permute(0, 3, 1, 2, 4)
    result = selected.gather(
        2,
        indices[:, :, None, None, None].expand(
            batch, nodes, 1, values.shape[2], values.shape[4]
        ),
    ).squeeze(2)
    return result * available[..., None, None].to(result.dtype)


__all__ = [
    "DeltaRouteConfig", "DeltaRouteTrajectory", "DeltaRouteTrajectoryOutput",
    "LayerConditionedChannelPool", "RouteDistribution", "RouteDynamicsCore",
    "RouteRolloutOutput", "centered_logit_huber", "gather_node_horizon",
    "predicted_route_distribution",
]
