"""Expert-conditioned direct factual prediction from the causal MTP tree."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .route_dynamics import DeltaRouteConfig, LayerConditionedChannelPool


@dataclass(frozen=True)
class TreeGlobalFactualOutput:
    score_delta: Tensor
    expert_branch_weights: Tensor
    node_context: Tensor


class TreeGlobalFactualHead(nn.Module):
    """Map all causal tree states directly to factual per-expert evidence.

    This deliberately bypasses counterfactual target-route reconstruction.
    Every expert forms its own posterior-conditioned attention over nodes and
    explicit OTHER.  A frozen reference computation is subtracted from the
    live computation: output is exactly zero initially while the complete
    causal encoder and expert-attention path receive first-step gradients.
    """

    def __init__(
        self,
        config: DeltaRouteConfig,
        expert_keys: Tensor,
    ) -> None:
        super().__init__()
        config.validate()
        if expert_keys.shape != (
            config.layers, config.experts, config.router_rank
        ):
            raise ValueError("expert keys disagree with factual-head config")
        self.config = config
        self.register_buffer("expert_keys", expert_keys.detach().float().clone())
        self.channels = LayerConditionedChannelPool(config)
        self.context = nn.Linear(config.latent_width, config.latent_width)
        self.other = nn.Linear(config.latent_width, config.latent_width)
        self.node_key = nn.Linear(config.latent_width, config.latent_width)
        self.node_value = nn.Linear(config.latent_width, config.latent_width)
        self.expert_query = nn.Linear(
            config.router_rank, config.latent_width, bias=False
        )
        self.expert_value = nn.Linear(
            config.router_rank, config.latent_width, bias=False
        )
        self.horizon_bias = nn.Parameter(torch.zeros(
            config.horizons, config.layers, config.experts
        ))
        self._snapshot_reference()

    def _snapshot_reference(self) -> None:
        for name, parameter in (
            ("ref_node_key_weight", self.node_key.weight),
            ("ref_node_key_bias", self.node_key.bias),
            ("ref_node_value_weight", self.node_value.weight),
            ("ref_node_value_bias", self.node_value.bias),
            ("ref_expert_query_weight", self.expert_query.weight),
            ("ref_expert_value_weight", self.expert_value.weight),
        ):
            self.register_buffer(name, parameter.detach().clone())

    def _evidence(
        self,
        contexts: Tensor,
        posterior: Tensor,
        *,
        reference: bool,
    ) -> tuple[Tensor, Tensor]:
        # contexts [B,H,L,N+1,W], posterior [B,H,N+1]
        if reference:
            contexts = contexts.detach()
            keys = F.linear(
                contexts, self.ref_node_key_weight, self.ref_node_key_bias
            )
            values = F.linear(
                contexts, self.ref_node_value_weight, self.ref_node_value_bias
            )
            expert_query = F.linear(
                self.expert_keys, self.ref_expert_query_weight
            )
            expert_value = F.linear(
                self.expert_keys, self.ref_expert_value_weight
            )
        else:
            keys = self.node_key(contexts)
            values = self.node_value(contexts)
            expert_query = self.expert_query(self.expert_keys)
            expert_value = self.expert_value(self.expert_keys)
        reliability = torch.einsum(
            "bhlnw,lew->bhlen", keys, expert_query
        ) / math.sqrt(self.config.latent_width)
        logits = reliability.float() + posterior.clamp_min(1e-30).log()[
            :, :, None, None, :
        ]
        weights = torch.softmax(logits, dim=-1)
        node_scores = torch.einsum(
            "bhlnw,lew->bhlen", values, expert_value
        ) / math.sqrt(self.config.latent_width)
        evidence = (weights * node_scores.float()).sum(-1)
        return evidence, weights

    def forward(
        self,
        *,
        context_states: Tensor,
        posterior: Tensor,
        fused: Tensor,
        post_ffn: Tensor,
        router_input: Tensor,
        vocabulary: Tensor,
        token: Tensor,
        router_logits: Tensor,
        metadata: Tensor,
        parents: Tensor,
        available: Tensor,
    ) -> TreeGlobalFactualOutput:
        config = self.config
        batch = fused.shape[0]
        if context_states.shape != (
            batch, config.horizons, config.layers, config.latent_width
        ):
            raise ValueError("factual-head context states have invalid geometry")
        if posterior.shape != (
            batch, config.horizons, config.nodes + 1
        ):
            raise ValueError("factual-head posterior has invalid geometry")
        pooled, _ = self.channels(
            fused=fused, post_ffn=post_ffn, router_input=router_input,
            vocabulary=vocabulary, token=token, router_logits=router_logits,
            metadata=metadata, parents=parents, available=available,
        )
        node_context = pooled + self.context(context_states)[:, :, :, None]
        other_context = self.other(context_states)[..., None, :]
        contexts = torch.cat((node_context, other_context), dim=3)
        live, weights = self._evidence(contexts, posterior, reference=False)
        reference, _ = self._evidence(contexts, posterior, reference=True)
        delta = live - reference + self.horizon_bias[None]
        delta = delta.clone(); delta[:, 0] = 0.0
        return TreeGlobalFactualOutput(delta, weights, node_context)


__all__ = ["TreeGlobalFactualHead", "TreeGlobalFactualOutput"]
