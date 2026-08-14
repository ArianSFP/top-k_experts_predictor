"""Context-conditioned factual branch selector for HARP-DeltaRoute."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .route_dynamics import DeltaRouteConfig


@dataclass(frozen=True)
class ContextualPathOutput:
    logits: Tensor
    probabilities: Tensor
    mask: Tensor


class ContextualFactualPathSelector(nn.Module):
    """Choose the factual branch using target context × complete tree state.

    The v3 selector scores every node from its tree state alone.  It cannot ask
    whether a node is compatible with the completed target token/context.  This
    head pools the forty causal target-context cells separately per horizon,
    globally contextualizes the causal tree, and scores their interaction.
    It retains the MTP path probability as a strong additive prior and includes
    an explicit OTHER branch.
    """

    def __init__(self, config: DeltaRouteConfig, *, blocks: int = 2) -> None:
        super().__init__(); config.validate(); self.config = config
        width = config.latent_width
        self.layer_logits = nn.Parameter(torch.zeros(
            config.horizons, config.layers
        ))
        self.context = nn.Sequential(
            nn.RMSNorm(width), nn.Linear(width, width), nn.SiLU(),
        )
        self.node = nn.Sequential(
            nn.RMSNorm(width), nn.Linear(width, width), nn.SiLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=config.attention_heads,
            dim_feedforward=config.transition_width,
            dropout=config.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.tree = nn.TransformerEncoder(
            layer, num_layers=blocks, norm=nn.RMSNorm(width)
        )
        self.horizon = nn.Embedding(config.horizons, width)
        self.interaction = nn.Sequential(
            nn.RMSNorm(width), nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 1),
        )
        self.bilinear_query = nn.Linear(width, width, bias=False)
        self.bilinear_key = nn.Linear(width, width, bias=False)
        self.other = nn.Sequential(
            nn.RMSNorm(width), nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 1),
        )
        # Zero final projections reproduce the frozen learned parent posterior.
        # Context/tree features open independently after the first update.
        nn.init.zeros_(self.interaction[-1].weight)
        nn.init.zeros_(self.interaction[-1].bias)
        nn.init.zeros_(self.bilinear_query.weight)
        nn.init.zeros_(self.other[-1].weight)
        nn.init.zeros_(self.other[-1].bias)

    def forward(
        self, *, tree_states: Tensor, context_states: Tensor,
        base_probabilities: Tensor, horizon_mask: Tensor,
        node_available: Tensor,
    ) -> ContextualPathOutput:
        config = self.config; batch, nodes, width = tree_states.shape
        if (nodes, width) != (config.nodes, config.latent_width):
            raise ValueError("contextual selector tree geometry is invalid")
        if context_states.shape != (
            batch, config.horizons, config.layers, config.latent_width
        ):
            raise ValueError("contextual selector target context is invalid")
        if base_probabilities.shape != (
            batch, config.horizons, nodes + 1
        ):
            raise ValueError("contextual selector parent posterior is invalid")
        if horizon_mask.shape != (batch, config.horizons, nodes):
            raise ValueError("contextual selector horizon mask is invalid")
        if node_available.shape != (batch, nodes):
            raise ValueError("contextual selector availability is invalid")

        layer_weights = torch.softmax(self.layer_logits.float(), dim=-1)
        context = torch.einsum(
            "hl,bhlw->bhw", layer_weights, context_states.float()
        ) + self.horizon.weight[None]
        context = self.context(context)
        nodes_encoded = self.node(tree_states.float())
        nodes_encoded = self.tree(
            nodes_encoded, src_key_padding_mask=~node_available.bool()
        )
        joint = context[:, :, None] + nodes_encoded[:, None]
        correction = self.interaction(joint).squeeze(-1)
        correction = correction + torch.einsum(
            "bhw,bnw->bhn", self.bilinear_query(context),
            self.bilinear_key(nodes_encoded),
        ) / math.sqrt(width)
        visible = horizon_mask.bool() & node_available[:, None].bool()
        base = base_probabilities.float()
        captured = torch.where(
            visible, base[..., :-1], torch.zeros_like(base[..., :-1])
        ).sum(-1)
        base = torch.cat((
            torch.where(visible, base[..., :-1], torch.zeros_like(base[..., :-1])),
            (1.0 - captured).clamp_min(0.0)[..., None],
        ), dim=-1)
        base = base / base.sum(-1, keepdim=True).clamp_min(1e-12)
        correction = torch.cat((
            correction, self.other(context).squeeze(-1)[..., None],
        ), dim=-1)
        logits = base.clamp_min(1e-12).log() + correction
        mask = torch.cat((
            visible,
            torch.ones(batch, config.horizons, 1, dtype=torch.bool,
                       device=visible.device),
        ), dim=-1)
        probabilities = torch.softmax(
            logits.masked_fill(~mask, -torch.inf), dim=-1
        )
        return ContextualPathOutput(logits, probabilities, mask)


__all__ = ["ContextualFactualPathSelector", "ContextualPathOutput"]
