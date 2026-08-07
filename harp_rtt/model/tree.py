"""Ancestor/sibling-aware adaptive MTP tree encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .common import SwiGLUResidual, safe_padding_mask
from .config import HARPRTTConfig


@dataclass(frozen=True)
class TreeEncoding:
    states: Tensor
    posterior_logits: Tensor
    horizon_mask: Tensor
    available: Tensor


def _structural_attention_mask(
    parent_ids: Tensor,
    available: Tensor,
    heads: int,
) -> Tensor:
    """Build the per-example ancestor/self/sibling attention mask."""

    batch, nodes = parent_ids.shape
    device = parent_ids.device
    parent = parent_ids.long()
    if ((parent >= nodes) | (parent < -1)).any():
        raise ValueError("tree parent ID lies outside [-1, node_count)")
    ancestors = torch.zeros(batch, nodes, nodes, dtype=torch.bool, device=device)
    current = parent.clone()
    rows = torch.arange(batch, device=device)[:, None].expand(batch, nodes)
    queries = torch.arange(nodes, device=device)[None].expand(batch, nodes)
    for _ in range(nodes):
        valid = current >= 0
        if valid.any():
            ancestors[rows[valid], queries[valid], current[valid]] = True
        next_parent = parent.gather(1, current.clamp_min(0))
        current = torch.where(valid, next_parent, torch.full_like(current, -1))
    eye = torch.eye(nodes, dtype=torch.bool, device=device)[None]
    siblings = (
        (parent[:, :, None] == parent[:, None, :])
        & (parent[:, :, None] >= 0)
        & available[:, :, None]
        & available[:, None, :]
    )
    allowed = ancestors | siblings | eye
    safe, _ = safe_padding_mask(available)
    allowed &= safe[:, None, :]
    # Padded query rows attend to the first safe key and are zeroed after each
    # block.  This avoids all-masked MultiheadAttention rows.
    padded_queries = ~available
    if padded_queries.any():
        first_safe = safe.float().argmax(dim=-1)
        allowed[padded_queries] = False
        allowed[
            rows[padded_queries],
            queries[padded_queries],
            first_safe[:, None].expand(-1, nodes)[padded_queries],
        ] = True
    blocked = ~allowed
    return blocked[:, None].expand(-1, heads, -1, -1).reshape(
        batch * heads, nodes, nodes
    )


class _TreeBlock(nn.Module):
    def __init__(self, config: HARPRTTConfig) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(config.tree_width)
        self.attention = nn.MultiheadAttention(
            config.tree_width,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.feedforward = SwiGLUResidual(
            config.tree_width, config.tree_ffn_width, config.dropout
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        structural_mask: Tensor,
        padding_mask: Tensor,
        available: Tensor,
    ) -> Tensor:
        value = self.norm(inputs)
        update, _ = self.attention(
            value,
            value,
            value,
            attn_mask=structural_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        result = self.feedforward(inputs + self.dropout(update))
        return result * available[..., None].to(result.dtype)


class AdaptiveTreeEncoder(nn.Module):
    """Encode unordered rich MTP nodes with explicit tree structure."""

    def __init__(self, config: HARPRTTConfig) -> None:
        super().__init__()
        self.config = config
        component_width = config.tree_width // 4
        self.hidden_projection = nn.Sequential(
            nn.RMSNorm(config.tree_hidden_width),
            nn.Linear(config.tree_hidden_width, component_width),
        )
        self.fused_projection = nn.Sequential(
            nn.RMSNorm(config.tree_fused_width),
            nn.Linear(config.tree_fused_width, component_width),
        )
        self.router_input_projection = nn.Sequential(
            nn.RMSNorm(config.tree_router_input_width),
            nn.Linear(config.tree_router_input_width, component_width),
        )
        self.router_logit_projection = nn.Sequential(
            nn.LayerNorm(config.experts),
            nn.Linear(config.experts, component_width),
        )
        self.token_projection = nn.Sequential(
            nn.RMSNorm(config.tree_token_width),
            nn.Linear(config.tree_token_width, component_width),
        )
        self.metadata_projection = nn.Sequential(
            nn.LayerNorm(config.tree_metadata_width),
            nn.Linear(config.tree_metadata_width, component_width),
        )
        self.node_fusion = nn.Sequential(
            nn.Linear(component_width * 6, config.tree_ffn_width),
            nn.SiLU(),
            nn.Linear(config.tree_ffn_width, config.tree_width),
            nn.RMSNorm(config.tree_width),
        )
        self.depth_embedding = nn.Embedding(
            config.max_tree_depth + 1, config.tree_width
        )
        self.branch_embedding = nn.Embedding(
            config.max_tree_branches + 1, config.tree_width
        )
        self.blocks = nn.ModuleList(
            [_TreeBlock(config) for _ in range(config.tree_blocks)]
        )
        self.horizon_embedding = nn.Embedding(
            config.active_horizons, config.tree_width
        )
        self.posterior_state = nn.Linear(config.tree_width, config.tree_width)
        self.posterior = nn.Linear(config.tree_width, 1, bias=False)
        self.output_norm = nn.RMSNorm(config.tree_width)

    def forward(
        self,
        *,
        hidden: Tensor,
        fused: Tensor,
        router_input: Tensor,
        router_logits: Tensor,
        token_embeddings: Tensor,
        metadata: Tensor,
        depth_ids: Tensor,
        parent_ids: Tensor,
        branch_ids: Tensor,
        available: Tensor,
        horizon_mask: Tensor | None = None,
        horizon_ids: Tensor | None = None,
    ) -> TreeEncoding:
        config = self.config
        if hidden.ndim != 3:
            raise ValueError("tree hidden states must be [B,N,D]")
        batch, nodes, _ = hidden.shape
        if not 1 <= nodes <= config.max_tree_nodes:
            raise ValueError("tree node count lies outside the configured budget")
        expected_prefix = (batch, nodes)
        for name, value in {
            "fused": fused,
            "router_input": router_input,
            "router_logits": router_logits,
            "token_embeddings": token_embeddings,
            "metadata": metadata,
        }.items():
            if value.shape[:2] != expected_prefix:
                raise ValueError(f"tree {name} must begin [B,N]")
        for name, value in {
            "depth_ids": depth_ids,
            "parent_ids": parent_ids,
            "branch_ids": branch_ids,
            "available": available,
        }.items():
            if value.shape != expected_prefix:
                raise ValueError(f"tree {name} must have shape [B,N]")
        available = available.bool()
        invalid_depth = (depth_ids < 0) | (depth_ids > config.max_tree_depth)
        if (invalid_depth & available).any():
            raise ValueError("valid tree depth ID lies outside the configured range")
        invalid_branch = (branch_ids < 0) | (
            branch_ids > config.max_tree_branches
        )
        if (invalid_branch & available).any():
            raise ValueError("valid tree branch ID lies outside the configured range")
        centered_router = router_logits - router_logits.mean(dim=-1, keepdim=True)
        components = [
            self.hidden_projection(hidden),
            self.fused_projection(fused),
            self.router_input_projection(router_input),
            self.router_logit_projection(centered_router),
            self.token_projection(token_embeddings),
            self.metadata_projection(metadata),
        ]
        states = self.node_fusion(torch.cat(components, dim=-1))
        states = (
            states
            + self.depth_embedding(depth_ids.clamp(0, config.max_tree_depth))
            + self.branch_embedding(branch_ids.clamp(0, config.max_tree_branches))
        )
        states = states * available[..., None].to(states.dtype)
        safe, padding = safe_padding_mask(available)
        if (~available.any(dim=-1)).any():
            states = states.clone()
            states[~available.any(dim=-1), 0] = 0
        structural = _structural_attention_mask(
            parent_ids, available, config.attention_heads
        )
        for block in self.blocks:
            states = block(
                states,
                structural_mask=structural,
                padding_mask=padding,
                available=available,
            )
        states = self.output_norm(states)
        horizon_embedding = self.horizon_embedding.weight
        logits = self.posterior(
            torch.tanh(
                self.posterior_state(states)[:, None]
                + horizon_embedding[None, :, None]
            )
        ).squeeze(-1)
        if horizon_mask is not None:
            expected = (batch, nodes, config.active_horizons)
            if horizon_mask.shape != expected:
                raise ValueError(f"tree horizon_mask must have shape {expected}")
            per_horizon = horizon_mask.permute(0, 2, 1).bool()
        elif horizon_ids is not None:
            if horizon_ids.shape != (batch, nodes):
                raise ValueError("tree horizon_ids must be [B,N]")
            horizon = torch.arange(
                1, config.active_horizons + 1, device=states.device
            )
            per_horizon = horizon_ids[:, None, :] == horizon[None, :, None]
            # Zero is an explicitly shared/root node.
            per_horizon |= horizon_ids[:, None, :] == 0
        else:
            per_horizon = torch.ones(
                batch,
                config.active_horizons,
                nodes,
                dtype=torch.bool,
                device=states.device,
            )
        per_horizon &= available[:, None]
        # If capture supplies no node for a horizon, retain every causal node
        # rather than constructing an invalid all-masked posterior.
        missing = ~per_horizon.any(dim=-1)
        if missing.any():
            per_horizon = torch.where(
                missing[..., None], available[:, None], per_horizon
            )
        # Keep raw finite logits public for branch/path auxiliary losses.  The
        # exact branch mixture applies ``per_horizon`` only while normalizing.
        return TreeEncoding(
            states=states,
            posterior_logits=logits,
            horizon_mask=per_horizon,
            available=available,
        )


__all__ = ["AdaptiveTreeEncoder", "TreeEncoding"]
