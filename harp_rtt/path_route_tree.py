"""Causal adaptive-tree adapters for token-conditioned route surrogates."""

from __future__ import annotations

import torch
from torch import Tensor


def reconstruct_tree_token_prefixes(
    token_ids: Tensor,
    parents: Tensor,
    depths: Tensor,
    node_mask: Tensor,
    *,
    horizons: int = 4,
) -> tuple[Tensor, Tensor]:
    """Return ancestor-ordered token prefixes without consulting labels.

    Padding after each node's depth is zero and explicitly described by the
    returned mask. Parent indices must be sample-local and parent-before-child.
    """

    if token_ids.ndim != 2 or parents.shape != token_ids.shape:
        raise ValueError("tree token IDs and parents must be [B,N]")
    if depths.shape != token_ids.shape or node_mask.shape != token_ids.shape:
        raise ValueError("tree depths/mask must match [B,N]")
    if horizons < 1:
        raise ValueError("tree prefix horizon count must be positive")
    batch, nodes = token_ids.shape
    valid = node_mask.bool()
    if bool(((depths < 1) & valid).any()) or bool(((depths > horizons) & valid).any()):
        raise ValueError("valid tree depth lies outside the route horizon range")
    local = torch.arange(nodes, device=token_ids.device)[None].expand(batch, -1)
    if bool(((parents >= local) & valid & (parents >= 0)).any()):
        raise ValueError("tree parent must precede its child")
    if bool(((parents < -1) & valid).any()):
        raise ValueError("tree parent uses an invalid negative sentinel")

    prefix = torch.zeros(
        batch, nodes, horizons, device=token_ids.device, dtype=token_ids.dtype
    )
    current = local.clone()
    active = valid.clone()
    position = depths.long() - 1
    written = torch.zeros_like(depths, dtype=torch.int64)
    for _ in range(horizons):
        safe = current.clamp(0, max(0, nodes - 1))
        values = token_ids.gather(1, safe)
        write = active & (position >= 0)
        written = written + write.long()
        slot = position.clamp(0, horizons - 1)[..., None]
        previous = prefix.gather(2, slot).squeeze(-1)
        prefix.scatter_(
            2, slot, torch.where(write, values, previous)[..., None]
        )
        next_parent = parents.gather(1, safe)
        if bool((write & (position == 0) & (next_parent >= 0)).any()):
            raise ValueError("tree depth is shorter than its parent chain")
        if bool((write & (position > 0) & (next_parent < 0)).any()):
            raise ValueError("tree depth is longer than its parent chain")
        active = write & (next_parent >= 0)
        current = torch.where(active, next_parent, torch.zeros_like(next_parent))
        position = position - 1
    path_mask = (
        torch.arange(horizons, device=token_ids.device)[None, None]
        < depths.long()[..., None]
    ) & valid[..., None]
    if bool((prefix[path_mask] < 0).any()):
        raise ValueError("valid tree prefix contains a negative token ID")
    if bool((valid & (written != depths.long())).any()):
        raise ValueError("tree depth is longer than its parent chain")
    return prefix, path_mask


__all__ = ["reconstruct_tree_token_prefixes"]
