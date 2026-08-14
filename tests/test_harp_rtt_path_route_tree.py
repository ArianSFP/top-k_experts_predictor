from __future__ import annotations

import pytest
import torch

from harp_rtt.path_route_tree import reconstruct_tree_token_prefixes


def test_tree_prefixes_are_ancestor_ordered_and_padded() -> None:
    tokens = torch.tensor([[11, 21, 22, 31, 41, 32]])
    parents = torch.tensor([[-1, 0, 0, 1, 3, 2]])
    depths = torch.tensor([[1, 2, 2, 3, 4, 3]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    paths, available = reconstruct_tree_token_prefixes(
        tokens, parents, depths, mask
    )
    assert paths.tolist() == [[
        [11, 0, 0, 0], [11, 21, 0, 0], [11, 22, 0, 0],
        [11, 21, 31, 0], [11, 21, 31, 41], [11, 22, 32, 0],
    ]]
    assert available.sum(-1).tolist() == depths.tolist()


def test_tree_prefix_reconstruction_rejects_noncausal_parent() -> None:
    with pytest.raises(ValueError, match="precede"):
        reconstruct_tree_token_prefixes(
            torch.tensor([[1, 2]]), torch.tensor([[-1, 1]]),
            torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.bool),
        )
