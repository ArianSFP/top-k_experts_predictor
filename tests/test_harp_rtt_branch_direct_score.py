from __future__ import annotations

import inspect

import torch

from harp_rtt.branch_direct_score import DirectHighRankBranchRouteSurrogate
from harp_rtt.path_route_surrogate import PathRouteSurrogateConfig
from harp_rtt.path_route_trajectory import LayerwiseTokenRouteSurrogate


def test_direct_score_is_exact_at_zero_and_branch_sensitive_when_open() -> None:
    config = PathRouteSurrogateConfig(
        experts=16, layers=3, horizons=4, history=2, exact_k=2,
        router_rank=7, state_roles=4, state_rank=5, token_width=12,
        width=16, ffn_width=32, attention_heads=4, blocks=1,
        route_width=6, output_adapter_rank=3, free_rank=3, dropout=0.0,
    )
    generator = torch.Generator().manual_seed(19)
    keys = torch.randn(3, 16, 7, generator=generator)
    bias = torch.randn(3, 16, generator=generator)
    mask = torch.ones(3, 7)
    parent = LayerwiseTokenRouteSurrogate(config, keys, bias, mask).eval()
    model = DirectHighRankBranchRouteSurrogate(
        config, keys, bias, mask, raw_rank=4
    ).eval()
    missing = model.load_state_dict(parent.state_dict(), strict=False)
    assert not missing.unexpected_keys
    batch = 2
    values = {
        "state_coordinates": torch.randn(batch, 4, 3, 5, generator=generator),
        "current_queries": torch.randn(batch, 3, 7, generator=generator),
        "history_selected_ids": torch.randint(16, (batch, 2, 3, 2), generator=generator),
        "history_selected_weights": torch.rand(batch, 2, 3, 2, generator=generator),
        "path_token_embeddings": torch.randn(batch, 4, 12, generator=generator),
        "branch_states": torch.randn(batch, 4, 12, generator=generator),
        "branch_router_logits": torch.randn(batch, 16, generator=generator),
        "branch_selected_ids": torch.randint(16, (batch, 2), generator=generator),
        "branch_selected_weights": torch.rand(batch, 2, generator=generator),
        "branch_vocab_embedding": torch.randn(batch, 12, generator=generator),
        "branch_vocab_statistics": torch.randn(batch, 6, generator=generator),
        "branch_scalars": torch.randn(batch, 8, generator=generator),
    }
    parent_values = {
        name: value for name, value in values.items()
        if name in inspect.signature(parent.forward).parameters
    }
    with torch.no_grad():
        expected = parent(**parent_values)
        actual = model(**values)
    assert torch.equal(actual.queries, expected.queries)
    assert torch.equal(actual.scores, expected.scores)

    with torch.no_grad():
        model.score_up_direct.normal_(std=0.05)
        first = model(**values).scores
        changed = dict(values)
        changed["branch_states"] = values["branch_states"].clone()
        changed["branch_states"][0].add_(1.0)
        second = model(**changed).scores
    assert not torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])

    model.train(); model.zero_grad(set_to_none=True)
    model(**values).scores.square().mean().backward()
    assert model.score_up_direct.grad is not None
    assert model.branch_state_down.grad is not None
