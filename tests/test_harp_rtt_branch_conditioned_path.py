from __future__ import annotations

import inspect

import torch

from harp_rtt.branch_conditioned_path import BranchConditionedLayerwiseRouteSurrogate
from harp_rtt.path_route_surrogate import PathRouteSurrogateConfig
from harp_rtt.path_route_trajectory import LayerwiseTokenRouteSurrogate


def configuration() -> PathRouteSurrogateConfig:
    return PathRouteSurrogateConfig(
        experts=16, layers=3, horizons=4, history=2, exact_k=2,
        router_rank=7, state_roles=4, state_rank=5, token_width=12,
        width=16, ffn_width=32, attention_heads=4, blocks=1,
        route_width=6, output_adapter_rank=3, free_rank=3, dropout=0.0,
    )


def inputs(config: PathRouteSurrogateConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    batch = 2
    return {
        "state_coordinates": torch.randn(
            batch, config.state_roles, config.layers, config.state_rank,
            generator=generator,
        ),
        "current_queries": torch.randn(
            batch, config.layers, config.router_rank, generator=generator,
        ),
        "history_selected_ids": torch.randint(
            config.experts,
            (batch, config.history, config.layers, config.exact_k),
            generator=generator,
        ),
        "history_selected_weights": torch.rand(
            batch, config.history, config.layers, config.exact_k,
            generator=generator,
        ),
        "path_token_embeddings": torch.randn(
            batch, config.horizons, config.token_width, generator=generator,
        ),
        "branch_states": torch.randn(
            batch, 4, config.token_width, generator=generator,
        ),
        "branch_router_logits": torch.randn(
            batch, config.experts, generator=generator,
        ),
        "branch_selected_ids": torch.randint(
            config.experts, (batch, config.exact_k), generator=generator,
        ),
        "branch_selected_weights": torch.rand(
            batch, config.exact_k, generator=generator,
        ),
        "branch_vocab_embedding": torch.randn(
            batch, config.token_width, generator=generator,
        ),
        "branch_vocab_statistics": torch.randn(batch, 6, generator=generator),
        "branch_scalars": torch.randn(batch, 8, generator=generator),
    }


def models() -> tuple[
    LayerwiseTokenRouteSurrogate, BranchConditionedLayerwiseRouteSurrogate
]:
    config = configuration()
    generator = torch.Generator().manual_seed(11)
    keys = torch.randn(
        config.layers, config.experts, config.router_rank, generator=generator,
    )
    bias = torch.randn(config.layers, config.experts, generator=generator)
    mask = torch.ones(config.layers, config.router_rank)
    parent = LayerwiseTokenRouteSurrogate(config, keys, bias, mask).eval()
    child = BranchConditionedLayerwiseRouteSurrogate(
        config, keys, bias, mask, branch_state_rank=3, branch_vocab_rank=2,
    ).eval()
    incompatible = child.load_state_dict(parent.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(name.startswith("branch_") for name in incompatible.missing_keys)
    return parent, child


def test_zero_gate_exactly_reproduces_parent() -> None:
    parent, child = models()
    values = inputs(parent.config)
    parent_values = {
        name: value for name, value in values.items()
        if name in inspect.signature(parent.forward).parameters
    }
    with torch.no_grad():
        expected = parent(**parent_values)
        actual = child(**values)
    assert torch.equal(actual.queries, expected.queries)
    assert torch.equal(actual.scores, expected.scores)
    assert torch.equal(actual.selected_ids, expected.selected_ids)


def test_open_gate_uses_branch_identity_and_has_gradients() -> None:
    _, child = models()
    values = inputs(child.config)
    with torch.no_grad():
        child.branch_gate.fill_(0.2)
        first = child(**values).scores
        changed = dict(values)
        changed["branch_states"] = values["branch_states"].clone()
        changed["branch_states"][0].add_(2.0)
        second = child(**changed).scores
    assert not torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])

    child.train(); child.zero_grad(set_to_none=True)
    child(**values).scores.square().mean().backward()
    assert child.branch_state_down.grad is not None
    assert child.branch_gate.grad is not None


def test_serving_api_has_no_target_argument() -> None:
    parameters = inspect.signature(
        BranchConditionedLayerwiseRouteSurrogate.forward
    ).parameters
    assert not any("target" in name or "label" in name for name in parameters)
