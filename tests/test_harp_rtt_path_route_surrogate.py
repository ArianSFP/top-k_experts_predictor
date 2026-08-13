from __future__ import annotations

import inspect

import torch

from harp_rtt.path_route_surrogate import (
    PathRouteSurrogateConfig,
    TokenConditionedRouteSurrogate,
)


def _fixture() -> tuple[TokenConditionedRouteSurrogate, dict[str, torch.Tensor]]:
    torch.manual_seed(83)
    config = PathRouteSurrogateConfig(
        experts=11,
        layers=4,
        horizons=4,
        history=3,
        exact_k=2,
        router_rank=5,
        state_roles=4,
        state_rank=6,
        token_width=12,
        width=16,
        ffn_width=32,
        attention_heads=4,
        blocks=1,
        route_width=7,
        output_adapter_rank=3,
        free_rank=3,
        dropout=0.0,
    )
    model = TokenConditionedRouteSurrogate(
        config,
        torch.randn(config.layers, config.experts, config.router_rank),
        torch.randn(config.layers, config.experts),
        torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0],
             [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]],
            dtype=torch.bool,
        ),
    )
    batch = 2
    inputs = {
        "state_coordinates": torch.randn(
            batch, config.state_roles, config.layers, config.state_rank
        ),
        "current_queries": torch.randn(
            batch, config.layers, config.router_rank
        ),
        "history_selected_ids": torch.randint(
            0, config.experts,
            (batch, config.history, config.layers, config.exact_k),
        ),
        "history_selected_weights": torch.rand(
            batch, config.history, config.layers, config.exact_k
        ),
        "path_token_embeddings": torch.randn(
            batch, config.horizons, config.token_width
        ),
    }
    return model, inputs


def test_path_route_surrogate_shapes_rank_mask_and_gradients() -> None:
    model, inputs = _fixture()
    output = model(**inputs)
    config = model.config
    assert output.queries.shape == (2, 4, 4, 5)
    assert output.scores.shape == (2, 4, 4, 11)
    assert output.selected_ids.shape == (2, 4, 4, 2)
    assert output.hidden.shape == (2, 4, 4, 16)
    assert output.path_states.shape == (2, 4, 16)
    assert torch.equal(
        output.queries * (~model.rank_mask.bool())[None, None],
        torch.zeros_like(output.queries),
    )
    output.scores.square().mean().backward()
    assert model.token_projection[1].weight.grad is not None
    assert float(model.token_projection[1].weight.grad.abs().sum()) > 0
    assert model.route_embedding.grad is not None
    assert float(model.route_embedding.grad.abs().sum()) > 0
    assert model.state_projection[0].weight.grad is not None
    assert float(model.state_projection[0].weight.grad.abs().sum()) > 0


def test_path_and_weighted_route_history_change_predictions() -> None:
    model, inputs = _fixture()
    model.eval()
    baseline = model(**inputs).scores
    changed_path = {key: value.clone() for key, value in inputs.items()}
    changed_path["path_token_embeddings"][0, 1] += 3.0
    assert not torch.equal(baseline[0], model(**changed_path).scores[0])

    changed_weights = {key: value.clone() for key, value in inputs.items()}
    changed_weights["history_selected_weights"][0, 0, 0] = torch.tensor(
        [0.999, 0.001]
    )
    assert not torch.equal(baseline[0], model(**changed_weights).scores[0])


def test_serving_api_has_no_target_label_argument() -> None:
    parameters = inspect.signature(TokenConditionedRouteSurrogate.forward).parameters
    forbidden = {
        "target_centered_logits", "target_selected_ids", "future_router_inputs",
        "counterfactual_queries", "counterfactual_router_logits",
    }
    assert forbidden.isdisjoint(parameters)
