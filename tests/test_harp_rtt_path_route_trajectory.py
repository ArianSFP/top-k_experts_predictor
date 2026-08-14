from __future__ import annotations

import torch

from harp_rtt.path_route_surrogate import PathRouteSurrogateConfig
from harp_rtt.path_route_trajectory import LayerwiseTokenRouteSurrogate


def _trajectory_fixture() -> tuple[
    LayerwiseTokenRouteSurrogate, dict[str, torch.Tensor]
]:
    torch.manual_seed(97)
    config = PathRouteSurrogateConfig(
        experts=11, layers=4, horizons=4, history=3, exact_k=2,
        router_rank=5, state_roles=4, state_rank=6, token_width=12,
        width=16, ffn_width=32, attention_heads=4, blocks=1,
        route_width=7, output_adapter_rank=3, free_rank=3, dropout=0.0,
    )
    model = LayerwiseTokenRouteSurrogate(
        config, torch.randn(4, 11, 5), torch.randn(4, 11),
        torch.ones(4, 5, dtype=torch.bool),
    )
    inputs = {
        "state_coordinates": torch.randn(2, 4, 4, 6),
        "current_queries": torch.randn(2, 4, 5),
        "history_selected_ids": torch.randint(0, 11, (2, 3, 4, 2)),
        "history_selected_weights": torch.rand(2, 3, 4, 2),
        "path_token_embeddings": torch.randn(2, 4, 12),
    }
    return model, inputs


def test_layerwise_trajectory_shapes_and_feedback_gradients() -> None:
    model, inputs = _trajectory_fixture()
    output = model(**inputs)
    assert output.queries.shape == (2, 4, 4, 5)
    assert output.scores.shape == (2, 4, 4, 11)
    output.scores[:, :, 1:].square().mean().backward()
    assert model.expert_effect_embedding.grad is not None
    assert float(model.expert_effect_embedding.grad.abs().sum()) > 0
    assert model.feedback_query.weight.grad is not None
    assert float(model.feedback_query.weight.grad.abs().sum()) > 0
    assert model.trajectory_cell.weight_hh.grad is not None
    assert float(model.trajectory_cell.weight_hh.grad.abs().sum()) > 0


def test_expert_effect_change_is_strictly_downstream() -> None:
    model, inputs = _trajectory_fixture()
    model.eval()
    before = model(**inputs).scores.detach()
    with torch.no_grad():
        model.expert_effect_embedding[0].add_(10.0 * torch.randn_like(
            model.expert_effect_embedding[0]
        ))
    after = model(**inputs).scores.detach()
    assert torch.equal(before[:, :, 0], after[:, :, 0])
    assert not torch.equal(before[:, :, 1:], after[:, :, 1:])


def test_later_path_tokens_remain_causally_masked_under_rollout() -> None:
    model, inputs = _trajectory_fixture()
    model.eval()
    before = model(**inputs).scores.detach()
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["path_token_embeddings"][:, 2:] += 100.0
    after = model(**changed).scores.detach()
    assert torch.equal(before[:, :2], after[:, :2])
    assert not torch.equal(before[:, 2:], after[:, 2:])
