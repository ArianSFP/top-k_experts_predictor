from __future__ import annotations

import torch

from tests.test_harp_rtt_path_route_surrogate import _fixture


def test_later_path_tokens_cannot_change_earlier_horizon_routes() -> None:
    model, inputs = _fixture()
    model.eval()
    baseline = model(**inputs).scores.detach()
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["path_token_embeddings"][:, 2:] += 100.0
    perturbed = model(**changed).scores.detach()
    assert torch.equal(baseline[:, :2], perturbed[:, :2])
    assert not torch.equal(baseline[:, 2:], perturbed[:, 2:])
