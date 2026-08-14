from __future__ import annotations

from types import SimpleNamespace

import torch

from runpod.train_harp_contextual_path_selector import _native_top8


def test_native_top8_masks_sentinel_expert_ids_before_scatter() -> None:
    experts = 12
    native = torch.full((1, 3, 2, 2), -1, dtype=torch.long)
    native[0, 0] = torch.tensor([[2, 5], [3, 7]])
    counterfactual = {
        "selected_ids": native,
        "valid": torch.tensor([[[True, True], [False, False], [False, False]]]),
        "node_mask": torch.tensor([[True, True, False]]),
    }
    output = SimpleNamespace(
        counterfactual=counterfactual,
        branch_mask=torch.tensor([[[True, False, False], [True, False, False]]]),
        semantic=object(),
        anchor_scores=torch.zeros(1, 2, 2, experts),
    )
    anchor_marginals = torch.zeros(1, 2, 2, experts)
    anchor_marginals[..., :2] = 1.0
    core = SimpleNamespace(
        semantic_marginals=lambda semantic, scores: (
            anchor_marginals, None, None, None
        )
    )
    parent = SimpleNamespace(
        config=SimpleNamespace(experts=experts, exact_k=2), core=core
    )
    posterior = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
    )

    selected = _native_top8(output, posterior, parent)

    assert selected.shape == (1, 2, 2, 2)
    assert set(selected[0, 0, 0].tolist()) == {2, 5}
