from __future__ import annotations

import numpy as np
import pytest
import torch

from harp8.router_geometry import router_svd_keys


def test_full_rank_router_keys_preserve_gram() -> None:
    generator = torch.Generator().manual_seed(8)
    weights = torch.randn(3, 7, 11, generator=generator)
    keys, diagnostics = router_svd_keys(weights)
    assert keys.shape == (3, 7, 7)
    assert diagnostics["rank"] == 7
    for layer in range(3):
        assert torch.allclose(
            keys[layer] @ keys[layer].T,
            weights[layer] @ weights[layer].T,
            atol=2e-5,
            rtol=2e-5,
        )
    assert diagnostics["retained_energy_min"] == pytest.approx(1.0, abs=1e-6)


def test_truncated_router_keys_report_loss_and_validate_rank() -> None:
    weights = np.random.default_rng(4).normal(size=(2, 6, 9)).astype(np.float32)
    keys, diagnostics = router_svd_keys(weights, rank=3)
    assert keys.shape == (2, 6, 3)
    assert 0.0 < diagnostics["retained_energy_min"] < 1.0
    assert diagnostics["relative_gram_error_max"] > 0.0
    with pytest.raises(ValueError, match="rank"):
        router_svd_keys(weights, rank=0)
