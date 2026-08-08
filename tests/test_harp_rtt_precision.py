from __future__ import annotations

import pytest
import torch

from harp_rtt.geometry import build_centered_router_geometry
from harp_rtt.model import HARPRTTConfig
from harp_rtt.model.heads import HybridRouterScoreHead
from harp_rtt.model.tree import TreeEncoding


DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is required"
        ),
    ),
]


@pytest.mark.parametrize("device_type", DEVICES)
def test_frozen_geometry_is_fp32_under_bfloat16_autocast(
    device_type: str,
) -> None:
    device = torch.device(device_type)
    generator = torch.Generator().manual_seed(781)
    weights = torch.randn(2, 17, 23, generator=generator)
    bias = torch.randn(2, 17, generator=generator)
    inputs = torch.randn(7, 2, 23, generator=generator)
    geometry = build_centered_router_geometry(weights, bias).to(device)
    inputs = inputs.to(device)

    expected_coordinates = geometry.encode_router_inputs(inputs)
    expected_scores = geometry.score_coordinates(expected_coordinates)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        coordinates = geometry.encode_router_inputs(inputs)
        scores = geometry.score_coordinates(coordinates)

    assert coordinates.dtype == torch.float32
    assert scores.dtype == torch.float32
    assert torch.equal(coordinates, expected_coordinates)
    assert torch.equal(scores, expected_scores)


@pytest.mark.parametrize("device_type", DEVICES)
def test_hybrid_head_scores_frozen_keys_in_fp32_under_bfloat16_autocast(
    device_type: str,
) -> None:
    device = torch.device(device_type)
    config = HARPRTTConfig(
        layers=2,
        experts=17,
        anchor_horizons=4,
        active_horizons=4,
        exact_k=4,
        candidate_width=8,
        router_rank=5,
        model_width=5,
        tree_width=5,
        decoder_ffn_width=10,
    )
    generator = torch.Generator().manual_seed(991)
    keys = torch.randn(2, 17, 5, generator=generator)
    bias = torch.randn(2, 17, generator=generator)
    rank_mask = torch.ones(2, 5, dtype=torch.bool)
    head = HybridRouterScoreHead(config, keys, rank_mask, bias).to(device).eval()
    endpoint = torch.randn(2, 4, 2, 5, generator=generator).to(device)
    states = torch.randn(2, 3, 5, generator=generator).to(device)
    tree = TreeEncoding(
        states=states,
        posterior_logits=torch.zeros(2, 4, 3, device=device),
        horizon_mask=torch.ones(2, 4, 3, dtype=torch.bool, device=device),
        available=torch.ones(2, 3, dtype=torch.bool, device=device),
    )
    anchor = torch.zeros(2, 4, 2, 17, device=device)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        output = head(endpoint, tree, anchor)

    assert output.geometry_scores.dtype == torch.float32
    expected = torch.einsum(
        "bhlnr,ler->bhlne", output.predicted_queries.float(), head.expert_keys
    )
    expected = expected + head.centered_bias[None, None, :, None]
    assert torch.equal(output.geometry_scores, expected)
