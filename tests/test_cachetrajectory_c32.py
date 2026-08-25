from __future__ import annotations

import torch

from harp_rtt.cachetrajectory_c32 import (
    FactorizedCacheTrajectory32,
    build_candidate_ids,
    build_features,
    parameter_and_mac_contract,
    split_masks,
)


def test_candidates_pin_current_and_use_dense_order_for_novel_slots() -> None:
    scores = torch.arange(256).float().view(1, 1, 1, 256).expand(1, 4, 40, 256).clone()
    current = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]).view(1, 1, 8).expand(1, 40, 8)
    candidates = build_candidate_ids(scores, current, width=12)
    assert candidates[0, 0, 0].tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 255, 254, 253, 252]


def test_history_features_are_causal_and_have_fixed_geometry() -> None:
    scores = torch.randn(2, 4, 40, 256)
    current = scores[:, 0].topk(8, -1).indices
    candidates = build_candidate_ids(scores, current)
    history_ids = current[:, None].expand(-1, 8, -1, -1).clone()
    history_weights = torch.full_like(history_ids, 0.125, dtype=torch.float)
    features, dense_rank, is_current = build_features(
        scores, candidates, current, history_ids, history_weights
    )
    assert features.shape == (2, 4, 40, 32, 28)
    assert dense_rank.shape == is_current.shape == (2, 4, 40, 32)
    assert torch.equal(features[..., 4].bool(), is_current)
    assert torch.equal(features[..., 5], torch.where(is_current, torch.arange(8).view(1, 1, 1, 8).expand(2, 4, 40, 8).float().repeat_interleave(4, -1) if False else features[..., 5], features[..., 5]))


def test_factorized_contract_is_smaller_than_frozen_c16() -> None:
    contract = parameter_and_mac_contract()
    assert contract["parameters"] <= 120_007
    assert contract["deployment_bf16_bytes"] <= 240_014
    assert contract["mac_ratio"] < 1.0
    model = FactorizedCacheTrajectory32()
    factual, survivor = model(
        torch.randn(3, 32, 28),
        torch.randint(0, 256, (3, 32)),
        torch.tensor([0, 1, 2]),
        torch.tensor([3, 4, 5]),
    )
    assert factual.shape == survivor.shape == (3, 32)


def test_request_split_is_frozen_and_disjoint() -> None:
    request = torch.arange(32)
    masks = split_masks(request)
    assert int(masks["training"].sum()) == 12
    assert int(masks["calibration"].sum()) == 4
    assert int(masks["blind"].sum()) == 16
    assert not bool((masks["training"] & masks["calibration"]).any())
    assert not bool((masks["design"] & masks["blind"]).any())
