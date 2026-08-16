import pytest
import torch

from harp_rtt.dataset import HarpRTTDataset

from runpod.screen_route_quant import (
    parse_candidates,
    parse_layers,
    parse_upgrade_fractions,
    upgrade_schedule,
)
from runpod.train_shadow_experts_local import ShadowFactualStateDataset


def test_routequant_driver_parses_frozen_ladder():
    assert parse_layers("0,6,20,32") == (0, 6, 20, 32)
    assert parse_candidates("2:mse,3:mse,4:amax") == (
        (2, "mse"),
        (3, "mse"),
        (4, "amax"),
    )


@pytest.mark.parametrize("value", ("", "0,0", "39", "-1"))
def test_routequant_driver_rejects_invalid_layer_set(value: str):
    with pytest.raises(ValueError):
        parse_layers(value)


@pytest.mark.parametrize("value", ("", "2", "0:mse", "5:mse", "2:bad", "2:mse,2:mse"))
def test_routequant_driver_rejects_invalid_candidate_ladder(value: str):
    with pytest.raises(ValueError):
        parse_candidates(value)



def test_shadow_factual_wrapper_can_retarget_layer_without_losing_subset():
    source = object.__new__(ShadowFactualStateDataset)
    source.layer = 0
    source.next_router_agreement = True
    source.source = object.__new__(HarpRTTDataset)
    source.indices = (3, 7, 11)
    retargeted = ShadowFactualStateDataset(
        source, layer=6, next_router_agreement=True
    )
    assert retargeted.source is source.source
    assert retargeted.indices == source.indices
    assert retargeted.layer == 6


def test_routequant_mixed_upgrade_schedule_is_deterministic_and_nested():
    scores = torch.tensor([1.0, 3.0, 3.0, -1.0])
    quarter = upgrade_schedule(scores, fraction=0.25)
    half = upgrade_schedule(scores, fraction=0.5)
    assert quarter.tolist() == [3, 4, 3, 3]
    assert half.tolist() == [3, 4, 4, 3]
    assert set((quarter == 4).nonzero().flatten().tolist()) <= set(
        (half == 4).nonzero().flatten().tolist()
    )
    assert parse_upgrade_fractions("0.25,0.5,0.75") == (0.25, 0.5, 0.75)


@pytest.mark.parametrize("value", ("0", "1", "0.5,0.25", "0.5,0.5"))
def test_routequant_rejects_invalid_upgrade_fractions(value: str):
    with pytest.raises(ValueError):
        parse_upgrade_fractions(value)
