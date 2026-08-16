import pytest

from runpod.screen_route_quant import parse_candidates, parse_layers


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

