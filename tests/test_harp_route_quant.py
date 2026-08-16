import pytest
import torch

from harp_rtt.route_quant import (
    PackedRouteQuantExperts,
    dequantize_groupwise_nbit,
    pack_unsigned_codes,
    quantize_groupwise_nbit,
    uniform_routequant_projected_bytes,
    unpack_unsigned_codes,
)


@pytest.mark.parametrize("bits", (1, 2, 3, 4))
@pytest.mark.parametrize("values", (1, 2, 7, 8, 9, 17))
def test_routequant_code_packing_round_trip(bits: int, values: int):
    maximum = 1 << bits
    codes = (torch.arange(3 * values).reshape(3, values) % maximum).to(torch.uint8)
    packed = pack_unsigned_codes(codes, bits=bits)
    assert packed.shape == (3, (values * bits + 7) // 8)
    assert torch.equal(
        unpack_unsigned_codes(packed, bits=bits, values=values), codes
    )


def test_routequant_rejects_code_outside_declared_width():
    with pytest.raises(ValueError, match="outside"):
        pack_unsigned_codes(torch.tensor([[4]], dtype=torch.uint8), bits=2)


@pytest.mark.parametrize("bits", (1, 2, 3, 4))
def test_routequant_groupwise_round_trip_is_finite(bits: int):
    torch.manual_seed(11)
    weight = torch.randn(3, 5, 8)
    packed, scales = quantize_groupwise_nbit(
        weight, bits=bits, group_size=4, scale_method="mse"
    )
    restored = dequantize_groupwise_nbit(
        packed, scales, bits=bits, group_size=4, dtype=torch.float32
    )
    assert restored.shape == weight.shape
    assert torch.isfinite(restored).all()
    assert float((restored - weight).square().mean()) < float(weight.square().mean())


def test_routequant_mse_scale_does_not_worsen_weight_error():
    torch.manual_seed(19)
    weight = torch.randn(4, 7, 16) * torch.linspace(0.2, 3.0, 16)
    errors = {}
    for method in ("amax", "mse"):
        packed, scales = quantize_groupwise_nbit(
            weight, bits=2, group_size=8, scale_method=method
        )
        restored = dequantize_groupwise_nbit(
            packed, scales, bits=2, group_size=8, dtype=torch.float32
        )
        errors[method] = float((restored - weight).square().mean())
    assert errors["mse"] <= errors["amax"] + 1e-7


def test_routequant_runtime_preserves_identity_weights_and_omission():
    torch.manual_seed(23)
    gate_up = torch.randn(4, 4, 4)
    down = torch.randn(4, 4, 2)
    schedule = torch.tensor([4, 3, 0, 2], dtype=torch.int8)
    module = PackedRouteQuantExperts.from_target(
        gate_up,
        down,
        gate_up_bits=schedule,
        exact_k=2,
        group_size=2,
        item_chunk=2,
    )
    hidden = torch.randn(2, 4)
    ids = torch.tensor([[0, 1], [2, 3]])
    weights = torch.tensor([[0.7, 0.2], [0.9, 0.1]])
    values = module.selected_unweighted(hidden, ids)
    output = module(hidden, ids, weights)
    assert torch.equal(values[1, 0], torch.zeros(4))
    assert torch.allclose(output, (values * weights[..., None]).sum(1))
    renormalized = (
        values * (weights / weights.sum(-1, keepdim=True))[..., None]
    ).sum(1)
    assert not torch.allclose(output, renormalized)
    assert not any(True for _ in module.parameters())
    module.enable_dequantized_cache()
    cached = module(hidden, ids, weights)
    assert torch.equal(cached, output)
    assert set(module._gate_up_cache) == {0, 1, 3}
    module.enable_dequantized_cache(False)
    assert not module._gate_up_cache and not module._down_cache


def test_routequant_mixed_bit_bank_storage_round_trip():
    torch.manual_seed(29)
    gate_up = torch.randn(4, 4, 4)
    down = torch.randn(4, 4, 2)
    schedule = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
    module = PackedRouteQuantExperts.from_target(
        gate_up, down, gate_up_bits=schedule, exact_k=2,
        group_size=2, item_chunk=1
    )
    state = module.state_dict()
    assert "gate_up.packed_b3" in state
    assert module.persistent_nbytes() == sum(
        value.numel() * value.element_size() for value in state.values()
    )
    assert module.gate_up.bit_widths.tolist() == [1, 2, 3, 4]
    assert module.down.bit_widths.tolist() == [1, 2, 3, 4]


def test_routequant_requires_complete_native_topk_and_complete_cell_omission():
    gate_up = torch.randn(2, 4, 4)
    down = torch.randn(2, 4, 2)
    module = PackedRouteQuantExperts.from_target(
        gate_up, down, gate_up_bits=2, exact_k=2, group_size=2
    )
    with pytest.raises(ValueError, match="complete native top-k"):
        module(torch.ones(1, 4), torch.tensor([[0]]), torch.ones(1, 1))
    with pytest.raises(ValueError, match="complete expert cell"):
        PackedRouteQuantExperts.from_target(
            gate_up,
            down,
            gate_up_bits=torch.tensor([0, 2]),
            down_bits=torch.tensor([2, 2]),
            exact_k=2,
            group_size=2,
        )


def test_routequant_projected_bytes_are_monotonic_and_include_overhead():
    sizes = [uniform_routequant_projected_bytes(bits=bits) for bits in (1, 2, 3, 4)]
    assert sizes == sorted(sizes)
    raw_two_bit = 32_212_254_720 * 2 // 8
    assert sizes[1] > raw_two_bit
    assert sizes[3] > 15 * 2**30

