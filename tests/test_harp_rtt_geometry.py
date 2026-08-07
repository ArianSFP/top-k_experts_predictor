from __future__ import annotations

import pytest
import torch

from harp_rtt.geometry import (
    CenteredRouterGeometry,
    assert_router_geometry_equivalent,
    audit_centered_router_geometry,
    build_centered_router_geometry,
)


def test_full_centered_geometry_reconstructs_affine_logits_and_topk() -> None:
    generator = torch.Generator().manual_seed(908)
    weights = torch.randn(3, 9, 13, generator=generator)
    bias = torch.randn(3, 9, generator=generator)
    inputs = torch.randn(11, 3, 13, generator=generator)
    geometry = build_centered_router_geometry(weights, bias)

    assert geometry.K.shape == (3, 9, 8)
    assert geometry.V.shape == (3, 13, 8)
    assert geometry.rank_mask.shape == (3, 8)
    assert geometry.ranks.tolist() == [8, 8, 8]
    expected_weights = weights - weights.mean(dim=1, keepdim=True)
    expected_bias = bias - bias.mean(dim=-1, keepdim=True)
    assert torch.allclose(
        geometry.reconstruct_centered_weights(),
        expected_weights,
        atol=5e-6,
        rtol=5e-6,
    )
    assert torch.allclose(geometry.centered_bias, expected_bias, atol=1e-7, rtol=0)
    assert torch.allclose(
        geometry.row_norms,
        torch.linalg.vector_norm(expected_weights, dim=-1),
        atol=2e-6,
        rtol=2e-6,
    )

    audit = assert_router_geometry_equivalent(
        geometry,
        weights,
        bias,
        router_inputs=inputs,
        k=4,
    )
    assert audit.identical_topk is True
    assert audit.topk_matching_rows == 33
    assert audit.maximum_absolute_logit_error is not None
    assert audit.maximum_absolute_logit_error < 5e-5


def test_coordinates_are_v_transpose_inputs_and_include_centered_bias() -> None:
    generator = torch.Generator().manual_seed(12)
    weights = torch.randn(2, 7, 10, generator=generator)
    bias = torch.randn(2, 7, generator=generator)
    inputs = torch.randn(4, 2, 10, generator=generator)
    geometry = build_centered_router_geometry(weights, bias)
    coordinates = geometry.encode_router_inputs(inputs)
    assert torch.allclose(
        coordinates,
        torch.einsum("bld,ldr->blr", inputs, geometry.V),
        atol=1e-6,
        rtol=1e-6,
    )
    direct = torch.einsum("bld,led->ble", inputs, weights) + bias
    direct = direct - direct.mean(-1, keepdim=True)
    assert torch.allclose(
        geometry.score_coordinates(coordinates), direct, atol=3e-5, rtol=3e-5
    )
    zeros = torch.zeros(1, 2, 10)
    assert torch.allclose(geometry.centered_logits(zeros)[0], geometry.centered_bias)


def test_variable_numerical_ranks_are_padded_and_masked() -> None:
    generator = torch.Generator().manual_seed(7)
    full = torch.randn(6, 8, generator=generator)
    rank_one = torch.randn(6, 1, generator=generator) @ torch.randn(
        1, 8, generator=generator
    )
    zero = torch.zeros(6, 8)
    weights = torch.stack((full, rank_one, zero))
    geometry = build_centered_router_geometry(weights)
    assert geometry.ranks.tolist() == [5, 1, 0]
    assert geometry.maximum_rank == 5
    assert geometry.rank_mask.tolist() == [
        [True, True, True, True, True],
        [True, False, False, False, False],
        [False, False, False, False, False],
    ]
    assert torch.equal(geometry.K[1, :, 1:], torch.zeros(6, 4))
    assert torch.equal(geometry.V[2], torch.zeros(8, 5))
    geometry.validate()


def test_v2_tensor_payload_round_trips_all_geometry() -> None:
    weights = torch.randn(2, 7, 9, generator=torch.Generator().manual_seed(44))
    geometry = build_centered_router_geometry(weights)
    restored = CenteredRouterGeometry.from_tensor_dict(geometry.to_tensor_dict())
    assert restored.relative_rank_threshold == pytest.approx(
        geometry.relative_rank_threshold
    )
    for name in (
        "expert_keys",
        "input_basis",
        "rank_mask",
        "singular_values",
        "ranks",
        "row_norms",
        "centered_bias",
    ):
        assert torch.equal(getattr(restored, name), getattr(geometry, name))


def test_audit_reports_and_assertion_rejects_lossy_geometry() -> None:
    generator = torch.Generator().manual_seed(91)
    weights = torch.randn(2, 10, 12, generator=generator)
    inputs = torch.randn(5, 2, 12, generator=generator)
    lossy = build_centered_router_geometry(weights, relative_rank_threshold=0.8)
    audit = audit_centered_router_geometry(lossy, weights, router_inputs=inputs, k=4)
    assert audit.relative_weight_rms_error > 0.01
    with pytest.raises(ValueError, match="geometry audit failed"):
        assert_router_geometry_equivalent(
            lossy,
            weights,
            router_inputs=inputs,
            k=4,
        )


def test_geometry_rejects_invalid_sources_and_shapes() -> None:
    with pytest.raises(ValueError, match=r"\[layers, experts, hidden\]"):
        build_centered_router_geometry(torch.randn(5, 7))
    weights = torch.randn(2, 6, 8)
    with pytest.raises(ValueError, match="bias must have shape"):
        build_centered_router_geometry(weights, torch.randn(6))
    bad = weights.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_centered_router_geometry(bad)
