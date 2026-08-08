from __future__ import annotations

import copy

import pytest
import torch

from harp8.config import HARPConfig
from harp8.model import HARP8Teacher
from harp_rtt.anchor import LegacyHARPAnchorBridge


def _config() -> HARPConfig:
    return HARPConfig(
        experts=12,
        layers=2,
        horizons=4,
        route_history=3,
        mtp_depths=3,
        target_state_channels=1,
        target_state_width=4,
        mtp_state_channels=1,
        mtp_state_width=4,
        mtp_state_projection_width=8,
        mtp_metadata_width=4,
        mtp_vocab_width=0,
        route_width=8,
        state_width=8,
        mtp_width=8,
        model_width=8,
        route_ffn_width=16,
        mtp_ffn_width=16,
        fusion_ffn_width=16,
        attention_heads=2,
        temporal_blocks=1,
        layer_blocks=1,
        state_blocks=1,
        mtp_cross_blocks=1,
        fusion_blocks=1,
        dropout=0.0,
        dense_output=True,
        output_rank=4,
        future_latent_width=0,
        mtp_source_dropout=0.0,
        target_state_source_dropout=0.0,
    )


def _batch() -> dict[str, object]:
    torch.manual_seed(9)
    batch, history, layers, experts, nodes, hidden = 2, 3, 2, 12, 4, 6
    meta = torch.zeros(batch, nodes, 13, dtype=torch.int64)
    meta[:, 0, 4] = 1
    meta[:, 1, 4] = 2
    meta[:, 2, 4] = 2
    scalars = torch.zeros(batch, nodes, 8)
    scalars[:, 0, 2] = 0.9
    scalars[:, 1, 2] = 0.4
    scalars[:, 2, 2] = 0.8  # strongest depth-two branch is node two
    mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
    return {
        "metadata": {"request_id": ["a", "b"]},
        "inputs": {
            "history": {
                "logits": torch.randn(batch, history, layers, experts),
                "available": torch.ones(batch, history, layers, dtype=torch.bool),
            },
            "current": {
                "post_moe_residual_xplus": torch.randn(batch, layers, hidden),
            },
            "tree": {
                "states": torch.randn(batch, nodes, 4, hidden),
                "router_logits": torch.randn(batch, nodes, experts),
                "meta": meta,
                "scalars": scalars,
                "mask": mask,
            },
            "within_request": torch.tensor([2.0, 7.0]),
        },
        "targets": {
            "future_router_logits": torch.randn(batch, 4, layers, experts),
            "tree_acceptance": torch.randint(0, 2, (batch, nodes), dtype=torch.bool),
        },
    }


def _adaptive_config() -> HARPConfig:
    return HARPConfig(**{**_config().to_dict(), "mtp_depths": 8})


def _adaptive_batch() -> dict[str, object]:
    """Rich batch whose graph maxima do not describe one coherent branch."""

    batch = _batch()
    inputs = batch["inputs"]
    assert isinstance(inputs, dict)
    tree = inputs["tree"]
    assert isinstance(tree, dict)
    batch_size = 2
    hidden = 6
    experts = 12
    depths = 6

    # If the old independent per-depth graph scan runs, these unmistakable
    # values win.  The adaptive bridge must ignore them for legacy MTP PCA.
    tree["states"] = torch.full((batch_size, 4, 4, hidden), 9_000.0)
    tree["router_logits"] = torch.full((batch_size, 4, experts), -9_000.0)
    tree["adaptive_contract"] = torch.ones(batch_size, dtype=torch.bool)

    anchor_hidden = (
        torch.arange(batch_size * depths * hidden, dtype=torch.float32)
        .reshape(batch_size, depths, hidden)
        .div(10.0)
    )
    anchor_router = (
        torch.arange(batch_size * depths * experts, dtype=torch.float32)
        .reshape(batch_size, depths, experts)
        .div(100.0)
    )
    batch["anchor_inputs"] = {
        "mtp_spine": {
            "hidden_states": anchor_hidden,
            "router_logits": anchor_router,
            "depth": torch.arange(1, depths + 1).expand(batch_size, -1),
            "parent": torch.tensor([-1, 0, 1, 2, 3, 4]).expand(batch_size, -1),
            "token_ids": torch.tensor([77, 2, 303, 404, 505, 606]).expand(
                batch_size, -1
            ),
            "mask": torch.ones(batch_size, depths, dtype=torch.bool),
            "contract": torch.ones(batch_size, dtype=torch.bool),
        }
    }
    return batch


def _move(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def test_anchor_bridge_matches_direct_legacy_forward_and_ignores_labels() -> None:
    torch.manual_seed(4)
    config = _config()
    anchor = HARP8Teacher(config).eval()
    target_means = torch.randn(config.layers, 6)
    target_components = torch.randn(config.layers, 6, config.target_state_width)
    mtp_means = torch.randn(2, 6)
    mtp_components = torch.randn(2, 6, config.mtp_state_width)
    bridge = LegacyHARPAnchorBridge(
        anchor,
        target_means=target_means,
        target_components=target_components,
        mtp_means=mtp_means,
        mtp_components=mtp_components,
    ).eval()
    batch = _batch()
    legacy = bridge.legacy_inputs(batch)
    expected = anchor(**legacy)
    actual = bridge(batch=batch)
    assert torch.equal(
        legacy["route_history"],
        batch["inputs"]["history"]["logits"].permute(0, 2, 1, 3),
    )
    assert legacy["mtp_available"].tolist() == [
        [True, True, False],
        [True, True, False],
    ]
    for key in expected:
        assert torch.equal(actual[key], expected[key])

    changed = copy.deepcopy(batch)
    changed["targets"]["future_router_logits"].normal_(mean=1000.0, std=1.0)
    changed["targets"]["tree_acceptance"].logical_not_()
    ignored = bridge(batch=changed)
    for key in actual:
        assert torch.equal(ignored[key], actual[key])


def test_adaptive_anchor_bridge_uses_explicit_h1_h6_spine_not_graph_maxima() -> None:
    torch.manual_seed(24)
    config = _adaptive_config()
    mtp_means = torch.randn(6, 6)
    mtp_components = torch.randn(6, 6, config.mtp_state_width)
    bridge = LegacyHARPAnchorBridge(
        HARP8Teacher(config),
        target_means=torch.randn(config.layers, 6),
        target_components=torch.randn(config.layers, 6, config.target_state_width),
        mtp_means=mtp_means,
        mtp_components=mtp_components,
    ).eval()
    batch = _adaptive_batch()
    legacy = bridge.legacy_inputs(batch)
    assert "anchor_inputs" not in batch["inputs"]
    anchor_inputs = batch["anchor_inputs"]
    assert isinstance(anchor_inputs, dict)
    spine = anchor_inputs["mtp_spine"]
    assert isinstance(spine, dict)

    expected_features = torch.einsum(
        "bdh,dhq->bdq",
        spine["hidden_states"].float() - mtp_means[None],
        mtp_components,
    )
    assert torch.equal(legacy["mtp_states"][:, :6, 0], expected_features)
    assert torch.equal(legacy["mtp_router_logits"][:, :6], spine["router_logits"])
    assert legacy["mtp_available"].tolist() == [
        [True, True, True, True, True, True, False, False],
        [True, True, True, True, True, True, False, False],
    ]
    # Sentinel graph values prove that no per-depth adaptive-tree argmax fed
    # the incumbent preprocessing channel.
    assert not torch.any(legacy["mtp_router_logits"][:, :6] == -9_000.0)

    expected = bridge.anchor(**legacy)
    actual = bridge(batch=batch)
    for name in expected:
        assert torch.equal(actual[name], expected[name])


@pytest.mark.parametrize(
    "mutation",
    ("missing", "contract", "mask", "depth", "parent"),
)
def test_adaptive_anchor_bridge_fails_closed_on_missing_or_incoherent_spine(
    mutation: str,
) -> None:
    config = _adaptive_config()
    bridge = LegacyHARPAnchorBridge(
        HARP8Teacher(config),
        target_means=torch.zeros(config.layers, 6),
        target_components=torch.randn(config.layers, 6, config.target_state_width),
        mtp_means=torch.zeros(6, 6),
        mtp_components=torch.randn(6, 6, config.mtp_state_width),
    ).eval()
    batch = _adaptive_batch()
    anchor_inputs = batch["anchor_inputs"]
    assert isinstance(anchor_inputs, dict)
    if mutation == "missing":
        del batch["anchor_inputs"]
    else:
        spine = anchor_inputs["mtp_spine"]
        assert isinstance(spine, dict)
        if mutation == "contract":
            spine["contract"][0] = False
        elif mutation == "mask":
            spine["mask"][0, -1] = False
        elif mutation == "depth":
            spine["depth"][0, -1] = 5
        elif mutation == "parent":
            spine["parent"][0, -1] = 2

    with pytest.raises(
        (KeyError, TypeError, ValueError),
        match="(?i)anchor|spine|depth|parent|contract|mask",
    ):
        bridge.legacy_inputs(batch)


@pytest.mark.parametrize(
    "device_type",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_anchor_frozen_pca_inputs_are_exact_under_bfloat16_autocast(
    device_type: str,
) -> None:
    device = torch.device(device_type)
    torch.manual_seed(41)
    config = _config()
    bridge = (
        LegacyHARPAnchorBridge(
            HARP8Teacher(config),
            target_means=torch.randn(config.layers, 6),
            target_components=torch.randn(config.layers, 6, config.target_state_width),
            mtp_means=torch.randn(2, 6),
            mtp_components=torch.randn(2, 6, config.mtp_state_width),
        )
        .to(device)
        .eval()
    )
    batch = _move(_batch(), device)
    assert isinstance(batch, dict)
    reference_inputs = bridge.legacy_inputs(batch)
    reference_outputs = bridge(batch=batch)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        autocast_inputs = bridge.legacy_inputs(batch)
        # This is the incumbent under its production mixed-precision policy,
        # supplied with the non-autocast canonical bridge inputs.
        expected_outputs = bridge.anchor(**reference_inputs)
        actual_outputs = bridge(batch=batch)

    for name in ("target_states", "mtp_states"):
        assert autocast_inputs[name].dtype == torch.float32
        assert torch.equal(autocast_inputs[name], reference_inputs[name])
    for name in expected_outputs:
        # The bridge adds no autocast drift beyond the incumbent network's own
        # mixed-precision execution.
        assert torch.equal(actual_outputs[name], expected_outputs[name])
        assert torch.allclose(
            actual_outputs[name].float(),
            reference_outputs[name].float(),
            atol=2e-2,
            rtol=2e-2,
        )


def test_anchor_bridge_is_permanently_frozen() -> None:
    config = _config()
    bridge = LegacyHARPAnchorBridge(
        HARP8Teacher(config),
        target_means=torch.zeros(config.layers, 6),
        target_components=torch.randn(config.layers, 6, config.target_state_width),
        mtp_means=torch.zeros(2, 6),
        mtp_components=torch.randn(2, 6, config.mtp_state_width),
    )
    bridge.train()
    assert not bridge.anchor.training
    assert not any(parameter.requires_grad for parameter in bridge.anchor.parameters())


def test_anchor_bridge_clamps_position_to_legacy_training_support() -> None:
    config = _config()
    bridge = LegacyHARPAnchorBridge(
        HARP8Teacher(config),
        target_means=torch.zeros(config.layers, 6),
        target_components=torch.randn(config.layers, 6, config.target_state_width),
        mtp_means=torch.zeros(2, 6),
        mtp_components=torch.randn(2, 6, config.mtp_state_width),
    ).eval()
    late = _batch()
    late["inputs"]["within_request"] = torch.tensor([32.0, 400.0])
    clipped = copy.deepcopy(late)
    clipped["inputs"]["within_request"] = torch.tensor([32.0, 32.0])

    late_inputs = bridge.legacy_inputs(late)
    assert late_inputs["within_request"].tolist() == [32.0, 32.0]
    late_outputs = bridge(batch=late)
    clipped_outputs = bridge(batch=clipped)
    for name in late_outputs:
        assert torch.equal(late_outputs[name], clipped_outputs[name])

    invalid = copy.deepcopy(late)
    invalid["inputs"]["within_request"] = torch.tensor([-1.0, 2.0])
    with pytest.raises(ValueError, match="finite and non-negative"):
        bridge.legacy_inputs(invalid)


def test_anchor_bridge_allows_selected_phase3_gradients() -> None:
    config = _config()
    bridge = LegacyHARPAnchorBridge(
        HARP8Teacher(config),
        target_means=torch.zeros(config.layers, 6),
        target_components=torch.randn(config.layers, 6, config.target_state_width),
        mtp_means=torch.zeros(2, 6),
        mtp_components=torch.randn(2, 6, config.mtp_state_width),
    ).eval()
    # Phase 3 deliberately re-enables only selected legacy upper parameters.
    bridge.anchor.output_bias.requires_grad_(True)
    loss = bridge(batch=_batch())["future_router_scores"].sum()
    loss.backward()
    assert bridge.anchor.output_bias.grad is not None
    assert torch.isfinite(bridge.anchor.output_bias.grad).all()
