from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from harp8.config import HARPConfig, LossConfig
from harp8.data import CompactHARPData
from harp8.evaluate import evaluate_checkpoint
from harp8.features import make_inner_split
from harp8.losses import endpoint_loss
from harp8.metrics import evaluate_split
from harp8.model import HARP8Teacher


def _config(**overrides: object) -> HARPConfig:
    values: dict[str, object] = {
        "experts": 16,
        "layers": 4,
        "horizons": 3,
        "route_history": 3,
        "mtp_depths": 3,
        "target_state_channels": 2,
        "target_state_width": 8,
        "mtp_state_channels": 2,
        "mtp_state_width": 8,
        "mtp_metadata_width": 4,
        "mtp_vocab_width": 0,
        "route_width": 32,
        "state_width": 32,
        "mtp_width": 32,
        "model_width": 32,
        "route_ffn_width": 64,
        "mtp_ffn_width": 64,
        "fusion_ffn_width": 64,
        "attention_heads": 4,
        "temporal_blocks": 1,
        "layer_blocks": 1,
        "state_blocks": 1,
        "mtp_cross_blocks": 1,
        "fusion_blocks": 1,
        "dropout": 0.0,
        "future_latent_width": 8,
        "mtp_source_dropout": 0.0,
        "target_state_source_dropout": 0.0,
    }
    values.update(overrides)
    return HARPConfig(**values)


def _inputs(config: HARPConfig, batch_size: int = 2) -> dict[str, torch.Tensor]:
    nodes = config.mtp_depths
    return {
        "route_history": torch.randn(
            batch_size,
            config.layers,
            config.route_history,
            config.experts,
        ),
        "route_available": torch.ones(
            batch_size, config.route_history, dtype=torch.bool
        ),
        "target_states": torch.randn(
            batch_size,
            config.layers,
            config.target_state_channels,
            config.target_state_width,
        ),
        "target_state_available": torch.ones(
            batch_size,
            config.layers,
            config.target_state_channels,
            dtype=torch.bool,
        ),
        "mtp_states": torch.randn(
            batch_size,
            nodes,
            config.mtp_state_channels,
            config.mtp_state_width,
        ),
        "mtp_router_logits": torch.randn(batch_size, nodes, config.experts),
        "mtp_metadata": torch.randn(
            batch_size, nodes, config.mtp_metadata_width
        ),
        "mtp_depth_ids": torch.arange(1, nodes + 1)[None].expand(
            batch_size, -1
        ),
        "mtp_available": torch.ones(batch_size, nodes, dtype=torch.bool),
    }


def _labels(
    config: HARPConfig, inputs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    teacher = torch.randn(
        inputs["route_history"].shape[0],
        config.horizons,
        config.layers,
        config.experts,
    )
    return {
        **inputs,
        "teacher_router_scores": teacher,
        "target_top8": torch.topk(teacher, 8, dim=-1).indices,
        "valid_future": torch.ones(teacher.shape[:2], dtype=torch.bool),
        "future_latent_target": torch.randn(
            teacher.shape[0], config.horizons, config.future_latent_width
        ),
    }


def test_harp_outputs_all_layers_horizons_and_experts() -> None:
    config = _config()
    model = HARP8Teacher(config).eval()
    outputs = model(**_inputs(config))
    assert outputs["future_router_scores"].shape == (2, 3, 4, 16)
    assert outputs["future_inclusion_probabilities"].shape == (2, 3, 4, 16)
    assert outputs["source_gate_weights"].shape == (2, 3, 4, 3)
    assert outputs["generator_context"].shape == (2, 3, 4, 32)
    assert outputs["future_latent"].shape == (2, 3, 8)
    assert torch.isfinite(outputs["future_router_scores"]).all()
    assert torch.allclose(
        outputs["source_gate_weights"].sum(dim=-1),
        torch.ones(2, 3, 4),
        atol=1e-6,
    )
    assert model.output_weight is not None
    assert model.output_weight.shape == (3, 4, 32, 16)


def test_mtp_node_order_is_not_an_input_order_signal() -> None:
    torch.manual_seed(11)
    config = _config()
    model = HARP8Teacher(config).eval()
    inputs = _inputs(config)
    first = model(**inputs)["future_router_scores"]
    permutation = torch.tensor([2, 0, 1])
    for name in (
        "mtp_states",
        "mtp_router_logits",
        "mtp_metadata",
        "mtp_depth_ids",
        "mtp_available",
    ):
        inputs[name] = inputs[name][:, permutation]
    second = model(**inputs)["future_router_scores"]
    assert torch.allclose(first, second, atol=1e-6)


def test_all_missing_mtp_is_finite_and_fully_masked() -> None:
    config = _config()
    model = HARP8Teacher(config).eval()
    inputs = _inputs(config)
    inputs["mtp_available"].zero_()
    outputs = model(**inputs)
    assert torch.isfinite(outputs["future_router_scores"]).all()
    assert torch.equal(
        outputs["source_gate_weights"][..., HARP8Teacher.SOURCE_MTP],
        torch.zeros_like(
            outputs["source_gate_weights"][..., HARP8Teacher.SOURCE_MTP]
        ),
    )


def test_forward_kl_direction_and_h2_only_mask() -> None:
    config = _config(future_latent_width=0)
    inputs = _inputs(config)
    labels = _labels(config, inputs)
    teacher = labels["teacher_router_scores"]
    loss_config = LossConfig(
        hard_negative_end_rank=16, future_latent=0.0
    )
    exact = endpoint_loss(
        {"future_router_scores": teacher.clone()},
        labels,
        loss_config,
        active_horizons=(2,),
    )
    changed = teacher.clone()
    changed[:, 0] += torch.randn_like(changed[:, 0]) * 100
    changed[:, 2] += torch.randn_like(changed[:, 2]) * 100
    masked = endpoint_loss(
        {"future_router_scores": changed},
        labels,
        loss_config,
        active_horizons=(2,),
    )
    noisy = teacher.clone()
    noisy[:, 1] = -noisy[:, 1]
    wrong = endpoint_loss(
        {"future_router_scores": noisy},
        labels,
        loss_config,
        active_horizons=(2,),
    )
    assert exact.components["router_kl"].abs() < 1e-6
    assert torch.allclose(exact.total, masked.total, atol=1e-6)
    assert wrong.components["router_kl"] > exact.components["router_kl"]




def test_explicit_horizon_weights_mask_inactive_horizon() -> None:
    config = _config(future_latent_width=0)
    inputs = _inputs(config)
    labels = _labels(config, inputs)
    loss_config = LossConfig(
        hard_negative_end_rank=16,
        future_latent=0.0,
        horizon_weights=(1.0, 0.0, 4.0),
    )
    changed = labels["teacher_router_scores"].clone()
    changed[:, 1] += torch.randn_like(changed[:, 1]) * 100
    baseline = endpoint_loss(
        {"future_router_scores": labels["teacher_router_scores"]},
        labels,
        loss_config,
        active_horizons=(1, 3),
    )
    masked = endpoint_loss(
        {"future_router_scores": changed},
        labels,
        loss_config,
        active_horizons=(1, 3),
    )
    assert torch.allclose(baseline.total, masked.total, atol=1e-6)


def test_candidate16_boundary_loss_responds_to_ninth_negative() -> None:
    config = _config(experts=24, future_latent_width=0)
    inputs = _inputs(config)
    labels = _labels(config, inputs)
    teacher = labels["teacher_router_scores"]
    teacher_top8 = labels["target_top8"]
    good = torch.full_like(teacher, -1.0)
    good.scatter_(-1, teacher_top8, 2.0)
    bad = good.clone()
    bad.scatter_(-1, teacher_top8[..., :1], -2.0)
    loss_config = LossConfig(
        hard_negative_end_rank=24,
        future_latent=0.0,
        router_kl=0.0,
        boundary=0.0,
        inclusion=0.0,
        centered_score=0.0,
        candidate16=1.0,
        candidate_count=16,
    )
    good_loss = endpoint_loss(
        {"future_router_scores": good}, labels, loss_config, active_horizons=(1, 2, 3)
    )
    bad_loss = endpoint_loss(
        {"future_router_scores": bad}, labels, loss_config, active_horizons=(1, 2, 3)
    )
    assert bad_loss.components["candidate16"] > good_loss.components["candidate16"]


def test_full_membership_loss_uses_all_non_top8_experts() -> None:
    config = _config(experts=24, future_latent_width=0)
    inputs = _inputs(config)
    labels = _labels(config, inputs)
    teacher = labels["teacher_router_scores"]
    top8 = labels["target_top8"]
    first = teacher.clone()
    second = teacher.clone()
    excluded = {int(item) for item in top8[0, 0, 0]}
    negative = next(value for value in range(config.experts) if value not in excluded)
    second[:, :, :, negative] += 100.0
    loss_config = LossConfig(
        hard_negative_end_rank=24,
        future_latent=0.0,
        router_kl=0.0,
        boundary=0.0,
        inclusion=0.0,
        centered_score=0.0,
        full_membership=1.0,
    )
    first_loss = endpoint_loss(
        {"future_router_scores": first}, labels, loss_config, active_horizons=(1, 2, 3)
    )
    second_loss = endpoint_loss(
        {"future_router_scores": second}, labels, loss_config, active_horizons=(1, 2, 3)
    )
    assert second_loss.components["full_membership"] > first_loss.components[
        "full_membership"
    ]


def test_inner_split_is_domain_stratified_and_deterministic() -> None:
    requests = [
        {"request_id": 1, "offline_split": "train", "domain": "a"},
        {"request_id": 2, "offline_split": "train", "domain": "a"},
        {"request_id": 3, "offline_split": "train", "domain": "b"},
        {"request_id": 4, "offline_split": "train", "domain": "b"},
        {"request_id": 5, "offline_split": "validation", "domain": "a"},
    ]
    first = make_inner_split(requests, development_fraction=0.5, seed=7)
    second = make_inner_split(requests, development_fraction=0.5, seed=7)
    assert first == second
    assert first["counts"] == {"train": 2, "validation": 2}
    assert set(first["assignments"]) == {"1", "2", "3", "4"}
def _write_fixture(
    root: Path,
) -> tuple[Path, Path, np.ndarray, np.ndarray, HARPConfig]:
    capture = root / "capture"
    mtp = root / "mtp"
    capture.mkdir()
    mtp.mkdir()
    config = _config(
        target_state_channels=1,
        target_state_width=4,
        mtp_state_channels=1,
        mtp_state_width=4,
    )
    request_count = 3
    rows_per_request = 5
    rows = request_count * rows_per_request
    rng = np.random.default_rng(9)
    raw = rng.normal(
        size=(rows, config.layers, config.experts)
    ).astype(np.float32)
    np.save(capture / "raw_router_logits.npy", raw)
    np.save(
        capture / "top8_expert_ids.npy",
        np.argsort(-raw, axis=-1)[..., :8].astype(np.uint16),
    )
    np.save(
        mtp / "mtp_router_logits_depths.npy",
        rng.normal(size=(rows, 2, config.experts)).astype(np.float32),
    )
    with (capture / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for request_id, split in enumerate(
            ("train", "validation", "test"), 1
        ):
            handle.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "offline_split": split,
                        "domain": "synthetic",
                    }
                )
                + "\n"
            )
    target = rng.normal(
        size=(rows, config.layers, 4)
    ).astype(np.float16)
    mtp_state = rng.normal(size=(rows, 2, 4)).astype(np.float16)
    return capture, mtp, target, mtp_state, config


def test_compact_adapter_masks_missing_depths_and_seals_test(
    tmp_path: Path,
) -> None:
    capture, mtp, target, mtp_state, config = _write_fixture(tmp_path)
    data = CompactHARPData(
        capture,
        mtp,
        target,
        mtp_state,
        config,
        rows_per_request=5,
    )
    with pytest.raises(PermissionError, match="sealed test"):
        data.indices("test")
    assert len(data.indices("test", allow_test=True)) == 4
    batch = data.batch(np.asarray([0, 1]), "cpu")
    assert batch["route_history"].shape == (2, 4, 3, 16)
    assert torch.equal(
        batch["mtp_available"],
        torch.tensor([[True, True, False], [True, True, False]]),
    )
    assert not batch["route_available"][0, 1:].any()
    assert batch["route_available"][1, :2].all()


def test_checkpoint_round_trip_is_exact(tmp_path: Path) -> None:
    config = _config()
    model = HARP8Teacher(config).eval()
    inputs = _inputs(config)
    expected = model(**inputs)["future_router_scores"]
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": config.to_dict(), "model": model.state_dict()}, path)
    payload = torch.load(path, weights_only=False)
    restored = HARP8Teacher(HARPConfig(**payload["config"])).eval()
    restored.load_state_dict(payload["model"], strict=True)
    actual = restored(**inputs)["future_router_scores"]
    assert torch.equal(expected, actual)


def test_vectorized_evaluation_preserves_grouped_contract(tmp_path: Path) -> None:
    capture, mtp, target, mtp_state, config = _write_fixture(tmp_path)
    data = CompactHARPData(
        capture,
        mtp,
        target,
        mtp_state,
        config,
        rows_per_request=5,
    )
    result = evaluate_split(
        HARP8Teacher(config).eval(),
        data,
        "validation",
        batch_size=2,
        device="cpu",
    )
    assert len(result.horizon_metrics) == config.horizons
    assert len(result.layer_metrics) == config.horizons * config.layers
    assert len(result.domain_metrics) == config.horizons
    assert result.horizon_metrics[0]["source_rows"] == 4
    assert result.horizon_metrics[2]["source_rows"] == 2
    assert 0.0 <= result.h2_request_macro_recall <= 1.0


def test_sealed_test_requires_explicit_confirmation(tmp_path: Path) -> None:
    capture, mtp, target, mtp_state, config = _write_fixture(tmp_path)
    data = CompactHARPData(
        capture,
        mtp,
        target,
        mtp_state,
        config,
        rows_per_request=5,
    )
    with pytest.raises(PermissionError, match="test evaluation requires"):
        evaluate_checkpoint(
            tmp_path / "not_opened.pt",
            data,
            tmp_path / "test-output",
            split="test",
            device="cpu",
        )


def test_tiny_trace_can_be_overfit() -> None:
    torch.manual_seed(17)
    config = _config(
        experts=10,
        layers=2,
        horizons=2,
        route_history=2,
        mtp_depths=2,
        route_width=16,
        state_width=16,
        mtp_width=16,
        model_width=16,
        route_ffn_width=32,
        mtp_ffn_width=32,
        fusion_ffn_width=32,
        target_state_width=4,
        mtp_state_width=4,
        future_latent_width=0,
    )
    inputs = _inputs(config, batch_size=4)
    labels = _labels(config, inputs)
    model = HARP8Teacher(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_config = LossConfig(
        hard_negative_end_rank=10,
        future_latent=0.0,
    )

    def loss_value() -> torch.Tensor:
        return endpoint_loss(
            model(**inputs),
            labels,
            loss_config,
            active_horizons=(2,),
        ).total

    initial = float(loss_value().detach())
    for _step in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        optimizer.step()
    final = float(loss_value().detach())
    assert final < initial * 0.35
