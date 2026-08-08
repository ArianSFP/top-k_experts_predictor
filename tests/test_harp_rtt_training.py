from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from harp_rtt.losses import GradientAudit
from harp_rtt.training import (
    OptimizerStepState,
    PhaseSpec,
    autocast_context,
    configure_training_phase,
    cosine_warmup_multiplier,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    seed_everything,
)


class _Legacy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lower = nn.Linear(3, 3)
        self.layer_route_blocks = nn.ModuleList([nn.Linear(3, 3)])
        self.target_state_blocks = nn.ModuleList([nn.Linear(3, 3)])
        self.output_bias = nn.Parameter(torch.zeros(3))


class _Bridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = _Legacy()


class _TinyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = _Bridge()
        self.token_embedding = nn.Embedding(8, 3)
        self.route_encoder = nn.Linear(3, 3)
        self.target_encoder = nn.Linear(3, 3)
        self.score_head = nn.Linear(3, 3)
        self.branch_residual = nn.Linear(3, 3)
        self.trajectory = nn.Module()
        self.trajectory.correction_head = nn.Linear(3, 3)
        self.decoder = nn.Module()
        self.decoder.exact_adapter = nn.Linear(3, 3)
        self.reranker = nn.Linear(3, 3)
        self._anchor_frozen = True


def _trainable(model: nn.Module) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def test_phase_ownership_is_explicit_and_phase3_uses_lower_legacy_lr() -> None:
    model = _TinyTeacher()
    groups, report = configure_training_phase(
        model, PhaseSpec("phase2", epochs=1, new_learning_rate=2e-4)
    )
    assert len(groups) == 1
    assert report.legacy_parameter_count == 0
    assert all(not name.startswith("anchor.") for name in _trainable(model))
    assert all(not name.startswith("reranker.") for name in _trainable(model))
    assert model._anchor_frozen

    groups, report = configure_training_phase(
        model,
        PhaseSpec(
            "phase3",
            epochs=1,
            new_learning_rate=1e-4,
            legacy_learning_rate=1e-5,
        ),
    )
    assert [group["group_name"] for group in groups] == ["new", "legacy_upper"]
    assert groups[1]["lr"] == pytest.approx(groups[0]["lr"] / 10)
    assert report.legacy_parameter_count > 0
    assert any("layer_route_blocks" in name for name in report.legacy_names)
    assert all("lower" not in name for name in report.legacy_names)
    assert not model._anchor_frozen

    configure_training_phase(
        model, PhaseSpec("phase4", epochs=1, new_learning_rate=2e-4)
    )
    assert _trainable(model) == {"reranker.weight", "reranker.bias"}
    assert model._anchor_frozen

    configure_training_phase(
        model, PhaseSpec("phase5", epochs=1, new_learning_rate=5e-5)
    )
    names = _trainable(model)
    assert {"reranker.weight", "score_head.weight", "branch_residual.weight"} <= names
    assert not any(name.startswith("route_encoder.") for name in names)


def test_phase3_rejects_insufficient_legacy_lr_separation() -> None:
    with pytest.raises(ValueError, match="at least 5x"):
        PhaseSpec(
            "phase3",
            epochs=1,
            new_learning_rate=1e-4,
            legacy_learning_rate=3e-5,
        ).validate()


def test_nested_move_preserves_role_tuple_and_strings() -> None:
    batch = {
        "inputs": {"tensor": torch.ones(2), "current_roles": ("a", "b")},
        "metadata": {"request_id": ["r0", "r1"]},
    }
    moved = move_to_device(batch, "cpu")
    assert moved["inputs"]["tensor"].device.type == "cpu"
    assert moved["inputs"]["current_roles"] == ("a", "b")
    assert moved["metadata"]["request_id"] == ["r0", "r1"]


def test_cpu_diagnostic_autocast_accepts_real_bf16_capture_features() -> None:
    """CPU smoke must reproduce the production learned-network dtype policy."""

    projection = nn.Sequential(nn.RMSNorm(8), nn.Linear(8, 4))
    captured = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    with autocast_context("cpu", enabled=True):
        output = projection(captured)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_cosine_warmup_boundaries() -> None:
    assert cosine_warmup_multiplier(0, 100, 10) == pytest.approx(0.1)
    assert cosine_warmup_multiplier(9, 100, 10) == pytest.approx(1.0)
    assert cosine_warmup_multiplier(100, 100, 10) == pytest.approx(0.0)


def test_seed_policy_reserves_ieee_fp32_for_router_geometry() -> None:
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        seed_everything(81)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def test_checkpoint_round_trip_is_hash_verified_and_non_overwriting(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = _TinyTeacher()
    groups, _ = configure_training_phase(
        model, PhaseSpec("phase2", epochs=1, new_learning_rate=2e-4)
    )
    optimizer = torch.optim.AdamW(groups)
    state = OptimizerStepState(global_step=7, micro_step=2, auxiliary_scale=0.5)
    phase = PhaseSpec("phase2", epochs=1, new_learning_rate=2e-4)
    audit = GradientAudit()
    manifest = save_checkpoint(
        tmp_path,
        tag="epoch_001",
        model=model,
        optimizer=optimizer,
        state=state,
        phase=phase,
        gradient_audit=audit,
        provenance={"test": True},
    )
    manifest_path = tmp_path / "epoch_001.manifest.json"
    assert manifest["model"]["sha256"]

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    restored, loaded_manifest = load_checkpoint(
        manifest_path, model=model, optimizer=optimizer, gradient_audit=audit
    )
    assert restored.global_step == 7
    assert restored.auxiliary_scale == pytest.approx(0.5)
    assert loaded_manifest == manifest
    with pytest.raises(FileExistsError):
        save_checkpoint(
            tmp_path,
            tag="epoch_001",
            model=model,
            optimizer=optimizer,
            state=state,
            phase=phase,
            gradient_audit=audit,
            provenance={},
        )
