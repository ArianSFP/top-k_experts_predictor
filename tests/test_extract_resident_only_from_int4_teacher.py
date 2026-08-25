from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from runpod.extract_resident_only_from_int4_teacher import (
    ExtractionContract,
    _canonical_ids_sha256,
    extract_layers,
    load_teacher_layer,
    sha256_file,
    validate_allocation_plan,
    validate_teacher_bundle,
)


CONTRACT = ExtractionContract(
    layers=2,
    experts=4,
    hidden_width=4,
    intermediate_width=2,
    group_size=2,
    exact_k=2,
    core_size=2,
    total_residents=4,
)
TEACHER_SOURCE = "a" * 40
OUTPUT_SOURCE = "b" * 40
TARGET_SHA = "c" * 64


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tensor(shape: tuple[int, ...], dtype: torch.dtype, offset: int) -> torch.Tensor:
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
    values = values.reshape(shape) + offset
    if dtype == torch.uint8:
        return values.remainder(251).to(dtype)
    return (values / 17.0).to(dtype)


def _teacher_state(layer: int) -> dict[str, torch.Tensor]:
    shapes = CONTRACT.teacher_shapes
    return {
        "gate_up_packed": _tensor(shapes["gate_up_packed"], torch.uint8, layer),
        "gate_up_scales": _tensor(
            shapes["gate_up_scales"], torch.bfloat16, 10 + layer
        ),
        "down_packed": _tensor(shapes["down_packed"], torch.uint8, 20 + layer),
        "down_scales": _tensor(shapes["down_scales"], torch.bfloat16, 30 + layer),
    }


def _build_teacher(root: Path) -> None:
    root.mkdir()
    manifest = {
        "schema": "harp_shadowroute_int4_top4_export_v1",
        "source_commit": TEACHER_SOURCE,
        "target_checkpoint_index_sha256": TARGET_SHA,
        "group_size": CONTRACT.group_size,
        "active_slots": CONTRACT.exact_k,
        "quantization": "symmetric_signed_int4_per_output_group",
        "optimizer_constructed": False,
        "training_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    _json(root / "run_manifest.json", manifest)
    metrics = []
    for layer in range(CONTRACT.layers):
        directory = root / f"layer_{layer:02d}"
        directory.mkdir()
        path = directory / f"shadow_int4_top4_layer_{layer:02d}.pt"
        torch.save(
            {
                "schema": "harp_shadowroute_local_expert_layer_v1",
                "mode": "int4_top4",
                "layer": layer,
                "source_commit": TEACHER_SOURCE,
                "target_checkpoint_index_sha256": TARGET_SHA,
                "model_state_dict": _teacher_state(layer),
                "group_size": CONTRACT.group_size,
                "active_slots": CONTRACT.exact_k,
                "optimizer_constructed": False,
                "training_started": False,
                "formal_validation_opened": False,
                "calibration_opened": False,
                "sealed_test_opened": False,
            },
            path,
        )
        metrics.append({"layer": layer, "checkpoint_sha256": sha256_file(path)})
    _json(
        root / "STAGE_RESULT.json",
        {
            "schema": "harp_shadowroute_int4_top4_export_result_v1",
            "layers": CONTRACT.layers,
            "group_size": CONTRACT.group_size,
            "active_slots": CONTRACT.exact_k,
            "layer_metrics": metrics,
            "closed_loop_authorized": False,
            "optimizer_constructed": False,
            "training_started": False,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        },
    )
    files = sorted(path for path in root.rglob("*") if path.is_file())
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def _plan() -> dict:
    frequencies = [[10, 9, 2, 1], [1, 10, 9, 2]]
    core = [[0, 1], [1, 2]]
    residents = [[0, 1], [2, 1]]
    return {
        "schema": "harp_shadowroute_resident_allocation_v2",
        "source_commit": OUTPUT_SOURCE,
        "allocation_objective": "frequency_core_locked_test",
        "expert_counts": frequencies,
        "resident_expert_ids_by_layer": residents,
        "resident_counts_by_layer": [2, 2],
        "coverage_by_layer": [0.75, 0.8],
        "weight_mass_coverage_by_layer": [0.7, 0.85],
        "total_residents": CONTRACT.total_residents,
        "resident_cell_bytes": CONTRACT.resident_cell_bytes,
        "packed_int4_bytes": CONTRACT.resident_cell_bytes
        * CONTRACT.total_residents,
        "fallback_bytes": 0,
        "frequency_core_inclusion_verified": True,
        "frequency_core_size_per_layer": CONTRACT.core_size,
        "frequency_core_cells": CONTRACT.layers * CONTRACT.core_size,
        "frequency_core_sha256": _canonical_ids_sha256(core),
        "resident_ids_sha256": _canonical_ids_sha256(residents),
        "partition_manifest_sha256": "d" * 64,
        "selection_uses_train_routes_only": True,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }


def test_extractor_selects_teacher_tensors_byte_exactly(tmp_path: Path) -> None:
    teacher_root = tmp_path / "teacher"
    plan_path = tmp_path / "resident_allocation.json"
    output = tmp_path / "output"
    _build_teacher(teacher_root)
    _json(plan_path, _plan())

    result = extract_layers(
        teacher_bundle=teacher_root,
        allocation_plan=plan_path,
        output=output,
        source_commit=OUTPUT_SOURCE,
        target_checkpoint_index_sha256=TARGET_SHA,
        layers=[1],
        contract=CONTRACT,
    )

    assert result["layer_count"] == 1
    directory = output / "layer_01"
    checkpoint_path = directory / "shadow_resident_int4_only_layer_01.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    teacher = torch.load(
        teacher_root / "layer_01/shadow_int4_top4_layer_01.pt",
        map_location="cpu",
        weights_only=True,
    )["model_state_dict"]
    ids = torch.tensor([2, 1])
    state = checkpoint["model_state_dict"]
    for name in ("gate_up_packed", "gate_up_scales", "down_packed", "down_scales"):
        assert torch.equal(state[name], teacher[name].index_select(0, ids))
    assert state["resident_ids"].tolist() == [2, 1]
    assert state["expert_to_resident"].tolist() == [-1, 1, 0, -1]
    assert checkpoint["source_commit"] == OUTPUT_SOURCE
    assert checkpoint["target_checkpoint_index_sha256"] == TARGET_SHA
    assert checkpoint["closed_loop_authorized"] is False

    stage = json.loads((directory / "STAGE_RESULT.json").read_text())
    assert stage["schema"] == "harp_shadowroute_local_expert_layer_result_v1"
    assert stage["checkpoint_sha256"] == sha256_file(checkpoint_path)
    assert stage["byte_exact_tensor_verification_passed"] is True
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha256_file(directory / name) == digest


def test_allocation_recomputes_core_instead_of_trusting_flag() -> None:
    plan = copy.deepcopy(_plan())
    plan["resident_expert_ids_by_layer"][0] = [0, 2]
    plan["resident_ids_sha256"] = _canonical_ids_sha256(
        plan["resident_expert_ids_by_layer"]
    )
    with pytest.raises(ValueError, match="omits mandatory frequency core"):
        validate_allocation_plan(
            plan, source_commit=OUTPUT_SOURCE, contract=CONTRACT
        )


def test_teacher_target_and_selected_checkpoint_hashes_are_enforced(
    tmp_path: Path,
) -> None:
    teacher_root = tmp_path / "teacher"
    _build_teacher(teacher_root)
    with pytest.raises(ValueError, match="target_checkpoint_index_sha256 mismatch"):
        validate_teacher_bundle(
            teacher_root,
            target_checkpoint_index_sha256="e" * 64,
            contract=CONTRACT,
        )

    teacher = validate_teacher_bundle(
        teacher_root,
        target_checkpoint_index_sha256=TARGET_SHA,
        contract=CONTRACT,
    )
    with teacher.paths[1].open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_teacher_layer(teacher, 1, contract=CONTRACT)
