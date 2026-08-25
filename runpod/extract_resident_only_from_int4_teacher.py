#!/usr/bin/env python3
"""Extract fallback-free resident shards from an immutable full INT4 bundle.

This is a tensor-selection operation only.  It never opens the target model,
constructs an optimizer, reads a capture, or requantizes a weight.  Each output
packed/scaled tensor must be byte-identical to the corresponding rows in the
all-256-expert group-64 teacher shard.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


PLAN_SCHEMA = "harp_shadowroute_resident_allocation_v2"
TEACHER_MANIFEST_SCHEMA = "harp_shadowroute_int4_top4_export_v1"
TEACHER_RESULT_SCHEMA = "harp_shadowroute_int4_top4_export_result_v1"
LOCAL_SCHEMA = "harp_shadowroute_local_expert_layer_v1"
LAYER_RESULT_SCHEMA = "harp_shadowroute_local_expert_layer_result_v1"
TEACHER_MODE = "int4_top4"
OUTPUT_MODE = "resident_int4_only"
TEACHER_STATE_KEYS = {
    "gate_up_packed",
    "gate_up_scales",
    "down_packed",
    "down_scales",
}
OUTPUT_STATE_KEYS = TEACHER_STATE_KEYS | {"resident_ids", "expert_to_resident"}
SAFETY_FIELDS = {
    "optimizer_constructed": False,
    "training_started": False,
    "formal_validation_opened": False,
    "calibration_opened": False,
    "sealed_test_opened": False,
}


@dataclass(frozen=True)
class ExtractionContract:
    layers: int = 40
    experts: int = 256
    hidden_width: int = 2048
    intermediate_width: int = 512
    group_size: int = 64
    exact_k: int = 8
    core_size: int = 64
    total_residents: int = 3850

    @property
    def resident_cell_bytes(self) -> int:
        parameters = 3 * self.hidden_width * self.intermediate_width
        return parameters // 2 + 2 * (parameters // self.group_size)

    @property
    def teacher_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "gate_up_packed": (
                self.experts,
                2 * self.intermediate_width,
                self.hidden_width // 2,
            ),
            "gate_up_scales": (
                self.experts,
                2 * self.intermediate_width,
                self.hidden_width // self.group_size,
            ),
            "down_packed": (
                self.experts,
                self.hidden_width,
                self.intermediate_width // 2,
            ),
            "down_scales": (
                self.experts,
                self.hidden_width,
                self.intermediate_width // self.group_size,
            ),
        }

    def validate(self) -> None:
        values = (
            self.layers,
            self.experts,
            self.hidden_width,
            self.intermediate_width,
            self.group_size,
            self.exact_k,
            self.core_size,
            self.total_residents,
        )
        if any(value < 1 for value in values):
            raise ValueError("extraction contract values must be positive")
        if self.core_size > self.experts or self.exact_k > self.experts:
            raise ValueError("core/top-k exceeds the expert namespace")
        if self.hidden_width % self.group_size or self.intermediate_width % self.group_size:
            raise ValueError("group size must divide both expert projection widths")
        if self.hidden_width % 2 or self.intermediate_width % 2:
            raise ValueError("packed INT4 widths must be even")


PRODUCTION_CONTRACT = ExtractionContract()


@dataclass(frozen=True)
class TeacherBundle:
    root: Path
    source_commit: str
    target_checkpoint_index_sha256: str
    paths: tuple[Path, ...]
    checkpoint_sha256: tuple[str, ...]
    manifest_sha256: str
    result_sha256: str
    sums_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bundle", type=Path, required=True)
    parser.add_argument("--allocation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--target-checkpoint-index-sha256", required=True)
    parser.add_argument(
        "--layer",
        type=int,
        choices=range(PRODUCTION_CONTRACT.layers),
        action="append",
        help="extract only this layer; repeat as needed (default: all 40)",
    )
    return parser.parse_args()


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hex(value: Any, length: int, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_ids_sha256(ids_by_layer: Sequence[Sequence[int]]) -> str:
    payload = json.dumps(
        [[int(expert) for expert in layer] for layer in ids_by_layer],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_byte_sha256(value: Tensor) -> str:
    contiguous = value.detach().to(device="cpu").contiguous().view(torch.uint8)
    digest = hashlib.sha256()
    digest.update(memoryview(contiguous.numpy()).cast("B"))
    return digest.hexdigest()


def _parse_checksum_inventory(root: Path) -> dict[str, str]:
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise ValueError(f"teacher bundle lacks SHA256SUMS: {root}")
    result: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        relative = Path(name)
        if (
            separator != "  "
            or name in result
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
        ):
            raise ValueError(f"malformed teacher checksum inventory: {sums}")
        _validate_hex(digest, 64, "teacher checksum")
        result[name] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != sums
    }
    if set(result) != actual:
        raise ValueError("teacher checksum inventory does not match its files")
    return result


def _verify_inventory_entry(
    root: Path, inventory: Mapping[str, str], relative: str
) -> str:
    expected = inventory.get(relative)
    if expected is None:
        raise ValueError(f"teacher checksum inventory lacks {relative}")
    path = root / relative
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"teacher checksum mismatch: {path}")
    return expected


def validate_allocation_plan(
    plan: Mapping[str, Any],
    *,
    source_commit: str,
    contract: ExtractionContract = PRODUCTION_CONTRACT,
) -> tuple[tuple[int, ...], ...]:
    """Validate the complete allocation and recompute its mandatory core."""

    contract.validate()
    _validate_hex(source_commit, 40, "source commit")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("source_commit") != source_commit:
        raise ValueError("resident allocation schema/source lineage mismatch")
    expected = {
        "total_residents": contract.total_residents,
        "frequency_core_inclusion_verified": True,
        "frequency_core_size_per_layer": contract.core_size,
        "frequency_core_cells": contract.layers * contract.core_size,
        "fallback_bytes": 0,
        "selection_uses_train_routes_only": True,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    for name, wanted in expected.items():
        if plan.get(name) != wanted:
            raise ValueError(f"resident allocation {name} mismatch")

    raw_ids = plan.get("resident_expert_ids_by_layer")
    raw_counts = plan.get("resident_counts_by_layer")
    raw_frequency = plan.get("expert_counts")
    if (
        not isinstance(raw_ids, list)
        or len(raw_ids) != contract.layers
        or not isinstance(raw_counts, list)
        or len(raw_counts) != contract.layers
        or not isinstance(raw_frequency, list)
        or len(raw_frequency) != contract.layers
    ):
        raise ValueError("resident allocation must contain every layer")

    ids_by_layer: list[tuple[int, ...]] = []
    frequency_rows: list[list[int]] = []
    for layer in range(contract.layers):
        ids = raw_ids[layer]
        frequencies = raw_frequency[layer]
        if not isinstance(ids, list) or not isinstance(frequencies, list):
            raise ValueError(f"resident allocation layer {layer} is malformed")
        if len(frequencies) != contract.experts or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in frequencies
        ):
            raise ValueError(f"resident allocation expert counts are invalid at layer {layer}")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= contract.experts
            for value in ids
        ):
            raise ValueError(f"resident allocation IDs are invalid at layer {layer}")
        values = tuple(int(value) for value in ids)
        if len(values) != len(set(values)) or len(values) != int(raw_counts[layer]):
            raise ValueError(f"resident allocation namespace/count mismatch at layer {layer}")
        ids_by_layer.append(values)
        frequency_rows.append([int(value) for value in frequencies])

    if sum(map(len, ids_by_layer)) != contract.total_residents:
        raise ValueError("resident allocation does not contain the required total cells")
    frequencies = torch.tensor(frequency_rows, dtype=torch.int64)
    core = torch.argsort(
        frequencies, dim=-1, descending=True, stable=True
    )[:, : contract.core_size]
    for layer, (resident, required) in enumerate(
        zip(ids_by_layer, core.tolist(), strict=True)
    ):
        missing = sorted(set(required) - set(resident))
        if missing:
            preview = ",".join(map(str, missing[:8]))
            raise ValueError(
                f"resident allocation omits mandatory frequency core at layer {layer}: {preview}"
            )
    core_hash = _canonical_ids_sha256(core.tolist())
    if plan.get("frequency_core_sha256") != core_hash:
        raise ValueError("resident allocation frequency-core hash mismatch")
    resident_hash = _canonical_ids_sha256(ids_by_layer)
    if plan.get("resident_ids_sha256") not in (None, resident_hash):
        raise ValueError("resident allocation namespace hash mismatch")

    logical_bytes = contract.resident_cell_bytes * contract.total_residents
    if plan.get("resident_cell_bytes") != contract.resident_cell_bytes:
        raise ValueError("resident allocation cell-byte contract mismatch")
    if plan.get("packed_int4_bytes") != logical_bytes:
        raise ValueError("resident allocation packed-byte contract mismatch")
    for name in (
        "coverage_by_layer",
        "weight_mass_coverage_by_layer",
    ):
        value = plan.get(name)
        if not isinstance(value, list) or len(value) != contract.layers:
            raise ValueError(f"resident allocation lacks {name}")
    _validate_hex(plan.get("partition_manifest_sha256"), 64, "partition manifest hash")
    return tuple(ids_by_layer)


def validate_teacher_bundle(
    root: Path,
    *,
    target_checkpoint_index_sha256: str,
    contract: ExtractionContract = PRODUCTION_CONTRACT,
) -> TeacherBundle:
    """Validate teacher metadata and bind all forty claimed shard digests."""

    contract.validate()
    target_sha = _validate_hex(
        target_checkpoint_index_sha256, 64, "target checkpoint index hash"
    )
    if not root.is_dir():
        raise FileNotFoundError(root)
    inventory = _parse_checksum_inventory(root)
    manifest_sha = _verify_inventory_entry(root, inventory, "run_manifest.json")
    result_sha = _verify_inventory_entry(root, inventory, "STAGE_RESULT.json")
    manifest = _read_json(root / "run_manifest.json")
    result = _read_json(root / "STAGE_RESULT.json")
    if manifest.get("schema") != TEACHER_MANIFEST_SCHEMA:
        raise ValueError("full INT4 teacher manifest schema mismatch")
    teacher_source = _validate_hex(manifest.get("source_commit"), 40, "teacher source commit")
    expected_manifest = {
        "target_checkpoint_index_sha256": target_sha,
        "group_size": contract.group_size,
        "active_slots": contract.exact_k,
        "quantization": "symmetric_signed_int4_per_output_group",
        **SAFETY_FIELDS,
    }
    for name, wanted in expected_manifest.items():
        if manifest.get(name) != wanted:
            raise ValueError(f"full INT4 teacher manifest {name} mismatch")
    if result.get("schema") != TEACHER_RESULT_SCHEMA:
        raise ValueError("full INT4 teacher result schema mismatch")
    expected_result = {
        "layers": contract.layers,
        "group_size": contract.group_size,
        "active_slots": contract.exact_k,
        "closed_loop_authorized": False,
        **SAFETY_FIELDS,
    }
    for name, wanted in expected_result.items():
        if result.get(name) != wanted:
            raise ValueError(f"full INT4 teacher result {name} mismatch")
    metrics = result.get("layer_metrics")
    if not isinstance(metrics, list) or len(metrics) != contract.layers:
        raise ValueError("full INT4 teacher result lacks every layer")

    paths: list[Path] = []
    checkpoint_hashes: list[str] = []
    for layer, metric in enumerate(metrics):
        if not isinstance(metric, dict) or metric.get("layer") != layer:
            raise ValueError("full INT4 teacher layer metrics are not ordered")
        claimed = _validate_hex(
            metric.get("checkpoint_sha256"), 64, f"teacher layer {layer} hash"
        )
        relative = f"layer_{layer:02d}/shadow_int4_top4_layer_{layer:02d}.pt"
        path = root / relative
        if not path.is_file() or inventory.get(relative) != claimed:
            raise ValueError(f"full INT4 teacher inventory disagrees at layer {layer}")
        paths.append(path)
        checkpoint_hashes.append(claimed)
    return TeacherBundle(
        root=root,
        source_commit=teacher_source,
        target_checkpoint_index_sha256=target_sha,
        paths=tuple(paths),
        checkpoint_sha256=tuple(checkpoint_hashes),
        manifest_sha256=manifest_sha,
        result_sha256=result_sha,
        sums_sha256=sha256_file(root / "SHA256SUMS"),
    )


def load_teacher_layer(
    teacher: TeacherBundle,
    layer: int,
    *,
    contract: ExtractionContract = PRODUCTION_CONTRACT,
) -> dict[str, Tensor]:
    path = teacher.paths[layer]
    if sha256_file(path) != teacher.checkpoint_sha256[layer]:
        raise ValueError(f"full INT4 teacher checksum mismatch at layer {layer}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"full INT4 teacher layer {layer} is not a mapping")
    expected = {
        "schema": LOCAL_SCHEMA,
        "mode": TEACHER_MODE,
        "layer": layer,
        "source_commit": teacher.source_commit,
        "target_checkpoint_index_sha256": teacher.target_checkpoint_index_sha256,
        "group_size": contract.group_size,
        "active_slots": contract.exact_k,
        "optimizer_constructed": False,
        "training_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    for name, wanted in expected.items():
        if value.get(name) != wanted:
            raise ValueError(f"full INT4 teacher layer {layer} {name} mismatch")
    state = value.get("model_state_dict")
    if not isinstance(state, dict) or set(state) != TEACHER_STATE_KEYS:
        raise ValueError(f"full INT4 teacher state changed at layer {layer}")
    shapes = contract.teacher_shapes
    dtypes = {
        "gate_up_packed": torch.uint8,
        "gate_up_scales": torch.bfloat16,
        "down_packed": torch.uint8,
        "down_scales": torch.bfloat16,
    }
    result: dict[str, Tensor] = {}
    for name in sorted(TEACHER_STATE_KEYS):
        tensor = state.get(name)
        if (
            not isinstance(tensor, Tensor)
            or tuple(tensor.shape) != shapes[name]
            or tensor.dtype != dtypes[name]
            or tensor.device.type != "cpu"
        ):
            raise ValueError(f"full INT4 teacher tensor {name} changed at layer {layer}")
        result[name] = tensor
    return result


def select_resident_state(
    teacher_state: Mapping[str, Tensor],
    resident_ids: Sequence[int],
    *,
    contract: ExtractionContract = PRODUCTION_CONTRACT,
) -> dict[str, Tensor]:
    ids = torch.tensor(tuple(int(value) for value in resident_ids), dtype=torch.int64)
    if (
        ids.ndim != 1
        or ids.numel() < contract.core_size
        or ids.unique().numel() != ids.numel()
        or bool(((ids < 0) | (ids >= contract.experts)).any())
    ):
        raise ValueError("resident layer namespace is invalid")
    mapping = torch.full((contract.experts,), -1, dtype=torch.int64)
    mapping[ids] = torch.arange(ids.numel(), dtype=torch.int64)
    result = {
        name: teacher_state[name].index_select(0, ids).contiguous().clone()
        for name in TEACHER_STATE_KEYS
    }
    result["resident_ids"] = ids
    result["expert_to_resident"] = mapping
    return result


def _write_checksums(directory: Path) -> None:
    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    with (directory / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verify_output_state(
    output_state: Mapping[str, Tensor], source_state: Mapping[str, Tensor]
) -> None:
    if set(output_state) != OUTPUT_STATE_KEYS:
        raise ValueError("resident output state contains unexpected tensors")
    for name in OUTPUT_STATE_KEYS:
        if not isinstance(output_state[name], Tensor) or not torch.equal(
            output_state[name], source_state[name]
        ):
            raise ValueError(f"resident output tensor {name} is not byte-exact")


def extract_layer(
    *,
    teacher: TeacherBundle,
    allocation_plan_path: Path,
    plan: Mapping[str, Any],
    resident_ids: Sequence[int],
    output_root: Path,
    source_commit: str,
    layer: int,
    contract: ExtractionContract = PRODUCTION_CONTRACT,
) -> dict[str, Any]:
    final = output_root / f"layer_{layer:02d}"
    temporary = output_root / f".layer_{layer:02d}.tmp"
    if final.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite resident layer {layer}")
    temporary.mkdir()

    teacher_state = load_teacher_layer(teacher, layer, contract=contract)
    state = select_resident_state(teacher_state, resident_ids, contract=contract)
    selected_tensor_hashes = {
        name: _tensor_byte_sha256(state[name]) for name in sorted(TEACHER_STATE_KEYS)
    }
    plan_sha = sha256_file(allocation_plan_path)
    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": LOCAL_SCHEMA,
        "created_utc": created,
        "source_commit": source_commit,
        "mode": OUTPUT_MODE,
        "layer": layer,
        "target_checkpoint_index_sha256": teacher.target_checkpoint_index_sha256,
        "resident_allocation_plan_sha256": plan_sha,
        "resident_count": len(resident_ids),
        "resident_expert_ids": list(map(int, resident_ids)),
        "logical_resident_bytes": contract.resident_cell_bytes * len(resident_ids),
        "extraction_method": "byte_exact_index_select_from_full_group64_int4_teacher",
        "teacher_source_commit": teacher.source_commit,
        "teacher_checkpoint_sha256": teacher.checkpoint_sha256[layer],
        "teacher_manifest_sha256": teacher.manifest_sha256,
        "teacher_result_sha256": teacher.result_sha256,
        "teacher_sums_sha256": teacher.sums_sha256,
        "selected_tensor_byte_sha256": selected_tensor_hashes,
        "trainable_names": [],
        "trainable_parameters": 0,
        "target_checkpoint_opened": False,
        "capture_opened": False,
        "target_expert_loads_permitted": False,
        **SAFETY_FIELDS,
    }
    _write_json_exclusive(temporary / "run_manifest.json", manifest)

    checkpoint_path = temporary / f"shadow_resident_int4_only_layer_{layer:02d}.pt"
    checkpoint = {
        "schema": LOCAL_SCHEMA,
        "source_commit": source_commit,
        "mode": OUTPUT_MODE,
        "layer": layer,
        "seed": 42,
        "model_state_dict": state,
        "resident_expert_ids": state["resident_ids"].clone(),
        "resident_count": len(resident_ids),
        "resident_train_slot_coverage": plan["coverage_by_layer"][layer],
        "resident_train_weight_mass_coverage": plan[
            "weight_mass_coverage_by_layer"
        ][layer],
        "target_checkpoint_index_sha256": teacher.target_checkpoint_index_sha256,
        "partition_manifest_sha256": plan["partition_manifest_sha256"],
        "resident_allocation_plan_sha256": plan_sha,
        "teacher_source_commit": teacher.source_commit,
        "teacher_checkpoint_sha256": teacher.checkpoint_sha256[layer],
        "selected_tensor_byte_sha256": selected_tensor_hashes,
        "diagnostic_only": True,
        "closed_loop_authorized": False,
        "target_checkpoint_opened": False,
        "capture_opened": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())

    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(reloaded, dict) or not isinstance(
        reloaded.get("model_state_dict"), dict
    ):
        raise ValueError("resident output checkpoint failed to reload")
    _verify_output_state(reloaded["model_state_dict"], state)
    for name, digest in selected_tensor_hashes.items():
        if _tensor_byte_sha256(reloaded["model_state_dict"][name]) != digest:
            raise ValueError(f"resident output tensor hash changed for {name}")

    checkpoint_sha = sha256_file(checkpoint_path)
    result = {
        "schema": LAYER_RESULT_SCHEMA,
        "mode": OUTPUT_MODE,
        "layer": layer,
        "resident_count": len(resident_ids),
        "checkpoint_sha256": checkpoint_sha,
        "extraction_method": manifest["extraction_method"],
        "teacher_checkpoint_sha256": teacher.checkpoint_sha256[layer],
        "byte_exact_tensor_verification_passed": True,
        "target_checkpoint_opened": False,
        "capture_opened": False,
        "closed_loop_evaluation_started": False,
        **SAFETY_FIELDS,
    }
    _write_json_exclusive(temporary / "STAGE_RESULT.json", result)
    _write_checksums(temporary)
    for line in (temporary / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or sha256_file(temporary / name) != digest:
            raise ValueError("resident output checksum verification failed")
    temporary.rename(final)
    return {
        "layer": layer,
        "path": str(final),
        "resident_count": len(resident_ids),
        "checkpoint_sha256": checkpoint_sha,
        "teacher_checkpoint_sha256": teacher.checkpoint_sha256[layer],
    }


def extract_layers(
    *,
    teacher_bundle: Path,
    allocation_plan: Path,
    output: Path,
    source_commit: str,
    target_checkpoint_index_sha256: str,
    layers: Sequence[int] | None = None,
    contract: ExtractionContract = PRODUCTION_CONTRACT,
) -> dict[str, Any]:
    contract.validate()
    _validate_hex(source_commit, 40, "source commit")
    _validate_hex(
        target_checkpoint_index_sha256, 64, "target checkpoint index hash"
    )
    plan = _read_json(allocation_plan)
    ids_by_layer = validate_allocation_plan(
        plan, source_commit=source_commit, contract=contract
    )
    teacher = validate_teacher_bundle(
        teacher_bundle,
        target_checkpoint_index_sha256=target_checkpoint_index_sha256,
        contract=contract,
    )
    selected = tuple(range(contract.layers)) if layers is None else tuple(layers)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(layer < 0 or layer >= contract.layers for layer in selected)
    ):
        raise ValueError("selected extraction layers are empty, duplicated, or out of range")
    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError(output)
        if (output / "BUNDLE_RESULT.json").exists() or (
            output / "BUNDLE_SHA256SUMS"
        ).exists():
            raise PermissionError("refusing to modify a sealed resident bundle")
    else:
        output.mkdir(parents=True)
    extracted = [
        extract_layer(
            teacher=teacher,
            allocation_plan_path=allocation_plan,
            plan=plan,
            resident_ids=ids_by_layer[layer],
            output_root=output,
            source_commit=source_commit,
            layer=layer,
            contract=contract,
        )
        for layer in selected
    ]
    return {
        "schema": "harp_shadowroute_resident_int4_teacher_extraction_v1",
        "source_commit": source_commit,
        "target_checkpoint_index_sha256": target_checkpoint_index_sha256,
        "teacher_source_commit": teacher.source_commit,
        "teacher_manifest_sha256": teacher.manifest_sha256,
        "teacher_result_sha256": teacher.result_sha256,
        "teacher_sums_sha256": teacher.sums_sha256,
        "allocation_plan_sha256": sha256_file(allocation_plan),
        "layers": extracted,
        "layer_count": len(extracted),
        "target_checkpoint_opened": False,
        "capture_opened": False,
        "optimizer_constructed": False,
        "training_started": False,
    }


def main() -> None:
    args = parse_args()
    result = extract_layers(
        teacher_bundle=args.teacher_bundle,
        allocation_plan=args.allocation_plan,
        output=args.output,
        source_commit=args.source_commit,
        target_checkpoint_index_sha256=args.target_checkpoint_index_sha256,
        layers=args.layer,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
