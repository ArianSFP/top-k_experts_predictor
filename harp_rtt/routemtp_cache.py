"""Checkpoint-derived RouteMTP prefix-cache hydration contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
from safetensors.torch import load_file, save_file


ROUTEMTP_CACHE_SCHEMA = "harp_routemtp_prefix_cache_v1"
ROUTEMTP_CACHE_RECORD_SCHEMA = "harp_routemtp_prefix_cache_record_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RouteMTPCacheGeometry:
    mtp_layers: int
    kv_heads: int
    head_dimension: int
    sequence_axis: int
    physical_layout: str
    dtype: str
    rope_implementation: str
    transformers_version: str
    checkpoint_sha256: str
    engine_commit: str

    def validate(self) -> None:
        if min(self.mtp_layers, self.kv_heads, self.head_dimension) < 1:
            raise ValueError("cache dimensions must be positive")
        if self.sequence_axis not in (0, 1, 2, 3):
            raise ValueError("cache sequence axis is invalid")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("unsupported RouteMTP cache dtype")
        for name in (
            "physical_layout", "rope_implementation", "transformers_version",
            "checkpoint_sha256", "engine_commit",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"cache geometry lacks {name}")
        if len(self.checkpoint_sha256) != 64:
            raise ValueError("checkpoint SHA256 must contain 64 hex digits")
        try:
            int(self.checkpoint_sha256, 16)
        except ValueError as error:
            raise ValueError("checkpoint SHA256 is not hexadecimal") from error


@dataclass(frozen=True)
class RouteMTPSourceOffset:
    request_id: str
    source_position: int
    prefix_length: int
    prefix_hash: str

    def validate(self, maximum_cache_length: int) -> None:
        if not self.request_id or self.source_position < 0:
            raise ValueError("invalid RouteMTP source offset identity")
        if not 0 <= self.prefix_length <= maximum_cache_length:
            raise ValueError("RouteMTP prefix slice exceeds hydrated cache")
        if len(self.prefix_hash) != 64:
            raise ValueError("RouteMTP prefix hash must be SHA256")


def validate_cache_tensors(
    tensors: Mapping[str, Tensor],
    geometry: RouteMTPCacheGeometry,
) -> int:
    geometry.validate()
    expected = {
        *(f"key.{layer}" for layer in range(geometry.mtp_layers)),
        *(f"value.{layer}" for layer in range(geometry.mtp_layers)),
        "shifted_token_ids",
    }
    if set(tensors) != expected:
        raise ValueError(
            f"RouteMTP cache tensor keys differ: missing={sorted(expected-set(tensors))}, "
            f"unexpected={sorted(set(tensors)-expected)}"
        )
    lengths: set[int] = set()
    for layer in range(geometry.mtp_layers):
        key = tensors[f"key.{layer}"]; value = tensors[f"value.{layer}"]
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("RouteMTP K/V tensors must share rank-four geometry")
        if key.shape[0] != 1 or key.shape[1] != geometry.kv_heads or key.shape[3] != geometry.head_dimension:
            raise ValueError("RouteMTP K/V tensor geometry differs from checkpoint")
        if key.dtype != value.dtype or str(key.dtype).removeprefix("torch.") != geometry.dtype:
            raise TypeError("RouteMTP K/V dtype differs from cache manifest")
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError("RouteMTP K/V cache contains NaN or Inf")
        lengths.add(int(key.shape[geometry.sequence_axis]))
    token_ids = tensors["shifted_token_ids"]
    if token_ids.ndim != 1 or token_ids.dtype != torch.int64:
        raise ValueError("shifted token IDs must be int64 [T]")
    lengths.add(int(token_ids.numel()))
    if len(lengths) != 1:
        raise ValueError("RouteMTP cache streams have different lengths")
    return lengths.pop()


def write_cache_record(
    path: Path,
    tensors: Mapping[str, Tensor],
    geometry: RouteMTPCacheGeometry,
    *,
    request_id: str,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite RouteMTP cache {path}")
    length = validate_cache_tensors(tensors, geometry)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale RouteMTP cache temporary file {temporary}")
    save_file({name: value.detach().cpu().contiguous() for name, value in tensors.items()}, temporary)
    os.replace(temporary, path)
    return {
        "schema": ROUTEMTP_CACHE_RECORD_SCHEMA,
        "request_id": request_id,
        "relative_path": path.name,
        "cache_length": length,
        "sha256": sha256_file(path),
        "causal_slice_required": True,
    }


def load_causal_cache_slice(
    path: Path,
    geometry: RouteMTPCacheGeometry,
    offset: RouteMTPSourceOffset,
    *,
    expected_sha256: str,
) -> dict[str, Tensor]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("RouteMTP cache checksum mismatch")
    tensors = dict(load_file(path, device="cpu"))
    maximum = validate_cache_tensors(tensors, geometry)
    offset.validate(maximum)
    result: dict[str, Tensor] = {}
    for name, value in tensors.items():
        if name == "shifted_token_ids":
            result[name] = value[: offset.prefix_length].clone()
            continue
        slices = [slice(None)] * value.ndim
        slices[geometry.sequence_axis] = slice(0, offset.prefix_length)
        result[name] = value[tuple(slices)].clone()
    if validate_cache_tensors(result, geometry) != offset.prefix_length:
        raise RuntimeError("RouteMTP causal cache slicing failed")
    return result


def restore_transformers_dynamic_cache(
    tensors: Mapping[str, Tensor],
    geometry: RouteMTPCacheGeometry,
    *,
    config: Any,
    device: torch.device | str,
) -> Any:
    """Reconstruct a fresh Transformers DynamicCache from a causal slice."""

    length = validate_cache_tensors(tensors, geometry)
    try:
        from transformers import DynamicCache
    except ImportError as error:  # pragma: no cover - optional capture extra
        raise RuntimeError("Transformers is required for RouteMTP cache replay") from error
    try:
        cache = DynamicCache(config=config)
    except TypeError:  # older compatible API
        cache = DynamicCache()
    for layer in range(geometry.mtp_layers):
        key = tensors[f"key.{layer}"].to(device=device)
        value = tensors[f"value.{layer}"].to(device=device)
        if hasattr(cache, "update"):
            cache.update(key, value, layer)
        else:  # pragma: no cover - guarded compatibility seam
            raise TypeError("Transformers DynamicCache lacks update()")
    if hasattr(cache, "get_seq_length") and int(cache.get_seq_length()) != length:
        raise RuntimeError("restored RouteMTP cache length differs from causal slice")
    return cache


def write_hydration_manifest(
    path: Path,
    *,
    geometry: RouteMTPCacheGeometry,
    records: list[Mapping[str, Any]],
    offsets: list[RouteMTPSourceOffset],
    bindings: Mapping[str, str],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite RouteMTP manifest {path}")
    geometry.validate()
    request_ids = [str(record["request_id"]) for record in records]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("duplicate request in RouteMTP hydration manifest")
    maximum_by_request = {
        str(record["request_id"]): int(record["cache_length"]) for record in records
    }
    keys: set[tuple[str, int]] = set()
    for offset in offsets:
        if offset.request_id not in maximum_by_request:
            raise ValueError("RouteMTP source offset lacks a hydrated request")
        offset.validate(maximum_by_request[offset.request_id])
        key = (offset.request_id, offset.source_position)
        if key in keys:
            raise ValueError("duplicate RouteMTP request/source offset")
        keys.add(key)
    required_bindings = {
        "source_commit", "target_checkpoint_sha256", "mtp_checkpoint_sha256",
        "split_manifest_sha256", "base_capture_audit_sha256",
        "counterfactual_companion_sha256",
    }
    if set(bindings) != required_bindings or any(not str(value) for value in bindings.values()):
        raise ValueError("RouteMTP hydration bindings are incomplete")
    payload = {
        "schema": ROUTEMTP_CACHE_SCHEMA,
        "geometry": asdict(geometry),
        "records": [dict(value) for value in records],
        "source_offsets": [asdict(value) for value in offsets],
        "bindings": dict(bindings),
        "outer_split": "train",
        "label_only": False,
        "runtime_available": True,
        "causal_slice_required": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "training_started": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


__all__ = [
    "ROUTEMTP_CACHE_RECORD_SCHEMA",
    "ROUTEMTP_CACHE_SCHEMA",
    "RouteMTPCacheGeometry",
    "RouteMTPSourceOffset",
    "load_causal_cache_slice",
    "restore_transformers_dynamic_cache",
    "sha256_file",
    "validate_cache_tensors",
    "write_cache_record",
    "write_hydration_manifest",
]
