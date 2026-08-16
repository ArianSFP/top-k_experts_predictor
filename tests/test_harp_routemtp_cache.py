from __future__ import annotations

import hashlib

import pytest
import torch

from harp_rtt.routemtp_cache import (
    RouteMTPCacheGeometry,
    RouteMTPSourceOffset,
    load_causal_cache_slice,
    write_cache_record,
    write_hydration_manifest,
)


def _geometry() -> RouteMTPCacheGeometry:
    return RouteMTPCacheGeometry(
        mtp_layers=1,
        kv_heads=2,
        head_dimension=4,
        sequence_axis=2,
        physical_layout="batch,kv_head,position,head_dimension",
        dtype="bfloat16",
        rope_implementation="qwen35_mrope_v1",
        transformers_version="5.14.1",
        checkpoint_sha256="a" * 64,
        engine_commit="test-engine",
    )


def test_cache_is_never_loaded_without_a_causal_prefix_slice(tmp_path) -> None:
    geometry = _geometry()
    tensors = {
        "key.0": torch.randn(1, 2, 5, 4, dtype=torch.bfloat16),
        "value.0": torch.randn(1, 2, 5, 4, dtype=torch.bfloat16),
        "shifted_token_ids": torch.arange(5, dtype=torch.int64),
    }
    path = tmp_path / "request.safetensors"
    record = write_cache_record(path, tensors, geometry, request_id="request-1")
    offset = RouteMTPSourceOffset("request-1", 7, 3, "b" * 64)
    loaded = load_causal_cache_slice(
        path, geometry, offset, expected_sha256=record["sha256"]
    )
    assert loaded["key.0"].shape == (1, 2, 3, 4)
    assert loaded["shifted_token_ids"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="exceeds"):
        load_causal_cache_slice(
            path,
            geometry,
            RouteMTPSourceOffset("request-1", 7, 6, "b" * 64),
            expected_sha256=record["sha256"],
        )


def test_hydration_manifest_is_hash_bound_and_sealed(tmp_path) -> None:
    geometry = _geometry()
    tensors = {
        "key.0": torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16),
        "value.0": torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16),
        "shifted_token_ids": torch.arange(2, dtype=torch.int64),
    }
    cache = tmp_path / "request.safetensors"
    record = write_cache_record(cache, tensors, geometry, request_id="request-1")
    manifest = tmp_path / "manifest.json"
    write_hydration_manifest(
        manifest,
        geometry=geometry,
        records=[record],
        offsets=[RouteMTPSourceOffset("request-1", 2, 2, "c" * 64)],
        bindings={
            "source_commit": "d" * 40,
            "target_checkpoint_sha256": "e" * 64,
            "mtp_checkpoint_sha256": "f" * 64,
            "split_manifest_sha256": "1" * 64,
            "base_capture_audit_sha256": "2" * 64,
            "counterfactual_companion_sha256": "3" * 64,
        },
    )
    payload = manifest.read_bytes()
    assert b'"sealed_test_opened": false' in payload
    assert b'"causal_slice_required": true' in payload
    assert hashlib.sha256(payload).hexdigest()
