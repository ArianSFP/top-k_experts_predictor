from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

import runpod.audit_harp_path_surrogate_cache as audit


def _cache(root: Path, *, duplicate: bool = False) -> Path:
    root.mkdir()
    rows, rank = 2, 3
    arrays = {
        "state_coordinates": np.zeros((rows, 4, 40, rank), np.float16),
        "current_queries": np.zeros((rows, 40, 255), np.float16),
        "history_selected_ids": np.zeros((rows, 8, 40, 8), np.int16),
        "history_selected_weights": np.ones((rows, 8, 40, 8), np.float16),
        "path_token_ids": np.ones((rows, 4), np.int32),
        "target_centered_logits": np.zeros((rows, 4, 40, 256), np.float16),
        "target_selected_ids": np.broadcast_to(
            np.arange(8, dtype=np.int16), (rows, 4, 40, 8)
        ).copy(),
        "request_index": np.array([0, 1], np.int16),
        "source_position": np.array([7, 7 if duplicate else 8], np.int32),
        "split": np.array([0, 1], np.uint8),
    }
    if duplicate:
        arrays["request_index"][1] = 0
        arrays["split"][1] = 0
    for name, value in arrays.items():
        np.save(root / f"{name}.npy", value)
    (root / "manifest.json").write_text(json.dumps({
        "schema": "harp_path_surrogate_cache_v1", "complete": True,
        "formal_validation_opened": False, "calibration_opened": False,
        "sealed_test_opened": False, "outer_split": "train",
        "counterfactual_labels_read": False, "rows": rows,
        "state_rank": rank, "train_rows": 2 if duplicate else 1,
        "tune_rows": 0 if duplicate else 1,
        "train_requests": ["train"], "tune_requests": ["tune"],
    }))
    return root


def _run(monkeypatch: pytest.MonkeyPatch, cache: Path, builder: Path) -> None:
    monkeypatch.setattr(sys, "argv", [
        "audit", "--cache", str(cache), "--source-commit", "abc123",
        "--builder", str(builder), "--chunk-rows", "1",
    ])
    audit.main()


def test_cache_audit_accepts_disjoint_complete_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = tmp_path / "builder.py"; builder.write_text("# frozen\n")
    cache = _cache(tmp_path / "cache")
    _run(monkeypatch, cache, builder)
    result = json.loads((cache / "CACHE_AUDIT.json").read_text())
    assert result["complete"] is True
    assert result["request_group_disjoint"] is True
    assert result["native_top8_agreement_from_fp16_logits"] == 1.0
    assert (cache / "SHA256SUMS").is_file()


def test_cache_audit_rejects_duplicate_request_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = tmp_path / "builder.py"; builder.write_text("# frozen\n")
    cache = _cache(tmp_path / "cache", duplicate=True)
    with pytest.raises(ValueError, match="duplicate"):
        _run(monkeypatch, cache, builder)
