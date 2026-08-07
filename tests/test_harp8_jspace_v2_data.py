from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from harp8.jspace_v2_data import (
    AlignedJContextCandidateData,
    assert_no_label_only_inputs,
    causal_v2_model_inputs,
)


def _memmap(path: Path, dtype: str, values: np.ndarray) -> None:
    output = np.memmap(path, mode="w+", dtype=dtype, shape=values.shape)
    output[...] = values
    output.flush()
    del output


def _pool(root: Path, *, store_context: bool) -> None:
    root.mkdir()
    rows, horizons, layers, candidates, experts, width = 2, 8, 2, 10, 12, 6
    shape = (rows, horizons, layers, candidates)
    arrays: list[dict[str, object]] = []

    def write(name: str, dtype: str, values: np.ndarray) -> None:
        _memmap(root / name, dtype, values)
        arrays.append({"path": name, "bytes": (root / name).stat().st_size, "sha256": "fixture"})

    write("candidate_scores.f32", "<f4", np.zeros(shape, np.float32))
    write(
        "candidate_ids.u2",
        "<u2",
        np.broadcast_to(np.arange(candidates, dtype=np.uint16), shape).copy(),
    )
    membership = np.zeros(shape, np.uint8)
    membership[..., :8] = 1
    write("target_membership.u1", "u1", membership)
    write("teacher_candidate_scores.f32", "<f4", np.ones(shape, np.float32))
    write("valid_future.u1", "u1", np.ones((rows, horizons), np.uint8))
    write("current_scores.f32", "<f4", np.zeros(shape, np.float32))
    write("current_rank.f32", "<f4", np.zeros(shape, np.float32))
    write("source_gates.f16", "<f2", np.zeros((rows, horizons, layers, 3), np.float16))
    write("copy_gates.f16", "<f2", np.zeros(shape, np.float16))
    if store_context:
        context = np.arange(
            rows * horizons * layers * width, dtype=np.float16
        ).reshape(rows, horizons, layers, width)
        write("generator_context.f16", "<f2", context)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "request_ids": [22, 11],
                "within": [2, 0],
                "domains": ["unit", "unit"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "harp8_candidate_pool_v1",
                "split": "train",
                "rows": rows,
                "horizons": horizons,
                "layers": layers,
                "experts": experts,
                "native_k": 8,
                "candidate_count": candidates,
                "model_width": width,
                "store_context": store_context,
                "arrays": arrays,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _data(tmp_path: Path, *, store_context: bool = True) -> AlignedJContextCandidateData:
    capture = tmp_path / "capture"
    mtp = tmp_path / "mtp"
    capture.mkdir()
    mtp.mkdir()
    (capture / "requests.jsonl").write_text(
        json.dumps({"request_id": 11}) + "\n" + json.dumps({"request_id": 22}) + "\n",
        encoding="utf-8",
    )
    rows_per_request, rows, layers, experts = 4, 8, 2, 12
    router = np.arange(rows * layers * experts, dtype=np.float32).reshape(
        rows, layers, experts
    )
    np.save(capture / "raw_router_logits.npy", router)
    np.save(
        capture / "top8_expert_ids.npy",
        np.argsort(-router, axis=-1)[..., :8].astype(np.uint16),
    )
    np.save(mtp / "mtp_hidden_depths.npy", np.ones((rows, 6, 7), np.float16))
    np.save(
        mtp / "mtp_router_logits_depths.npy",
        np.ones((rows, 6, experts), np.float32),
    )
    features = tmp_path / "features.npy"
    np.save(features, np.ones((rows, layers, 5), np.float16))
    pool = tmp_path / "pool"
    _pool(pool, store_context=store_context)
    return AlignedJContextCandidateData(
        pool,
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        rows_per_request=rows_per_request,
        history=3,
    )


def test_v2_data_loads_only_active_generator_context_prefix(tmp_path: Path) -> None:
    data = _data(tmp_path)
    batch = data.batch(
        np.asarray([0, 1], dtype=np.int64),
        "cpu",
        active_horizons=4,
        include_context=True,
        compact=True,
    )
    assert batch["generator_context"].shape == (2, 4, 2, 6)
    assert batch["candidate_scores"].shape == (2, 4, 2, 10)
    assert batch["j_states"].shape == (2, 3, 2, 5)
    assert batch["mtp_states"].shape == (2, 6, 7)
    assert "context" not in batch
    assert "capture_row" not in batch
    # Pool row zero belongs to request 22, within two, independently of the
    # capture request ordering.  Generator context itself remains pool-aligned.
    expected = np.asarray(data.pool.context[0, :4], dtype=np.float32)
    assert np.array_equal(batch["generator_context"][0].numpy(), expected)


def test_v2_data_fails_closed_without_context_or_when_disabled(tmp_path: Path) -> None:
    data = _data(tmp_path)
    with pytest.raises(ValueError, match="include_context=True"):
        data.batch(np.asarray([0]), "cpu", include_context=False)

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    with pytest.raises(ValueError, match="context-enabled"):
        _data(missing_root, store_context=False)


def test_label_only_fields_are_rejected() -> None:
    assert_no_label_only_inputs({"candidate_scores": torch.ones(1)})
    assert callable(causal_v2_model_inputs)
    with pytest.raises(ValueError, match="label-only"):
        assert_no_label_only_inputs(
            {
                "candidate_scores": torch.ones(1),
                "mtp_draft_target_token_ids": torch.ones(1),
            }
        )
