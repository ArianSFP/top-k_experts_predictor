from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from harp8.candidates import POOL_SCHEMA_V2, REQUEST_IDS_HASH_ENCODING, request_ids_sha256
from harp8.jspace_router_data import (
    FullRouterForecastData,
    JSPACE_ROUTER_CAUSAL_MODEL_KEYS,
    JSPACE_ROUTER_SECONDARY_CAUSAL_MODEL_KEYS,
    causal_router_forecaster_inputs,
)


def _memmap(path: Path, dtype: str, values: np.ndarray) -> dict[str, object]:
    output = np.memmap(path, mode="w+", dtype=dtype, shape=values.shape)
    output[...] = values
    output.flush()
    del output
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": "fixture"}


def _pool(root: Path, *, split: str, request_ids: list[int], within: list[int]) -> None:
    root.mkdir()
    rows, horizons, layers, candidates, experts, width = (
        len(request_ids), 8, 2, 10, 12, 6
    )
    shape = (rows, horizons, layers, candidates)
    scores = np.broadcast_to(
        np.linspace(3.0, -3.0, candidates, dtype=np.float32), shape
    ).copy()
    ids = np.broadcast_to(np.arange(candidates, dtype=np.uint16), shape).copy()
    valid = np.asarray(
        [[horizon <= 3 - position for horizon in range(1, 9)] for position in within],
        dtype=np.uint8,
    )
    arrays = [
        _memmap(root / "candidate_scores.f32", "<f4", scores),
        _memmap(root / "candidate_ids.u2", "<u2", ids),
        _memmap(root / "target_membership.u1", "u1", np.zeros(shape, np.uint8)),
        _memmap(root / "teacher_candidate_scores.f32", "<f4", np.zeros(shape, np.float32)),
        _memmap(root / "valid_future.u1", "u1", valid),
        _memmap(root / "current_scores.f32", "<f4", np.zeros(shape, np.float32)),
        _memmap(root / "current_rank.f32", "<f4", np.zeros(shape, np.float32)),
        _memmap(root / "source_gates.f16", "<f2", np.zeros((rows, horizons, layers, 3), np.float16)),
        _memmap(root / "copy_gates.f16", "<f2", np.zeros(shape, np.float16)),
        _memmap(
            root / "generator_context.f16",
            "<f2",
            np.arange(rows * horizons * layers * width, dtype=np.float16).reshape(
                rows, horizons, layers, width
            ),
        ),
    ]
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "request_ids": request_ids,
                "within": within,
                "domains": ["unit"] * rows,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": POOL_SCHEMA_V2,
                "split": split,
                "split_field_semantics": "deprecated_alias_of_level2_split",
                "level2_split": split,
                "source_data_split": split,
                "source_offline_split": split,
                "base_fold_id": "synthetic-fold-0",
                "base_fit_excluded": True,
                "base_fit_overlap_count": 0,
                "base_train_request_ids": [999],
                "base_train_request_ids_sha256": request_ids_sha256([999]),
                "exported_request_ids_sha256": request_ids_sha256(request_ids),
                "request_ids_hash_encoding": REQUEST_IDS_HASH_ENCODING,
                "allow_test": False,
                "rows": rows,
                "horizons": horizons,
                "layers": layers,
                "experts": experts,
                "native_k": 8,
                "candidate_count": candidates,
                "model_width": width,
                "store_context": True,
                "arrays": arrays,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    capture = tmp_path / "capture"
    mtp = tmp_path / "mtp"
    capture.mkdir()
    mtp.mkdir()
    (capture / "requests.jsonl").write_text(
        json.dumps({"request_id": 11}) + "\n" + json.dumps({"request_id": 22}) + "\n",
        encoding="utf-8",
    )
    rows, layers, experts = 8, 2, 12
    router = np.arange(rows * layers * experts, dtype=np.float32).reshape(
        rows, layers, experts
    )
    np.save(capture / "raw_router_logits.npy", router)
    np.save(
        capture / "top8_expert_ids.npy",
        np.argsort(-router, axis=-1)[..., :8].astype(np.uint16),
    )
    np.save(
        mtp / "mtp_hidden_depths.npy",
        np.arange(rows * 6 * 7, dtype=np.float16).reshape(rows, 6, 7),
    )
    np.save(
        mtp / "mtp_router_logits_depths.npy",
        np.arange(rows * 6 * experts, dtype=np.float32).reshape(rows, 6, experts),
    )
    features = tmp_path / "features.npy"
    np.save(features, np.arange(rows * layers * 5, dtype=np.float16).reshape(rows, layers, 5))
    return capture, mtp, features


def test_full_router_data_aligns_future_labels_and_preserves_harp_top8(tmp_path: Path) -> None:
    capture, mtp, features = _artifacts(tmp_path)
    pool = tmp_path / "pool"
    _pool(pool, split="train", request_ids=[22, 11], within=[2, 0])
    data = FullRouterForecastData(
        pool,
        expected_split="train",
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        rows_per_request=4,
        history=3,
    )
    batch = data.batch(np.asarray([0, 1]), "cpu")
    assert batch["base_router_scores"].shape == (2, 8, 2, 12)
    assert batch["teacher_router_scores"].shape == (2, 8, 2, 12)
    assert batch["target_top8"].shape == (2, 8, 2, 8)
    assert batch["j_states"].shape == (2, 3, 2, 5)
    assert batch["route_history"].shape == (2, 3, 2, 12)
    assert batch["generator_context"].shape == (2, 8, 2, 6)
    assert batch["valid_future"][0].tolist() == [True, False, False, False, False, False, False, False]
    # Pool row zero is request 22, within two -> capture row six; H1 is row seven.
    expected = np.load(capture / "raw_router_logits.npy")[7]
    assert np.array_equal(batch["teacher_router_scores"][0, 0].numpy(), expected)
    predicted = torch.topk(batch["base_router_scores"], 8, dim=-1).indices
    assert set(predicted[0, 0, 0].tolist()) == set(range(8))
    assert batch["route_mask"][0, :, 0].tolist() == [True, True, True]
    assert batch["route_mask"][1, :, 0].tolist() == [True, False, False]


def test_full_router_data_rejects_tampered_exported_request_hash(
    tmp_path: Path,
) -> None:
    capture, mtp, features = _artifacts(tmp_path)
    pool = tmp_path / "pool"
    _pool(pool, split="train", request_ids=[22, 11], within=[2, 0])
    metadata_path = pool / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["request_ids"] = [22, 22]
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exported request-ID hash"):
        FullRouterForecastData(
            pool,
            expected_split="train",
            capture_dir=capture,
            mtp_dir=mtp,
            target_features=features,
            rows_per_request=4,
            history=3,
        )


def test_causal_allowlist_strips_sanctioned_supervision_and_rejects_leakage() -> None:
    values = {
        name: torch.ones(1) for name in JSPACE_ROUTER_CAUSAL_MODEL_KEYS
    }
    values.update(
        {
            "teacher_router_scores": torch.full((1,), 999.0),
            "target_top8": torch.full((1,), 9),
            "valid_future": torch.ones(1, dtype=torch.bool),
        }
    )
    causal = causal_router_forecaster_inputs(values)
    assert set(causal) == JSPACE_ROUTER_CAUSAL_MODEL_KEYS
    assert not {"teacher_router_scores", "target_top8", "valid_future"}.intersection(causal)
    with pytest.raises(ValueError, match="label-only"):
        causal_router_forecaster_inputs(
            {**values, "prefix_matches_committed": torch.ones(1)}
        )


@pytest.mark.parametrize(
    "corrupt",
    (
        "target_features",
        "target_feature_rms",
        "route_history",
        "mtp_states",
        "mtp_router_logits",
        "generator_context",
        "base_router_scores",
    ),
)
def test_full_router_data_rejects_non_finite_causal_arrays(
    tmp_path: Path, corrupt: str
) -> None:
    capture, mtp, features = _artifacts(tmp_path)
    pool = tmp_path / "pool"
    _pool(pool, split="train", request_ids=[22, 11], within=[2, 0])
    rms_path: Path | None = None
    if corrupt == "target_features":
        values = np.load(features, mmap_mode="r+")
        values[0, 0, 0] = np.nan
        values.flush()
    elif corrupt == "target_feature_rms":
        rms_path = tmp_path / "log_rms.npy"
        np.save(rms_path, np.zeros((8, 2), dtype=np.float16))
        values = np.load(rms_path, mmap_mode="r+")
        values[0, 0] = np.inf
        values.flush()
    elif corrupt == "route_history":
        values = np.load(capture / "raw_router_logits.npy", mmap_mode="r+")
        values[0, 0, 0] = np.nan
        values.flush()
    elif corrupt == "mtp_states":
        values = np.load(mtp / "mtp_hidden_depths.npy", mmap_mode="r+")
        values[0, 0, 0] = np.inf
        values.flush()
    elif corrupt == "mtp_router_logits":
        values = np.load(mtp / "mtp_router_logits_depths.npy", mmap_mode="r+")
        values[0, 0, 0] = np.nan
        values.flush()
    elif corrupt == "generator_context":
        values = np.memmap(
            pool / "generator_context.f16",
            mode="r+",
            dtype="<f2",
            shape=(2, 8, 2, 6),
        )
        values[1, 0, 0, 0] = np.inf
        values.flush()
    else:
        values = np.memmap(
            pool / "candidate_scores.f32",
            mode="r+",
            dtype="<f4",
            shape=(2, 8, 2, 10),
        )
        values[1, 0, 0, 0] = np.nan
        values.flush()
    del values

    data = FullRouterForecastData(
        pool,
        expected_split="train",
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        target_feature_rms=rms_path,
        rows_per_request=4,
        history=3,
    )
    with pytest.raises(ValueError, match="non-finite"):
        # Pool row one is request 11, within zero -> capture row zero.
        data.batch(np.asarray([1]), "cpu")


@pytest.mark.parametrize("violation", ("duplicate", "out_of_range"))
def test_full_router_data_validates_authoritative_target_top8(
    tmp_path: Path, violation: str
) -> None:
    capture, mtp, features = _artifacts(tmp_path)
    pool = tmp_path / "pool"
    _pool(pool, split="train", request_ids=[22, 11], within=[2, 0])
    top8 = np.load(capture / "top8_expert_ids.npy", mmap_mode="r+")
    # Pool row one is request 11, within zero; its H1 target is capture row one.
    if violation == "duplicate":
        top8[1, 0, 1] = top8[1, 0, 0]
    else:
        top8[1, 0, 0] = 99
    top8.flush()
    del top8
    data = FullRouterForecastData(
        pool,
        expected_split="train",
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        rows_per_request=4,
        history=3,
    )
    message = "unique" if violation == "duplicate" else "outside expert namespace"
    with pytest.raises(ValueError, match=message):
        data.batch(np.asarray([1]), "cpu")


def test_pool_censoring_remains_masked_inside_request_boundary(tmp_path: Path) -> None:
    capture, mtp, features = _artifacts(tmp_path)
    pool = tmp_path / "pool"
    _pool(pool, split="train", request_ids=[22, 11], within=[2, 0])
    valid = np.memmap(
        pool / "valid_future.u1", mode="r+", dtype="u1", shape=(2, 8)
    )
    # H1 and H2 are geometrically in-bounds at within zero, but the pool may
    # censor either independently. The loader must never revive those labels.
    valid[1, :2] = 0
    valid.flush()
    del valid
    data = FullRouterForecastData(
        pool,
        expected_split="train",
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        rows_per_request=4,
        history=3,
    )
    batch = data.batch(np.asarray([1]), "cpu")
    assert batch["valid_future"][0, :3].tolist() == [False, False, True]


def test_pool_future_mask_rejects_non_binary_values(tmp_path: Path) -> None:
    capture, mtp, features = _artifacts(tmp_path)
    pool = tmp_path / "pool"
    _pool(pool, split="train", request_ids=[22, 11], within=[2, 0])
    valid = np.memmap(
        pool / "valid_future.u1", mode="r+", dtype="u1", shape=(2, 8)
    )
    valid[1, 0] = 2
    valid.flush()
    del valid
    data = FullRouterForecastData(
        pool,
        expected_split="train",
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        rows_per_request=4,
        history=3,
    )
    with pytest.raises(ValueError, match="zero or one"):
        data.batch(np.asarray([1]), "cpu")


def test_test_pool_is_rejected_before_tensor_open(tmp_path: Path) -> None:
    pool = tmp_path / "test_pool"
    pool.mkdir()
    (pool / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "harp8_candidate_pool_v1",
                "split": "test",
                "store_context": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected 'validation' pool"):
        FullRouterForecastData(
            pool,
            expected_split="validation",
            capture_dir=tmp_path / "does_not_exist",
            mtp_dir=tmp_path / "does_not_exist",
            target_features=tmp_path / "does_not_exist.npy",
        )


def test_dual_target_stream_uses_identical_capture_rows_lags_and_masks(
    tmp_path: Path,
) -> None:
    capture, mtp, features = _artifacts(tmp_path)
    pool = tmp_path / "pool"
    _pool(pool, split="train", request_ids=[22, 11], within=[2, 0])
    secondary = tmp_path / "secondary.npy"
    secondary_rms = tmp_path / "secondary_rms.npy"
    values = (
        1000
        + np.arange(8 * 2 * 4, dtype=np.float16).reshape(8, 2, 4)
    )
    rms = 2000 + np.arange(8 * 2, dtype=np.float16).reshape(8, 2)
    np.save(secondary, values)
    np.save(secondary_rms, rms)
    dual = FullRouterForecastData(
        pool,
        expected_split="train",
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        secondary_target_features=secondary,
        secondary_target_feature_rms=secondary_rms,
        rows_per_request=4,
        history=3,
    )
    batch = dual.batch(np.asarray([0, 1]), "cpu")
    assert batch["secondary_target_states"].shape == (2, 3, 2, 5)
    assert torch.equal(batch["secondary_target_mask"], batch["j_mask"])
    # Pool row zero maps to request 22/within 2 -> capture row 6, then lags 5/4.
    for lag, capture_row in enumerate((6, 5, 4)):
        assert np.array_equal(
            batch["secondary_target_states"][0, lag, :, :4].numpy(),
            values[capture_row].astype(np.float32),
        )
        assert np.array_equal(
            batch["secondary_target_states"][0, lag, :, 4].numpy(),
            rms[capture_row].astype(np.float32),
        )
    causal = causal_router_forecaster_inputs(
        batch, secondary_target_enabled=True
    )
    assert set(causal) == (
        JSPACE_ROUTER_CAUSAL_MODEL_KEYS
        | JSPACE_ROUTER_SECONDARY_CAUSAL_MODEL_KEYS
    )
    with pytest.raises(ValueError, match="explicit dual-stream contract"):
        causal_router_forecaster_inputs(batch)

    single = FullRouterForecastData(
        pool,
        expected_split="train",
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        rows_per_request=4,
        history=3,
    )
    single_batch = single.batch(np.asarray([0]), "cpu")
    assert not JSPACE_ROUTER_SECONDARY_CAUSAL_MODEL_KEYS.intersection(single_batch)


def test_dual_target_stream_rejects_shape_and_nonfinite_values(tmp_path: Path) -> None:
    capture, mtp, features = _artifacts(tmp_path)
    pool = tmp_path / "pool"
    _pool(pool, split="train", request_ids=[22, 11], within=[2, 0])
    wrong = tmp_path / "wrong.npy"
    np.save(wrong, np.zeros((7, 2, 4), dtype=np.float16))
    with pytest.raises(ValueError, match="align with the primary stream"):
        FullRouterForecastData(
            pool,
            expected_split="train",
            capture_dir=capture,
            mtp_dir=mtp,
            target_features=features,
            secondary_target_features=wrong,
            rows_per_request=4,
        )

    secondary = tmp_path / "secondary.npy"
    values = np.zeros((8, 2, 4), dtype=np.float16)
    values[0, 0, 0] = np.nan
    np.save(secondary, values)
    data = FullRouterForecastData(
        pool,
        expected_split="train",
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        secondary_target_features=secondary,
        rows_per_request=4,
    )
    with pytest.raises(ValueError, match="secondary target-state.*non-finite"):
        # Pool row one maps to capture row zero.
        data.batch(np.asarray([1]), "cpu")


def test_full_baseline_matches_legacy_bits_and_stable_candidate_ties(
    tmp_path: Path,
) -> None:
    capture, mtp, features = _artifacts(tmp_path)
    pool = tmp_path / "pool"
    _pool(pool, split="train", request_ids=[22, 11], within=[2, 0])
    score_map = np.memmap(
        pool / "candidate_scores.f32",
        mode="r+",
        dtype="<f4",
        shape=(2, 8, 2, 10),
    )
    id_map = np.memmap(
        pool / "candidate_ids.u2",
        mode="r+",
        dtype="<u2",
        shape=(2, 8, 2, 10),
    )
    score_map[0] = 3.25
    permutation = np.asarray([9, 4, 7, 0, 1, 2, 3, 5, 6, 8], dtype=np.uint16)
    id_map[0] = permutation
    score_map.flush()
    id_map.flush()
    del score_map, id_map

    data = FullRouterForecastData(
        pool,
        expected_split="train",
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        rows_per_request=4,
        history=3,
    )
    rows = np.asarray([0], dtype=np.int64)
    scores = np.asarray(data.pool.candidate_scores[rows], dtype=np.float32)
    ids = np.asarray(data.pool.candidate_ids[rows], dtype=np.int64)
    legacy_floor = scores.min(axis=-1, keepdims=True) - data.base_floor_margin
    legacy = np.broadcast_to(
        legacy_floor, scores.shape[:-1] + (data.experts,)
    ).copy()
    np.put_along_axis(legacy, ids, scores, axis=-1)

    expanded, expected_top8 = data._full_harp_baseline(rows)
    assert np.array_equal(expanded, legacy)
    ascending_tie_ids = list(range(8))
    assert expected_top8[0, 0, 0].tolist() == ascending_tie_ids
    full_top8 = np.argsort(-expanded, axis=-1, kind="stable")[..., :8]
    assert np.array_equal(expected_top8, full_top8)


def test_full_baseline_reads_candidate_scores_and_ids_once_per_batch(
    tmp_path: Path,
) -> None:
    capture, mtp, features = _artifacts(tmp_path)
    pool = tmp_path / "pool"
    _pool(pool, split="train", request_ids=[22, 11], within=[2, 0])
    data = FullRouterForecastData(
        pool,
        expected_split="train",
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        rows_per_request=4,
        history=3,
    )

    class CountingArray:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.reads = 0

        def __getitem__(self, key: object) -> np.ndarray:
            self.reads += 1
            return self.values[key]

    scores = CountingArray(data.pool.candidate_scores)
    ids = CountingArray(data.pool.candidate_ids)
    data.pool.candidate_scores = scores
    data.pool.candidate_ids = ids
    data.batch(np.asarray([0, 1]), "cpu")
    assert scores.reads == 1
    assert ids.reads == 1


@pytest.mark.parametrize("violation", ("duplicate", "out_of_range"))
def test_full_baseline_candidate_id_integrity_is_unchanged(
    tmp_path: Path, violation: str
) -> None:
    capture, mtp, features = _artifacts(tmp_path)
    pool = tmp_path / "pool"
    _pool(pool, split="train", request_ids=[22, 11], within=[2, 0])
    ids = np.memmap(
        pool / "candidate_ids.u2",
        mode="r+",
        dtype="<u2",
        shape=(2, 8, 2, 10),
    )
    ids[0, 0, 0, 0] = ids[0, 0, 0, 1] if violation == "duplicate" else 99
    ids.flush()
    del ids
    data = FullRouterForecastData(
        pool,
        expected_split="train",
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        rows_per_request=4,
        history=3,
    )
    message = "unique" if violation == "duplicate" else "outside expert namespace"
    with pytest.raises(ValueError, match=message):
        data._full_harp_baseline(np.asarray([0]))


def test_full_baseline_revalidates_positive_finite_floor_margin(tmp_path: Path) -> None:
    capture, mtp, features = _artifacts(tmp_path)
    pool = tmp_path / "pool"
    _pool(pool, split="train", request_ids=[22, 11], within=[2, 0])
    data = FullRouterForecastData(
        pool,
        expected_split="train",
        capture_dir=capture,
        mtp_dir=mtp,
        target_features=features,
        rows_per_request=4,
    )
    for invalid in (0.0, -1.0, np.inf):
        data.base_floor_margin = invalid
        with pytest.raises(ValueError, match="finite and positive"):
            data._full_harp_baseline(np.asarray([0]))
