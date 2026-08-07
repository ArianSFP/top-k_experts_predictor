from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harp8.build_m3_derived_shards import (
    CAUSAL_ARRAYS,
    COMPACT_REPRESENTATION,
    COMPACT_SCHEMA,
    LABEL_ONLY_ARRAYS,
    M3_DERIVED_SCHEMA,
    MTP_DEPTH_SCHEMA,
    _sha256_file,
    build_m3_derived_shards,
)
from harp8.candidates import POOL_SCHEMA


VOCAB_SIZE = 1000


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _file_record(path: Path, *, include_filename: bool = False) -> dict[str, object]:
    record: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if include_filename:
        record["filename"] = path.name
    return record


def _fixture(
    tmp_path: Path, *, split: str = "train"
) -> tuple[Path, Path, Path, Path, np.ndarray]:
    capture = tmp_path / "capture"
    mtp = tmp_path / "mtp"
    pool = tmp_path / "pool"
    split_path = tmp_path / "split.json"
    capture.mkdir()
    mtp.mkdir()
    pool.mkdir()

    request_specs = [
        (10, [100 + index for index in range(10)]),
        (20, [200 + index for index in range(10)]),
    ]
    requests = []
    rows = []
    for request_id, generated in request_specs:
        requests.append(
            {
                "request_id": request_id,
                "offline_split": "train",
                "prompt_token_ids": [1, 2],
                "generated_token_ids": generated,
                "routed_positions": 9,
            }
        )
        for within in range(9):
            rows.append(
                {
                    "request_id": request_id,
                    "within_request_index": within,
                    "router_state_position": 2 + within,
                    "state_token_id": generated[within],
                    "source_eligible": within < 8,
                }
            )
    requests_path = capture / "requests.jsonl"
    requests_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in requests),
        encoding="utf-8",
    )
    rows_path = capture / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in rows),
        encoding="utf-8",
    )
    capture_manifest = {
        "schema": COMPACT_SCHEMA,
        "representation": COMPACT_REPRESENTATION,
        "rows": len(rows),
        "requests": len(requests),
        "files": {
            "requests.jsonl": _file_record(requests_path),
            "rows.jsonl": _file_record(rows_path),
        },
    }
    _write_json(capture / "manifest.json", capture_manifest)

    drafts = np.full((len(rows), 6), 700, dtype="<i4")
    for request_number, (_request_id, generated) in enumerate(request_specs):
        for within in range(9):
            capture_row = request_number * 9 + within
            for depth in range(6):
                target = within + depth + 2
                drafts[capture_row, depth] = (
                    generated[target] if target < len(generated) else 700 + depth
                )
    # An isolated mismatch followed by later matches exercises cumulative AND.
    drafts[0, 2] = 900
    draft_path = mtp / "mtp_draft_token_ids.npy"
    np.save(draft_path, drafts)
    # These deliberately poisonous label-only files must never be opened.
    (mtp / "mtp_draft_target_token_ids.npy").write_bytes(b"DO-NOT-READ-TARGET-IDS")
    (mtp / "mtp_draft_target_logprobs.npy").write_bytes(b"DO-NOT-READ-LOGPROBS")
    mtp_manifest = {
        "schema": MTP_DEPTH_SCHEMA,
        "source_capture_manifest_sha256": _sha256_file(capture / "manifest.json"),
        "rows": len(rows),
        "requests": len(requests),
        "draft_depths": 6,
        "alignment": {"future_committed_tokens_used_as_features": False},
        "files": {
            "draft_token_ids": _file_record(draft_path, include_filename=True),
            "draft_target_token_ids": _file_record(
                mtp / "mtp_draft_target_token_ids.npy", include_filename=True
            ),
            "draft_target_logprobs": _file_record(
                mtp / "mtp_draft_target_logprobs.npy", include_filename=True
            ),
        },
    }
    _write_json(mtp / "manifest.json", mtp_manifest)

    assignments = {str(request_id): split for request_id, _generated in request_specs}
    _write_json(
        split_path,
        {"schema": "harp8_inner_split_v1", "assignments": assignments},
    )
    # Pool request order intentionally differs from capture order.
    request_ids = [20] * 8 + [10] * 8
    within = list(range(8)) + list(range(8))
    metadata = {
        "request_ids": request_ids,
        "within": within,
        "domains": ["fixture"] * len(request_ids),
    }
    _write_json(pool / "metadata.json", metadata)
    valid_future = np.ones((len(request_ids), 8), dtype="u1")
    valid_path = pool / "valid_future.u1"
    valid_future.tofile(valid_path)
    valid_record = _file_record(valid_path)
    valid_record["path"] = valid_path.name
    pool_manifest = {
        "schema": POOL_SCHEMA,
        "split": split,
        "allow_test": False,
        "rows": len(request_ids),
        "horizons": 8,
        "layers": 40,
        "experts": 256,
        "candidate_count": 64,
        "model_width": 384,
        "arrays": [valid_record],
        "split_manifest": {
            "path": str(split_path),
            "sha256": _sha256_file(split_path),
        },
    }
    _write_json(pool / "manifest.json", pool_manifest)
    return pool, capture, mtp, split_path, drafts


def _raw_array(output: Path, manifest: dict[str, object], name: str) -> np.ndarray:
    record = manifest["arrays"][name]
    return np.memmap(
        output / record["filename"],
        mode="r",
        dtype=np.dtype(record["dtype"]),
        shape=tuple(record["shape"]),
    )


def test_m3_builder_joins_pool_rows_and_emits_exact_causal_streams(
    tmp_path: Path,
) -> None:
    pool, capture, mtp, split_path, drafts = _fixture(tmp_path)
    output = tmp_path / "derived"
    manifest = build_m3_derived_shards(
        pool,
        capture,
        mtp,
        output,
        expected_split="train",
        vocab_size=VOCAB_SIZE,
        split_manifest_path=split_path,
    )

    assert manifest["schema"] == M3_DERIVED_SCHEMA
    assert manifest["leakage_contract"]["causal_arrays"] == sorted(CAUSAL_ARRAYS)
    assert manifest["leakage_contract"]["label_only_arrays"] == sorted(
        LABEL_ONLY_ARRAYS
    )
    assert manifest["leakage_contract"]["future_committed_token_ids_exported"] is False
    capture_rows = _raw_array(output, manifest, "capture_rows")
    assert capture_rows.tolist() == list(range(9, 17)) + list(range(0, 8))

    output_ids = _raw_array(output, manifest, "draft_output_token_ids")
    state_ids = _raw_array(output, manifest, "state_input_token_ids")
    assert np.array_equal(output_ids, drafts[np.asarray(capture_rows)])
    # First state input is authoritative x[t+1]; subsequent inputs are the
    # preceding causal outputs, never the later committed continuation.
    assert state_ids[0].tolist() == [201, *output_ids[0, :5].tolist()]
    assert state_ids[8].tolist() == [101, *output_ids[8, :5].tolist()]
    assert _raw_array(output, manifest, "state_input_valid").tolist() == [[1] * 6] * 16
    assert _raw_array(output, manifest, "draft_output_valid").tolist() == [[1] * 6] * 16

    expected_names = {
        "manifest.json",
        "state_input_token_ids.i4",
        "state_input_valid.u1",
        "draft_output_token_ids.i4",
        "draft_output_valid.u1",
        "path_token_match.u1",
        "path_prefix_reliability.u1",
        "reliability_valid.u1",
        "capture_rows.i8",
    }
    assert {path.name for path in output.iterdir()} == expected_names
    assert not any(
        "target" in path.name or "logprob" in path.name for path in output.iterdir()
    )


def test_m3_reliability_is_cumulative_and_censored_at_request_end(
    tmp_path: Path,
) -> None:
    pool, capture, mtp, split_path, _drafts = _fixture(tmp_path)
    output = tmp_path / "derived"
    manifest = build_m3_derived_shards(
        pool,
        capture,
        mtp,
        output,
        expected_split="train",
        vocab_size=VOCAB_SIZE,
        split_manifest_path=split_path,
    )
    match = _raw_array(output, manifest, "path_token_match")
    prefix = _raw_array(output, manifest, "path_prefix_reliability")
    valid = _raw_array(output, manifest, "reliability_valid")

    # Reversed pool order puts request 10, within 0 at derived row 8.
    assert match[8].tolist() == [1, 1, 0, 1, 1, 1]
    assert prefix[8].tolist() == [1, 1, 0, 0, 0, 0]
    assert valid[8].tolist() == [1, 1, 1, 1, 1, 1]
    # At within 7, only x[t+2] remains in the committed sequence.
    assert match[15].tolist() == [1, 0, 0, 0, 0, 0]
    assert valid[15].tolist() == [1, 0, 0, 0, 0, 0]
    assert prefix[15].tolist() == [1, 0, 0, 0, 0, 0]


def test_m3_ignores_future_target_id_and_logprob_capture_files(tmp_path: Path) -> None:
    pool, capture, mtp, split_path, _drafts = _fixture(tmp_path)
    # Hashes and contents of these label-only source files are intentionally
    # made invalid after manifest creation.  The builder must never touch them.
    (mtp / "mtp_draft_target_token_ids.npy").write_bytes(b"corrupted target IDs")
    (mtp / "mtp_draft_target_logprobs.npy").write_bytes(b"corrupted logprobs")
    manifest = build_m3_derived_shards(
        pool,
        capture,
        mtp,
        tmp_path / "derived",
        expected_split="train",
        vocab_size=VOCAB_SIZE,
        split_manifest_path=split_path,
    )
    assert manifest["leakage_contract"]["future_target_logprobs_read"] is False
    assert manifest["leakage_contract"]["future_target_logprobs_exported"] is False


def test_m3_rejects_test_pool_before_opening_metadata(tmp_path: Path) -> None:
    pool, capture, mtp, split_path, _drafts = _fixture(tmp_path)
    pool_manifest = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    pool_manifest["split"] = "test"
    _write_json(pool / "manifest.json", pool_manifest)
    (pool / "metadata.json").unlink()
    with pytest.raises(PermissionError, match="sealed/test"):
        build_m3_derived_shards(
            pool,
            capture,
            mtp,
            tmp_path / "derived",
            expected_split="train",
            vocab_size=VOCAB_SIZE,
            split_manifest_path=split_path,
        )


def test_m3_rejects_request_boundary_mismatch(tmp_path: Path) -> None:
    pool, capture, mtp, split_path, _drafts = _fixture(tmp_path)
    metadata = json.loads((pool / "metadata.json").read_text(encoding="utf-8"))
    metadata["within"][7] = 8
    _write_json(pool / "metadata.json", metadata)
    with pytest.raises(ValueError, match="causal source row|request boundary"):
        build_m3_derived_shards(
            pool,
            capture,
            mtp,
            tmp_path / "derived",
            expected_split="train",
            vocab_size=VOCAB_SIZE,
            split_manifest_path=split_path,
        )


@pytest.mark.parametrize("corruption", ["hash", "dtype", "vocab"])
def test_m3_rejects_corrupt_mtp_draft_source(tmp_path: Path, corruption: str) -> None:
    pool, capture, mtp, split_path, _drafts = _fixture(tmp_path)
    path = mtp / "mtp_draft_token_ids.npy"
    values = np.load(path)
    if corruption == "hash":
        values[0, 0] += 1
        np.save(path, values)
    elif corruption == "dtype":
        np.save(path, values.astype("<i8"))
    else:
        values[0, 0] = VOCAB_SIZE
        np.save(path, values)
    if corruption != "hash":
        mtp_manifest = json.loads((mtp / "manifest.json").read_text(encoding="utf-8"))
        mtp_manifest["files"]["draft_token_ids"] = _file_record(
            path, include_filename=True
        )
        _write_json(mtp / "manifest.json", mtp_manifest)
    pattern = "SHA-256" if corruption == "hash" else "shape/dtype|vocabulary"
    with pytest.raises(ValueError, match=pattern):
        build_m3_derived_shards(
            pool,
            capture,
            mtp,
            tmp_path / "derived",
            expected_split="train",
            vocab_size=VOCAB_SIZE,
            split_manifest_path=split_path,
        )


def test_m3_is_deterministic_and_never_overwrites(tmp_path: Path) -> None:
    pool, capture, mtp, split_path, _drafts = _fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = {
        "expected_split": "train",
        "vocab_size": VOCAB_SIZE,
        "split_manifest_path": split_path,
    }
    manifest_a = build_m3_derived_shards(pool, capture, mtp, first, **kwargs)
    manifest_b = build_m3_derived_shards(pool, capture, mtp, second, **kwargs)
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    for name in manifest_a["arrays"]:
        assert (
            manifest_a["arrays"][name]["sha256"] == manifest_b["arrays"][name]["sha256"]
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_m3_derived_shards(pool, capture, mtp, first, **kwargs)
