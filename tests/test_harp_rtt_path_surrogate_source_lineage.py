from __future__ import annotations

import json
from pathlib import Path

import pytest

from runpod.build_harp_path_surrogate_cache import source_lineage_partitions


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def test_source_lineage_resolves_adaptive_ids_to_original_requests(
    tmp_path: Path,
) -> None:
    diagnostic = tmp_path / "diagnostic.jsonl"
    reuse = tmp_path / "reuse.json"
    events = tmp_path / "events.jsonl"
    _write_jsonl(diagnostic, [
        {"request_id": f"source-dev-{index}"} for index in range(128)
    ])
    reuse.write_text(json.dumps({"inner_split": {
        "tuning_requests": [f"adaptive-tune-{index}" for index in range(32)]
    }}))
    _write_jsonl(events, [
        {"event": "sequence_start", "request_id": f"adaptive-tune-{index}",
         "source_request_id": f"source-tune-{index}"}
        for index in range(32)
    ])
    development, tuning = source_lineage_partitions(diagnostic, reuse, events)
    assert development == {f"source-dev-{index}" for index in range(128)}
    assert tuning == {f"source-tune-{index}" for index in range(32)}
    assert development.isdisjoint(tuning)


def test_source_lineage_rejects_missing_adaptive_mapping(tmp_path: Path) -> None:
    diagnostic = tmp_path / "diagnostic.jsonl"
    reuse = tmp_path / "reuse.json"
    events = tmp_path / "events.jsonl"
    _write_jsonl(diagnostic, [
        {"request_id": f"source-dev-{index}"} for index in range(128)
    ])
    reuse.write_text(json.dumps({"inner_split": {
        "tuning_requests": [f"adaptive-tune-{index}" for index in range(32)]
    }}))
    _write_jsonl(events, [])
    with pytest.raises(ValueError, match="lack source-lineage"):
        source_lineage_partitions(diagnostic, reuse, events)
