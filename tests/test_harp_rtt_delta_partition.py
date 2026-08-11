from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "runpod" / "transformers_mtp_bridge" / "prepare_harp_delta_20k_partition.py"
SPEC = importlib.util.spec_from_file_location("prepare_harp_delta_20k_partition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def scouts(usable: int) -> list[dict[str, float | int]]:
    return [
        {
            "source_position_offset": index,
            "mtp_entropy": float(index % 7),
            "mtp_margin": float(index % 11),
        }
        for index in range(usable)
    ]


def test_mixed_position_selection_is_deterministic_unique_and_long_aware() -> None:
    first, first_reason = MODULE.mixed_position_offsets(
        request_id="request-1", usable=512, scouts=scouts(512),
        entropy_threshold=3.0, margin_threshold=5.0, seed=42,
    )
    second, second_reason = MODULE.mixed_position_offsets(
        request_id="request-1", usable=512, scouts=scouts(512),
        entropy_threshold=3.0, margin_threshold=5.0, seed=42,
    )
    assert first == second
    assert first_reason == second_reason
    assert len(first) == len(set(first)) == 16
    assert all(0 <= offset < 512 for offset in first)
    assert any(offset > 32 for offset in first)
    assert any(reason.startswith("causal_entropy") for reason in first_reason.values())


def test_mixed_position_selection_backfills_sparse_scouts() -> None:
    offsets, reasons = MODULE.mixed_position_offsets(
        request_id="sparse", usable=20, scouts=[],
        entropy_threshold=0.0, margin_threshold=0.0, seed=42,
    )
    assert len(offsets) == len(set(offsets)) == 16
    assert any(reason == "deterministic_causal_backfill" for reason in reasons.values())


def test_mixed_position_selection_rejects_short_request() -> None:
    with pytest.raises(ValueError, match="fewer than 16"):
        MODULE.mixed_position_offsets(
            request_id="short", usable=15, scouts=scouts(15),
            entropy_threshold=3.0, margin_threshold=5.0, seed=42,
        )


def test_log_offsets_extend_beyond_legacy_first_33() -> None:
    values = MODULE.log_offsets(1000, 4)
    assert values[0] >= 33
    assert values[-1] == 999
    assert len(values) == len(set(values)) == 4
