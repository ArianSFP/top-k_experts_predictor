from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "runpod"
    / "transformers_mtp_bridge"
    / "prepare_b2_translator_fitting_prompts.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_b2_fitting", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rows() -> list[dict[str, float | int]]:
    result = []
    offset = 0
    for entropy in (0.1, 0.9):
        for margin in (0.1, 0.9):
            for _ in range(4):
                result.append(
                    {"offset": offset, "entropy": entropy, "margin": margin}
                )
                offset += 1
    return result


def test_causal_scout_selects_two_per_joint_stratum() -> None:
    selected, counts = MODULE.choose_stratified_offsets(
        rows(),
        request_id="tfprod-req000001-12345678",
        uniform_offsets=set(),
        entropy_median=0.5,
        margin_median=0.5,
        seed=42,
    )
    assert len(selected) == len(set(selected)) == 8
    assert selected == sorted(selected)
    assert set(counts.values()) == {2}


def test_causal_scout_is_deterministic_and_avoids_uniform_offsets() -> None:
    uniform = {0, 4, 8, 12}
    first = MODULE.choose_stratified_offsets(
        rows(),
        request_id="tfprod-req000001-12345678",
        uniform_offsets=uniform,
        entropy_median=0.5,
        margin_median=0.5,
        seed=42,
    )
    second = MODULE.choose_stratified_offsets(
        list(reversed(rows())),
        request_id="tfprod-req000001-12345678",
        uniform_offsets=uniform,
        entropy_median=0.5,
        margin_median=0.5,
        seed=42,
    )
    assert first == second
    assert not (set(first[0]) & uniform)


def test_causal_scout_backfills_sparse_strata_without_labels() -> None:
    sparse = [
        {"offset": offset, "entropy": 0.9, "margin": 0.1}
        for offset in range(12)
    ]
    selected, counts = MODULE.choose_stratified_offsets(
        sparse,
        request_id="tfprod-req000001-12345678",
        uniform_offsets={0, 1},
        entropy_median=0.5,
        margin_median=0.5,
        seed=44,
    )
    assert len(selected) == 8
    assert counts["high_entropy_low_margin"] == 8
    assert sum(counts.values()) == 8
