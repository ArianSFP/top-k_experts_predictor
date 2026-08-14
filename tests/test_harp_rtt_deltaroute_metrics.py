from __future__ import annotations

import pytest

from harp_rtt.deltaroute_metrics import paired_h2_h4_request_bootstrap


def _rows(value: float) -> list[dict[str, object]]:
    return [
        {"request_id": request, "horizon": horizon, "slot_recall_at_8": value}
        for request in ("a", "b", "c")
        for horizon in (1, 2, 3, 4)
    ]


def test_h2_h4_bootstrap_excludes_h1_and_is_deterministic() -> None:
    candidate = _rows(0.7)
    baseline = _rows(0.5)
    for row in candidate:
        if row["horizon"] == 1:
            row["slot_recall_at_8"] = 0.0
    first = paired_h2_h4_request_bootstrap(candidate, baseline)
    second = paired_h2_h4_request_bootstrap(candidate, baseline)
    assert first == second
    assert first["paired_gain"] == pytest.approx(0.2)
    assert first["lower_bound_positive"] is True


def test_h2_h4_bootstrap_rejects_incomplete_requests() -> None:
    with pytest.raises(ValueError, match="complete H2-H4"):
        paired_h2_h4_request_bootstrap(_rows(0.7)[:-1], _rows(0.5))
