from __future__ import annotations

from pathlib import Path
import sys

import pytest


BRIDGE = Path(__file__).resolve().parents[1] / "runpod" / "transformers_mtp_bridge"
if str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))

from prepare_causal_pilot_partitions import (  # noqa: E402
    pre_eos_horizon_usable_positions,
    uniform_offsets,
)


def test_h4_partition_removes_legacy_tail_and_post_eos_rows() -> None:
    assert (
        pre_eos_horizon_usable_positions(
            split_usable_t_plus_2_positions=254,
            prompt_token_count=481,
            eos_position=None,
            horizon=4,
        )
        == 252
    )

    usable = pre_eos_horizon_usable_positions(
        split_usable_t_plus_2_positions=254,
        prompt_token_count=481,
        eos_position=646,
        horizon=4,
    )
    assert usable == 162
    offsets = uniform_offsets(usable, 16)
    assert len(offsets) == 16
    assert offsets[-1] == 161
    assert offsets[-1] + 4 == 165


def test_h4_partition_rejects_eos_before_generated_continuation() -> None:
    with pytest.raises(ValueError, match="precedes"):
        pre_eos_horizon_usable_positions(
            split_usable_t_plus_2_positions=254,
            prompt_token_count=481,
            eos_position=480,
            horizon=4,
        )


def test_b15_confirmation_uses_only_the_64_unselected_outer_train_requests() -> None:
    from prepare_b15_confirmation_partition import select_confirmation_requests

    rows = [{"request_id": f"request-{index:03d}"} for index in range(452)]
    from prepare_causal_pilot_partitions import request_order

    ordered = sorted(rows, key=lambda row: request_order(row["request_id"], 42))
    assert select_confirmation_requests(rows, seed=42) == ordered[388:452]
