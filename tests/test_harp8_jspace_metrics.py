from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from harp8.jspace_metrics import evaluate_jspace_ranker, paired_request_bootstrap


class _ScoresFromBatch(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"scores": batch["predicted_scores"]}


class _MaskedMetricData:
    rows_per_request = 1

    def __init__(self) -> None:
        self.pool = SimpleNamespace(
            manifest={"native_k": 8},
            candidate_count=10,
            horizons=1,
            layers=1,
            request_ids=np.asarray([7], dtype=np.int64),
            domains=np.asarray(["unit"], dtype=object),
            within=np.asarray([0], dtype=np.int32),
        )

    def sequential_batches(self, batch_size: int):
        assert batch_size > 0
        yield np.asarray([0], dtype=np.int64)

    def batch(self, rows: np.ndarray, device: str) -> dict[str, torch.Tensor]:
        assert rows.tolist() == [0]
        # The two padded/invalid candidates deliberately receive the largest
        # scores.  Correct evaluation must exclude them before torch.topk.
        scores = torch.tensor(
            [[[[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 100.0, 99.0]]]],
            device=device,
        )
        membership = torch.zeros_like(scores)
        membership[..., :8] = 1.0
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[..., :8] = True
        return {
            "predicted_scores": scores,
            "candidate_scores": scores.clone(),
            "target_membership": membership,
            "candidate_mask": mask,
            "valid_future": torch.ones((1, 1), dtype=torch.bool, device=device),
        }


def test_paired_bootstrap_aggregates_complete_requests() -> None:
    rows = []
    for request_id, gain in ((1, 0.1), (2, 0.2), (3, 0.3)):
        for horizon in range(1, 5):
            rows.append({
                "request_id": request_id, "horizon": horizon,
                "recall_at_8": 0.5 + gain, "base_recall_at_8": 0.5,
            })
    result = paired_request_bootstrap(rows, replicates=200, seed=5)
    assert result["requests"] == 3
    assert result["paired_gain"] == pytest.approx(0.2)
    assert result["ci95_lower"] > 0.0


def test_paired_bootstrap_rejects_incomplete_horizons() -> None:
    with pytest.raises(ValueError, match="four horizons"):
        paired_request_bootstrap([
            {"request_id": 1, "horizon": 1, "recall_at_8": 0.5, "base_recall_at_8": 0.4}
        ])


def test_metric_topk_excludes_invalid_candidates() -> None:
    result = evaluate_jspace_ranker(
        _ScoresFromBatch(),
        _MaskedMetricData(),  # type: ignore[arg-type]
        batch_size=1,
        device="cpu",
        autocast=False,
    )
    horizon = result["horizon_metrics"][0]
    assert horizon["recall_at_8"] == pytest.approx(1.0)
    assert horizon["base_recall_at_8"] == pytest.approx(1.0)
    assert horizon["coverage_at_10"] == pytest.approx(1.0)
    assert result["native_k"] == 8
    assert result["candidate_count"] == 10
