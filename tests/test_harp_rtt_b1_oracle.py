from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


RUNPOD = Path(__file__).resolve().parents[1] / "runpod"
if str(RUNPOD) not in sys.path:
    sys.path.insert(0, str(RUNPOD))

from evaluate_harp_rtt_b1_oracle import (  # noqa: E402
    _anchor_branch_union,
    _bootstrap_delta,
    _first_divergence,
    _finite_mean_or_none,
    _oracle_branch_scores,
)


def test_finite_mean_serializes_empty_mask_as_null() -> None:
    assert _finite_mean_or_none(np.asarray([np.nan, np.nan])) is None
    assert _finite_mean_or_none(np.asarray([1.0, np.nan, 3.0])) == 2.0


def test_oracle_union_preserves_anchor48_and_fills_from_branch() -> None:
    anchor = -torch.arange(256, dtype=torch.float32).reshape(1, 1, 1, 256)
    branch = torch.arange(256, dtype=torch.float32).reshape(1, 1, 1, 256)
    result = _anchor_branch_union(anchor, branch).reshape(-1)
    assert result[:48].tolist() == list(range(48))
    assert result[48:].tolist() == list(range(255, 239, -1))
    assert len(set(result.tolist())) == 64


def test_oracle_branch_scores_deduplicate_shared_prefix_nodes() -> None:
    counterfactual = {
        "router_logits": torch.zeros(1, 4, 4, 40, 256),
        "valid": torch.zeros(1, 4, 4, 40, dtype=torch.bool),
        "node_local_indices": torch.full((1, 4, 4), -1, dtype=torch.int64),
        "target_path_logp": torch.full((1, 4, 4), float("nan")),
    }
    # Every H2 slot points at one shared node; it must contribute once, not 4x.
    counterfactual["valid"][:, :, 1:] = True
    counterfactual["node_local_indices"][:, :, 1] = 1
    counterfactual["node_local_indices"][:, :, 2] = torch.tensor([2, 2, 3, 3])
    counterfactual["node_local_indices"][:, :, 3] = torch.tensor([4, 5, 6, 7])
    counterfactual["target_path_logp"][:, :, 1] = np.log(0.2)
    counterfactual["target_path_logp"][:, :, 2] = np.log(0.1)
    counterfactual["target_path_logp"][:, :, 3] = np.log(0.05)
    counterfactual["router_logits"][..., 7] = 3.0
    scores, mass = _oracle_branch_scores(counterfactual)
    assert mass[0, 1].item() == pytest.approx(0.2)
    assert mass[0, 2].item() == pytest.approx(0.2)
    assert mass[0, 3].item() == pytest.approx(0.2)
    assert torch.all(scores[0, 1, :, 7] > scores[0, 1, :, 8])


def test_prefix_divergence_requires_nested_matches() -> None:
    values = np.asarray(
        [
            [True, True, True, True],
            [True, False, False, False],
            [True, True, False, False],
        ]
    )
    divergence, report = _first_divergence(values)
    assert divergence.tolist() == [0, 2, 3]
    assert report["verified"] is True
    with pytest.raises(ValueError, match="nesting"):
        _first_divergence(np.asarray([[True, False, True, False]]))


def test_paired_bootstrap_is_seeded_and_request_complete() -> None:
    candidate = {"a": 0.9, "b": 0.8, "c": 0.7}
    reference = {"a": 0.8, "b": 0.7, "c": 0.6}
    first = _bootstrap_delta(candidate, reference, replicates=1000, seed=42)
    second = _bootstrap_delta(candidate, reference, replicates=1000, seed=42)
    assert first == second
    assert first["requests"] == 3
    assert first["lower_bound_positive"] is True
