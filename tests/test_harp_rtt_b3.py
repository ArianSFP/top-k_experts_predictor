from __future__ import annotations

import json

import pytest
from torch import nn

from runpod.train_harp_rtt_b3 import (
    configure_b3_parameters,
    generator_selection,
    verify_b2_override,
)


class _TinyB3Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Linear(2, 2)
        self.token_embedding = nn.Embedding(4, 2)
        self.tree_encoder = nn.Linear(2, 2)
        self.score_head = nn.Linear(2, 2)
        self.reranker = nn.Linear(2, 2)
        self._anchor_frozen = True


def test_b3_ownership_keeps_generator_and_ranker_disjoint() -> None:
    generator = _TinyB3Model()
    _, generator_report = configure_b3_parameters(
        generator,
        stage="generator",
        epochs=2,
        learning_rate=2e-4,
        weight_decay=0.01,
    )
    assert set(generator_report["new_names"]) == {
        "tree_encoder.weight",
        "tree_encoder.bias",
        "score_head.weight",
        "score_head.bias",
    }
    assert not generator.reranker.weight.requires_grad
    assert not generator.anchor.weight.requires_grad

    ranker = _TinyB3Model()
    _, ranker_report = configure_b3_parameters(
        ranker,
        stage="ranker",
        epochs=2,
        learning_rate=2e-4,
        weight_decay=0.01,
    )
    assert set(ranker_report["new_names"]) == {
        "reranker.weight",
        "reranker.bias",
    }
    assert not ranker.tree_encoder.weight.requires_grad


def test_b2_override_is_operational_not_a_three_seed_claim(tmp_path) -> None:
    record = {
        "schema": "harp_rtt_b2_user_operational_override_v1",
        "user_authorized": True,
        "operational_pass": True,
        "move_to_b3_authorized": True,
        "thresholds_changed": False,
        "preregistered_three_seed_gate_completed": False,
        "preregistered_three_seed_gate_passed": None,
        "single_seed_statistical_claim": False,
        "completed_seed": {"seed": 42, "checkpoint_sha256": "0" * 64},
    }
    path = tmp_path / "override.json"
    path.write_text(json.dumps(record))
    assert verify_b2_override(path)["completed_seed"]["seed"] == 42
    record["preregistered_three_seed_gate_completed"] = True
    path.write_text(json.dumps(record))
    with pytest.raises(PermissionError, match="three_seed_gate_completed"):
        verify_b2_override(path)


def test_generator_selection_prioritizes_h2_h4_candidate_information() -> None:
    report = {
        "horizon_metrics": [
            {"request_macro_candidate_coverage_at_64": 0.80},
            {"request_macro_candidate_coverage_at_64": 0.98},
            {"request_macro_candidate_coverage_at_64": 0.99},
            {"request_macro_candidate_coverage_at_64": 0.97},
        ],
        "mean_h1_h4_request_macro_slot_recall_at_8": 0.81,
        "mean_h1_h4_request_macro_exact_set_nll": 2.0,
    }
    selection = generator_selection(report)
    assert selection == pytest.approx((0.98, 0.97, 0.81, -2.0))
