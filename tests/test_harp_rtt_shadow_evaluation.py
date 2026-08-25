import json

import torch

from runpod.evaluate_shadow_route_top8 import (
    bootstrap_factual_recall, load_current_expert_sets, native_parity_mask,
    summarize_cache_set,
)


def test_shadow_bootstrap_is_complete_request_deterministic() -> None:
    rows = []
    for request, value in (("request-a", 0.8), ("request-b", 1.0)):
        rows.append(
            {
                "condition": "shadow_root",
                "metric": "recall",
                "request_id": request,
                "horizon": 1,
                "value": value,
            }
        )
        for horizon in (2, 3, 4):
            rows.append(
                {
                    "condition": "shadow_lm",
                    "metric": "recall",
                    "request_id": request,
                    "horizon": horizon,
                    "value": value,
                }
            )

    first = bootstrap_factual_recall(rows, replicates=100, seed=42)
    second = bootstrap_factual_recall(rows, replicates=100, seed=42)

    assert first == second
    summary = first["conditions"]["shadow_lm"]
    assert summary["request_count"] == 2
    assert summary["h2_h4"]["point"] == 0.9
    assert summary["h1_h4"]["point"] == 0.9


def test_native_parity_excludes_valid_nodes_outside_runtime_budget() -> None:
    valid = torch.tensor(
        [[True, True], [True, True], [True, False], [True, True]]
    )
    node_mask = torch.tensor([True, True, False, False])

    active = native_parity_mask(valid, node_mask, count=4)

    assert torch.equal(
        active,
        torch.tensor(
            [[True, True], [True, True], [False, False], [False, False]]
        ),
    )


def test_load_current_expert_sets_uses_source_token_routes(tmp_path) -> None:
    rows = []
    for layer in range(40):
        rows.append({
            "event": "target_layer",
            "record_valid": True,
            "sequence_id": "sequence-a",
            "committed_token_position": 11,
            "target_layer": layer,
            "candidate_expert_ids": list(range(layer, layer + 8)),
        })
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    loaded = load_current_expert_sets(
        tmp_path, [{"sequence_id": "sequence-a", "source_position": 11}]
    )

    assert loaded[("sequence-a", 11)].shape == (40, 8)
    assert loaded[("sequence-a", 11)][39].tolist() == list(range(39, 47))


def test_summarize_cache_set_reports_macro_and_pooled_counts() -> None:
    cells = {}
    for horizon in (1, 2, 3, 4):
        cells[("request-a", horizon)] = [(3, 4, 2)]
        cells[("request-b", horizon)] = [(1, 2, 2)]

    summary, rows = summarize_cache_set(cells)

    h1 = summary["horizons"]["1"]
    assert h1["request_macro_recall"] == 0.625
    assert h1["pooled_recall"] == 4 / 6
    assert h1["pooled_mean_true_intersection_experts"] == 1.5
    assert h1["true_intersection_experts_total"] == 6
    assert len(rows) == 8
