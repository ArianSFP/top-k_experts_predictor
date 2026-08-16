import torch

from runpod.evaluate_shadow_route_top8 import (
    assert_prediction_record_label_free, bootstrap_factual_recall,
    native_parity_mask, resident_omission_telemetry,
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


def test_prediction_sidecar_rejects_label_like_fields() -> None:
    assert_prediction_record_label_free({"router_logits": torch.zeros(1)})
    try:
        assert_prediction_record_label_free({"target_ids": torch.zeros(1)})
    except PermissionError:
        pass
    else:
        raise AssertionError("target label reached the prediction-only sidecar")


def test_resident_telemetry_excludes_current_layer_from_prior_exposure() -> None:
    selected = torch.full((2, 40, 8), 100, dtype=torch.long)
    weights = torch.full((2, 40, 8), 0.125)
    residents = tuple(torch.arange(64) for _ in range(40))
    telemetry = resident_omission_telemetry(
        selected_ids=selected, selected_weights=weights,
        parent_indices=torch.tensor([-1, 0]), node_mask=torch.tensor([True, True]),
        static_resident_ids=residents, current_selected_ids=None,
    )
    assert telemetry["immediate_missing_mass"][0, 0] == 1
    assert telemetry["prior_missing_mass"][0, 0] == 0
    assert telemetry["prior_missing_mass"][0, 1] == 1
    assert telemetry["prior_missing_mass"][1, 0] == 40
