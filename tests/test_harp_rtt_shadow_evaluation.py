from runpod.evaluate_shadow_route_top8 import bootstrap_factual_recall


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
