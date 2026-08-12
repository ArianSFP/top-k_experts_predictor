from __future__ import annotations

import pytest
import torch

from harp_rtt.delta import HARPDeltaConfig
from harp_rtt.deltaroute_batch import prepare_deltaroute_batch

from test_harp_rtt_delta_batch import config


def _batch() -> dict[str, object]:
    c = config()
    batch, history, nodes, hidden = 1, 3, 4, 6
    hashes = torch.zeros(batch, nodes, 32, dtype=torch.uint8)
    hashes[0, :, 0] = torch.arange(1, nodes + 1)
    tree = {
        "states": torch.randn(batch, nodes, 4, hidden),
        "router_logits": torch.randn(batch, nodes, c.experts),
        "vocab_top64_ids": torch.arange(64)[None, None].expand(batch, nodes, -1),
        "vocab_top64_log_probabilities": torch.full((batch, nodes, 64), -5.0),
        "vocab_statistics": torch.randn(batch, nodes, 6),
        "meta": torch.zeros(batch, nodes, 13, dtype=torch.long),
        "path_log_probabilities": torch.log(torch.tensor([[0.8, 0.4, 0.2, 0.1]])),
        "local_probabilities": torch.tensor([[0.8, 0.5, 0.5, 0.5]]),
        "source_ready": torch.ones(batch, nodes),
        "depth": torch.tensor([[1, 2, 3, 4]]),
        "child_ranks": torch.zeros(batch, nodes),
        "first_divergence_depths": torch.zeros(batch, nodes),
        "cumulative_path_ranks": torch.ones(batch, nodes),
        "sibling_counts": torch.ones(batch, nodes),
        "parent": torch.tensor([[-1, 0, 1, 2]]),
        "mask": torch.ones(batch, nodes, dtype=torch.bool),
        "horizon_mask": torch.eye(4, dtype=torch.bool)[None],
        "exact_prefix_hashes": hashes,
    }
    tree["meta"][..., 7] = torch.tensor([[1, 2, 3, 4]])
    return {
        "inputs": {
            "history": {
                "logits": torch.randn(batch, history, c.layers, c.experts),
                "selected_ids": torch.zeros(batch, history, c.layers, 2, dtype=torch.long),
                "execution_weights": torch.full((batch, history, c.layers, 2), 0.5),
                "available": torch.ones(batch, history, c.layers, dtype=torch.bool),
            },
            "current": {"normalized_target_router_input_a": torch.randn(batch, c.layers, hidden)},
            "tree": tree,
            "exact_next_token_id": torch.tensor([5]),
            "within_request": torch.tensor([101.0]),
            "final_hidden": torch.randn(batch, hidden),
        },
        "targets": {"future_prefix_hashes": hashes.clone()},
    }


def test_deltaroute_batch_preserves_channel_identity_and_seals_labels() -> None:
    batch = _batch()
    c = config()
    prepared = prepare_deltaroute_batch(
        batch,
        anchor_scores=torch.randn(1, 4, c.layers, c.experts),
        token_embedding=torch.randn(128, 6),
        input_basis=torch.randn(c.layers, 6, c.router_rank),
        rank_mask=torch.ones(c.layers, c.router_rank, dtype=torch.bool),
        config=c,
    )
    states = batch["inputs"]["tree"]["states"]
    assert torch.equal(prepared.trajectory_inputs["fused"], states[:, :, 0])
    assert torch.equal(prepared.trajectory_inputs["post_ffn"], states[:, :, 1])
    assert torch.equal(prepared.trajectory_inputs["router_input"], states[:, :, 2])
    assert "selected_ids" not in prepared.trajectory_inputs
    assert set(prepared.history_targets) == {
        "logits", "selected_ids", "execution_weights", "available"
    }


def test_deltaroute_batch_rejects_counterfactual_inputs() -> None:
    batch = _batch()
    batch["inputs"]["counterfactual"] = {"query_coordinates": torch.ones(1)}
    with pytest.raises(PermissionError, match="target-only"):
        prepare_deltaroute_batch(
            batch,
            anchor_scores=torch.randn(1, 4, 2, 64),
            token_embedding=torch.randn(128, 6),
            input_basis=torch.randn(2, 6, 3),
            rank_mask=torch.ones(2, 3, dtype=torch.bool),
            config=config(),
        )
