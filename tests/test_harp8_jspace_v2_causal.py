from __future__ import annotations

import torch

from harp8.jspace_v2_data import causal_v2_model_inputs


def test_causal_forward_allowlist_excludes_every_supervision_tensor() -> None:
    outer_training_batch = {
        "candidate_scores": torch.ones(1),
        "candidate_ids": torch.zeros(1, dtype=torch.long),
        "candidate_mask": torch.ones(1, dtype=torch.bool),
        "j_states": torch.ones(1),
        "j_mask": torch.ones(1, dtype=torch.bool),
        "mtp_states": torch.ones(1),
        "mtp_router_logits": torch.ones(1),
        "mtp_mask": torch.ones(1, dtype=torch.bool),
        "generator_context": torch.ones(1),
        "target_membership": torch.ones(1),
        "teacher_candidate_scores": torch.ones(1),
        "valid_future": torch.ones(1),
    }
    model_inputs = causal_v2_model_inputs(outer_training_batch)
    assert "target_membership" not in model_inputs
    assert "teacher_candidate_scores" not in model_inputs
    assert "valid_future" not in model_inputs
    assert set(model_inputs) == {
        "candidate_scores",
        "candidate_ids",
        "candidate_mask",
        "j_states",
        "j_mask",
        "mtp_states",
        "mtp_router_logits",
        "mtp_mask",
        "generator_context",
    }
