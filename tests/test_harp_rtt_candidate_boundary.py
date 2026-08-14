from __future__ import annotations

import torch

from harp_rtt.candidate_boundary import candidate_entry_loss


def test_candidate_entry_is_zero_when_all_true_are_contained() -> None:
    scores = torch.arange(12, dtype=torch.float32)[None, None]
    anchor = scores.clone()
    truth = torch.tensor([[[11, 10]]])
    loss = candidate_entry_loss(
        scores, anchor, truth, torch.ones(1, 1, dtype=torch.bool),
        anchor_quota=2, candidate_width=4,
    )
    assert float(loss) == 0.0


def test_candidate_entry_pushes_only_missing_true_expert_up() -> None:
    aligned = torch.arange(12, dtype=torch.float32)[None, None].requires_grad_()
    anchor = torch.arange(12, dtype=torch.float32)[None, None]
    truth = torch.tensor([[[11, 0]]])
    loss = candidate_entry_loss(
        aligned, anchor, truth, torch.ones(1, 1, dtype=torch.bool),
        anchor_quota=2, candidate_width=4,
    )
    loss.backward()
    assert aligned.grad is not None
    assert float(aligned.grad[0, 0, 0]) < 0.0
    assert int((aligned.grad != 0).sum()) == 1


def test_candidate_entry_respects_invalid_rows_and_stable_ties() -> None:
    aligned = torch.arange(
        12, dtype=torch.float32
    )[None, None].expand(2, 1, 12).clone().requires_grad_()
    anchor = torch.arange(12, dtype=torch.float32)[None, None].expand(2, 1, 12)
    truth = torch.tensor([[[0, 1]], [[0, 1]]])
    valid = torch.tensor([[True], [False]])
    loss = candidate_entry_loss(
        aligned, anchor, truth, valid, anchor_quota=2, candidate_width=4,
    )
    loss.backward()
    assert float(aligned.grad[0].abs().sum()) > 0.0
    assert float(aligned.grad[1].abs().sum()) == 0.0
