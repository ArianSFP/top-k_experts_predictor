from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNPOD = ROOT / "runpod"
if str(RUNPOD) not in sys.path:
    sys.path.insert(0, str(RUNPOD))

from evaluate_harp_rtt_b15_oracle import _oracle_batch, _target_greedy_candidates  # noqa: E402
from harp_rtt.counterfactual import empty_counterfactual_tensors  # noqa: E402
from harp_rtt.node_counterfactual import empty_node_counterfactual_tensors  # noqa: E402


def _anchor() -> torch.Tensor:
    torch.manual_seed(19)
    return torch.randn(1, 4, 40, 256)


def test_node_oracle_uses_native_selected_membership_and_other() -> None:
    labels = empty_node_counterfactual_tensors()
    labels["node_mask"][:4] = True
    labels["node_local_indices"][:4] = torch.arange(4)
    labels["parent_local_indices"][:4] = torch.tensor([-1, 0, 1, 2])
    labels["depth"][:4] = torch.tensor([1, 2, 3, 4])
    labels["budget_node_masks"][:, :4] = True
    labels["valid"][1:4] = True
    labels["selected_ids"][1:4] = torch.arange(8, dtype=torch.int32)
    labels["target_path_logp"][1:4] = torch.tensor([-0.2, -0.4, -0.7])
    labels["source_path_logp"][:4] = torch.tensor([0.0, -0.1, -0.3, -0.6])
    batched = {name: value.unsqueeze(0) for name, value in labels.items()}
    result = _oracle_batch(batched, _anchor(), budget="all")
    assert torch.allclose(
        result["target_captured_mass"][0, 1, 0], torch.tensor(-0.2).exp()
    )
    assert torch.allclose(
        result["target_branch"][0, 1, 0, :8],
        torch.full((8,), torch.tensor(-0.2).exp()),
    )
    assert torch.allclose(
        result["target_full"].sum(-1), torch.full((1, 4, 40), 8.0), atol=1e-4
    )


def test_target_greedy_diagnostic_follows_argmax_children() -> None:
    labels = empty_node_counterfactual_tensors()
    labels["node_mask"][:4] = True
    labels["node_local_indices"][:4] = torch.arange(4)
    labels["parent_local_indices"][:4] = torch.tensor([-1, 0, 1, 2])
    labels["depth"][:4] = torch.tensor([1, 2, 3, 4])
    labels["budget_node_masks"][:, :4] = True
    labels["target_next_token_ids"][:4] = torch.tensor([20, 30, 40, 50])
    labels["target_next_token_valid"][:4] = True
    labels["valid"][1:4] = True
    labels["selected_ids"][1:4] = torch.arange(8, dtype=torch.int32)
    batched = {name: value.unsqueeze(0) for name, value in labels.items()}
    tree = {
        "token_ids": torch.tensor([[10, 20, 30, 40] + [0] * 28]),
        "parent": torch.tensor([[-1, 0, 1, 2] + [-1] * 28]),
    }
    result = _target_greedy_candidates(
        batched, tree, _anchor(), budget="all"
    )
    assert result is not None
    candidates, occurrence = result
    assert occurrence.tolist() == [[True, True, True, True]]
    truth = batched["selected_ids"][:, 3]
    assert (
        (candidates[:, 3].unsqueeze(-1) == truth.unsqueeze(-2))
        .any(-2)
        .all()
    )


def test_legacy_four_path_oracle_is_still_readable() -> None:
    labels = empty_counterfactual_tensors()
    labels["path_mask"][0] = True
    labels["path_depths"][0] = 4
    labels["node_local_indices"][0] = torch.tensor([0, 1, 2, 3])
    labels["valid"][0, 1:] = True
    labels["selected_ids"][0, 1:] = torch.arange(8, dtype=torch.int32)
    labels["target_path_logp"][0, 1:] = torch.tensor([-0.2, -0.4, -0.7])
    labels["source_path_logp"][0] = torch.tensor([0.0, -0.1, -0.3, -0.6])
    batched = {name: value.unsqueeze(0) for name, value in labels.items()}
    result = _oracle_batch(batched, _anchor(), budget="4")
    assert torch.allclose(
        result["target_captured_mass"][0, 3, 0], torch.tensor(-0.7).exp()
    )
    assert torch.equal(
        result["target_branch"][0, 3, 0, :8] > 0, torch.ones(8, dtype=torch.bool)
    )


def test_node_budget_diagnostics_preserve_batch_dimension() -> None:
    labels = empty_node_counterfactual_tensors()
    batched = {
        name: torch.stack([value, value], dim=0)
        for name, value in labels.items()
    }
    batch_size = batched["budget_realized"].shape[0]
    rows = torch.cat(
        [
            batched["budget_realized"].long(),
            batched["budget_category_counts"].long().reshape(batch_size, -1),
            batched["budget_node_masks"].sum(-1).long(),
        ],
        dim=-1,
    )
    assert rows.shape == (2, 18)
