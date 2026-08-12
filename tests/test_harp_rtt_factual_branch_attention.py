from __future__ import annotations

import torch

from harp_rtt.b31 import quota_candidate_union
from harp_rtt.factual_branch_attention import (
    ExpertConditionedBranchAttention,
    SummaryFactualAligner,
    parent_branch_marginals,
)


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    batch, horizons, layers, nodes, experts, width = 2, 4, 3, 4, 16, 8
    anchor_scores = torch.randn(batch, horizons, layers, experts)
    anchor_marginals = torch.softmax(anchor_scores, -1) * 2.0
    node_logits = torch.randn(batch, horizons, layers, nodes, experts)
    node_marginals = torch.softmax(node_logits, -1) * 2.0
    posterior = torch.tensor(
        [[[0.35, 0.25, 0.15, 0.05, 0.20]]]
    ).expand(batch, horizons, -1).clone()
    node_mask = torch.ones(batch, horizons, nodes, dtype=torch.bool)
    return {
        "anchor_scores": anchor_scores,
        "anchor_marginals": anchor_marginals,
        "node_marginals": node_marginals,
        "posterior": posterior,
        "mtp_probabilities": posterior[..., :-1].clone(),
        "node_mask": node_mask,
        "first_divergence_depth": torch.tensor([[2, 3, 4, -1]]).expand(batch, -1),
        "tree_states": torch.randn(batch, nodes, width),
        "context_states": torch.randn(batch, horizons, layers, width),
    }


def test_m0_zero_gate_reproduces_parent_candidate_order() -> None:
    values = _inputs()
    model = SummaryFactualAligner(
        layers=3, experts=16, exact_k=2, hidden_width=8,
        candidate_width=8, anchor_quota=4,
    )
    output = model(**{key: values[key] for key in (
        "anchor_scores", "anchor_marginals", "node_marginals", "posterior",
        "mtp_probabilities", "node_mask", "first_divergence_depth",
    )})
    parent = parent_branch_marginals(
        values["node_marginals"], values["posterior"], values["node_mask"],
        values["anchor_marginals"], k=2,
    )
    expected = quota_candidate_union(
        values["anchor_scores"], parent, anchor_quota=4, width=8
    ).expert_ids
    assert torch.equal(output.candidate_ids, expected)
    assert torch.equal(output.marginals, parent)
    assert torch.allclose(output.marginals.sum(-1), torch.full_like(parent[..., 0], 2.0))


def test_m0_gate_and_expert_head_receive_factual_gradients() -> None:
    values = _inputs()
    model = SummaryFactualAligner(
        layers=3, experts=16, exact_k=2, hidden_width=8,
        candidate_width=8, anchor_quota=4,
    )
    output = model(**{key: values[key] for key in (
        "anchor_scores", "anchor_marginals", "node_marginals", "posterior",
        "mtp_probabilities", "node_mask", "first_divergence_depth",
    )})
    output.scores.square().mean().backward()
    assert model.gate.grad is not None and model.gate.grad.abs().sum() > 0
    # The zero gate intentionally protects the parent on the first step.
    assert model.output.weight.grad is not None
    assert model.output.weight.grad.abs().sum() == 0


def test_m1_tau_zero_reproduces_posterior_mixture_and_normalizes_other() -> None:
    values = _inputs()
    keys = torch.randn(3, 16, 5)
    model = ExpertConditionedBranchAttention(
        keys, horizons=4, exact_k=2, tree_width=8, hidden_width=8,
        candidate_width=8, anchor_quota=4,
    )
    output = model(**{key: values[key] for key in (
        "anchor_scores", "anchor_marginals", "node_marginals", "posterior",
        "node_mask", "tree_states", "context_states",
    )})
    parent = parent_branch_marginals(
        values["node_marginals"], values["posterior"], values["node_mask"],
        values["anchor_marginals"], k=2,
    )
    assert output.expert_branch_weights is not None
    assert torch.allclose(output.expert_branch_weights.sum(3), torch.ones_like(parent))
    assert torch.equal(output.marginals, parent)
    expected = quota_candidate_union(
        values["anchor_scores"], parent, anchor_quota=4, width=8
    ).expert_ids
    assert torch.equal(output.candidate_ids, expected)


def test_m1_is_equivariant_to_node_permutation_and_masks_nodes() -> None:
    values = _inputs()
    values["node_mask"][:, :, -1] = False
    values["posterior"][:, :, -1] += values["posterior"][:, :, 3]
    values["posterior"][:, :, 3] = 0.0
    keys = torch.randn(3, 16, 5)
    model = ExpertConditionedBranchAttention(
        keys, horizons=4, exact_k=2, tree_width=8, hidden_width=8,
        candidate_width=8, anchor_quota=4,
    )
    with torch.no_grad():
        model.tau.fill_(0.25)
    original = model(**{key: values[key] for key in (
        "anchor_scores", "anchor_marginals", "node_marginals", "posterior",
        "node_mask", "tree_states", "context_states",
    )})
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = dict(values)
    permuted["node_marginals"] = values["node_marginals"][:, :, :, permutation]
    permuted["posterior"] = torch.cat(
        (values["posterior"][..., :-1][..., permutation], values["posterior"][..., -1:]), -1
    )
    permuted["node_mask"] = values["node_mask"][..., permutation]
    permuted["tree_states"] = values["tree_states"][:, permutation]
    changed = model(**{key: permuted[key] for key in (
        "anchor_scores", "anchor_marginals", "node_marginals", "posterior",
        "node_mask", "tree_states", "context_states",
    )})
    assert torch.allclose(original.marginals, changed.marginals, atol=2e-6, rtol=2e-6)
    assert original.expert_branch_weights is not None
    assert torch.equal(
        original.expert_branch_weights[:, :, :, 3],
        torch.zeros_like(original.expert_branch_weights[:, :, :, 3]),
    )


def test_m1_expert_specific_weights_change_after_tau_opens() -> None:
    values = _inputs()
    model = ExpertConditionedBranchAttention(
        torch.randn(3, 16, 5), horizons=4, exact_k=2,
        tree_width=8, hidden_width=8, candidate_width=8, anchor_quota=4,
    )
    with torch.no_grad():
        model.tau.fill_(0.5)
    output = model(**{key: values[key] for key in (
        "anchor_scores", "anchor_marginals", "node_marginals", "posterior",
        "node_mask", "tree_states", "context_states",
    )})
    assert output.expert_branch_weights is not None
    assert output.expert_branch_weights.var(-1).mean() > 0
    assert torch.isfinite(output.coherence_kl) and output.coherence_kl >= 0
