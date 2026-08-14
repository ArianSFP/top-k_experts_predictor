from __future__ import annotations

import torch

from harp_rtt.content_student import CausalContentStudent


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(19)
    batch, horizons, layers, nodes, width, experts = 2, 3, 4, 5, 8, 11
    posterior = torch.rand(batch, horizons, nodes + 1)
    posterior = posterior / posterior.sum(-1, keepdim=True)
    mask = torch.ones(batch, horizons, nodes, dtype=torch.bool)
    mask[0, 1, 3:] = False
    posterior[0, 1, 3:-1] = 0
    posterior[0, 1] = posterior[0, 1] / posterior[0, 1].sum()
    return {
        "node_context": torch.randn(batch, horizons, layers, nodes, width),
        "other_context": torch.randn(batch, horizons, layers, width),
        "posterior": posterior,
        "node_mask": mask,
        "parent_scores": torch.randn(batch, horizons, layers, experts),
    }


def _student() -> CausalContentStudent:
    return CausalContentStudent(
        horizons=3, layers=4, nodes=5, experts=11, width=8, score_rank=3
    )


def test_epoch_zero_exactly_reproduces_parent_scores() -> None:
    inputs = _inputs()
    output = _student()(**inputs)
    assert torch.equal(output.scores, inputs["parent_scores"].float())
    assert torch.allclose(output.branch_weights.sum(-1), torch.ones(2, 3, 4))
    assert not bool(output.branch_weights[0, 1, :, 3:5].any())


def test_node_permutation_is_equivariant() -> None:
    inputs = _inputs()
    model = _student().eval()
    permutation = torch.tensor([2, 0, 4, 1, 3])
    permuted = dict(inputs)
    permuted["node_context"] = inputs["node_context"][:, :, :, permutation]
    permuted["node_mask"] = inputs["node_mask"][:, :, permutation]
    permuted["posterior"] = torch.cat((
        inputs["posterior"][..., :-1][..., permutation],
        inputs["posterior"][..., -1:],
    ), dim=-1)
    with torch.no_grad():
        original = model(**inputs)
        changed = model(**permuted)
    assert torch.allclose(original.content_latent, changed.content_latent, atol=1e-6)
    assert torch.equal(original.scores, changed.scores)


def test_score_and_reliability_gates_receive_gradients() -> None:
    inputs = _inputs()
    model = _student()
    with torch.no_grad():
        model.score_gate.fill_(0.1)
        model.reliability_gate.fill_(0.1)
    output = model(**inputs)
    loss = output.scores.square().mean() + output.content_latent.square().mean()
    loss.backward()
    assert model.score_gate.grad is not None
    assert model.reliability_gate.grad is not None
    assert torch.isfinite(model.score_gate.grad).all()
    assert torch.isfinite(model.reliability_gate.grad).all()


def test_unavailable_node_cannot_change_output() -> None:
    inputs = _inputs()
    model = _student().eval()
    changed = dict(inputs)
    changed["node_context"] = inputs["node_context"].clone()
    changed["node_context"][0, 1, :, 3:] = 1e6
    with torch.no_grad():
        before = model(**inputs)
        after = model(**changed)
    assert torch.equal(before.content_latent[0, 1], after.content_latent[0, 1])
    assert torch.equal(before.scores[0, 1], after.scores[0, 1])
