from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from types import SimpleNamespace

from harp_rtt.factual_branch_attention import FactualAlignmentOutput
from harp_rtt.route_dynamics import DeltaRouteConfig, DeltaRouteTrajectory


SCRIPT = Path(__file__).parents[1] / "runpod" / "train_harp_deltaroute_v4_dynamics.py"
SPEC = importlib.util.spec_from_file_location("train_harp_deltaroute_v4_dynamics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _trajectory() -> DeltaRouteTrajectory:
    config = DeltaRouteConfig(
        experts=8, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=6, metadata_width=4, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    return DeltaRouteTrajectory(
        config, torch.randn(4, 8, 3), torch.randn(4, 8),
        torch.ones(4, 3, dtype=torch.bool),
    )


def test_aligner_failure_explicitly_keeps_dynamics_authorized(tmp_path: Path) -> None:
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps({
        "schema": MODULE.ALIGNER_AGGREGATE_SCHEMA,
        "selected_aligner": None,
        "aligner_promoted": False,
        "route_dynamics_remains_authorized_if_aligner_fails": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }), encoding="utf-8")
    value = MODULE.validate_aligner_aggregate(path)
    assert value["aligner_promoted"] is False


def test_r2_delays_only_transition_core_and_keeps_initializer_trainable() -> None:
    trajectory = _trajectory()
    primary, delayed = MODULE.configure_stage_parameters(
        trajectory, None, "rollout_r2"
    )
    assert primary and delayed
    assert any(name.startswith("trajectory.seed") for name, _ in primary)
    assert all(name.startswith("trajectory.dynamics") for name, _ in delayed)
    assert all(parameter.requires_grad for _, parameter in primary)
    assert all(not parameter.requires_grad for _, parameter in delayed)
    assert all("free_" not in name for name, _ in primary)


def test_r0_r1_train_causal_context_and_dynamics_but_not_r2_gates() -> None:
    for stage in ("transition_r0", "rollout_r1"):
        trajectory = _trajectory()
        primary, delayed = MODULE.configure_stage_parameters(
            trajectory, None, stage
        )
        names = {name for name, _ in primary}
        assert delayed == []
        assert any(name.startswith("trajectory.channels") for name in names)
        assert any(name.startswith("trajectory.dynamics") for name in names)
        assert not any(name.startswith("trajectory.seed") for name in names)


def test_gate_constants_match_preregistered_recovery() -> None:
    expected = MODULE.R0_PARENT_ALLNODE_RECALL + 0.5 * (
        1 - MODULE.R0_PARENT_ALLNODE_RECALL
    )
    assert MODULE.R0_REQUIRED_RECALL == pytest.approx(expected)
    assert MODULE.PREDECESSOR == {
        "transition_r0": None,
        "rollout_r1": "transition_r0",
        "rollout_r2": "rollout_r1",
        "factual_joint": "rollout_r2",
    }


def test_driver_preserves_allnode_pretraining_and_budget16_deployment() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"allnode_transition_supervision": True' in source
    assert '"runtime_tree_nodes": 32' in source
    assert '"factual_supervision_budget": 16' in source
    assert '"OPTIMIZER_START.json"' in source
    assert '"ranker_authorized"' in source
    assert "All 32 causal runtime nodes participate" in source


def test_joint_objective_scatter_maps_node_scores_to_matching_horizon() -> None:
    trajectory = _trajectory()
    batch, horizons, layers, nodes, experts, exact_k = 1, 4, 4, 3, 8, 2
    parent_scores = torch.randn(batch, horizons, layers, nodes, experts)
    route_scores = torch.randn(batch, nodes, layers, experts, requires_grad=True)
    depth = torch.tensor([[2, 3, 4]])
    node_mask = torch.ones(batch, nodes, dtype=torch.bool)
    target_ids = torch.argsort(
        route_scores.detach(), dim=-1, descending=True, stable=True
    )[..., :exact_k]
    target_queries = torch.randn(batch, nodes, layers, 3)
    counterfactual = {
        "node_mask": node_mask,
        "budget_node_masks": node_mask[:, None].expand(batch, 3, nodes).clone(),
        "depth": depth,
        "valid": torch.ones(batch, nodes, layers, dtype=torch.bool),
        "query_coordinates": target_queries,
        "router_logits": route_scores.detach(),
        "selected_ids": target_ids,
    }
    future_ids = torch.tensor([0, 1]).reshape(1, 1, 1, 2).expand(
        batch, horizons, layers, exact_k
    ).clone()
    posterior = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0], [0.8, 0.0, 0.0, 0.2],
          [0.0, 0.8, 0.0, 0.2], [0.0, 0.0, 0.8, 0.2]]]
    )
    branch_mask = posterior[..., :-1] > 0
    aligned_scores = torch.randn(batch, horizons, layers, experts, requires_grad=True)
    aligned = FactualAlignmentOutput(
        aligned_scores, torch.sigmoid(aligned_scores),
        torch.zeros_like(aligned_scores), None,
        aligned_scores.sum() * 0.0,
        torch.argsort(
            aligned_scores, dim=-1, descending=True, stable=True
        )[..., :4],
    )
    output = MODULE.BatchRouteOutput(
        queries=target_queries,
        scores=route_scores,
        selected_ids=target_ids,
        affine_queries=None,
        targets={
            "future_selected_ids": future_ids,
            "future_available": torch.ones(batch, horizons, layers, dtype=torch.bool),
        },
        counterfactual=counterfactual,
        anchor_scores=torch.randn(batch, horizons, layers, experts),
        semantic=SimpleNamespace(
            node_scores=parent_scores,
            factual_path_posterior=posterior,
        ),
        branch_mask=branch_mask,
        aligned=aligned,
    )
    loss = MODULE.objective(output, trajectory, stage="factual_joint")
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert route_scores.grad is not None and route_scores.grad.abs().sum() > 0


def test_runtime_candidate_builder_uses_every_visible_node() -> None:
    config = DeltaRouteConfig(
        experts=80, layers=4, horizons=4, nodes=3, router_rank=3,
        raw_width=6, metadata_width=4, latent_width=8, effect_width=4,
        transition_width=16, layer_adapter_rank=2, free_rank=2,
        attention_heads=2, exact_k=2, dropout=0.0,
    )
    trajectory = DeltaRouteTrajectory(
        config, torch.randn(4, 80, 3), torch.randn(4, 80),
        torch.ones(4, 3, dtype=torch.bool),
    )
    batch, horizons, layers, nodes, experts = 1, 4, 4, 3, 80
    scores = torch.full((batch, horizons, layers, nodes, experts), -100.0)
    for node, expert in enumerate((2, 3, 4)):
        scores[..., node, expert] = 5.0
        scores[..., node, expert + 1] = 4.0
    depth = torch.tensor([[2, 3, 4]])
    node_scores = torch.stack(
        [scores[:, depth[0, node] - 1, :, node] for node in range(nodes)], dim=1
    )
    posterior = torch.tensor(
        [[[0.2, 0.2, 0.2, 0.4]]]
    ).expand(batch, horizons, nodes + 1).clone()
    branch_mask = torch.ones(batch, horizons, nodes, dtype=torch.bool)
    anchor_scores = torch.arange(experts, dtype=torch.float32).reshape(
        1, 1, 1, experts
    ).expand(batch, horizons, layers, experts).clone()
    anchor_marginals = torch.softmax(anchor_scores, -1) * 2

    class Core:
        def semantic_marginals(self, _semantic, _scores):
            return anchor_marginals, None, None, None

    parent = SimpleNamespace(
        core=Core(), config=SimpleNamespace(candidate_width=64)
    )
    output = MODULE.BatchRouteOutput(
        queries=torch.randn(batch, nodes, layers, 3),
        scores=node_scores,
        selected_ids=torch.argsort(
            node_scores, dim=-1, descending=True, stable=True
        )[..., :2],
        affine_queries=None,
        targets={},
        counterfactual={"depth": depth, "node_mask": torch.ones(batch, nodes, dtype=torch.bool)},
        anchor_scores=anchor_scores,
        semantic=SimpleNamespace(node_scores=scores, factual_path_posterior=posterior),
        branch_mask=branch_mask,
        aligned=None,
    )
    candidates = MODULE._runtime_candidate_ids(output, parent, trajectory)
    assert candidates.shape == (batch, horizons, layers, 64)
    chosen = torch.zeros(batch, horizons, layers, experts, dtype=torch.bool)
    chosen.scatter_(-1, candidates, True)
    # These low-anchor-rank experts are retained only because every visible
    # runtime node contributes exact-set evidence.
    assert bool(chosen[..., 2:6].all())
