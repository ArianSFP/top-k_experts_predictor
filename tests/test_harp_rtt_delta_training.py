from __future__ import annotations

import torch

from harp_rtt.delta import HARPDeltaConfig, HARPDeltaTeacher, HARPDeltaTree
from harp_rtt.delta_training import (
    candidate_loss,
    configure_delta_stage,
    ranker_loss,
    semantic_loss,
    set_delta_stage_mode,
)
from harp_rtt.exact_k import stable_topk


def model() -> HARPDeltaTree:
    config = HARPDeltaConfig(
        experts=64,
        layers=2,
        exact_k=2,
        candidate_width=64,
        max_tree_nodes=4,
        router_rank=4,
        context_input_width=8,
        root_input_width=8,
        node_input_width=8,
        tree_width=16,
        set_width=16,
        ranker_width=16,
        tree_ffn_width=32,
        ranker_ffn_width=32,
        attention_heads=4,
        tree_blocks=1,
        ranker_blocks=1,
        free_rank=4,
        position_frequencies=2,
        maximum_swaps=2,
        dropout=0.0,
    )
    return HARPDeltaTree(
        config,
        torch.zeros(config.layers, config.experts, config.router_rank),
        torch.zeros(config.layers, config.experts),
    )


def test_delta_stage_ownership_is_disjoint() -> None:
    semantic_model = model()
    semantic = configure_delta_stage(semantic_model, "semantic")
    assert semantic.trainable_names
    assert all(not name.startswith("candidates.") for name in semantic.trainable_names)
    assert all(not name.startswith("ranker.") for name in semantic.trainable_names)

    candidate_model = model()
    candidate = configure_delta_stage(candidate_model, "candidate")
    assert all(name.startswith("candidates.") for name in candidate.trainable_names)

    ranker_model = model()
    ranker = configure_delta_stage(ranker_model, "ranker")
    assert all(name.startswith("ranker.") for name in ranker.trainable_names)

    calibration_model = model()
    calibration = configure_delta_stage(calibration_model, "calibration")
    assert set(name.split(".")[1] for name in calibration.trainable_names) <= {
        "gain", "quota", "swap"
    }


def test_teacher_stage_ownership_trains_adapter_only_with_semantics() -> None:
    core = model()
    config = core.config
    teacher = HARPDeltaTeacher(
        config,
        core.set_head.expert_keys,
        core.set_head.centered_bias,
        raw_width=12,
        target_control_width=5,
        metadata_width=3,
    )
    semantic = configure_delta_stage(teacher, "semantic")
    assert any(name.startswith("adapter.") for name in semantic.trainable_names)
    assert all(not name.startswith("core.candidates.") for name in semantic.trainable_names)
    assert all(not name.startswith("core.ranker.") for name in semantic.trainable_names)
    candidate = configure_delta_stage(teacher, "candidate")
    assert all(name.startswith("core.candidates.") for name in candidate.trainable_names)
    assert all(not name.startswith("adapter.") for name in candidate.trainable_names)

    set_delta_stage_mode(teacher, "candidate")
    assert teacher.training is False
    assert teacher.adapter.training is False
    assert teacher.core.tree.training is False
    assert teacher.core.candidates.training is True
    assert teacher.core.ranker.training is False


def _forward() -> tuple[HARPDeltaTree, object, torch.Tensor]:
    torch.manual_seed(7)
    delta = model()
    config = delta.config
    batch, nodes = 2, 4
    anchor = torch.randn(batch, config.horizons, config.layers, config.experts)
    output = delta(
        anchor_scores=anchor,
        context_features=torch.randn(
            batch, config.horizons, config.layers, config.context_input_width
        ),
        root_features=torch.randn(batch, config.layers, config.root_input_width),
        node_features=torch.randn(batch, nodes, config.node_input_width),
        node_parent_ids=torch.tensor([[-1, 0, 1, 2], [-1, 0, 1, 2]]),
        node_available=torch.ones(batch, nodes, dtype=torch.bool),
        node_path_log_probabilities=torch.log(
            torch.tensor([[0.5, 0.25, 0.125, 0.0625]]).repeat(batch, 1)
        ),
        node_horizon_mask=torch.ones(
            batch, config.horizons, nodes, dtype=torch.bool
        ),
        source_positions=torch.tensor([2, 200]),
    )
    return delta, output, anchor


def test_all_delta_stage_losses_have_a_gradient_path() -> None:
    delta, output, anchor = _forward()
    labels = stable_topk(output.root_scores.detach(), delta.config.exact_k)
    labels = labels[:, None].expand(-1, delta.config.horizons, -1, -1).clone()
    targets = {
        "future_selected_ids": labels,
        "future_available": torch.ones(
            labels.shape[:-1], dtype=torch.bool
        ),
        "factual_branch_index": torch.zeros(
            labels.shape[:2], dtype=torch.long
        ),
        "future_query_coordinates": output.root_queries.detach(),
    }
    counterfactual = {
        "selected_ids": stable_topk(
            output.node_scores.detach().permute(0, 3, 1, 2, 4)[:, :, 0],
            delta.config.exact_k,
        ),
        "valid": torch.ones(
            labels.shape[0], 4, delta.config.layers, dtype=torch.bool
        ),
        "depth": torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]),
        "budget_node_masks": torch.ones(
            labels.shape[0], 4, 4, dtype=torch.bool
        ),
        "query_coordinates": output.node_queries.detach()[:, 0].permute(0, 2, 1, 3),
        "target_path_distribution": output.factual_path_posterior.detach(),
    }
    configure_delta_stage(delta, "semantic")
    semantic = semantic_loss(output, targets, counterfactual)
    semantic.total.backward(retain_graph=True)
    assert delta.set_head.free_basis.grad is not None

    delta.zero_grad(set_to_none=True)
    configure_delta_stage(delta, "candidate")
    candidate = candidate_loss(delta, output, anchor, targets)
    candidate.total.backward(retain_graph=True)
    assert delta.candidates.gain.weight.grad is not None

    delta.zero_grad(set_to_none=True)
    configure_delta_stage(delta, "ranker")
    ranked = ranker_loss(output, anchor, targets)
    ranked.total.backward()
    assert delta.ranker.correction.weight.grad is not None
