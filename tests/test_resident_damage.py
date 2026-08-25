import pytest
import torch

from harp_rtt.resident_damage import (
    allocate_core_locked_utility,
    frequency_core_ids,
    omission_damage_per_slot,
    reduce_expert_damage,
    router_logits_for_routed_variants,
    validate_core_inclusion,
)


def _synthetic_next_router() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    routed = torch.tensor([[[1.0, 0.4]]])
    states = {
        "post_attention_residual_u": torch.zeros_like(routed),
        "post_moe_residual_xplus": routed.clone(),
        "shared_expert_output_delta_s": torch.zeros_like(routed),
        "next_post_attention_residual_u": routed.clone(),
        "next_selected_expert_ids": torch.tensor([[[0]]]),
    }
    norm_weight = torch.zeros(2)
    router_weight = torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
    states["next_raw_target_router_logits"] = router_logits_for_routed_variants(
        routed.unsqueeze(-2),
        states,
        norm_weight=norm_weight,
        router_weight=router_weight,
    )[..., 0, :]
    return states, norm_weight, router_weight


def test_exact_omission_damage_detects_router_recall_loss() -> None:
    states, norm_weight, router_weight = _synthetic_next_router()
    routed = states["post_moe_residual_xplus"]
    damage = omission_damage_per_slot(
        routed,
        torch.tensor([[[[1.0, 0.0]]]]),
        torch.ones(1, 1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        states,
        norm_weight=norm_weight,
        router_weight=router_weight,
        include_exact_objective=True,
    )
    assert damage["slot_recall_damage"].item() == 1.0
    assert damage["router_logit_mse"].item() > 0.0
    assert damage["objective_positive_damage"].item() > 0.0


def test_zero_contribution_and_invalid_rows_have_zero_damage() -> None:
    states, norm_weight, router_weight = _synthetic_next_router()
    routed = states["post_moe_residual_xplus"]
    zero = omission_damage_per_slot(
        routed,
        torch.zeros(1, 1, 1, 2),
        torch.ones(1, 1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        states,
        norm_weight=norm_weight,
        router_weight=router_weight,
    )
    assert all(torch.equal(value, torch.zeros_like(value)) for value in zero.values())

    invalid = omission_damage_per_slot(
        routed,
        torch.tensor([[[[1.0, 0.0]]]]),
        torch.ones(1, 1, 1),
        torch.zeros(1, 1, dtype=torch.bool),
        states,
        norm_weight=norm_weight,
        router_weight=router_weight,
    )
    assert all(torch.equal(value, torch.zeros_like(value)) for value in invalid.values())


def test_reduce_expert_damage_scatter_adds_only_valid_slots() -> None:
    selected = torch.tensor([[[0, 1], [1, 2]]])
    weights = torch.tensor([[[0.7, 0.3], [0.4, 0.6]]])
    valid = torch.tensor([[True, False]])
    values = torch.tensor([[[2.0, 3.0], [100.0, 100.0]]])
    reduced = reduce_expert_damage(
        selected, weights, valid, {"damage": values}, experts=4
    )
    assert reduced["occurrences"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert reduced["selected_weight_mass"].tolist() == pytest.approx(
        [0.7, 0.3, 0.0, 0.0]
    )
    assert reduced["damage"].tolist() == [2.0, 3.0, 0.0, 0.0]


def test_frequency_core_is_stable_and_validator_is_fail_closed() -> None:
    counts = torch.tensor([[4, 4, 3, 2], [1, 5, 5, 0]])
    core, digest = frequency_core_ids(counts, core_size=2)
    assert core.tolist() == [[0, 1], [1, 2]]
    assert len(digest) == 64
    assert frequency_core_ids(counts, core_size=2)[1] == digest
    validate_core_inclusion([[0, 1, 3], [1, 2]], core)
    with pytest.raises(ValueError, match="mandatory frequency-core"):
        validate_core_inclusion([[0, 3], [1, 2]], core)


def test_core_locked_allocator_meets_cells_core_and_hit_cap() -> None:
    counts = torch.tensor(
        [[10, 9, 8, 1, 0, 0], [10, 9, 8, 1, 0, 0]], dtype=torch.int64
    )
    utility = torch.tensor(
        [[10.0, 9.0, 100.0, 2.0, 1.0, 0.0], [10.0, 9.0, 100.0, 2.0, 1.0, 0.0]]
    )
    allocation = allocate_core_locked_utility(
        utility,
        counts,
        total_residents=6,
        core_size=2,
        maximum_per_layer=4,
        hit_count_cap=40,
    )
    core, core_hash = frequency_core_ids(counts, core_size=2)
    validate_core_inclusion(allocation.resident_expert_ids_by_layer, core)
    assert sum(allocation.resident_counts_by_layer) == 6
    assert max(allocation.resident_counts_by_layer) <= 4
    assert allocation.resident_hit_count <= 40
    assert allocation.frequency_core_sha256 == core_hash
    assert allocation.optional_residents == 2
    assert len(allocation.resident_ids_sha256) == 64


def test_core_locked_allocator_rejects_infeasible_hit_cap() -> None:
    counts = torch.tensor(
        [[10, 9, 8, 1, 0, 0], [10, 9, 8, 1, 0, 0]], dtype=torch.int64
    )
    utility = torch.ones(2, 6)
    with pytest.raises(ValueError, match="infeasible"):
        allocate_core_locked_utility(
            utility,
            counts,
            total_residents=6,
            core_size=2,
            maximum_per_layer=4,
            hit_count_cap=37,
        )
