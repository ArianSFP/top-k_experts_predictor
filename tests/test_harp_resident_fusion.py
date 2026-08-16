import torch

from harp_rtt.shadow_expert import (
    PackedInt4ResidentExperts, target_selected_expert_outputs,
)

from harp_rtt.resident_fusion import (
    ExpertHarpRescue,
    ScalarHarpRescue,
    candidate_recall,
    candidate_union,
    cumulative_prior_omission,
    expert_availability_mask,
    hot_resident_subset,
    oracle_expert_merge_recall,
    oracle_model_switch_recall,
    paired_set_overlap,
    rescue_parameter_count,
    selected_route_availability,
)


def _marginals(ids: torch.Tensor, experts: int = 256) -> torch.Tensor:
    result = torch.zeros(*ids.shape[:-1], experts)
    result.scatter_(-1, ids, 1.0)
    return result


def test_hot25_is_stable_prefix_of_train_ordering() -> None:
    residents = tuple(torch.arange(95 - layer % 3) for layer in range(40))
    hot = hot_resident_subset(residents)
    assert len(hot) == 40
    assert all(ids.numel() == 64 for ids in hot)
    assert torch.equal(hot[17], torch.arange(64))


def test_current_route_extends_availability_without_changing_residents() -> None:
    residents = tuple(torch.arange(64) for _ in range(40))
    current = torch.stack([torch.arange(8) + 64 for _ in range(40)])
    available = expert_availability_mask(residents, current_selected_ids=current)
    assert available.shape == (40, 256)
    assert bool(available[:, :72].all())
    assert not bool(available[:, 72:].any())
    route = current[None]
    assert bool(selected_route_availability(route, available).all())


def test_cumulative_omission_excludes_current_router_miss() -> None:
    parent = torch.tensor([-1, 0, 0, 1])
    mask = torch.tensor([True, True, True, True])
    immediate = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]]
    )
    exposure = cumulative_prior_omission(parent, mask, immediate)
    assert torch.equal(exposure[0], torch.tensor([0.0, 1.0, 3.0]))
    assert torch.equal(exposure[1], torch.tensor([6.0, 10.0, 15.0]))
    assert torch.equal(exposure[2], torch.tensor([6.0, 13.0, 21.0]))
    assert torch.equal(exposure[3], torch.tensor([21.0, 22.0, 23.0]))


def test_paired_overlap_and_oracles_measure_unique_rescue() -> None:
    target = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    shadow = torch.tensor([[0, 1, 2, 3, 8, 9, 10, 11]])
    harp = torch.tensor([[0, 1, 4, 5, 12, 13, 14, 15]])
    overlap = paired_set_overlap(shadow, harp, target)
    assert torch.equal(overlap.both, torch.tensor([0.25]))
    assert torch.equal(overlap.shadow_unique, torch.tensor([0.25]))
    assert torch.equal(overlap.harp_unique, torch.tensor([0.25]))
    assert torch.equal(overlap.neither, torch.tensor([0.25]))
    assert torch.equal(
        oracle_model_switch_recall(shadow, harp, target), torch.tensor([0.5])
    )
    union = candidate_union(shadow, harp)
    assert torch.equal(candidate_recall(union, target), torch.tensor([0.75]))
    swaps = oracle_expert_merge_recall(shadow, harp, target)
    assert torch.equal(swaps[1], torch.tensor([0.625]))
    assert torch.equal(swaps[2], torch.tensor([0.75]))
    assert torch.equal(swaps[8], torch.tensor([0.75]))


def test_cold_only_oracle_cannot_claim_resident_harp_rescue() -> None:
    target = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    shadow = torch.tensor([[0, 1, 8, 9, 10, 11, 12, 13]])
    harp = torch.tensor([[2, 3, 4, 5, 14, 15, 16, 17]])
    cold = torch.tensor([[False, False, False, False, True, True, True, True]])
    result = oracle_expert_merge_recall(
        shadow, harp, target, cold_expert_mask=cold
    )
    assert torch.equal(result[8], torch.tensor([0.5]))


def test_scalar_rescue_is_exact_parent_at_zero_and_mass_preserving() -> None:
    shadow_ids = torch.arange(8).reshape(1, 1, 1, 8).expand(1, 4, 40, 8)
    harp_ids = (torch.arange(8) + 8).reshape(1, 1, 1, 8).expand(1, 4, 40, 8)
    shadow = _marginals(shadow_ids)
    harp = _marginals(harp_ids)
    captured_probability = torch.full((1, 4), 0.75)
    captured = shadow * captured_probability[..., None, None]
    baseline = captured + (1 - captured_probability)[..., None, None] * harp
    model = ScalarHarpRescue()
    epoch_zero = model(
        baseline_marginals=baseline,
        captured_shadow_marginals=captured,
        captured_probability=captured_probability,
        harp_marginals=harp,
    )
    assert torch.equal(epoch_zero, baseline)
    with torch.no_grad():
        model.raw_gate.fill_(1.0)
    opened = model(
        baseline_marginals=baseline,
        captured_shadow_marginals=captured,
        captured_probability=captured_probability,
        harp_marginals=harp,
    )
    assert torch.allclose(opened.sum(-1), torch.full((1, 4, 40), 8.0))
    assert torch.equal(opened, harp)


def test_expert_rescue_is_small_and_exact_parent_at_zero() -> None:
    shadow_ids = torch.arange(8).reshape(1, 1, 1, 8).expand(2, 4, 40, 8)
    harp_ids = (torch.arange(8) + 8).reshape(1, 1, 1, 8).expand(2, 4, 40, 8)
    shadow = _marginals(shadow_ids)
    harp = _marginals(harp_ids)
    features = torch.zeros(2, 4, 40, 256, 7)
    model = ExpertHarpRescue(feature_width=7)
    assert rescue_parameter_count(model) < 1_000_000
    output = model(
        baseline_marginals=shadow,
        harp_marginals=harp,
        features=features,
    )
    assert torch.equal(output, shadow)



class _RecordingNative(torch.nn.Module):
    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("gate_up", gate_up)
        self.register_buffer("down", down)
        self.seen: list[int] = []

    def forward(
        self, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        self.seen.extend(ids.flatten().tolist())
        values = target_selected_expert_outputs(
            hidden, ids, self.gate_up, self.down
        )
        return (values * weights[..., None]).sum(-2)


def test_runtime_resident_mask_and_opportunistic_cache_are_nonpersistent() -> None:
    torch.manual_seed(42)
    gate_up = torch.randn(5, 4, 4)
    down = torch.randn(5, 4, 2)
    module = PackedInt4ResidentExperts.from_target(
        torch.tensor([0, 1, 2]), None, gate_up, down,
        exact_k=4, group_size=2,
    )
    native = _RecordingNative(gate_up, down)
    original_state = set(module.state_dict())
    module.configure_runtime_policy(
        active_resident_ids=torch.tensor([0]),
        opportunistic_ids=torch.tensor([3]),
        native_experts=native,
    )
    hidden = torch.randn(1, 4)
    ids = torch.tensor([[0, 1, 3, 4]])
    weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    result = module.forward_components(hidden, ids, weights)
    assert torch.equal(
        result.resident_mask, torch.tensor([[True, False, False, False]])
    )
    assert torch.equal(
        result.opportunistic_mask, torch.tensor([[False, False, True, False]])
    )
    assert torch.equal(
        result.missing_mask, torch.tensor([[False, True, False, True]])
    )
    assert torch.allclose(result.missing_mass, torch.tensor([[0.4]]))
    zero_missing = module.forward_components(
        hidden, ids, torch.tensor([[0.4, 0.0, 0.2, 0.0]])
    )
    assert torch.equal(result.output, zero_missing.output)
    assert native.seen == [3, 3]
    assert set(module.state_dict()) == original_state
    module.reset_runtime_policy()
    assert torch.equal(module.runtime_active_resident_ids, torch.tensor([0, 1, 2]))
    assert module.runtime_opportunistic_ids.numel() == 0


def test_zero_rescue_gates_receive_gradients_without_changing_forward() -> None:
    shadow_ids = torch.arange(8).reshape(1, 1, 1, 8).expand(1, 4, 40, 8)
    harp_ids = (torch.arange(8) + 8).reshape(1, 1, 1, 8).expand(1, 4, 40, 8)
    shadow = _marginals(shadow_ids)
    harp = _marginals(harp_ids)
    scalar = ScalarHarpRescue()
    captured_probability = torch.ones(1, 4)
    output = scalar(
        baseline_marginals=shadow, captured_shadow_marginals=shadow,
        captured_probability=captured_probability, harp_marginals=harp,
    )
    assert torch.equal(output, shadow)
    output[..., 8].sum().backward()
    assert scalar.raw_gate.grad is not None
    assert bool((scalar.raw_gate.grad != 0).any())

    expert = ExpertHarpRescue(feature_width=3)
    features = torch.zeros(1, 4, 40, 256, 3)
    output = expert(
        baseline_marginals=shadow, harp_marginals=harp, features=features,
    )
    assert torch.equal(output, shadow)
    output[..., 8].sum().backward()
    assert expert.raw_open_gate.grad is not None
    assert bool((expert.raw_open_gate.grad != 0).any())
