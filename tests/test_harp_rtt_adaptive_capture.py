from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch


BRIDGE = (
    Path(__file__).resolve().parents[1] / "runpod" / "transformers_mtp_bridge"
)
if str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))

from adaptive_mtp_tree import (  # noqa: E402
    AnchorSpineNode,
    AdaptiveExpansionPolicy,
    AdaptiveMTPTreeBuilder,
    FixedBeamPolicy,
    NodeObservation,
    build_anchor_spine,
    build_acceptance_labels,
    build_fixed_beam_tree,
)
from adaptive_capture_contract import (  # noqa: E402
    assert_causal_mtp_node_record,
    assert_prompt_capture_allowed,
)
from native_mtp_branch import (  # noqa: E402
    NativeMTPBranchRunner,
    stable_topk_log_probabilities,
)
from mtp_router_semantics import native_topk_router_distribution  # noqa: E402


def observation(path: tuple[int, ...], *, uncertain: bool) -> NodeObservation:
    offset = (sum(path) * 29 + len(path) * 13) % 10_000
    ids = tuple((offset + 101 * rank) % 32_000 for rank in range(8))
    if uncertain:
        logps = tuple(-1.55 - 0.2 * rank for rank in range(8))
        entropy = 9.4
    else:
        logps = (-0.05, -3.2, -4.1, -4.8, -5.3, -5.8, -6.2, -6.7)
        entropy = 0.9
    return NodeObservation(ids, logps, 32_000, entropy)


def structural_rows(nodes):
    return [
        (
            node.local_index,
            node.parent_local_index,
            node.depth,
            node.token_id,
            node.token_rank_under_parent,
            node.token_path_ids,
            node.path_log_probability,
            node.path_id,
        )
        for node in nodes
    ]


def test_uncertain_policy_is_deterministic_budgeted_and_parent_ordered() -> None:
    policy = AdaptiveExpansionPolicy(max_nodes=32)
    builder = AdaptiveMTPTreeBuilder(policy)
    first = builder.build(
        tree_id="seq:tree0",
        exact_h1_token_id=77,
        evaluate=lambda path: observation(path, uncertain=True),
    )
    second = builder.build(
        tree_id="seq:tree0",
        exact_h1_token_id=77,
        evaluate=lambda path: observation(path, uncertain=True),
    )
    assert structural_rows(first) == structural_rows(second)
    assert len(first) == 32
    assert first[0].token_id == 77
    assert first[0].depth == 1
    assert first[0].parent_local_index is None
    counts = {depth: sum(node.depth == depth for node in first) for depth in range(1, 5)}
    assert counts[1] == 1
    assert counts[2] > 1  # uncertain context is wide
    assert counts[4] > 1  # the depth bonus preserves material H4 coverage
    for node in first[1:]:
        assert node.parent_local_index is not None
        assert node.parent_local_index < node.local_index
        parent = first[node.parent_local_index]
        assert node.depth == parent.depth + 1
        assert node.token_path_ids[:-1] == parent.token_path_ids
        assert node.path_log_probability == pytest.approx(
            parent.path_log_probability + node.local_token_log_probability
        )


def test_adaptive_16_is_canonical_prefix_of_adaptive_32() -> None:
    build = AdaptiveMTPTreeBuilder
    full = build(AdaptiveExpansionPolicy(max_nodes=32)).build(
        tree_id="seq:prefix",
        exact_h1_token_id=77,
        evaluate=lambda path: observation(path, uncertain=True),
    )
    independent = build(AdaptiveExpansionPolicy(max_nodes=16)).build(
        tree_id="seq:prefix",
        exact_h1_token_id=77,
        evaluate=lambda path: observation(path, uncertain=True),
    )
    assert structural_rows(full[:16]) == structural_rows(independent)


@pytest.mark.parametrize(
    ("widths", "budget"),
    [((5, 5, 5), 16), ((11, 10, 10), 32)],
)
def test_fixed_beam_controls_are_exact_budgeted_and_deterministic(
    widths: tuple[int, int, int], budget: int
) -> None:
    def wide_observation(path: tuple[int, ...]) -> NodeObservation:
        offset = (sum(path) * 29 + len(path) * 13) % 10_000
        ids = tuple((offset + 101 * rank) % 32_000 for rank in range(16))
        return NodeObservation(
            ids,
            tuple(-1.55 - 0.2 * rank for rank in range(16)),
            32_000,
            9.4,
        )

    policy = FixedBeamPolicy(widths)
    first = build_fixed_beam_tree(
        tree_id=f"seq:fixed{budget}",
        exact_h1_token_id=77,
        evaluate=wide_observation,
        policy=policy,
    )
    second = build_fixed_beam_tree(
        tree_id=f"seq:fixed{budget}",
        exact_h1_token_id=77,
        evaluate=wide_observation,
        policy=policy,
    )
    assert structural_rows(first) == structural_rows(second)
    assert len(first) == budget
    assert [sum(node.depth == depth for node in first) for depth in range(1, 5)] == [
        1,
        *widths,
    ]
    assert all(
        node.parent_local_index is None or node.parent_local_index < node.local_index
        for node in first
    )


def test_confident_policy_is_deep_and_narrow() -> None:
    nodes = AdaptiveMTPTreeBuilder(AdaptiveExpansionPolicy(max_nodes=32)).build(
        tree_id="seq:confident",
        exact_h1_token_id=9,
        evaluate=lambda path: observation(path, uncertain=False),
    )
    assert len(nodes) == 4
    assert [node.depth for node in nodes] == [1, 2, 3, 4]
    assert all(node.token_rank_under_parent == 0 for node in nodes)


def test_anchor_spine_is_exact_h1_through_h6_parent_coherent_top1() -> None:
    """The legacy anchor spine is separate from the bounded H1--H4 tree.

    Token ID two deliberately occupies the first local top-1 edge.  It is a
    common EOS ID, but this pinned preprocessing channel has no EOS early-stop:
    every one of its six historical depths must be captured.
    """

    next_tokens = (2, 303, 404, 505, 606, 707)
    evaluated_paths: list[tuple[int, ...]] = []
    callback_payloads: list[object] = []

    def evaluate(path: tuple[int, ...]) -> NodeObservation:
        evaluated_paths.append(path)
        depth = len(path)
        return NodeObservation(
            top_token_ids=(next_tokens[depth - 1], 30_000 + depth),
            top_log_probabilities=(-0.1 * depth, -3.0 - depth),
            vocabulary_size=32_000,
            vocabulary_entropy=0.5 + depth,
            payload={"depth": depth},
        )

    nodes = build_anchor_spine(
        tree_id="seq:anchor0",
        exact_h1_token_id=77,
        evaluate=evaluate,
        on_node=lambda node: callback_payloads.append(node.observation.payload),
        required_depth=6,
    )

    expected_paths = [
        (77,),
        (77, 2),
        (77, 2, 303),
        (77, 2, 303, 404),
        (77, 2, 303, 404, 505),
        (77, 2, 303, 404, 505, 606),
    ]
    assert evaluated_paths == expected_paths
    assert len(nodes) == 6
    assert all(isinstance(node, AnchorSpineNode) for node in nodes)
    assert [node.depth for node in nodes] == list(range(1, 7))
    assert [node.local_index for node in nodes] == list(range(6))
    assert [node.parent_local_index for node in nodes] == [None, 0, 1, 2, 3, 4]
    assert [node.token_id for node in nodes] == [77, 2, 303, 404, 505, 606]
    assert [node.token_rank_under_parent for node in nodes] == [None, 0, 0, 0, 0, 0]
    assert [node.token_path_ids for node in nodes] == expected_paths
    assert callback_payloads == [{"depth": depth} for depth in range(1, 7)]
    assert all(node.observation.payload is None for node in nodes)

    for node in nodes[1:]:
        assert node.parent_local_index is not None
        parent = nodes[node.parent_local_index]
        assert node.token_id == parent.observation.top_token_ids[0]
        assert node.local_token_log_probability == pytest.approx(
            parent.observation.top_log_probabilities[0]
        )
        assert node.path_log_probability == pytest.approx(
            parent.path_log_probability + node.local_token_log_probability
        )


def test_acceptance_is_derived_later_and_never_enters_observation() -> None:
    nodes = AdaptiveMTPTreeBuilder(AdaptiveExpansionPolicy(max_nodes=4)).build(
        tree_id="seq:labels",
        exact_h1_token_id=7,
        evaluate=lambda path: NodeObservation(
            top_token_ids=(8, 18),
            top_log_probabilities=(-0.1, -2.4),
            vocabulary_size=128,
            vocabulary_entropy=0.4,
        ),
    )
    # Prefix occupies positions 0--1.  H1/H2 match; H3 and therefore H4 do not.
    committed = [100, 101, 7, 8, 99, 8]
    labels = build_acceptance_labels(
        nodes,
        committed_token_ids=committed,
        committed_prefix_position=1,
    )
    assert [row["branch_path_accepted"] for row in labels] == [True, True, False, False]
    assert all(row["label_only"] for row in labels)
    assert all(node.observation.payload is None for node in nodes)


def test_causal_node_guard_rejects_acceptance_and_realized_future() -> None:
    valid = {
        "authoritative_prefix_hash": "abc",
        "engine_draft_depth": 1,
        "exact_committed_h1_root": True,
        "acceptance_fields_present": False,
    }
    assert_causal_mtp_node_record(valid)
    with pytest.raises(ValueError, match="acceptance leaked"):
        assert_causal_mtp_node_record({**valid, "draft_token_accepted": True})
    with pytest.raises(ValueError, match="realized future leaked"):
        assert_causal_mtp_node_record({**valid, "realized_h2_token_id": 17})
    with pytest.raises(ValueError, match="realized future leaked"):
        assert_causal_mtp_node_record(
            {**valid, "nested": {"future_target_router_input": [1.0]}}
        )
    with pytest.raises(ValueError, match="acceptance absent"):
        assert_causal_mtp_node_record({**valid, "acceptance_fields_present": True})


def test_sealed_or_external_prompt_rows_fail_closed() -> None:
    assert_prompt_capture_allowed({"original_split": "train"})
    with pytest.raises(PermissionError, match="sealed split"):
        assert_prompt_capture_allowed({"original_split": "test"})
    with pytest.raises(PermissionError, match="external-evaluation"):
        assert_prompt_capture_allowed({"external_evaluation": True})


class FakeMTP:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        input_ids: torch.Tensor,
        previous_hidden: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        past_key_values,
        use_cache: bool,
    ):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "previous_hidden": previous_hidden.clone(),
                "position_ids": position_ids.clone(),
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )
        batch, length = input_ids.shape
        hidden = 4
        experts = 256
        vocab = 64
        token = input_ids.float().unsqueeze(-1).expand(batch, length, hidden)
        router_logits = torch.arange(experts, dtype=torch.bfloat16).view(1, 1, -1)
        router_logits = router_logits.expand(batch, length, -1) / 100.0
        router_probs = torch.softmax(router_logits, dtype=torch.float, dim=-1)
        top_weights, top_ids = torch.topk(router_probs, 8, dim=-1)
        top_weights = (
            top_weights / top_weights.sum(dim=-1, keepdim=True)
        ).to(router_logits.dtype)
        vocab_logits = torch.arange(vocab, dtype=torch.float32).view(1, 1, -1)
        vocab_logits = vocab_logits.expand(batch, length, -1) / 10.0
        return {
            "token_embedding": token,
            "normalized_embedding": token + 1,
            "normalized_previous_hidden": previous_hidden + 2,
            "fused_hidden": token + 3,
            "router_input": token + 4,
            "router_logits": router_logits,
            "router_probabilities": router_probs,
            "top8_ids": top_ids,
            "top8_weights": top_weights,
            "post_ffn_hidden": token + 5,
            "head_input": token + 6,
            "vocabulary_logits": vocab_logits,
            "past_key_values": object(),
        }


def test_native_branch_runner_recomputes_isolated_prefix_without_future_state() -> None:
    fake = FakeMTP()
    runner = NativeMTPBranchRunner(
        mtp=fake,
        authoritative_prefix_token_ids=[10, 11],
        authoritative_target_hidden_through_t=[torch.zeros(4), torch.ones(4)],
        exact_h1_token_id=7,
    )
    root = runner.evaluate((7,))
    child_token = root.top_token_ids[0]
    runner.evaluate((7, child_token))
    assert fake.calls[0]["input_ids"].tolist() == [[11, 7]]
    assert fake.calls[1]["input_ids"].tolist() == [[11, 7, child_token]]
    assert tuple(fake.calls[0]["previous_hidden"].shape) == (1, 2, 4)
    assert tuple(fake.calls[1]["previous_hidden"].shape) == (1, 3, 4)
    assert all(call["past_key_values"] is None for call in fake.calls)
    assert all(call["use_cache"] is False for call in fake.calls)
    with pytest.raises(ValueError, match="exact committed"):
        runner.evaluate((8,))


def test_native_anchor_runner_reaches_h6_while_adaptive_evaluate_stays_h4() -> None:
    fake = FakeMTP()
    runner = NativeMTPBranchRunner(
        mtp=fake,
        authoritative_prefix_token_ids=[10, 11],
        authoritative_target_hidden_through_t=[torch.zeros(4), torch.ones(4)],
        exact_h1_token_id=7,
    )

    path = (7,)
    for _depth in range(1, 7):
        observation_at_depth = runner.evaluate_anchor(path)
        if len(path) < 6:
            path = path + (observation_at_depth.top_token_ids[0],)

    assert len(fake.calls) == 6
    assert fake.calls[-1]["input_ids"].tolist() == [[11, 7, 63, 63, 63, 63, 63]]
    assert tuple(fake.calls[-1]["previous_hidden"].shape) == (1, 7, 4)
    assert all(call["past_key_values"] is None for call in fake.calls)
    assert all(call["use_cache"] is False for call in fake.calls)

    with pytest.raises(ValueError, match="H4"):
        runner.evaluate((7, 63, 63, 63, 63))


def test_native_vocabulary_topk_has_deterministic_token_id_ties() -> None:
    values = torch.zeros(70)
    top_values, top_ids = stable_topk_log_probabilities(values, 64)
    assert top_values.tolist() == [0.0] * 64
    assert top_ids.tolist() == list(range(64))


def test_native_router_execution_weights_match_transformers_bf16_semantics() -> None:
    logits = torch.tensor(
        [[[0.25, -0.5, 1.125, 0.75, -1.0, 0.0, 0.5, 0.375, -0.25]]],
        dtype=torch.bfloat16,
    )
    probabilities, weights, ids = native_topk_router_distribution(logits, 4)

    reference_probabilities = torch.nn.functional.softmax(
        logits, dtype=torch.float, dim=-1
    )
    reference_values, reference_ids = torch.topk(
        reference_probabilities, 4, dim=-1
    )
    reference_weights = (
        reference_values / reference_values.sum(dim=-1, keepdim=True)
    ).to(logits.dtype)

    assert probabilities.dtype == torch.float32
    assert weights.dtype == logits.dtype == torch.bfloat16
    assert torch.equal(probabilities, reference_probabilities)
    assert torch.equal(ids, reference_ids)
    assert torch.equal(weights, reference_weights)
