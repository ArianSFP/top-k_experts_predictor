from __future__ import annotations

import pytest
import torch

from harp_rtt.model.candidates import CandidateUnion
from harp_rtt.model.config import HARPRTTConfig
from harp_rtt.model.tree import AdaptiveTreeEncoder, ancestor_closed_visibility
from runpod.transformers_mtp_bridge.hydrate_selected_train_sequence_starts import segment_name


def tiny_tree_config() -> HARPRTTConfig:
    return HARPRTTConfig(
        experts=12,
        layers=2,
        exact_k=2,
        candidate_width=6,
        max_tree_nodes=4,
        max_tree_depth=4,
        router_rank=3,
        tree_hidden_width=6,
        tree_fused_width=6,
        tree_router_input_width=6,
        tree_token_width=6,
        tree_metadata_width=8,
        tree_width=16,
        tree_ffn_width=32,
        attention_heads=4,
        tree_blocks=1,
        dropout=0.0,
    )


def test_formal_adaptive_depth_is_exactly_four() -> None:
    tiny_tree_config().validate()
    with pytest.raises(ValueError, match="exactly H1--H4"):
        HARPRTTConfig(max_tree_depth=32).validate()


def test_anytime_visibility_is_ancestor_closed_and_rejects_late_parent() -> None:
    parents = torch.tensor([[-1, 0, 1, 2, 0]])
    available = torch.ones_like(parents, dtype=torch.bool)
    visible = ancestor_closed_visibility(parents, available, 4)
    assert visible.tolist() == [[True, True, True, True, False]]
    bad = torch.tensor([[-1, 4, 1, 2, 0]])
    with pytest.raises(ValueError, match="ancestor closed"):
        ancestor_closed_visibility(bad, available, 4)


@torch.no_grad()
def test_vocab_pool_is_order_invariant_and_branch_hash_ids_are_not_semantic() -> None:
    torch.manual_seed(91)
    config = tiny_tree_config()
    encoder = AdaptiveTreeEncoder(config).eval()
    batch, nodes, candidates = 1, 4, 5
    common = {
        "hidden": torch.randn(batch, nodes, 6),
        "fused": torch.randn(batch, nodes, 6),
        "router_input": torch.randn(batch, nodes, 6),
        "router_logits": torch.randn(batch, nodes, 12),
        "token_embeddings": torch.randn(batch, nodes, 6),
        "metadata": torch.zeros(batch, nodes, 8),
        "depth_ids": torch.tensor([[1, 2, 3, 4]]),
        "parent_ids": torch.tensor([[-1, 0, 1, 2]]),
        "available": torch.ones(batch, nodes, dtype=torch.bool),
    }
    vocab = torch.randn(batch, nodes, candidates, 6)
    logp = torch.log_softmax(torch.randn(batch, nodes, candidates), dim=-1)
    first = encoder(
        **common,
        branch_ids=torch.tensor([[1, 2, 3, 4]]),
        vocab_token_embeddings=vocab,
        vocab_log_probabilities=logp,
    )
    permutation = torch.tensor([3, 1, 4, 0, 2])
    second = encoder(
        **common,
        branch_ids=torch.tensor([[8, 7, 6, 5]]),
        vocab_token_embeddings=vocab[:, :, permutation],
        vocab_log_probabilities=logp[:, :, permutation],
    )
    assert torch.allclose(first.states, second.states, atol=2e-6)
    assert torch.allclose(first.posterior_logits, second.posterior_logits, atol=2e-6)


def test_candidate_curriculum_preserves_anchor_then_opens_twenty_four_slots() -> None:
    config = HARPRTTConfig(experts=80, layers=1, candidate_width=64)
    union = CandidateUnion(config)
    anchor = torch.arange(80, dtype=torch.float32).view(1, 1, 1, 80)
    branch = torch.zeros_like(anchor)
    branch[..., :24] = torch.arange(100, 76, -1, dtype=torch.float32)
    sources = [anchor, branch] + [torch.randn_like(anchor) for _ in range(4)]
    active = [True, True, False, False, False, False]

    initial = union(sources, training_progress=0.0, active_sources=active)
    expected_anchor = torch.argsort(
        anchor, dim=-1, descending=True, stable=True
    )[..., :64]
    assert initial.anchor_quota == 64
    assert torch.equal(initial.expert_ids, expected_anchor)

    opened = union(sources, training_progress=1.0, active_sources=active)
    assert opened.anchor_quota == 40
    assert set(opened.expert_ids.flatten().tolist()) == (
        set(range(40, 80)) | set(range(24))
    )

    union.eval()
    with pytest.raises(ValueError, match="training-only"):
        union(
            sources,
            training_progress=1.0,
            active_sources=active,
            anchor_quota_override=32,
        )

def test_capture_segment_mapping_respects_one_based_boundaries() -> None:
    assert segment_name("tfprod-req000000-1234abcd").endswith("000000_000000")
    assert segment_name("tfprod-req000016-1234abcd").endswith("000001_000016")
    assert segment_name("tfprod-req000017-1234abcd").endswith("000017_000032")
    assert segment_name("tfprod-req000425-1234abcd").endswith("000417_000432")
    assert segment_name("tfprod-req000608-1234abcd").endswith("000593_000608")
