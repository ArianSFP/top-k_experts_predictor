from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt
from harp_rtt.index import (
    MTP_REQUIRED_ROLES,
    PROMPT_ROLES,
    TARGET_REQUIRED_ROLES,
    build_segment,
)


SHAPES = {
    "post_attention_residual_u": (2048,),
    "normalized_target_router_input_a": (2048,),
    "post_moe_residual_xplus": (2048,),
    "routed_expert_output_delta_r": (2048,),
    "shared_expert_output_delta_s": (2048,),
    "raw_target_router_logits": (256,),
    "selected_expert_ids": (8,),
    "selected_execution_weights": (8,),
    "mtp_fused_state": (2048,),
    "mtp_router_input": (2048,),
    "mtp_post_ffn_hidden": (2048,),
    "mtp_vocabulary_head_input": (2048,),
    "raw_mtp_router_logits": (256,),
    "mtp_selected_expert_ids": (8,),
    "mtp_selected_execution_weights": (8,),
    "vocab_top64_token_ids": (64,),
    "vocab_top64_log_probabilities": (64,),
    "harp_anchor_mtp_vocabulary_head_input": (2048,),
    "harp_anchor_raw_mtp_router_logits": (256,),
    "harp_anchor_vocab_top64_token_ids": (64,),
    "harp_anchor_vocab_top64_log_probabilities": (64,),
}
ID_ROLES = {
    "selected_expert_ids",
    "mtp_selected_expert_ids",
    "vocab_top64_token_ids",
    "harp_anchor_vocab_top64_token_ids",
}


def _prefix_hash(tokens: list[int]) -> str:
    digest = hashlib.sha256(b"GCRP2_PREFIX_V1")
    for token in tokens:
        digest.update(struct.pack("<i", int(token)))
    return digest.hexdigest()


class CaptureBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.events: list[dict[str, object]] = []
        self.next_event = 1
        self.payload = bytearray(64)

    def emit(self, event: str, **values: object) -> int:
        event_id = self.next_event
        self.next_event += 1
        self.events.append({"event": event, "event_id": event_id, **values})
        return event_id

    def tensor(self, parent: int, role: str, value: float = 0.0) -> None:
        shape = SHAPES[role]
        if role in ID_ROLES:
            if shape == (8,):
                array = np.arange(8, dtype="<i4")
            else:
                array = np.arange(64, dtype="<i4")
            dtype = "int32"
        else:
            array = np.full(shape, value, dtype="<f4")
            dtype = "float32"
        offset = len(self.payload)
        raw = array.tobytes()
        self.payload.extend(raw)
        self.emit(
            "tensor",
            parent_event_id=parent,
            tensor_role=role,
            payload_offset=offset,
            payload_bytes=len(raw),
            native_dtype=dtype,
            shape=list(shape),
            payload_file="payload.bin",
        )

    def finish(self) -> None:
        self.root.mkdir(parents=True)
        (self.root / "events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.events)
        )
        (self.root / "payload.bin").write_bytes(self.payload)


def _rich_index(
    tmp_path: Path,
    *,
    adaptive: bool = False,
    drop_adaptive_parent_at: int | None = None,
    drop_anchor_ready: bool = False,
    corrupt_anchor_parent: bool = False,
) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    segment = corpus / "segments" / "seg_000"
    capture = CaptureBuilder(segment)
    sequence = "seq-a"
    request = "req-a"
    capture.emit(
        "sequence_start",
        sequence_id=sequence,
        request_id=request,
        dataset_source="synthetic",
    )
    prompt_ids = list(range(100, 108))
    generated_ids = list(range(108, 115))

    for position, token_id in enumerate(prompt_ids):
        token_event = capture.emit(
            "target_token",
            sequence_id=sequence,
            absolute_sequence_position=position,
            committed_input_token_id=token_id,
        )
        for layer in range(40):
            parent = capture.emit(
                "prompt_route_layer",
                sequence_id=sequence,
                target_token_event_id=token_event,
                absolute_sequence_position=position,
                target_layer=layer,
            )
            for role in PROMPT_ROLES:
                capture.tensor(parent, role, value=position + layer / 100.0)

    for position, token_id in zip(range(8, 11), generated_ids[:3], strict=True):
        token_event = capture.emit(
            "target_token",
            sequence_id=sequence,
            absolute_sequence_position=position,
            committed_input_token_id=token_id,
        )
        for layer in range(40):
            parent = capture.emit(
                "target_layer",
                sequence_id=sequence,
                target_token_event_id=token_event,
                absolute_sequence_position=position,
                target_layer=layer,
                record_valid=True,
                post_mixer_pre_moe_residual_rms=1.0,
                cos_routed_output_to_pre_moe_residual=0.1,
                cos_shared_output_to_pre_moe_residual=0.2,
                shared_expert_gate_scalar=0.3,
            )
            for role in TARGET_REQUIRED_ROLES:
                capture.tensor(parent, role, value=position + layer / 100.0)

    if adaptive:
        # Greedy spine plus a second complete H2--H4 sibling path. Creation
        # order is parent-before-child but deliberately repeats depths.
        node_specs = (
            # depth, parent-local, token, local-logp, path-logp
            (1, None, 111, 0.0, 0.0),
            (2, 0, 201, -0.10, -0.10),
            (3, 1, 202, -0.20, -0.30),
            (4, 2, 203, -0.30, -0.60),
            (2, 0, 211, -0.40, -0.40),
            (3, 4, 212, -0.50, -0.90),
            (4, 5, 213, -0.60, -1.50),
        )
        full_tokens = prompt_ids + generated_ids
        source_hash = _prefix_hash(full_tokens[:11])
        event_ids: list[int] = []
        tree_id = "seq-a:tree:10"
        for local, (depth, parent_local, token, local_logp, path_logp) in enumerate(
            node_specs
        ):
            parent_event = (
                None if parent_local is None else event_ids[int(parent_local)]
            )
            if local == drop_adaptive_parent_at:
                parent_event = None
            node = capture.emit(
                "mtp_node",
                sequence_id=sequence,
                committed_prefix_position=10,
                gcrp_target_position=10 + depth,
                gcrp_horizon=depth,
                engine_draft_depth=depth,
                verifier_cycle_id=1,
                tree_id=tree_id,
                tree_node_id=f"{tree_id}:n{local}",
                node_local_index=local,
                parent_node_event_id=parent_event,
                parent_local_index=parent_local,
                parent_tree_node_id=(
                    None if parent_local is None else f"{tree_id}:n{parent_local}"
                ),
                branch_id=f"branch:{local}",
                branch_path_id=f"path:{local}",
                path_id=f"path:{local}",
                node_token_id=token,
                draft_token_id=310 + local,
                conditioning_class=(
                    "authoritative_exact_root"
                    if depth == 1
                    else "position_aligned_speculative_context"
                ),
                exact_committed_h1_root=depth == 1,
                acceptance_fields_present=False,
                authoritative_prefix_hash=source_hash,
                exact_prefix_hash=(bytes([local + 1]) * 32).hex(),
                vocabulary_prediction_position=11 + depth,
                target_position_mapping={
                    "source_t": 10,
                    "node_horizon": depth,
                    "node_target_position": 10 + depth,
                    "vocabulary_prediction_position": 11 + depth,
                },
                source_ready_event_order=10_000 + local,
                source_ready_timestamp_utc=f"2026-08-07T12:00:{local:02d}+00:00",
                source_ready_monotonic_ns=500_000 + local,
                structural_validity=True,
                feature_available=True,
                record_valid=True,
                node_local_token_log_probability=local_logp,
                node_local_token_probability=math.exp(local_logp),
                path_log_probability=path_logp,
                path_probability=math.exp(path_logp),
                branch_probability=math.exp(path_logp),
                vocabulary_entropy=1.0 + local,
                mtp_router_entropy=2.0 + local,
            )
            event_ids.append(node)
            for role in MTP_REQUIRED_ROLES:
                capture.tensor(node, role, value=float(local + 1))
            capture.emit(
                "mtp_acceptance_label",
                mtp_node_event_id=node,
                draft_token_accepted=local < 4,
                acceptance_label_valid=True,
                label_only=True,
            )

        anchor_event_ids: list[int] = []
        anchor_paths: list[int] = [111]
        anchor_spine_id = f"{tree_id}:legacy-anchor-h1-h6"
        for local in range(6):
            depth = local + 1
            if depth > 1:
                anchor_paths.append(0)
            local_logp = 0.0 if depth == 1 else -0.1 * (depth - 1)
            path_logp = -0.1 * sum(range(1, depth))
            parent_event = None if local == 0 else anchor_event_ids[-1]
            exact_hash = _prefix_hash(full_tokens[:11] + anchor_paths)
            node = capture.emit(
                "harp_anchor_spine_node",
                sequence_id=sequence,
                request_id=request,
                verifier_cycle_id=1,
                tree_id=tree_id,
                anchor_spine_id=anchor_spine_id,
                anchor_spine_schema="harp_rtt_legacy_anchor_spine_v1",
                anchor_local_index=local,
                parent_anchor_node_event_id=parent_event,
                parent_anchor_local_index=None if local == 0 else local - 1,
                committed_prefix_position=10,
                authoritative_prefix_hash=source_hash,
                exact_prefix_hash=exact_hash,
                mtp_conditioning_prefix_hash=exact_hash,
                engine_draft_depth=depth,
                gcrp_target_position=10 + depth,
                vocabulary_prediction_position=11 + depth,
                node_token_id=anchor_paths[-1],
                vocabulary_top_token_id=0,
                token_rank_under_parent=None if local == 0 else 0,
                branch_path_token_ids=list(anchor_paths),
                branch_path_token_log_probabilities=[0.0]
                + [-0.1 * value for value in range(1, depth)],
                node_local_token_log_probability=local_logp,
                node_local_token_probability=math.exp(local_logp),
                path_log_probability=path_logp,
                path_probability=math.exp(path_logp),
                source_ready_event_order=capture.next_event,
                source_ready_timestamp_utc=f"2026-08-07T13:00:{local:02d}+00:00",
                source_ready_monotonic_ns=600_000 + local,
                exact_committed_h1_root=depth == 1,
                anchor_only=True,
                adaptive_model_input=False,
                consumes_adaptive_node_budget=False,
                continues_through_eos=True,
                eos_termination_applies=False,
                required_anchor_depth=6,
                acceptance_fields_present=False,
                label_records_present=False,
                structural_validity=True,
                feature_available=True,
                record_valid=True,
            )
            anchor_event_ids.append(node)
            for role in (
                "harp_anchor_mtp_vocabulary_head_input",
                "harp_anchor_raw_mtp_router_logits",
                "harp_anchor_vocab_top64_token_ids",
                "harp_anchor_vocab_top64_log_probabilities",
            ):
                capture.tensor(node, role, value=float(depth))
        capture.emit(
            "harp_anchor_spine_ready",
            sequence_id=sequence,
            request_id=request,
            verifier_cycle_id=1,
            tree_id=tree_id,
            anchor_spine_id=anchor_spine_id,
            anchor_spine_schema="harp_rtt_legacy_anchor_spine_v1",
            committed_prefix_position=10,
            authoritative_prefix_hash=source_hash,
            exact_h1_token_id=111,
            exact_h1_root_event_id=anchor_event_ids[0],
            anchor_node_event_ids=anchor_event_ids,
            node_count=6,
            required_depth=6,
            synchronous_before_target_h1_execution=True,
            local_top1_parent_coherent=True,
            continues_through_eos=True,
            eos_termination_applies=False,
            consumes_adaptive_node_budget=False,
            adaptive_model_input=False,
            expansion_uses_realized_future=False,
            expansion_uses_acceptance=False,
            labels_emitted=False,
        )
    else:
        parent_event = None
        for depth in range(1, 5):
            node = capture.emit(
                "mtp_node",
                sequence_id=sequence,
                committed_prefix_position=10,
                gcrp_target_position=10 + depth,
                gcrp_horizon=depth,
                engine_draft_depth=depth,
                verifier_cycle_id=1,
                parent_node_event_id=parent_event,
                node_token_id=110 + depth,
                draft_token_id=210 + depth,
                conditioning_class="authoritative_context_equivalent"
                if depth == 1
                else "position_aligned_speculative_context",
                source_ready_event_order=10_000 + depth,
                branch_probability=0.8**depth,
                record_valid=True,
            )
            parent_event = node
            for role in MTP_REQUIRED_ROLES:
                capture.tensor(node, role, value=float(depth))
            capture.emit(
                "mtp_acceptance_label",
                mtp_node_event_id=node,
                draft_token_accepted=depth <= 2,
                acceptance_label_valid=True,
            )

    for position, token_id in zip(range(11, 15), generated_ids[3:], strict=True):
        token_event = capture.emit(
            "target_token",
            sequence_id=sequence,
            absolute_sequence_position=position,
            committed_input_token_id=token_id,
        )
        for layer in range(40):
            parent = capture.emit(
                "target_layer",
                sequence_id=sequence,
                target_token_event_id=token_event,
                absolute_sequence_position=position,
                target_layer=layer,
                record_valid=True,
                post_mixer_pre_moe_residual_rms=1.0,
                cos_routed_output_to_pre_moe_residual=0.1,
                cos_shared_output_to_pre_moe_residual=0.2,
                shared_expert_gate_scalar=0.3,
            )
            for role in TARGET_REQUIRED_ROLES:
                capture.tensor(parent, role, value=position + layer / 100.0)

    if drop_anchor_ready:
        capture.events = [
            row
            for row in capture.events
            if row.get("event") != "harp_anchor_spine_ready"
        ]
    if corrupt_anchor_parent:
        child = next(
            row
            for row in capture.events
            if row.get("event") == "harp_anchor_spine_node"
            and row.get("anchor_local_index") == 1
        )
        child["parent_anchor_node_event_id"] = -999

    capture.emit(
        "sequence_end",
        sequence_id=sequence,
        prompt_length=len(prompt_ids),
        generated_length=len(generated_ids),
        prompt_token_ids=prompt_ids,
        committed_generated_token_ids=generated_ids,
        termination_reason="length",
    )
    capture.finish()
    index_root = tmp_path / "index"
    build_segment(segment, index_root / "seg_000", {sequence: "train"})
    return corpus, index_root


def test_rich_dataset_shapes_and_label_isolation(tmp_path: Path) -> None:
    corpus, index_root = _rich_index(tmp_path)
    dataset = HarpRTTDataset(
        index_root,
        "train",
        corpus_root=corpus,
        max_tree_nodes=8,
        include_optional_current=False,
    )
    assert len(dataset) == 3
    item = next(item for item in dataset if item["metadata"]["position"] == 10)
    assert item["metadata"]["position"] == 10
    assert item["inputs"]["history"]["logits"].shape == (8, 40, 256)
    assert item["inputs"]["history"]["selected_ids"].shape == (8, 40, 8)
    assert item["inputs"]["exact_next_token_id"].item() == 111
    assert item["inputs"]["within_request"].item() == 2
    assert item["inputs"]["final_hidden"].shape == (2048,)
    assert item["inputs"]["tree"]["states"].shape == (8, 4, 2048)
    assert not item["inputs"]["tree"]["adaptive_contract"].item()
    assert item["inputs"]["tree"]["mask"].sum().item() == 4
    assert item["inputs"]["tree"]["parent"].tolist()[:4] == [-1, 0, 1, 2]
    assert item["inputs"]["tree"]["branch"].tolist()[:4] == [1, 1, 1, 1]
    assert "acceptance" not in item["inputs"]["tree"]
    assert item["targets"]["tree_acceptance_valid"].sum().item() == 4
    assert item["targets"]["future_router_logits"].shape == (4, 40, 256)
    assert item["targets"]["future_router_inputs"].shape == (4, 40, 2048)
    assert item["targets"]["future_selected_ids"].shape == (4, 40, 8)
    assert torch.allclose(
        item["inputs"]["history"]["logits"].mean(-1),
        torch.zeros(8, 40),
        atol=2e-6,
    )

    batch = collate_harp_rtt([item, item])
    assert batch["inputs"]["history"]["logits"].shape == (2, 8, 40, 256)
    assert batch["targets"]["future_selected_ids"].shape == (2, 4, 40, 8)


def test_test_split_is_sealed_by_default(tmp_path: Path) -> None:
    corpus, index_root = _rich_index(tmp_path)
    manifest_path = index_root / "seg_000" / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sequences"][0]["split"] = "test"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PermissionError, match="sealed"):
        HarpRTTDataset(index_root, "test", corpus_root=corpus)


def test_adaptive_tree_fields_and_siblings_survive_index_and_dataset(
    tmp_path: Path,
) -> None:
    corpus, index_root = _rich_index(tmp_path, adaptive=True)
    dataset = HarpRTTDataset(
        index_root,
        "train",
        corpus_root=corpus,
        max_tree_nodes=32,
        include_optional_current=False,
    )
    item = next(item for item in dataset if item["metadata"]["position"] == 10)
    tree = item["inputs"]["tree"]
    assert tree["adaptive_contract"].item()
    count = int(tree["mask"].sum())
    assert count == 7
    assert tree["depth"][:count].tolist() == [1, 2, 3, 4, 2, 3, 4]
    assert tree["parent"][:count].tolist() == [-1, 0, 1, 2, 0, 4, 5]
    assert tree["branch"][:count].tolist() == list(range(1, 8))
    assert torch.unique(tree["path_ids"][:count]).numel() == count
    assert tree["target_positions"][:count].tolist() == [1, 2, 3, 4, 2, 3, 4]
    assert tree["absolute_target_positions"][:count].tolist() == [11, 12, 13, 14, 12, 13, 14]
    assert tree["vocabulary_prediction_positions"][:count].tolist() == [12, 13, 14, 15, 13, 14, 15]
    assert tree["token_ids"][:count].tolist() == [111, 201, 202, 203, 211, 212, 213]
    assert torch.equal(
        tree["conditioning_token_ids"][:count], tree["token_ids"][:count]
    )
    assert tree["router_logits"].shape == (32, 256)
    assert tree["vocab_top64_ids"].shape == (32, 64)
    assert tree["vocab_top64_log_probabilities"].shape == (32, 64)
    assert tree["vocab_statistics"].shape == (32, 6)
    assert tree["vocab_statistics"][:count, 2].tolist() == pytest.approx(
        [1.0 + local for local in range(count)]
    )
    assert tree["capture_scalars"].shape == (32, 9)
    assert tree["path_log_probabilities"][:count].tolist() == pytest.approx(
        [0.0, -0.1, -0.3, -0.6, -0.4, -0.9, -1.5]
    )
    assert tree["local_probabilities"][:count].tolist() == pytest.approx(
        [1.0, math.exp(-0.1), math.exp(-0.2), math.exp(-0.3), math.exp(-0.4), math.exp(-0.5), math.exp(-0.6)]
    )
    assert tree["source_ready_event_order"][:count].tolist() == list(
        range(10_000, 10_007)
    )
    assert tree["source_ready_monotonic_ns"][:count].tolist() == list(
        range(500_000, 500_007)
    )
    assert tree["source_ready"][:count].tolist() == pytest.approx(
        [index / 6 for index in range(7)]
    )
    assert tree["structural_valid"][:count].all()
    assert tree["feature_available"][:count].all()
    assert tree["conditioning_classes"][:count].tolist() == [2, 1, 1, 1, 1, 1, 1]
    assert tree["exact_committed_h1_root"][:count].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert tree["exact_root_match"][:count].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    for local in range(count):
        assert tree["exact_prefix_hashes"][local].tolist() == [local + 1] * 32

    # Raw wall-clock timestamps remain in the immutable index for provenance;
    # only normalized causal readiness enters the tensor model contract.
    segment = dataset.segments[0]
    assert segment.adaptive_mtp_tree
    assert segment.mtp_source_ready_timestamp_utc[:count].tolist() == [
        f"2026-08-07T12:00:{local:02d}+00:00" for local in range(count)
    ]
    assert "acceptance" not in tree
    assert "tree_acceptance" not in item["inputs"]
    assert item["targets"]["tree_acceptance_valid"][:count].all()


def test_adaptive_child_without_explicit_parent_never_uses_legacy_fallback(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires an explicit parent"):
        _rich_index(tmp_path, adaptive=True, drop_adaptive_parent_at=1)


def test_adaptive_index_rejects_missing_anchor_readiness(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="anchor-spine tree mismatch"):
        _rich_index(tmp_path, adaptive=True, drop_anchor_ready=True)


def test_adaptive_index_rejects_incoherent_anchor_parent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parent-coherent local-top1"):
        _rich_index(tmp_path, adaptive=True, corrupt_anchor_parent=True)
