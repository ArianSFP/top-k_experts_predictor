#!/usr/bin/env python3
"""Immutable HARP-RTT native-MTP adaptive-tree capture (schema v1).

This is a capture-only driver.  For each authoritative source position ``t`` it
selects the exact greedy target token ``x[t+1]`` before executing target token
``t+1``, roots a native checkpoint-MTP tree on that token, and expands causal
H2--H4 alternatives with the frozen policy in :mod:`adaptive_mtp_tree`.

Branches never share a mutable KV cache.  Each path is evaluated by an isolated
full-prefix MTP call using target final-hidden rows only through ``t`` plus the
already-produced MTP hidden rows along that speculative path.  This is slower
than a runtime tree kernel, but it gives true branch semantics with the current
Transformers API and cannot contaminate siblings through cache mutation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adaptive_mtp_tree import (  # noqa: E402
    AdaptiveExpansionPolicy,
    AdaptiveMTPTreeBuilder,
    AdaptiveTreeNode,
    AnchorSpineNode,
    FixedBeamPolicy,
    build_anchor_spine,
    build_acceptance_labels,
    build_fixed_beam_tree,
)
from adaptive_capture_contract import (  # noqa: E402
    ANCHOR_SPINE_NODE_EVENT,
    ANCHOR_SPINE_READY_EVENT,
    ANCHOR_SPINE_REQUIRED_DEPTH,
    ANCHOR_SPINE_ROLES,
    ANCHOR_SPINE_SCHEMA,
    CAPTURE_PROFILE,
    FORMAT_VERSION,
    MANIFEST_SCHEMA,
    MAX_CAPTURE_DEPTH,
    SCHEMA,
    assert_causal_anchor_spine_node_record,
    assert_causal_mtp_node_record,
    assert_prompt_capture_allowed,
)
from capture_transformers_segment import (  # noqa: E402
    CaptureRun,
    DEFAULT_PROMPTS,
    Sidecar,
    TargetCaptureHooks,
    build_manifest as build_chain_manifest,
    canonical_json,
    load_target,
    prefix_hash,
    prompt_ids,
    sha256_bytes,
    sha256_file,
)
from qwen35_mtp import load_checkpoint_mtp  # noqa: E402
from native_mtp_branch import NativeMTPBranchRunner  # noqa: E402
from matched_tree_controls import (  # noqa: E402
    CONTROL_NAMES,
    CONTROL_READY_EVENT,
    CONTROL_RESOLVED_EVENT,
    CONTROL_SCHEMA,
    assert_identical_structure,
    greedy_spine_prefix,
    assert_ready_record,
    assert_resolved_record,
    resolved_labels,
    serialize_nodes,
    structural_hash,
)


class AdaptiveCaptureRun(CaptureRun):
    def __init__(
        self,
        output: Path,
        run_id: str,
        model_path: Path,
        *,
        policy: AdaptiveExpansionPolicy,
        full_vocab_audit_percent: int,
        candidate_vector_audit_percent: int,
        prompt_history_tokens: int,
        production_capture: bool,
        emit_matched_controls: bool,
    ) -> None:
        super().__init__(
            output,
            run_id,
            model_path,
            MAX_CAPTURE_DEPTH,
            full_vocab_audit_percent,
            candidate_vector_audit_percent,
            prompt_history_tokens,
            production_capture,
            event_schema=SCHEMA,
            event_format_version=FORMAT_VERSION,
            capture_profile=CAPTURE_PROFILE,
        )
        self.policy = policy
        self.emit_matched_controls = bool(emit_matched_controls)
        self.adaptive_tree_count = 0
        self.adaptive_source_positions = 0
        self.native_mtp_calls = 0
        self.anchor_native_mtp_calls = 0
        self.anchor_spine_count = 0
        self.anchor_spine_nodes = 0
        self.control_tree_count: Counter[str] = Counter()
        self.control_node_count: Counter[str] = Counter()
        self.control_native_mtp_calls: Counter[str] = Counter()
        self.nodes_by_depth: Counter[int] = Counter()
        sidecars = output / "sidecars"
        self.anchor_states = Sidecar(
            sidecars / "harp_anchor_states.bin", b"HARPANCHSTATE1", self.events
        )
        self.anchor_routes = Sidecar(
            sidecars / "harp_anchor_routes.bin", b"HARPANCHROUTE1", self.events
        )
        self.anchor_vocab = Sidecar(
            sidecars / "harp_anchor_vocab.bin", b"HARPANCHVOCAB1", self.events
        )

    def close(self) -> None:
        for sidecar in (self.anchor_states, self.anchor_routes, self.anchor_vocab):
            sidecar.close()
        super().close()

    def write_anchor_spine_node(
        self,
        node: AnchorSpineNode,
        *,
        sequence_id: str,
        request_id: str,
        cycle_id: int,
        committed_prefix_position: int,
        authoritative_prefix_token_ids: list[int],
        parent_event_ids: dict[int, int],
    ) -> int:
        """Persist one label-free node of the separate incumbent anchor spine."""

        payload = node.observation.payload
        if not isinstance(payload, dict):
            raise RuntimeError("anchor spine payload was cleared before persistence")
        parent_event_id = (
            None
            if node.parent_local_index is None
            else parent_event_ids[node.parent_local_index]
        )
        router_position = committed_prefix_position + node.depth
        vocabulary_position = router_position + 1
        conditioning_hash = prefix_hash(
            authoritative_prefix_token_ids + list(node.token_path_ids)
        )
        fields = dict(
            run_id=self.run_id,
            sequence_id=sequence_id,
            request_id=request_id,
            verifier_cycle_id=cycle_id,
            tree_id=node.tree_id,
            anchor_spine_id=node.anchor_spine_id,
            anchor_spine_schema=ANCHOR_SPINE_SCHEMA,
            anchor_local_index=node.local_index,
            parent_anchor_node_event_id=parent_event_id,
            parent_anchor_local_index=node.parent_local_index,
            committed_prefix_position=committed_prefix_position,
            authoritative_prefix_hash=prefix_hash(authoritative_prefix_token_ids),
            exact_prefix_hash=conditioning_hash,
            mtp_conditioning_prefix_hash=conditioning_hash,
            engine_draft_depth=node.depth,
            depth=node.depth,
            gcrp_horizon=node.depth,
            gcrp_target_position=router_position,
            router_state_position=router_position,
            vocabulary_prediction_position=vocabulary_position,
            node_token_id=node.token_id,
            token_rank_under_parent=node.token_rank_under_parent,
            branch_path_token_ids=[int(value) for value in node.token_path_ids],
            branch_path_token_log_probabilities=[
                float(value) for value in node.token_path_log_probabilities
            ],
            path_id=node.path_id,
            node_local_token_log_probability=node.local_token_log_probability,
            node_local_token_probability=node.local_token_probability,
            path_log_probability=node.path_log_probability,
            path_probability=node.path_probability,
            vocabulary_top_token_id=int(node.observation.top_token_ids[0]),
            top_token_probability=node.observation.top_probability,
            source_ready_event_order=self.events.next_id,
            source_ready_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            source_ready_monotonic_ns=time.monotonic_ns(),
            source_clock_domain_id="host_monotonic_after_native_mtp_materialization",
            native_branch_execution_mode=(
                "isolated_full_prefix_recomputation_no_shared_mutable_kv"
            ),
            structural_validity=True,
            feature_available=True,
            exact_committed_h1_root=node.depth == 1,
            anchor_only=True,
            adaptive_model_input=False,
            consumes_adaptive_node_budget=False,
            continues_through_eos=True,
            eos_termination_applies=False,
            required_anchor_depth=ANCHOR_SPINE_REQUIRED_DEPTH,
            acceptance_fields_present=False,
            label_records_present=False,
            record_valid=True,
        )
        assert_causal_anchor_spine_node_record(fields)
        node_event = self.events.emit(ANCHOR_SPINE_NODE_EVENT, **fields)
        tensors = (
            (
                self.anchor_states,
                "mtp_vocabulary_head_input",
                ANCHOR_SPINE_ROLES[0],
                router_position,
            ),
            (
                self.anchor_routes,
                "raw_mtp_router_logits",
                ANCHOR_SPINE_ROLES[1],
                router_position,
            ),
            (
                self.anchor_vocab,
                "vocab_top64_token_ids",
                ANCHOR_SPINE_ROLES[2],
                vocabulary_position,
            ),
            (
                self.anchor_vocab,
                "vocab_top64_log_probabilities",
                ANCHOR_SPINE_ROLES[3],
                vocabulary_position,
            ),
        )
        for sidecar, payload_role, event_role, position in tensors:
            self.write_tensor(
                sidecar,
                payload[payload_role],
                role=event_role,
                parent_event_id=node_event,
                entity_kind=ANCHOR_SPINE_NODE_EVENT,
                sequence_id=sequence_id,
                absolute_position=position,
                depth=node.depth,
            )
        parent_event_ids[node.local_index] = node_event
        self.anchor_spine_nodes += 1
        return node_event

    def write_adaptive_node(
        self,
        node: AdaptiveTreeNode,
        *,
        sequence_id: str,
        request_id: str,
        cycle_id: int,
        committed_prefix_position: int,
        authoritative_prefix_token_ids: list[int],
        parent_event_ids: dict[int, int],
    ) -> int:
        payload = node.observation.payload
        if not isinstance(payload, dict):
            raise RuntimeError("adaptive node payload was cleared before persistence")
        parent_event_id = (
            None
            if node.parent_local_index is None
            else parent_event_ids[node.parent_local_index]
        )
        router_position = committed_prefix_position + node.depth
        vocabulary_position = router_position + 1
        router_logits = payload["raw_mtp_router_logits"]
        router_probabilities = payload["full_mtp_router_probabilities"].float()
        router_entropy = float(
            (
                -router_probabilities
                * torch.log(router_probabilities.clamp_min(1e-30))
            ).sum().item()
        )
        top_logps = node.observation.top_log_probabilities
        top1_margin = (
            math.inf if len(top_logps) < 2 else float(top_logps[0] - top_logps[1])
        )
        top8_mass = float(sum(math.exp(value) for value in top_logps[:8]))
        ready_timestamp = datetime.now(timezone.utc).isoformat()
        ready_monotonic_ns = time.monotonic_ns()
        conditioning = (
            "authoritative_exact_root"
            if node.depth == 1
            else "position_aligned_speculative_context"
        )
        conditioning_hash = prefix_hash(
            authoritative_prefix_token_ids + list(node.token_path_ids)
        )
        node_fields = dict(
            run_id=self.run_id,
            sequence_id=sequence_id,
            request_id=request_id,
            verifier_cycle_id=cycle_id,
            tree_id=node.tree_id,
            tree_node_id=node.tree_node_id,
            node_local_index=node.local_index,
            parent_node_event_id=parent_event_id,
            parent_tree_node_id=node.parent_tree_node_id,
            parent_local_index=node.parent_local_index,
            branch_id=node.branch_id,
            branch_path_id=node.path_id,
            path_id=node.path_id,
            committed_prefix_position=committed_prefix_position,
            authoritative_prefix_hash=prefix_hash(authoritative_prefix_token_ids),
            exact_prefix_hash=conditioning_hash,
            mtp_conditioning_prefix_hash=conditioning_hash,
            engine_draft_depth=node.depth,
            depth=node.depth,
            router_state_position=router_position,
            mtp_input_token_position=router_position,
            vocabulary_prediction_position=vocabulary_position,
            gcrp_target_position=router_position,
            gcrp_horizon=node.depth,
            target_position_mapping={
                "source_t": committed_prefix_position,
                "node_horizon": node.depth,
                "node_target_position": router_position,
                "vocabulary_prediction_position": vocabulary_position,
            },
            execution_phase="MTP_ADAPTIVE_DRAFT",
            conditioning_class=conditioning,
            structural_validity=True,
            structural_validity_reason="native_isolated_parent_before_child",
            exact_committed_h1_root=node.depth == 1,
            node_token_id=node.token_id,
            node_token_position=router_position,
            token_rank_under_parent=node.token_rank_under_parent,
            branch_path_token_ids=[int(v) for v in node.token_path_ids],
            draft_prefix_token_ids=[int(v) for v in node.token_path_ids],
            branch_path_token_log_probabilities=[
                float(v) for v in node.token_path_log_probabilities
            ],
            node_local_token_log_probability=node.local_token_log_probability,
            node_local_token_probability=node.local_token_probability,
            path_log_probability=node.path_log_probability,
            path_probability=node.path_probability,
            # Compatibility names consumed by the current immutable rich index.
            draft_token_log_probability=node.local_token_log_probability,
            cumulative_draft_logprobability=node.path_log_probability,
            branch_probability=node.path_probability,
            draft_token_id=int(node.observation.top_token_ids[0]),
            vocabulary_top_token_id=int(node.observation.top_token_ids[0]),
            top_token_probability=node.observation.top_probability,
            top1_top2_logprob_margin=top1_margin,
            vocabulary_entropy=node.observation.vocabulary_entropy,
            vocabulary_entropy_valid=True,
            vocabulary_top8_mass=top8_mass,
            vocabulary_top8_mass_valid=True,
            mtp_router_entropy=router_entropy,
            mtp_router_logit_rms=float(
                router_logits.float().square().mean().sqrt().item()
            ),
            policy_expansion_width=self.policy.width(
                node.observation, depth=node.depth
            ),
            source_ready_event_order=self.events.next_id,
            source_ready_timestamp_utc=ready_timestamp,
            source_ready_monotonic_ns=ready_monotonic_ns,
            source_clock_domain_id="host_monotonic_after_native_mtp_materialization",
            source_graph_node_id="checkpoint_mtp.layers.0",
            source_mtp_layer_id=0,
            native_branch_execution_mode=(
                "isolated_full_prefix_recomputation_no_shared_mutable_kv"
            ),
            feature_available=True,
            acceptance_fields_present=False,
            record_valid=True,
        )
        assert_causal_mtp_node_record(node_fields)
        node_event = self.events.emit("mtp_node", **node_fields)

        state_roles = (
            "frozen_target_token_embedding",
            "mtp_normalized_token_embedding",
            "mtp_normalized_previous_hidden",
            "mtp_fused_state",
            "mtp_router_input",
            "mtp_hidden_state",
            "mtp_post_ffn_hidden",
            "mtp_vocabulary_head_input",
        )
        route_roles = (
            "raw_mtp_router_logits",
            "full_mtp_router_probabilities",
            "mtp_selected_expert_ids",
            "mtp_selected_execution_weights",
        )
        vocab_roles = (
            "vocab_top64_token_ids",
            "vocab_top64_log_probabilities",
            "vocab_top64_probabilities",
        )
        for role in state_roles:
            self.write_tensor(
                self.mtp_states,
                payload[role],
                role=role,
                parent_event_id=node_event,
                entity_kind="mtp_node",
                sequence_id=sequence_id,
                absolute_position=router_position,
                depth=node.depth,
            )
        for role in route_roles:
            self.write_tensor(
                self.mtp_routes,
                payload[role],
                role=role,
                parent_event_id=node_event,
                entity_kind="mtp_node",
                sequence_id=sequence_id,
                absolute_position=router_position,
                depth=node.depth,
            )
        for role in vocab_roles:
            self.write_tensor(
                self.mtp_vocab,
                payload[role],
                role=role,
                parent_event_id=node_event,
                entity_kind="mtp_node",
                sequence_id=sequence_id,
                absolute_position=vocabulary_position,
                depth=node.depth,
            )
        audit_key = int(
            hashlib.sha256(
                f"{sequence_id}:{cycle_id}:{node.path_id}".encode()
            ).hexdigest()[:8],
            16,
        )
        if (cycle_id == 0 and node.local_index == 0) or (
            audit_key % 100 < self.full_vocab_audit_percent
        ):
            self.write_tensor(
                self.mtp_vocab,
                payload["full_vocabulary_logits_audit"],
                role="full_vocabulary_logits_audit",
                parent_event_id=node_event,
                entity_kind="mtp_node",
                sequence_id=sequence_id,
                absolute_position=vocabulary_position,
                depth=node.depth,
            )
        self.mtp_nodes += 1
        self.nodes_by_depth[node.depth] += 1
        self.ready_mtp_nodes.append(
            {"event_id": node_event, "gcrp_target_position": router_position}
        )
        parent_event_ids[node.local_index] = node_event
        return node_event


def _build_matched_controls(
    run: AdaptiveCaptureRun,
    *,
    mtp,
    tree_id: str,
    nodes: list[AdaptiveTreeNode],
    sequence_id: str,
    request_id: str,
    cycle_id: int,
    committed_prefix_position: int,
    root_prefix: list[int],
    final_hidden_history: list[torch.Tensor],
    exact_h1_token: int,
    eos_token_id: int | None,
) -> tuple[dict[str, list[AdaptiveTreeNode]], dict[str, int]]:
    """Build all causal B1 views and emit their label-free ready records."""

    if len(nodes) != run.policy.max_nodes:
        raise RuntimeError("canonical adaptive tree did not fill its declared budget")
    if run.policy.max_nodes != 32:
        raise ValueError("matched B1 controls require a canonical adaptive-32 tree")

    independent_runner = NativeMTPBranchRunner(
        mtp=mtp,
        authoritative_prefix_token_ids=root_prefix,
        authoritative_target_hidden_through_t=final_hidden_history,
        exact_h1_token_id=exact_h1_token,
    )
    independent16 = AdaptiveMTPTreeBuilder(
        AdaptiveExpansionPolicy(max_nodes=16)
    ).build(
        tree_id=tree_id,
        exact_h1_token_id=exact_h1_token,
        evaluate=independent_runner.evaluate,
        eos_token_id=eos_token_id,
    )
    assert_identical_structure(nodes[:16], independent16)
    run.control_native_mtp_calls["adaptive16_independent"] += (
        independent_runner.native_call_count
    )

    fixed_views: dict[str, list[AdaptiveTreeNode]] = {}
    fixed_manifests: dict[str, dict[str, Any]] = {}
    for name, widths in (("fixed16", (5, 5, 5)), ("fixed32", (11, 10, 10))):
        policy = FixedBeamPolicy(widths)
        runner = NativeMTPBranchRunner(
            mtp=mtp,
            authoritative_prefix_token_ids=root_prefix,
            authoritative_target_hidden_through_t=final_hidden_history,
            exact_h1_token_id=exact_h1_token,
        )
        fixed_views[name] = build_fixed_beam_tree(
            tree_id=f"{tree_id}:control:{name}",
            exact_h1_token_id=exact_h1_token,
            evaluate=runner.evaluate,
            policy=policy,
        )
        fixed_manifests[name] = policy.manifest()
        run.control_native_mtp_calls[name] += runner.native_call_count

    greedy = greedy_spine_prefix(nodes)

    views = {
        "greedy": greedy,
        "fixed16": fixed_views["fixed16"],
        "adaptive16": nodes[:16],
        "fixed32": fixed_views["fixed32"],
        "adaptive32": nodes,
    }
    policy_manifests = {
        "greedy": {
            "schema": "harp_rtt_greedy_spine_control_v1",
            "maximum_depth": 4,
            "selection": "local_rank_zero_parent_coherent",
            "source": "canonical_adaptive32_prefix",
            "terminal_shortening": "stop_when_rank_zero_child_is_absent_after_eos",
            "uses_target_labels": False,
        },
        "fixed16": fixed_manifests["fixed16"],
        "adaptive16": AdaptiveExpansionPolicy(max_nodes=16).manifest(),
        "fixed32": fixed_manifests["fixed32"],
        "adaptive32": run.policy.manifest(),
    }

    ready_event_ids: dict[str, int] = {}
    for name in CONTROL_NAMES:
        control_nodes = views[name]
        serialized = serialize_nodes(control_nodes)
        record = {
            "run_id": run.run_id,
            "sequence_id": sequence_id,
            "request_id": request_id,
            "verifier_cycle_id": cycle_id,
            "control_schema": CONTROL_SCHEMA,
            "control_name": name,
            "control_tree_id": f"{tree_id}:view:{name}",
            "canonical_adaptive_tree_id": tree_id,
            "committed_prefix_position": committed_prefix_position,
            "authoritative_prefix_hash": prefix_hash(root_prefix),
            "exact_h1_token_id": exact_h1_token,
            "node_count": len(control_nodes),
            "nodes_by_horizon": {
                str(depth): sum(node.depth == depth for node in control_nodes)
                for depth in range(1, MAX_CAPTURE_DEPTH + 1)
            },
            "nodes": serialized,
            "structural_hash": structural_hash(control_nodes),
            "policy": policy_manifests[name],
            "policy_hash": sha256_bytes(canonical_json(policy_manifests[name])),
            "adaptive16_definition": (
                "first_16_parent_before_child_nodes_of_canonical_adaptive32"
                if name == "adaptive16"
                else None
            ),
            "independent_adaptive16_structural_hash": (
                structural_hash(independent16) if name == "adaptive16" else None
            ),
            "independent_adaptive16_exact_match": (
                True if name == "adaptive16" else None
            ),
            "source_ready_event_order": run.events.next_id,
            "synchronous_before_target_h1_execution": True,
            "causal_input_only": True,
            "diagnostic_only": True,
            "model_input": False,
            "target_labels_present": False,
            "acceptance_present": False,
            "factual_continuation_used": False,
            "sealed_test_opened": False,
        }
        assert_ready_record(record)
        ready_event_ids[name] = run.events.emit(CONTROL_READY_EVENT, **record)
        run.control_tree_count[name] += 1
        run.control_node_count[name] += len(control_nodes)
    return views, ready_event_ids

@torch.no_grad()
def capture_adaptive_sequence(
    run: AdaptiveCaptureRun,
    target,
    mtp,
    hooks: TargetCaptureHooks,
    tokenizer,
    prompt: dict[str, Any],
    ordinal: int,
    *,
    source_positions: int,
    label_lookahead: int,
) -> None:
    assert_prompt_capture_allowed(prompt)
    sample_id = str(prompt.get("sample_id", prompt.get("prompt_id", ordinal)))
    prompt_sha = str(
        prompt.get(
            "prompt_sha256",
            sha256_bytes(
                str(prompt.get("text", prompt.get("prompt_token_ids", ""))).encode()
            ),
        )
    )
    sequence_id = f"tfadaptive-seq{ordinal:06d}-{prompt_sha[:8]}"
    requested_offsets = prompt.get("source_position_offsets")
    if requested_offsets is None:
        source_position_offsets = list(range(source_positions))
    else:
        source_position_offsets = [int(value) for value in requested_offsets]
        if len(source_position_offsets) != source_positions:
            raise ValueError(
                "prompt source_position_offsets must match --source-positions"
            )
        if (
            source_position_offsets != sorted(set(source_position_offsets))
            or any(value < 0 for value in source_position_offsets)
        ):
            raise ValueError(
                "source_position_offsets must be sorted unique non-negative integers"
            )
    capture_offsets = frozenset(source_position_offsets)
    request_id = f"tfadaptive-req{ordinal:06d}-{prompt_sha[:8]}"
    prompt_token_ids = (
        [int(value) for value in prompt["prompt_token_ids"]]
        if "prompt_token_ids" in prompt
        else prompt_ids(tokenizer, prompt["text"])
    )
    group_id = str(
        prompt.get(
            "group_id",
            prompt.get("conversation_group_id", prompt.get("family", sample_id)),
        )
    )
    run.ready_mtp_nodes.clear()
    run.sequence_count += 1
    sequence_start = run.events.emit(
        "sequence_start",
        run_id=run.run_id,
        sequence_id=sequence_id,
        request_id=request_id,
        prompt_id=sample_id,
        manifest_ordinal=ordinal,
        conversation_or_source_family_id=str(
            prompt.get("conversation_group_id", group_id)
        ),
        dataset_item_family_id=group_id,
        dataset_source=str(prompt.get("source", "synthetic_pilot")),
        domain_label=prompt.get("domain"),
        language_label=prompt.get("language"),
        original_split=prompt.get("original_split"),
        source_request_id=prompt.get("source_request_id"),
        source_sequence_id=prompt.get("source_sequence_id"),
        assigned_split=prompt.get("split"),
        split_manifest_sha256=prompt.get("split_manifest_sha256"),
        inner_partition=prompt.get("partition"),
        external_evaluation=False,
        prompt_token_ids=prompt_token_ids,
        prompt_hash=prefix_hash(prompt_token_ids),
        source_prompt_sha256=prompt_sha,
        split_group_id=group_id,
        offline_split_assigned=False,
        sealed_test_opened=False,
    )

    committed_tokens = list(prompt_token_ids)
    generated_tokens: list[int] = []
    final_hidden_history: list[torch.Tensor] = []
    target_cache = None
    call_id = 0
    cycle_id = 0
    captured_trees: list[
        tuple[int, int, list[AdaptiveTreeNode], dict[int, int], str]
    ] = []
    captured_controls: list[
        tuple[
            int,
            int,
            str,
            dict[str, list[AdaptiveTreeNode]],
            dict[str, int],
        ]
    ] = []

    def target_call(token_ids: list[int], phase: str, start_position: int):
        nonlocal target_cache, call_id
        hooks.clear()
        result = target(
            input_ids=torch.tensor([token_ids], device=target.device, dtype=torch.long),
            past_key_values=target_cache,
            use_cache=True,
            output_hidden_states=True,
            output_router_logits=True,
            return_dict=True,
        )
        target_cache = result.past_key_values
        final_hidden_history.extend(
            run.flush_target_call(
                hooks,
                result,
                sequence_id=sequence_id,
                request_id=request_id,
                phase=phase,
                start_position=start_position,
                full_prefix_tokens=committed_tokens,
                call_input_ids=token_ids,
                call_id=call_id,
            )
        )
        call_id += 1
        return result

    current_output = target_call(prompt_token_ids, "PREFILL_TARGET", 0)
    computed_length = len(prompt_token_ids)
    eos_id = tokenizer.eos_token_id
    termination = "source_positions_plus_label_lookahead"
    total_generation_steps = max(source_position_offsets) + 1 + label_lookahead
    post_warmup_steps = 0

    # Prompt rows persist route history only in the established rich schema.
    # Commit and execute one target token before opening the first adaptive tree
    # so every source t has all required target state/residual channels.
    warmup_token = int(current_output.logits[0, -1].argmax().item())
    generated_tokens.append(warmup_token)
    committed_tokens.append(warmup_token)
    current_output = target_call(
        [warmup_token], "VERIFICATION_TARGET", computed_length
    )
    computed_length += 1
    if warmup_token == eos_id:
        termination = "eos_during_source_warmup"

    while warmup_token != eos_id and post_warmup_steps < total_generation_steps:
        committed_prefix_position = computed_length - 1
        root_prefix = list(committed_tokens[:computed_length])
        exact_h1_token = int(current_output.logits[0, -1].argmax().item())

        if post_warmup_steps in capture_offsets:
            if len(final_hidden_history) != computed_length:
                raise RuntimeError("target hidden history crossed the causal source boundary")
            tree_id = (
                f"{sequence_id}:tree{cycle_id:06d}:p{committed_prefix_position:08d}"
            )
            parent_event_ids: dict[int, int] = {}
            runner = NativeMTPBranchRunner(
                mtp=mtp,
                authoritative_prefix_token_ids=root_prefix,
                authoritative_target_hidden_through_t=final_hidden_history,
                exact_h1_token_id=exact_h1_token,
            )

            def persist(node: AdaptiveTreeNode) -> None:
                run.write_adaptive_node(
                    node,
                    sequence_id=sequence_id,
                    request_id=request_id,
                    cycle_id=cycle_id,
                    committed_prefix_position=committed_prefix_position,
                    authoritative_prefix_token_ids=root_prefix,
                    parent_event_ids=parent_event_ids,
                )

            nodes = AdaptiveMTPTreeBuilder(run.policy).build(
                tree_id=tree_id,
                exact_h1_token_id=exact_h1_token,
                evaluate=runner.evaluate,
                on_node=persist,
                eos_token_id=eos_id,
            )
            run.native_mtp_calls += runner.native_call_count
            run.adaptive_tree_count += 1
            run.adaptive_source_positions += 1
            run.events.emit(
                "mtp_tree_ready",
                run_id=run.run_id,
                sequence_id=sequence_id,
                request_id=request_id,
                verifier_cycle_id=cycle_id,
                tree_id=tree_id,
                committed_prefix_position=committed_prefix_position,
                authoritative_prefix_hash=prefix_hash(root_prefix),
                exact_h1_token_id=exact_h1_token,
                eos_token_id=eos_id,
                exact_h1_root_event_id=parent_event_ids[0],
                mtp_node_event_ids=[parent_event_ids[node.local_index] for node in nodes],
                tree_node_ids=[node.tree_node_id for node in nodes],
                node_count=len(nodes),
                maximum_node_budget=run.policy.max_nodes,
                nodes_by_horizon={
                    str(depth): sum(node.depth == depth for node in nodes)
                    for depth in range(1, MAX_CAPTURE_DEPTH + 1)
                },
                source_ready_event_order=run.events.next_id,
                synchronous_before_target_h1_execution=True,
                expansion_uses_realized_future=False,
                expansion_uses_acceptance=False,
            )

            if run.emit_matched_controls:
                control_views, control_ready_event_ids = _build_matched_controls(
                    run,
                    mtp=mtp,
                    tree_id=tree_id,
                    nodes=nodes,
                    sequence_id=sequence_id,
                    request_id=request_id,
                    cycle_id=cycle_id,
                    committed_prefix_position=committed_prefix_position,
                    root_prefix=root_prefix,
                    final_hidden_history=final_hidden_history,
                    exact_h1_token=exact_h1_token,
                    eos_token_id=eos_id,
                )
                captured_controls.append(
                    (
                        cycle_id,
                        committed_prefix_position,
                        tree_id,
                        control_views,
                        control_ready_event_ids,
                    )
                )

            # Re-run an independent, anchor-only greedy path through every
            # pinned preprocessing depth.  This neither consumes the adaptive
            # 32-node budget nor enters ready_mtp_nodes/model inputs.  Unlike
            # the adaptive graph, it intentionally continues after EOS to
            # preserve incumbent legacy-capture semantics exactly.
            anchor_parent_event_ids: dict[int, int] = {}
            anchor_runner = NativeMTPBranchRunner(
                mtp=mtp,
                authoritative_prefix_token_ids=root_prefix,
                authoritative_target_hidden_through_t=final_hidden_history,
                exact_h1_token_id=exact_h1_token,
            )

            def persist_anchor(node: AnchorSpineNode) -> None:
                run.write_anchor_spine_node(
                    node,
                    sequence_id=sequence_id,
                    request_id=request_id,
                    cycle_id=cycle_id,
                    committed_prefix_position=committed_prefix_position,
                    authoritative_prefix_token_ids=root_prefix,
                    parent_event_ids=anchor_parent_event_ids,
                )

            anchor_nodes = build_anchor_spine(
                tree_id=tree_id,
                exact_h1_token_id=exact_h1_token,
                evaluate=anchor_runner.evaluate_anchor,
                on_node=persist_anchor,
                required_depth=ANCHOR_SPINE_REQUIRED_DEPTH,
            )
            run.anchor_native_mtp_calls += anchor_runner.native_call_count
            run.anchor_spine_count += 1
            run.events.emit(
                ANCHOR_SPINE_READY_EVENT,
                run_id=run.run_id,
                sequence_id=sequence_id,
                request_id=request_id,
                verifier_cycle_id=cycle_id,
                tree_id=tree_id,
                anchor_spine_id=anchor_nodes[0].anchor_spine_id,
                anchor_spine_schema=ANCHOR_SPINE_SCHEMA,
                committed_prefix_position=committed_prefix_position,
                authoritative_prefix_hash=prefix_hash(root_prefix),
                exact_h1_token_id=exact_h1_token,
                exact_h1_root_event_id=anchor_parent_event_ids[0],
                anchor_node_event_ids=[
                    anchor_parent_event_ids[node.local_index]
                    for node in anchor_nodes
                ],
                node_count=len(anchor_nodes),
                required_depth=ANCHOR_SPINE_REQUIRED_DEPTH,
                source_ready_event_order=run.events.next_id,
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
            captured_trees.append(
                (
                    cycle_id,
                    committed_prefix_position,
                    nodes,
                    parent_event_ids,
                    tree_id,
                )
            )
            cycle_id += 1

        # Commit exactly the target-selected H1 token.  No speculative branch
        # changes target generation or supplies future state to another tree.
        generated_tokens.append(exact_h1_token)
        committed_tokens.append(exact_h1_token)
        current_output = target_call(
            [exact_h1_token], "VERIFICATION_TARGET", computed_length
        )
        computed_length += 1
        post_warmup_steps += 1
        if exact_h1_token == eos_id:
            termination = "eos"
            break

    for (
        tree_cycle_id,
        committed_prefix_position,
        nodes,
        event_ids,
        tree_id,
    ) in captured_trees:
        labels = build_acceptance_labels(
            nodes,
            committed_token_ids=committed_tokens,
            committed_prefix_position=committed_prefix_position,
        )
        accepted_event_ids: list[int] = []
        valid_count = 0
        for node, label in zip(nodes, labels):
            valid_count += int(label["acceptance_label_valid"])
            if label["branch_path_accepted"] is True:
                accepted_event_ids.append(event_ids[node.local_index])
            run.events.emit(
                "mtp_acceptance_label",
                run_id=run.run_id,
                sequence_id=sequence_id,
                request_id=request_id,
                verifier_cycle_id=tree_cycle_id,
                tree_id=tree_id,
                mtp_node_event_id=event_ids[node.local_index],
                **label,
            )
        run.events.emit(
            "mtp_tree_resolved",
            run_id=run.run_id,
            sequence_id=sequence_id,
            request_id=request_id,
            tree_id=tree_id,
            committed_prefix_position=committed_prefix_position,
            node_event_ids=[event_ids[node.local_index] for node in nodes],
            acceptance_valid_node_count=valid_count,
            accepted_node_event_ids=accepted_event_ids,
            labels_only=True,
        )

    for (
        control_cycle_id,
        committed_prefix_position,
        tree_id,
        control_views,
        control_ready_event_ids,
    ) in captured_controls:
        for name in CONTROL_NAMES:
            control_nodes = control_views[name]
            resolved = resolved_labels(
                control_nodes,
                committed_token_ids=committed_tokens,
                committed_prefix_position=committed_prefix_position,
            )
            record = {
                "run_id": run.run_id,
                "sequence_id": sequence_id,
                "request_id": request_id,
                "verifier_cycle_id": control_cycle_id,
                "control_schema": CONTROL_SCHEMA,
                "control_name": name,
                "control_tree_id": f"{tree_id}:view:{name}",
                "canonical_adaptive_tree_id": tree_id,
                "control_ready_event_id": control_ready_event_ids[name],
                "committed_prefix_position": committed_prefix_position,
                "node_count": len(control_nodes),
                "structural_hash": structural_hash(control_nodes),
                **resolved,
                "labels_only": True,
                "available_at_runtime": False,
                "model_input": False,
                "sealed_test_opened": False,
            }
            assert_resolved_record(record)
            run.events.emit(CONTROL_RESOLVED_EVENT, **record)

    run.events.emit(
        "sequence_end",
        run_id=run.run_id,
        sequence_id=sequence_id,
        request_id=request_id,
        sequence_start_event_id=sequence_start,
        prompt_token_ids=prompt_token_ids,
        committed_generated_token_ids=generated_tokens,
        full_committed_token_ids=committed_tokens,
        prompt_length=len(prompt_token_ids),
        generated_length=len(generated_tokens),
        captured_authoritative_position_count=computed_length,
        source_state_warmup_generated_token_count=1,
        adaptive_tree_source_position_count=len(captured_trees),
        legacy_anchor_spine_source_position_count=len(captured_trees),
        requested_source_positions=source_positions,
        requested_label_lookahead=label_lookahead,
        requested_source_position_offsets=source_position_offsets,
        termination_reason=termination,
        eos_position=(
            len(prompt_token_ids) + generated_tokens.index(eos_id)
            if eos_id in generated_tokens
            else None
        ),
        sealed_test_opened=False,
    )


def build_adaptive_manifest(
    args,
    target,
    config,
    tokenizer,
    mtp,
    run: AdaptiveCaptureRun,
    started_at: str,
) -> dict[str, Any]:
    # Reuse the existing frozen-model/router provenance machinery, then replace
    # every chain-specific semantic field with the versioned adaptive contract.
    manifest = build_chain_manifest(
        args, target, config, tokenizer, mtp, run, started_at
    )
    policy_manifest = run.policy.manifest()
    graph_manifest = {
        "schema": "harp_rtt_native_mtp_adaptive_graph_v1",
        "checkpoint_mtp_tensor_mapping": mtp.load_report,
        "root": (
            "exact target argmax x[t+1], selected after target token t and before "
            "target token t+1 execution"
        ),
        "semantic_position_rule": {
            "source_target_state": "p=t",
            "tree_node_depth_h_token_and_router_state": "p+h",
            "tree_node_vocabulary_prediction": "p+h+1",
        },
        "native_branch_execution": (
            "isolated full-prefix recomputation per path; no mutable KV cache is "
            "shared between siblings"
        ),
        "maximum_nodes_including_root": run.policy.max_nodes,
        "depths": [1, 2, 3, 4],
        "policy": policy_manifest,
        "acceptance_storage": "separate mtp_acceptance_label events only",
        "forbidden_future_inputs": [
            "target routes/states from t+1 onward",
            "realized H2-H4 token IDs",
            "eventual branch acceptance",
        ],
    }
    anchor_manifest = {
        "schema": ANCHOR_SPINE_SCHEMA,
        "purpose": "epoch-zero LegacyHARPAnchorBridge compatibility only",
        "root": "same exact committed target H1 token as the adaptive tree",
        "required_depth": ANCHOR_SPINE_REQUIRED_DEPTH,
        "depths": list(range(1, ANCHOR_SPINE_REQUIRED_DEPTH + 1)),
        "parent_rule": "H2-H6 consume the immediately preceding node local top-1",
        "eos_rule": "continue through all six depths even when an earlier token is EOS",
        "native_branch_execution": (
            "independent isolated full-prefix recomputation; no adaptive sibling state"
        ),
        "tensor_roles": list(ANCHOR_SPINE_ROLES),
        "adaptive_node_budget_consumed": False,
        "adaptive_model_input": False,
        "labels_emitted": False,
        "realized_future_inputs": False,
        "source_ready_before_target_h1_execution": True,
    }
    matched_control_manifest = {
        "schema": CONTROL_SCHEMA,
        "enabled": run.emit_matched_controls,
        "control_names": list(CONTROL_NAMES),
        "canonical_view": "adaptive32",
        "adaptive16_definition": (
            "first_16_parent_before_child_nodes_of_canonical_adaptive32"
        ),
        "adaptive16_independent_rebuild_required": True,
        "fixed16_depth_widths": {"1": 1, "2": 5, "3": 5, "4": 5},
        "fixed32_depth_widths": {"1": 1, "2": 11, "3": 10, "4": 10},
        "selection_inputs": "causal_native_mtp_only",
        "ready_records_contain_target_labels": False,
        "resolved_records_label_only": True,
        "model_input": False,
    }
    manifest.update(
        {
            "schema": MANIFEST_SCHEMA,
            "format_version": FORMAT_VERSION,
            "capture_profile": CAPTURE_PROFILE,
            "decoding_mode": (
                "deterministic greedy frozen target with non-intervening adaptive "
                "native-MTP H1--H4 capture"
            ),
            "sampling_parameters": {"temperature": 0.0, "top_p": 1.0},
            "mtp_draft_max": MAX_CAPTURE_DEPTH,
            "mtp_graph_manifest": graph_manifest,
            "mtp_graph_manifest_hash": sha256_bytes(canonical_json(graph_manifest)),
            "legacy_harp_anchor_spine_manifest": anchor_manifest,
            "legacy_harp_anchor_spine_manifest_hash": sha256_bytes(
                canonical_json(anchor_manifest)
            ),
            "adaptive_expansion_policy": policy_manifest,
            "adaptive_expansion_policy_hash": sha256_bytes(
                canonical_json(policy_manifest)
            ),
            "matched_tree_control_manifest": matched_control_manifest,
            "matched_tree_control_manifest_hash": sha256_bytes(
                canonical_json(matched_control_manifest)
            ),
            "clock_domain_definitions": {
                **manifest["clock_domain_definitions"],
                "host_monotonic_after_native_mtp_materialization": (
                    "host time.monotonic_ns sampled after native-MTP scalar "
                    "metadata has been materialized on the host; comparable only "
                    "within this process and not a performance benchmark"
                ),
            },
            "capture_policy": {
                **manifest["capture_policy"],
                "exact_committed_h1_root": True,
                "adaptive_native_mtp_h2_h4": True,
                "maximum_tree_nodes_including_root": run.policy.max_nodes,
                "exact_tree_node_budget": True,
                "matched_controls_emitted": run.emit_matched_controls,
                "parent_before_child": True,
                "isolated_sibling_execution": True,
                "acceptance_labels_separate": True,
                "realized_h2_h4_available_to_expansion": False,
                "future_target_state_available_to_expansion": False,
                "legacy_anchor_spine_separate": True,
                "legacy_anchor_required_depth": ANCHOR_SPINE_REQUIRED_DEPTH,
                "legacy_anchor_continues_through_eos": True,
                "legacy_anchor_consumes_adaptive_budget": False,
                "legacy_anchor_enters_adaptive_model_inputs": False,
                "legacy_anchor_labels_present": False,
                "source_positions_per_sequence": args.source_positions,
                "label_lookahead": args.label_lookahead,
                "source_position_selection": "per_prompt_uniform_offsets_or_consecutive",
            },
            "counts": {
                **manifest["counts"],
                "mtp_nodes": run.mtp_nodes,
                "adaptive_trees": run.adaptive_tree_count,
                "adaptive_source_positions": run.adaptive_source_positions,
                "native_mtp_branch_calls": run.native_mtp_calls,
                "matched_control_trees": dict(run.control_tree_count),
                "matched_control_nodes": dict(run.control_node_count),
                "matched_control_native_mtp_calls": dict(
                    run.control_native_mtp_calls
                ),
                "legacy_anchor_spines": run.anchor_spine_count,
                "legacy_anchor_spine_nodes": run.anchor_spine_nodes,
                "legacy_anchor_native_mtp_branch_calls": run.anchor_native_mtp_calls,
                "mtp_nodes_by_depth": {
                    str(depth): run.nodes_by_depth[depth]
                    for depth in range(1, MAX_CAPTURE_DEPTH + 1)
                },
            },
            "sealed_test_opened": False,
            "training_started": False,
            "stop_before_training": True,
        }
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-id", default="harp_rtt_adaptive_mtp_tree_pilot_v1"
    )
    parser.add_argument("--source-positions", type=int, default=2)
    parser.add_argument("--label-lookahead", type=int, default=3)
    parser.add_argument("--max-tree-nodes", type=int, default=32)
    parser.add_argument("--full-vocab-audit-percent", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda",
        help="single execution device: cpu, cuda, or cuda:<index> (default: cuda)",
    )
    parser.add_argument("--hash-all-weights", action="store_true")
    parser.add_argument("--model-weight-manifest", type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--candidate-vector-audit-percent", type=int, default=5)
    parser.add_argument("--prompt-history-tokens", type=int, default=8)
    parser.add_argument("--production-capture", action="store_true")
    parser.add_argument(
        "--emit-matched-controls",
        action="store_true",
        help="emit causal greedy/fixed/adaptive B1 diagnostic views",
    )
    args = parser.parse_args()
    # The shared provenance builder expects this legacy attribute.  It is
    # immediately replaced by the adaptive graph manifest.
    args.max_draft_depth = MAX_CAPTURE_DEPTH
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable capture {args.output}")
    if args.source_positions <= 0:
        raise ValueError("source_positions must be positive")
    if args.label_lookahead < 3:
        raise ValueError("H4 label completeness requires label_lookahead >= 3")
    if not 4 <= args.max_tree_nodes <= 32:
        raise ValueError("the H1--H4 first-teacher capture budget must be in [4, 32]")
    if not 0 <= args.full_vocab_audit_percent <= 100:
        raise ValueError("full-vocab audit percent must be in [0, 100]")
    if args.production_capture and args.prompt_manifest is None:
        raise ValueError("production adaptive capture requires an explicit prompt manifest")
    policy = AdaptiveExpansionPolicy(max_nodes=args.max_tree_nodes)

    args.output.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    target, config = load_target(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    mtp = load_checkpoint_mtp(
        args.model,
        device=target.device,
        embed_tokens=target.model.embed_tokens,
        lm_head=target.lm_head,
        config=config,
    )
    run = AdaptiveCaptureRun(
        args.output,
        args.run_id,
        args.model,
        policy=policy,
        full_vocab_audit_percent=args.full_vocab_audit_percent,
        candidate_vector_audit_percent=args.candidate_vector_audit_percent,
        prompt_history_tokens=args.prompt_history_tokens,
        production_capture=args.production_capture,
        emit_matched_controls=args.emit_matched_controls,
    )
    run.model = target
    if args.prompt_manifest:
        all_prompts = [
            json.loads(line)
            for line in args.prompt_manifest.read_text().splitlines()
            if line.strip()
        ]
        stop = None if args.limit is None else args.skip + args.limit
        prompts = all_prompts[args.skip:stop]
        if not prompts:
            raise ValueError("selected prompt-manifest slice is empty")
    else:
        prompts = DEFAULT_PROMPTS[:1]
    for prompt in prompts:
        assert_prompt_capture_allowed(prompt)

    try:
        with TargetCaptureHooks(target) as hooks:
            for index, prompt in enumerate(prompts, args.skip + 1):
                capture_adaptive_sequence(
                    run,
                    target,
                    mtp,
                    hooks,
                    tokenizer,
                    prompt,
                    index,
                    source_positions=args.source_positions,
                    label_lookahead=args.label_lookahead,
                )
    finally:
        run.close()

    manifest = build_adaptive_manifest(
        args, target, config, tokenizer, mtp, run, started_at
    )
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    checksum_files = [
        path
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    with (args.output / "SHA256SUMS").open("w") as handle:
        for path in checksum_files:
            handle.write(
                f"{sha256_file(path)}  {path.relative_to(args.output)}\n"
            )
    (args.output / "STOP_BEFORE_TRAINING.json").write_text(
        json.dumps(
            {
                "schema": "harp_rtt_adaptive_capture_stop_v1",
                "run_id": args.run_id,
                "capture_complete": True,
                "audit_complete": False,
                "production_capture_started": args.production_capture,
                "training_started": False,
                "sealed_test_opened": False,
                "instruction": "Run the adaptive blocking auditor; do not train.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "adaptive_trees": run.adaptive_tree_count,
                "mtp_nodes": run.mtp_nodes,
                "mtp_nodes_by_depth": dict(sorted(run.nodes_by_depth.items())),
                "native_mtp_branch_calls": run.native_mtp_calls,
                "legacy_anchor_spines": run.anchor_spine_count,
                "legacy_anchor_spine_nodes": run.anchor_spine_nodes,
                "legacy_anchor_native_mtp_branch_calls": run.anchor_native_mtp_calls,
                "generated_positions": run.generated_positions,
                "production_capture": args.production_capture,
                "sealed_test_opened": False,
                "training_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
