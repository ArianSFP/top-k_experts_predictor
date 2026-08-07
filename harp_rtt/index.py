"""Build the immutable HARP-RTT rich-capture index.

The index stores compact metadata and byte offsets only.  Tensor payloads stay
in the audited Transformers sidecars and are read lazily by ``dataset.py``.
Future labels, acceptance labels, and counterfactual records remain separate
from causal model inputs by construction.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


INDEX_SCHEMA = "harp_rtt_rich_event_index_v1"
COLLECTION_SCHEMA = "harp_rtt_rich_event_index_collection_v1"
SPLITS = {"train", "validation", "calibration", "test"}

TARGET_REQUIRED_ROLES = (
    "post_attention_residual_u",
    "normalized_target_router_input_a",
    "post_moe_residual_xplus",
    "routed_expert_output_delta_r",
    "shared_expert_output_delta_s",
    "raw_target_router_logits",
    "selected_expert_ids",
    "selected_execution_weights",
)
TARGET_OPTIONAL_ROLES = (
    "block_input_residual_x",
    "complete_moe_delta",
    "full_target_router_probabilities",
)
TARGET_ROLES = TARGET_REQUIRED_ROLES + TARGET_OPTIONAL_ROLES

PROMPT_ROLES = (
    "raw_target_router_logits",
    "selected_expert_ids",
    "selected_execution_weights",
)

MTP_REQUIRED_ROLES = (
    "mtp_fused_state",
    "mtp_router_input",
    "mtp_post_ffn_hidden",
    "mtp_vocabulary_head_input",
    "raw_mtp_router_logits",
    "mtp_selected_expert_ids",
    "mtp_selected_execution_weights",
    "vocab_top64_token_ids",
    "vocab_top64_log_probabilities",
)
MTP_OPTIONAL_ROLES = ("full_mtp_router_probabilities",)
MTP_ROLES = MTP_REQUIRED_ROLES + MTP_OPTIONAL_ROLES

ANCHOR_SPINE_SCHEMA = "harp_rtt_legacy_anchor_spine_v1"
ANCHOR_SPINE_REQUIRED_DEPTH = 6
ANCHOR_SPINE_NODE_EVENT = "harp_anchor_spine_node"
ANCHOR_SPINE_READY_EVENT = "harp_anchor_spine_ready"
ANCHOR_SPINE_ROLES = (
    "harp_anchor_mtp_vocabulary_head_input",
    "harp_anchor_raw_mtp_router_logits",
    "harp_anchor_vocab_top64_token_ids",
    "harp_anchor_vocab_top64_log_probabilities",
)
ANCHOR_SPINE_META_COLUMNS = (
    "sequence_index",
    "committed_prefix_position",
    "depth",
    "node_event_id",
    "parent_node_event_id",
    "node_token_id",
    "next_top_token_id",
    "source_ready_event_order",
    "record_valid",
    "exact_committed_h1_root",
    "anchor_local_index",
    "target_position",
    "vocabulary_prediction_position",
    "source_ready_monotonic_ns",
    "tree_identity_hash_i63",
    "spine_identity_hash_i63",
    "token_rank_under_parent",
)
ANCHOR_SPINE_SCALAR_COLUMNS = (
    "local_token_log_probability",
    "path_log_probability",
    "local_token_probability",
    "path_probability",
)

CONDITIONING_CODE = {
    "authoritative_context_equivalent": 0,
    "position_aligned_speculative_context": 1,
    "authoritative_exact_root": 2,
    "counterfactual_teacher_forced": 3,
}

# The first thirteen columns are the frozen v1 layout and must not move: old
# greedy-chain indices are consumed in place. Adaptive captures append the
# audit/structure fields that were not present in the original corpus.
MTP_META_COLUMNS = (
    "sequence_index",
    "committed_prefix_position",
    "target_position",
    "horizon",
    "depth",
    "node_event_id",
    "parent_node_event_id",
    "conditioning_token_id",
    "draft_token_id",
    "source_ready_event_order",
    "conditioning_code",
    "branch_identity_hash_i63",
    "record_valid",
    "node_local_index",
    "structural_validity",
    "feature_available",
    "exact_committed_h1_root",
    "source_ready_monotonic_ns",
    "vocabulary_prediction_position",
    "path_identity_hash_i63",
    "tree_identity_hash_i63",
    "token_rank_under_parent",
)

MTP_SCALAR_COLUMNS = (
    "local_token_log_probability",
    "path_log_probability",
    "path_probability",
    "next_top_token_probability",
    "next_top1_top2_logprob_margin",
    "vocabulary_entropy",
    "vocabulary_top8_mass",
    "mtp_router_entropy",
    "local_token_probability",
)


def _is_adaptive_mtp_node(row: Mapping[str, Any]) -> bool:
    """Return whether *row* carries the explicit adaptive-tree contract."""

    return any(
        key in row
        for key in (
            "tree_id",
            "tree_node_id",
            "node_local_index",
            "parent_local_index",
            "exact_committed_h1_root",
        )
    )


def _contains_forbidden_causal_field(value: object) -> str | None:
    """Find acceptance/realized-future fields nested in a causal node event."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if "accept" in lowered and key != "acceptance_fields_present":
                return key
            if lowered.startswith("realized_") or "future_target_" in lowered:
                return key
            nested = _contains_forbidden_causal_field(child)
            if nested is not None:
                return f"{key}.{nested}"
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            nested = _contains_forbidden_causal_field(child)
            if nested is not None:
                return f"[{index}].{nested}"
    return None


def _canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_i63(value: object) -> int:
    raw = hashlib.sha256(str(value).encode("utf-8")).digest()[:8]
    return int.from_bytes(raw, "little") & ((1 << 63) - 1)


def _fixed_hash(value: str | None) -> bytes:
    if not value:
        return bytes(32)
    raw = bytes.fromhex(value)
    if len(raw) != 32:
        raise ValueError("prefix hashes must be SHA-256 values")
    return raw


def _prefix_hash(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256(b"GCRP2_PREFIX_V1")
    for token in tokens:
        digest.update(struct.pack("<i", int(token)))
    return digest.hexdigest()


def _save(path: Path, values: object, dtype: object | None = None) -> None:
    np.save(path, np.asarray(values, dtype=dtype), allow_pickle=False)


def _descriptor(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dtype": str(row["native_dtype"]),
        "shape": [int(value) for value in row["shape"]],
        "bytes": int(row["payload_bytes"]),
        "payload_file": str(row["payload_file"]),
    }


def _register_descriptor(
    registry: dict[str, dict[str, Any]], role: str, row: Mapping[str, Any]
) -> None:
    value = _descriptor(row)
    previous = registry.get(role)
    if previous is None:
        registry[role] = value
    elif previous != value:
        raise ValueError(f"non-constant descriptor for {role}: {previous} != {value}")


def _validate_offsets(
    offsets: np.ndarray,
    roles: Sequence[str],
    required: Iterable[str],
    kind: str,
) -> None:
    required_columns = [roles.index(role) for role in required]
    if offsets.size and (offsets[:, required_columns] < 0).any():
        row, relative_column = np.argwhere(offsets[:, required_columns] < 0)[0]
        role = required_columns[int(relative_column)]
        raise ValueError(f"missing required {kind} role {roles[role]} at row {row}")


def build_segment(
    segment: Path,
    output: Path,
    split_by_sequence: Mapping[str, str],
) -> dict[str, Any]:
    """Index one already-audited capture segment."""
    output.mkdir(parents=True, exist_ok=False)
    sequences: list[dict[str, Any]] = []
    sequence_index: dict[str, int] = {}
    sequence_end: dict[str, dict[str, Any]] = {}
    target_tokens: dict[int, tuple[int, int, int]] = {}

    target_meta: list[list[int]] = []
    target_prefix: list[bytes] = []
    target_offsets: list[list[int]] = []
    target_available: list[list[bool]] = []
    target_scalars: list[list[float]] = []

    prompt_meta: list[list[int]] = []
    prompt_offsets: list[list[int]] = []

    mtp_meta: list[list[int]] = []
    mtp_prefix: list[bytes] = []
    mtp_exact_prefix: list[bytes] = []
    mtp_ready_timestamp_utc: list[str] = []
    mtp_offsets: list[list[int]] = []
    mtp_available: list[list[bool]] = []
    mtp_scalars: list[list[float]] = []
    mtp_acceptance: list[list[int]] = []
    mtp_event_to_row: dict[int, int] = {}
    mtp_event_structure: dict[int, tuple[int, int, int, int, str | None]] = {}
    adaptive_tree_counts: defaultdict[tuple[int, str], int] = defaultdict(int)
    adaptive_mtp_events: set[int] = set()
    acceptance_events: set[int] = set()
    cycle_depth_event: dict[tuple[int, int, int], int] = {}

    anchor_meta: list[list[int]] = []
    anchor_prefix: list[bytes] = []
    anchor_exact_prefix: list[bytes] = []
    anchor_ready_timestamp_utc: list[str] = []
    anchor_offsets: list[list[int]] = []
    anchor_available: list[list[bool]] = []
    anchor_scalars: list[list[float]] = []
    anchor_paths: list[tuple[int, ...]] = []
    anchor_event_to_row: dict[int, int] = {}
    anchor_tree_rows: defaultdict[tuple[int, str], list[int]] = defaultdict(list)
    anchor_ready_by_tree: dict[tuple[int, str], dict[str, Any]] = {}
    adaptive_root_by_tree: dict[tuple[int, str], tuple[int, int, int, str]] = {}
    target_event_by_position: dict[tuple[int, int], int] = {}

    tensor_parent: dict[int, tuple[str, int]] = {}
    descriptors: dict[str, dict[str, Any]] = {}
    counters: defaultdict[str, int] = defaultdict(int)

    def sequence_number(sequence_id: str) -> int:
        if sequence_id not in sequence_index:
            raise ValueError(f"event references unknown sequence {sequence_id}")
        return sequence_index[sequence_id]

    with (segment / "events.jsonl").open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{segment}/events.jsonl:{line_number}: {exc}") from exc
            kind = str(row["event"])
            counters[kind] += 1

            if kind == "sequence_start":
                sid = str(row["sequence_id"])
                if sid in sequence_index:
                    raise ValueError(f"duplicate sequence {sid}")
                split = split_by_sequence.get(sid)
                if split not in SPLITS:
                    raise ValueError(f"sequence {sid} has no valid frozen split")
                sequence_index[sid] = len(sequences)
                sequences.append(
                    {
                        "sequence_id": sid,
                        "request_id": str(row["request_id"]),
                        "dataset_source": row.get("dataset_source"),
                        "domain_label": row.get("domain_label"),
                        "language_label": row.get("language_label"),
                        "split_group_id": row.get("split_group_id"),
                        "prompt_hash": row.get("prompt_hash"),
                        "split": split,
                    }
                )
                continue

            if kind == "sequence_end":
                sequence_end[str(row["sequence_id"])] = row
                continue

            if kind == "target_token":
                event_id = int(row["event_id"])
                target_identity = (
                    sequence_number(str(row["sequence_id"])),
                    int(row["absolute_sequence_position"]),
                    int(row["committed_input_token_id"]),
                )
                target_tokens[event_id] = target_identity
                target_event_by_position[target_identity[:2]] = event_id
                continue

            if kind in {"target_layer", "prompt_route_layer"}:
                token_event = int(row["target_token_event_id"])
                if token_event not in target_tokens:
                    raise ValueError(f"{kind} precedes target token {token_event}")
                sequence, position, token_id = target_tokens[token_event]
                if sequence != sequence_number(str(row["sequence_id"])):
                    raise ValueError("target token/layer sequence mismatch")
                if position != int(row["absolute_sequence_position"]):
                    raise ValueError("target token/layer position mismatch")
                event_id = int(row["event_id"])
                if kind == "target_layer":
                    index = len(target_meta)
                    tensor_parent[event_id] = (kind, index)
                    target_meta.append(
                        [
                            sequence,
                            position,
                            int(row["target_layer"]),
                            token_id,
                            event_id,
                            int(bool(row.get("record_valid", True))),
                        ]
                    )
                    target_prefix.append(_fixed_hash(row.get("authoritative_prefix_hash")))
                    target_offsets.append([-1] * len(TARGET_ROLES))
                    target_available.append([False] * len(TARGET_ROLES))
                    target_scalars.append(
                        [
                            float(row.get("post_mixer_pre_moe_residual_rms", 0.0)),
                            float(row.get("cos_routed_output_to_pre_moe_residual", 0.0)),
                            float(row.get("cos_shared_output_to_pre_moe_residual", 0.0)),
                            float(row.get("shared_expert_gate_scalar", 0.0)),
                        ]
                    )
                else:
                    index = len(prompt_meta)
                    tensor_parent[event_id] = (kind, index)
                    prompt_meta.append(
                        [sequence, position, int(row["target_layer"]), token_id, event_id]
                    )
                    prompt_offsets.append([-1] * len(PROMPT_ROLES))
                continue

            if kind == "mtp_node":
                sid = str(row["sequence_id"])
                sequence = sequence_number(sid)
                cycle = int(row.get("verifier_cycle_id", 0))
                depth = int(row.get("engine_draft_depth", row.get("depth", 0)))
                if depth <= 0:
                    raise ValueError("MTP node depth must be positive")
                event_id = int(row["event_id"])
                adaptive = _is_adaptive_mtp_node(row)
                tree_id = str(row["tree_id"]) if adaptive and "tree_id" in row else None
                if adaptive:
                    if tree_id is None or "node_local_index" not in row:
                        raise ValueError(
                            f"adaptive MTP node {event_id} lacks tree/local identity"
                        )
                    local_index = int(row["node_local_index"])
                    expected_local_index = adaptive_tree_counts[(sequence, tree_id)]
                    if local_index != expected_local_index:
                        raise ValueError(
                            f"adaptive MTP tree {tree_id} local index {local_index} "
                            f"does not follow {expected_local_index}"
                        )
                    if row.get("acceptance_fields_present") is not False:
                        raise ValueError(
                            f"adaptive MTP node {event_id} does not declare acceptance absent"
                        )
                    forbidden = _contains_forbidden_causal_field(row)
                    if forbidden is not None:
                        raise ValueError(
                            f"adaptive MTP node {event_id} leaks label/future field {forbidden}"
                        )
                    if not row.get("exact_prefix_hash"):
                        raise ValueError(
                            f"adaptive MTP node {event_id} lacks exact-prefix hash"
                        )
                else:
                    local_index = int(row.get("node_local_index", -1))
                explicit_parent = row.get("parent_node_event_id")
                # The depth fallback exists solely for the historical greedy
                # corpus. Applying it to a branching tree silently attaches a
                # child to the last sibling at depth-1 and destroys topology.
                if explicit_parent is None and depth > 1 and not adaptive:
                    explicit_parent = cycle_depth_event.get((sequence, cycle, depth - 1))
                if adaptive and depth == 1 and explicit_parent is not None:
                    raise ValueError(f"adaptive MTP root {event_id} cannot have a parent")
                if adaptive and depth > 1 and explicit_parent is None:
                    raise ValueError(
                        f"adaptive MTP node {event_id} requires an explicit parent"
                    )
                parent_event = -1 if explicit_parent is None else int(explicit_parent)
                if adaptive and parent_event >= 0:
                    parent_structure = mtp_event_structure.get(parent_event)
                    if parent_structure is None:
                        raise ValueError(
                            f"adaptive MTP node {event_id} references a parent not yet emitted"
                        )
                    parent_sequence, parent_cycle, parent_depth, _, parent_tree = (
                        parent_structure
                    )
                    if (
                        parent_sequence != sequence
                        or parent_cycle != cycle
                        or parent_tree != tree_id
                        or parent_depth + 1 != depth
                    ):
                        raise ValueError(
                            f"adaptive MTP node {event_id} has a cross-tree or wrong-depth parent"
                        )
                prefix_tokens = [int(value) for value in row.get("draft_prefix_token_ids", [])]
                node_token = row.get("node_token_id")
                if node_token is None and prefix_tokens:
                    node_token = prefix_tokens[-1]
                if node_token is None:
                    raise ValueError(f"MTP node {event_id} has no conditioning token")
                branch_value = row.get(
                    "branch_id",
                    row.get("branch_path_id", row.get("draft_branch_id", 0)),
                )
                path_value = row.get("path_id", row.get("branch_path_id", branch_value))
                conditioning = str(row.get("conditioning_class", "authoritative_context_equivalent"))
                conditioning_code = CONDITIONING_CODE.setdefault(
                    conditioning, max(CONDITIONING_CODE.values()) + 1
                )
                index = len(mtp_meta)
                tensor_parent[event_id] = (kind, index)
                mtp_event_to_row[event_id] = index
                mtp_event_structure[event_id] = (
                    sequence,
                    cycle,
                    depth,
                    int(row["committed_prefix_position"]),
                    tree_id,
                )
                cycle_depth_event[(sequence, cycle, depth)] = event_id
                if adaptive:
                    adaptive_tree_counts[(sequence, tree_id)] += 1
                    adaptive_mtp_events.add(event_id)
                structural_validity = bool(row.get("structural_validity", True))
                feature_available = bool(row.get("feature_available", True))
                record_valid = (
                    bool(row.get("record_valid", True))
                    and structural_validity
                    and feature_available
                )
                exact_root = bool(row.get("exact_committed_h1_root", depth == 1))
                if adaptive and exact_root != (depth == 1):
                    raise ValueError(
                        f"adaptive MTP node {event_id} has inconsistent exact-H1 marker"
                    )
                target_position = int(
                    row.get("gcrp_target_position", row.get("router_state_position"))
                )
                mtp_meta.append(
                    [
                        sequence,
                        int(row["committed_prefix_position"]),
                        target_position,
                        int(row.get("gcrp_horizon", depth)),
                        depth,
                        event_id,
                        parent_event,
                        int(node_token),
                        int(row.get("draft_token_id", -1)),
                        int(row.get("source_ready_event_order", event_id)),
                        conditioning_code,
                        _stable_i63(branch_value),
                        int(record_valid),
                        local_index,
                        int(structural_validity),
                        int(feature_available),
                        int(exact_root),
                        int(row.get("source_ready_monotonic_ns", 0)),
                        int(row.get("vocabulary_prediction_position", target_position + 1)),
                        _stable_i63(path_value),
                        _stable_i63(tree_id if tree_id is not None else cycle),
                        int(row.get("token_rank_under_parent", -1)),
                    ]
                )
                if adaptive and depth == 1:
                    tree_key = (sequence, str(tree_id))
                    if tree_key in adaptive_root_by_tree:
                        raise ValueError(f"adaptive tree {tree_id} has multiple roots")
                    adaptive_root_by_tree[tree_key] = (
                        event_id,
                        int(node_token),
                        int(row["committed_prefix_position"]),
                        str(row.get("authoritative_prefix_hash", "")),
                    )
                mtp_prefix.append(_fixed_hash(row.get("authoritative_prefix_hash")))
                mtp_exact_prefix.append(
                    _fixed_hash(
                        row.get(
                            "exact_prefix_hash",
                            row.get(
                                "mtp_conditioning_prefix_hash",
                                row.get("authoritative_prefix_hash"),
                            ),
                        )
                    )
                )
                mtp_ready_timestamp_utc.append(
                    str(row.get("source_ready_timestamp_utc", ""))
                )
                mtp_offsets.append([-1] * len(MTP_ROLES))
                mtp_available.append([False] * len(MTP_ROLES))
                mtp_scalars.append(
                    [
                        float(
                            row.get(
                                "node_local_token_log_probability",
                                row.get("draft_token_log_probability", 0.0),
                            )
                        ),
                        float(
                            row.get(
                                "path_log_probability",
                                row.get("cumulative_draft_logprobability", 0.0),
                            )
                        ),
                        float(
                            row.get(
                                "path_probability",
                                row.get("branch_probability", 1.0),
                            )
                        ),
                        float(row.get("top_token_probability", 0.0)),
                        float(row.get("top1_top2_logprob_margin", 0.0)),
                        float(row.get("vocabulary_entropy", 0.0)),
                        float(row.get("vocabulary_top8_mass", 0.0)),
                        float(row.get("mtp_router_entropy", 0.0)),
                        float(
                            row.get(
                                "node_local_token_probability",
                                np.exp(
                                    float(
                                        row.get(
                                            "node_local_token_log_probability",
                                            row.get("draft_token_log_probability", 0.0),
                                        )
                                    )
                                ),
                            )
                        ),
                    ]
                )
                mtp_acceptance.append([0, 0])
                continue

            if kind == ANCHOR_SPINE_NODE_EVENT:
                sid = str(row["sequence_id"])
                sequence = sequence_number(sid)
                event_id = int(row["event_id"])
                tree_id = str(row.get("tree_id", ""))
                spine_id = str(row.get("anchor_spine_id", ""))
                tree_key = (sequence, tree_id)
                rows = anchor_tree_rows[tree_key]
                local = int(row.get("anchor_local_index", -1))
                depth = int(row.get("engine_draft_depth", -1))
                if row.get("anchor_spine_schema") != ANCHOR_SPINE_SCHEMA:
                    raise ValueError(f"anchor spine node {event_id} has an unknown schema")
                forbidden = _contains_forbidden_causal_field(row)
                if forbidden is not None:
                    raise ValueError(
                        f"anchor spine node {event_id} leaks label/future field {forbidden}"
                    )
                required_flags = {
                    "acceptance_fields_present": False,
                    "label_records_present": False,
                    "anchor_only": True,
                    "adaptive_model_input": False,
                    "consumes_adaptive_node_budget": False,
                    "continues_through_eos": True,
                    "eos_termination_applies": False,
                    "record_valid": True,
                    "structural_validity": True,
                    "feature_available": True,
                }
                for field, expected in required_flags.items():
                    if row.get(field) is not expected:
                        raise ValueError(
                            f"anchor spine node {event_id} has invalid {field}"
                        )
                if int(row.get("required_anchor_depth", -1)) != ANCHOR_SPINE_REQUIRED_DEPTH:
                    raise ValueError(f"anchor spine node {event_id} has wrong required depth")
                if local != len(rows) or depth != local + 1:
                    raise ValueError(
                        f"anchor spine {spine_id} is not contiguous exact H1-H6"
                    )
                if not tree_id or not spine_id:
                    raise ValueError(f"anchor spine node {event_id} lacks tree/spine identity")
                path = tuple(int(value) for value in row.get("branch_path_token_ids", []))
                path_logps = tuple(
                    float(value)
                    for value in row.get("branch_path_token_log_probabilities", [])
                )
                if len(path) != depth or len(path_logps) != depth:
                    raise ValueError(f"anchor spine node {event_id} path/depth mismatch")
                node_token = int(row["node_token_id"])
                if path[-1] != node_token:
                    raise ValueError(f"anchor spine node {event_id} token/path mismatch")
                parent_value = row.get("parent_anchor_node_event_id")
                parent_local_value = row.get("parent_anchor_local_index")
                rank_value = row.get("token_rank_under_parent")
                local_logp = float(row["node_local_token_log_probability"])
                path_logp = float(row["path_log_probability"])
                if depth == 1:
                    if (
                        parent_value is not None
                        or parent_local_value is not None
                        or rank_value is not None
                        or row.get("exact_committed_h1_root") is not True
                        or abs(local_logp) > 1e-7
                        or abs(path_logp) > 1e-7
                    ):
                        raise ValueError(f"anchor spine {spine_id} has invalid exact-H1 root")
                    parent_event = -1
                else:
                    previous_row = rows[-1]
                    previous_meta = anchor_meta[previous_row]
                    parent_event = int(parent_value) if parent_value is not None else -1
                    if (
                        int(parent_local_value) != local - 1
                        or parent_event != previous_meta[3]
                        or int(rank_value) != 0
                        or node_token != previous_meta[6]
                        or path[:-1] != anchor_paths[previous_row]
                    ):
                        raise ValueError(
                            f"anchor spine {spine_id} is not a parent-coherent local-top1 chain"
                        )
                    expected_path_logp = anchor_scalars[previous_row][1] + local_logp
                    if abs(path_logp - expected_path_logp) > 2e-5:
                        raise ValueError(f"anchor spine node {event_id} path log-probability mismatch")
                local_probability = float(row["node_local_token_probability"])
                path_probability = float(row["path_probability"])
                if (
                    abs(local_probability - float(np.exp(local_logp))) > 2e-5
                    or abs(path_probability - float(np.exp(path_logp))) > 2e-5
                ):
                    raise ValueError(f"anchor spine node {event_id} probability mismatch")
                prefix_value = str(row.get("authoritative_prefix_hash", ""))
                exact_prefix_value = str(
                    row.get("exact_prefix_hash", row.get("mtp_conditioning_prefix_hash", ""))
                )
                if exact_prefix_value != str(row.get("mtp_conditioning_prefix_hash", "")):
                    raise ValueError(f"anchor spine node {event_id} exact-prefix hash mismatch")
                target_position = int(row["gcrp_target_position"])
                base_position = int(row["committed_prefix_position"])
                if target_position != base_position + depth:
                    raise ValueError(f"anchor spine node {event_id} target-position mismatch")
                index = len(anchor_meta)
                tensor_parent[event_id] = (kind, index)
                anchor_event_to_row[event_id] = index
                rows.append(index)
                anchor_paths.append(path)
                anchor_meta.append(
                    [
                        sequence,
                        base_position,
                        depth,
                        event_id,
                        parent_event,
                        node_token,
                        int(row["vocabulary_top_token_id"]),
                        int(row["source_ready_event_order"]),
                        1,
                        int(depth == 1),
                        local,
                        target_position,
                        int(row["vocabulary_prediction_position"]),
                        int(row.get("source_ready_monotonic_ns", 0)),
                        _stable_i63(tree_id),
                        _stable_i63(spine_id),
                        -1 if rank_value is None else int(rank_value),
                    ]
                )
                anchor_prefix.append(_fixed_hash(prefix_value))
                anchor_exact_prefix.append(_fixed_hash(exact_prefix_value))
                anchor_ready_timestamp_utc.append(
                    str(row.get("source_ready_timestamp_utc", ""))
                )
                anchor_offsets.append([-1] * len(ANCHOR_SPINE_ROLES))
                anchor_available.append([False] * len(ANCHOR_SPINE_ROLES))
                anchor_scalars.append(
                    [local_logp, path_logp, local_probability, path_probability]
                )
                continue

            if kind == ANCHOR_SPINE_READY_EVENT:
                sequence = sequence_number(str(row["sequence_id"]))
                tree_id = str(row.get("tree_id", ""))
                tree_key = (sequence, tree_id)
                if tree_key in anchor_ready_by_tree:
                    raise ValueError(f"adaptive tree {tree_id} has duplicate anchor readiness")
                anchor_ready_by_tree[tree_key] = row
                continue

            if kind == "mtp_acceptance_label":
                event_id = int(row["mtp_node_event_id"])
                if event_id not in mtp_event_to_row:
                    raise ValueError(f"acceptance label references missing MTP node {event_id}")
                if event_id in acceptance_events:
                    raise ValueError(f"duplicate acceptance label for MTP node {event_id}")
                if event_id in adaptive_mtp_events and row.get("label_only") is not True:
                    raise ValueError(
                        f"adaptive acceptance label for MTP node {event_id} is not label-only"
                    )
                acceptance_events.add(event_id)
                mtp_acceptance[mtp_event_to_row[event_id]] = [
                    int(bool(row.get("draft_token_accepted", row.get("accepted_prefix_label", False)))),
                    int(bool(row.get("acceptance_label_valid", False))),
                ]
                continue

            if kind == "tensor":
                parent_event = int(row["parent_event_id"])
                parent = tensor_parent.get(parent_event)
                if parent is None:
                    continue
                parent_kind, index = parent
                role = str(row["tensor_role"])
                _register_descriptor(descriptors, role, row)
                offset = int(row["payload_offset"])
                if parent_kind == "target_layer" and role in TARGET_ROLES:
                    column = TARGET_ROLES.index(role)
                    target_offsets[index][column] = offset
                    target_available[index][column] = True
                elif parent_kind == "prompt_route_layer" and role in PROMPT_ROLES:
                    prompt_offsets[index][PROMPT_ROLES.index(role)] = offset
                elif parent_kind == "mtp_node" and role in MTP_ROLES:
                    column = MTP_ROLES.index(role)
                    mtp_offsets[index][column] = offset
                    mtp_available[index][column] = True
                elif parent_kind == ANCHOR_SPINE_NODE_EVENT and role in ANCHOR_SPINE_ROLES:
                    column = ANCHOR_SPINE_ROLES.index(role)
                    anchor_offsets[index][column] = offset
                    anchor_available[index][column] = True

    missing_adaptive_labels = adaptive_mtp_events - acceptance_events
    if missing_adaptive_labels:
        first_missing = min(missing_adaptive_labels)
        raise ValueError(
            f"adaptive MTP node {first_missing} has no separate acceptance label"
        )

    adaptive_tree_keys = set(adaptive_root_by_tree)
    anchor_tree_keys = set(anchor_tree_rows)
    ready_tree_keys = set(anchor_ready_by_tree)
    if adaptive_tree_keys:
        if anchor_tree_keys != adaptive_tree_keys or ready_tree_keys != adaptive_tree_keys:
            missing_nodes = sorted(adaptive_tree_keys - anchor_tree_keys)
            missing_ready = sorted(adaptive_tree_keys - ready_tree_keys)
            extra = sorted((anchor_tree_keys | ready_tree_keys) - adaptive_tree_keys)
            raise ValueError(
                "adaptive/anchor-spine tree mismatch; "
                f"missing_nodes={missing_nodes[:2]} missing_ready={missing_ready[:2]} "
                f"extra={extra[:2]}"
            )
    elif anchor_tree_keys or ready_tree_keys:
        raise ValueError("legacy-only segment unexpectedly contains anchor-spine events")

    for tree_key in sorted(adaptive_tree_keys):
        sequence, tree_id = tree_key
        rows = anchor_tree_rows[tree_key]
        ready = anchor_ready_by_tree[tree_key]
        root_event, adaptive_h1_token, base_position, source_hash = adaptive_root_by_tree[
            tree_key
        ]
        if len(rows) != ANCHOR_SPINE_REQUIRED_DEPTH:
            raise ValueError(
                f"adaptive tree {tree_id} anchor spine has {len(rows)} rows, expected 6"
            )
        expected_events = [anchor_meta[index][3] for index in rows]
        required_ready_flags = {
            "anchor_spine_schema": ANCHOR_SPINE_SCHEMA,
            "synchronous_before_target_h1_execution": True,
            "local_top1_parent_coherent": True,
            "continues_through_eos": True,
            "eos_termination_applies": False,
            "consumes_adaptive_node_budget": False,
            "adaptive_model_input": False,
            "expansion_uses_realized_future": False,
            "expansion_uses_acceptance": False,
            "labels_emitted": False,
        }
        for field, expected in required_ready_flags.items():
            if ready.get(field) != expected:
                raise ValueError(f"anchor spine {tree_id} readiness has invalid {field}")
        if (
            int(ready.get("required_depth", -1)) != ANCHOR_SPINE_REQUIRED_DEPTH
            or int(ready.get("node_count", -1)) != ANCHOR_SPINE_REQUIRED_DEPTH
            or [int(value) for value in ready.get("anchor_node_event_ids", [])]
            != expected_events
            or int(ready.get("exact_h1_root_event_id", -1)) != expected_events[0]
            or int(ready.get("exact_h1_token_id", -1)) != adaptive_h1_token
            or int(anchor_meta[rows[0]][5]) != adaptive_h1_token
            or int(ready.get("committed_prefix_position", -1)) != base_position
            or str(ready.get("authoritative_prefix_hash", "")) != source_hash
        ):
            raise ValueError(f"adaptive tree {tree_id} anchor readiness/root mismatch")
        ready_event = int(ready["event_id"])
        target_h1_event = target_event_by_position.get((sequence, base_position + 1))
        if target_h1_event is None or ready_event >= target_h1_event:
            raise ValueError(f"anchor spine {tree_id} was not ready before target H1 execution")
        end = sequence_end.get(str(sequences[sequence]["sequence_id"]))
        if end is None:
            raise ValueError(f"anchor spine {tree_id} sequence has no sequence_end")
        full_tokens = [int(value) for value in end.get("full_committed_token_ids", [])]
        if not full_tokens:
            full_tokens = [int(value) for value in end["prompt_token_ids"]] + [
                int(value) for value in end["committed_generated_token_ids"]
            ]
        expected_source_hash = _prefix_hash(full_tokens[: base_position + 1])
        if expected_source_hash != source_hash:
            raise ValueError(f"anchor spine {tree_id} authoritative prefix hash mismatch")
        for row_index in rows:
            metadata = anchor_meta[row_index]
            depth = metadata[2]
            expected_exact = _prefix_hash(
                full_tokens[: base_position + 1] + list(anchor_paths[row_index])
            )
            if bytes(anchor_prefix[row_index]).hex() != expected_source_hash:
                raise ValueError(f"anchor spine {tree_id} source hashes are not coherent")
            if bytes(anchor_exact_prefix[row_index]).hex() != expected_exact:
                raise ValueError(
                    f"anchor spine {tree_id} H{depth} conditioning hash mismatch"
                )
            if metadata[7] > ready_event or metadata[3] > ready_event:
                raise ValueError(f"anchor spine {tree_id} readiness precedes a node")

    for sequence in sequences:
        sid = str(sequence["sequence_id"])
        end = sequence_end.get(sid)
        if end is None:
            raise ValueError(f"sequence {sid} has no sequence_end")
        prompt_ids = [int(value) for value in end["prompt_token_ids"]]
        generated_ids = [int(value) for value in end["committed_generated_token_ids"]]
        sequence.update(
            {
                "prompt_length": int(end["prompt_length"]),
                "generated_length": int(end["generated_length"]),
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated_ids,
                "termination_reason": str(end["termination_reason"]),
            }
        )

    target_offsets_array = np.asarray(target_offsets, dtype=np.int64)
    prompt_offsets_array = np.asarray(prompt_offsets, dtype=np.int64)
    mtp_offsets_array = np.asarray(mtp_offsets, dtype=np.int64)
    anchor_offsets_array = np.asarray(anchor_offsets, dtype=np.int64).reshape(
        (-1, len(ANCHOR_SPINE_ROLES))
    )
    _validate_offsets(
        target_offsets_array, TARGET_ROLES, TARGET_REQUIRED_ROLES, "target"
    )
    _validate_offsets(prompt_offsets_array, PROMPT_ROLES, PROMPT_ROLES, "prompt")
    _validate_offsets(mtp_offsets_array, MTP_ROLES, MTP_REQUIRED_ROLES, "MTP")
    _validate_offsets(
        anchor_offsets_array,
        ANCHOR_SPINE_ROLES,
        ANCHOR_SPINE_ROLES,
        "anchor spine",
    )

    _save(output / "target_meta.npy", target_meta, np.int64)
    _save(output / "target_prefix_sha256.npy", target_prefix, "S32")
    _save(output / "target_offsets.npy", target_offsets_array, np.int64)
    _save(output / "target_available.npy", target_available, np.bool_)
    _save(output / "target_scalars.npy", target_scalars, np.float32)
    _save(output / "prompt_meta.npy", prompt_meta, np.int64)
    _save(output / "prompt_offsets.npy", prompt_offsets_array, np.int64)
    mtp_meta_array = np.asarray(mtp_meta, dtype=np.int64).reshape(
        (-1, len(MTP_META_COLUMNS))
    )
    mtp_scalars_array = np.asarray(mtp_scalars, dtype=np.float32).reshape(
        (-1, len(MTP_SCALAR_COLUMNS))
    )
    _save(output / "mtp_meta.npy", mtp_meta_array, np.int64)
    _save(output / "mtp_prefix_sha256.npy", mtp_prefix, "S32")
    _save(output / "mtp_exact_prefix_sha256.npy", mtp_exact_prefix, "S32")
    maximum_timestamp_length = max(
        (len(value) for value in mtp_ready_timestamp_utc), default=1
    )
    _save(
        output / "mtp_source_ready_timestamp_utc.npy",
        mtp_ready_timestamp_utc,
        f"<U{maximum_timestamp_length}",
    )
    _save(output / "mtp_offsets.npy", mtp_offsets_array, np.int64)
    _save(output / "mtp_available.npy", mtp_available, np.bool_)
    _save(output / "mtp_scalars.npy", mtp_scalars_array, np.float32)
    _save(output / "mtp_acceptance_labels.npy", mtp_acceptance, np.uint8)
    anchor_meta_array = np.asarray(anchor_meta, dtype=np.int64).reshape(
        (-1, len(ANCHOR_SPINE_META_COLUMNS))
    )
    anchor_scalars_array = np.asarray(anchor_scalars, dtype=np.float32).reshape(
        (-1, len(ANCHOR_SPINE_SCALAR_COLUMNS))
    )
    _save(output / "anchor_spine_meta.npy", anchor_meta_array, np.int64)
    _save(output / "anchor_spine_prefix_sha256.npy", anchor_prefix, "S32")
    _save(output / "anchor_spine_exact_prefix_sha256.npy", anchor_exact_prefix, "S32")
    anchor_timestamp_length = max(
        (len(value) for value in anchor_ready_timestamp_utc), default=1
    )
    _save(
        output / "anchor_spine_source_ready_timestamp_utc.npy",
        anchor_ready_timestamp_utc,
        f"<U{anchor_timestamp_length}",
    )
    _save(output / "anchor_spine_offsets.npy", anchor_offsets_array, np.int64)
    _save(output / "anchor_spine_available.npy", anchor_available, np.bool_)
    _save(output / "anchor_spine_scalars.npy", anchor_scalars_array, np.float32)

    manifest = {
        "schema": INDEX_SCHEMA,
        "segment": segment.name,
        "source_segment": str(segment),
        "sequences": sequences,
        "target_roles": list(TARGET_ROLES),
        "target_required_roles": list(TARGET_REQUIRED_ROLES),
        "prompt_roles": list(PROMPT_ROLES),
        "mtp_roles": list(MTP_ROLES),
        "mtp_required_roles": list(MTP_REQUIRED_ROLES),
        "mtp_meta_columns": list(MTP_META_COLUMNS),
        "mtp_scalar_columns": list(MTP_SCALAR_COLUMNS),
        "anchor_spine_roles": list(ANCHOR_SPINE_ROLES),
        "anchor_spine_required_roles": list(ANCHOR_SPINE_ROLES),
        "anchor_spine_meta_columns": list(ANCHOR_SPINE_META_COLUMNS),
        "anchor_spine_scalar_columns": list(ANCHOR_SPINE_SCALAR_COLUMNS),
        "tensor_descriptors": descriptors,
        "counts": dict(sorted(counters.items())),
        "target_layer_rows": len(target_meta),
        "prompt_layer_rows": len(prompt_meta),
        "mtp_nodes": len(mtp_meta),
        "adaptive_mtp_nodes": len(adaptive_mtp_events),
        "legacy_mtp_nodes": len(mtp_meta) - len(adaptive_mtp_events),
        "anchor_spine_contract": bool(adaptive_mtp_events),
        "anchor_spine_schema": (
            ANCHOR_SPINE_SCHEMA if adaptive_mtp_events else None
        ),
        "anchor_spine_required_depth": (
            ANCHOR_SPINE_REQUIRED_DEPTH if adaptive_mtp_events else 0
        ),
        "anchor_spines": len(anchor_tree_rows),
        "anchor_spine_nodes": len(anchor_meta),
        "anchor_spine_labels": 0,
        "anchor_spine_consumes_adaptive_node_budget": False,
        "anchor_spine_enters_adaptive_model_inputs": False,
        "anchor_spine_continues_through_eos": bool(adaptive_mtp_events),
        "decoding_profile": (
            "exact_h1_native_mtp_adaptive_h2_h4"
            if adaptive_mtp_events
            else "token_end_greedy_argmax"
        ),
        "adaptive_mtp_tree": bool(adaptive_mtp_events),
        "acceptance_is_label_only": True,
    }
    manifest_path = output / "index_manifest.json"
    manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    hashes = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "INDEX_SHA256SUMS.json"
    }
    (output / "INDEX_SHA256SUMS.json").write_text(
        _canonical_json(hashes) + "\n", encoding="utf-8"
    )
    return manifest


def build_collection(corpus: Path, split_manifest: Path, output: Path) -> dict[str, Any]:
    split_by_sequence: dict[str, str] = {}
    with split_manifest.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row["sequence_id"])
            split = str(row["split"])
            if split not in SPLITS:
                raise ValueError(f"{split_manifest}:{line_number}: invalid split {split}")
            if sid in split_by_sequence:
                raise ValueError(f"duplicate split assignment for {sid}")
            split_by_sequence[sid] = split
    output.mkdir(parents=True, exist_ok=False)
    manifests = []
    for segment in sorted((corpus / "segments").iterdir()):
        if not (segment / "events.jsonl").is_file():
            continue
        print(f"indexing {segment.name}", flush=True)
        manifests.append(
            build_segment(segment, output / segment.name, split_by_sequence)
        )
    indexed_sequences = {
        str(sequence["sequence_id"])
        for manifest in manifests
        for sequence in manifest["sequences"]
    }
    if indexed_sequences != set(split_by_sequence):
        missing = sorted(set(split_by_sequence) - indexed_sequences)
        extra = sorted(indexed_sequences - set(split_by_sequence))
        raise ValueError(f"split/index sequence mismatch; missing={missing[:4]} extra={extra[:4]}")
    summary = {
        "schema": COLLECTION_SCHEMA,
        "source_corpus": str(corpus),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": _sha256(split_manifest),
        "segments": len(manifests),
        "sequences": len(indexed_sequences),
        "target_layer_rows": sum(int(item["target_layer_rows"]) for item in manifests),
        "prompt_layer_rows": sum(int(item["prompt_layer_rows"]) for item in manifests),
        "mtp_nodes": sum(int(item["mtp_nodes"]) for item in manifests),
        "anchor_spines": sum(int(item["anchor_spines"]) for item in manifests),
        "anchor_spine_nodes": sum(
            int(item["anchor_spine_nodes"]) for item in manifests
        ),
        "test_indexed_but_not_authorized_for_loading": True,
    }
    (output / "INDEX_SUMMARY.json").write_text(
        _canonical_json(summary) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        _canonical_json(build_collection(args.corpus, args.split_manifest, args.output))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
