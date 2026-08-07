#!/usr/bin/env python3
"""Blocking audit for event-aligned Transformers GCRP-2R pilot traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np
import torch


PREFIX_DOMAIN = b"GCRP2_PREFIX_V1"
TARGET_MAIN_REQUIRED = {
    "post_attention_residual_u": (2048,),
    "normalized_target_router_input_a": (2048,),
    "post_moe_residual_xplus": (2048,),
    "routed_expert_output_delta_r": (2048,),
    "shared_expert_output_delta_s": (2048,),
    "raw_target_router_logits": (256,),
    "full_target_router_probabilities": (256,),
    "selected_expert_ids": (8,),
    "selected_execution_weights": (8,),
}
TARGET_AUDIT_REQUIRED = {
    "block_input_residual_x": (2048,),
    "complete_moe_delta": (2048,),
    "candidate_unweighted_output_vectors": (8, 2048),
    "candidate_weighted_output_vectors": (8, 2048),
    "shared_expert_output_before_gate": (2048,),
    "shared_expert_gate_logit": (1,),
}
PROMPT_ROUTE_REQUIRED = {
    "raw_target_router_logits": (256,),
    "full_target_router_probabilities": (256,),
    "selected_expert_ids": (8,),
    "selected_execution_weights": (8,),
}
MTP_REQUIRED = {
    "mtp_fused_state": (2048,),
    "mtp_router_input": (2048,),
    "mtp_post_ffn_hidden": (2048,),
    "mtp_vocabulary_head_input": (2048,),
    "raw_mtp_router_logits": (256,),
    "full_mtp_router_probabilities": (256,),
    "mtp_selected_expert_ids": (8,),
    "mtp_selected_execution_weights": (8,),
    "vocab_top64_token_ids": (64,),
    "vocab_top64_log_probabilities": (64,),
}
DTYPES = {
    "bf16": (np.dtype("<u2"), "bf16"),
    "float16": (np.dtype("<f2"), "float"),
    "float32": (np.dtype("<f4"), "float"),
    "float64": (np.dtype("<f8"), "float"),
    "int64": (np.dtype("<i8"), "int"),
    "int32": (np.dtype("<i4"), "int"),
    "int16": (np.dtype("<i2"), "int"),
    "uint8": (np.dtype("u1"), "int"),
    "bool": (np.dtype("u1"), "int"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prefix_hash(tokens: list[int]) -> str:
    digest = hashlib.sha256(PREFIX_DOMAIN)
    for token in tokens:
        digest.update(struct.pack("<i", int(token)))
    return digest.hexdigest()


class AuditError(ValueError):
    pass


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = json.loads((root / "run_manifest.json").read_text())
        self.rows = [
            json.loads(line)
            for line in (root / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.by_id = {int(row["event_id"]): row for row in self.rows}
        self.tensors: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        self.maps: dict[Path, np.memmap] = {}
        for row in self.rows:
            if row["event"] == "tensor":
                parent = int(row["parent_event_id"])
                role = str(row["tensor_role"])
                if role in self.tensors[parent]:
                    raise AuditError(f"duplicate tensor role {role} for event {parent}")
                self.tensors[parent][role] = row

    def tensor(self, parent: int, role: str) -> np.ndarray:
        row = self.tensors[parent][role]
        name = row["native_dtype"]
        if name not in DTYPES:
            raise AuditError(f"unsupported dtype {name}")
        dtype, category = DTYPES[name]
        path = self.root / row["payload_file"]
        mapping = self.maps.get(path)
        if mapping is None:
            if not path.is_file() or path.stat().st_size < 64:
                raise AuditError(f"missing or short sidecar {path}")
            mapping = np.memmap(path, mode="r", dtype=np.uint8)
            self.maps[path] = mapping
        offset = int(row["payload_offset"])
        length = int(row["payload_bytes"])
        if offset < 64 or offset + length > mapping.size:
            raise AuditError(f"bad sidecar range for tensor event {row['event_id']}")
        values = np.frombuffer(memoryview(mapping[offset : offset + length]), dtype=dtype)
        shape = tuple(int(v) for v in row["shape"])
        if values.size != math.prod(shape):
            raise AuditError(f"shape/byte mismatch for tensor event {row['event_id']}")
        values = values.reshape(shape)
        if category == "bf16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        if category in {"float", "bf16"} and not np.isfinite(values).all():
            raise AuditError(f"nonfinite tensor event {row['event_id']}")
        return values

    def close(self) -> None:
        self.maps.clear()


def rel_rms(actual: np.ndarray, expected: np.ndarray) -> float:
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    denominator = max(float(np.sqrt(np.mean(np.asarray(expected, dtype=np.float64) ** 2))), 1e-12)
    return float(np.sqrt(np.mean(delta**2)) / denominator)


def cosine(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, bool]:
    lhs = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denominator <= 1e-12:
        return 0.0, False
    return float(np.dot(lhs, rhs) / denominator), True


def check_checksums(root: Path) -> int:
    rows = (root / "SHA256SUMS").read_text().splitlines()
    for line in rows:
        expected, relative = line.split("  ", 1)
        actual = sha256_file(root / relative)
        if actual != expected:
            raise AuditError(f"checksum mismatch {relative}")
    return len(rows)


def check_tensor_roles(store: Store, parent: int, required: dict[str, tuple[int, ...]]) -> None:
    roles = store.tensors.get(parent, {})
    missing = required.keys() - roles.keys()
    if missing:
        raise AuditError(f"event {parent} missing tensors {sorted(missing)}")
    for role, shape in required.items():
        row_shape = tuple(store.tensors[parent][role]["shape"])
        if row_shape != shape:
            raise AuditError(f"event {parent} role {role} shape {row_shape} != {shape}")
        store.tensor(parent, role)


def _audit_route(
    store: Store,
    event_id: int,
    *,
    logits_role: str,
    probs_role: str,
    ids_role: str,
    weights_role: str,
) -> tuple[float, float, int]:
    logits = store.tensor(event_id, logits_role).astype(np.float32)
    probs = store.tensor(event_id, probs_role).astype(np.float32)
    stable = logits - logits.max()
    recomputed = np.exp(stable)
    recomputed /= recomputed.sum()
    probability_error = float(np.max(np.abs(recomputed - probs)))
    ids = store.tensor(event_id, ids_role).astype(np.int64)
    weights = store.tensor(event_id, weights_role).astype(np.float32)
    if len(set(ids.tolist())) != 8 or np.any(ids < 0) or np.any(ids >= 256):
        raise AuditError(f"invalid selected IDs at event {event_id}")
    predicted = torch.topk(torch.from_numpy(probs), 8).indices.numpy()
    failure = 0
    if set(predicted.tolist()) != set(ids.tolist()):
        selected_min = float(probs[ids].min())
        unselected = np.delete(probs, ids)
        if selected_min + 1e-7 < float(unselected.max()):
            failure = 1
    expected_weights = probs[ids] / probs[ids].sum()
    weight_error = float(np.max(np.abs(expected_weights - weights)))
    return probability_error, weight_error, failure


def audit_target(store: Store) -> dict[str, Any]:
    rows = [row for row in store.rows if row["event"] == "target_layer"]
    prompt_rows = [
        row for row in store.rows if row["event"] == "prompt_route_layer"
    ]
    if not rows or not prompt_rows:
        raise AuditError("missing generated target rows or prompt route history")
    keys = set()
    layers_by_position: dict[tuple[str, int], set[int]] = defaultdict(set)
    prompt_layers_by_position: dict[tuple[str, int], set[int]] = defaultdict(set)
    residual_errors, complete_errors = [], []
    weighted_errors, shared_errors = [], []
    probability_errors, weight_errors, scalar_errors = [], [], []
    selected_set_failures = 0
    audit_layers = set()
    audit_rows = 0
    dtype_counts = Counter()

    for row in rows:
        event_id = int(row["event_id"])
        key = (
            row["sequence_id"],
            int(row["absolute_sequence_position"]),
            int(row["target_layer"]),
        )
        if key in keys:
            raise AuditError(f"duplicate target key {key}")
        keys.add(key)
        layers_by_position[key[:2]].add(key[2])
        if row["cycle_phase"] == "PREFILL_TARGET":
            raise AuditError(f"prefill row entered generated target table {key}")
        if not row.get("record_valid") or not row.get("authoritative_prefix_hash"):
            raise AuditError(f"invalid target row {key}")
        check_tensor_roles(store, event_id, TARGET_MAIN_REQUIRED)
        for role in TARGET_MAIN_REQUIRED:
            dtype_counts[(role, store.tensors[event_id][role]["native_dtype"])] += 1

        u = store.tensor(event_id, "post_attention_residual_u")
        a = store.tensor(event_id, "normalized_target_router_input_a")
        xplus = store.tensor(event_id, "post_moe_residual_xplus")
        routed = store.tensor(event_id, "routed_expert_output_delta_r")
        shared = store.tensor(event_id, "shared_expert_output_delta_s")
        residual_errors.append(rel_rms(xplus, u + routed + shared))
        if np.allclose(a, 0):
            raise AuditError(f"all-zero router input {key}")

        probability_error, weight_error, failure = _audit_route(
            store, event_id,
            logits_role="raw_target_router_logits",
            probs_role="full_target_router_probabilities",
            ids_role="selected_expert_ids",
            weights_role="selected_execution_weights",
        )
        probability_errors.append(probability_error)
        weight_errors.append(weight_error)
        selected_set_failures += failure
        ids = store.tensor(event_id, "selected_expert_ids").astype(np.int64)
        if ids.tolist() != [int(v) for v in row["candidate_expert_ids"]]:
            raise AuditError(f"candidate ID alignment mismatch {key}")

        for ready_id in row.get("ready_mtp_node_event_ids", []):
            ready_id = int(ready_id)
            node = store.by_id.get(ready_id)
            if node is None or node.get("event") != "mtp_node":
                raise AuditError(f"unknown ready MTP node {ready_id} at {key}")
            if node.get("sequence_id") != row["sequence_id"]:
                raise AuditError(f"cross-sequence ready MTP node at {key}")
            if int(node["gcrp_target_position"]) not in (
                key[1] + 1, key[1] + 2
            ):
                raise AuditError(f"misaligned ready MTP node {ready_id} at {key}")
            if ready_id >= event_id:
                raise AuditError(f"noncausal ready MTP node {ready_id} at {key}")

        scalar_arrays = (
            row["candidate_unweighted_output_rms"],
            row["candidate_weighted_output_rms"],
            row["candidate_weighted_output_fraction"],
            row["candidate_output_cosine_to_routed_sum"],
        )
        if any(
            len(values) != 8 or not np.isfinite(np.asarray(values)).all()
            for values in scalar_arrays
        ):
            raise AuditError(f"invalid candidate reductions {key}")
        if len(row["candidate_output_cosine_valid"]) != 8:
            raise AuditError(f"invalid candidate cosine masks {key}")
        scalar_errors.append(
            abs(sum(row["candidate_weighted_output_fraction"]) - 1.0)
        )
        u_rms = float(np.sqrt(np.mean(u.astype(np.float64) ** 2)))
        scalar_errors.append(
            abs(u_rms - float(row["post_mixer_pre_moe_residual_rms"]))
        )

        if bool(row.get("audit_subset")):
            audit_rows += 1
            audit_layers.add(key[2])
            check_tensor_roles(store, event_id, TARGET_AUDIT_REQUIRED)
            for role in TARGET_AUDIT_REQUIRED:
                dtype_counts[(role, store.tensors[event_id][role]["native_dtype"])] += 1
            complete = store.tensor(event_id, "complete_moe_delta")
            complete_errors.append(rel_rms(complete, routed + shared))
            unweighted = store.tensor(
                event_id, "candidate_unweighted_output_vectors"
            )
            weighted = store.tensor(
                event_id, "candidate_weighted_output_vectors"
            )
            weights = store.tensor(
                event_id, "selected_execution_weights"
            ).astype(np.float32)
            weighted_errors.append(
                rel_rms(weighted, unweighted * weights[:, None])
            )
            complete_errors.append(rel_rms(routed, weighted.sum(axis=0)))
            shared_raw = store.tensor(
                event_id, "shared_expert_output_before_gate"
            )
            gate_logit = float(
                store.tensor(event_id, "shared_expert_gate_logit")[0]
            )
            gate = 1.0 / (1.0 + math.exp(-gate_logit))
            shared_errors.append(rel_rms(shared, shared_raw * gate))
            wrms = np.sqrt(np.mean(weighted.astype(np.float64) ** 2, axis=-1))
            fractions = wrms / max(float(wrms.sum()), 1e-12)
            scalar_errors.extend(
                np.abs(
                    fractions
                    - np.asarray(row["candidate_weighted_output_fraction"])
                ).tolist()
            )
            for index in range(8):
                value, valid = cosine(weighted[index], routed)
                if bool(row["candidate_output_cosine_valid"][index]) != valid:
                    raise AuditError(f"candidate cosine mask mismatch {key}")
                scalar_errors.append(
                    abs(
                        value
                        - float(
                            row["candidate_output_cosine_to_routed_sum"][index]
                        )
                    )
                )
        elif TARGET_AUDIT_REQUIRED.keys() & store.tensors[event_id].keys():
            raise AuditError(f"non-audit row carries sampled heavy tensors {key}")

    for key, layers in layers_by_position.items():
        if layers != set(range(40)):
            raise AuditError(f"incomplete target layers at {key}: {sorted(layers)}")
    if not audit_rows or audit_layers != set(range(40)):
        raise AuditError(
            f"audit subset does not cover every layer: rows={audit_rows}, "
            f"layers={sorted(audit_layers)}"
        )

    for row in prompt_rows:
        event_id = int(row["event_id"])
        key = (row["sequence_id"], int(row["absolute_sequence_position"]))
        prompt_layers_by_position[key].add(int(row["target_layer"]))
        check_tensor_roles(store, event_id, PROMPT_ROUTE_REQUIRED)
        probability_error, weight_error, failure = _audit_route(
            store, event_id,
            logits_role="raw_target_router_logits",
            probs_role="full_target_router_probabilities",
            ids_role="selected_expert_ids",
            weights_role="selected_execution_weights",
        )
        probability_errors.append(probability_error)
        weight_errors.append(weight_error)
        selected_set_failures += failure
    for key, layers in prompt_layers_by_position.items():
        if layers != set(range(40)):
            raise AuditError(f"incomplete prompt route layers at {key}")

    metrics = {
        "residual_relative_rms_max": max(residual_errors),
        "complete_moe_relative_rms_max": max(complete_errors),
        "candidate_weight_relative_rms_max": max(weighted_errors),
        "shared_relative_rms_max": max(shared_errors),
        "router_probability_abs_max": max(probability_errors),
        "selected_positive_gap_failures": selected_set_failures,
        "selected_weight_abs_max": max(weight_errors),
        "scalar_abs_max": max(scalar_errors),
    }
    if metrics["residual_relative_rms_max"] > 0.03:
        raise AuditError(f"target residual identity failed: {metrics}")
    if metrics["complete_moe_relative_rms_max"] > 0.03:
        raise AuditError(f"target MoE decomposition failed: {metrics}")
    if metrics["candidate_weight_relative_rms_max"] > 0.01:
        raise AuditError(f"candidate weighting failed: {metrics}")
    if metrics["shared_relative_rms_max"] > 0.01:
        raise AuditError(f"shared gate identity failed: {metrics}")
    if metrics["router_probability_abs_max"] > 2e-6:
        raise AuditError(f"router probability audit failed: {metrics}")
    if selected_set_failures:
        raise AuditError(f"native selected-set audit failed: {metrics}")
    if metrics["selected_weight_abs_max"] > 0.002:
        raise AuditError(f"selected weight audit failed: {metrics}")
    if metrics["scalar_abs_max"] > 2e-4:
        raise AuditError(f"target reduction audit failed: {metrics}")

    return {
        "target_layer_rows": len(rows),
        "authoritative_generated_positions": len(layers_by_position),
        "positions_with_all_40_layers": len(layers_by_position),
        "prompt_route_layer_rows": len(prompt_rows),
        "prompt_history_positions": len(prompt_layers_by_position),
        "audit_subset_rows": audit_rows,
        "audit_subset_layers": len(audit_layers),
        "identity_metrics": metrics,
        "native_dtype_counts": {
            f"{role}:{dtype}": count
            for (role, dtype), count in sorted(dtype_counts.items())
        },
    }


def audit_mtp(store: Store) -> dict[str, Any]:
    nodes = [row for row in store.rows if row["event"] == "mtp_node"]
    labels = {
        int(row["mtp_node_event_id"]): row
        for row in store.rows
        if row["event"] == "mtp_acceptance_label"
    }
    target_positions = {
        (row["sequence_id"], int(row["absolute_sequence_position"]))
        for row in store.rows
        if row["event"] == "target_token"
    }
    if not nodes:
        raise AuditError("no MTP nodes")
    keys = set()
    depth_counts = Counter()
    probability_errors = []
    weight_errors = []
    full_vocab_audits = 0
    for row in nodes:
        event_id = int(row["event_id"])
        key = (
            row["sequence_id"],
            int(row["verifier_cycle_id"]),
            int(row["engine_draft_depth"]),
            int(row["draft_branch_id"]),
        )
        if key in keys:
            raise AuditError(f"duplicate MTP node {key}")
        keys.add(key)
        depth = key[2]
        depth_counts[depth] += 1
        if row.get("acceptance_fields_present") is not False:
            raise AuditError(f"acceptance leaked into MTP input row {event_id}")
        if event_id not in labels or not labels[event_id].get("label_only"):
            raise AuditError(f"missing label-only MTP resolution for {event_id}")
        if not labels[event_id].get("acceptance_label_valid"):
            raise AuditError(f"invalid acceptance label {event_id}")
        base = int(row["committed_prefix_position"])
        expected_router = base + depth
        if int(row["router_state_position"]) != expected_router:
            raise AuditError(f"MTP router-position mismatch {event_id}")
        if int(row["mtp_input_token_position"]) != expected_router:
            raise AuditError(f"MTP input-position mismatch {event_id}")
        if int(row["vocabulary_prediction_position"]) != expected_router + 1:
            raise AuditError(f"MTP vocabulary-position mismatch {event_id}")
        if int(row["draft_token_position"]) != expected_router + 1:
            raise AuditError(f"MTP draft-position mismatch {event_id}")
        if int(row["gcrp_target_position"]) != expected_router:
            raise AuditError(f"MTP GCRP target-position mismatch {event_id}")
        if int(row["gcrp_horizon"]) != depth:
            raise AuditError(f"MTP GCRP horizon mismatch {event_id}")
        if (row["sequence_id"], base) not in target_positions:
            raise AuditError(f"MTP root lacks same-run target row {event_id}")
        check_tensor_roles(store, event_id, MTP_REQUIRED)
        logits = store.tensor(event_id, "raw_mtp_router_logits").astype(np.float32)
        probs = store.tensor(event_id, "full_mtp_router_probabilities").astype(np.float32)
        stable = logits - logits.max()
        recomputed = np.exp(stable)
        recomputed /= recomputed.sum()
        probability_errors.append(float(np.max(np.abs(recomputed - probs))))
        ids = store.tensor(event_id, "mtp_selected_expert_ids").astype(np.int64)
        actual = set(ids.tolist())
        predicted = set(torch.topk(torch.from_numpy(probs), 8).indices.numpy().tolist())
        if actual != predicted:
            selected_min = float(probs[ids].min())
            unselected = np.delete(probs, ids)
            if selected_min + 1e-7 < float(unselected.max()):
                raise AuditError(f"MTP selected-set positive-gap mismatch {event_id}")
        weights = store.tensor(event_id, "mtp_selected_execution_weights").astype(np.float32)
        expected_weights = probs[ids] / probs[ids].sum()
        weight_errors.append(float(np.max(np.abs(expected_weights - weights))))
        top_ids = store.tensor(event_id, "vocab_top64_token_ids").astype(np.int64)
        top_log_probs = store.tensor(event_id, "vocab_top64_log_probabilities")
        if int(top_ids[0]) != int(row["draft_token_id"]):
            raise AuditError(f"MTP draft/top1 mismatch {event_id}")
        if np.any(np.diff(top_log_probs) > 1e-6):
            raise AuditError(f"MTP top64 not sorted {event_id}")
        if "full_vocabulary_logits_audit" in store.tensors[event_id]:
            full_vocab_audits += 1
            full = store.tensor(event_id, "full_vocabulary_logits_audit").astype(np.float32)
            stable = full - full.max()
            log_probs = stable - math.log(float(np.exp(stable).sum()))
            cutoff = float(np.partition(log_probs, -64)[-64])
            if float(log_probs[top_ids].min()) + 1e-7 < cutoff:
                raise AuditError(f"full vocabulary top64 positive-gap mismatch {event_id}")
            if int(top_ids[0]) != int(np.argmax(log_probs)):
                raise AuditError(f"full vocabulary top1 mismatch {event_id}")
            if float(np.max(np.abs(log_probs[top_ids] - top_log_probs))) > 2e-5:
                raise AuditError(f"full vocabulary log-probability mismatch {event_id}")
        if row["conditioning_class"] not in {
            "authoritative_context_equivalent",
            "position_aligned_speculative_context",
        }:
            raise AuditError(f"bad conditioning class {event_id}")

    if set(depth_counts) != set(range(1, 7)):
        raise AuditError(f"missing MTP depths {dict(depth_counts)}")
    if len(set(depth_counts.values())) != 1:
        raise AuditError(f"incomplete depth-six chains {dict(depth_counts)}")
    if max(probability_errors) > 2e-6 or max(weight_errors) > 0.002:
        raise AuditError("MTP router reconstruction tolerance failed")
    if full_vocab_audits < 1:
        raise AuditError("no full-vocabulary audit sample")

    return {
        "mtp_nodes": len(nodes),
        "mtp_nodes_by_depth": dict(sorted(depth_counts.items())),
        "acceptance_labels": len(labels),
        "full_vocabulary_audit_nodes": full_vocab_audits,
        "router_probability_abs_max": max(probability_errors),
        "selected_weight_abs_max": max(weight_errors),
    }


def audit_sequences(store: Store) -> dict[str, Any]:
    starts = {
        row["sequence_id"]: row
        for row in store.rows
        if row["event"] == "sequence_start"
    }
    ends = {
        row["sequence_id"]: row
        for row in store.rows
        if row["event"] == "sequence_end"
    }
    if starts.keys() != ends.keys() or not starts:
        raise AuditError("sequence start/end mismatch")
    complete_h1 = 0
    complete_h2 = 0
    for sequence_id, end in ends.items():
        tokens = [int(v) for v in end["full_committed_token_ids"]]
        prompt_length = int(end["prompt_length"])
        if tokens[:prompt_length] != [int(v) for v in starts[sequence_id]["prompt_token_ids"]]:
            raise AuditError(f"prompt token mismatch {sequence_id}")
        if prefix_hash(tokens[:prompt_length]) != starts[sequence_id]["prompt_hash"]:
            raise AuditError(f"prompt hash mismatch {sequence_id}")
        captured = int(end["captured_authoritative_position_count"])
        if captured != len(tokens):
            raise AuditError(
                f"not every committed token has an authoritative target state in {sequence_id}"
            )
        complete_h1 += max(0, captured - prompt_length - 1)
        complete_h2 += max(0, captured - prompt_length - 2)
        per_position = Counter(
            int(row["absolute_sequence_position"])
            for row in store.rows
            if row["event"] == "target_token" and row["sequence_id"] == sequence_id
        )
        if set(per_position) != set(range(captured)) or any(v != 1 for v in per_position.values()):
            raise AuditError(f"nonmonotonic or duplicate target positions {sequence_id}")
    return {
        "sequences": len(starts),
        "complete_generated_t_plus_1_positions": complete_h1,
        "complete_generated_t_plus_2_positions": complete_h2,
    }


def audit(root: Path) -> dict[str, Any]:
    store = Store(root)
    try:
        manifest = store.manifest
        if manifest.get("training_started") is not False:
            raise AuditError("training flag is not false")
        if not isinstance(manifest.get("production_capture"), bool):
            raise AuditError("production_capture flag is not Boolean")
        header = store.rows[0]
        if (
            header.get("event") != "trace_header"
            or header.get("trace_header") is not True
            or header.get("schema") != "gcrp2r_transformers_joint_capture_v1"
        ):
            raise AuditError("invalid trace header")
        if len(store.by_id) != len(store.rows):
            raise AuditError("duplicate event IDs")
        if set(store.by_id) != set(range(len(store.rows))):
            raise AuditError("event IDs are not contiguous")
        checksum_count = check_checksums(root)
        target = audit_target(store)
        mtp = audit_mtp(store)
        sequences = audit_sequences(store)
        result = {
            "schema": "gcrp2r_transformers_joint_capture_audit_v2",
            "passed": True,
            "run_id": manifest["run_id"],
            "checksum_files_verified": checksum_count,
            "target": target,
            "mtp": mtp,
            "sequences": sequences,
            "gates": {
                "same_execution_target_mtp": True,
                "all_target_u_a_xplus": True,
                "all_target_routed_shared_outputs": True,
                "candidate_order_explicit": True,
                "position_semantics_explicit": True,
                "acceptance_label_only": True,
                "depth_six_complete": True,
                "future_t_plus_2_labels_available": (
                    sequences["complete_generated_t_plus_2_positions"] > 0
                ),
                "training_started": False,
                "production_capture_started": manifest["production_capture"],
            },
        }
        return result
    finally:
        store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.capture / "CAPTURE_AUDIT.json"
    try:
        report = audit(args.capture)
    except Exception as exc:
        report = {
            "schema": "gcrp2r_transformers_joint_capture_audit_v2",
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "training_started": False,
        }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
