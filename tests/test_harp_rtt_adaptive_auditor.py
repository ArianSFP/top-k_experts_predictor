from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


BRIDGE = (
    Path(__file__).resolve().parents[1] / "runpod" / "transformers_mtp_bridge"
)
if str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))


def _load_auditor(monkeypatch: pytest.MonkeyPatch):
    """Load the auditor without the pod-pinned Qwen Transformers revision."""

    audit_dependency = ModuleType("audit_transformers_segment")

    class AuditError(RuntimeError):
        pass

    audit_dependency.AuditError = AuditError
    audit_dependency.Store = object
    for name in (
        "_audit_route",
        "audit_sequences",
        "audit_target",
        "check_checksums",
        "check_tensor_roles",
    ):
        setattr(audit_dependency, name, lambda *_args, **_kwargs: None)
    capture_dependency = ModuleType("capture_transformers_segment")
    capture_dependency.prefix_hash = lambda _tokens: "unused-in-depth-test"
    monkeypatch.setitem(
        sys.modules, "audit_transformers_segment", audit_dependency
    )
    monkeypatch.setitem(
        sys.modules, "capture_transformers_segment", capture_dependency
    )

    path = BRIDGE / "audit_adaptive_tree_capture.py"
    spec = importlib.util.spec_from_file_location(
        "harp_rtt_adaptive_auditor_depth_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(depth: int, token_id: int) -> dict[str, int]:
    return {"engine_draft_depth": depth, "node_token_id": token_id}


def test_exact_h1_eos_may_terminate_without_h2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = _load_auditor(monkeypatch)
    auditor._audit_eos_terminated_depths(
        [_row(1, 2)], eos_token_id=2, tree_id="h1-eos"
    )


def test_h2_eos_may_terminate_without_h3_or_h4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = _load_auditor(monkeypatch)
    auditor._audit_eos_terminated_depths(
        [_row(1, 17), _row(2, 2)],
        eos_token_id=2,
        tree_id="h2-eos",
    )


@pytest.mark.parametrize(
    "rows",
    [
        [_row(1, 17)],
        [_row(1, 17), _row(2, 23)],
        [_row(1, 17), _row(2, 2), _row(2, 23)],
    ],
)
def test_non_eos_terminal_frontier_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, int]],
) -> None:
    auditor = _load_auditor(monkeypatch)
    with pytest.raises(auditor.AuditError, match="without an EOS-only frontier"):
        auditor._audit_eos_terminated_depths(
            rows, eos_token_id=2, tree_id="non-eos-truncation"
        )


def test_depth_gap_after_eos_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = _load_auditor(monkeypatch)
    with pytest.raises(auditor.AuditError, match="nodes after missing H2"):
        auditor._audit_eos_terminated_depths(
            [_row(1, 2), _row(3, 2)],
            eos_token_id=2,
            tree_id="depth-gap",
        )


class _AnchorStore:
    def __init__(
        self,
        rows: list[dict[str, object]],
        descriptors: dict[int, dict[str, dict[str, object]]],
        values: dict[tuple[int, str], np.ndarray],
    ) -> None:
        self.rows = rows
        self.tensors = descriptors
        self._values = values

    def tensor(self, parent: int, role: str) -> np.ndarray:
        return self._values[(parent, role)]


def _anchor_fixture(auditor):
    tree_id = "seq:tree0"
    spine_id = f"{tree_id}:legacy-anchor-h1-h6"
    sequence_id = "seq"
    base = 2
    # EOS occurs at H3. The incumbent anchor must still materialize H4--H6.
    path_tokens = [10, 11, 2, 13, 14, 15]
    next_tokens = [11, 2, 13, 14, 15, 16]
    rows: list[dict[str, object]] = []
    descriptors: dict[int, dict[str, dict[str, object]]] = {}
    values: dict[tuple[int, str], np.ndarray] = {}
    event_ids = [10 + 5 * local for local in range(6)]
    path_logps: list[float] = []
    for local, (event_id, token, next_token) in enumerate(
        zip(event_ids, path_tokens, next_tokens)
    ):
        local_logp = 0.0 if local == 0 else -0.1
        path_logps.append(local_logp)
        cumulative = sum(path_logps)
        top_ids = np.asarray(
            [next_token] + [1000 + 100 * local + index for index in range(1, 64)],
            dtype=np.int32,
        )
        top_logps = np.linspace(-0.1, -6.4, 64, dtype=np.float32)
        row = {
            "event": auditor.ANCHOR_SPINE_NODE_EVENT,
            "event_id": event_id,
            "tree_id": tree_id,
            "anchor_spine_id": spine_id,
            "anchor_spine_schema": auditor.ANCHOR_SPINE_SCHEMA,
            "sequence_id": sequence_id,
            "committed_prefix_position": base,
            "anchor_local_index": local,
            "parent_anchor_node_event_id": (
                None if local == 0 else event_ids[local - 1]
            ),
            "parent_anchor_local_index": None if local == 0 else local - 1,
            "engine_draft_depth": local + 1,
            "node_token_id": token,
            "token_rank_under_parent": None if local == 0 else 0,
            "branch_path_token_ids": path_tokens[: local + 1],
            "branch_path_token_log_probabilities": path_logps.copy(),
            "path_id": f"anchor-path-{local}",
            "node_local_token_log_probability": local_logp,
            "node_local_token_probability": math.exp(local_logp),
            "path_log_probability": cumulative,
            "path_probability": math.exp(cumulative),
            "vocabulary_top_token_id": next_token,
            "top_token_probability": math.exp(float(top_logps[0])),
            "gcrp_target_position": base + local + 1,
            "authoritative_prefix_hash": "unused-in-depth-test",
            "mtp_conditioning_prefix_hash": "unused-in-depth-test",
            "exact_prefix_hash": "unused-in-depth-test",
            "source_ready_event_order": event_id,
            "source_ready_timestamp_utc": f"2026-01-01T00:00:0{local}+00:00",
            "source_ready_monotonic_ns": 100 + local,
            "source_clock_domain_id": (
                "host_monotonic_after_native_mtp_materialization"
            ),
            "native_branch_execution_mode": (
                "isolated_full_prefix_recomputation_no_shared_mutable_kv"
            ),
            "structural_validity": True,
            "record_valid": True,
            "feature_available": True,
            "exact_committed_h1_root": local == 0,
            "anchor_only": True,
            "adaptive_model_input": False,
            "consumes_adaptive_node_budget": False,
            "continues_through_eos": True,
            "eos_termination_applies": False,
            "required_anchor_depth": 6,
            "acceptance_fields_present": False,
            "label_records_present": False,
        }
        rows.append(row)
        role_values = {
            "harp_anchor_mtp_vocabulary_head_input": np.ones(2048),
            "harp_anchor_raw_mtp_router_logits": np.ones(256),
            "harp_anchor_vocab_top64_token_ids": top_ids,
            "harp_anchor_vocab_top64_log_probabilities": top_logps,
        }
        descriptors[event_id] = {}
        for offset, (role, tensor) in enumerate(role_values.items(), start=1):
            descriptors[event_id][role] = {
                "event_id": event_id + offset,
                "parent_event_id": event_id,
                "entity_kind": auditor.ANCHOR_SPINE_NODE_EVENT,
                "sequence_id": sequence_id,
                "engine_draft_depth": local + 1,
                "absolute_position": (
                    base
                    + local
                    + 1
                    + int(role.startswith("harp_anchor_vocab_top64_"))
                ),
                "payload_file": (
                    "sidecars/harp_anchor_states.bin"
                    if role == "harp_anchor_mtp_vocabulary_head_input"
                    else "sidecars/harp_anchor_routes.bin"
                    if role == "harp_anchor_raw_mtp_router_logits"
                    else "sidecars/harp_anchor_vocab.bin"
                ),
                "native_dtype": (
                    "bf16"
                    if role
                    in {
                        "harp_anchor_mtp_vocabulary_head_input",
                        "harp_anchor_raw_mtp_router_logits",
                    }
                    else "int32" if role.endswith("token_ids") else "float32"
                ),
                "shape": list(tensor.shape),
            }
            values[(event_id, role)] = tensor

    ready_event_id = 40
    rows.append(
        {
            "event": auditor.ANCHOR_SPINE_READY_EVENT,
            "event_id": ready_event_id,
            "tree_id": tree_id,
            "anchor_spine_id": spine_id,
            "anchor_spine_schema": auditor.ANCHOR_SPINE_SCHEMA,
            "sequence_id": sequence_id,
            "committed_prefix_position": base,
            "authoritative_prefix_hash": "unused-in-depth-test",
            "exact_h1_token_id": path_tokens[0],
            "exact_h1_root_event_id": event_ids[0],
            "anchor_node_event_ids": event_ids,
            "node_count": 6,
            "required_depth": 6,
            "source_ready_event_order": ready_event_id,
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
    )
    store = _AnchorStore(rows, descriptors, values)
    kwargs = {
        "adaptive_trees": {
            tree_id: [
                {
                    "node_local_index": 0,
                    "sequence_id": sequence_id,
                    "committed_prefix_position": base,
                    "node_token_id": path_tokens[0],
                }
            ]
        },
        "sequence_end": {
            sequence_id: {"full_committed_token_ids": [7, 8, 9, path_tokens[0]]}
        },
        "target_event_by_position": {
            (sequence_id, base + 1): {"event_id": 50}
        },
        "label_by_node": {},
    }
    return store, kwargs


def test_anchor_spine_is_exact_h1_h6_even_through_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = _load_auditor(monkeypatch)
    store, kwargs = _anchor_fixture(auditor)
    report = auditor._audit_anchor_spines(store, **kwargs)
    assert report["anchor_nodes"] == 6
    assert report["anchor_nodes_by_depth"] == {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
    }
    assert report["continues_through_eos"] is True


def test_anchor_spine_missing_post_eos_depth_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = _load_auditor(monkeypatch)
    store, kwargs = _anchor_fixture(auditor)
    store.rows = [row for row in store.rows if row.get("event_id") != 35]
    with pytest.raises(auditor.AuditError, match="expected H1--H6"):
        auditor._audit_anchor_spines(store, **kwargs)


def test_anchor_spine_child_must_be_parent_local_top1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = _load_auditor(monkeypatch)
    store, kwargs = _anchor_fixture(auditor)
    child = next(row for row in store.rows if row.get("event_id") == 25)
    child["node_token_id"] = 999
    child["branch_path_token_ids"] = [10, 11, 2, 999]
    with pytest.raises(auditor.AuditError, match="local top-1 child"):
        auditor._audit_anchor_spines(store, **kwargs)


def test_anchor_spine_rejects_acceptance_label_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = _load_auditor(monkeypatch)
    store, kwargs = _anchor_fixture(auditor)
    store.rows.append(
        {
            "event": "mtp_acceptance_label",
            "event_id": 41,
            "mtp_node_event_id": 10,
        }
    )
    with pytest.raises(auditor.AuditError, match="has an acceptance label"):
        auditor._audit_anchor_spines(store, **kwargs)


def test_anchor_manifest_is_hashed_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = _load_auditor(monkeypatch)
    anchor = {
        "schema": auditor.ANCHOR_SPINE_SCHEMA,
        "purpose": "epoch-zero LegacyHARPAnchorBridge compatibility only",
        "root": "same exact committed target H1 token as the adaptive tree",
        "required_depth": 6,
        "depths": [1, 2, 3, 4, 5, 6],
        "parent_rule": "H2-H6 consume the immediately preceding node local top-1",
        "eos_rule": "continue through all six depths even when an earlier token is EOS",
        "native_branch_execution": (
            "independent isolated full-prefix recomputation; no adaptive sibling state"
        ),
        "tensor_roles": list(auditor.ANCHOR_SPINE_ROLES),
        "adaptive_node_budget_consumed": False,
        "adaptive_model_input": False,
        "labels_emitted": False,
        "realized_future_inputs": False,
        "source_ready_before_target_h1_execution": True,
    }
    manifest = {
        "legacy_harp_anchor_spine_manifest": anchor,
        "legacy_harp_anchor_spine_manifest_hash": auditor._canonical_hash(anchor),
        "capture_policy": {
            "legacy_anchor_spine_separate": True,
            "legacy_anchor_required_depth": 6,
            "legacy_anchor_continues_through_eos": True,
            "legacy_anchor_consumes_adaptive_budget": False,
            "legacy_anchor_enters_adaptive_model_inputs": False,
            "legacy_anchor_labels_present": False,
        },
        "counts": {
            "legacy_anchor_spines": 1,
            "legacy_anchor_spine_nodes": 6,
            "legacy_anchor_native_mtp_branch_calls": 6,
        },
    }
    report = {"anchor_spines": 1, "anchor_nodes": 6}
    assert auditor._audit_anchor_manifest(manifest, report)[
        "capture_counts_match"
    ] is True

    corrupt = copy.deepcopy(manifest)
    corrupt["legacy_harp_anchor_spine_manifest_hash"] = "0" * 64
    with pytest.raises(auditor.AuditError, match="manifest hash mismatch"):
        auditor._audit_anchor_manifest(corrupt, report)
