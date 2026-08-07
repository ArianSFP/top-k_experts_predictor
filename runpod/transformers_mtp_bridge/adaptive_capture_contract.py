"""Pure schema and leakage guards for adaptive native-MTP capture."""

from __future__ import annotations

from typing import Any


SCHEMA = "gcrp2r_transformers_adaptive_mtp_tree_capture_v2"
MANIFEST_SCHEMA = "gcrp2r_transformers_adaptive_mtp_tree_run_manifest_v2"
FORMAT_VERSION = 3
CAPTURE_PROFILE = (
    "harp_rtt_exact_h1_native_mtp_adaptive_h2_h4_with_legacy_anchor_h1_h6"
)
MAX_CAPTURE_DEPTH = 4
FULL_VOCAB_TOP_K = 64
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
SEALED_SPLIT_NAMES = {
    "test",
    "sealed_test",
    "sealed-test",
    "outer_test",
    "outer-test",
    "evaluation_test",
}


def assert_causal_mtp_node_record(fields: dict[str, Any]) -> None:
    """Reject any label/future field before a causal node event is emitted."""
    if fields.get("acceptance_fields_present") is not False:
        raise ValueError("causal MTP node must explicitly declare acceptance absent")

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for raw_key, child in value.items():
                key = str(raw_key)
                current = f"{path}.{key}" if path else key
                lowered = key.lower()
                if "accept" in lowered and current != "acceptance_fields_present":
                    raise ValueError(
                        f"acceptance leaked into causal MTP node field {current!r}"
                    )
                if lowered.startswith("realized_") or "future_target_" in lowered:
                    raise ValueError(
                        f"realized future leaked into causal MTP node field {current!r}"
                    )
                walk(child, current)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(fields)
    if "authoritative_prefix_hash" not in fields:
        raise ValueError("causal MTP node lacks authoritative source-prefix hash")
    depth = int(fields.get("engine_draft_depth", -1))
    if bool(fields.get("exact_committed_h1_root")) != (depth == 1):
        raise ValueError("exact-H1 root marker is inconsistent with node depth")


def assert_causal_anchor_spine_node_record(fields: dict[str, Any]) -> None:
    """Fail closed on an anchor-only H1--H6 causal-spine record."""

    assert_causal_mtp_node_record(fields)
    depth = int(fields.get("engine_draft_depth", -1))
    local = int(fields.get("anchor_local_index", -1))
    if fields.get("anchor_spine_schema") != ANCHOR_SPINE_SCHEMA:
        raise ValueError("anchor spine node has an unknown schema")
    if not 1 <= depth <= ANCHOR_SPINE_REQUIRED_DEPTH or local != depth - 1:
        raise ValueError("anchor spine local index/depth is not exact H1--H6 order")
    if fields.get("anchor_only") is not True:
        raise ValueError("anchor spine node is not marked anchor-only")
    if fields.get("adaptive_model_input") is not False:
        raise ValueError("anchor spine node may not enter the adaptive model inputs")
    if fields.get("consumes_adaptive_node_budget") is not False:
        raise ValueError("anchor spine node may not consume the adaptive node budget")
    if fields.get("continues_through_eos") is not True:
        raise ValueError("anchor spine must preserve legacy continuation through EOS")
    if fields.get("eos_termination_applies") is not False:
        raise ValueError("EOS termination may not truncate the legacy anchor spine")
    if int(fields.get("required_anchor_depth", -1)) != ANCHOR_SPINE_REQUIRED_DEPTH:
        raise ValueError("anchor spine node has the wrong pinned preprocessing depth")
    rank = fields.get("token_rank_under_parent")
    if (rank is None) != (depth == 1):
        raise ValueError("only the exact-H1 anchor root may omit a parent rank")
    if depth > 1 and int(rank) != 0:
        raise ValueError("anchor spine child must be the local top-1 under its parent")


def assert_prompt_capture_allowed(prompt: dict[str, Any]) -> None:
    """Fail closed if a manifest row identifies any sealed/evaluation split."""
    for key in ("split", "assigned_split", "offline_split", "original_split"):
        value = prompt.get(key)
        if value is not None and str(value).strip().lower() in SEALED_SPLIT_NAMES:
            raise PermissionError(
                f"adaptive recapture refuses sealed split row ({key}={value!r})"
            )
    if bool(prompt.get("external_evaluation", False)):
        raise PermissionError("adaptive recapture refuses external-evaluation rows")
