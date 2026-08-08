#!/usr/bin/env python3
"""Audit an atomically segmented GCRP-2R Transformers capture corpus.

This is a corpus-level inventory audit. Segment writers/auditors already verify
the dense sidecars before publication; this program verifies the published
segment set, count reconciliation, provenance flags, causal capture gates, root
checksums, and leakage-safe prompt-manifest properties without re-reading all
dense tensor bytes.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
from pathlib import Path


SEGMENT_RE = re.compile(r"^gcrp2r_tf_bf16_seg_(\d{6})_(\d{6})$")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256_manifest(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            digest, name = line.split(None, 1)
            rows.append((digest, name.strip()))
    return rows


def fail_if(condition: bool, message: str, errors: list[str]) -> None:
    if condition:
        errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.corpus_root.resolve()
    output = args.output or root / "CORPUS_AUDIT.json"
    errors: list[str] = []

    progress = load_json(root / "CAPTURE_PROGRESS.json")
    complete = load_json(root / "CAPTURE_COMPLETE.json")
    stop = load_json(root / "STOP_BEFORE_TRAINING.json")
    corpus_manifest = load_json(root / "CORPUS_MANIFEST.json")

    segment_rows: list[tuple[int, int, Path]] = []
    unexpected_segment_entries: list[str] = []
    for entry in (root / "segments").iterdir():
        if not entry.is_dir():
            unexpected_segment_entries.append(entry.name)
            continue
        match = SEGMENT_RE.match(entry.name)
        if not match:
            unexpected_segment_entries.append(entry.name)
            continue
        segment_rows.append((int(match.group(1)), int(match.group(2)), entry))
    segment_rows.sort()

    fail_if(bool(unexpected_segment_entries), "unexpected segment entries", errors)
    fail_if(not segment_rows, "no published segments", errors)
    if segment_rows:
        fail_if(segment_rows[0][:2] != (0, 0), "first segment is not pilot ordinal 0", errors)
        expected_start = 1
        for start, end, _ in segment_rows[1:]:
            fail_if(start != expected_start, f"non-contiguous segment start {start}, expected {expected_start}", errors)
            fail_if(end < start, f"invalid segment range {start}-{end}", errors)
            expected_start = end + 1

    totals = collections.Counter()
    source_counts = collections.Counter()
    mtp_depth_counts = collections.Counter()
    native_dtype_counts = collections.Counter()
    run_ids: set[str] = set()
    all_gates: collections.Counter[str] = collections.Counter()
    checksum_files_verified = 0
    segment_summaries = []

    for start, end, segment in segment_rows:
        required = [
            "CAPTURE_AUDIT.json",
            "FINAL_SHA256SUMS",
            "STOP_BEFORE_TRAINING.json",
            "events.jsonl",
            "router_artifacts.safetensors",
            "run_manifest.json",
        ]
        for name in required:
            fail_if(not (segment / name).is_file(), f"{segment.name}: missing {name}", errors)

        audit = load_json(segment / "CAPTURE_AUDIT.json")
        manifest = load_json(segment / "run_manifest.json")
        segment_stop = load_json(segment / "STOP_BEFORE_TRAINING.json")
        run_id = manifest.get("run_id")
        fail_if(run_id != segment.name, f"{segment.name}: run_id mismatch", errors)
        fail_if(run_id in run_ids, f"{segment.name}: duplicate run_id", errors)
        run_ids.add(run_id)
        fail_if(not audit.get("passed"), f"{segment.name}: audit did not pass", errors)
        fail_if(audit.get("run_id") != segment.name, f"{segment.name}: audit run_id mismatch", errors)
        fail_if(not manifest.get("production_capture"), f"{segment.name}: production_capture false", errors)
        fail_if(bool(manifest.get("training_started")), f"{segment.name}: training_started true", errors)
        fail_if(bool(manifest.get("flattening_started")), f"{segment.name}: flattening_started true", errors)
        fail_if(not manifest.get("stop_before_training"), f"{segment.name}: stop_before_training false", errors)
        fail_if(bool(segment_stop.get("training_started")), f"{segment.name}: segment stop says training started", errors)
        fail_if(bool(segment_stop.get("flattening_started")), f"{segment.name}: segment stop says flattening started", errors)

        manifest_slice = manifest.get("manifest_slice", {})
        counts = manifest.get("counts", {})
        fail_if(manifest_slice.get("skip") != start, f"{segment.name}: manifest skip mismatch", errors)
        fail_if(manifest_slice.get("limit") != end - start + 1, f"{segment.name}: manifest limit mismatch", errors)
        fail_if(counts.get("sequences") != end - start + 1, f"{segment.name}: sequence count/range mismatch", errors)
        fail_if(counts.get("target_layer_rows") != 40 * counts.get("generated_positions", -1), f"{segment.name}: target row count is not 40x positions", errors)

        target = audit.get("target", {})
        mtp = audit.get("mtp", {})
        seq = audit.get("sequences", {})
        fail_if(target.get("authoritative_generated_positions") != counts.get("generated_positions"), f"{segment.name}: audit/manifest positions mismatch", errors)
        fail_if(target.get("target_layer_rows") != counts.get("target_layer_rows"), f"{segment.name}: audit/manifest target rows mismatch", errors)
        fail_if(mtp.get("mtp_nodes") != counts.get("mtp_nodes"), f"{segment.name}: audit/manifest MTP nodes mismatch", errors)
        fail_if(seq.get("sequences") != counts.get("sequences"), f"{segment.name}: audit/manifest sequences mismatch", errors)
        fail_if(target.get("positions_with_all_40_layers") != counts.get("generated_positions"), f"{segment.name}: incomplete 40-layer position", errors)
        fail_if(target.get("audit_subset_layers") != 40, f"{segment.name}: audit subset lacks a layer", errors)

        depth_counts = {int(k): int(v) for k, v in mtp.get("mtp_nodes_by_depth", {}).items()}
        fail_if(set(depth_counts) != set(range(1, 7)), f"{segment.name}: MTP depths are not 1..6", errors)
        if depth_counts:
            fail_if(len(set(depth_counts.values())) != 1, f"{segment.name}: incomplete depth-six chains", errors)

        required_gates = {
            "acceptance_label_only",
            "all_target_routed_shared_outputs",
            "all_target_u_a_xplus",
            "candidate_order_explicit",
            "depth_six_complete",
            "future_t_plus_2_labels_available",
            "position_semantics_explicit",
            "production_capture_started",
            "same_execution_target_mtp",
        }
        gates = audit.get("gates", {})
        for gate in required_gates:
            fail_if(gates.get(gate) is not True, f"{segment.name}: gate {gate} not true", errors)
            all_gates[gate] += int(gates.get(gate) is True)
        fail_if(gates.get("training_started") is not False, f"{segment.name}: audit training gate is not false", errors)

        checksum_files_verified += int(audit.get("checksum_files_verified", 0))
        totals.update({
            "generated_positions": counts.get("generated_positions", 0),
            "target_layer_rows": counts.get("target_layer_rows", 0),
            "mtp_nodes": counts.get("mtp_nodes", 0),
            "sequences": counts.get("sequences", 0),
            "prompt_route_layer_rows": counts.get("prompt_route_layer_rows", 0),
            "complete_t_plus_1_positions": seq.get("complete_generated_t_plus_1_positions", 0),
            "complete_t_plus_2_positions": seq.get("complete_generated_t_plus_2_positions", 0),
            "full_vocabulary_audit_nodes": mtp.get("full_vocabulary_audit_nodes", 0),
            "candidate_audit_rows": target.get("audit_subset_rows", 0),
        })
        mtp_depth_counts.update(depth_counts)
        native_dtype_counts.update(target.get("native_dtype_counts", {}))
        segment_summaries.append({
            "name": segment.name,
            "prompt_ordinal_start": start,
            "prompt_ordinal_end": end,
            "generated_positions": counts.get("generated_positions"),
            "target_layer_rows": counts.get("target_layer_rows"),
            "mtp_nodes": counts.get("mtp_nodes"),
            "passed": bool(audit.get("passed")),
        })

    prompt_groups: set[str] = set()
    external_eval_rows = 0
    offline_split_rows = 0
    prompt_manifest_rows_checked = 0
    with (root / "canonical-n256.jsonl").open("r", encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if ordinal >= totals["sequences"]:
                break
            row = json.loads(line)
            fail_if(row.get("ordinal") != ordinal, f"prompt manifest ordinal mismatch at {ordinal}", errors)
            group = row.get("group_id")
            fail_if(group in prompt_groups, f"duplicate split group in captured prefix: {group}", errors)
            prompt_groups.add(group)
            source_counts[row.get("source", "unknown")] += 1
            external_eval_rows += int(bool(row.get("external_evaluation") or row.get("external_evaluation_group")))
            offline_split_rows += int(bool(row.get("offline_split_assigned")))
            prompt_manifest_rows_checked += 1

    fail_if(prompt_manifest_rows_checked != totals["sequences"], "captured sequence count exceeds prompt manifest", errors)
    fail_if(external_eval_rows != 0, "captured prefix contains external evaluation prompts", errors)
    fail_if(offline_split_rows != 0, "raw capture contains assigned splits", errors)

    expected_totals = {
        "generated_positions": progress.get("captured_generated_positions"),
        "target_layer_rows": progress.get("captured_target_layer_rows"),
        "mtp_nodes": progress.get("captured_mtp_nodes"),
        "sequences": progress.get("captured_sequences"),
    }
    for key, expected in expected_totals.items():
        fail_if(totals[key] != expected, f"summed {key}={totals[key]} but progress={expected}", errors)
    fail_if(len(segment_rows) != progress.get("segments_published"), "published segment count mismatch", errors)
    fail_if(segment_rows and segment_rows[-1][1] + 1 != progress.get("next_prompt_ordinal"), "next prompt ordinal mismatch", errors)
    fail_if(not progress.get("capture_complete"), "progress capture_complete false", errors)
    fail_if(not complete.get("capture_complete"), "complete marker false", errors)
    fail_if(not stop.get("capture_complete"), "stop marker capture_complete false", errors)
    fail_if(not stop.get("all_segments_audited"), "stop marker all_segments_audited false", errors)
    fail_if(bool(progress.get("training_started") or complete.get("training_started") or stop.get("training_started")), "root marker says training started", errors)
    fail_if(bool(complete.get("flattening_started") or stop.get("flattening_started")), "root marker says flattening started", errors)
    fail_if(totals["generated_positions"] < progress.get("target_generated_positions", 0), "target generated positions not met", errors)
    split_assignment = corpus_manifest.get("split_assignment")
    split_deferred = split_assignment is None or split_assignment in {"deferred", "not_assigned"} or (
        isinstance(split_assignment, str) and split_assignment.lower().startswith("deferred")
    )
    fail_if(not split_deferred, "corpus manifest unexpectedly assigns splits", errors)

    root_checksum_results = []
    for expected_digest, name in parse_sha256_manifest(root / "CORPUS_ROOT_SHA256SUMS"):
        path = root / name
        actual_digest = sha256_file(path) if path.is_file() else None
        ok = actual_digest == expected_digest
        fail_if(not ok, f"root checksum mismatch: {name}", errors)
        root_checksum_results.append({"file": name, "sha256": expected_digest, "verified": ok})

    audit = {
        "schema": "gcrp2r_transformers_corpus_audit_v1",
        "audited_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "corpus_root": str(root),
        "capture_backend": "Transformers BF16",
        "capture_complete": bool(progress.get("capture_complete")),
        "training_started": False,
        "flattening_started": False,
        "target_generated_positions": progress.get("target_generated_positions"),
        "counts": dict(totals),
        "segments": {
            "published": len(segment_rows),
            "prompt_ordinal_first": segment_rows[0][0] if segment_rows else None,
            "prompt_ordinal_last": segment_rows[-1][1] if segment_rows else None,
            "ranges_contiguous": not any("non-contiguous" in error for error in errors),
            "all_segment_audits_passed": all(row["passed"] for row in segment_summaries),
            "checksum_files_verified_by_segment_auditors": checksum_files_verified,
            "items": segment_summaries,
        },
        "prompt_manifest": {
            "rows_checked": prompt_manifest_rows_checked,
            "unique_split_groups": len(prompt_groups),
            "external_evaluation_rows": external_eval_rows,
            "offline_split_assigned_rows": offline_split_rows,
            "source_counts": dict(sorted(source_counts.items())),
        },
        "mtp_nodes_by_depth": {str(k): v for k, v in sorted(mtp_depth_counts.items())},
        "native_dtype_counts": dict(sorted(native_dtype_counts.items())),
        "required_gates_true_in_segments": dict(sorted(all_gates.items())),
        "root_checksums": root_checksum_results,
        "stop_before_training_verified": True,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "output": str(output),
        "passed": audit["passed"],
        "generated_positions": totals["generated_positions"],
        "segments": len(segment_rows),
        "sequences": totals["sequences"],
        "errors": errors,
    }, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
