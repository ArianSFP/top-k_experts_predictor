#!/usr/bin/env python3
"""Resumable capture-only supervisor for the BF16 Transformers GCRP-2R corpus."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


SCHEMA = "gcrp2r_transformers_corpus_supervisor_v1"
CORPUS_SCHEMA = "gcrp2r_transformers_bf16_corpus_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_checksums(root: Path, filename: str, excluded: set[str]) -> int:
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]
    destination = root / filename
    with destination.open("w") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(root)}\n")
    return len(files)


def verify_checksums(root: Path, filename: str) -> int:
    count = 0
    for line in (root / filename).read_text().splitlines():
        expected, relative = line.split("  ", 1)
        actual = sha256_file(root / relative)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch {root / relative}")
        count += 1
    return count


def initialize_corpus(args: argparse.Namespace) -> dict[str, Any]:
    args.corpus_root.mkdir(parents=True, exist_ok=True)
    (args.corpus_root / "segments").mkdir(exist_ok=True)
    (args.corpus_root / "code").mkdir(exist_ok=True)
    manifest_path = args.corpus_root / "CORPUS_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != CORPUS_SCHEMA:
            raise RuntimeError("existing corpus root has incompatible schema")
        return manifest

    sources = {
        "GCRP-2R_v1.3_formal_architecture.md": args.architecture,
        "canonical-n256.jsonl": args.prompt_manifest,
        "MODEL_WEIGHT_SHA256SUMS.json": args.model_weight_manifest,
    }
    for name, source in sources.items():
        shutil.copy2(source, args.corpus_root / name)
    for source in (
        args.capture_script,
        args.audit_script,
        args.mtp_module,
        Path(__file__),
    ):
        shutil.copy2(source, args.corpus_root / "code" / source.name)

    prompt_rows = sum(
        1 for line in args.prompt_manifest.read_text().splitlines() if line.strip()
    )
    manifest = {
        "schema": CORPUS_SCHEMA,
        "created_utc": now(),
        "architecture_file": "GCRP-2R_v1.3_formal_architecture.md",
        "architecture_sha256": sha256_file(args.architecture),
        "prompt_manifest_file": "canonical-n256.jsonl",
        "prompt_manifest_sha256": sha256_file(args.prompt_manifest),
        "prompt_manifest_rows": prompt_rows,
        "model_weight_manifest_file": "MODEL_WEIGHT_SHA256SUMS.json",
        "model_weight_manifest_sha256": sha256_file(args.model_weight_manifest),
        "engine": "Transformers BF16 eager attention/eager experts",
        "target_generated_positions": args.target_positions,
        "max_new_tokens_per_request": args.max_new_tokens,
        "mtp_depth": 6,
        "segment_prompt_count": args.segment_prompts,
        "candidate_vector_audit_percent": args.candidate_audit_percent,
        "full_vocab_audit_percent": args.full_vocab_audit_percent,
        "prompt_route_history_tokens": 8,
        "split_assignment": "deferred until corpus-wide deduplication",
        "training_started": False,
        "flattening_started": False,
        "segments": [],
    }
    atomic_json(manifest_path, manifest)
    progress = {
        "schema": SCHEMA,
        "status": "capturing",
        "updated_utc": now(),
        "next_prompt_ordinal": 0,
        "captured_sequences": 0,
        "captured_generated_positions": 0,
        "captured_target_layer_rows": 0,
        "captured_mtp_nodes": 0,
        "segments_published": 0,
        "target_generated_positions": args.target_positions,
        "training_started": False,
    }
    atomic_json(args.corpus_root / "CAPTURE_PROGRESS.json", progress)
    atomic_json(
        args.corpus_root / "STOP_BEFORE_TRAINING.json",
        {
            "capture_complete": False,
            "capture_in_progress": True,
            "training_started": False,
            "instruction": "Capture and audit only. Do not preprocess or train.",
        },
    )
    return manifest


def add_segment_static_files(
    local: Path, args: argparse.Namespace
) -> None:
    shutil.copy2(args.architecture, local / args.architecture.name)
    (local / "code").mkdir(exist_ok=True)
    for source in (args.capture_script, args.audit_script, args.mtp_module):
        shutil.copy2(source, local / "code" / source.name)
    manifest = json.loads((local / "run_manifest.json").read_text())
    manifest["architecture_file"] = args.architecture.name
    manifest["architecture_sha256"] = sha256_file(args.architecture)
    manifest["capture_source_files"] = {
        str(path.relative_to(local)): sha256_file(path)
        for path in sorted((local / "code").iterdir())
        if path.is_file()
    }
    manifest["training_started"] = False
    manifest["flattening_started"] = False
    manifest["segment_finalization_started_utc"] = now()
    atomic_json(local / "run_manifest.json", manifest)


def run_audit(local: Path, args: argparse.Namespace) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(args.audit_script),
            "--capture",
            str(local),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        print(result.stdout, flush=True)
        raise RuntimeError(f"blocking audit failed for {local.name}")
    audit = json.loads((local / "CAPTURE_AUDIT.json").read_text())
    if not audit.get("passed"):
        raise RuntimeError(f"audit JSON did not pass for {local.name}")
    print(
        json.dumps(
            {
                "event": "segment_audit_passed",
                "segment": local.name,
                "generated_positions": audit["target"][
                    "authoritative_generated_positions"
                ],
                "sequences": audit["sequences"]["sequences"],
                "mtp_nodes": audit["mtp"]["mtp_nodes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return audit


def finalize_local_segment(
    local: Path, args: argparse.Namespace
) -> dict[str, Any]:
    add_segment_static_files(local, args)
    write_checksums(
        local,
        "SHA256SUMS",
        {
            "SHA256SUMS",
            "FINAL_SHA256SUMS",
            "CAPTURE_AUDIT.json",
            "STOP_BEFORE_TRAINING.json",
        },
    )
    audit = run_audit(local, args)
    marker_path = local / "STOP_BEFORE_TRAINING.json"
    marker = json.loads(marker_path.read_text())
    marker.update(
        {
            "capture_complete": True,
            "audit_complete": True,
            "audit_passed": True,
            "production_capture_started": True,
            "training_started": False,
            "flattening_started": False,
            "audit_sha256": sha256_file(local / "CAPTURE_AUDIT.json"),
            "raw_checksum_manifest_sha256": sha256_file(local / "SHA256SUMS"),
            "instruction": "Segment is immutable. Do not train from one segment.",
        }
    )
    atomic_json(marker_path, marker)
    final_count = write_checksums(
        local, "FINAL_SHA256SUMS", {"FINAL_SHA256SUMS"}
    )
    verify_checksums(local, "FINAL_SHA256SUMS")
    manifest = json.loads((local / "run_manifest.json").read_text())
    return {
        "segment_id": local.name,
        "prompt_skip": int(manifest["manifest_slice"]["skip"]),
        "prompt_limit": int(manifest["manifest_slice"]["limit"]),
        "sequences": int(audit["sequences"]["sequences"]),
        "generated_positions": int(
            audit["target"]["authoritative_generated_positions"]
        ),
        "target_layer_rows": int(audit["target"]["target_layer_rows"]),
        "mtp_nodes": int(audit["mtp"]["mtp_nodes"]),
        "complete_t_plus_2_positions": int(
            audit["sequences"]["complete_generated_t_plus_2_positions"]
        ),
        "audit_sha256": sha256_file(local / "CAPTURE_AUDIT.json"),
        "final_checksum_manifest_sha256": sha256_file(
            local / "FINAL_SHA256SUMS"
        ),
        "final_checksum_files": final_count,
        "bytes": sum(
            path.stat().st_size
            for path in local.rglob("*")
            if path.is_file()
        ),
    }


def publish_segment(
    local: Path, segment: dict[str, Any], args: argparse.Namespace
) -> Path:
    segments_root = args.corpus_root / "segments"
    final = segments_root / local.name
    partial = segments_root / f".{local.name}.partial"
    if final.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite published segment {local.name}")
    shutil.copytree(local, partial)
    verify_checksums(partial, "FINAL_SHA256SUMS")
    os.replace(partial, final)
    segment["published_path"] = str(final)
    segment["published_utc"] = now()
    segment["persistent_final_checksum_manifest_sha256"] = sha256_file(
        final / "FINAL_SHA256SUMS"
    )
    if (
        segment["persistent_final_checksum_manifest_sha256"]
        != segment["final_checksum_manifest_sha256"]
    ):
        raise RuntimeError("persistent checksum manifest changed during publish")
    return final


def load_progress(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads((args.corpus_root / "CAPTURE_PROGRESS.json").read_text())


def commit_progress(
    args: argparse.Namespace,
    corpus: dict[str, Any],
    progress: dict[str, Any],
    segment: dict[str, Any],
) -> None:
    corpus["segments"].append(segment)
    corpus["updated_utc"] = now()
    progress.update(
        {
            "updated_utc": now(),
            "next_prompt_ordinal": (
                segment["prompt_skip"] + segment["prompt_limit"]
            ),
            "captured_sequences": (
                progress["captured_sequences"] + segment["sequences"]
            ),
            "captured_generated_positions": (
                progress["captured_generated_positions"]
                + segment["generated_positions"]
            ),
            "captured_target_layer_rows": (
                progress["captured_target_layer_rows"]
                + segment["target_layer_rows"]
            ),
            "captured_mtp_nodes": (
                progress["captured_mtp_nodes"] + segment["mtp_nodes"]
            ),
            "segments_published": progress["segments_published"] + 1,
            "last_segment": segment["segment_id"],
            "training_started": False,
        }
    )
    atomic_json(args.corpus_root / "CORPUS_MANIFEST.json", corpus)
    atomic_json(args.corpus_root / "CAPTURE_PROGRESS.json", progress)
    print(
        json.dumps(
            {
                "event": "corpus_progress",
                "segments": progress["segments_published"],
                "sequences": progress["captured_sequences"],
                "generated_positions": progress[
                    "captured_generated_positions"
                ],
                "target": progress["target_generated_positions"],
                "next_prompt_ordinal": progress["next_prompt_ordinal"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def capture_segment(
    skip: int, limit: int, args: argparse.Namespace
) -> Path:
    stop = skip + limit - 1
    name = f"gcrp2r_tf_bf16_seg_{skip:06d}_{stop:06d}"
    local = args.local_root / name
    if local.exists():
        raise FileExistsError(f"local segment already exists: {local}")
    command = [
        sys.executable,
        str(args.capture_script),
        "--model",
        str(args.model),
        "--model-weight-manifest",
        str(args.model_weight_manifest),
        "--prompt-manifest",
        str(args.prompt_manifest),
        "--skip",
        str(skip),
        "--limit",
        str(limit),
        "--output",
        str(local),
        "--run-id",
        name,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-draft-depth",
        "6",
        "--full-vocab-audit-percent",
        str(args.full_vocab_audit_percent),
        "--candidate-vector-audit-percent",
        str(args.candidate_audit_percent),
        "--prompt-history-tokens",
        "8",
        "--production-capture",
    ]
    print(
        json.dumps(
            {
                "event": "segment_capture_start",
                "segment": name,
                "skip": skip,
                "limit": limit,
                "utc": now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    subprocess.run(command, check=True)
    return local


def complete_corpus(
    args: argparse.Namespace,
    corpus: dict[str, Any],
    progress: dict[str, Any],
) -> None:
    complete = progress["captured_generated_positions"] >= args.target_positions
    progress["status"] = "complete" if complete else "manifest_exhausted"
    progress["capture_complete"] = complete
    progress["updated_utc"] = now()
    corpus["capture_completed_utc"] = now()
    corpus["capture_complete"] = complete
    corpus["final_counts"] = {
        key: progress[key]
        for key in (
            "segments_published",
            "captured_sequences",
            "captured_generated_positions",
            "captured_target_layer_rows",
            "captured_mtp_nodes",
        )
    }
    corpus["training_started"] = False
    atomic_json(args.corpus_root / "CORPUS_MANIFEST.json", corpus)
    atomic_json(args.corpus_root / "CAPTURE_PROGRESS.json", progress)
    atomic_json(
        args.corpus_root / "CAPTURE_COMPLETE.json",
        {
            "schema": "gcrp2r_transformers_capture_complete_v1",
            "capture_complete": complete,
            "completed_utc": now(),
            "target_generated_positions": args.target_positions,
            "actual_generated_positions": progress[
                "captured_generated_positions"
            ],
            "segments_published": progress["segments_published"],
            "training_started": False,
            "flattening_started": False,
        },
    )
    atomic_json(
        args.corpus_root / "STOP_BEFORE_TRAINING.json",
        {
            "capture_complete": complete,
            "capture_in_progress": False,
            "all_segments_audited": True,
            "training_started": False,
            "flattening_started": False,
            "instruction": (
                "Raw event-aligned capture is complete. Stop before "
                "preprocessing, splitting, or training."
            ),
        },
    )
    root_files = [
        path
        for path in sorted(args.corpus_root.iterdir())
        if path.is_file() and path.name != "CORPUS_ROOT_SHA256SUMS"
    ]
    with (args.corpus_root / "CORPUS_ROOT_SHA256SUMS").open("w") as handle:
        for path in root_files:
            handle.write(f"{sha256_file(path)}  {path.name}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--model-weight-manifest", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--capture-script", type=Path, required=True)
    parser.add_argument("--audit-script", type=Path, required=True)
    parser.add_argument("--mtp-module", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--target-positions", type=int, default=150000)
    parser.add_argument("--segment-prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--candidate-audit-percent", type=int, default=5)
    parser.add_argument("--full-vocab-audit-percent", type=int, default=1)
    parser.add_argument("--adopt-local-segment", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.local_root.mkdir(parents=True, exist_ok=True)
    corpus = initialize_corpus(args)
    progress = load_progress(args)
    prompt_count = int(corpus["prompt_manifest_rows"])

    if args.adopt_local_segment:
        local = args.adopt_local_segment.resolve()
        expected_root = args.local_root.resolve()
        if expected_root not in local.parents:
            raise RuntimeError("adopted segment is outside the declared local root")
        segment = finalize_local_segment(local, args)
        publish_segment(local, segment, args)
        commit_progress(args, corpus, progress, segment)
        shutil.rmtree(local)

    while (
        progress["captured_generated_positions"] < args.target_positions
        and progress["next_prompt_ordinal"] < prompt_count
    ):
        skip = int(progress["next_prompt_ordinal"])
        limit = min(args.segment_prompts, prompt_count - skip)
        local = capture_segment(skip, limit, args)
        segment = finalize_local_segment(local, args)
        publish_segment(local, segment, args)
        commit_progress(args, corpus, progress, segment)
        resolved_root = local.resolve()
        if args.local_root.resolve() not in resolved_root.parents:
            raise RuntimeError("refusing to remove local segment outside local root")
        shutil.rmtree(resolved_root)

    complete_corpus(args, corpus, progress)
    print(
        json.dumps(
            {
                "event": "capture_supervisor_complete",
                "status": progress["status"],
                "generated_positions": progress[
                    "captured_generated_positions"
                ],
                "sequences": progress["captured_sequences"],
                "segments": progress["segments_published"],
                "training_started": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
