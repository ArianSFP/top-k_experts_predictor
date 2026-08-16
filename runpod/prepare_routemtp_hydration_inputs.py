#!/usr/bin/env python3
"""Build hash-bound RouteMTP hydration inputs from immutable adaptive data.

The rich index already records the exact prompt and generated token sequence
for each adaptive request.  Reusing it avoids scanning the multi-gigabyte raw
event stream and keeps hydration identities aligned with the indexed tree and
counterfactual companion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Mapping


SCHEMA = "harp_routemtp_hydration_inputs_v1"
PREFIX_DOMAIN = b"GCRP2_PREFIX_V1"


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


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def write_jsonl_exclusive(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-source-positions", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite RouteMTP inputs {args.output}")
    if args.maximum_source_positions < 0:
        raise ValueError("maximum source positions must be non-negative")
    index = json.loads(args.index_manifest.read_text(encoding="utf-8"))
    companion_path = args.companion / "manifest.json"
    companion = json.loads(companion_path.read_text(encoding="utf-8"))
    if companion.get("split") != "train" or companion.get("label_only") is not True:
        raise PermissionError("RouteMTP companion must be label-only outer-train")
    if companion.get("runtime_available") is not False:
        raise PermissionError("RouteMTP companion labels cannot be runtime inputs")
    if companion.get("sealed_test_opened") is not False:
        raise PermissionError("RouteMTP companion crossed the sealed test")
    sequences: dict[str, dict[str, Any]] = {}
    for row in index.get("sequences", []):
        if row.get("split") != "train":
            raise PermissionError("RouteMTP index contains a non-train sequence")
        request_id = str(row["request_id"])
        if request_id in sequences:
            raise ValueError(f"duplicate RouteMTP index request {request_id}")
        prompt = [int(value) for value in row["prompt_token_ids"]]
        generated = [int(value) for value in row["generated_token_ids"]]
        if len(prompt) != int(row["prompt_length"]) or len(generated) != int(row["generated_length"]):
            raise ValueError("RouteMTP index token lengths are inconsistent")
        sequences[request_id] = {
            "request_id": request_id,
            "sequence_id": str(row["sequence_id"]),
            "split": "train",
            "external_evaluation": False,
            "prompt_length": len(prompt),
            "full_committed_token_ids": prompt + generated,
        }
    records = list(companion.get("records", []))
    if args.maximum_source_positions:
        records = records[: args.maximum_source_positions]
    if not records:
        raise ValueError("RouteMTP companion selection is empty")
    offsets: list[dict[str, Any]] = []
    authorized: set[str] = set()
    seen: set[tuple[str, int]] = set()
    for record in records:
        request_id = str(record["request_id"])
        if request_id not in sequences:
            raise KeyError(f"companion request is absent from index: {request_id}")
        source_position = int(record["source_position"])
        tokens = sequences[request_id]["full_committed_token_ids"]
        if not 0 <= source_position < len(tokens) - 1:
            raise ValueError("RouteMTP source position cannot provide exact H1")
        key = (request_id, source_position)
        if key in seen:
            raise ValueError("duplicate RouteMTP hydration source position")
        seen.add(key); authorized.add(request_id)
        offsets.append({
            "request_id": request_id,
            "source_position": source_position,
            "split": "train",
            "prefix_hash": prefix_hash(tokens[: source_position + 1]),
        })
    request_rows = [sequences[request_id] for request_id in sorted(authorized)]
    args.output.mkdir(parents=True)
    requests_path = args.output / "requests.jsonl"
    offsets_path = args.output / "source_offsets.jsonl"
    write_jsonl_exclusive(requests_path, request_rows)
    write_jsonl_exclusive(offsets_path, offsets)
    result = {
        "schema": SCHEMA,
        "requests": len(request_rows),
        "source_positions": len(offsets),
        "maximum_source_positions": args.maximum_source_positions,
        "index_manifest_sha256": sha256_file(args.index_manifest),
        "companion_manifest_sha256": sha256_file(companion_path),
        "requests_sha256": sha256_file(requests_path),
        "source_offsets_sha256": sha256_file(offsets_path),
        "outer_split": "train",
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "training_started": False,
        "optimizer_constructed": False,
    }
    write_json_exclusive(args.output / "MANIFEST.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
