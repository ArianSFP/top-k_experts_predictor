#!/usr/bin/env python3
"""Freeze disjoint outer-train request partitions for the HARP-RTT v2 pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SPLIT_SHA256 = (
    "be03308606e3c6ae524f0268bbd8a8b3590a76332f42ae165c203758bafc970b"
)
PARTITION_SCHEMA = "harp_rtt_causal_branch_inner_partitions_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_offsets(usable_positions: int, count: int) -> list[int]:
    if usable_positions < count or count < 1:
        raise ValueError("request does not contain enough usable positions")
    if count == 1:
        return [0]
    result = [
        round(index * (usable_positions - 1) / (count - 1))
        for index in range(count)
    ]
    if result != sorted(set(result)):
        raise ValueError("uniform source-position selection produced duplicates")
    return result


def request_order(request_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}|{request_id}".encode()).hexdigest()
    return digest, request_id


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-sequence-starts", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-split-sha256", default=EXPECTED_SPLIT_SHA256
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen partitions {args.output}")
    split_sha = sha256_file(args.split_manifest)
    if split_sha != args.expected_split_sha256:
        raise ValueError("split manifest hash disagrees with the frozen v1 split")
    train = []
    for line in args.split_manifest.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("split") == "train":
            train.append(row)
    if len(train) != 452:
        raise ValueError(f"expected 452 outer-train requests, found {len(train)}")
    train.sort(key=lambda row: request_order(str(row["request_id"]), args.seed))
    engineering = train[:4]
    probe = train[4:132]
    translator = train[132:388]
    identities = [
        {str(row["request_id"]) for row in partition}
        for partition in (engineering, probe, translator)
    ]
    if any(identities[i] & identities[j] for i in range(3) for j in range(i)):
        raise AssertionError("inner partitions are not request-disjoint")

    args.output.mkdir(parents=True)
    partition_rows: dict[str, list[dict[str, Any]]] = {}
    for name, rows, position_count in (
        ("engineering", engineering, 8),
        ("diagnostic_probe", probe, 16),
        ("translator_fitting", translator, 16),
    ):
        prepared = []
        for row in rows:
            value = dict(row)
            usable = int(row["usable_t_plus_2_positions"])
            if name == "translator_fitting":
                value["uniform_source_position_offsets"] = uniform_offsets(usable, 8)
                value["causal_stratified_position_count"] = 8
                value["causal_stratified_selector_status"] = (
                    "pending_train_only_mtp_entropy_margin_scout"
                )
            else:
                value["source_position_offsets"] = uniform_offsets(
                    usable, position_count
                )
            value["partition"] = name
            prepared.append(value)
        partition_rows[name] = prepared
        write_jsonl(args.output / f"{name}_requests.jsonl", prepared)

    hydrated = False
    if args.train_sequence_starts is not None:
        starts = {}
        for line in args.train_sequence_starts.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") != "sequence_start":
                raise ValueError("hydration input contains a non-sequence-start event")
            request_id = str(row["request_id"])
            if request_id in starts:
                raise ValueError(f"duplicate sequence-start row {request_id}")
            starts[request_id] = row
        selected = set().union(*identities)
        missing = selected - starts.keys()
        unexpected = starts.keys() - selected
        if missing or unexpected:
            raise ValueError(
                f"hydration identity mismatch: missing={len(missing)}, "
                f"unexpected={len(unexpected)}"
            )
        for name in ("engineering", "diagnostic_probe"):
            prompts = []
            for row in partition_rows[name]:
                start = starts[str(row["request_id"])]
                prompts.append(
                    {
                        "sample_id": str(row["split_group_id"]),
                        "prompt_token_ids": [
                            int(value) for value in start["prompt_token_ids"]
                        ],
                        "source": str(row["dataset_source"]),
                        "group_id": str(row["split_group_id"]),
                        "original_split": "train",
                        "split": "train",
                        "external_evaluation": False,
                        "source_request_id": str(row["request_id"]),
                        "source_sequence_id": str(row["sequence_id"]),
                        "split_manifest_sha256": split_sha,
                        "partition": name,
                        "source_position_offsets": row["source_position_offsets"],
                    }
                )
            write_jsonl(args.output / f"{name}_capture_prompts.jsonl", prompts)
        hydrated = True

    manifest = {
        "schema": PARTITION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "split_manifest_sha256": split_sha,
        "outer_split": "train",
        "sealed_test_opened": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "partitions": {
            "engineering": {
                "requests": 4,
                "positions_per_request": 8,
                "positions": 32,
                "selection": "uniform",
            },
            "diagnostic_probe": {
                "requests": 128,
                "positions_per_request": 16,
                "positions": 2048,
                "selection": "uniform",
            },
            "translator_fitting": {
                "requests": 256,
                "positions_per_request": 16,
                "positions": 4096,
                "selection": "eight_uniform_plus_eight_causal_entropy_margin_stratified",
                "stratified_offsets_status": "pending_B1_gate_and_train_only_scout",
            },
        },
        "request_order": "sha256(seed|request_id), request_id",
        "hydrated_capture_prompts": hydrated,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    files = [
        path for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    with (args.output / "SHA256SUMS").open("w") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
