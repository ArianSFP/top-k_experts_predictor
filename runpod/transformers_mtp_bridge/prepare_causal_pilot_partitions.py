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
PARTITION_SCHEMA = "harp_rtt_causal_branch_inner_partitions_v3"


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
        round(index * (usable_positions - 1) / (count - 1)) for index in range(count)
    ]
    if result != sorted(set(result)):
        raise ValueError("uniform source-position selection produced duplicates")
    return result


def pre_eos_horizon_usable_positions(
    *,
    split_usable_t_plus_2_positions: int,
    prompt_token_count: int,
    eos_position: int | None,
    horizon: int = 4,
) -> int:
    """Return source positions with all requested tokens before/at first EOS."""

    if horizon < 2:
        raise ValueError("partition horizon must be at least two")
    if split_usable_t_plus_2_positions < 0 or prompt_token_count < 1:
        raise ValueError("invalid split usability or prompt length")
    usable = max(0, split_usable_t_plus_2_positions - (horizon - 2))
    if eos_position is None:
        return usable
    eos_generated_index = int(eos_position) - prompt_token_count
    if eos_generated_index < 0:
        raise ValueError("first EOS position precedes the generated continuation")
    return min(usable, max(0, eos_generated_index - (horizon - 1)))


def request_order(request_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}|{request_id}".encode()).hexdigest()
    return digest, request_id


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_event_rows(
    path: Path, *, event: str, expected_request_ids: set[str]
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("event") != event:
            raise ValueError(f"{path} contains a non-{event} event")
        request_id = str(row["request_id"])
        if request_id in rows:
            raise ValueError(f"duplicate {event} row {request_id}")
        rows[request_id] = row
    missing = expected_request_ids - rows.keys()
    unexpected = rows.keys() - expected_request_ids
    if missing or unexpected:
        raise ValueError(
            f"{event} identity mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-sequence-starts", type=Path)
    parser.add_argument("--train-sequence-ends", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-split-sha256", default=EXPECTED_SPLIT_SHA256)
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

    selected = set().union(*identities)
    if args.train_sequence_starts is None or args.train_sequence_ends is None:
        raise ValueError(
            "causal partition freezing requires train sequence starts and ends"
        )
    starts = load_event_rows(
        args.train_sequence_starts,
        event="sequence_start",
        expected_request_ids=selected,
    )
    ends = load_event_rows(
        args.train_sequence_ends,
        event="sequence_end",
        expected_request_ids=selected,
    )

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
            request_id = str(row["request_id"])
            start = starts[request_id]
            end = ends[request_id]
            eos_position = end.get("eos_position")
            usable = pre_eos_horizon_usable_positions(
                split_usable_t_plus_2_positions=int(row["usable_t_plus_2_positions"]),
                prompt_token_count=len(start["prompt_token_ids"]),
                eos_position=None if eos_position is None else int(eos_position),
                horizon=4,
            )
            value["legacy_usable_t_plus_2_positions"] = int(
                row["usable_t_plus_2_positions"]
            )
            value["pre_eos_h1_h4_usable_positions"] = usable
            value["first_eos_absolute_position"] = eos_position
            value["source_horizon"] = 4
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
                        "source_horizon": 4,
                        "pre_eos_h1_h4_usable_positions": row[
                            "pre_eos_h1_h4_usable_positions"
                        ],
                        "first_eos_absolute_position": row[
                            "first_eos_absolute_position"
                        ],
                    }
                )
            write_jsonl(args.output / f"{name}_capture_prompts.jsonl", prompts)
        hydrated = True

    manifest = {
        "schema": PARTITION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "split_manifest_sha256": split_sha,
        "sequence_start_metadata_sha256": sha256_file(args.train_sequence_starts),
        "sequence_end_metadata_sha256": sha256_file(args.train_sequence_ends),
        "source_horizon": 4,
        "position_eligibility": {
            "base": "legacy_usable_t_plus_2_minus_two_for_h4",
            "first_eos_rule": "all_H1_H4_tokens_exist_and_H4_may_equal_first_EOS",
            "post_eos_rows_allowed": False,
            "selected_requests_with_first_eos": sum(
                row["first_eos_absolute_position"] is not None
                for rows in partition_rows.values()
                for row in rows
            ),
            "selected_requests_shortened_from_legacy": sum(
                row["pre_eos_h1_h4_usable_positions"]
                < row["legacy_usable_t_plus_2_positions"]
                for rows in partition_rows.values()
                for row in rows
            ),
            "minimum_pre_eos_h1_h4_usable_positions": min(
                row["pre_eos_h1_h4_usable_positions"]
                for rows in partition_rows.values()
                for row in rows
            ),
        },
        "outer_split": "train",
        "sealed_test_opened": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "partitions": {
            "engineering": {
                "requests": 4,
                "positions_per_request": 8,
                "positions": 32,
                "selection": "uniform_over_pre_eos_h1_h4_span",
            },
            "diagnostic_probe": {
                "requests": 128,
                "positions_per_request": 16,
                "positions": 2048,
                "selection": "uniform_over_pre_eos_h1_h4_span",
            },
            "translator_fitting": {
                "requests": 256,
                "positions_per_request": 16,
                "positions": 4096,
                "selection": "eight_pre_eos_uniform_plus_eight_causal_entropy_margin_stratified",
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
        path
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    with (args.output / "SHA256SUMS").open("w") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
