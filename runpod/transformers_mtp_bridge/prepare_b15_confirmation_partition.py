#!/usr/bin/env python3
"""Freeze the 64 untouched outer-train requests for B1.5 confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_causal_pilot_partitions import (
    EXPECTED_SPLIT_SHA256,
    load_event_rows,
    pre_eos_horizon_usable_positions,
    request_order,
    sha256_file,
    uniform_offsets,
    write_jsonl,
)


SCHEMA = "harp_rtt_b15_confirmation_partition_v2_group_disjoint"
PARTITION = "b15_confirmation"
REQUESTS = 64
POSITIONS_PER_REQUEST = 32


def select_confirmation_requests(
    train: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    if len(train) != 452:
        raise ValueError(f"expected 452 outer-train requests, found {len(train)}")
    ordered = sorted(
        train, key=lambda row: request_order(str(row["request_id"]), seed)
    )
    selected = ordered[388:452]
    if len(selected) != REQUESTS:
        raise AssertionError("B1.5 confirmation partition is not exactly 64 requests")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--base-partitions", type=Path, required=True)
    parser.add_argument("--train-sequence-starts", type=Path, required=True)
    parser.add_argument("--train-sequence-ends", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-split-sha256", default=EXPECTED_SPLIT_SHA256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen confirmation {args.output}")
    split_sha = sha256_file(args.split_manifest)
    if split_sha != args.expected_split_sha256:
        raise ValueError("split manifest hash disagrees with the frozen v1 split")
    base_manifest_path = args.base_partitions / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    if (
        base_manifest.get("split_manifest_sha256") != split_sha
        or base_manifest.get("seed") != args.seed
        or base_manifest.get("outer_split") != "train"
        or base_manifest.get("sealed_test_opened") is not False
        or base_manifest.get("formal_validation_opened") is not False
        or base_manifest.get("calibration_opened") is not False
    ):
        raise ValueError("base causal partition manifest is not the frozen v3 source")

    train = []
    for line in args.split_manifest.read_text().splitlines():
        if line:
            row = json.loads(line)
            if row.get("split") == "train":
                train.append(row)
    selected = select_confirmation_requests(train, seed=args.seed)
    selected_ids = {str(row["request_id"]) for row in selected}
    selected_groups = {str(row["split_group_id"]) for row in selected}
    prior_ids = set()
    prior_groups = set()
    for name in ("engineering", "diagnostic_probe", "translator_fitting"):
        for line in (args.base_partitions / f"{name}_requests.jsonl").read_text().splitlines():
            if line:
                prior = json.loads(line)
                prior_ids.add(str(prior["request_id"]))
                prior_groups.add(str(prior["split_group_id"]))
    if selected_ids & prior_ids or len(prior_ids) != 388:
        raise ValueError("B1.5 confirmation is not disjoint from all frozen v3 partitions")

    if selected_groups & prior_groups:
        raise ValueError(
            "B1.5 confirmation has split-group/lineage overlap with frozen v3 partitions"
        )
    starts = load_event_rows(
        args.train_sequence_starts,
        event="sequence_start",
        expected_request_ids=selected_ids,
    )
    ends = load_event_rows(
        args.train_sequence_ends,
        event="sequence_end",
        expected_request_ids=selected_ids,
    )
    requests, prompts = [], []
    for row in selected:
        value = dict(row)
        request_id = str(row["request_id"])
        start = starts[request_id]
        end = ends[request_id]
        eos = end.get("eos_position")
        usable = pre_eos_horizon_usable_positions(
            split_usable_t_plus_2_positions=int(row["usable_t_plus_2_positions"]),
            prompt_token_count=len(start["prompt_token_ids"]),
            eos_position=None if eos is None else int(eos),
            horizon=4,
        )
        if usable < POSITIONS_PER_REQUEST:
            raise ValueError(
                f"confirmation request {request_id} has only {usable} complete H1-H4 positions"
            )
        offsets = uniform_offsets(usable, POSITIONS_PER_REQUEST)
        value.update(
            {
                "legacy_usable_t_plus_2_positions": int(row["usable_t_plus_2_positions"]),
                "pre_eos_h1_h4_usable_positions": usable,
                "first_eos_absolute_position": eos,
                "source_horizon": 4,
                "source_position_offsets": offsets,
                "partition": PARTITION,
            }
        )
        requests.append(value)
        prompts.append(
            {
                "sample_id": str(row["split_group_id"]),
                "prompt_token_ids": [int(token) for token in start["prompt_token_ids"]],
                "source": str(row["dataset_source"]),
                "group_id": str(row["split_group_id"]),
                "original_split": "train",
                "split": "train",
                "external_evaluation": False,
                "source_request_id": request_id,
                "source_sequence_id": str(row["sequence_id"]),
                "split_manifest_sha256": split_sha,
                "partition": PARTITION,
                "source_position_offsets": offsets,
                "source_horizon": 4,
                "pre_eos_h1_h4_usable_positions": usable,
                "first_eos_absolute_position": eos,
            }
        )

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "b15_confirmation_requests.jsonl", requests)
    write_jsonl(args.output / "b15_confirmation_capture_prompts.jsonl", prompts)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "split_manifest_sha256": split_sha,
        "base_partitions_manifest_sha256": sha256_file(base_manifest_path),
        "sequence_start_metadata_sha256": sha256_file(args.train_sequence_starts),
        "sequence_end_metadata_sha256": sha256_file(args.train_sequence_ends),
        "outer_split": "train",
        "partition": PARTITION,
        "requests": REQUESTS,
        "positions_per_request": POSITIONS_PER_REQUEST,
        "positions": REQUESTS * POSITIONS_PER_REQUEST,
        "selection": "all previously unused seed42 outer-train requests; uniform complete pre-EOS H1-H4 positions",
        "request_slice": [388, 452],
        "request_disjoint_from_frozen_v3": True,
        "split_group_disjoint_from_frozen_v3": True,
        "minimum_pre_eos_h1_h4_usable_positions": min(
            int(row["pre_eos_h1_h4_usable_positions"]) for row in requests
        ),
        "sealed_test_opened": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    files = sorted(path for path in args.output.iterdir() if path.is_file())
    with (args.output / "SHA256SUMS").open("w") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
