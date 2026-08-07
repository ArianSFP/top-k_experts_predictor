#!/usr/bin/env python3
"""Hydrate selected outer-train partitions from ordinal-addressed capture segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


REQUEST_PATTERN = re.compile(r"^tfprod-req(?P<ordinal>[0-9]{6})-[0-9a-f]{8}$")


def segment_name(request_id: str) -> str:
    match = REQUEST_PATTERN.fullmatch(request_id)
    if match is None:
        raise ValueError(f"unexpected production request ID {request_id!r}")
    ordinal = int(match.group("ordinal"))
    if ordinal == 0:
        return "gcrp2r_tf_bf16_seg_000000_000000"
    start = ((ordinal - 1) // 16) * 16 + 1
    end = start + 15
    return f"gcrp2r_tf_bf16_seg_{start:06d}_{end:06d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-dir", type=Path, required=True)
    parser.add_argument("--segment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite hydration output {args.output}")
    selected: dict[str, list[str]] = {}
    for name in (
        "engineering_requests.jsonl",
        "diagnostic_probe_requests.jsonl",
        "translator_fitting_requests.jsonl",
    ):
        for line in (args.partition_dir / name).read_text().splitlines():
            row = json.loads(line)
            if row.get("split") != "train":
                raise PermissionError("hydration refuses a non-train partition row")
            request_id = str(row["request_id"])
            selected.setdefault(segment_name(request_id), []).append(request_id)
    expected = {request for values in selected.values() for request in values}
    if len(expected) != 388:
        raise ValueError(f"expected 388 selected train requests, found {len(expected)}")

    captured: dict[str, dict] = {}
    for segment, request_ids in sorted(selected.items()):
        events = args.segment_root / segment / "events.jsonl"
        if not events.is_file():
            raise FileNotFoundError(f"capture segment is missing: {events}")
        request_alternation = "|".join(re.escape(value) for value in request_ids)
        pattern = (
            r'"event":"sequence_start".*"request_id":"('
            + request_alternation
            + r')"'
        )
        result = subprocess.run(
            ["grep", "-E", pattern, str(events)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(
                f"grep failed for {segment}: {result.stderr.strip()}"
            )
        for line in result.stdout.splitlines():
            row = json.loads(line)
            request_id = str(row["request_id"])
            if request_id not in expected:
                raise PermissionError("hydration emitted an unselected request")
            if (
                row.get("event") != "sequence_start"
                or bool(row.get("external_evaluation", False))
            ):
                raise PermissionError(
                    "hydration emitted an external-evaluation sequence start"
                )
            if request_id in captured:
                raise ValueError(f"duplicate hydrated request {request_id}")
            captured[request_id] = row
    missing = expected - captured.keys()
    if missing:
        raise ValueError(f"hydration missed {len(missing)} selected requests")
    with args.output.open("w") as handle:
        for request_id in sorted(captured):
            handle.write(json.dumps(captured[request_id], sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "schema": "harp_rtt_selected_train_sequence_starts_v2",
                "records": len(captured),
                "segments_read": len(selected),
                "outer_split": "train",
                "sealed_test_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
