#!/usr/bin/env python3
"""Freeze the request-disjoint 20k-position HARP-DeltaTree v3 pilot.

The selector uses only outer-train identity, pre-EOS availability, absolute
position, and causal MTP entropy/margin scouts. It never reads future target
routes, factual prefix mismatch, acceptance, validation, calibration, or test.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "harp_delta_v3_20k_partition_v1"
REQUESTS = 1_250
POSITIONS_PER_REQUEST = 16
REQUEST_SPLITS = (("train", 1_000), ("tune", 125), ("development", 125))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_order(seed: int, *parts: object) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode()).hexdigest()


def uniform_offsets(usable: int, count: int, *, start: int = 0) -> list[int]:
    if count < 1 or usable <= start:
        return []
    if count == 1:
        return [start]
    end = usable - 1
    return sorted({round(start + index * (end - start) / (count - 1)) for index in range(count)})


def log_offsets(usable: int, count: int, *, start: int = 33) -> list[int]:
    if count < 1 or usable <= start:
        return []
    low, high = math.log1p(start), math.log(usable)
    if count == 1:
        return [start]
    return sorted(
        {
            min(usable - 1, max(start, round(math.exp(low + index * (high - low) / (count - 1)) - 1)))
            for index in range(count)
        }
    )


def pre_eos_usable(row: Mapping[str, Any], start: Mapping[str, Any], end: Mapping[str, Any]) -> int:
    legacy = int(row["usable_t_plus_2_positions"])
    usable = max(0, legacy - 2)
    eos = end.get("eos_position")
    if eos is None:
        return usable
    prompt = len(start["prompt_token_ids"])
    generated_eos = int(eos) - prompt
    if generated_eos < 0:
        raise ValueError("first EOS precedes the generated continuation")
    return min(usable, max(0, generated_eos - 3))


def causal_thresholds(scouts: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[float, float]:
    entropy = [float(row["mtp_entropy"]) for rows in scouts.values() for row in rows]
    margin = [float(row["mtp_margin"]) for rows in scouts.values() for row in rows]
    if not entropy or len(entropy) != len(margin):
        raise ValueError("causal scout contains no entropy/margin rows")
    if not all(math.isfinite(value) for value in entropy + margin):
        raise ValueError("causal scout contains non-finite values")
    return float(median(entropy)), float(median(margin))


def mixed_position_offsets(
    *,
    request_id: str,
    usable: int,
    scouts: Sequence[Mapping[str, Any]],
    entropy_threshold: float,
    margin_threshold: float,
    seed: int,
) -> tuple[list[int], dict[int, str]]:
    """Choose 4 short, 4 log-late, 4 causal strata, and 4 uniform rows."""

    if usable < POSITIONS_PER_REQUEST:
        raise ValueError("request has fewer than 16 complete pre-EOS H1-H4 positions")
    chosen: list[int] = []
    reason: dict[int, str] = {}

    def add(values: Iterable[int], category: str, limit: int) -> None:
        added = 0
        for value in values:
            value = int(value)
            if added >= limit:
                break
            if 0 <= value < usable and value not in reason:
                chosen.append(value)
                reason[value] = category
                added += 1

    add(uniform_offsets(min(usable, 33), 4), "short_0_32", 4)
    add(log_offsets(usable, 4), "log_uniform_late", 4)

    bins: dict[tuple[bool, bool], list[Mapping[str, Any]]] = {
        (False, False): [], (False, True): [], (True, False): [], (True, True): []
    }
    for row in scouts:
        offset = int(row["source_position_offset"])
        if 0 <= offset < usable:
            key = (
                float(row["mtp_entropy"]) >= entropy_threshold,
                float(row["mtp_margin"]) >= margin_threshold,
            )
            bins[key].append(row)
    for key in ((False, False), (False, True), (True, False), (True, True)):
        ordered = sorted(
            bins[key],
            key=lambda row: stable_order(
                seed, request_id, "causal", key, int(row["source_position_offset"])
            ),
        )
        add(
            (int(row["source_position_offset"]) for row in ordered),
            f"causal_entropy_{int(key[0])}_margin_{int(key[1])}",
            1,
        )

    add(uniform_offsets(usable, 8), "global_uniform", 4)
    backfill = sorted(
        range(usable),
        key=lambda offset: stable_order(seed, request_id, "backfill", offset),
    )
    add(backfill, "deterministic_causal_backfill", POSITIONS_PER_REQUEST - len(chosen))
    if len(chosen) != POSITIONS_PER_REQUEST or len(set(chosen)) != POSITIONS_PER_REQUEST:
        raise AssertionError("mixed selector did not return exactly 16 unique offsets")
    return chosen, reason


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_events(path: Path, event: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("event") != event:
            raise ValueError(f"{path} contains a non-{event} row")
        request_id = str(row["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate {event} request {request_id}")
        result[request_id] = row
    return result


def load_scouts(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for row in load_jsonl(path):
        request_id = str(row["request_id"])
        identity = (request_id, int(row["source_position_offset"]))
        if identity in seen:
            raise ValueError(f"duplicate causal scout row {identity}")
        seen.add(identity)
        result.setdefault(request_id, []).append(row)
    return result


def excluded_identities(paths: Sequence[Path]) -> tuple[set[str], set[str]]:
    request_ids: set[str] = set()
    groups: set[str] = set()
    for path in paths:
        candidates = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for candidate in candidates:
            for row in load_jsonl(candidate):
                if "request_id" in row:
                    request_ids.add(str(row["request_id"]))
                if "source_request_id" in row:
                    request_ids.add(str(row["source_request_id"]))
                if "split_group_id" in row:
                    groups.add(str(row["split_group_id"]))
                if "group_id" in row:
                    groups.add(str(row["group_id"]))
    return request_ids, groups


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--sequence-starts", type=Path, required=True)
    parser.add_argument("--sequence-ends", type=Path, required=True)
    parser.add_argument("--causal-scout", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen v3 partition {args.output}")
    split_sha = sha256_file(args.split_manifest)
    if split_sha != args.expected_split_sha256:
        raise ValueError("split manifest hash differs from the declared v3 source")
    starts = load_events(args.sequence_starts, "sequence_start")
    ends = load_events(args.sequence_ends, "sequence_end")
    scouts = load_scouts(args.causal_scout)
    excluded_requests, excluded_groups = excluded_identities(args.exclude)

    eligible: list[tuple[dict[str, Any], int]] = []
    for row in load_jsonl(args.split_manifest):
        if row.get("split") != "train":
            continue
        if row.get("external_evaluation") is True:
            raise PermissionError("external evaluation row appeared inside outer-train")
        request_id = str(row["request_id"])
        group_id = str(row["split_group_id"])
        if request_id in excluded_requests or group_id in excluded_groups:
            continue
        if request_id not in starts or request_id not in ends or request_id not in scouts:
            continue
        usable = pre_eos_usable(row, starts[request_id], ends[request_id])
        if usable >= POSITIONS_PER_REQUEST:
            eligible.append((row, usable))
    eligible.sort(key=lambda item: stable_order(args.seed, "request", item[0]["request_id"]))
    if len(eligible) < REQUESTS:
        raise ValueError(f"v3 pilot needs {REQUESTS} eligible disjoint requests; found {len(eligible)}")
    selected = eligible[:REQUESTS]
    selected_groups = [str(row["split_group_id"]) for row, _ in selected]
    if len(set(selected_groups)) != REQUESTS:
        raise ValueError("selected v3 requests are not lineage/group disjoint")
    entropy_threshold, margin_threshold = causal_thresholds(
        {str(row["request_id"]): scouts[str(row["request_id"])] for row, _ in selected}
    )

    args.output.mkdir(parents=True)
    cursor = 0
    files: dict[str, dict[str, Any]] = {}
    for split_name, count in REQUEST_SPLITS:
        rows_out, prompts = [], []
        for row, usable in selected[cursor : cursor + count]:
            request_id = str(row["request_id"])
            offsets, reasons = mixed_position_offsets(
                request_id=request_id,
                usable=usable,
                scouts=scouts[request_id],
                entropy_threshold=entropy_threshold,
                margin_threshold=margin_threshold,
                seed=args.seed,
            )
            value = dict(row)
            value.update(
                {
                    "partition": f"delta_{split_name}",
                    "source_horizon": 4,
                    "pre_eos_h1_h4_usable_positions": usable,
                    "source_position_offsets": offsets,
                    "position_selection_reasons": [reasons[offset] for offset in offsets],
                }
            )
            rows_out.append(value)
            prompts.append(
                {
                    "sample_id": str(row["split_group_id"]),
                    "prompt_token_ids": [int(token) for token in starts[request_id]["prompt_token_ids"]],
                    "source": str(row["dataset_source"]),
                    "group_id": str(row["split_group_id"]),
                    "original_split": "train",
                    "split": "train",
                    "external_evaluation": False,
                    "source_request_id": request_id,
                    "source_sequence_id": str(row["sequence_id"]),
                    "split_manifest_sha256": split_sha,
                    "partition": f"delta_{split_name}",
                    "source_position_offsets": offsets,
                    "source_horizon": 4,
                    "pre_eos_h1_h4_usable_positions": usable,
                }
            )
        cursor += count
        request_path = args.output / f"{split_name}_requests.jsonl"
        prompt_path = args.output / f"{split_name}_capture_prompts.jsonl"
        write_jsonl(request_path, rows_out)
        write_jsonl(prompt_path, prompts)
        files[request_path.name] = {"sha256": sha256_file(request_path), "rows": len(rows_out)}
        files[prompt_path.name] = {"sha256": sha256_file(prompt_path), "rows": len(prompts)}

    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "split_manifest_sha256": split_sha,
        "sequence_starts_sha256": sha256_file(args.sequence_starts),
        "sequence_ends_sha256": sha256_file(args.sequence_ends),
        "causal_scout_sha256": sha256_file(args.causal_scout),
        "exclusion_sources": [str(path) for path in args.exclude],
        "excluded_requests": len(excluded_requests),
        "excluded_groups": len(excluded_groups),
        "outer_split": "train",
        "requests": REQUESTS,
        "positions_per_request": POSITIONS_PER_REQUEST,
        "positions": REQUESTS * POSITIONS_PER_REQUEST,
        "request_partitions": {name: count for name, count in REQUEST_SPLITS},
        "selection": "4 short + 4 log-late + 4 causal entropy/margin strata + 4 global uniform with deterministic causal backfill",
        "entropy_threshold": entropy_threshold,
        "margin_threshold": margin_threshold,
        "request_disjoint": True,
        "split_group_disjoint": True,
        "counterfactual_labels_are_targets_only": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "files": files,
    }
    manifest_path = args.output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
