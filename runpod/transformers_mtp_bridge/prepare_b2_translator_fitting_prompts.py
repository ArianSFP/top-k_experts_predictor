#!/usr/bin/env python3
"""Freeze the train-only B2 fitting prompts from causal MTP telemetry only."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from harp_rtt.index import MTP_META_COLUMNS, MTP_SCALAR_COLUMNS


SCHEMA = "harp_rtt_b2_translator_fitting_partition_v1"
REQUEST_PATTERN = re.compile(r"^tfprod-req(?P<ordinal>[0-9]{6})-[0-9a-f]{8}$")
STRATA = (
    "low_entropy_low_margin",
    "low_entropy_high_margin",
    "high_entropy_low_margin",
    "high_entropy_high_margin",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_order(seed: int, request_id: str, offset: int) -> tuple[bytes, int]:
    value = f"harp-rtt-b2-scout\0{seed}\0{request_id}\0{offset}".encode()
    return hashlib.sha256(value).digest(), int(offset)


def segment_name(request_id: str) -> str:
    match = REQUEST_PATTERN.fullmatch(request_id)
    if match is None:
        raise ValueError(f"unexpected production request ID {request_id!r}")
    ordinal = int(match.group("ordinal"))
    if ordinal == 0:
        return "gcrp2r_tf_bf16_seg_000000_000000"
    start = ((ordinal - 1) // 16) * 16 + 1
    return f"gcrp2r_tf_bf16_seg_{start:06d}_{start + 15:06d}"


def stratum(
    entropy: float,
    margin: float,
    *,
    entropy_median: float,
    margin_median: float,
) -> str:
    entropy_side = "high_entropy" if entropy >= entropy_median else "low_entropy"
    margin_side = "low_margin" if margin <= margin_median else "high_margin"
    return f"{entropy_side}_{margin_side}"


def choose_stratified_offsets(
    rows: list[dict[str, Any]],
    *,
    request_id: str,
    uniform_offsets: set[int],
    entropy_median: float,
    margin_median: float,
    seed: int,
    count: int = 8,
) -> tuple[list[int], dict[str, int]]:
    """Choose two offsets per joint stratum, then causally backfill."""

    if count != 8:
        raise ValueError("the frozen B2 fitting contract requires eight scout offsets")
    available = [row for row in rows if int(row["offset"]) not in uniform_offsets]
    by_stratum: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATA}
    for row in available:
        name = stratum(
            float(row["entropy"]),
            float(row["margin"]),
            entropy_median=entropy_median,
            margin_median=margin_median,
        )
        by_stratum[name].append(row)
    selected: list[dict[str, Any]] = []
    for name in STRATA:
        ordered = sorted(
            by_stratum[name],
            key=lambda row: stable_order(seed, request_id, int(row["offset"])),
        )
        selected.extend(ordered[:2])
    selected_offsets = {int(row["offset"]) for row in selected}
    if len(selected_offsets) < count:
        remaining = sorted(
            (
                row
                for row in available
                if int(row["offset"]) not in selected_offsets
            ),
            key=lambda row: stable_order(seed, request_id, int(row["offset"])),
        )
        for row in remaining:
            selected.append(row)
            selected_offsets.add(int(row["offset"]))
            if len(selected_offsets) == count:
                break
    if len(selected_offsets) != count:
        raise ValueError(
            f"{request_id} exposes only {len(selected_offsets)} usable causal scout offsets"
        )
    chosen = sorted(selected_offsets)
    counts = Counter(
        stratum(
            float(row["entropy"]),
            float(row["margin"]),
            entropy_median=entropy_median,
            margin_median=margin_median,
        )
        for row in available
        if int(row["offset"]) in selected_offsets
    )
    return chosen, {name: int(counts.get(name, 0)) for name in STRATA}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-dir", type=Path, required=True)
    parser.add_argument("--rich-index-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_requests(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 256:
        raise ValueError(f"expected 256 translator-fitting requests, found {len(rows)}")
    identities = {str(row["request_id"]) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("translator-fitting requests are not unique")
    for row in rows:
        if row.get("partition") != "translator_fitting" or row.get("split") != "train":
            raise PermissionError("B2 fitting selection is outer-train only")
        if row.get("causal_stratified_selector_status") != (
            "pending_train_only_mtp_entropy_margin_scout"
        ):
            raise ValueError("translator-fitting request is not in the frozen pending state")
        uniform = [int(value) for value in row["uniform_source_position_offsets"]]
        if len(uniform) != 8 or uniform != sorted(set(uniform)):
            raise ValueError("frozen uniform fitting offsets are invalid")
    return rows


def load_causal_rows(
    requests: list[dict[str, Any]],
    index_root: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, str],
    int,
]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in requests:
        grouped.setdefault(segment_name(str(row["request_id"])), []).append(row)
    causal: dict[str, list[dict[str, Any]]] = {}
    sequences: dict[str, dict[str, Any]] = {}
    manifest_hashes: dict[str, str] = {}
    duplicate_positions = 0
    for segment, segment_requests in sorted(grouped.items()):
        root = index_root / segment
        manifest_path = root / "index_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"rich index segment is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "harp_rtt_rich_event_index_v1":
            raise ValueError("causal scout index schema mismatch")
        manifest_hashes[segment] = sha256_file(manifest_path)
        meta = np.load(root / "mtp_meta.npy", mmap_mode="r")
        scalars = np.load(root / "mtp_scalars.npy", mmap_mode="r")
        meta_columns = tuple(manifest.get("mtp_meta_columns", MTP_META_COLUMNS[: meta.shape[1]]))
        scalar_columns = tuple(
            manifest.get("mtp_scalar_columns", MTP_SCALAR_COLUMNS[: scalars.shape[1]])
        )
        required_meta = {
            "sequence_index",
            "target_position",
            "depth",
            "source_ready_event_order",
            "record_valid",
        }
        required_scalars = {
            "next_top1_top2_logprob_margin",
            "vocabulary_entropy",
        }
        if not required_meta.issubset(meta_columns) or not required_scalars.issubset(
            scalar_columns
        ):
            raise ValueError("rich index lacks the causal entropy/margin scout fields")
        mc = {name: meta_columns.index(name) for name in required_meta}
        sc = {name: scalar_columns.index(name) for name in required_scalars}
        sequence_rows = list(manifest.get("sequences", []))
        by_request = {
            str(value["request_id"]): (index, value)
            for index, value in enumerate(sequence_rows)
        }
        for request in segment_requests:
            request_id = str(request["request_id"])
            if request_id not in by_request:
                raise KeyError(f"{segment} lacks reserved request {request_id}")
            sequence_index, sequence = by_request[request_id]
            if sequence.get("split") != "train":
                raise PermissionError("causal scout refused a non-train index sequence")
            prompt_length = int(sequence["prompt_length"])
            usable = int(request["pre_eos_h1_h4_usable_positions"])
            candidates: dict[int, tuple[tuple[int, int], dict[str, Any]]] = {}
            rows = np.nonzero(
                (meta[:, mc["sequence_index"]] == sequence_index)
                & (meta[:, mc["record_valid"]] == 1)
            )[0]
            for row_index in rows.tolist():
                target_position = int(meta[row_index, mc["target_position"]])
                offset = target_position - prompt_length
                if not 0 <= offset < usable:
                    continue
                # Earliest source-ready draft is selected on duplicate target
                # positions. This uses causal event order and depth only.
                order = (
                    int(meta[row_index, mc["source_ready_event_order"]]),
                    int(meta[row_index, mc["depth"]]),
                )
                value = {
                    "offset": offset,
                    "entropy": float(scalars[row_index, sc["vocabulary_entropy"]]),
                    "margin": float(
                        scalars[row_index, sc["next_top1_top2_logprob_margin"]]
                    ),
                    "depth": int(meta[row_index, mc["depth"]]),
                    "source_ready_event_order": order[0],
                    "index_row": int(row_index),
                    "segment": segment,
                }
                previous = candidates.get(offset)
                if previous is not None:
                    duplicate_positions += 1
                if previous is None or order < previous[0]:
                    candidates[offset] = (order, value)
            causal[request_id] = [value for _, value in sorted(candidates.values())]
            sequences[request_id] = dict(sequence)
            if len(causal[request_id]) < 16:
                raise ValueError(f"{request_id} has too little causal scout telemetry")
    return causal, sequences, manifest_hashes, duplicate_positions


def main() -> None:
    args = parse_args()
    partition = args.partition_dir.expanduser().resolve()
    index_root = args.rich_index_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen B2 partition {output}")
    source_manifest = json.loads((partition / "manifest.json").read_text())
    if (
        source_manifest.get("outer_split") != "train"
        or source_manifest.get("formal_validation_opened") is not False
        or source_manifest.get("calibration_opened") is not False
        or source_manifest.get("sealed_test_opened") is not False
    ):
        raise PermissionError("source inner-partition manifest is not train-only")
    requests_path = partition / "translator_fitting_requests.jsonl"
    requests = load_requests(requests_path)
    causal, sequences, index_hashes, duplicates = load_causal_rows(requests, index_root)

    eligible = [
        row
        for request in requests
        for row in causal[str(request["request_id"])]
        if int(row["offset"]) not in set(
            int(value) for value in request["uniform_source_position_offsets"]
        )
    ]
    entropy_median = float(np.median([float(row["entropy"]) for row in eligible]))
    margin_median = float(np.median([float(row["margin"]) for row in eligible]))

    output.mkdir(parents=True)
    resolved, prompts, scout = [], [], []
    aggregate_counts: Counter[str] = Counter()
    for request in requests:
        request_id = str(request["request_id"])
        uniform = {int(value) for value in request["uniform_source_position_offsets"]}
        selected, counts = choose_stratified_offsets(
            causal[request_id],
            request_id=request_id,
            uniform_offsets=uniform,
            entropy_median=entropy_median,
            margin_median=margin_median,
            seed=args.seed,
        )
        aggregate_counts.update(counts)
        offsets = sorted(uniform | set(selected))
        if len(offsets) != 16:
            raise AssertionError("uniform and causal-stratified offsets overlap")
        value = dict(request)
        value.update(
            {
                "source_position_offsets": offsets,
                "causal_stratified_source_position_offsets": selected,
                "causal_stratified_selector_status": "frozen_train_only_causal_scout",
                "causal_stratified_counts": counts,
            }
        )
        resolved.append(value)
        sequence = sequences[request_id]
        prompts.append(
            {
                "sample_id": str(request["split_group_id"]),
                "prompt_token_ids": [int(token) for token in sequence["prompt_token_ids"]],
                "source": str(request["dataset_source"]),
                "group_id": str(request["split_group_id"]),
                "original_split": "train",
                "split": "train",
                "external_evaluation": False,
                "source_request_id": request_id,
                "source_sequence_id": str(request["sequence_id"]),
                "split_manifest_sha256": source_manifest["split_manifest_sha256"],
                "partition": "translator_fitting",
                "source_position_offsets": offsets,
                "source_horizon": 4,
                "pre_eos_h1_h4_usable_positions": int(
                    request["pre_eos_h1_h4_usable_positions"]
                ),
                "first_eos_absolute_position": request["first_eos_absolute_position"],
            }
        )
        selected_set = set(selected)
        for row in causal[request_id]:
            if int(row["offset"]) in selected_set:
                scout.append(
                    {
                        "request_id": request_id,
                        **row,
                        "stratum": stratum(
                            float(row["entropy"]),
                            float(row["margin"]),
                            entropy_median=entropy_median,
                            margin_median=margin_median,
                        ),
                    }
                )

    def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")

    write_jsonl(output / "translator_fitting_requests.jsonl", resolved)
    write_jsonl(output / "translator_fitting_capture_prompts.jsonl", prompts)
    write_jsonl(output / "causal_scout_selected_rows.jsonl", scout)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "outer_split": "train",
        "requests": 256,
        "positions_per_request": 16,
        "positions": 4096,
        "uniform_positions_per_request": 8,
        "causal_stratified_positions_per_request": 8,
        "selection": "8_uniform_plus_2_per_global_entropy_margin_quadrant_with_causal_hash_backfill",
        "causal_duplicate_policy": "earliest_source_ready_event_order_then_depth",
        "causal_features_read": [
            "mtp_meta.sequence_index",
            "mtp_meta.target_position",
            "mtp_meta.depth",
            "mtp_meta.source_ready_event_order",
            "mtp_meta.record_valid",
            "mtp_scalars.vocabulary_entropy",
            "mtp_scalars.next_top1_top2_logprob_margin",
        ],
        "forbidden_features_read": [],
        "acceptance_read": False,
        "factual_continuation_read": False,
        "target_labels_read": False,
        "entropy_median": entropy_median,
        "margin_median": margin_median,
        "selected_stratum_counts": {
            name: int(aggregate_counts.get(name, 0)) for name in STRATA
        },
        "duplicate_target_positions_resolved_causally": int(duplicates),
        "source_partition_manifest_sha256": sha256_file(partition / "manifest.json"),
        "source_translator_requests_sha256": sha256_file(requests_path),
        "split_manifest_sha256": source_manifest["split_manifest_sha256"],
        "rich_index_manifest_sha256": index_hashes,
        "selector_script_sha256": sha256_file(Path(__file__).resolve()),
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    files = sorted(path for path in output.iterdir() if path.is_file())
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
