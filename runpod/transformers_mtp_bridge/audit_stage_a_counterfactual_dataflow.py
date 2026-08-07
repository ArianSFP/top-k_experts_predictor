#!/usr/bin/env python3
"""Exercise the real Stage-A capture -> index -> dataset -> label join path.

This is an optimizer-free blocking audit.  It creates a small collection view
of one audited capture, builds the normal rich index, joins an audited
counterfactual companion under ``targets`` only, and proves that non-training
or non-train access is rejected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.counterfactual import (
    CounterfactualDatasetAdapter,
    load_counterfactual_companion,
)
from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt
from harp_rtt.index import build_collection


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sequence_starts(capture: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (capture / "events.jsonl").open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("event") == "sequence_start":
                rows.append(row)
    if not rows:
        raise ValueError("capture contains no sequence_start events")
    for row in rows:
        if (
            row.get("assigned_split") != "train"
            or row.get("original_split") != "train"
            or row.get("external_evaluation") is not False
            or row.get("sealed_test_opened") is not False
        ):
            raise PermissionError("Stage-A indexing is restricted to outer-train rows")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture = args.capture.expanduser().resolve()
    companion = args.companion.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse Stage-A audit output {output}")
    output.mkdir(parents=True)

    capture_audit = json.loads((capture / "CAPTURE_AUDIT_ADAPTIVE.json").read_text())
    companion_audit = json.loads((companion / "COUNTERFACTUAL_AUDIT.json").read_text())
    if capture_audit.get("passed") is not True or companion_audit.get("passed") is not True:
        raise ValueError("both source capture and companion must pass blocking audits")

    starts = _sequence_starts(capture)
    split_manifest = output / "stage_a_train_split.jsonl"
    with split_manifest.open("x", encoding="utf-8") as sink:
        for row in starts:
            sink.write(_canonical_json({"sequence_id": row["sequence_id"], "split": "train"}) + "\n")

    corpus = output / "corpus"
    segments = corpus / "segments"
    segments.mkdir(parents=True)
    (segments / "stage_a").symlink_to(capture, target_is_directory=True)
    index_root = output / "index"
    index_summary = build_collection(corpus, split_manifest, index_root)

    base = HarpRTTDataset(index_root, "train", corpus_root=corpus, max_tree_nodes=32)
    labels, companion_manifest = load_counterfactual_companion(
        companion, split="train", training=True
    )
    joined = CounterfactualDatasetAdapter(
        base, labels, split="train", training=True
    )
    if len(joined) != len(labels):
        raise ValueError(
            f"dataset/companion join is not one-to-one: dataset={len(joined)} labels={len(labels)}"
        )
    batch_size = min(max(int(args.batch_size), 1), len(joined))
    batch = collate_harp_rtt([joined[index] for index in range(batch_size)])
    if "counterfactual" in batch["inputs"]:
        raise AssertionError("counterfactual labels leaked into model inputs")
    counterfactual = batch["targets"].get("counterfactual")
    if counterfactual is None:
        raise AssertionError("counterfactual labels missing from targets")
    expected_shapes = {
        "path_mask": (batch_size, 4),
        "path_depths": (batch_size, 4),
        "node_local_indices": (batch_size, 4, 4),
        "source_path_logp": (batch_size, 4, 4),
        "target_path_logp": (batch_size, 4, 4),
        "query_coordinates": (batch_size, 4, 4, 40, 255),
        "router_logits": (batch_size, 4, 4, 40, 256),
        "selected_ids": (batch_size, 4, 4, 40, 8),
        "selected_weights": (batch_size, 4, 4, 40, 8),
        "valid": (batch_size, 4, 4, 40),
    }
    observed_shapes = {
        name: tuple(int(value) for value in counterfactual[name].shape)
        for name in expected_shapes
    }
    if observed_shapes != expected_shapes:
        raise ValueError(
            f"counterfactual batch geometry mismatch: {observed_shapes} != {expected_shapes}"
        )

    rejected: dict[str, bool] = {}
    for split, training in (
        ("validation", True),
        ("calibration", True),
        ("test", True),
        ("train", False),
    ):
        name = f"{split}:{'training' if training else 'inference'}"
        try:
            load_counterfactual_companion(companion, split=split, training=training)
        except PermissionError:
            rejected[name] = True
        else:
            rejected[name] = False
    if not all(rejected.values()):
        raise PermissionError(f"counterfactual access did not fail closed: {rejected}")

    report = {
        "schema": "harp_rtt_stage_a_counterfactual_dataflow_audit_v2",
        "passed": True,
        "optimizer_started": False,
        "training_started": False,
        "sealed_test_opened": False,
        "capture": str(capture),
        "companion": str(companion),
        "sequences": len(starts),
        "dataset_samples": len(joined),
        "companion_records": len(labels),
        "batch_size": batch_size,
        "counterfactual_only_in_targets": True,
        "observed_shapes": {name: list(shape) for name, shape in observed_shapes.items()},
        "rejected_access": rejected,
        "index_summary": index_summary,
        "companion_schema": companion_manifest["schema"],
    }
    (output / "STAGE_A_DATAFLOW_AUDIT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
