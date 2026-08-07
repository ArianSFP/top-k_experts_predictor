#!/usr/bin/env python3
"""Fail-closed audit for a HARP-RTT v2 counterfactual companion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.counterfactual import (
    COUNTERFACTUAL_RECORD_SCHEMA,
    COUNTERFACTUAL_SCHEMA,
    NATIVE_BF16_CENTERED_LOGIT_TOLERANCE,
    audit_counterfactual_geometry,
    selector_sha256,
)
from harp_rtt.static_artifacts import load_static_target_artifacts


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_checksums(root: Path) -> dict[str, str]:
    rows = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        rows[relative] = digest
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument(
        "--maximum-logit-error",
        type=float,
        default=NATIVE_BF16_CENTERED_LOGIT_TOLERANCE,
    )
    parser.add_argument(
        "--authoritative-topk-device", default="cuda"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.maximum_logit_error <= 0.0
        or args.maximum_logit_error > NATIVE_BF16_CENTERED_LOGIT_TOLERANCE
    ):
        raise ValueError(
            "counterfactual logit tolerance must be positive and may not exceed "
            f"the frozen {NATIVE_BF16_CENTERED_LOGIT_TOLERANCE}"
        )
    manifest = load_json(args.companion / "manifest.json")
    if manifest.get("schema") != COUNTERFACTUAL_SCHEMA:
        raise ValueError("counterfactual companion schema mismatch")
    if manifest.get("label_only") is not True:
        raise ValueError("companion is not marked label-only")
    if manifest.get("runtime_available") is not False:
        raise ValueError("companion is incorrectly marked runtime-available")
    if manifest.get("split") != "train":
        raise ValueError("counterfactual companion is not outer-train only")
    if manifest.get("sealed_test_opened") is not False:
        raise ValueError("counterfactual companion reports sealed-test access")
    tensor_contract = manifest.get("tensor_contract")
    if not isinstance(tensor_contract, dict) or (
        tensor_contract.get("h1_masked") is not True
        or tensor_contract.get("target_path_probability_condition")
        != "exact_committed_h1"
        or tensor_contract.get("target_path_probability_origin") != "h2_edge"
    ):
        raise ValueError(
            "companion target path probabilities are not conditioned on exact H1"
        )
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("companion bindings are missing")
    if bindings.get("selector_sha256") != selector_sha256():
        raise ValueError("counterfactual selector hash mismatch")
    geometry_path = args.static_dir / "router_geometry.safetensors"
    if bindings.get("router_geometry_sha256") != sha256_file(geometry_path):
        raise ValueError("companion is bound to a different router geometry")
    if bindings.get("static_manifest_sha256") != sha256_file(
        args.static_dir / "manifest.json"
    ):
        raise ValueError("companion is bound to a different static manifest")

    checksums = parse_checksums(args.companion)
    expected_inventory = {
        str(path.relative_to(args.companion))
        for path in args.companion.rglob("*")
        if path.is_file()
        and path.name not in {
            "SHA256SUMS",
            "STOP_BEFORE_TRAINING.json",
            "COUNTERFACTUAL_AUDIT.json",
        }
    }
    if set(checksums) != expected_inventory:
        raise ValueError("counterfactual checksum inventory mismatch")
    for relative, expected in checksums.items():
        if sha256_file(args.companion / relative) != expected:
            raise ValueError(f"counterfactual checksum mismatch: {relative}")

    static = load_static_target_artifacts(
        args.static_dir, device="cpu", load_embedding=False
    )
    geometry = static.geometry
    reports = []
    router_input_maximum = 0.0
    audit_rows = 0
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("counterfactual companion contains no records")
    seen = set()
    for record in records:
        key = (
            str(record["request_id"]),
            int(record["source_position"]),
            str(record["tree_id"]),
        )
        if key in seen:
            raise ValueError(f"duplicate counterfactual record key {key}")
        seen.add(key)
        relative = record["record"]["path"]
        path = args.companion / relative
        if record["record"]["sha256"] != sha256_file(path):
            raise ValueError("record-level checksum mismatch")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        if metadata.get("schema") != COUNTERFACTUAL_RECORD_SCHEMA:
            raise ValueError("record safetensors schema mismatch")
        if metadata.get("label_only") != "true":
            raise ValueError("record safetensors is not label-only")
        if metadata.get("runtime_available") != "false":
            raise ValueError("record safetensors is runtime-available")
        tensors = load_file(path, device="cpu")
        report = audit_counterfactual_geometry(
            tensors,
            geometry,
            maximum_logit_error=args.maximum_logit_error,
            authoritative_topk_device=args.authoritative_topk_device,
        )
        report["tree_id"] = record["tree_id"]
        reports.append(report)
        audit_record = record.get("router_input_audit")
        if audit_record is not None:
            audit_path = args.companion / audit_record["path"]
            if audit_record["sha256"] != sha256_file(audit_path):
                raise ValueError("router-input audit checksum mismatch")
            values = load_file(audit_path, device="cpu")
            valid = values["valid"].bool()
            if valid[:, 0].any() or not valid.any():
                raise ValueError("router-input audit validity is malformed")
            coordinates = geometry.encode_router_inputs(
                values["router_inputs"].float()
            )
            difference = (
                coordinates - tensors["query_coordinates"].float()
            )[valid]
            router_input_maximum = max(
                router_input_maximum, float(difference.abs().max().item())
            )
            audit_rows += int(valid.sum().item())

    passed = bool(
        reports
        and all(report["passed"] for report in reports)
        and router_input_maximum <= 5e-3
    )
    result = {
        "schema": "harp_rtt_counterfactual_companion_audit_v2",
        "passed": passed,
        "records": len(reports),
        "valid_layer_rows": sum(
            int(report["valid_layer_rows"]) for report in reports
        ),
        "maximum_absolute_centered_logit_error": max(
            float(report["maximum_absolute_centered_logit_error"])
            for report in reports
        ),
        "native_selected_ids_exact": all(
            bool(report["native_selected_ids_exact"]) for report in reports
        ),
        "native_selected_set_matching_rows": sum(
            int(report["native_selected_set_matching_rows"]) for report in reports
        ),
        "geometry_selected_ids_exact": all(
            bool(report["geometry_selected_ids_exact"]) for report in reports
        ),
        "geometry_selected_set_matching_rows": sum(
            int(report["geometry_selected_set_matching_rows"]) for report in reports
        ),
        "geometry_numerical_boundary_rows": sum(
            int(report["geometry_numerical_boundary_rows"]) for report in reports
        ),
        "geometry_selected_set_agreement": (
            sum(int(report["geometry_selected_set_matching_rows"]) for report in reports)
            / sum(int(report["valid_layer_rows"]) for report in reports)
        ),
        "maximum_logit_error_tolerance": args.maximum_logit_error,
        "authoritative_topk_device": args.authoritative_topk_device,
        "router_input_audit_layer_rows": audit_rows,
        "maximum_router_input_coordinate_error": router_input_maximum,
        "label_only": True,
        "runtime_available": False,
        "sealed_test_opened": False,
        "training_started": False,
        "per_record": reports,
    }
    (args.companion / "COUNTERFACTUAL_AUDIT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    stop_path = args.companion / "STOP_BEFORE_TRAINING.json"
    stop = load_json(stop_path)
    stop["audit_complete"] = passed
    stop["instruction"] = (
        "Blocking companion audit passed; Stage-A integration may continue."
        if passed else
        "Blocking companion audit failed; do not train or index this artifact."
    )
    stop_path.write_text(json.dumps(stop, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
