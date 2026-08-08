#!/usr/bin/env python3
"""Fail-closed blocking audit for a B1.5 node-indexed companion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.b15 import b15_selector_sha256  # noqa: E402
from harp_rtt.node_counterfactual import (  # noqa: E402
    NODE_COUNTERFACTUAL_RECORD_SCHEMA,
    NODE_COUNTERFACTUAL_SCHEMA,
    NODE_ROUTER_AUDIT_SCHEMA,
    audit_node_counterfactual_geometry,
)
from harp_rtt.counterfactual import (  # noqa: E402
    NATIVE_BF16_CENTERED_LOGIT_TOLERANCE,
)
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument(
        "--maximum-logit-error",
        type=float,
        default=NATIVE_BF16_CENTERED_LOGIT_TOLERANCE,
    )
    parser.add_argument("--authoritative-topk-device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.maximum_logit_error <= NATIVE_BF16_CENTERED_LOGIT_TOLERANCE:
        raise ValueError("node counterfactual logit tolerance exceeds the frozen limit")
    manifest = load_json(args.companion / "manifest.json")
    if manifest.get("schema") != NODE_COUNTERFACTUAL_SCHEMA:
        raise ValueError("node counterfactual companion schema mismatch")
    if (
        manifest.get("label_only") is not True
        or manifest.get("runtime_available") is not False
        or manifest.get("split") != "train"
        or manifest.get("sealed_test_opened") is not False
        or manifest.get("training_started") is not False
    ):
        raise ValueError("node companion violates its safety contract")
    contract = manifest.get("tensor_contract")
    if not isinstance(contract, dict) or (
        contract.get("layout") != "unique_parent_before_child_nodes"
        or contract.get("h1_masked") is not True
        or contract.get("target_path_probability_condition") != "exact_committed_h1"
    ):
        raise ValueError("node tensor contract is invalid")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict) or bindings.get("selector_sha256") != b15_selector_sha256():
        raise ValueError("node selector binding mismatch")
    if bindings.get("router_geometry_sha256") != sha256_file(
        args.static_dir / "router_geometry.safetensors"
    ):
        raise ValueError("node companion geometry binding mismatch")
    if bindings.get("static_manifest_sha256") != sha256_file(
        args.static_dir / "manifest.json"
    ):
        raise ValueError("node companion static binding mismatch")

    checksums = {}
    for line in (args.companion / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest
    inventory = {
        str(path.relative_to(args.companion))
        for path in args.companion.rglob("*")
        if path.is_file()
        and path.name
        not in {"SHA256SUMS", "STOP_BEFORE_TRAINING.json", "COUNTERFACTUAL_AUDIT.json"}
    }
    if set(checksums) != inventory:
        raise ValueError("node counterfactual checksum inventory mismatch")
    for relative, digest in checksums.items():
        if sha256_file(args.companion / relative) != digest:
            raise ValueError(f"node counterfactual checksum mismatch: {relative}")

    static = load_static_target_artifacts(
        args.static_dir, device="cpu", load_embedding=False
    )
    geometry = static.geometry
    reports = []
    seen = set()
    audit_rows = 0
    router_input_maximum = 0.0
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("node companion contains no records")
    for record in records:
        key = (
            str(record["request_id"]),
            int(record["source_position"]),
            str(record["tree_id"]),
        )
        if key in seen:
            raise ValueError(f"duplicate node record key {key}")
        seen.add(key)
        prefixes = record.get("node_prefix_hashes")
        if not isinstance(prefixes, list) or len(prefixes) != contract["max_nodes"]:
            raise ValueError("node prefix-hash inventory is incomplete")
        relative = str(record["record"]["path"])
        path = args.companion / relative
        if record["record"].get("sha256") != sha256_file(path):
            raise ValueError("node record checksum mismatch")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        if (
            metadata.get("schema") != NODE_COUNTERFACTUAL_RECORD_SCHEMA
            or metadata.get("label_only") != "true"
            or metadata.get("runtime_available") != "false"
        ):
            raise ValueError("node record metadata is invalid")
        tensors = load_file(path, device="cpu")
        report = audit_node_counterfactual_geometry(
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
            if audit_record.get("sha256") != sha256_file(audit_path):
                raise ValueError("node router-input audit checksum mismatch")
            with safe_open(str(audit_path), framework="pt", device="cpu") as handle:
                audit_metadata = dict(handle.metadata() or {})
            if audit_metadata.get("schema") != NODE_ROUTER_AUDIT_SCHEMA:
                raise ValueError("node router-input audit schema mismatch")
            values = load_file(audit_path, device="cpu")
            valid = values["valid"].bool()
            if valid[0].any() or not valid.any():
                raise ValueError("node router-input audit validity is malformed")
            coordinates = geometry.encode_router_inputs(values["router_inputs"].float())
            difference = (coordinates - tensors["query_coordinates"].float())[valid]
            router_input_maximum = max(
                router_input_maximum, float(difference.abs().max().item())
            )
            audit_rows += int(valid.sum().item())

    total_rows = sum(int(report["valid_layer_rows"]) for report in reports)
    passed = bool(
        reports
        and all(report["passed"] for report in reports)
        and router_input_maximum <= 5e-3
    )
    result = {
        "schema": "harp_rtt_counterfactual_nodes_audit_v3",
        "passed": passed,
        "records": len(reports),
        "valid_layer_rows": total_rows,
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
            / total_rows
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
        "Blocking node audit passed; B1.5 oracle evaluation may continue."
        if passed
        else "Blocking node audit failed; do not index or evaluate this artifact."
    )
    stop_path.write_text(json.dumps(stop, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
