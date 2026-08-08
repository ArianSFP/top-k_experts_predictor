#!/usr/bin/env python3
"""Compare reference and cloned-parent-cache node companions for promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_manifest(root: Path) -> dict:
    value = json.loads((root / "manifest.json").read_text())
    if not isinstance(value, dict):
        raise ValueError("companion manifest must be a mapping")
    return value


def keyed_records(manifest: dict) -> dict[tuple[str, int, str], dict]:
    return {
        (
            str(row["request_id"]),
            int(row["source_position"]),
            str(row["tree_id"]),
        ): row
        for row in manifest["records"]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-logit-error", type=float, default=2e-2)
    parser.add_argument("--maximum-query-error", type=float, default=5e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_manifest = load_manifest(args.reference)
    optimized_manifest = load_manifest(args.optimized)
    if reference_manifest.get("execution_mode") != (
        "isolated_full_prefix_replay_reference"
    ):
        raise ValueError("reference companion has the wrong execution mode")
    if optimized_manifest.get("execution_mode") != (
        "sibling_isolated_cloned_parent_cache"
    ):
        raise ValueError("optimized companion has the wrong execution mode")
    for binding in (
        "source_commit",
        "selector_sha256",
        "base_capture_manifest_sha256",
        "router_geometry_sha256",
        "split_manifest_sha256",
    ):
        if (
            reference_manifest["bindings"].get(binding)
            != optimized_manifest["bindings"].get(binding)
        ):
            raise ValueError(f"companion parity binding mismatch: {binding}")
    reference = keyed_records(reference_manifest)
    optimized = keyed_records(optimized_manifest)
    if set(reference) != set(optimized):
        raise ValueError("companion parity record keys disagree")

    maximum_logit = 0.0
    maximum_query = 0.0
    maximum_weight = 0.0
    selected_exact = True
    structural_exact = True
    probability_exact = True
    rows = 0
    for key in sorted(reference):
        left = load_file(
            args.reference / reference[key]["record"]["path"], device="cpu"
        )
        right = load_file(
            args.optimized / optimized[key]["record"]["path"], device="cpu"
        )
        for name in (
            "node_mask",
            "node_local_indices",
            "parent_local_indices",
            "depth",
            "first_divergence_depth",
            "budget_node_masks",
            "budget_endpoint_masks",
            "budget_realized",
            "budget_category_counts",
            "target_next_token_ids",
            "target_next_token_valid",
            "valid",
        ):
            structural_exact &= torch.equal(left[name], right[name])
        for name in (
            "source_edge_logp", "source_path_logp", "target_edge_logp",
            "target_path_logp", "target_next_token_logp",
        ):
            probability_exact &= torch.allclose(
                left[name].float(), right[name].float(), rtol=0.0, atol=0.0, equal_nan=True
            )
        valid = left["valid"].bool()
        if not torch.equal(valid, right["valid"].bool()):
            structural_exact = False
            continue
        selected_exact &= torch.equal(
            left["selected_ids"][valid], right["selected_ids"][valid]
        )
        maximum_logit = max(
            maximum_logit,
            float(
                (
                    left["router_logits"].float()
                    - right["router_logits"].float()
                )[valid].abs().max().item()
            ),
        )
        maximum_query = max(
            maximum_query,
            float(
                (
                    left["query_coordinates"].float()
                    - right["query_coordinates"].float()
                )[valid].abs().max().item()
            ),
        )
        maximum_weight = max(
            maximum_weight,
            float(
                (
                    left["selected_weights"].float()
                    - right["selected_weights"].float()
                )[valid].abs().max().item()
            ),
        )
        rows += int(valid.sum().item())

    passed = bool(
        structural_exact
        and probability_exact
        and selected_exact
        and maximum_logit <= args.maximum_logit_error
        and maximum_query <= args.maximum_query_error
    )
    report = {
        "schema": "harp_rtt_counterfactual_node_cache_parity_v3",
        "passed": passed,
        "records": len(reference),
        "valid_layer_rows": rows,
        "structural_tensors_exact": structural_exact,
        "path_probabilities_exact": probability_exact,
        "selected_ids_bit_identical": selected_exact,
        "maximum_router_logit_difference": maximum_logit,
        "maximum_query_coordinate_difference": maximum_query,
        "maximum_selected_weight_difference": maximum_weight,
        "router_logit_tolerance": args.maximum_logit_error,
        "query_coordinate_tolerance": args.maximum_query_error,
        "optimized_promoted": passed,
        "training_started": False,
        "sealed_test_opened": False,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite parity report {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
