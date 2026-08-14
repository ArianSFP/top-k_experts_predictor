#!/usr/bin/env python3
"""Evaluate ShadowRoute on immutable adaptive trees with an exact target prefix."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "runpod" / "transformers_mtp_bridge"
for candidate in (str(REPO_ROOT), str(BRIDGE)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from harp_rtt.node_counterfactual import load_node_counterfactual_companion  # noqa: E402
from harp_rtt.exact_k import stable_topk  # noqa: E402
from harp_rtt.route_ceiling import factual_branch_topk, posterior_native_topk  # noqa: E402
from harp_rtt.shadow_backbone import exact_prefix_experts, install_shadow_experts  # noqa: E402
from harp_rtt.shadow_bundle import load_shadow_bundle  # noqa: E402
from harp_rtt.shadow_checkpoint import IndexedCheckpoint, sha256_file  # noqa: E402
from harp_rtt.shadow_rollout import ShadowRouteHooks, run_shadow_tree  # noqa: E402


SCHEMA = "harp_shadowroute_closed_loop_evaluation_v1"
RESULT_SCHEMA = "harp_shadowroute_closed_loop_result_v1"
CEILING_BUNDLE_SCHEMA = "harp_deltaroute_v4_ceiling_bundle_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-capture", type=Path, required=True)
    parser.add_argument("--native-companion", type=Path, required=True)
    parser.add_argument(
        "--ceiling-bundle",
        type=Path,
        help=(
            "optional immutable Stage-0 bundle used to evaluate factual branch "
            "mixtures with learned, target, and raw-MTP posteriors"
        ),
    )
    parser.add_argument(
        "--ceiling-parent-sha256",
        help="required selected-parent checkpoint SHA256 when --ceiling-bundle is used",
    )
    parser.add_argument("--layer-checkpoint-root", type=Path)
    parser.add_argument("--s1-fallback-root", type=Path)
    parser.add_argument("--shadow-width", type=int, choices=(16, 32, 64, 96, 128), default=16)
    parser.add_argument("--indexed-active-slots", type=int, choices=(4, 8), default=8)
    parser.add_argument("--native-exact-slots", type=int, choices=(1, 2, 4, 6, 8))
    parser.add_argument(
        "--mode",
        choices=("exact_top1_plus_draft", "shared_width128", "indexed_width16", "int4_top4"),
        required=True,
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--execution-source-commit",
        required=True,
        help="full Git SHA of the evaluator implementation executing this run",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--native-parity", action="store_true")
    parser.add_argument(
        "--diagnostic-unpromoted-bundle",
        action="store_true",
        help=(
            "load component-gate failures for a non-promoting Stage-A diagnostic; "
            "restricted to S0/S2/INT4"
        ),
    )
    parser.add_argument(
        "--native-parity-only",
        action="store_true",
        help=(
            "audit exact prefix/cache replay against authoritative native routes "
            "without loading a learned ShadowRoute bundle"
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_checksums(output: Path) -> None:
    paths = sorted(path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())


def route_overlap(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape or predicted.shape[-1] != 8:
        raise ValueError("route overlap requires aligned native top-eight sets")
    return (
        target[..., None] == predicted[..., None, :]
    ).any(-1).float().mean(-1)


def slot_coverage(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape[:-1] != target.shape[:-1] or target.shape[-1] != 8:
        raise ValueError("slot coverage requires aligned prediction/target cells")
    return (
        target[..., None] == predicted[..., None, :]
    ).any(-1).float().mean(-1)


def load_ceiling_bundle(path: Path, *, parent_sha256: str) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != CEILING_BUNDLE_SCHEMA:
        raise ValueError("ShadowRoute factual evaluation ceiling-bundle schema mismatch")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("ShadowRoute factual evaluation lacks bundle provenance")
    if provenance.get("outer_split") != "train":
        raise PermissionError("ShadowRoute factual evaluation is outer-train only")
    if provenance.get("parent_checkpoint_sha256") != parent_sha256:
        raise ValueError("ShadowRoute factual evaluation parent checkpoint mismatch")
    for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if provenance.get(key) is not False:
            raise PermissionError(f"ShadowRoute factual bundle violates {key}")
    request_ids = value.get("request_ids")
    if not isinstance(request_ids, list) or not request_ids:
        raise ValueError("ShadowRoute factual bundle lacks request IDs")
    expected_rows = len(request_ids)
    expected = {
        "anchor_marginals": (expected_rows, 4, 40, 256),
        "target_ids": (expected_rows, 4, 40, 8),
        "future_valid": (expected_rows, 4, 40),
        "node_native_ids": (expected_rows, 32, 40, 8),
        "branch_mask": (expected_rows, 4, 32),
        "factual_branch_indices": (expected_rows, 4),
        "prefix_mismatch": (expected_rows, 4),
    }
    for name, shape in expected.items():
        tensor = value.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"ShadowRoute factual bundle field {name} has invalid geometry")
    posteriors = value.get("posteriors")
    if not isinstance(posteriors, Mapping):
        raise ValueError("ShadowRoute factual bundle lacks posteriors")
    for name in ("learned", "target", "mtp"):
        condition = posteriors.get(name)
        if not isinstance(condition, Mapping):
            raise ValueError(f"ShadowRoute factual bundle lacks {name} posterior")
        if tuple(condition["captured"].shape) != (expected_rows, 4, 32):
            raise ValueError(f"ShadowRoute {name} captured posterior has invalid geometry")
        if tuple(condition["other"].shape) != (expected_rows, 4):
            raise ValueError(f"ShadowRoute {name} OTHER posterior has invalid geometry")
    return value


def resolve_ceiling_row(
    bundle: Mapping[str, Any],
    remaining: dict[str, list[int]],
    *,
    request_id: str,
    native_ids: torch.Tensor,
    native_valid: torch.Tensor,
) -> int:
    candidates = remaining.get(request_id, [])
    if not candidates:
        raise KeyError(f"ceiling bundle has no unused row for request {request_id!r}")
    count = native_ids.shape[0]
    matches: list[int] = []
    for row in candidates:
        reference = bundle["node_native_ids"][row, :count].long()
        if torch.equal(reference[native_valid], native_ids.long()[native_valid]):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"native-route fingerprint join for {request_id!r} produced {len(matches)} rows"
        )
    row = matches[0]
    candidates.remove(row)
    return row


def summarize_factual(
    cells: Mapping[tuple[str, str, str, int], list[torch.Tensor]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    request_rows: list[dict[str, Any]] = []
    for (condition, metric, request, horizon), values in sorted(cells.items()):
        request_rows.append({
            "condition": condition,
            "metric": metric,
            "request_id": request,
            "horizon": horizon,
            "value": float(torch.cat(values).mean()),
        })
    metrics: dict[str, float] = {}
    conditions = sorted({row["condition"] for row in request_rows})
    kinds = sorted({row["metric"] for row in request_rows})
    for condition in conditions:
        for metric in kinds:
            horizons: list[float] = []
            complete = True
            for horizon in (2, 3, 4):
                values = [
                    float(row["value"])
                    for row in request_rows
                    if row["condition"] == condition
                    and row["metric"] == metric
                    and row["horizon"] == horizon
                ]
                if not values:
                    complete = False
                    break
                value = sum(values) / len(values)
                metrics[f"{condition}_{metric}_h{horizon}"] = value
                horizons.append(value)
            if complete:
                metrics[f"{condition}_{metric}_h2_h4"] = sum(horizons) / 3.0
    return metrics, request_rows


def summarize(
    cells: Mapping[tuple[str, int], list[torch.Tensor]]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    for (request, horizon), values in sorted(cells.items()):
        rows.append(
            {
                "request_id": request,
                "horizon": horizon,
                "route_recall_at_8": float(torch.cat(values).mean()),
            }
        )
    metrics = {}
    for horizon in (2, 3, 4):
        values = [
            float(row["route_recall_at_8"])
            for row in rows if row["horizon"] == horizon
        ]
        if not values:
            raise ValueError(f"closed-loop evaluation has no H{horizon} rows")
        metrics[f"route_recall_h{horizon}"] = sum(values) / len(values)
    metrics["route_recall_h2_h4"] = sum(
        metrics[f"route_recall_h{horizon}"] for horizon in (2, 3, 4)
    ) / 3.0
    return metrics, rows


def main() -> None:
    args = parse_args()
    try:
        from capture_counterfactual_companion import frozen_nodes, load_base_capture
        from capture_transformers_segment import load_target, prefix_hash
    except ImportError as exc:
        raise RuntimeError(
            "closed-loop evaluation requires the pinned Qwen3.5-MoE Transformers build"
        ) from exc
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation {args.output}")
    for field, value in (
        ("source commit", args.source_commit),
        ("execution source commit", args.execution_source_commit),
    ):
        if len(value) != 40 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{field} must be a full lowercase Git SHA")
    if args.limit is not None and args.limit < 1:
        raise ValueError("evaluation limit must be positive")
    if args.native_parity_only:
        args.native_parity = True
    elif args.layer_checkpoint_root is None and args.native_exact_slots is None:
        raise ValueError("learned evaluation requires --layer-checkpoint-root")
    if args.native_exact_slots is not None and args.mode != "exact_top1_plus_draft":
        raise ValueError("native top-k diagnostic requires the exact-top1 S0 wrapper")
    if args.native_exact_slots is not None and args.layer_checkpoint_root is not None:
        raise ValueError("native top-k diagnostic may not load a learned bundle")
    if args.diagnostic_unpromoted_bundle and args.mode not in {
        "exact_top1_plus_draft", "indexed_width16", "int4_top4"
    }:
        raise ValueError("the unpromoted diagnostic override is restricted to S0/S2/INT4")
    if (args.ceiling_bundle is None) != (args.ceiling_parent_sha256 is None):
        raise ValueError(
            "--ceiling-bundle and --ceiling-parent-sha256 must be provided together"
        )
    if args.ceiling_parent_sha256 is not None and (
        len(args.ceiling_parent_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.ceiling_parent_sha256)
    ):
        raise ValueError("ceiling parent SHA256 must be 64 lowercase hex characters")

    base_manifest, trees, sequence_tokens = load_base_capture(args.base_capture)
    labels, companion_manifest = load_node_counterfactual_companion(
        args.native_companion, split="train", training=True
    )
    ceiling = None
    remaining_ceiling_rows: dict[str, list[int]] = {}
    if args.ceiling_bundle is not None:
        assert args.ceiling_parent_sha256 is not None
        ceiling = load_ceiling_bundle(
            args.ceiling_bundle, parent_sha256=args.ceiling_parent_sha256
        )
        for row, request_id in enumerate(ceiling["request_ids"]):
            remaining_ceiling_rows.setdefault(str(request_id), []).append(row)
    if args.limit is not None:
        trees = trees[: args.limit]
    if not trees:
        raise ValueError("closed-loop evaluation contains no trees")
    for tree in trees:
        key = (str(tree["request_id"]), int(tree["source_position"]))
        if key not in labels:
            raise ValueError(f"native companion lacks tree join {key}")
        if tree.get("assigned_split") != "train" or tree.get("external_evaluation") is not False:
            raise PermissionError("closed-loop evaluation is restricted to outer-train")

    checkpoint = IndexedCheckpoint(args.model)
    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "execution_source_commit": args.execution_source_commit,
        "mode": args.mode,
        "shadow_width": args.shadow_width,
        "native_exact_slots": args.native_exact_slots,
        "indexed_active_slots": args.indexed_active_slots,
        "draft_scale": 0.0 if args.native_exact_slots is not None else 1.0,
        "trees": len(trees),
        "native_parity_requested": args.native_parity,
        "native_parity_only": args.native_parity_only,
        "diagnostic_unpromoted_bundle": args.diagnostic_unpromoted_bundle,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "base_capture_manifest_sha256": sha256_file(args.base_capture / "run_manifest.json"),
        "native_companion_manifest_sha256": sha256_file(args.native_companion / "manifest.json"),
        "ceiling_bundle_sha256": (
            None if args.ceiling_bundle is None else sha256_file(args.ceiling_bundle)
        ),
        "ceiling_parent_sha256": args.ceiling_parent_sha256,
        "factual_mixture_evaluation": ceiling is not None,
        "label_only_native_routes": True,
        "native_routes_used_as_model_inputs": False,
        "optimizer_constructed": False,
        "training_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "model_revision": base_manifest.get("model_revision"),
        "companion_schema": companion_manifest.get("schema"),
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)

    target, _config = load_target(args.model, device=args.device)
    installed = install_shadow_experts(
        target,
        args.mode,
        retain_native=True,
        shadow_width=args.shadow_width,
        exact_slots=(1 if args.native_exact_slots is None else args.native_exact_slots),
        draft_scale=(1.0 if args.native_exact_slots is None else 0.0),
        indexed_active_slots=args.indexed_active_slots,
    )
    if not args.native_parity_only and args.native_exact_slots is None:
        assert args.layer_checkpoint_root is not None
        layer_paths = load_shadow_bundle(
            installed,
            args.layer_checkpoint_root,
            source_commit=args.source_commit,
            target_checkpoint_index_sha256=checkpoint.index_sha256,
            s1_fallback_root=args.s1_fallback_root,
            allow_unpromoted_diagnostic=args.diagnostic_unpromoted_bundle,
        )
        write_json_exclusive(
            args.output / "BUNDLE_AUDIT.json",
            {
                "layers": len(layer_paths),
                "checkpoint_sha256": [sha256_file(path) for path in layer_paths],
                "complete": len(layer_paths) == 40,
            },
        )

    cells: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    factual_cells: dict[
        tuple[str, str, str, int], list[torch.Tensor]
    ] = defaultdict(list)
    native_mismatches = 0
    peak_gib = 0.0
    with ShadowRouteHooks(target) as hooks:
        for ordinal, tree in enumerate(trees):
            tokens = sequence_tokens.get(tree["sequence_id"])
            if tokens is None:
                raise ValueError("base capture sequence has no committed ending")
            position = int(tree["source_position"])
            authoritative_prefix = tokens[: position + 1]
            if prefix_hash(authoritative_prefix) != tree["authoritative_prefix_hash"]:
                raise ValueError("authoritative prefix hash changed")
            nodes = frozen_nodes(tree["rows"])
            token_ids = torch.tensor([node.token_id for node in nodes], dtype=torch.int64)
            parents = torch.tensor([
                -1 if node.parent_local_index is None else node.parent_local_index
                for node in nodes
            ], dtype=torch.int64)
            node_mask = torch.ones(len(nodes), dtype=torch.bool)
            native = labels[(str(tree["request_id"]), position)]
            count = int(native["node_mask"].sum())
            if count != len(nodes):
                raise ValueError("native companion/base tree node count differs")
            ceiling_row = None
            if ceiling is not None:
                ceiling_row = resolve_ceiling_row(
                    ceiling,
                    remaining_ceiling_rows,
                    request_id=str(tree["request_id"]),
                    native_ids=native["selected_ids"][:count].long(),
                    native_valid=native["valid"][:count].bool(),
                )

            if args.native_parity:
                with exact_prefix_experts(installed):
                    reference = run_shadow_tree(
                        installed, hooks, authoritative_prefix=authoritative_prefix,
                        token_ids=token_ids, parent_indices=parents, node_mask=node_mask,
                    )
                active = native["valid"][:count]
                native_mismatches += int(
                    (
                        reference.selected_ids[:count].cpu()[active]
                        != native["selected_ids"][:count].long()[active]
                    ).sum()
                )
                if native_mismatches:
                    raise RuntimeError("exact-prefix native cache parity failed")
                del reference

            if not args.native_parity_only:
                result = run_shadow_tree(
                    installed, hooks, authoritative_prefix=authoritative_prefix,
                    token_ids=token_ids, parent_indices=parents, node_mask=node_mask,
                )
                predicted = result.selected_ids[:count].cpu()
                truth = native["selected_ids"][:count].long()
                depth = native["depth"][:count].long()
                valid = native["valid"][:count]
                overlap = route_overlap(predicted, truth)
                request = str(tree["source_request_id"] or tree["request_id"])
                for horizon in (2, 3, 4):
                    active = valid & (depth[:, None] == horizon)
                    if active.any():
                        cells[(request, horizon)].append(overlap[active])
                if ceiling is not None:
                    assert ceiling_row is not None
                    node_ids = torch.full((1, 32, 40, 8), -1, dtype=torch.long)
                    node_ids[0, :count] = predicted
                    anchor_marginals = ceiling["anchor_marginals"][
                        ceiling_row : ceiling_row + 1
                    ].float()
                    target_ids = ceiling["target_ids"][
                        ceiling_row : ceiling_row + 1
                    ].long()
                    future_valid = ceiling["future_valid"][
                        ceiling_row : ceiling_row + 1
                    ].bool()
                    branch_mask = ceiling["branch_mask"][
                        ceiling_row : ceiling_row + 1
                    ].bool()
                    prefix_mismatch = ceiling["prefix_mismatch"][
                        ceiling_row : ceiling_row + 1
                    ].bool()
                    anchor_ids = stable_topk(anchor_marginals, 8)

                    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                    for condition_name in ("learned", "target", "mtp"):
                        posterior = ceiling["posteriors"][condition_name]
                        predicted_factual, inclusion = posterior_native_topk(
                            node_ids,
                            posterior["captured"][ceiling_row : ceiling_row + 1].float(),
                            posterior["other"][ceiling_row : ceiling_row + 1].float(),
                            branch_mask,
                            anchor_marginals,
                        )
                        conditions[condition_name] = (predicted_factual, inclusion)
                    conditions["factual_branch"] = (
                        factual_branch_topk(
                            node_ids,
                            ceiling["factual_branch_indices"][
                                ceiling_row : ceiling_row + 1
                            ].long(),
                            anchor_ids,
                        ),
                        torch.empty(0),
                    )

                    for condition_name, (predicted_factual, inclusion) in conditions.items():
                        recall = slot_coverage(predicted_factual, target_ids)
                        coverage64 = (
                            None
                            if inclusion.numel() == 0
                            else slot_coverage(stable_topk(inclusion, 64), target_ids)
                        )
                        for horizon in (2, 3, 4):
                            active = future_valid[0, horizon - 1]
                            factual_cells[(
                                condition_name, "recall", request, horizon
                            )].append(recall[0, horizon - 1][active])
                            if coverage64 is not None:
                                factual_cells[(
                                    condition_name, "candidate64", request, horizon
                                )].append(coverage64[0, horizon - 1][active])
                            stratum = (
                                "mismatch_recall"
                                if bool(prefix_mismatch[0, horizon - 1])
                                else "matched_recall"
                            )
                            factual_cells[(
                                condition_name, stratum, request, horizon
                            )].append(recall[0, horizon - 1][active])
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                peak_gib = max(peak_gib, torch.cuda.max_memory_reserved() / 2**30)
            print(json.dumps({"tree": ordinal + 1, "total": len(trees)}), flush=True)

    if args.native_parity_only:
        metrics: dict[str, float] = {}
        rows: list[dict[str, Any]] = []
    else:
        metrics, rows = summarize(cells)
    if ceiling is None:
        factual_metrics: dict[str, float] = {}
        factual_rows: list[dict[str, Any]] = []
    else:
        factual_metrics, factual_rows = summarize_factual(factual_cells)
    write_rows(args.output / "request_route_predictions.jsonl", rows)
    write_rows(args.output / "request_factual_predictions.jsonl", factual_rows)
    accuracy_gate_met = bool(
        args.native_parity_only
        or (
            metrics["route_recall_h2_h4"] >= 0.85
            and metrics["route_recall_h4"] >= 0.80
        )
    )
    promotion_eligible = not args.diagnostic_unpromoted_bundle
    result = {
        "schema": RESULT_SCHEMA,
        "mode": args.mode,
        "native_parity_only": args.native_parity_only,
        "diagnostic_unpromoted_bundle": args.diagnostic_unpromoted_bundle,
        "promotion_eligible": promotion_eligible,
        "metrics": metrics,
        "factual_metrics": factual_metrics,
        "native_parity_mismatches": native_mismatches,
        "peak_reserved_gib": peak_gib,
        "gate": {
            "route_recall_h2_h4_required": 0.85,
            "route_recall_h4_required": 0.80,
            "accuracy_thresholds_met": accuracy_gate_met,
            "passed": bool(accuracy_gate_met and promotion_eligible),
        },
        "factual_gate": {
            "route_recall_h2_h4_required": 0.80,
            "route_recall_h4_required": 0.75,
            "accuracy_thresholds_met": bool(
                ceiling is not None
                and factual_metrics.get("mtp_recall_h2_h4", -1.0) >= 0.80
                and factual_metrics.get("mtp_recall_h4", -1.0) >= 0.75
            ),
            "passed": bool(
                ceiling is not None
                and promotion_eligible
                and factual_metrics.get("mtp_recall_h2_h4", -1.0) >= 0.80
                and factual_metrics.get("mtp_recall_h4", -1.0) >= 0.75
            ),
        },
        "training_started": False,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    write_checksums(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
