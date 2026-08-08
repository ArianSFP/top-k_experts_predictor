#!/usr/bin/env python3
"""Evaluate the HARP-RTT v2 B1 adaptive-branch information gate.

This is an optimizer-free, outer-train diagnostic.  It compares matched causal
path occurrence, evaluates a target-route oracle source under the frozen C64
anchor-quota policy, verifies prefix-match nesting, and performs a paired
complete-request bootstrap.  Counterfactual tensors remain labels throughout.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.anchor import LegacyHARPAnchorBridge  # noqa: E402
from harp_rtt.counterfactual import (  # noqa: E402
    CounterfactualDatasetAdapter,
    load_counterfactual_companion,
)
from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt  # noqa: E402
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402


SCHEMA = "harp_rtt_b1_branch_information_gate_v1"
CONTROL_READY = "mtp_matched_control_tree_ready"
CONTROL_RESOLVED = "mtp_matched_control_tree_resolved"
CONTROL_NAMES = ("greedy", "fixed16", "adaptive16", "fixed32", "adaptive32")
HORIZONS = 4
LAYERS = 40
EXPERTS = 256
TOP_K = 8
CANDIDATES = 64
ANCHOR_QUOTA = 48
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42
GATES = {
    "mean_h1_h4": 0.985,
    "h4": 0.970,
    "h4_prefix_mismatch": 0.930,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _load_control_labels(
    capture: Path,
) -> tuple[
    dict[tuple[str, int], dict[str, dict[int, bool]]],
    dict[tuple[str, int], dict[str, float]],
    dict[str, Any],
]:
    occurrence: dict[tuple[str, int], dict[str, dict[int, bool]]] = defaultdict(dict)
    uncertainty: dict[tuple[str, int], dict[str, float]] = {}
    ready_ids: dict[tuple[str, int, str], int] = {}
    resolved_ids: set[tuple[str, int, str]] = set()
    starts = 0
    with (capture / "events.jsonl").open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            event = row.get("event")
            if event == "sequence_start":
                starts += 1
                if (
                    row.get("assigned_split") != "train"
                    or row.get("original_split") != "train"
                    or row.get("inner_partition") != "diagnostic_probe"
                    or row.get("external_evaluation") is not False
                    or row.get("sealed_test_opened") is not False
                ):
                    raise PermissionError(
                        "B1 controls must contain only the frozen outer-train diagnostic probe"
                    )
            elif event == CONTROL_READY:
                request = str(row["request_id"])
                position = int(row["committed_prefix_position"])
                name = str(row["control_name"])
                key = (request, position, name)
                if key in ready_ids:
                    raise ValueError(f"duplicate control ready record {key}")
                ready_ids[key] = int(row["event_id"])
                if name == "adaptive32":
                    nodes = row["nodes"]
                    if len(nodes) != 32:
                        raise ValueError("B1 adaptive32 view is not exact-budgeted")
                    greedy = nodes[:4]
                    logps = greedy[0]["vocab_top64_log_probabilities"]
                    uncertainty[(request, position)] = {
                        "root_entropy": float(greedy[0]["vocabulary_entropy"]),
                        "root_margin": float(logps[0] - logps[1]),
                    }
            elif event == CONTROL_RESOLVED:
                request = str(row["request_id"])
                position = int(row["committed_prefix_position"])
                name = str(row["control_name"])
                key = (request, position, name)
                if key in resolved_ids:
                    raise ValueError(f"duplicate control resolved record {key}")
                resolved_ids.add(key)
                occurrence[(request, position)][name] = {
                    depth: bool(row["path_occurrence_by_horizon"][str(depth)])
                    for depth in range(1, HORIZONS + 1)
                }
    if not starts or set(ready_ids) != resolved_ids:
        raise ValueError("matched control ready/resolved inventory is incomplete")
    expected = set(CONTROL_NAMES)
    if any(set(value) != expected for value in occurrence.values()):
        raise ValueError("one or more B1 sources lack all five controls")
    if set(occurrence) != set(uncertainty):
        raise ValueError("B1 uncertainty/control source inventories differ")
    return (
        dict(occurrence),
        uncertainty,
        {"sequences": starts, "sources": len(occurrence)},
    )


def _stable_top(scores: Tensor, width: int) -> Tensor:
    return torch.argsort(scores.float(), dim=-1, descending=True, stable=True)[
        ..., :width
    ]


def _coverage(ids: Tensor, target: Tensor) -> Tensor:
    return (ids.unsqueeze(-1) == target.unsqueeze(-2)).any(dim=-2).float().mean(dim=-1)


def _finite_mean_or_none(values: np.ndarray) -> float | None:
    finite = np.isfinite(values)
    if not finite.any():
        return None
    return float(values[finite].mean())


def _anchor_branch_union(anchor: Tensor, branch: Tensor) -> Tensor:
    """Anchor top-48 plus the best 16 distinct oracle-branch experts."""

    if anchor.shape != branch.shape or anchor.shape[-1] != EXPERTS:
        raise ValueError("anchor/branch dense score geometry mismatch")
    leading = anchor.shape[:-1]
    rows = int(np.prod(leading))
    anchor_order = _stable_top(anchor, EXPERTS).reshape(rows, EXPERTS)
    branch_order = _stable_top(branch, EXPERTS).reshape(rows, EXPERTS)
    result = torch.full((rows, CANDIDATES), -1, dtype=torch.int64)
    selected = torch.zeros((rows, EXPERTS), dtype=torch.bool)
    result[:, :ANCHOR_QUOTA] = anchor_order[:, :ANCHOR_QUOTA].cpu()
    selected.scatter_(1, result[:, :ANCHOR_QUOTA], True)
    counts = torch.full((rows,), ANCHOR_QUOTA, dtype=torch.int64)
    row_ids = torch.arange(rows)
    for rank in range(EXPERTS):
        ids = branch_order[:, rank].cpu()
        keep = (~selected.gather(1, ids[:, None]).squeeze(1)) & (counts < CANDIDATES)
        if keep.any():
            active_rows = row_ids[keep]
            active_ids = ids[keep]
            result[active_rows, counts[keep]] = active_ids
            selected[active_rows, active_ids] = True
            counts[keep] += 1
        if bool((counts == CANDIDATES).all()):
            break
    if (result < 0).any():
        raise RuntimeError("oracle branch source could not fill C64")
    return result.reshape(*leading, CANDIDATES)


def _oracle_branch_scores(
    counterfactual: Mapping[str, Tensor]
) -> tuple[Tensor, Tensor]:
    """Target-path-probability-weighted route evidence, deduplicated by node."""

    logits = counterfactual["router_logits"].float().cpu()
    valid = counterfactual["valid"].bool().cpu()
    node_ids = counterfactual["node_local_indices"].long().cpu()
    target_logp = counterfactual["target_path_logp"].float().cpu()
    batch = int(logits.shape[0])
    scores = torch.zeros((batch, HORIZONS, LAYERS, EXPERTS), dtype=torch.float32)
    captured_mass = torch.zeros((batch, HORIZONS), dtype=torch.float32)
    for sample in range(batch):
        for horizon in range(1, HORIZONS):
            seen: set[int] = set()
            for path in range(logits.shape[1]):
                node = int(node_ids[sample, path, horizon].item())
                if node < 0 or node in seen:
                    continue
                if not bool(valid[sample, path, horizon].all()):
                    continue
                seen.add(node)
                probability = math.exp(float(target_logp[sample, path, horizon]))
                if not math.isfinite(probability) or probability < 0.0:
                    raise ValueError("invalid target counterfactual path probability")
                captured_mass[sample, horizon] += probability
                scores[sample, horizon] += probability * torch.softmax(
                    logits[sample, path, horizon], dim=-1
                )
            if not seen:
                raise ValueError(
                    f"sample {sample} lacks oracle branches at H{horizon + 1}"
                )
            if captured_mass[sample, horizon] > 1.0 + 2e-5:
                raise ValueError("deduplicated captured target path mass exceeds one")
    return scores, captured_mass


def _request_macro(
    values: np.ndarray,
    request_ids: Sequence[str],
    sample_mask: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    if values.shape[0] != len(request_ids):
        raise ValueError("metric sample dimension differs from request IDs")
    if sample_mask is None:
        sample_mask = np.ones(values.shape, dtype=bool)
    elif sample_mask.shape != values.shape:
        sample_mask = np.broadcast_to(sample_mask, values.shape)
    sample_mask = np.asarray(sample_mask, dtype=bool) & np.isfinite(values)
    grouped_values: dict[str, list[float]] = defaultdict(list)
    grouped_masks: dict[str, list[bool]] = defaultdict(list)
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = sample_mask.reshape(sample_mask.shape[0], -1)
    for index, request in enumerate(request_ids):
        grouped_values[request].extend(flat_values[index].tolist())
        grouped_masks[request].extend(flat_mask[index].tolist())
    per_request: dict[str, float] = {}
    for request in sorted(grouped_values):
        selected = np.asarray(grouped_masks[request], dtype=bool)
        if selected.any():
            per_request[request] = float(
                np.asarray(grouped_values[request], dtype=np.float64)[selected].mean()
            )
    if not per_request:
        return float("nan"), {}
    return float(np.mean(list(per_request.values()))), per_request


def _metric_summary(
    values: np.ndarray,
    request_ids: Sequence[str],
    *,
    sample_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    mean, per_request = _request_macro(values, request_ids, sample_mask)
    return {
        "request_macro_mean": mean,
        "requests": len(per_request),
        "per_request": per_request,
    }


def _bootstrap_delta(
    candidate: dict[str, float],
    reference: dict[str, float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    common = sorted(set(candidate) & set(reference))
    if not common:
        raise ValueError("paired bootstrap has no complete requests")
    delta = np.asarray([candidate[key] - reference[key] for key in common])
    rng = np.random.default_rng(seed)
    sampled = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = rng.integers(0, len(common), size=len(common))
        sampled[replicate] = delta[indices].mean()
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return {
        "requests": len(common),
        "replicates": replicates,
        "seed": seed,
        "mean_delta": float(delta.mean()),
        "ci95": [float(lower), float(upper)],
        "lower_bound_positive": bool(lower > 0.0),
    }


def _first_divergence(greedy_match: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if greedy_match.shape[1] != HORIZONS:
        raise ValueError("greedy match must be [N,4]")
    if not greedy_match[:, 0].all():
        raise ValueError("exact committed H1 must always match")
    nesting_failures = 0
    divergence = np.zeros(len(greedy_match), dtype=np.int64)
    for index, row in enumerate(greedy_match):
        for horizon in range(1, HORIZONS):
            if row[horizon] and not row[horizon - 1]:
                nesting_failures += 1
            if not row[horizon] and divergence[index] == 0:
                divergence[index] = horizon + 1
    if nesting_failures:
        raise ValueError(f"prefix-match nesting failed on {nesting_failures} rows")
    return divergence, {
        "verified": True,
        "failures": 0,
        "first_divergence_counts": {
            f"H{depth}": int((divergence == depth).sum()) for depth in (2, 3, 4)
        },
        "fully_matched_h4": int((divergence == 0).sum()),
    }


def _quartiles(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    thresholds = np.quantile(values, [0.25, 0.5, 0.75]).tolist()
    return np.digitize(values, thresholds, right=True), [float(v) for v in thresholds]


def _stratified_metrics(
    metrics: Mapping[str, np.ndarray],
    request_ids: Sequence[str],
    greedy_match: np.ndarray,
    divergence: np.ndarray,
    entropy_bin: np.ndarray,
    margin_bin: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        output: dict[str, Any] = {}
        expanded = mask[:, :, None]
        for name, values in metrics.items():
            mean, per_request = _request_macro(values, request_ids, expanded)
            output[name] = {"request_macro_mean": mean, "requests": len(per_request)}
        return output

    result["prefix_match"] = {
        f"H{horizon + 1}_matched": summarize(
            np.eye(HORIZONS, dtype=bool)[None, horizon, :]
            & greedy_match[:, horizon, None]
        )
        for horizon in range(HORIZONS)
    }
    for depth in (2, 3, 4):
        sample = divergence == depth
        mask = np.broadcast_to(sample[:, None], (len(sample), HORIZONS))
        result[f"first_divergence_H{depth}"] = summarize(mask)
    for prefix, bins in (("entropy", entropy_bin), ("margin", margin_bin)):
        result[prefix] = {}
        for bucket in range(4):
            sample = bins == bucket
            mask = np.broadcast_to(sample[:, None], (len(sample), HORIZONS))
            result[prefix][f"Q{bucket + 1}"] = summarize(mask)

    result["horizon"] = {}
    for horizon in range(HORIZONS):
        mask = np.zeros((len(request_ids), HORIZONS), dtype=bool)
        mask[:, horizon] = True
        result["horizon"][f"H{horizon + 1}"] = summarize(mask)

    result["layer"] = {}
    for layer in range(LAYERS):
        layer_result: dict[str, Any] = {}
        for name, values in metrics.items():
            mean, per_request = _request_macro(
                values[:, :, layer : layer + 1], request_ids
            )
            layer_result[name] = {
                "request_macro_mean": mean,
                "requests": len(per_request),
            }
        result["layer"][str(layer)] = layer_result

    result["block_phase_layer_mod_4"] = {}
    for phase in range(4):
        layer_mask = np.asarray([layer % 4 == phase for layer in range(LAYERS)])
        phase_result: dict[str, Any] = {}
        for name, values in metrics.items():
            selected = values[:, :, layer_mask]
            mean, per_request = _request_macro(selected, request_ids)
            phase_result[name] = {
                "request_macro_mean": mean,
                "requests": len(per_request),
            }
        result["block_phase_layer_mod_4"][str(phase)] = phase_result
    return result


def _path_occurrence_summary(
    occurrence_rows: Sequence[dict[str, dict[int, bool]]],
    request_ids: Sequence[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in CONTROL_NAMES:
        values = np.asarray(
            [
                [float(row[name][horizon]) for horizon in range(1, HORIZONS + 1)]
                for row in occurrence_rows
            ],
            dtype=np.float64,
        )
        horizon_report = {}
        for horizon in range(HORIZONS):
            mean, per_request = _request_macro(
                values[:, horizon : horizon + 1], request_ids
            )
            horizon_report[f"H{horizon + 1}"] = {
                "request_macro_path_occurrence": mean,
                "requests": len(per_request),
            }
        mean, per_request = _request_macro(values, request_ids)
        report[name] = {
            "mean_h1_h4_request_macro_path_occurrence": mean,
            "horizons": horizon_report,
            "requests": len(per_request),
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite B1 output {args.output}")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    for path in (
        args.capture,
        args.index_root,
        args.corpus_root,
        args.companion,
        args.static_dir,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)
    controls, uncertainty, control_inventory = _load_control_labels(args.capture)
    labels, companion_manifest = load_counterfactual_companion(
        args.companion, split="train", training=True
    )
    base = HarpRTTDataset(
        args.index_root,
        "train",
        corpus_root=args.corpus_root,
        max_tree_nodes=32,
    )
    dataset = CounterfactualDatasetAdapter(base, labels, split="train", training=True)
    if len(dataset) != 2048 or len(labels) != 2048 or len(controls) != 2048:
        raise ValueError(
            "B1 requires exactly 2,048 one-to-one source/control/companion records"
        )

    device = torch.device(args.device)
    bridge, anchor_provenance = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint,
        args.target_preprocessing,
        args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    bridge.to(device).eval()
    static = load_static_target_artifacts(
        args.static_dir, device="cpu", load_embedding=False
    )
    geometry = static.geometry

    request_ids: list[str] = []
    source_positions: list[int] = []
    occurrence_rows: list[dict[str, dict[int, bool]]] = []
    entropy: list[float] = []
    margin: list[float] = []
    metric_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    captured_masses: list[np.ndarray] = []
    cf_metric_rows: dict[str, list[np.ndarray]] = defaultdict(list)

    with torch.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            items = [
                dataset[index]
                for index in range(start, min(start + args.batch_size, len(dataset)))
            ]
            batch = collate_harp_rtt(items)
            metadata = batch["metadata"]
            batch_requests = [str(value) for value in metadata["request_id"]]
            batch_positions = [int(value) for value in metadata["position"]]
            keys = list(zip(batch_requests, batch_positions, strict=True))
            if any(key not in controls or key not in uncertainty for key in keys):
                raise KeyError("dataset row is not aligned to a B1 control record")
            request_ids.extend(batch_requests)
            source_positions.extend(batch_positions)
            occurrence_rows.extend(controls[key] for key in keys)
            entropy.extend(uncertainty[key]["root_entropy"] for key in keys)
            margin.extend(uncertainty[key]["root_margin"] for key in keys)

            device_batch = {
                "inputs": _move(batch["inputs"], device),
                "anchor_inputs": _move(batch["anchor_inputs"], device),
            }
            enabled = device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=enabled,
            ):
                anchor_outputs = bridge(batch=device_batch)
            anchor_scores = (
                anchor_outputs["future_router_scores"][:, :HORIZONS].float().cpu()
            )
            target_ids = batch["targets"]["future_selected_ids"].long().cpu()
            counterfactual = batch["targets"]["counterfactual"]
            branch_scores, captured_mass = _oracle_branch_scores(counterfactual)
            captured_masses.append(captured_mass.numpy())

            anchor_top8 = _stable_top(anchor_scores, TOP_K)
            anchor_top64 = _stable_top(anchor_scores, CANDIDATES)
            oracle_c64 = _anchor_branch_union(anchor_scores, branch_scores)
            oracle_c64[:, 0] = anchor_top64[:, 0]
            route_top8 = _stable_top(branch_scores, TOP_K)
            route_top8[:, 0] = anchor_top8[:, 0]

            metric_rows["anchor_recall_at_8"].append(
                _coverage(anchor_top8, target_ids).numpy()
            )
            metric_rows["anchor_coverage_at_64"].append(
                _coverage(anchor_top64, target_ids).numpy()
            )
            metric_rows["oracle_route_recall_at_8"].append(
                _coverage(route_top8, target_ids).numpy()
            )
            metric_rows["oracle_coverage_at_64"].append(
                _coverage(oracle_c64, target_ids).numpy()
            )

            q = counterfactual["query_coordinates"].float().cpu()
            native = counterfactual["router_logits"].float().cpu()
            valid = counterfactual["valid"].bool().cpu()
            reconstructed = geometry.score_coordinates(q)
            centered = native - native.mean(dim=-1, keepdim=True)
            native_probability = torch.softmax(native, dim=-1)
            reconstructed_log_probability = torch.log_softmax(reconstructed, dim=-1)
            kl = (
                native_probability
                * (
                    torch.log(native_probability.clamp_min(1e-30))
                    - reconstructed_log_probability
                )
            ).sum(dim=-1)
            huber = torch.nn.functional.smooth_l1_loss(
                reconstructed, centered, reduction="none"
            ).mean(dim=-1)
            cosine = torch.nn.functional.cosine_similarity(
                reconstructed, centered, dim=-1
            )
            geometry_ids = _stable_top(reconstructed, TOP_K)
            stored_ids = counterfactual["selected_ids"].long().cpu()
            top8_recall = _coverage(geometry_ids, stored_ids)
            for name, values in (
                ("router_kl", kl),
                ("centered_logit_huber", huber),
                ("centered_logit_cosine", cosine),
                ("counterfactual_top8_recall", top8_recall),
            ):
                values = values.masked_fill(~valid, float("nan"))
                # Average unique/selected path rows into one sample/horizon/layer grid.
                cf_metric_rows[name].append(torch.nanmean(values, dim=1).numpy())

    metrics = {name: np.concatenate(rows, axis=0) for name, rows in metric_rows.items()}
    cf_metrics = {
        name: np.concatenate(rows, axis=0) for name, rows in cf_metric_rows.items()
    }
    masses = np.concatenate(captured_masses, axis=0)
    greedy_match = np.asarray(
        [
            [row["greedy"][horizon] for horizon in range(1, HORIZONS + 1)]
            for row in occurrence_rows
        ],
        dtype=bool,
    )
    divergence, nesting = _first_divergence(greedy_match)
    entropy_array = np.asarray(entropy, dtype=np.float64)
    margin_array = np.asarray(margin, dtype=np.float64)
    entropy_bin, entropy_thresholds = _quartiles(entropy_array)
    margin_bin, margin_thresholds = _quartiles(margin_array)

    aggregate: dict[str, Any] = {}
    per_request: dict[str, dict[str, float]] = {}
    for name, values in metrics.items():
        mean, requests = _request_macro(values, request_ids)
        horizons = []
        for horizon in range(HORIZONS):
            horizon_mean, horizon_requests = _request_macro(
                values[:, horizon : horizon + 1], request_ids
            )
            horizons.append(
                {
                    "horizon": horizon + 1,
                    "request_macro_mean": horizon_mean,
                    "requests": len(horizon_requests),
                }
            )
        aggregate[name] = {
            "mean_h1_h4_request_macro": mean,
            "horizons": horizons,
        }
        per_request[name] = requests
    for recall_name, coverage_name in (
        ("anchor_recall_at_8", "anchor_coverage_at_64"),
        ("oracle_route_recall_at_8", "oracle_coverage_at_64"),
    ):
        recall = aggregate[recall_name]["mean_h1_h4_request_macro"]
        coverage = aggregate[coverage_name]["mean_h1_h4_request_macro"]
        aggregate[recall_name]["recall_over_coverage"] = recall / coverage

    h4_mismatch_mask = np.zeros_like(metrics["oracle_coverage_at_64"], dtype=bool)
    h4_mismatch_mask[:, 3, :] = ~greedy_match[:, 3, None]
    mismatch_mean, mismatch_requests = _request_macro(
        metrics["oracle_coverage_at_64"], request_ids, h4_mismatch_mask
    )
    aggregate["oracle_coverage_at_64"][
        "h4_prefix_mismatch_request_macro"
    ] = mismatch_mean
    aggregate["oracle_coverage_at_64"]["h4_prefix_mismatch_requests"] = len(
        mismatch_requests
    )

    oracle_mean = aggregate["oracle_coverage_at_64"]["mean_h1_h4_request_macro"]
    oracle_h4 = aggregate["oracle_coverage_at_64"]["horizons"][3]["request_macro_mean"]
    gate_results = {
        "mean_h1_h4": {
            "value": oracle_mean,
            "threshold": GATES["mean_h1_h4"],
            "passed": bool(oracle_mean >= GATES["mean_h1_h4"]),
        },
        "h4": {
            "value": oracle_h4,
            "threshold": GATES["h4"],
            "passed": bool(oracle_h4 >= GATES["h4"]),
        },
        "h4_prefix_mismatch": {
            "value": mismatch_mean,
            "threshold": GATES["h4_prefix_mismatch"],
            "passed": bool(mismatch_mean >= GATES["h4_prefix_mismatch"]),
        },
    }
    all_gates = all(item["passed"] for item in gate_results.values())

    cf_summary: dict[str, Any] = {}
    for name, values in cf_metrics.items():
        cf_summary[name] = {
            "mean": _finite_mean_or_none(values),
            "by_horizon": [
                _finite_mean_or_none(values[:, horizon]) for horizon in range(HORIZONS)
            ],
            "by_layer": [
                _finite_mean_or_none(values[:, :, layer]) for layer in range(LAYERS)
            ],
            "by_block_phase_layer_mod_4": [
                _finite_mean_or_none(values[:, :, phase::4]) for phase in range(4)
            ],
        }

    report = {
        "schema": SCHEMA,
        "passed": all_gates,
        "decision": (
            "promote_to_B2" if all_gates else "stop_before_B2_change_tree_policy"
        ),
        "stage": "B1_oracle",
        "split": "outer_train_inner_diagnostic_probe",
        "source_positions": len(request_ids),
        "requests": len(set(request_ids)),
        "layers": LAYERS,
        "candidate_contract": {
            "width": CANDIDATES,
            "h1": "frozen_anchor_top64",
            "h2_h4": "frozen_anchor_top48_plus_16_distinct_target_route_oracle_experts",
            "branch_score": (
                "sum_over_unique_captured_nodes(target_path_probability * "
                "softmax(native_target_router_logits))"
            ),
            "path_duplicates_deduplicated_by_node_local_index": True,
            "other_mass": "preserved_implicitly_by_the_anchor_top48",
            "tie_policy": "descending_score_stable_expert_id",
        },
        "aggregate_metrics": aggregate,
        "path_occurrence": _path_occurrence_summary(occurrence_rows, request_ids),
        "prefix_match_nesting": nesting,
        "uncertainty_strata": {
            "root_entropy_quartile_thresholds": entropy_thresholds,
            "root_margin_quartile_thresholds": margin_thresholds,
        },
        "stratified_metrics": _stratified_metrics(
            metrics,
            request_ids,
            greedy_match,
            divergence,
            entropy_bin,
            margin_bin,
        ),
        "counterfactual_geometry_metrics": cf_summary,
        "counterfactual_stratified_metrics": _stratified_metrics(
            cf_metrics,
            request_ids,
            greedy_match,
            divergence,
            entropy_bin,
            margin_bin,
        ),
        "path_occurrence_stratified": _stratified_metrics(
            {
                f"{name}_path_occurrence": np.repeat(
                    np.asarray(
                        [
                            [
                                float(row[name][horizon])
                                for horizon in range(1, HORIZONS + 1)
                            ]
                            for row in occurrence_rows
                        ],
                        dtype=np.float64,
                    )[:, :, None],
                    LAYERS,
                    axis=2,
                )
                for name in CONTROL_NAMES
            },
            request_ids,
            greedy_match,
            divergence,
            entropy_bin,
            margin_bin,
        ),
        "query_prediction_metrics": {
            "applicable": False,
            "reason": "B1 has no learned branch translator; query Huber/cosine begin at B2",
        },
        "captured_target_path_mass": {
            f"H{horizon + 1}": {
                "mean": float(masses[:, horizon].mean()),
                "minimum": float(masses[:, horizon].min()),
                "maximum": float(masses[:, horizon].max()),
            }
            for horizon in range(1, HORIZONS)
        },
        "paired_complete_request_bootstrap": {
            "oracle_vs_anchor_mean_h1_h4_coverage": _bootstrap_delta(
                per_request["oracle_coverage_at_64"],
                per_request["anchor_coverage_at_64"],
            ),
            "h4_prefix_mismatch_oracle_vs_anchor": _bootstrap_delta(
                mismatch_requests,
                _request_macro(
                    metrics["anchor_coverage_at_64"], request_ids, h4_mismatch_mask
                )[1],
            ),
        },
        "gates": gate_results,
        "all_gates_passed": all_gates,
        "control_inventory": control_inventory,
        "bindings": {
            "capture": str(args.capture.resolve()),
            "capture_manifest_sha256": _sha256(args.capture / "run_manifest.json"),
            "companion": str(args.companion.resolve()),
            "companion_manifest_sha256": _sha256(args.companion / "manifest.json"),
            "index": str(args.index_root.resolve()),
            "static_geometry_sha256": static.manifest["files"][
                "router_geometry.safetensors"
            ]["sha256"],
            "anchor": anchor_provenance,
        },
        "safety": {
            "optimizer_started": False,
            "training_started": False,
            "counterfactual_model_input": False,
            "sealed_test_opened": False,
            "validation_opened": False,
            "calibration_opened": False,
        },
    }

    args.output.mkdir(parents=True)
    (args.output / "B1_ORACLE_GATE_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "B1_REQUEST_METRICS.npz",
        request_ids=np.asarray(request_ids),
        source_positions=np.asarray(source_positions, dtype=np.int64),
        greedy_match=greedy_match,
        first_divergence=divergence,
        root_entropy=entropy_array,
        root_margin=margin_array,
        captured_target_path_mass=masses,
        **metrics,
        **{f"cf_{name}": values for name, values in cf_metrics.items()},
    )
    hashes = {
        path.name: _sha256(path)
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    (args.output / "SHA256SUMS.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
