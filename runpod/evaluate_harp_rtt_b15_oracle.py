#!/usr/bin/env python3
"""Evaluate HARP-RTT B1.1/B1.5 with exact selected-set inclusion oracles."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = REPO_ROOT / "runpod" / "transformers_mtp_bridge"
for candidate in (str(REPO_ROOT), str(BRIDGE_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from harp_rtt.anchor import LegacyHARPAnchorBridge  # noqa: E402
from harp_rtt.b15 import (  # noqa: E402
    B15_ORACLE_SCHEMA,
    H1_C64_GATE,
    add_other_anchor_mass,
    b15_candidate_union_sha256,
    candidate_union,
    global_candidates,
    selected_set_inclusion_mass,
)
from harp_rtt.counterfactual import (  # noqa: E402
    COUNTERFACTUAL_SCHEMA,
    CounterfactualDatasetAdapter,
    load_counterfactual_companion,
)
from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt  # noqa: E402
from harp_rtt.node_counterfactual import (  # noqa: E402
    NODE_COUNTERFACTUAL_SCHEMA,
    NodeCounterfactualDatasetAdapter,
    load_node_counterfactual_companion,
)


CONTROL_READY = "mtp_matched_control_tree_ready"
CONTROL_RESOLVED = "mtp_matched_control_tree_resolved"
CONTROL_NAMES = ("greedy", "fixed16", "adaptive16", "fixed32", "adaptive32")
HORIZONS = 4
LAYERS = 40
EXPERTS = 256
TOP_K = 8
CANDIDATES = 64
QUOTA_POLICIES = ((48, 16), (40, 24), (32, 32))
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42
GATES = {
    "mean_h2_h4": 0.985,
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


def _stable_top(scores: Tensor, width: int) -> Tensor:
    return torch.argsort(scores.float(), dim=-1, descending=True, stable=True)[..., :width]


def _coverage(ids: Tensor, target: Tensor) -> Tensor:
    return (ids.unsqueeze(-1) == target.unsqueeze(-2)).any(-2).float().mean(-1)


def _load_controls(
    capture: Path, *, expected_inner_partition: str
) -> tuple[
    dict[tuple[str, int], dict[str, dict[int, bool]]],
    dict[tuple[str, int], dict[str, float]],
    dict[str, Any],
]:
    occurrence: dict[tuple[str, int], dict[str, dict[int, bool]]] = defaultdict(dict)
    uncertainty: dict[tuple[str, int], dict[str, float]] = {}
    ready: set[tuple[str, int, str]] = set()
    resolved: set[tuple[str, int, str]] = set()
    sequences = 0
    with (capture / "events.jsonl").open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            event = row.get("event")
            if event == "sequence_start":
                sequences += 1
                if (
                    row.get("assigned_split") != "train"
                    or row.get("original_split") != "train"
                    or row.get("inner_partition") != expected_inner_partition
                    or row.get("external_evaluation") is not False
                    or row.get("sealed_test_opened") is not False
                ):
                    raise PermissionError("B1.5 controls are not the declared outer-train partition")
            elif event == CONTROL_READY:
                key = (
                    str(row["request_id"]),
                    int(row["committed_prefix_position"]),
                    str(row["control_name"]),
                )
                if key in ready:
                    raise ValueError(f"duplicate control ready row {key}")
                ready.add(key)
                if key[2] == "adaptive32":
                    nodes = row["nodes"]
                    if len(nodes) != 32:
                        raise ValueError("adaptive32 control is not exact-budgeted")
                    logp = nodes[0]["vocab_top64_log_probabilities"]
                    uncertainty[key[:2]] = {
                        "root_entropy": float(nodes[0]["vocabulary_entropy"]),
                        "root_margin": float(logp[0] - logp[1]),
                    }
            elif event == CONTROL_RESOLVED:
                key = (
                    str(row["request_id"]),
                    int(row["committed_prefix_position"]),
                    str(row["control_name"]),
                )
                if key in resolved:
                    raise ValueError(f"duplicate control resolved row {key}")
                resolved.add(key)
                occurrence[key[:2]][key[2]] = {
                    depth: bool(row["path_occurrence_by_horizon"][str(depth)])
                    for depth in range(1, HORIZONS + 1)
                }
    if not sequences or ready != resolved:
        raise ValueError("matched-control inventory is incomplete")
    if any(set(row) != set(CONTROL_NAMES) for row in occurrence.values()):
        raise ValueError("a source lacks one or more matched controls")
    if set(occurrence) != set(uncertainty):
        raise ValueError("control and uncertainty identities disagree")
    return dict(occurrence), uncertainty, {
        "sequences": sequences,
        "sources": len(occurrence),
        "inner_partition": expected_inner_partition,
    }


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
    mask = np.asarray(sample_mask, dtype=bool) & np.isfinite(values)
    grouped_values: dict[str, list[float]] = defaultdict(list)
    grouped_masks: dict[str, list[bool]] = defaultdict(list)
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    for index, request in enumerate(request_ids):
        grouped_values[request].extend(flat_values[index].tolist())
        grouped_masks[request].extend(flat_mask[index].tolist())
    per_request = {}
    for request in sorted(grouped_values):
        selected = np.asarray(grouped_masks[request], dtype=bool)
        if selected.any():
            per_request[request] = float(
                np.asarray(grouped_values[request], dtype=np.float64)[selected].mean()
            )
    if not per_request:
        return float("nan"), {}
    return float(np.mean(list(per_request.values()))), per_request


def _bootstrap_delta(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    common = sorted(set(candidate) & set(reference))
    if not common:
        raise ValueError("paired bootstrap has no complete requests")
    delta = np.asarray([candidate[key] - reference[key] for key in common])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, len(common), size=len(common))
        sampled[replicate] = delta[indices].mean()
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return {
        "requests": len(common),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "mean_delta": float(delta.mean()),
        "ci95": [float(lower), float(upper)],
        "lower_bound_positive": bool(lower > 0),
    }


def _first_divergence(greedy_match: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if greedy_match.shape[1] != HORIZONS or not greedy_match[:, 0].all():
        raise ValueError("greedy prefix contract requires exact H1")
    divergence = np.zeros(len(greedy_match), dtype=np.int64)
    failures = 0
    for index, row in enumerate(greedy_match):
        for horizon in range(1, HORIZONS):
            if row[horizon] and not row[horizon - 1]:
                failures += 1
            if not row[horizon] and divergence[index] == 0:
                divergence[index] = horizon + 1
    if failures:
        raise ValueError(f"prefix-match nesting failed on {failures} rows")
    return divergence, {
        "verified": True,
        "failures": 0,
        "first_divergence_counts": {
            f"H{depth}": int((divergence == depth).sum()) for depth in (2, 3, 4)
        },
        "fully_matched_h4": int((divergence == 0).sum()),
    }


def _selection_mask(counterfactual: Mapping[str, Tensor], budget: str) -> Tensor | None:
    if "node_mask" not in counterfactual:
        if budget != "4":
            raise ValueError("v2 path companion supports only the frozen budget 4")
        return None
    if budget == "all":
        return counterfactual["node_mask"].bool().cpu()
    index = {"4": 0, "8": 1, "16": 2}.get(budget)
    if index is None:
        raise ValueError("path budget must be 4, 8, 16, or all")
    return counterfactual["budget_node_masks"][:, index].bool().cpu()


def _horizon_rows(
    counterfactual: Mapping[str, Tensor],
    *,
    sample: int,
    horizon: int,
    budget: str,
    probability_key: str,
) -> tuple[Tensor, Tensor, Tensor, list[int]]:
    if "node_mask" in counterfactual:
        selection = _selection_mask(counterfactual, budget)
        assert selection is not None
        depth = counterfactual["depth"][sample].long().cpu()
        valid_nodes = selection[sample] & (depth == horizon + 1)
        indices = torch.nonzero(valid_nodes, as_tuple=False).flatten()
        if not len(indices):
            raise ValueError(f"node companion has no selected H{horizon + 1} route")
        ids = counterfactual["selected_ids"][sample, indices].long().cpu()
        valid = counterfactual["valid"][sample, indices].bool().cpu()
        logp = counterfactual[probability_key][sample, indices].float().cpu()
        return ids, logp.exp(), valid, indices.tolist()

    node_ids = counterfactual["node_local_indices"][sample, :, horizon].long().cpu()
    valid = counterfactual["valid"][sample, :, horizon].bool().cpu()
    logp = counterfactual[probability_key][sample, :, horizon].float().cpu()
    selected = counterfactual["selected_ids"][sample, :, horizon].long().cpu()
    rows, ids, probabilities, validity = [], [], [], []
    seen = set()
    for path, node in enumerate(node_ids.tolist()):
        if node < 0 or node in seen or not bool(valid[path].all()):
            continue
        seen.add(node)
        rows.append(node)
        ids.append(selected[path])
        probabilities.append(logp[path].exp())
        validity.append(valid[path])
    if not rows:
        raise ValueError(f"path companion has no selected H{horizon + 1} route")
    return torch.stack(ids), torch.stack(probabilities), torch.stack(validity), rows


def _oracle_batch(
    counterfactual: Mapping[str, Tensor], anchor_scores: Tensor, *, budget: str
) -> dict[str, Tensor]:
    batch = anchor_scores.shape[0]
    target_branch = torch.zeros_like(anchor_scores, dtype=torch.float32)
    source_branch = torch.zeros_like(anchor_scores, dtype=torch.float32)
    target_mass = torch.zeros(anchor_scores.shape[:-1], dtype=torch.float32)
    source_mass = torch.zeros_like(target_mass)
    for sample in range(batch):
        for horizon in range(1, HORIZONS):
            for key, branch, captured in (
                ("target_path_logp", target_branch, target_mass),
                ("source_path_logp", source_branch, source_mass),
            ):
                ids, probabilities, valid, _ = _horizon_rows(
                    counterfactual,
                    sample=sample,
                    horizon=horizon,
                    budget=budget,
                    probability_key=key,
                )
                mass, layer_mass = selected_set_inclusion_mass(
                    ids, probabilities, valid, experts=EXPERTS
                )
                branch[sample, horizon] = mass
                captured[sample, horizon] = layer_mass
    target_full, target_other, target_error = add_other_anchor_mass(
        target_branch, target_mass, anchor_scores, exact_k=TOP_K
    )
    source_full, source_other, source_error = add_other_anchor_mass(
        source_branch, source_mass, anchor_scores, exact_k=TOP_K
    )
    return {
        "target_branch": target_branch,
        "target_full": target_full,
        "target_captured_mass": target_mass,
        "target_other_mass": target_other,
        "target_cardinality_error": target_error,
        "source_branch": source_branch,
        "source_full": source_full,
        "source_captured_mass": source_mass,
        "source_other_mass": source_other,
        "source_cardinality_error": source_error,
    }


def _tree_token_path(tree: Mapping[str, Tensor], sample: int, node: int) -> tuple[int, ...]:
    tokens = tree["token_ids"][sample].long().cpu()
    parents = tree["parent"][sample].long().cpu()
    path = []
    current = node
    while current >= 0:
        path.append(int(tokens[current]))
        current = int(parents[current])
    return tuple(reversed(path))


def _factual_candidates(
    counterfactual: Mapping[str, Tensor],
    tree: Mapping[str, Tensor],
    factual_tokens: Tensor,
    anchor_scores: Tensor,
    *,
    budget: str,
) -> tuple[Tensor, Tensor]:
    anchor_order = _stable_top(anchor_scores, EXPERTS)
    result = anchor_order[..., :CANDIDATES].clone()
    occurrence = torch.zeros(anchor_scores.shape[:2], dtype=torch.bool)
    occurrence[:, 0] = True
    for sample in range(anchor_scores.shape[0]):
        for horizon in range(1, HORIZONS):
            _, _, _, nodes = _horizon_rows(
                counterfactual,
                sample=sample,
                horizon=horizon,
                budget=budget,
                probability_key="target_path_logp",
            )
            wanted = tuple(int(value) for value in factual_tokens[sample, : horizon + 1])
            match = next(
                (node for node in nodes if _tree_token_path(tree, sample, node) == wanted),
                None,
            )
            if match is None:
                continue
            occurrence[sample, horizon] = True
            if "node_mask" in counterfactual:
                exact = counterfactual["selected_ids"][sample, match].long().cpu()
            else:
                node_ids = counterfactual["node_local_indices"][sample, :, horizon]
                path = int(torch.nonzero(node_ids == match, as_tuple=False)[0])
                exact = counterfactual["selected_ids"][sample, path, horizon].long().cpu()
            for layer in range(LAYERS):
                chosen = exact[layer].tolist()
                selected = set(chosen)
                chosen.extend(
                    int(expert)
                    for expert in anchor_order[sample, horizon, layer].tolist()
                    if int(expert) not in selected
                )
                result[sample, horizon, layer] = torch.tensor(chosen[:CANDIDATES])
    return result, occurrence


def _target_greedy_candidates(
    counterfactual: Mapping[str, Tensor],
    tree: Mapping[str, Tensor],
    anchor_scores: Tensor,
    *,
    budget: str,
) -> tuple[Tensor, Tensor] | None:
    if "target_next_token_ids" not in counterfactual:
        return None
    selection = _selection_mask(counterfactual, budget)
    if selection is None:
        raise ValueError("target-greedy diagnostics require node-indexed labels")
    anchor_order = _stable_top(anchor_scores, EXPERTS)
    result = anchor_order[..., :CANDIDATES].clone()
    occurrence = torch.zeros(anchor_scores.shape[:2], dtype=torch.bool)
    occurrence[:, 0] = True
    token_ids = tree["token_ids"].long().cpu()
    parents = tree["parent"].long().cpu()
    depth = counterfactual["depth"].long().cpu()
    next_ids = counterfactual["target_next_token_ids"].long().cpu()
    next_valid = counterfactual["target_next_token_valid"].bool().cpu()
    for sample in range(anchor_scores.shape[0]):
        current = 0
        for horizon in range(1, HORIZONS):
            if not bool(next_valid[sample, current]):
                break
            wanted = next_ids[sample, current]
            children = torch.nonzero(
                selection[sample]
                & (parents[sample] == current)
                & (depth[sample] == horizon + 1)
                & (token_ids[sample] == wanted),
                as_tuple=False,
            ).flatten()
            if len(children) > 1:
                raise ValueError("target-greedy tree path has duplicate children")
            if not len(children):
                break
            current = int(children[0])
            occurrence[sample, horizon] = True
            exact = counterfactual["selected_ids"][sample, current].long().cpu()
            for layer in range(LAYERS):
                chosen = exact[layer].tolist()
                selected = set(chosen)
                chosen.extend(
                    int(expert)
                    for expert in anchor_order[sample, horizon, layer].tolist()
                    if int(expert) not in selected
                )
                result[sample, horizon, layer] = torch.tensor(chosen[:CANDIDATES])
    return result, occurrence


def _aggregate_metric(values: np.ndarray, request_ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, float]]:
    mean, per_request = _request_macro(values, request_ids)
    horizons = []
    for horizon in range(HORIZONS):
        horizon_mean, requests = _request_macro(values[:, horizon : horizon + 1], request_ids)
        horizons.append({
            "horizon": horizon + 1,
            "request_macro_mean": horizon_mean,
            "requests": len(requests),
        })
    h2_h4, _ = _request_macro(values[:, 1:], request_ids)
    return {
        "mean_h1_h4_request_macro": mean,
        "mean_h2_h4_request_macro": h2_h4,
        "horizons": horizons,
    }, per_request


def _gates(values: np.ndarray, request_ids: Sequence[str], greedy_match: np.ndarray) -> tuple[dict[str, Any], dict[str, float]]:
    mean, mean_requests = _request_macro(values[:, 1:], request_ids)
    h4, _ = _request_macro(values[:, 3:4], request_ids)
    mismatch_mask = np.zeros_like(values, dtype=bool)
    mismatch_mask[:, 3, :] = ~greedy_match[:, 3, None]
    mismatch, mismatch_requests = _request_macro(values, request_ids, mismatch_mask)
    results = {
        "mean_h2_h4": {"value": mean, "threshold": GATES["mean_h2_h4"], "passed": bool(mean >= GATES["mean_h2_h4"])},
        "h4": {"value": h4, "threshold": GATES["h4"], "passed": bool(h4 >= GATES["h4"])},
        "h4_prefix_mismatch": {"value": mismatch, "threshold": GATES["h4_prefix_mismatch"], "passed": bool(mismatch >= GATES["h4_prefix_mismatch"])},
    }
    return {"passed": all(value["passed"] for value in results.values()), "metrics": results}, mismatch_requests



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
        output = {}
        expanded = mask[:, :, None]
        for name, values in metrics.items():
            mean, requests = _request_macro(values, request_ids, expanded)
            output[name] = {
                "request_macro_mean": mean if requests else None,
                "requests": len(requests),
            }
        return output

    result["prefix"] = {}
    for horizon in range(HORIZONS):
        matched = np.zeros_like(greedy_match, dtype=bool)
        matched[:, horizon] = greedy_match[:, horizon]
        mismatched = np.zeros_like(greedy_match, dtype=bool)
        mismatched[:, horizon] = ~greedy_match[:, horizon]
        result["prefix"][f"H{horizon + 1}_matched"] = summarize(matched)
        result["prefix"][f"H{horizon + 1}_mismatched"] = summarize(mismatched)
    for depth in (2, 3, 4):
        mask = np.broadcast_to((divergence == depth)[:, None], greedy_match.shape)
        result[f"first_divergence_H{depth}"] = summarize(mask)
    matched = np.broadcast_to((divergence == 0)[:, None], greedy_match.shape)
    result["fully_matched_H4"] = summarize(matched)
    for name, bins in (("entropy", entropy_bin), ("margin", margin_bin)):
        result[name] = {}
        for bucket in range(4):
            mask = np.broadcast_to((bins == bucket)[:, None], greedy_match.shape)
            result[name][f"Q{bucket + 1}"] = summarize(mask)
    result["horizon"] = {}
    for horizon in range(HORIZONS):
        mask = np.zeros_like(greedy_match, dtype=bool)
        mask[:, horizon] = True
        result["horizon"][f"H{horizon + 1}"] = summarize(mask)
    result["layer"] = {}
    for layer in range(LAYERS):
        result["layer"][str(layer)] = {
            name: {
                "request_macro_mean": value,
                "requests": len(requests),
            }
            for name, values in metrics.items()
            for value, requests in [
                _request_macro(values[:, :, layer : layer + 1], request_ids)
            ]
        }
    result["block_phase_layer_mod_4"] = {}
    for phase in range(4):
        result["block_phase_layer_mod_4"][str(phase)] = {
            name: {
                "request_macro_mean": value,
                "requests": len(requests),
            }
            for name, values in metrics.items()
            for value, requests in [
                _request_macro(values[:, :, phase::4], request_ids)
            ]
        }
    return result


def _path_occurrence(
    occurrence_rows: Sequence[dict[str, dict[int, bool]]],
    request_ids: Sequence[str],
) -> dict[str, Any]:
    report = {}
    for control in CONTROL_NAMES:
        values = np.asarray(
            [
                [float(row[control][horizon]) for horizon in range(1, HORIZONS + 1)]
                for row in occurrence_rows
            ]
        )
        report[control] = {
            f"H{horizon + 1}": _request_macro(
                values[:, horizon : horizon + 1], request_ids
            )[0]
            for horizon in range(HORIZONS)
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
    parser.add_argument("--path-budget", choices=("4", "8", "16", "all"), default="4")
    parser.add_argument("--expected-inner-partition", default="diagnostic_probe")
    parser.add_argument("--expected-positions", type=int, default=2048)
    parser.add_argument("--expected-requests", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite B1.5 output {args.output}")
    if args.batch_size <= 0 or args.expected_positions <= 0 or args.expected_requests <= 0:
        raise ValueError("batch and expected inventory counts must be positive")
    for path in (args.capture, args.index_root, args.corpus_root, args.companion, args.static_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    controls, uncertainty, control_inventory = _load_controls(
        args.capture, expected_inner_partition=args.expected_inner_partition
    )
    companion_manifest = json.loads((args.companion / "manifest.json").read_text())
    schema = companion_manifest.get("schema")
    if schema == COUNTERFACTUAL_SCHEMA:
        if args.path_budget != "4":
            raise ValueError("the immutable v2 companion supports only budget 4")
        labels, companion_manifest = load_counterfactual_companion(
            args.companion, split="train", training=True
        )
        adapter = CounterfactualDatasetAdapter
    elif schema == NODE_COUNTERFACTUAL_SCHEMA:
        labels, companion_manifest = load_node_counterfactual_companion(
            args.companion, split="train", training=True
        )
        adapter = NodeCounterfactualDatasetAdapter
    else:
        raise ValueError(f"unsupported companion schema {schema!r}")
    base = HarpRTTDataset(
        args.index_root, "train", corpus_root=args.corpus_root, max_tree_nodes=32
    )
    dataset = adapter(base, labels, split="train", training=True)
    if (
        len(dataset) != args.expected_positions
        or len(labels) != args.expected_positions
        or len(controls) != args.expected_positions
        or len({key[0] for key in controls}) != args.expected_requests
    ):
        raise ValueError("B1.5 source/control/companion inventory disagrees with the declared partition")

    device = torch.device(args.device)
    bridge, anchor_provenance = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint,
        args.target_preprocessing,
        args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    bridge.to(device).eval()

    request_ids: list[str] = []
    source_positions: list[int] = []
    occurrence_rows = []
    entropy, margin = [], []
    metric_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    target_masses, source_masses = [], []
    factual_occurrence = []
    target_greedy_occurrence = []
    budget_rows = []

    with torch.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            items = [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
            batch = collate_harp_rtt(items)
            metadata = batch["metadata"]
            batch_requests = [str(value) for value in metadata["request_id"]]
            batch_positions = [int(value) for value in metadata["position"]]
            keys = list(zip(batch_requests, batch_positions, strict=True))
            if any(key not in controls or key not in uncertainty for key in keys):
                raise KeyError("dataset row is not aligned to a B1.5 control")
            request_ids.extend(batch_requests)
            source_positions.extend(batch_positions)
            occurrence_rows.extend(controls[key] for key in keys)
            entropy.extend(uncertainty[key]["root_entropy"] for key in keys)
            margin.extend(uncertainty[key]["root_margin"] for key in keys)

            device_batch = {
                "inputs": _move(batch["inputs"], device),
                "anchor_inputs": _move(batch["anchor_inputs"], device),
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                anchor_outputs = bridge(batch=device_batch)
            anchor_scores = anchor_outputs["future_router_scores"][:, :HORIZONS].float().cpu()
            target_ids = batch["targets"]["future_selected_ids"].long().cpu()
            factual_tokens = batch["targets"]["future_meta"][:, :, 3].long().cpu()
            counterfactual = batch["targets"]["counterfactual"]
            oracle = _oracle_batch(counterfactual, anchor_scores, budget=args.path_budget)
            target_masses.append(oracle["target_captured_mass"].numpy())
            source_masses.append(oracle["source_captured_mass"].numpy())

            anchor_top8 = _stable_top(anchor_scores, TOP_K)
            anchor_top64 = _stable_top(anchor_scores, CANDIDATES)
            metric_rows["anchor_recall_at_8"].append(_coverage(anchor_top8, target_ids).numpy())
            metric_rows["anchor_coverage_at_64"].append(_coverage(anchor_top64, target_ids).numpy())

            for prefix, full, branch in (
                ("target", oracle["target_full"], oracle["target_branch"]),
                ("mtp", oracle["source_full"], oracle["source_branch"]),
            ):
                top8 = _stable_top(full, TOP_K)
                top8[:, 0] = anchor_top8[:, 0]
                metric_rows[f"{prefix}_inclusion_recall_at_8"].append(_coverage(top8, target_ids).numpy())
                global_c64 = global_candidates(full, width=CANDIDATES)
                global_c64[:, 0] = anchor_top64[:, 0]
                metric_rows[f"{prefix}_global_coverage_at_64"].append(_coverage(global_c64, target_ids).numpy())
                for anchor_quota, branch_quota in QUOTA_POLICIES:
                    candidates = candidate_union(
                        anchor_scores,
                        branch,
                        anchor_quota=anchor_quota,
                        width=CANDIDATES,
                    )
                    candidates[:, 0] = anchor_top64[:, 0]
                    metric_rows[f"{prefix}_quota_{anchor_quota}_{branch_quota}_coverage_at_64"].append(
                        _coverage(candidates, target_ids).numpy()
                    )

            factual_c64, factual_present = _factual_candidates(
                counterfactual,
                batch["inputs"]["tree"],
                factual_tokens,
                anchor_scores,
                budget=args.path_budget,
            )
            metric_rows["captured_factual_coverage_at_64"].append(
                _coverage(factual_c64, target_ids).numpy()
            )
            factual_occurrence.append(factual_present.numpy())
            target_greedy = _target_greedy_candidates(
                counterfactual, batch["inputs"]["tree"], anchor_scores,
                budget=args.path_budget,
            )
            if target_greedy is not None:
                target_greedy_c64, target_greedy_present = target_greedy
                metric_rows["captured_target_greedy_coverage_at_64"].append(
                    _coverage(target_greedy_c64, target_ids).numpy()
                )
                target_greedy_occurrence.append(target_greedy_present.numpy())
            if "node_mask" in counterfactual:
                budget_rows.append(
                    torch.cat(
                        [
                            counterfactual["budget_realized"].long(),
                            counterfactual["budget_category_counts"].long().reshape(-1),
                            counterfactual["budget_node_masks"].sum(-1).long(),
                        ],
                        dim=-1,
                    ).numpy()
                )

    metrics = {name: np.concatenate(rows) for name, rows in metric_rows.items()}
    masses_target = np.concatenate(target_masses)
    masses_source = np.concatenate(source_masses)
    factual_path_occurrence = np.concatenate(factual_occurrence)
    target_greedy_path_occurrence = (
        None
        if not target_greedy_occurrence
        else np.concatenate(target_greedy_occurrence)
    )
    greedy_match = np.asarray(
        [[row["greedy"][h] for h in range(1, HORIZONS + 1)] for row in occurrence_rows],
        dtype=bool,
    )
    divergence, nesting = _first_divergence(greedy_match)
    entropy_array = np.asarray(entropy, dtype=np.float64)
    margin_array = np.asarray(margin, dtype=np.float64)
    entropy_thresholds = np.quantile(entropy_array, [0.25, 0.5, 0.75])
    margin_thresholds = np.quantile(margin_array, [0.25, 0.5, 0.75])
    entropy_bin = np.digitize(entropy_array, entropy_thresholds, right=True)
    margin_bin = np.digitize(margin_array, margin_thresholds, right=True)

    aggregate, per_request = {}, {}
    for name, values in metrics.items():
        aggregate[name], per_request[name] = _aggregate_metric(values, request_ids)

    policy_order = [
        "target_quota_48_16_coverage_at_64",
        "target_quota_40_24_coverage_at_64",
        "target_quota_32_32_coverage_at_64",
        "target_global_coverage_at_64",
    ]
    policy_gates = {}
    mismatch_requests = {}
    for name in policy_order:
        policy_gates[name], mismatch_requests[name] = _gates(
            metrics[name], request_ids, greedy_match
        )
    promotion_policy = next((name for name in policy_order if policy_gates[name]["passed"]), None)
    passed = promotion_policy is not None

    anchor_h1 = aggregate["anchor_coverage_at_64"]["horizons"][0]["request_macro_mean"]
    selected_name = promotion_policy or "target_global_coverage_at_64"
    anchor_h2_h4_requests = _request_macro(metrics["anchor_coverage_at_64"][:, 1:], request_ids)[1]
    selected_h2_h4_requests = _request_macro(metrics[selected_name][:, 1:], request_ids)[1]
    h4_mismatch_mask = np.zeros_like(metrics[selected_name], dtype=bool)
    h4_mismatch_mask[:, 3, :] = ~greedy_match[:, 3, None]
    anchor_mismatch_requests = _request_macro(
        metrics["anchor_coverage_at_64"], request_ids, h4_mismatch_mask
    )[1]

    report = {
        "schema": B15_ORACLE_SCHEMA,
        "passed": passed,
        "decision": "information_gate_passed_stop_before_B2" if passed else "stop_before_B2_information_gate_failed",
        "stage": "B1.1" if schema == COUNTERFACTUAL_SCHEMA else "B1.5",
        "split": f"outer_train_inner_{args.expected_inner_partition}",
        "path_budget": args.path_budget,
        "source_positions": len(request_ids),
        "requests": len(set(request_ids)),
        "layers": LAYERS,
        "oracle_contract": {
            "authoritative_route_labels": "native_BF16_selected_ids",
            "inclusion_mass": "sum_unique_nodes(path_probability * selected_set_indicator)",
            "other": "residual_probability_times_exact_k_anchor_marginal",
            "candidate_union_algorithm_sha256": b15_candidate_union_sha256(),
            "target_and_causal_mtp_priors_reported_separately": True,
            "target_greedy_oracle_is_diagnostic_not_promotion": True,
            "quota_policies": [list(value) for value in QUOTA_POLICIES],
            "global_policy": "top64_of_full_inclusion_mass",
            "h1_excluded_from_branch_promotion": True,
        },
        "aggregate_metrics": aggregate,
        "policy_gates": policy_gates,
        "promotion_policy": promotion_policy,
        "all_branch_gates_passed": passed,
        "h1_independent_gate": {
            "current_source": "frozen_anchor_only",
            "value": anchor_h1,
            "threshold": H1_C64_GATE,
            "passed": bool(anchor_h1 >= H1_C64_GATE),
            "blocking_for_b15": False,
            "optimizer_started": False,
        },
        "captured_factual_path_occurrence": {
            f"H{h + 1}": float(factual_path_occurrence[:, h].mean())
            for h in range(HORIZONS)
        },
        "captured_target_greedy_path_occurrence": (
            None
            if target_greedy_path_occurrence is None
            else {
                f"H{h + 1}": float(target_greedy_path_occurrence[:, h].mean())
                for h in range(HORIZONS)
            }
        ),
        "captured_target_path_mass": {
            f"H{h + 1}": {
                "mean": float(masses_target[:, h].mean()),
                "minimum": float(masses_target[:, h].min()),
                "maximum": float(masses_target[:, h].max()),
            }
            for h in range(1, HORIZONS)
        },
        "captured_mtp_path_mass": {
            f"H{h + 1}": {
                "mean": float(masses_source[:, h].mean()),
                "minimum": float(masses_source[:, h].min()),
                "maximum": float(masses_source[:, h].max()),
            }
            for h in range(1, HORIZONS)
        },
        "prefix_match_nesting": nesting,
        "path_occurrence": _path_occurrence(occurrence_rows, request_ids),
        "stratified_metrics": _stratified_metrics(
            metrics, request_ids, greedy_match, divergence, entropy_bin, margin_bin
        ),
        "first_divergence": {
            "H2": int((divergence == 2).sum()),
            "H3": int((divergence == 3).sum()),
            "H4": int((divergence == 4).sum()),
            "fully_matched": int((divergence == 0).sum()),
        },
        "uncertainty": {
            "root_entropy_quartiles": [float(value) for value in entropy_thresholds],
            "root_margin_quartiles": [float(value) for value in margin_thresholds],
        },
        "paired_complete_request_bootstrap": {
            "selected_vs_anchor_h2_h4": _bootstrap_delta(
                selected_h2_h4_requests, anchor_h2_h4_requests
            ),
            "selected_vs_anchor_h4_mismatch": _bootstrap_delta(
                mismatch_requests[selected_name], anchor_mismatch_requests
            ),
        },
        "budget_diagnostics": None if not budget_rows else {
            "columns": [
                "realized_4", "realized_8", "realized_16",
                *[f"budget_{budget}_category_{category}" for budget in (4, 8, 16) for category in ("greedy", "H2", "H3", "H4")],
                "unique_nodes_4", "unique_nodes_8", "unique_nodes_16",
            ],
            "mean": np.concatenate(budget_rows).mean(0).tolist(),
            "minimum": np.concatenate(budget_rows).min(0).tolist(),
            "maximum": np.concatenate(budget_rows).max(0).tolist(),
        },
        "control_inventory": control_inventory,
        "bindings": {
            "capture": str(args.capture.resolve()),
            "capture_manifest_sha256": _sha256(args.capture / "run_manifest.json"),
            "companion": str(args.companion.resolve()),
            "companion_schema": schema,
            "companion_manifest_sha256": _sha256(args.companion / "manifest.json"),
            "companion_audit_sha256": _sha256(args.companion / "COUNTERFACTUAL_AUDIT.json"),
            "index": str(args.index_root.resolve()),
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

    request_metric_extras = {}
    if target_greedy_path_occurrence is not None:
        request_metric_extras["target_greedy_path_occurrence"] = (
            target_greedy_path_occurrence
        )
    args.output.mkdir(parents=True)
    (args.output / "B15_ORACLE_GATE_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        args.output / "B15_REQUEST_METRICS.npz",
        request_ids=np.asarray(request_ids),
        source_positions=np.asarray(source_positions),
        greedy_match=greedy_match,
        first_divergence=divergence,
        captured_target_path_mass=masses_target,
        captured_mtp_path_mass=masses_source,
        factual_path_occurrence=factual_path_occurrence,
        **request_metric_extras,
        **metrics,
    )
    hashes = {
        path.name: _sha256(path)
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    (args.output / "SHA256SUMS.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
