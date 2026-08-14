#!/usr/bin/env python3
"""Optimizer-free Stage-0 ceilings for HARP-DeltaRoute v4."""

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
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.b31 import quota_candidate_union, selected_ids_inclusion_mass  # noqa: E402
from harp_rtt.exact_k import stable_topk  # noqa: E402
from harp_rtt.route_ceiling import (  # noqa: E402
    factual_branch_topk,
    posterior_native_topk,
    slot_recall_at_k,
    swap_cap_oracle_recall,
    true_expert_support_audit,
)
from harp_rtt.training import sha256_file  # noqa: E402


BUNDLE_SCHEMA = "harp_deltaroute_v4_ceiling_bundle_v1"
REPORT_SCHEMA = "harp_deltaroute_v4_ceiling_report_v1"
RUN_SCHEMA = "harp_deltaroute_v4_run_manifest_v1"


def _tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"v4 ceiling field {name} must be a tensor")
    return value.cpu()


def load_bundle(path: Path, *, expected_parent_sha256: str) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("DeltaRoute ceiling bundle schema mismatch")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("DeltaRoute ceiling bundle lacks provenance")
    if provenance.get("outer_split") != "train":
        raise PermissionError("DeltaRoute ceilings are restricted to outer-train")
    if provenance.get("parent_checkpoint_sha256") != expected_parent_sha256:
        raise ValueError("DeltaRoute ceiling parent checkpoint hash mismatch")
    for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if provenance.get(key) is not False:
            raise PermissionError(f"DeltaRoute ceiling bundle violates {key}")
    request_ids = value.get("request_ids")
    if not isinstance(request_ids, list) or not request_ids:
        raise ValueError("DeltaRoute ceiling bundle lacks request IDs")
    if len(set(map(str, request_ids))) < 2:
        raise ValueError("DeltaRoute ceiling requires multiple request groups")
    return value


def _posterior(value: Mapping[str, Any], name: str) -> tuple[Tensor, Tensor]:
    posteriors = value.get("posteriors")
    if not isinstance(posteriors, Mapping) or not isinstance(posteriors.get(name), Mapping):
        raise ValueError(f"DeltaRoute ceiling lacks {name} posterior")
    condition = posteriors[name]
    assert isinstance(condition, Mapping)
    return _tensor(condition.get("captured"), f"posteriors.{name}.captured"), _tensor(
        condition.get("other"), f"posteriors.{name}.other"
    )


def _request_macro(
    values: Tensor,
    valid: Tensor,
    request_ids: list[str],
) -> dict[str, Any]:
    if values.ndim != 3 or valid.shape != values.shape:
        raise ValueError("request metrics must be [B,H,L]")
    if len(request_ids) != values.shape[0]:
        raise ValueError("request IDs disagree with metric rows")
    grouped: dict[str, list[int]] = defaultdict(list)
    for row, request in enumerate(request_ids):
        grouped[str(request)].append(row)
    horizons: list[float] = []
    rows: list[dict[str, Any]] = []
    for horizon in range(values.shape[1]):
        per_request: list[float] = []
        for request in sorted(grouped):
            selected = grouped[request]
            active = valid[selected, horizon]
            if not bool(active.any()):
                raise ValueError(f"request {request!r} lacks H{horizon + 1} rows")
            score = float(values[selected, horizon].masked_select(active).mean())
            per_request.append(score)
            rows.append({
                "request_id": request,
                "horizon": horizon + 1,
                "slot_recall_at_8": score,
            })
        horizons.append(sum(per_request) / len(per_request))
    return {
        "request_macro": sum(horizons) / len(horizons),
        "request_macro_h2_h4": sum(horizons[1:]) / 3.0,
        **{f"request_macro_h{index + 1}": value for index, value in enumerate(horizons)},
        "request_metrics": rows,
    }


def _condition(
    predicted: Tensor,
    target: Tensor,
    valid: Tensor,
    request_ids: list[str],
) -> dict[str, Any]:
    return _request_macro(slot_recall_at_k(predicted, target), valid, request_ids)


def evaluate_bundle(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor_scores = _tensor(value.get("anchor_scores"), "anchor_scores").float()
    anchor_marginals = _tensor(value.get("anchor_marginals"), "anchor_marginals").float()
    target_ids = _tensor(value.get("target_ids"), "target_ids").long()
    valid = _tensor(value.get("future_valid"), "future_valid").bool()
    native_ids = _tensor(value.get("node_native_ids"), "node_native_ids").long()
    branch_mask = _tensor(value.get("branch_mask"), "branch_mask").bool()
    factual = _tensor(value.get("factual_branch_indices"), "factual_branch_indices").long()
    request_ids = [str(item) for item in value["request_ids"]]
    batch, horizons, layers, experts = anchor_scores.shape
    exact_k = target_ids.shape[-1]
    if anchor_marginals.shape != anchor_scores.shape:
        raise ValueError("anchor score/marginal geometry differs")
    if target_ids.shape != (batch, horizons, layers, exact_k):
        raise ValueError("target selected-set geometry is invalid")
    if valid.shape == (batch, horizons):
        valid = valid[..., None].expand(batch, horizons, layers)
    if valid.shape != (batch, horizons, layers):
        raise ValueError("future validity geometry is invalid")
    if native_ids.shape[0] != batch or native_ids.shape[2:] != (layers, exact_k):
        raise ValueError("native node-route geometry is invalid")
    nodes = native_ids.shape[1]
    if branch_mask.shape != (batch, horizons, nodes) or factual.shape != (batch, horizons):
        raise ValueError("branch mask/factual index geometry is invalid")
    anchor_ids = stable_topk(anchor_scores, exact_k)
    factual_for_ceiling = factual.clone()
    factual_for_ceiling[:, 0] = nodes
    factual_ids = factual_branch_topk(native_ids, factual_for_ceiling, anchor_ids)

    conditions: dict[str, Any] = {
        "anchor": _condition(anchor_ids, target_ids, valid, request_ids),
        "native_factual_branch_with_anchor_fallback": _condition(
            factual_ids, target_ids, valid, request_ids
        ),
    }
    posterior_masses: dict[str, Tensor] = {}
    posterior_ids: dict[str, Tensor] = {}
    for name in ("target", "learned", "mtp"):
        captured, other = _posterior(value, name)
        ids, mass = posterior_native_topk(
            native_ids, captured, other, branch_mask, anchor_marginals,
            exact_k=exact_k,
        )
        posterior_ids[name] = ids
        posterior_masses[name] = mass
        conditions[f"native_{name}_posterior"] = _condition(
            ids, target_ids, valid, request_ids
        )

    learned, _ = _posterior(value, "learned")
    captured_learned = selected_ids_inclusion_mass(
        native_ids[:, None].expand(batch, horizons, nodes, layers, exact_k).permute(
            0, 1, 3, 2, 4
        ),
        learned, branch_mask, experts=experts, normalize=False,
    )
    candidate = quota_candidate_union(
        anchor_scores, captured_learned, anchor_quota=32,
        width=int(value.get("candidate_width", 64)),
    ).expert_ids
    candidate_metrics = _condition(candidate, target_ids, valid, request_ids)
    swap = swap_cap_oracle_recall(
        anchor_ids, candidate, target_ids, (1, 2, 4, 6, 8)
    )
    swap_metrics = {
        str(cap): _request_macro(values, valid, request_ids)
        for cap, values in swap.items()
    }

    mismatch = value.get("prefix_mismatch")
    mismatch_h4 = None
    if isinstance(mismatch, Tensor):
        mismatch = mismatch.bool()
        if mismatch.shape != (batch, horizons):
            raise ValueError("prefix mismatch strata must be [B,H]")
        active = valid[:, 3] & mismatch[:, 3, None]
        recall = slot_recall_at_k(candidate, target_ids)[:, 3]
        mismatch_h4 = float(recall.masked_select(active).mean()) if bool(active.any()) else None

    learned_scores = value.get("learned_node_scores")
    learned_marginals = value.get("learned_node_marginals")
    compact_support = value.get("true_expert_support")
    soft_sidecar: dict[str, Any] = {
        "schema": "harp_deltaroute_v4_missing_true_support_v1",
        "provenance": dict(value["provenance"]),
        "available": False,
    }
    audit_fields = (
        "anchor_rank", "best_branch_rank", "learned_mixture_rank",
        "mtp_mixture_rank", "factual_branch_rank",
        "geometry_best_rank", "supporting_branches",
    )
    compact_values: dict[str, Tensor] | None = None
    if isinstance(compact_support, Mapping):
        compact_values = {
            field: _tensor(compact_support.get(field), f"true_expert_support.{field}")
            for field in audit_fields
        }
        if any(tensor.shape != target_ids.shape for tensor in compact_values.values()):
            raise ValueError("compact true-expert support geometry is invalid")
    if compact_values is not None or (
        isinstance(learned_scores, Tensor) and isinstance(learned_marginals, Tensor)
    ):
        mtp, _ = _posterior(value, "mtp")
        audit = None if compact_values is not None else true_expert_support_audit(
            target_ids=target_ids, anchor_scores=anchor_scores,
            node_scores=learned_scores.float(),
            node_marginals=learned_marginals.float(),
            learned_probabilities=learned, mtp_probabilities=mtp,
            branch_mask=branch_mask, factual_branch_indices=factual,
            geometry_scores=(None if not isinstance(value.get("geometry_node_scores"), Tensor)
                             else value["geometry_node_scores"].float()),
        )
        contained = (
            target_ids[..., None] == candidate[..., None, :]
        ).any(-1)
        missing = ~contained & valid[..., None]
        indices = missing.nonzero(as_tuple=False)
        soft_sidecar = {
            "schema": "harp_deltaroute_v4_missing_true_support_v1",
            "provenance": dict(value["provenance"]),
            "available": True,
            "rows": int(indices.shape[0]),
            "indices_b_h_l_slot": indices,
            "target_expert_ids": target_ids[missing],
            **{field: (
                compact_values[field][missing]
                if compact_values is not None else getattr(audit, field)[missing]
            ) for field in audit_fields},
        }

    factual_gate = conditions["native_factual_branch_with_anchor_fallback"][
        "request_macro_h2_h4"
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": dict(value["provenance"]),
        "conditions": conditions,
        "native_learned_posterior_candidate_c64": candidate_metrics,
        "native_learned_posterior_candidate_h4_mismatch": mismatch_h4,
        "swap_cap_oracle": swap_metrics,
        "native_factual_h2_h4_gate": {
            "threshold": 0.89,
            "value": factual_gate,
            "passed": factual_gate >= 0.89,
        },
        "soft_support_sidecar_available": bool(soft_sidecar["available"]),
        "training_started": False,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    return report, soft_sidecar


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite v4 ceiling run {args.output}")
    value = load_bundle(
        args.bundle, expected_parent_sha256=args.parent_checkpoint_sha256
    )
    args.output.mkdir(parents=True)
    manifest = {
        "schema": RUN_SCHEMA,
        "stage": "ceiling",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "bundle_sha256": sha256_file(args.bundle),
        "parent_checkpoint_sha256": args.parent_checkpoint_sha256,
        "training_started": False,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    _write_json_exclusive(args.output / "run_manifest.json", manifest)
    report, sidecar = evaluate_bundle(value)
    report["run_manifest_sha256"] = sha256_file(args.output / "run_manifest.json")
    _write_json_exclusive(args.output / "STAGE_RESULT.json", report)
    with (args.output / "SOFT_SUPPORT_MISSING.pt").open("xb") as handle:
        torch.save(sidecar, handle); handle.flush(); os.fsync(handle.fileno())
    files = ("run_manifest.json", "STAGE_RESULT.json", "SOFT_SUPPORT_MISSING.pt")
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for name in files:
            handle.write(f"{sha256_file(args.output / name)}  {name}\n")
        handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
