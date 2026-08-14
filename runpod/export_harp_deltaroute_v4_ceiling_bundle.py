#!/usr/bin/env python3
"""Export the no-recapture DeltaRoute Stage-0 bundle from frozen v3."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.anchor import LegacyHARPAnchorBridge  # noqa: E402
from harp_rtt.b31 import anchor_spine_prefix_matches  # noqa: E402
from harp_rtt.dataset import HarpRTTDataset  # noqa: E402
from harp_rtt.delta import HARPDeltaConfig, HARPDeltaTeacher  # noqa: E402
from harp_rtt.model.heads import exact_projected_marginals  # noqa: E402
from harp_rtt.node_counterfactual import (  # noqa: E402
    NodeCounterfactualDatasetAdapter,
    load_node_counterfactual_companion,
)
from harp_rtt.route_ceiling import true_expert_support_audit  # noqa: E402
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.train import runtime_static_artifacts  # noqa: E402
from harp_rtt.training import seed_everything, sha256_file  # noqa: E402
from runpod.evaluate_harp_deltaroute_v4_ceiling import BUNDLE_SCHEMA  # noqa: E402
from runpod.train_harp_delta_v3 import (  # noqa: E402
    COUNTERFACTUAL_BUDGET_INDEX,
    SCHEMA as PARENT_SCHEMA,
    forward_batch,
    loader,
)


EXPECTED_ROWS = 2_048
EXPECTED_REQUESTS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--probe-index", type=Path, required=True)
    parser.add_argument("--probe-corpus", type=Path, required=True)
    parser.add_argument("--probe-companion", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--capture-source-commit", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_parent(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != PARENT_SCHEMA:
        raise ValueError("DeltaRoute requires a v3 stage checkpoint")
    if value.get("stage") != "semantic" or value.get("counterfactual_budget") != "16":
        raise ValueError("DeltaRoute parent must be the budget-16 semantic checkpoint")
    for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
        if value.get(key) is not False:
            raise PermissionError(f"DeltaRoute parent violates {key}")
    return value


def _append(storage: dict[str, list[Tensor]], name: str, value: Tensor) -> None:
    storage.setdefault(name, []).append(value.detach().cpu())


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite v4 ceiling bundle {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    seed_everything(42, deterministic=True)
    checkpoint = _load_parent(args.parent_checkpoint)
    static = load_static_target_artifacts(args.static_dir, device="cpu")
    config = HARPDeltaConfig(**checkpoint["config"])
    if (
        config.experts != static.geometry.experts
        or config.layers != static.geometry.layers
        or config.router_rank != static.geometry.maximum_rank
    ):
        raise ValueError("v3 parent and frozen router geometry differ")
    model = HARPDeltaTeacher(
        config, static.geometry.expert_keys, static.geometry.centered_bias,
        raw_width=static.geometry.hidden_width,
        target_control_width=static.geometry.maximum_rank,
        metadata_width=8,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False).eval()
    anchor, anchor_provenance = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint, args.target_preprocessing, args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    anchor = anchor.to(device).requires_grad_(False).eval()
    labels, companion_manifest = load_node_counterfactual_companion(
        args.probe_companion, split="train", training=True,
        expected_bindings={"source_commit": args.capture_source_commit},
    )
    base = HarpRTTDataset(
        args.probe_index, "train", corpus_root=args.probe_corpus,
        max_tree_nodes=config.max_tree_nodes,
    )
    probe = NodeCounterfactualDatasetAdapter(base, labels, split="train", training=True)
    requests = [
        str(base.segments[row.segment].sequences[row.sequence]["request_id"])
        for row in base.records
    ]
    counts = Counter(requests)
    if (
        len(probe) != EXPECTED_ROWS or len(labels) != EXPECTED_ROWS
        or len(counts) != EXPECTED_REQUESTS or set(counts.values()) != {16}
    ):
        raise ValueError("v4 ceiling requires the frozen 128x16 outer-train probe")
    runtime_static = runtime_static_artifacts(static, device)
    token_embedding = static.token_embedding
    if token_embedding is None:
        raise RuntimeError("DeltaRoute requires frozen token embeddings")
    token_embedding = token_embedding.to(device)
    input_basis = static.geometry.input_basis.to(device)
    rank_mask = static.geometry.rank_mask.to(device)
    expert_keys = static.geometry.expert_keys.to(device)
    centered_bias = static.geometry.centered_bias.to(device)

    dense: dict[str, list[Tensor]] = {}
    request_order: list[str] = []
    learned_posterior: dict[str, list[Tensor]] = {"captured": [], "other": []}
    target_posterior: dict[str, list[Tensor]] = {"captured": [], "other": []}
    mtp_posterior: dict[str, list[Tensor]] = {"captured": [], "other": []}
    support_fields: dict[str, list[Tensor]] = {}
    budget_index = COUNTERFACTUAL_BUDGET_INDEX["16"]
    for host in loader(
        probe, batch=args.batch_size, shuffle=False, seed=42,
        workers=args.num_workers, device=device,
    ):
        output, targets, counterfactual, anchor_scores = forward_batch(
            model=model, anchor=anchor, host=host, runtime_static=runtime_static,
            token_embedding=token_embedding, input_basis=input_basis,
            rank_mask=rank_mask, device=device, semantic_only=True,
            counterfactual_budget_index=budget_index,
        )
        anchor_marginals, _, node_marginals, _ = model.core.semantic_marginals(
            output, anchor_scores
        )
        selection = counterfactual["semantic_selection_mask"].bool()
        label_valid = counterfactual["valid"].bool().any(-1)
        horizon_mask = host["inputs"]["tree"]["horizon_mask"].to(device).bool()
        branch_mask = horizon_mask & selection[:, None] & label_valid[:, None]
        learned = output.factual_path_posterior.float()
        target = counterfactual["target_path_distribution"].float().clone()
        target[:, 0] = learned[:, 0]
        path = host["inputs"]["tree"]["path_log_probabilities"].to(device).float()
        mtp_captured = path.exp()[:, None].expand_as(branch_mask) * branch_mask.float()
        mtp_other = (1.0 - mtp_captured.sum(-1)).clamp(0.0, 1.0)
        native = counterfactual["selected_ids"].long()
        geometry = torch.einsum(
            "bhlnr,ler->bhlne", output.node_queries.float(), expert_keys.float()
        ) + centered_bias[None, None, :, None]
        support = true_expert_support_audit(
            target_ids=targets["future_selected_ids"].long(),
            anchor_scores=anchor_scores,
            node_scores=output.node_scores.float(),
            node_marginals=node_marginals.float(),
            learned_probabilities=learned[..., :-1],
            mtp_probabilities=mtp_captured,
            branch_mask=branch_mask,
            factual_branch_indices=targets["factual_branch_index"].long(),
            geometry_scores=geometry,
        )

        _append(dense, "anchor_scores", anchor_scores.to(torch.bfloat16))
        _append(dense, "anchor_marginals", anchor_marginals.to(torch.bfloat16))
        _append(dense, "target_ids", targets["future_selected_ids"].long())
        _append(dense, "future_valid", targets["future_available"].bool())
        _append(dense, "node_native_ids", native.to(torch.int16))
        _append(dense, "branch_mask", branch_mask)
        _append(dense, "factual_branch_indices", targets["factual_branch_index"].long())
        for name in (
            "anchor_rank", "best_branch_rank", "learned_mixture_rank",
            "mtp_mixture_rank", "factual_branch_rank", "geometry_best_rank",
            "supporting_branches",
        ):
            _append(support_fields, name, getattr(support, name).to(torch.int16))
        learned_posterior["captured"].append(learned[..., :-1].cpu())
        learned_posterior["other"].append(learned[..., -1].cpu())
        target_posterior["captured"].append(target[..., :-1].cpu())
        target_posterior["other"].append(target[..., -1].cpu())
        mtp_posterior["captured"].append(mtp_captured.cpu())
        mtp_posterior["other"].append(mtp_other.cpu())
        greedy_match = anchor_spine_prefix_matches(
            host["anchor_inputs"]["mtp_spine"]["exact_prefix_hashes"],
            host["targets"]["future_prefix_hashes"],
        )
        _append(dense, "prefix_mismatch", ~greedy_match)
        metadata = host["metadata"]
        request_order.extend(str(item) for item in metadata["request_id"])

    bundle: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "outer_split": "train",
            "partition": "diagnostic_development_128x16",
            "rows": EXPECTED_ROWS,
            "requests": EXPECTED_REQUESTS,
            "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
            "parent_source_commit": checkpoint["source_commit"],
            "export_source_commit": args.source_commit,
            "companion_manifest_sha256": sha256_file(args.probe_companion / "manifest.json"),
            "companion_source_commit": companion_manifest.get("source_commit"),
            "anchor_provenance": anchor_provenance,
            "budget": 16,
            "runtime_tree_nodes": 32,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        },
        "request_ids": request_order,
        **{name: torch.cat(values) for name, values in dense.items()},
        "true_expert_support": {
            name: torch.cat(values) for name, values in support_fields.items()
        },
        "posteriors": {
            name: {key: torch.cat(values) for key, values in condition.items()}
            for name, condition in (
                ("learned", learned_posterior),
                ("target", target_posterior),
                ("mtp", mtp_posterior),
            )
        },
        "exact_k": config.exact_k,
        "candidate_width": config.candidate_width,
        "training_started": False,
        "optimizer_constructed": False,
    }
    if request_order != requests:
        raise RuntimeError("v4 ceiling DataLoader changed immutable request ordering")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        torch.save(bundle, handle); handle.flush(); os.fsync(handle.fileno())
    result = {
        "schema": "harp_deltaroute_v4_ceiling_bundle_export_v1",
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "bytes": args.output.stat().st_size,
        "rows": EXPECTED_ROWS,
        "requests": EXPECTED_REQUESTS,
        "training_started": False,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
