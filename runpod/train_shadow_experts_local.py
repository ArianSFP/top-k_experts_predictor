#!/usr/bin/env python3
"""Train one layer-local HARP-ShadowRoute S0, S1, or S2 expert shard.

This stage reuses already-indexed outer-train factual states.  It loads only
one native target expert layer, never constructs the 67 GiB target, and never
opens formal validation, calibration, or sealed-test rows.  Its result is a
component checkpoint; only the later exact-backbone closed-loop evaluation can
authorize ShadowRoute.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.dataset import HarpRTTDataset  # noqa: E402
from harp_rtt.exact_k import exact_set_nll, stable_topk  # noqa: E402
from harp_rtt.losses import boundary_loss_per_endpoint  # noqa: E402
from harp_rtt.shadow_checkpoint import (  # noqa: E402
    IndexedCheckpoint,
    load_target_layer_experts,
    sha256_file,
)
from harp_rtt.shadow_expert import (  # noqa: E402
    BasisDraftConfig,
    IndexedShadowExperts,
    PackedInt4ResidentExperts,
    RouteConditionedBasisExperts,
    RouterVisibleTailControl,
    ShadowExpertConfig,
    SharedResidualExperts,
    SwiGLUDraftExpert,
    target_neuron_importance,
    target_selected_expert_outputs,
)
from harp_rtt.shadow_training import selected_expert_distillation_loss  # noqa: E402
from harp_rtt.training import move_to_device, seed_everything  # noqa: E402
from runpod.train_harp_delta_v3 import (  # noqa: E402
    DATA_PROFILES,
    RequestSubset,
    _manifest as validate_partition,
    _record_request_id,
    _request_counts,
    _reuse_split_manifest as validate_reuse_split,
    loader,
)


SCHEMA = "harp_shadowroute_local_expert_layer_v1"
RESULT_SCHEMA = "harp_shadowroute_local_expert_layer_result_v1"
MODES = (
    "s0_exact_top1_plus_draft",
    "s1_shared",
    "s1_shared_width512",
    "s2_indexed",
    "basisdraft_all8",
    "resident_int4_shared",
    "resident_int4_tail_control",
)
EFFECTIVE_BATCH = 32
MICROBATCH_CHOICES = (32, 16, 8, 4, 2, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "tune", "development"):
        parser.add_argument(f"--{split}-index", type=Path, required=True)
        parser.add_argument(f"--{split}-corpus", type=Path, required=True)
        parser.add_argument(f"--{split}-companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--scale-train-index", type=Path)
    parser.add_argument("--scale-train-corpus", type=Path)
    parser.add_argument("--outer-split-manifest", type=Path)
    parser.add_argument("--initializer-checkpoint", type=Path)
    parser.add_argument("--next-router-agreement", action="store_true")
    parser.add_argument("--router-agreement-weight", type=float, default=0.1)
    parser.add_argument("--aggregate-loss-weight", type=float, default=1.0)
    parser.add_argument("--individual-loss-weight", type=float, default=1.0)
    parser.add_argument("--cosine-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--basis-coefficients-only",
        action="store_true",
        help=(
            "freeze the shared nonlinear basis while training route-specific "
            "coefficients and expert residual adapters"
        ),
    )
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--layer", type=int, choices=range(40), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--microbatch-size", type=int, choices=(0, *MICROBATCH_CHOICES), default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--minimum-expert-count", type=int, default=128)
    parser.add_argument(
        "--shadow-width", type=int, choices=(16, 32, 64, 96, 128), default=16
    )
    parser.add_argument("--indexed-active-slots", type=int, choices=(4, 8), default=8)
    parser.add_argument("--basis-count", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--basis-width", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument(
        "--basis-expert-residual-width", type=int,
        choices=(2, 4, 8, 16), default=2,
    )
    parser.add_argument(
        "--resident-count", type=int,
        choices=(32, 48, 64, 80, 92, 96), default=80,
    )
    parser.add_argument("--resident-eval-only", action="store_true")
    parser.add_argument("--resident-export-only", action="store_true")
    parser.add_argument("--resident-allocation-plan", type=Path)
    parser.add_argument(
        "--resident-v2-phase",
        choices=("tail", "control", "joint"),
        default="joint",
    )
    parser.add_argument(
        "--resident-control-rank", type=int, choices=(32, 64, 128), default=128
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


class ShadowFactualStateDataset(Dataset[dict[str, Any]]):
    """Minimal label-only factual reader for local shadow-expert training."""

    def __init__(
        self, base: Dataset[Any], *, layer: int, next_router_agreement: bool = False
    ) -> None:
        if layer not in range(40):
            raise ValueError("ShadowRoute factual reader layer is out of range")
        self.layer = int(layer)
        self.next_router_agreement = bool(next_router_agreement)
        if self.next_router_agreement and self.layer == 39:
            raise ValueError("layer 39 has no within-token next-router target")
        if isinstance(base, RequestSubset):
            self.source = base.base
            self.indices = tuple(int(index) for index in base.indices)
        elif isinstance(base, HarpRTTDataset):
            self.source = base
            self.indices = tuple(range(len(base)))
        else:
            raise TypeError("ShadowRoute factual reader requires a frozen HARP dataset")
        if not isinstance(self.source, HarpRTTDataset):
            raise TypeError("ShadowRoute factual reader cannot locate HarpRTTDataset")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.source.records[self.indices[index]]
        segment = self.source.segments[record.segment]
        roles = (
            "normalized_target_router_input_a",
            "selected_expert_ids",
            "selected_execution_weights",
            "routed_expert_output_delta_r",
        )
        future = []
        for row in record.future_rows:
            values = {
                role: segment.read(
                    "target", [int(row) + self.layer], role
                )[0]
                for role in roles
            }
            if self.next_router_agreement:
                for role in (
                    "post_attention_residual_u",
                    "post_moe_residual_xplus",
                    "shared_expert_output_delta_s",
                ):
                    values[role] = segment.read(
                        "target", [int(row) + self.layer], role
                    )[0]
                for role in (
                    "post_attention_residual_u",
                    "raw_target_router_logits",
                    "selected_expert_ids",
                ):
                    values[f"next_{role}"] = segment.read(
                        "target", [int(row) + self.layer + 1], role
                    )[0]
            future.append(values)
        states = {
            "routed_expert_output_delta_r": torch.stack(
                [row["routed_expert_output_delta_r"] for row in future]
            )
        }
        if self.next_router_agreement:
            for role in (
                "post_attention_residual_u",
                "post_moe_residual_xplus",
                "shared_expert_output_delta_s",
                "next_post_attention_residual_u",
                "next_raw_target_router_logits",
                "next_selected_expert_ids",
            ):
                states[role] = torch.stack([row[role] for row in future])
        return {
            "metadata": {
                "request_id": str(segment.sequences[record.sequence]["request_id"]),
                "position": int(record.position),
            },
            # Serving inputs are deliberately absent from this layer-local
            # component probe.  Every tensor below is a train-only teacher.
            "inputs": {},
            "targets": {
                "future_router_inputs": torch.stack(
                    [row["normalized_target_router_input_a"] for row in future]
                ),
                "future_selected_ids": torch.stack(
                    [row["selected_expert_ids"].to(torch.int64) for row in future]
                ),
                "future_execution_weights": torch.stack(
                    [row["selected_execution_weights"].float() for row in future]
                ),
                "future_available": torch.ones(4, dtype=torch.bool),
                "future_states": states,
            },
        }


class ShadowGeneratedTokenDataset(Dataset[dict[str, Any]]):
    """Deduplicated outer-train target tokens for no-recapture distillation."""

    def __init__(
        self,
        base: HarpRTTDataset,
        *,
        layer: int,
        allowed_request_ids: set[str],
        next_router_agreement: bool = False,
    ) -> None:
        if layer not in range(40):
            raise ValueError("ShadowRoute generated-token reader layer is out of range")
        self.layer = int(layer)
        self.next_router_agreement = bool(next_router_agreement)
        if self.next_router_agreement and self.layer == 39:
            raise ValueError("layer 39 has no within-token next-router target")
        self.source = base
        self.rows: list[tuple[int, int, int, int]] = []
        self.requests: set[str] = set()
        for segment_index, segment in enumerate(base.segments):
            for (sequence, position), row in segment.target_by_position.items():
                metadata = segment.sequences[sequence]
                request = str(metadata["request_id"])
                if request not in allowed_request_ids:
                    continue
                if str(metadata.get("split")) != "train":
                    raise PermissionError("scaled ShadowRoute token is not outer-train")
                self.rows.append((segment_index, sequence, int(position), int(row)))
                self.requests.add(request)
        self.rows.sort(key=lambda value: (value[0], value[1], value[2]))
        if not self.rows:
            raise ValueError("scaled ShadowRoute corpus contains no permitted tokens")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        segment_index, sequence, position, row = self.rows[index]
        segment = self.source.segments[segment_index]
        roles = (
            "normalized_target_router_input_a",
            "selected_expert_ids",
            "selected_execution_weights",
            "routed_expert_output_delta_r",
        )
        values = {
            role: segment.read("target", [row + self.layer], role)[0]
            for role in roles
        }
        if self.next_router_agreement:
            for role in (
                "post_attention_residual_u",
                "post_moe_residual_xplus",
                "shared_expert_output_delta_s",
            ):
                values[role] = segment.read(
                    "target", [row + self.layer], role
                )[0]
            for role in (
                "post_attention_residual_u",
                "raw_target_router_logits",
                "selected_expert_ids",
            ):
                values[f"next_{role}"] = segment.read(
                    "target", [row + self.layer + 1], role
                )[0]
        states = {
            "routed_expert_output_delta_r": values[
                "routed_expert_output_delta_r"
            ].unsqueeze(0)
        }
        if self.next_router_agreement:
            for role in (
                "post_attention_residual_u",
                "post_moe_residual_xplus",
                "shared_expert_output_delta_s",
                "next_post_attention_residual_u",
                "next_raw_target_router_logits",
                "next_selected_expert_ids",
            ):
                states[role] = values[role].unsqueeze(0)
        request = str(segment.sequences[sequence]["request_id"])
        return {
            "metadata": {"request_id": request, "position": position},
            "inputs": {},
            "targets": {
                "future_router_inputs": values[
                    "normalized_target_router_input_a"
                ].unsqueeze(0),
                "future_selected_ids": values["selected_expert_ids"].to(
                    torch.int64
                ).unsqueeze(0),
                "future_execution_weights": values[
                    "selected_execution_weights"
                ].float().unsqueeze(0),
                "future_available": torch.ones(1, dtype=torch.bool),
                "future_states": states,
            },
        }


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
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


def load_split(
    args: argparse.Namespace,
    split: str,
    *,
    selected_requests: set[str] | None,
) -> tuple[ShadowFactualStateDataset, set[str]]:
    index = getattr(args, f"{split}_index")
    corpus = getattr(args, f"{split}_corpus")
    companion = getattr(args, f"{split}_companion")
    base = HarpRTTDataset(index, "train", corpus_root=corpus, max_tree_nodes=32)

    # Local expert distillation consumes factual future states only.  Loading
    # every 2 MiB counterfactual-node record here would deserialize 4,096
    # label-only tensors once per target layer despite never using one.  Keep
    # the lineage fail-closed by checking the immutable manifest and exact
    # base join instead.
    companion_manifest_path = companion / "manifest.json"
    companion_audit_path = companion / "COUNTERFACTUAL_AUDIT.json"
    manifest = json.loads(companion_manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(companion_audit_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "harp_rtt_counterfactual_target_companion_v4_nodes":
        raise ValueError(f"ShadowRoute {split} companion schema changed")
    if (
        manifest.get("label_only") is not True
        or manifest.get("runtime_available") is not False
        or manifest.get("sealed_test_opened") is not False
        or audit.get("passed") is not True
        or audit.get("label_only") is not True
    ):
        raise PermissionError(f"ShadowRoute {split} companion is not sealed label-only data")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(base):
        raise ValueError(f"ShadowRoute {split} companion/base row count differs")
    companion_joins = {
        (str(record["request_id"]), int(record["source_position"]))
        for record in records
    }
    if len(companion_joins) != len(records):
        raise ValueError(f"ShadowRoute {split} companion joins are not unique")
    base_joins = set()
    for record in base.records:
        segment = base.segments[record.segment]
        request = str(segment.sequences[record.sequence]["request_id"])
        base_joins.add((request, int(record.position)))
    if base_joins != companion_joins:
        raise ValueError(f"ShadowRoute {split} companion/base joins changed")

    filtered: Dataset[Any]
    if selected_requests is None:
        filtered = base
        counts = _request_counts(base)
    else:
        filtered = RequestSubset(base, selected_requests)
        counts = Counter(
            _record_request_id(base, index)
            for index in filtered.indices  # type: ignore[attr-defined]
        )
    contract = DATA_PROFILES[args.data_profile]
    if (
        len(filtered) != int(contract["rows"][split])
        or len(counts) != int(contract["requests"][split])
        or set(counts.values()) != {16}
    ):
        raise ValueError(f"ShadowRoute {split} split has invalid row/request counts")
    return (
        ShadowFactualStateDataset(
            filtered,
            layer=args.layer,
            next_router_agreement=args.next_router_agreement,
        ),
        set(counts),
    )


def dataset_lineage(dataset: Dataset[Any]) -> tuple[set[str], set[str]]:
    if not isinstance(dataset, ShadowFactualStateDataset):
        raise TypeError("lineage inspection requires a factual-state dataset")
    source_requests: set[str] = set()
    split_groups: set[str] = set()
    for index in dataset.indices:
        record = dataset.source.records[index]
        segment = dataset.source.segments[record.segment]
        metadata = segment.sequences[record.sequence]
        source_requests.add(
            str(metadata.get("source_request_id", metadata["request_id"]))
        )
        split_groups.add(str(metadata["split_group_id"]))
    return source_requests, split_groups


def build_scaled_training_dataset(
    args: argparse.Namespace,
    *,
    partition: Mapping[str, Any],
    tune: Dataset[Any],
    development: Dataset[Any],
) -> tuple[ShadowGeneratedTokenDataset, dict[str, Any]]:
    paths = (
        args.scale_train_index,
        args.scale_train_corpus,
        args.outer_split_manifest,
    )
    if any(path is None for path in paths):
        raise ValueError("scaled training requires index, corpus, and outer manifest")
    assert args.scale_train_index is not None
    assert args.scale_train_corpus is not None
    assert args.outer_split_manifest is not None
    split_sha = sha256_file(args.outer_split_manifest)
    if split_sha != str(partition["split_manifest_sha256"]):
        raise ValueError("scaled corpus split manifest differs from the frozen partition")
    entries: dict[str, Mapping[str, Any]] = {}
    with args.outer_split_manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            request = str(entry["request_id"])
            if request in entries:
                raise ValueError("outer split manifest request IDs are not unique")
            entries[request] = entry
    holdout_sources: set[str] = set()
    holdout_groups: set[str] = set()
    for dataset in (tune, development):
        sources, groups = dataset_lineage(dataset)
        holdout_sources.update(sources)
        holdout_groups.update(groups)
    manifest_groups = {str(entry["split_group_id"]) for entry in entries.values()}
    missing_groups = sorted(holdout_groups - manifest_groups)
    if missing_groups:
        raise ValueError(
            f"outer split manifest lacks holdout groups {missing_groups[:3]}"
        )
    matched_sources = holdout_sources & entries.keys()
    blocked_components = {
        str(entry["dedup_component_id"])
        for request, entry in entries.items()
        if request in matched_sources
        or str(entry["split_group_id"]) in holdout_groups
    }
    allowed_requests = {
        request
        for request, entry in entries.items()
        if str(entry["split"]) == "train"
        and str(entry["split_group_id"]) not in holdout_groups
        and str(entry["dedup_component_id"]) not in blocked_components
    }
    if allowed_requests & holdout_sources:
        raise PermissionError("scaled training includes an exact holdout request")
    base = HarpRTTDataset(
        args.scale_train_index,
        "train",
        corpus_root=args.scale_train_corpus,
        max_tree_nodes=1,
        include_optional_current=False,
    )
    dataset = ShadowGeneratedTokenDataset(
        base,
        layer=args.layer,
        allowed_request_ids=allowed_requests,
        next_router_agreement=args.next_router_agreement,
    )
    if dataset.requests - allowed_requests:
        raise PermissionError("scaled token dataset escaped its outer-train allowlist")
    return dataset, {
        "outer_split_manifest_sha256": split_sha,
        "training_tokens": len(dataset),
        "training_requests": len(dataset.requests),
        "blocked_holdout_requests": len(holdout_sources),
        "matched_holdout_source_requests": len(matched_sources),
        "holdout_split_groups": len(holdout_groups),
        "blocked_dedup_components": len(blocked_components),
        "token_identity": "segment,sequence,absolute_target_position",
        "legacy_single_chain_features_used": False,
        "factual_target_state_reuse_only": True,
    }


def build_student(
    mode: str,
    device: torch.device,
    *,
    shadow_width: int = 16,
    indexed_active_slots: int = 8,
    basis_count: int = 16,
    basis_width: int = 32,
    basis_expert_residual_width: int = 2,
) -> nn.Module:
    if mode == "s0_exact_top1_plus_draft":
        return SwiGLUDraftExpert(2048, 512).to(device=device, dtype=torch.bfloat16)
    if mode == "s1_shared":
        return SwiGLUDraftExpert(2048, 128).to(device=device, dtype=torch.bfloat16)
    if mode == "s1_shared_width512":
        return SwiGLUDraftExpert(2048, 512).to(device=device, dtype=torch.bfloat16)
    if mode in {"resident_int4_shared", "resident_int4_tail_control"}:
        return SwiGLUDraftExpert(2048, 512).to(device=device, dtype=torch.bfloat16)
    if mode == "basisdraft_all8":
        return RouteConditionedBasisExperts(BasisDraftConfig(
            basis_count=basis_count,
            basis_width=basis_width,
            expert_residual_width=basis_expert_residual_width,
        )).to(device=device, dtype=torch.bfloat16)
    return IndexedShadowExperts(
        ShadowExpertConfig(
            shadow_width=shadow_width,
            active_slots=indexed_active_slots,
        ),
        fallback=None,
    ).to(device=device, dtype=torch.bfloat16)


def batch_tensors(
    host: Mapping[str, Any], *, layer: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[str], Mapping[str, Tensor]]:
    batch = move_to_device(host, device)
    targets = batch.get("targets")
    metadata = batch.get("metadata")
    if not isinstance(targets, Mapping) or not isinstance(metadata, Mapping):
        raise TypeError("ShadowRoute local batch lacks targets or metadata")
    states = targets.get("future_states")
    if not isinstance(states, Mapping):
        raise TypeError("ShadowRoute full-state labels are not target-only")
    if "future_states" in batch.get("inputs", {}):
        raise PermissionError("future target states leaked into ShadowRoute inputs")
    requests = metadata.get("request_id")
    if not isinstance(requests, list):
        raise ValueError("ShadowRoute batch lacks request IDs")
    router_inputs = targets["future_router_inputs"]
    routed = states["routed_expert_output_delta_r"]
    selected_ids = targets["future_selected_ids"]
    execution_weights = targets["future_execution_weights"]
    available = targets["future_available"]
    # The production local reader emits only the declared target layer.  Keep
    # the full-grid branch for synthetic/legacy tests and reject ambiguity.
    if router_inputs.ndim == 4:
        router_inputs = router_inputs[:, :, layer]
        routed = routed[:, :, layer]
        selected_ids = selected_ids[:, :, layer]
        execution_weights = execution_weights[:, :, layer]
        available = available[:, :, layer]
    elif not (
        router_inputs.ndim == 3
        and routed.ndim == 3
        and selected_ids.ndim == 3
        and execution_weights.ndim == 3
        and available.ndim == 2
    ):
        raise ValueError("ShadowRoute layer-local teacher tensor ranks changed")
    return (
        router_inputs,
        routed,
        selected_ids,
        execution_weights,
        available.bool(),
        [str(value) for value in requests],
        states,
    )


def next_router_agreement_loss(
    predicted_routed: Tensor,
    states: Mapping[str, Tensor],
    valid: Tensor,
    *,
    norm_weight: Tensor,
    router_weight: Tensor,
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    required = (
        "post_attention_residual_u",
        "post_moe_residual_xplus",
        "shared_expert_output_delta_s",
        "next_post_attention_residual_u",
        "next_raw_target_router_logits",
        "next_selected_expert_ids",
    )
    if any(name not in states for name in required):
        raise ValueError("next-router agreement labels are incomplete")
    current_u = states["post_attention_residual_u"]
    current_xplus = states["post_moe_residual_xplus"]
    shared = states["shared_expert_output_delta_s"]
    next_u = states["next_post_attention_residual_u"]
    teacher_logits = states["next_raw_target_router_logits"]
    teacher_ids = states["next_selected_expert_ids"].long()
    if any(value.shape[:-1] != predicted_routed.shape[:-1] for value in (
        current_u, current_xplus, shared, next_u, teacher_logits, teacher_ids
    )):
        raise ValueError("next-router agreement leading axes differ")
    predicted_xplus = current_u + shared + predicted_routed
    frozen_attention_delta = next_u - current_xplus
    predicted_next_u = predicted_xplus + frozen_attention_delta
    values = predicted_next_u.float()
    normalized = values * torch.rsqrt(
        values.square().mean(-1, keepdim=True) + 1e-6
    )
    # Qwen3.5-MoE stores an RMSNorm delta and applies ``1 + weight``.
    # Multiplying by the raw checkpoint tensor rotates the router input into a
    # near-zero, semantically wrong space.
    normalized = normalized * (1.0 + norm_weight.float())
    predicted_logits = F.linear(normalized, router_weight.float())
    teacher_probability = torch.softmax(teacher_logits.detach().float(), dim=-1)
    kl_rows = F.kl_div(
        torch.log_softmax(predicted_logits, dim=-1),
        teacher_probability,
        reduction="none",
    ).sum(-1)
    active = valid.bool()
    kl = (kl_rows * active.float()).sum() / active.sum().clamp_min(1)
    safe_ids = torch.where(active[..., None], teacher_ids, 0)
    exact = exact_set_nll(predicted_logits, safe_ids, valid=active, k=8)
    boundary_rows = boundary_loss_per_endpoint(
        predicted_logits,
        teacher_logits.detach(),
        safe_ids,
        margin=0.125,
        model_rank_start=9,
        teacher_rank_start=9,
        rank_end=32,
    )
    boundary = (boundary_rows * active.float()).sum() / active.sum().clamp_min(1)
    total = kl + exact + 0.2 * boundary
    return (
        total,
        {
            "next_router_kl": kl,
            "next_router_exact_set": exact,
            "next_router_boundary": boundary,
        },
        predicted_logits,
    )


def teacher_values(
    inputs: Tensor,
    ids: Tensor,
    weights: Tensor,
    gate_up: Tensor,
    down: Tensor,
) -> tuple[Tensor, Tensor]:
    with torch.no_grad():
        individual = target_selected_expert_outputs(inputs, ids, gate_up, down)
        aggregate = (individual * weights[..., None].to(individual)).sum(-2)
    return individual, aggregate


def objective(
    model: nn.Module,
    host: Mapping[str, Any],
    *,
    mode: str,
    layer: int,
    device: torch.device,
    gate_up: Tensor,
    down: Tensor,
    next_norm_weight: Tensor | None = None,
    next_router_weight: Tensor | None = None,
    router_agreement_weight: float = 0.0,
    aggregate_loss_weight: float = 1.0,
    individual_loss_weight: float = 1.0,
    cosine_loss_weight: float = 0.1,
) -> tuple[
    Tensor, dict[str, float], Tensor, Tensor, Tensor, list[str], Tensor | None
]:
    inputs, routed_target, ids, weights, valid, requests, states = batch_tensors(
        host, layer=layer, device=device
    )
    if mode == "s0_exact_top1_plus_draft":
        with torch.no_grad():
            top1_target = target_selected_expert_outputs(
                inputs, ids[..., :1], gate_up, down
            )[..., 0, :]
    elif mode in {"s2_indexed", "basisdraft_all8"}:
        if mode == "s2_indexed":
            assert isinstance(model, IndexedShadowExperts)
            active_slots = model.config.routed_slots
        else:
            assert isinstance(model, RouteConditionedBasisExperts)
            active_slots = model.config.exact_k
        ids = ids[..., :active_slots]
        weights = weights[..., :active_slots]
        individual_target: Tensor | None = None
        if mode != "basisdraft_all8" or individual_loss_weight > 0:
            individual_target, _reconstructed = teacher_values(
                inputs, ids, weights, gate_up, down
            )
            routed_target = _reconstructed
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        if mode == "s0_exact_top1_plus_draft":
            exact_top1 = top1_target * weights[..., 0, None].to(inputs)
            predicted = exact_top1 + model(inputs)
            individual_loss = predicted.sum() * 0.0
        elif mode in {"s1_shared", "s1_shared_width512"}:
            predicted = model(inputs)
            individual_loss = predicted.sum() * 0.0
        elif mode in {"resident_int4_shared", "resident_int4_tail_control"}:
            assert isinstance(model, PackedInt4ResidentExperts)
            components = model.forward_components(inputs, ids, weights)
            predicted = components.output
            individual_loss = predicted.sum() * 0.0
        elif mode == "basisdraft_all8":
            assert isinstance(model, RouteConditionedBasisExperts)
            predicted = model(inputs, ids, weights)
            if individual_loss_weight > 0:
                assert individual_target is not None
                predicted_individual = model.selected_unweighted(inputs, ids)
                individual_loss = selected_expert_distillation_loss(
                    predicted_individual, individual_target, weights, valid
                )
            else:
                individual_loss = predicted.sum() * 0.0
        else:
            assert isinstance(model, IndexedShadowExperts)
            predicted_individual = model.selected_unweighted(inputs, ids)
            predicted = (predicted_individual * weights[..., None].to(inputs)).sum(-2)
            individual_loss = selected_expert_distillation_loss(
                predicted_individual, individual_target, weights, valid
            )
    rows = F.huber_loss(
        predicted.float(), routed_target.float(), reduction="none", delta=1.0
    ).mean(-1)
    aggregate = (rows * valid.float()).sum() / valid.sum().clamp_min(1)
    cosine_rows = 1.0 - F.cosine_similarity(
        predicted.float(), routed_target.float(), dim=-1, eps=1e-8
    )
    cosine = (cosine_rows * valid.float()).sum() / valid.sum().clamp_min(1)
    tail_huber = predicted.sum() * 0.0
    normalized_tail_huber = predicted.sum() * 0.0
    control_huber = predicted.sum() * 0.0
    if mode == "resident_int4_tail_control":
        assert isinstance(model, PackedInt4ResidentExperts)
        target_tail = routed_target.float() - components.resident_output.detach().float()
        tail_rows = F.huber_loss(
            components.tail_output.float(), target_tail,
            reduction="none", delta=1.0,
        ).mean(-1)
        tail_huber = (
            tail_rows * valid.float()
        ).sum() / valid.sum().clamp_min(1)
        mass = components.missing_mass.float()
        normalized_valid = valid & (mass[..., 0] >= 0.05)
        normalized_rows = F.huber_loss(
            components.tail_output.float() / mass.clamp_min(0.05),
            target_tail / mass.clamp_min(0.05),
            reduction="none", delta=1.0,
        ).mean(-1)
        normalized_tail_huber = (
            normalized_rows * normalized_valid.float()
        ).sum() / normalized_valid.sum().clamp_min(1)
        precontrol = (
            components.resident_output.detach().float()
            + components.tail_output.float()
        )
        control_target = routed_target.float() - precontrol
        control_rows = F.huber_loss(
            components.control_output.float(), control_target,
            reduction="none", delta=1.0,
        ).mean(-1)
        control_huber = (
            control_rows * valid.float()
        ).sum() / valid.sum().clamp_min(1)
    agreement = predicted.sum() * 0.0
    agreement_parts: dict[str, Tensor] = {}
    next_overlap: Tensor | None = None
    if router_agreement_weight > 0:
        if next_norm_weight is None or next_router_weight is None:
            raise ValueError("router-agreement weights were not loaded")
        agreement, agreement_parts, next_logits = next_router_agreement_loss(
            predicted,
            states,
            valid,
            norm_weight=next_norm_weight,
            router_weight=next_router_weight,
        )
        teacher_ids = states["next_selected_expert_ids"].long()
        next_ids = stable_topk(next_logits, k=8)
        next_overlap = (
            teacher_ids[..., None] == next_ids[..., None, :]
        ).any(-1).float().mean(-1).detach()
    loss = (
        float(aggregate_loss_weight) * aggregate
        + float(cosine_loss_weight) * cosine
        + float(individual_loss_weight) * individual_loss
        + float(router_agreement_weight) * agreement
        + (0.5 * tail_huber + 0.1 * normalized_tail_huber + 0.25 * control_huber)
    )
    return (
        loss,
        {
            "aggregate_huber": float(aggregate.detach()),
            "aggregate_cosine_distance": float(cosine.detach()),
            "individual_huber": float(individual_loss.detach()),
            "native_reconstruction_checked_in_epoch_zero": 1.0,
            "tail_huber": float(tail_huber.detach()),
            "normalized_tail_huber": float(normalized_tail_huber.detach()),
            "control_huber": float(control_huber.detach()),
            "next_router_agreement": float(agreement.detach()),
            **{name: float(value.detach()) for name, value in agreement_parts.items()},
            "total": float(loss.detach()),
        },
        predicted.detach(),
        routed_target.detach(),
        valid.detach(),
        requests,
        next_overlap,
    )


@torch.no_grad()
def native_reconstruction_audit(
    dataset: Dataset[Any],
    *,
    layer: int,
    device: torch.device,
    gate_up: Tensor,
    down: Tensor,
    workers: int,
) -> float:
    host = next(iter(loader(
        dataset, batch=1, shuffle=False, seed=0, workers=workers, device=device
    )))
    inputs, routed_target, ids, weights, valid, _requests, _states = batch_tensors(
        host, layer=layer, device=device
    )
    _individual, reconstructed = teacher_values(inputs, ids, weights, gate_up, down)
    rows = (reconstructed.float() - routed_target.float()).square().mean(-1)
    maximum = float(rows[valid].max() if valid.any() else 0.0)
    if maximum > 0.05:
        raise ValueError("native expert replay does not reconstruct captured routed residual")
    return maximum


@torch.no_grad()
def next_router_teacher_audit(
    dataset: Dataset[Any],
    *,
    layer: int,
    device: torch.device,
    norm_weight: Tensor,
    router_weight: Tensor,
    workers: int,
) -> dict[str, float]:
    host = next(iter(loader(
        dataset, batch=32, shuffle=False, seed=0, workers=workers, device=device
    )))
    _inputs, routed, _ids, _weights, valid, _requests, states = batch_tensors(
        host, layer=layer, device=device
    )
    loss, parts, logits = next_router_agreement_loss(
        routed,
        states,
        valid,
        norm_weight=norm_weight,
        router_weight=router_weight,
    )
    teacher_ids = states["next_selected_expert_ids"].long()
    predicted_ids = stable_topk(logits, k=8)
    overlap = (
        teacher_ids[..., None] == predicted_ids[..., None, :]
    ).any(-1).float().mean(-1)
    recall = float((overlap * valid.float()).sum() / valid.sum().clamp_min(1))
    return {
        "teacher_forced_recall_at_8": recall,
        "teacher_forced_total": float(loss),
        **{name: float(value) for name, value in parts.items()},
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Dataset[Any],
    *,
    mode: str,
    layer: int,
    device: torch.device,
    gate_up: Tensor,
    down: Tensor,
    microbatch: int,
    workers: int,
    next_norm_weight: Tensor | None = None,
    next_router_weight: Tensor | None = None,
    router_agreement_weight: float = 0.0,
    aggregate_loss_weight: float = 1.0,
    individual_loss_weight: float = 1.0,
    cosine_loss_weight: float = 0.1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    request_cells: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
    error_sum = energy_sum = dot_sum = pred_norm = target_norm = 0.0
    elements = 0
    next_overlap_sum = next_overlap_rows = 0.0
    for host in loader(
        dataset, batch=microbatch, shuffle=False, seed=0,
        workers=workers, device=device,
    ):
        _loss, _parts, predicted, target, valid, requests, next_overlap = objective(
            model, host, mode=mode, layer=layer, device=device,
            gate_up=gate_up, down=down,
            next_norm_weight=next_norm_weight,
            next_router_weight=next_router_weight,
            router_agreement_weight=router_agreement_weight,
            aggregate_loss_weight=aggregate_loss_weight,
            individual_loss_weight=individual_loss_weight,
            cosine_loss_weight=cosine_loss_weight,
        )
        if next_overlap is not None:
            next_overlap_sum += float((next_overlap * valid.float()).sum())
            next_overlap_rows += float(valid.sum())
        error = (predicted.float() - target.float()).square().mean(-1)
        energy = target.float().square().mean(-1)
        cosine = F.cosine_similarity(predicted.float(), target.float(), dim=-1, eps=1e-8)
        for row, request in enumerate(requests):
            for horizon in range(4):
                if bool(valid[row, horizon]):
                    request_cells[(request, horizon + 1)].append(
                        (
                            float(error[row, horizon]),
                            float(energy[row, horizon]),
                            float(cosine[row, horizon]),
                        )
                    )
        active_pred = predicted.float()[valid]
        active_target = target.float()[valid]
        error_sum += float((active_pred - active_target).square().sum())
        energy_sum += float(active_target.square().sum())
        dot_sum += float((active_pred * active_target).sum())
        pred_norm += float(active_pred.square().sum())
        target_norm += float(active_target.square().sum())
        elements += active_target.numel()
    if elements == 0:
        raise ValueError("ShadowRoute evaluation contains no valid states")
    normalized_rmse = math.sqrt(error_sum / max(energy_sum, 1e-12))
    cosine = dot_sum / max(math.sqrt(pred_norm * target_norm), 1e-12)
    rows = []
    for (request, horizon), values in sorted(request_cells.items()):
        rows.append(
            {
                "request_id": request,
                "horizon": horizon,
                "layer": layer,
                "mse": sum(value[0] for value in values) / len(values),
                "target_energy": sum(value[1] for value in values) / len(values),
                "cosine": sum(value[2] for value in values) / len(values),
            }
        )
    metrics = {
            "normalized_rmse": normalized_rmse,
            "cosine": cosine,
            "relative_mse_reduction_vs_zero": 1.0 - error_sum / max(energy_sum, 1e-12),
            "rows": len(rows),
    }
    if next_overlap_rows:
        metrics["next_router_recall_at_8"] = next_overlap_sum / next_overlap_rows
    return metrics, rows


def expert_counts(
    dataset: Dataset[Any], *, layer: int, device: torch.device, workers: int,
    active_slots: int = 8,
) -> Tensor:
    counts = torch.zeros(256, dtype=torch.int64)
    for host in loader(
        dataset, batch=32, shuffle=False, seed=0, workers=workers, device=device
    ):
        _inputs, _routed, ids, _weights, valid, _requests, _states = batch_tensors(
            host, layer=layer, device=device
        )
        active = ids[valid][..., :active_slots].detach().cpu().flatten()
        counts += torch.bincount(active, minlength=256)
    return counts


def autotune(
    model: nn.Module,
    dataset: Dataset[Any],
    args: argparse.Namespace,
    *,
    device: torch.device,
    gate_up: Tensor,
    down: Tensor,
    next_norm_weight: Tensor | None = None,
    next_router_weight: Tensor | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    choices = (args.microbatch_size,) if args.microbatch_size else MICROBATCH_CHOICES
    trace: list[dict[str, Any]] = []
    for size in choices:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            host = next(iter(loader(
                dataset, batch=size, shuffle=False, seed=0,
                workers=args.num_workers, device=device,
            )))
            model.zero_grad(set_to_none=True)
            objective(
                model, host, mode=args.mode, layer=args.layer, device=device,
                gate_up=gate_up, down=down,
                next_norm_weight=next_norm_weight,
                next_router_weight=next_router_weight,
                router_agreement_weight=(
                    args.router_agreement_weight if args.next_router_agreement else 0.0
                ),
                aggregate_loss_weight=args.aggregate_loss_weight,
                individual_loss_weight=args.individual_loss_weight,
                cosine_loss_weight=args.cosine_loss_weight,
            )[0].backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
            accepted = device.type != "cuda" or peak <= 21.0
            trace.append({"microbatch": size, "peak_reserved_gib": peak, "accepted": accepted})
            if accepted:
                return size, trace
        except torch.OutOfMemoryError:
            trace.append({"microbatch": size, "oom": True, "accepted": False})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("ShadowRoute layer shard exceeds the 21 GiB training gate")


def main() -> None:
    args = parse_args()
    args.data_profile = "b2_reuse_4096"
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite ShadowRoute run {args.output}")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    if args.next_router_agreement and (
        not math.isfinite(args.router_agreement_weight)
        or args.router_agreement_weight <= 0
    ):
        raise ValueError("router-agreement weight must be finite and positive")
    for name in (
        "aggregate_loss_weight", "individual_loss_weight", "cosine_loss_weight"
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if args.basis_coefficients_only and args.mode != "basisdraft_all8":
        raise ValueError("coefficient-only training requires BasisDraft mode")
    resident_stage = args.resident_eval_only or args.resident_export_only
    if args.resident_eval_only and args.resident_export_only:
        raise ValueError("resident evaluation and export modes are mutually exclusive")
    if resident_stage != (args.mode == "resident_int4_shared"):
        raise ValueError(
            "resident INT4 shared mode requires an optimizer-free eval/export stage"
        )
    if args.mode == "resident_int4_tail_control" and resident_stage:
        raise ValueError("resident v2 is a trainable stage, not a v1 export")
    if args.resident_eval_only and not args.next_router_agreement:
        raise ValueError("resident hybrid evaluation requires next-router labels")
    if args.resident_allocation_plan is not None and args.mode not in {
        "resident_int4_shared", "resident_int4_tail_control"
    }:
        raise ValueError("resident allocation plan is valid only for resident modes")
    if (
        args.mode == "resident_int4_tail_control"
        and args.layer == 39
        and args.next_router_agreement
    ):
        raise ValueError("layer 39 has no next-router control transition")
    partition = validate_partition(args.partition_manifest, args.data_profile)
    reuse = validate_reuse_split(args.reuse_split_manifest)
    selected = {
        "train": set(reuse["inner_split"]["training_requests"]),
        "tune": set(reuse["inner_split"]["tuning_requests"]),
        "development": None,
    }
    datasets: dict[str, Dataset[Any]] = {}
    groups: dict[str, set[str]] = {}
    scale_requested = any(
        value is not None
        for value in (
            args.scale_train_index,
            args.scale_train_corpus,
            args.outer_split_manifest,
        )
    )
    if scale_requested and not all(
        value is not None
        for value in (
            args.scale_train_index,
            args.scale_train_corpus,
            args.outer_split_manifest,
        )
    ):
        raise ValueError("scaled training arguments must be supplied together")
    load_splits = ("tune", "development") if scale_requested else (
        "train", "tune", "development"
    )
    for split in load_splits:
        datasets[split], groups[split] = load_split(
            args, split, selected_requests=selected[split]
        )
    scale_provenance: dict[str, Any] | None = None
    if scale_requested:
        datasets["train"], scale_provenance = build_scaled_training_dataset(
            args,
            partition=partition,
            tune=datasets["tune"],
            development=datasets["development"],
        )
        assert isinstance(datasets["train"], ShadowGeneratedTokenDataset)
        groups["train"] = set(datasets["train"].requests)
    if any(
        groups[left] & groups[right]
        for left, right in (("train", "tune"), ("train", "development"), ("tune", "development"))
    ):
        raise PermissionError("ShadowRoute request groups overlap")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(args.seed, deterministic=args.deterministic)
    checkpoint = IndexedCheckpoint(args.target_model)
    gate_up, down = load_target_layer_experts(
        checkpoint, args.layer, device=device, dtype=torch.bfloat16
    )
    next_norm_weight: Tensor | None = None
    next_router_weight: Tensor | None = None
    initial_router_tune: dict[str, Any] | None = None
    if args.next_router_agreement:
        if args.layer >= 39:
            raise ValueError("layer 39 has no within-token next router")
        prefix = f"model.language_model.layers.{args.layer + 1}."
        names = (
            prefix + "post_attention_layernorm.weight",
            prefix + "mlp.gate.weight",
        )
        values = checkpoint.tensors(names)
        next_norm_weight = values[names[0]].to(
            device=device, dtype=torch.bfloat16
        )
        next_router_weight = values[names[1]].to(
            device=device, dtype=torch.bfloat16
        )
        if next_norm_weight.shape != (2048,) or next_router_weight.shape != (
            256, 2048
        ):
            raise ValueError("next target router geometry changed")
    model = build_student(
        args.mode,
        device,
        shadow_width=args.shadow_width,
        indexed_active_slots=args.indexed_active_slots,
        basis_count=args.basis_count,
        basis_width=args.basis_width,
        basis_expert_residual_width=args.basis_expert_residual_width,
    )
    initializer_provenance: dict[str, Any] | None = None
    deferred_resident_state: Mapping[str, Tensor] | None = None
    if args.initializer_checkpoint is not None:
        initializer = torch.load(
            args.initializer_checkpoint, map_location="cpu", weights_only=False
        )
        expected_mode = (
            "resident_int4_shared"
            if args.mode == "resident_int4_tail_control"
            else (
                "s1_shared_width512"
                if args.mode in {"basisdraft_all8", "resident_int4_shared"}
                and initializer.get("mode") == "s1_shared_width512"
                else args.mode
            )
        )
        expected = {
            "schema": SCHEMA,
            "mode": expected_mode,
            "layer": args.layer,
            "target_checkpoint_index_sha256": checkpoint.index_sha256,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        }
        for field, value in expected.items():
            if initializer.get(field) != value:
                raise ValueError(f"ShadowRoute initializer {field} mismatch")
        state = initializer.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("ShadowRoute initializer lacks a model state")
        if args.mode == "resident_int4_tail_control":
            deferred_resident_state = state
        elif args.mode == "basisdraft_all8" and expected_mode == "s1_shared_width512":
            assert isinstance(model, RouteConditionedBasisExperts)
            shared = SwiGLUDraftExpert(2048, 512)
            shared.load_state_dict(state, strict=True)
            model.initialize_from_shared(shared)
        else:
            model.load_state_dict(state, strict=True)
        initializer_provenance = {
            "checkpoint_sha256": sha256_file(args.initializer_checkpoint),
            "source_commit": str(initializer["source_commit"]),
            "best_epoch": int(initializer.get("best_epoch", 0)),
            "diagnostic_only": bool(initializer.get("diagnostic_only", False)),
            "closed_loop_authorized": bool(
                initializer.get("closed_loop_authorized", False)
            ),
        }
    resident_ids: Tensor | None = None
    resident_plan_sha256: str | None = None
    if args.mode in {"resident_int4_shared", "resident_int4_tail_control"}:
        if args.initializer_checkpoint is None:
            raise ValueError("resident hybrid requires a frozen initializer")
        assert isinstance(model, SwiGLUDraftExpert)
        resident_counts = expert_counts(
            datasets["train"], layer=args.layer, device=device,
            workers=args.num_workers, active_slots=8,
        )
        if args.resident_allocation_plan is None:
            resident_ids = torch.argsort(
                resident_counts, descending=True, stable=True
            )[: args.resident_count].to(device)
        else:
            plan = json.loads(args.resident_allocation_plan.read_text(encoding="utf-8"))
            if plan.get("schema") not in {
                "harp_shadowroute_resident_allocation_v1",
                "harp_shadowroute_resident_allocation_v2",
            }:
                raise ValueError("resident allocation plan schema changed")
            expected_plan = {
                "source_commit": args.source_commit,
                "partition_manifest_sha256": sha256_file(args.partition_manifest),
                "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
                "selection_uses_train_routes_only": True,
                "formal_validation_opened": False,
                "calibration_opened": False,
                "sealed_test_opened": False,
            }
            for field, expected_value in expected_plan.items():
                if plan.get(field) != expected_value:
                    raise ValueError(f"resident allocation plan {field} mismatch")
            ids_by_layer = plan.get("resident_expert_ids_by_layer")
            if not isinstance(ids_by_layer, list) or len(ids_by_layer) != 40:
                raise ValueError("resident allocation plan lacks forty layers")
            resident_ids = torch.as_tensor(
                ids_by_layer[args.layer], dtype=torch.long, device=device
            )
            if (
                resident_ids.ndim != 1
                or resident_ids.unique().numel() != resident_ids.numel()
                or bool(((resident_ids < 0) | (resident_ids >= 256)).any())
            ):
                raise ValueError("resident allocation plan contains invalid expert IDs")
            resident_plan_sha256 = sha256_file(args.resident_allocation_plan)
        control = None
        if args.mode == "resident_int4_tail_control" and args.layer < 39:
            if next_router_weight is None:
                raise ValueError("resident v2 control requires next-router geometry")
            control = RouterVisibleTailControl(
                rank=args.resident_control_rank,
                device=device,
                dtype=torch.bfloat16,
            )
            control.bind_next_router(next_router_weight)
        model = PackedInt4ResidentExperts.from_target(
            resident_ids,
            SharedResidualExperts(model),
            gate_up,
            down,
            exact_k=8,
            group_size=64,
            router_control=control,
        )
        if args.mode == "resident_int4_tail_control":
            if deferred_resident_state is None:
                raise ValueError("resident v2 lacks the sealed v1 parent state")
            incompatible = model.load_state_dict(
                deferred_resident_state, strict=False
            )
            expected_missing = (
                {
                    "router_control.expert_codes.weight",
                    "router_control.hidden_projection.weight",
                    "router_control.router_delta_projection.weight",
                }
                if args.layer < 39 else set()
            )
            if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
                raise ValueError("resident v2 parent state is not exactly compatible")
            model.requires_grad_(False)
            if args.resident_v2_phase in {"tail", "joint"}:
                model.fallback.requires_grad_(True)
            if args.layer < 39 and args.resident_v2_phase in {"control", "joint"}:
                assert model.router_control is not None
                model.router_control.requires_grad_(True)
        else:
            model.requires_grad_(False)
    if args.basis_coefficients_only:
        assert isinstance(model, RouteConditionedBasisExperts)
        model.gate_up_proj.requires_grad_(False)
        model.down_proj.requires_grad_(False)
        model.expert_coefficients.requires_grad_(True)
    selected_neurons = None
    if isinstance(model, IndexedShadowExperts):
        selected_neurons = model.initialize_from_target_neurons(
            gate_up, down, target_neuron_importance(gate_up, down)
        ).cpu()
    if args.mode == "s2_indexed":
        counts = expert_counts(
            datasets["train"], layer=args.layer, device=device, workers=args.num_workers,
            active_slots=args.indexed_active_slots,
        )
        trained_mass = float(
            counts[counts >= args.minimum_expert_count].sum() / counts.sum().clamp_min(1)
        )
    elif args.mode in {"resident_int4_shared", "resident_int4_tail_control"}:
        counts = resident_counts
        trained_mass = float(counts[resident_ids.cpu()].sum() / counts.sum().clamp_min(1))
    else:
        counts = torch.zeros(256, dtype=torch.int64)
        trained_mass = 1.0

    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "mode": args.mode,
        "layer": args.layer,
        "seed": args.seed,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "companion_manifest_sha256": {
            split: sha256_file(getattr(args, f"{split}_companion") / "manifest.json")
            for split in ("train", "tune", "development")
        },
        "counterfactual_payload_loaded": False,
        "partition_schema": partition["schema"],
        "diagnostic_reuse": True,
        "scaled_outer_train_factual_reuse": scale_provenance,
        "initializer": initializer_provenance,
        "next_router_agreement": args.next_router_agreement,
        "router_agreement_weight": (
            args.router_agreement_weight if args.next_router_agreement else 0.0
        ),
        "aggregate_loss_weight": args.aggregate_loss_weight,
        "individual_loss_weight": args.individual_loss_weight,
        "cosine_loss_weight": args.cosine_loss_weight,
        "basis_coefficients_only": args.basis_coefficients_only,
        "frozen_attention_delta_teacher_forcing": args.next_router_agreement,
        "target_state_is_label_only": True,
        "native_target_layer_loaded": True,
        "complete_target_model_loaded": False,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "trainable_names": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "minimum_expert_count": args.minimum_expert_count,
        "shadow_width": args.shadow_width,
        "indexed_active_slots": args.indexed_active_slots,
        "basis_count": args.basis_count,
        "basis_width": args.basis_width,
        "basis_expert_residual_width": args.basis_expert_residual_width,
        "resident_count": (
            int(resident_ids.numel()) if resident_ids is not None else None
        ),
        "resident_allocation_plan_sha256": resident_plan_sha256,
        "resident_v2_phase": (
            args.resident_v2_phase
            if args.mode == "resident_int4_tail_control" else None
        ),
        "resident_control_rank": (
            args.resident_control_rank
            if args.mode == "resident_int4_tail_control" and args.layer < 39
            else None
        ),
        "resident_v2_loss_weights": (
            {"tail": 0.5, "normalized_tail": 0.1, "control": 0.25}
            if args.mode == "resident_int4_tail_control" else None
        ),
        "resident_expert_ids": (
            resident_ids.detach().cpu().tolist() if resident_ids is not None else None
        ),
        "resident_train_slot_coverage": (
            trained_mass if args.mode in {
                "resident_int4_shared", "resident_int4_tail_control"
            } else None
        ),
        "expert_frequency_gate_applicable": args.mode == "s2_indexed",
        "trained_expert_count": (
            int((counts >= args.minimum_expert_count).sum())
            if args.mode == "s2_indexed" else (
                int(resident_ids.numel())
                if args.mode in {
                    "resident_int4_shared", "resident_int4_tail_control"
                } and resident_ids is not None
                else 256
            )
        ),
        "trained_selected_slot_mass": trained_mass,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)
    reconstruction_max_mse = native_reconstruction_audit(
        datasets["train"], layer=args.layer, device=device,
        gate_up=gate_up, down=down, workers=args.num_workers,
    )
    write_json_exclusive(
        args.output / "EPOCH_ZERO_AUDIT.json",
        {
            "native_expert_reconstruction_max_mse": reconstruction_max_mse,
            "maximum_allowed_mse": 0.05,
            "passed": reconstruction_max_mse <= 0.05,
            "optimizer_constructed": False,
        },
    )
    if args.next_router_agreement:
        assert next_norm_weight is not None and next_router_weight is not None
        teacher_audit = next_router_teacher_audit(
            datasets["train"],
            layer=args.layer,
            device=device,
            norm_weight=next_norm_weight,
            router_weight=next_router_weight,
            workers=args.num_workers,
        )
        initial_router_tune, initial_tune_rows = evaluate(
            model,
            datasets["tune"],
            mode=args.mode,
            layer=args.layer,
            device=device,
            gate_up=gate_up,
            down=down,
            microbatch=32,
            workers=args.num_workers,
            next_norm_weight=next_norm_weight,
            next_router_weight=next_router_weight,
            router_agreement_weight=args.router_agreement_weight,
            aggregate_loss_weight=args.aggregate_loss_weight,
            individual_loss_weight=args.individual_loss_weight,
            cosine_loss_weight=args.cosine_loss_weight,
        )
        router_audit = {
            **teacher_audit,
            "initializer_tune": initial_router_tune,
            "minimum_teacher_forced_recall_at_8": 0.98,
            "passed": teacher_audit["teacher_forced_recall_at_8"] >= 0.98,
            "optimizer_constructed": False,
        }
        write_json_exclusive(
            args.output / "EPOCH_ZERO_ROUTER_AUDIT.json", router_audit
        )
        if not router_audit["passed"]:
            raise ValueError("teacher-forced next-router reconstruction failed")
    else:
        initial_tune_rows = []
    if resident_stage:
        assert isinstance(model, PackedInt4ResidentExperts)
        if initial_router_tune is None:
            initial_router_tune, initial_tune_rows = evaluate(
                model,
                datasets["tune"],
                mode=args.mode,
                layer=args.layer,
                device=device,
                gate_up=gate_up,
                down=down,
                microbatch=32,
                workers=args.num_workers,
                aggregate_loss_weight=args.aggregate_loss_weight,
                individual_loss_weight=0.0,
                cosine_loss_weight=args.cosine_loss_weight,
            )
        development, development_rows = evaluate(
            model,
            datasets["development"],
            mode=args.mode,
            layer=args.layer,
            device=device,
            gate_up=gate_up,
            down=down,
            microbatch=32,
            workers=args.num_workers,
            next_norm_weight=next_norm_weight,
            next_router_weight=next_router_weight,
            router_agreement_weight=(
                args.router_agreement_weight if args.next_router_agreement else 0.0
            ),
            aggregate_loss_weight=args.aggregate_loss_weight,
            individual_loss_weight=0.0,
            cosine_loss_weight=args.cosine_loss_weight,
        )
        write_rows(args.output / "tune_residual_predictions.jsonl", initial_tune_rows)
        write_rows(
            args.output / "development_residual_predictions.jsonl",
            development_rows,
        )
        checkpoint_path = args.output / f"shadow_{args.mode}_layer_{args.layer:02d}.pt"
        checkpoint_value = {
            "schema": SCHEMA,
            "source_commit": args.source_commit,
            "mode": args.mode,
            "layer": args.layer,
            "seed": args.seed,
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "resident_expert_ids": model.resident_ids.detach().cpu(),
            "resident_count": model.resident_count,
            "resident_train_slot_coverage": trained_mass,
            "initializer": initializer_provenance,
            "tune": initial_router_tune,
            "development": development,
            "target_checkpoint_index_sha256": checkpoint.index_sha256,
            "partition_manifest_sha256": sha256_file(args.partition_manifest),
            "diagnostic_only": True,
            "closed_loop_authorized": False,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        }
        with checkpoint_path.open("xb") as handle:
            torch.save(checkpoint_value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        result = {
            "schema": RESULT_SCHEMA,
            "mode": args.mode,
            "layer": args.layer,
            "resident_count": model.resident_count,
            "resident_train_slot_coverage": trained_mass,
            "tune": initial_router_tune,
            "development": development,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_started": False,
            "optimizer_constructed": False,
            "closed_loop_evaluation_started": False,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        }
        write_json_exclusive(args.output / "STAGE_RESULT.json", result)
        write_checksums(args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    microbatch, trace = autotune(
        model,
        datasets["train"],
        args,
        device=device,
        gate_up=gate_up,
        down=down,
        next_norm_weight=next_norm_weight,
        next_router_weight=next_router_weight,
    )
    write_json_exclusive(
        args.output / "MEMORY_AUTOTUNE.json",
        {
            "selected_microbatch": microbatch,
            "effective_batch": EFFECTIVE_BATCH,
            "trace": trace,
            "optimizer_constructed": False,
        },
    )
    if args.preflight_only:
        write_json_exclusive(
            args.output / "PREFLIGHT_RESULT.json",
            {
                "training_started": False,
                "closed_loop_authorized": False,
                "selected_microbatch": microbatch,
            },
        )
        write_checksums(args.output)
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    write_json_exclusive(
        args.output / "OPTIMIZER_START.json",
        {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
    )
    accumulation = math.ceil(EFFECTIVE_BATCH / microbatch)
    if args.next_router_agreement:
        assert initial_router_tune is not None
        best_value = -float(initial_router_tune["next_router_recall_at_8"])
        best_epoch = 0
        best_state: dict[str, Tensor] | None = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    else:
        best_value = math.inf
        best_epoch = 0
        best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = batches = 0.0
        for step, host in enumerate(
            loader(
                datasets["train"], batch=microbatch, shuffle=True,
                seed=args.seed + epoch, workers=args.num_workers, device=device,
            ),
            start=1,
        ):
            loss, _parts, *_ = objective(
                model, host, mode=args.mode, layer=args.layer, device=device,
                gate_up=gate_up, down=down,
                next_norm_weight=next_norm_weight,
                next_router_weight=next_router_weight,
                router_agreement_weight=(
                    args.router_agreement_weight if args.next_router_agreement else 0.0
                ),
                aggregate_loss_weight=args.aggregate_loss_weight,
                individual_loss_weight=args.individual_loss_weight,
                cosine_loss_weight=args.cosine_loss_weight,
            )
            (loss / accumulation).backward()
            total += float(loss.detach())
            batches += 1
            if step % accumulation == 0 or step * microbatch >= len(datasets["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        tune, _ = evaluate(
            model, datasets["tune"], mode=args.mode, layer=args.layer,
            device=device, gate_up=gate_up, down=down,
            microbatch=microbatch, workers=args.num_workers,
            next_norm_weight=next_norm_weight,
            next_router_weight=next_router_weight,
            router_agreement_weight=(
                args.router_agreement_weight if args.next_router_agreement else 0.0
            ),
            aggregate_loss_weight=args.aggregate_loss_weight,
            individual_loss_weight=args.individual_loss_weight,
            cosine_loss_weight=args.cosine_loss_weight,
        )
        append_jsonl(
            args.output / "metrics.jsonl",
            {"epoch": epoch, "train_loss": total / max(batches, 1.0), "tune": tune},
        )
        value = (
            -float(tune["next_router_recall_at_8"])
            if args.next_router_agreement
            else float(tune["normalized_rmse"])
        )
        if value < best_value:
            best_value = value
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("ShadowRoute local trainer produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    development, rows = evaluate(
        model, datasets["development"], mode=args.mode, layer=args.layer,
        device=device, gate_up=gate_up, down=down,
        microbatch=microbatch, workers=args.num_workers,
        next_norm_weight=next_norm_weight,
        next_router_weight=next_router_weight,
        router_agreement_weight=(
            args.router_agreement_weight if args.next_router_agreement else 0.0
        ),
        aggregate_loss_weight=args.aggregate_loss_weight,
        individual_loss_weight=args.individual_loss_weight,
        cosine_loss_weight=args.cosine_loss_weight,
    )
    write_rows(args.output / "development_residual_predictions.jsonl", rows)
    component_gate = bool(
        (
            development.get("next_router_recall_at_8", 0.0) >= 0.80
            and development["relative_mse_reduction_vs_zero"] >= 0.40
        )
        if args.next_router_agreement
        else (
            development["relative_mse_reduction_vs_zero"] >= 0.50
            and development["cosine"] >= 0.70
        )
    )
    checkpoint_value = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "mode": (
            "resident_tail_control_v2"
            if args.mode == "resident_int4_tail_control" else args.mode
        ),
        "layer": args.layer,
        "seed": args.seed,
        "model_state_dict": best_state,
        "resident_expert_ids": (
            model.resident_ids.detach().cpu()
            if isinstance(model, PackedInt4ResidentExperts) else None
        ),
        "resident_count": (
            model.resident_count
            if isinstance(model, PackedInt4ResidentExperts) else None
        ),
        "resident_v2_phase": (
            args.resident_v2_phase
            if args.mode == "resident_int4_tail_control" else None
        ),
        "resident_control_rank": (
            args.resident_control_rank
            if args.mode == "resident_int4_tail_control" and args.layer < 39
            else None
        ),
        "selected_neurons": selected_neurons,
        "expert_counts": counts,
        "minimum_expert_count": args.minimum_expert_count,
        "shadow_width": args.shadow_width,
        "indexed_active_slots": args.indexed_active_slots,
        "basis_count": args.basis_count,
        "basis_width": args.basis_width,
        "basis_expert_residual_width": args.basis_expert_residual_width,
        "trained_selected_slot_mass": trained_mass,
        "best_epoch": best_epoch,
        "selection_metric": (
            "next_router_recall_at_8"
            if args.next_router_agreement else "normalized_rmse"
        ),
        "best_selection_value": (
            -best_value if args.next_router_agreement else best_value
        ),
        "best_tune_normalized_rmse": (
            None if args.next_router_agreement else best_value
        ),
        "development": development,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "diagnostic_only": True,
        "closed_loop_authorized": component_gate,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    checkpoint_mode = (
        "resident_tail_control_v2"
        if args.mode == "resident_int4_tail_control" else args.mode
    )
    checkpoint_path = args.output / f"shadow_{checkpoint_mode}_layer_{args.layer:02d}.pt"
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint_value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    result = {
        "schema": RESULT_SCHEMA,
        "mode": args.mode,
        "layer": args.layer,
        "best_epoch": best_epoch,
        "development": development,
        "component_gate_passed": component_gate,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_started": True,
        "closed_loop_evaluation_started": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    write_checksums(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
