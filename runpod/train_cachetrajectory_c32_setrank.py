#!/usr/bin/env python3
"""Leak-safe design/blind driver for the compact C32 set ranker."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.cachetrajectory_c32 import (  # noqa: E402
    build_candidate_ids,
    build_features,
    request_macro_metrics,
    split_masks,
    target_membership,
)
from harp_rtt.cachetrajectory_c32_setrank import (  # noqa: E402
    EquivariantCacheTrajectory32,
    continuous_policy_score,
    parameter_and_mac_contract,
    ranking_loss,
    select_mean_constrained_policies,
)


SCHEMA = "cachetrajectory_c32_setrank_v1"
AUDIT_SHA256 = "587ed9df000fc57117a1f393311b6001988c758f11ae5f1413e0be776e6bc3de"
HISTORY_SHA256 = "c69686f93187d9b87d9ee29d24dc16951c097f30c0488ee56c5305227e497e21"
REFERENCE_SHA256 = "9fe2ee545a51a7605f5d3873602464c8b4c6b126aaddbef5b50f092651e38b5c"
FROZEN_C16_SHA256 = "341373b64a5eca88c5b790cfb096b25de9b635dc07dbddd5edc77d52893e8ced"
ACTIVE_PARAMETERS = 84_714
ACTIVE_BYTES = 169_428
ACTIVE_MACS = 79_872
CALIBRATION_FACTUAL_GATE = 0.88
CALIBRATION_CACHE_GATE = 0.95
SEEDS = (701, 702)
RESIDUAL_GRID = (0.0, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0)
SURVIVOR_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
CURRENT_BIAS_GRID = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("design", "blind"), default="design")
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(
            "/workspace/LLM_prefetch_study/artifacts/harp_rtt/"
            "resident_shadow_v2_cacheset_dev_20260825/evaluations/"
            "v2_screen_512_score_audit_v2/cache_set_audit.pt"
        ),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("/tmp/cacheset_audit_exact_history_v2.pt"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/tmp/cacheset_cachetrajectory16_dataset_v3.pt"),
    )
    parser.add_argument(
        "--active-result",
        type=Path,
        default=Path("/workspace/cachetrajectory_c32_joint_v1/STAGE_RESULT.json"),
    )
    parser.add_argument(
        "--frozen-c16",
        type=Path,
        default=Path("/tmp/cachetrajectory16_final_frozen.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stable_indices(values: Tensor, dense_rank: Tensor, count: int = 8) -> Tensor:
    rank_order = dense_rank.long().argsort(
        dim=-1, descending=False, stable=True
    )
    ordered_values = values.float().gather(-1, rank_order)
    value_order = ordered_values.argsort(
        dim=-1, descending=True, stable=True
    )[..., :count]
    return rank_order.gather(-1, value_order)


def select_from_score(
    candidate_ids: Tensor, score: Tensor, dense_rank: Tensor
) -> Tensor:
    return candidate_ids.long().gather(
        -1, stable_indices(score, dense_rank)
    )


def metrics_for_mask(
    selected: Tensor,
    data: Mapping[str, Any],
    mask: Tensor,
) -> dict[str, Any]:
    return request_macro_metrics(
        selected,
        data["current_ids"],
        data["target_ids"],
        data["valid"],
        data["request_index"],
        mask,
    )


def _request_counts(request_index: Tensor, masks: Mapping[str, Tensor]) -> dict[str, int]:
    return {
        name: int(request_index[mask].unique().numel())
        for name, mask in masks.items()
        if name in {"training", "calibration", "design", "blind"}
    }


def _validate_lineage(args: argparse.Namespace) -> dict[str, Any]:
    input_hashes = {
        "audit": sha256_file(args.audit),
        "history": sha256_file(args.history),
        "reference": sha256_file(args.reference),
        "active_result": sha256_file(args.active_result),
        "frozen_c16": sha256_file(args.frozen_c16),
    }
    expected = {
        "audit": AUDIT_SHA256,
        "history": HISTORY_SHA256,
        "reference": REFERENCE_SHA256,
        "frozen_c16": FROZEN_C16_SHA256,
    }
    for name, digest in expected.items():
        if input_hashes[name] != digest:
            raise ValueError(f"{name} SHA changed")

    active = json.loads(args.active_result.read_text(encoding="utf-8"))
    active_contract = active.get("contract", {})
    required_active = {
        "parameters": ACTIVE_PARAMETERS,
        "deployment_bf16_bytes": ACTIVE_BYTES,
        "linear_macs_per_cell": ACTIVE_MACS,
    }
    for name, value in required_active.items():
        if int(active_contract.get(name, -1)) != value:
            raise ValueError(f"active C32 {name} contract changed")
    active_inputs = set(active.get("input_sha256", {}).values())
    if not {AUDIT_SHA256, HISTORY_SHA256, REFERENCE_SHA256} <= active_inputs:
        raise ValueError("active C32 input lineage changed")

    frozen_c16 = torch.load(
        args.frozen_c16, map_location="cpu", weights_only=False
    )
    architecture = frozen_c16.get("architecture", {})
    lineage = frozen_c16.get("lineage", {})
    if lineage.get("score_audit_sha256") != AUDIT_SHA256:
        raise ValueError("frozen C16 score-audit lineage changed")
    if architecture.get("candidate_policy") != (
        "unique(current_top8_then_shadowroute_dense_rank_until_width16)"
    ):
        raise ValueError("frozen C16 candidate policy changed")
    return input_hashes


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    input_hashes = _validate_lineage(args)
    audit = torch.load(args.audit, map_location="cpu", weights_only=False)
    history = torch.load(args.history, map_location="cpu", weights_only=False)
    reference = torch.load(
        args.reference, map_location="cpu", weights_only=False
    )
    scores = audit["scores"].float()
    current_ids = audit["current_ids"].long()
    target_ids = audit["target_ids"].long()
    valid = audit["valid"].bool()
    request_index = reference["request_index"].long()
    if len(audit["request_ids"]) != len(request_index):
        raise AssertionError("audit/reference row counts differ")
    for row, request in enumerate(audit["request_ids"]):
        if request != reference["request_ids"][int(request_index[row])]:
            raise AssertionError(f"request alignment failed at row {row}")
    if not torch.equal(current_ids, reference["current_ids"].long()):
        raise AssertionError("current IDs differ from retained reference")
    if not torch.equal(target_ids, reference["target_ids"].long()):
        raise AssertionError("target IDs differ from retained reference")
    if not torch.equal(target_ids, history["future_ids"].long()):
        raise AssertionError("history future IDs differ from audit targets")
    if not torch.equal(valid, reference["valid"].bool()):
        raise AssertionError("valid masks differ from retained reference")
    if not torch.equal(
        history["history_ids"][:, 0].long().sort(-1).values,
        current_ids.sort(-1).values,
    ):
        raise AssertionError("history lag0 does not set-equal current routes")

    candidates = build_candidate_ids(scores, current_ids, width=args.width)
    features, dense_rank, is_current = build_features(
        scores,
        candidates,
        current_ids,
        history["history_ids"],
        history["history_weights"],
    )
    target_all = target_membership(candidates, target_ids)
    ref_candidates = reference["candidate_ids"].long()
    ref_continuous = reference["continuous"]
    if args.width < 16:
        raise ValueError("the frozen C16 reconstruction requires width>=16")
    if not torch.equal(candidates[..., :16], ref_candidates):
        raise AssertionError("first sixteen candidates do not reconstruct C16")
    if not torch.equal(dense_rank[..., :16], reference["dense_rank"].long()):
        raise AssertionError("first sixteen dense ranks differ")
    if not torch.equal(is_current[..., :16], reference["is_current"].bool()):
        raise AssertionError("first sixteen current flags differ")
    if not torch.equal(
        target_all[..., :16], reference["target_membership"].bool()
    ):
        raise AssertionError("first sixteen target labels differ")
    if not torch.equal(features[..., :16, 2].half(), ref_continuous[..., 4]):
        raise AssertionError("normalized dense-rank feature differs")
    if not torch.equal(features[..., :16, 3].half(), ref_continuous[..., 5]):
        raise AssertionError("current-membership feature differs")
    for lag in range(8):
        for column in range(3):
            new_column = 4 + 3 * lag + column
            old_column = 43 + 6 * lag + 3 + column
            if not torch.equal(
                features[..., :16, new_column].half(),
                ref_continuous[..., old_column],
            ):
                raise AssertionError(
                    f"history feature differs at lag {lag} column {column}"
                )

    masks = split_masks(request_index)
    counts = _request_counts(request_index, masks)
    if counts != {
        "training": 12,
        "calibration": 4,
        "design": 16,
        "blind": 16,
    }:
        raise AssertionError(f"request split changed: {counts}")
    if bool((masks["training"] & masks["calibration"]).any()):
        raise AssertionError("training/calibration overlap")
    if bool((masks["design"] & masks["blind"]).any()):
        raise AssertionError("design/blind overlap")
    if not torch.equal(
        masks["design"], masks["training"] | masks["calibration"]
    ):
        raise AssertionError("design split is not train union calibration")
    target = target_all
    if args.mode == "design":
        # Blind labels are touched above only for immutable audit/reference
        # alignment. They are then masked out before every objective, oracle,
        # model inference, policy choice, and reported design metric.
        target = target_all & masks["design"][:, None, None, None]

    base_score = features[..., 1].float()
    deployed = scores.argsort(
        dim=-1, descending=True, stable=True
    )[..., :8]
    base_selected = select_from_score(candidates, base_score, dense_rank)
    if not torch.equal(
        deployed.sort(-1).values, base_selected.sort(-1).values
    ):
        raise AssertionError("candidate baseline does not reproduce dense top8")
    design_model = EquivariantCacheTrajectory32(
        candidate_width=args.width
    )
    if not bool(
        design_model.factual_residual_head.weight.eq(0).all()
        and design_model.factual_residual_head.bias.eq(0).all()
    ):
        raise AssertionError("fresh factual residual is not exactly zero")
    design_rows = masks["design"]
    zero_score = continuous_policy_score(
        base_score[design_rows],
        torch.zeros_like(base_score[design_rows]),
        torch.zeros_like(base_score[design_rows]),
        is_current[design_rows],
        residual_scale=1.0,
        survivor_scale=0.0,
        current_bias=0.0,
    )
    zero_selected = select_from_score(
        candidates[design_rows], zero_score, dense_rank[design_rows]
    )
    if not torch.equal(
        zero_selected.sort(-1).values,
        deployed[design_rows].sort(-1).values,
    ):
        raise AssertionError("zero residual does not preserve deployed top8")

    oracle = select_from_score(
        candidates,
        base_score + target.float() * 1_000_000.0,
        dense_rank,
    )
    oracle_design = metrics_for_mask(oracle, {
        "current_ids": current_ids,
        "target_ids": target_ids,
        "valid": valid,
        "request_index": request_index,
    }, design_rows)
    if (
        oracle_design["factual_mean"] < 0.90
        or oracle_design["cache_set_mean"] != 1.0
    ):
        raise AssertionError("design C32 oracle gate failed")

    contract = parameter_and_mac_contract(args.width)
    if not (
        int(contract["parameters"]) < ACTIVE_PARAMETERS
        and int(contract["deployment_bf16_bytes"]) < ACTIVE_BYTES
        and int(contract["linear_macs_per_cell"]) < ACTIVE_MACS
    ):
        raise AssertionError("replacement does not strictly improve active contract")
    return {
        "scores": scores,
        "current_ids": current_ids,
        "target_ids": target_ids,
        "valid": valid,
        "request_index": request_index,
        "candidates": candidates,
        "features": features,
        "dense_rank": dense_rank,
        "is_current": is_current,
        "target": target,
        "base_score": base_score,
        "deployed": deployed,
        "oracle": oracle,
        "oracle_design": oracle_design,
        "masks": masks,
        "request_counts": counts,
        "contract": contract,
        "input_hashes": input_hashes,
    }


def flatten_data(data: Mapping[str, Any]) -> dict[str, Tensor]:
    features = data["features"]
    rows, horizons, layers, width, feature_width = features.shape
    return {
        "features": features.reshape(-1, width, feature_width),
        "candidates": data["candidates"].reshape(-1, width),
        "target": data["target"].reshape(-1, width),
        "is_current": data["is_current"].reshape(-1, width),
        "base": data["base_score"].reshape(-1, width),
        "valid": data["valid"].reshape(-1),
        "horizons": torch.arange(horizons)
        .view(1, horizons, 1)
        .expand(rows, horizons, layers)
        .reshape(-1),
        "layers": torch.arange(layers)
        .view(1, 1, layers)
        .expand(rows, horizons, layers)
        .reshape(-1),
    }


def cell_indices(data: Mapping[str, Any], row_mask: Tensor) -> Tensor:
    cells = (
        row_mask[:, None, None].expand_as(data["valid"]).reshape(-1)
        & data["valid"].reshape(-1)
    )
    return cells.nonzero(as_tuple=False).flatten()


def forward_cells(
    model: EquivariantCacheTrajectory32,
    flat: Mapping[str, Tensor],
    indices: Tensor,
) -> tuple[Tensor, Tensor]:
    return model(
        flat["features"].index_select(0, indices),
        flat["candidates"].index_select(0, indices),
        flat["horizons"].index_select(0, indices),
        flat["layers"].index_select(0, indices),
    )


@torch.no_grad()
def mean_ranking_loss(
    model: EquivariantCacheTrajectory32,
    flat: Mapping[str, Tensor],
    indices: Tensor,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    seen = 0
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        factual, survivor = forward_cells(model, flat, chosen)
        loss, _pieces = ranking_loss(
            factual,
            survivor,
            flat["target"].index_select(0, chosen),
            flat["is_current"].index_select(0, chosen),
            flat["base"].index_select(0, chosen),
        )
        total += float(loss) * len(chosen)
        seen += len(chosen)
    return total / max(seen, 1)


@torch.no_grad()
def infer_rows(
    model: EquivariantCacheTrajectory32,
    data: Mapping[str, Any],
    row_mask: Tensor,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    model.eval()
    row_ids = row_mask.nonzero(as_tuple=False).flatten()
    features = data["features"].index_select(0, row_ids)
    candidates = data["candidates"].index_select(0, row_ids)
    rows, horizons, layers, width, feature_width = features.shape
    flat_features = features.reshape(-1, width, feature_width)
    flat_candidates = candidates.reshape(-1, width)
    flat_horizons = torch.arange(horizons).view(1, horizons, 1).expand(
        rows, horizons, layers
    ).reshape(-1)
    flat_layers = torch.arange(layers).view(1, 1, layers).expand(
        rows, horizons, layers
    ).reshape(-1)
    factual_parts: list[Tensor] = []
    survivor_parts: list[Tensor] = []
    for start in range(0, len(flat_features), batch_size):
        stop = min(len(flat_features), start + batch_size)
        factual, survivor = model(
            flat_features[start:stop],
            flat_candidates[start:stop],
            flat_horizons[start:stop],
            flat_layers[start:stop],
        )
        factual_parts.append(factual)
        survivor_parts.append(survivor)
    return (
        torch.cat(factual_parts).reshape(rows, horizons, layers, width),
        torch.cat(survivor_parts).reshape(rows, horizons, layers, width),
    )


def horizon_metrics(
    selected: Tensor,
    current: Tensor,
    target: Tensor,
    valid: Tensor,
    request_index: Tensor,
    horizon: int,
) -> dict[str, float]:
    factual = (
        selected.unsqueeze(-1) == target[:, horizon].unsqueeze(-2)
    ).any(-1).sum(-1).float() / 8.0
    target_current = (
        target[:, horizon].unsqueeze(-1)
        == current.unsqueeze(-2)
    ).any(-1)
    target_selected = (
        target[:, horizon].unsqueeze(-1)
        == selected.unsqueeze(-2)
    ).any(-1)
    factual_requests: list[Tensor] = []
    cache_requests: list[Tensor] = []
    for request in request_index.unique(sorted=True):
        cells = request_index.eq(request)[:, None] & valid[:, horizon]
        factual_requests.append(factual[cells].mean())
        cache_requests.append(
            (
                target_current
                & target_selected
                & cells.unsqueeze(-1)
            ).sum().float()
            / (
                target_current & cells.unsqueeze(-1)
            ).sum().clamp_min(1)
        )
    return {
        "factual": float(torch.stack(factual_requests).mean()),
        "cache_set": float(torch.stack(cache_requests).mean()),
    }


def calibrate_policy(
    data: Mapping[str, Any],
    row_mask: Tensor,
    factual: Tensor,
    survivor: Tensor,
) -> dict[str, Any]:
    candidates = data["candidates"][row_mask]
    base = data["base_score"][row_mask]
    current_flag = data["is_current"][row_mask]
    dense_rank = data["dense_rank"][row_mask]
    current_ids = data["current_ids"][row_mask]
    target_ids = data["target_ids"][row_mask]
    valid = data["valid"][row_mask]
    request_index = data["request_index"][row_mask]
    options_by_horizon: list[list[Mapping[str, Any]]] = []
    for horizon in range(4):
        options: list[Mapping[str, Any]] = []
        for residual_scale in RESIDUAL_GRID:
            for survivor_scale in SURVIVOR_GRID:
                for current_bias in CURRENT_BIAS_GRID:
                    score = continuous_policy_score(
                        base[:, horizon],
                        factual[:, horizon],
                        survivor[:, horizon],
                        current_flag[:, horizon],
                        residual_scale=residual_scale,
                        survivor_scale=survivor_scale,
                        current_bias=current_bias,
                    )
                    selected = select_from_score(
                        candidates[:, horizon],
                        score,
                        dense_rank[:, horizon],
                    )
                    metrics = horizon_metrics(
                        selected,
                        current_ids,
                        target_ids,
                        valid,
                        request_index,
                        horizon,
                    )
                    options.append(
                        {
                            "residual_scale": residual_scale,
                            "survivor_scale": survivor_scale,
                            "current_bias": current_bias,
                            **metrics,
                        }
                    )
        options_by_horizon.append(options)
    selected = select_mean_constrained_policies(
        options_by_horizon,
        cache_target=CALIBRATION_CACHE_GATE,
    )
    return {
        **selected,
        "grid_size_per_horizon": (
            len(RESIDUAL_GRID)
            * len(SURVIVOR_GRID)
            * len(CURRENT_BIAS_GRID)
        ),
    }


def apply_policy(
    data: Mapping[str, Any],
    row_mask: Tensor,
    factual: Tensor,
    survivor: Tensor,
    policies: list[Mapping[str, float]],
) -> Tensor:
    candidates = data["candidates"][row_mask]
    base = data["base_score"][row_mask]
    current_flag = data["is_current"][row_mask]
    dense_rank = data["dense_rank"][row_mask]
    selected = []
    for horizon, policy in enumerate(policies):
        score = continuous_policy_score(
            base[:, horizon],
            factual[:, horizon],
            survivor[:, horizon],
            current_flag[:, horizon],
            residual_scale=float(policy["residual_scale"]),
            survivor_scale=float(policy["survivor_scale"]),
            current_bias=float(policy["current_bias"]),
        )
        selected.append(
            select_from_score(
                candidates[:, horizon], score, dense_rank[:, horizon]
            )
        )
    return torch.stack(selected, dim=1)


def subset_metrics(
    selected: Tensor,
    data: Mapping[str, Any],
    row_mask: Tensor,
) -> dict[str, Any]:
    rows = row_mask.nonzero(as_tuple=False).flatten()
    local_mask = torch.ones(len(rows), dtype=torch.bool)
    return request_macro_metrics(
        selected,
        data["current_ids"].index_select(0, rows),
        data["target_ids"].index_select(0, rows),
        data["valid"].index_select(0, rows),
        data["request_index"].index_select(0, rows),
        local_mask,
    )


def source_hashes() -> dict[str, str]:
    driver = Path(__file__).resolve()
    module = REPO_ROOT / "harp_rtt" / "cachetrajectory_c32_setrank.py"
    feature_module = REPO_ROOT / "harp_rtt" / "cachetrajectory_c32.py"
    return {
        "driver": sha256_file(driver),
        "setrank_module": sha256_file(module),
        "feature_module": sha256_file(feature_module),
    }


def deployment_state(
    state: Mapping[str, Tensor],
    expected_bytes: int,
) -> tuple[dict[str, Tensor], int]:
    converted: dict[str, Tensor] = {}
    logical_bytes = 0
    for name, value in state.items():
        tensor = value.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.to(torch.bfloat16)
        else:
            tensor = tensor.clone()
        converted[name] = tensor
        logical_bytes += tensor.numel() * tensor.element_size()
    if logical_bytes != int(expected_bytes):
        raise AssertionError(
            f"BF16 logical bytes {logical_bytes} != {expected_bytes}"
        )
    return converted, logical_bytes


def metric_mean_delta(
    primary: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "factual_mean": (
            float(primary["factual_mean"]) - float(reference["factual_mean"])
        ),
        "cache_set_mean": (
            float(primary["cache_set_mean"])
            - float(reference["cache_set_mean"])
        ),
    }


def diagnostic_metadata() -> dict[str, Any]:
    return {
        "diagnostic_unpromoted_bundle": True,
        "promotion_eligible": False,
        "old_holdout_reused": True,
        "diagnostic_reason": (
            "retained score audit was produced by the pre-hard-invariant "
            "allocation that misses 14 mandatory frequency-top64 cells"
        ),
    }


def design(args: argparse.Namespace, data: Mapping[str, Any]) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    flat = flatten_data(data)
    train_cells = cell_indices(data, data["masks"]["training"])
    calibration_cells = cell_indices(data, data["masks"]["calibration"])
    seed_results: list[dict[str, Any]] = []
    states: dict[int, dict[str, Tensor]] = {}
    fp32_states: dict[int, dict[str, Tensor]] = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = EquivariantCacheTrajectory32(candidate_width=args.width)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        best_loss = mean_ranking_loss(
            model, flat, calibration_cells, args.batch_size
        )
        best_epoch = 0
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        stale = 0
        epochs: list[dict[str, float | int]] = []
        for epoch in range(1, args.epochs + 1):
            started = time.monotonic()
            generator = torch.Generator().manual_seed(seed * 1000 + epoch)
            order = train_cells[
                torch.randperm(len(train_cells), generator=generator)
            ]
            model.train()
            train_total = 0.0
            seen = 0
            for start in range(0, len(order), args.batch_size):
                chosen = order[start : start + args.batch_size]
                factual, survivor = forward_cells(model, flat, chosen)
                loss, _pieces = ranking_loss(
                    factual,
                    survivor,
                    flat["target"].index_select(0, chosen),
                    flat["is_current"].index_select(0, chosen),
                    flat["base"].index_select(0, chosen),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_total += float(loss.detach()) * len(chosen)
                seen += len(chosen)
            calibration_loss = mean_ranking_loss(
                model, flat, calibration_cells, args.batch_size
            )
            record = {
                "epoch": epoch,
                "train_loss": train_total / max(seen, 1),
                "calibration_loss": calibration_loss,
                "seconds": time.monotonic() - started,
            }
            epochs.append(record)
            print(
                json.dumps({"event": "epoch", "seed": seed, **record}),
                flush=True,
            )
            if calibration_loss < best_loss - 1.0e-4:
                best_loss = calibration_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break

        model.load_state_dict(best_state)
        cal_factual_fp32, cal_survivor_fp32 = infer_rows(
            model,
            data,
            data["masks"]["calibration"],
            args.batch_size,
        )
        deployed_state, logical_bytes = deployment_state(
            best_state,
            int(data["contract"]["deployment_bf16_bytes"]),
        )
        deploy_model = EquivariantCacheTrajectory32(
            candidate_width=args.width
        ).to(torch.bfloat16)
        deploy_model.load_state_dict(deployed_state)
        cal_factual, cal_survivor = infer_rows(
            deploy_model,
            data,
            data["masks"]["calibration"],
            args.batch_size,
        )
        calibrated = calibrate_policy(
            data,
            data["masks"]["calibration"],
            cal_factual,
            cal_survivor,
        )
        cal_selected = apply_policy(
            data,
            data["masks"]["calibration"],
            cal_factual,
            cal_survivor,
            calibrated["policies_by_horizon"],
        )
        calibration_metrics = subset_metrics(
            cal_selected, data, data["masks"]["calibration"]
        )
        cal_selected_fp32 = apply_policy(
            data,
            data["masks"]["calibration"],
            cal_factual_fp32,
            cal_survivor_fp32,
            calibrated["policies_by_horizon"],
        )
        calibration_metrics_fp32 = subset_metrics(
            cal_selected_fp32,
            data,
            data["masks"]["calibration"],
        )
        if abs(
            float(calibrated["factual_mean"])
            - float(calibration_metrics["factual_mean"])
        ) > 1.0e-7 or abs(
            float(calibrated["cache_set_mean"])
            - float(calibration_metrics["cache_set_mean"])
        ) > 1.0e-7:
            raise AssertionError("joint calibration metric reconstruction drift")
        seed_results.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_calibration_loss_fp32": best_loss,
                "epochs": epochs,
                "calibration": calibrated,
                "calibration_metrics": calibration_metrics,
                "calibration_metrics_float32_same_policy": (
                    calibration_metrics_fp32
                ),
                "bfloat16_minus_float32_mean_delta": metric_mean_delta(
                    calibration_metrics,
                    calibration_metrics_fp32,
                ),
                "deployment_dtype": "torch.bfloat16",
                "deployment_logical_parameter_bytes": logical_bytes,
            }
        )
        states[seed] = deployed_state
        fp32_states[seed] = best_state

    seed_results.sort(
        key=lambda row: (
            row["calibration_metrics"]["factual_mean"],
            row["calibration_metrics"]["cache_set_mean"],
            -int(row["seed"]),
        ),
        reverse=True,
    )
    frozen = seed_results[0]
    seed = int(frozen["seed"])
    policies = frozen["calibration"]["policies_by_horizon"]
    model = EquivariantCacheTrajectory32(
        candidate_width=args.width
    ).to(torch.bfloat16)
    model.load_state_dict(states[seed])
    model_fp32 = EquivariantCacheTrajectory32(candidate_width=args.width)
    model_fp32.load_state_dict(fp32_states[seed])
    frozen_metrics: dict[str, Any] = {}
    frozen_metrics_fp32: dict[str, Any] = {}
    frozen_metric_delta: dict[str, Any] = {}
    for name in ("training", "calibration", "design"):
        mask = data["masks"][name]
        factual, survivor = infer_rows(
            model, data, mask, args.batch_size
        )
        selected = apply_policy(
            data, mask, factual, survivor, policies
        )
        frozen_metrics[name] = subset_metrics(selected, data, mask)
        factual_fp32, survivor_fp32 = infer_rows(
            model_fp32, data, mask, args.batch_size
        )
        selected_fp32 = apply_policy(
            data, mask, factual_fp32, survivor_fp32, policies
        )
        frozen_metrics_fp32[name] = subset_metrics(
            selected_fp32, data, mask
        )
        frozen_metric_delta[name] = metric_mean_delta(
            frozen_metrics[name],
            frozen_metrics_fp32[name],
        )

    calibration_metrics = frozen_metrics["calibration"]
    promotion_gate = {
        "purpose": "permission to open reused diagnostic blind only",
        "deployment_promotion_gate": False,
        "required_factual_mean": CALIBRATION_FACTUAL_GATE,
        "required_cache_set_mean": CALIBRATION_CACHE_GATE,
        "observed_factual_mean": calibration_metrics["factual_mean"],
        "observed_cache_set_mean": calibration_metrics["cache_set_mean"],
        "passed": (
            calibration_metrics["factual_mean"] >= CALIBRATION_FACTUAL_GATE
            and calibration_metrics["cache_set_mean"] >= CALIBRATION_CACHE_GATE
        ),
    }
    preoptimizer: dict[str, Any] = {}
    for name in ("training", "calibration", "design"):
        mask = data["masks"][name]
        preoptimizer[name] = {
            "dense_baseline": metrics_for_mask(
                data["deployed"], data, mask
            ),
            "c32_oracle": metrics_for_mask(data["oracle"], data, mask),
        }
    logical_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in states[seed].values()
    )
    if logical_bytes != int(data["contract"]["deployment_bf16_bytes"]):
        raise AssertionError("frozen BF16 logical byte count drift")
    if any(
        tensor.is_floating_point() and tensor.dtype != torch.bfloat16
        for tensor in states[seed].values()
    ):
        raise AssertionError("frozen state contains a non-BF16 float tensor")

    sources = source_hashes()
    diagnostic = diagnostic_metadata()
    checkpoint = {
        "schema": SCHEMA,
        "seed": seed,
        "candidate_width": args.width,
        "contract": data["contract"],
        "deployment_dtype": "torch.bfloat16",
        "logical_parameter_bytes": logical_bytes,
        "state_dict": states[seed],
        "input_hashes": data["input_hashes"],
        "source_hashes": sources,
        **diagnostic,
    }
    torch.save(checkpoint, args.output / "FROZEN_MODEL.pt")
    design_result = {
        "schema": SCHEMA,
        "mode": "design",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": data["contract"],
        "deployment_dtype": "torch.bfloat16",
        "logical_parameter_bytes": logical_bytes,
        "request_counts": data["request_counts"],
        "input_hashes": data["input_hashes"],
        "source_hashes": sources,
        "preoptimizer": preoptimizer,
        "seeds": seed_results,
        "frozen_seed": seed,
        "frozen_policy": policies,
        "frozen_joint_calibration": frozen["calibration"],
        "frozen_prediction_metrics": frozen_metrics,
        "frozen_prediction_metrics_float32_same_policy": frozen_metrics_fp32,
        "bfloat16_minus_float32_mean_delta": frozen_metric_delta,
        "promotion_gate": promotion_gate,
        "active_replacement_contract": {
            "active_parameters": ACTIVE_PARAMETERS,
            "active_deployment_bf16_bytes": ACTIVE_BYTES,
            "active_linear_macs_per_cell": ACTIVE_MACS,
            "replacement_parameters": data["contract"]["parameters"],
            "replacement_deployment_bf16_bytes": data["contract"][
                "deployment_bf16_bytes"
            ],
            "replacement_linear_macs_per_cell": data["contract"][
                "linear_macs_per_cell"
            ],
            "strict_parameter_improvement": True,
            "strict_byte_improvement": True,
            "strict_mac_improvement": True,
            "active_result_sha256": data["input_hashes"]["active_result"],
            "frozen_c16_sha256": data["input_hashes"]["frozen_c16"],
        },
        **diagnostic,
        "blind_labels_used_for_alignment_only": True,
        "blind_labels_in_optimizer_policy_or_design_metrics": False,
        "blind_model_predictions_opened": False,
        "new_capture": False,
        "gpu_used": False,
        "formal_validation_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "DESIGN_RESULT.json", design_result)
    policy_result = {
        "schema": SCHEMA,
        "candidate_width": args.width,
        "seed": seed,
        "deployment_dtype": "torch.bfloat16",
        "logical_parameter_bytes": logical_bytes,
        "policies_by_horizon": policies,
        "per_horizon_calibration_metrics": frozen["calibration"][
            "per_horizon_metrics"
        ],
        "calibration_factual_mean": frozen["calibration"]["factual_mean"],
        "calibration_cache_set_mean": frozen["calibration"]["cache_set_mean"],
        "promotion_gate": promotion_gate,
        "selection_basis": (
            "four calibration requests; exact joint maximum factual mean "
            "subject to H1-H4 mean CacheSet >= 0.95"
        ),
        "input_hashes": data["input_hashes"],
        "source_hashes": sources,
        "frozen_model_sha256": sha256_file(args.output / "FROZEN_MODEL.pt"),
        "design_result_sha256": sha256_file(args.output / "DESIGN_RESULT.json"),
        **diagnostic,
        "blind_model_predictions_opened": False,
    }
    write_json_exclusive(args.output / "FROZEN_POLICY.json", policy_result)
    print(json.dumps({"event": "design_complete", **policy_result}), flush=True)


def blind(args: argparse.Namespace, data: Mapping[str, Any]) -> None:
    if not args.output.is_dir():
        raise FileNotFoundError("blind mode requires the frozen design directory")
    for name in ("BLIND_RESULT.json", "STAGE_RESULT.json"):
        if (args.output / name).exists():
            raise FileExistsError(f"refusing to reopen {(args.output / name)}")
    model_path = args.output / "FROZEN_MODEL.pt"
    policy_path = args.output / "FROZEN_POLICY.json"
    design_path = args.output / "DESIGN_RESULT.json"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    design_result = json.loads(design_path.read_text(encoding="utf-8"))
    expected_diagnostic = diagnostic_metadata()
    for name, artifact in (
        ("checkpoint", checkpoint),
        ("policy", policy),
        ("design", design_result),
    ):
        for field, expected in expected_diagnostic.items():
            if artifact.get(field) != expected:
                raise ValueError(f"{name} diagnostic lineage field {field} changed")
    if design_result.get("promotion_gate", {}).get("passed") is not True:
        raise PermissionError("diagnostic calibration gate did not pass")
    if policy.get("promotion_gate") != design_result.get("promotion_gate"):
        raise ValueError("frozen policy diagnostic gate differs from design result")
    if policy.get("frozen_model_sha256") != sha256_file(model_path):
        raise ValueError("frozen model hash changed")
    if policy.get("design_result_sha256") != sha256_file(design_path):
        raise ValueError("design result hash changed")
    if checkpoint.get("input_hashes") != data["input_hashes"]:
        raise ValueError("checkpoint input hashes changed")
    if checkpoint.get("source_hashes") != source_hashes():
        raise ValueError("checkpoint source hashes changed")
    if policy.get("input_hashes") != data["input_hashes"]:
        raise ValueError("policy input hashes changed")
    if policy.get("source_hashes") != source_hashes():
        raise ValueError("policy source hashes changed")
    if checkpoint.get("contract") != data["contract"]:
        raise ValueError("deployment contract changed")
    if int(checkpoint.get("candidate_width", -1)) != args.width:
        raise ValueError("candidate width changed")
    if checkpoint.get("deployment_dtype") != "torch.bfloat16":
        raise ValueError("checkpoint deployment dtype changed")
    expected_bytes = int(data["contract"]["deployment_bf16_bytes"])
    if int(checkpoint.get("logical_parameter_bytes", -1)) != expected_bytes:
        raise ValueError("checkpoint logical parameter bytes changed")
    state = checkpoint["state_dict"]
    logical_bytes = sum(
        value.numel() * value.element_size() for value in state.values()
    )
    if logical_bytes != expected_bytes:
        raise ValueError("checkpoint state bytes differ from deployment contract")
    if any(
        value.is_floating_point() and value.dtype != torch.bfloat16
        for value in state.values()
    ):
        raise ValueError("checkpoint contains non-BF16 floating state")
    model = EquivariantCacheTrajectory32(
        candidate_width=args.width
    ).to(torch.bfloat16)
    model.load_state_dict(state)
    blind_mask = data["masks"]["blind"]
    factual, survivor = infer_rows(model, data, blind_mask, args.batch_size)
    selected = apply_policy(
        data,
        blind_mask,
        factual,
        survivor,
        policy["policies_by_horizon"],
    )
    metrics = subset_metrics(selected, data, blind_mask)
    result = {
        "schema": SCHEMA,
        "mode": "diagnostic_holdout",
        "execution_mode": "blind",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": data["contract"],
        "deployment_dtype": "torch.bfloat16",
        "logical_parameter_bytes": logical_bytes,
        "blind_requests": data["request_counts"]["blind"],
        "blind_metrics": metrics,
        "diagnostic_goal_met": (
            metrics["cache_set_mean"] >= 0.95
            and metrics["factual_mean"] >= 0.90
        ),
        "passed": False,
        "input_hashes": data["input_hashes"],
        "source_hashes": source_hashes(),
        "frozen_model_sha256": sha256_file(model_path),
        "frozen_policy_sha256": sha256_file(policy_path),
        "design_result_sha256": sha256_file(design_path),
        "blind_model_predictions_opened": True,
        "new_capture": False,
        "gpu_used": False,
        "formal_validation_opened": False,
        "sealed_test_opened": False,
        **expected_diagnostic,
    }
    write_json_exclusive(args.output / "BLIND_RESULT.json", result)
    stage = {
        **result,
        "blind_result_sha256": sha256_file(args.output / "BLIND_RESULT.json"),
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", stage)
    print(json.dumps({"event": "blind_complete", **stage}), flush=True)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.epochs > 12:
        raise ValueError("epochs must lie in [1,12]")
    if args.patience != 2:
        raise ValueError("this screen freezes patience at two")
    if tuple(args.seeds) != SEEDS:
        raise ValueError("this screen freezes seeds at 701/702")
    if args.width != 32:
        raise ValueError("this frozen screen requires candidate width 32")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    data = prepare(args)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "mode": args.mode,
                    "request_counts": data["request_counts"],
                    "contract": data["contract"],
                    "oracle_design": data["oracle_design"],
                    "zero_residual_dense_parity": True,
                    "input_hashes": data["input_hashes"],
                    "source_hashes": source_hashes(),
                    "blind_model_predictions_opened": False,
                    **diagnostic_metadata(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.mode == "design":
        design(args, data)
    else:
        blind(args, data)


if __name__ == "__main__":
    main()
