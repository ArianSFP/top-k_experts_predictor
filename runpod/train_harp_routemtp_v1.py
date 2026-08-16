#!/usr/bin/env python3
"""Train one immutable RouteMTP v1 stage on existing outer-train data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "runpod" / "transformers_mtp_bridge"
for value in (REPO_ROOT, BRIDGE):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from harp_rtt.anchor import LegacyHARPAnchorBridge  # noqa: E402
from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt  # noqa: E402
from harp_rtt.exact_k import exact_k_logz_with_marginals, stable_topk  # noqa: E402
from harp_rtt.node_counterfactual import (  # noqa: E402
    NodeCounterfactualDatasetAdapter,
    load_node_counterfactual_companion,
)
from harp_rtt.routemtp import (  # noqa: E402
    ANYTIME_BUDGETS,
    RecurrentRouteResidual,
    RouteMTPConfig,
    RouteMTPPredictor,
    install_cache_coherent_adapters,
)
from harp_rtt.routemtp_adapters import (  # noqa: E402
    SwitchableExpertLoRA,
    SwitchableMTPRouterResidual,
)
from harp_rtt.routemtp_batch import prepare_routemtp_batch  # noqa: E402
from harp_rtt.routemtp_cache import (  # noqa: E402
    ROUTEMTP_CACHE_SCHEMA,
    RouteMTPCacheGeometry,
    RouteMTPSourceOffset,
    load_causal_cache_slice,
    restore_transformers_dynamic_cache,
)
from harp_rtt.routemtp_loss import RouteMTPObjective  # noqa: E402
from harp_rtt.routemtp_replay import RouteMTPTreeRunner  # noqa: E402
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402
from harp_rtt.train import prepare_model_batch, runtime_static_artifacts  # noqa: E402
from harp_rtt.training import move_to_device, seed_everything, sha256_file  # noqa: E402
SCHEMA = "harp_routemtp_stage_training_v1"
RESULT_SCHEMA = "harp_routemtp_stage_result_v1"
STAGES = (
    "f0_captured", "f0_replay", "f1_replay", "f1_path",
    "r2_qo", "r2_path", "r2_joint", "r3_residual",
    "r3_expert", "r3_router",
)
PREDECESSOR = {
    "f0_captured": None,
    "f0_replay": None,
    "f1_replay": "f0_replay",
    "f1_path": "f1_replay",
    "r2_qo": "f1_replay",
    "r2_path": "r2_qo",
    "r2_joint": "r2_path",
    "r3_residual": "r2_joint",
    "r3_expert": "r3_residual",
    "r3_router": "r3_expert",
}
EFFECTIVE_BATCH = 32
EXPECTED_B2_REQUESTS = 256
EXPECTED_ROWS_PER_REQUEST = 16
OFFICIAL_TUNE_REQUESTS = 32


class IndexedSubset(Dataset[Any]):
    def __init__(self, base: Dataset[Any], indices: list[int]) -> None:
        self.base = base; self.indices = tuple(indices)
    def __len__(self) -> int: return len(self.indices)
    def __getitem__(self, index: int) -> Any: return self.base[self.indices[index]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--train-companion", type=Path, required=True)
    parser.add_argument("--hydration", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--target-preprocessing", type=Path, required=True)
    parser.add_argument("--mtp-preprocessing", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--parity-hydration", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--hydration-source-commit",
        help="immutable source commit recorded by the predecessor hydration",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--microbatch-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--internal-dev-requests", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def request_groups(base: HarpRTTDataset) -> dict[str, list[int]]:
    by_request: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(base.records):
        request = str(base.segments[record.segment].sequences[record.sequence]["request_id"])
        by_request[request].append(index)
    return dict(by_request)


def protected_training_split(
    dataset: Dataset[Any],
    base: HarpRTTDataset,
    dev_requests: int,
) -> tuple[Dataset[Any], Dataset[Any], dict[str, Any]]:
    """Freeze B2's official tune split, then split only its 224 train requests.

    Request identities are read from the base index rather than ``dataset`` so
    counterfactual labels belonging to the protected official tune requests are
    not opened while constructing the training subsets.
    """
    by_request = request_groups(base)
    if len(by_request) != EXPECTED_B2_REQUESTS or len(base) != (
        EXPECTED_B2_REQUESTS * EXPECTED_ROWS_PER_REQUEST
    ):
        raise ValueError("RouteMTP requires the frozen 256-request B2 inventory")
    if any(len(indices) != EXPECTED_ROWS_PER_REQUEST for indices in by_request.values()):
        raise ValueError("every RouteMTP B2 request must contribute exactly 16 rows")
    b2_order = sorted(
        by_request,
        key=lambda request: (
            hashlib.sha256(f"harp-rtt-b2-tune\0{request}".encode()).digest(),
            request,
        ),
    )
    official_tune = set(b2_order[:OFFICIAL_TUNE_REQUESTS])
    training_pool = set(b2_order[OFFICIAL_TUNE_REQUESTS:])
    if len(training_pool) <= dev_requests:
        raise ValueError("not enough fitting requests for RouteMTP internal development")
    ordered = sorted(
        training_pool,
        key=lambda value: (
            hashlib.sha256(("RouteMTP-internal-42:" + value).encode()).digest(),
            value,
        ),
    )
    development = set(ordered[:dev_requests]); fitting = training_pool - development
    fit_indices = [index for request in sorted(fitting) for index in by_request[request]]
    dev_indices = [index for request in sorted(development) for index in by_request[request]]
    if fitting & development or fitting & official_tune or development & official_tune:
        raise AssertionError("RouteMTP request partitions overlap")
    return IndexedSubset(dataset, fit_indices), IndexedSubset(dataset, dev_indices), {
        "schema": "harp_routemtp_protected_request_split_v2",
        "seed": 42,
        "official_tune_selection": "sha256(harp-rtt-b2-tune\\0request_id)",
        "official_tuning_requests": sorted(official_tune),
        "official_tuning_rows": sum(len(by_request[value]) for value in official_tune),
        "fit_requests": sorted(fitting),
        "internal_development_requests": sorted(development),
        "fit_rows": len(fit_indices),
        "internal_development_rows": len(dev_indices),
        "request_disjoint": True,
        "official_tune_opened": False,
        "diagnostic_development_opened": False,
    }


class HydrationStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        value = json.loads((root / "HYDRATION_MANIFEST.json").read_text())
        if value.get("schema") != ROUTEMTP_CACHE_SCHEMA:
            raise ValueError("RouteMTP hydration schema mismatch")
        if value.get("outer_split") != "train" or value.get("causal_slice_required") is not True:
            raise PermissionError("RouteMTP hydration is not causal outer-train")
        for key in ("formal_validation_opened", "calibration_opened", "sealed_test_opened"):
            if value.get(key) is not False:
                raise PermissionError(f"RouteMTP hydration violates {key}")
        self.value = value
        self.geometry = RouteMTPCacheGeometry(**value["geometry"]); self.geometry.validate()
        self.records = {str(row["request_id"]): row for row in value["records"]}
        self.offsets = {
            (str(row["request_id"]), int(row["source_position"])): RouteMTPSourceOffset(**row)
            for row in value["source_offsets"]
        }

    def batch_factories(
        self,
        metadata: Mapping[str, Any],
        *,
        mtp: nn.Module,
        device: torch.device,
    ) -> tuple[list[Any], Tensor, Tensor, list[Tensor], list[Tensor]]:
        request_ids = [str(value) for value in metadata["request_id"]]
        positions_value = metadata["position"]
        positions = positions_value.tolist() if isinstance(positions_value, Tensor) else list(positions_value)
        slices: list[dict[str, Tensor]] = []; offsets: list[RouteMTPSourceOffset] = []
        for request_id, position in zip(request_ids, positions, strict=True):
            offset = self.offsets[(request_id, int(position))]
            record = self.records[request_id]
            slices.append(load_causal_cache_slice(
                self.root / str(record["relative_path"]), self.geometry, offset,
                expected_sha256=str(record["sha256"]),
            ))
            offsets.append(offset)
        factories = [
            (lambda tensors=tensors: restore_transformers_dynamic_cache(
                tensors, self.geometry, config=mtp.decoder.config, device=device
            ))
            for tensors in slices
        ]
        lengths = torch.tensor([value.prefix_length for value in offsets], device=device)
        seeds = torch.stack([
            tensors["target_final_hidden"][-1] for tensors in slices
        ]).to(device=device)
        prefix_ids = [tensors["shifted_token_ids"] for tensors in slices]
        hidden_histories = [tensors["target_final_hidden"] for tensors in slices]
        return factories, lengths, seeds, prefix_ids, hidden_histories


def collated_loader(dataset: Dataset[Any], *, batch: int, shuffle: bool, seed: int, workers: int) -> DataLoader[Any]:
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=workers,
        collate_fn=collate_harp_rtt, persistent_workers=False, drop_last=False,
    )


def anchor_scores(
    host: Mapping[str, Any], *, anchor: LegacyHARPAnchorBridge,
    runtime_static: Any, device: torch.device,
) -> tuple[dict[str, Any], Tensor]:
    batch = move_to_device(host, device)
    prepared = prepare_model_batch(batch, runtime_static)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        scores = anchor(batch=prepared)["future_router_scores"][:, :4].float()
    return prepared, scores


def stage_uses_replay(stage: str) -> bool:
    return stage != "f0_captured"


def stage_uses_target_bank(stage: str) -> bool:
    return stage not in {"f0_captured", "f0_replay"}


def compact_state(
    predictor: RouteMTPPredictor,
    adapters: Any,
    residual: RecurrentRouteResidual | None,
    expert: SwitchableExpertLoRA | None,
    router: SwitchableMTPRouterResidual | None,
) -> dict[str, Any]:
    predictor_parameters = dict(predictor.named_parameters())
    result: dict[str, Any] = {
        # Router geometry is loaded from the hash-bound static artifact and is
        # deliberately not duplicated in the compact predictor checkpoint.
        "predictor": {
            name: value.detach().cpu() for name, value in predictor_parameters.items()
        },
        "qo": {
            "fusion_down": adapters.fusion.down.weight.detach().cpu(),
            "fusion_up": adapters.fusion.up.weight.detach().cpu(),
            "query_down": adapters.query.down.weight.detach().cpu(),
            "query_up": adapters.query.up.weight.detach().cpu(),
            "output_down": adapters.output.down.weight.detach().cpu(),
            "output_up": adapters.output.up.weight.detach().cpu(),
        },
    }
    if residual is not None:
        result["residual"] = {name: value.detach().cpu() for name, value in residual.state_dict().items()}
    if expert is not None:
        result["expert"] = {
            name: getattr(expert, name).detach().cpu() for name in
            ("gate_up_down", "gate_up_up", "down_down", "down_up")
        }
    if router is not None:
        result["router"] = {
            "down": router.down.weight.detach().cpu(),
            "up": router.up.weight.detach().cpu(),
        }
    return result


def load_compact_state(
    path: Path, *, expected_stage: str,
    predictor: RouteMTPPredictor, adapters: Any,
    residual: RecurrentRouteResidual | None,
    expert: SwitchableExpertLoRA | None,
    router: SwitchableMTPRouterResidual | None,
) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema") != SCHEMA or value.get("stage") != expected_stage:
        raise ValueError(f"RouteMTP initializer must be completed {expected_stage}")
    predictor_parameters = dict(predictor.named_parameters())
    stored_parameters = value["state"]["predictor"]
    if set(stored_parameters) != set(predictor_parameters):
        raise ValueError("RouteMTP predictor parameter namespace mismatch")
    with torch.no_grad():
        for name, tensor in stored_parameters.items():
            if tensor.shape != predictor_parameters[name].shape:
                raise ValueError(f"RouteMTP predictor parameter shape mismatch: {name}")
            predictor_parameters[name].copy_(tensor)
    qo = value["state"]["qo"]
    for module, prefix in ((adapters.fusion, "fusion"), (adapters.query, "query"), (adapters.output, "output")):
        module.down.weight.data.copy_(qo[prefix + "_down"])
        module.up.weight.data.copy_(qo[prefix + "_up"])
    if residual is not None and "residual" in value["state"]:
        residual.load_state_dict(value["state"]["residual"], strict=True)
    if expert is not None and "expert" in value["state"]:
        for name, tensor in value["state"]["expert"].items(): getattr(expert, name).data.copy_(tensor)
    if router is not None and "router" in value["state"]:
        router.down.weight.data.copy_(value["state"]["router"]["down"])
        router.up.weight.data.copy_(value["state"]["router"]["up"])
    return value


def configure_stage(
    stage: str, predictor: RouteMTPPredictor, adapters: Any,
    residual: RecurrentRouteResidual | None,
    expert: SwitchableExpertLoRA | None,
    router: SwitchableMTPRouterResidual | None,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[str]]:
    predictor.requires_grad_(False)
    for parameter in adapters.parameters(): parameter.requires_grad_(False)
    modules: list[nn.Module] = []; lora: list[nn.Parameter] = []
    nonlinear: list[nn.Parameter] = []
    if stage in {"f0_captured", "f0_replay", "f1_replay", "r2_qo", "r2_joint", "r3_residual", "r3_expert", "r3_router"}:
        modules.append(predictor.route_head)
    if stage in {"f1_path", "r2_path", "r2_joint", "r3_residual", "r3_expert", "r3_router"}:
        modules.append(predictor.path_head)
    if stage in {"r2_qo", "r2_joint", "r3_residual", "r3_expert", "r3_router"}:
        lora.extend(adapters.parameters())
    if residual is not None and stage in {"r3_residual", "r3_expert", "r3_router"}:
        modules.append(residual)
    if expert is not None and stage in {"r3_expert", "r3_router"}:
        nonlinear.extend((
            expert.gate_up_down, expert.gate_up_up,
            expert.down_down, expert.down_up,
        ))
    if router is not None and stage == "r3_router":
        nonlinear.extend(router.down.parameters()); nonlinear.extend(router.up.parameters())
    for module in modules: module.requires_grad_(True)
    for parameter in lora: parameter.requires_grad_(True)
    for parameter in nonlinear: parameter.requires_grad_(True)
    head = [parameter for module in modules for parameter in module.parameters() if parameter.requires_grad]
    head.extend(nonlinear)
    head_ids = {id(parameter) for parameter in head}
    lora = [parameter for parameter in lora if id(parameter) not in head_ids]
    names = [name for name, parameter in predictor.named_parameters() if parameter.requires_grad]
    for prefix, module in (
        ("mtp.fusion", adapters.fusion),
        ("mtp.query", adapters.query),
        ("mtp.output", adapters.output),
    ):
        names.extend(
            f"{prefix}.{name}" for name, parameter in module.named_parameters()
            if parameter.requires_grad
        )
    names.extend(["mtp." + name for name, parameter in (
        list((residual or nn.Identity()).named_parameters())
        + list((expert or nn.Identity()).named_parameters())
        + list((router or nn.Identity()).named_parameters())
    ) if parameter.requires_grad])
    return head, lora, sorted(set(names))


def deployed_parameter_bytes(
    predictor: RouteMTPPredictor,
    adapters: Any,
    residual: RecurrentRouteResidual | None,
    expert: SwitchableExpertLoRA | None,
    router: SwitchableMTPRouterResidual | None,
) -> int:
    """BF16 bytes for every added deployed parameter, not just this stage's trainables."""

    parameters: list[nn.Parameter] = list(predictor.parameters())
    parameters.extend(adapters.parameters())
    if residual is not None:
        parameters.extend(residual.parameters())
    if expert is not None:
        parameters.extend((
            expert.gate_up_down, expert.gate_up_up,
            expert.down_down, expert.down_up,
        ))
    if router is not None:
        parameters.extend(router.down.parameters())
        parameters.extend(router.up.parameters())
    unique = {id(parameter): parameter for parameter in parameters}
    return sum(parameter.numel() * 2 for parameter in unique.values())


def _bootstrap_interval(values: list[float], *, replicates: int = 1_000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    generator = random.Random(42)
    size = len(values)
    estimates = sorted(
        sum(values[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(replicates)
    )
    return [estimates[int(0.025 * replicates)], estimates[int(0.975 * replicates) - 1]]


def forward_one(
    *, stage: str, predictor: RouteMTPPredictor, runner: RouteMTPTreeRunner,
    objective: RouteMTPObjective, anchor: LegacyHARPAnchorBridge,
    runtime_static: Any, geometry: Any, token_embedding: Tensor,
    hydration: HydrationStore, host: Mapping[str, Any], device: torch.device,
    budget_index: int,
    execute_visible_only: bool = False,
) -> tuple[Any, Any, Any, Tensor]:
    prepared_host, base_scores = anchor_scores(
        host, anchor=anchor, runtime_static=runtime_static, device=device
    )
    prepared = prepare_routemtp_batch(
        prepared_host, geometry=geometry, token_embedding=token_embedding,
        config=predictor.config,
    )
    inputs = prepared.model_inputs
    visible = inputs["anytime_node_masks"][:, budget_index]
    execution_mask = visible if execute_visible_only else inputs["node_mask"]
    if stage_uses_replay(stage):
        factories, lengths, target_seeds, prefix_ids, hidden_histories = hydration.batch_factories(
            host["metadata"], mtp=runner.mtp, device=device
        )
        replay = runner(
            node_token_ids=inputs["node_token_ids"],
            parent_ids=inputs["node_parent_ids"],
            node_mask=execution_mask,
            current_target_hidden=target_seeds,
            base_cache_factory=lambda index: factories[index](),
            base_cache_lengths=lengths,
            base_prefix_token_ids=prefix_ids,
            base_target_hidden_history=hidden_histories,
        )
        channels = dict(
            fused_state=replay.fused_state,
            router_input=replay.router_input,
            post_moe_hidden=replay.post_moe_hidden,
            vocabulary_head_input=replay.vocabulary_head_input,
            mtp_router_logits=replay.router_logits,
            mtp_selected_ids=replay.selected_ids,
            mtp_selected_weights=replay.selected_weights,
            recurrent_state=replay.vocabulary_head_input,
        )
    else:
        channels = dict(
            fused_state=inputs["captured_fused_state"],
            router_input=inputs["captured_router_input"],
            post_moe_hidden=inputs["captured_post_moe_hidden"],
            vocabulary_head_input=inputs["captured_vocabulary_head_input"],
            mtp_router_logits=inputs["captured_mtp_router_logits"],
            mtp_selected_ids=inputs["captured_mtp_selected_ids"],
            mtp_selected_weights=inputs["captured_mtp_selected_weights"],
            recurrent_state=inputs["captured_vocabulary_head_input"],
        )
    _, anchor_marginals = exact_k_logz_with_marginals(base_scores, predictor.config.exact_k)
    output = predictor(
        **channels,
        node_token_embeddings=inputs["node_token_embeddings"],
        node_parent_ids=inputs["node_parent_ids"],
        node_depths=inputs["node_depths"],
        node_child_ranks=inputs["node_child_ranks"],
        node_mask=execution_mask,
        visible_mask=visible,
        native_edge_log_probabilities=inputs["native_edge_log_probabilities"],
        current_post_layer=inputs["current_post_layer"],
        current_queries=inputs["current_queries"],
        current_centered_router_logits=inputs["current_centered_router_logits"],
        current_selected_ids=inputs["current_selected_ids"],
        current_selected_weights=inputs["current_selected_weights"],
        anchor_marginals=anchor_marginals,
        use_target_bank=stage_uses_target_bank(stage),
        path_gradient_scale=0.1 if stage == "r2_joint" else 1.0,
    )
    factual_index = prepared.factual_branch_indices.clone()
    nodes = visible.shape[-1]
    for horizon in range(predictor.config.horizons):
        index = factual_index[:, horizon]
        inside = (index >= 0) & (index < nodes)
        safe = index.clamp(0, nodes - 1)
        shown = visible.gather(1, safe[:, None]).squeeze(1)
        factual_index[:, horizon] = torch.where(inside & shown, index, torch.full_like(index, nodes))
    loss = objective(
        route=output.route,
        path=output.path,
        node_depths=inputs["node_depths"],
        visible_mask=visible,
        branch_supervision_mask=execution_mask,
        native_path_log_probabilities=inputs["native_path_log_probabilities"],
        teacher_node_logits=prepared.teacher_node_logits,
        teacher_node_ids=prepared.teacher_node_ids,
        teacher_node_queries=prepared.teacher_node_queries,
        branch_valid=prepared.branch_valid,
        anchor_scores=base_scores,
        factual_ids=prepared.factual_ids,
        factual_valid=prepared.factual_valid,
        factual_branch_indices=factual_index,
    )
    if stage == "r3_router" and runner.router_adapter is not None:
        trust, balance = runner.router_adapter.collected_regularization()
        loss = replace(loss, total=loss.total + 0.01 * trust + 0.01 * balance)
    return output, loss, prepared, base_scores


@torch.no_grad()
def evaluate(
    *, stage: str, predictor: RouteMTPPredictor, runner: RouteMTPTreeRunner,
    objective: RouteMTPObjective, anchor: LegacyHARPAnchorBridge,
    runtime_static: Any, geometry: Any, token_embedding: Tensor,
    hydration: HydrationStore, dataset: Dataset[Any], device: torch.device,
    microbatch: int, workers: int,
    budgets_to_evaluate: tuple[int, ...] = ANYTIME_BUDGETS,
) -> dict[str, Any]:
    predictor.eval(); runner.eval(); anchor.eval(); budgets: dict[str, Any] = {}
    if runner.router_adapter is not None:
        runner.router_adapter.set_inference_mode()
    for budget in budgets_to_evaluate:
        if budget not in ANYTIME_BUDGETS:
            raise ValueError(f"undeclared anytime budget {budget}")
        budget_index = ANYTIME_BUDGETS.index(budget)
        branch_hits = branch_slots = factual_hits = factual_slots = candidate_hits = 0
        horizon_hits = [0] * 4; horizon_slots = [0] * 4
        candidate_horizon_hits = [0] * 4; loss_total = rows = 0
        executed_nodes = 0
        request_counts: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "branch_hits": 0, "branch_slots": 0,
                "horizon_hits": [0] * 4, "horizon_slots": [0] * 4,
                "candidate_horizon_hits": [0] * 4,
            }
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for host in collated_loader(dataset, batch=microbatch, shuffle=False, seed=0, workers=workers):
            output, loss, prepared, _ = forward_one(
                stage=stage, predictor=predictor, runner=runner, objective=objective,
                anchor=anchor, runtime_static=runtime_static, geometry=geometry,
                token_embedding=token_embedding, hydration=hydration, host=host,
                device=device, budget_index=budget_index,
                execute_visible_only=True,
            )
            visible = prepared.model_inputs["anytime_node_masks"][:, budget_index]
            executed_nodes += int(visible.sum())
            branch_valid = (
                prepared.branch_valid
                & visible[..., None]
                & (prepared.model_inputs["node_depths"] >= 2)[..., None]
            )
            branch_match = (
                output.route.selected_ids[..., :, None]
                == prepared.teacher_node_ids[..., None, :]
            ).any(-2)
            per_request_branch_hits = (branch_match & branch_valid[..., None]).sum((1, 2, 3))
            per_request_branch_slots = branch_valid.sum((1, 2)) * predictor.config.exact_k
            branch_hits += int(per_request_branch_hits.sum())
            branch_slots += int(per_request_branch_slots.sum())
            factual_match = (
                output.factual.selected_ids[..., :, None]
                == prepared.factual_ids[..., None, :]
            ).any(-2)
            valid = prepared.factual_valid
            factual_hits += int((factual_match & valid[..., None]).sum())
            factual_slots += int(valid.sum()) * predictor.config.exact_k
            candidate = stable_topk(output.factual.projected_marginals, 64)
            covered = (candidate[..., :, None] == prepared.factual_ids[..., None, :]).any(-2)
            candidate_hits += int((covered & valid[..., None]).sum())
            per_request_horizon_hits = (factual_match & valid[..., None]).sum((2, 3))
            per_request_horizon_slots = valid.sum(2) * predictor.config.exact_k
            per_request_candidate_hits = (covered & valid[..., None]).sum((2, 3))
            for horizon in range(4):
                horizon_hits[horizon] += int((factual_match[:, horizon] & valid[:, horizon, :, None]).sum())
                horizon_slots[horizon] += int(valid[:, horizon].sum()) * predictor.config.exact_k
                candidate_horizon_hits[horizon] += int(
                    (covered[:, horizon] & valid[:, horizon, :, None]).sum()
                )
            request_ids = [str(value) for value in host["metadata"]["request_id"]]
            for batch_index, request_id in enumerate(request_ids):
                counts = request_counts[request_id]
                counts["branch_hits"] += int(per_request_branch_hits[batch_index])
                counts["branch_slots"] += int(per_request_branch_slots[batch_index])
                for horizon in range(4):
                    counts["horizon_hits"][horizon] += int(
                        per_request_horizon_hits[batch_index, horizon]
                    )
                    counts["horizon_slots"][horizon] += int(
                        per_request_horizon_slots[batch_index, horizon]
                    )
                    counts["candidate_horizon_hits"][horizon] += int(
                        per_request_candidate_hits[batch_index, horizon]
                    )
            active = int(valid.sum()); loss_total += float(loss.total) * active; rows += active
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        request_rows: list[dict[str, Any]] = []
        for request_id, counts in sorted(request_counts.items()):
            h_hits = counts["horizon_hits"]; h_slots = counts["horizon_slots"]
            c_hits = counts["candidate_horizon_hits"]
            request_rows.append({
                "request_id": request_id,
                "branch_recall_at_8": counts["branch_hits"] / max(1, counts["branch_slots"]),
                "factual_h1_h4_recall_at_8": sum(h_hits) / max(1, sum(h_slots)),
                "factual_h2_h4_recall_at_8": sum(h_hits[1:]) / max(1, sum(h_slots[1:])),
                "factual_h4_recall_at_8": h_hits[3] / max(1, h_slots[3]),
                "factual_h2_h4_c64_coverage": sum(c_hits[1:]) / max(1, sum(h_slots[1:])),
            })
        macro_keys = (
            "branch_recall_at_8", "factual_h1_h4_recall_at_8",
            "factual_h2_h4_recall_at_8", "factual_h4_recall_at_8",
            "factual_h2_h4_c64_coverage",
        )
        request_macro = {
            key: sum(float(row[key]) for row in request_rows) / max(1, len(request_rows))
            for key in macro_keys
        }
        bootstrap = {
            key: _bootstrap_interval([float(row[key]) for row in request_rows])
            for key in macro_keys
        }
        budgets[str(budget)] = {
            "loss": loss_total / max(1, rows),
            "branch_recall_at_8": branch_hits / max(1, branch_slots),
            "factual_h1_h4_recall_at_8": factual_hits / max(1, factual_slots),
            "factual_h2_h4_recall_at_8": sum(horizon_hits[1:]) / max(1, sum(horizon_slots[1:])),
            "per_horizon_recall_at_8": [horizon_hits[h] / max(1, horizon_slots[h]) for h in range(4)],
            "factual_c64_coverage": candidate_hits / max(1, factual_slots),
            "factual_h2_h4_c64_coverage": sum(candidate_horizon_hits[1:]) / max(1, sum(horizon_slots[1:])),
            "per_horizon_c64_coverage": [
                candidate_horizon_hits[h] / max(1, horizon_slots[h]) for h in range(4)
            ],
            "request_macro": request_macro,
            "request_bootstrap_95": bootstrap,
            "request_rows": request_rows,
            "bootstrap_replicates": 1_000,
            "bootstrap_seed": 42,
            "wall_seconds": wall_seconds,
            "milliseconds_per_source_position": 1_000.0 * wall_seconds / max(1, len(dataset)),
            "executed_nodes": executed_nodes,
            "executed_nodes_per_source_position": executed_nodes / max(1, len(dataset)),
        }
    selection_budget = "16" if "16" in budgets else str(max(budgets_to_evaluate))
    return {"budgets": budgets, "selection_value": budgets[selection_budget]["branch_recall_at_8"]}


def main() -> None:
    args = parse_args(); stage = args.stage
    from qwen35_mtp import load_checkpoint_mtp
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(args.source_commit) != 40: raise ValueError("RouteMTP requires a full source commit")
    hydration_source_commit = args.hydration_source_commit or args.source_commit
    if len(hydration_source_commit) != 40:
        raise ValueError("RouteMTP hydration requires a full predecessor commit")
    parity = json.loads(args.parity_report.read_text())
    if parity.get("passed") is not True or parity.get("adapter_off") is not True:
        raise PermissionError("RouteMTP replay parity has not passed")
    if args.microbatch_size > EFFECTIVE_BATCH or EFFECTIVE_BATCH % args.microbatch_size:
        raise ValueError("microbatch must divide effective batch 32")
    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)

    base = HarpRTTDataset(args.train_index, "train", corpus_root=args.train_corpus, max_tree_nodes=32)
    labels, companion_manifest = load_node_counterfactual_companion(
        args.train_companion, split="train", training=True
    )
    joined = NodeCounterfactualDatasetAdapter(base, labels, split="train", training=True)
    fit, internal_dev, split = protected_training_split(
        joined, base, args.internal_dev_requests
    )
    hydration = HydrationStore(args.hydration)
    parity_hydration = HydrationStore(args.parity_hydration)
    hydration_manifest_sha256 = sha256_file(args.hydration / "HYDRATION_MANIFEST.json")
    parity_hydration_manifest_sha256 = sha256_file(
        args.parity_hydration / "HYDRATION_MANIFEST.json"
    )
    companion_manifest_sha256 = sha256_file(args.train_companion / "manifest.json")
    if parity.get("hydration_manifest_sha256") != parity_hydration_manifest_sha256:
        raise ValueError("RouteMTP parity report and Stage-A hydration differ")
    if parity_hydration.geometry != hydration.geometry:
        raise ValueError("RouteMTP Stage-A/full hydration geometry differs")
    if parity_hydration.value.get("bindings") != hydration.value.get("bindings"):
        raise ValueError("RouteMTP Stage-A/full hydration bindings differ")
    for key, offset in parity_hydration.offsets.items():
        if hydration.offsets.get(key) != offset:
            raise ValueError("RouteMTP Stage-A source offset is absent from full hydration")
    for request_id, record in parity_hydration.records.items():
        full_record = hydration.records.get(request_id)
        if full_record is None or full_record.get("sha256") != record.get("sha256"):
            raise ValueError("RouteMTP Stage-A request record differs in full hydration")
    if parity.get("counterfactual_companion_manifest_sha256") != companion_manifest_sha256:
        raise ValueError("RouteMTP parity report was produced from a different companion")
    bindings = hydration.value.get("bindings", {})
    if bindings.get("source_commit") != hydration_source_commit:
        raise ValueError("RouteMTP hydration source commit differs from training source")
    if bindings.get("counterfactual_companion_sha256") != companion_manifest_sha256:
        raise ValueError("RouteMTP hydration companion binding differs from training companion")
    missing_offsets = []
    for dataset in (fit, internal_dev):
        for index in range(len(dataset)):
            item = dataset[index]; key = (str(item["metadata"]["request_id"]), int(item["metadata"]["position"]))
            if key not in hydration.offsets: missing_offsets.append(key)
    if missing_offsets: raise KeyError(f"RouteMTP hydration misses {len(missing_offsets)} training rows")

    static = load_static_target_artifacts(args.static_dir, device="cpu")
    if static.token_embedding is None: raise RuntimeError("RouteMTP requires the frozen embedding")
    config = RouteMTPConfig(
        hidden_width=static.geometry.hidden_width,
        target_layers=static.geometry.layers,
        experts=static.geometry.experts,
        router_rank=static.geometry.maximum_rank,
    )
    geometry = static.geometry.to(device)
    predictor = RouteMTPPredictor(
        config, geometry.expert_keys, geometry.centered_bias, geometry.rank_mask
    ).to(device)
    mtp = load_checkpoint_mtp(args.model, device=device)
    adapters = install_cache_coherent_adapters(mtp, rank=config.lora_rank)
    residual = RecurrentRouteResidual(config.hidden_width, config.residual_width).to(device) if stage.startswith("r3_") else None
    expert = None; router = None
    if stage in {"r3_expert", "r3_router"}:
        expert = SwitchableExpertLoRA(mtp.decoder.layers[0].mlp.experts, rank=8).to(device)
        mtp.decoder.layers[0].mlp.experts = expert
    if stage == "r3_router":
        router = SwitchableMTPRouterResidual(
            mtp.decoder.layers[0].mlp.gate, config.hidden_width, config.experts,
            rank=16, top_k=config.exact_k,
        ).to(device)
        mtp.decoder.layers[0].mlp.gate = router
    runner = RouteMTPTreeRunner(
        mtp, adapters, recurrent_residual=residual,
        expert_adapter=expert, router_adapter=router,
    ).to(device)
    initializer = None
    expected = PREDECESSOR[stage]
    if expected is None:
        if args.initialize_from is not None: raise ValueError(f"{stage} does not accept an initializer")
    else:
        if args.initialize_from is None: raise ValueError(f"{stage} requires --initialize-from {expected}")
        initializer = load_compact_state(
            args.initialize_from, expected_stage=expected, predictor=predictor,
            adapters=adapters, residual=residual, expert=expert, router=router,
        )
        if stage == "r2_path":
            nn.init.zeros_(predictor.path_head.pre[-1].weight); nn.init.zeros_(predictor.path_head.pre[-1].bias)
            nn.init.zeros_(predictor.path_head.post[-1].weight); nn.init.zeros_(predictor.path_head.post[-1].bias)
            predictor.path_head.other.data.zero_()
    head_parameters, lora_parameters, trainable_names = configure_stage(
        stage, predictor, adapters, residual, expert, router
    )
    trainable_parameter_bytes = sum(
        parameter.numel() * 2 for parameter in head_parameters + lora_parameters
    )
    parameter_bytes = deployed_parameter_bytes(
        predictor, adapters, residual, expert, router
    )
    if parameter_bytes > 1 << 30:
        raise RuntimeError("RouteMTP deployed BF16 weights exceed 1 GiB")
    anchor, anchor_provenance = LegacyHARPAnchorBridge.from_artifacts(
        args.anchor_checkpoint, args.target_preprocessing, args.mtp_preprocessing,
        expected_checkpoint_sha256=args.anchor_sha256,
    )
    anchor = anchor.to(device).requires_grad_(False).eval()
    token_embedding = static.token_embedding.to(device)
    runtime_static = runtime_static_artifacts(static, device); del static
    objective = RouteMTPObjective(geometry.expert_keys, geometry.centered_bias, exact_k=config.exact_k).to(device)

    args.output.mkdir(parents=True)
    write_json_exclusive(args.output / "INTERNAL_SPLIT.json", split)
    manifest = {
        "schema": SCHEMA, "stage": stage, "seed": args.seed,
        "source_commit": args.source_commit,
        "hydration_source_commit": hydration_source_commit,
        "config": config.__dict__, "trainable_names": trainable_names,
        "trainable_bf16_bytes": trainable_parameter_bytes,
        "added_deployed_bf16_bytes": parameter_bytes,
        "hard_size_limit_bytes": 1 << 30,
        "effective_batch": EFFECTIVE_BATCH, "microbatch": args.microbatch_size,
        "optimizer_constructed": False, "training_started": False,
        "adapter_contract": "fusion_fc_plus_qo_kv_frozen",
        "native_mtp_generates_tokens": True, "route_mtp_generates_tokens": False,
        "official_tune_opened": False, "diagnostic_development_opened": False,
        "formal_validation_opened": False, "calibration_opened": False, "sealed_test_opened": False,
        "hydration_manifest_sha256": hydration_manifest_sha256,
        "parity_hydration_manifest_sha256": parity_hydration_manifest_sha256,
        "counterfactual_companion_manifest_sha256": companion_manifest_sha256,
        "parity_report_sha256": sha256_file(args.parity_report),
        "companion_sealed_test_opened": companion_manifest.get("sealed_test_opened"),
        "anchor": anchor_provenance,
        "initializer_sha256": sha256_file(args.initialize_from) if args.initialize_from else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_exclusive(args.output / "RUN_MANIFEST.json", manifest)
    if args.preflight_only:
        write_json_exclusive(args.output / "STAGE_RESULT.json", {
            **manifest, "schema": RESULT_SCHEMA, "preflight_only": True,
            "optimizer_constructed": False, "training_started": False,
        }); return

    epoch_zero = evaluate(
        stage=stage, predictor=predictor, runner=runner, objective=objective,
        anchor=anchor, runtime_static=runtime_static, geometry=geometry,
        token_embedding=token_embedding, hydration=hydration,
        dataset=internal_dev, device=device, microbatch=args.microbatch_size,
        workers=args.num_workers,
    )
    write_json_exclusive(args.output / "EPOCH_ZERO_AUDIT.json", {
        "schema": "harp_routemtp_epoch_zero_audit_v1",
        "stage": stage,
        "metrics": epoch_zero,
        "optimizer_constructed": False,
        "training_started": False,
        "official_tune_opened": False,
        "diagnostic_development_opened": False,
        "sealed_test_opened": False,
    })
    groups = []
    if head_parameters: groups.append({"params": head_parameters, "lr": args.learning_rate})
    if lora_parameters: groups.append({"params": lora_parameters, "lr": args.lora_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    write_json_exclusive(args.output / "OPTIMIZER_START.json", {
        "schema": "harp_routemtp_optimizer_start_v1", "stage": stage,
        "constructed_at": datetime.now(timezone.utc).isoformat(),
        "trainable_names": trainable_names,
    })
    accumulation = EFFECTIVE_BATCH // args.microbatch_size
    best_value = -math.inf; best_epoch = 0; best_state = None; stale = 0
    for epoch in range(1, args.epochs + 1):
        predictor.train(); runner.eval(); anchor.eval(); optimizer.zero_grad(set_to_none=True)
        if router is not None:
            router.set_training_progress((epoch - 1) / max(1, args.epochs - 1))
        total = batches = 0.0
        for step, host in enumerate(collated_loader(
            fit, batch=args.microbatch_size, shuffle=True, seed=args.seed + epoch,
            workers=args.num_workers,
        ), 1):
            budget_index = (step + epoch - 2) % len(ANYTIME_BUDGETS)
            _, loss, _, _ = forward_one(
                stage=stage, predictor=predictor, runner=runner, objective=objective,
                anchor=anchor, runtime_static=runtime_static, geometry=geometry,
                token_embedding=token_embedding, hydration=hydration, host=host,
                device=device, budget_index=budget_index,
            )
            (loss.total / accumulation).backward()
            if step % accumulation == 0:
                nn.utils.clip_grad_norm_(head_parameters + lora_parameters, args.gradient_clip)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.total.detach()); batches += 1
        if batches % accumulation:
            nn.utils.clip_grad_norm_(head_parameters + lora_parameters, args.gradient_clip)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(
            stage=stage, predictor=predictor, runner=runner, objective=objective,
            anchor=anchor, runtime_static=runtime_static, geometry=geometry,
            token_embedding=token_embedding, hydration=hydration,
            dataset=internal_dev, device=device, microbatch=args.microbatch_size,
            workers=args.num_workers, budgets_to_evaluate=(16,),
        )
        append_jsonl(args.output / "metrics.jsonl", {
            "epoch": epoch, "train_loss": total / max(1, batches), "internal_dev": metrics
        })
        value = float(metrics["selection_value"])
        if value > best_value:
            best_value = value; best_epoch = epoch; stale = 0
            best_state = compact_state(predictor, adapters, residual, expert, router)
        else:
            stale += 1
        if stale >= args.patience: break
    if best_state is None: raise RuntimeError("RouteMTP produced no checkpoint")
    checkpoint = {
        "schema": SCHEMA, "stage": stage, "seed": args.seed,
        "source_commit": args.source_commit, "config": config.__dict__,
        "hydration_source_commit": hydration_source_commit,
        "best_epoch": best_epoch, "best_internal_dev_value": best_value,
        "state": best_state, "added_bf16_bytes": parameter_bytes,
        "hydration_manifest_sha256": hydration_manifest_sha256,
        "parity_hydration_manifest_sha256": parity_hydration_manifest_sha256,
        "training_companion_manifest_sha256": companion_manifest_sha256,
        "internal_split_sha256": sha256_file(args.output / "INTERNAL_SPLIT.json"),
        "run_manifest_sha256": sha256_file(args.output / "RUN_MANIFEST.json"),
        "official_tune_opened": False, "diagnostic_development_opened": False,
        "formal_validation_opened": False, "calibration_opened": False, "sealed_test_opened": False,
    }
    torch.save(checkpoint, args.output / "best_checkpoint.pt")
    result = {
        "schema": RESULT_SCHEMA, "stage": stage, "best_epoch": best_epoch,
        "best_internal_dev_branch_recall_at_8": best_value,
        "checkpoint_sha256": sha256_file(args.output / "best_checkpoint.pt"),
        "parity_hydration_manifest_sha256": parity_hydration_manifest_sha256,
        "trainable_bf16_bytes": trainable_parameter_bytes,
        "added_deployed_bf16_bytes": parameter_bytes,
        "size_limit_passed": parameter_bytes <= 1 << 30,
        "optimizer_constructed": True, "training_started": True,
        "official_tune_opened": False, "diagnostic_development_opened": False,
        "formal_validation_opened": False, "calibration_opened": False, "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    checksum_names = (
        "RUN_MANIFEST.json", "INTERNAL_SPLIT.json", "EPOCH_ZERO_AUDIT.json",
        "OPTIMIZER_START.json", "best_checkpoint.pt", "STAGE_RESULT.json",
    )
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for name in checksum_names:
            handle.write(f"{sha256_file(args.output / name)}  {name}\n")
        handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
