#!/usr/bin/env python3
"""End-to-end factual closed-loop distillation for HARP-ShadowRoute S0.

This is the first ShadowRoute stage that optimizes the deployed 40-layer
future-token rollout.  It uses only outer-train factual labels already stored
in the rich corpus.  Target labels supervise the loss and are never accepted
by the model forward API.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "runpod" / "transformers_mtp_bridge"
for candidate in (str(REPO_ROOT), str(BRIDGE)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from harp_rtt.dataset import HarpRTTDataset  # noqa: E402
from harp_rtt.exact_k import exact_set_nll  # noqa: E402
from harp_rtt.losses import boundary_loss_per_endpoint  # noqa: E402
from harp_rtt.shadow_backbone import (  # noqa: E402
    freeze_except_shadow,
    install_shadow_experts,
)
from harp_rtt.shadow_bundle import (  # noqa: E402
    CHECKPOINT_MODE,
    LOCAL_SCHEMA,
    load_shadow_bundle,
)
from harp_rtt.shadow_checkpoint import IndexedCheckpoint, sha256_file  # noqa: E402
from harp_rtt.shadow_expert import ExactTop1PlusDraftExperts  # noqa: E402
from harp_rtt.shadow_rollout_training import (  # noqa: E402
    ShadowTrainingHooks,
    cache_to_cpu,
    clone_detached_hybrid_cache,
    exact_prefix_cache_for_training,
)
from harp_rtt.training import seed_everything  # noqa: E402


SCHEMA = "harp_shadowroute_factual_rollout_training_v1"
RESULT_SCHEMA = "harp_shadowroute_factual_rollout_result_v1"
SELECTION_DOMAIN = b"HARP_SHADOWROUTE_FACTUAL_ROLLOUT_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--initializer-root", type=Path, required=True)
    parser.add_argument("--initializer-source-commit", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--router-temperature", type=float, default=2.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--limit-train-requests", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _full_sha(value: str, name: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a full lowercase Git SHA")
    return value


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


def write_checksums(output: Path) -> None:
    paths = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.relative_to(output)}\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class FactualPath:
    record_index: int
    request_id: str
    position: int
    prefix_token_ids: tuple[int, ...]
    future_token_ids: tuple[int, int, int, int]


class FactualRolloutCorpus:
    """One causally selected H1--H4 path per request with label-only targets."""

    roles = (
        "raw_target_router_logits",
        "selected_expert_ids",
        "routed_expert_output_delta_r",
        "post_moe_residual_xplus",
    )

    def __init__(
        self,
        base: HarpRTTDataset,
        *,
        requests: set[str],
        seed: int,
    ) -> None:
        self.base = base
        selected: dict[str, tuple[bytes, FactualPath]] = {}
        for record_index, record in enumerate(base.records):
            segment = base.segments[record.segment]
            sequence = segment.sequences[record.sequence]
            request = str(sequence["request_id"])
            if request not in requests:
                continue
            if str(sequence.get("split")) != "train":
                raise PermissionError("factual rollout encountered a non-train request")
            full_tokens = tuple(
                int(token)
                for token in (
                    list(sequence["prompt_token_ids"])
                    + list(sequence["generated_token_ids"])
                )
            )
            position = int(record.position)
            if len(full_tokens) <= position + 4:
                raise ValueError("factual rollout source lacks complete H1--H4 tokens")
            payload = (
                SELECTION_DOMAIN
                + b"\0"
                + str(seed).encode()
                + b"\0"
                + request.encode()
                + b"\0"
                + str(position).encode()
            )
            rank = hashlib.sha256(payload).digest()
            path = FactualPath(
                record_index=record_index,
                request_id=request,
                position=position,
                prefix_token_ids=full_tokens[: position + 1],
                future_token_ids=tuple(full_tokens[position + 1 : position + 5]),
            )
            previous = selected.get(request)
            if previous is None or rank < previous[0]:
                selected[request] = (rank, path)
        missing = sorted(requests - set(selected))
        if missing:
            raise ValueError(f"factual rollout lacks requests: {missing[:4]}")
        self.paths = tuple(selected[request][1] for request in sorted(selected))

    def __len__(self) -> int:
        return len(self.paths)

    def item(self, ordinal: int) -> dict[str, Any]:
        path = self.paths[ordinal]
        record = self.base.records[path.record_index]
        segment = self.base.segments[record.segment]
        future = [
            segment.read_layer_token("target", row, self.roles)
            for row in record.future_rows
        ]
        return {
            "request_id": path.request_id,
            "position": path.position,
            "prefix_token_ids": list(path.prefix_token_ids),
            "future_token_ids": list(path.future_token_ids),
            "target_router_logits": torch.stack(
                [row["raw_target_router_logits"].float() for row in future]
            ),
            "target_selected_ids": torch.stack(
                [row["selected_expert_ids"].long() for row in future]
            ),
            "target_routed_delta": torch.stack(
                [row["routed_expert_output_delta_r"].float() for row in future]
            ),
            "target_hidden_state": torch.stack(
                [row["post_moe_residual_xplus"].float() for row in future]
            ),
        }


def _split_requests(path: Path) -> tuple[set[str], set[str], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("formal_validation_opened") is not False:
        raise PermissionError("reuse split opened formal validation")
    if value.get("calibration_opened") is not False or value.get("sealed_test_opened") is not False:
        raise PermissionError("reuse split crossed a sealed partition")
    inner = value.get("inner_split")
    if not isinstance(inner, dict) or inner.get("request_disjoint") is not True:
        raise ValueError("reuse split is not request-disjoint")
    train = {str(item) for item in inner.get("training_requests", [])}
    tune = {str(item) for item in inner.get("tuning_requests", [])}
    if len(train) != 224 or len(tune) != 32 or train & tune:
        raise ValueError("frozen factual train/tune request contract changed")
    return train, tune, value


def rollout_loss(
    predicted_logits: Tensor,
    predicted_routed: Tensor,
    predicted_hidden: Tensor,
    teacher_logits: Tensor,
    teacher_ids: Tensor,
    teacher_routed: Tensor,
    teacher_hidden: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Metric-aligned route loss plus residual/trajectory stabilizers."""

    predicted = predicted_logits.float()
    teacher = teacher_logits.detach().float()
    ids = teacher_ids.detach().long()
    probability = torch.softmax(teacher / temperature, dim=-1)
    router_kl = F.kl_div(
        torch.log_softmax(predicted / temperature, dim=-1),
        probability,
        reduction="batchmean",
    ) * (temperature * temperature)
    set_nll = exact_set_nll(predicted, ids, k=8)
    boundary = boundary_loss_per_endpoint(
        predicted,
        teacher,
        ids,
        margin=0.125,
        model_rank_start=9,
        teacher_rank_start=9,
        rank_end=32,
    ).mean()
    routed = F.huber_loss(predicted_routed.float(), teacher_routed.detach().float())
    hidden = F.huber_loss(predicted_hidden.float(), teacher_hidden.detach().float())
    hidden_cosine = (
        1.0
        - F.cosine_similarity(
            predicted_hidden.float(), teacher_hidden.detach().float(), dim=-1
        )
    ).mean()
    total = (
        set_nll
        + router_kl
        + 0.2 * boundary
        + 0.5 * routed
        + 0.25 * hidden
        + 0.1 * hidden_cosine
    )
    return total, {
        "set_nll": set_nll,
        "router_kl": router_kl,
        "boundary": boundary,
        "routed_huber": routed,
        "hidden_huber": hidden,
        "hidden_cosine": hidden_cosine,
    }


def _run_path(
    installed,
    hooks: ShadowTrainingHooks,
    item: Mapping[str, Any],
    host_cache: Any,
    *,
    device: torch.device,
    temperature: float,
    backward_scale: float | None,
) -> dict[str, Any]:
    cache = clone_detached_hybrid_cache(host_cache, device=device)
    labels_logits = item["target_router_logits"]
    labels_ids = item["target_selected_ids"]
    labels_routed = item["target_routed_delta"]
    labels_hidden = item["target_hidden_state"]
    horizon_recall: list[float] = []
    totals: dict[str, float] = {}
    layer_zero_native_match = None
    for horizon, token_id in enumerate(item["future_token_ids"]):
        hooks.clear()
        output = installed.model(
            input_ids=torch.tensor([[token_id]], dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=False,
            output_router_logits=True,
            return_dict=True,
        )
        logits, ids, _weights, routed, hidden = hooks.stacked()
        next_cache = clone_detached_hybrid_cache(output.past_key_values)
        loss, components = rollout_loss(
            logits,
            routed,
            hidden,
            labels_logits[horizon].to(device),
            labels_ids[horizon].to(device),
            labels_routed[horizon].to(device),
            labels_hidden[horizon].to(device),
            temperature=temperature,
        )
        if backward_scale is not None:
            (loss * backward_scale).backward()
        overlap = (
            ids.detach().cpu()[..., None, :]
            == labels_ids[horizon][..., :, None]
        ).any(-1).float().mean()
        horizon_recall.append(float(overlap))
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        if horizon == 0:
            layer_zero_native_match = bool(
                torch.equal(ids.detach().cpu()[0], labels_ids[0, 0])
            )
        del output, loss, components, logits, ids, routed, hidden
        cache = next_cache
    return {
        "request_id": str(item["request_id"]),
        "position": int(item["position"]),
        "recall": horizon_recall,
        "layer_zero_h1_native_match": layer_zero_native_match,
        **{name: value / 4.0 for name, value in totals.items()},
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize an empty rollout")
    result = {
        f"route_recall_h{horizon + 1}": sum(row["recall"][horizon] for row in rows) / len(rows)
        for horizon in range(4)
    }
    result["route_recall_h2_h4"] = sum(
        result[f"route_recall_h{horizon}"] for horizon in (2, 3, 4)
    ) / 3.0
    result["loss"] = sum(float(row["loss"]) for row in rows) / len(rows)
    return result


def _host_prefix_cache(installed, item: Mapping[str, Any]) -> Any:
    exact = exact_prefix_cache_for_training(installed, list(item["prefix_token_ids"]))
    return cache_to_cpu(exact)


def evaluate(
    installed,
    hooks: ShadowTrainingHooks,
    corpus: FactualRolloutCorpus,
    prefix_caches: dict[tuple[str, int], Any],
    *,
    device: torch.device,
    temperature: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    installed.model.eval()
    with torch.no_grad():
        for ordinal in range(len(corpus)):
            item = corpus.item(ordinal)
            key = (str(item["request_id"]), int(item["position"]))
            if key not in prefix_caches:
                prefix_caches[key] = _host_prefix_cache(installed, item)
            rows.append(
                _run_path(
                    installed,
                    hooks,
                    item,
                    prefix_caches[key],
                    device=device,
                    temperature=temperature,
                    backward_scale=None,
                )
            )
            print(json.dumps({"evaluation": ordinal + 1, "total": len(corpus)}), flush=True)
    return _summary(rows), rows


def _state_dicts(installed) -> list[dict[str, Tensor]]:
    result = []
    for module in installed.shadow_experts:
        if not isinstance(module, ExactTop1PlusDraftExperts):
            raise TypeError("factual rollout v1 is restricted to S0")
        result.append({
            name: value.detach().cpu().clone()
            for name, value in module.draft_expert.state_dict().items()
        })
    return result


def _load_state_dicts(installed, states: list[dict[str, Tensor]]) -> None:
    if len(states) != 40:
        raise ValueError("factual rollout state must contain 40 layers")
    for module, state in zip(installed.shadow_experts, states, strict=True):
        assert isinstance(module, ExactTop1PlusDraftExperts)
        module.draft_expert.load_state_dict(state, strict=True)


def _save_bundle(
    installed,
    output: Path,
    *,
    source_commit: str,
    parent_source_commit: str,
    target_sha: str,
    authorized: bool,
    metrics: Mapping[str, float],
) -> None:
    for layer, module in enumerate(installed.shadow_experts):
        assert isinstance(module, ExactTop1PlusDraftExperts)
        directory = output / f"layer_{layer:02d}"
        directory.mkdir()
        path = directory / f"shadow_{CHECKPOINT_MODE['exact_top1_plus_draft']}_layer_{layer:02d}.pt"
        value = {
            "schema": LOCAL_SCHEMA,
            "mode": CHECKPOINT_MODE["exact_top1_plus_draft"],
            "layer": layer,
            "source_commit": source_commit,
            "parent_source_commit": parent_source_commit,
            "target_checkpoint_index_sha256": target_sha,
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in module.draft_expert.state_dict().items()
            },
            "closed_loop_authorized": bool(authorized),
            "closed_loop_metrics": dict(metrics),
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        }
        torch.save(value, path)


def main() -> None:
    args = parse_args()
    _full_sha(args.source_commit, "source commit")
    _full_sha(args.initializer_source_commit, "initializer source commit")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite factual rollout {args.output}")
    if args.epochs < 1 or args.accumulation < 1:
        raise ValueError("epochs and accumulation must be positive")
    if args.learning_rate <= 0 or args.router_temperature <= 0:
        raise ValueError("learning rate and router temperature must be positive")
    if args.limit_train_requests is not None and args.limit_train_requests < 1:
        raise ValueError("training request limit must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = IndexedCheckpoint(args.model)
    train_requests, tune_requests, _reuse = _split_requests(args.reuse_split_manifest)
    if args.limit_train_requests is not None:
        train_requests = set(sorted(train_requests)[: args.limit_train_requests])
    base = HarpRTTDataset(args.index, "train", corpus_root=args.corpus, max_tree_nodes=32)
    train = FactualRolloutCorpus(base, requests=train_requests, seed=args.seed)
    tune = FactualRolloutCorpus(base, requests=tune_requests, seed=args.seed)

    args.output.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "initializer_source_commit": args.initializer_source_commit,
        "initializer_root": str(args.initializer_root),
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "train_requests": len(train),
        "tune_requests": len(tune),
        "one_causal_position_per_request": True,
        "selection_domain": SELECTION_DOMAIN.decode(),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "accumulation": args.accumulation,
        "router_temperature": args.router_temperature,
        "labels_used_as_model_inputs": False,
        "optimizer_constructed": False,
        "training_started": False,
        "preflight_only": args.preflight_only,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)

    try:
        from capture_transformers_segment import load_target
    except ImportError as exc:  # pragma: no cover - production environment
        raise RuntimeError("factual rollout requires the pinned target bridge") from exc
    target, _config = load_target(args.model, device=args.device)
    installed = install_shadow_experts(
        target, "exact_top1_plus_draft", retain_native=True
    )
    load_shadow_bundle(
        installed,
        args.initializer_root,
        source_commit=args.initializer_source_commit,
        target_checkpoint_index_sha256=checkpoint.index_sha256,
    )
    target.eval()
    trainable_names = freeze_except_shadow(target)
    trainable = [parameter for parameter in target.parameters() if parameter.requires_grad]
    initial_states = _state_dicts(installed)
    prefix_caches: dict[tuple[str, int], Any] = {}
    torch.cuda.reset_peak_memory_stats(device)

    with ShadowTrainingHooks(target) as hooks:
        item = train.item(0)
        key = (str(item["request_id"]), int(item["position"]))
        prefix_caches[key] = _host_prefix_cache(installed, item)
        target.zero_grad(set_to_none=True)
        audit_row = _run_path(
            installed,
            hooks,
            item,
            prefix_caches[key],
            device=device,
            temperature=args.router_temperature,
            backward_scale=0.25,
        )
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        gradient_layers = sum(
            any(
                parameter.grad is not None and bool(parameter.grad.abs().max() > 0)
                for parameter in module.draft_expert.parameters()
            )
            for module in installed.shadow_experts
            if isinstance(module, ExactTop1PlusDraftExperts)
        )
        peak_gib = torch.cuda.max_memory_reserved(device) / 2**30
        preflight = {
            "request_id": item["request_id"],
            "position": item["position"],
            "four_token_backward_completed": True,
            "h1_layer0_native_route_match": audit_row["layer_zero_h1_native_match"],
            "finite_gradients": finite_gradients,
            "gradient_layers": gradient_layers,
            "expected_gradient_layers": 40,
            "peak_reserved_gib": peak_gib,
            "maximum_reserved_gib": 90.0,
            "passed": bool(
                audit_row["layer_zero_h1_native_match"]
                and finite_gradients
                and gradient_layers == 40
                and peak_gib <= 90.0
            ),
        }
        write_json_exclusive(args.output / "MEMORY_PREFLIGHT.json", preflight)
        target.zero_grad(set_to_none=True)
        if not preflight["passed"]:
            raise RuntimeError(f"factual rollout preflight failed: {preflight}")
        if args.preflight_only:
            result = {
                "schema": RESULT_SCHEMA,
                "preflight_only": True,
                "preflight": preflight,
                "optimizer_constructed": False,
                "training_started": False,
                "formal_validation_opened": False,
                "calibration_opened": False,
                "sealed_test_opened": False,
            }
            write_json_exclusive(args.output / "STAGE_RESULT.json", result)
            write_checksums(args.output)
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        epoch_zero, epoch_zero_rows = evaluate(
            installed,
            hooks,
            tune,
            prefix_caches,
            device=device,
            temperature=args.router_temperature,
        )
        write_json_exclusive(
            args.output / "EPOCH_ZERO_AUDIT.json",
            {"metrics": epoch_zero, "requests": len(epoch_zero_rows)},
        )
        write_json_exclusive(
            args.output / "OPTIMIZER_START.json",
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "optimizer": "AdamW",
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "trainable_names": list(trainable_names),
            },
        )
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        best_metric = float(epoch_zero["route_recall_h2_h4"])
        best_h4 = float(epoch_zero["route_recall_h4"])
        best_states = initial_states
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            target.eval()
            optimizer.zero_grad(set_to_none=True)
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
            train_rows = []
            for step, ordinal in enumerate(order, start=1):
                item = train.item(ordinal)
                key = (str(item["request_id"]), int(item["position"]))
                if key not in prefix_caches:
                    prefix_caches[key] = _host_prefix_cache(installed, item)
                row = _run_path(
                    installed,
                    hooks,
                    item,
                    prefix_caches[key],
                    device=device,
                    temperature=args.router_temperature,
                    backward_scale=1.0 / (4.0 * args.accumulation),
                )
                train_rows.append(row)
                if step % args.accumulation == 0 or step == len(order):
                    torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                print(json.dumps({"epoch": epoch, "train": step, "total": len(order)}), flush=True)
            tune_metrics, tune_rows = evaluate(
                installed,
                hooks,
                tune,
                prefix_caches,
                device=device,
                temperature=args.router_temperature,
            )
            row = {
                "epoch": epoch,
                "train": _summary(train_rows),
                "tune": tune_metrics,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }
            append_jsonl(args.output / "metrics.jsonl", row)
            metric = float(tune_metrics["route_recall_h2_h4"])
            h4 = float(tune_metrics["route_recall_h4"])
            if metric > best_metric and h4 >= best_h4:
                best_metric = metric
                best_h4 = h4
                best_epoch = epoch
                best_states = _state_dicts(installed)

        _load_state_dicts(installed, best_states)
        gain = best_metric - float(epoch_zero["route_recall_h2_h4"])
        h4_gain = best_h4 - float(epoch_zero["route_recall_h4"])
        authorized = bool(best_epoch > 0 and gain > 0 and h4_gain >= 0)
        final_metrics = {
            "route_recall_h2_h4": best_metric,
            "route_recall_h4": best_h4,
            "gain_h2_h4": gain,
            "gain_h4": h4_gain,
        }
        _save_bundle(
            installed,
            args.output,
            source_commit=args.source_commit,
            parent_source_commit=args.initializer_source_commit,
            target_sha=checkpoint.index_sha256,
            authorized=authorized,
            metrics=final_metrics,
        )
        result = {
            "schema": RESULT_SCHEMA,
            "preflight_only": False,
            "preflight": preflight,
            "epoch_zero": epoch_zero,
            "best_epoch": best_epoch,
            "metrics": final_metrics,
            "closed_loop_authorized": authorized,
            "optimizer_constructed": True,
            "training_started": True,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        }
        write_json_exclusive(args.output / "STAGE_RESULT.json", result)
        write_checksums(args.output)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
