#!/usr/bin/env python3
"""Score exact train-only resident omission damage at selected MoE layers.

For each factual outer-train endpoint, this program reexecutes only the eight
selected native BF16 experts.  It removes each weighted expert contribution
from the captured full routed output and measures the resulting frozen
next-router damage.  It never rolls out the model, captures new activations,
constructs an optimizer, or opens tune/development/formal-test rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.metrics import slot_recall_at_k  # noqa: E402
from harp_rtt.resident_damage import (  # noqa: E402
    frequency_core_ids,
    omission_damage_per_slot,
    reduce_expert_damage,
    router_logits_for_routed_variants,
    validate_core_inclusion,
)
from harp_rtt.shadow_checkpoint import (  # noqa: E402
    IndexedCheckpoint,
    load_target_layer_experts,
    sha256_file,
)
from harp_rtt.shadow_expert import target_selected_expert_outputs  # noqa: E402
from harp_rtt.training import seed_everything  # noqa: E402
from runpod.train_harp_delta_v3 import (  # noqa: E402
    _manifest as validate_partition,
    _reuse_split_manifest as validate_reuse_split,
    loader,
)
from runpod.train_shadow_experts_local import batch_tensors, load_split  # noqa: E402


SCHEMA = "harp_shadowroute_resident_omission_damage_v1"
LAYER_SCHEMA = "harp_shadowroute_resident_omission_damage_layer_v1"
RESULT_SCHEMA = "harp_shadowroute_resident_omission_damage_result_v1"
LAYERS = 40
EXPERTS = 256
SLOTS = 8
HORIZONS = 4
TRAIN_ROWS = 3_584
TRAIN_ENDPOINTS_PER_LAYER = TRAIN_ROWS * HORIZONS
TRAIN_OMISSIONS_PER_LAYER = TRAIN_ENDPOINTS_PER_LAYER * SLOTS


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not layers or len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError("layers must be non-empty and unique")
    if any(layer not in range(LAYERS - 1) for layer in layers):
        raise argparse.ArgumentTypeError("omission scoring supports layers 0--38")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--train-companion", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--reuse-split-manifest", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--reference-allocation", type=Path, required=True)
    parser.add_argument("--expected-frequency-core-sha256", required=True)
    parser.add_argument("--layers", type=parse_layers, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--score-mode", choices=("router_energy", "exact"), default="router_energy"
    )
    parser.add_argument("--microbatch-size", type=int, choices=(1, 2, 4, 8, 16, 32), default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--maximum-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_checksums(output: Path) -> None:
    paths = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_reference(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, str]:
    value = json.loads(args.reference_allocation.read_text(encoding="utf-8"))
    if value.get("schema") not in {
        "harp_shadowroute_resident_allocation_v1",
        "harp_shadowroute_resident_allocation_v2",
    }:
        raise ValueError("reference resident allocation schema changed")
    expected = {
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "selection_uses_train_routes_only": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
        "total_residents": 3850,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ValueError(f"reference resident allocation {field} mismatch")
    counts = torch.as_tensor(value.get("expert_counts"), dtype=torch.int64)
    mass = torch.as_tensor(value.get("expert_weight_mass"), dtype=torch.float64)
    if counts.shape != (LAYERS, EXPERTS) or mass.shape != counts.shape:
        raise ValueError("reference allocation route statistics changed geometry")
    ids = value.get("resident_expert_ids_by_layer")
    if not isinstance(ids, list) or len(ids) != LAYERS:
        raise ValueError("reference allocation lacks forty resident layers")
    core, core_hash = frequency_core_ids(counts, core_size=64)
    if core_hash != args.expected_frequency_core_sha256:
        raise ValueError("frequency-core hash differs from the declared hard constraint")
    # The old mass allocation is allowed to be noncompliant, but its exact
    # omissions are surfaced in the manifest instead of silently accepted.
    missing_core: list[dict[str, int]] = []
    for layer, (resident, required) in enumerate(zip(ids, core.tolist())):
        resident_set = {int(expert) for expert in resident}
        missing_core.extend(
            {"layer": layer, "expert": int(expert)}
            for expert in required
            if int(expert) not in resident_set
        )
    value["observed_missing_frequency_core"] = missing_core
    return value, counts, mass, core_hash


def target_router_weights(
    checkpoint: IndexedCheckpoint,
    layer: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer + 1}."
    names = (
        prefix + "post_attention_layernorm.weight",
        prefix + "mlp.gate.weight",
    )
    values = checkpoint.tensors(names)
    norm = values[names[0]].to(device=device, dtype=torch.bfloat16)
    router = values[names[1]].to(device=device, dtype=torch.bfloat16)
    if norm.shape != (2048,) or router.shape != (EXPERTS, 2048):
        raise ValueError("next target router geometry changed")
    return norm, router


@torch.no_grad()
def score_layer(
    args: argparse.Namespace,
    *,
    layer: int,
    dataset: Any,
    checkpoint: IndexedCheckpoint,
    reference_counts: torch.Tensor,
    reference_mass: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    gate_up, down = load_target_layer_experts(
        checkpoint, layer, device=device, dtype=torch.bfloat16
    )
    norm_weight, router_weight = target_router_weights(checkpoint, layer, device)
    totals: dict[str, torch.Tensor] = {}
    rows_seen = 0
    valid_endpoints = 0
    batches_seen = 0
    reconstruction_error = 0.0
    reconstruction_energy = 0.0
    full_recall_sum = 0.0
    observed_counts = torch.zeros(EXPERTS, dtype=torch.float64)
    observed_mass = torch.zeros(EXPERTS, dtype=torch.float64)
    layer_started = time.monotonic()

    for batch_index, host in enumerate(loader(
        dataset,
        batch=args.microbatch_size,
        shuffle=False,
        seed=args.seed,
        workers=args.num_workers,
        device=device,
    )):
        if args.maximum_batches and batch_index >= args.maximum_batches:
            break
        (
            router_inputs,
            routed,
            selected_ids,
            selected_weights,
            valid,
            _requests,
            states,
        ) = batch_tensors(host, layer=layer, device=device)
        individual = target_selected_expert_outputs(
            router_inputs.to(torch.bfloat16), selected_ids, gate_up, down
        )
        reconstructed = (
            individual * selected_weights.to(individual)[..., None]
        ).sum(-2)
        active = valid.bool().unsqueeze(-1)
        difference = (reconstructed.float() - routed.float()) * active.float()
        reconstruction_error += float(difference.square().sum())
        reconstruction_energy += float((routed.float() * active.float()).square().sum())

        per_slot = omission_damage_per_slot(
            routed,
            individual,
            selected_weights,
            valid,
            states,
            norm_weight=norm_weight,
            router_weight=router_weight,
            include_exact_objective=args.score_mode == "exact",
        )
        reduced = reduce_expert_damage(
            selected_ids, selected_weights, valid, per_slot, experts=EXPERTS
        )
        for name, values in reduced.items():
            if name == "occurrences":
                observed_counts += values.cpu()
            elif name == "selected_weight_mass":
                observed_mass += values.cpu()
            else:
                totals.setdefault(name, torch.zeros(EXPERTS, dtype=torch.float64))
                totals[name] += values.cpu()

        full_logits = router_logits_for_routed_variants(
            routed.unsqueeze(-2),
            states,
            norm_weight=norm_weight,
            router_weight=router_weight,
        )[..., 0, :]
        teacher_ids = states["next_selected_expert_ids"].long()
        safe_ids = torch.where(valid.bool().unsqueeze(-1), teacher_ids, 0)
        full_recall = slot_recall_at_k(full_logits, safe_ids, None, k=SLOTS)
        full_recall_sum += float((full_recall * valid.float()).sum())
        rows_seen += int(valid.shape[0])
        valid_endpoints += int(valid.sum())
        batches_seen += 1
        if batches_seen == 1 or batches_seen % 8 == 0:
            print(json.dumps({
                "event": "omission_damage_progress",
                "layer": layer,
                "batches_seen": batches_seen,
                "rows_seen": rows_seen,
                "elapsed_seconds": time.monotonic() - layer_started,
            }), flush=True)

    del gate_up, down, norm_weight, router_weight
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if batches_seen == 0:
        raise ValueError("maximum-batches configuration scored no data")
    full_scan = rows_seen == len(dataset)
    if full_scan:
        if rows_seen != TRAIN_ROWS or valid_endpoints != TRAIN_ENDPOINTS_PER_LAYER:
            raise ValueError("complete scorer pass changed the frozen train geometry")
        if not torch.equal(observed_counts.to(torch.int64), reference_counts[layer]):
            raise ValueError("full-scan expert counts differ from the reference allocation")
        if not torch.allclose(
            observed_mass,
            reference_mass[layer],
            rtol=1e-10,
            atol=1e-6,
        ):
            difference = float((observed_mass - reference_mass[layer]).abs().max())
            raise ValueError(f"full-scan expert weight mass differs by {difference}")

    occurrence_denominator = observed_counts.clamp_min(1.0)
    metric_sums = {name: values.tolist() for name, values in sorted(totals.items())}
    metric_means = {
        name: (values / occurrence_denominator).tolist()
        for name, values in sorted(totals.items())
    }
    return {
        "schema": LAYER_SCHEMA,
        "layer": layer,
        "score_mode": args.score_mode,
        "rows_seen": rows_seen,
        "expected_full_train_rows": TRAIN_ROWS,
        "valid_endpoints": valid_endpoints,
        "expected_full_train_endpoints": TRAIN_ENDPOINTS_PER_LAYER,
        "selected_omissions": int(observed_counts.sum()),
        "expected_full_train_omissions": TRAIN_OMISSIONS_PER_LAYER,
        "batches_seen": batches_seen,
        "maximum_batches": args.maximum_batches,
        "complete_train_scan": full_scan,
        "eligible_for_final_allocation": full_scan,
        "expert_occurrences": observed_counts.to(torch.int64).tolist(),
        "expert_selected_weight_mass": observed_mass.tolist(),
        "metric_sums_by_expert": metric_sums,
        "metric_means_per_occurrence_by_expert": metric_means,
        "exact_bf16_selected_experts_reexecuted": True,
        "captured_full_routed_output_used": True,
        "frozen_attention_delta_teacher_forcing": True,
        "routed_reconstruction_relative_rmse": (
            reconstruction_error / max(reconstruction_energy, 1e-30)
        ) ** 0.5,
        "full_routed_next_router_slot_recall_at_8": (
            full_recall_sum / max(valid_endpoints, 1)
        ),
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }


def main() -> None:
    args = parse_args()
    if len(args.source_commit) != 40 or any(
        value not in "0123456789abcdef" for value in args.source_commit
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    if len(args.expected_frequency_core_sha256) != 64 or any(
        value not in "0123456789abcdef"
        for value in args.expected_frequency_core_sha256
    ):
        raise ValueError("expected frequency-core hash must be lowercase SHA-256")
    if args.maximum_batches < 0:
        raise ValueError("maximum-batches must be non-negative")
    validate_partition(args.partition_manifest, "b2_reuse_4096")
    reuse = validate_reuse_split(args.reuse_split_manifest)
    reference, reference_counts, reference_mass, core_hash = validate_reference(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(args.seed, deterministic=args.deterministic)
    checkpoint = IndexedCheckpoint(args.target_model)

    args.data_profile = "b2_reuse_4096"
    args.next_router_agreement = True
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "layers": list(args.layers),
        "score_mode": args.score_mode,
        "microbatch_size": args.microbatch_size,
        "num_workers": args.num_workers,
        "maximum_batches": args.maximum_batches,
        "diagnostic_partial_scan": args.maximum_batches > 0,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "partition_manifest_sha256": sha256_file(args.partition_manifest),
        "reuse_split_manifest_sha256": sha256_file(args.reuse_split_manifest),
        "train_companion_manifest_sha256": sha256_file(
            args.train_companion / "manifest.json"
        ),
        "reference_allocation_sha256": sha256_file(args.reference_allocation),
        "reference_allocation_objective": reference.get("allocation_objective"),
        "reference_missing_frequency_core": reference[
            "observed_missing_frequency_core"
        ],
        "frequency_core_size_per_layer": 64,
        "frequency_core_cells": 64 * LAYERS,
        "frequency_core_sha256": core_hash,
        "frequency_core_hash_verified": True,
        "optional_cell_budget": 3850 - 64 * LAYERS,
        "expected_full_train_rows_per_layer": TRAIN_ROWS,
        "expected_full_train_endpoints_per_layer": TRAIN_ENDPOINTS_PER_LAYER,
        "expected_full_train_omissions_per_layer": TRAIN_OMISSIONS_PER_LAYER,
        "selection_uses_train_routes_only": True,
        "new_capture_performed": False,
        "model_rollout_performed": False,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "RUN_MANIFEST.json", manifest)

    rows: list[dict[str, Any]] = []
    training_requests = set(reuse["inner_split"]["training_requests"])
    for layer in args.layers:
        args.layer = layer
        dataset, groups = load_split(
            args, "train", selected_requests=training_requests
        )
        if len(dataset) != TRAIN_ROWS or len(groups) != 224:
            raise ValueError("scorer train split differs from the frozen contract")
        row = score_layer(
            args,
            layer=layer,
            dataset=dataset,
            checkpoint=checkpoint,
            reference_counts=reference_counts,
            reference_mass=reference_mass,
            device=device,
        )
        write_json_exclusive(args.output / f"layer_{layer:02d}.json", row)
        rows.append(row)

    result = {
        "schema": RESULT_SCHEMA,
        "layers": list(args.layers),
        "score_mode": args.score_mode,
        "complete_train_scan_all_layers": all(
            bool(row["complete_train_scan"]) for row in rows
        ),
        "eligible_for_final_allocation": all(
            bool(row["eligible_for_final_allocation"]) for row in rows
        ),
        "frequency_core_sha256": core_hash,
        "frequency_core_hash_verified": True,
        "layer_results": [f"layer_{layer:02d}.json" for layer in args.layers],
        "selection_uses_train_routes_only": True,
        "new_capture_performed": False,
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
