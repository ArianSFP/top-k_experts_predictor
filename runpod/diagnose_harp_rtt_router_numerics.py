#!/usr/bin/env python3
"""Diagnose real-corpus target-router reconstruction without opening test data.

This is an audit-only probe.  It compares several mathematically equivalent
rank-255 paths against captured BF16 router logits on a small non-test batch.
It never writes model/static artifacts and refuses the sealed test split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from harp_rtt.dataset import HarpRTTDataset, collate_harp_rtt
from harp_rtt.exact_k import stable_topk
from harp_rtt.static_artifacts import load_static_target_artifacts


SCHEMA = "harp_rtt_router_numerics_diagnostic_v2"


def _load_router_weights(model_root: Path, layers: int) -> Tensor:
    index = json.loads(
        (model_root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    keys = [
        f"model.language_model.layers.{layer}.mlp.gate.weight"
        for layer in range(layers)
    ]
    loaded: dict[str, Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index[key], []).append(key)
    for shard, shard_keys in by_shard.items():
        with safe_open(str(model_root / shard), framework="pt", device="cpu") as handle:
            for key in shard_keys:
                loaded[key] = handle.get_tensor(key)
    return torch.stack([loaded[key] for key in keys])


def _capture_provenance(batch: dict[str, Any], corpus_root: Path) -> list[dict[str, Any]]:
    values = batch["metadata"]["segment"]
    segments = [values] if isinstance(values, str) else list(values)
    result = []
    for segment in sorted(set(str(value) for value in segments)):
        path = corpus_root / "segments" / segment / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        result.append(
            {
                "segment": segment,
                "run_manifest": str(path),
                "engine": manifest.get("engine"),
                "engine_version": manifest.get("engine_version"),
                "engine_build_flags": manifest.get("engine_build_flags"),
                "torch": (manifest.get("hardware_topology_manifest") or {}).get("torch"),
                "cuda_runtime": (
                    manifest.get("hardware_topology_manifest") or {}
                ).get("cuda_runtime"),
                "accelerator": (
                    manifest.get("hardware_topology_manifest") or {}
                ).get("accelerator"),
                "cuda_capability": (
                    manifest.get("hardware_topology_manifest") or {}
                ).get("cuda_capability"),
                "model_revision": manifest.get("model_revision"),
                "router_artifact_sha256": manifest.get("router_artifact_sha256"),
                "quantization_format": manifest.get("quantization_format"),
            }
        )
    return result


def _metrics(scores: Tensor, captured: Tensor, labels: Tensor, valid: Tensor) -> dict[str, Any]:
    scores = scores.float()
    captured = captured.float()
    scores = scores - scores.mean(dim=-1, keepdim=True)
    centered = captured - captured.mean(dim=-1, keepdim=True)
    difference = (scores - centered)[valid]
    predicted = stable_topk(scores, 8)
    captured_ids = stable_topk(centered, 8)
    captured_membership = torch.zeros_like(centered, dtype=torch.bool)
    captured_membership.scatter_(-1, captured_ids, True)
    captured_recall = (
        captured_membership.gather(-1, predicted).sum(-1).float() / 8.0
    )[valid]
    label_membership = torch.zeros_like(centered, dtype=torch.bool)
    label_membership.scatter_(-1, labels.long(), True)
    label_recall = (
        label_membership.gather(-1, predicted).sum(-1).float() / 8.0
    )[valid]
    return {
        "max_abs_error": float(difference.abs().max()),
        "rmse": math.sqrt(float(difference.square().mean())),
        "captured_top8_slot_recall": float(captured_recall.mean()),
        "captured_top8_exact_set": float((captured_recall == 1).float().mean()),
        "selected_label_slot_recall": float(label_recall.mean()),
        "selected_label_exact_set": float((label_recall == 1).float().mean()),
    }


def _pair_metrics(first: Tensor, second: Tensor, valid: Tensor) -> dict[str, Any]:
    first = first.float() - first.float().mean(dim=-1, keepdim=True)
    second = second.float() - second.float().mean(dim=-1, keepdim=True)
    difference = (first - second)[valid]
    first_ids = stable_topk(first, 8)
    second_ids = stable_topk(second, 8)
    membership = torch.zeros_like(first, dtype=torch.bool)
    membership.scatter_(-1, second_ids, True)
    recall = (membership.gather(-1, first_ids).sum(-1).float() / 8.0)[valid]
    return {
        "max_abs_error": float(difference.abs().max()),
        "rmse": math.sqrt(float(difference.square().mean())),
        "slot_recall": float(recall.mean()),
        "exact_set": float((recall == 1).float().mean()),
        "nonexact_endpoints": int((recall != 1).sum()),
        "missing_slots": int(((1.0 - recall) * 8).sum()),
    }


def _boundary_report(scores: Tensor, reference: Tensor, valid: Tensor) -> dict[str, Any]:
    """Condition set disagreements on the reference rank-8/rank-9 margin."""

    scores = scores.float() - scores.float().mean(dim=-1, keepdim=True)
    reference = reference.float() - reference.float().mean(dim=-1, keepdim=True)
    predicted = stable_topk(scores, 8)
    reference_ids = stable_topk(reference, 8)
    membership = torch.zeros_like(reference, dtype=torch.bool)
    membership.scatter_(-1, reference_ids, True)
    recall = membership.gather(-1, predicted).sum(-1).float() / 8.0
    ordered = torch.sort(reference, dim=-1, descending=True, stable=True).values
    margin = ordered[..., 7] - ordered[..., 8]
    active_recall = recall[valid]
    active_margin = margin[valid]
    edges = (0.0, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, math.inf)
    bins = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (active_margin >= lower) & (active_margin < upper)
        count = int(selected.sum())
        if not count:
            continue
        values = active_recall[selected]
        bins.append(
            {
                "lower_inclusive": lower,
                "upper_exclusive": upper if math.isfinite(upper) else None,
                "endpoints": count,
                "nonexact_endpoints": int((values != 1).sum()),
                "missing_slots": int(((1.0 - values) * 8).sum()),
                "slot_recall": float(values.mean()),
                "exact_set": float((values == 1).float().mean()),
            }
        )
    mismatched = active_recall != 1
    return {
        "rank8_minus_rank9_margin_bins": bins,
        "mismatch_margin_max": (
            float(active_margin[mismatched].max()) if bool(mismatched.any()) else None
        ),
        "mismatch_margin_mean": (
            float(active_margin[mismatched].mean()) if bool(mismatched.any()) else None
        ),
        "all_margin_min": float(active_margin.min()),
        "all_margin_median": float(active_margin.median()),
    }


def _strict_boundary_mask(reference: Tensor, valid: Tensor) -> Tensor:
    centered = reference.float() - reference.float().mean(dim=-1, keepdim=True)
    ordered = torch.sort(centered, dim=-1, descending=True, stable=True).values
    return valid & ((ordered[..., 7] - ordered[..., 8]) > 0)


def _reference_contrast_geometry(weights: Tensor) -> tuple[Tensor, Tensor]:
    """Exact algebraic centered rank-(E-1) factorization using expert E-1."""

    layers, experts, _ = weights.shape
    differences = weights[:, :-1] - weights[:, -1:]
    keys = torch.full(
        (layers, experts, experts - 1),
        -1.0 / experts,
        dtype=weights.dtype,
    )
    identity = torch.eye(experts - 1, dtype=weights.dtype)
    keys[:, :-1] += identity
    return keys, differences.transpose(1, 2).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--static-artifacts", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "calibration"), default="validation"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--native-device",
        default="cpu",
        help="device used only for the checkpoint-dtype BF16 F.linear replay",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = HarpRTTDataset(
        args.index_root, args.split, corpus_root=args.corpus_root, allow_test=False
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_harp_rtt,
            )
        )
    )
    inputs_bf16 = batch["targets"]["future_router_inputs"].to(torch.bfloat16)
    inputs_fp32 = inputs_bf16.float()
    captured = batch["targets"]["future_router_logits"].float()
    labels = batch["targets"]["future_selected_ids"].long()
    valid = batch["targets"]["future_available"].bool()
    static = load_static_target_artifacts(
        args.static_artifacts, device="cpu", load_embedding=False
    )
    weights_bf16 = _load_router_weights(args.model_root, static.geometry.layers)
    weights_fp32 = weights_bf16.float()
    centered_weights = weights_fp32 - weights_fp32.mean(dim=1, keepdim=True)

    with torch.no_grad():
        direct_fp32 = torch.einsum("bhld,led->bhle", inputs_fp32, weights_fp32)
        native_device = torch.device(args.native_device)
        if native_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"native replay device is unavailable: {native_device}")
        native_inputs = inputs_bf16.to(native_device)
        native_layers = []
        for layer in range(weights_bf16.shape[0]):
            native_layers.append(
                F.linear(
                    native_inputs[:, :, layer], weights_bf16[layer].to(native_device)
                )
                .float()
                .cpu()
            )
        direct_bf16 = torch.stack(native_layers, dim=2)
        geometry = static.geometry
        factorized = geometry.centered_logits(inputs_fp32)
        runtime_autocast_factorized = None
        runtime_autocast_dtype = None
        if native_device.type == "cuda":
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                runtime_coordinates = torch.einsum(
                    "bhld,ldr->bhlr",
                    inputs_bf16.to(native_device),
                    geometry.input_basis.to(native_device),
                )
                runtime_autocast = (
                    torch.einsum(
                        "bhlr,ler->bhle",
                        runtime_coordinates,
                        geometry.expert_keys.to(native_device),
                    )
                    + geometry.centered_bias.to(native_device)
                )
            runtime_autocast_dtype = str(runtime_autocast.dtype)
            runtime_autocast_factorized = runtime_autocast.float().cpu()
        reconstructed_weights = geometry.reconstruct_centered_weights()
        reconstructed_direct = torch.einsum(
            "bhld,led->bhle", inputs_fp32, reconstructed_weights
        )
        contrast_keys, contrast_basis = _reference_contrast_geometry(weights_fp32)
        contrast_coordinates = torch.einsum(
            "bhld,ldr->bhlr", inputs_fp32, contrast_basis
        )
        contrast = torch.einsum(
            "bhlr,ler->bhle", contrast_coordinates, contrast_keys
        )
        # This path is not rank compressed; it is the canonical FP32 score floor.
        centered_direct = torch.einsum(
            "bhld,led->bhle", inputs_fp32, centered_weights
        )

    variants = {
        f"checkpoint_bf16_f_linear_{str(native_device).replace(':', '_')}": direct_bf16,
        "checkpoint_fp32_direct": direct_fp32,
        "checkpoint_fp32_precentered_direct": centered_direct,
        "static_v1_svd_two_stage": factorized,
        "static_v1_reconstructed_weight_one_stage": reconstructed_direct,
        "reference_contrast_rank255_two_stage": contrast,
    }
    if runtime_autocast_factorized is not None:
        variants["static_v1_svd_two_stage_runtime_bf16_autocast"] = (
            runtime_autocast_factorized
        )
    canonical_gate = _pair_metrics(factorized, direct_fp32, valid)
    strict_capture_mask = _strict_boundary_mask(captured, valid)
    strict_capture_gate = _pair_metrics(
        factorized, captured, strict_capture_mask
    )
    factorization_pass = (
        canonical_gate["exact_set"] == 1.0
        and canonical_gate["slot_recall"] == 1.0
        and canonical_gate["max_abs_error"] <= 2e-4
    )
    strict_capture_pass = (
        strict_capture_gate["exact_set"] == 1.0
        and strict_capture_gate["slot_recall"] == 1.0
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "split": args.split,
        "test_accessed": False,
        "samples": int(inputs_bf16.shape[0]),
        "valid_endpoints": int(valid.sum()),
        "input_dtype": str(inputs_bf16.dtype),
        "captured_logit_dtype_after_loader": str(captured.dtype),
        "captured_logits_exactly_bf16_representable": bool(
            torch.equal(captured, captured.to(torch.bfloat16).float())
        ),
        "captured_strict_boundary_endpoints": int(strict_capture_mask.sum()),
        "captured_tied_boundary_endpoints": int((valid & ~strict_capture_mask).sum()),
        "capture_provenance": _capture_provenance(batch, args.corpus_root),
        "source_identity": {
            "static_artifact": str(args.static_artifacts.resolve()),
            "static_geometry_sha256": static.manifest["files"][
                "router_geometry.safetensors"
            ]["sha256"],
            "checkpoint": str(args.model_root.resolve()),
            "checkpoint_revision": static.manifest["model"]["repository_revision"],
            "checkpoint_index_sha256": static.manifest["model"]["index_sha256"],
        },
        "selected_ids_vs_captured_logits": _metrics(captured, captured, labels, valid),
        "variants_vs_captured_bf16": {
            name: _metrics(scores, captured, labels, valid)
            for name, scores in variants.items()
        },
        "variants_vs_checkpoint_fp32_direct": {
            name: _pair_metrics(scores, direct_fp32, valid)
            for name, scores in variants.items()
        },
        "boundary_conditioned_vs_captured_bf16": {
            name: _boundary_report(scores, captured, valid)
            for name, scores in variants.items()
        },
        "key_pairwise_comparisons": {
            "native_bf16_vs_static_v1_svd": _pair_metrics(
                direct_bf16, factorized, valid
            ),
            "static_v1_svd_vs_native_bf16": _pair_metrics(
                factorized, direct_bf16, valid
            ),
            "native_bf16_vs_checkpoint_fp32": _pair_metrics(
                direct_bf16, direct_fp32, valid
            ),
            "static_v1_svd_vs_checkpoint_fp32": _pair_metrics(
                factorized, direct_fp32, valid
            ),
        },
        "key_pairwise_boundary_conditioned": {
            "static_v1_svd_vs_native_bf16_margin": _boundary_report(
                factorized, direct_bf16, valid
            ),
            "native_bf16_vs_static_v1_svd_margin": _boundary_report(
                direct_bf16, factorized, valid
            ),
        },
        "static_v1_weight_reconstruction": {
            "max_abs_error": float((reconstructed_weights - centered_weights).abs().max()),
            "relative_rmse": float(
                (reconstructed_weights - centered_weights).square().mean().sqrt()
                / centered_weights.square().mean().sqrt()
            ),
        },
        "reference_contrast_weight_reconstruction": {
            "max_abs_error": float(
                (
                    torch.einsum("ler,ldr->led", contrast_keys, contrast_basis)
                    - centered_weights
                )
                .abs()
                .max()
            )
        },
        "gate_assessment": {
            "canonical_geometry_reference": (
                "checkpoint FP32 affine scores on exact decoded BF16 router inputs"
            ),
            "canonical_factorization_thresholds": {
                "exact_top8_set": 1.0,
                "slot_recall_at_8": 1.0,
                "maximum_absolute_centered_logit_error": 2e-4,
            },
            "canonical_factorization_observed": canonical_gate,
            "canonical_factorization_passed": factorization_pass,
            "captured_bf16_strict_boundary_observed": strict_capture_gate,
            "captured_bf16_strict_boundary_passed": strict_capture_pass,
            "captured_bf16_ties_are_diagnostic_not_factorization_failures": True,
            "cross_architecture_bf16_exact_replay_required": False,
            "training_supported_by_static_v1": (
                factorization_pass and strict_capture_pass
            ),
            "recommendation": (
                "reuse immutable static v1; no geometry re-extraction"
                if factorization_pass and strict_capture_pass
                else "investigate and build a new immutable version"
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "static_geometry_device": "cpu",
            "native_bf16_replay_device": str(native_device),
            "native_bf16_replay_device_name": (
                torch.cuda.get_device_name(native_device)
                if native_device.type == "cuda"
                else None
            ),
            "native_bf16_replay_compute_capability": (
                list(torch.cuda.get_device_capability(native_device))
                if native_device.type == "cuda"
                else None
            ),
            "runtime_autocast_factorized_output_dtype": runtime_autocast_dtype,
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
