#!/usr/bin/env python3
"""Hydrate request-deduplicated, adapter-off RouteMTP prefix K/V caches.

This replays only already-authorized outer-train requests.  It creates no new
prompts and no future-route labels.  Full request caches are stored once, but
the public loader requires a hash-bound causal source-position slice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.routemtp_cache import (  # noqa: E402
    RouteMTPCacheGeometry,
    RouteMTPSourceOffset,
    sha256_file,
    write_cache_record,
    write_hydration_manifest,
)
SCHEMA = "harp_routemtp_hydration_run_v1"
PREFIX_DOMAIN = b"GCRP2_PREFIX_V1"
MINIMUM_VRAM_GIB = 90.0


def prefix_hash(tokens: Iterable[int]) -> str:
    import struct

    digest = hashlib.sha256(PREFIX_DOMAIN)
    for token in tokens:
        digest.update(struct.pack("<i", int(token)))
    return digest.hexdigest()


def git_commit() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if len(value) != 40:
        raise RuntimeError("RouteMTP hydration requires a full source commit")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--source-offsets-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-checkpoint-sha256", required=True)
    parser.add_argument("--mtp-checkpoint-sha256", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--base-capture-audit-sha256", required=True)
    parser.add_argument("--counterfactual-companion-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-source-positions", type=int, default=0)
    parser.add_argument("--allow-low-vram-for-tests", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def load_requests(path: Path) -> dict[str, list[int]]:
    rows = _read_jsonl(path)
    sequence_starts: dict[str, dict[str, Any]] = {}
    requests: dict[str, list[int]] = {}
    for row in rows:
        event = row.get("event")
        if event == "sequence_start":
            if bool(row.get("external_evaluation", False)):
                raise PermissionError("RouteMTP hydration refuses external evaluation")
            sequence_starts[str(row["request_id"])] = row
            continue
        if event == "sequence_end":
            request_id = str(row["request_id"])
            if request_id not in sequence_starts:
                raise ValueError("sequence_end lacks an authorized sequence_start")
            tokens = row.get("full_committed_token_ids")
        else:
            request_id = str(row["request_id"])
            if str(row.get("split", "train")).lower() != "train":
                raise PermissionError("RouteMTP hydration is outer-train only")
            if bool(row.get("external_evaluation", False)):
                raise PermissionError("RouteMTP hydration refuses external evaluation")
            tokens = row.get("full_committed_token_ids")
        if not isinstance(tokens, list) or len(tokens) < 2:
            raise ValueError(f"request {request_id} lacks a complete token sequence")
        if request_id in requests:
            raise ValueError(f"duplicate hydrated request {request_id}")
        requests[request_id] = [int(value) for value in tokens]
    if not requests:
        raise ValueError("RouteMTP request hydration input is empty")
    return requests


def load_offsets(path: Path, requests: Mapping[str, list[int]]) -> list[RouteMTPSourceOffset]:
    result: list[RouteMTPSourceOffset] = []
    seen: set[tuple[str, int]] = set()
    for row in _read_jsonl(path):
        if str(row.get("split", "train")).lower() != "train":
            raise PermissionError("RouteMTP source offsets are outer-train only")
        request_id = str(row["request_id"]); source_position = int(row["source_position"])
        tokens = requests.get(request_id)
        if tokens is None:
            raise KeyError(f"source offset references missing request {request_id}")
        if not 0 <= source_position < len(tokens) - 1:
            raise ValueError("RouteMTP source position cannot supply exact H1")
        # Shifted MTP prefix pairs token[1:t+1] with target hidden[0:t].
        length = source_position
        expected_hash = prefix_hash(tokens[: source_position + 1])
        supplied_hash = str(row.get("prefix_hash", expected_hash))
        if supplied_hash != expected_hash:
            raise ValueError("RouteMTP source prefix hash mismatch")
        key = (request_id, source_position)
        if key in seen:
            raise ValueError("duplicate RouteMTP source offset")
        seen.add(key)
        result.append(RouteMTPSourceOffset(request_id, source_position, length, expected_hash))
    if not result:
        raise ValueError("RouteMTP source offset input is empty")
    return sorted(result, key=lambda value: (value.request_id, value.source_position))


def _cache_layers(cache: Any) -> list[tuple[Tensor, Tensor]]:
    if hasattr(cache, "layers"):
        result: list[tuple[Tensor, Tensor]] = []
        for layer in cache.layers:
            key = getattr(layer, "keys", getattr(layer, "key_cache", None))
            value = getattr(layer, "values", getattr(layer, "value_cache", None))
            if key is None or value is None:
                continue
            result.append((key, value))
        if result:
            return result
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache, strict=True))
    try:
        result = []
        for layer in cache:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                result.append((layer[0], layer[1]))
        if result:
            return result
    except TypeError:
        pass
    raise TypeError("unsupported Transformers MTP cache layout")


def _geometry(
    layers: list[tuple[Tensor, Tensor]],
    *,
    sequence_length: int,
    args: argparse.Namespace,
) -> RouteMTPCacheGeometry:
    key = layers[0][0]
    if key.ndim != 4 or key.shape[0] != 1:
        raise ValueError("unexpected RouteMTP K/V rank")
    candidates = [index for index, value in enumerate(key.shape) if value == sequence_length]
    sequence_axis = 2 if key.shape[2] == sequence_length else (candidates[0] if len(candidates) == 1 else -1)
    if sequence_axis != 2:
        raise ValueError("current Qwen RouteMTP cache must use [B,H,T,D]")
    return RouteMTPCacheGeometry(
        mtp_layers=len(layers),
        kv_heads=int(key.shape[1]),
        head_dimension=int(key.shape[3]),
        sequence_axis=sequence_axis,
        physical_layout="batch,kv_head,position,head_dimension",
        dtype=str(key.dtype).removeprefix("torch."),
        rope_implementation="checkpoint_qwen3_5_mrope",
        transformers_version=str(transformers.__version__),
        checkpoint_sha256=args.mtp_checkpoint_sha256,
        engine_commit=git_commit(),
    )


def main() -> None:
    args = parse_args()
    from runpod.transformers_mtp_bridge.capture_transformers_segment import load_target
    from runpod.transformers_mtp_bridge.qwen35_mtp import load_checkpoint_mtp
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite RouteMTP hydration {args.output}")
    if args.maximum_source_positions < 0:
        raise ValueError("maximum source positions must be non-negative")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RouteMTP hydration requires CUDA")
    total_gib = torch.cuda.get_device_properties(torch.device(args.device)).total_memory / 2**30
    if total_gib < MINIMUM_VRAM_GIB and not args.allow_low_vram_for_tests:
        raise RuntimeError(
            f"RouteMTP hydration requires >= {MINIMUM_VRAM_GIB:.0f} GiB VRAM, found {total_gib:.2f}"
        )
    requests = load_requests(args.requests_jsonl)
    offsets = load_offsets(args.source_offsets_jsonl, requests)
    if args.maximum_source_positions:
        offsets = offsets[: args.maximum_source_positions]
        authorized = {value.request_id for value in offsets}
        requests = {key: value for key, value in requests.items() if key in authorized}

    args.output.mkdir(parents=True)
    (args.output / "records").mkdir()
    initial = {
        "schema": SCHEMA,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "requests": len(requests),
        "source_positions": len(offsets),
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "total_vram_gib": total_gib,
        "training_started": False,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    with (args.output / "RUN_MANIFEST.json").open("x", encoding="utf-8") as handle:
        json.dump(initial, handle, indent=2, sort_keys=True); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())

    target, config = load_target(args.model, device=args.device)
    mtp = load_checkpoint_mtp(
        args.model,
        device=args.device,
        embed_tokens=target.model.embed_tokens,
        lm_head=target.lm_head,
        config=config,
    )
    records: list[dict[str, Any]] = []
    geometry: RouteMTPCacheGeometry | None = None
    with torch.inference_mode():
        for request_id, tokens in sorted(requests.items()):
            ids = torch.tensor([tokens], device=args.device, dtype=torch.long)
            target_output = target(
                input_ids=ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = target_output.hidden_states[-1]
            mtp_output = mtp(
                ids[:, 1:],
                hidden[:, :-1],
                position_ids=torch.arange(
                    len(tokens) - 1, device=args.device, dtype=torch.long
                )[None],
                past_key_values=None,
                use_cache=True,
                compute_vocabulary_logits=False,
            )
            layers = _cache_layers(mtp_output["past_key_values"])
            current_geometry = _geometry(
                layers, sequence_length=len(tokens) - 1, args=args
            )
            if geometry is None:
                geometry = current_geometry
            elif geometry != current_geometry:
                raise ValueError("RouteMTP cache geometry changed between requests")
            tensors: dict[str, Tensor] = {
                "shifted_token_ids": ids[0, 1:].detach().cpu(),
            }
            for layer, (key, value) in enumerate(layers):
                tensors[f"key.{layer}"] = key.detach().cpu()
                tensors[f"value.{layer}"] = value.detach().cpu()
            record = write_cache_record(
                args.output / "records" / f"{request_id}.safetensors",
                tensors,
                geometry,
                request_id=request_id,
            )
            record["relative_path"] = str(Path("records") / f"{request_id}.safetensors")
            records.append(record)
            del target_output, hidden, mtp_output, layers, tensors
    assert geometry is not None
    write_hydration_manifest(
        args.output / "HYDRATION_MANIFEST.json",
        geometry=geometry,
        records=records,
        offsets=offsets,
        bindings={
            "source_commit": git_commit(),
            "target_checkpoint_sha256": args.target_checkpoint_sha256,
            "mtp_checkpoint_sha256": args.mtp_checkpoint_sha256,
            "split_manifest_sha256": args.split_manifest_sha256,
            "base_capture_audit_sha256": args.base_capture_audit_sha256,
            "counterfactual_companion_sha256": args.counterfactual_companion_sha256,
        },
    )
    maximum_reserved = torch.cuda.max_memory_reserved(torch.device(args.device)) / 2**30
    result = {
        **initial,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "maximum_reserved_gib": maximum_reserved,
        "maximum_reserved_gate_gib": MINIMUM_VRAM_GIB,
        "maximum_reserved_gate_passed": maximum_reserved <= MINIMUM_VRAM_GIB,
        "hydration_manifest_sha256": sha256_file(args.output / "HYDRATION_MANIFEST.json"),
        "records": len(records),
        "source_positions": len(offsets),
    }
    with (args.output / "STAGE_RESULT.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    if maximum_reserved > MINIMUM_VRAM_GIB:
        raise RuntimeError("RouteMTP hydration exceeded the predeclared 90-GiB gate")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
