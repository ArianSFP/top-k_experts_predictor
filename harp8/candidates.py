"""Export fixed-width HARP candidate pools for offline scheduler studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .config import HARPConfig
from .data import CompactHARPData
from .metrics import model_inputs
from .model import HARP8Teacher
from .train import TRAINING_SCHEMA, sha256_file


POOL_SCHEMA = "harp8_candidate_pool_v1"


def _load_checkpoint(path: Path, device: str) -> tuple[HARP8Teacher, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != TRAINING_SCHEMA:
        raise ValueError("checkpoint is not an HARP training checkpoint")
    config = HARPConfig(**payload["model_config"])
    model = HARP8Teacher(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload


def export_candidate_pool(
    data: CompactHARPData,
    checkpoint: Path,
    output_dir: Path,
    *,
    split: str = "validation",
    candidate_count: int = 16,
    batch_size: int = 64,
    device: str = "cuda:0",
    allow_test: bool = False,
    store_context: bool = True,
) -> dict[str, Any]:
    """Export one request-grouped, immutable candidate pool."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse candidate directory {output_dir}")
    if not 8 < candidate_count <= data.config.experts:
        raise ValueError("candidate_count must exceed native top-8 and fit experts")
    output_dir.mkdir(parents=True)
    model, checkpoint_payload = _load_checkpoint(checkpoint, device)
    indices = data.indices(split, allow_test=allow_test)
    count = len(indices)
    h, layers, experts = data.config.horizons, data.config.layers, data.config.experts
    def mm(name: str, dtype: str, shape: tuple[int, ...]) -> np.memmap:
        return np.memmap(output_dir / name, mode="w+", dtype=dtype, shape=shape)
    candidate_scores = mm("candidate_scores.f32", "<f4", (count, h, layers, candidate_count))
    candidate_ids = mm("candidate_ids.u2", "<u2", (count, h, layers, candidate_count))
    target_membership = mm("target_membership.u1", "u1", (count, h, layers, candidate_count))
    teacher_candidate_scores = mm("teacher_candidate_scores.f32", "<f4", (count, h, layers, candidate_count))
    valid_future = mm("valid_future.u1", "u1", (count, h))
    current_scores = mm("current_scores.f32", "<f4", (count, h, layers, candidate_count))
    current_rank = mm("current_rank.f32", "<f4", (count, h, layers, candidate_count))
    source_gates = mm("source_gates.f16", "<f2", (count, h, layers, 3))
    copy_gates = mm("copy_gates.f16", "<f2", (count, h, layers, candidate_count))
    contexts = mm("generator_context.f16", "<f2", (count, h, layers, model.config.model_width)) if store_context else None
    request_ids = np.empty(count, dtype=np.int64)
    within = np.empty(count, dtype=np.int32)
    domains: list[str] = [""] * count

    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            batch_indices = indices[start:stop]
            batch = data.batch(batch_indices, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                outputs = model(**model_inputs(batch))
            scores = outputs["future_router_scores"].float()
            values, ids = torch.topk(scores, candidate_count, dim=-1, sorted=True)
            teacher = batch["teacher_router_scores"].float()
            target = batch["target_top8"].long()
            member = (ids.unsqueeze(-1) == target.unsqueeze(-2)).any(dim=-1)
            current = batch["route_history"][:, :, 0].float()
            current = current - current.mean(dim=-1, keepdim=True)
            current_values = current[:, None].expand(-1, h, -1, -1).gather(-1, ids)
            full_rank = torch.argsort(torch.argsort(-current, dim=-1), dim=-1).float()
            current_ranks = full_rank[:, None].expand(-1, h, -1, -1).gather(-1, ids) / max(1, experts - 1)
            request, domain, local_within = data.metadata(batch_indices)
            request_ids[start:stop], within[start:stop] = request, local_within
            domains[start:stop] = [str(value) for value in domain]
            candidate_scores[start:stop] = values.cpu().numpy()
            candidate_ids[start:stop] = ids.cpu().numpy()
            target_membership[start:stop] = member.cpu().numpy().astype(np.uint8)
            teacher_candidate_scores[start:stop] = teacher.gather(-1, ids).cpu().numpy()
            valid_future[start:stop] = batch["valid_future"].cpu().numpy().astype(np.uint8)
            current_scores[start:stop] = current_values.cpu().numpy()
            current_rank[start:stop] = current_ranks.cpu().numpy()
            source_gates[start:stop] = outputs["source_gate_weights"].float().cpu().numpy()
            # The generator exposes a copy gate for every expert. Candidate pools
            # retain only the selected candidate namespace, exactly as the score
            # and rank arrays do; this also supports widths greater than 16.
            candidate_copy_gates = outputs["copy_gate"].gather(-1, ids)
            copy_gates[start:stop] = candidate_copy_gates.float().cpu().numpy()
            if contexts is not None:
                contexts[start:stop] = outputs["generator_context"].float().cpu().numpy()

    for value in (candidate_scores, candidate_ids, target_membership, teacher_candidate_scores,
                  valid_future, current_scores, current_rank, source_gates, copy_gates, contexts):
        if value is not None:
            value.flush()
    (output_dir / "metadata.json").write_text(json.dumps(
        {"request_ids": request_ids.tolist(), "within": within.tolist(), "domains": domains},
        sort_keys=True) + "\n", encoding="utf-8")
    arrays = [{"path": path.name, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
              for path in sorted(output_dir.iterdir()) if path.suffix in (".f32", ".f16", ".u1", ".u2")]
    manifest = {
        "schema": POOL_SCHEMA, "split": split, "rows": count, "horizons": h,
        "layers": layers, "experts": experts, "native_k": 8,
        "candidate_count": candidate_count, "model_width": model.config.model_width,
        "store_context": store_context,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint),
                       "epoch": checkpoint_payload.get("epoch"), "seed": checkpoint_payload.get("seed")},
        "split_manifest": ({"path": str(data.split_manifest_path), "sha256": sha256_file(data.split_manifest_path)}
                           if data.split_manifest_path is not None else None),
        "allow_test": bool(allow_test), "arrays": arrays,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-dir", "mtp-dir", "target-state-features", "mtp-state-features"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = HARPConfig(**payload["model_config"])
    data = CompactHARPData(args.capture_dir, args.mtp_dir, args.target_state_features,
                           args.mtp_state_features, config, split_manifest=args.split_manifest)
    manifest = export_candidate_pool(data, args.checkpoint, args.output_dir,
                                     split=args.split, candidate_count=args.candidate_count,
                                     batch_size=args.batch_size, device=args.device,
                                     allow_test=args.allow_test, store_context=not args.no_context)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
