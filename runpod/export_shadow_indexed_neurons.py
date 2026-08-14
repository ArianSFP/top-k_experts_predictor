#!/usr/bin/env python3
"""Export an optimizer-free expert-indexed ShadowRoute subnetwork bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.shadow_bundle import LOCAL_SCHEMA  # noqa: E402
from harp_rtt.shadow_checkpoint import (  # noqa: E402
    IndexedCheckpoint,
    load_target_layer_experts,
    sha256_file,
)
from harp_rtt.shadow_expert import (  # noqa: E402
    IndexedShadowExperts,
    ShadowExpertConfig,
    target_neuron_importance,
)


SCHEMA = "harp_shadowroute_indexed_neuron_export_v1"
RESULT_SCHEMA = "harp_shadowroute_indexed_neuron_export_result_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--shadow-width", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite indexed export {args.output}")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    args.output.mkdir(parents=True)
    checkpoint = IndexedCheckpoint(args.model)
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "target_checkpoint_index_sha256": checkpoint.index_sha256,
        "shadow_width": args.shadow_width,
        "layers": 40,
        "experts": 256,
        "optimizer_constructed": False,
        "training_started": False,
        "exact_target_neuron_subnetwork": True,
        "diagnostic_only": True,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "run_manifest.json", manifest)
    counts = torch.ones(256, dtype=torch.int64)
    selected_hashes = []
    for layer in range(40):
        gate_up, down = load_target_layer_experts(
            checkpoint, layer, device=args.device, dtype=torch.bfloat16
        )
        module = IndexedShadowExperts(
            ShadowExpertConfig(shadow_width=args.shadow_width), fallback=None
        ).to(device=args.device, dtype=torch.bfloat16)
        selected = module.initialize_from_target_neurons(
            gate_up, down, target_neuron_importance(gate_up, down)
        )
        module.set_trained_counts(counts, minimum=1)
        directory = args.output / f"layer_{layer:02d}"
        directory.mkdir()
        path = directory / f"shadow_s2_indexed_layer_{layer:02d}.pt"
        value = {
            "schema": LOCAL_SCHEMA,
            "mode": "s2_indexed",
            "layer": layer,
            "source_commit": args.source_commit,
            "target_checkpoint_index_sha256": checkpoint.index_sha256,
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in module.state_dict().items()
                if name in {"gate_up_proj", "down_proj", "trained_experts"}
            },
            "selected_neurons": selected.cpu(),
            "expert_counts": counts,
            "minimum_expert_count": 1,
            "trained_selected_slot_mass": 1.0,
            "exact_target_neuron_subnetwork": True,
            "diagnostic_only": True,
            "closed_loop_authorized": False,
            "optimizer_constructed": False,
            "training_started": False,
            "formal_validation_opened": False,
            "calibration_opened": False,
            "sealed_test_opened": False,
        }
        with path.open("xb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        selected_path = directory / "selected_neurons.pt"
        with selected_path.open("xb") as handle:
            torch.save(selected.cpu(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        selected_hashes.append(sha256_file(selected_path))
        del module, gate_up, down, selected
        torch.cuda.empty_cache()
        print(json.dumps({"layer": layer, "layers": 40}), flush=True)
    result = {
        "schema": RESULT_SCHEMA,
        "layers": 40,
        "shadow_width": args.shadow_width,
        "selected_neuron_sha256": selected_hashes,
        "optimizer_constructed": False,
        "training_started": False,
        "closed_loop_authorized": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    write_json_exclusive(args.output / "STAGE_RESULT.json", result)
    paths = sorted(
        path for path in args.output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    with (args.output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.relative_to(args.output)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
