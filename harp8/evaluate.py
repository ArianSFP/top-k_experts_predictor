"""Evaluate a frozen HARP checkpoint; test requires an explicit one-time gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import torch

from .config import HARPConfig
from .data import CompactHARPData
from .metrics import evaluate_split, request_bootstrap_h2
from .model import HARP8Teacher
from .train import TRAINING_SCHEMA, sha256_file, write_csv


TEST_CONFIRMATION = "OPEN_HARP8_SEALED_TEST_ONCE"


def evaluate_checkpoint(
    checkpoint_path: Path,
    data: CompactHARPData,
    output_dir: Path,
    *,
    split: str = "validation",
    batch_size: int = 256,
    device: str = "cuda:0",
    test_confirmation: str | None = None,
    training_manifest_path: Path | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse evaluation directory {output_dir}")
    allow_test = split == "test"
    if allow_test:
        if test_confirmation != TEST_CONFIRMATION:
            raise PermissionError(
                f"test evaluation requires --test-confirmation {TEST_CONFIRMATION}"
            )
        if training_manifest_path is None:
            raise ValueError("sealed test requires the frozen training manifest")
        training_manifest = json.loads(
            training_manifest_path.read_text(encoding="utf-8")
        )
        if training_manifest.get("schema") != TRAINING_SCHEMA:
            raise ValueError("training manifest schema is incompatible")
        if training_manifest.get("sealed_test_evaluated") is not False:
            raise ValueError("training manifest does not attest an unopened test")
        expected = training_manifest["outputs"]["best.pt"]["sha256"]
        if sha256_file(checkpoint_path) != expected:
            raise ValueError("selected checkpoint differs from training manifest")
    elif split != "validation":
        raise ValueError("only validation and explicitly opened test are supported")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != TRAINING_SCHEMA:
        raise ValueError("checkpoint schema is incompatible")
    config = HARPConfig(**checkpoint["model_config"])
    if config.to_dict() != data.config.to_dict():
        raise ValueError("checkpoint and data configuration differ")
    model = HARP8Teacher(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    result = evaluate_split(
        model,
        data,
        split,
        batch_size=batch_size,
        device=device,
        allow_test=allow_test,
    )
    bootstrap = request_bootstrap_h2(
        result.request_metrics,
        seed=int(checkpoint.get("seed", 42)),
    )
    output_dir.mkdir(parents=True)
    write_csv(output_dir / f"{split}_metrics.csv", result.horizon_metrics)
    write_csv(
        output_dir / f"{split}_request_metrics.csv", result.request_metrics
    )
    write_csv(output_dir / f"{split}_layer_metrics.csv", result.layer_metrics)
    write_csv(output_dir / f"{split}_domain_metrics.csv", result.domain_metrics)
    report: dict[str, object] = {
        "schema": "harp8t_evaluation_v1",
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "h2_request_macro_slot_recall_at_8": result.h2_request_macro_recall,
        "h2_bootstrap": bootstrap,
        "single_model_target": 0.80,
        "target_passed": result.h2_request_macro_recall >= 0.80,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "horizons": result.horizon_metrics,
    }
    report_path = output_dir / f"{split}_evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if allow_test:
        marker = {
            "schema": "harp8t_sealed_test_opened_v1",
            **report,
            "training_manifest": str(training_manifest_path),
            "training_manifest_sha256": sha256_file(training_manifest_path),
        }
        (output_dir / "SEALED_TEST_OPENED.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mtp-dir", type=Path, required=True)
    parser.add_argument("--target-state-features", type=Path, required=True)
    parser.add_argument("--mtp-state-features", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--test-confirmation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = HARPConfig(**checkpoint["model_config"])
    data = CompactHARPData(
        args.capture_dir,
        args.mtp_dir,
        args.target_state_features,
        args.mtp_state_features,
        config,
        split_manifest=args.split_manifest,
    )
    report = evaluate_checkpoint(
        args.checkpoint,
        data,
        args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        test_confirmation=args.test_confirmation,
        training_manifest_path=args.training_manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
