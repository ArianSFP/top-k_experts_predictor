"""Overfit a fixed real-trace subset as a HARP implementation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import HARPConfig, LossConfig
from .data import CompactHARPData
from .losses import endpoint_loss
from .metrics import model_inputs, slot_overlap
from .model import HARP8Teacher
from .train import TRAINING_SCHEMA, sha256_file


def _recall_h2(
    model: HARP8Teacher,
    data: CompactHARPData,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> float:
    overlaps = 0.0
    decisions = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = data.batch(indices[start : start + batch_size], device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                outputs = model(**model_inputs(batch))
            predicted = torch.topk(
                outputs["future_router_scores"][:, 1], 8, dim=-1
            ).indices
            overlaps += float(
                slot_overlap(predicted, batch["target_top8"][:, 1]).sum()
            )
            decisions += predicted.shape[0] * predicted.shape[1]
    return overlaps / max(1, decisions * 8)


def run_overfit(
    data: CompactHARPData,
    output_dir: Path,
    *,
    rows: int = 256,
    maximum_steps: int = 2000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    target_recall: float = 0.95,
    seed: int = 42,
    device: str = "cuda:0",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse overfit directory {output_dir}")
    output_dir.mkdir(parents=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    train_indices = data.indices("train")
    if rows > len(train_indices):
        raise ValueError("overfit subset exceeds available training rows")
    indices = np.sort(rng.choice(train_indices, rows, replace=False))
    np.save(output_dir / "source_row_indices.npy", indices)
    model = HARP8Teacher(data.config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=device.startswith("cuda"),
    )
    loss_config = LossConfig()
    initial_recall = _recall_h2(
        model, data, indices, batch_size=batch_size, device=device
    )
    history: list[dict[str, float | int]] = []
    step = 0
    passed = False
    while step < maximum_steps and not passed:
        order = indices.copy()
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), batch_size):
            batch = data.batch(order[start : start + batch_size], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                outputs = model(**model_inputs(batch))
                loss = endpoint_loss(
                    outputs,
                    batch,
                    loss_config,
                    active_horizons=(2,),
                )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 50 == 0 or step == maximum_steps:
                recall = _recall_h2(
                    model,
                    data,
                    indices,
                    batch_size=batch_size,
                    device=device,
                )
                record = {
                    "step": step,
                    "loss": float(loss.total.detach()),
                    "h2_slot_recall_at_8": recall,
                }
                history.append(record)
                print(json.dumps({"event": "harp8_real_overfit", **record}), flush=True)
                passed = recall >= target_recall
                model.train()
            if step >= maximum_steps or passed:
                break
    checkpoint = output_dir / "overfit.pt"
    torch.save(
        {
            "schema": TRAINING_SCHEMA,
            "kind": "real_trace_overfit_gate",
            "model_config": data.config.to_dict(),
            "model_state": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "seed": seed,
            "steps": step,
        },
        checkpoint,
    )
    report: dict[str, object] = {
        "schema": "harp8t_real_trace_overfit_v1",
        "rows": rows,
        "initial_h2_slot_recall_at_8": initial_recall,
        "final_h2_slot_recall_at_8": (
            history[-1]["h2_slot_recall_at_8"] if history else initial_recall
        ),
        "target_h2_slot_recall_at_8": target_recall,
        "passed": passed,
        "steps": step,
        "maximum_steps": maximum_steps,
        "history": history,
        "test_rows_used": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mtp-dir", type=Path, required=True)
    parser.add_argument("--target-state-features", type=Path, required=True)
    parser.add_argument("--mtp-state-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--maximum-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = HARPConfig.compact_compatibility()
    data = CompactHARPData(
        args.capture_dir,
        args.mtp_dir,
        args.target_state_features,
        args.mtp_state_features,
        config,
    )
    report = run_overfit(
        data,
        args.output_dir,
        rows=args.rows,
        maximum_steps=args.maximum_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        target_recall=args.target_recall,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
