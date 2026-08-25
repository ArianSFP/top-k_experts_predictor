#!/usr/bin/env python3
"""Decompose the C32 ranking ceiling into survivor and novel decisions.

This is a read-only diagnostic over the retained score audit.  It does not
train a model or open any formal-validation/sealed-test partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.cachetrajectory_c32 import (  # noqa: E402
    build_candidate_ids,
    request_macro_metrics,
    split_masks,
    target_membership,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def select(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    dense_rank: torch.Tensor,
) -> torch.Tensor:
    """Select with authoritative full-256 dense rank as the tie-break."""
    rank_order = torch.argsort(
        dense_rank.long(), dim=-1, descending=False, stable=True
    )
    ordered_scores = scores.float().gather(-1, rank_order)
    score_order = torch.argsort(
        ordered_scores, dim=-1, descending=True, stable=True
    )[..., :8]
    return candidates.gather(-1, rank_order.gather(-1, score_order))


def metrics(
    selected: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    request_index: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, object]:
    return request_macro_metrics(
        selected, current, target, valid, request_index, mask
    )


def main() -> None:
    args = parse_args()
    audit = torch.load(args.audit, map_location="cpu", weights_only=False)
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    scores = audit["scores"].float()
    current = audit["current_ids"].long()
    target = audit["target_ids"].long()
    valid = audit["valid"].bool()
    request_index = reference["request_index"].long()
    if not torch.equal(current, reference["current_ids"].long()):
        raise ValueError("reference current IDs differ")
    if not torch.equal(target, reference["target_ids"].long()):
        raise ValueError("reference target IDs differ")
    if not torch.equal(valid, reference["valid"].bool()):
        raise ValueError("reference valid mask differs")

    candidates = build_candidate_ids(scores, current, width=32)
    base = scores.gather(-1, candidates)
    dense_order = scores.argsort(dim=-1, descending=True, stable=True)
    inverse_rank = torch.empty_like(dense_order)
    inverse_rank.scatter_(
        -1,
        dense_order,
        torch.arange(scores.shape[-1]).view(1, 1, 1, -1).expand_as(dense_order),
    )
    dense_rank = inverse_rank.gather(-1, candidates)
    target_member = target_membership(candidates, target)
    current_member = (
        candidates.unsqueeze(-1) == current[:, None, :, None, :]
    ).any(-1)
    dense = select(base, candidates, dense_rank)
    perfect = select(
        base + target_member.float() * 1_000_000.0,
        candidates,
        dense_rank,
    )
    survivor_oracle = select(
        base + (target_member & current_member).float() * 1_000_000.0,
        candidates,
        dense_rank,
    )
    novel_oracle = select(
        base + (target_member & ~current_member).float() * 1_000_000.0,
        candidates,
        dense_rank,
    )

    masks = split_masks(request_index)
    conditions = {
        "dense_baseline": dense,
        "survivor_oracle_base_fill": survivor_oracle,
        "novel_oracle_base_fill": novel_oracle,
        "full_c32_oracle": perfect,
    }
    result = {
        "schema": "cachetrajectory_c32_survivor_oracle_audit_v1",
        "conditions": {
            condition: {
                split: metrics(
                    selected,
                    current,
                    target,
                    valid,
                    request_index,
                    masks[split],
                )
                for split in ("training", "calibration", "design", "blind", "full")
            }
            for condition, selected in conditions.items()
        },
        "lineage": {
            "audit_sha256": sha256_file(args.audit),
            "reference_sha256": sha256_file(args.reference),
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
        "training_started": False,
        "optimizer_constructed": False,
        "new_capture": False,
        "formal_validation_opened": False,
        "sealed_test_opened": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
