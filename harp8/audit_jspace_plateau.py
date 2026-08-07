"""Read-only diagnostics for a residual J-HARP candidate-ranker plateau.

This module deliberately does not train or rewrite an artifact.  It checks the
immutable capture's stable top-k contract, measures the frozen candidate-pool
ceiling, and diagnoses whether the four preregistered surrogate losses point in
compatible score-space directions at the frozen-base initialization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .jspace_reranker import JSpaceRerankerLossConfig, jspace_reranker_loss


def stable_topk(
    scores: np.ndarray, k: int, *, expert_ids: np.ndarray | None = None
) -> np.ndarray:
    """Return top-k indices with the capture's stable expert-ID tie rule.

    ``argpartition`` is intentionally forbidden here: BF16 router logits have
    exact ties at the top-8 boundary, and partitioning may choose a different
    tied expert even though the stored authoritative route is correct.
    """

    values = np.asarray(scores)
    if values.ndim < 1 or not 1 <= k <= values.shape[-1]:
        raise ValueError("k must lie within the score axis")
    if expert_ids is None:
        return np.argsort(-values, axis=-1, kind="stable")[..., :k]
    ids = np.asarray(expert_ids)
    if ids.shape != values.shape:
        raise ValueError("expert_ids must match scores")
    # np.lexsort uses the last key as primary: descending score first, then
    # ascending expert ID to reproduce a stable sort in the full namespace.
    return np.lexsort((ids, -values), axis=-1)[..., :k]


def recall_from_membership(
    membership: np.ndarray, selected_indices: np.ndarray, k: int
) -> float:
    labels = np.asarray(membership, dtype=np.bool_)
    chosen = np.asarray(selected_indices, dtype=np.int64)
    if chosen.shape != labels.shape[:-1] + (k,):
        raise ValueError("selected_indices geometry disagrees with membership")
    hits = np.take_along_axis(labels, chosen, axis=-1).sum(axis=-1)
    return float(np.mean(hits / k))


def _pool_arrays(root: Path) -> tuple[dict[str, Any], dict[str, np.memmap]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    shape = (
        int(manifest["rows"]),
        int(manifest["horizons"]),
        int(manifest["layers"]),
        int(manifest["candidate_count"]),
    )
    arrays = {
        "base": np.memmap(root / "candidate_scores.f32", mode="r", dtype="<f4", shape=shape),
        "teacher": np.memmap(root / "teacher_candidate_scores.f32", mode="r", dtype="<f4", shape=shape),
        "ids": np.memmap(root / "candidate_ids.u2", mode="r", dtype="<u2", shape=shape),
        "membership": np.memmap(root / "target_membership.u1", mode="r", dtype="u1", shape=shape),
        "valid": np.memmap(
            root / "valid_future.u1",
            mode="r",
            dtype="u1",
            shape=(shape[0], shape[1]),
        ),
    }
    return manifest, arrays


def capture_topk_audit(capture_dir: Path, rows: np.ndarray, native_k: int) -> dict[str, Any]:
    logits = np.load(capture_dir / "raw_router_logits.npy", mmap_mode="r", allow_pickle=False)
    topk = np.load(capture_dir / "top8_expert_ids.npy", mmap_mode="r", allow_pickle=False)
    values = np.asarray(logits[rows], dtype=np.float32)
    authoritative = np.asarray(topk[rows], dtype=np.int64)
    predicted = stable_topk(values, native_k)
    predicted_ids = predicted
    overlap = (
        predicted_ids[..., :, None] == authoritative[..., None, :]
    ).any(axis=-1).sum(axis=-1)
    ordered = np.sort(values, axis=-1)
    cutoff_tie = ordered[..., -native_k] == ordered[..., -native_k - 1]
    return {
        "rows": int(len(rows)),
        "slot_recall": float(np.mean(overlap / native_k)),
        "exact_set_rate": float(np.mean(overlap == native_k)),
        "cutoff_tie_rate": float(np.mean(cutoff_tie)),
        "stable_sort_required": True,
    }


def pool_score_audit(
    pool_root: Path, rows: np.ndarray, active_horizons: int = 4
) -> dict[str, Any]:
    manifest, arrays = _pool_arrays(pool_root)
    native_k = int(manifest.get("native_k", 8))
    base = np.asarray(arrays["base"][rows, :active_horizons], dtype=np.float32)
    teacher = np.asarray(arrays["teacher"][rows, :active_horizons], dtype=np.float32)
    ids = np.asarray(arrays["ids"][rows, :active_horizons], dtype=np.int64)
    membership = np.asarray(
        arrays["membership"][rows, :active_horizons], dtype=np.bool_
    )
    valid = np.asarray(arrays["valid"][rows, :active_horizons], dtype=np.bool_)
    horizon_rows: list[dict[str, Any]] = []
    for column in range(active_horizons):
        keep = valid[:, column]
        b = base[:, column][keep]
        t = teacher[:, column][keep]
        e = ids[:, column][keep]
        m = membership[:, column][keep]
        base_order = stable_topk(b, native_k)
        teacher_order = stable_topk(t, native_k, expert_ids=e)
        centered_base = b - b.mean(axis=-1, keepdims=True)
        centered_teacher = t - t.mean(axis=-1, keepdims=True)
        correlation = np.sum(centered_base * centered_teacher, axis=-1) / (
            np.sqrt(
                np.sum(centered_base**2, axis=-1)
                * np.sum(centered_teacher**2, axis=-1)
            )
            + 1e-12
        )
        sorted_teacher = np.sort(t, axis=-1)
        horizon_rows.append(
            {
                "horizon": column + 1,
                "valid_rows": int(keep.sum()),
                "base_recall_at_8": recall_from_membership(m, base_order, native_k),
                "coverage_at_candidate_count": float(m.sum(axis=-1).mean() / native_k),
                "teacher_stable_recall_at_8": recall_from_membership(
                    m, teacher_order, native_k
                ),
                "teacher_cutoff_tie_rate": float(
                    np.mean(
                        sorted_teacher[..., -native_k]
                        == sorted_teacher[..., -native_k - 1]
                    )
                ),
                "base_teacher_centered_pearson": float(correlation.mean()),
            }
        )

    # Score-space loss gradients at the exact frozen-base initialization.
    predicted = torch.tensor(base, requires_grad=True)
    tensor_batch = {
        "target_membership": torch.tensor(membership),
        "teacher_candidate_scores": torch.tensor(teacher),
        "valid_future": torch.tensor(valid),
        "candidate_mask": torch.ones_like(torch.tensor(membership), dtype=torch.bool),
    }
    defaults = JSpaceRerankerLossConfig(
        horizon_weights=JSpaceRerankerLossConfig().horizon_weights[:active_horizons]
    )
    loss = jspace_reranker_loss(predicted, tensor_batch, defaults)
    gradients: dict[str, torch.Tensor] = {}
    for name, component in loss.components.items():
        gradients[name] = torch.autograd.grad(
            component, predicted, retain_graph=True
        )[0].detach().double().flatten()
    coefficients = {
        "boundary": defaults.boundary,
        "balanced_bce": defaults.balanced_bce,
        "listwise": defaults.listwise,
        "restricted_kl": defaults.restricted_kl,
    }
    cosines: dict[str, float] = {}
    names = list(gradients)
    for offset, left in enumerate(names):
        for right in names[offset + 1 :]:
            denominator = torch.linalg.vector_norm(gradients[left]) * torch.linalg.vector_norm(
                gradients[right]
            )
            cosines[f"{left}__{right}"] = float(
                torch.dot(gradients[left], gradients[right]) / denominator.clamp_min(1e-30)
            )
    return {
        "rows": int(len(rows)),
        "horizons": horizon_rows,
        "initial_loss_components": {
            name: float(value.detach()) for name, value in loss.components.items()
        },
        "initial_weighted_gradient_norms": {
            name: float(coefficients[name] * torch.linalg.vector_norm(value))
            for name, value in gradients.items()
        },
        "initial_gradient_cosines": cosines,
    }


def _rank_metrics(
    scores: torch.Tensor, batch: dict[str, torch.Tensor], native_k: int = 8
) -> dict[str, Any]:
    candidate_mask = batch.get("candidate_mask")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(batch["target_membership"], dtype=torch.bool)
    predicted = scores.masked_fill(~candidate_mask, -torch.inf).topk(native_k, dim=-1).indices
    frozen = batch["candidate_scores"].masked_fill(
        ~candidate_mask, -torch.inf
    ).topk(native_k, dim=-1).indices
    membership = batch["target_membership"].bool()
    recall = membership.gather(-1, predicted).sum(dim=-1).float() / native_k
    base_recall = membership.gather(-1, frozen).sum(dim=-1).float() / native_k
    swaps = (
        predicted.unsqueeze(-1) != frozen.unsqueeze(-2)
    ).all(dim=-1).sum(dim=-1).float()
    return {
        "mean_recall_at_8": float(recall.mean()),
        "base_mean_recall_at_8": float(base_recall.mean()),
        "recall_at_8_by_horizon": [
            float(value) for value in recall.mean(dim=(0, 2))
        ],
        "mean_swaps_from_base_top8": float(swaps.mean()),
    }


def tiny_real_overfit(
    train_pool: Path,
    *,
    capture_dir: Path,
    mtp_dir: Path,
    target_features: Path,
    target_feature_rms: Path,
    router_keys_path: Path,
    rows: int = 4,
    steps: int = 60,
    seed: int = 1234,
    learning_rate: float = 3e-3,
) -> dict[str, Any]:
    """Compare loss profiles on a deliberately tiny, real, CPU overfit."""

    from .jspace_data import AlignedJCandidateData
    from .jspace_reranker import JSpaceCandidateReranker
    from .train_jspace_reranker import derive_model_config, prepare_model_batch

    data = AlignedJCandidateData(
        train_pool,
        capture_dir=capture_dir,
        mtp_dir=mtp_dir,
        target_features=target_features,
        target_feature_rms=target_feature_rms,
    )
    valid = np.asarray(data.pool.valid_future[:, :4], dtype=np.bool_)
    eligible = np.flatnonzero(valid.all(axis=1))
    if rows <= 0 or rows > len(eligible):
        raise ValueError("tiny-overfit row count is invalid")
    selected_rows = eligible[np.linspace(0, len(eligible) - 1, rows, dtype=np.int64)]
    batch = prepare_model_batch(
        data.batch(selected_rows, "cpu", active_horizons=4, include_context=False),
        include_feature_rms=True,
        include_candidate_features=True,
        active_horizons=4,
    )
    router_keys = torch.from_numpy(
        np.array(np.load(router_keys_path, mmap_mode="r", allow_pickle=False), copy=True)
    ).float()
    model_config = derive_model_config(
        data,
        router_keys,
        include_feature_rms=True,
        include_candidate_features=True,
        active_horizons=4,
        overrides={
            "model_width": 32,
            "attention_heads": 4,
            "feedforward_width": 64,
            "expert_embedding_width": 16,
            "temporal_blocks": 1,
            "axial_blocks": 1,
            "mtp_blocks": 1,
            "set_blocks": 1,
            "inducing_points": 4,
            "dropout": 0.0,
        },
    )
    horizon_weights = (1.0, 1.0, 1.25, 1.5)
    profiles = {
        "preregistered_v1": JSpaceRerankerLossConfig(horizon_weights=horizon_weights),
        "no_restricted_kl": JSpaceRerankerLossConfig(
            restricted_kl=0.0, horizon_weights=horizon_weights
        ),
        "membership_ranking_v1": JSpaceRerankerLossConfig(
            boundary=1.0,
            balanced_bce=0.0,
            listwise=1.0,
            restricted_kl=0.0,
            horizon_weights=horizon_weights,
        ),
    }
    result: dict[str, Any] = {
        "rows": selected_rows.tolist(),
        "steps": int(steps),
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "model_config": model_config.to_dict(),
        "candidate_coverage": float(
            batch["target_membership"].sum(dim=-1).float().mean() / 8
        ),
        "profiles": {},
    }
    for name, loss_config in profiles.items():
        torch.manual_seed(seed)
        model = JSpaceCandidateReranker(model_config, router_keys)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.0
        )
        snapshots: dict[str, Any] = {}
        for step in range(steps + 1):
            if step in {0, 1, 2, 5, 10, 20, 40, steps}:
                model.eval()
                with torch.inference_mode():
                    evaluation = model(batch)
                snapshots[str(step)] = {
                    **_rank_metrics(evaluation.scores, batch),
                    "delta_rms": float(evaluation.delta.float().square().mean().sqrt()),
                }
            if step == steps:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = jspace_reranker_loss(output, batch, loss_config)
            loss.total.backward()
            optimizer.step()
        result["profiles"][name] = {
            "loss_config": loss_config.to_dict(),
            "snapshots": snapshots,
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active-horizons", type=int, default=4)
    parser.add_argument("--tiny-overfit-pool", type=Path)
    parser.add_argument("--mtp-dir", type=Path)
    parser.add_argument("--target-features", type=Path)
    parser.add_argument("--target-feature-rms", type=Path)
    parser.add_argument("--router-keys", type=Path)
    parser.add_argument("--tiny-overfit-rows", type=int, default=4)
    parser.add_argument("--tiny-overfit-steps", type=int, default=60)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = json.loads((args.pool_root / "manifest.json").read_text(encoding="utf-8"))
    pool_rows = int(manifest["rows"])
    capture = np.load(
        args.capture_dir / "raw_router_logits.npy", mmap_mode="r", allow_pickle=False
    )
    if args.sample_rows <= 0:
        raise ValueError("sample_rows must be positive")
    rng = np.random.default_rng(args.seed)
    pool_sample = np.sort(
        rng.choice(pool_rows, size=min(args.sample_rows, pool_rows), replace=False)
    )
    capture_sample = np.sort(
        rng.choice(len(capture), size=min(args.sample_rows, len(capture)), replace=False)
    )
    result = {
        "schema": "harp8_jspace_plateau_audit_v1",
        "seed": int(args.seed),
        "capture_topk": capture_topk_audit(
            args.capture_dir, capture_sample, int(manifest.get("native_k", 8))
        ),
        "pool": pool_score_audit(args.pool_root, pool_sample, args.active_horizons),
    }
    if args.tiny_overfit_pool is not None:
        required = {
            "mtp_dir": args.mtp_dir,
            "target_features": args.target_features,
            "target_feature_rms": args.target_feature_rms,
            "router_keys": args.router_keys,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"tiny overfit requires {missing}")
        result["tiny_real_overfit"] = tiny_real_overfit(
            args.tiny_overfit_pool,
            capture_dir=args.capture_dir,
            mtp_dir=args.mtp_dir,
            target_features=args.target_features,
            target_feature_rms=args.target_feature_rms,
            router_keys_path=args.router_keys,
            rows=args.tiny_overfit_rows,
            steps=args.tiny_overfit_steps,
        )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "capture_topk_audit",
    "pool_score_audit",
    "recall_from_membership",
    "stable_topk",
    "tiny_real_overfit",
]
