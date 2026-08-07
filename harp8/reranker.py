"""Conditional permutation-equivariant reranking of a fixed candidate pool."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .candidates import POOL_SCHEMA
from .train import sha256_file


RERANKER_SCHEMA = "harp8_fixed16_reranker_v1"


class CandidatePool:
    """Memory-mapped candidate pool with fail-closed shape checks."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != POOL_SCHEMA:
            raise ValueError("candidate pool has an incompatible schema")
        self.manifest = manifest
        self.rows = int(manifest["rows"])
        self.horizons = int(manifest["horizons"])
        self.layers = int(manifest["layers"])
        self.experts = int(manifest["experts"])
        self.candidate_count = int(manifest["candidate_count"])
        self.model_width = int(manifest["model_width"])
        shape = (self.rows, self.horizons, self.layers, self.candidate_count)
        self.candidate_scores = np.memmap(root / "candidate_scores.f32", mode="r", dtype="<f4", shape=shape)
        self.candidate_ids = np.memmap(root / "candidate_ids.u2", mode="r", dtype="<u2", shape=shape)
        self.target_membership = np.memmap(root / "target_membership.u1", mode="r", dtype="u1", shape=shape)
        self.teacher_candidate_scores = np.memmap(root / "teacher_candidate_scores.f32", mode="r", dtype="<f4", shape=shape)
        self.valid_future = np.memmap(root / "valid_future.u1", mode="r", dtype="u1", shape=(self.rows, self.horizons))
        self.current_scores = np.memmap(root / "current_scores.f32", mode="r", dtype="<f4", shape=(self.rows, self.horizons, self.layers, self.candidate_count))
        self.current_rank = np.memmap(root / "current_rank.f32", mode="r", dtype="<f4", shape=(self.rows, self.horizons, self.layers, self.candidate_count))
        self.source_gates = np.memmap(root / "source_gates.f16", mode="r", dtype="<f2", shape=(self.rows, self.horizons, self.layers, 3))
        self.copy_gates = np.memmap(root / "copy_gates.f16", mode="r", dtype="<f2", shape=shape)
        context_path = root / "generator_context.f16"
        self.context = (
            np.memmap(context_path, mode="r", dtype="<f2", shape=(self.rows, self.horizons, self.layers, self.model_width))
            if manifest.get("store_context") else None
        )
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        self.request_ids = np.asarray(metadata["request_ids"], dtype=np.int64)
        self.within = np.asarray(metadata["within"], dtype=np.int32)
        self.domains = np.asarray(metadata["domains"], dtype=object)
        if len(self.request_ids) != self.rows:
            raise ValueError("candidate metadata row count disagrees with manifest")

    def batch(self, rows: np.ndarray, device: str) -> dict[str, torch.Tensor]:
        values = {
            "candidate_scores": np.asarray(self.candidate_scores[rows], dtype=np.float32),
            "candidate_ids": np.asarray(self.candidate_ids[rows], dtype=np.int64),
            "target_membership": np.asarray(self.target_membership[rows], dtype=np.float32),
            "teacher_candidate_scores": np.asarray(self.teacher_candidate_scores[rows], dtype=np.float32),
            "valid_future": np.asarray(self.valid_future[rows], dtype=np.bool_),
            "current_scores": np.asarray(self.current_scores[rows], dtype=np.float32),
            "current_rank": np.asarray(self.current_rank[rows], dtype=np.float32),
            "source_gates": np.asarray(self.source_gates[rows], dtype=np.float32),
            "copy_gates": np.asarray(self.copy_gates[rows], dtype=np.float32),
        }
        if self.context is not None:
            values["context"] = np.asarray(self.context[rows], dtype=np.float32)
        return {name: torch.as_tensor(value, device=device) for name, value in values.items()}


class Fixed16SetReranker(nn.Module):
    """Transformer over an unordered candidate set, with no candidate positions."""

    def __init__(
        self,
        *,
        experts: int,
        layers: int,
        candidate_count: int = 16,
        context_width: int = 384,
        hidden_width: int = 256,
        object_width: int = 64,
        heads: int = 8,
        blocks: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden_width % heads:
            raise ValueError("reranker hidden width must divide attention heads")
        self.experts = experts
        self.layers = layers
        self.candidate_count = candidate_count
        self.context_width = context_width
        self.hidden_width = hidden_width
        self.object_embedding = nn.Embedding(layers * experts, object_width)
        self.scalar_projection = nn.Sequential(
            nn.LayerNorm(10), nn.Linear(10, hidden_width - object_width), nn.SiLU()
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(context_width), nn.Linear(context_width, hidden_width - object_width), nn.SiLU()
        )
        self.input_projection = nn.Sequential(
            nn.Linear(hidden_width - object_width, hidden_width - object_width), nn.SiLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_width, nhead=heads, dim_feedforward=hidden_width * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=blocks)
        self.norm = nn.LayerNorm(hidden_width)
        self.delta = nn.Sequential(nn.Linear(hidden_width, hidden_width), nn.SiLU(), nn.Linear(hidden_width, 1))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.object_embedding.weight, std=0.02)
        nn.init.zeros_(self.delta[-1].bias)
        nn.init.zeros_(self.delta[-1].weight)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        base = batch["candidate_scores"].float()
        ids = batch["candidate_ids"].long()
        b, h, l, c = ids.shape
        if c != self.candidate_count:
            raise ValueError("candidate pool width disagrees with reranker")
        rank = torch.argsort(torch.argsort(-base, dim=-1), dim=-1).to(base.dtype) / max(1, c - 1)
        base_center = base - base.mean(dim=-1, keepdim=True)
        margin8 = base - base[..., 7:8]
        margin16 = base - base[..., -1:]
        current = batch["current_scores"].float()
        current_rank = batch["current_rank"].float()
        if current.ndim == 3:
            current = current[:, None].expand(-1, h, -1, -1)
            current_rank = current_rank[:, None].expand(-1, h, -1, -1)
        current = current - current.mean(dim=-1, keepdim=True)
        gates = batch["source_gates"].float().unsqueeze(-2).expand(-1, -1, -1, c, -1)
        scalars = torch.cat([
            base_center[..., None], margin8[..., None], margin16[..., None], rank[..., None],
            current[..., None], current_rank[..., None], batch["copy_gates"].float()[..., None], gates,
        ], dim=-1)
        object_ids = ids + torch.arange(l, device=ids.device).view(1, 1, l, 1) * self.experts
        objects = self.object_embedding(object_ids)
        scalar = self.scalar_projection(scalars)
        if "context" in batch:
            context = self.context_projection(batch["context"].float()).unsqueeze(-2).expand(-1, -1, -1, c, -1)
        else:
            context = torch.zeros_like(scalar[..., : self.hidden_width - self.object_embedding.embedding_dim])
        content = self.input_projection(scalar + context)
        hidden = torch.cat([objects, content], dim=-1)
        flat = hidden.reshape(b * h * l, c, self.hidden_width)
        encoded = self.norm(self.encoder(flat))
        delta = self.delta(encoded).reshape(b, h, l, c)
        return base + delta


def reranker_loss(
    predicted: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    temperature: float = 2.0,
    pairwise_weight: float = 0.25,
    distill_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch["valid_future"].bool()[:, :, None]
    membership = batch["target_membership"].bool()
    pos = F.softplus(-predicted).masked_fill(~membership, 0.0)
    neg = F.softplus(predicted).masked_fill(membership, 0.0)
    positive_count = membership.sum(dim=-1).clamp_min(1)
    negative_count = (~membership).sum(dim=-1).clamp_min(1)
    bce_rows = 0.5 * (pos.sum(dim=-1) / positive_count + neg.sum(dim=-1) / negative_count)
    positive = predicted
    negative = predicted
    pair_matrix = F.softplus(negative[..., None, :] - positive[..., :, None])
    pair_mask = membership[..., :, None] & (~membership)[..., None, :]
    pair_rows = (pair_matrix * pair_mask.to(pair_matrix.dtype)).sum(dim=(-1, -2)) / pair_mask.sum(dim=(-1, -2)).clamp_min(1)
    teacher = batch["teacher_candidate_scores"].float()
    distill = F.kl_div(
        torch.log_softmax(predicted / temperature, dim=-1),
        torch.softmax(teacher / temperature, dim=-1), reduction="none",
    ).sum(dim=-1) * temperature**2
    mask = valid.expand_as(bce_rows)
    denom = mask.sum().clamp_min(1)
    total = ((bce_rows * mask).sum() + pairwise_weight * (pair_rows * mask).sum() + distill_weight * (distill * mask).sum()) / denom
    return total, {"bce": float((bce_rows * mask).sum().detach() / denom),
                   "pairwise": float((pair_rows * mask).sum().detach() / denom),
                   "distill": float((distill * mask).sum().detach() / denom),
                   "loss": float(total.detach())}


def evaluate_pool(model: Fixed16SetReranker, pool: CandidatePool, *, batch_size: int, device: str) -> dict[str, Any]:
    request_values: dict[tuple[int, int], list[float]] = {}
    totals = np.zeros(pool.horizons, dtype=np.float64)
    counts = np.zeros(pool.horizons, dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, pool.rows, batch_size):
            rows = np.arange(start, min(pool.rows, start + batch_size), dtype=np.int64)
            batch = pool.batch(rows, device)
            scores = model(batch)
            order = torch.topk(scores, 8, dim=-1).indices
            membership = batch["target_membership"].bool()
            hits = membership.gather(-1, order).sum(dim=-1).float() / 8.0
            valid = batch["valid_future"].bool()[:, :, None].expand(-1, -1, pool.layers)
            hits = hits.cpu().numpy()
            valid_np = valid.cpu().numpy()
            for local, global_row in enumerate(rows.tolist()):
                request_id = int(pool.request_ids[global_row])
                for horizon in range(pool.horizons):
                    if not valid_np[local, horizon, 0]:
                        continue
                    value = float(hits[local, horizon].mean())
                    key = (request_id, horizon)
                    request_values.setdefault(key, []).append(value)
                    totals[horizon] += value
                    counts[horizon] += 1
    horizon_rows = []
    for horizon in range(pool.horizons):
        values = [v for (req, h), row in request_values.items() if h == horizon for v in [float(np.mean(row))]]
        horizon_rows.append({"horizon": horizon + 1, "request_macro_recall_at_8": float(np.mean(values)) if values else 0.0,
                             "micro_recall_at_8": float(totals[horizon] / max(1, counts[horizon]))})
    return {"horizon_metrics": horizon_rows, "mean_h1_h4_recall_at_8": float(np.mean([row["request_macro_recall_at_8"] for row in horizon_rows[:4]]))}


def candidate_gate(pool: CandidatePool) -> dict[str, float | bool]:
    """Check the preregistered fixed-pool gate before reranker training."""
    values = []
    h4 = []
    membership = np.asarray(pool.target_membership, dtype=np.uint8)
    valid = np.asarray(pool.valid_future, dtype=np.uint8)
    for horizon in range(min(4, pool.horizons)):
        row = membership[:, horizon].sum(axis=-1) / 8.0
        row = row[valid[:, horizon].astype(bool)]
        values.append(float(row.mean()) if len(row) else 0.0)
        if horizon == 3:
            h4.append(float(row.mean()) if len(row) else 0.0)
    mean = float(np.mean(values)) if values else 0.0
    h4_value = h4[0] if h4 else 0.0
    return {"mean_h1_h4_coverage_at_16": mean, "h4_coverage_at_16": h4_value,
            "passes": bool(mean >= 0.95 and h4_value >= 0.93)}


def train_reranker(train_pool: CandidatePool, validation_pool: CandidatePool, output_dir: Path, *, device: str = "cuda:0", epochs: int = 20, batch_size: int = 32, learning_rate: float = 3e-4, seed: int = 42, force: bool = False) -> dict[str, Any]:
    gate = candidate_gate(validation_pool)
    if not force and not gate["passes"]:
        raise RuntimeError("fixed-16 candidate gate has not passed; use force only for diagnostics")
    if train_pool.candidate_count != validation_pool.candidate_count:
        raise ValueError("train and validation pools have different widths")
    torch.manual_seed(seed)
    np.random.seed(seed)
    context_width = train_pool.model_width
    model = Fixed16SetReranker(experts=train_pool.experts, layers=train_pool.layers,
                               candidate_count=train_pool.candidate_count,
                               context_width=context_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse reranker directory {output_dir}")
    output_dir.mkdir(parents=True)
    history: list[dict[str, Any]] = []
    best = -math.inf
    best_path = output_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(train_pool.rows)
        losses = []
        for start in range(0, train_pool.rows, batch_size):
            batch = train_pool.batch(order[start:start + batch_size], device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = reranker_loss(model(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(metrics["loss"])
        evaluation = evaluate_pool(model, validation_pool, batch_size=batch_size, device=device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "validation_mean_h1_h4_recall_at_8": evaluation["mean_h1_h4_recall_at_8"]}
        history.append(record)
        print(json.dumps({"event": "harp8_reranker_epoch", **record}, sort_keys=True), flush=True)
        if record["validation_mean_h1_h4_recall_at_8"] > best:
            best = record["validation_mean_h1_h4_recall_at_8"]
            torch.save({"schema": RERANKER_SCHEMA, "model_config": {
                "experts": train_pool.experts, "layers": train_pool.layers,
                "candidate_count": train_pool.candidate_count, "context_width": context_width},
                "model_state": model.state_dict(), "epoch": epoch, "seed": seed,
                "candidate_gate": gate, "train_manifest_sha256": sha256_file(train_pool.root / "manifest.json"),
                "validation_manifest_sha256": sha256_file(validation_pool.root / "manifest.json")}, best_path)
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema": RERANKER_SCHEMA, "best_epoch": int(torch.load(best_path, map_location="cpu", weights_only=False)["epoch"]),
                "candidate_gate": gate, "best_validation_mean_h1_h4_recall_at_8": best,
                "outputs": {"best.pt": sha256_file(best_path), "training_history.json": sha256_file(output_dir / "training_history.json")}}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_pool = CandidatePool(args.train_pool)
    validation_pool = CandidatePool(args.validation_pool)
    manifest = train_reranker(train_pool, validation_pool, args.output_dir, device=args.device,
                               epochs=args.epochs, batch_size=args.batch_size,
                               learning_rate=args.learning_rate, seed=args.seed, force=args.force)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
