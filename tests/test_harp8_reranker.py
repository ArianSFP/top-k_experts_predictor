from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from harp8.candidates import export_candidate_pool
from harp8.config import HARPConfig
from harp8.data import CompactHARPData
from harp8.model import HARP8Teacher
from harp8.reranker import CandidatePool, Fixed16SetReranker, reranker_loss


def _write_pool(root: Path, *, rows: int = 3, horizons: int = 2, layers: int = 2,
                experts: int = 24, candidates: int = 16, width: int = 16) -> Path:
    root.mkdir()
    shape = (rows, horizons, layers, candidates)
    rng = np.random.default_rng(4)
    arrays = {
        "candidate_scores.f32": ("<f4", shape, rng.normal(size=shape).astype(np.float32)),
        "candidate_ids.u2": ("<u2", shape, np.broadcast_to(np.arange(candidates, dtype=np.uint16), shape).copy()),
        "target_membership.u1": ("u1", shape, np.zeros(shape, dtype=np.uint8)),
        "teacher_candidate_scores.f32": ("<f4", shape, rng.normal(size=shape).astype(np.float32)),
        "valid_future.u1": ("u1", (rows, horizons), np.ones((rows, horizons), dtype=np.uint8)),
        "current_scores.f32": ("<f4", shape, rng.normal(size=shape).astype(np.float32)),
        "current_rank.f32": ("<f4", shape, rng.random(size=shape).astype(np.float32)),
        "source_gates.f16": ("<f2", (rows, horizons, layers, 3), rng.random((rows, horizons, layers, 3)).astype(np.float16)),
        "copy_gates.f16": ("<f2", shape, rng.random(size=shape).astype(np.float16)),
        "generator_context.f16": ("<f2", (rows, horizons, layers, width), rng.normal(size=(rows, horizons, layers, width)).astype(np.float16)),
    }
    membership = arrays["target_membership.u1"][2]
    membership[..., :8] = 1
    for name, (dtype, array_shape, value) in arrays.items():
        output = np.memmap(root / name, mode="w+", dtype=dtype, shape=array_shape)
        output[...] = value
        output.flush()
    (root / "metadata.json").write_text(json.dumps({
        "request_ids": [1, 1, 2], "within": [0, 1, 0], "domains": ["a", "a", "b"]
    }) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({
        "schema": "harp8_candidate_pool_v1", "rows": rows, "horizons": horizons,
        "layers": layers, "experts": experts, "native_k": 8,
        "candidate_count": candidates, "model_width": width, "store_context": True,
    }) + "\n", encoding="utf-8")
    return root


def test_reranker_forward_loss_and_permutation_equivariance(tmp_path: Path) -> None:
    pool = CandidatePool(_write_pool(tmp_path / "pool"))
    batch = pool.batch(np.arange(pool.rows), "cpu")
    model = Fixed16SetReranker(experts=pool.experts, layers=pool.layers,
                               candidate_count=pool.candidate_count,
                               context_width=pool.model_width, hidden_width=32,
                               object_width=8, heads=4, blocks=1, dropout=0.0).eval()
    with torch.inference_mode():
        original = model(batch)
    loss, metrics = reranker_loss(original, batch)
    assert torch.isfinite(loss)
    assert set(metrics) == {"bce", "pairwise", "distill", "loss"}
    permutation = torch.tensor([3, 0, 15, 4, 1, 12, 9, 2, 10, 7, 6, 5, 8, 11, 13, 14])
    permuted = {name: value.clone() for name, value in batch.items()}
    for name in ("candidate_scores", "candidate_ids", "target_membership",
                 "teacher_candidate_scores", "current_scores", "current_rank", "copy_gates"):
        permuted[name] = permuted[name].index_select(-1, permutation)
    with torch.inference_mode():
        reordered = model(permuted).index_select(-1, torch.argsort(permutation))
    assert torch.allclose(original, reordered, atol=1e-5)


def test_candidate_pool_rejects_wrong_schema(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    try:
        CandidatePool(root)
    except ValueError as exc:
        assert "schema" in str(exc)
    else:
        raise AssertionError("wrong candidate schema was accepted")


def test_candidate_exporter_writes_horizon_aligned_pool(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    mtp = tmp_path / "mtp"
    capture.mkdir()
    mtp.mkdir()
    config = HARPConfig(
        experts=16, layers=2, horizons=2, route_history=2, mtp_depths=2,
        target_state_channels=1, target_state_width=4, mtp_state_channels=1,
        mtp_state_width=4, mtp_state_projection_width=8, mtp_metadata_width=4,
        mtp_vocab_width=0, route_width=16, state_width=16, mtp_width=16,
        model_width=16, route_ffn_width=32, mtp_ffn_width=32,
        fusion_ffn_width=32, attention_heads=4, temporal_blocks=1,
        layer_blocks=1, state_blocks=1, mtp_cross_blocks=1, fusion_blocks=1,
        dropout=0.0, future_latent_width=0,
    )
    rows_per_request = 4
    rows = 3 * rows_per_request
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(rows, config.layers, config.experts)).astype(np.float32)
    np.save(capture / "raw_router_logits.npy", raw)
    np.save(capture / "top8_expert_ids.npy", np.argsort(-raw, axis=-1)[..., :8].astype(np.uint16))
    np.save(mtp / "mtp_router_logits_depths.npy", rng.normal(size=(rows, 2, config.experts)).astype(np.float32))
    target = rng.normal(size=(rows, config.layers, 4)).astype(np.float16)
    states = rng.normal(size=(rows, 2, 4)).astype(np.float16)
    requests = [
        {"request_id": 1, "offline_split": "train", "domain": "a"},
        {"request_id": 2, "offline_split": "validation", "domain": "a"},
        {"request_id": 3, "offline_split": "test", "domain": "b"},
    ]
    (capture / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8")
    data = CompactHARPData(capture, mtp, target, states, config, rows_per_request=rows_per_request)
    model = HARP8Teacher(config).eval()
    checkpoint = tmp_path / "best.pt"
    torch.save({"schema": "harp8t_training_v1", "model_config": config.to_dict(),
                "model_state": model.state_dict(), "epoch": 1, "seed": 1}, checkpoint)
    output = tmp_path / "pool-out"
    manifest = export_candidate_pool(data, checkpoint, output, split="validation",
                                     candidate_count=16, batch_size=2, device="cpu")
    assert manifest["rows"] == 3
    pool = CandidatePool(output)
    assert pool.current_scores.shape == (3, 2, 2, 16)
    assert pool.candidate_ids.shape == (3, 2, 2, 16)
    assert pool.valid_future.shape == (3, 2)
