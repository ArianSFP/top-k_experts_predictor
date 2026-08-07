from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from harp8.jspace_data import (
    AlignedJCandidateData,
    move_tensor_batch,
    ordered_prefetched_batches,
    pin_tensor_batch,
    slice_tensor_batch,
)

def _write_pool(root: Path, request_ids: list[int], within: list[int]) -> None:
    root.mkdir()
    rows, h, layers, candidates, experts = len(request_ids), 2, 2, 10, 12
    shape = (rows, h, layers, candidates)
    arrays = {
        "candidate_scores.f32": ("<f4", np.zeros(shape, np.float32)),
        "candidate_ids.u2": ("<u2", np.broadcast_to(np.arange(candidates, dtype=np.uint16), shape)),
        "target_membership.u1": ("u1", np.zeros(shape, np.uint8)),
        "teacher_candidate_scores.f32": ("<f4", np.zeros(shape, np.float32)),
        "valid_future.u1": ("u1", np.ones((rows, h), np.uint8)),
        "current_scores.f32": ("<f4", np.zeros(shape, np.float32)),
        "current_rank.f32": ("<f4", np.zeros(shape, np.float32)),
        "source_gates.f16": ("<f2", np.zeros((rows, h, layers, 3), np.float16)),
        "copy_gates.f16": ("<f2", np.zeros(shape, np.float16)),
    }
    arrays["target_membership.u1"][1][..., :8] = 1
    for name, (dtype, value) in arrays.items():
        mm = np.memmap(root / name, mode="w+", dtype=dtype, shape=value.shape)
        mm[...] = value
        mm.flush()
    (root / "manifest.json").write_text(json.dumps({
        "schema": "harp8_candidate_pool_v1", "rows": rows, "horizons": h,
        "layers": layers, "experts": experts, "candidate_count": candidates,
        "model_width": 8, "store_context": False,
    }) + "\n", encoding="utf-8")
    (root / "metadata.json").write_text(json.dumps({
        "request_ids": request_ids, "within": within, "domains": ["x"] * rows,
    }) + "\n", encoding="utf-8")


def test_aligned_data_join_and_prefetch_preserve_exact_shuffled_rows(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    mtp = tmp_path / "mtp"
    capture.mkdir(); mtp.mkdir()
    request_ids = [91, 17]
    (capture / "requests.jsonl").write_text("".join(
        json.dumps({"request_id": value}) + "\n" for value in request_ids
    ), encoding="utf-8")
    rows_per_request, layers, experts, width = 4, 2, 12, 5
    rows = len(request_ids) * rows_per_request
    router = np.arange(rows * layers * experts, dtype=np.float32).reshape(rows, layers, experts)
    np.save(capture / "raw_router_logits.npy", router)
    np.save(capture / "top8_expert_ids.npy", np.argsort(-router, axis=-1)[..., :8].astype(np.uint16))
    features = np.arange(rows * layers * width, dtype=np.float32).reshape(rows, layers, width)
    feature_path = tmp_path / "features.npy"; np.save(feature_path, features)
    np.save(mtp / "mtp_hidden_depths.npy", np.ones((rows, 3, 7), np.float16))
    np.save(mtp / "mtp_router_logits_depths.npy", np.ones((rows, 3, experts), np.float32))
    pool = tmp_path / "pool"
    # Reverse request order to prove that this is an ID join, not a row join.
    _write_pool(pool, [17, 91], [2, 0])
    data = AlignedJCandidateData(
        pool, capture_dir=capture, mtp_dir=mtp, target_features=feature_path,
        rows_per_request=rows_per_request, history=3,
    )
    batch = data.batch(np.asarray([0, 1]), "cpu")
    assert batch["capture_row"].tolist() == [6, 0]
    assert batch["history_available"].tolist() == [[True, True, True], [True, False, False]]
    assert np.array_equal(batch["target_history"][0, 0].numpy(), features[6])
    assert np.count_nonzero(batch["target_history"][1, 1:].numpy()) == 0
    assert batch["candidate_route_scores"].shape == (2, 2, 2, 10, 3)
    assert batch["candidate_history_membership"].shape == (2, 2, 2, 10, 3)
    assert batch["j_states"].shape == (2, 3, 2, 5)
    assert batch["j_mask"].shape == (2, 3, 2)
    assert batch["candidate_features"].shape == (2, 2, 2, 10, 13)

    shuffled = np.asarray([1, 0], dtype=np.int64)
    full = data.batch(
        shuffled,
        "cpu",
        active_horizons=1,
        include_context=False,
    )
    compact = data.batch(
        shuffled,
        "cpu",
        active_horizons=1,
        include_context=False,
        compact=True,
    )
    for name, value in compact.items():
        assert torch.equal(value, full[name]), name
    full_bytes = sum(value.numel() * value.element_size() for value in full.values())
    compact_bytes = sum(
        value.numel() * value.element_size() for value in compact.values()
    )
    assert compact_bytes < full_bytes

    # Group materialization followed by microbatch views must equal the old
    # one-row gather path bit for bit, including a non-monotonic row order.
    for local, row in enumerate(shuffled.tolist()):
        sliced = slice_tensor_batch(compact, local, local + 1)
        direct = data.batch(
            np.asarray([row], dtype=np.int64),
            "cpu",
            active_horizons=1,
            include_context=False,
            compact=True,
        )
        assert sliced.keys() == direct.keys()
        for name in direct:
            assert torch.equal(sliced[name], direct[name]), name

    direct_batches = [
        data.batch(
            np.asarray([row], dtype=np.int64), "cpu",
            active_horizons=1, include_context=False, compact=True,
        )
        for row in shuffled.tolist()
    ]
    grouped_batches = [
        slice_tensor_batch(compact, index, index + 1)
        for index in range(len(shuffled))
    ]

    def dropout_trace(
        batches: list[dict[str, torch.Tensor]],
    ) -> tuple[list[float], torch.Tensor, torch.Tensor]:
        projection = torch.nn.Linear(5, 1, bias=False)
        with torch.no_grad():
            projection.weight.copy_(
                torch.arange(1, 6, dtype=torch.float32).view(1, 5)
            )
        torch.manual_seed(319)
        losses: list[float] = []
        for values in batches:
            hidden = torch.nn.functional.dropout(
                values["j_states"].float(), p=0.25, training=True
            )
            loss = projection(hidden).sum()
            losses.append(float(loss.detach()))
            loss.backward()
        assert projection.weight.grad is not None
        return losses, projection.weight.grad.clone(), torch.get_rng_state().clone()

    direct_trace = dropout_trace(direct_batches)
    grouped_trace = dropout_trace(grouped_batches)

    row_batches = [
        np.asarray([1], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
    ]
    prefetched = list(ordered_prefetched_batches(
        data,
        row_batches,
        active_horizons=1,
        include_context=False,
        compact=True,
        prefetch=True,
        pin_memory=False,
    ))
    assert [rows.tolist() for rows, _values in prefetched] == [[1], [0]]
    for (rows, values), expected_row in zip(prefetched, (1, 0), strict=True):
        direct = data.batch(
            np.asarray([expected_row], dtype=np.int64), "cpu",
            active_horizons=1, include_context=False, compact=True,
        )
        moved = move_tensor_batch(values, "cpu")
        assert rows.tolist() == [expected_row]
        for name in direct:
            assert torch.equal(moved[name], direct[name]), name


def test_pin_tensor_batch_materializes_expanded_views_only(monkeypatch) -> None:
    expanded = torch.arange(3, dtype=torch.float32).view(1, 3).expand(4, 3)
    contiguous = torch.arange(12, dtype=torch.float32).view(4, 3).clone()
    assert not expanded.is_contiguous()
    assert contiguous.is_contiguous()
    original_pointers = {
        "expanded": expanded.data_ptr(),
        "contiguous": contiguous.data_ptr(),
    }
    pinned_sources: list[tuple[int, bool]] = []

    def fake_pin_memory(value: torch.Tensor) -> torch.Tensor:
        pinned_sources.append((value.data_ptr(), value.is_contiguous()))
        return value.clone()

    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin_memory)
    pinned = pin_tensor_batch({
        "expanded": expanded,
        "contiguous": contiguous,
    })

    assert all(is_contiguous for _pointer, is_contiguous in pinned_sources)
    assert pinned_sources[0][0] != original_pointers["expanded"]
    assert pinned_sources[1][0] == original_pointers["contiguous"]
    assert pinned["expanded"].shape == expanded.shape
    assert torch.equal(pinned["expanded"], expanded)
    assert torch.equal(pinned["contiguous"], contiguous)
