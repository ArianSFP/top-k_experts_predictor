from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import pytest
import torch
from torch.utils.data import Dataset
from safetensors.torch import save_file

from harp_rtt.counterfactual import (
    CounterfactualDatasetAdapter,
    assert_counterfactual_access_allowed,
    audit_counterfactual_geometry,
    empty_counterfactual_tensors,
    select_counterfactual_paths,
    load_counterfactual_companion,
    selector_sha256,
)
from harp_rtt.geometry import build_centered_router_geometry


@dataclass
class Node:
    local_index: int
    parent_local_index: int | None
    depth: int
    token_id: int
    token_rank_under_parent: int
    token_path_ids: tuple[int, ...]
    token_path_log_probabilities: tuple[float, ...]
    path_log_probability: float


def _node(
    local: int,
    parent: int | None,
    path: tuple[int, ...],
    ranks: tuple[int, ...],
    logps: tuple[float, ...],
) -> Node:
    return Node(
        local,
        parent,
        len(path),
        path[-1],
        ranks[-1],
        path,
        logps,
        sum(logps),
    )


def test_divergence_selector_balances_h2_h3_h4_without_labels() -> None:
    nodes = [
        _node(0, None, (10,), (0,), (0.0,)),
        _node(1, 0, (10, 20), (0, 0), (0.0, -0.1)),
        _node(2, 1, (10, 20, 30), (0, 0, 0), (0.0, -0.1, -0.1)),
        _node(3, 2, (10, 20, 30, 40), (0, 0, 0, 0), (0.0, -0.1, -0.1, -0.1)),
        _node(4, 0, (10, 21), (0, 1), (0.0, -0.2)),
        _node(5, 4, (10, 21, 31), (0, 1, 0), (0.0, -0.2, -0.2)),
        _node(6, 5, (10, 21, 31, 41), (0, 1, 0, 0), (0.0, -0.2, -0.2, -0.2)),
        _node(7, 1, (10, 20, 32), (0, 0, 1), (0.0, -0.1, -0.15)),
        _node(8, 7, (10, 20, 32, 42), (0, 0, 1, 0), (0.0, -0.1, -0.15, -0.1)),
        _node(9, 2, (10, 20, 30, 43), (0, 0, 0, 1), (0.0, -0.1, -0.1, -0.05)),
    ]
    chosen = select_counterfactual_paths(nodes)
    assert [path.first_divergence_depth for path in chosen if path] == [None, 2, 3, 4]
    assert [path.token_ids for path in chosen if path] == [
        (10, 20, 30, 40),
        (10, 21, 31, 41),
        (10, 20, 32, 42),
        (10, 20, 30, 43),
    ]
    assert chosen[2] is not None
    assert chosen[2].node_local_indices == (0, 1, 7, 8)


def test_missing_divergence_slot_is_masked_not_backfilled() -> None:
    nodes = [
        _node(0, None, (10,), (0,), (0.0,)),
        _node(1, 0, (10, 20), (0, 0), (0.0, -0.1)),
        _node(2, 1, (10, 20, 30), (0, 0, 0), (0.0, -0.1, -0.1)),
        _node(3, 2, (10, 20, 30, 40), (0, 0, 0, 0), (0.0, -0.1, -0.1, -0.1)),
    ]
    selected = select_counterfactual_paths(nodes)
    assert selected[0] is not None
    assert selected[1:] == (None, None, None)


def test_counterfactual_geometry_audit_reconstructs_rotated_layer_bases() -> None:
    torch.manual_seed(4)
    layers, experts, hidden = 3, 12, 6
    weights = torch.randn(layers, experts, hidden)
    geometry = build_centered_router_geometry(weights, relative_rank_threshold=1e-7)
    tensors = empty_counterfactual_tensors(
        layers=layers, rank=geometry.maximum_rank, experts=experts
    )
    inputs = torch.randn(layers, hidden)
    query = geometry.encode_router_inputs(inputs)
    logits = torch.einsum("ld,led->le", inputs.to(torch.bfloat16), weights.to(torch.bfloat16))
    ids = torch.argsort(logits.float(), dim=-1, descending=True, stable=True)[:, :8]
    probs = torch.softmax(logits.float(), dim=-1)
    values = probs.gather(-1, ids)
    values = (values / values.sum(-1, keepdim=True)).to(torch.bfloat16)
    tensors["path_mask"][0] = True
    tensors["path_depths"][0] = 2
    tensors["node_local_indices"][0, :2] = torch.tensor([0, 1])
    tensors["query_coordinates"][0, 1] = query
    tensors["router_logits"][0, 1] = logits
    tensors["selected_ids"][0, 1] = ids.to(torch.int32)
    tensors["selected_weights"][0, 1] = values
    tensors["valid"][0, 1] = True
    report = audit_counterfactual_geometry(tensors, geometry, maximum_logit_error=0.1)
    assert report["passed"]
    assert report["valid_layer_rows"] == layers


class _OneItem(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        assert index == 0
        return {
            "metadata": {"request_id": "r", "position": 7},
            "inputs": {"x": torch.tensor(1)},
            "targets": {"y": torch.tensor(2)},
        }


def test_companion_is_train_only_and_joins_under_targets() -> None:
    labels = {("r", 7): empty_counterfactual_tensors()}
    joined = CounterfactualDatasetAdapter(
        _OneItem(), labels, split="train", training=True
    )[0]
    assert "counterfactual" in joined["targets"]
    assert "counterfactual" not in joined["inputs"]
    with pytest.raises(PermissionError, match="outer-train"):
        CounterfactualDatasetAdapter(
            _OneItem(), labels, split="validation", training=True
        )
    with pytest.raises(PermissionError, match="inference"):
        assert_counterfactual_access_allowed(
            split="train", training=False, enabled=True
        )


def test_malicious_counterfactual_input_fails_closed() -> None:
    class Bad(_OneItem):
        def __getitem__(self, index: int) -> dict[str, object]:
            item = super().__getitem__(index)
            item["inputs"]["counterfactual"] = torch.tensor(3)  # type: ignore[index]
            return item

    with pytest.raises(ValueError, match="leaked"):
        CounterfactualDatasetAdapter(
            Bad(), {("r", 7): empty_counterfactual_tensors()},
            split="train", training=True,
        )[0]


def test_audited_companion_loads_by_request_position_and_rejects_sealed_split(
    tmp_path,
) -> None:
    root = tmp_path / "companion"
    records = root / "records"
    records.mkdir(parents=True)
    tensors = empty_counterfactual_tensors()
    record = records / "source_000000.safetensors"
    save_file(
        tensors,
        record,
        metadata={"label_only": "true", "runtime_available": "false"},
    )
    digest = hashlib.sha256(record.read_bytes()).hexdigest()
    manifest = {
        "schema": "harp_rtt_counterfactual_target_companion_v2",
        "label_only": True,
        "runtime_available": False,
        "split": "train",
        "sealed_test_opened": False,
        "bindings": {
            "selector_sha256": selector_sha256(),
            "router_geometry_sha256": "geometry",
        },
        "records": [
            {
                "request_id": "r",
                "source_position": 7,
                "record": {
                    "path": "records/source_000000.safetensors",
                    "sha256": digest,
                },
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "COUNTERFACTUAL_AUDIT.json").write_text(
        json.dumps({"passed": True})
    )
    (root / "SHA256SUMS").write_text(
        f"{digest}  records/source_000000.safetensors\n"
    )
    labels, loaded = load_counterfactual_companion(
        root,
        split="train",
        training=True,
        expected_bindings={"router_geometry_sha256": "geometry"},
    )
    assert ("r", 7) in labels
    assert loaded["label_only"] is True
    with pytest.raises(PermissionError, match="outer-train"):
        load_counterfactual_companion(
            root, split="test", training=True
        )
