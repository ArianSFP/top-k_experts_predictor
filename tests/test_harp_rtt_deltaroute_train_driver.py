from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from harp_rtt.delta import HARPDeltaConfig, HARPDeltaTeacher


SCRIPT = Path(__file__).parents[1] / "runpod" / "train_harp_deltaroute_v4.py"
SPEC = importlib.util.spec_from_file_location("train_harp_deltaroute_v4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ceiling_gate_binds_parent_and_requires_native_top8(tmp_path: Path) -> None:
    parent = "a" * 64
    path = tmp_path / "ceiling.json"
    path.write_text(json.dumps({
        "schema": MODULE.CEILING_SCHEMA,
        "provenance": {"parent_checkpoint_sha256": parent},
        "native_factual_h2_h4_gate": {
            "passed": True, "value": 0.91, "threshold": 0.89,
        },
        "training_started": False,
        "optimizer_constructed": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }), encoding="utf-8")
    assert MODULE.validate_ceiling(path, parent)["native_factual_h2_h4"] == 0.91
    with pytest.raises(ValueError, match="parent checkpoint"):
        MODULE.validate_ceiling(path, "b" * 64)


def test_aligner_builder_keeps_parent_small_and_frozen() -> None:
    config = HARPDeltaConfig(
        experts=80, layers=3, horizons=4, exact_k=2, candidate_width=64,
        max_tree_nodes=4, router_rank=5, context_input_width=8,
        root_input_width=8, node_input_width=8, tree_width=8, set_width=8,
        ranker_width=8, tree_ffn_width=16, ranker_ffn_width=16,
        attention_heads=2, tree_blocks=1, ranker_blocks=1, free_rank=2,
        maximum_swaps=2,
    )
    parent = HARPDeltaTeacher(
        config, torch.randn(3, 80, 5), torch.randn(3, 80),
        raw_width=6, target_control_width=5, metadata_width=8,
    )
    parent.requires_grad_(False)
    for stage in MODULE.STAGES:
        aligner = MODULE.build_aligner(stage, parent)
        assert sum(parameter.numel() for parameter in aligner.parameters()) < 1_000_000
        assert all(not parameter.requires_grad for parameter in parent.parameters())


def test_driver_records_optimizer_start_separately_and_preserves_32_nodes() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"optimizer_constructed": False' in source
    assert '"OPTIMIZER_START.json"' in source
    assert '"runtime_tree_nodes": 32' in source
    assert '"counterfactual_supervision_budget": 16' in source
    assert "candidate_coverage_h2_h4" in source
