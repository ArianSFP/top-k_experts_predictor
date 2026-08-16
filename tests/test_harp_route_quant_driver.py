import json

import pytest
import torch

from harp_rtt.dataset import HarpRTTDataset

from runpod.screen_route_quant import (
    parse_candidates,
    parse_layers,
    parse_upgrade_fractions,
    resolve_scale_storage,
    upgrade_schedule,
)
from runpod.train_shadow_experts_local import ShadowFactualStateDataset


def test_routequant_driver_parses_frozen_ladder():
    assert parse_layers("0,6,20,32") == (0, 6, 20, 32)
    assert parse_candidates("2:mse,3:mse,4:amax") == (
        (2, "mse"),
        (3, "mse"),
        (4, "amax"),
    )


@pytest.mark.parametrize("value", ("", "0,0", "39", "-1"))
def test_routequant_driver_rejects_invalid_layer_set(value: str):
    with pytest.raises(ValueError):
        parse_layers(value)


@pytest.mark.parametrize("value", ("", "2", "0:mse", "5:mse", "2:bad", "2:mse,2:mse"))
def test_routequant_driver_rejects_invalid_candidate_ladder(value: str):
    with pytest.raises(ValueError):
        parse_candidates(value)



def test_shadow_factual_wrapper_can_retarget_layer_without_losing_subset():
    source = object.__new__(ShadowFactualStateDataset)
    source.layer = 0
    source.next_router_agreement = True
    source.source = object.__new__(HarpRTTDataset)
    source.indices = (3, 7, 11)
    retargeted = ShadowFactualStateDataset(
        source, layer=6, next_router_agreement=True
    )
    assert retargeted.source is source.source
    assert retargeted.indices == source.indices
    assert retargeted.layer == 6


def test_routequant_mixed_upgrade_schedule_is_deterministic_and_nested():
    scores = torch.tensor([1.0, 3.0, 3.0, -1.0])
    quarter = upgrade_schedule(scores, fraction=0.25)
    half = upgrade_schedule(scores, fraction=0.5)
    assert quarter.tolist() == [3, 4, 3, 3]
    assert half.tolist() == [3, 4, 4, 3]
    assert set((quarter == 4).nonzero().flatten().tolist()) <= set(
        (half == 4).nonzero().flatten().tolist()
    )
    assert parse_upgrade_fractions("0.25,0.5,0.75") == (0.25, 0.5, 0.75)


@pytest.mark.parametrize("value", ("0", "1", "0.5,0.25", "0.5,0.5"))
def test_routequant_rejects_invalid_upgrade_fractions(value: str):
    with pytest.raises(ValueError):
        parse_upgrade_fractions(value)


def test_routequant_scale_storage_contract():
    assert resolve_scale_storage("bf16") == (torch.bfloat16, 2)
    assert resolve_scale_storage("log8") == (torch.uint8, 1)
    if hasattr(torch, "float8_e4m3fn"):
        assert resolve_scale_storage("fp8_e4m3") == (torch.float8_e4m3fn, 1)
    with pytest.raises(ValueError):
        resolve_scale_storage("int8")


def test_routequant_export_schedule_is_hash_bound(tmp_path):
    from runpod.export_route_quant import SCHEDULE_SCHEMA, load_schedule

    source = "a" * 40
    target = "b" * 64
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps({
        "schema": SCHEDULE_SCHEMA,
        "source_commit": source,
        "target_checkpoint_index_sha256": target,
        "bit_widths": [[3] * 256 for _ in range(40)],
    }))
    widths, digest = load_schedule(
        path, uniform_bits=None, source_commit=source,
        target_checkpoint_index_sha256=target,
    )
    assert widths.shape == (40, 256)
    assert digest is not None and len(digest) == 64
    with pytest.raises(ValueError, match="lineage"):
        load_schedule(
            path, uniform_bits=None, source_commit="c" * 40,
            target_checkpoint_index_sha256=target,
        )


def test_routequant_finalizer_seals_complete_interrupted_rows(tmp_path):
    from runpod.finalize_route_quant_screen import finalize

    run = tmp_path / "run"
    run.mkdir()
    manifest = {
        "source_commit": "a" * 40,
        "layers": [0],
        "candidates": ["3:mse"],
        "mixed_upgrade_fractions": [0.5],
        "mixed_base_bits": 3,
        "mixed_upgrade_bits": 4,
        "optimizer_constructed": False,
        "training_started": False,
        "development_opened_for_metrics": False,
        "formal_validation_opened": False,
        "calibration_opened": False,
        "sealed_test_opened": False,
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    base = {
        "layer": 0,
        "metrics": {
            "request_macro_next_router_recall_at_8": 0.9,
            "normalized_residual_rmse": 0.2,
        },
        "exact_baseline": {"request_macro_next_router_recall_at_8": 0.99},
        "uniform_40_layer_projected_bytes": 100,
        "uniform_40_layer_projected_gib": 100 / 2**30,
    }
    rows = [
        {**base, "bits": 3, "scale_method": "mse"},
        {**base, "variant": "mixed_int3_int4", "upgrade_fraction": 0.5},
    ]
    (run / "candidate_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (run / "layer_00_train_utility.json").write_text("{}")
    result = finalize(run, source_commit="b" * 40)
    assert result["candidate_rows"] == 2
    assert result["recovered_after_post_metric_interruption"] is True
    assert (run / "SHA256SUMS").is_file()
    with pytest.raises(FileExistsError):
        finalize(run, source_commit="b" * 40)


def test_routequant_global_schedule_is_stable_and_forces_final_layer():
    from runpod.build_route_quant_schedule import allocate_global_schedule

    scores = torch.arange(40 * 256, dtype=torch.float64).reshape(40, 256)
    scores[0, 0] = scores[0, 1]
    schedule = allocate_global_schedule(
        scores, base_bits=2, upgrade_bits=3, upgrade_fraction=0.125
    )
    assert schedule.shape == (40, 256)
    assert torch.equal(schedule[39], torch.full((256,), 3, dtype=torch.int8))
    assert int((schedule == 3).sum()) == round(40 * 256 * 0.125)
    assert schedule[38, 255] == 3


def test_routequant_global_schedule_never_upgrades_unobserved_experts(tmp_path):
    from runpod.build_route_quant_schedule import load_global_upgrade_scores

    manifest = {
        "schema": "harp_routequant_representative_screen_v1",
        "source_commit": "a" * 40,
        "target_checkpoint_index_sha256": "b" * 64,
        "partition_manifest_sha256": "c" * 64,
        "reuse_split_manifest_sha256": "d" * 64,
        "layers": list(range(39)),
        "mixed_base_bits": 1,
        "mixed_upgrade_bits": 2,
        "mixed_utility_split": "train",
        "development_opened_for_metrics": False,
        "formal_validation_opened": False,
        "sealed_test_opened": False,
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    for layer in range(39):
        score = [1.0] * 256
        occurrences = [1] * 256
        score[7] = 1e6
        occurrences[7] = 0
        (tmp_path / f"layer_{layer:02d}_train_utility.json").write_text(
            json.dumps({"layer": layer, "score": score, "occurrences": occurrences})
        )
    _, scores, _ = load_global_upgrade_scores(
        tmp_path, base_bits=1, upgrade_bits=2
    )
    assert torch.isneginf(scores[:39, 7]).all()
