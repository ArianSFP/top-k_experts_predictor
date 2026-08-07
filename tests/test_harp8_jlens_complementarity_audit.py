from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harp8.audit_jlens_complementarity import (
    AUDIT_SCHEMA,
    build_parser,
    paired_request_bootstrap,
    request_source_rows,
    run_complementarity_audit,
    stable_topk,
)
from harp8.jspace_features import FEATURE_SCHEMA
from harp8.prepare_jspace_features import PREPARATION_SCHEMA


def _write_fixture(root: Path) -> tuple[Path, Path]:
    feature_root = root / "features"
    capture = root / "capture"
    feature_root.mkdir()
    capture.mkdir()
    rows_per_request = 6
    requests = [
        {"request_id": 101, "offline_split": "train"},
        {"request_id": 102, "offline_split": "train"},
        {"request_id": 103, "offline_split": "train"},
        {"request_id": 201, "offline_split": "validation"},
        {"request_id": 202, "offline_split": "validation"},
        {"request_id": 301, "offline_split": "test"},
    ]
    (capture / "requests.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in requests),
        encoding="utf-8",
    )
    total_rows, layers, experts, rank = len(requests) * rows_per_request, 40, 12, 3
    row = np.arange(total_rows, dtype=np.float32)[:, None, None]
    layer = np.arange(layers, dtype=np.float32)[None, :, None]
    coordinate = np.arange(rank, dtype=np.float32)[None, None, :]
    raw = np.sin(0.17 * row + 0.03 * layer + 0.41 * coordinate).astype(np.float16)
    j_lens = np.cos(0.11 * row - 0.02 * layer + 0.37 * coordinate).astype(np.float16)
    expert = np.arange(experts, dtype=np.float32)[None, None, :]
    logits = (
        np.sin(0.13 * row + 0.07 * layer + 0.19 * expert)
        + np.cos(0.23 * row - 0.05 * layer + 0.31 * expert)
    ).astype(np.float32)
    top8 = stable_topk(logits, 8).astype(np.uint16)

    # Deliberately poison the sealed split. A valid audit proves that no test
    # feature, router target, or top-k row was indexed.
    test_start = 5 * rows_per_request
    raw[test_start:] = np.float16(np.nan)
    j_lens[test_start:] = np.float16(np.nan)
    logits[test_start:] = np.float32(np.nan)
    top8[test_start:] = np.uint16(999)

    records: dict[str, dict[str, object]] = {}
    for representation, values in (("j_lens", j_lens), ("raw_residual", raw)):
        rank_dir = feature_root / representation / f"rank{rank}"
        rank_dir.mkdir(parents=True)
        path = rank_dir / "features_normalized.npy"
        np.save(path, values, allow_pickle=False)
        records[representation] = {
            "ranks": {
                str(rank): {
                    "feature_store": {
                        "schema": FEATURE_SCHEMA,
                        "representation": representation,
                        "rows": total_rows,
                        "layers": layers,
                        "feature_width": rank,
                        "pca": {
                            "fit_pairs_sha256": "a" * 64,
                            "captured_train_variance_fraction": 0.75,
                        },
                        "outputs": {
                            "features_normalized.npy": {
                                "path": str(path),
                                "bytes": path.stat().st_size,
                                "sha256": ("b" if representation == "j_lens" else "c")
                                * 64,
                            }
                        },
                    }
                }
            }
        }
    (feature_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": PREPARATION_SCHEMA,
                "immutable_output": True,
                "geometry": {"rows_per_request": rows_per_request},
                "fit": {"split": "train", "pairs_sha256": "a" * 64},
                "representations": records,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(capture / "raw_router_logits.npy", logits, allow_pickle=False)
    np.save(capture / "top8_expert_ids.npy", top8, allow_pickle=False)
    (capture / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "synthetic_capture",
                "arrays": {
                    "router": {
                        "path": "raw_router_logits.npy",
                        "sha256": "d" * 64,
                    },
                    "top8": {
                        "path": "top8_expert_ids.npy",
                        "sha256": "e" * 64,
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return feature_root, capture


def test_stable_topk_uses_ascending_expert_id_at_ties() -> None:
    scores = np.zeros((1, 12), dtype=np.float32)
    assert stable_topk(scores, 8).tolist() == [list(range(8))]


def test_request_rows_are_train_validation_only_and_request_major() -> None:
    requests = [
        {"request_id": 11, "offline_split": "train"},
        {"request_id": 22, "offline_split": "validation"},
        {"request_id": 33, "offline_split": "test"},
    ]
    rows, request_ids, within = request_source_rows(
        requests, "validation", rows_per_request=6, maximum_horizon=4
    )
    assert rows.tolist() == [6, 7]
    assert request_ids.tolist() == [22, 22]
    assert within.tolist() == [0, 1]
    with pytest.raises(PermissionError, match="train/validation"):
        request_source_rows(requests, "test", rows_per_request=6, maximum_horizon=4)


def test_paired_bootstrap_is_complete_request_paired_and_deterministic() -> None:
    raw = np.zeros((4, 4), dtype=np.float64)
    dual = np.asarray([[0.0, 0.1, 0.2, 0.3], [0.1] * 4, [0.2] * 4, [0.3] * 4])
    first = paired_request_bootstrap(
        dual, raw, horizons=(1, 2, 3, 4), replicates=100, seed=17
    )
    second = paired_request_bootstrap(
        dual, raw, horizons=(1, 2, 3, 4), replicates=100, seed=17
    )
    assert first == second
    assert first["mean_h1_h4"]["gain"] == pytest.approx(float(dual.mean()))
    assert first["resampling_unit"] == "complete validation request"


def test_end_to_end_audit_ignores_poisoned_test_rows_and_exports_predictions(
    tmp_path: Path,
) -> None:
    feature_root, capture = _write_fixture(tmp_path)
    output = tmp_path / "audit"
    summary = run_complementarity_audit(
        feature_root=feature_root,
        capture_dir=capture,
        output_dir=output,
        rank=3,
        train_sample=4,
        bootstrap_replicates=50,
        device="cpu",
        export_predictions=True,
        command_argv=("harp8-audit-jlens-complementarity", "--synthetic"),
    )
    assert summary["schema"] == AUDIT_SCHEMA
    assert summary["sealed_test_accessed"] is False
    assert summary["fit_split"] == "train"
    assert summary["evaluation_split"] == "validation"
    assert summary["data"]["validation_source_rows"] == 4
    assert summary["data"]["validation_requests"] == 2
    assert summary["data"]["request_counts"]["test_metadata_only"] == 1
    probe = summary["future_router_linear_probe"]
    assert set(probe["j3_recall_at_8"]) == {"h1", "h2", "h3", "h4", "mean_h1_h4"}
    assert len(probe["per_layer_metrics"]) == 40
    assert probe["paired_dual3_minus_raw3"]["bootstrap_replicates"] == 50
    assert (output / "audit_summary.json").is_file()
    request_rows = (
        (output / "validation_request_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(request_rows) == 8
    with np.load(output / "validation_predicted_top8.npz", allow_pickle=False) as rows:
        assert rows["j3"].shape == (4, 4, 40, 8)
        assert rows["raw3"].shape == (4, 4, 40, 8)
        assert rows["dual3"].shape == (4, 4, 40, 8)
        assert rows["request_id"].tolist() == [201, 201, 202, 202]
        assert not bool(rows["contains_target_top8"].item())
        assert not bool(rows["contains_router_targets"].item())
        assert "target_top8" not in rows.files
        assert "router_targets" not in rows.files
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        run_complementarity_audit(
            feature_root=feature_root,
            capture_dir=capture,
            output_dir=output,
            rank=3,
            train_sample=4,
            bootstrap_replicates=5,
        )


def test_invalid_non_train_pca_is_rejected_before_tensor_open(tmp_path: Path) -> None:
    feature_root, capture = _write_fixture(tmp_path)
    manifest_path = feature_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fit"]["split"] = "validation"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    # Remove a tensor entirely: the split gate must trigger before any NPY open.
    (capture / "raw_router_logits.npy").unlink()
    with pytest.raises(PermissionError, match="train split only"):
        run_complementarity_audit(
            feature_root=feature_root,
            capture_dir=capture,
            output_dir=tmp_path / "refused",
            rank=3,
            train_sample=4,
            bootstrap_replicates=5,
        )


def test_cli_has_no_test_or_split_selection_surface() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "split" not in destinations
    assert "test_pool" not in destinations
    assert "test" not in destinations
    args = parser.parse_args(
        [
            "--feature-root",
            "features",
            "--capture-dir",
            "capture",
            "--output-dir",
            "audit",
        ]
    )
    assert args.train_sample == 4096
    assert args.horizons == (1, 2, 3, 4)
    assert args.bootstrap_replicates == 5000
