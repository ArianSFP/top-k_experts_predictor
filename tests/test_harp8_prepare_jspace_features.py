from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from harp8.jspace_features import (
    load_j_lens,
    load_shared_pca,
    make_random_orthogonal_control,
    sha256_file,
)
from harp8.prepare_jspace_features import (
    AUDIT_SCHEMA,
    PREPARATION_SCHEMA,
    SUPPORTED_REPRESENTATIONS,
    _transform_fit_pairs,
    _transform_fit_pairs_device,
    build_parser,
    export_feature_store_device,
    prepare_feature_stores,
    run_precision_audit,
)


def _write_inputs(root: Path) -> tuple[Path, Path, Path, np.ndarray]:
    requests_path = root / "requests.jsonl"
    requests = [
        {"request_id": 101, "offline_split": "train", "domain": "reasoning"},
        {"request_id": 102, "offline_split": "train", "domain": "code"},
        {"request_id": 201, "offline_split": "validation", "domain": "reasoning"},
        {"request_id": 301, "offline_split": "test", "domain": "code"},
    ]
    requests_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in requests),
        encoding="utf-8",
    )

    rng = np.random.default_rng(17)
    residuals = rng.normal(size=(8, 40, 4)).astype(np.float16)
    residuals += np.linspace(0.1, 0.8, 8, dtype=np.float16)[:, None, None]
    residual_path = root / "post_layer_residuals.npy"
    np.save(residual_path, residuals, allow_pickle=False)

    matrices = {layer: torch.eye(4, dtype=torch.float32) for layer in range(39)}
    matrices[0] = torch.tensor(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.25, 0.0],
            [0.0, 0.0, 1.0, -0.5],
            [0.25, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    lens_path = root / "j_lens.pt"
    torch.save({"J": matrices, "n_prompts": 1_000}, lens_path)
    return residual_path, lens_path, requests_path, residuals


def test_cli_defaults_match_production_contract() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--residual-npy",
            "residual.npy",
            "--lens-checkpoint",
            "lens.pt",
            "--requests-jsonl",
            "requests.jsonl",
            "--output-root",
            "features",
        ]
    )
    assert args.representations == SUPPORTED_REPRESENTATIONS
    assert args.pca_ranks == (512,)
    assert args.fit_split == "train"
    assert args.max_fit_rows == 30_000
    assert args.seed == 42


def test_prepare_exports_every_control_with_train_only_shared_pca(tmp_path: Path) -> None:
    residual_path, lens_path, requests_path, residuals = _write_inputs(tmp_path)
    output = tmp_path / "features"
    manifest = prepare_feature_stores(
        residual_path,
        lens_path,
        requests_path,
        output,
        representations=SUPPORTED_REPRESENTATIONS,
        pca_ranks=(1, 2),
        maximum_fit_vectors=24,
        pca_device="numpy",
        export_chunk_rows=3,
        command_argv=("harp8-prepare-jspace", "--unit-test"),
    )

    assert manifest["schema"] == PREPARATION_SCHEMA
    assert manifest["command"] == "harp8-prepare-jspace --unit-test"
    assert manifest["fit"]["split"] == "train"
    assert manifest["fit"]["requests"] == 2
    assert manifest["fit"]["rows"] == 4
    assert manifest["fit"]["selected_vectors"] == 24
    assert manifest["splits"] == {
        "test": {"requests": 1, "rows": 2},
        "train": {"requests": 2, "rows": 4},
        "validation": {"requests": 1, "rows": 2},
    }
    disk_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert disk_manifest == manifest

    pair_hashes: set[str] = set()
    for representation in SUPPORTED_REPRESENTATIONS:
        record = manifest["representations"][representation]
        rank = record["ranks"]["2"]
        pca_path = output / representation / "shared_pca_rank2.npz"
        features_path = output / representation / "rank2" / "features_normalized.npy"
        magnitude_path = output / representation / "rank2" / "log_rms.npy"
        assert rank["pca_artifact"]["file_sha256"] == sha256_file(pca_path)
        assert np.load(features_path, mmap_mode="r").shape == (8, 40, 2)
        assert np.load(magnitude_path, mmap_mode="r").shape == (8, 40)
        assert np.isfinite(np.load(features_path)).all()
        assert np.isfinite(np.load(magnitude_path)).all()
        pair_hashes.add(load_shared_pca(pca_path).fit_pairs_sha256)
    assert pair_hashes == {manifest["fit"]["pairs_sha256"]}
    for representation in SUPPORTED_REPRESENTATIONS:
        rank1 = np.load(
            output / representation / "rank1" / "features_normalized.npy"
        )
        rank2 = np.load(
            output / representation / "rank2" / "features_normalized.npy"
        )
        assert np.array_equal(rank1, rank2[..., :1])
        rank1_manifest = manifest["representations"][representation]["ranks"]["1"][
            "feature_store"
        ]
        assert rank1_manifest["derived_from_maximum_rank"] == 2

    # Layer 39 is an identity endpoint before PCA, so every representation has
    # the same pre-projection magnitude there despite different control maps.
    expected_rms = np.sqrt(np.mean(residuals[:, 39].astype(np.float32) ** 2, axis=-1))
    j_magnitude = np.load(output / "j_lens" / "rank2" / "log_rms.npy")
    raw_magnitude = np.load(output / "raw_residual" / "rank2" / "log_rms.npy")
    assert np.allclose(j_magnitude[:, 39], np.log(expected_rms), atol=1e-6)
    assert np.array_equal(j_magnitude[:, 39], raw_magnitude[:, 39])

    with pytest.raises(FileExistsError):
        prepare_feature_stores(
            residual_path,
            lens_path,
            requests_path,
            output,
            representations=("raw_residual",),
            pca_ranks=(2,),
            maximum_fit_vectors=24,
            pca_device="numpy",
        )


def test_validation_and_test_values_cannot_change_fitted_pca(tmp_path: Path) -> None:
    residual_path, lens_path, requests_path, residuals = _write_inputs(tmp_path)
    first_output = tmp_path / "first"
    first = prepare_feature_stores(
        residual_path,
        lens_path,
        requests_path,
        first_output,
        representations=("raw_residual",),
        pca_ranks=(2,),
        maximum_fit_vectors=80,
        pca_device="numpy",
    )

    changed = residuals.copy()
    changed[4:] = np.float16(1_000.0)
    changed_path = tmp_path / "changed.npy"
    np.save(changed_path, changed, allow_pickle=False)
    second_output = tmp_path / "second"
    second = prepare_feature_stores(
        changed_path,
        lens_path,
        requests_path,
        second_output,
        representations=("raw_residual",),
        pca_ranks=(2,),
        maximum_fit_vectors=80,
        pca_device="numpy",
    )

    first_pca = first["representations"]["raw_residual"]["ranks"]["2"][
        "pca_artifact"
    ]
    second_pca = second["representations"]["raw_residual"]["ranks"]["2"][
        "pca_artifact"
    ]
    assert first_pca["sha256"] == second_pca["sha256"]
    assert first_pca["file_sha256"] == second_pca["file_sha256"]
    assert first["source"]["residual_npy"]["sha256"] != second["source"][
        "residual_npy"
    ]["sha256"]

    with pytest.raises(ValueError, match="may not be used"):
        prepare_feature_stores(
            residual_path,
            lens_path,
            requests_path,
            tmp_path / "forbidden",
            representations=("raw_residual",),
            pca_ranks=(2,),
            fit_split="validation",
            maximum_fit_vectors=80,
            pca_device="numpy",
        )


def test_audit_only_records_hashes_and_transport_gate(tmp_path: Path) -> None:
    residual_path, lens_path, requests_path, residuals = _write_inputs(tmp_path)
    reference_path = tmp_path / "reference.npy"
    np.save(reference_path, residuals.astype(np.float32), allow_pickle=False)
    output = tmp_path / "audit"
    manifest = run_precision_audit(
        residual_path,
        reference_path,
        lens_path,
        requests_path,
        output,
        maximum_rows=5,
        layers=(0, 12, 38, 39),
        seed=9,
        chunk_rows=2,
        reference_probe_recall=0.8000,
        candidate_probe_recall=0.7995,
        command_argv=("harp8-prepare-jspace", "--audit-only"),
    )
    assert manifest["schema"] == AUDIT_SCHEMA
    assert manifest["audit"]["accepted"] is True
    assert manifest["audit"]["mean_transport_cosine"] >= 0.9995
    assert manifest["audit"]["probe_recall_delta"] == pytest.approx(0.0005)
    assert manifest["sample"]["selected_rows"] == 5
    assert manifest["candidate"]["sha256"] == sha256_file(residual_path)
    assert manifest["reference"]["sha256"] == sha256_file(reference_path)
    assert json.loads((output / "precision_audit.json").read_text()) == manifest

    with pytest.raises(FileExistsError):
        run_precision_audit(
            residual_path,
            reference_path,
            lens_path,
            requests_path,
            output,
        )



def test_numpy_and_torch_device_export_are_equivalent(tmp_path: Path) -> None:
    residual_path, lens_path, requests_path, _residuals = _write_inputs(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_feature_stores(
        residual_path,
        lens_path,
        requests_path,
        prepared,
        representations=("raw_residual",),
        pca_ranks=(2,),
        maximum_fit_vectors=40,
        pca_device="numpy",
    )
    pca = load_shared_pca(prepared / "raw_residual" / "shared_pca_rank2.npz")
    matrices, lens_provenance = load_j_lens(lens_path, expected_width=4)

    numpy_dir = tmp_path / "numpy_export"
    selected_dir = tmp_path / "selected_export"
    export_feature_store_device(
        residual_path,
        numpy_dir,
        representation="j_lens",
        matrices=matrices,
        lens_provenance=lens_provenance,
        orthogonal_control=None,
        pca=pca,
        device="numpy",
        feature_dtype="float32",
        chunk_rows=3,
    )
    selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    selected_manifest = export_feature_store_device(
        residual_path,
        selected_dir,
        representation="j_lens",
        matrices=matrices,
        lens_provenance=lens_provenance,
        orthogonal_control=None,
        pca=pca,
        device=selected_device,
        feature_dtype="float32",
        chunk_rows=2,
    )
    numpy_features = np.load(numpy_dir / "features_normalized.npy")
    selected_features = np.load(selected_dir / "features_normalized.npy")
    numpy_rms = np.load(numpy_dir / "log_rms.npy")
    selected_rms = np.load(selected_dir / "log_rms.npy")
    assert np.allclose(selected_features, numpy_features, atol=3e-5, rtol=3e-5)
    assert np.allclose(selected_rms, numpy_rms, atol=3e-6, rtol=3e-6)
    assert selected_manifest["execution_device"] == selected_device
    assert selected_manifest["fp32_accumulation"] is True
    assert selected_manifest["tf32_disabled"] is torch.cuda.is_available()



def test_fit_pair_transport_matches_numpy_on_selected_device(tmp_path: Path) -> None:
    residual_path, lens_path, _requests_path, residuals = _write_inputs(tmp_path)
    matrices, _provenance = load_j_lens(lens_path, expected_width=4)
    pairs = np.asarray(
        [
            [0, 0],
            [1, 0],
            [2, 7],
            [3, 7],
            [4, 38],
            [5, 38],
            [6, 39],
            [7, 39],
        ],
        dtype=np.int64,
    )
    selected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    controls = {
        "j_lens": None,
        "raw_residual": None,
        "random_orthogonal_shared": make_random_orthogonal_control(
            4, seed=42, shared_across_layers=True
        ),
        "random_orthogonal_per_layer": make_random_orthogonal_control(
            4, seed=42, shared_across_layers=False
        ),
    }
    for representation, control in controls.items():
        expected = _transform_fit_pairs(
            residuals,
            pairs,
            representation=representation,
            matrices=matrices,
            control=control,
        )
        actual = _transform_fit_pairs_device(
            residuals,
            pairs,
            representation=representation,
            matrices=matrices,
            control=control,
            device=selected_device,
            chunk_vectors=1,
        )
        assert np.allclose(actual, expected, atol=3e-6, rtol=3e-6)
