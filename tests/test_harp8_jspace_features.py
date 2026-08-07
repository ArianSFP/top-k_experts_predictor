from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harp8.jspace_features import (
    FEATURE_SCHEMA,
    PrecisionAuditThresholds,
    audit_feature_precision,
    build_train_fit_pairs,
    export_feature_store,
    fit_shared_pca,
    load_shared_pca,
    make_random_orthogonal_control,
    normalize_with_log_rms,
    save_shared_pca,
    sha256_file,
    transport_all_layers,
    transport_layer,
)


def _lens(width: int) -> dict[int, np.ndarray]:
    return {layer: np.eye(width, dtype=np.float32) for layer in range(39)}


def test_row_vector_transport_uses_j_transpose_and_layer39_is_identity() -> None:
    matrices = _lens(2)
    matrices[0] = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    row = np.asarray([[5.0, 7.0]], dtype=np.float16)
    transported = transport_layer(row, 0, matrices)
    assert transported.dtype == np.float32
    assert np.array_equal(transported, row.astype(np.float32) @ matrices[0].T)
    assert not np.array_equal(transported, row.astype(np.float32) @ matrices[0])

    endpoint = transport_layer(row, 39, matrices)
    assert endpoint.dtype == np.float32
    assert np.array_equal(endpoint, row.astype(np.float32))

    all_layers = np.broadcast_to(row, (3, 40, 2)).copy()
    result = transport_all_layers(all_layers, matrices)
    assert result.shape == (3, 40, 2)
    assert np.array_equal(result[:, 0], transported.repeat(3, axis=0))
    assert np.array_equal(result[:, 39], all_layers[:, 39].astype(np.float32))


def test_normalized_feature_and_log_rms_preserve_scale() -> None:
    values = np.asarray([[3.0, 4.0], [6.0, 8.0]], dtype=np.float16)
    normalized, log_rms = normalize_with_log_rms(values)
    assert normalized.dtype == np.float32
    assert log_rms.dtype == np.float32
    assert np.allclose(np.sqrt(np.mean(normalized**2, axis=-1)), 1.0)
    expected_rms = np.sqrt(np.mean(values.astype(np.float32) ** 2, axis=-1))
    assert np.allclose(log_rms, np.log(expected_rms))
    assert np.allclose(normalized * np.exp(log_rms[:, None]), values)


def test_train_fit_pairs_are_train_only_balanced_and_deterministic() -> None:
    train_mask = np.asarray([True, True, False, True, False, True])
    domains = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
    requests = np.asarray([10, 11, 12, 13, 14, 15])
    first = build_train_fit_pairs(
        train_mask,
        layers=(0, 1, 2),
        maximum_vectors=8,
        seed=19,
        row_strata=(domains, requests),
    )
    second = build_train_fit_pairs(
        train_mask,
        layers=(0, 1, 2),
        maximum_vectors=8,
        seed=19,
        row_strata=(domains, requests),
    )
    assert np.array_equal(first, second)
    assert len(first) == 8
    assert train_mask[first[:, 0]].all()
    assert set(first[:, 1]).issubset({0, 1, 2})
    counts = np.bincount(first[:, 1], minlength=3)
    assert counts.max() - counts.min() <= 1


def test_shared_pca_uses_only_explicit_fit_pairs_and_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=(8, 3, 5)).astype(np.float32)
    fit_pairs = np.asarray(
        [[row, layer] for row in range(5) for layer in range(3)],
        dtype=np.int64,
    )
    first = fit_shared_pca(values, fit_pairs, rank=3, device="numpy")
    modified = values.copy()
    modified[5:] = 1_000_000.0  # validation rows must not affect the fit
    second = fit_shared_pca(modified, fit_pairs, rank=3, device="numpy")
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.components, second.components)
    assert first.fit_pairs_sha256 == second.fit_pairs_sha256
    projected = first.transform(values[:2])
    assert projected.shape == (2, 3, 3)

    path = tmp_path / "shared_pca.npz"
    metadata = save_shared_pca(first, path)
    loaded = load_shared_pca(path)
    assert metadata["file_sha256"] == sha256_file(path)
    assert loaded.sha256() == first.sha256()
    assert np.array_equal(loaded.components, first.components)


def test_random_orthogonal_controls_are_deterministic_and_norm_preserving() -> None:
    first = make_random_orthogonal_control(9, layers=4, seed=7)
    second = make_random_orthogonal_control(9, layers=4, seed=7)
    other = make_random_orthogonal_control(9, layers=4, seed=8)
    assert first.sha256() == second.sha256()
    assert first.sha256() != other.sha256()
    row = np.arange(9, dtype=np.float32)[None]
    transformed = first.apply_layer(row, 2)
    assert np.allclose(np.linalg.norm(transformed), np.linalg.norm(row))
    matrix = first.matrix(2)
    assert np.array_equal(matrix @ matrix.T, np.eye(9, dtype=np.float32))
    assert np.array_equal(transformed, row @ matrix.T)

    shared = make_random_orthogonal_control(
        9, layers=4, seed=7, shared_across_layers=True
    )
    assert np.array_equal(shared.matrix(0), shared.matrix(3))


def test_chunked_memmap_export_records_hashes_shapes_and_identity_endpoint(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "residuals.npy"
    source = np.lib.format.open_memmap(
        source_path, mode="w+", dtype="<f2", shape=(5, 40, 3)
    )
    source[...] = np.arange(source.size, dtype=np.float32).reshape(source.shape) + 1
    source.flush()
    matrices = _lens(3)
    matrices[0] = np.asarray(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    output_dir = tmp_path / "features"
    manifest = export_feature_store(
        source_path,
        output_dir,
        representation="j_lens",
        matrices=matrices,
        lens_provenance={"sha256": "lens-hash"},
        feature_dtype="<f2",
        chunk_rows=2,
    )
    assert manifest["schema"] == FEATURE_SCHEMA
    assert manifest["fp32_accumulation"] is True
    assert manifest["layer_39_transport"] == "identity"
    assert manifest["source"]["sha256"] == sha256_file(source_path)
    for name, record in manifest["outputs"].items():
        assert record["sha256"] == sha256_file(output_dir / name)
        assert record["bytes"] == (output_dir / name).stat().st_size
    disk_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert disk_manifest == manifest

    features = np.load(output_dir / "features_normalized.npy", mmap_mode="r")
    magnitude = np.load(output_dir / "log_rms.npy", mmap_mode="r")
    expected, expected_log_rms = normalize_with_log_rms(
        np.asarray(source[:, 39], dtype=np.float32)
    )
    assert features.shape == (5, 40, 3)
    assert features.dtype == np.dtype("<f2")
    assert magnitude.shape == (5, 40)
    assert np.allclose(features[:, 39], expected, atol=5e-4)
    assert np.allclose(magnitude[:, 39], expected_log_rms, atol=1e-6)
    with pytest.raises(FileExistsError):
        export_feature_store(
            source_path,
            output_dir,
            representation="raw",
        )


def test_precision_audit_applies_transport_thresholds_and_probe_gate() -> None:
    rng = np.random.default_rng(14)
    reference = rng.normal(size=(4, 40, 4)).astype(np.float32)
    candidate = reference.astype(np.float16)
    matrices = _lens(4)
    accepted = audit_feature_precision(
        reference,
        candidate,
        matrices,
        layers=(0, 12, 38, 39),
        reference_probe_recall=0.8,
        candidate_probe_recall=0.7995,
    )
    assert accepted["accepted"] is True
    assert accepted["mean_transport_cosine"] >= 0.9995
    assert accepted["probe_recall_delta"] == pytest.approx(0.0005)

    corrupted = candidate.copy()
    corrupted[:, 12] = rng.normal(size=corrupted[:, 12].shape)
    rejected = audit_feature_precision(
        reference,
        corrupted,
        matrices,
        layers=(12,),
        thresholds=PrecisionAuditThresholds(
            minimum_mean_cosine=0.99,
            minimum_row_cosine=0.98,
            maximum_probe_recall_delta=0.001,
        ),
    )
    assert rejected["accepted"] is False
    assert rejected["mean_transport_cosine"] < 0.99
