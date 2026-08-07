"""Prepare immutable dense J-Lens and matched-control feature stores.

The command is intentionally separate from model training.  It joins the
post-layer residual rows to ``requests.jsonl``, selects PCA fit vectors only
from complete requests in the requested training split, fits one coordinate
basis shared by all target layers, and exports rank-specific memory-mappable
stores with complete provenance.

Typical invocation::

    python -m harp8.prepare_jspace_features \
        --residual-npy post_layer_residuals.npy \
        --lens-checkpoint j_lens.pt \
        --requests-jsonl requests.jsonl \
        --output-root artifacts/harp8/jspace_features \
        --representations j_lens,raw_residual,random_orthogonal_shared \
        --pca-ranks 256,512,1024
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .jspace_features import (
    FEATURE_SCHEMA,
    LENS_LAYERS,
    TARGET_LAYERS,
    PrecisionAuditThresholds,
    RandomOrthogonalControl,
    SharedPCA,
    audit_feature_precision,
    build_train_fit_pairs,
    fit_shared_pca,
    load_j_lens,
    make_random_orthogonal_control,
    normalize_with_log_rms,
    save_shared_pca,
    sha256_file,
    transport_layer,
)


PREPARATION_SCHEMA = "harp8_jspace_feature_preparation_v1"
AUDIT_SCHEMA = "harp8_jspace_precision_audit_command_v1"
SUPPORTED_REPRESENTATIONS = (
    "j_lens",
    "raw_residual",
    "random_orthogonal_shared",
    "random_orthogonal_per_layer",
)
_FORBIDDEN_FIT_SPLITS = {"validation", "val", "test", "eval", "evaluation"}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_requests(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"request at {path}:{line_number} is not an object")
            if "request_id" not in record:
                raise ValueError(f"request at {path}:{line_number} lacks request_id")
            records.append(record)
    if not records:
        raise ValueError("requests JSONL is empty")
    request_ids = [str(record["request_id"]) for record in records]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("requests JSONL contains duplicate request_id values")
    return records


def _request_split(record: Mapping[str, Any]) -> str:
    offline = record.get("offline_split")
    generic = record.get("split")
    if offline is not None and generic is not None and str(offline) != str(generic):
        raise ValueError(
            f"request {record.get('request_id')} has conflicting offline_split and split"
        )
    value = offline if offline is not None else generic
    if value is None:
        raise ValueError(f"request {record.get('request_id')} lacks a split field")
    return str(value)


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("comma-separated list must not be empty")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("comma-separated list contains duplicates")
    return values


def _parse_ranks(value: str) -> tuple[int, ...]:
    raw = _parse_csv_strings(value)
    try:
        ranks = tuple(int(item) for item in raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("PCA ranks must be integers") from exc
    if any(rank <= 0 for rank in ranks):
        raise argparse.ArgumentTypeError("PCA ranks must be positive")
    return tuple(sorted(ranks))


def _parse_layers(value: str) -> tuple[int, ...]:
    raw = _parse_csv_strings(value)
    try:
        layers = tuple(int(item) for item in raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("audit layers must be integers") from exc
    if any(layer < 0 or layer >= TARGET_LAYERS for layer in layers):
        raise argparse.ArgumentTypeError("audit layers must lie in 0..39")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("audit layers must be unique")
    return layers


def _resolve_geometry(
    residuals: np.ndarray,
    requests: Sequence[Mapping[str, Any]],
    rows_per_request: int | None,
) -> int:
    if residuals.ndim != 3 or residuals.shape[1] != TARGET_LAYERS:
        raise ValueError("residual array must have shape [rows,40,width]")
    if residuals.shape[0] == 0 or residuals.shape[2] == 0:
        raise ValueError("residual array must be non-empty")
    if rows_per_request is None:
        inferred, remainder = divmod(int(residuals.shape[0]), len(requests))
        if remainder or inferred <= 0:
            raise ValueError(
                "cannot infer a constant rows-per-request value from residuals and requests"
            )
        return inferred
    if rows_per_request <= 0:
        raise ValueError("rows_per_request must be positive")
    if len(requests) * rows_per_request != residuals.shape[0]:
        raise ValueError("rows_per_request disagrees with residual/request geometry")
    return int(rows_per_request)


def _request_metadata(
    requests: Sequence[Mapping[str, Any]],
    *,
    rows_per_request: int,
    fit_split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if fit_split.lower() in _FORBIDDEN_FIT_SPLITS:
        raise ValueError("validation and test requests may not be used to fit PCA")
    request_splits = np.asarray([_request_split(record) for record in requests], dtype=object)
    if fit_split not in set(request_splits.tolist()):
        raise ValueError(f"fit split {fit_split!r} does not occur in requests JSONL")
    request_domains = np.asarray(
        [str(record.get("domain", "unknown")) for record in requests], dtype=object
    )
    request_ids = np.asarray([str(record["request_id"]) for record in requests], dtype=object)
    row_splits = np.repeat(request_splits, rows_per_request)
    row_domains = np.repeat(request_domains, rows_per_request)
    row_requests = np.repeat(request_ids, rows_per_request)
    train_mask = np.asarray(row_splits == fit_split, dtype=bool)

    split_counts: dict[str, dict[str, int]] = {}
    for split in sorted(set(request_splits.tolist())):
        request_count = int(np.count_nonzero(request_splits == split))
        split_counts[split] = {
            "requests": request_count,
            "rows": request_count * rows_per_request,
        }
    fit_request_ids = sorted(request_ids[request_splits == fit_split].tolist())
    fit_summary = {
        "split": fit_split,
        "requests": len(fit_request_ids),
        "rows": int(train_mask.sum()),
        "request_ids_sha256": hashlib.sha256(
            _canonical_json(fit_request_ids).encode("utf-8")
        ).hexdigest(),
    }
    return train_mask, row_domains, row_requests, {
        "split_counts": split_counts,
        "fit": fit_summary,
    }


def _control_for_representation(
    representation: str,
    *,
    width: int,
    seed: int,
) -> RandomOrthogonalControl | None:
    if representation == "random_orthogonal_shared":
        return make_random_orthogonal_control(
            width, seed=seed, shared_across_layers=True
        )
    if representation == "random_orthogonal_per_layer":
        return make_random_orthogonal_control(
            width, seed=seed, shared_across_layers=False
        )
    return None


def _transform_fit_pairs(
    residuals: np.ndarray,
    fit_pairs: np.ndarray,
    *,
    representation: str,
    matrices: Mapping[int, Any],
    control: RandomOrthogonalControl | None,
) -> np.ndarray:
    """Transform and unit-RMS-normalize only the sampled PCA vectors."""

    values = np.empty((len(fit_pairs), residuals.shape[2]), dtype=np.float32)
    for layer in sorted(set(fit_pairs[:, 1].tolist())):
        selected = np.flatnonzero(fit_pairs[:, 1] == layer)
        rows = fit_pairs[selected, 0]
        source = np.asarray(residuals[rows, layer], dtype=np.float32)
        if representation == "j_lens":
            transformed = transport_layer(source, int(layer), matrices)
        elif representation == "raw_residual":
            transformed = source
        elif representation.startswith("random_orthogonal"):
            if control is None:
                raise AssertionError("random control was not constructed")
            transformed = control.apply_layer(source, int(layer))
        else:  # validated by the public entry point
            raise AssertionError(f"unexpected representation {representation}")
        values[selected], _log_rms = normalize_with_log_rms(transformed)
    return values



def _transform_fit_pairs_device(
    residuals: np.ndarray,
    fit_pairs: np.ndarray,
    *,
    representation: str,
    matrices: Mapping[int, Any],
    control: RandomOrthogonalControl | None,
    device: str,
    chunk_vectors: int = 4096,
) -> np.ndarray:
    """Device-aware FP32 transport for the sampled shared-PCA vectors."""

    if chunk_vectors <= 0:
        raise ValueError("fit transform chunk size must be positive")
    if device == "numpy":
        return _transform_fit_pairs(
            residuals,
            fit_pairs,
            representation=representation,
            matrices=matrices,
            control=control,
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for CPU/CUDA fit transport") from exc
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fit transport requested but CUDA is unavailable")
    values = np.empty((len(fit_pairs), residuals.shape[2]), dtype=np.float32)
    previous_tf32 = None
    if target.type == "cuda":
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for layer in sorted(set(fit_pairs[:, 1].tolist())):
            selected = np.flatnonzero(fit_pairs[:, 1] == layer)
            matrix_tensor = None
            permutation = None
            signs = None
            if representation == "j_lens" and layer < LENS_LAYERS:
                matrix_tensor = torch.as_tensor(
                    np.asarray(matrices[int(layer)], dtype=np.float32),
                    dtype=torch.float32,
                    device=target,
                )
            elif representation.startswith("random_orthogonal"):
                if control is None:
                    raise AssertionError("random control was not constructed")
                permutation = torch.as_tensor(
                    control.permutations[int(layer)],
                    dtype=torch.int64,
                    device=target,
                )
                signs = torch.as_tensor(
                    control.signs[int(layer)],
                    dtype=torch.float32,
                    device=target,
                )
            for offset in range(0, len(selected), chunk_vectors):
                destination = selected[offset : offset + chunk_vectors]
                rows = fit_pairs[destination, 0]
                source = np.asarray(residuals[rows, int(layer)], dtype=np.float32)
                host = torch.from_numpy(source)
                if target.type == "cuda":
                    staged = torch.empty(host.shape, dtype=torch.float32, pin_memory=True)
                    staged.copy_(host)
                    tensor = staged.to(target, non_blocking=True)
                else:
                    tensor = host.to(target)
                with torch.no_grad(), torch.autocast(
                    device_type=target.type, enabled=False
                ):
                    if representation == "j_lens" and matrix_tensor is not None:
                        transformed = torch.matmul(tensor, matrix_tensor.transpose(0, 1))
                    elif representation == "raw_residual" or (
                        representation == "j_lens" and layer == TARGET_LAYERS - 1
                    ):
                        transformed = tensor
                    elif representation.startswith("random_orthogonal"):
                        assert permutation is not None and signs is not None
                        transformed = tensor[:, permutation] * signs
                    else:
                        raise AssertionError(
                            f"unexpected representation {representation}"
                        )
                    rms = transformed.square().mean(dim=-1, keepdim=True).sqrt()
                    normalized = transformed / rms.clamp_min(1e-12)
                    if not bool(torch.isfinite(normalized).all()):
                        raise ValueError("non-finite PCA fit feature")
                    values[destination] = normalized.cpu().numpy()
    finally:
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return values


def _fit_representation_pca(
    residuals: np.ndarray,
    fit_pairs: np.ndarray,
    *,
    representation: str,
    matrices: Mapping[int, Any],
    control: RandomOrthogonalControl | None,
    maximum_rank: int,
    seed: int,
    iterations: int,
    device: str,
    transform_chunk_vectors: int,
) -> SharedPCA:
    matrix = _transform_fit_pairs_device(
        residuals,
        fit_pairs,
        representation=representation,
        matrices=matrices,
        control=control,
        device=device,
        chunk_vectors=transform_chunk_vectors,
    )
    synthetic = matrix[:, None, :]
    synthetic_pairs = np.stack(
        [np.arange(len(matrix), dtype=np.int64), np.zeros(len(matrix), dtype=np.int64)],
        axis=1,
    )
    fitted = fit_shared_pca(
        synthetic,
        synthetic_pairs,
        rank=maximum_rank,
        seed=seed,
        iterations=iterations,
        device=device,
    )
    actual_pairs_hash = hashlib.sha256(
        np.ascontiguousarray(fit_pairs, dtype="<i8").tobytes()
    ).hexdigest()
    return replace(fitted, fit_pairs_sha256=actual_pairs_hash)


def _truncate_pca(pca: SharedPCA, rank: int) -> SharedPCA:
    if not 0 < rank <= pca.rank:
        raise ValueError("truncated PCA rank lies outside the fitted basis")
    retained_energy = float(np.square(pca.singular_values, dtype=np.float64).sum())
    total_energy = retained_energy / max(pca.captured_variance_fraction, 1e-30)
    captured = float(
        np.square(pca.singular_values[:rank], dtype=np.float64).sum() / total_energy
    )
    return SharedPCA(
        mean=np.asarray(pca.mean, dtype=np.float32),
        components=np.asarray(pca.components[:, :rank], dtype=np.float32),
        singular_values=np.asarray(pca.singular_values[:rank], dtype=np.float32),
        fit_pairs_sha256=pca.fit_pairs_sha256,
        fit_vectors=pca.fit_vectors,
        seed=pca.seed,
        captured_variance_fraction=captured,
    )



def _transform_full_chunk_numpy(
    chunk: np.ndarray,
    *,
    representation: str,
    matrices: Mapping[int, Any],
    control: RandomOrthogonalControl | None,
) -> np.ndarray:
    values = np.asarray(chunk, dtype=np.float32)
    if representation == "raw_residual":
        return np.array(values, copy=True)
    output = np.empty_like(values, dtype=np.float32)
    if representation == "j_lens":
        for layer in range(TARGET_LAYERS):
            output[:, layer] = transport_layer(values[:, layer], layer, matrices)
        return output
    if representation.startswith("random_orthogonal"):
        if control is None:
            raise AssertionError("random control was not constructed")
        for layer in range(TARGET_LAYERS):
            output[:, layer] = control.apply_layer(values[:, layer], layer)
        return output
    raise AssertionError(f"unexpected representation {representation}")


def export_feature_store_device(
    source: Path | np.ndarray,
    output_dir: Path,
    *,
    representation: str,
    matrices: Mapping[int, Any],
    lens_provenance: Mapping[str, Any],
    orthogonal_control: RandomOrthogonalControl | None,
    pca: SharedPCA,
    device: str,
    feature_dtype: str | np.dtype = "float16",
    log_rms_dtype: str | np.dtype = "float32",
    chunk_rows: int = 32,
    source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export with FP32 NumPy or true FP32 Torch/CUDA transport.

    CUDA keeps the 39-layer lens resident on device, transfers one residual-row
    chunk through pinned host staging, disables TF32, and performs transport,
    normalization, and PCA before copying compact output to NPY memmaps.
    """

    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(f"unsupported representation {representation!r}")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse feature directory {output_dir}")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        source_array = np.load(source_path, mmap_mode="r", allow_pickle=False)
        provenance = dict(source_provenance or {})
        if not provenance:
            provenance = {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "bytes": source_path.stat().st_size,
            }
    else:
        source_path = None
        source_array = np.asanyarray(source)
        digest = hashlib.sha256()
        digest.update(str(source_array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(source_array.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(source_array).tobytes())
        provenance = {"path": None, "array_sha256": digest.hexdigest()}
    if source_array.ndim != 3 or source_array.shape[1] != TARGET_LAYERS:
        raise ValueError("source residuals must have shape [rows,40,width]")
    width = int(source_array.shape[2])
    if pca.input_width != width:
        raise ValueError("PCA width disagrees with source residuals")
    if representation == "j_lens":
        if set(int(value) for value in matrices) != set(range(LENS_LAYERS)):
            raise ValueError("J-Lens coverage must be exactly layers 0..38")
    if representation.startswith("random_orthogonal"):
        if orthogonal_control is None:
            raise ValueError("random representation requires an orthogonal control")
        if orthogonal_control.layers != TARGET_LAYERS or orthogonal_control.width != width:
            raise ValueError("random-control geometry disagrees with residuals")
    feature_dtype = np.dtype(feature_dtype)
    log_rms_dtype = np.dtype(log_rms_dtype)
    if feature_dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("device exporter supports float16 or float32 features")
    if log_rms_dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("device exporter supports float16 or float32 log RMS")

    output_dir.mkdir(parents=True)
    features_path = output_dir / "features_normalized.npy"
    magnitude_path = output_dir / "log_rms.npy"
    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=feature_dtype,
        shape=(source_array.shape[0], TARGET_LAYERS, pca.rank),
    )
    log_rms = np.lib.format.open_memmap(
        magnitude_path,
        mode="w+",
        dtype=log_rms_dtype,
        shape=(source_array.shape[0], TARGET_LAYERS),
    )

    execution_device = str(device)
    use_numpy = execution_device == "numpy"
    torch = None
    target = None
    lens_tensor = None
    permutations = None
    signs = None
    pca_mean = None
    pca_components = None
    previous_tf32 = None
    if not use_numpy:
        try:
            import torch as torch_module
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for CPU/CUDA device export") from exc
        torch = torch_module
        target = torch.device(execution_device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA feature export requested but CUDA is unavailable")
        if target.type == "cuda":
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
        if representation == "j_lens":
            lens_tensor = torch.stack(
                [
                    torch.as_tensor(
                        np.asarray(matrices[layer], dtype=np.float32),
                        dtype=torch.float32,
                    )
                    for layer in range(LENS_LAYERS)
                ],
                dim=0,
            ).to(target)
        if representation.startswith("random_orthogonal"):
            assert orthogonal_control is not None
            permutations = torch.as_tensor(
                orthogonal_control.permutations, dtype=torch.int64, device=target
            )
            signs = torch.as_tensor(
                orthogonal_control.signs, dtype=torch.float32, device=target
            )
        pca_mean = torch.as_tensor(pca.mean, dtype=torch.float32, device=target)
        pca_components = torch.as_tensor(
            pca.components, dtype=torch.float32, device=target
        )

    try:
        for start in range(0, source_array.shape[0], chunk_rows):
            stop = min(source_array.shape[0], start + chunk_rows)
            source_chunk = np.asarray(source_array[start:stop], dtype=np.float32)
            if use_numpy:
                transformed = _transform_full_chunk_numpy(
                    source_chunk,
                    representation=representation,
                    matrices=matrices,
                    control=orthogonal_control,
                )
                normalized, magnitude = normalize_with_log_rms(transformed)
                projected = pca.transform(normalized)
            else:
                assert torch is not None and target is not None
                host = torch.from_numpy(source_chunk)
                if target.type == "cuda":
                    staged = torch.empty(host.shape, dtype=torch.float32, pin_memory=True)
                    staged.copy_(host)
                    tensor = staged.to(target, non_blocking=True)
                else:
                    tensor = host.to(target)
                with torch.no_grad(), torch.autocast(
                    device_type=target.type, enabled=False
                ):
                    if representation == "raw_residual":
                        transformed_tensor = tensor
                    elif representation == "j_lens":
                        assert lens_tensor is not None
                        transformed_tensor = torch.empty_like(tensor)
                        transformed_tensor[:, :LENS_LAYERS] = torch.bmm(
                            tensor[:, :LENS_LAYERS].transpose(0, 1),
                            lens_tensor.transpose(1, 2),
                        ).transpose(0, 1)
                        transformed_tensor[:, TARGET_LAYERS - 1] = tensor[
                            :, TARGET_LAYERS - 1
                        ]
                    else:
                        assert permutations is not None and signs is not None
                        index = permutations.unsqueeze(0).expand(tensor.shape[0], -1, -1)
                        transformed_tensor = torch.gather(tensor, 2, index) * signs.unsqueeze(0)
                    rms = transformed_tensor.square().mean(dim=-1, keepdim=True).sqrt()
                    rms = rms.clamp_min(1e-12)
                    normalized_tensor = transformed_tensor / rms
                    assert pca_mean is not None and pca_components is not None
                    projected_tensor = torch.matmul(
                        normalized_tensor - pca_mean, pca_components
                    )
                    if not bool(torch.isfinite(projected_tensor).all()) or not bool(
                        torch.isfinite(rms).all()
                    ):
                        raise ValueError(f"non-finite feature in source rows {start}:{stop}")
                    output_tensor = projected_tensor
                    if feature_dtype == np.dtype("float16"):
                        output_tensor = output_tensor.to(torch.float16)
                    projected = output_tensor.cpu().numpy()
                    magnitude = rms.log()[..., 0].cpu().numpy()
            if not np.isfinite(projected).all() or not np.isfinite(magnitude).all():
                raise ValueError(f"non-finite feature in source rows {start}:{stop}")
            features[start:stop] = projected
            log_rms[start:stop] = magnitude
    finally:
        if previous_tf32 is not None:
            assert torch is not None
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    features.flush()
    log_rms.flush()

    outputs: dict[str, Any] = {}
    for path, shape, dtype in (
        (features_path, features.shape, features.dtype),
        (magnitude_path, log_rms.shape, log_rms.dtype),
    ):
        outputs[path.name] = {
            "path": str(path),
            "shape": [int(value) for value in shape],
            "dtype": str(dtype),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest: dict[str, Any] = {
        "schema": FEATURE_SCHEMA,
        "representation": representation,
        "transport_definition": "row_vector_xplus_matmul_J_transpose",
        "fp32_accumulation": True,
        "tf32_disabled": bool(not use_numpy and target is not None and target.type == "cuda"),
        "execution_device": execution_device,
        "host_io": "chunked_npy_memmap_with_pinned_staging_on_cuda",
        "normalization": "unit_rms_with_separate_natural_log_rms",
        "rows": int(source_array.shape[0]),
        "layers": TARGET_LAYERS,
        "input_width": width,
        "feature_width": pca.rank,
        "layer_39_transport": "identity"
        if representation in {"j_lens", "raw_residual"}
        else "random_orthogonal_control",
        "source": provenance,
        "outputs": outputs,
        "pca": pca.metadata(),
    }
    if representation == "j_lens":
        manifest["lens"] = dict(lens_provenance)
    if orthogonal_control is not None:
        manifest["random_orthogonal_control"] = {
            "construction": "signed_permutation",
            "seed": orthogonal_control.seed,
            "shared_across_layers": orthogonal_control.shared_across_layers,
            "sha256": orthogonal_control.sha256(),
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _export_rank_prefix(
    maximum_store_dir: Path,
    output_dir: Path,
    *,
    pca: SharedPCA,
    chunk_rows: int,
) -> dict[str, Any]:
    """Derive a nested lower-rank store without repeating dense transport."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse feature directory {output_dir}")
    maximum_store_dir = Path(maximum_store_dir)
    base_manifest = json.loads(
        (maximum_store_dir / "manifest.json").read_text(encoding="utf-8")
    )
    source_features = np.load(
        maximum_store_dir / "features_normalized.npy", mmap_mode="r", allow_pickle=False
    )
    if pca.rank >= source_features.shape[-1]:
        raise ValueError("prefix rank must be smaller than maximum store width")
    output_dir.mkdir(parents=True)
    features_path = output_dir / "features_normalized.npy"
    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=source_features.dtype,
        shape=(*source_features.shape[:-1], pca.rank),
    )
    for start in range(0, len(source_features), chunk_rows):
        stop = min(len(source_features), start + chunk_rows)
        features[start:stop] = source_features[start:stop, :, : pca.rank]
    features.flush()
    magnitude_path = output_dir / "log_rms.npy"
    shutil.copyfile(maximum_store_dir / "log_rms.npy", magnitude_path)
    magnitude = np.load(magnitude_path, mmap_mode="r", allow_pickle=False)

    manifest = copy.deepcopy(base_manifest)
    manifest["feature_width"] = pca.rank
    manifest["pca"] = pca.metadata()
    manifest["derived_from_maximum_rank"] = int(source_features.shape[-1])
    manifest["outputs"] = {
        "features_normalized.npy": {
            "path": str(features_path),
            "shape": [int(value) for value in features.shape],
            "dtype": str(features.dtype),
            "bytes": features_path.stat().st_size,
            "sha256": sha256_file(features_path),
        },
        "log_rms.npy": {
            "path": str(magnitude_path),
            "shape": [int(value) for value in magnitude.shape],
            "dtype": str(magnitude.dtype),
            "bytes": magnitude_path.stat().st_size,
            "sha256": sha256_file(magnitude_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_feature_stores(
    residual_npy: Path,
    lens_checkpoint: Path,
    requests_jsonl: Path,
    output_root: Path,
    *,
    representations: Sequence[str] = SUPPORTED_REPRESENTATIONS,
    pca_ranks: Sequence[int] = (512,),
    fit_split: str = "train",
    maximum_fit_vectors: int = 30_000,
    rows_per_request: int | None = None,
    seed: int = 42,
    pca_iterations: int = 6,
    pca_device: str = "cuda:0",
    fit_transform_chunk_vectors: int = 4096,
    export_chunk_rows: int = 32,
    feature_dtype: str = "float16",
    command_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit and export all requested representations into a new output root."""

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output root {output_root}")
    names = tuple(str(value) for value in representations)
    unknown = sorted(set(names) - set(SUPPORTED_REPRESENTATIONS))
    if not names or len(set(names)) != len(names) or unknown:
        raise ValueError(
            f"representations must be unique members of {SUPPORTED_REPRESENTATIONS}; "
            f"unknown={unknown}"
        )
    ranks = tuple(sorted(int(value) for value in pca_ranks))
    if not ranks or len(set(ranks)) != len(ranks) or ranks[0] <= 0:
        raise ValueError("PCA ranks must be unique positive integers")
    if maximum_fit_vectors <= 0:
        raise ValueError("maximum_fit_vectors must be positive")
    if pca_iterations <= 0 or export_chunk_rows <= 0:
        raise ValueError("PCA iterations and export chunk rows must be positive")
    if fit_transform_chunk_vectors <= 0:
        raise ValueError("fit transform chunk vectors must be positive")

    residual_npy = Path(residual_npy)
    lens_checkpoint = Path(lens_checkpoint)
    requests_jsonl = Path(requests_jsonl)
    residuals = np.load(residual_npy, mmap_mode="r", allow_pickle=False)
    requests = _read_requests(requests_jsonl)
    rows_per_request = _resolve_geometry(residuals, requests, rows_per_request)
    width = int(residuals.shape[2])
    if ranks[-1] > min(width, maximum_fit_vectors - 1):
        raise ValueError("largest PCA rank exceeds width or maximum fit capacity")
    matrices, lens_provenance = load_j_lens(lens_checkpoint, expected_width=width)
    train_mask, row_domains, row_requests, request_summary = _request_metadata(
        requests,
        rows_per_request=rows_per_request,
        fit_split=fit_split,
    )
    fit_pairs = build_train_fit_pairs(
        train_mask,
        maximum_vectors=maximum_fit_vectors,
        seed=seed,
        row_strata=(row_domains, row_requests),
    )
    if ranks[-1] >= len(fit_pairs):
        raise ValueError("largest PCA rank requires at least rank+1 fit vectors")

    residual_provenance = {
        "path": str(residual_npy),
        "sha256": sha256_file(residual_npy),
        "bytes": residual_npy.stat().st_size,
        "shape": [int(value) for value in residuals.shape],
        "dtype": str(residuals.dtype),
    }
    output_root.mkdir(parents=True)
    representation_records: dict[str, Any] = {}
    for representation in names:
        control = _control_for_representation(representation, width=width, seed=seed)
        fitted = _fit_representation_pca(
            residuals,
            fit_pairs,
            representation=representation,
            matrices=matrices,
            control=control,
            maximum_rank=ranks[-1],
            seed=seed,
            iterations=pca_iterations,
            device=pca_device,
            transform_chunk_vectors=fit_transform_chunk_vectors,
        )
        representation_dir = output_root / representation
        representation_dir.mkdir()
        pcas = {rank: _truncate_pca(fitted, rank) for rank in ranks}
        pca_records = {
            rank: save_shared_pca(
                pcas[rank], representation_dir / f"shared_pca_rank{rank}.npz"
            )
            for rank in ranks
        }
        maximum_rank = ranks[-1]
        maximum_store_dir = representation_dir / f"rank{maximum_rank}"
        stores = {
            maximum_rank: export_feature_store_device(
                residual_npy,
                maximum_store_dir,
                representation=representation,
                matrices=matrices,
                lens_provenance=lens_provenance,
                orthogonal_control=control,
                pca=pcas[maximum_rank],
                device=pca_device,
                feature_dtype=feature_dtype,
                chunk_rows=export_chunk_rows,
                source_provenance=residual_provenance,
            )
        }
        for rank in ranks[:-1]:
            stores[rank] = _export_rank_prefix(
                maximum_store_dir,
                representation_dir / f"rank{rank}",
                pca=pcas[rank],
                chunk_rows=export_chunk_rows,
            )
        rank_records = {
            str(rank): {
                "pca_artifact": pca_records[rank],
                "feature_store": stores[rank],
            }
            for rank in ranks
        }
        representation_records[representation] = {
            "fit": fitted.metadata(),
            "random_orthogonal_control": None
            if control is None
            else {
                "construction": "signed_permutation",
                "shared_across_layers": control.shared_across_layers,
                "seed": control.seed,
                "sha256": control.sha256(),
            },
            "ranks": rank_records,
        }

    invocation = list(command_argv) if command_argv is not None else list(sys.argv)
    manifest: dict[str, Any] = {
        "schema": PREPARATION_SCHEMA,
        "immutable_output": True,
        "command": shlex.join(invocation),
        "commands": {"prepare": shlex.join(invocation)},
        "argv": invocation,
        "source": {
            "residual_npy": residual_provenance,
            "requests_jsonl": {
                "path": str(requests_jsonl),
                "sha256": sha256_file(requests_jsonl),
                "requests": len(requests),
            },
            "lens_checkpoint": lens_provenance,
        },
        "geometry": {
            "rows_per_request": rows_per_request,
            "target_layers": TARGET_LAYERS,
            "lens_layers": LENS_LAYERS,
            "hidden_width": width,
            "j_lens_layer_39_transport": "identity",
            "random_control_layer_39_transport": "signed_permutation",
        },
        "splits": request_summary["split_counts"],
        "fit": {
            **request_summary["fit"],
            "maximum_vectors": maximum_fit_vectors,
            "selected_vectors": int(len(fit_pairs)),
            "pairs_sha256": hashlib.sha256(
                np.ascontiguousarray(fit_pairs, dtype="<i8").tobytes()
            ).hexdigest(),
            "stratification": ["domain", "request_id", "target_layer"],
            "seed": seed,
            "pca_device": pca_device,
            "pca_iterations": pca_iterations,
            "transform_chunk_vectors": fit_transform_chunk_vectors,
            "export_chunk_rows": export_chunk_rows,
        },
        "representations": representation_records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_precision_audit(
    candidate_residual_npy: Path,
    reference_residual_npy: Path,
    lens_checkpoint: Path,
    requests_jsonl: Path,
    output_root: Path,
    *,
    maximum_rows: int = 256,
    layers: Sequence[int] = (0, 12, 20, 28, 36, 38, 39),
    rows_per_request: int | None = None,
    seed: int = 42,
    chunk_rows: int = 32,
    reference_probe_recall: float | None = None,
    candidate_probe_recall: float | None = None,
    command_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the reduced-precision J-transport audit without exporting features."""

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output root {output_root}")
    if maximum_rows <= 0:
        raise ValueError("maximum_rows must be positive")
    candidate_path = Path(candidate_residual_npy)
    reference_path = Path(reference_residual_npy)
    requests_path = Path(requests_jsonl)
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    requests = _read_requests(requests_path)
    _resolve_geometry(candidate, requests, rows_per_request)
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference residual arrays must have identical shapes")
    matrices, lens_provenance = load_j_lens(
        Path(lens_checkpoint), expected_width=int(candidate.shape[2])
    )
    rng = np.random.default_rng(seed)
    if len(candidate) > maximum_rows:
        selected = np.sort(rng.choice(len(candidate), maximum_rows, replace=False))
    else:
        selected = np.arange(len(candidate), dtype=np.int64)
    audit = audit_feature_precision(
        np.asarray(reference[selected]),
        np.asarray(candidate[selected]),
        matrices,
        layers=layers,
        thresholds=PrecisionAuditThresholds(),
        reference_probe_recall=reference_probe_recall,
        candidate_probe_recall=candidate_probe_recall,
        chunk_rows=chunk_rows,
    )
    invocation = list(command_argv) if command_argv is not None else list(sys.argv)
    manifest = {
        "schema": AUDIT_SCHEMA,
        "command": shlex.join(invocation),
        "commands": {"precision_audit": shlex.join(invocation)},
        "argv": invocation,
        "candidate": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "dtype": str(candidate.dtype),
        },
        "reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "dtype": str(reference.dtype),
        },
        "requests_jsonl": {
            "path": str(requests_path),
            "sha256": sha256_file(requests_path),
            "requests": len(requests),
        },
        "lens_checkpoint": lens_provenance,
        "sample": {
            "seed": seed,
            "maximum_rows": maximum_rows,
            "selected_rows": int(len(selected)),
            "selected_rows_sha256": hashlib.sha256(
                np.ascontiguousarray(selected, dtype="<i8").tobytes()
            ).hexdigest(),
            "layers": [int(value) for value in layers],
        },
        "audit": audit,
    }
    output_root.mkdir(parents=True)
    (output_root / "precision_audit.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-npy", type=Path, required=True)
    parser.add_argument("--lens-checkpoint", type=Path, required=True)
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--representations",
        type=_parse_csv_strings,
        default=SUPPORTED_REPRESENTATIONS,
        help="comma list: " + ",".join(SUPPORTED_REPRESENTATIONS),
    )
    parser.add_argument("--pca-ranks", type=_parse_ranks, default=(512,))
    parser.add_argument("--fit-split", default="train")
    parser.add_argument("--max-fit-rows", type=int, default=30_000)
    parser.add_argument("--rows-per-request", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-iterations", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit-transform-chunk-vectors", type=int, default=4096)
    parser.add_argument("--export-chunk-rows", type=int, default=32)
    parser.add_argument("--feature-dtype", default="float16")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-reference-residual", type=Path)
    parser.add_argument("--audit-max-rows", type=int, default=256)
    parser.add_argument(
        "--audit-layers", type=_parse_layers, default=(0, 12, 20, 28, 36, 38, 39)
    )
    parser.add_argument("--reference-probe-recall", type=float)
    parser.add_argument("--candidate-probe-recall", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command_argv = [sys.executable, "-m", "harp8.prepare_jspace_features"] + list(
        sys.argv[1:] if argv is None else argv
    )
    if args.audit_only:
        if args.audit_reference_residual is None:
            raise SystemExit("--audit-only requires --audit-reference-residual")
        manifest = run_precision_audit(
            args.residual_npy,
            args.audit_reference_residual,
            args.lens_checkpoint,
            args.requests_jsonl,
            args.output_root,
            maximum_rows=args.audit_max_rows,
            layers=args.audit_layers,
            rows_per_request=args.rows_per_request,
            seed=args.seed,
            chunk_rows=args.export_chunk_rows,
            reference_probe_recall=args.reference_probe_recall,
            candidate_probe_recall=args.candidate_probe_recall,
            command_argv=command_argv,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["audit"]["accepted"] else 2
    manifest = prepare_feature_stores(
        args.residual_npy,
        args.lens_checkpoint,
        args.requests_jsonl,
        args.output_root,
        representations=args.representations,
        pca_ranks=args.pca_ranks,
        fit_split=args.fit_split,
        maximum_fit_vectors=args.max_fit_rows,
        rows_per_request=args.rows_per_request,
        seed=args.seed,
        pca_iterations=args.pca_iterations,
        pca_device=args.device,
        fit_transform_chunk_vectors=args.fit_transform_chunk_vectors,
        export_chunk_rows=args.export_chunk_rows,
        feature_dtype=args.feature_dtype,
        command_argv=command_argv,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
