"""Dense J-Lens feature preparation and matched residual controls.

The target J-Lens is defined at the post-block residual boundary.  For source
layers 0..38 it transports a row-vector residual with ``xplus @ J[layer].T``.
Layer 39 is already in the final-layer coordinate system and is therefore an
explicit identity endpoint.  All transport and PCA arithmetic is accumulated
in FP32; storage precision is selected independently by the caller.

This module deliberately contains no model or training code.  Its outputs are
immutable, provenance-rich feature stores that can be shared by matched J,
raw-residual, and deterministic random-orthogonal experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FEATURE_SCHEMA = "harp8_dense_jspace_features_v1"
PCA_SCHEMA = "harp8_shared_train_pca_v1"
PRECISION_AUDIT_SCHEMA = "harp8_jspace_precision_audit_v1"
TARGET_LAYERS = 40
LENS_LAYERS = 39


def sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without reading a large array at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray, *, chunk_rows: int = 256) -> str:
    """Hash array metadata and C-order values deterministically in chunks."""

    array = np.asanyarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    if array.ndim == 0:
        digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()
    for start in range(0, array.shape[0], chunk_rows):
        digest.update(np.ascontiguousarray(array[start : start + chunk_rows]).tobytes())
    return digest.hexdigest()


def _to_numpy_float32(value: Any) -> np.ndarray:
    """Convert NumPy or CPU/GPU tensor-like input to a NumPy FP32 array."""

    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def load_j_lens(
    path: Path,
    *,
    expected_width: int | None = 2048,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Load the project J-Lens checkpoint without importing J-Route internals."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency gate
        raise RuntimeError("torch is required to load a J-Lens checkpoint") from exc
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("J"), Mapping):
        raise ValueError("J-Lens checkpoint lacks a J matrix mapping")
    matrices = {int(layer): _to_numpy_float32(value) for layer, value in artifact["J"].items()}
    width = validate_j_lens(matrices, expected_width=expected_width)
    metadata = {
        "path": str(Path(path)),
        "sha256": sha256_file(Path(path)),
        "width": width,
        "layers": list(range(LENS_LAYERS)),
        "n_prompts": int(artifact.get("n_prompts", -1)),
    }
    return matrices, metadata


def validate_j_lens(
    matrices: Mapping[int, Any],
    *,
    expected_width: int | None = None,
) -> int:
    """Validate exact 0..38 coverage and return the common hidden width."""

    keys = {int(layer) for layer in matrices}
    if keys != set(range(LENS_LAYERS)):
        missing = sorted(set(range(LENS_LAYERS)) - keys)
        extra = sorted(keys - set(range(LENS_LAYERS)))
        raise ValueError(
            "J-Lens must contain exactly layers 0..38; "
            f"missing={missing}, extra={extra}"
        )
    widths: set[int] = set()
    for layer in range(LENS_LAYERS):
        matrix = matrices[layer]
        shape = tuple(int(value) for value in matrix.shape)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"J-Lens layer {layer} is not square: {shape}")
        widths.add(shape[0])
        if not np.isfinite(_to_numpy_float32(matrix)).all():
            raise ValueError(f"J-Lens layer {layer} contains a non-finite value")
    if len(widths) != 1:
        raise ValueError(f"J-Lens matrices disagree on width: {sorted(widths)}")
    width = widths.pop()
    if expected_width is not None and width != expected_width:
        raise ValueError(f"J-Lens width {width} does not equal expected width {expected_width}")
    return width


def transport_layer(
    xplus: Any,
    layer: int,
    matrices: Mapping[int, Any],
) -> np.ndarray:
    """Transport one layer in FP32 using the row-vector ``xplus @ J.T`` rule."""

    if not 0 <= layer < TARGET_LAYERS:
        raise ValueError("target layer must lie in 0..39")
    values = _to_numpy_float32(xplus)
    if values.ndim < 1:
        raise ValueError("post-layer residual must have a hidden dimension")
    if layer == TARGET_LAYERS - 1:
        return np.array(values, dtype=np.float32, copy=True)
    if layer not in matrices:
        raise ValueError(f"J-Lens is missing source layer {layer}")
    matrix = _to_numpy_float32(matrices[layer])
    if matrix.shape != (values.shape[-1], values.shape[-1]):
        raise ValueError(
            f"layer {layer} residual width {values.shape[-1]} and J matrix {matrix.shape} disagree"
        )
    result = np.matmul(values, matrix.T)
    return np.asarray(result, dtype=np.float32)


def transport_all_layers(
    xplus: Any,
    matrices: Mapping[int, Any],
) -> np.ndarray:
    """Transport an ``[..., 40, D]`` post-layer residual array in FP32."""

    values = _to_numpy_float32(xplus)
    if values.ndim < 2 or values.shape[-2] != TARGET_LAYERS:
        raise ValueError("all-layer residuals must have shape [...,40,D]")
    validate_j_lens(matrices, expected_width=int(values.shape[-1]))
    return _transport_all_layers_validated(values, matrices)


def _transport_all_layers_validated(
    values: np.ndarray,
    matrices: Mapping[int, Any],
) -> np.ndarray:
    """Transport after the caller has performed the expensive full lens audit."""

    output = np.empty_like(values, dtype=np.float32)
    for layer in range(TARGET_LAYERS):
        output[..., layer, :] = transport_layer(values[..., layer, :], layer, matrices)
    return output


def normalize_with_log_rms(
    values: Any,
    *,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unit-RMS features and their scalar log-RMS magnitude."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    array = _to_numpy_float32(values)
    if array.ndim < 1 or array.shape[-1] == 0:
        raise ValueError("features must have a non-empty hidden dimension")
    if not np.isfinite(array).all():
        raise ValueError("features contain a non-finite value")
    rms = np.sqrt(np.mean(np.square(array), axis=-1, keepdims=True, dtype=np.float32))
    safe_rms = np.maximum(rms, np.float32(epsilon))
    normalized = np.asarray(array / safe_rms, dtype=np.float32)
    log_rms = np.asarray(np.log(safe_rms[..., 0]), dtype=np.float32)
    return normalized, log_rms


@dataclass(frozen=True)
class RandomOrthogonalControl:
    """A scalable random orthogonal map represented by signed permutations."""

    permutations: np.ndarray
    signs: np.ndarray
    seed: int
    shared_across_layers: bool

    @property
    def layers(self) -> int:
        return int(self.permutations.shape[0])

    @property
    def width(self) -> int:
        return int(self.permutations.shape[1])

    def apply_layer(self, values: Any, layer: int) -> np.ndarray:
        if not 0 <= layer < self.layers:
            raise ValueError(f"control layer must lie in 0..{self.layers - 1}")
        array = _to_numpy_float32(values)
        if array.shape[-1] != self.width:
            raise ValueError("random-control input width disagrees with its map")
        return np.asarray(
            array[..., self.permutations[layer]] * self.signs[layer],
            dtype=np.float32,
        )

    def matrix(self, layer: int) -> np.ndarray:
        """Materialize Q where ``apply_layer(x) == x @ Q.T`` (for audits/tests)."""

        matrix = np.zeros((self.width, self.width), dtype=np.float32)
        matrix[np.arange(self.width), self.permutations[layer]] = self.signs[layer]
        return matrix

    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.permutations, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(self.signs, dtype="i1").tobytes())
        digest.update(str(self.seed).encode("ascii"))
        digest.update(str(int(self.shared_across_layers)).encode("ascii"))
        return digest.hexdigest()


def make_random_orthogonal_control(
    width: int,
    *,
    layers: int = TARGET_LAYERS,
    seed: int = 42,
    shared_across_layers: bool = False,
) -> RandomOrthogonalControl:
    """Create a deterministic exact-orthogonal signed-permutation control."""

    if width <= 0 or layers <= 0:
        raise ValueError("width and layers must be positive")
    permutations = np.empty((layers, width), dtype=np.int64)
    signs = np.empty((layers, width), dtype=np.int8)
    shared_permutation: np.ndarray | None = None
    shared_signs: np.ndarray | None = None
    for layer in range(layers):
        if shared_across_layers and shared_permutation is not None:
            permutations[layer] = shared_permutation
            signs[layer] = shared_signs
            continue
        effective_layer = 0 if shared_across_layers else layer
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), effective_layer]))
        permutation = rng.permutation(width).astype(np.int64)
        layer_signs = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=width)
        permutations[layer] = permutation
        signs[layer] = layer_signs
        if shared_across_layers:
            shared_permutation = permutation
            shared_signs = layer_signs
    return RandomOrthogonalControl(
        permutations=permutations,
        signs=signs,
        seed=int(seed),
        shared_across_layers=bool(shared_across_layers),
    )


def deterministic_stratified_sample(
    indices: Sequence[int] | np.ndarray,
    strata: Sequence[Any] | np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Sample as evenly as possible across strata, deterministically.

    The returned *values* come from ``indices`` rather than being positions into
    that array.  Group order and within-group order are both seeded, avoiding a
    lexical bias when ``maximum`` is smaller than the number of strata.
    """

    items = np.asarray(indices, dtype=np.int64)
    # ``np.asarray(list_of_tuples, dtype=object)`` is still two-dimensional.
    # Preserve compound labels as scalar Python objects instead.
    labels = np.empty(len(strata), dtype=object)
    labels[:] = list(strata)
    if items.ndim != 1 or labels.ndim != 1 or len(items) != len(labels):
        raise ValueError("indices and strata must be aligned one-dimensional arrays")
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if len(items) == 0:
        raise ValueError("cannot sample an empty population")
    if len(np.unique(items)) != len(items):
        raise ValueError("sampling indices must be unique")
    if maximum >= len(items):
        return np.sort(items.copy())

    groups: dict[str, list[int]] = {}
    for item, label in zip(items.tolist(), labels.tolist(), strict=True):
        key = json.dumps(label, ensure_ascii=False, sort_keys=True, default=str)
        groups.setdefault(key, []).append(int(item))
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    group_order = [keys[position] for position in rng.permutation(len(keys))]
    shuffled: dict[str, np.ndarray] = {}
    for key in keys:
        values = np.asarray(sorted(groups[key]), dtype=np.int64)
        rng.shuffle(values)
        shuffled[key] = values

    offsets = {key: 0 for key in keys}
    selected: list[int] = []
    while len(selected) < maximum:
        progressed = False
        for key in group_order:
            offset = offsets[key]
            if offset >= len(shuffled[key]):
                continue
            selected.append(int(shuffled[key][offset]))
            offsets[key] += 1
            progressed = True
            if len(selected) == maximum:
                break
        if not progressed:  # defensive: total group capacity was exhausted
            break
    if len(selected) != maximum:
        raise AssertionError("stratified sampler did not fill the requested sample")
    return np.sort(np.asarray(selected, dtype=np.int64))


def build_train_fit_pairs(
    train_mask: Sequence[bool] | np.ndarray,
    *,
    layers: Sequence[int] = tuple(range(TARGET_LAYERS)),
    maximum_vectors: int = 30_000,
    seed: int = 42,
    row_strata: Sequence[Sequence[Any] | np.ndarray] | None = None,
) -> np.ndarray:
    """Build deterministic ``[row, layer]`` PCA samples from train rows only.

    ``row_strata`` is a hook for immutable request/domain/group labels.  Multiple
    arrays are combined into one stratum before crossing with source layer.
    """

    mask = np.asarray(train_mask, dtype=bool)
    if mask.ndim != 1 or not mask.any():
        raise ValueError("train_mask must be one-dimensional and contain train rows")
    selected_layers = np.asarray(tuple(int(layer) for layer in layers), dtype=np.int64)
    if (
        selected_layers.ndim != 1
        or len(selected_layers) == 0
        or len(np.unique(selected_layers)) != len(selected_layers)
        or (selected_layers < 0).any()
        or (selected_layers >= TARGET_LAYERS).any()
    ):
        raise ValueError("layers must be unique values in 0..39")
    if maximum_vectors <= 0:
        raise ValueError("maximum_vectors must be positive")

    strata_arrays = (
        []
        if row_strata is None
        else [np.asarray(value, dtype=object) for value in row_strata]
    )
    if any(value.ndim != 1 or len(value) != len(mask) for value in strata_arrays):
        raise ValueError("every row-stratum array must align with train_mask")
    rows = np.flatnonzero(mask).astype(np.int64)
    if strata_arrays:
        row_labels = [
            tuple(value[row] for value in strata_arrays)
            for row in rows.tolist()
        ]
    else:
        row_labels = [("all",) for _ in rows]
    target = min(maximum_vectors, len(rows) * len(selected_layers))
    base, remainder = divmod(target, len(selected_layers))
    layer_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x4A]))
    extra_layers = set(
        int(value)
        for value in layer_rng.choice(
            selected_layers,
            size=remainder,
            replace=False,
        ).tolist()
    )
    pair_blocks: list[np.ndarray] = []
    for layer in selected_layers.tolist():
        quota = base + int(layer in extra_layers)
        if quota == 0:
            continue
        sampled_rows = deterministic_stratified_sample(
            rows,
            row_labels,
            maximum=quota,
            seed=int(
                np.random.SeedSequence([int(seed), int(layer), 0x50])
                .generate_state(1)[0]
            ),
        )
        pair_blocks.append(
            np.stack(
                [sampled_rows, np.full(len(sampled_rows), layer, dtype=np.int64)],
                axis=1,
            )
        )
    pairs = np.concatenate(pair_blocks, axis=0)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return np.asarray(pairs[order], dtype=np.int64)


def _canonicalize_component_signs(components: np.ndarray) -> np.ndarray:
    result = np.asarray(components, dtype=np.float32).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1
    return result


@dataclass(frozen=True)
class SharedPCA:
    """A shared coordinate projection fitted only on explicit train pairs."""

    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    fit_pairs_sha256: str
    fit_vectors: int
    seed: int
    captured_variance_fraction: float

    @property
    def input_width(self) -> int:
        return int(self.components.shape[0])

    @property
    def rank(self) -> int:
        return int(self.components.shape[1])

    def transform(self, values: Any) -> np.ndarray:
        array = _to_numpy_float32(values)
        if array.shape[-1] != self.input_width:
            raise ValueError("PCA input width disagrees with fitted components")
        return np.asarray((array - self.mean) @ self.components, dtype=np.float32)

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for value in (self.mean, self.components, self.singular_values):
            digest.update(np.ascontiguousarray(value, dtype="<f4").tobytes())
        digest.update(self.fit_pairs_sha256.encode("ascii"))
        digest.update(str(self.fit_vectors).encode("ascii"))
        digest.update(str(self.seed).encode("ascii"))
        return digest.hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": PCA_SCHEMA,
            "input_width": self.input_width,
            "rank": self.rank,
            "fit_vectors": self.fit_vectors,
            "fit_pairs_sha256": self.fit_pairs_sha256,
            "seed": self.seed,
            "captured_train_variance_fraction": self.captured_variance_fraction,
            "sha256": self.sha256(),
        }


def fit_shared_pca(
    values: np.ndarray,
    fit_pairs: np.ndarray,
    *,
    rank: int,
    seed: int = 42,
    iterations: int = 6,
    device: str = "cpu",
) -> SharedPCA:
    """Fit one PCA basis across layers using only caller-supplied train pairs."""

    source = np.asanyarray(values)
    pairs = np.asarray(fit_pairs, dtype=np.int64)
    if source.ndim != 3:
        raise ValueError("shared PCA source must have shape [rows,layers,width]")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) < 2:
        raise ValueError("fit_pairs must have shape [N,2] with at least two vectors")
    if (
        (pairs[:, 0] < 0).any()
        or (pairs[:, 0] >= source.shape[0]).any()
        or (pairs[:, 1] < 0).any()
        or (pairs[:, 1] >= source.shape[1]).any()
    ):
        raise ValueError("fit pair lies outside the source array")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("fit pairs must be unique")
    width = int(source.shape[2])
    maximum_rank = min(len(pairs) - 1, width)
    if not 0 < rank <= maximum_rank:
        raise ValueError(f"rank must lie in 1..{maximum_rank}")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    matrix = np.asarray(source[pairs[:, 0], pairs[:, 1]], dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("PCA fit vectors contain a non-finite value")
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = np.asarray(matrix - mean, dtype=np.float32)
    total_energy = float(np.square(centered, dtype=np.float64).sum())
    if total_energy <= 0:
        raise ValueError("PCA fit vectors have zero centered variance")

    if device == "numpy":
        _u, singular, vectors_t = np.linalg.svd(centered, full_matrices=False)
        components = vectors_t[:rank].T
        singular = singular[:rank]
    else:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - production dependency gate
            raise RuntimeError("torch is required for CPU/CUDA randomized PCA") from exc
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA PCA requested but CUDA is unavailable")
        devices = [target.index or 0] if target.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if target.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            tensor = torch.as_tensor(centered, dtype=torch.float32, device=target)
            _u, singular_tensor, components_tensor = torch.pca_lowrank(
                tensor,
                q=rank,
                center=False,
                niter=iterations,
            )
            components = components_tensor.cpu().numpy()
            singular = singular_tensor.cpu().numpy()
    components = _canonicalize_component_signs(components)
    singular = np.asarray(singular, dtype=np.float32)
    captured = float(np.square(singular, dtype=np.float64).sum() / total_energy)
    pairs_hash = hashlib.sha256(np.ascontiguousarray(pairs, dtype="<i8").tobytes()).hexdigest()
    return SharedPCA(
        mean=mean,
        components=components,
        singular_values=singular,
        fit_pairs_sha256=pairs_hash,
        fit_vectors=int(len(pairs)),
        seed=int(seed),
        captured_variance_fraction=captured,
    )


def fit_shared_pca_from_train(
    values: np.ndarray,
    train_mask: Sequence[bool] | np.ndarray,
    *,
    rank: int,
    layers: Sequence[int] = tuple(range(TARGET_LAYERS)),
    maximum_vectors: int = 30_000,
    row_strata: Sequence[Sequence[Any] | np.ndarray] | None = None,
    seed: int = 42,
    iterations: int = 6,
    device: str = "cpu",
) -> tuple[SharedPCA, np.ndarray]:
    """Fit shared PCA through the explicit train-only stratified sampler."""

    pairs = build_train_fit_pairs(
        train_mask,
        layers=layers,
        maximum_vectors=maximum_vectors,
        seed=seed,
        row_strata=row_strata,
    )
    pca = fit_shared_pca(
        values,
        pairs,
        rank=rank,
        seed=seed,
        iterations=iterations,
        device=device,
    )
    return pca, pairs


def save_shared_pca(pca: SharedPCA, path: Path) -> dict[str, Any]:
    """Persist PCA arrays losslessly and return hash-bearing metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            schema=np.asarray(PCA_SCHEMA),
            mean=np.asarray(pca.mean, dtype="<f4"),
            components=np.asarray(pca.components, dtype="<f4"),
            singular_values=np.asarray(pca.singular_values, dtype="<f4"),
            fit_pairs_sha256=np.asarray(pca.fit_pairs_sha256),
            fit_vectors=np.asarray(pca.fit_vectors, dtype="<i8"),
            seed=np.asarray(pca.seed, dtype="<i8"),
            captured_variance_fraction=np.asarray(
                pca.captured_variance_fraction, dtype="<f8"
            ),
        )
    return {**pca.metadata(), "path": str(path), "file_sha256": sha256_file(path)}


def load_shared_pca(path: Path) -> SharedPCA:
    """Load a lossless shared-PCA artifact and validate its internal digest."""

    with np.load(Path(path), allow_pickle=False) as artifact:
        if str(artifact["schema"].item()) != PCA_SCHEMA:
            raise ValueError("unsupported shared-PCA schema")
        return SharedPCA(
            mean=np.asarray(artifact["mean"], dtype=np.float32),
            components=np.asarray(artifact["components"], dtype=np.float32),
            singular_values=np.asarray(artifact["singular_values"], dtype=np.float32),
            fit_pairs_sha256=str(artifact["fit_pairs_sha256"].item()),
            fit_vectors=int(artifact["fit_vectors"].item()),
            seed=int(artifact["seed"].item()),
            captured_variance_fraction=float(artifact["captured_variance_fraction"].item()),
        )


def _source_array(source: Path | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        provenance = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    else:
        array = np.asanyarray(source)
        provenance = {"path": None, "array_sha256": _array_sha256(array)}
    return array, provenance


def _transform_chunk(
    chunk: np.ndarray,
    *,
    representation: str,
    matrices: Mapping[int, Any] | None,
    orthogonal_control: RandomOrthogonalControl | None,
) -> np.ndarray:
    if representation == "j_lens":
        if matrices is None:
            raise ValueError("j_lens export requires matrices")
        return _transport_all_layers_validated(
            np.asarray(chunk, dtype=np.float32), matrices
        )
    values = np.asarray(chunk, dtype=np.float32)
    if representation == "raw":
        return np.array(values, copy=True)
    if representation in {"random_orthogonal_shared", "random_orthogonal_per_layer"}:
        if orthogonal_control is None:
            raise ValueError("random-orthogonal export requires a control map")
        expected_shared = representation == "random_orthogonal_shared"
        if orthogonal_control.shared_across_layers != expected_shared:
            raise ValueError("random-control sharing mode disagrees with representation")
        output = np.empty_like(values, dtype=np.float32)
        for layer in range(TARGET_LAYERS):
            output[:, layer] = orthogonal_control.apply_layer(values[:, layer], layer)
        return output
    raise ValueError(f"unsupported feature representation {representation!r}")


def export_feature_store(
    source: Path | np.ndarray,
    output_dir: Path,
    *,
    representation: str,
    matrices: Mapping[int, Any] | None = None,
    lens_provenance: Mapping[str, Any] | None = None,
    orthogonal_control: RandomOrthogonalControl | None = None,
    pca: SharedPCA | None = None,
    feature_dtype: str | np.dtype = "<f2",
    log_rms_dtype: str | np.dtype = "<f4",
    chunk_rows: int = 32,
) -> dict[str, Any]:
    """Chunk-transform an all-layer residual array into immutable NPY stores."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse feature directory {output_dir}")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    source_array, source_provenance = _source_array(source)
    if source_array.ndim != 3 or source_array.shape[1] != TARGET_LAYERS:
        raise ValueError("source residuals must have shape [rows,40,width]")
    if source_array.shape[0] == 0 or source_array.shape[2] == 0:
        raise ValueError("source residual array must be non-empty")
    width = int(source_array.shape[2])
    if representation == "j_lens":
        if matrices is None:
            raise ValueError("j_lens export requires matrices")
        validate_j_lens(matrices, expected_width=width)
    if representation.startswith("random_orthogonal"):
        if orthogonal_control is None:
            raise ValueError("random-orthogonal export requires a control map")
        if orthogonal_control.layers != TARGET_LAYERS or orthogonal_control.width != width:
            raise ValueError("random-control geometry disagrees with source residuals")
    output_width = width if pca is None else pca.rank
    if pca is not None and pca.input_width != width:
        raise ValueError("PCA width disagrees with source residuals")
    feature_dtype = np.dtype(feature_dtype)
    log_rms_dtype = np.dtype(log_rms_dtype)
    if feature_dtype.kind != "f" or log_rms_dtype.kind != "f":
        raise ValueError("feature and log-RMS storage dtypes must be floating-point")

    output_dir.mkdir(parents=True)
    features_path = output_dir / "features_normalized.npy"
    magnitude_path = output_dir / "log_rms.npy"
    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=feature_dtype,
        shape=(source_array.shape[0], TARGET_LAYERS, output_width),
    )
    log_rms = np.lib.format.open_memmap(
        magnitude_path,
        mode="w+",
        dtype=log_rms_dtype,
        shape=(source_array.shape[0], TARGET_LAYERS),
    )
    for start in range(0, source_array.shape[0], chunk_rows):
        stop = min(source_array.shape[0], start + chunk_rows)
        transformed = _transform_chunk(
            np.asarray(source_array[start:stop]),
            representation=representation,
            matrices=matrices,
            orthogonal_control=orthogonal_control,
        )
        normalized, magnitude = normalize_with_log_rms(transformed)
        if pca is not None:
            normalized = pca.transform(normalized)
        if not np.isfinite(normalized).all() or not np.isfinite(magnitude).all():
            raise ValueError(f"non-finite feature in source rows {start}:{stop}")
        features[start:stop] = normalized
        log_rms[start:stop] = magnitude
    features.flush()
    log_rms.flush()

    outputs = {}
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
        "normalization": "unit_rms_with_separate_natural_log_rms",
        "rows": int(source_array.shape[0]),
        "layers": TARGET_LAYERS,
        "input_width": width,
        "feature_width": output_width,
        "layer_39_transport": "identity",
        "source": source_provenance,
        "outputs": outputs,
        "pca": None if pca is None else pca.metadata(),
    }
    if representation == "j_lens":
        manifest["lens"] = dict(lens_provenance or {})
        if not manifest["lens"].get("sha256"):
            digest = hashlib.sha256()
            assert matrices is not None
            for layer in range(LENS_LAYERS):
                digest.update(np.ascontiguousarray(_to_numpy_float32(matrices[layer])).tobytes())
            manifest["lens"]["matrix_content_sha256"] = digest.hexdigest()
    if orthogonal_control is not None:
        manifest["random_orthogonal_control"] = {
            "construction": "signed_permutation",
            "seed": orthogonal_control.seed,
            "shared_across_layers": orthogonal_control.shared_across_layers,
            "sha256": orthogonal_control.sha256(),
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


@dataclass(frozen=True)
class PrecisionAuditThresholds:
    """Pre-training gates for reduced-precision residual/J feature storage."""

    minimum_mean_cosine: float = 0.9995
    minimum_row_cosine: float = 0.995
    maximum_probe_recall_delta: float = 0.001

    def __post_init__(self) -> None:
        if not -1.0 <= self.minimum_mean_cosine <= 1.0:
            raise ValueError("minimum_mean_cosine must lie in [-1,1]")
        if not -1.0 <= self.minimum_row_cosine <= 1.0:
            raise ValueError("minimum_row_cosine must lie in [-1,1]")
        if self.maximum_probe_recall_delta < 0:
            raise ValueError("maximum_probe_recall_delta must be non-negative")


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1, dtype=np.float64)
    left_norm = np.sqrt(np.sum(np.square(left), axis=-1, dtype=np.float64))
    right_norm = np.sqrt(np.sum(np.square(right), axis=-1, dtype=np.float64))
    denominator = left_norm * right_norm
    if (denominator <= 0).any():
        raise ValueError("precision audit encountered a zero-norm transported state")
    return np.asarray(numerator / denominator, dtype=np.float64)


def audit_feature_precision(
    reference_residuals: np.ndarray,
    candidate_residuals: np.ndarray,
    matrices: Mapping[int, Any],
    *,
    layers: Iterable[int] = tuple(range(TARGET_LAYERS)),
    thresholds: PrecisionAuditThresholds = PrecisionAuditThresholds(),
    reference_probe_recall: float | None = None,
    candidate_probe_recall: float | None = None,
    chunk_rows: int = 32,
) -> dict[str, Any]:
    """Audit precision after applying the same J transport to both captures."""

    reference = np.asanyarray(reference_residuals)
    candidate = np.asanyarray(candidate_residuals)
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("precision-audit residual arrays must share [rows,layers,width]")
    if reference.shape[1] != TARGET_LAYERS:
        raise ValueError("precision audit expects 40 target layers")
    validate_j_lens(matrices, expected_width=int(reference.shape[2]))
    selected_layers = tuple(int(layer) for layer in layers)
    if (
        not selected_layers
        or len(set(selected_layers)) != len(selected_layers)
        or any(layer < 0 or layer >= TARGET_LAYERS for layer in selected_layers)
    ):
        raise ValueError("precision-audit layers must be unique values in 0..39")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if (reference_probe_recall is None) != (candidate_probe_recall is None):
        raise ValueError("probe recalls must either both be supplied or both omitted")

    cosine_sum = 0.0
    cosine_count = 0
    minimum_cosine = 1.0
    square_error = 0.0
    value_count = 0
    maximum_absolute_error = 0.0
    finite = True
    for start in range(0, reference.shape[0], chunk_rows):
        stop = min(reference.shape[0], start + chunk_rows)
        for layer in selected_layers:
            left = transport_layer(reference[start:stop, layer], layer, matrices)
            right = transport_layer(candidate[start:stop, layer], layer, matrices)
            finite = finite and bool(np.isfinite(left).all() and np.isfinite(right).all())
            if not finite:
                continue
            cosines = _row_cosine(left, right)
            cosine_sum += float(cosines.sum())
            cosine_count += int(cosines.size)
            minimum_cosine = min(minimum_cosine, float(cosines.min()))
            difference = left.astype(np.float64) - right.astype(np.float64)
            square_error += float(np.square(difference).sum())
            value_count += int(difference.size)
            maximum_absolute_error = max(maximum_absolute_error, float(np.abs(difference).max()))
    if not finite or cosine_count == 0:
        mean_cosine = float("nan")
        rmse = float("nan")
    else:
        mean_cosine = cosine_sum / cosine_count
        rmse = float(np.sqrt(square_error / value_count))
    recall_delta = (
        None
        if reference_probe_recall is None
        else abs(float(reference_probe_recall) - float(candidate_probe_recall))
    )
    accepted = (
        finite
        and mean_cosine >= thresholds.minimum_mean_cosine
        and minimum_cosine >= thresholds.minimum_row_cosine
        and (
            recall_delta is None
            or recall_delta <= thresholds.maximum_probe_recall_delta
        )
    )
    return {
        "schema": PRECISION_AUDIT_SCHEMA,
        "accepted": bool(accepted),
        "rows": int(reference.shape[0]),
        "layers": list(selected_layers),
        "vectors_compared": cosine_count,
        "finite": bool(finite),
        "mean_transport_cosine": mean_cosine,
        "minimum_transport_cosine": minimum_cosine,
        "transport_rmse": rmse,
        "maximum_transport_absolute_error": maximum_absolute_error,
        "probe_recall_delta": recall_delta,
        "thresholds": {
            "minimum_mean_cosine": thresholds.minimum_mean_cosine,
            "minimum_row_cosine": thresholds.minimum_row_cosine,
            "maximum_probe_recall_delta": thresholds.maximum_probe_recall_delta,
        },
    }
