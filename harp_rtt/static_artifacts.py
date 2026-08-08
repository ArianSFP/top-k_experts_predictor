"""Extract immutable frozen-target artifacts for HARP-RTT.

The extractor reads only named tensors from a sharded Hugging Face
``model.safetensors.index.json``.  It never constructs a Transformers model
and never deserializes unrelated checkpoint tensors.  The output contains the
centered-router geometry used by the predictor and the exact frozen token /
final-normalization tensors needed to reconstruct causal token features.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .geometry import (
    ROUTER_GEOMETRY_SCHEMA,
    CenteredRouterGeometry,
    assert_router_geometry_equivalent,
    build_centered_router_geometry,
)


STATIC_ARTIFACT_SCHEMA = "harp_rtt_static_target_artifacts_v1"
FROZEN_TARGET_SCHEMA = "harp_rtt_frozen_target_features_v1"
MANIFEST_FILENAME = "manifest.json"
GEOMETRY_FILENAME = "router_geometry.safetensors"
FROZEN_TARGET_FILENAME = "frozen_target.safetensors"
SOURCE_CONFIG_FILENAME = "source_config.json"


@dataclass(frozen=True)
class StaticTargetContract:
    """Exact checkpoint contract expected by one static-artifact build."""

    architecture_names: tuple[str, ...]
    outer_model_type: str
    text_model_type: str
    layers: int
    experts: int
    experts_per_token: int
    hidden_width: int
    vocabulary_size: int
    rms_norm_epsilon: float
    checkpoint_dtype: str = "bfloat16"
    require_shared_mtp_embedding: bool = True

    def validate(self) -> None:
        if not self.architecture_names or any(not name for name in self.architecture_names):
            raise ValueError("architecture_names must be nonempty strings")
        if not self.outer_model_type or not self.text_model_type:
            raise ValueError("model types must be nonempty")
        if self.layers < 1 or self.experts < 2 or self.hidden_width < 1:
            raise ValueError("contract requires positive layers/hidden and at least two experts")
        if not 1 <= self.experts_per_token <= self.experts:
            raise ValueError("experts_per_token lies outside the expert count")
        if self.vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        if not math.isfinite(self.rms_norm_epsilon) or self.rms_norm_epsilon <= 0:
            raise ValueError("rms_norm_epsilon must be finite and positive")
        if self.checkpoint_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("unsupported checkpoint_dtype")


HARP_RTT_TARGET_CONTRACT = StaticTargetContract(
    architecture_names=("Qwen3_5MoeForConditionalGeneration",),
    outer_model_type="qwen3_5_moe",
    text_model_type="qwen3_5_moe_text",
    layers=40,
    experts=256,
    experts_per_token=8,
    hidden_width=2048,
    vocabulary_size=248320,
    rms_norm_epsilon=1e-6,
    checkpoint_dtype="bfloat16",
    require_shared_mtp_embedding=True,
)


@dataclass(frozen=True)
class ResolvedTargetKeys:
    router_weights: tuple[str, ...]
    router_biases: tuple[str, ...]
    token_embedding: str
    final_rmsnorm_weight: str

    @property
    def has_router_bias(self) -> bool:
        return bool(self.router_biases)


@dataclass(frozen=True)
class TensorHeader:
    shape: tuple[int, ...]
    dtype: str
    shard: str


@dataclass(frozen=True)
class StaticTargetArtifacts:
    """Verified tensors loaded from a completed static-artifact directory."""

    geometry: CenteredRouterGeometry
    token_embedding: Tensor | None
    final_rmsnorm_weight: Tensor
    final_rmsnorm_epsilon: float
    manifest: Mapping[str, Any]

    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        if self.token_embedding is None:
            raise RuntimeError("token embedding was not loaded")
        return frozen_token_embeddings(token_ids, self.token_embedding)

    def reconstruct_final_hidden(
        self,
        hidden_states: Tensor,
        *,
        activation_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> Tensor:
        return reconstruct_final_hidden_rmsnorm(
            hidden_states,
            self.final_rmsnorm_weight,
            self.final_rmsnorm_epsilon,
            activation_dtype=activation_dtype,
            output_dtype=output_dtype,
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_content_sha256(tensor: Tensor) -> str:
    """Hash the exact contiguous tensor bytes without a large bytes copy."""

    value = tensor.detach().to(device="cpu").contiguous()
    byte_view = value.view(torch.uint8).numpy()
    digest = hashlib.sha256()
    digest.update(memoryview(byte_view))
    return digest.hexdigest()


def _tensor_collection_sha256(values: Sequence[tuple[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in values:
        value = tensor.detach().to(device="cpu").contiguous()
        header = _canonical_json(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(memoryview(value.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return normalized


def _require_revision(value: str) -> str:
    normalized = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{40,64}", normalized) is None:
        raise ValueError("expected_revision must be a pinned hexadecimal commit")
    return normalized


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read valid {label} JSON at {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _safetensors_api() -> tuple[Any, Any]:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("safetensors is required for static artifact extraction") from exc
    return safe_open, save_file


class _IndexedSafetensors:
    """Read selected tensors through a validated sharded index."""

    def __init__(self, model_path: Path, index: Mapping[str, Any]) -> None:
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model index has no nonempty weight_map")
        parsed: dict[str, str] = {}
        for key, shard in weight_map.items():
            if not isinstance(key, str) or not key or not isinstance(shard, str) or not shard:
                raise ValueError("model index weight_map must map nonempty strings")
            shard_path = Path(shard)
            if shard_path.is_absolute() or len(shard_path.parts) != 1 or shard_path.name != shard:
                raise ValueError(f"unsafe shard path in model index: {shard!r}")
            parsed[key] = shard
        metadata = index.get("metadata")
        total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
        if (
            not isinstance(total_size, (int, float))
            or isinstance(total_size, bool)
            or not math.isfinite(float(total_size))
            or not float(total_size).is_integer()
            or int(total_size) <= 0
        ):
            raise ValueError(
                "model index metadata.total_size must be a positive integer-valued number"
            )
        self.model_path = model_path
        self.weight_map = parsed
        for shard in self.shards:
            path = self.model_path / shard
            if not path.is_file():
                raise ValueError(f"model index references missing shard {shard}")

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self.weight_map)

    @property
    def shards(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.weight_map.values())))

    def _group(self, names: Sequence[str]) -> dict[str, list[str]]:
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for name in names:
            shard = self.weight_map.get(name)
            if shard is None:
                raise ValueError(f"model index does not map required tensor {name}")
            grouped[shard].append(name)
        return dict(grouped)

    def inspect(self, names: Sequence[str]) -> dict[str, TensorHeader]:
        safe_open, _ = _safetensors_api()
        result: dict[str, TensorHeader] = {}
        for shard, shard_names in sorted(self._group(names).items()):
            with safe_open(
                str(self.model_path / shard), framework="pt", device="cpu"
            ) as handle:
                actual_keys = set(handle.keys())
                for name in shard_names:
                    if name not in actual_keys:
                        raise ValueError(
                            f"index maps {name} to {shard}, but the shard does not contain it"
                        )
                    value = handle.get_slice(name)
                    result[name] = TensorHeader(
                        shape=tuple(int(item) for item in value.get_shape()),
                        dtype=str(value.get_dtype()),
                        shard=shard,
                    )
        return result

    def load(self, names: Sequence[str]) -> dict[str, Tensor]:
        safe_open, _ = _safetensors_api()
        result: dict[str, Tensor] = {}
        for shard, shard_names in sorted(self._group(names).items()):
            with safe_open(
                str(self.model_path / shard), framework="pt", device="cpu"
            ) as handle:
                actual_keys = set(handle.keys())
                for name in shard_names:
                    if name not in actual_keys:
                        raise ValueError(
                            f"index maps {name} to {shard}, but the shard does not contain it"
                        )
                    result[name] = handle.get_tensor(name).contiguous()
        return result


def _validate_config(
    config: Mapping[str, Any], contract: StaticTargetContract
) -> Mapping[str, Any]:
    contract.validate()
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or tuple(architectures) != contract.architecture_names:
        raise ValueError(
            f"config architectures mismatch: expected {contract.architecture_names}, "
            f"found {architectures!r}"
        )
    if config.get("model_type") != contract.outer_model_type:
        raise ValueError("outer model_type disagrees with the static target contract")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ValueError("config text_config must be an object")
    expected = {
        "model_type": contract.text_model_type,
        "num_hidden_layers": contract.layers,
        "num_experts": contract.experts,
        "num_experts_per_tok": contract.experts_per_token,
        "hidden_size": contract.hidden_width,
        "vocab_size": contract.vocabulary_size,
    }
    for key, wanted in expected.items():
        if text_config.get(key) != wanted:
            raise ValueError(
                f"text config {key} mismatch: expected {wanted!r}, "
                f"found {text_config.get(key)!r}"
            )
    configured_dtype = text_config.get("dtype", text_config.get("torch_dtype"))
    if configured_dtype != contract.checkpoint_dtype:
        raise ValueError(
            f"text config dtype mismatch: expected {contract.checkpoint_dtype!r}, "
            f"found {configured_dtype!r}"
        )
    epsilon = text_config.get("rms_norm_eps")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise ValueError("text config rms_norm_eps must be numeric")
    if float(epsilon) != float(contract.rms_norm_epsilon):
        raise ValueError(
            f"text config rms_norm_eps mismatch: expected {contract.rms_norm_epsilon}, "
            f"found {epsilon}"
        )
    if contract.require_shared_mtp_embedding and text_config.get(
        "mtp_use_dedicated_embeddings"
    ) is not False:
        raise ValueError("HARP-RTT requires mtp_use_dedicated_embeddings=false")
    layer_types = text_config.get("layer_types")
    if layer_types is not None and (
        not isinstance(layer_types, list) or len(layer_types) != contract.layers
    ):
        raise ValueError("text config layer_types length disagrees with num_hidden_layers")
    return text_config


def _resolve_target_keys(
    keys: frozenset[str], contract: StaticTargetContract
) -> ResolvedTargetKeys:
    router_prefixes = (
        "model.language_model.layers",
        "model.layers",
        "language_model.layers",
    )
    complete: list[tuple[str, tuple[str, ...]]] = []
    for prefix in router_prefixes:
        candidates = tuple(
            f"{prefix}.{layer}.mlp.gate.weight" for layer in range(contract.layers)
        )
        present = sum(name in keys for name in candidates)
        if present == contract.layers:
            complete.append((prefix, candidates))
        elif present:
            raise ValueError(
                f"partial target router key family {prefix}: "
                f"found {present}/{contract.layers} weights"
            )
    if len(complete) != 1:
        raise ValueError(
            "expected exactly one complete target router key family; "
            f"found {[prefix for prefix, _ in complete]}"
        )
    prefix, router_weights = complete[0]
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.mlp\.gate\.weight$")
    indexed_layers = sorted(
        int(match.group(1))
        for key in keys
        if (match := pattern.fullmatch(key)) is not None
    )
    if indexed_layers != list(range(contract.layers)):
        raise ValueError(
            "target router layer keys disagree with config: "
            f"found layers {indexed_layers}"
        )

    router_bias_candidates = tuple(name.removesuffix(".weight") + ".bias" for name in router_weights)
    bias_count = sum(name in keys for name in router_bias_candidates)
    if bias_count not in (0, contract.layers):
        raise ValueError(
            f"partial target router bias family: found {bias_count}/{contract.layers} biases"
        )
    router_biases = router_bias_candidates if bias_count else ()

    embedding_candidates = (
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
    )
    embeddings = tuple(name for name in embedding_candidates if name in keys)
    if len(embeddings) != 1:
        raise ValueError(
            f"expected exactly one shared token embedding key, found {list(embeddings)}"
        )
    norm_candidates = (
        "model.language_model.norm.weight",
        "model.norm.weight",
        "language_model.norm.weight",
    )
    norms = tuple(name for name in norm_candidates if name in keys)
    if len(norms) != 1:
        raise ValueError(f"expected exactly one final RMSNorm key, found {list(norms)}")
    if contract.require_shared_mtp_embedding and any(
        key in keys
        for key in ("mtp.embed_tokens.weight", "mtp.embedding.weight")
    ):
        raise ValueError("checkpoint contains a dedicated MTP embedding despite shared contract")
    return ResolvedTargetKeys(
        router_weights=router_weights,
        router_biases=router_biases,
        token_embedding=embeddings[0],
        final_rmsnorm_weight=norms[0],
    )


def _expected_safetensors_dtype(dtype: str) -> str:
    return {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}[dtype]


def _validate_headers(
    headers: Mapping[str, TensorHeader],
    keys: ResolvedTargetKeys,
    contract: StaticTargetContract,
) -> None:
    expected_dtype = _expected_safetensors_dtype(contract.checkpoint_dtype)
    for name in keys.router_weights:
        header = headers[name]
        if header.shape != (contract.experts, contract.hidden_width):
            raise ValueError(
                f"router {name} shape mismatch: expected "
                f"{(contract.experts, contract.hidden_width)}, found {header.shape}"
            )
        if header.dtype != expected_dtype:
            raise ValueError(
                f"router {name} dtype mismatch: expected {expected_dtype}, found {header.dtype}"
            )
    for name in keys.router_biases:
        header = headers[name]
        if header.shape != (contract.experts,):
            raise ValueError(
                f"router bias {name} shape mismatch: expected {(contract.experts,)}, "
                f"found {header.shape}"
            )
        if header.dtype != expected_dtype:
            raise ValueError(
                f"router bias {name} dtype mismatch: expected {expected_dtype}, "
                f"found {header.dtype}"
            )
    embedding = headers[keys.token_embedding]
    if embedding.shape != (contract.vocabulary_size, contract.hidden_width):
        raise ValueError(
            "token embedding shape mismatch: expected "
            f"{(contract.vocabulary_size, contract.hidden_width)}, found {embedding.shape}"
        )
    if embedding.dtype != expected_dtype:
        raise ValueError(
            f"token embedding dtype mismatch: expected {expected_dtype}, found {embedding.dtype}"
        )
    norm = headers[keys.final_rmsnorm_weight]
    if norm.shape != (contract.hidden_width,):
        raise ValueError(
            f"final RMSNorm shape mismatch: expected {(contract.hidden_width,)}, "
            f"found {norm.shape}"
        )
    if norm.dtype != expected_dtype:
        raise ValueError(
            f"final RMSNorm dtype mismatch: expected {expected_dtype}, found {norm.dtype}"
        )


def _revision_provenance(
    model_path: Path,
    shards: Sequence[str],
    *,
    expected_revision: str,
) -> list[dict[str, Any]]:
    metadata_root = model_path / ".cache" / "huggingface" / "download"
    records: list[dict[str, Any]] = []
    revisions: set[str] = set()
    for shard in shards:
        metadata_path = metadata_root / f"{shard}.metadata"
        if not metadata_path.is_file():
            raise ValueError(f"missing Hugging Face revision metadata for {shard}")
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or not lines[0].strip() or not lines[1].strip():
            raise ValueError(f"invalid Hugging Face revision metadata for {shard}")
        revision = lines[0].strip().lower()
        revisions.add(revision)
        records.append(
            {
                "filename": shard,
                "bytes": (model_path / shard).stat().st_size,
                "repository_revision": revision,
                "huggingface_etag": lines[1].strip(),
            }
        )
    if revisions != {expected_revision}:
        raise ValueError(
            f"checkpoint revision mismatch: expected {expected_revision}, "
            f"found {sorted(revisions)}"
        )
    return records


def reconstruct_final_hidden_rmsnorm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
    *,
    activation_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """Reproduce the Qwen final RMSNorm operation exactly.

    Qwen computes variance in FP32, casts the normalized activation back to
    the incoming activation dtype, and only then applies the frozen gain.  If
    stored residuals have been promoted to FP32, pass ``activation_dtype`` as
    ``torch.bfloat16`` to recover the deployed BF16 operation.
    """

    if not isinstance(hidden_states, Tensor) or not hidden_states.is_floating_point():
        raise TypeError("hidden_states must be a floating-point torch tensor")
    if not isinstance(weight, Tensor) or not weight.is_floating_point() or weight.ndim != 1:
        raise TypeError("weight must be a one-dimensional floating-point torch tensor")
    if hidden_states.ndim < 1 or hidden_states.shape[-1] != weight.numel():
        raise ValueError("RMSNorm input and weight widths disagree")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be finite and positive")
    target_activation_dtype = activation_dtype or hidden_states.dtype
    probe = torch.empty((), dtype=target_activation_dtype)
    if not probe.is_floating_point():
        raise TypeError("activation_dtype must be floating-point")
    if output_dtype is not None and not torch.empty((), dtype=output_dtype).is_floating_point():
        raise TypeError("output_dtype must be floating-point")
    activation = hidden_states.to(dtype=target_activation_dtype)
    normalized_fp32 = activation.float()
    variance = normalized_fp32.square().mean(dim=-1, keepdim=True)
    normalized = (
        normalized_fp32 * torch.rsqrt(variance + float(epsilon))
    ).to(target_activation_dtype)
    result = weight.to(device=hidden_states.device) * normalized
    return result.to(output_dtype) if output_dtype is not None else result


def frozen_token_embeddings(token_ids: Tensor, token_embedding: Tensor) -> Tensor:
    """Perform exact frozen embedding lookup with strict token-ID checks."""

    if not isinstance(token_ids, Tensor) or token_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("token_ids must be an int32 or int64 torch tensor")
    if not isinstance(token_embedding, Tensor) or token_embedding.ndim != 2:
        raise ValueError("token_embedding must be [vocabulary, hidden]")
    if token_ids.numel() and (
        int(token_ids.min()) < 0 or int(token_ids.max()) >= token_embedding.shape[0]
    ):
        raise ValueError("token ID lies outside the frozen embedding vocabulary")
    return torch.nn.functional.embedding(
        token_ids.to(device=token_embedding.device, dtype=torch.int64), token_embedding
    )


def _artifact_metadata(
    *,
    schema: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    values: Mapping[str, Any],
) -> dict[str, str]:
    metadata = {
        "schema": schema,
        "repository_revision": revision,
        "source_config_sha256": config_sha256,
        "source_index_sha256": index_sha256,
    }
    metadata.update({key: str(value) for key, value in values.items()})
    return metadata


def _implementation_provenance(
    source_root: str | Path | None,
) -> dict[str, Any]:
    """Hash either the explicit immutable source tree or core runtime files."""

    runtime_files = {
        Path(__file__).resolve(),
        Path(build_centered_router_geometry.__code__.co_filename).resolve(),
    }
    if source_root is None:
        root = Path(__file__).resolve().parents[1]
        files = sorted(runtime_files)
        scope = "core_runtime_files"
    else:
        root = Path(source_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"implementation_source_root is not a directory: {root}")
        try:
            for path in runtime_files:
                path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "implementation_source_root must contain the executing static and geometry modules"
            ) from exc
        files = sorted(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
        scope = "complete_explicit_source_tree"
    if not files or not runtime_files.issubset(set(files)):
        raise ValueError("implementation source inventory omits a required runtime module")
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "root": str(root),
        "scope": scope,
        "files": records,
        "inventory_sha256": hashlib.sha256(
            _canonical_json(records).encode("utf-8")
        ).hexdigest(),
    }


def extract_static_target_artifacts(
    model_path: str | Path,
    output_dir: str | Path,
    *,
    expected_revision: str,
    expected_config_sha256: str,
    expected_index_sha256: str,
    relative_rank_threshold: float = 1e-6,
    audit_rows: int = 8,
    audit_seed: int = 90817,
    device: str | torch.device = "cpu",
    implementation_source_root: str | Path | None = None,
    contract: StaticTargetContract = HARP_RTT_TARGET_CONTRACT,
) -> dict[str, Any]:
    """Build a pinned, non-overwriting HARP-RTT static artifact directory."""

    source = Path(model_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"model_path is not a directory: {source}")
    if destination.exists():
        raise FileExistsError(f"immutable output directory already exists: {destination}")
    if audit_rows < 1:
        raise ValueError("audit_rows must be positive")
    revision = _require_revision(expected_revision)
    config_hash = _require_sha256(expected_config_sha256, name="expected_config_sha256")
    index_hash = _require_sha256(expected_index_sha256, name="expected_index_sha256")
    contract.validate()
    implementation = _implementation_provenance(implementation_source_root)
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise ValueError("model directory must contain config.json and model.safetensors.index.json")
    actual_config_hash = sha256_file(config_path)
    actual_index_hash = sha256_file(index_path)
    if actual_config_hash != config_hash:
        raise ValueError(
            f"config SHA-256 mismatch: expected {config_hash}, found {actual_config_hash}"
        )
    if actual_index_hash != index_hash:
        raise ValueError(
            f"index SHA-256 mismatch: expected {index_hash}, found {actual_index_hash}"
        )
    config = _load_json(config_path, label="model config")
    text_config = _validate_config(config, contract)
    index = _load_json(index_path, label="model safetensors index")
    reader = _IndexedSafetensors(source, index)
    shard_records = _revision_provenance(
        source, reader.shards, expected_revision=revision
    )
    keys = _resolve_target_keys(reader.keys, contract)
    required_names = (
        *keys.router_weights,
        *keys.router_biases,
        keys.token_embedding,
        keys.final_rmsnorm_weight,
    )
    headers = reader.inspect(required_names)
    _validate_headers(headers, keys, contract)

    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {compute_device}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=destination.parent)
    )
    try:
        _, save_file = _safetensors_api()

        frozen = reader.load((keys.token_embedding, keys.final_rmsnorm_weight))
        token_embedding = frozen[keys.token_embedding]
        final_norm = frozen[keys.final_rmsnorm_weight]
        embedding_hash = tensor_content_sha256(token_embedding)
        norm_hash = tensor_content_sha256(final_norm)
        epsilon_tensor = torch.tensor(
            float(contract.rms_norm_epsilon), dtype=torch.float64
        )
        frozen_path = temporary / FROZEN_TARGET_FILENAME
        save_file(
            {
                "token_embedding": token_embedding,
                "final_rmsnorm_weight": final_norm,
                "final_rmsnorm_epsilon": epsilon_tensor,
            },
            str(frozen_path),
            metadata=_artifact_metadata(
                schema=FROZEN_TARGET_SCHEMA,
                revision=revision,
                config_sha256=config_hash,
                index_sha256=index_hash,
                values={
                    "token_embedding_source_key": keys.token_embedding,
                    "token_embedding_content_sha256": embedding_hash,
                    "final_rmsnorm_source_key": keys.final_rmsnorm_weight,
                    "final_rmsnorm_weight_content_sha256": norm_hash,
                    "final_rmsnorm_epsilon": repr(float(contract.rms_norm_epsilon)),
                    "implementation_inventory_sha256": implementation[
                        "inventory_sha256"
                    ],
                },
            ),
        )
        del frozen, token_embedding, final_norm

        router_values = reader.load((*keys.router_weights, *keys.router_biases))
        router_weights = torch.stack(
            [router_values[name] for name in keys.router_weights]
        ).to(device=compute_device)
        if keys.has_router_bias:
            router_bias = torch.stack(
                [router_values[name] for name in keys.router_biases]
            ).to(device=compute_device)
            bias_storage = "checkpoint_tensor"
        else:
            router_bias = torch.zeros(
                (contract.layers, contract.experts),
                dtype=router_weights.dtype,
                device=compute_device,
            )
            bias_storage = "implicit_zero"
        router_weight_hash = _tensor_collection_sha256(
            [(name, router_values[name]) for name in keys.router_weights]
        )
        router_bias_hash = _tensor_collection_sha256(
            (
                [(name, router_values[name]) for name in keys.router_biases]
                if keys.has_router_bias
                else [("implicit_zero_router_bias", router_bias.cpu())]
            )
        )
        del router_values

        geometry = build_centered_router_geometry(
            router_weights,
            router_bias,
            relative_rank_threshold=relative_rank_threshold,
        )
        generator = torch.Generator(device="cpu").manual_seed(int(audit_seed))
        audit_inputs = torch.randn(
            (audit_rows, contract.layers, contract.hidden_width),
            generator=generator,
            dtype=torch.float32,
        ).to(compute_device)
        audit = assert_router_geometry_equivalent(
            geometry,
            router_weights,
            router_bias,
            router_inputs=audit_inputs,
            k=contract.experts_per_token,
        )
        geometry = geometry.to("cpu")
        geometry_path = temporary / GEOMETRY_FILENAME
        save_file(
            geometry.to_tensor_dict(),
            str(geometry_path),
            metadata=_artifact_metadata(
                schema=ROUTER_GEOMETRY_SCHEMA,
                revision=revision,
                config_sha256=config_hash,
                index_sha256=index_hash,
                values={
                    "static_artifact_schema": STATIC_ARTIFACT_SCHEMA,
                    "router_weight_content_sha256": router_weight_hash,
                    "router_bias_content_sha256": router_bias_hash,
                    "router_bias_storage": bias_storage,
                    "orientation": "[layer,expert,hidden]",
                    "implementation_inventory_sha256": implementation[
                        "inventory_sha256"
                    ],
                },
            ),
        )
        del router_weights, router_bias, audit_inputs

        source_config_path = temporary / SOURCE_CONFIG_FILENAME
        source_config_path.write_bytes(config_path.read_bytes())
        files = {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": sha256_file(temporary / name),
            }
            for name in (
                GEOMETRY_FILENAME,
                FROZEN_TARGET_FILENAME,
                SOURCE_CONFIG_FILENAME,
            )
        }
        manifest: dict[str, Any] = {
            "schema": STATIC_ARTIFACT_SCHEMA,
            "immutable": True,
            "model": {
                "source_path": str(source),
                "repository_revision": revision,
                "config_sha256": config_hash,
                "index_sha256": index_hash,
                "architectures": list(contract.architecture_names),
                "outer_model_type": contract.outer_model_type,
                "text_model_type": contract.text_model_type,
            },
            "contract": asdict(contract),
            "resolved_tensors": {
                "router_weights": list(keys.router_weights),
                "router_biases": list(keys.router_biases),
                "router_bias_storage": bias_storage,
                "token_embedding": keys.token_embedding,
                "final_rmsnorm_weight": keys.final_rmsnorm_weight,
                "final_rmsnorm_epsilon_config_key": "text_config.rms_norm_eps",
            },
            "geometry": {
                "schema": ROUTER_GEOMETRY_SCHEMA,
                "path": GEOMETRY_FILENAME,
                "relative_rank_threshold": float(relative_rank_threshold),
                "shape": {
                    "layers": geometry.layers,
                    "experts": geometry.experts,
                    "hidden_width": geometry.hidden_width,
                    "maximum_rank": geometry.maximum_rank,
                },
                "ranks": [int(value) for value in geometry.ranks.tolist()],
                "router_weight_content_sha256": router_weight_hash,
                "router_bias_content_sha256": router_bias_hash,
                "audit": audit.to_dict(),
                "audit_source": "deterministic_synthetic_gaussian_checkpoint_independent",
                "audit_rows": int(audit_rows),
                "audit_seed": int(audit_seed),
            },
            "frozen_target": {
                "schema": FROZEN_TARGET_SCHEMA,
                "path": FROZEN_TARGET_FILENAME,
                "token_embedding_shape": [
                    contract.vocabulary_size,
                    contract.hidden_width,
                ],
                "token_embedding_dtype": contract.checkpoint_dtype,
                "token_embedding_content_sha256": embedding_hash,
                "final_rmsnorm_shape": [contract.hidden_width],
                "final_rmsnorm_dtype": contract.checkpoint_dtype,
                "final_rmsnorm_weight_content_sha256": norm_hash,
                "final_rmsnorm_epsilon": float(text_config["rms_norm_eps"]),
            },
            "checkpoint_shards": shard_records,
            "checkpoint_shard_inventory_sha256": hashlib.sha256(
                _canonical_json(shard_records).encode("utf-8")
            ).hexdigest(),
            "data_access": {
                "source": "static_checkpoint_only",
                "corpus_accessed": False,
                "test_split_accessed": False,
            },
            "implementation_source": implementation,
            "files": files,
        }
        (temporary / MANIFEST_FILENAME).write_text(
            _canonical_json(manifest) + "\n", encoding="utf-8"
        )
        if destination.exists():
            raise FileExistsError(
                f"immutable output directory appeared during build: {destination}"
            )
        temporary.rename(destination)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_static_target_artifacts(
    artifact_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    load_embedding: bool = True,
    verify_hashes: bool = True,
) -> StaticTargetArtifacts:
    """Load and validate geometry plus frozen target tensors."""

    root = Path(artifact_dir).expanduser().resolve()
    manifest = _load_json(root / MANIFEST_FILENAME, label="static artifact manifest")
    if manifest.get("schema") != STATIC_ARTIFACT_SCHEMA or manifest.get("immutable") is not True:
        raise ValueError("directory is not an immutable HARP-RTT static artifact")
    expected_names = {
        MANIFEST_FILENAME,
        GEOMETRY_FILENAME,
        FROZEN_TARGET_FILENAME,
        SOURCE_CONFIG_FILENAME,
    }
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ValueError(
            f"static artifact file inventory mismatch: expected {sorted(expected_names)}, "
            f"found {sorted(actual_names)}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("static artifact manifest has no file inventory")
    for name in expected_names - {MANIFEST_FILENAME}:
        record = files.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"manifest is missing file record for {name}")
        path = root / name
        if path.stat().st_size != record.get("bytes"):
            raise ValueError(f"artifact size mismatch for {name}")
        if verify_hashes and sha256_file(path) != record.get("sha256"):
            raise ValueError(f"artifact SHA-256 mismatch for {name}")

    safe_open, _ = _safetensors_api()
    geometry_path = root / GEOMETRY_FILENAME
    with safe_open(str(geometry_path), framework="pt", device="cpu") as handle:
        geometry_metadata = dict(handle.metadata() or {})
        geometry_tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    if geometry_metadata.get("schema") != ROUTER_GEOMETRY_SCHEMA:
        raise ValueError("router geometry safetensors schema mismatch")
    geometry = CenteredRouterGeometry.from_tensor_dict(geometry_tensors).to(device)

    frozen_record = manifest.get("frozen_target")
    if not isinstance(frozen_record, dict):
        raise ValueError("manifest has no frozen_target record")
    frozen_path = root / FROZEN_TARGET_FILENAME
    with safe_open(str(frozen_path), framework="pt", device="cpu") as handle:
        if dict(handle.metadata() or {}).get("schema") != FROZEN_TARGET_SCHEMA:
            raise ValueError("frozen target safetensors schema mismatch")
        expected_keys = {
            "token_embedding",
            "final_rmsnorm_weight",
            "final_rmsnorm_epsilon",
        }
        if set(handle.keys()) != expected_keys:
            raise ValueError("frozen target tensor key inventory mismatch")
        embedding_shape = tuple(handle.get_slice("token_embedding").get_shape())
        if embedding_shape != tuple(frozen_record.get("token_embedding_shape", ())):
            raise ValueError("frozen token embedding shape disagrees with manifest")
        token_embedding = handle.get_tensor("token_embedding") if load_embedding else None
        norm = handle.get_tensor("final_rmsnorm_weight")
        epsilon_tensor = handle.get_tensor("final_rmsnorm_epsilon")
    if epsilon_tensor.numel() != 1:
        raise ValueError("frozen final_rmsnorm_epsilon must be scalar")
    epsilon = float(epsilon_tensor.item())
    if epsilon != float(frozen_record.get("final_rmsnorm_epsilon", float("nan"))):
        raise ValueError("frozen RMSNorm epsilon disagrees with manifest")
    if tuple(norm.shape) != tuple(frozen_record.get("final_rmsnorm_shape", ())):
        raise ValueError("frozen RMSNorm weight shape disagrees with manifest")
    target_device = torch.device(device)
    return StaticTargetArtifacts(
        geometry=geometry,
        token_embedding=(
            token_embedding.to(device=target_device) if token_embedding is not None else None
        ),
        final_rmsnorm_weight=norm.to(device=target_device),
        final_rmsnorm_epsilon=epsilon,
        manifest=manifest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--relative-rank-threshold", type=float, default=1e-6)
    parser.add_argument("--audit-rows", type=int, default=8)
    parser.add_argument("--audit-seed", type=int, default=90817)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--implementation-source-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = extract_static_target_artifacts(
        args.model,
        args.output_dir,
        expected_revision=args.expected_revision,
        expected_config_sha256=args.expected_config_sha256,
        expected_index_sha256=args.expected_index_sha256,
        relative_rank_threshold=args.relative_rank_threshold,
        audit_rows=args.audit_rows,
        audit_seed=args.audit_seed,
        device=args.device,
        implementation_source_root=args.implementation_source_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "FROZEN_TARGET_FILENAME",
    "FROZEN_TARGET_SCHEMA",
    "GEOMETRY_FILENAME",
    "HARP_RTT_TARGET_CONTRACT",
    "MANIFEST_FILENAME",
    "SOURCE_CONFIG_FILENAME",
    "STATIC_ARTIFACT_SCHEMA",
    "StaticTargetArtifacts",
    "StaticTargetContract",
    "extract_static_target_artifacts",
    "frozen_token_embeddings",
    "load_static_target_artifacts",
    "reconstruct_final_hidden_rmsnorm",
    "sha256_file",
    "tensor_content_sha256",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
