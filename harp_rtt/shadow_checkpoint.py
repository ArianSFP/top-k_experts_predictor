"""Fail-closed target contract and selective checkpoint access for ShadowRoute."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


SHADOW_CHECKPOINT_SCHEMA = "harp_shadowroute_selective_checkpoint_v1"
TARGET_PREFIX = "model.language_model."
ROUTED_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)$"
)
EXPECTED_LAYER_TYPES = tuple(
    "full_attention" if layer % 4 == 3 else "linear_attention"
    for layer in range(40)
)


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _value(config: Any, *names: str) -> Any:
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    raise ValueError(f"target config lacks required field aliases {names}")


@dataclass(frozen=True)
class ShadowTargetContract:
    hidden_size: int = 2048
    layers: int = 40
    experts: int = 256
    exact_k: int = 8
    expert_intermediate: int = 512
    shared_intermediate: int = 512
    attention_heads: int = 16
    kv_heads: int = 2
    head_dim: int = 256
    linear_key_heads: int = 16
    linear_value_heads: int = 32
    linear_key_dim: int = 128
    linear_value_dim: int = 128
    linear_conv_kernel: int = 4
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_interleaved: bool = True
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    rms_norm_epsilon: float = 1e-6

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, int) and value < 1:
                raise ValueError(f"{name} must be positive")
            if isinstance(value, float) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.exact_k > self.experts:
            raise ValueError("top-k exceeds expert count")


def validate_shadow_target_config(
    config: Any,
    contract: ShadowTargetContract = ShadowTargetContract(),
) -> dict[str, Any]:
    """Validate the exact hybrid Qwen contract used by the audited captures."""

    contract.validate()
    expected = {
        ("hidden_size",): contract.hidden_size,
        ("num_hidden_layers",): contract.layers,
        ("num_experts",): contract.experts,
        ("num_experts_per_tok",): contract.exact_k,
        ("moe_intermediate_size",): contract.expert_intermediate,
        ("shared_expert_intermediate_size",): contract.shared_intermediate,
        ("num_attention_heads",): contract.attention_heads,
        ("num_key_value_heads",): contract.kv_heads,
        ("head_dim",): contract.head_dim,
        ("linear_num_key_heads", "linear_key_heads"): contract.linear_key_heads,
        ("linear_num_value_heads", "linear_value_heads"): contract.linear_value_heads,
        ("linear_key_head_dim", "linear_key_dim"): contract.linear_key_dim,
        ("linear_value_head_dim", "linear_value_dim"): contract.linear_value_dim,
        ("linear_conv_kernel_dim", "linear_conv_kernel", "conv_kernel_size"):
            contract.linear_conv_kernel,
    }
    observed: dict[str, Any] = {}
    for aliases, wanted in expected.items():
        found = _value(config, *aliases)
        if int(found) != int(wanted):
            raise ValueError(f"target config {aliases[0]}={found!r}, expected {wanted!r}")
        observed[aliases[0]] = int(found)
    rope_parameters = None
    try:
        candidate = _value(config, "rope_parameters")
        if isinstance(candidate, Mapping):
            rope_parameters = candidate
    except ValueError:
        pass

    def normalized_value(*aliases: str) -> Any:
        try:
            return _value(config, *aliases)
        except ValueError:
            if rope_parameters is not None:
                for alias in aliases:
                    if alias in rope_parameters:
                        return rope_parameters[alias]
            raise

    float_expected = {
        ("rope_theta",): contract.rope_theta,
        ("partial_rotary_factor",): contract.partial_rotary_factor,
        ("rms_norm_eps", "rms_norm_epsilon"): contract.rms_norm_epsilon,
    }
    for aliases, wanted in float_expected.items():
        found = float(normalized_value(*aliases))
        if not math.isclose(found, float(wanted), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"target config {aliases[0]}={found!r}, expected {wanted!r}")
        observed[aliases[0]] = found
    if rope_parameters is not None:
        if bool(rope_parameters.get("mrope_interleaved")) is not contract.mrope_interleaved:
            raise ValueError("target mRoPE interleaving differs from the frozen contract")
        section = tuple(int(value) for value in rope_parameters.get("mrope_section", ()))
        if section != contract.mrope_section:
            raise ValueError("target mRoPE section differs from the frozen contract")
        if str(rope_parameters.get("rope_type")) != "default":
            raise ValueError("target RoPE type differs from the frozen contract")
        observed["mrope_interleaved"] = contract.mrope_interleaved
        observed["mrope_section"] = contract.mrope_section
    layer_types = tuple(str(value) for value in _value(config, "layer_types"))
    if layer_types != EXPECTED_LAYER_TYPES:
        raise ValueError("target layer_types do not match the frozen 30-linear/10-full pattern")
    if bool(_value(config, "attention_bias")):
        raise ValueError("target attention unexpectedly uses bias")
    if float(_value(config, "attention_dropout")) != 0.0:
        raise ValueError("target attention dropout must be zero")
    observed["layer_types"] = layer_types
    return observed


class IndexedCheckpoint:
    """Read named tensors from a validated sharded safetensors checkpoint."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        index_path = self.model_path / "model.safetensors.index.json"
        value = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = value.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint index has no weight map")
        self.weight_map: dict[str, str] = {}
        for name, shard in weight_map.items():
            if not isinstance(name, str) or not isinstance(shard, str):
                raise ValueError("checkpoint weight map is malformed")
            path = Path(shard)
            if path.is_absolute() or len(path.parts) != 1:
                raise ValueError("checkpoint index contains an unsafe shard path")
            if not (self.model_path / path).is_file():
                raise ValueError(f"checkpoint shard is missing: {shard}")
            self.weight_map[name] = shard
        self.index_sha256 = sha256_file(index_path)

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self.weight_map)

    @property
    def routed_expert_keys(self) -> tuple[str, ...]:
        return tuple(sorted(name for name in self.weight_map if ROUTED_EXPERT_RE.match(name)))

    def tensors(self, names: Sequence[str]) -> dict[str, Tensor]:
        missing = sorted(set(names) - self.keys)
        if missing:
            raise ValueError(f"checkpoint lacks required tensors: {missing[:4]}")
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required for target loading") from exc
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for name in names:
            grouped[self.weight_map[name]].append(name)
        result: dict[str, Tensor] = {}
        for shard, shard_names in grouped.items():
            with safe_open(self.model_path / shard, framework="pt", device="cpu") as handle:
                for name in shard_names:
                    result[name] = handle.get_tensor(name)
        return result

    def tensor(self, name: str) -> Tensor:
        return self.tensors((name,))[name]


def validate_routed_expert_inventory(
    checkpoint: IndexedCheckpoint,
    *,
    layers: int = 40,
) -> tuple[str, ...]:
    keys = checkpoint.routed_expert_keys
    expected = {
        f"{TARGET_PREFIX}layers.{layer}.mlp.experts.{role}"
        for layer in range(layers)
        for role in ("gate_up_proj", "down_proj")
    }
    if set(keys) != expected:
        missing = sorted(expected - set(keys))
        extra = sorted(set(keys) - expected)
        raise ValueError(
            f"routed expert inventory mismatch; missing={missing[:3]} extra={extra[:3]}"
        )
    return keys


def load_target_layer_experts(
    checkpoint: IndexedCheckpoint,
    layer: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor]:
    """Load exactly one native routed-expert layer for local distillation."""

    if not 0 <= int(layer) < 40:
        raise ValueError("target layer must lie in [0,40)")
    validate_routed_expert_inventory(checkpoint)
    prefix = f"{TARGET_PREFIX}layers.{int(layer)}.mlp.experts."
    values = checkpoint.tensors(
        (prefix + "gate_up_proj", prefix + "down_proj")
    )
    gate_up = values[prefix + "gate_up_proj"]
    down = values[prefix + "down_proj"]
    if gate_up.shape != (256, 1024, 2048) or down.shape != (256, 2048, 512):
        raise ValueError("native target expert tensor geometry is not Qwen-compatible")
    target_device = torch.device(device)
    return (
        gate_up.to(device=target_device, dtype=dtype),
        down.to(device=target_device, dtype=dtype),
    )


def _resolve_parent(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    current: Any = module
    for part in parts[:-1]:
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"parameter parent for {name!r} is not a module")
    return current, parts[-1]


def assign_parameter(module: nn.Module, name: str, value: Tensor, *, device: torch.device) -> None:
    parent, leaf = _resolve_parent(module, name)
    existing = getattr(parent, leaf)
    if not isinstance(existing, nn.Parameter):
        raise TypeError(f"{name!r} is not a parameter")
    if tuple(existing.shape) != tuple(value.shape):
        raise ValueError(f"checkpoint shape mismatch for {name}")
    parameter = nn.Parameter(
        value.to(device=device, dtype=existing.dtype),
        requires_grad=False,
    )
    setattr(parent, leaf, parameter)


@dataclass(frozen=True)
class SelectiveLoadReport:
    schema: str
    checkpoint_index_sha256: str
    loaded_tensor_names: tuple[str, ...]
    omitted_routed_tensor_names: tuple[str, ...]
    shadow_parameter_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_exact_nonexpert_parameters(
    text_model: nn.Module,
    checkpoint: IndexedCheckpoint,
    *,
    device: str | torch.device,
    checkpoint_prefix: str = TARGET_PREFIX,
) -> SelectiveLoadReport:
    """Materialize exact text-model parameters while omitting routed experts."""

    target_device = torch.device(device)
    validate_routed_expert_inventory(checkpoint)
    loaded: list[str] = []
    shadow: list[str] = []
    for name, parameter in tuple(text_model.named_parameters()):
        source = checkpoint_prefix + name
        if ".mlp.experts." in name:
            if parameter.device.type == "meta":
                raise ValueError(
                    "shadow experts must be materialized before selective target loading"
                )
            shadow.append(name)
            continue
        value = checkpoint.tensor(source)
        assign_parameter(text_model, name, value, device=target_device)
        loaded.append(source)
    unresolved = [
        name for name, value in text_model.named_buffers()
        if value.device.type == "meta"
    ]
    if unresolved:
        raise ValueError(
            "official runtime buffers remain on meta after construction: "
            + ", ".join(unresolved[:8])
        )
    text_model.eval()
    for name, parameter in text_model.named_parameters():
        parameter.requires_grad_(".mlp.experts." in name)
    return SelectiveLoadReport(
        schema=SHADOW_CHECKPOINT_SCHEMA,
        checkpoint_index_sha256=checkpoint.index_sha256,
        loaded_tensor_names=tuple(loaded),
        omitted_routed_tensor_names=checkpoint.routed_expert_keys,
        shadow_parameter_names=tuple(shadow),
    )


__all__ = [
    "EXPECTED_LAYER_TYPES",
    "IndexedCheckpoint",
    "SelectiveLoadReport",
    "ShadowTargetContract",
    "TARGET_PREFIX",
    "assign_parameter",
    "load_exact_nonexpert_parameters",
    "load_target_layer_experts",
    "sha256_file",
    "validate_routed_expert_inventory",
    "validate_shadow_target_config",
]
