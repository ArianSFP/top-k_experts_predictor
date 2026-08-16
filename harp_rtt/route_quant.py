"""Identity-preserving low-bit routed experts for HARP-ShadowRoute.

RouteQuant compresses the *weights* of each target expert independently.  It
does not replace one expert with another expert or with a generic fallback.
The frozen target router therefore remains authoritative: native expert IDs
and execution weights enter unchanged and are never renormalised.

This module is the deliberately simple scientific reference implementation.
It dequantizes the selected matrices before executing the native SwiGLU.  A
packed CUDA kernel may replace that reference only after arithmetic and route
parity have been established.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SUPPORTED_BITS = (0, 1, 2, 3, 4)


def _validate_bits(bits: int) -> int:
    value = int(bits)
    if value not in SUPPORTED_BITS:
        raise ValueError(f"RouteQuant bits must lie in {SUPPORTED_BITS}")
    return value


def pack_unsigned_codes(codes: Tensor, *, bits: int) -> Tensor:
    """Pack unsigned integer codes along the final axis.

    The representation is little-endian within each byte.  It supports the
    non-byte-aligned three-bit case without padding between rows.
    """

    bits = _validate_bits(bits)
    if bits == 0:
        if codes.shape[-1] != 0:
            raise ValueError("zero-bit packing accepts only an empty value axis")
        return codes.to(torch.uint8)
    if codes.dtype not in {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    }:
        raise TypeError("packed codes must use an integer dtype")
    if codes.ndim < 1:
        raise ValueError("packed codes require a final value axis")
    maximum = (1 << bits) - 1
    if codes.numel() and bool(((codes < 0) | (codes > maximum)).any()):
        raise ValueError("code lies outside the declared bit width")
    width = int(codes.shape[-1])
    rows = int(codes.numel() // max(width, 1))
    byte_width = (width * bits + 7) // 8
    flat = codes.reshape(rows, width).to(torch.int32)
    starts = torch.arange(width, device=codes.device, dtype=torch.int64) * bits
    byte_index = (starts // 8).expand(rows, -1)
    offsets = (starts % 8).to(torch.int32)
    packed = torch.zeros(rows, byte_width, dtype=torch.int32, device=codes.device)
    packed.scatter_add_(1, byte_index, flat << offsets)
    crossing = offsets + bits > 8
    if bool(crossing.any()):
        high_index = byte_index[:, crossing] + 1
        high_values = flat[:, crossing] >> (8 - offsets[crossing])
        packed.scatter_add_(1, high_index, high_values)
    return packed.to(torch.uint8).reshape(*codes.shape[:-1], byte_width).contiguous()


def unpack_unsigned_codes(packed: Tensor, *, bits: int, values: int) -> Tensor:
    """Inverse of :func:`pack_unsigned_codes`."""

    bits = _validate_bits(bits)
    if values < 0:
        raise ValueError("unpacked value count must be non-negative")
    if packed.dtype != torch.uint8 or packed.ndim < 1:
        raise ValueError("packed tensor must be rank-one-or-greater uint8")
    expected = (values * bits + 7) // 8 if bits else 0
    if packed.shape[-1] != expected:
        raise ValueError("packed byte width disagrees with bits/value count")
    if bits == 0:
        return torch.empty(*packed.shape[:-1], values, dtype=torch.uint8, device=packed.device)
    rows = int(packed.numel() // max(expected, 1))
    flat = packed.reshape(rows, expected).to(torch.int32)
    starts = torch.arange(values, device=packed.device, dtype=torch.int64) * bits
    byte_index = (starts // 8).expand(rows, -1)
    offsets = (starts % 8).to(torch.int32)
    words = flat.gather(1, byte_index)
    crossing = offsets + bits > 8
    if bool(crossing.any()):
        high = flat.gather(1, byte_index[:, crossing] + 1)
        words[:, crossing] |= high << 8
    mask = (1 << bits) - 1
    values_tensor = (words >> offsets) & mask
    return values_tensor.to(torch.uint8).reshape(*packed.shape[:-1], values)


def quantize_groupwise_nbit(
    weight: Tensor,
    *,
    bits: int,
    group_size: int = 64,
    scale_method: str = "mse",
    refinement_steps: int = 4,
) -> tuple[Tensor, Tensor]:
    """Symmetric groupwise quantization with packed 1--4 bit codes.

    ``mse`` alternates discrete codes and the least-squares scale.  ``amax``
    matches the established INT4 reference's max-absolute scale policy.
    One-bit weights use a binary sign code and their least-squares mean
    absolute scale.
    """

    bits = _validate_bits(bits)
    if bits == 0:
        raise ValueError("zero-bit cells are represented by omission, not quantization")
    if weight.ndim < 2 or not weight.is_floating_point():
        raise ValueError("RouteQuant weights must be floating point rank >=2")
    if group_size < 1 or weight.shape[-1] % group_size:
        raise ValueError("group size must divide the input width")
    if scale_method not in {"amax", "mse"}:
        raise ValueError("scale method must be 'amax' or 'mse'")
    if refinement_steps < 0:
        raise ValueError("refinement steps must be non-negative")
    grouped = weight.float().reshape(*weight.shape[:-1], -1, group_size)
    if bits == 1:
        signed = torch.where(grouped >= 0, 1.0, -1.0)
        if scale_method == "amax":
            scales = grouped.abs().amax(-1).clamp_min(1e-8)
        else:
            scales = grouped.abs().mean(-1).clamp_min(1e-8)
        codes = (signed > 0).to(torch.uint8)
    else:
        qmax = (1 << (bits - 1)) - 1
        scales = grouped.abs().amax(-1).div(float(qmax)).clamp_min(1e-8)
        steps = refinement_steps if scale_method == "mse" else 0
        quantized = torch.zeros_like(grouped)
        for _ in range(steps + 1):
            quantized = torch.round(grouped / scales[..., None]).clamp(-qmax, qmax)
            if _ < steps:
                numerator = (grouped * quantized).sum(-1)
                denominator = quantized.square().sum(-1).clamp_min(1e-12)
                scales = (numerator / denominator).clamp_min(1e-8)
        codes = (quantized.to(torch.int16) + qmax).to(torch.uint8)
    flat_codes = codes.reshape(*weight.shape[:-1], -1)
    return (
        pack_unsigned_codes(flat_codes, bits=bits),
        scales.to(torch.bfloat16).contiguous(),
    )


def dequantize_groupwise_nbit(
    packed: Tensor,
    scales: Tensor,
    *,
    bits: int,
    group_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Dequantize a packed groupwise tensor into its compute dtype."""

    bits = _validate_bits(bits)
    if bits == 0:
        raise ValueError("zero-bit cells have no packed tensor")
    if scales.ndim < 2 or packed.ndim != scales.ndim:
        raise ValueError("packed/scales ranks differ")
    groups = int(scales.shape[-1])
    values = groups * group_size
    codes = unpack_unsigned_codes(packed, bits=bits, values=values)
    grouped = codes.reshape(*codes.shape[:-1], groups, group_size)
    if bits == 1:
        quantized = grouped.to(dtype).mul(2).sub(1)
    else:
        qmax = (1 << (bits - 1)) - 1
        quantized = grouped.to(dtype).sub(qmax)
    return (
        quantized * scales.to(dtype)[..., None]
    ).reshape(*codes.shape[:-1], values)


@dataclass(frozen=True)
class RouteQuantConfig:
    hidden_width: int = 2048
    intermediate_width: int = 512
    experts: int = 256
    exact_k: int = 8
    group_size: int = 64

    def validate(self) -> None:
        for name in (
            "hidden_width", "intermediate_width", "experts", "exact_k", "group_size"
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.exact_k > self.experts:
            raise ValueError("exact_k exceeds the expert count")
        if self.hidden_width % self.group_size or self.intermediate_width % self.group_size:
            raise ValueError("group size must divide both expert input widths")


class PackedNBitMatrixBank(nn.Module):
    """A mixed-bit collection of expert matrices with deterministic buckets."""

    def __init__(
        self,
        *,
        bit_widths: Tensor,
        input_width: int,
        output_width: int,
        group_size: int,
        buckets: Mapping[int, tuple[Tensor, Tensor, Tensor]],
    ) -> None:
        super().__init__()
        widths = torch.as_tensor(bit_widths, dtype=torch.int8).cpu()
        if widths.ndim != 1 or widths.numel() < 1:
            raise ValueError("matrix-bank bit widths must be a non-empty vector")
        if any(int(value) not in SUPPORTED_BITS for value in widths.tolist()):
            raise ValueError("matrix-bank contains an unsupported bit width")
        if input_width < 1 or output_width < 1 or input_width % group_size:
            raise ValueError("matrix-bank geometry is invalid")
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.group_size = int(group_size)
        self.items = int(widths.numel())
        self.register_buffer("bit_widths", widths)
        item_to_bucket = torch.full((self.items,), -1, dtype=torch.int32)
        for bits in range(1, 5):
            ids, packed, scales = buckets.get(
                bits,
                (
                    torch.empty(0, dtype=torch.int16),
                    torch.empty(0, output_width, 0, dtype=torch.uint8),
                    torch.empty(0, output_width, input_width // group_size, dtype=torch.bfloat16),
                ),
            )
            ids = torch.as_tensor(ids, dtype=torch.int16).cpu()
            if ids.ndim != 1 or ids.unique().numel() != ids.numel():
                raise ValueError("matrix-bank bucket IDs must be a unique vector")
            expected_bytes = (input_width * bits + 7) // 8
            if packed.shape != (ids.numel(), output_width, expected_bytes):
                raise ValueError("matrix-bank packed bucket geometry changed")
            if scales.shape != (ids.numel(), output_width, input_width // group_size):
                raise ValueError("matrix-bank scale bucket geometry changed")
            if ids.numel():
                ids_long = ids.long()
                if bool(((ids_long < 0) | (ids_long >= self.items)).any()):
                    raise ValueError("matrix-bank bucket ID is out of range")
                if not torch.equal(widths[ids_long], torch.full_like(widths[ids_long], bits)):
                    raise ValueError("matrix-bank bucket disagrees with bit-width table")
                item_to_bucket[ids_long] = torch.arange(ids.numel(), dtype=torch.int32)
            self.register_buffer(f"ids_b{bits}", ids.contiguous())
            self.register_buffer(f"packed_b{bits}", packed.to(torch.uint8).cpu().contiguous())
            self.register_buffer(f"scales_b{bits}", scales.to(torch.bfloat16).cpu().contiguous())
        active = widths > 0
        if bool((item_to_bucket[active] < 0).any()) or bool((item_to_bucket[~active] >= 0).any()):
            raise ValueError("matrix-bank bucket inventory is incomplete")
        self.register_buffer("item_to_bucket", item_to_bucket)

    @classmethod
    def from_weights(
        cls,
        weight: Tensor,
        *,
        bit_widths: Tensor,
        group_size: int = 64,
        scale_method: str = "mse",
        item_chunk: int = 2,
    ) -> "PackedNBitMatrixBank":
        if weight.ndim != 3:
            raise ValueError("matrix-bank source weights must be [items,out,in]")
        if item_chunk < 1:
            raise ValueError("matrix-bank item chunk must be positive")
        widths = torch.as_tensor(bit_widths, dtype=torch.int8).cpu()
        if widths.shape != (weight.shape[0],):
            raise ValueError("matrix-bank bit schedule does not match source items")
        buckets: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        for bits in range(1, 5):
            ids = (widths == bits).nonzero(as_tuple=False).flatten()
            byte_width = (weight.shape[-1] * bits + 7) // 8
            packed = torch.empty(
                ids.numel(), weight.shape[1], byte_width, dtype=torch.uint8
            )
            scales = torch.empty(
                ids.numel(), weight.shape[1], weight.shape[-1] // group_size,
                dtype=torch.bfloat16,
            )
            for start in range(0, ids.numel(), item_chunk):
                stop = min(start + item_chunk, ids.numel())
                selected = ids[start:stop].to(weight.device)
                current_packed, current_scales = quantize_groupwise_nbit(
                    weight.index_select(0, selected),
                    bits=bits,
                    group_size=group_size,
                    scale_method=scale_method,
                )
                packed[start:stop].copy_(current_packed.cpu())
                scales[start:stop].copy_(current_scales.cpu())
            buckets[bits] = (ids.to(torch.int16), packed, scales)
        return cls(
            bit_widths=widths,
            input_width=int(weight.shape[-1]),
            output_width=int(weight.shape[1]),
            group_size=group_size,
            buckets=buckets,
        )

    def dequantize(self, item: int, *, dtype: torch.dtype) -> Tensor | None:
        if not 0 <= int(item) < self.items:
            raise ValueError("matrix-bank item is out of range")
        bits = int(self.bit_widths[int(item)])
        if bits == 0:
            return None
        bucket = int(self.item_to_bucket[int(item)])
        if bucket < 0:
            raise RuntimeError("matrix-bank active item has no packed payload")
        packed = getattr(self, f"packed_b{bits}")[bucket]
        scales = getattr(self, f"scales_b{bits}")[bucket]
        return dequantize_groupwise_nbit(
            packed, scales, bits=bits, group_size=self.group_size, dtype=dtype
        )

    def persistent_nbytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in self.state_dict().values()
        )


class PackedRouteQuantExperts(nn.Module):
    """Full-width mixed-bit Qwen experts using the native routed call contract."""

    def __init__(
        self,
        config: RouteQuantConfig,
        gate_up: PackedNBitMatrixBank,
        down: PackedNBitMatrixBank,
    ) -> None:
        super().__init__()
        config.validate()
        if gate_up.items != config.experts or down.items != config.experts:
            raise ValueError("RouteQuant bank expert counts differ")
        if (
            gate_up.input_width != config.hidden_width
            or gate_up.output_width != 2 * config.intermediate_width
            or down.input_width != config.intermediate_width
            or down.output_width != config.hidden_width
        ):
            raise ValueError("RouteQuant bank geometry differs from Qwen experts")
        gate_missing = gate_up.bit_widths == 0
        down_missing = down.bit_widths == 0
        if not torch.equal(gate_missing, down_missing):
            raise ValueError("RouteQuant omission must remove the complete expert cell")
        self.config = config
        self.gate_up = gate_up
        self.down = down
        self.cache_dequantized = False
        self.cache_max_experts = 0
        self._gate_up_cache: dict[int, Tensor] = {}
        self._down_cache: dict[int, Tensor] = {}

    def enable_dequantized_cache(
        self, enabled: bool = True, *, max_experts: int | None = None
    ) -> None:
        """Cache exact reference dequantization without changing arithmetic."""

        self.cache_dequantized = bool(enabled)
        if max_experts is None:
            max_experts = self.config.experts
        if self.cache_dequantized and not 1 <= int(max_experts) <= self.config.experts:
            raise ValueError("RouteQuant dequantization cache capacity is invalid")
        self.cache_max_experts = int(max_experts) if self.cache_dequantized else 0
        if not self.cache_dequantized:
            self._gate_up_cache.clear()
            self._down_cache.clear()

    def _expert_weights(
        self, expert: int, *, dtype: torch.dtype
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.cache_dequantized and expert in self._gate_up_cache:
            return self._gate_up_cache[expert], self._down_cache[expert]
        gate_up = self.gate_up.dequantize(expert, dtype=dtype)
        down = self.down.dequantize(expert, dtype=dtype)
        if self.cache_dequantized and gate_up is not None and down is not None:
            if len(self._gate_up_cache) >= self.cache_max_experts:
                oldest = next(iter(self._gate_up_cache))
                self._gate_up_cache.pop(oldest)
                self._down_cache.pop(oldest)
            self._gate_up_cache[expert] = gate_up
            self._down_cache[expert] = down
        return gate_up, down

    @classmethod
    def from_target(
        cls,
        gate_up: Tensor,
        down: Tensor,
        *,
        gate_up_bits: int | Tensor,
        down_bits: int | Tensor | None = None,
        exact_k: int = 8,
        group_size: int = 64,
        scale_method: str = "mse",
        item_chunk: int = 2,
    ) -> "PackedRouteQuantExperts":
        if gate_up.ndim != 3 or down.ndim != 3:
            raise ValueError("RouteQuant target weights must be rank three")
        experts, twice_intermediate, hidden = gate_up.shape
        if twice_intermediate % 2:
            raise ValueError("RouteQuant gate/up width must be even")
        intermediate = twice_intermediate // 2
        if down.shape != (experts, hidden, intermediate):
            raise ValueError("RouteQuant target expert shapes differ")
        gate_schedule = torch.full(
            (experts,), int(gate_up_bits), dtype=torch.int8
        ) if isinstance(gate_up_bits, int) else torch.as_tensor(gate_up_bits, dtype=torch.int8)
        if down_bits is None:
            down_bits = gate_up_bits
        down_schedule = torch.full(
            (experts,), int(down_bits), dtype=torch.int8
        ) if isinstance(down_bits, int) else torch.as_tensor(down_bits, dtype=torch.int8)
        config = RouteQuantConfig(
            hidden_width=hidden,
            intermediate_width=intermediate,
            experts=experts,
            exact_k=exact_k,
            group_size=group_size,
        )
        return cls(
            config,
            PackedNBitMatrixBank.from_weights(
                gate_up,
                bit_widths=gate_schedule,
                group_size=group_size,
                scale_method=scale_method,
                item_chunk=item_chunk,
            ),
            PackedNBitMatrixBank.from_weights(
                down,
                bit_widths=down_schedule,
                group_size=group_size,
                scale_method=scale_method,
                item_chunk=item_chunk,
            ),
        )

    def selected_unweighted(self, hidden_states: Tensor, selected_ids: Tensor) -> Tensor:
        cfg = self.config
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != cfg.hidden_width:
            raise ValueError("RouteQuant hidden width changed")
        leading = tuple(hidden_states.shape[:-1])
        if selected_ids.shape[:-1] != leading:
            raise ValueError("RouteQuant route IDs do not align with hidden states")
        hidden = hidden_states.reshape(-1, cfg.hidden_width)
        ids = selected_ids.reshape(-1, selected_ids.shape[-1]).long()
        if ids.numel() and bool(((ids < 0) | (ids >= cfg.experts)).any()):
            raise ValueError("RouteQuant route ID lies outside the expert namespace")
        output = torch.zeros(
            hidden.shape[0], ids.shape[1], cfg.hidden_width,
            device=hidden.device, dtype=hidden.dtype,
        )
        for expert in torch.unique(ids).tolist():
            gate_up, down = self._expert_weights(int(expert), dtype=hidden.dtype)
            if gate_up is None:
                if down is not None:  # pragma: no cover - constructor invariant
                    raise RuntimeError("partially omitted RouteQuant expert")
                continue
            assert down is not None
            positions = (ids == int(expert)).nonzero(as_tuple=False)
            token_index, slot_index = positions[:, 0], positions[:, 1]
            projected = F.linear(hidden[token_index], gate_up)
            gate, up = projected.chunk(2, dim=-1)
            output[token_index, slot_index] = F.linear(F.silu(gate) * up, down)
        return output.reshape(*leading, ids.shape[-1], cfg.hidden_width)

    def forward(
        self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor
    ) -> Tensor:
        cfg = self.config
        if top_k_index.shape != top_k_weights.shape:
            raise ValueError("RouteQuant IDs/weights do not align")
        if top_k_index.shape[:-1] != hidden_states.shape[:-1]:
            raise ValueError("RouteQuant route leading axes changed")
        if top_k_index.shape[-1] != cfg.exact_k:
            raise ValueError("RouteQuant requires the complete native top-k route")
        if not torch.isfinite(top_k_weights).all() or bool((top_k_weights < 0).any()):
            raise ValueError("RouteQuant weights must be finite and non-negative")
        values = self.selected_unweighted(hidden_states, top_k_index)
        return (values * top_k_weights.to(values)[..., None]).sum(-2)

    def persistent_nbytes(self) -> int:
        return self.gate_up.persistent_nbytes() + self.down.persistent_nbytes()


def uniform_routequant_projected_bytes(
    *,
    bits: int,
    layers: int = 40,
    experts: int = 256,
    hidden_width: int = 2048,
    intermediate_width: int = 512,
    group_size: int = 64,
) -> int:
    """Exact logical tensor bytes for a uniform full-pool reference bundle."""

    bits = _validate_bits(bits)
    if bits == 0:
        return layers * 2 * experts * (1 + 4)  # bit table + bucket map
    gate_values = experts * 2 * intermediate_width * hidden_width
    down_values = experts * hidden_width * intermediate_width
    packed = math.ceil(gate_values * bits / 8) + math.ceil(down_values * bits / 8)
    groups = (
        experts * 2 * intermediate_width * (hidden_width // group_size)
        + experts * hidden_width * (intermediate_width // group_size)
    )
    scales = groups * 2
    metadata = 2 * (experts + experts * 4 + experts * 2)
    return layers * (packed + scales + metadata)


def mixed_routequant_projected_bytes(
    bit_widths: Tensor,
    *,
    hidden_width: int = 2048,
    intermediate_width: int = 512,
    group_size: int = 64,
) -> int:
    """Exact logical tensor bytes for a per-layer/per-expert bit schedule.

    The bit-width tensor is [layers, experts]. A schedule value applies to the
    complete gate/up/down expert cell, so expert identity and the native
    SwiGLU remain intact at every non-zero precision.
    """

    widths = torch.as_tensor(bit_widths, dtype=torch.int8).cpu()
    if widths.ndim != 2 or widths.shape[0] < 1 or widths.shape[1] < 1:
        raise ValueError("mixed RouteQuant schedule must be [layers,experts]")
    if any(int(value) not in SUPPORTED_BITS for value in widths.flatten().tolist()):
        raise ValueError("mixed RouteQuant schedule contains an unsupported bit width")
    if hidden_width % group_size or intermediate_width % group_size:
        raise ValueError("group size must divide both expert input widths")
    cell_weights = 3 * hidden_width * intermediate_width
    scale_values = (
        2 * intermediate_width * (hidden_width // group_size)
        + hidden_width * (intermediate_width // group_size)
    )
    payload = 0
    for layer in widths:
        active = layer > 0
        payload += sum(
            math.ceil(int((layer == bits).sum()) * cell_weights * bits / 8)
            for bits in range(1, 5)
        )
        payload += int(active.sum()) * scale_values * 2
        # Two matrix banks each persist one int8 bit table, one int32 bucket
        # map, and int16 expert IDs for every active bucket entry.
        payload += 2 * (
            layer.numel() * (1 + 4) + int(active.sum()) * 2
        )
    return int(payload)


__all__ = [
    "PackedNBitMatrixBank",
    "PackedRouteQuantExperts",
    "RouteQuantConfig",
    "SUPPORTED_BITS",
    "dequantize_groupwise_nbit",
    "pack_unsigned_codes",
    "quantize_groupwise_nbit",
    "mixed_routequant_projected_bytes",
    "uniform_routequant_projected_bytes",
    "unpack_unsigned_codes",
]
