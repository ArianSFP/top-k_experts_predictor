"""Centered target-router geometry for HARP-RTT.

For each target layer this module factorizes the centered affine router as

``center(W) = K @ V.T`` and ``center(b) = b_c``.

Variable numerical ranks are padded only at the storage boundary.  The rank
mask remains authoritative, so padded coordinates can never become an
accidental learned router direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor


ROUTER_GEOMETRY_SCHEMA = "harp_rtt_centered_router_geometry_v2"


def _as_float_tensor(value: Tensor | Any, *, name: str) -> Tensor:
    result = torch.as_tensor(value)
    if not result.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    result = result.float()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _center_experts(values: Tensor) -> Tensor:
    return values - values.mean(dim=-2, keepdim=True)


def _center_bias(values: Tensor) -> Tensor:
    return values - values.mean(dim=-1, keepdim=True)


def _stable_topk(scores: Tensor, k: int) -> Tensor:
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]


@dataclass(frozen=True)
class CenteredRouterGeometry:
    """Padded full numerical-rank factorization of a layered target router.

    Shapes are ``expert_keys[L,E,R]``, ``input_basis[L,D,R]``,
    ``rank_mask[L,R]``, ``singular_values[L,R]``, ``ranks[L]``,
    ``row_norms[L,E]``, and ``centered_bias[L,E]``.
    """

    expert_keys: Tensor
    input_basis: Tensor
    rank_mask: Tensor
    singular_values: Tensor
    ranks: Tensor
    row_norms: Tensor
    centered_bias: Tensor
    relative_rank_threshold: float

    @property
    def K(self) -> Tensor:
        """Architecture notation alias for :attr:`expert_keys`."""

        return self.expert_keys

    @property
    def V(self) -> Tensor:
        """Architecture notation alias for :attr:`input_basis`."""

        return self.input_basis

    @property
    def layers(self) -> int:
        return int(self.expert_keys.shape[0])

    @property
    def experts(self) -> int:
        return int(self.expert_keys.shape[1])

    @property
    def hidden_width(self) -> int:
        return int(self.input_basis.shape[1])

    @property
    def maximum_rank(self) -> int:
        return int(self.expert_keys.shape[2])

    @property
    def log_row_norms(self) -> Tensor:
        return self.row_norms.clamp_min(1e-12).log()

    def validate(self) -> None:
        if self.expert_keys.ndim != 3 or self.input_basis.ndim != 3:
            raise ValueError("expert_keys and input_basis must be rank-three tensors")
        l, e, r = self.expert_keys.shape
        expected = {
            "input_basis": (l, self.input_basis.shape[1], r),
            "rank_mask": (l, r),
            "singular_values": (l, r),
            "ranks": (l,),
            "row_norms": (l, e),
            "centered_bias": (l, e),
        }
        actual = {
            "input_basis": tuple(self.input_basis.shape),
            "rank_mask": tuple(self.rank_mask.shape),
            "singular_values": tuple(self.singular_values.shape),
            "ranks": tuple(self.ranks.shape),
            "row_norms": tuple(self.row_norms.shape),
            "centered_bias": tuple(self.centered_bias.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} has shape {actual[name]}, expected {shape}")
        if self.rank_mask.dtype != torch.bool:
            raise TypeError("rank_mask must be boolean")
        if self.ranks.dtype not in (torch.int32, torch.int64):
            raise TypeError("ranks must use an integer dtype")
        if not all(
            tensor.device == self.expert_keys.device
            for tensor in (
                self.input_basis,
                self.rank_mask,
                self.singular_values,
                self.ranks,
                self.row_norms,
                self.centered_bias,
            )
        ):
            raise ValueError("all geometry tensors must occupy the same device")
        for name, tensor in (
            ("expert_keys", self.expert_keys),
            ("input_basis", self.input_basis),
            ("singular_values", self.singular_values),
            ("row_norms", self.row_norms),
            ("centered_bias", self.centered_bias),
        ):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite and floating-point")
        if (self.singular_values < 0).any() or (self.row_norms < 0).any():
            raise ValueError("singular values and row norms must be non-negative")
        if (self.ranks < 0).any() or (self.ranks > r).any():
            raise ValueError("ranks lie outside padded rank geometry")
        if not torch.equal(self.rank_mask.sum(-1).to(self.ranks.dtype), self.ranks):
            raise ValueError("rank_mask cardinality disagrees with ranks")
        if r:
            canonical_mask = (
                torch.arange(r, device=self.ranks.device)[None, :] < self.ranks[:, None]
            )
            if not torch.equal(self.rank_mask, canonical_mask):
                raise ValueError("rank_mask must be left-packed")
            if (self.expert_keys.masked_select(~self.rank_mask[:, None, :]) != 0).any():
                raise ValueError("padded expert-key coordinates must be zero")
            if (self.input_basis.masked_select(~self.rank_mask[:, None, :]) != 0).any():
                raise ValueError("padded input-basis coordinates must be zero")
        if not math.isfinite(float(self.relative_rank_threshold)) or not (
            0.0 <= float(self.relative_rank_threshold) < 1.0
        ):
            raise ValueError("relative_rank_threshold must lie in [0, 1)")

    def reconstruct_centered_weights(self) -> Tensor:
        """Return ``K @ V.T`` with shape ``[L,E,D]``."""

        with torch.autocast(device_type=self.expert_keys.device.type, enabled=False):
            return torch.einsum(
                "ler,ldr->led",
                self.expert_keys.float(),
                self.input_basis.float(),
            )

    def encode_router_inputs(self, router_inputs: Tensor) -> Tensor:
        """Project ``[...,L,D]`` router inputs into padded ``[...,L,R]`` q."""

        if router_inputs.shape[-2:] != (self.layers, self.hidden_width):
            raise ValueError(
                "router_inputs must end in [layers, hidden]: expected "
                f"{(self.layers, self.hidden_width)}, got {tuple(router_inputs.shape[-2:])}"
            )
        values = router_inputs.to(device=self.input_basis.device)
        with torch.autocast(device_type=values.device.type, enabled=False):
            coordinates = torch.einsum(
                "...ld,ldr->...lr", values.float(), self.input_basis.float()
            )
            return coordinates * self.rank_mask.to(coordinates.dtype)

    def score_coordinates(self, coordinates: Tensor) -> Tensor:
        """Score padded ``[...,L,R]`` coordinates into centered logits."""

        if coordinates.shape[-2:] != (self.layers, self.maximum_rank):
            raise ValueError(
                "coordinates must end in [layers, maximum_rank]: expected "
                f"{(self.layers, self.maximum_rank)}, got {tuple(coordinates.shape[-2:])}"
            )
        values = coordinates.to(device=self.expert_keys.device)
        with torch.autocast(device_type=values.device.type, enabled=False):
            values = values.float() * self.rank_mask.to(torch.float32)
            return (
                torch.einsum("...lr,ler->...le", values, self.expert_keys.float())
                + self.centered_bias.float()
            )

    def centered_logits(self, router_inputs: Tensor) -> Tensor:
        """Project and score ``[...,L,D]`` inputs through the frozen geometry."""

        return self.score_coordinates(self.encode_router_inputs(router_inputs))

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "CenteredRouterGeometry":
        """Move geometry while preserving boolean/integer contract tensors."""

        def move_float(value: Tensor) -> Tensor:
            return value.to(device=device, dtype=dtype or value.dtype)

        result = CenteredRouterGeometry(
            expert_keys=move_float(self.expert_keys),
            input_basis=move_float(self.input_basis),
            rank_mask=self.rank_mask.to(device=device),
            singular_values=move_float(self.singular_values),
            ranks=self.ranks.to(device=device),
            row_norms=move_float(self.row_norms),
            centered_bias=move_float(self.centered_bias),
            relative_rank_threshold=self.relative_rank_threshold,
        )
        result.validate()
        return result

    def to_tensor_dict(self) -> dict[str, Tensor]:
        """Return a safetensors-compatible v2 payload without source weights."""

        self.validate()
        return {
            "expert_keys": self.expert_keys.detach().contiguous(),
            "input_basis": self.input_basis.detach().contiguous(),
            "rank_mask": self.rank_mask.detach().contiguous(),
            "singular_values": self.singular_values.detach().contiguous(),
            "ranks": self.ranks.detach().contiguous(),
            "row_norms": self.row_norms.detach().contiguous(),
            "centered_bias": self.centered_bias.detach().contiguous(),
            "relative_rank_threshold": torch.tensor(
                self.relative_rank_threshold,
                dtype=torch.float64,
                device=self.expert_keys.device,
            ),
        }

    @classmethod
    def from_tensor_dict(
        cls, tensors: Mapping[str, Tensor]
    ) -> "CenteredRouterGeometry":
        """Restore and validate a payload tagged with ``ROUTER_GEOMETRY_SCHEMA``."""

        required = {
            "expert_keys",
            "input_basis",
            "rank_mask",
            "singular_values",
            "ranks",
            "row_norms",
            "centered_bias",
            "relative_rank_threshold",
        }
        missing = sorted(required - tensors.keys())
        unexpected = sorted(tensors.keys() - required)
        if missing or unexpected:
            raise ValueError(
                f"router geometry tensor keys disagree with v2: missing={missing}, "
                f"unexpected={unexpected}"
            )
        threshold = tensors["relative_rank_threshold"]
        if threshold.numel() != 1:
            raise ValueError("relative_rank_threshold must be scalar")
        result = cls(
            expert_keys=tensors["expert_keys"].contiguous(),
            input_basis=tensors["input_basis"].contiguous(),
            rank_mask=tensors["rank_mask"].bool().contiguous(),
            singular_values=tensors["singular_values"].contiguous(),
            ranks=tensors["ranks"].to(torch.int64).contiguous(),
            row_norms=tensors["row_norms"].contiguous(),
            centered_bias=tensors["centered_bias"].contiguous(),
            relative_rank_threshold=float(threshold.item()),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class RouterGeometryAudit:
    """Reconstruction and decision-equivalence evidence for one geometry."""

    maximum_absolute_weight_error: float
    relative_weight_rms_error: float
    maximum_absolute_bias_error: float
    maximum_absolute_logit_error: float | None
    topk_rows: int
    topk_matching_rows: int
    topk_agreement: float | None
    ranks: tuple[int, ...]

    @property
    def identical_topk(self) -> bool | None:
        if self.topk_agreement is None:
            return None
        return self.topk_matching_rows == self.topk_rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_absolute_weight_error": self.maximum_absolute_weight_error,
            "relative_weight_rms_error": self.relative_weight_rms_error,
            "maximum_absolute_bias_error": self.maximum_absolute_bias_error,
            "maximum_absolute_logit_error": self.maximum_absolute_logit_error,
            "topk_rows": self.topk_rows,
            "topk_matching_rows": self.topk_matching_rows,
            "topk_agreement": self.topk_agreement,
            "identical_topk": self.identical_topk,
            "ranks": list(self.ranks),
        }


def build_centered_router_geometry(
    weights: Tensor | Any,
    bias: Tensor | Any | None = None,
    *,
    relative_rank_threshold: float = 1e-6,
) -> CenteredRouterGeometry:
    """Build padded per-layer ``K``, ``V``, rank masks, norms, and bias."""

    if not math.isfinite(float(relative_rank_threshold)) or not (
        0.0 <= float(relative_rank_threshold) < 1.0
    ):
        raise ValueError("relative_rank_threshold must lie in [0, 1)")
    source = _as_float_tensor(weights, name="weights")
    if source.ndim != 3:
        raise ValueError("weights must be [layers, experts, hidden]")
    layers, experts, hidden = map(int, source.shape)
    if layers < 1 or experts < 2 or hidden < 1:
        raise ValueError("router geometry requires L>=1, E>=2, and D>=1")
    if bias is None:
        source_bias = torch.zeros(
            (layers, experts), dtype=source.dtype, device=source.device
        )
    else:
        source_bias = _as_float_tensor(bias, name="bias").to(source.device)
        if source_bias.shape != (layers, experts):
            raise ValueError(
                f"bias must have shape {(layers, experts)}, got {tuple(source_bias.shape)}"
            )

    centered_weights = _center_experts(source)
    centered_bias = _center_bias(source_bias)
    factorizations: list[tuple[Tensor, Tensor, Tensor, int]] = []
    ranks: list[int] = []
    for layer in range(layers):
        u, singular, vh = torch.linalg.svd(
            centered_weights[layer], full_matrices=False
        )
        if singular.numel() == 0 or float(singular[0]) == 0.0:
            rank = 0
        else:
            rank = int(
                (
                    singular
                    > singular[0] * float(relative_rank_threshold)
                ).sum()
            )
        factorizations.append((u, singular, vh, rank))
        ranks.append(rank)

    maximum_rank = max(ranks)
    expert_keys = torch.zeros(
        (layers, experts, maximum_rank), dtype=source.dtype, device=source.device
    )
    input_basis = torch.zeros(
        (layers, hidden, maximum_rank), dtype=source.dtype, device=source.device
    )
    singular_values = torch.zeros(
        (layers, maximum_rank), dtype=source.dtype, device=source.device
    )
    rank_mask = torch.zeros(
        (layers, maximum_rank), dtype=torch.bool, device=source.device
    )
    for layer, (u, singular, vh, rank) in enumerate(factorizations):
        if rank:
            expert_keys[layer, :, :rank] = u[:, :rank] * singular[:rank]
            input_basis[layer, :, :rank] = vh[:rank].T
            singular_values[layer, :rank] = singular[:rank]
            rank_mask[layer, :rank] = True

    result = CenteredRouterGeometry(
        expert_keys=expert_keys,
        input_basis=input_basis,
        rank_mask=rank_mask,
        singular_values=singular_values,
        ranks=torch.tensor(ranks, dtype=torch.int64, device=source.device),
        row_norms=torch.linalg.vector_norm(centered_weights, dim=-1),
        centered_bias=centered_bias,
        relative_rank_threshold=float(relative_rank_threshold),
    )
    result.validate()
    return result


def audit_centered_router_geometry(
    geometry: CenteredRouterGeometry,
    weights: Tensor | Any,
    bias: Tensor | Any | None = None,
    *,
    router_inputs: Tensor | Any | None = None,
    k: int = 8,
) -> RouterGeometryAudit:
    """Audit centered affine reconstruction and stable top-``k`` decisions."""

    geometry.validate()
    source = _as_float_tensor(weights, name="weights").to(geometry.expert_keys.device)
    expected_shape = (geometry.layers, geometry.experts, geometry.hidden_width)
    if source.shape != expected_shape:
        raise ValueError(f"weights must have shape {expected_shape}, got {tuple(source.shape)}")
    if bias is None:
        source_bias = torch.zeros(
            (geometry.layers, geometry.experts),
            dtype=source.dtype,
            device=source.device,
        )
    else:
        source_bias = _as_float_tensor(bias, name="bias").to(source.device)
        if source_bias.shape != (geometry.layers, geometry.experts):
            raise ValueError("bias geometry disagrees with weights")
    if not 1 <= int(k) <= geometry.experts:
        raise ValueError(f"k must lie in 1..{geometry.experts}")

    expected_weights = _center_experts(source)
    expected_bias = _center_bias(source_bias)
    reconstructed = geometry.reconstruct_centered_weights()
    weight_difference = reconstructed - expected_weights
    denominator = expected_weights.square().mean().sqrt().clamp_min(1e-30)
    maximum_absolute_logit_error: float | None = None
    topk_rows = 0
    topk_matching_rows = 0
    agreement: float | None = None
    if router_inputs is not None:
        inputs = _as_float_tensor(router_inputs, name="router_inputs").to(source.device)
        if inputs.shape[-2:] != (geometry.layers, geometry.hidden_width):
            raise ValueError(
                "router_inputs must end in [layers, hidden]: expected "
                f"{(geometry.layers, geometry.hidden_width)}, got {tuple(inputs.shape[-2:])}"
            )
        direct = torch.einsum("...ld,led->...le", inputs, source) + source_bias
        direct = direct - direct.mean(dim=-1, keepdim=True)
        factored = geometry.centered_logits(inputs)
        maximum_absolute_logit_error = float((factored - direct).abs().max())
        direct_ids = _stable_topk(direct, int(k))
        factored_ids = _stable_topk(factored, int(k))
        matches = (direct_ids == factored_ids).all(dim=-1)
        topk_rows = int(matches.numel())
        topk_matching_rows = int(matches.sum())
        agreement = topk_matching_rows / max(topk_rows, 1)

    return RouterGeometryAudit(
        maximum_absolute_weight_error=float(weight_difference.abs().max()),
        relative_weight_rms_error=float(
            weight_difference.square().mean().sqrt() / denominator
        ),
        maximum_absolute_bias_error=float(
            (geometry.centered_bias - expected_bias).abs().max()
        ),
        maximum_absolute_logit_error=maximum_absolute_logit_error,
        topk_rows=topk_rows,
        topk_matching_rows=topk_matching_rows,
        topk_agreement=agreement,
        ranks=tuple(int(value) for value in geometry.ranks.tolist()),
    )


def assert_router_geometry_equivalent(
    geometry: CenteredRouterGeometry,
    weights: Tensor | Any,
    bias: Tensor | Any | None = None,
    *,
    router_inputs: Tensor | Any,
    k: int = 8,
    maximum_relative_weight_rms_error: float = 2e-6,
    maximum_absolute_logit_error: float = 2e-4,
) -> RouterGeometryAudit:
    """Raise when reconstruction or any audited stable top-k row differs."""

    audit = audit_centered_router_geometry(
        geometry,
        weights,
        bias,
        router_inputs=router_inputs,
        k=k,
    )
    failures: list[str] = []
    if audit.relative_weight_rms_error > maximum_relative_weight_rms_error:
        failures.append(
            "relative weight RMS error "
            f"{audit.relative_weight_rms_error:.3e} exceeds "
            f"{maximum_relative_weight_rms_error:.3e}"
        )
    if (
        audit.maximum_absolute_logit_error is None
        or audit.maximum_absolute_logit_error > maximum_absolute_logit_error
    ):
        failures.append(
            "maximum absolute logit error "
            f"{audit.maximum_absolute_logit_error} exceeds "
            f"{maximum_absolute_logit_error:.3e}"
        )
    if audit.identical_topk is not True:
        failures.append(
            f"stable top-{k} agreement is {audit.topk_matching_rows}/{audit.topk_rows}"
        )
    if failures:
        raise ValueError("centered router geometry audit failed: " + "; ".join(failures))
    return audit


__all__ = [
    "ROUTER_GEOMETRY_SCHEMA",
    "CenteredRouterGeometry",
    "RouterGeometryAudit",
    "assert_router_geometry_equivalent",
    "audit_centered_router_geometry",
    "build_centered_router_geometry",
]
