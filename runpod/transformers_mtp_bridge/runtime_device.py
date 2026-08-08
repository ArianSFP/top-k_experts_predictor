"""Execution-device validation and capture-manifest provenance helpers.

This module deliberately has no dependency on a particular Transformers model
class, which keeps CPU/CUDA selection independently testable on hosts that do
not have the pinned Qwen Transformers revision installed.
"""

from __future__ import annotations

import os
import platform
import sys

import torch


def resolve_execution_device(value: str | torch.device) -> torch.device:
    """Return a supported, available execution device.

    HARP-RTT capture is intentionally restricted to CPU or CUDA. There is no
    silent fallback: a requested CUDA capture must not turn into a CPU capture
    with different numerical behavior and provenance.
    """

    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid capture device {value!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("capture device must be 'cpu', 'cuda', or 'cuda:<index>'")
    if device.type == "cpu" and device.index is not None:
        raise ValueError("CPU capture device must not include an index")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA capture device {device} was requested but CUDA is unavailable"
        )
    if device.type == "cuda" and device.index is not None:
        device_count = torch.cuda.device_count()
        if not 0 <= device.index < device_count:
            raise ValueError(
                f"CUDA device index {device.index} is outside [0, {device_count})"
            )
    return device


def target_device_map(value: str | torch.device) -> dict[str, str]:
    """Build the single-device Accelerate map used by the frozen target."""

    return {"": str(resolve_execution_device(value))}


def hardware_topology_manifest(
    value: str | torch.device,
) -> dict[str, object]:
    """Describe the actual execution device without CUDA-only calls on CPU."""

    device = torch.device(value)
    common: dict[str, object] = {
        "execution_device": str(device),
        "device_type": device.type,
        "accelerator": None,
        "accelerator_index": None,
        "cuda_capability": None,
        # ``cuda_runtime`` historically described the active CUDA execution
        # path. Keep it null for CPU while recording the PyTorch build
        # separately, since a CUDA-enabled wheel can execute entirely on CPU.
        "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_architecture": platform.machine(),
        "cpu_processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "host": platform.node(),
        "python": sys.version,
    }
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        common.update(
            {
                "execution_device": f"cuda:{index}",
                "accelerator": torch.cuda.get_device_name(index),
                "accelerator_index": index,
                "cuda_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    elif device.type != "cpu":
        raise ValueError(f"unsupported capture execution device {device}")
    return common


def clock_domain_definitions(
    value: str | torch.device,
) -> dict[str, str]:
    """Return clock descriptions that match CPU or CUDA execution semantics."""

    device = torch.device(value)
    if device.type == "cpu":
        execution_order = (
            "logical host execution order; CPU model calls complete before the "
            "following event write; not a performance timestamp"
        )
    elif device.type == "cuda":
        execution_order = (
            "logical host issue order on the selected CUDA stream; host scalar and "
            "tensor materialization synchronizes the values used by event writes, "
            "but there is no explicit per-call CUDA timing barrier; not a "
            "performance timestamp"
        )
    else:
        raise ValueError(f"unsupported capture execution device {device}")
    return {"synchronous_transformers_execution_order": execution_order}
