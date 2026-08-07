"""Frozen target-router geometry for candidate-specific HARP scoring.

The geometry is deliberately derived from the authoritative BF16 router rows,
not from expert IDs or from an MTP router.  A thin SVD writes each target
expert as a row of ``U @ diag(S)``.  At full rank these keys preserve the
router-row Gram matrix exactly and avoid an avoidable rank bottleneck.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .train import canonical_json, sha256_file


ROUTER_KEY_SCHEMA = "harp8_router_svd_keys_v1"


def load_target_router_weights(path: Path) -> torch.Tensor:
    """Load ``[layers, experts, hidden]`` weights from supported artifacts."""
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - dependency is required remotely
        raise RuntimeError("safetensors is required for router artifacts") from exc

    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as source:
        names = set(source.keys())
        if "target_router_weights" in names:
            weights = source.get_tensor("target_router_weights")
        else:
            layer_names = sorted(
                name for name in names if name.startswith("target_router_weight.layer_")
            )
            if not layer_names:
                raise ValueError("router artifact contains no target router weights")
            expected = [f"target_router_weight.layer_{index:02d}" for index in range(len(layer_names))]
            if layer_names != expected:
                raise ValueError("per-layer target router weights are incomplete or unordered")
            weights = torch.stack([source.get_tensor(name) for name in layer_names])
    if weights.ndim != 3:
        raise ValueError(f"target router weights must be [L,E,D], got {tuple(weights.shape)}")
    if not torch.isfinite(weights.float()).all():
        raise ValueError("target router weights contain NaN or Inf")
    return weights.float().contiguous()


def router_svd_keys(
    weights: torch.Tensor | np.ndarray,
    *,
    rank: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return per-layer SVD keys ``U[:, :R] * S[:R]`` and diagnostics."""
    value = torch.as_tensor(weights, dtype=torch.float32, device="cpu")
    if value.ndim != 3:
        raise ValueError("router weights must be [layers, experts, hidden]")
    layers, experts, hidden = map(int, value.shape)
    maximum = min(experts, hidden)
    selected_rank = maximum if rank is None else int(rank)
    if not 1 <= selected_rank <= maximum:
        raise ValueError(f"rank must lie in 1..{maximum}")
    keys = torch.empty((layers, experts, selected_rank), dtype=torch.float32)
    relative_gram_errors: list[float] = []
    retained_energy: list[float] = []
    for layer in range(layers):
        matrix = value[layer]
        u, singular, _vh = torch.linalg.svd(matrix, full_matrices=False)
        keys[layer] = u[:, :selected_rank] * singular[:selected_rank]
        gram = matrix @ matrix.T
        reconstructed = keys[layer] @ keys[layer].T
        relative_gram_errors.append(
            float(torch.linalg.vector_norm(reconstructed - gram) / torch.linalg.vector_norm(gram).clamp_min(1e-30))
        )
        retained_energy.append(
            float(singular[:selected_rank].square().sum() / singular.square().sum().clamp_min(1e-30))
        )
    diagnostics = {
        "layers": layers,
        "experts": experts,
        "hidden_width": hidden,
        "rank": selected_rank,
        "relative_gram_error_max": max(relative_gram_errors),
        "relative_gram_error_mean": float(np.mean(relative_gram_errors)),
        "retained_energy_min": min(retained_energy),
        "retained_energy_mean": float(np.mean(retained_energy)),
    }
    return keys, diagnostics


def export_router_keys(
    router_artifact: Path,
    output_dir: Path,
    *,
    rank: int | None = None,
) -> dict[str, Any]:
    """Export immutable NumPy keys and a provenance manifest."""
    router_artifact = Path(router_artifact)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse router-key directory {output_dir}")
    output_dir.mkdir(parents=True)
    weights = load_target_router_weights(router_artifact)
    keys, diagnostics = router_svd_keys(weights, rank=rank)
    key_path = output_dir / "router_svd_keys.npy"
    np.save(key_path, keys.numpy().astype(np.float32, copy=False), allow_pickle=False)
    manifest = {
        "schema": ROUTER_KEY_SCHEMA,
        "router_artifact": str(router_artifact),
        "router_artifact_sha256": sha256_file(router_artifact),
        "keys": {
            "path": key_path.name,
            "sha256": sha256_file(key_path),
            "dtype": "float32",
            "shape": list(keys.shape),
        },
        "diagnostics": diagnostics,
    }
    (output_dir / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_router_keys(args.router_artifact, args.output_dir, rank=args.rank)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
