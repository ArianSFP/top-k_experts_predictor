"""Prepare leakage-safe nested PCA and standardized raw MTP features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


FEATURE_SCHEMA = "harp8_mtp_features_v1"
SPLIT_SCHEMA = "harp8_inner_split_v1"


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_inner_split(
    requests: list[dict[str, Any]],
    *,
    development_fraction: float = 0.20,
    seed: int = 20260807,
) -> dict[str, Any]:
    """Stratify the original train requests by domain without touching outer sets."""

    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development fraction must lie in (0,1)")
    grouped: dict[str, list[int]] = {}
    for row in requests:
        if str(row["offline_split"]) != "train":
            continue
        grouped.setdefault(str(row.get("domain", "unknown")), []).append(
            int(row["request_id"])
        )
    total = sum(len(values) for values in grouped.values())
    target_development = int(round(total * development_fraction))
    exact = {
        domain: len(values) * development_fraction
        for domain, values in grouped.items()
    }
    allocation = {domain: int(np.floor(value)) for domain, value in exact.items()}
    remainder = target_development - sum(allocation.values())
    order = sorted(
        grouped,
        key=lambda domain: (-(exact[domain] - allocation[domain]), domain),
    )
    for domain in order[:remainder]:
        allocation[domain] += 1

    assignments: dict[str, str] = {}
    domain_counts: dict[str, dict[str, int]] = {}
    for domain, request_ids in sorted(grouped.items()):
        domain_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{domain}".encode()).digest()[:8], "little"
        )
        rng = np.random.default_rng(domain_seed)
        shuffled = np.asarray(sorted(request_ids), dtype=np.int64)
        rng.shuffle(shuffled)
        development = set(
            int(value) for value in shuffled[: allocation[domain]].tolist()
        )
        for request_id in request_ids:
            assignments[str(request_id)] = (
                "validation" if request_id in development else "train"
            )
        domain_counts[domain] = {
            "train": len(request_ids) - len(development),
            "validation": len(development),
        }
    counts = {
        split: sum(value == split for value in assignments.values())
        for split in ("train", "validation")
    }
    if counts != {"train": total - target_development, "validation": target_development}:
        raise AssertionError("inner split allocation disagrees with requested size")
    return {
        "schema": SPLIT_SCHEMA,
        "seed": seed,
        "development_fraction": development_fraction,
        "source_offline_split": "train",
        "outer_validation_and_test_excluded": True,
        "counts": counts,
        "domain_counts": domain_counts,
        "assignments": dict(sorted(assignments.items(), key=lambda item: int(item[0]))),
    }


def _fit_indices(
    requests: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    rows_per_request: int,
    fit_split: str,
) -> np.ndarray:
    request_to_number = {
        int(row["request_id"]): number for number, row in enumerate(requests)
    }
    if fit_split == "inner_train":
        request_ids = [
            int(request_id)
            for request_id, value in split["assignments"].items()
            if value == "train"
        ]
    elif fit_split == "full_train":
        request_ids = [
            int(row["request_id"])
            for row in requests
            if str(row["offline_split"]) == "train"
        ]
    else:
        raise ValueError("fit split must be inner_train or full_train")
    return np.asarray(
        [
            request_to_number[request_id] * rows_per_request + within
            for request_id in request_ids
            for within in range(rows_per_request - 1)
        ],
        dtype=np.int64,
    )


def _fit_depth_pca(
    values: np.ndarray,
    selected: np.ndarray,
    rank: int,
    *,
    device: str,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    matrix = torch.as_tensor(
        np.asarray(values[selected], dtype=np.float32),
        device=device,
    )
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    q = min(rank, centered.shape[0] - 1, centered.shape[1])
    _u, singular, components = torch.pca_lowrank(
        centered,
        q=q,
        center=False,
        niter=iterations,
    )
    total_energy = centered.square().sum().clamp_min(1e-12)
    cumulative = singular.square().cumsum(dim=0) / total_energy
    metadata = {
        "fit_rows": int(len(selected)),
        "rank": int(q),
        "captured_train_variance_fraction": float(cumulative[-1]),
        "captured_variance_at_rank": {
            str(value): float(cumulative[min(value, q) - 1])
            for value in (128, 256, 512, 1024)
            if value <= q
        },
        "singular_values": singular.detach().cpu().numpy().astype(float).tolist(),
    }
    return (
        mean.detach().cpu().numpy().astype(np.float32),
        components.detach().cpu().numpy().astype(np.float32),
        singular.detach().cpu().numpy().astype(np.float32),
        metadata,
    )


def prepare_features(
    capture_dir: Path,
    mtp_dir: Path,
    output_dir: Path,
    *,
    maximum_rank: int = 1024,
    maximum_pca_rows: int = 30000,
    rows_per_request: int = 34,
    development_fraction: float = 0.20,
    split_seed: int = 20260807,
    pca_seed: int = 42,
    pca_iterations: int = 6,
    fit_split: str = "inner_train",
    device: str = "cuda:0",
    write_raw_standardized: bool = True,
    chunk_rows: int = 512,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse feature directory {output_dir}")
    output_dir.mkdir(parents=True)
    requests = _read_jsonl(Path(capture_dir) / "requests.jsonl")
    split = make_inner_split(
        requests,
        development_fraction=development_fraction,
        seed=split_seed,
    )
    split_path = output_dir / "inner_split_manifest.json"
    split_path.write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raw_path = Path(mtp_dir) / "mtp_hidden_depths.npy"
    raw = np.load(raw_path, mmap_mode="r")
    if raw.ndim != 3 or raw.shape[2] != 2048:
        raise ValueError("raw MTP hidden states must be [rows,depth,2048]")
    if raw.shape[0] != len(requests) * rows_per_request:
        raise ValueError("raw MTP rows disagree with request geometry")
    fit_indices = _fit_indices(
        requests,
        split,
        rows_per_request=rows_per_request,
        fit_split=fit_split,
    )
    rng = np.random.default_rng(pca_seed)
    if len(fit_indices) > maximum_pca_rows:
        selected = np.sort(
            rng.choice(fit_indices, maximum_pca_rows, replace=False)
        )
    else:
        selected = fit_indices

    pca_output = output_dir / f"mtp_pca_rank{maximum_rank}.npy"
    pca = np.lib.format.open_memmap(
        pca_output,
        mode="w+",
        dtype="<f2",
        shape=(raw.shape[0], raw.shape[1], maximum_rank),
    )
    raw_output = output_dir / "mtp_raw_standardized.npy"
    standardized = (
        np.lib.format.open_memmap(
            raw_output,
            mode="w+",
            dtype="<f2",
            shape=raw.shape,
        )
        if write_raw_standardized
        else None
    )
    means: list[np.ndarray] = []
    components: list[np.ndarray] = []
    singular_values: list[np.ndarray] = []
    raw_stds: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for depth in range(raw.shape[1]):
        mean, basis, singular, record = _fit_depth_pca(
            raw[:, depth],
            selected,
            maximum_rank,
            device=device,
            iterations=pca_iterations,
        )
        record["draft_depth"] = depth + 1
        fit_values = np.asarray(raw[selected, depth], dtype=np.float32)
        raw_std = fit_values.std(axis=0, dtype=np.float64).astype(np.float32)
        raw_std = np.maximum(raw_std, 1e-6)
        for start in range(0, raw.shape[0], chunk_rows):
            stop = min(raw.shape[0], start + chunk_rows)
            chunk = np.asarray(raw[start:stop, depth], dtype=np.float32)
            pca[start:stop, depth] = (chunk - mean) @ basis
            if standardized is not None:
                standardized[start:stop, depth] = np.clip(
                    (chunk - mean) / raw_std,
                    -8.0,
                    8.0,
                )
        means.append(mean)
        components.append(basis)
        singular_values.append(singular)
        raw_stds.append(raw_std)
        records.append(record)
    pca.flush()
    if standardized is not None:
        standardized.flush()
    preprocessing_path = output_dir / "preprocessing.pt"
    torch.save(
        {
            "schema": FEATURE_SCHEMA,
            "fit_split": fit_split,
            "fit_indices_sha256": hashlib.sha256(
                np.asarray(selected, dtype=np.int64).tobytes()
            ).hexdigest(),
            "pca_mean": torch.from_numpy(np.stack(means)),
            "pca_components": torch.from_numpy(np.stack(components)),
            "pca_singular_values": torch.from_numpy(np.stack(singular_values)),
            "raw_std": torch.from_numpy(np.stack(raw_stds)),
        },
        preprocessing_path,
    )
    manifest = {
        "schema": FEATURE_SCHEMA,
        "fit_split": fit_split,
        "maximum_rank": maximum_rank,
        "maximum_pca_rows": maximum_pca_rows,
        "fit_rows": int(len(selected)),
        "rows_per_request": rows_per_request,
        "raw_source": str(raw_path),
        "raw_source_sha256": sha256_file(raw_path),
        "depths": records,
        "training_request_only": True,
        "split_manifest": {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
        },
        "outputs": {
            pca_output.name: {
                "shape": list(pca.shape),
                "dtype": str(pca.dtype),
                "sha256": sha256_file(pca_output),
            },
            preprocessing_path.name: {
                "sha256": sha256_file(preprocessing_path),
            },
        },
    }
    if standardized is not None:
        manifest["outputs"][raw_output.name] = {
            "shape": list(standardized.shape),
            "dtype": str(standardized.dtype),
            "sha256": sha256_file(raw_output),
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mtp-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-rank", type=int, default=1024)
    parser.add_argument("--maximum-pca-rows", type=int, default=30000)
    parser.add_argument("--rows-per-request", type=int, default=34)
    parser.add_argument("--development-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=20260807)
    parser.add_argument("--pca-seed", type=int, default=42)
    parser.add_argument("--pca-iterations", type=int, default=6)
    parser.add_argument(
        "--fit-split",
        choices=("inner_train", "full_train"),
        default="inner_train",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-raw-standardized", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_features(
        args.capture_dir,
        args.mtp_dir,
        args.output_dir,
        maximum_rank=args.maximum_rank,
        maximum_pca_rows=args.maximum_pca_rows,
        rows_per_request=args.rows_per_request,
        development_fraction=args.development_fraction,
        split_seed=args.split_seed,
        pca_seed=args.pca_seed,
        pca_iterations=args.pca_iterations,
        fit_split=args.fit_split,
        device=args.device,
        write_raw_standardized=not args.skip_raw_standardized,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
