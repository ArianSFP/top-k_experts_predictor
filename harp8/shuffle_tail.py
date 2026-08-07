"""Create the preregistered tail-shuffled MTP feature control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .features import sha256_file


SHUFFLE_SCHEMA = "harp8_mtp_tail_shuffle_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_shuffled_tail(
    input_path: Path,
    output_path: Path,
    *,
    capture_dir: Path,
    split_manifest: Path,
    active_width: int = 128,
    rows_per_request: int = 34,
    seed: int = 20260807,
) -> dict[str, Any]:
    """Shuffle only PCA coordinates after ``active_width`` within exact strata.

    The first coordinates remain bit-identical. Extra coordinates are shuffled
    independently within (inner split, domain, within-position, depth), with
    outer validation/test rows left unchanged because they are not used for
    the nested design decision.
    """
    if output_path.exists():
        raise FileExistsError(f"refusing to reuse shuffled feature file {output_path}")
    raw = np.load(input_path, mmap_mode="r")
    if raw.ndim != 3 or active_width <= 0 or active_width >= raw.shape[-1]:
        raise ValueError("expected [rows,depth,width] with active width inside feature width")
    requests = _read_jsonl(Path(capture_dir) / "requests.jsonl")
    split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    assignments = {int(key): str(value) for key, value in split["assignments"].items()}
    if raw.shape[0] != len(requests) * rows_per_request:
        raise ValueError("feature rows disagree with request geometry")
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=raw.dtype, shape=raw.shape
    )
    for start in range(0, raw.shape[0], 512):
        stop = min(raw.shape[0], start + 512)
        output[start:stop] = np.asarray(raw[start:stop])
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str, int], list[int]] = {}
    for request_number, request in enumerate(requests):
        request_id = int(request["request_id"])
        split_name = assignments.get(request_id)
        if split_name is None:
            continue
        domain = str(request.get("domain", "unknown"))
        for within in range(rows_per_request - 1):
            row = request_number * rows_per_request + within
            groups.setdefault((split_name, domain, within), []).append(row)
    for depth in range(raw.shape[1]):
        for key, rows in sorted(groups.items()):
            row_array = np.asarray(rows, dtype=np.int64)
            if len(row_array) < 2:
                continue
            permutation = rng.permutation(len(row_array))
            values = np.asarray(raw[row_array, depth, active_width:], dtype=raw.dtype)
            output[row_array, depth, active_width:] = values[permutation]
    output.flush()
    manifest = {
        "schema": SHUFFLE_SCHEMA,
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "active_width": active_width,
        "rows_per_request": rows_per_request,
        "seed": seed,
        "strata": ["inner_split", "domain", "within_position", "depth"],
        "outer_rows_unchanged": True,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--active-width", type=int, default=128)
    parser.add_argument("--rows-per-request", type=int, default=34)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(write_shuffled_tail(
        args.input, args.output, capture_dir=args.capture_dir,
        split_manifest=args.split_manifest, active_width=args.active_width,
        rows_per_request=args.rows_per_request, seed=args.seed,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
