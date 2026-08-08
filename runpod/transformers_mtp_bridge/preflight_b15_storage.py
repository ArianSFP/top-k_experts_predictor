#!/usr/bin/env python3
"""Fail closed unless B1.5 has enough free artifact storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp_rtt.b15 import required_b15_free_bytes  # noqa: E402


def parse_component(value: str) -> tuple[str, int]:
    name, separator, raw = value.partition("=")
    if not separator or not name or not raw:
        raise argparse.ArgumentTypeError("component must be NAME=BYTES")
    try:
        size = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("component bytes must be an integer") from error
    if size < 0:
        raise argparse.ArgumentTypeError("component bytes must be non-negative")
    return name, size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--component-bytes",
        action="append",
        type=parse_component,
        required=True,
        metavar="NAME=BYTES",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    components = dict(args.component_bytes)
    if len(components) != len(args.component_bytes):
        raise ValueError("storage component names must be unique")
    estimated = sum(components.values())
    required = required_b15_free_bytes(estimated)
    free = shutil.disk_usage(path).free
    passed = free >= required
    report = {
        "schema": "harp_rtt_b15_storage_preflight_v1",
        "passed": passed,
        "path": str(path),
        "components_bytes": components,
        "estimated_total_bytes": estimated,
        "free_bytes": free,
        "required_free_bytes": required,
        "formula": "max(100_GiB, ceil(1.25 * estimated_total_bytes))",
        "optimizer_started": False,
        "training_started": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
