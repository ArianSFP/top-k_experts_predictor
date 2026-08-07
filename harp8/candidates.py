"""Export fixed-width HARP candidate pools for offline scheduler studies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .config import HARPConfig
from .data import CompactHARPData
from .metrics import model_inputs
from .model import HARP8Teacher
from .train import TRAINING_SCHEMA, sha256_file


POOL_SCHEMA = "harp8_candidate_pool_v1"
POOL_SCHEMA_V2 = "harp8_candidate_pool_v2"
SUPPORTED_POOL_SCHEMAS = frozenset({POOL_SCHEMA, POOL_SCHEMA_V2})
REQUEST_IDS_HASH_ENCODING = "sorted_unique_decimal_lf_v1"
DIRECT_EXTENSION_PREFIXES = (
    "direct_route_projection.",
    "direct_mtp_projection.",
    "direct_target_projection.",
    "position_projection.",
)


def canonical_request_ids(request_ids: Iterable[int]) -> tuple[int, ...]:
    """Canonicalize a request set for portable provenance hashing."""

    try:
        values = tuple(sorted({int(value) for value in request_ids}))
    except (TypeError, ValueError) as exc:
        raise ValueError("request IDs must be integers") from exc
    if not values:
        raise ValueError("request-ID set cannot be empty")
    return values


def request_ids_sha256(request_ids: Iterable[int]) -> str:
    """Hash a request *set*, independent of row order and repetition."""

    values = canonical_request_ids(request_ids)
    encoded = "".join(f"{value}\n" for value in values).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read request-ID manifest {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"request-ID manifest {path} must be a JSON object")
    return value


def request_ids_from_split_manifest(path: Path) -> tuple[int, ...]:
    """Return the exact base-fit request IDs from an inner split manifest."""

    manifest = _json_mapping(Path(path))
    if manifest.get("schema") != "harp8_inner_split_v1":
        raise ValueError("base-fit split manifest has an incompatible schema")
    assignments = manifest.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError("base-fit split manifest lacks request assignments")
    invalid = sorted(
        {str(value) for value in assignments.values()} - {"train", "validation"}
    )
    if invalid:
        raise ValueError(f"base-fit split manifest has invalid roles: {invalid}")
    train = [int(request_id) for request_id, role in assignments.items() if role == "train"]
    return canonical_request_ids(train)


def request_ids_from_file(path: Path) -> tuple[int, ...]:
    """Read explicit base-train request IDs from JSON, JSONL, or text."""

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read base-train request IDs {path}") from exc
    if not text.strip():
        raise ValueError("base-train request-ID file is empty")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        return canonical_request_ids(value)
    if isinstance(value, Mapping):
        if "assignments" in value:
            assignments = value["assignments"]
            if not isinstance(assignments, Mapping):
                raise ValueError("base-train assignments must be a mapping")
            return canonical_request_ids(
                int(request_id)
                for request_id, role in assignments.items()
                if str(role) == "train"
            )
        request_ids = value.get("request_ids")
        if isinstance(request_ids, list):
            return canonical_request_ids(request_ids)
        raise ValueError("base-train JSON must contain request_ids or assignments")

    values: list[int] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = line.strip()
        try:
            values.append(
                int(record["request_id"])
                if isinstance(record, Mapping)
                else int(record)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid request ID at {path}:{line_number}"
            ) from exc
    return canonical_request_ids(values)


def _resolve_base_train_request_ids(
    *,
    split_manifest: Path | None,
    request_ids_file: Path | None,
) -> tuple[tuple[int, ...] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve redundant base-fit sources and require exact agreement."""

    from_split = (
        request_ids_from_split_manifest(split_manifest)
        if split_manifest is not None
        else None
    )
    from_file = (
        request_ids_from_file(request_ids_file)
        if request_ids_file is not None
        else None
    )
    if from_split is not None and from_file is not None and from_split != from_file:
        raise ValueError(
            "base-fit split manifest and explicit base-train request IDs disagree"
        )
    values = from_split if from_split is not None else from_file
    split_record = (
        {
            "path": str(split_manifest),
            "sha256": sha256_file(Path(split_manifest)),
            "schema": "harp8_inner_split_v1",
        }
        if split_manifest is not None
        else None
    )
    ids_record = (
        {"path": str(request_ids_file), "sha256": sha256_file(Path(request_ids_file))}
        if request_ids_file is not None
        else None
    )
    return values, split_record, ids_record


def _load_model_state_for_export(
    model: HARP8Teacher,
    state: Mapping[str, torch.Tensor],
) -> str:
    """Load a generator checkpoint without weakening its architecture contract.

    Accuracy-expansion HARP added four additive direct projections whose final
    linear maps are explicitly zero-initialized. A checkpoint predating that
    extension is therefore exactly function-preserving in the extended model.
    No other missing or unexpected state is accepted.
    """

    try:
        model.load_state_dict(state, strict=True)
        return "strict"
    except RuntimeError as strict_error:
        missing, unexpected = model.load_state_dict(state, strict=False)
        expected_missing = {
            key
            for key in model.state_dict()
            if key.startswith(DIRECT_EXTENSION_PREFIXES)
        }
        actual_missing = set(missing)
        if unexpected or actual_missing != expected_missing:
            raise ValueError(
                "checkpoint does not omit exactly the complete zero-initialized "
                "direct extension; "
                f"missing={sorted(actual_missing)}, "
                f"expected_missing={sorted(expected_missing)}, "
                f"unexpected={unexpected}"
            ) from strict_error
        return "zero_initialized_direct_extension"


def _load_checkpoint(
    path: Path, device: str
) -> tuple[HARP8Teacher, dict[str, Any], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != TRAINING_SCHEMA:
        raise ValueError("checkpoint is not an HARP training checkpoint")
    config = HARPConfig(**payload["model_config"])
    model = HARP8Teacher(config)
    load_profile = _load_model_state_for_export(model, payload["model_state"])
    model.to(device).eval()
    return model, payload, load_profile


def export_candidate_pool(
    data: CompactHARPData,
    checkpoint: Path,
    output_dir: Path,
    *,
    source_data_split: str,
    level2_split: str,
    source_offline_split: str,
    base_fold_id: str,
    base_fit_excluded: bool,
    base_fit_split_manifest: Path | None = None,
    base_train_request_ids: Path | None = None,
    candidate_count: int = 16,
    batch_size: int = 64,
    device: str = "cuda:0",
    allow_test: bool = False,
    store_context: bool = True,
) -> dict[str, Any]:
    """Export one request-grouped pool with explicit level-2 OOF lineage.

    ``source_data_split`` selects rows from ``CompactHARPData``.  It is not the
    downstream role: inner-validation rows, for example, may legitimately be
    exported as ``level2_split='train'`` only when a disjoint base-fit request
    set proves that they are out of fold.
    """
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse candidate directory {output_dir}")
    if not 8 < candidate_count <= data.config.experts:
        raise ValueError("candidate_count must exceed native top-8 and fit experts")
    if source_data_split not in {"train", "validation", "test"}:
        raise ValueError("source_data_split must be train, validation, or test")
    if level2_split not in {"train", "validation"}:
        raise ValueError("level2_split must be train or validation")
    if source_offline_split not in {"train", "validation", "test"}:
        raise ValueError("source_offline_split must be train, validation, or test")
    if not str(base_fold_id).strip():
        raise ValueError("base_fold_id must be a non-empty stable identifier")
    if source_data_split == "test" and not allow_test:
        raise PermissionError("sealed test export requires allow_test=True")

    indices = data.indices(source_data_split, allow_test=allow_test)
    exported_request_ids, _domains, _within = data.metadata(indices)
    exported_set = canonical_request_ids(exported_request_ids.tolist())
    request_offline_splits = {
        int(record["request_id"]): str(record["offline_split"])
        for record in data.requests
    }
    actual_offline = {
        request_offline_splits[int(request_id)] for request_id in exported_set
    }
    if actual_offline != {source_offline_split}:
        raise ValueError(
            "source_offline_split disagrees with exported request records: "
            f"declared={source_offline_split!r}, actual={sorted(actual_offline)!r}"
        )
    base_ids, base_split_record, base_ids_record = _resolve_base_train_request_ids(
        split_manifest=(
            Path(base_fit_split_manifest)
            if base_fit_split_manifest is not None
            else None
        ),
        request_ids_file=(
            Path(base_train_request_ids)
            if base_train_request_ids is not None
            else None
        ),
    )
    if level2_split == "train" and not base_fit_excluded:
        raise ValueError("level-2 training export requires base_fit_excluded=true")
    if level2_split == "train" and base_ids is None:
        raise ValueError(
            "level-2 training export requires verifiable base-train request IDs"
        )
    if base_fit_excluded and base_ids is None:
        raise ValueError(
            "base_fit_excluded=true requires a base-fit split manifest or ID file"
        )
    overlap = sorted(set(exported_set).intersection(base_ids or ()))
    if base_fit_excluded and overlap:
        preview = overlap[:8]
        raise ValueError(
            "exported requests overlap the base-fit set despite "
            f"base_fit_excluded=true: {preview}"
        )

    output_dir.mkdir(parents=True)
    model, checkpoint_payload, checkpoint_load_profile = _load_checkpoint(
        checkpoint, device
    )
    count = len(indices)
    h, layers, experts = data.config.horizons, data.config.layers, data.config.experts
    def mm(name: str, dtype: str, shape: tuple[int, ...]) -> np.memmap:
        return np.memmap(output_dir / name, mode="w+", dtype=dtype, shape=shape)
    candidate_scores = mm("candidate_scores.f32", "<f4", (count, h, layers, candidate_count))
    candidate_ids = mm("candidate_ids.u2", "<u2", (count, h, layers, candidate_count))
    target_membership = mm("target_membership.u1", "u1", (count, h, layers, candidate_count))
    teacher_candidate_scores = mm("teacher_candidate_scores.f32", "<f4", (count, h, layers, candidate_count))
    valid_future = mm("valid_future.u1", "u1", (count, h))
    current_scores = mm("current_scores.f32", "<f4", (count, h, layers, candidate_count))
    current_rank = mm("current_rank.f32", "<f4", (count, h, layers, candidate_count))
    source_gates = mm("source_gates.f16", "<f2", (count, h, layers, 3))
    copy_gates = mm("copy_gates.f16", "<f2", (count, h, layers, candidate_count))
    contexts = mm("generator_context.f16", "<f2", (count, h, layers, model.config.model_width)) if store_context else None
    request_ids = np.empty(count, dtype=np.int64)
    within = np.empty(count, dtype=np.int32)
    domains: list[str] = [""] * count

    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            batch_indices = indices[start:stop]
            batch = data.batch(batch_indices, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                outputs = model(**model_inputs(batch))
            scores = outputs["future_router_scores"].float()
            values, ids = torch.topk(scores, candidate_count, dim=-1, sorted=True)
            teacher = batch["teacher_router_scores"].float()
            target = batch["target_top8"].long()
            member = (ids.unsqueeze(-1) == target.unsqueeze(-2)).any(dim=-1)
            current = batch["route_history"][:, :, 0].float()
            current = current - current.mean(dim=-1, keepdim=True)
            current_values = current[:, None].expand(-1, h, -1, -1).gather(-1, ids)
            full_rank = torch.argsort(torch.argsort(-current, dim=-1), dim=-1).float()
            current_ranks = full_rank[:, None].expand(-1, h, -1, -1).gather(-1, ids) / max(1, experts - 1)
            request, domain, local_within = data.metadata(batch_indices)
            request_ids[start:stop], within[start:stop] = request, local_within
            domains[start:stop] = [str(value) for value in domain]
            candidate_scores[start:stop] = values.cpu().numpy()
            candidate_ids[start:stop] = ids.cpu().numpy()
            target_membership[start:stop] = member.cpu().numpy().astype(np.uint8)
            teacher_candidate_scores[start:stop] = teacher.gather(-1, ids).cpu().numpy()
            valid_future[start:stop] = batch["valid_future"].cpu().numpy().astype(np.uint8)
            current_scores[start:stop] = current_values.cpu().numpy()
            current_rank[start:stop] = current_ranks.cpu().numpy()
            source_gates[start:stop] = outputs["source_gate_weights"].float().cpu().numpy()
            # The generator exposes a copy gate for every expert. Candidate pools
            # retain only the selected candidate namespace, exactly as the score
            # and rank arrays do; this also supports widths greater than 16.
            candidate_copy_gates = outputs["copy_gate"].gather(-1, ids)
            copy_gates[start:stop] = candidate_copy_gates.float().cpu().numpy()
            if contexts is not None:
                contexts[start:stop] = outputs["generator_context"].float().cpu().numpy()

    for value in (candidate_scores, candidate_ids, target_membership, teacher_candidate_scores,
                  valid_future, current_scores, current_rank, source_gates, copy_gates, contexts):
        if value is not None:
            value.flush()
    (output_dir / "metadata.json").write_text(json.dumps(
        {"request_ids": request_ids.tolist(), "within": within.tolist(), "domains": domains},
        sort_keys=True) + "\n", encoding="utf-8")
    arrays = [{"path": path.name, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
              for path in sorted(output_dir.iterdir()) if path.suffix in (".f32", ".f16", ".u1", ".u2")]
    source_split_record = (
        {
            "path": str(data.split_manifest_path),
            "sha256": sha256_file(data.split_manifest_path),
        }
        if data.split_manifest_path is not None
        else None
    )
    manifest = {
        "schema": POOL_SCHEMA_V2,
        # ``split`` is retained only for old role-aware consumers.  It is
        # deliberately an alias of level2_split, never the source row selector.
        "split": level2_split,
        "split_field_semantics": "deprecated_alias_of_level2_split",
        "level2_split": level2_split,
        "source_data_split": source_data_split,
        "source_offline_split": source_offline_split,
        "base_fold_id": str(base_fold_id),
        "base_fit_excluded": bool(base_fit_excluded),
        "exported_request_ids_sha256": request_ids_sha256(exported_set),
        "base_train_request_ids_sha256": (
            request_ids_sha256(base_ids) if base_ids is not None else None
        ),
        # Canonical IDs make the exclusion claim independently checkable when
        # the original base-fit manifest path is no longer mounted.
        "base_train_request_ids": list(base_ids) if base_ids is not None else None,
        "request_ids_hash_encoding": REQUEST_IDS_HASH_ENCODING,
        "source_row_split_manifest": source_split_record,
        "base_fit_split_manifest": base_split_record,
        "base_train_request_ids_file": base_ids_record,
        "base_fit_overlap_count": len(overlap),
        "rows": count, "horizons": h,
        "layers": layers, "experts": experts, "native_k": 8,
        "candidate_count": candidate_count, "model_width": model.config.model_width,
        "store_context": store_context,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint),
                       "epoch": checkpoint_payload.get("epoch"), "seed": checkpoint_payload.get("seed")},
        "checkpoint_load_profile": checkpoint_load_profile,
        # Historical alias retained for readers that use the source row split
        # manifest for feature alignment.  Base-fit lineage is always separate.
        "split_manifest": source_split_record,
        "allow_test": bool(allow_test), "arrays": arrays,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-dir", "mtp-dir", "target-state-features", "mtp-state-features"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-data-split",
        choices=("train", "validation", "test"),
        required=True,
        help="row selector in CompactHARPData; distinct from downstream role",
    )
    parser.add_argument(
        "--level2-split", choices=("train", "validation"), required=True
    )
    parser.add_argument(
        "--source-offline-split",
        choices=("train", "validation", "test"),
        required=True,
    )
    parser.add_argument("--base-fold-id", required=True)
    parser.add_argument("--base-fit-excluded", action="store_true")
    parser.add_argument("--base-fit-split-manifest", type=Path)
    parser.add_argument("--base-train-request-ids", type=Path)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = HARPConfig(**payload["model_config"])
    data = CompactHARPData(args.capture_dir, args.mtp_dir, args.target_state_features,
                           args.mtp_state_features, config, split_manifest=args.split_manifest)
    manifest = export_candidate_pool(data, args.checkpoint, args.output_dir,
                                     source_data_split=args.source_data_split,
                                     level2_split=args.level2_split,
                                     source_offline_split=args.source_offline_split,
                                     base_fold_id=args.base_fold_id,
                                     base_fit_excluded=args.base_fit_excluded,
                                     base_fit_split_manifest=args.base_fit_split_manifest,
                                     base_train_request_ids=args.base_train_request_ids,
                                     candidate_count=args.candidate_count,
                                     batch_size=args.batch_size, device=args.device,
                                     allow_test=args.allow_test, store_context=not args.no_context)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
