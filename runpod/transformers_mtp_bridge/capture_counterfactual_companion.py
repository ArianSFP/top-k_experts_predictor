#!/usr/bin/env python3
"""Write the label-only HARP-RTT v2 counterfactual target companion.

Path selection reads only the captured MTP tree.  Frozen-target execution occurs
strictly after selection and the resulting tensors are written to a separate,
hash-bound artifact which ordinary inference loaders cannot request.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for candidate in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from capture_transformers_segment import load_target, prefix_hash, sha256_file
from harp_rtt.counterfactual import (
    COUNTERFACTUAL_RECORD_SCHEMA,
    COUNTERFACTUAL_SCHEMA,
    empty_counterfactual_tensors,
    select_counterfactual_paths,
    selector_manifest,
    selector_sha256,
    validate_counterfactual_tensors,
)
from harp_rtt.static_artifacts import load_static_target_artifacts


EXPECTED_SPLIT_SHA256 = (
    "be03308606e3c6ae524f0268bbd8a8b3590a76332f42ae165c203758bafc970b"
)


@dataclass(frozen=True)
class FrozenTreeNode:
    local_index: int
    parent_local_index: int | None
    depth: int
    token_id: int
    token_rank_under_parent: int
    token_path_ids: tuple[int, ...]
    token_path_log_probabilities: tuple[float, ...]
    path_log_probability: float


class CounterfactualRouteHooks(AbstractContextManager):
    """Capture only target router inputs and native routing outputs."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.rows: dict[int, dict[str, torch.Tensor]] = {
            layer: {} for layer in range(len(model.model.layers))
        }
        self.handles: list[Any] = []

    def __enter__(self):
        for layer_id, layer in enumerate(self.model.model.layers):
            row = self.rows[layer_id]

            def norm_post(_module, _args, output, row=row):
                row["router_input"] = output.detach()

            def gate_post(_module, _args, output, row=row):
                logits, weights, ids = output
                row["router_logits"] = logits.detach()
                row["selected_weights"] = weights.detach()
                row["selected_ids"] = ids.detach()

            self.handles.extend(
                [
                    layer.post_attention_layernorm.register_forward_hook(norm_post),
                    layer.mlp.gate.register_forward_hook(gate_post),
                ]
            )
        return self

    def clear(self) -> None:
        for row in self.rows.values():
            row.clear()

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_base_capture(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[int]]]:
    manifest_path = root / "run_manifest.json"
    audit_path = root / "CAPTURE_AUDIT_ADAPTIVE.json"
    checksum_path = root / "SHA256SUMS"
    for path in (manifest_path, audit_path, checksum_path, root / "events.jsonl"):
        if not path.is_file():
            raise FileNotFoundError(f"base capture is incomplete: {path}")
    audit = load_json(audit_path)
    if audit.get("passed") is not True:
        raise ValueError("base adaptive capture audit did not pass")
    trees: dict[str, list[dict[str, Any]]] = {}
    endings: dict[str, list[int]] = {}
    starts: dict[str, dict[str, Any]] = {}
    for line in (root / "events.jsonl").read_text().splitlines():
        if not line:
            continue
        event = json.loads(line)
        if event.get("event") == "sequence_start":
            starts[str(event["sequence_id"])] = event
        elif event.get("event") == "mtp_node" and event.get("record_valid") is True:
            trees.setdefault(str(event["tree_id"]), []).append(event)
        elif event.get("event") == "sequence_end":
            endings[str(event["sequence_id"])] = [
                int(token) for token in event["full_committed_token_ids"]
            ]
    if not trees:
        raise ValueError("base capture contains no valid adaptive tree")
    ordered = []
    for tree_id in sorted(trees):
        rows = sorted(trees[tree_id], key=lambda row: int(row["node_local_index"]))
        if rows[0].get("exact_committed_h1_root") is not True:
            raise ValueError(f"tree {tree_id} lacks an exact H1 root")
        sequence_id = str(rows[0]["sequence_id"])
        start = starts.get(sequence_id)
        if start is None:
            raise ValueError(f"sequence {sequence_id} lacks a start event")
        ordered.append(
            {
                "tree_id": tree_id,
                "request_id": str(rows[0]["request_id"]),
                "sequence_id": sequence_id,
                "source_request_id": start.get("source_request_id"),
                "assigned_split": start.get("assigned_split"),
                "original_split": start.get("original_split"),
                "split_manifest_sha256": start.get("split_manifest_sha256"),
                "external_evaluation": start.get("external_evaluation"),
                "source_position": int(rows[0]["committed_prefix_position"]),
                "authoritative_prefix_hash": str(rows[0]["authoritative_prefix_hash"]),
                "rows": rows,
            }
        )
    return load_json(manifest_path), ordered, endings


def frozen_nodes(rows: list[dict[str, Any]]) -> list[FrozenTreeNode]:
    nodes = [
        FrozenTreeNode(
            local_index=int(row["node_local_index"]),
            parent_local_index=(
                None if row["parent_local_index"] is None
                else int(row["parent_local_index"])
            ),
            depth=int(row["depth"]),
            token_id=int(row["node_token_id"]),
            token_rank_under_parent=int(row["token_rank_under_parent"]),
            token_path_ids=tuple(int(token) for token in row["branch_path_token_ids"]),
            token_path_log_probabilities=tuple(
                float(value) for value in row["branch_path_token_log_probabilities"]
            ),
            path_log_probability=float(row["path_log_probability"]),
        )
        for row in rows
    ]
    return nodes


def capture_tree(
    *,
    target: Any,
    hooks: CounterfactualRouteHooks,
    geometry: Any,
    authoritative_prefix: list[int],
    nodes: list[FrozenTreeNode],
    audit_router_inputs: bool,
    execution_mode: str = "isolated_full_prefix_replay_reference",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    if execution_mode not in {
        "isolated_full_prefix_replay_reference",
        "sibling_isolated_cloned_prefix_cache",
    }:
        raise ValueError("unknown counterfactual target execution mode")
    selected = select_counterfactual_paths(nodes)
    tensors = empty_counterfactual_tensors(
        layers=geometry.layers,
        rank=geometry.maximum_rank,
        experts=geometry.experts,
    )
    audit = None
    if audit_router_inputs:
        audit = {
            "router_inputs": torch.zeros(
                4, 4, geometry.layers, geometry.hidden_width, dtype=torch.bfloat16
            ),
            "valid": torch.zeros(4, 4, geometry.layers, dtype=torch.bool),
        }
    prefix_length = len(authoritative_prefix)
    prefix_output = None
    prefix_cache = None
    prefix_token_logp = None
    if execution_mode == "sibling_isolated_cloned_prefix_cache":
        hooks.clear()
        prefix_output = target(
            input_ids=torch.tensor(
                [authoritative_prefix], dtype=torch.long, device=target.device
            ),
            use_cache=True,
            output_hidden_states=False,
            output_router_logits=True,
            return_dict=True,
        )
        prefix_cache = prefix_output.past_key_values
        prefix_token_logp = torch.log_softmax(
            prefix_output.logits[0, -1].float(), dim=-1
        )

    for slot, path in enumerate(selected):
        if path is None:
            continue
        tensors["path_mask"][slot] = True
        tensors["path_depths"][slot] = path.path_depth
        tensors["first_divergence_depths"][slot] = (
            -1 if path.first_divergence_depth is None
            else path.first_divergence_depth
        )
        tensors["node_local_indices"][slot] = torch.tensor(
            path.node_local_indices, dtype=torch.int64
        )
        for depth, node_index in enumerate(path.node_local_indices, start=1):
            if node_index >= 0:
                tensors["source_path_logp"][slot, depth - 1] = (
                    nodes[node_index].path_log_probability
                )

        hooks.clear()
        if execution_mode == "sibling_isolated_cloned_prefix_cache":
            import copy

            call_tokens = list(path.token_ids)
            output = target(
                input_ids=torch.tensor(
                    [call_tokens], dtype=torch.long, device=target.device
                ),
                past_key_values=copy.deepcopy(prefix_cache),
                use_cache=False,
                output_hidden_states=False,
                output_router_logits=True,
                return_dict=True,
            )
        else:
            call_tokens = authoritative_prefix + list(path.token_ids)
            output = target(
                input_ids=torch.tensor(
                    [call_tokens], dtype=torch.long, device=target.device
                ),
                use_cache=False,
                output_hidden_states=False,
                output_router_logits=True,
                return_dict=True,
            )
        token_logp = torch.log_softmax(output.logits[0].float(), dim=-1)
        cumulative = 0.0
        for depth, token_id in enumerate(path.token_ids, start=1):
            if execution_mode == "sibling_isolated_cloned_prefix_cache":
                edge = float(
                    (
                        prefix_token_logp[token_id]
                        if depth == 1
                        else token_logp[depth - 2, token_id]
                    ).item()
                )
                route_index = depth - 1
            else:
                prediction_index = prefix_length + depth - 2
                edge = float(token_logp[prediction_index, token_id].item())
                route_index = prefix_length + depth - 1
            cumulative += edge
            tensors["target_edge_logp"][slot, depth - 1] = edge
            tensors["target_path_logp"][slot, depth - 1] = cumulative
            if depth == 1:
                continue
            router_inputs = torch.stack(
                [
                    hooks.rows[layer]["router_input"][0, route_index]
                    for layer in range(geometry.layers)
                ]
            )
            logits = torch.stack(
                [
                    hooks.rows[layer]["router_logits"][route_index]
                    for layer in range(geometry.layers)
                ]
            )
            selected_ids = torch.stack(
                [
                    hooks.rows[layer]["selected_ids"][route_index]
                    for layer in range(geometry.layers)
                ]
            )
            selected_weights = torch.stack(
                [
                    hooks.rows[layer]["selected_weights"][route_index]
                    for layer in range(geometry.layers)
                ]
            )
            coordinates = geometry.encode_router_inputs(router_inputs)
            tensors["query_coordinates"][slot, depth - 1] = coordinates.cpu()
            tensors["router_logits"][slot, depth - 1] = logits.to(
                torch.bfloat16
            ).cpu()
            tensors["selected_ids"][slot, depth - 1] = selected_ids.to(
                torch.int32
            ).cpu()
            tensors["selected_weights"][slot, depth - 1] = selected_weights.to(
                torch.bfloat16
            ).cpu()
            tensors["valid"][slot, depth - 1] = True
            if audit is not None:
                audit["router_inputs"][slot, depth - 1] = router_inputs.to(
                    torch.bfloat16
                ).cpu()
                audit["valid"][slot, depth - 1] = True
        del output
    del prefix_output
    validate_counterfactual_tensors(
        tensors,
        layers=geometry.layers,
        rank=geometry.maximum_rank,
        experts=geometry.experts,
    )
    return tensors, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--base-capture", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--expected-split-sha256", default=EXPECTED_SPLIT_SHA256
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--execution-mode",
        choices=(
            "isolated_full_prefix_replay_reference",
            "sibling_isolated_cloned_prefix_cache",
        ),
        default="isolated_full_prefix_replay_reference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable companion {args.output}")
    if sha256_file(args.split_manifest) != args.expected_split_sha256:
        raise ValueError("split manifest hash does not match the frozen v1 manifest")
    if len(args.source_commit) != 40 or any(
        value not in "0123456789abcdef" for value in args.source_commit.lower()
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    base_manifest, trees, sequence_tokens = load_base_capture(args.base_capture)
    if args.limit is not None:
        trees = trees[: args.limit]
    if not trees:
        raise ValueError("counterfactual selection contains no source tree")
    train_request_ids = {
        str(row["request_id"])
        for line in args.split_manifest.read_text().splitlines()
        if line
        for row in (json.loads(line),)
        if row.get("split") == "train"
    }
    for tree in trees:
        if (
            tree.get("assigned_split") != "train"
            or tree.get("original_split") != "train"
            or tree.get("external_evaluation") is not False
            or tree.get("split_manifest_sha256") != args.expected_split_sha256
            or tree.get("source_request_id") not in train_request_ids
        ):
            raise PermissionError(
                "counterfactual replay refuses a tree not bound to outer-train"
            )

    args.output.mkdir(parents=True)
    records_dir = args.output / "records"
    audit_dir = args.output / "router_input_audits"
    records_dir.mkdir()
    audit_dir.mkdir()

    static = load_static_target_artifacts(
        args.static_dir, device=args.device, load_embedding=False
    )
    geometry = static.geometry
    target, _ = load_target(args.model, device=args.device)
    record_manifest = []
    with torch.inference_mode(), CounterfactualRouteHooks(target) as hooks:
        for ordinal, tree in enumerate(trees):
            tokens = sequence_tokens.get(tree["sequence_id"])
            if tokens is None:
                raise ValueError(f"sequence {tree['sequence_id']} has no ending")
            source_position = tree["source_position"]
            authoritative_prefix = tokens[: source_position + 1]
            if prefix_hash(authoritative_prefix) != tree["authoritative_prefix_hash"]:
                raise ValueError("authoritative prefix hash disagrees with base capture")
            nodes = frozen_nodes(tree["rows"])
            audit_sample = (
                int(hashlib.sha256(tree["tree_id"].encode()).hexdigest()[:8], 16)
                % 20 == 0
            )
            tensors, router_audit = capture_tree(
                target=target,
                hooks=hooks,
                geometry=geometry,
                authoritative_prefix=authoritative_prefix,
                nodes=nodes,
                audit_router_inputs=audit_sample,
                execution_mode=args.execution_mode,
            )
            stem = f"source_{ordinal:06d}"
            record_path = records_dir / f"{stem}.safetensors"
            save_file(
                {name: value.contiguous() for name, value in tensors.items()},
                record_path,
                metadata={
                    "schema": COUNTERFACTUAL_RECORD_SCHEMA,
                    "label_only": "true",
                    "runtime_available": "false",
                    "tree_id": tree["tree_id"],
                },
            )
            audit_record = None
            if router_audit is not None:
                audit_path = audit_dir / f"{stem}.safetensors"
                save_file(
                    {name: value.contiguous() for name, value in router_audit.items()},
                    audit_path,
                    metadata={
                        "schema": "harp_rtt_counterfactual_router_input_audit_v2",
                        "label_only": "true",
                        "runtime_available": "false",
                    },
                )
                audit_record = {
                    "path": str(audit_path.relative_to(args.output)),
                    "sha256": sha256_file(audit_path),
                }
            record_manifest.append(
                {
                    "ordinal": ordinal,
                    "request_id": tree["request_id"],
                    "source_request_id": tree["source_request_id"],
                    "sequence_id": tree["sequence_id"],
                    "tree_id": tree["tree_id"],
                    "source_position": source_position,
                    "record": {
                        "path": str(record_path.relative_to(args.output)),
                        "sha256": sha256_file(record_path),
                    },
                    "router_input_audit": audit_record,
                }
            )

    manifest = {
        "schema": COUNTERFACTUAL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label_only": True,
        "runtime_available": False,
        "split": "train",
        "sealed_test_opened": False,
        "training_started": False,
        "execution_mode": args.execution_mode,
        "selector": selector_manifest(),
        "bindings": {
            "source_commit": args.source_commit,
            "selector_sha256": selector_sha256(),
            "base_capture_manifest_sha256": sha256_file(
                args.base_capture / "run_manifest.json"
            ),
            "base_capture_audit_sha256": sha256_file(
                args.base_capture / "CAPTURE_AUDIT_ADAPTIVE.json"
            ),
            "base_capture_checksums_sha256": sha256_file(
                args.base_capture / "SHA256SUMS"
            ),
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "static_manifest_sha256": sha256_file(args.static_dir / "manifest.json"),
            "router_geometry_sha256": sha256_file(
                args.static_dir / "router_geometry.safetensors"
            ),
            "model_config_sha256": base_manifest.get("model_config_hash"),
            "model_identity_hashes": base_manifest.get("identity_file_hashes"),
            "model_revision": base_manifest.get("model_revision"),
        },
        "tensor_contract": {
            "path_slots": 4,
            "depths": 4,
            "layers": geometry.layers,
            "rank": geometry.maximum_rank,
            "experts": geometry.experts,
            "top_k": 8,
            "h1_masked": True,
        },
        "records": record_manifest,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    files = [
        path for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name not in {"SHA256SUMS", "STOP_BEFORE_TRAINING.json"}
    ]
    with (args.output / "SHA256SUMS").open("w") as handle:
        for path in files:
            handle.write(
                f"{sha256_file(path)}  {path.relative_to(args.output)}\n"
            )
    (args.output / "STOP_BEFORE_TRAINING.json").write_text(
        json.dumps(
            {
                "schema": "harp_rtt_counterfactual_stop_v2",
                "capture_complete": True,
                "audit_complete": False,
                "training_started": False,
                "sealed_test_opened": False,
                "instruction": "Run the companion blocking auditor; do not train.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(record_manifest),
                "router_input_audit_records": sum(
                    row["router_input_audit"] is not None for row in record_manifest
                ),
                "label_only": True,
                "training_started": False,
                "sealed_test_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
