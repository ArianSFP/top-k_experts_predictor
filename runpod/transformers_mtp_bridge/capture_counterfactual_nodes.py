#!/usr/bin/env python3
"""Capture every unique adaptive-32 target-route node once for B1.5."""

from __future__ import annotations

import argparse
import copy
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

from capture_counterfactual_companion import (  # noqa: E402
    CounterfactualRouteHooks,
    FrozenTreeNode,
    frozen_nodes,
    load_base_capture,
)
from capture_transformers_segment import (  # noqa: E402
    load_target,
    prefix_hash,
    sha256_file,
)
from harp_rtt.b15 import (  # noqa: E402
    b15_selector_manifest,
    b15_selector_sha256,
    select_path_budget,
)
from harp_rtt.node_counterfactual import (  # noqa: E402
    MAX_TREE_NODES,
    NODE_COUNTERFACTUAL_RECORD_SCHEMA,
    NODE_COUNTERFACTUAL_SCHEMA,
    NODE_ROUTER_AUDIT_SCHEMA,
    empty_node_counterfactual_tensors,
    validate_node_counterfactual_tensors,
)
from harp_rtt.static_artifacts import load_static_target_artifacts  # noqa: E402


EXPECTED_SPLIT_SHA256 = (
    "be03308606e3c6ae524f0268bbd8a8b3590a76332f42ae165c203758bafc970b"
)
EXECUTION_MODES = (
    "isolated_full_prefix_replay_reference",
    "sibling_isolated_cloned_parent_cache",
)


def _greedy_tokens(nodes: list[FrozenTreeNode]) -> tuple[int, ...]:
    tokens = [nodes[0].token_id]
    parent = 0
    for depth in range(2, 5):
        children = [
            node
            for node in nodes
            if node.parent_local_index == parent
            and node.depth == depth
            and node.token_rank_under_parent == 0
        ]
        if not children:
            break
        chosen = min(children, key=lambda node: (node.token_id, node.local_index))
        tokens.append(chosen.token_id)
        parent = chosen.local_index
    return tuple(tokens)


def _first_divergence(path: tuple[int, ...], greedy: tuple[int, ...]) -> int:
    for depth, (token, reference) in enumerate(zip(path, greedy), start=1):
        if token != reference:
            return depth
    return -1


def _capture_route(
    tensors: dict[str, torch.Tensor],
    audit: dict[str, torch.Tensor] | None,
    *,
    node: int,
    hooks: CounterfactualRouteHooks,
    geometry: Any,
) -> None:
    router_inputs = torch.stack(
        [hooks.rows[layer]["router_input"][0, 0] for layer in range(geometry.layers)]
    )
    logits = torch.stack(
        [hooks.rows[layer]["router_logits"][0] for layer in range(geometry.layers)]
    )
    selected_ids = torch.stack(
        [hooks.rows[layer]["selected_ids"][0] for layer in range(geometry.layers)]
    )
    selected_weights = torch.stack(
        [hooks.rows[layer]["selected_weights"][0] for layer in range(geometry.layers)]
    )
    tensors["query_coordinates"][node] = geometry.encode_router_inputs(
        router_inputs
    ).cpu()
    tensors["router_logits"][node] = logits.to(torch.bfloat16).cpu()
    tensors["selected_ids"][node] = selected_ids.to(torch.int32).cpu()
    tensors["selected_weights"][node] = selected_weights.to(torch.bfloat16).cpu()
    tensors["valid"][node] = True
    if audit is not None:
        audit["router_inputs"][node] = router_inputs.to(torch.bfloat16).cpu()
        audit["valid"][node] = True


def capture_node_tree(
    *,
    target: Any,
    hooks: CounterfactualRouteHooks,
    geometry: Any,
    authoritative_prefix: list[int],
    nodes: list[FrozenTreeNode],
    audit_router_inputs: bool,
    execution_mode: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    if execution_mode not in EXECUTION_MODES:
        raise ValueError("unknown node counterfactual execution mode")
    if not nodes or len(nodes) > MAX_TREE_NODES:
        raise ValueError("node companion requires one adaptive tree of at most 32 nodes")
    tensors = empty_node_counterfactual_tensors(
        nodes=MAX_TREE_NODES,
        layers=geometry.layers,
        rank=geometry.maximum_rank,
        experts=geometry.experts,
    )
    audit = None
    if audit_router_inputs:
        audit = {
            "router_inputs": torch.zeros(
                MAX_TREE_NODES,
                geometry.layers,
                geometry.hidden_width,
                dtype=torch.bfloat16,
            ),
            "valid": torch.zeros(
                MAX_TREE_NODES, geometry.layers, dtype=torch.bool
            ),
        }
    greedy = _greedy_tokens(nodes)
    for node in nodes:
        index = node.local_index
        tensors["node_mask"][index] = True
        tensors["node_local_indices"][index] = index
        tensors["parent_local_indices"][index] = (
            -1 if node.parent_local_index is None else node.parent_local_index
        )
        tensors["depth"][index] = node.depth
        tensors["first_divergence_depth"][index] = _first_divergence(
            node.token_path_ids, greedy
        )
        tensors["source_edge_logp"][index] = (
            0.0 if index == 0 else node.token_path_log_probabilities[-1]
        )
        tensors["source_path_logp"][index] = (
            0.0 if index == 0 else node.path_log_probability
        )
    for budget_index, budget in enumerate((4, 8, 16)):
        selection = select_path_budget(nodes, budget)
        tensors["budget_node_masks"][budget_index, : len(nodes)] = torch.tensor(
            selection.node_mask, dtype=torch.bool
        )
        tensors["budget_endpoint_masks"][
            budget_index, list(selection.endpoint_local_indices)
        ] = True
        tensors["budget_realized"][budget_index] = selection.realized_budget
        tensors["budget_category_counts"][budget_index] = torch.tensor(
            selection.category_counts, dtype=torch.int64
        )

    def replay_prefix() -> tuple[Any, torch.Tensor, Any]:
        hooks.clear()
        output = target(
            input_ids=torch.tensor(
                [authoritative_prefix], dtype=torch.long, device=target.device
            ),
            use_cache=True,
            output_hidden_states=False,
            output_router_logits=True,
            return_dict=True,
        )
        return (
            output.past_key_values,
            torch.log_softmax(output.logits[0, -1].float(), dim=-1),
            output,
        )

    if execution_mode == "isolated_full_prefix_replay_reference":
        for node in nodes[1:]:
            cache, previous_logp, prefix_output = replay_prefix()
            cumulative = 0.0
            output = None
            for depth, token_id in enumerate(node.token_path_ids, start=1):
                edge = float(previous_logp[token_id].item())
                if depth > 1:
                    cumulative += edge
                hooks.clear()
                output = target(
                    input_ids=torch.tensor(
                        [[token_id]], dtype=torch.long, device=target.device
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=False,
                    output_router_logits=True,
                    return_dict=True,
                )
                cache = output.past_key_values
                previous_logp = torch.log_softmax(
                    output.logits[0, -1].float(), dim=-1
                )
                if depth == node.depth:
                    tensors["target_edge_logp"][node.local_index] = edge
                    tensors["target_path_logp"][node.local_index] = cumulative
                    _capture_route(
                        tensors,
                        audit,
                        node=node.local_index,
                        hooks=hooks,
                        geometry=geometry,
                    )
            del output, prefix_output
    else:
        prefix_cache, prefix_logp, prefix_output = replay_prefix()
        root = nodes[0]
        hooks.clear()
        root_output = target(
            input_ids=torch.tensor(
                [[root.token_id]], dtype=torch.long, device=target.device
            ),
            past_key_values=prefix_cache,
            use_cache=True,
            output_hidden_states=False,
            output_router_logits=True,
            return_dict=True,
        )
        root_cache = root_output.past_key_values
        root_logp = torch.log_softmax(root_output.logits[0, -1].float(), dim=-1)
        children: dict[int, list[int]] = {node.local_index: [] for node in nodes}
        for node in nodes[1:]:
            assert node.parent_local_index is not None
            children[node.parent_local_index].append(node.local_index)
        for values in children.values():
            values.sort()

        def visit(parent: int, parent_cache: Any, parent_logp: torch.Tensor, cumulative: float) -> None:
            for child_id in children[parent]:
                child = nodes[child_id]
                edge = float(parent_logp[child.token_id].item())
                child_cumulative = cumulative + edge
                hooks.clear()
                output = target(
                    input_ids=torch.tensor(
                        [[child.token_id]], dtype=torch.long, device=target.device
                    ),
                    past_key_values=copy.deepcopy(parent_cache),
                    use_cache=True,
                    output_hidden_states=False,
                    output_router_logits=True,
                    return_dict=True,
                )
                child_cache = output.past_key_values
                child_logp = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
                tensors["target_edge_logp"][child_id] = edge
                tensors["target_path_logp"][child_id] = child_cumulative
                _capture_route(
                    tensors,
                    audit,
                    node=child_id,
                    hooks=hooks,
                    geometry=geometry,
                )
                visit(child_id, child_cache, child_logp, child_cumulative)
                del output, child_cache

        visit(0, root_cache, root_logp, 0.0)
        del prefix_output, root_output, prefix_cache, root_cache, prefix_logp

    validate_node_counterfactual_tensors(
        tensors,
        nodes=MAX_TREE_NODES,
        layers=geometry.layers,
        rank=geometry.maximum_rank,
        experts=geometry.experts,
    )
    return tensors, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--base-capture", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-split-sha256", default=EXPECTED_SPLIT_SHA256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default=EXECUTION_MODES[0])
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
        raise ValueError("node counterfactual capture contains no source tree")
    train_ids = {
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
            or tree.get("source_request_id") not in train_ids
        ):
            raise PermissionError("node replay refuses a tree outside outer-train")

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
    records = []
    with torch.inference_mode(), CounterfactualRouteHooks(target) as hooks:
        for ordinal, tree in enumerate(trees):
            tokens = sequence_tokens.get(tree["sequence_id"])
            if tokens is None:
                raise ValueError(f"sequence {tree['sequence_id']} has no ending")
            source_position = int(tree["source_position"])
            authoritative_prefix = tokens[: source_position + 1]
            if prefix_hash(authoritative_prefix) != tree["authoritative_prefix_hash"]:
                raise ValueError("authoritative prefix hash disagrees with base capture")
            nodes = frozen_nodes(tree["rows"])
            audit_sample = (
                int(hashlib.sha256(tree["tree_id"].encode()).hexdigest()[:8], 16)
                % 20
                == 0
            )
            tensors, router_audit = capture_node_tree(
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
                    "schema": NODE_COUNTERFACTUAL_RECORD_SCHEMA,
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
                        "schema": NODE_ROUTER_AUDIT_SCHEMA,
                        "label_only": "true",
                        "runtime_available": "false",
                    },
                )
                audit_record = {
                    "path": str(audit_path.relative_to(args.output)),
                    "sha256": sha256_file(audit_path),
                }
            records.append(
                {
                    "ordinal": ordinal,
                    "request_id": tree["request_id"],
                    "source_request_id": tree["source_request_id"],
                    "sequence_id": tree["sequence_id"],
                    "tree_id": tree["tree_id"],
                    "source_position": source_position,
                    "node_prefix_hashes": [
                        prefix_hash(authoritative_prefix + list(node.token_path_ids))
                        for node in nodes
                    ],
                    "record": {
                        "path": str(record_path.relative_to(args.output)),
                        "sha256": sha256_file(record_path),
                    },
                    "router_input_audit": audit_record,
                }
            )

    manifest = {
        "schema": NODE_COUNTERFACTUAL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label_only": True,
        "runtime_available": False,
        "split": "train",
        "sealed_test_opened": False,
        "training_started": False,
        "execution_mode": args.execution_mode,
        "selector": b15_selector_manifest(),
        "bindings": {
            "source_commit": args.source_commit,
            "selector_sha256": b15_selector_sha256(),
            "base_capture_manifest_sha256": sha256_file(args.base_capture / "run_manifest.json"),
            "base_capture_audit_sha256": sha256_file(args.base_capture / "CAPTURE_AUDIT_ADAPTIVE.json"),
            "base_capture_checksums_sha256": sha256_file(args.base_capture / "SHA256SUMS"),
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "static_manifest_sha256": sha256_file(args.static_dir / "manifest.json"),
            "router_geometry_sha256": sha256_file(args.static_dir / "router_geometry.safetensors"),
            "model_config_sha256": base_manifest.get("model_config_hash"),
            "model_identity_hashes": base_manifest.get("identity_file_hashes"),
            "model_revision": base_manifest.get("model_revision"),
        },
        "tensor_contract": {
            "layout": "unique_parent_before_child_nodes",
            "max_nodes": MAX_TREE_NODES,
            "depths": 4,
            "layers": geometry.layers,
            "rank": geometry.maximum_rank,
            "experts": geometry.experts,
            "top_k": 8,
            "h1_masked": True,
            "target_path_probability_condition": "exact_committed_h1",
            "target_path_probability_origin": "h2_edge",
        },
        "records": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    files = [
        path
        for path in sorted(args.output.rglob("*"))
        if path.is_file()
        and path.name not in {"SHA256SUMS", "STOP_BEFORE_TRAINING.json"}
    ]
    with (args.output / "SHA256SUMS").open("w") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(args.output)}\n")
    (args.output / "STOP_BEFORE_TRAINING.json").write_text(
        json.dumps(
            {
                "schema": "harp_rtt_counterfactual_nodes_stop_v3",
                "capture_complete": True,
                "audit_complete": False,
                "training_started": False,
                "sealed_test_opened": False,
                "instruction": "Run the node companion blocking auditor; do not train.",
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
                "records": len(records),
                "router_input_audit_records": sum(
                    row["router_input_audit"] is not None for row in records
                ),
                "label_only": True,
                "training_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
