"""HARP-RTT B1.5 information-ceiling primitives.

This module is deliberately optimizer-free.  It contains only causal path
selection, exact selected-set oracle aggregation, candidate construction, and
the dormant H1 root-supervision contract used by the B1.5 gate evaluator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .counterfactual import (
    MAX_DEPTH,
    TOP_K,
    TreeNodeLike,
    select_counterfactual_paths,
)
from .exact_k import exact_set_nll
from .model.heads import exact_projected_marginals


B15_SELECTOR_SCHEMA = "harp_rtt_b15_causal_path_budget_selector_v2_nested"
B15_CANDIDATE_UNION_SCHEMA = "harp_rtt_b15_positive_branch_candidate_union_v1"
B15_ORACLE_SCHEMA = "harp_rtt_b15_exact_selected_set_oracle_v1"
PATH_BUDGETS = (4, 8, 16)
PATH_QUOTAS = {
    8: (1, 2, 2, 3),
    16: (1, 3, 5, 7),
}
H1_C64_GATE = 0.98


def required_b15_free_bytes(estimated_total_bytes: int) -> int:
    if estimated_total_bytes < 0:
        raise ValueError("estimated B1.5 storage must be non-negative")
    return max(100 * (1 << 30), (5 * int(estimated_total_bytes) + 3) // 4)


@dataclass(frozen=True)
class BudgetSelection:
    requested_budget: int | str
    endpoint_local_indices: tuple[int, ...]
    node_mask: tuple[bool, ...]
    category_counts: tuple[int, int, int, int]
    realized_budget: int
    unique_node_count: int


@dataclass(frozen=True)
class H1RootLoss:
    total: Tensor
    exact_set: Tensor
    query: Tensor
    router_kl: Tensor


def b15_selector_manifest() -> dict[str, Any]:
    return {
        "schema": B15_SELECTOR_SCHEMA,
        "budgets": {
            "4": "bit-for-bit legacy divergence-depth selector",
            "8": list(PATH_QUOTAS[8]),
            "16": list(PATH_QUOTAS[16]),
            "all": "all valid adaptive-32 H2-H4 nodes",
        },
        "categories": ["greedy", "first_divergence_h2", "first_divergence_h3", "first_divergence_h4"],
        "underfill": (
            "seed each larger budget from the previous budget; pool unfilled quota "
            "slots; require one new H2-H4 node; then order "
            "by deeper endpoint, descending cumulative MTP probability, "
            "lexicographic token path, local index"
        ),
        "nested_endpoints": "E4 subset E8 subset E16 subset all H2-H4 nodes",
        "nested_nodes": "N4 subset N8 subset N16 subset all nodes",
        "uses_target_labels": False,
        "uses_target_probabilities": False,
        "uses_acceptance": False,
        "uses_factual_continuation": False,
    }


def b15_selector_sha256() -> str:
    payload = json.dumps(
        b15_selector_manifest(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def b15_candidate_union_manifest() -> dict[str, Any]:
    return {
        "schema": B15_CANDIDATE_UNION_SCHEMA,
        "steps": [
            "insert stable anchor top-K_A",
            (
                "insert distinct strictly-positive branch-mass experts in stable "
                "descending order"
            ),
            "fill unoccupied positions from the next stable anchor experts",
        ],
        "zero_branch_invariant": (
            "all-zero branch mass exactly reproduces anchor top-width"
        ),
        "ties": (
            "descending stable argsort; lower expert ID wins exact input-order ties"
        ),
    }


def b15_candidate_union_sha256() -> str:
    payload = json.dumps(
        b15_candidate_union_manifest(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_tree(nodes: Sequence[TreeNodeLike]) -> None:
    if not nodes:
        raise ValueError("B1.5 selector received an empty tree")
    if len(nodes) > 32:
        raise ValueError("B1.5 adaptive-32 selector received more than 32 nodes")
    for expected, node in enumerate(nodes):
        if int(node.local_index) != expected:
            raise ValueError("tree local indices must be contiguous")
        if not 1 <= int(node.depth) <= MAX_DEPTH:
            raise ValueError("B1.5 selector accepts only H1-H4 nodes")
        if expected == 0:
            if int(node.depth) != 1 or node.parent_local_index is not None:
                raise ValueError("node zero must be the exact H1 root")
        else:
            parent = node.parent_local_index
            if parent is None or not 0 <= int(parent) < expected:
                raise ValueError("each parent must precede its child")


def _greedy_tokens(nodes: Sequence[TreeNodeLike]) -> tuple[int, ...]:
    tokens = [int(nodes[0].token_id)]
    parent = 0
    for depth in range(2, MAX_DEPTH + 1):
        candidates = [
            node
            for node in nodes
            if node.parent_local_index == parent
            and int(node.depth) == depth
            and int(node.token_rank_under_parent) == 0
        ]
        if not candidates:
            break
        chosen = min(candidates, key=lambda node: (int(node.token_id), int(node.local_index)))
        tokens.append(int(chosen.token_id))
        parent = int(chosen.local_index)
    return tuple(tokens)


def _first_divergence(
    path: tuple[int, ...], greedy: tuple[int, ...]
) -> int | None:
    for depth, (token, reference) in enumerate(zip(path, greedy), start=1):
        if token != reference:
            return depth
    return None


def _ancestors(nodes: Sequence[TreeNodeLike], endpoint: int) -> tuple[int, ...]:
    result = []
    current: int | None = endpoint
    while current is not None:
        result.append(current)
        parent = nodes[current].parent_local_index
        current = None if parent is None else int(parent)
    return tuple(reversed(result))


def _endpoint_key(node: TreeNodeLike) -> tuple[Any, ...]:
    return (
        -int(node.depth),
        -float(node.path_log_probability),
        tuple(int(value) for value in node.token_path_ids),
        int(node.local_index),
    )


def select_path_budget(
    nodes: Sequence[TreeNodeLike], budget: int | str
) -> BudgetSelection:
    """Return a causal endpoint selection and its ancestor-closed node mask."""

    _validate_tree(nodes)
    greedy = _greedy_tokens(nodes)
    categories = (None, 2, 3, 4)

    if budget == "all":
        endpoints = tuple(
            int(node.local_index)
            for node in nodes
            if int(node.depth) >= 2
        )
        counts = [0, 0, 0, 0]
        for endpoint in endpoints:
            divergence = _first_divergence(
                tuple(int(v) for v in nodes[endpoint].token_path_ids), greedy
            )
            counts[0 if divergence is None else divergence - 1] += 1
        return BudgetSelection(
            requested_budget="all",
            endpoint_local_indices=endpoints,
            node_mask=tuple(True for _ in nodes),
            category_counts=tuple(counts),  # type: ignore[arg-type]
            realized_budget=len(endpoints),
            unique_node_count=len(nodes),
        )

    if not isinstance(budget, int) or budget not in PATH_BUDGETS:
        raise ValueError("path budget must be one of 4, 8, 16, or 'all'")

    if budget == 4:
        selected = select_counterfactual_paths(nodes)
        endpoints = tuple(
            int(path.endpoint_local_index) for path in selected if path is not None
        )
        selected_nodes = {
            node
            for endpoint in endpoints
            for node in _ancestors(nodes, endpoint)
        }
        counts = tuple(int(path is not None) for path in selected)
        return BudgetSelection(
            requested_budget=4,
            endpoint_local_indices=endpoints,
            node_mask=tuple(index in selected_nodes for index in range(len(nodes))),
            category_counts=counts,  # type: ignore[arg-type]
            realized_budget=len(endpoints),
            unique_node_count=len(selected_nodes),
        )

    quota = PATH_QUOTAS[budget]
    categorized: dict[int | None, list[TreeNodeLike]] = {key: [] for key in categories}
    for node in nodes:
        if int(node.depth) < 2:
            continue
        divergence = _first_divergence(
            tuple(int(value) for value in node.token_path_ids), greedy
        )
        if divergence in categorized:
            categorized[divergence].append(node)
    for values in categorized.values():
        values.sort(key=_endpoint_key)

    seed = select_path_budget(nodes, 4 if budget == 8 else 8)
    endpoints = list(seed.endpoint_local_indices)
    selected_nodes = {
        index for index, selected in enumerate(seed.node_mask) if selected
    }
    counts = list(seed.category_counts)

    def add(node: TreeNodeLike) -> bool:
        endpoint = int(node.local_index)
        if endpoint in endpoints:
            return False
        ancestors = _ancestors(nodes, endpoint)
        if not any(int(nodes[index].depth) >= 2 and index not in selected_nodes for index in ancestors):
            return False
        endpoints.append(endpoint)
        selected_nodes.update(ancestors)
        divergence = _first_divergence(
            tuple(int(value) for value in node.token_path_ids), greedy
        )
        counts[0 if divergence is None else divergence - 1] += 1
        return True

    for category_index, (category, required) in enumerate(
        zip(categories, quota, strict=True)
    ):
        if counts[category_index] >= required:
            continue
        for node in categorized[category]:
            add(node)
            if counts[category_index] >= required:
                break

    remaining = sorted(
        (node for node in nodes if int(node.depth) >= 2 and int(node.local_index) not in endpoints),
        key=_endpoint_key,
    )
    for node in remaining:
        if len(endpoints) >= budget:
            break
        add(node)

    result = BudgetSelection(
        requested_budget=budget,
        endpoint_local_indices=tuple(endpoints),
        node_mask=tuple(index in selected_nodes for index in range(len(nodes))),
        category_counts=tuple(counts),  # type: ignore[arg-type]
        realized_budget=len(endpoints),
        unique_node_count=len(selected_nodes),
    )
    if not set(seed.endpoint_local_indices).issubset(result.endpoint_local_indices):
        raise AssertionError("nested endpoint budget invariant failed")
    if any(
        previous and not current
        for previous, current in zip(seed.node_mask, result.node_mask, strict=True)
    ):
        raise AssertionError(
            "nested ancestor-closed node budget invariant failed"
        )
    return result


def selected_set_inclusion_mass(
    selected_ids: Tensor,
    probabilities: Tensor,
    valid: Tensor,
    *,
    experts: int,
) -> tuple[Tensor, Tensor]:
    """Aggregate exact selected-set membership into Bayes inclusion mass.

    ``selected_ids`` is ``[N,L,K]``, ``probabilities`` is ``[N]``, and
    ``valid`` is ``[N,L]``.  The result is ``[L,E]`` plus captured mass per
    layer.  Rows are assumed to identify unique tree nodes.
    """

    if selected_ids.ndim != 3 or valid.shape != selected_ids.shape[:2]:
        raise ValueError("selected-set oracle expects [N,L,K] IDs and [N,L] validity")
    if probabilities.shape != (selected_ids.shape[0],):
        raise ValueError("selected-set probabilities disagree with node dimension")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("selected-set probabilities must be finite and non-negative")
    ids = selected_ids.long()
    active_ids = ids[valid]
    if active_ids.numel() and ((active_ids < 0) | (active_ids >= experts)).any():
        raise ValueError("selected expert ID is outside the oracle namespace")
    if active_ids.numel() and any(
        torch.unique(row).numel() != row.numel() for row in active_ids
    ):
        raise ValueError("authoritative selected sets must not contain duplicates")
    layers = selected_ids.shape[1]
    mass = probabilities[:, None] * valid.to(probabilities.dtype)
    captured = mass.sum(dim=0)
    if (captured > 1.0 + 2e-5).any():
        raise ValueError("captured path probability mass exceeds one")
    result = probabilities.new_zeros((layers, experts))
    source = mass[..., None].expand_as(ids).reshape(layers, -1)
    index = ids.clamp_min(0).permute(1, 0, 2).reshape(layers, -1)
    source = source.reshape(selected_ids.shape[0], layers, -1).permute(1, 0, 2).reshape(layers, -1)
    active = valid[..., None].expand_as(ids).permute(1, 0, 2).reshape(layers, -1)
    result.scatter_add_(1, index, source * active.to(source.dtype))
    return result, captured.clamp(max=1.0)


def add_other_anchor_mass(
    branch_mass: Tensor,
    captured_mass: Tensor,
    anchor_scores: Tensor,
    *,
    exact_k: int = TOP_K,
) -> tuple[Tensor, Tensor, Tensor]:
    if branch_mass.shape != anchor_scores.shape:
        raise ValueError("branch and anchor expert geometry disagree")
    if captured_mass.shape != branch_mass.shape[:-1]:
        raise ValueError("captured mass disagrees with branch geometry")
    if (captured_mass > 1.0 + 2e-5).any() or (captured_mass < 0).any():
        raise ValueError("captured mass lies outside [0,1]")
    _, anchor_marginals, cardinality_error = exact_projected_marginals(
        anchor_scores.float(), exact_k
    )
    other = (1.0 - captured_mass).clamp_min(0.0)
    full = branch_mass + other[..., None] * anchor_marginals
    return full, other, cardinality_error


def candidate_union(
    anchor_scores: Tensor,
    branch_mass: Tensor,
    *,
    anchor_quota: int,
    width: int = 64,
) -> Tensor:
    if anchor_scores.shape != branch_mass.shape:
        raise ValueError("anchor and branch candidate sources disagree")
    experts = anchor_scores.shape[-1]
    if not 0 <= anchor_quota <= width <= experts:
        raise ValueError("invalid candidate width/quota")
    leading = anchor_scores.shape[:-1]
    rows = math.prod(leading)
    anchor_order = torch.argsort(
        anchor_scores.float(), dim=-1, descending=True, stable=True
    ).reshape(rows, experts)
    branch_order = torch.argsort(
        branch_mass.float(), dim=-1, descending=True, stable=True
    ).reshape(rows, experts)
    branch_values = branch_mass.float().reshape(rows, experts)
    if not torch.isfinite(branch_values).all() or (branch_values < 0).any():
        raise ValueError("branch candidate mass must be finite and non-negative")
    result = torch.full((rows, width), -1, dtype=torch.long, device=anchor_scores.device)
    selected = torch.zeros((rows, experts), dtype=torch.bool, device=anchor_scores.device)
    result[:, :anchor_quota] = anchor_order[:, :anchor_quota]
    if anchor_quota:
        selected.scatter_(1, result[:, :anchor_quota], True)
    counts = torch.full((rows,), anchor_quota, dtype=torch.long, device=anchor_scores.device)
    row_ids = torch.arange(rows, device=anchor_scores.device)
    for rank in range(experts):
        ids = branch_order[:, rank]
        positive = branch_values.gather(1, ids[:, None]).squeeze(1) > 0
        keep = (
            positive
            & ~selected.gather(1, ids[:, None]).squeeze(1)
            & (counts < width)
        )
        if keep.any():
            active_rows = row_ids[keep]
            active_ids = ids[keep]
            result[active_rows, counts[keep]] = active_ids
            selected[active_rows, active_ids] = True
            counts[keep] += 1
        if bool((counts == width).all()):
            break
    for rank in range(experts):
        ids = anchor_order[:, rank]
        keep = ~selected.gather(1, ids[:, None]).squeeze(1) & (counts < width)
        if keep.any():
            active_rows = row_ids[keep]
            active_ids = ids[keep]
            result[active_rows, counts[keep]] = active_ids
            selected[active_rows, active_ids] = True
            counts[keep] += 1
        if bool((counts == width).all()):
            break
    if (result < 0).any():
        raise RuntimeError("candidate policy failed to fill its fixed width")
    return result.reshape(*leading, width)


def global_candidates(full_inclusion_mass: Tensor, *, width: int = 64) -> Tensor:
    return torch.argsort(
        full_inclusion_mass.float(), dim=-1, descending=True, stable=True
    )[..., :width]


def factual_branch_candidates(
    anchor_scores: Tensor,
    factual_selected_ids: Tensor | None,
    *,
    width: int = 64,
) -> Tensor:
    anchor = torch.argsort(
        anchor_scores.float(), dim=-1, descending=True, stable=True
    )
    if factual_selected_ids is None:
        return anchor[..., :width]
    if factual_selected_ids.shape != anchor_scores.shape[:-1] + (TOP_K,):
        raise ValueError("factual selected-set geometry disagrees with anchor")
    if not TOP_K <= width <= anchor_scores.shape[-1]:
        raise ValueError("factual candidate width must contain the exact top-k set")
    leading = anchor_scores.shape[:-1]
    rows = math.prod(leading)
    exact = factual_selected_ids.long().reshape(rows, TOP_K)
    ordered = anchor.reshape(rows, anchor_scores.shape[-1])
    result = torch.full(
        (rows, width), -1, dtype=torch.long, device=anchor_scores.device
    )
    for row in range(rows):
        chosen = [int(value) for value in exact[row].tolist()]
        if len(set(chosen)) != TOP_K:
            raise ValueError("factual selected set contains duplicate experts")
        selected = set(chosen)
        chosen.extend(
            int(expert)
            for expert in ordered[row].tolist()
            if int(expert) not in selected
        )
        result[row] = torch.tensor(
            chosen[:width], dtype=torch.long, device=anchor_scores.device
        )
    return result.reshape(*leading, width)


def h1_root_supervision_loss(
    predicted_scores: Tensor,
    predicted_queries: Tensor,
    target_selected_ids: Tensor,
    target_queries: Tensor,
    target_router_logits: Tensor,
    *,
    valid: Tensor | None = None,
    exact_k: int = TOP_K,
    query_weight: float = 0.2,
    router_kl_weight: float = 0.1,
) -> H1RootLoss:
    """Dormant H1 root loss contract; B1.5 must not run its optimizer."""

    exact = exact_set_nll(
        predicted_scores,
        target_selected_ids.long(),
        valid=valid,
        k=exact_k,
    )
    huber = F.huber_loss(
        predicted_queries.float(), target_queries.detach().float(), reduction="none"
    ).mean(-1)
    cosine = 1.0 - F.cosine_similarity(
        predicted_queries.float(), target_queries.detach().float(), dim=-1, eps=1e-8
    )
    query_cell = huber + 0.1 * cosine
    target_probability = torch.softmax(target_router_logits.detach().float(), dim=-1)
    router_cell = F.kl_div(
        torch.log_softmax(predicted_scores.float(), dim=-1),
        target_probability,
        reduction="none",
    ).sum(-1)
    if valid is not None:
        mask = valid.to(query_cell.dtype)
        query = (query_cell * mask).sum() / mask.sum().clamp_min(1.0)
        router = (router_cell * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        query = query_cell.mean()
        router = router_cell.mean()
    total = exact + float(query_weight) * query + float(router_kl_weight) * router
    return H1RootLoss(total=total, exact_set=exact, query=query, router_kl=router)


__all__ = [
    "B15_CANDIDATE_UNION_SCHEMA",
    "B15_ORACLE_SCHEMA",
    "B15_SELECTOR_SCHEMA",
    "BudgetSelection",
    "H1_C64_GATE",
    "H1RootLoss",
    "add_other_anchor_mass",
    "b15_candidate_union_manifest",
    "b15_candidate_union_sha256",
    "b15_selector_manifest",
    "b15_selector_sha256",
    "candidate_union",
    "factual_branch_candidates",
    "global_candidates",
    "h1_root_supervision_loss",
    "required_b15_free_bytes",
    "select_path_budget",
    "selected_set_inclusion_mass",
]
