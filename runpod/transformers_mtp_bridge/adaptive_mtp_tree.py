#!/usr/bin/env python3
"""Deterministic, causal adaptive-tree policy for native checkpoint MTP capture.

The policy in this module is deliberately independent of the target model and
capture writer.  Its evaluator receives only a speculative token path rooted on
the already-selected H1 token.  It may return native MTP outputs for that path,
but it never receives realized H2--H4 tokens or acceptance decisions.

Expansion is confidence-adaptive and exact-budgeted.  A greedy spine first
guarantees coverage through H4 when the root is non-terminal.  Preferred edges
are filled best-first: low-entropy parents initially expose fewer alternatives,
high-entropy parents initially expose more, and a depth bonus favors completing
promising paths instead of spending the whole budget at H2.  If that preferred
frontier is exhausted, deterministic lower-ranked edges refill it so
``adaptive-16`` is always the canonical prefix of ``adaptive-32``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
from typing import Any, Callable, Iterable


POLICY_SCHEMA = "harp_rtt_native_mtp_adaptive_policy_v1"
PATH_DOMAIN = b"HARP_RTT_NATIVE_MTP_PATH_V1\0"


def _canonical_path_digest(tree_id: str, tokens: Iterable[int]) -> str:
    digest = hashlib.sha256(PATH_DOMAIN)
    encoded_tree = tree_id.encode("utf-8")
    digest.update(len(encoded_tree).to_bytes(4, "little"))
    digest.update(encoded_tree)
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=True))
    return digest.hexdigest()


@dataclass
class NodeObservation:
    """Compact vocabulary observation returned by an isolated MTP execution.

    ``payload`` may temporarily hold tensors for the capture callback.  The
    builder clears it immediately after ``on_node`` returns, so a 32-node tree
    cannot accidentally retain 32 full-vocabulary tensors or KV caches.
    """

    top_token_ids: tuple[int, ...]
    top_log_probabilities: tuple[float, ...]
    vocabulary_size: int
    vocabulary_entropy: float
    payload: Any = field(default=None, repr=False)

    def validate(self) -> None:
        if not self.top_token_ids or len(self.top_token_ids) != len(
            self.top_log_probabilities
        ):
            raise ValueError("MTP observation must contain aligned top tokens/log-probabilities")
        if len(set(self.top_token_ids)) != len(self.top_token_ids):
            raise ValueError("MTP top-token IDs must be unique")
        if self.vocabulary_size <= max(self.top_token_ids) or min(self.top_token_ids) < 0:
            raise ValueError("MTP top-token ID outside declared vocabulary")
        if not math.isfinite(self.vocabulary_entropy) or self.vocabulary_entropy < 0.0:
            raise ValueError("MTP vocabulary entropy must be finite and non-negative")
        previous = math.inf
        for value in self.top_log_probabilities:
            if not math.isfinite(value) or value > 1e-6:
                raise ValueError("MTP log probabilities must be finite and <= 0")
            if value > previous + 1e-7:
                raise ValueError("MTP candidates must be sorted by descending probability")
            previous = value

    @property
    def top_probability(self) -> float:
        return math.exp(self.top_log_probabilities[0])

    @property
    def top1_top2_margin(self) -> float:
        if len(self.top_log_probabilities) < 2:
            return math.inf
        return self.top_log_probabilities[0] - self.top_log_probabilities[1]

    @property
    def normalized_entropy(self) -> float:
        denominator = math.log(max(2, self.vocabulary_size))
        return min(1.0, max(0.0, self.vocabulary_entropy / denominator))


@dataclass(frozen=True)
class AdaptiveExpansionPolicy:
    """Frozen first-teacher expansion policy from the HARP-RTT specification."""

    max_nodes: int = 32
    max_depth: int = 4
    maximum_width: int = 8
    very_confident_probability: float = 0.75
    confident_probability: float = 0.45
    moderate_probability: float = 0.25
    very_confident_margin: float = 2.0
    confident_margin: float = 1.0
    # Roughly offsets one plausible-token log-probability per additional level.
    # This leaves uncertain roots wide while reserving material budget for H4.
    depth_priority_bonus: float = 1.20
    uncertainty_priority_bonus: float = 0.40

    def __post_init__(self) -> None:
        if not 1 <= self.max_nodes <= 64:
            raise ValueError("max_nodes must be in [1, 64]")
        if self.max_depth != 4:
            raise ValueError("the HARP-RTT first teacher targets exactly H1--H4")
        if not 1 <= self.maximum_width <= 64:
            raise ValueError("maximum_width must be in [1, 64]")

    def width(self, observation: NodeObservation, *, depth: int) -> int:
        if depth >= self.max_depth:
            return 0
        probability = observation.top_probability
        margin = observation.top1_top2_margin
        if (
            probability >= self.very_confident_probability
            or margin >= self.very_confident_margin
        ):
            width = 1
        elif probability >= self.confident_probability or margin >= self.confident_margin:
            width = 2
        elif probability >= self.moderate_probability:
            width = 4
        else:
            width = self.maximum_width
        return min(width, len(observation.top_token_ids))

    def proposal_priority(
        self,
        *,
        child_path_log_probability: float,
        child_depth: int,
        parent_observation: NodeObservation,
    ) -> float:
        return (
            child_path_log_probability
            + self.depth_priority_bonus * float(child_depth - 1)
            + self.uncertainty_priority_bonus * parent_observation.normalized_entropy
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": POLICY_SCHEMA,
            "max_nodes_including_exact_h1_root": self.max_nodes,
            "max_depth": self.max_depth,
            "maximum_branch_width": self.maximum_width,
            "width_rule": {
                "one": {
                    "top_probability_gte": self.very_confident_probability,
                    "or_top1_top2_logprob_margin_gte": self.very_confident_margin,
                },
                "two": {
                    "top_probability_gte": self.confident_probability,
                    "or_top1_top2_logprob_margin_gte": self.confident_margin,
                },
                "four": {"top_probability_gte": self.moderate_probability},
                "eight_otherwise": self.maximum_width,
            },
            "greedy_spine_through_h4_first": True,
            "best_first_priority": {
                "path_log_probability": 1.0,
                "child_depth_minus_one": self.depth_priority_bonus,
                "parent_normalized_entropy": self.uncertainty_priority_bonus,
            },
            "tie_break": [
                "child_depth",
                "parent_local_index",
                "candidate_rank",
                "token_id",
            ],
            "exact_node_budget_when_nonterminal_root": True,
            "frontier_refill": {
                "trigger": "preferred_frontier_exhausted_before_node_budget",
                "maximum_rank_exclusive": self.maximum_width,
                "ranking": "same_best_first_priority_and_stable_tie_break",
            },
            "uses_realized_future_tokens": False,
            "uses_acceptance_labels": False,
        }


@dataclass
class AdaptiveTreeNode:
    local_index: int
    tree_id: str
    tree_node_id: str
    parent_local_index: int | None
    parent_tree_node_id: str | None
    depth: int
    token_id: int
    token_rank_under_parent: int
    token_path_ids: tuple[int, ...]
    token_path_log_probabilities: tuple[float, ...]
    local_token_log_probability: float
    path_log_probability: float
    path_id: str
    branch_id: str
    observation: NodeObservation

    @property
    def local_token_probability(self) -> float:
        return math.exp(self.local_token_log_probability)

    @property
    def path_probability(self) -> float:
        return math.exp(self.path_log_probability)


@dataclass
class AnchorSpineNode:
    """One explicit node of the incumbent-compatible greedy H1--H6 spine."""

    local_index: int
    tree_id: str
    anchor_spine_id: str
    parent_local_index: int | None
    depth: int
    token_id: int
    token_rank_under_parent: int | None
    token_path_ids: tuple[int, ...]
    token_path_log_probabilities: tuple[float, ...]
    local_token_log_probability: float
    path_log_probability: float
    path_id: str
    observation: NodeObservation

    @property
    def local_token_probability(self) -> float:
        return math.exp(self.local_token_log_probability)

    @property
    def path_probability(self) -> float:
        return math.exp(self.path_log_probability)


def build_anchor_spine(
    *,
    tree_id: str,
    exact_h1_token_id: int,
    evaluate: Callable[[tuple[int, ...]], NodeObservation],
    on_node: Callable[[AnchorSpineNode], None] | None = None,
    required_depth: int = 6,
) -> list[AnchorSpineNode]:
    """Materialize the parent-coherent legacy greedy spine through every depth.

    This channel is independent of the adaptive H1--H4 graph. It always
    follows the preceding node's local rank-zero vocabulary token and does not
    stop on EOS, matching the incumbent capture that unconditionally produced
    every available preprocessing depth.
    """

    if exact_h1_token_id < 0:
        raise ValueError("exact H1 token ID must be non-negative")
    if required_depth <= 0:
        raise ValueError("anchor spine depth must be positive")
    anchor_spine_id = f"{tree_id}:legacy-anchor-h1-h{required_depth}"
    nodes: list[AnchorSpineNode] = []
    parent: AnchorSpineNode | None = None
    for depth in range(1, required_depth + 1):
        if parent is None:
            token_id = int(exact_h1_token_id)
            rank: int | None = None
            local_logp = 0.0
            path = (token_id,)
            edge_logps = (0.0,)
            path_logp = 0.0
        else:
            token_id = int(parent.observation.top_token_ids[0])
            rank = 0
            local_logp = float(parent.observation.top_log_probabilities[0])
            path = parent.token_path_ids + (token_id,)
            edge_logps = parent.token_path_log_probabilities + (local_logp,)
            path_logp = parent.path_log_probability + local_logp
        observation = evaluate(path)
        observation.validate()
        node = AnchorSpineNode(
            local_index=depth - 1,
            tree_id=tree_id,
            anchor_spine_id=anchor_spine_id,
            parent_local_index=None if parent is None else parent.local_index,
            depth=depth,
            token_id=token_id,
            token_rank_under_parent=rank,
            token_path_ids=path,
            token_path_log_probabilities=edge_logps,
            local_token_log_probability=local_logp,
            path_log_probability=path_logp,
            path_id=f"anchor-path:{_canonical_path_digest(anchor_spine_id, path)}",
            observation=observation,
        )
        nodes.append(node)
        if on_node is not None:
            on_node(node)
        observation.payload = None
        parent = node

    for expected, node in enumerate(nodes):
        if node.local_index != expected or node.depth != expected + 1:
            raise ValueError("anchor spine is not contiguous H1--H6")
        if expected == 0:
            if (
                node.parent_local_index is not None
                or node.token_id != exact_h1_token_id
                or node.token_rank_under_parent is not None
            ):
                raise ValueError("anchor spine has an invalid exact-H1 root")
            continue
        previous = nodes[expected - 1]
        if (
            node.parent_local_index != expected - 1
            or node.token_rank_under_parent != 0
            or node.token_path_ids[:-1] != previous.token_path_ids
            or node.token_id != previous.observation.top_token_ids[0]
        ):
            raise ValueError("anchor spine is not a parent-coherent local-top1 chain")
    return nodes


@dataclass(frozen=True)
class FixedBeamPolicy:
    """Matched level-budget control for the adaptive H1--H4 policy."""

    widths_h2_h4: tuple[int, int, int]

    def __post_init__(self) -> None:
        if any(width < 1 for width in self.widths_h2_h4):
            raise ValueError("fixed beam widths must be positive")
        if self.widths_h2_h4 not in {(5, 5, 5), (11, 10, 10)}:
            raise ValueError("formal fixed controls are beam-16 or beam-32")

    @property
    def max_nodes(self) -> int:
        return 1 + sum(self.widths_h2_h4)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "harp_rtt_fixed_probability_beam_control_v1",
            "maximum_nodes_including_root": self.max_nodes,
            "depth_widths": {
                "1": 1,
                "2": self.widths_h2_h4[0],
                "3": self.widths_h2_h4[1],
                "4": self.widths_h2_h4[2],
            },
            "ranking": [
                "descending_cumulative_mtp_log_probability",
                "lexicographic_token_path",
                "parent_local_index",
                "candidate_rank",
                "token_id",
            ],
            "uses_target_labels": False,
            "uses_acceptance": False,
            "uses_factual_continuation": False,
        }


def build_fixed_beam_tree(
    *,
    tree_id: str,
    exact_h1_token_id: int,
    evaluate: Callable[[tuple[int, ...]], NodeObservation],
    policy: FixedBeamPolicy,
    on_node: Callable[[AdaptiveTreeNode], None] | None = None,
) -> list[AdaptiveTreeNode]:
    """Build a stable levelwise probability beam with an exact H1 root."""

    nodes: list[AdaptiveTreeNode] = []

    def add(
        parent: AdaptiveTreeNode | None,
        token_id: int,
        rank: int,
        local_logp: float,
    ) -> AdaptiveTreeNode:
        depth = 1 if parent is None else parent.depth + 1
        path = (token_id,) if parent is None else parent.token_path_ids + (token_id,)
        edge = (0.0,) if parent is None else (
            parent.token_path_log_probabilities + (local_logp,)
        )
        cumulative = 0.0 if parent is None else parent.path_log_probability + local_logp
        observation = evaluate(path)
        observation.validate()
        local = len(nodes)
        digest = _canonical_path_digest(tree_id, path)
        node = AdaptiveTreeNode(
            local_index=local,
            tree_id=tree_id,
            tree_node_id=f"{tree_id}:n{local:02d}",
            parent_local_index=None if parent is None else parent.local_index,
            parent_tree_node_id=None if parent is None else parent.tree_node_id,
            depth=depth,
            token_id=int(token_id),
            token_rank_under_parent=int(rank),
            token_path_ids=path,
            token_path_log_probabilities=edge,
            local_token_log_probability=0.0 if parent is None else float(local_logp),
            path_log_probability=float(cumulative),
            path_id=f"path:{digest}",
            branch_id=f"branch:{digest[:32]}",
            observation=observation,
        )
        nodes.append(node)
        if on_node is not None:
            on_node(node)
        observation.payload = None
        return node

    root = add(None, exact_h1_token_id, 0, 0.0)
    parents = [root]
    for target_depth, width in enumerate(policy.widths_h2_h4, start=2):
        proposals: list[tuple[float, tuple[int, ...], int, int, int]] = []
        for parent in parents:
            for rank, (token, logp) in enumerate(
                zip(
                    parent.observation.top_token_ids,
                    parent.observation.top_log_probabilities,
                    strict=True,
                )
            ):
                path = parent.token_path_ids + (int(token),)
                proposals.append(
                    (
                        -(parent.path_log_probability + float(logp)),
                        path,
                        parent.local_index,
                        rank,
                        int(token),
                    )
                )
        proposals.sort()
        chosen = proposals[:width]
        if len(chosen) != width:
            raise ValueError(f"fixed beam cannot fill H{target_depth} width {width}")
        parents = [
            add(nodes[parent], token, rank, nodes[parent].observation.top_log_probabilities[rank])
            for _negative, _path, parent, rank, token in chosen
        ]
    if len(nodes) != policy.max_nodes:
        raise RuntimeError("fixed beam did not consume its exact node budget")
    validate_tree_structure(
        nodes,
        policy=AdaptiveExpansionPolicy(max_nodes=policy.max_nodes),
        exact_h1_token_id=exact_h1_token_id,
    )
    return nodes


class AdaptiveMTPTreeBuilder:
    """Build one tree using only evaluator-produced native MTP distributions."""

    def __init__(self, policy: AdaptiveExpansionPolicy) -> None:
        self.policy = policy

    def build(
        self,
        *,
        tree_id: str,
        exact_h1_token_id: int,
        evaluate: Callable[[tuple[int, ...]], NodeObservation],
        on_node: Callable[[AdaptiveTreeNode], None] | None = None,
        eos_token_id: int | None = None,
    ) -> list[AdaptiveTreeNode]:
        if exact_h1_token_id < 0:
            raise ValueError("exact H1 token ID must be non-negative")
        nodes: list[AdaptiveTreeNode] = []
        occupied_edges: set[tuple[int, int]] = set()

        def add_node(
            *,
            parent: AdaptiveTreeNode | None,
            token_id: int,
            rank: int,
            local_log_probability: float,
        ) -> AdaptiveTreeNode:
            if len(nodes) >= self.policy.max_nodes:
                raise RuntimeError("adaptive tree budget exceeded")
            depth = 1 if parent is None else parent.depth + 1
            if depth > self.policy.max_depth:
                raise RuntimeError("adaptive tree depth exceeded")
            path = (token_id,) if parent is None else parent.token_path_ids + (token_id,)
            edge_logps = (
                (0.0,)
                if parent is None
                else parent.token_path_log_probabilities + (local_log_probability,)
            )
            path_log_probability = 0.0 if parent is None else (
                parent.path_log_probability + local_log_probability
            )
            observation = evaluate(path)
            observation.validate()
            local_index = len(nodes)
            path_digest = _canonical_path_digest(tree_id, path)
            node = AdaptiveTreeNode(
                local_index=local_index,
                tree_id=tree_id,
                tree_node_id=f"{tree_id}:n{local_index:02d}",
                parent_local_index=None if parent is None else parent.local_index,
                parent_tree_node_id=None if parent is None else parent.tree_node_id,
                depth=depth,
                token_id=token_id,
                token_rank_under_parent=rank,
                token_path_ids=path,
                token_path_log_probabilities=edge_logps,
                local_token_log_probability=0.0 if parent is None else local_log_probability,
                path_log_probability=path_log_probability,
                path_id=f"path:{path_digest}",
                branch_id=f"branch:{path_digest[:32]}",
                observation=observation,
            )
            nodes.append(node)
            if on_node is not None:
                on_node(node)
            # The writer has consumed native tensors.  Retain only compact top-k
            # values used by the frozen policy.
            observation.payload = None
            return node

        root = add_node(
            parent=None,
            token_id=exact_h1_token_id,
            rank=0,
            local_log_probability=0.0,
        )

        # A greedy causal spine guarantees an H1/H2/H3/H4 path before width can
        # consume the node budget.  It stops naturally on EOS.
        spine = root
        while (
            len(nodes) < self.policy.max_nodes
            and spine.depth < self.policy.max_depth
            and (eos_token_id is None or spine.token_id != eos_token_id)
        ):
            width = self.policy.width(spine.observation, depth=spine.depth)
            if width == 0:
                break
            token_id = spine.observation.top_token_ids[0]
            local_logp = spine.observation.top_log_probabilities[0]
            occupied_edges.add((spine.local_index, token_id))
            spine = add_node(
                parent=spine,
                token_id=token_id,
                rank=0,
                local_log_probability=local_logp,
            )

        # Heap entries are fully ordered, so equal model scores never make the
        # expansion depend on Python object identity or hash randomization.
        frontier: list[tuple[float, int, int, int, int]] = []
        queued_edges: set[tuple[int, int]] = set()

        def enqueue(parent: AdaptiveTreeNode, *, preferred_only: bool) -> None:
            if parent.depth >= self.policy.max_depth:
                return
            if eos_token_id is not None and parent.token_id == eos_token_id:
                return
            width = (
                self.policy.width(parent.observation, depth=parent.depth)
                if preferred_only
                else min(
                    self.policy.maximum_width,
                    len(parent.observation.top_token_ids),
                )
            )
            for rank in range(width):
                token_id = parent.observation.top_token_ids[rank]
                edge = (parent.local_index, token_id)
                if edge in occupied_edges or edge in queued_edges:
                    continue
                local_logp = parent.observation.top_log_probabilities[rank]
                priority = self.policy.proposal_priority(
                    child_path_log_probability=parent.path_log_probability + local_logp,
                    child_depth=parent.depth + 1,
                    parent_observation=parent.observation,
                )
                heapq.heappush(
                    frontier,
                    (-priority, parent.depth + 1, parent.local_index, rank, token_id),
                )
                queued_edges.add(edge)

        for node in nodes:
            enqueue(node, preferred_only=True)

        while len(nodes) < self.policy.max_nodes:
            if not frontier:
                # Thresholds allocate the preferred frontier. They must not
                # silently turn a declared view into a variable-size tree.
                for node in nodes:
                    enqueue(node, preferred_only=False)
                if not frontier:
                    break
            _negative_priority, _depth, parent_index, rank, token_id = heapq.heappop(
                frontier
            )
            edge = (parent_index, token_id)
            queued_edges.discard(edge)
            if edge in occupied_edges:
                continue
            parent = nodes[parent_index]
            occupied_edges.add(edge)
            child = add_node(
                parent=parent,
                token_id=token_id,
                rank=rank,
                local_log_probability=parent.observation.top_log_probabilities[rank],
            )
            enqueue(child, preferred_only=True)

        validate_tree_structure(nodes, policy=self.policy, exact_h1_token_id=exact_h1_token_id)
        if (
            (eos_token_id is None or exact_h1_token_id != eos_token_id)
            and len(nodes) != self.policy.max_nodes
        ):
            raise RuntimeError("adaptive tree did not consume its exact node budget")
        return nodes


def validate_tree_structure(
    nodes: list[AdaptiveTreeNode],
    *,
    policy: AdaptiveExpansionPolicy,
    exact_h1_token_id: int,
) -> None:
    if not nodes or len(nodes) > policy.max_nodes:
        raise ValueError("adaptive tree is empty or over budget")
    root = nodes[0]
    if (
        root.depth != 1
        or root.parent_local_index is not None
        or root.token_id != exact_h1_token_id
        or root.local_token_log_probability != 0.0
        or root.path_log_probability != 0.0
    ):
        raise ValueError("invalid exact committed H1 root")
    ids: set[str] = set()
    paths: set[str] = set()
    for expected_index, node in enumerate(nodes):
        if node.local_index != expected_index:
            raise ValueError("node local indices must be contiguous in creation order")
        if node.tree_node_id in ids or node.path_id in paths:
            raise ValueError("duplicate adaptive node/path identity")
        ids.add(node.tree_node_id)
        paths.add(node.path_id)
        if not 1 <= node.depth <= policy.max_depth:
            raise ValueError("node depth outside H1--H4")
        if node.parent_local_index is None:
            if expected_index != 0:
                raise ValueError("only node zero may be a root")
            continue
        if not 0 <= node.parent_local_index < node.local_index:
            raise ValueError("parent must be emitted before its child")
        parent = nodes[node.parent_local_index]
        if node.depth != parent.depth + 1:
            raise ValueError("child depth is not parent depth plus one")
        if node.parent_tree_node_id != parent.tree_node_id:
            raise ValueError("parent node identity mismatch")
        if node.token_path_ids[:-1] != parent.token_path_ids:
            raise ValueError("child token path does not extend its parent")
        if abs(
            node.path_log_probability
            - (parent.path_log_probability + node.local_token_log_probability)
        ) > 1e-7:
            raise ValueError("path log probability is not the sum of local edges")


def build_acceptance_labels(
    nodes: list[AdaptiveTreeNode],
    *,
    committed_token_ids: list[int],
    committed_prefix_position: int,
) -> list[dict[str, Any]]:
    """Derive label-only branch acceptance after the continuation is committed."""
    labels: list[dict[str, Any]] = []
    for node in nodes:
        start = committed_prefix_position + 1
        stop = start + node.depth
        realized = tuple(int(v) for v in committed_token_ids[start:stop])
        valid = len(realized) == node.depth
        matched = 0
        for predicted, actual in zip(node.token_path_ids, realized):
            if predicted != actual:
                break
            matched += 1
        accepted = valid and matched == node.depth
        labels.append(
            {
                "tree_node_id": node.tree_node_id,
                "path_id": node.path_id,
                "engine_draft_depth": node.depth,
                "accepted_prefix_length": matched if valid else None,
                "accepted_prefix_label": accepted if valid else None,
                "draft_token_accepted": accepted if valid else None,
                "branch_path_accepted": accepted if valid else None,
                "acceptance_label_valid": valid,
                "label_only": True,
            }
        )
    return labels


def _synthetic_observation(path: tuple[int, ...], *, uncertain: bool) -> NodeObservation:
    base = (sum(path) * 17 + len(path) * 101) % 10000
    ids = tuple((base + index * 37 + 11) % 32000 for index in range(8))
    if uncertain:
        logps = tuple(-1.65 - 0.18 * index for index in range(8))
        entropy = 9.5
    else:
        logps = (-0.08, -3.2, -4.1, -4.8, -5.2, -5.6, -6.0, -6.4)
        entropy = 1.2
    return NodeObservation(ids, logps, 32000, entropy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=32)
    parser.add_argument("--exact-h1-token-id", type=int, default=42)
    parser.add_argument("--scenario", choices=("confident", "uncertain"), default="uncertain")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("this policy utility only executes with --dry-run")
    policy = AdaptiveExpansionPolicy(max_nodes=args.max_nodes)
    nodes = AdaptiveMTPTreeBuilder(policy).build(
        tree_id="dry-run:tree000000",
        exact_h1_token_id=args.exact_h1_token_id,
        evaluate=lambda path: _synthetic_observation(
            path, uncertain=args.scenario == "uncertain"
        ),
    )
    print(
        json.dumps(
            {
                "policy": policy.manifest(),
                "scenario": args.scenario,
                "node_count": len(nodes),
                "nodes_by_depth": {
                    str(depth): sum(node.depth == depth for node in nodes)
                    for depth in range(1, policy.max_depth + 1)
                },
                "nodes": [
                    {
                        "local_index": node.local_index,
                        "parent_local_index": node.parent_local_index,
                        "depth": node.depth,
                        "token_id": node.token_id,
                        "token_rank_under_parent": node.token_rank_under_parent,
                        "path_log_probability": node.path_log_probability,
                        "tree_node_id": node.tree_node_id,
                        "branch_id": node.branch_id,
                        "path_id": node.path_id,
                    }
                    for node in nodes
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
