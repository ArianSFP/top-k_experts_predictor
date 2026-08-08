from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch

BRIDGE = Path(__file__).resolve().parents[1] / "runpod" / "transformers_mtp_bridge"
if str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))

capture_dependency = ModuleType("capture_transformers_segment")
capture_dependency.load_target = lambda *_args, **_kwargs: None
capture_dependency.prefix_hash = lambda values: str(tuple(values))
capture_dependency.sha256_file = lambda _path: "unused"
sys.modules["capture_transformers_segment"] = capture_dependency

from capture_counterfactual_companion import FrozenTreeNode  # noqa: E402
from capture_counterfactual_nodes import capture_node_tree  # noqa: E402


class FakeHooks:
    def __init__(self, layers: int) -> None:
        self.rows = {layer: {} for layer in range(layers)}

    def clear(self) -> None:
        for row in self.rows.values():
            row.clear()


class FakeGeometry:
    layers = 2
    hidden_width = 3
    maximum_rank = 3
    experts = 12

    def encode_router_inputs(self, values: torch.Tensor) -> torch.Tensor:
        return values.float()


class FakeTarget:
    device = torch.device("cpu")

    def __init__(self, hooks: FakeHooks) -> None:
        self.hooks = hooks

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values=None,
        **_kwargs,
    ):
        prefix = () if past_key_values is None else tuple(past_key_values)
        values = tuple(int(value) for value in input_ids.flatten().tolist())
        state = prefix + values
        vocab = torch.arange(64, dtype=torch.float32)
        logits = (vocab * 0.01 + sum(state) * 0.0001)[None, None]
        for layer in range(2):
            hidden = torch.tensor(
                [float(sum(state)), float(len(state)), float(layer)],
                dtype=torch.float32,
            )
            router = torch.arange(12, dtype=torch.float32) * 0.05
            router = router + hidden.sum() * 0.001
            ids = torch.topk(torch.softmax(router, dim=-1), 8).indices
            probability = torch.softmax(router, dim=-1).gather(0, ids)
            self.hooks.rows[layer].update(
                {
                    "router_input": hidden[None, None],
                    "router_logits": router[None],
                    "selected_ids": ids[None],
                    "selected_weights": (probability / probability.sum())[None],
                }
            )
        return SimpleNamespace(past_key_values=state, logits=logits)


def nodes() -> list[FrozenTreeNode]:
    rows = [
        (None, 1, 10, 0, (10,), (0.0,), 0.0),
        (0, 2, 20, 0, (10, 20), (0.0, -0.1), -0.1),
        (0, 2, 21, 1, (10, 21), (0.0, -0.2), -0.2),
        (1, 3, 30, 0, (10, 20, 30), (0.0, -0.1, -0.1), -0.2),
        (1, 3, 31, 1, (10, 20, 31), (0.0, -0.1, -0.3), -0.4),
        (2, 3, 32, 0, (10, 21, 32), (0.0, -0.2, -0.2), -0.4),
        (3, 4, 40, 0, (10, 20, 30, 40), (0.0, -0.1, -0.1, -0.1), -0.3),
        (3, 4, 41, 1, (10, 20, 30, 41), (0.0, -0.1, -0.1, -0.4), -0.6),
        (4, 4, 42, 0, (10, 20, 31, 42), (0.0, -0.1, -0.3, -0.1), -0.5),
        (5, 4, 43, 0, (10, 21, 32, 43), (0.0, -0.2, -0.2, -0.1), -0.5),
    ]
    return [
        FrozenTreeNode(
            local_index=index,
            parent_local_index=row[0],
            depth=row[1],
            token_id=row[2],
            token_rank_under_parent=row[3],
            token_path_ids=row[4],
            token_path_log_probabilities=row[5],
            path_log_probability=row[6],
        )
        for index, row in enumerate(rows)
    ]


def test_reference_and_cloned_parent_cache_are_exact_on_unique_nodes() -> None:
    geometry = FakeGeometry()
    hooks = FakeHooks(geometry.layers)
    target = FakeTarget(hooks)
    reference, reference_audit = capture_node_tree(
        target=target,
        hooks=hooks,
        geometry=geometry,
        authoritative_prefix=[1, 2, 3],
        nodes=nodes(),
        audit_router_inputs=True,
        execution_mode="isolated_full_prefix_replay_reference",
    )
    optimized, optimized_audit = capture_node_tree(
        target=target,
        hooks=hooks,
        geometry=geometry,
        authoritative_prefix=[1, 2, 3],
        nodes=nodes(),
        audit_router_inputs=True,
        execution_mode="sibling_isolated_cloned_parent_cache",
    )
    assert reference_audit is not None and optimized_audit is not None
    for name in reference:
        if reference[name].is_floating_point():
            assert torch.allclose(
                reference[name].float(), optimized[name].float(), equal_nan=True
            ), name
        else:
            assert torch.equal(reference[name], optimized[name]), name
    assert int(reference["valid"].sum()) == 9 * geometry.layers
    assert torch.equal(reference_audit["valid"], optimized_audit["valid"])
