"""Independent parent-recursive RouteMTP execution over a fixed native tree."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn

from .routemtp import InstalledRouteMTPAdapters, RecurrentRouteResidual


@dataclass(frozen=True)
class RouteMTPReplayOutput:
    token_embedding: Tensor
    normalized_embedding: Tensor
    normalized_previous_hidden: Tensor
    fused_state: Tensor
    router_input: Tensor
    router_logits: Tensor
    selected_ids: Tensor
    selected_weights: Tensor
    post_moe_hidden: Tensor
    vocabulary_head_input: Tensor
    call_count: int


class RouteMTPTreeRunner(nn.Module):
    """Replay each endpoint from an independently cloned base prefix cache.

    Recomputing short H1--H4 paths is intentionally the training reference:
    it avoids mutable-cache DAG aliasing and gives exact sibling isolation.
    A parent-cache reuse implementation may be promoted only after parity.
    """

    OUTPUT_KEYS = (
        "token_embedding",
        "normalized_embedding",
        "normalized_previous_hidden",
        "fused_hidden",
        "router_input",
        "router_logits",
        "top8_ids",
        "top8_weights",
        "post_ffn_hidden",
        "head_input",
    )

    def __init__(
        self,
        mtp: nn.Module,
        adapters: InstalledRouteMTPAdapters,
        *,
        recurrent_residual: RecurrentRouteResidual | None = None,
        expert_adapter: nn.Module | None = None,
        router_adapter: nn.Module | None = None,
        replay_mode: str = "isolated_full_prefix",
    ) -> None:
        super().__init__()
        if not hasattr(mtp, "forward_with_grad"):
            raise TypeError("RouteMTP replay requires a gradient-enabled MTP module")
        self.mtp = mtp
        self.adapters = adapters
        self.recurrent_residual = recurrent_residual
        self.expert_adapter = expert_adapter
        self.router_adapter = router_adapter
        if replay_mode not in {"isolated_full_prefix", "cached_append"}:
            raise ValueError("unsupported RouteMTP replay mode")
        self.replay_mode = replay_mode

    @staticmethod
    def _path(parent_ids: Tensor, node: int) -> list[int]:
        path: list[int] = []
        seen: set[int] = set()
        current = int(node)
        while current >= 0:
            if current in seen:
                raise ValueError("RouteMTP tree contains a parent cycle")
            if current > node:
                raise ValueError("RouteMTP tree is not parent-before-child")
            seen.add(current); path.append(current)
            current = int(parent_ids[current].item())
        path.reverse()
        if not path or path[0] != 0:
            raise ValueError("RouteMTP path is not rooted on exact H1")
        return path

    def forward(
        self,
        *,
        node_token_ids: Tensor,
        parent_ids: Tensor,
        node_mask: Tensor,
        current_target_hidden: Tensor,
        base_cache_factory: Callable[[int], Any],
        base_cache_lengths: Tensor,
        base_prefix_token_ids: list[Tensor] | None = None,
        base_target_hidden_history: list[Tensor] | None = None,
    ) -> RouteMTPReplayOutput:
        batch, nodes = node_token_ids.shape
        if parent_ids.shape != (batch, nodes) or node_mask.shape != (batch, nodes):
            raise ValueError("RouteMTP replay topology tensors disagree")
        if current_target_hidden.ndim != 2 or current_target_hidden.shape[0] != batch:
            raise ValueError("current target hidden must be [B,D]")
        if base_cache_lengths.shape != (batch,):
            raise ValueError("base cache lengths must be [B]")
        if self.replay_mode == "isolated_full_prefix":
            if base_prefix_token_ids is None or base_target_hidden_history is None:
                raise ValueError("full-prefix replay requires hydrated prefix tensors")
            if len(base_prefix_token_ids) != batch or len(base_target_hidden_history) != batch:
                raise ValueError("hydrated full-prefix batch geometry differs")
        if not bool(node_mask[:, 0].all()) or not bool((parent_ids[:, 0] == -1).all()):
            raise ValueError("RouteMTP replay requires the exact H1 root")

        records: list[list[dict[str, Tensor] | None]] = [
            [None for _ in range(nodes)] for _ in range(batch)
        ]
        calls = 0
        for batch_index in range(batch):
            topology = parent_ids[batch_index]
            for node in range(nodes):
                    if not bool(node_mask[batch_index, node]):
                        continue
                    path = self._path(topology, node)
                    if self.replay_mode == "isolated_full_prefix":
                        prefix_ids = base_prefix_token_ids[batch_index].to(
                            device=node_token_ids.device, dtype=torch.long
                        )
                        history = base_target_hidden_history[batch_index].to(
                            device=current_target_hidden.device,
                            dtype=current_target_hidden.dtype,
                        )
                        path_ids = node_token_ids[batch_index, path].long()
                        ancestors = [
                            records[batch_index][path_node]["head_input"][0]
                            for path_node in path[:-1]
                        ]
                        previous_rows = [history]
                        if ancestors:
                            previous_rows.append(torch.stack(ancestors))
                        previous = torch.cat(previous_rows, dim=0)[None]
                        full_ids = torch.cat((prefix_ids, path_ids))[None]
                        if full_ids.shape[1] != previous.shape[1]:
                            raise RuntimeError("full-prefix IDs/hidden rows do not align")
                        with ExitStack() as stack:
                            stack.enter_context(
                                self.adapters.active(True, tail_rows=len(path))
                            )
                            if self.expert_adapter is not None:
                                stack.enter_context(
                                    self.expert_adapter.active(True, tail_rows=len(path))
                                )
                            if self.router_adapter is not None:
                                stack.enter_context(
                                    self.router_adapter.active(True, tail_rows=len(path))
                                )
                            output = self.mtp.forward_with_grad(
                                full_ids,
                                previous,
                                position_ids=torch.arange(
                                    full_ids.shape[1],
                                    device=node_token_ids.device,
                                    dtype=torch.long,
                                )[None],
                                past_key_values=None,
                                use_cache=False,
                                compute_vocabulary_logits=False,
                            )
                        calls += 1
                        recurrent = output["head_input"][:, -1]
                        if self.recurrent_residual is not None:
                            recurrent = self.recurrent_residual(recurrent)
                        final = {key: output[key][:, -1] for key in self.OUTPUT_KEYS}
                        final["head_input"] = recurrent
                        records[batch_index][node] = final
                        continue
                    cache = base_cache_factory(batch_index)
                    # Hydration supplies the exact final-normalized target
                    # row consumed by native MTP at source t.  Reconstructing
                    # it from a BF16 persisted layer-39 xplus is measurably
                    # different near MTP router boundaries.
                    previous = current_target_hidden[
                        batch_index : batch_index + 1, None
                    ]
                    final: dict[str, Tensor] | None = None
                    for step, path_node in enumerate(path):
                        with ExitStack() as stack:
                            stack.enter_context(
                                self.adapters.active(True, tail_rows=1)
                            )
                            if self.expert_adapter is not None:
                                stack.enter_context(
                                    self.expert_adapter.active(True, tail_rows=1)
                                )
                            if self.router_adapter is not None:
                                stack.enter_context(
                                    self.router_adapter.active(True, tail_rows=1)
                                )
                            output = self.mtp.forward_with_grad(
                                node_token_ids[
                                    batch_index : batch_index + 1,
                                    path_node : path_node + 1,
                                ],
                                previous,
                                position_ids=torch.tensor(
                                    [[int(base_cache_lengths[batch_index].item()) + step]],
                                    device=node_token_ids.device,
                                    dtype=torch.long,
                                ),
                                past_key_values=cache,
                                use_cache=True,
                                compute_vocabulary_logits=False,
                            )
                        calls += 1
                        cache = output["past_key_values"]
                        recurrent = output["head_input"][:, -1]
                        if self.recurrent_residual is not None:
                            recurrent = self.recurrent_residual(recurrent)
                        previous = recurrent[:, None]
                        final = {
                            key: output[key][:, -1] for key in self.OUTPUT_KEYS
                        }
                        final["head_input"] = recurrent
                    if final is None:
                        raise RuntimeError("RouteMTP replay produced no node output")
                    records[batch_index][node] = final

        template = next(
            record for row in records for record in row if record is not None
        )
        stacked: dict[str, Tensor] = {}
        for key in self.OUTPUT_KEYS:
            zero = torch.zeros_like(template[key][0])
            rows: list[Tensor] = []
            for batch_index in range(batch):
                rows.append(torch.stack([
                    records[batch_index][node][key][0]
                    if records[batch_index][node] is not None else zero
                    for node in range(nodes)
                ]))
            stacked[key] = torch.stack(rows)
        return RouteMTPReplayOutput(
            token_embedding=stacked["token_embedding"],
            normalized_embedding=stacked["normalized_embedding"],
            normalized_previous_hidden=stacked["normalized_previous_hidden"],
            fused_state=stacked["fused_hidden"],
            router_input=stacked["router_input"],
            router_logits=stacked["router_logits"],
            selected_ids=stacked["top8_ids"].long(),
            selected_weights=stacked["top8_weights"],
            post_moe_hidden=stacked["post_ffn_hidden"],
            vocabulary_head_input=stacked["head_input"],
            call_count=calls,
        )


__all__ = ["RouteMTPReplayOutput", "RouteMTPTreeRunner"]
