"""Isolated full-prefix native-MTP branch execution for HARP-RTT capture."""

from __future__ import annotations

import torch

from adaptive_capture_contract import (
    ANCHOR_SPINE_REQUIRED_DEPTH,
    FULL_VOCAB_TOP_K,
    MAX_CAPTURE_DEPTH,
)
from adaptive_mtp_tree import NodeObservation


def stable_topk_log_probabilities(
    log_probabilities: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k by descending value with ascending token-ID tie breaking."""
    if log_probabilities.ndim != 1 or not 0 < k <= log_probabilities.numel():
        raise ValueError("stable top-k expects one vector and a valid k")
    provisional = torch.topk(log_probabilities, k, sorted=False).values
    cutoff = provisional.min()
    strict_ids = torch.nonzero(log_probabilities > cutoff, as_tuple=False).flatten()
    tied_ids = torch.nonzero(log_probabilities == cutoff, as_tuple=False).flatten()
    needed = k - strict_ids.numel()
    if needed < 0 or tied_ids.numel() < needed:
        raise RuntimeError("could not resolve deterministic top-k boundary")
    candidate_ids = torch.cat((strict_ids, tied_ids[:needed]))
    # nonzero emits ascending IDs; make this invariant explicit, then use a
    # stable score sort so exact ties retain ascending token ID.
    candidate_ids = candidate_ids[torch.argsort(candidate_ids)]
    order = torch.argsort(
        log_probabilities[candidate_ids], descending=True, stable=True
    )
    ids = candidate_ids[order]
    return log_probabilities[ids], ids


class NativeMTPBranchRunner:
    """Evaluate native MTP paths without future target state or shared KV state."""

    def __init__(
        self,
        *,
        mtp,
        authoritative_prefix_token_ids: list[int],
        authoritative_target_hidden_through_t: list[torch.Tensor],
        exact_h1_token_id: int,
    ) -> None:
        if not authoritative_prefix_token_ids:
            raise ValueError("authoritative MTP root prefix is empty")
        if len(authoritative_prefix_token_ids) != len(
            authoritative_target_hidden_through_t
        ):
            raise ValueError("target hidden history must end exactly at source token t")
        self.mtp = mtp
        self.prefix = tuple(int(v) for v in authoritative_prefix_token_ids)
        self.target_hidden = tuple(
            value.detach() for value in authoritative_target_hidden_through_t
        )
        self.exact_h1_token_id = int(exact_h1_token_id)
        self.head_input_by_path: dict[tuple[int, ...], torch.Tensor] = {}
        self.native_call_count = 0

    @torch.no_grad()
    def evaluate(self, path: tuple[int, ...]) -> NodeObservation:
        """Evaluate an adaptive-graph path, hard-limited to H1--H4."""

        return self._evaluate(path, maximum_depth=MAX_CAPTURE_DEPTH, channel="adaptive")

    @torch.no_grad()
    def evaluate_anchor(self, path: tuple[int, ...]) -> NodeObservation:
        """Evaluate the separate incumbent anchor path through pinned H1--H6."""

        return self._evaluate(
            path,
            maximum_depth=ANCHOR_SPINE_REQUIRED_DEPTH,
            channel="legacy anchor",
        )

    def _evaluate(
        self,
        path: tuple[int, ...],
        *,
        maximum_depth: int,
        channel: str,
    ) -> NodeObservation:
        if not path or path[0] != self.exact_h1_token_id:
            raise ValueError("every MTP branch must be rooted on exact committed x[t+1]")
        if len(path) > maximum_depth:
            raise ValueError(f"{channel} MTP path extends beyond H{maximum_depth}")
        ancestors: list[torch.Tensor] = []
        for length in range(1, len(path)):
            ancestor_path = path[:length]
            if ancestor_path not in self.head_input_by_path:
                raise ValueError("MTP parent must be evaluated before its child")
            ancestors.append(self.head_input_by_path[ancestor_path])

        input_ids = list(self.prefix[1:]) + [int(v) for v in path]
        previous_rows = list(self.target_hidden) + ancestors
        if len(input_ids) != len(previous_rows):
            raise RuntimeError("shifted MTP IDs and previous-hidden rows do not align")
        device = self.target_hidden[0].device
        output = self.mtp(
            torch.tensor([input_ids], device=device, dtype=torch.long),
            torch.stack(previous_rows, dim=0).unsqueeze(0),
            position_ids=torch.arange(
                len(input_ids), device=device, dtype=torch.long
            ).unsqueeze(0),
            past_key_values=None,
            use_cache=False,
        )
        self.native_call_count += 1

        router_logits = output["router_logits"]
        execution_weights = output["top8_weights"]
        if execution_weights.dtype != router_logits.dtype:
            raise RuntimeError(
                "MTP execution weights must retain the native router-logit dtype"
            )

        vocab_logits = output["vocabulary_logits"][0, -1]
        log_probs = torch.log_softmax(vocab_logits.float(), dim=-1)
        top_log_probs, top_ids = stable_topk_log_probabilities(
            log_probs, FULL_VOCAB_TOP_K
        )
        probabilities = log_probs.exp()
        entropy = float((-(probabilities * log_probs)).sum().item())
        top_probabilities = top_log_probs.exp()
        # Clone the single recurrence row: retaining a view here would pin the
        # entire full-prefix decoder output once per branch node.
        head_input = output["head_input"][0, -1].detach().clone()
        self.head_input_by_path[path] = head_input

        payload = {
            "frozen_target_token_embedding": output["token_embedding"][0, -1],
            "mtp_normalized_token_embedding": output["normalized_embedding"][0, -1],
            "mtp_normalized_previous_hidden": output[
                "normalized_previous_hidden"
            ][0, -1],
            "mtp_fused_state": output["fused_hidden"][0, -1],
            "mtp_router_input": output["router_input"][0, -1],
            "mtp_hidden_state": output["post_ffn_hidden"][0, -1],
            "mtp_post_ffn_hidden": output["post_ffn_hidden"][0, -1],
            "mtp_vocabulary_head_input": head_input,
            "raw_mtp_router_logits": router_logits[0, -1],
            "full_mtp_router_probabilities": output["router_probabilities"][0, -1],
            "mtp_selected_expert_ids": output["top8_ids"][0, -1].to(torch.int32),
            "mtp_selected_execution_weights": execution_weights[0, -1],
            "vocab_top64_token_ids": top_ids.to(torch.int32),
            "vocab_top64_log_probabilities": top_log_probs,
            "vocab_top64_probabilities": top_probabilities,
            "full_vocabulary_logits_audit": vocab_logits,
        }
        observation = NodeObservation(
            top_token_ids=tuple(int(v) for v in top_ids.detach().cpu().tolist()),
            top_log_probabilities=tuple(
                float(v) for v in top_log_probs.detach().cpu().tolist()
            ),
            vocabulary_size=int(vocab_logits.numel()),
            vocabulary_entropy=entropy,
            payload=payload,
        )
        del output, probabilities, log_probs
        return observation
