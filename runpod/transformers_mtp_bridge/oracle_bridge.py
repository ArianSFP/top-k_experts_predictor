#!/usr/bin/env python3
"""Compare the frozen Transformers checkpoint-MTP module to a SGLang oracle."""

from __future__ import annotations

import argparse
import json
import math
import mmap
from pathlib import Path
from typing import Any

import torch

from qwen35_mtp import load_checkpoint_mtp


DTYPES = {
    "bf16": torch.uint16,
    "f16": torch.float16,
    "f32": torch.float32,
    "i32": torch.int32,
    "i64": torch.int64,
    "bool": torch.bool,
}


class OracleStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        workers = sorted(root.glob("worker-*"))
        if len(workers) != 1:
            raise ValueError(f"expected one worker, found {workers}")
        self.worker = workers[0]
        self.events = [
            json.loads(line)
            for line in (self.worker / "events.jsonl").open(encoding="utf-8")
        ]
        self.starts = {
            event["call_id"]: event
            for event in self.events
            if event["event"] == "forward_begin"
        }
        self.tensor_events = {
            (event["call_id"], event["kind"], event.get("target_layer_id")): event
            for event in self.events
            if event["event"] == "tensor"
        }
        self.payload_stream = (self.worker / "tensors.bin").open("rb")
        self.payload = mmap.mmap(
            self.payload_stream.fileno(), 0, access=mmap.ACCESS_READ
        )

    def close(self) -> None:
        self.payload.close()
        self.payload_stream.close()

    def tensor(
        self, call_id: int, kind: str, layer: int | None = None
    ) -> torch.Tensor:
        event = self.tensor_events[(call_id, kind, layer)]
        start = event["payload_offset"]
        end = start + event["payload_bytes"]
        view = memoryview(self.payload)[start:end]
        value = torch.frombuffer(view, dtype=DTYPES[event["dtype"]]).clone()
        if event["dtype"] == "bf16":
            value = value.view(torch.bfloat16)
        return value.reshape(event["shape"])

    def first_cycle(self, rid_substring: str) -> dict[str, Any]:
        ordered_starts = sorted(self.starts.values(), key=lambda x: x["event_seq"])
        matching = [
            item
            for item in ordered_starts
            if any(rid_substring in rid for rid in (item.get("rids") or []))
        ]
        target_extend = next(
            item
            for item in matching
            if item["namespace"] == "target" and item["forward_mode"] == "EXTEND"
        )
        verify = next(
            item
            for item in matching
            if item["event_seq"] > target_extend["event_seq"]
            and item["namespace"] == "target"
            and item["forward_mode"] == "TARGET_VERIFY"
        )
        mtp = [
            item
            for item in matching
            if target_extend["event_seq"] < item["event_seq"] < verify["event_seq"]
            and item["namespace"] == "mtp"
        ]
        if [item["engine_draft_depth"] for item in mtp] != list(range(1, 7)):
            raise ValueError("first oracle cycle is not an unbranched depth-six chain")
        acceptance = next(
            item
            for item in self.events
            if item["event"] == "speculation_acceptance"
            and item["event_seq"] > verify["event_seq"]
            and any(rid_substring in rid for rid in (item.get("rids") or []))
        )
        response_path = next(
            path
            for path in sorted((self.root / "responses").glob("*.json"))
            if rid_substring in json.loads(path.read_text())["meta_info"]["id"]
        )
        return {
            "target_extend": target_extend,
            "mtp": mtp,
            "verify": verify,
            "acceptance": acceptance,
            "response": json.loads(response_path.read_text()),
        }


def _as_rows(value: torch.Tensor) -> torch.Tensor:
    if value.ndim >= 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 1:
        value = value.unsqueeze(0)
    return value


def vector_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = _as_rows(actual).float().cpu()
    reference = _as_rows(reference).float().cpu()
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch {actual.shape} vs {reference.shape}")
    error = actual - reference
    row_dot = (actual * reference).sum(dim=-1)
    row_norm = (
        torch.linalg.vector_norm(actual, dim=-1)
        * torch.linalg.vector_norm(reference, dim=-1)
    ).clamp_min(1e-30)
    cosine = row_dot / row_norm
    return {
        "cosine_mean": float(cosine.mean().item()),
        "cosine_min": float(cosine.min().item()),
        "rms_error": float(torch.sqrt(error.square().mean()).item()),
        "relative_rms_error": float(
            torch.sqrt(error.square().sum() / reference.square().sum().clamp_min(1e-30)).item()
        ),
        "max_abs_error": float(error.abs().max().item()),
    }


def centered_correlation(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = _as_rows(actual).float().cpu()
    reference = _as_rows(reference).float().cpu()
    actual = actual - actual.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    corr = (actual * reference).sum(dim=-1) / (
        torch.linalg.vector_norm(actual, dim=-1)
        * torch.linalg.vector_norm(reference, dim=-1)
    ).clamp_min(1e-30)
    return {"mean": float(corr.mean().item()), "min": float(corr.min().item())}


def topk_overlap(actual: torch.Tensor, reference: torch.Tensor, k: int) -> float:
    actual_ids = torch.topk(_as_rows(actual).float(), k, dim=-1).indices.cpu()
    reference_ids = torch.topk(_as_rows(reference).float(), k, dim=-1).indices.cpu()
    matches = (
        actual_ids.unsqueeze(-1) == reference_ids.unsqueeze(-2)
    ).any(dim=-1).sum(dim=-1)
    return float((matches.float() / k).mean().item())


def router_metrics(
    actual_logits: torch.Tensor,
    actual_ids: torch.Tensor,
    reference_logits: torch.Tensor,
    reference_native_ids: torch.Tensor,
) -> dict[str, Any]:
    actual_logits = _as_rows(actual_logits)
    actual_ids = _as_rows(actual_ids).long().cpu()
    reference_logits = _as_rows(reference_logits)
    reference_native_ids = _as_rows(reference_native_ids).long().cpu()
    set_match = torch.all(
        torch.sort(actual_ids, dim=-1).values
        == torch.sort(reference_native_ids, dim=-1).values,
        dim=-1,
    )
    slot_matches = (
        actual_ids.unsqueeze(-1) == reference_native_ids.unsqueeze(-2)
    ).any(dim=-1).sum(dim=-1)
    top9 = torch.topk(reference_logits.float().cpu(), 9, dim=-1).values
    cutoff_gap = top9[:, 7] - top9[:, 8]
    mismatch = ~set_match
    return {
        "logits": vector_metrics(actual_logits, reference_logits),
        "exact_set_rate": float(set_match.float().mean().item()),
        "slot_recall": float((slot_matches.float() / 8).mean().item()),
        "tie_mismatch_rows": int((mismatch & (cutoff_gap == 0)).sum().item()),
        "positive_gap_mismatch_rows": int(
            (mismatch & (cutoff_gap > 0)).sum().item()
        ),
        "reference_cutoff_gap_min": float(cutoff_gap.min().item()),
    }


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def compare_call(
    store: OracleStore,
    start: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    call_id = start["call_id"]
    ref_fused = store.tensor(call_id, "mtp_fused_hidden")
    ref_router_input = store.tensor(call_id, "mtp_router_input", 0)
    ref_router_logits = store.tensor(call_id, "mtp_raw_affine_router_logits", 0)
    ref_router_ids = store.tensor(
        call_id, "mtp_native_kernel_top8_expert_ids", 0
    )
    ref_head_input = store.tensor(call_id, "mtp_vocabulary_head_input")
    ref_vocab = store.tensor(call_id, "mtp_full_vocabulary_logits_audit")
    actual_vocab = output["vocabulary_logits"]
    reference_vocab_rows = _as_rows(ref_vocab).shape[0]
    actual_vocab_for_reference = _as_rows(actual_vocab)[-reference_vocab_rows:]
    actual_last = _as_rows(actual_vocab)[-1]
    ref_last = _as_rows(ref_vocab)[-1]
    return {
        "call_id": call_id,
        "engine_draft_depth": start["engine_draft_depth"],
        "model_positions": start["model_positions_raw"],
        "source_target_hidden_positions": start[
            "mtp_source_target_hidden_positions"
        ],
        "router_state_positions": start["mtp_router_state_positions"],
        "vocabulary_prediction_positions": start[
            "mtp_vocab_prediction_positions"
        ],
        "input_ids": start["input_ids"],
        "fused_state": vector_metrics(output["fused_hidden"], ref_fused),
        "router_input": vector_metrics(output["router_input"], ref_router_input),
        "router": router_metrics(
            output["router_logits"],
            output["top8_ids"],
            ref_router_logits,
            ref_router_ids,
        ),
        "head_input": vector_metrics(output["head_input"], ref_head_input),
        "vocabulary_logits": {
            **vector_metrics(actual_vocab_for_reference, ref_vocab),
            "centered_correlation": centered_correlation(actual_vocab_for_reference, ref_vocab),
            "top1_overlap": topk_overlap(actual_vocab_for_reference, ref_vocab, 1),
            "top5_overlap": topk_overlap(actual_vocab_for_reference, ref_vocab, 5),
            "top10_overlap": topk_overlap(actual_vocab_for_reference, ref_vocab, 10),
            "top64_overlap": topk_overlap(actual_vocab_for_reference, ref_vocab, 64),
        },
        "transformers_draft_token_id": int(actual_last.argmax().item()),
        "sglang_draft_token_id": int(ref_last.argmax().item()),
        "draft_token_match": bool(actual_last.argmax() == ref_last.argmax()),
    }


def run_mode(
    module,
    store: OracleStore,
    cycle: dict[str, Any],
    *,
    mode: str,
    device: torch.device,
) -> dict[str, Any]:
    if mode not in {
        "exact_hidden_forced_tokens",
        "recurrent_hidden_forced_tokens",
        "recurrent_hidden_free_tokens",
    }:
        raise ValueError(mode)
    cache = None
    previous_output = None
    calls = []
    draft_tokens: list[int] = []
    for index, start in enumerate(cycle["mtp"]):
        call_id = start["call_id"]
        reference_previous = store.tensor(
            call_id, "mtp_previous_target_hidden"
        ).unsqueeze(0).to(device)
        if index == 0 or mode == "exact_hidden_forced_tokens":
            previous_hidden = reference_previous
        else:
            previous_hidden = previous_output["head_input"][:, -1:, :]

        reference_ids = list(start["input_ids"])
        if index == 0 or mode != "recurrent_hidden_free_tokens":
            input_ids = reference_ids
        else:
            input_ids = [draft_tokens[-1]]
        position_ids = list(start["model_positions_raw"])
        if len(input_ids) != previous_hidden.shape[1]:
            raise ValueError(
                f"mode {mode} depth {index + 1}: token/hidden rows disagree"
            )
        output = module(
            torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(0),
            previous_hidden,
            position_ids=torch.tensor(
                position_ids, device=device, dtype=torch.long
            ).unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
        )
        cache = output["past_key_values"]
        result = compare_call(store, start, output)
        result["mode_input_ids"] = input_ids
        calls.append(result)
        draft_tokens.append(result["transformers_draft_token_id"])
        previous_output = output

    verify_draft = cycle["verify"]["spec_primitives"]["draft_token"]
    bonus_token = int(verify_draft[0])
    tree = [bonus_token] + draft_tokens
    committed = list(cycle["response"]["output_ids"])
    acceptance_by_prefix = common_prefix_length(tree, committed)
    native_acceptance = int(cycle["acceptance"]["accept_lens"][0])
    return {
        "mode": mode,
        "calls": calls,
        "transformers_tree_token_ids": tree,
        "sglang_verify_tree_token_ids": verify_draft,
        "all_six_draft_tokens_match": all(
            call["draft_token_match"] for call in calls
        ),
        "tree_token_ids_exact": tree == verify_draft,
        "acceptance_from_committed_prefix": acceptance_by_prefix,
        "sglang_native_accept_length": native_acceptance,
        "acceptance_outcome_match": acceptance_by_prefix == native_acceptance,
    }


def summarize_mode(mode: dict[str, Any]) -> dict[str, Any]:
    calls = mode["calls"]
    return {
        "mode": mode["mode"],
        "all_six_draft_tokens_match": mode["all_six_draft_tokens_match"],
        "tree_token_ids_exact": mode["tree_token_ids_exact"],
        "acceptance_outcome_match": mode["acceptance_outcome_match"],
        "fused_cosine_min": min(c["fused_state"]["cosine_min"] for c in calls),
        "router_logits_cosine_min": min(
            c["router"]["logits"]["cosine_min"] for c in calls
        ),
        "router_slot_recall_min": min(c["router"]["slot_recall"] for c in calls),
        "router_positive_gap_mismatch_rows": sum(
            c["router"]["positive_gap_mismatch_rows"] for c in calls
        ),
        "vocabulary_centered_correlation_min": min(
            c["vocabulary_logits"]["centered_correlation"]["min"] for c in calls
        ),
        "vocabulary_top1_overlap_min": min(
            c["vocabulary_logits"]["top1_overlap"] for c in calls
        ),
        "vocabulary_top64_overlap_min": min(
            c["vocabulary_logits"]["top64_overlap"] for c in calls
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--request-substring", default="seq0001")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    store = OracleStore(args.oracle)
    try:
        cycle = store.first_cycle(args.request_substring)
        module = load_checkpoint_mtp(args.model, device=device)
        modes = [
            run_mode(module, store, cycle, mode=mode, device=device)
            for mode in (
                "exact_hidden_forced_tokens",
                "recurrent_hidden_forced_tokens",
                "recurrent_hidden_free_tokens",
            )
        ]
        report = {
            "schema": "qwen36_transformers_sglang_mtp_bridge_v1",
            "model": str(args.model),
            "oracle": str(args.oracle),
            "request_substring": args.request_substring,
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "module_load_report": module.load_report,
            "position_contract": {
                "model_rope": "p",
                "source_target_hidden": "p",
                "input_token": "p+1",
                "router_state": "p+1",
                "vocabulary_prediction": "p+2",
                "draft_token": "p+2",
            },
            "modes": modes,
            "summaries": [summarize_mode(mode) for mode in modes],
            "training_started": False,
            "large_capture_started": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report["summaries"], indent=2, sort_keys=True))
    finally:
        store.close()


if __name__ == "__main__":
    main()
