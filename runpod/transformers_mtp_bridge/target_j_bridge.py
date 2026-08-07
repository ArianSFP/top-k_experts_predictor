#!/usr/bin/env python3
"""Forced-token target residual, J-space, and end-to-end MTP bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
)

from oracle_bridge import (
    OracleStore,
    centered_correlation,
    common_prefix_length,
    router_metrics,
    topk_overlap,
    vector_metrics,
)
from qwen35_mtp import load_checkpoint_mtp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--request-substring", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_target(model_path: Path) -> Qwen3_5MoeForCausalLM:
    full_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config = full_config.text_config
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = Qwen3_5MoeForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": "cuda"},
    )
    return model.eval().requires_grad_(False)


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    row = logits.reshape(-1)
    return int((row > row[token_id]).sum().item()) + 1


def j_metrics(
    target,
    target_block: torch.Tensor,
    sglang_block: torch.Tensor,
    matrix: torch.Tensor,
    selected_token: int,
) -> dict[str, Any]:
    device = target.device
    matrix = matrix.float().to(device)
    target_j = target_block.float().to(device) @ matrix.T
    sglang_j = sglang_block.float().to(device) @ matrix.T
    state_metrics = vector_metrics(target_j, sglang_j)

    target_normed = target.model.norm(target_j.to(torch.bfloat16))
    sglang_normed = target.model.norm(sglang_j.to(torch.bfloat16))
    target_logits = target.lm_head(target_normed)
    sglang_logits = target.lm_head(sglang_normed)
    target_last = target_logits[-1]
    sglang_last = sglang_logits[-1]
    result = {
        "transported_state": state_metrics,
        "centered_logit_correlation": centered_correlation(
            target_logits, sglang_logits
        ),
        "top1_overlap": topk_overlap(target_logits, sglang_logits, 1),
        "top5_overlap": topk_overlap(target_logits, sglang_logits, 5),
        "top10_overlap": topk_overlap(target_logits, sglang_logits, 10),
        "top50_overlap": topk_overlap(target_logits, sglang_logits, 50),
        "selected_token_id": selected_token,
        "transformers_selected_token_rank": rank_of(target_last, selected_token),
        "sglang_selected_token_rank": rank_of(sglang_last, selected_token),
    }
    result["selected_token_rank_difference"] = (
        result["transformers_selected_token_rank"]
        - result["sglang_selected_token_rank"]
    )
    return result


@torch.no_grad()
def run_mtp_chain(
    target,
    target_output,
    target_input_ids: list[int],
    module,
    cycle: dict[str, Any],
) -> dict[str, Any]:
    device = target.device
    bonus_token = int(target_output.logits[0, -1].argmax().item())
    mtp_input_ids = target_input_ids[1:] + [bonus_token]
    previous_hidden = target_output.hidden_states[-1]
    position_ids = torch.arange(
        len(target_input_ids), device=device, dtype=torch.long
    ).unsqueeze(0)

    output = module(
        torch.tensor(mtp_input_ids, device=device, dtype=torch.long).unsqueeze(0),
        previous_hidden,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=True,
    )
    cache = output["past_key_values"]
    drafts = [int(output["vocabulary_logits"][0, -1].argmax().item())]
    per_depth = [
        {
            "depth": 1,
            "input_ids": mtp_input_ids,
            "draft_token_id": drafts[-1],
            "router_state_position": len(target_input_ids),
            "vocabulary_prediction_position": len(target_input_ids) + 1,
        }
    ]
    previous_hidden = output["head_input"][:, -1:, :]
    model_position = len(target_input_ids)

    for depth in range(2, 7):
        output = module(
            torch.tensor([[drafts[-1]]], device=device, dtype=torch.long),
            previous_hidden,
            position_ids=torch.tensor(
                [[model_position]], device=device, dtype=torch.long
            ),
            past_key_values=cache,
            use_cache=True,
        )
        cache = output["past_key_values"]
        drafts.append(int(output["vocabulary_logits"][0, -1].argmax().item()))
        per_depth.append(
            {
                "depth": depth,
                "input_ids": [drafts[-2]],
                "draft_token_id": drafts[-1],
                "router_state_position": model_position + 1,
                "vocabulary_prediction_position": model_position + 2,
            }
        )
        previous_hidden = output["head_input"][:, -1:, :]
        model_position += 1

    tree = [bonus_token] + drafts
    reference_tree = cycle["verify"]["spec_primitives"]["draft_token"]

    verifier_cache = target_output.past_key_values
    accepted = 1
    verifier_tokens: list[int] = []
    current_token = bonus_token
    for draft in drafts:
        verifier = target(
            input_ids=torch.tensor([[current_token]], device=device),
            past_key_values=verifier_cache,
            use_cache=True,
            return_dict=True,
        )
        verifier_cache = verifier.past_key_values
        verifier_token = int(verifier.logits[0, -1].argmax().item())
        verifier_tokens.append(verifier_token)
        if verifier_token != draft:
            break
        accepted += 1
        current_token = draft

    native_acceptance = int(cycle["acceptance"]["accept_lens"][0])
    committed = list(cycle["response"]["output_ids"])
    return {
        "transformers_tree_token_ids": tree,
        "sglang_tree_token_ids": reference_tree,
        "tree_token_ids_exact": tree == reference_tree,
        "draft_token_matches_by_depth": [
            actual == expected
            for actual, expected in zip(drafts, reference_tree[1:])
        ],
        "per_depth": per_depth,
        "transformers_verifier_tokens_until_rejection": verifier_tokens,
        "transformers_accept_length": accepted,
        "sglang_native_accept_length": native_acceptance,
        "acceptance_outcome_match": accepted == native_acceptance,
        "common_prefix_with_sglang_committed": common_prefix_length(
            tree, committed
        ),
    }


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    store = OracleStore(args.oracle)
    try:
        cycle = store.first_cycle(args.request_substring)
        target_start = cycle["target_extend"]
        target_call = target_start["call_id"]
        target_ids = list(target_start["input_ids"])
        target = load_target(args.model)
        block_outputs: dict[int, torch.Tensor] = {}
        handles = []
        for layer_id, layer in enumerate(target.model.layers):
            def capture(_module, _inputs, output, layer_id=layer_id):
                block_outputs[layer_id] = output.detach().cpu()
            handles.append(layer.register_forward_hook(capture))

        input_ids = torch.tensor(
            target_ids, device=target.device, dtype=torch.long
        ).unsqueeze(0)
        output = target(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            output_router_logits=True,
            return_dict=True,
        )
        for handle in handles:
            handle.remove()
        if sorted(block_outputs) != list(range(40)):
            raise RuntimeError("did not capture all 40 target block outputs")

        selected_token = int(output.logits[0, -1].argmax().item())
        target_layers = []
        for layer in range(40):
            reference_block = store.tensor(
                target_call, "post_moe_residual_xplus_target", layer
            )
            actual_block = block_outputs[layer]
            actual_router_logits = output.router_logits[layer]
            actual_router_probs = torch.softmax(
                actual_router_logits.float(), dim=-1
            )
            _, actual_router_ids = torch.topk(actual_router_probs, 8, dim=-1)
            reference_router_logits = store.tensor(
                target_call, "target_raw_affine_router_logits", layer
            )
            reference_router_ids = store.tensor(
                target_call, "target_native_kernel_top8_expert_ids", layer
            )
            target_layers.append(
                {
                    "layer": layer,
                    "post_block_residual": vector_metrics(
                        actual_block, reference_block
                    ),
                    "router": router_metrics(
                        actual_router_logits,
                        actual_router_ids,
                        reference_router_logits,
                        reference_router_ids,
                    ),
                }
            )

        reference_mtp_source = store.tensor(
            cycle["mtp"][0]["call_id"], "mtp_previous_target_hidden"
        )
        target_source_metrics = vector_metrics(
            output.hidden_states[-1], reference_mtp_source
        )

        lens_checkpoint = torch.load(
            args.lens, map_location="cpu", weights_only=True
        )
        j_layers = []
        for layer in lens_checkpoint["source_layers"]:
            reference_block = store.tensor(
                target_call, "post_moe_residual_xplus_target", layer
            )
            j_layers.append(
                {
                    "layer": layer,
                    **j_metrics(
                        target,
                        block_outputs[layer][0],
                        reference_block,
                        lens_checkpoint["J"][layer],
                        selected_token,
                    ),
                }
            )

        mtp = load_checkpoint_mtp(args.model, device=target.device)
        mtp_chain = run_mtp_chain(
            target, output, target_ids, mtp, cycle
        )

        report = {
            "schema": "qwen36_transformers_sglang_target_j_mtp_bridge_v1",
            "model": str(args.model),
            "oracle": str(args.oracle),
            "lens": str(args.lens),
            "request_substring": args.request_substring,
            "target_input_ids": target_ids,
            "transformers_bonus_token_id": selected_token,
            "sglang_bonus_token_id": int(
                cycle["verify"]["spec_primitives"]["draft_token"][0]
            ),
            "bonus_token_match": selected_token
            == int(cycle["verify"]["spec_primitives"]["draft_token"][0]),
            "target_layers": target_layers,
            "target_final_normalized_hidden_vs_mtp_source": target_source_metrics,
            "j_layers": j_layers,
            "end_to_end_mtp_chain": mtp_chain,
            "summary": {
                "target_residual_cosine_min": min(
                    row["post_block_residual"]["cosine_min"]
                    for row in target_layers
                ),
                "target_router_slot_recall_min": min(
                    row["router"]["slot_recall"] for row in target_layers
                ),
                "target_router_positive_gap_mismatch_rows": sum(
                    row["router"]["positive_gap_mismatch_rows"]
                    for row in target_layers
                ),
                "j_state_cosine_min": min(
                    row["transported_state"]["cosine_min"] for row in j_layers
                ),
                "j_centered_logit_correlation_min": min(
                    row["centered_logit_correlation"]["min"] for row in j_layers
                ),
                "bonus_token_match": selected_token
                == int(cycle["verify"]["spec_primitives"]["draft_token"][0]),
                "mtp_tree_exact": mtp_chain["tree_token_ids_exact"],
                "mtp_acceptance_match": mtp_chain["acceptance_outcome_match"],
            },
            "training_started": False,
            "large_capture_started": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
    finally:
        store.close()


if __name__ == "__main__":
    main()
