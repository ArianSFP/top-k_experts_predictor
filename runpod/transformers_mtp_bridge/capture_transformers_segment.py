#!/usr/bin/env python3
"""Joint BF16 Transformers target-plus-MTP pilot capture for GCRP-2R v1.3.

The target and checkpoint MTP module are frozen.  Every MTP chain is constructed
from the same target forward states and exact committed prefix that supplies the
authoritative target records.  This program only writes event-aligned raw traces;
it never creates flattened examples or starts training.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from types import MethodType
import struct
import time
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from qwen35_mtp import load_checkpoint_mtp
from runtime_device import (
    clock_domain_definitions,
    hardware_topology_manifest,
    target_device_map,
)


SCHEMA = "gcrp2r_transformers_joint_capture_v1"
FORMAT_VERSION = 1
PREFIX_DOMAIN = b"GCRP2_PREFIX_V1"
TOP_K = 8
HIDDEN = 2048
EXPERTS = 256
DEFAULT_PROMPTS = [
    {
        "prompt_id": "pilot_paris",
        "source": "synthetic_factual",
        "family": "pilot_factual",
        "text": "What is the capital of France? Answer briefly.",
    },
    {
        "prompt_id": "pilot_arithmetic",
        "source": "synthetic_reasoning",
        "family": "pilot_reasoning",
        "text": "Calculate 17 * 23. Give the result and one short line of working.",
    },
    {
        "prompt_id": "pilot_code",
        "source": "synthetic_code",
        "family": "pilot_code",
        "text": "Write a Python function is_even(n) that returns whether n is even.",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def prefix_hash(tokens: list[int]) -> str:
    digest = hashlib.sha256(PREFIX_DOMAIN)
    for token in tokens:
        digest.update(struct.pack("<i", int(token)))
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    work = tensor.detach().cpu().contiguous()
    if work.dtype == torch.bfloat16:
        payload = work.view(torch.uint16).numpy().tobytes()
    else:
        payload = work.numpy().tobytes()
    return sha256_bytes(payload)


def dtype_name(tensor: torch.Tensor) -> str:
    names = {
        torch.bfloat16: "bf16",
        torch.float16: "float16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.int16: "int16",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    if tensor.dtype not in names:
        raise TypeError(f"unsupported tensor dtype {tensor.dtype}")
    return names[tensor.dtype]


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    work = tensor.detach().cpu().contiguous()
    if work.dtype == torch.bfloat16:
        return work.view(torch.uint16).numpy().tobytes()
    if work.dtype == torch.bool:
        return work.to(torch.uint8).numpy().tobytes()
    return work.numpy().tobytes()


class EventWriter:
    def __init__(
        self,
        path: Path,
        run_id: str,
        production_capture: bool,
        *,
        schema: str = SCHEMA,
        format_version: int = FORMAT_VERSION,
        capture_profile: str = "single_greedy_chain",
    ) -> None:
        self.path = path
        self.handle = path.open("w", encoding="utf-8")
        self.next_id = 0
        self.emit(
            "trace_header",
            trace_header=True,
            schema=schema,
            format_version=format_version,
            capture_profile=capture_profile,
            run_id=run_id,
            training_started=False,
            production_capture=production_capture,
        )

    def emit(self, kind: str, **fields: Any) -> int:
        event_id = self.next_id
        self.next_id += 1
        row = {"event": kind, "event_id": event_id, **fields}
        self.handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        if event_id % 4096 == 0:
            self.handle.flush()
        return event_id

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


class Sidecar:
    def __init__(self, path: Path, magic: bytes, events: EventWriter) -> None:
        if len(magic) > 16:
            raise ValueError("sidecar magic too long")
        self.path = path
        self.events = events
        self.handle = path.open("wb")
        self.handle.write(magic.ljust(64, b"\0"))
        self.offset = 64

    def write(
        self,
        tensor: torch.Tensor,
        *,
        role: str,
        parent_event_id: int,
        entity_kind: str,
        sequence_id: str,
        absolute_position: int | None = None,
        layer: int | None = None,
        depth: int | None = None,
    ) -> int:
        payload = tensor_bytes(tensor)
        padding = (-self.offset) % 64
        if padding:
            self.handle.write(b"\0" * padding)
            self.offset += padding
        offset = self.offset
        self.handle.write(payload)
        self.offset += len(payload)
        return self.events.emit(
            "tensor",
            parent_event_id=parent_event_id,
            entity_kind=entity_kind,
            tensor_role=role,
            sequence_id=sequence_id,
            absolute_position=absolute_position,
            target_layer=layer,
            engine_draft_depth=depth,
            native_dtype=dtype_name(tensor),
            shape=list(tensor.shape),
            logical_width=tensor.numel(),
            payload_file=str(Path("sidecars") / self.path.name),
            payload_offset=offset,
            payload_bytes=len(payload),
        )

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


class TargetCaptureHooks(AbstractContextManager):
    """Capture target block boundaries and exact expert-path constituents."""

    def __init__(self, model: Qwen3_5MoeForCausalLM) -> None:
        self.model = model
        self.rows: dict[int, dict[str, torch.Tensor]] = {
            layer: {} for layer in range(len(model.model.layers))
        }
        self.handles: list[Any] = []
        self.original_expert_forwards: list[tuple[Any, Any]] = []

    @staticmethod
    def candidate_outputs(module, hidden, selected_ids) -> torch.Tensor:
        hidden = hidden.reshape(-1, hidden.shape[-1])
        selected_ids = selected_ids.reshape(hidden.shape[0], -1)
        result = torch.zeros(
            hidden.shape[0],
            selected_ids.shape[1],
            hidden.shape[1],
            dtype=hidden.dtype,
            device=hidden.device,
        )
        for expert_id in torch.unique(selected_ids):
            positions = (selected_ids == expert_id).nonzero(as_tuple=False)
            token_index = positions[:, 0]
            slot_index = positions[:, 1]
            current = hidden[token_index]
            gate, up = F.linear(current, module.gate_up_proj[expert_id]).chunk(2, dim=-1)
            current = module.act_fn(gate) * up
            current = F.linear(current, module.down_proj[expert_id])
            result[token_index, slot_index] = current
        return result

    def __enter__(self):
        for layer_id, layer in enumerate(self.model.model.layers):
            row = self.rows[layer_id]

            def block_pre(_module, args, row=row):
                row["block_input_x"] = args[0].detach()

            def norm_pre(_module, args, row=row):
                row["post_attention_u"] = args[0].detach()

            def norm_post(_module, _args, output, row=row):
                row["router_input_a"] = output.detach()

            def gate_post(_module, _args, output, row=row):
                logits, weights, ids = output
                row["router_logits"] = logits.detach()
                row["selected_weights"] = weights.detach()
                row["selected_ids"] = ids.detach()

            experts_module = layer.mlp.experts
            original_expert_forward = experts_module.forward
            self.original_expert_forwards.append(
                (experts_module, original_expert_forward)
            )

            def captured_expert_forward(
                module, hidden_states, top_k_index, top_k_weights, row=row
            ):
                hidden_states = hidden_states.reshape(
                    -1, hidden_states.shape[-1]
                )
                top_k_index = top_k_index.reshape(
                    hidden_states.shape[0], -1
                )
                final_hidden_states = torch.zeros_like(hidden_states)
                unweighted = torch.zeros(
                    hidden_states.shape[0],
                    top_k_index.shape[1],
                    hidden_states.shape[1],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                with torch.no_grad():
                    expert_mask = torch.nn.functional.one_hot(
                        top_k_index, num_classes=module.num_experts
                    )
                    expert_mask = expert_mask.permute(2, 1, 0)
                    expert_hit = torch.greater(
                        expert_mask.sum(dim=(-1, -2)), 0
                    ).nonzero()
                for expert_idx in expert_hit:
                    expert_idx = expert_idx[0]
                    top_k_pos, token_idx = torch.where(
                        expert_mask[expert_idx]
                    )
                    current_state = hidden_states[token_idx]
                    gate, up = F.linear(
                        current_state, module.gate_up_proj[expert_idx]
                    ).chunk(2, dim=-1)
                    raw_output = module.act_fn(gate) * up
                    raw_output = F.linear(
                        raw_output, module.down_proj[expert_idx]
                    )
                    unweighted[token_idx, top_k_pos] = raw_output
                    weighted_output = (
                        raw_output
                        * top_k_weights[token_idx, top_k_pos, None]
                    )
                    final_hidden_states.index_add_(
                        0, token_idx,
                        weighted_output.to(final_hidden_states.dtype),
                    )
                row["routed_delta"] = final_hidden_states.detach()
                row["candidate_unweighted"] = unweighted.detach()
                row["candidate_weighted"] = (
                    unweighted * top_k_weights.unsqueeze(-1)
                ).detach()
                return final_hidden_states

            experts_module.forward = MethodType(
                captured_expert_forward, experts_module
            )

            def shared_post(_module, _args, output, row=row):
                row["shared_raw"] = output.detach()

            def shared_gate_post(_module, _args, output, row=row):
                row["shared_gate_logit"] = output.detach()

            def mlp_post(_module, _args, output, row=row):
                row["complete_moe_delta"] = output.detach()

            def block_post(_module, _args, output, row=row):
                row["post_moe_xplus"] = output.detach()

            self.handles.extend(
                [
                    layer.register_forward_pre_hook(block_pre),
                    layer.post_attention_layernorm.register_forward_pre_hook(norm_pre),
                    layer.post_attention_layernorm.register_forward_hook(norm_post),
                    layer.mlp.gate.register_forward_hook(gate_post),
                    layer.mlp.shared_expert.register_forward_hook(shared_post),
                    layer.mlp.shared_expert_gate.register_forward_hook(shared_gate_post),
                    layer.mlp.register_forward_hook(mlp_post),
                    layer.register_forward_hook(block_post),
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
        for module, original in self.original_expert_forwards:
            module.forward = original
        self.original_expert_forwards.clear()
        return False


def rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().item())


def cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, bool]:
    left = lhs.float().reshape(-1)
    right = rhs.float().reshape(-1)
    valid = float(left.norm()) > 1e-12 and float(right.norm()) > 1e-12
    if not valid:
        return 0.0, False
    return float(F.cosine_similarity(left, right, dim=0).item()), True


def load_target(
    model_path: Path,
    *,
    device: str | torch.device = "cuda",
) -> tuple[Qwen3_5MoeForCausalLM, Any]:
    """Load the frozen BF16 eager target on one explicit CPU or CUDA device."""

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
        device_map=target_device_map(device),
    )
    model.eval().requires_grad_(False)
    return model, config


class CaptureRun:
    def __init__(
        self,
        output: Path,
        run_id: str,
        model_path: Path,
        max_draft_depth: int,
        full_vocab_audit_percent: int,
        candidate_vector_audit_percent: int,
        prompt_history_tokens: int,
        production_capture: bool,
        *,
        event_schema: str = SCHEMA,
        event_format_version: int = FORMAT_VERSION,
        capture_profile: str = "single_greedy_chain",
    ) -> None:
        self.output = output
        self.run_id = run_id
        self.model_path = model_path
        self.max_draft_depth = max_draft_depth
        self.full_vocab_audit_percent = full_vocab_audit_percent
        self.candidate_vector_audit_percent = candidate_vector_audit_percent
        self.prompt_history_tokens = prompt_history_tokens
        self.production_capture = production_capture
        self.events = EventWriter(
            output / "events.jsonl",
            run_id,
            production_capture,
            schema=event_schema,
            format_version=event_format_version,
            capture_profile=capture_profile,
        )
        sidecars = output / "sidecars"
        sidecars.mkdir()
        self.target_states = Sidecar(sidecars / "target_states.bin", b"G2RTSTATE1", self.events)
        self.target_routes = Sidecar(sidecars / "target_routes.bin", b"G2RTROUTE1", self.events)
        self.target_candidates = Sidecar(sidecars / "target_candidates.bin", b"G2RTCAND1", self.events)
        self.mtp_states = Sidecar(sidecars / "mtp_states.bin", b"G2RMSTATE1", self.events)
        self.mtp_routes = Sidecar(sidecars / "mtp_routes.bin", b"G2RMROUTE1", self.events)
        self.mtp_vocab = Sidecar(sidecars / "mtp_vocab.bin", b"G2RMVOCAB1", self.events)
        self.sequence_rows: list[dict[str, Any]] = []
        self.target_layer_rows = 0
        self.prompt_route_layer_rows = 0
        self.generated_positions = 0
        self.sequence_count = 0
        self.mtp_nodes = 0
        self.ready_mtp_nodes: list[dict[str, int]] = []

    def write_tensor(self, sidecar: Sidecar, tensor: torch.Tensor, **kwargs) -> int:
        return sidecar.write(tensor, **kwargs)

    def flush_target_call(
        self,
        hooks: TargetCaptureHooks,
        output: Any,
        *,
        sequence_id: str,
        request_id: str,
        phase: str,
        start_position: int,
        full_prefix_tokens: list[int],
        call_input_ids: list[int],
        call_id: int,
    ) -> list[torch.Tensor]:
        final_hidden_rows = [value.detach() for value in output.hidden_states[-1][0]]
        prompt_route_start = max(0, len(call_input_ids) - self.prompt_history_tokens)
        for local_index, token_id in enumerate(call_input_ids):
            absolute_position = start_position + local_index
            prefix = full_prefix_tokens[: absolute_position + 1]
            is_prompt = phase == "PREFILL_TARGET"
            token_event = self.events.emit(
                "target_token",
                run_id=self.run_id,
                sequence_id=sequence_id,
                request_id=request_id,
                target_call_id=call_id,
                phase=phase,
                authoritative_flag=True,
                absolute_sequence_position=absolute_position,
                committed_token_position=absolute_position,
                committed_input_token_id=int(token_id),
                is_prompt=is_prompt,
                authoritative_prefix_hash=prefix_hash(prefix),
                graph_execution_id=f"target:{sequence_id}:{call_id}",
            )
            persist_prompt_route = is_prompt and local_index >= prompt_route_start
            for layer_id, layer in enumerate(self.model.model.layers):
                row = hooks.rows[layer_id]
                required = {
                    "block_input_x", "post_attention_u", "router_input_a",
                    "router_logits", "selected_weights", "selected_ids",
                    "routed_delta", "candidate_unweighted",
                    "candidate_weighted", "shared_raw",
                    "shared_gate_logit", "complete_moe_delta",
                    "post_moe_xplus",
                }
                missing = required - row.keys()
                if missing:
                    raise RuntimeError(
                        f"target layer {layer_id} missing hooks {sorted(missing)}"
                    )
                if is_prompt and not persist_prompt_route:
                    continue
                index = local_index
                logits = row["router_logits"][index]
                probabilities = torch.softmax(logits.float(), dim=-1)
                ids = row["selected_ids"][index]
                weights = row["selected_weights"][index]
                route_tensors = {
                    "raw_target_router_logits": logits,
                    "full_target_router_probabilities": probabilities,
                    "selected_expert_ids": ids.to(torch.int32),
                    "selected_execution_weights": weights,
                }
                if is_prompt:
                    layer_event = self.events.emit(
                        "prompt_route_layer",
                        run_id=self.run_id,
                        sequence_id=sequence_id,
                        request_id=request_id,
                        target_token_event_id=token_event,
                        target_call_id=call_id,
                        absolute_sequence_position=absolute_position,
                        target_layer=layer_id,
                        block_type=layer.block_type,
                        cycle_phase=phase,
                        authoritative_prefix_hash=prefix_hash(prefix),
                        record_valid=True,
                    )
                    for role, tensor in route_tensors.items():
                        self.write_tensor(
                            self.target_routes, tensor, role=role,
                            parent_event_id=layer_event,
                            entity_kind="prompt_route_layer",
                            sequence_id=sequence_id,
                            absolute_position=absolute_position,
                            layer=layer_id,
                        )
                    self.prompt_route_layer_rows += 1
                    continue

                u = row["post_attention_u"][0, index]
                a = row["router_input_a"][0, index]
                x = row["block_input_x"][0, index]
                xplus = row["post_moe_xplus"][0, index]
                routed = row["routed_delta"][index]
                shared_raw = row["shared_raw"][index]
                shared_gate_logit = row["shared_gate_logit"][index].reshape(1)
                shared_gate = torch.sigmoid(shared_gate_logit)
                shared = shared_raw * shared_gate
                complete = row["complete_moe_delta"][0, index]
                unweighted = row["candidate_unweighted"][index]
                weighted = row["candidate_weighted"][index]
                weighted_rms = weighted.float().square().mean(dim=-1).sqrt()
                fractions = weighted_rms / (weighted_rms.sum() + 1e-12)
                candidate_cos, candidate_valid = [], []
                for candidate in weighted:
                    value, valid = cosine(candidate, routed)
                    candidate_cos.append(value)
                    candidate_valid.append(valid)
                cos_r, valid_r = cosine(routed, u)
                cos_s, valid_s = cosine(shared, u)
                audit_value = int(
                    hashlib.sha256(
                        f"{self.run_id}|{sequence_id}|{absolute_position}".encode()
                    ).hexdigest()[:8],
                    16,
                ) % 100
                audit_subset = (
                    self.generated_positions == 0
                    or audit_value < self.candidate_vector_audit_percent
                )
                ready = [
                    value["event_id"]
                    for value in self.ready_mtp_nodes
                    if value["gcrp_target_position"]
                    in (absolute_position + 1, absolute_position + 2)
                ]
                layer_event = self.events.emit(
                    "target_layer",
                    run_id=self.run_id,
                    sequence_id=sequence_id,
                    request_id=request_id,
                    target_token_event_id=token_event,
                    target_call_id=call_id,
                    absolute_sequence_position=absolute_position,
                    committed_token_position=absolute_position,
                    target_layer=layer_id,
                    block_type=layer.block_type,
                    cycle_phase=phase,
                    authoritative_prefix_hash=prefix_hash(prefix),
                    record_valid=True,
                    audit_subset=audit_subset,
                    candidate_expert_ids=[int(v) for v in ids.cpu().tolist()],
                    post_mixer_pre_moe_residual_rms=rms(u),
                    cos_routed_output_to_pre_moe_residual=cos_r,
                    cos_routed_output_valid=valid_r,
                    cos_shared_output_to_pre_moe_residual=cos_s,
                    cos_shared_output_valid=valid_s,
                    candidate_unweighted_output_rms=[
                        float(v) for v in
                        unweighted.float().square().mean(dim=-1).sqrt().cpu().tolist()
                    ],
                    candidate_weighted_output_rms=[
                        float(v) for v in weighted_rms.cpu().tolist()
                    ],
                    candidate_weighted_output_fraction=[
                        float(v) for v in fractions.cpu().tolist()
                    ],
                    candidate_output_cosine_to_routed_sum=candidate_cos,
                    candidate_output_cosine_valid=candidate_valid,
                    shared_expert_gate_scalar=float(shared_gate.item()),
                    clock_domain_id="synchronous_transformers_execution_order",
                    router_ready_event_order=token_event,
                    early_inputs_ready_event_order=token_event,
                    routed_output_ready_event_order=self.events.next_id,
                    late_inputs_ready_event_order=self.events.next_id,
                    ready_mtp_node_event_ids=ready,
                    early_prediction_emitted_timestamp=None,
                    late_prediction_emitted_timestamp=None,
                )
                state_tensors = {
                    "post_attention_residual_u": u,
                    "normalized_target_router_input_a": a,
                    "post_moe_residual_xplus": xplus,
                    "routed_expert_output_delta_r": routed,
                    "shared_expert_output_delta_s": shared,
                }
                if audit_subset:
                    state_tensors.update({
                        "block_input_residual_x": x,
                        "complete_moe_delta": complete,
                    })
                for role, tensor in state_tensors.items():
                    self.write_tensor(
                        self.target_states, tensor, role=role,
                        parent_event_id=layer_event,
                        entity_kind="target_layer",
                        sequence_id=sequence_id,
                        absolute_position=absolute_position,
                        layer=layer_id,
                    )
                for role, tensor in route_tensors.items():
                    self.write_tensor(
                        self.target_routes, tensor, role=role,
                        parent_event_id=layer_event,
                        entity_kind="target_layer",
                        sequence_id=sequence_id,
                        absolute_position=absolute_position,
                        layer=layer_id,
                    )
                if audit_subset:
                    candidate_tensors = {
                        "candidate_unweighted_output_vectors": unweighted,
                        "candidate_weighted_output_vectors": weighted,
                        "shared_expert_output_before_gate": shared_raw,
                        "shared_expert_gate_logit": shared_gate_logit,
                    }
                    for role, tensor in candidate_tensors.items():
                        self.write_tensor(
                            self.target_candidates, tensor, role=role,
                            parent_event_id=layer_event,
                            entity_kind="target_layer",
                            sequence_id=sequence_id,
                            absolute_position=absolute_position,
                            layer=layer_id,
                        )
                self.target_layer_rows += 1
            if not is_prompt:
                self.generated_positions += 1
        return final_hidden_rows

    def write_mtp_node(
        self,
        *,
        sequence_id: str,
        request_id: str,
        cycle_id: int,
        depth: int,
        base_position: int,
        root_prefix_tokens: list[int],
        draft_prefix_tokens: list[int],
        module_output: dict[str, Any],
        cumulative_log_probability: float,
    ) -> tuple[int, int, float]:
        vocab = module_output["vocabulary_logits"][0, -1].float()
        log_probs = torch.log_softmax(vocab, dim=-1)
        top_log_probs, top_ids = torch.topk(log_probs, 64)
        draft_token = int(top_ids[0].item())
        draft_log_probability = float(top_log_probs[0].item())
        probs = log_probs.exp()
        entropy = float((-(probs * log_probs)).sum().item())
        top8_mass = float(top_log_probs[:8].exp().sum().item())
        margin = float((top_log_probs[0] - top_log_probs[1]).item())
        router_logits = module_output["router_logits"][0, -1]
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        router_entropy = float(
            (-(router_probs * torch.log(router_probs.clamp_min(1e-30)))).sum().item()
        )
        cumulative = cumulative_log_probability + draft_log_probability
        router_position = base_position + depth
        vocabulary_position = base_position + depth + 1
        conditioning = (
            "authoritative_context_equivalent"
            if depth == 1
            else "position_aligned_speculative_context"
        )
        node_event = self.events.emit(
            "mtp_node",
            run_id=self.run_id,
            sequence_id=sequence_id,
            request_id=request_id,
            verifier_cycle_id=cycle_id,
            committed_prefix_position=base_position,
            authoritative_prefix_hash=prefix_hash(root_prefix_tokens),
            engine_draft_depth=depth,
            router_state_position=router_position,
            mtp_input_token_position=router_position,
            vocabulary_prediction_position=vocabulary_position,
            draft_token_position=vocabulary_position,
            gcrp_target_position=router_position,
            gcrp_horizon=depth,
            draft_branch_id=0,
            parent_draft_branch_id=None if depth == 1 else 0,
            execution_phase="MTP_DRAFT",
            conditioning_class=conditioning,
            draft_prefix_length=len(draft_prefix_tokens),
            draft_prefix_token_ids=[int(v) for v in draft_prefix_tokens],
            draft_prefix_hash=prefix_hash(root_prefix_tokens + draft_prefix_tokens),
            draft_token_id=draft_token,
            draft_token_log_probability=draft_log_probability,
            top_token_probability=float(math.exp(draft_log_probability)),
            top1_top2_logprob_margin=margin,
            vocabulary_entropy=entropy,
            vocabulary_entropy_valid=True,
            vocabulary_top8_mass=top8_mass,
            vocabulary_top8_mass_valid=True,
            mtp_router_entropy=router_entropy,
            mtp_router_logit_rms=rms(router_logits),
            cumulative_draft_logprobability=cumulative,
            source_ready_event_order=self.events.next_id,
            source_graph_node_id="checkpoint_mtp.layers.0",
            source_mtp_layer_id=0,
            branch_probability=float(math.exp(cumulative)),
            feature_available=True,
            acceptance_fields_present=False,
            record_valid=True,
        )
        state_tensors = {
            "mtp_fused_state": module_output["fused_hidden"][0, -1],
            "mtp_router_input": module_output["router_input"][0, -1],
            "mtp_post_ffn_hidden": module_output["post_ffn_hidden"][0, -1],
            "mtp_vocabulary_head_input": module_output["head_input"][0, -1],
        }
        for role, tensor in state_tensors.items():
            self.write_tensor(
                self.mtp_states,
                tensor,
                role=role,
                parent_event_id=node_event,
                entity_kind="mtp_node",
                sequence_id=sequence_id,
                absolute_position=router_position,
                depth=depth,
            )
        route_tensors = {
            "raw_mtp_router_logits": router_logits,
            "full_mtp_router_probabilities": router_probs,
            "mtp_selected_expert_ids": module_output["top8_ids"][0, -1].to(torch.int32),
            "mtp_selected_execution_weights": module_output["top8_weights"][0, -1],
        }
        for role, tensor in route_tensors.items():
            self.write_tensor(
                self.mtp_routes,
                tensor,
                role=role,
                parent_event_id=node_event,
                entity_kind="mtp_node",
                sequence_id=sequence_id,
                absolute_position=router_position,
                depth=depth,
            )
        vocab_tensors = {
            "vocab_top64_token_ids": top_ids.to(torch.int32),
            "vocab_top64_log_probabilities": top_log_probs,
        }
        audit_key = int(hashlib.sha256(f"{sequence_id}:{cycle_id}:{depth}".encode()).hexdigest()[:8], 16)
        if (cycle_id == 0 and depth == 1) or (
            audit_key % 100 < self.full_vocab_audit_percent
        ):
            vocab_tensors["full_vocabulary_logits_audit"] = module_output[
                "vocabulary_logits"
            ][0, -1]
        for role, tensor in vocab_tensors.items():
            self.write_tensor(
                self.mtp_vocab,
                tensor,
                role=role,
                parent_event_id=node_event,
                entity_kind="mtp_node",
                sequence_id=sequence_id,
                absolute_position=vocabulary_position,
                depth=depth,
            )
        self.mtp_nodes += 1
        self.ready_mtp_nodes.append(
            {"event_id": node_event, "gcrp_target_position": router_position}
        )
        return node_event, draft_token, cumulative

    def close(self) -> None:
        for sidecar in (
            self.target_states,
            self.target_routes,
            self.target_candidates,
            self.mtp_states,
            self.mtp_routes,
            self.mtp_vocab,
        ):
            sidecar.close()
        self.events.close()


def prompt_ids(tokenizer, text: str) -> list[int]:
    messages = [{"role": "user", "content": text}]
    value = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if hasattr(value, "keys"):
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


@torch.no_grad()
def capture_sequence(
    run: CaptureRun,
    target: Qwen3_5MoeForCausalLM,
    mtp,
    hooks: TargetCaptureHooks,
    tokenizer,
    prompt: dict[str, str],
    sequence_index: int,
    max_new_tokens: int,
) -> None:
    ordinal = int(prompt.get("ordinal", sequence_index - 1))
    sample_id = str(prompt.get("sample_id", f"pilot-{sequence_index:04d}"))
    prompt_sha = str(prompt.get("prompt_sha256", sha256_bytes(sample_id.encode())))
    sequence_id = f"tfprod-seq{ordinal:06d}-{prompt_sha[:8]}"
    request_id = f"tfprod-req{ordinal:06d}-{prompt_sha[:8]}"
    prompt_token_ids = (
        [int(value) for value in prompt["prompt_token_ids"]]
        if "prompt_token_ids" in prompt
        else prompt_ids(tokenizer, prompt["text"])
    )
    group_id = str(
        prompt.get(
            "group_id",
            prompt.get("conversation_group_id", prompt.get("family", sample_id)),
        )
    )
    run.ready_mtp_nodes.clear()
    run.sequence_count += 1
    sequence_start = run.events.emit(
        "sequence_start",
        run_id=run.run_id,
        sequence_id=sequence_id,
        request_id=request_id,
        prompt_id=sample_id,
        manifest_ordinal=ordinal,
        conversation_or_source_family_id=str(
            prompt.get("conversation_group_id", group_id)
        ),
        dataset_item_family_id=group_id,
        dataset_source=str(prompt.get("source", "synthetic_pilot")),
        domain_label=prompt.get("domain"),
        language_label=prompt.get("language"),
        original_split=prompt.get("original_split"),
        external_evaluation=bool(prompt.get("external_evaluation", False)),
        prompt_token_ids=prompt_token_ids,
        prompt_hash=prefix_hash(prompt_token_ids),
        source_prompt_sha256=prompt_sha,
        split_group_id=group_id,
        offline_split_assigned=False,
    )

    committed_tokens = list(prompt_token_ids)
    generated_tokens: list[int] = []
    final_hidden_history: list[torch.Tensor] = []
    target_cache = None
    call_id = 0
    cycle_id = 0

    def target_call(token_ids: list[int], phase: str, start_position: int):
        nonlocal target_cache, call_id
        hooks.clear()
        result = target(
            input_ids=torch.tensor([token_ids], device=target.device, dtype=torch.long),
            past_key_values=target_cache,
            use_cache=True,
            output_hidden_states=True,
            output_router_logits=True,
            return_dict=True,
        )
        target_cache = result.past_key_values
        hidden_rows = run.flush_target_call(
            hooks,
            result,
            sequence_id=sequence_id,
            request_id=request_id,
            phase=phase,
            start_position=start_position,
            full_prefix_tokens=committed_tokens,
            call_input_ids=token_ids,
            call_id=call_id,
        )
        final_hidden_history.extend(hidden_rows)
        call_id += 1
        return result

    current_output = target_call(prompt_token_ids, "PREFILL_TARGET", 0)
    computed_length = len(prompt_token_ids)
    pending_token: int | None = None
    eos_id = tokenizer.eos_token_id
    termination = "max_new_tokens"

    while len(generated_tokens) < max_new_tokens:
        if pending_token is not None:
            current_output = target_call(
                [pending_token], "VERIFICATION_TARGET", computed_length
            )
            computed_length += 1
            pending_token = None
            if len(generated_tokens) >= max_new_tokens:
                break

        base_position = computed_length - 1
        root_prefix = list(committed_tokens[:computed_length])
        bonus_token = int(current_output.logits[0, -1].argmax().item())
        mtp_input_ids = root_prefix[1:] + [bonus_token]
        previous_hidden = torch.stack(final_hidden_history[:computed_length], dim=0).unsqueeze(0)
        position_ids = torch.arange(
            computed_length, device=target.device, dtype=torch.long
        ).unsqueeze(0)
        mtp_output = mtp(
            torch.tensor([mtp_input_ids], device=target.device, dtype=torch.long),
            previous_hidden,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
        )
        mtp_cache = mtp_output["past_key_values"]
        node_ids: list[int] = []
        drafts: list[int] = []
        cumulative = 0.0
        draft_prefix = [bonus_token]
        node_id, draft, cumulative = run.write_mtp_node(
            sequence_id=sequence_id,
            request_id=request_id,
            cycle_id=cycle_id,
            depth=1,
            base_position=base_position,
            root_prefix_tokens=root_prefix,
            draft_prefix_tokens=draft_prefix,
            module_output=mtp_output,
            cumulative_log_probability=cumulative,
        )
        node_ids.append(node_id)
        drafts.append(draft)
        previous_mtp_hidden = mtp_output["head_input"][:, -1:, :]
        for depth in range(2, run.max_draft_depth + 1):
            position = base_position + depth - 1
            mtp_output = mtp(
                torch.tensor([[drafts[-1]]], device=target.device, dtype=torch.long),
                previous_mtp_hidden,
                position_ids=torch.tensor([[position]], device=target.device),
                past_key_values=mtp_cache,
                use_cache=True,
            )
            mtp_cache = mtp_output["past_key_values"]
            draft_prefix = [bonus_token] + drafts
            node_id, draft, cumulative = run.write_mtp_node(
                sequence_id=sequence_id,
                request_id=request_id,
                cycle_id=cycle_id,
                depth=depth,
                base_position=base_position,
                root_prefix_tokens=root_prefix,
                draft_prefix_tokens=draft_prefix,
                module_output=mtp_output,
                cumulative_log_probability=cumulative,
            )
            node_ids.append(node_id)
            drafts.append(draft)
            previous_mtp_hidden = mtp_output["head_input"][:, -1:, :]

        run.events.emit(
            "mtp_cycle_ready",
            run_id=run.run_id,
            sequence_id=sequence_id,
            request_id=request_id,
            verifier_cycle_id=cycle_id,
            committed_prefix_position=base_position,
            authoritative_prefix_hash=prefix_hash(root_prefix),
            mtp_node_event_ids=node_ids,
            synchronous_before_verification=True,
        )

        accepted_drafts = 0
        first_rejection_depth: int | None = None
        generated_tokens.append(bonus_token)
        committed_tokens.append(bonus_token)
        current_output = target_call(
            [bonus_token], "VERIFICATION_TARGET", computed_length
        )
        computed_length += 1
        if bonus_token == eos_id:
            termination = "eos"
            first_rejection_depth = 1
        else:
            for index, draft_token in enumerate(drafts):
                depth = index + 1
                truth = int(current_output.logits[0, -1].argmax().item())
                if truth != draft_token:
                    first_rejection_depth = depth
                    if len(generated_tokens) < max_new_tokens:
                        generated_tokens.append(truth)
                        committed_tokens.append(truth)
                        pending_token = truth
                    break
                accepted_drafts += 1
                if len(generated_tokens) >= max_new_tokens:
                    pending_token = draft_token
                    break
                generated_tokens.append(draft_token)
                committed_tokens.append(draft_token)
                pending_token = draft_token
                if draft_token == eos_id:
                    termination = "eos"
                    break
                if depth < len(drafts) and len(generated_tokens) < max_new_tokens:
                    current_output = target_call(
                        [draft_token], "VERIFICATION_TARGET", computed_length
                    )
                    computed_length += 1
                    pending_token = None
            else:
                pending_token = drafts[-1]

        for depth, node_id in enumerate(node_ids, 1):
            run.events.emit(
                "mtp_acceptance_label",
                run_id=run.run_id,
                sequence_id=sequence_id,
                request_id=request_id,
                verifier_cycle_id=cycle_id,
                mtp_node_event_id=node_id,
                engine_draft_depth=depth,
                accepted_prefix_length=accepted_drafts,
                accepted_prefix_label=accepted_drafts >= depth,
                draft_token_accepted=accepted_drafts >= depth,
                acceptance_label_valid=True,
                first_rejection_depth=first_rejection_depth,
                label_only=True,
            )
        run.events.emit(
            "mtp_cycle_resolved",
            run_id=run.run_id,
            sequence_id=sequence_id,
            request_id=request_id,
            verifier_cycle_id=cycle_id,
            accepted_draft_count=accepted_drafts,
            first_rejection_depth=first_rejection_depth,
            bonus_token_id=bonus_token,
            draft_token_ids=drafts,
            node_event_ids=node_ids,
        )
        cycle_id += 1
        if termination == "eos":
            break

    if pending_token is not None and computed_length < len(committed_tokens):
        current_output = target_call(
            [pending_token], "VERIFICATION_TARGET", computed_length
        )
        computed_length += 1

    generated_tokens = generated_tokens[:max_new_tokens]
    run.events.emit(
        "sequence_end",
        run_id=run.run_id,
        sequence_id=sequence_id,
        request_id=request_id,
        sequence_start_event_id=sequence_start,
        prompt_token_ids=prompt_token_ids,
        committed_generated_token_ids=generated_tokens,
        full_committed_token_ids=prompt_token_ids + generated_tokens,
        prompt_length=len(prompt_token_ids),
        generated_length=len(generated_tokens),
        captured_authoritative_position_count=computed_length,
        termination_reason=termination,
        eos_position=(
            len(prompt_token_ids) + generated_tokens.index(eos_id)
            if eos_id in generated_tokens
            else None
        ),
    )


def build_manifest(
    args,
    target,
    config,
    tokenizer,
    mtp,
    run: CaptureRun,
    started_at: str,
) -> dict[str, Any]:
    model_files = []
    if args.model_weight_manifest:
        weight_manifest = json.loads(args.model_weight_manifest.read_text())
        model_files = list(weight_manifest["files"])
        (run.output / "MODEL_WEIGHT_SHA256SUMS.json").write_text(
            json.dumps(weight_manifest, indent=2, sort_keys=True) + "\n"
        )
    else:
        for path in sorted(args.model.glob("*.safetensors")):
            row = {"name": path.name, "bytes": path.stat().st_size}
            if args.hash_all_weights:
                row["sha256"] = sha256_file(path)
            model_files.append(row)
    identity_files = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ):
        path = args.model / name
        if path.is_file():
            identity_files[name] = sha256_file(path)

    router_tensors: dict[str, torch.Tensor] = {}
    router_hashes = []
    for layer_id, layer in enumerate(target.model.layers):
        weight = layer.mlp.gate.weight.detach().float().cpu().contiguous()
        key = f"target_router_weight.layer_{layer_id:02d}"
        router_tensors[key] = weight
        router_hashes.append(tensor_hash(weight))
        router_tensors[f"target_router_bias.layer_{layer_id:02d}"] = torch.zeros(
            weight.shape[0], dtype=torch.float32
        )
    mtp_weight = mtp.decoder.layers[0].mlp.gate.weight.detach().float().cpu().contiguous()
    router_tensors["mtp_router_weight"] = mtp_weight
    router_tensors["mtp_router_bias"] = torch.zeros(
        mtp_weight.shape[0], dtype=torch.float32
    )
    router_path = run.output / "router_artifacts.safetensors"
    save_file(router_tensors, router_path)

    graph_manifest = {
        "checkpoint_mtp_tensor_mapping": mtp.load_report,
        "semantic_position_rule": {
            "source_target_state": "p",
            "mtp_input_and_router_state": "p+1",
            "mtp_vocabulary_and_draft": "p+2",
        },
        "chain": "one autoregressive branch",
        "depth": args.max_draft_depth,
    }
    manifest = {
        "schema": "gcrp2r_transformers_run_manifest_v1",
        "format_version": FORMAT_VERSION,
        "run_id": run.run_id,
        "capture_timestamp_utc": started_at,
        "capture_completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_identifier": str(args.model),
        "model_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "model_files": model_files,
        "identity_file_hashes": identity_files,
        "model_config_hash": identity_files.get("config.json"),
        "tokenizer_identifier": str(args.model),
        "tokenizer_hash": identity_files.get("tokenizer.json"),
        "chat_template_hash": sha256_bytes((tokenizer.chat_template or "").encode()),
        "router_tensor_hashes": router_hashes,
        "router_artifact_file": router_path.name,
        "router_artifact_sha256": sha256_file(router_path),
        "engine": "transformers",
        "engine_version": transformers.__version__,
        "engine_commit": getattr(transformers, "__commit__", None),
        "engine_build_flags": {
            "attention_implementation": "eager",
            "experts_implementation": "eager",
            "torch_compile": False,
        },
        "quantization_format": "BF16 frozen checkpoint",
        "activation_dtypes": ["bfloat16", "float32"],
        "random_seed": args.seed,
        "decoding_mode": "deterministic greedy custom depth-six MTP verification",
        "sampling_parameters": {"temperature": 0.0, "top_p": 1.0},
        "mtp_enabled": True,
        "mtp_draft_max": args.max_draft_depth,
        "mtp_graph_manifest": graph_manifest,
        "mtp_graph_manifest_hash": sha256_bytes(canonical_json(graph_manifest)),
        "hardware_topology_manifest": hardware_topology_manifest(target.device),
        "clock_domain_definitions": clock_domain_definitions(target.device),
        "prefix_hash_algorithm": "SHA-256 over domain tag plus little-endian int32 token IDs",
        "prefix_hash_version": 1,
        "dimensions": {
            "target_layers": len(target.model.layers),
            "experts": config.num_experts,
            "top_k": config.num_experts_per_tok,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "mtp_layers": 1,
        },
        "capture_policy": {
            "all_target_u_a_xplus": True,
            "all_target_routed_and_shared_residuals": True,
            "candidate_vector_audit_percent": args.candidate_vector_audit_percent,
            "prompt_history_tokens": args.prompt_history_tokens,
            "full_vocabulary_audit_percent": args.full_vocab_audit_percent,
            "splits_assigned_during_capture": False,
        },
        "counts": {
            "target_layer_rows": run.target_layer_rows,
            "mtp_nodes": run.mtp_nodes,
            "sequences": run.sequence_count,
            "prompt_route_layer_rows": run.prompt_route_layer_rows,
            "generated_positions": run.generated_positions,
        },
        "training_started": False,
        "production_capture": args.production_capture,
        "stop_before_training": True,
        "source_prompt_manifest": (
            str(args.prompt_manifest) if args.prompt_manifest else None
        ),
        "source_prompt_manifest_sha256": (
            sha256_file(args.prompt_manifest) if args.prompt_manifest else None
        ),
        "manifest_slice": {"skip": args.skip, "limit": args.limit},
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="gcrp2r_transformers_joint_pilot_v1")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-draft-depth", type=int, default=6)
    parser.add_argument("--full-vocab-audit-percent", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-all-weights", action="store_true")
    parser.add_argument("--model-weight-manifest", type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--candidate-vector-audit-percent", type=int, default=5)
    parser.add_argument("--prompt-history-tokens", type=int, default=8)
    parser.add_argument("--production-capture", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.max_draft_depth != 6:
        raise ValueError("the validated reference graph requires depth six")
    args.output.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    target, config = load_target(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    mtp = load_checkpoint_mtp(
        args.model,
        device=target.device,
        embed_tokens=target.model.embed_tokens,
        lm_head=target.lm_head,
        config=config,
    )
    run = CaptureRun(
        args.output,
        args.run_id,
        args.model,
        args.max_draft_depth,
        args.full_vocab_audit_percent,
        args.candidate_vector_audit_percent,
        args.prompt_history_tokens,
        args.production_capture,
    )
    run.model = target
    if args.prompt_manifest:
        all_prompts = [
            json.loads(line)
            for line in args.prompt_manifest.read_text().splitlines()
            if line.strip()
        ]
        stop = None if args.limit is None else args.skip + args.limit
        prompts = all_prompts[args.skip:stop]
        if not prompts:
            raise ValueError("selected prompt-manifest slice is empty")
    else:
        prompts = DEFAULT_PROMPTS
    try:
        with TargetCaptureHooks(target) as hooks:
            for index, prompt in enumerate(prompts, 1):
                capture_sequence(
                    run,
                    target,
                    mtp,
                    hooks,
                    tokenizer,
                    prompt,
                    index,
                    args.max_new_tokens,
                )
    finally:
        run.close()

    manifest = build_manifest(args, target, config, tokenizer, mtp, run, started_at)
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    checksum_files = [
        path
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    with (args.output / "SHA256SUMS").open("w") as handle:
        for path in checksum_files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(args.output)}\n")
    (args.output / "STOP_BEFORE_TRAINING.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "capture_complete": True,
                "audit_complete": False,
                "production_capture_started": args.production_capture,
                "training_started": False,
                "instruction": "Run the blocking auditor; do not train.",
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
                "target_layer_rows": run.target_layer_rows,
                "mtp_nodes": run.mtp_nodes,
                "generated_positions": run.generated_positions,
                "production_capture": args.production_capture,
                "training_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
