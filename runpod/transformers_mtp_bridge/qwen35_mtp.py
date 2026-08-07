"""Frozen Qwen3.6/Qwen3.5-MoE checkpoint MTP module for Transformers.

This module reuses the official Transformers Qwen3.5-MoE decoder layer and
loads only the checkpoint mtp.* tensors. Token embeddings and the language
model head are shared with the frozen target checkpoint, matching SGLang.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextModel,
)

from mtp_router_semantics import native_topk_router_distribution


MTP_PREFIX = "mtp."
SHARED_EMBED_KEY = "model.language_model.embed_tokens.weight"
SHARED_LM_HEAD_KEY = "lm_head.weight"


def _checkpoint_index(model_path: str | Path) -> dict[str, str]:
    path = Path(model_path)
    return json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]


def read_checkpoint_tensor(
    model_path: str | Path, index: dict[str, str], key: str
) -> torch.Tensor:
    path = Path(model_path)
    shard = index[key]
    with safe_open(path / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def load_shared_embedding_and_head(
    model_path: str | Path,
) -> tuple[nn.Embedding, nn.Linear, Any]:
    """Load the frozen target embedding/head without materializing the target model."""
    model_path = Path(model_path)
    full_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config = copy.deepcopy(full_config.text_config)
    index = _checkpoint_index(model_path)
    embedding_weight = read_checkpoint_tensor(model_path, index, SHARED_EMBED_KEY)
    head_weight = read_checkpoint_tensor(model_path, index, SHARED_LM_HEAD_KEY)

    embedding = nn.Embedding(
        config.vocab_size,
        config.hidden_size,
        padding_idx=config.pad_token_id,
        device="meta",
        dtype=embedding_weight.dtype,
    )
    embedding.weight = nn.Parameter(embedding_weight, requires_grad=False)
    lm_head = nn.Linear(
        config.hidden_size,
        config.vocab_size,
        bias=False,
        device="meta",
        dtype=head_weight.dtype,
    )
    lm_head.weight = nn.Parameter(head_weight, requires_grad=False)
    return embedding, lm_head, config


class Qwen35CheckpointMTP(nn.Module):
    """One-layer full-attention MTP module with exact checkpoint weight names."""

    def __init__(
        self,
        config: Any,
        embed_tokens: nn.Embedding,
        lm_head: nn.Linear,
    ) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

        self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        mtp_config = copy.deepcopy(config)
        mtp_config.num_hidden_layers = 1
        mtp_config.full_attention_interval = 1
        mtp_config.layer_types = ["full_attention"]
        mtp_config._attn_implementation = os.environ.get("GCRP_MTP_ATTENTION", "eager")
        mtp_config._experts_implementation = "eager"
        # Avoid allocating a second 248k x 2048 embedding. It is replaced by
        # the shared frozen target embedding immediately after construction.
        original_vocab_size = mtp_config.vocab_size
        mtp_config.vocab_size = 1
        self.decoder = Qwen3_5MoeTextModel(mtp_config)
        self.decoder.embed_tokens = embed_tokens
        self.decoder.config.vocab_size = original_vocab_size
        self.decoder.config._attn_implementation = mtp_config._attn_implementation

        self._router_input: torch.Tensor | None = None
        self._post_ffn_hidden: torch.Tensor | None = None
        self.decoder.layers[0].mlp.register_forward_pre_hook(
            self._capture_router_input
        )
        self.decoder.layers[0].register_forward_hook(self._capture_post_ffn_hidden)

    def _capture_router_input(self, _module, args) -> None:
        self._router_input = args[0]

    def _capture_post_ffn_hidden(self, _module, _args, output) -> None:
        self._post_ffn_hidden = output

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        previous_hidden: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        past_key_values: Any | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2 or previous_hidden.ndim != 3:
            raise ValueError("expected input_ids [B,T] and previous_hidden [B,T,D]")
        if input_ids.shape != previous_hidden.shape[:2]:
            raise ValueError("input IDs and previous hidden must align by batch/token")

        token_embedding = self.embed_tokens(input_ids)
        normalized_embedding = self.pre_fc_norm_embedding(token_embedding)
        normalized_previous_hidden = self.pre_fc_norm_hidden(previous_hidden)
        fused_hidden = self.fc(
            torch.cat((normalized_embedding, normalized_previous_hidden), dim=-1)
        )

        self._router_input = None
        self._post_ffn_hidden = None
        output = self.decoder(
            inputs_embeds=fused_hidden,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_router_logits=True,
            return_dict=True,
        )
        if self._router_input is None:
            raise RuntimeError("MTP router-input hook did not fire")
        if self._post_ffn_hidden is None:
            raise RuntimeError("MTP decoder-output hook did not fire")
        if not output.router_logits or len(output.router_logits) != 1:
            raise RuntimeError("expected exactly one MTP router-logit tensor")

        head_input = output.last_hidden_state
        vocabulary_logits = self.lm_head(head_input)
        router_logits = output.router_logits[0].reshape(
            input_ids.shape[0], input_ids.shape[1], -1
        )
        router_probabilities, selected_weights, selected_ids = (
            native_topk_router_distribution(
                router_logits, self.config.num_experts_per_tok
            )
        )
        return {
            "token_embedding": token_embedding,
            "normalized_embedding": normalized_embedding,
            "normalized_previous_hidden": normalized_previous_hidden,
            "fused_hidden": fused_hidden,
            "router_input": self._router_input,
            "router_logits": router_logits,
            "router_probabilities": router_probabilities,
            "top8_ids": selected_ids,
            "top8_weights": selected_weights,
            "post_ffn_hidden": self._post_ffn_hidden,
            "head_input": head_input,
            "vocabulary_logits": vocabulary_logits,
            "past_key_values": output.past_key_values,
        }


def load_checkpoint_mtp(
    model_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    embed_tokens: nn.Embedding | None = None,
    lm_head: nn.Linear | None = None,
    config: Any | None = None,
) -> Qwen35CheckpointMTP:
    """Instantiate and exactly load all 19 checkpoint mtp.* parameters."""
    model_path = Path(model_path)
    if embed_tokens is None or lm_head is None or config is None:
        if any(value is not None for value in (embed_tokens, lm_head, config)):
            raise ValueError(
                "embed_tokens, lm_head, and config must be provided together"
            )
        embed_tokens, lm_head, config = load_shared_embedding_and_head(model_path)

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        module = Qwen35CheckpointMTP(config, embed_tokens, lm_head)
    finally:
        torch.set_default_dtype(old_dtype)

    parameters = dict(module.named_parameters())
    index = _checkpoint_index(model_path)
    source_keys = sorted(key for key in index if key.startswith(MTP_PREFIX))
    loaded: list[tuple[str, str]] = []
    for source_key in source_keys:
        if source_key.startswith("mtp.layers."):
            destination = "decoder." + source_key[len(MTP_PREFIX) :]
        elif source_key.startswith("mtp.norm."):
            destination = "decoder." + source_key[len(MTP_PREFIX) :]
        else:
            destination = source_key[len(MTP_PREFIX) :]
        if destination not in parameters:
            raise KeyError(f"no Transformers MTP parameter for {source_key} -> {destination}")
        value = read_checkpoint_tensor(model_path, index, source_key)
        parameter = parameters[destination]
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(
                f"shape mismatch {source_key} {tuple(value.shape)} -> "
                f"{destination} {tuple(parameter.shape)}"
            )
        parameter.data.copy_(value.to(dtype=parameter.dtype))
        parameter.requires_grad_(False)
        loaded.append((source_key, destination))

    if len(loaded) != 19:
        raise RuntimeError(f"expected 19 MTP tensors, loaded {len(loaded)}")
    module.eval().requires_grad_(False)
    module.to(device=device, dtype=torch.bfloat16)
    module.load_report = {
        "source_tensor_count": len(loaded),
        "mapping": loaded,
        "shared_embedding_key": SHARED_EMBED_KEY,
        "shared_lm_head_key": SHARED_LM_HEAD_KEY,
        "attention_implementation": module.decoder.config._attn_implementation,
        "experts_implementation": "eager",
        "layer_types": ["full_attention"],
    }
    return module
