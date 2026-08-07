"""Pure tensor helpers matching native Qwen3.5-MoE router semantics."""

from __future__ import annotations

import torch


def native_topk_router_distribution(
    router_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP32 probabilities and the weights actually sent to experts.

    Transformers performs the softmax and top-k normalization in FP32, then
    casts normalized weights back to the router-logit dtype before expert
    execution. The final cast is therefore part of the capture contract.
    """

    if router_logits.ndim < 1 or not 0 < top_k <= router_logits.shape[-1]:
        raise ValueError("router logits/top-k shape mismatch")
    router_probabilities = torch.nn.functional.softmax(
        router_logits, dtype=torch.float, dim=-1
    )
    selected_probabilities, selected_ids = torch.topk(
        router_probabilities, top_k, dim=-1
    )
    selected_weights = selected_probabilities / selected_probabilities.sum(
        dim=-1, keepdim=True
    )
    selected_weights = selected_weights.to(router_logits.dtype)
    return router_probabilities, selected_weights, selected_ids
