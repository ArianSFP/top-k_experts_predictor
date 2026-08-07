"""Serializable geometry for the HARP-RTT teacher model.

The production defaults follow ``HARP_RTT_90_formal_architecture.md``.  The
input widths describe the uncompressed rich-capture tensors and can be made
smaller in unit tests without changing any semantic axis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HARPRTTConfig:
    """Model and tensor geometry for the H1--H4 residual teacher."""

    experts: int = 256
    layers: int = 40
    anchor_horizons: int = 8
    active_horizons: int = 4
    route_history: int = 8
    exact_k: int = 8
    candidate_width: int = 64
    max_tree_nodes: int = 32
    max_tree_depth: int = 32
    max_tree_branches: int = 64

    # Frozen centered-router geometry.
    router_rank: int = 255

    # Rich target/token capture widths.
    target_control_width: int = 255
    target_content_width: int = 2048
    target_post_attention_width: int = 2048
    target_post_moe_width: int = 2048
    target_routed_width: int = 2048
    target_shared_width: int = 2048
    exact_token_width: int = 2048
    final_hidden_width: int = 2048

    # MTP tree channel widths.  ``tree_states`` in the batch adapter maps the
    # four capture channels onto these projections.
    tree_hidden_width: int = 2048
    tree_fused_width: int = 2048
    tree_router_input_width: int = 2048
    tree_token_width: int = 2048
    tree_metadata_width: int = 8

    route_summary_width: int = 4
    route_key_width: int = 64
    target_width: int = 256
    route_width: int = 384
    tree_width: int = 512
    model_width: int = 384
    reranker_width: int = 384
    object_residual_width: int = 32
    candidate_feature_width: int = 0

    route_ffn_width: int = 768
    target_ffn_width: int = 512
    tree_ffn_width: int = 1024
    decoder_ffn_width: int = 768
    reranker_ffn_width: int = 768
    attention_heads: int = 8

    temporal_blocks: int = 2
    route_layer_blocks: int = 2
    target_blocks: int = 2
    tree_blocks: int = 4
    decoder_blocks: int = 6
    trajectory_rounds: int = 2
    reranker_set_blocks: int = 2
    reranker_summary_blocks: int = 2
    dropout: float = 0.05

    freeze_anchor: bool = True
    detach_self_conditioning: bool = True

    def validate(self) -> None:
        positive = {
            name: value
            for name, value in asdict(self).items()
            if name.endswith("_width")
            or name
            in {
                "experts",
                "layers",
                "anchor_horizons",
                "active_horizons",
                "route_history",
                "exact_k",
                "max_tree_nodes",
                "max_tree_depth",
                "max_tree_branches",
                "router_rank",
                "attention_heads",
                "temporal_blocks",
                "route_layer_blocks",
                "target_blocks",
                "tree_blocks",
                "decoder_blocks",
                "trajectory_rounds",
                "reranker_set_blocks",
                "reranker_summary_blocks",
            }
        }
        # Optional feature widths are the only dimensions allowed to be zero.
        optional_zero = {"candidate_feature_width"}
        invalid = [
            name
            for name, value in positive.items()
            if value < 0 or (value == 0 and name not in optional_zero)
        ]
        if invalid:
            raise ValueError(f"positive HARP-RTT dimensions required: {invalid}")
        if not 1 <= self.active_horizons <= self.anchor_horizons:
            raise ValueError("active_horizons must lie within anchor_horizons")
        if not 1 <= self.exact_k <= self.experts:
            raise ValueError("exact_k must lie within the expert namespace")
        if not self.exact_k <= self.candidate_width <= self.experts:
            raise ValueError("candidate_width must lie in [exact_k, experts]")
        if self.trajectory_rounds != 2:
            raise ValueError("HARP-RTT v1 requires exactly two trajectory rounds")
        for name in (
            "route_width",
            "target_width",
            "tree_width",
            "model_width",
            "reranker_width",
        ):
            if getattr(self, name) % self.attention_heads:
                raise ValueError(f"{name} must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["HARPRTTConfig"]
