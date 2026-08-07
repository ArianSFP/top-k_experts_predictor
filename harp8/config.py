"""Serializable configuration for the HARP-8T research teacher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HARPConfig:
    """Model geometry.

    The defaults are the accuracy-first configuration. Tests and diagnostic
    pilots use smaller values while preserving the same tensor contract.
    """

    experts: int = 256
    layers: int = 40
    horizons: int = 8
    route_history: int = 8
    mtp_depths: int = 8

    target_state_channels: int = 2
    target_state_width: int = 2048
    mtp_state_channels: int = 3
    mtp_state_width: int = 2048
    mtp_state_projection_width: int = 256
    mtp_metadata_width: int = 8
    mtp_vocab_width: int = 256

    route_width: int = 256
    state_width: int = 256
    mtp_width: int = 384
    model_width: int = 384
    route_ffn_width: int = 768
    mtp_ffn_width: int = 1024
    fusion_ffn_width: int = 768
    attention_heads: int = 8
    temporal_blocks: int = 2
    layer_blocks: int = 2
    state_blocks: int = 1
    mtp_cross_blocks: int = 2
    fusion_blocks: int = 2
    dropout: float = 0.05

    dense_output: bool = True
    output_rank: int = 128
    future_latent_width: int = 256
    mtp_source_dropout: float = 0.10
    target_state_source_dropout: float = 0.10
    use_target_state: bool = True
    use_mtp: bool = True
    # Zero means all configured depths; positive values define an ablation.
    mtp_active_depths: int = 0

    @classmethod
    def compact_compatibility(cls) -> "HARPConfig":
        """Profile for the currently captured PCA features and six MTP nodes."""

        return cls(
            mtp_depths=8,
            target_state_channels=1,
            target_state_width=128,
            mtp_state_channels=1,
            mtp_state_width=128,
            mtp_metadata_width=4,
            mtp_vocab_width=0,
            future_latent_width=128,
        )

    def validate(self) -> None:
        positive = {
            "experts": self.experts,
            "layers": self.layers,
            "horizons": self.horizons,
            "route_history": self.route_history,
            "mtp_depths": self.mtp_depths,
            "target_state_channels": self.target_state_channels,
            "target_state_width": self.target_state_width,
            "mtp_state_channels": self.mtp_state_channels,
            "mtp_state_width": self.mtp_state_width,
            "mtp_state_projection_width": self.mtp_state_projection_width,
            "mtp_metadata_width": self.mtp_metadata_width,
            "route_width": self.route_width,
            "state_width": self.state_width,
            "mtp_width": self.mtp_width,
            "model_width": self.model_width,
            "attention_heads": self.attention_heads,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive HARP dimensions required: {invalid}")
        for name, width in (
            ("route_width", self.route_width),
            ("state_width", self.state_width),
            ("mtp_width", self.mtp_width),
            ("model_width", self.model_width),
        ):
            if width % self.attention_heads:
                raise ValueError(f"{name} must be divisible by attention_heads")
        if self.mtp_vocab_width < 0 or self.future_latent_width < 0:
            raise ValueError("optional feature widths cannot be negative")
        if not 0 <= self.mtp_active_depths <= self.mtp_depths:
            raise ValueError("mtp_active_depths must be zero or lie within configured depths")
        if not self.dense_output and self.output_rank <= 0:
            raise ValueError("ranked output requires a positive output_rank")
        for probability in (
            self.dropout,
            self.mtp_source_dropout,
            self.target_state_source_dropout,
        ):
            if not 0.0 <= probability < 1.0:
                raise ValueError("dropout probabilities must lie in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LossConfig:
    temperature: float = 2.0
    router_kl: float = 1.0
    boundary: float = 0.30
    inclusion: float = 0.10
    centered_score: float = 0.05
    future_latent: float = 0.05
    full_membership: float = 0.0
    candidate16: float = 0.0
    candidate_count: int = 16
    candidate_margin: float = 0.0
    hard_negative_end_rank: int = 32
    h2_weight: float = 2.0
    # None preserves the original H2-emphasized objective. Otherwise this must
    # contain one non-negative weight per configured forecast horizon.
    horizon_weights: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    minimum_epochs: int = 10
    patience: int = 5
    batch_size: int = 8
    evaluation_batch_size: int = 8
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    minimum_learning_rate: float = 2e-5
    warmup_fraction: float = 0.03
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    gradient_clip: float = 1.0
    seed: int = 42
    active_horizons: tuple[int, ...] = tuple(range(1, 9))
    h2_only: bool = False
    selection_profile: str = "h2"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["active_horizons"] = list(self.active_horizons)
        if value["selection_profile"] not in ("h2", "h1_h4_candidate16"):
            raise ValueError("unknown checkpoint selection profile")
        return value
