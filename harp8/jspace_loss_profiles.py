"""Named, immutable loss profiles for J-space reranker experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .jspace_reranker import JSpaceRerankerLossConfig


PREREGISTERED_V1 = "preregistered_v1"
MEMBERSHIP_RANKING_V1 = "membership_ranking_v1"
CUSTOM_LOSS_PROFILE = "custom"
JSPACE_LOSS_PROFILE_NAMES = (PREREGISTERED_V1, MEMBERSHIP_RANKING_V1)


def _horizon_weights(horizons: int) -> tuple[float, ...]:
    available = JSpaceRerankerLossConfig().horizon_weights
    if horizons <= 0:
        raise ValueError("horizons must be positive")
    if horizons > len(available):
        raise ValueError("more than eight horizons require explicit loss design")
    return tuple(available[:horizons])


def resolve_jspace_loss_profile(
    name: str,
    horizons: int,
) -> JSpaceRerankerLossConfig:
    """Resolve a public profile name to its complete, validated objective."""

    horizon_weights = _horizon_weights(horizons)
    if name == PREREGISTERED_V1:
        config = JSpaceRerankerLossConfig(horizon_weights=horizon_weights)
    elif name == MEMBERSHIP_RANKING_V1:
        config = JSpaceRerankerLossConfig(
            boundary=1.0,
            balanced_bce=0.0,
            listwise=1.0,
            restricted_kl=0.0,
            temperature=2.0,
            hard_negative_count=24,
            horizon_weights=horizon_weights,
        )
    else:
        choices = ", ".join(JSPACE_LOSS_PROFILE_NAMES)
        raise ValueError(
            f"unknown J-space loss profile {name!r}; choose from {choices}"
        )
    config.validate(horizons)
    return config


def loss_config_from_mapping(
    value: Mapping[str, Any],
) -> JSpaceRerankerLossConfig:
    """Restore a loss config while normalizing JSON lists to immutable tuples."""

    fields = dict(value)
    if "horizon_weights" in fields:
        fields["horizon_weights"] = tuple(fields["horizon_weights"])
    return JSpaceRerankerLossConfig(**fields)


def infer_jspace_loss_profile(
    config: JSpaceRerankerLossConfig,
    horizons: int,
) -> str:
    """Return the matching public profile, or ``custom`` for Python callers."""

    config.validate(horizons)
    encoded = config.to_dict()
    for name in JSPACE_LOSS_PROFILE_NAMES:
        if encoded == resolve_jspace_loss_profile(name, horizons).to_dict():
            return name
    return CUSTOM_LOSS_PROFILE


def validate_jspace_loss_profile(
    config: JSpaceRerankerLossConfig,
    horizons: int,
    requested: str | None,
) -> str:
    """Bind a config to a named CLI profile or infer identity for Python use."""

    if requested is None:
        return infer_jspace_loss_profile(config, horizons)
    expected = resolve_jspace_loss_profile(requested, horizons)
    if config.to_dict() != expected.to_dict():
        raise ValueError(
            f"loss config differs from requested loss profile {requested!r}"
        )
    return requested


def recorded_jspace_loss_profile(
    payload: Mapping[str, Any],
    horizons: int,
) -> str:
    """Read profile identity, inferring it for checkpoints predating the field."""

    encoded = payload.get("loss_config")
    if not isinstance(encoded, Mapping):
        raise ValueError("resume checkpoint has no valid loss_config")
    config = loss_config_from_mapping(encoded)
    config.validate(horizons)
    explicit = payload.get("loss_profile")
    if explicit is None:
        return infer_jspace_loss_profile(config, horizons)
    if explicit == CUSTOM_LOSS_PROFILE:
        return explicit
    expected = resolve_jspace_loss_profile(str(explicit), horizons)
    if config.to_dict() != expected.to_dict():
        raise ValueError("resume checkpoint loss profile and loss_config disagree")
    return str(explicit)


def assert_resume_loss_profile(
    payload: Mapping[str, Any],
    *,
    expected_profile: str,
    expected_config: JSpaceRerankerLossConfig,
    horizons: int,
) -> None:
    """Fail closed when a resume would change either objective name or values."""

    recorded_profile = recorded_jspace_loss_profile(payload, horizons)
    if recorded_profile != expected_profile:
        raise ValueError("resume checkpoint loss_profile differs from this run")
    encoded = payload.get("loss_config")
    assert isinstance(encoded, Mapping)
    recorded_config = loss_config_from_mapping(encoded)
    if recorded_config.to_dict() != expected_config.to_dict():
        raise ValueError("resume checkpoint loss_config differs from this run")


def provenance_with_loss_profile(
    provenance: Mapping[str, Any],
    *,
    loss_profile: str,
    loss_config: JSpaceRerankerLossConfig,
) -> dict[str, Any]:
    """Attach objective identity without mutating caller-owned provenance."""

    result = deepcopy(dict(provenance))
    execution = dict(result.get("execution_contract", {}))
    expected = {
        "loss_profile": loss_profile,
        "resolved_loss_config": loss_config.to_dict(),
    }
    for name, value in expected.items():
        if name in execution and execution[name] != value:
            raise ValueError(f"execution contract {name} conflicts with loss objective")
        execution[name] = value
    result["execution_contract"] = execution
    return result


__all__ = [
    "CUSTOM_LOSS_PROFILE",
    "JSPACE_LOSS_PROFILE_NAMES",
    "MEMBERSHIP_RANKING_V1",
    "PREREGISTERED_V1",
    "assert_resume_loss_profile",
    "infer_jspace_loss_profile",
    "loss_config_from_mapping",
    "provenance_with_loss_profile",
    "recorded_jspace_loss_profile",
    "resolve_jspace_loss_profile",
    "validate_jspace_loss_profile",
]
