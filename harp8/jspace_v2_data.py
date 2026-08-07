"""Aligned, label-safe inputs for the experimental J-HARP-C64 v2 ranker.

Version 2 deliberately keeps the immutable v1 capture and candidate-pool
formats.  Its only additional required input is the frozen HARP
``generator_context.f16`` tensor already stored in context-enabled pools.
This module makes that dependency explicit and fails closed when the context
is absent instead of silently falling back to a different architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .jspace_data import AlignedJCandidateData


JSPACE_V2_ALIGNED_DATA_SCHEMA = "harp8_jspace_v2_aligned_candidate_data_v1"

# These values may be retained in trace artifacts as supervision or audit
# metadata, but none is an admissible observed input to the v2 ranker.
LABEL_ONLY_INPUT_FIELDS = frozenset(
    {
        "accepted_at_depth",
        "accepted_through_depth",
        "acceptance_label",
        "actual_acceptance",
        "committed_future_token_ids",
        "first_rejection_depth",
        "future_router_logits",
        "future_router_scores",
        "future_target_top8",
        "mtp_draft_target_logprobs",
        "mtp_draft_target_token_ids",
        "prefix_matches_committed",
        "target_acceptance",
    }
)

JSPACE_V2_MODEL_BATCH_KEYS = frozenset(
    {
        "candidate_scores",
        "candidate_ids",
        "target_membership",
        "teacher_candidate_scores",
        "valid_future",
        "candidate_mask",
        "j_states",
        "j_mask",
        "mtp_states",
        "mtp_router_logits",
        "mtp_mask",
        "mtp_metadata",
        "candidate_features",
        "generator_context",
    }
)

# Only these causal fields cross the model-forward boundary.  Supervision
# tensors remain in the surrounding training batch for loss calculation.
JSPACE_V2_CAUSAL_MODEL_KEYS = frozenset(
    {
        "candidate_scores",
        "candidate_ids",
        "candidate_mask",
        "j_states",
        "j_mask",
        "mtp_states",
        "mtp_router_logits",
        "mtp_mask",
        "mtp_metadata",
        "candidate_features",
        "generator_context",
    }
)


def assert_no_label_only_inputs(batch: Mapping[str, torch.Tensor]) -> None:
    """Reject any accidental future/acceptance feature before model entry."""

    forbidden = sorted(LABEL_ONLY_INPUT_FIELDS.intersection(batch))
    if forbidden:
        raise ValueError(f"label-only fields cannot be model inputs: {forbidden}")


def causal_v2_model_inputs(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the strict causal allowlist consumed by model forward.

    In particular, future membership and teacher candidate scores stay in the
    outer batch for the objective but cannot be observed by the ranker.
    """

    assert_no_label_only_inputs(batch)
    required = {
        "candidate_scores",
        "candidate_ids",
        "j_states",
        "j_mask",
        "mtp_states",
        "mtp_router_logits",
        "mtp_mask",
        "generator_context",
    }
    missing = required - batch.keys()
    if missing:
        raise KeyError(f"causal v2 model inputs are missing: {sorted(missing)}")
    return {
        name: value
        for name, value in batch.items()
        if name in JSPACE_V2_CAUSAL_MODEL_KEYS
    }


def compact_v2_model_batch(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Retain exactly the tensors consumed by v2 and its training loss."""

    assert_no_label_only_inputs(batch)
    required = {
        "candidate_scores",
        "candidate_ids",
        "target_membership",
        "teacher_candidate_scores",
        "valid_future",
        "j_states",
        "j_mask",
        "mtp_states",
        "mtp_router_logits",
        "mtp_mask",
        "generator_context",
    }
    missing = required - batch.keys()
    if missing:
        raise KeyError(f"v2 model batch is missing required tensors: {sorted(missing)}")
    return {
        name: value
        for name, value in batch.items()
        if name in JSPACE_V2_MODEL_BATCH_KEYS
    }


class AlignedJContextCandidateData(AlignedJCandidateData):
    """V1-aligned data plus mandatory local frozen-generator context.

    The candidate pool owns ``generator_context`` and is already request- and
    horizon-aligned.  Target J/history and MTP tensors continue to use the
    audited request-ID join implemented by :class:`AlignedJCandidateData`.
    """

    def __init__(
        self,
        pool_root: Path,
        *,
        capture_dir: Path,
        mtp_dir: Path,
        target_features: Path,
        target_feature_rms: Path | None = None,
        rows_per_request: int = 34,
        history: int = 3,
    ) -> None:
        super().__init__(
            pool_root,
            capture_dir=capture_dir,
            mtp_dir=mtp_dir,
            target_features=target_features,
            target_feature_rms=target_feature_rms,
            rows_per_request=rows_per_request,
            history=history,
        )
        if self.pool.context is None or not bool(
            self.pool.manifest.get("store_context", False)
        ):
            raise ValueError(
                "J-HARP-C64 v2 requires a context-enabled candidate pool "
                "with generator_context.f16"
            )
        context_path = self.pool.root / "generator_context.f16"
        declared = {
            str(record.get("path")): record
            for record in self.pool.manifest.get("arrays", [])
            if isinstance(record, Mapping)
        }
        if context_path.name not in declared:
            raise ValueError(
                "candidate manifest does not declare generator_context.f16"
            )
        expected_shape = (
            self.pool.rows,
            self.pool.horizons,
            self.pool.layers,
            self.pool.model_width,
        )
        if tuple(self.pool.context.shape) != expected_shape:
            raise ValueError("generator context disagrees with pool geometry")

    @property
    def generator_context_width(self) -> int:
        return int(self.pool.model_width)

    def batch(
        self,
        rows: np.ndarray,
        device: str | torch.device,
        *,
        active_horizons: int | None = None,
        include_context: bool = True,
        compact: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not include_context:
            raise ValueError(
                "J-HARP-C64 v2 requires include_context=True; use v1 for a "
                "matched no-context control"
            )
        result = super().batch(
            rows,
            device,
            active_horizons=active_horizons,
            include_context=True,
            compact=False,
        )
        context = result.pop("context", None)
        if context is None:
            raise RuntimeError("context-enabled pool produced no generator context")
        if context.ndim != 4:
            raise ValueError("generator context must have shape [B,H,L,D_G]")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("generator context contains non-finite values")
        result["generator_context"] = context
        assert_no_label_only_inputs(result)
        return compact_v2_model_batch(result) if compact else result


__all__ = [
    "AlignedJContextCandidateData",
    "JSPACE_V2_ALIGNED_DATA_SCHEMA",
    "JSPACE_V2_CAUSAL_MODEL_KEYS",
    "JSPACE_V2_MODEL_BATCH_KEYS",
    "LABEL_ONLY_INPUT_FIELDS",
    "assert_no_label_only_inputs",
    "causal_v2_model_inputs",
    "compact_v2_model_batch",
]
