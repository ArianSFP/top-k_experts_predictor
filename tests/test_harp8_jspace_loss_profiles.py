from __future__ import annotations

import pytest

from harp8.jspace_loss_profiles import (
    MEMBERSHIP_RANKING_V1,
    PREREGISTERED_V1,
    assert_resume_loss_profile,
    provenance_with_loss_profile,
    resolve_jspace_loss_profile,
)
from harp8.jspace_reranker import JSpaceRerankerLossConfig
from harp8.train_jspace_reranker import build_parser as build_v1_parser
from harp8.train_jspace_v2_reranker import build_parser as build_v2_parser


def test_preregistered_profile_preserves_previous_default_exactly() -> None:
    previous = JSpaceRerankerLossConfig(
        horizon_weights=JSpaceRerankerLossConfig().horizon_weights[:4]
    )
    resolved = resolve_jspace_loss_profile(PREREGISTERED_V1, 4)

    assert resolved.to_dict() == previous.to_dict()
    assert build_v1_parser().get_default("loss_profile") == PREREGISTERED_V1
    assert build_v2_parser().get_default("loss_profile") == PREREGISTERED_V1


def test_membership_ranking_profile_is_exactly_preregistered_ablation() -> None:
    resolved = resolve_jspace_loss_profile(MEMBERSHIP_RANKING_V1, 4)

    assert resolved.to_dict() == {
        "boundary": 1.0,
        "balanced_bce": 0.0,
        "listwise": 1.0,
        "restricted_kl": 0.0,
        "temperature": 2.0,
        "hard_negative_count": 24,
        "horizon_weights": (1.0, 1.0, 1.25, 1.5),
    }


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown J-space loss profile"):
        resolve_jspace_loss_profile("unregistered_experiment", 4)


def test_resume_profile_mismatch_fails_and_legacy_default_is_inferred() -> None:
    preregistered = resolve_jspace_loss_profile(PREREGISTERED_V1, 4)
    membership = resolve_jspace_loss_profile(MEMBERSHIP_RANKING_V1, 4)
    legacy = {"loss_config": preregistered.to_dict()}

    assert_resume_loss_profile(
        legacy,
        expected_profile=PREREGISTERED_V1,
        expected_config=preregistered,
        horizons=4,
    )
    with pytest.raises(ValueError, match="loss_profile"):
        assert_resume_loss_profile(
            legacy,
            expected_profile=MEMBERSHIP_RANKING_V1,
            expected_config=membership,
            horizons=4,
        )


def test_provenance_records_profile_and_complete_resolved_config() -> None:
    config = resolve_jspace_loss_profile(MEMBERSHIP_RANKING_V1, 4)
    source = {"execution_contract": {"active_horizons": 4}}

    result = provenance_with_loss_profile(
        source,
        loss_profile=MEMBERSHIP_RANKING_V1,
        loss_config=config,
    )

    assert source == {"execution_contract": {"active_horizons": 4}}
    assert result["execution_contract"]["loss_profile"] == MEMBERSHIP_RANKING_V1
    assert result["execution_contract"]["resolved_loss_config"] == config.to_dict()
