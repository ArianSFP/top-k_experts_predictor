from __future__ import annotations

import numpy as np

from harp8.audit_jspace_plateau import recall_from_membership, stable_topk


def test_stable_topk_uses_full_expert_id_to_break_candidate_ties() -> None:
    scores = np.asarray([[1.0, 1.0, 0.5]], dtype=np.float32)
    expert_ids = np.asarray([[9, 2, 4]], dtype=np.int64)
    selected = stable_topk(scores, 1, expert_ids=expert_ids)
    assert selected.tolist() == [[1]]


def test_stable_topk_without_ids_preserves_axis_order_at_ties() -> None:
    scores = np.asarray([[1.0, 1.0, 0.5]], dtype=np.float32)
    assert stable_topk(scores, 2).tolist() == [[0, 1]]


def test_recall_from_membership_has_fixed_native_k_denominator() -> None:
    membership = np.asarray([[True, False, True, False]])
    selected = np.asarray([[0, 1]])
    assert recall_from_membership(membership, selected, 2) == 0.5
