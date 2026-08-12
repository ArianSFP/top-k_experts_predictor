"""Accuracy-first HARP router-query trajectory primitives."""

from .exact_k import (
    DEFAULT_K,
    cardinality_project_marginals,
    exact_k_logz_marginals_fast,
    exact_k_logz_with_marginals,
    exact_k_marginals,
    exact_set_nll,
    log_esp_k,
    soft_cardinality_topk,
    soft_recall_loss,
    stable_topk,
    validate_exact_set_labels,
)
from .geometry import (
    ROUTER_GEOMETRY_SCHEMA,
    CenteredRouterGeometry,
    RouterGeometryAudit,
    assert_router_geometry_equivalent,
    audit_centered_router_geometry,
    build_centered_router_geometry,
)
from .delta import HARPDeltaConfig, HARPDeltaOutput, HARPDeltaTeacher, HARPDeltaTree
from .delta_batch import DeltaPreparedBatch, factual_branch_indices, prepare_delta_batch
from .deltaroute_batch import DeltaRoutePreparedBatch, prepare_deltaroute_batch
from .route_dynamics import DeltaRouteConfig, DeltaRouteTrajectory
from .schema import (
    CandidateBatch,
    CandidateDiagnostics,
    HARPRTTBatch,
    HARPRTTDimensions,
    HARPRTTOutput,
    TreeBatch,
)

__all__ = [
    "DEFAULT_K",
    "CenteredRouterGeometry",
    "CandidateBatch",
    "CandidateDiagnostics",
    "HARPRTTBatch",
    "HARPRTTDimensions",
    "HARPRTTOutput",
    "HARPDeltaConfig",
    "HARPDeltaOutput",
    "HARPDeltaTeacher",
    "HARPDeltaTree",
    "DeltaPreparedBatch",
    "DeltaRouteConfig",
    "DeltaRoutePreparedBatch",
    "DeltaRouteTrajectory",
    "factual_branch_indices",
    "prepare_delta_batch",
    "prepare_deltaroute_batch",
    "RouterGeometryAudit",
    "ROUTER_GEOMETRY_SCHEMA",
    "TreeBatch",
    "assert_router_geometry_equivalent",
    "audit_centered_router_geometry",
    "build_centered_router_geometry",
    "cardinality_project_marginals",
    "exact_k_logz_marginals_fast",
    "exact_k_logz_with_marginals",
    "exact_k_marginals",
    "exact_set_nll",
    "log_esp_k",
    "soft_cardinality_topk",
    "soft_recall_loss",
    "stable_topk",
    "validate_exact_set_labels",
]
