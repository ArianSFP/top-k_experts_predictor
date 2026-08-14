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
from .shadow_backbone import (
    InstalledShadowBackbone,
    exact_prefix_experts,
    install_shadow_experts,
)
from .shadow_bundle import load_shadow_bundle
from .shadow_cache import ShadowNodeResult, ShadowTreeResult, ShadowTreeRunner
from .shadow_capture import ShadowCaptureDimensions
from .shadow_expert import (
    ExactTop1PlusDraftExperts,
    IndexedShadowExperts,
    ShadowExpertConfig,
    SharedResidualExperts,
    SwiGLUDraftExpert,
)
from .shadow_route import ShadowRouteMixture, raw_mtp_prior_mixture
from .shadow_training import ShadowLossWeights, shadow_route_objective
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
    "ExactTop1PlusDraftExperts",
    "IndexedShadowExperts",
    "InstalledShadowBackbone",
    "ShadowCaptureDimensions",
    "ShadowExpertConfig",
    "ShadowLossWeights",
    "ShadowNodeResult",
    "ShadowRouteMixture",
    "ShadowTreeResult",
    "ShadowTreeRunner",
    "SharedResidualExperts",
    "SwiGLUDraftExpert",
    "factual_branch_indices",
    "prepare_delta_batch",
    "prepare_deltaroute_batch",
    "raw_mtp_prior_mixture",
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
    "exact_prefix_experts",
    "install_shadow_experts",
    "load_shadow_bundle",
    "log_esp_k",
    "soft_cardinality_topk",
    "soft_recall_loss",
    "shadow_route_objective",
    "stable_topk",
    "validate_exact_set_labels",
]
