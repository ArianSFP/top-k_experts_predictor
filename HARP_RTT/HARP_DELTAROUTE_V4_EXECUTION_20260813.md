# HARP-DeltaRoute v4 execution ledger

**Date:** 2026-08-13  
**Branch:** `agent/harp-deltaroute-v4`  
**Scope:** outer-train diagnostic reuse only; formal validation, calibration,
and sealed test remain unopened.

## 1. Bottleneck hierarchy

The native branch labels establish that the existing adaptive-32 tree is not
the dominant bottleneck: native routes with the learned posterior provide
approximately `0.9853` mean H2--H4 C64.  The selected learned budget-16 route
translator provides only `0.914467`, while replacing its learned posterior
with the target posterior changes coverage by only about `0.0003`.

The active hierarchy is therefore:

1. branch state to target route translation;
2. factual expert-specific aggregation of imperfect branch predictions;
3. only after C64 approaches `0.98`, within-C64 ranking.

The final ranker remains closed.

## 2. Implemented diagnostics

The static-head diagnosis was converted into a layerwise route model in
`harp_rtt/route_dynamics.py`.  Its corrected R0 condition predicts adjacent
target-layer route changes from the true current query, native selected IDs
and execution weights, and causal MTP branch context.  The predicted change is
attached as a residual over the selected budget-16 parent query and score, so
the parent's learned non-geometric free score is preserved exactly at epoch
zero.

The first R0 implementation discarded that free score and was stopped after
two epochs as an invalid architecture diagnostic.  The corrected run is bound
to source commit `9192ee0433c35a9d6883963755bf247a69850a39` and writes to
`transition_r0_residual_seed42_9192ee0` on local NVMe.

## 3. Emerging R0 result

At epoch 24, tuning counterfactual route Recall@8 had improved monotonically
to `0.614873`, with positive gains at H2, H3, and H4.  Mean H2--H4 factual C64
rose to `0.929976`; H4 reached `0.911371` and H4 prefix-mismatch reached
`0.866254`.  This is a useful reusable component, but remains far below the
predeclared R0 route threshold `0.7827185`.

The scientific interpretation is specific: even with the true current router
query and true current selected route, the query-visible transition recovers
only a modest fraction of the missing route information.  A likely omitted
state is the target residual component orthogonal to the current layer's
router basis.

The terminal epoch and untouched 2,048-row development result are recorded
only after the driver seals `STAGE_RESULT.json`.

## 4. No-recapture router-blind content probe

Before authorizing new target replay, the existing factual corpus can test
this hypothesis.  Every H1--H4 factual row already contains the complete
normalized 2,048-dimensional target router input for all 40 layers, selected
IDs, and native execution weights.

`harp_rtt/content_transition.py` and
`runpod/train_harp_deltaroute_content_probe.py` implement a teacher-only probe:

1. decompose each full state into frozen router-visible and router-blind
   components;
2. use the current query, native executed route, and blind content to predict
   the next layer's query and exact selected set;
3. ablate only the blind component with identical learned parameters;
4. report request-grouped H2--H4 results and a 1,000-replicate paired
   bootstrap.

The probe is explicitly not a serving model.  Its manifest declares the
future full state as a teacher input and sets `serving_authorized=false`.  A
pass (`>=0.80` mean H2--H4 one-step route recall, positive paired lower bound,
and positive content gain at every horizon) authorizes causal content-state
distillation from already-captured MTP/tree inputs.  A failure strengthens the
case for the narrowly specified router-blind counterfactual recapture.

## 5. Next architecture conditional on the probe

If router-blind content is decisive, freeze the probe's route-relevant content
encoder and train a causal student to predict that latent from the existing
layer-conditioned MTP channel tokens and adaptive tree.  Train the deployed
factual route/C64 objective directly, retain the corrected R0 branch residual
as a candidate source, and keep explicit OTHER/anchor fallback.  This uses the
existing factual full states as labels and requires no recapture.

If the full-state probe itself is weak, more MLP/ranker capacity is not
justified.  The next data capture must store a compact router-blind content
sketch plus projected attention and MoE transition deltas for counterfactual
nodes, exactly as preregistered in the v4 architecture plan.

Small diagnostics use seed 42 only per user instruction.  Additional seeds are
reserved for a large architectural gain near a promotion decision.
