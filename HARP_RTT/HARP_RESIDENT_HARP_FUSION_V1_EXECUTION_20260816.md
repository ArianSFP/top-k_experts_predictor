# HARP Resident + Legacy-HARP Fusion v1

## Objective

Determine whether the legacy causal HARP predictor contains unique expert-route
information lost when compact Resident-Shadow omits routed expert computation.
The experiment is an information gate first and a learned fusion experiment
only if the paired oracle shows a large recoverable gain.

This lineage does **not** claim that the legacy predictor repairs the shadow
hidden state. Score fusion can recover final prefetch decisions, but it cannot
reconstruct omitted routed residuals, later attention inputs, or the Shadow-LM
token trajectory. A positive oracle with weak learned fusion would motivate a
separate in-rollout missing-state correction; a weak oracle kills that path.

## Metric alignment

The old model is the 43.56M-parameter temperature-1 HARP anchor, not an
MTP-hidden-only classifier. It consumes the greedy MTP spine together with
current target PCA state and route history. Its historical matched/mismatched
numbers were factual future-route strata from an older population and are not
directly comparable to Resident-Shadow's per-counterfactual-node branch metric.

This experiment resolves that ambiguity by joining both predictions to the
same immutable factual H1--H4 target sets, request population, position mask,
and layer mask. The old HARP prediction is fused only after Shadow-LM branch
aggregation because it has one factual forecast per horizon, not one valid
counterfactual forecast per sibling branch.

## Frozen inputs

- Legacy HARP checkpoint SHA256:
  `410ced95e6082f6a9bfa962d082926ee1ff5c29d203a3869f713a5ee7e58a09d`.
- The DeltaRoute ceiling bundle already stores row-aligned legacy HARP scores
  and exact-k marginals. No old-model replay is required.
- The sealed Resident-Only v2 bundle contains 3,850 utility-ranked group-64
  INT4 layer-expert cells.
- The missing artifact is a prediction-only Resident-Shadow cache. Producing
  it requires exact closed-loop inference but no target/MTP capture.
- Formal validation, calibration, and sealed test remain unopened.

## Resident policies

The evaluator supports three non-persistent views of the same sealed bundle:

1. `resident_3850`: reproduce the compact Resident-Only v2 namespace.
2. `hot25`: execute exactly the first 64 train-utility-ranked resident cells
   per layer (2,560/10,240 cells).
3. `hot25_current`: execute hot-25 plus exact current-token experts already
   present in the target offload cache. It is forbidden to initiate a load.

The current-token whitelist is the exact route of the already completed source
token. Overlapping hot cells use the exact cache copy; disabled packed cells do
not execute. Original router weights are never renormalized. Runtime masks and
cache bindings are non-persistent and are reset after every source tree.

## Prediction-only sidecar

`evaluate_shadow_route_top8.py --prediction-sidecar` writes one hash-bound
record per source position containing:

- request/position/tree/prefix identity and an opaque ceiling-row join;
- budget-16 topology and structural branch masks;
- per-node Shadow router logits, selected IDs, and selected weights;
- Shadow-LM captured/OTHER probabilities;
- captured and final factual Shadow inclusion marginals;
- static and current-cache availability;
- immediate omission mass/count;
- causal prior omission exposure: all layers of ancestors plus layers before
  the current router, never the current-layer omission.

The writer rejects target, truth, prefix-mismatch, factual-branch, or acceptance
fields. Labels and diagnostic strata are joined later by the offline analyzer.

## Information gate

`analyze_harp_resident_complementarity.py` reports request-macro:

- Shadow and HARP Recall@8;
- shared correct slots, Shadow-unique slots, HARP-unique slots, and misses;
- HARP recovery of Shadow misses at top 8/16/32/64;
- union@16 and whole-model switch ceilings;
- 1/2/4/8-swap ceilings and unlimited merge at HARP top 8/16/32/64;
- cold-only merge, cold versus available rescue, and missed-truth HARP ranks;
- horizon, layer quartile, prefix, and causal omission strata;
- 1,000 paired complete-request bootstraps with seed 42.

The primary decision is made on `hot25_current`. Training remains forbidden
unless the conservative HARP-top-32 expert-merge oracle:

- adds at least 0.08 H1--H4 and 0.08 H4, or reaches 0.90 H1--H4;
- recovers at least half of the compact-to-full ShadowRoute gap; and
- recovers at least 40% of Shadow misses in a high-prior-omission stratum.

The analyzer also runs a bounded cache-only branch factorial. It reweights only
the conditional distribution among already captured branches using
HARP/Shadow route compatibility, while preserving captured mass and OTHER
mass. This is diagnostic and cannot silently become a trained selector.

## Learned rescue (only after the gate)

Two variants begin at exact Resident-Shadow output equality:

- F0: one scalar horizon/layer captured-evidence replacement gate.
- F1: a shared expert-conditioned residual stacker with fewer than one million
  parameters. `f1_cold` restricts corrections to unavailable experts;
  `f1_all` is the declared diagnostic ablation.

Both operate on separately calibrated exact-k inclusion marginals rather than
subtracting incompatible raw logit spaces. Causal inputs include Shadow/HARP
marginals and ranks, availability, posterior-weighted prior omission exposure,
captured support/mass, branch entropy, boundary membership, disagreement,
horizon, layer, and expert identity. Prefix match and factual branch identity
are labels/strata only and changing them cannot change model features.

Training is seed 42 only, AdamW at `3e-4`, weight decay `0.01`, effective batch
32, maximum 20 epochs, and patience 3. The loss is exact-set NLL plus a 0.1
missing-true/false-prediction swap boundary. A straight-through constrained
outer gate gives bit-exact epoch-zero output and a usable gradient at zero.

## Confirmation

The frozen checkpoint records every fit/tune request ID. The confirmation
driver rejects any overlap and constructs no optimizer. Promotion requires:

- H1--H4 Recall@8 at least 0.90;
- at least +0.07 over the paired resident baseline;
- H4 at least 0.85 and at least +0.06;
- paired complete-request 95% lower gain above zero;
- no horizon regression larger than 0.005.

## Hardware sequence

1. Local CPU: implementation tests and static leakage checks.
2. RTX PRO 6000 96GB (or H100 NVL/H200): generate prediction-only sidecars on
   the already-open 32-request/512-position Resident screen. Prior compact
   closed loop peaked at 86.166 GiB. No optimizer and no capture.
3. CPU: run the paired oracle and kill the path if it misses the large-gain
   gate.
4. RTX 3090/4090 24GB: only if the information gate passes, fit F0/F1. This is
   when a 3090 pod becomes necessary.
5. RTX PRO 6000 96GB: generate the frozen request-disjoint confirmation cache;
   evaluate the selected checkpoint once.

The 3090 must not be started speculatively. The expensive exact-cache replay is
needed first because prior completed runs retained only request-level metrics,
not paired per-position/per-layer Shadow predictions.

## Current implementation status

- Runtime resident policies and exact-current whitelist: implemented.
- Reusable exact-prefix hybrid cache and prediction-only sidecar: implemented.
- Complementarity/oracle/stratification analyzer: implemented.
- Cache-only branch-compatibility factorial: implemented.
- Zero-baseline F0/F1 trainer and immutable checkpoint contract: implemented.
- Request-disjoint confirmation driver: implemented.
- Synthetic leakage, causality, oracle, zero-gate, runtime-mask, and arithmetic
  tests: implemented.
- GPU prediction sidecars and empirical results: not yet produced.
