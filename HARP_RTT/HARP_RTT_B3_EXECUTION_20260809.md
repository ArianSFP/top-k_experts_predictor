# HARP-RTT B3 Factual Generator and Rich C64 Ranker

Date: 2026-08-09  
Branch: `agent/harp-rtt-b1.5-exact-set-oracle`

## Decision boundary

B2 seed 42 completed and recovered more than half of the oracle lift in every
declared mismatch stratum:

| Stratum | Recovery G |
| --- | ---: |
| H2 prefix mismatch | 0.610053 |
| H3 prefix mismatch | 0.620945 |
| H4 prefix mismatch | 0.637952 |

Seeds 43 and 44 were intentionally stopped after three and two complete
epochs. The user authorized an operational transition to B3 because their
early learning curves followed seed 42. The preregistered three-seed gate and
paired bootstrap remain explicitly not completed. This is not reported as a
three-seed statistical pass. The immutable authorization is
`B2_USER_OPERATIONAL_OVERRIDE_20260809.json`.

## Data boundary

B3 opens no new data:

- fitting uses the same 256 outer-train requests and 4,096 positions as B2;
- the unchanged request-hash partition supplies 224 training and 32 tuning
  requests;
- promotion is measured on the already-open, request-disjoint 128-request,
  2,048-position outer-train diagnostic probe;
- counterfactual budget-16 labels remain train-only retention targets;
- formal validation, calibration and sealed test remain unopened.

Every loader is bound to the audited adaptive-32 base capture and node-indexed
companion. The B3 command exposes no switch capable of selecting validation,
calibration or test.

## Generator stage

The generator loads the exact seed-42 checkpoint
`fad0da794ba229be80e5cb05e6cb69f135b765f1339acdc74be6e0b52431c1c6`.
Before constructing an optimizer it must prove:

- bitwise H1-H4 dense-score equality with the frozen HARP anchor;
- identical stable top-8 IDs;
- exact anchor top-64 candidate identity;
- complete non-anchor checkpoint coverage;
- untouched sealed-split flags.

It then trains all non-anchor, non-token-embedding, non-reranker modules with:

- factual exact-set and formal auxiliary objectives;
- 0.1 counterfactual semantic-retention weight, including the declared 0.1
  target posterior term inside that retention objective;
- effective batch 32, AdamW, peak learning rate 2e-4, 5% linear warmup then
  cosine decay, weight decay 0.01 and clipping 1.0;
- random ancestor-closed anytime budgets during generator training;
- tuning selection led first by mean H2-H4 C64 coverage.

The candidate curriculum preserves anchor top-64 through 10% of training and
anneals to the B1.5-frozen 40/24 policy by 50%. Only anchor, branch geometry,
branch mixture and trajectory sources are active; raw persistence and
transition sources remain closed.

The outer-train probe gate is:

\[
\operatorname{Mean}(C_{64,H2:H4})\ge0.985,
\qquad
C_{64,H4}\ge0.970,
\qquad
C_{64,H4,\mathrm{mismatch}}\ge0.930.
\]

H1 C64 and Recall@8 are reported independently. A generator failure stops
before constructing the ranker optimizer.

## Ranker stage

Only a gate-passing generator checkpoint may initialize the ranker stage.
Generator modules are frozen in evaluation mode, so the predicted C64 set is
deterministic during ranker training. Only `reranker.*` parameters train.

The objective is the factual HARP-RTT loss plus swap loss:

\[
0.1\,\operatorname{softplus}
\left(0.125+s_{\mathrm{intruding}}-s_{\mathrm{missing\ true}}\right).
\]

Outside-C64 misses are logged separately. The development gate is mean
H1-H4 Recall@8 at least 0.85 on the same request-disjoint outer-train probe.
Passing this gate authorizes a later, separately frozen formal-validation
evaluation; it does not open that split automatically.

## Execution order

1. Run a generator preflight on the RTX 4090; no optimizer may be constructed.
2. If memory, binding and epoch-zero audits pass, run generator seed 42.
3. Evaluate every epoch on the 32-request tuning subset.
4. Evaluate the best generator once on the 128-request probe.
5. Start the ranker only if all three learned candidate gates pass.
6. Evaluate the best ranker once on the same probe and record Recall/Coverage.
7. Checksum and mirror fresh local-NVMe outputs to persistent storage.
8. Publish the source and documentation; stop, never terminate, the RunPod.

The driver is `runpod/train_harp_rtt_b3.py`. Its preflight path is
optimizer-free and every result records `formal_validation_opened=false`,
`calibration_opened=false`, and `sealed_test_opened=false`.

## Terminal result — 2026-08-10

The generator stage completed all 12 declared epochs on the RTX 4090. Epoch 10
was selected by the frozen lexicographic tuple. Epochs 11 and 12 did not improve
candidate coverage. The ranker optimizer was never constructed.

The request-disjoint 128-request, 2,048-position probe result is:

| Metric | Result | Gate | Outcome |
| --- | ---: | ---: | --- |
| Mean H2-H4 C64 | 0.905447 | 0.985 | fail |
| H4 C64 | 0.885579 | 0.970 | fail |
| H4 prefix-mismatch C64 | 0.823856 | 0.930 | fail |
| Mean H1-H4 Recall@8 | 0.536401 | -- | reported |
| H1 C64 | 0.938570 | independent future gate 0.980 | below gate |

Per-horizon learned C64 was `0.938570 / 0.923939 / 0.906822 / 0.885579`,
and per-horizon Recall@8 was
`0.585223 / 0.548007 / 0.521884 / 0.490491`.

The selected checkpoint is `best_generator.pt`, SHA-256
`064e3207b724e47a8a77c7d44a0c4d7169df2c5184078760e2354a5e053637c2`.
All seven terminal files pass `SHA256SUMS`. The byte-identical persistent mirror
is:

```text
/workspace/LLM_prefetch_study/artifacts/harp_rtt/
  b3_generator_20260810_6a350f9/
```

The mirror also retains the successful optimizer-free preflight, the rejected
constant-learning-rate epoch-one diagnostic, and both throughput-only
microbatch smokes. The canonical source commit is
`6a350f99ab1d946fcf9b279d060614280fcf6452`.

This is a terminal generator-gate failure, not a ranker result. In particular,
the user-accepted request-disjoint B1.5 oracle reached approximately `0.982895`
mean H2-H4 C64, `0.979030` H4 C64, and `0.962117` H4 mismatch C64, whereas the
learned B3 generator retained only `0.905447`, `0.885579`, and `0.823856`.
The counterfactual tree contains substantially more useful route information
than the learned route/posterior/candidate interface recovers.

## Next experiment

Run a zero-recapture B3.1 oracle-substitution and candidate-calibration
factorial on the existing fitting and diagnostic-probe artifacts. For the
frozen epoch-ten generator, evaluate:

1. learned routes with learned branch posterior;
2. oracle selected-set routes with learned posterior;
3. learned routes with target posterior;
4. oracle routes with target posterior;
5. each condition under global top-64 and the frozen 40/24 union.

Export source-specific positive candidate mass, unique contribution, overlap
with anchor top-40, and coverage by matched prefix and first divergence depth.
This separates route translation, posterior weighting, and quota/calibration
loss without another target replay. If oracle routes plus learned posterior
fail, fix posterior supervision. If learned routes plus target posterior fail,
fix the semantic route head. If global top-64 passes while 40/24 fails, fix the
candidate union. Only after the learned source recovers a material fraction of
the accepted oracle coverage should the rich ranker or the independent H1 root
predictor be trained.
