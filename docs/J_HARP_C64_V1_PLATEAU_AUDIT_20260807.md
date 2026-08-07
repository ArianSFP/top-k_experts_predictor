# J-HARP-C64 v1 plateau audit — 2026-08-07

## Scope

This is a validation-only audit of the residual c64 reranker in:

```text
harp8/jspace_reranker.py
harp8/jspace_data.py
harp8/jspace_metrics.py
harp8/train_jspace_reranker.py
```

It investigates why the matched raw-residual run improved the frozen HARP
endpoint by only about 0.15 percentage points after two epochs and why the
matched J-Lens run showed similarly small first-epoch movement. No sealed test
partition was opened. The capture, pools, checkpoints, and training outputs
were treated as immutable.

## Executive conclusion

There is no evidence of a gross alignment, candidate-order, masking, Recall@8,
or zero-initialization implementation bug. A four-row real-data overfit reaches
98.0% Recall@8, proving that the ranker and membership labels can drive
corrective swaps.

The dominant observed failure is instead a surrogate/optimization shortcut:
the large reranker mostly learns a monotone sharpening of the already strong
frozen HARP scores. That reduces BCE, listwise, and pairwise losses on the many
already-correct candidate relations while changing too few top-8 decisions.
On a held-out checkpoint sample, the centered learned correction correlated
0.881 with the centered frozen score, and only about 0.25 of eight selections
changed per layer. The exact validation improvement corresponds to only 0.0124
net additional correct experts per layer.

The v1 information path makes that shortcut especially attractive. It drops
the stored 384-dimensional frozen HARP generator context, broadcasts one
horizon-only MTP summary to all layers, and emits one shared router-coordinate
query into 40 independently oriented SVD bases. The experimental v2 ranker
addresses these three limitations. Its first accuracy run should use an
explicit membership-ranking loss profile, without changing the preregistered
v1 default.

## Corrected router tie audit

An early exploratory calculation incorrectly used `numpy.argpartition` to
reconstruct top-8. That result is retracted. BF16-expanded router logits have
exact cutoff ties, and `argpartition` may choose an arbitrary tied expert.
The authoritative capture uses stable descending order in the full expert-ID
namespace.

The corrected audit uses:

```python
np.argsort(-raw_logits, axis=-1, kind="stable")[..., :8]
```

For candidate subsets whose array order differs from expert-ID order, it uses:

```python
np.lexsort((candidate_expert_ids, -candidate_scores), axis=-1)[..., :8]
```

On 2,048 deterministically sampled capture positions, covering all 40 layers:

| Check | Result |
|---|---:|
| stable raw-logit top-8 slot recall versus captured top-8 | 1.000000 |
| stable raw-logit exact-set rate | 1.000000 |
| top-8/top-9 exact-tie rate | 0.177820 |

Thus, the raw logits and top-8 IDs are aligned. The 17.8% cutoff-tie rate does
mean that a softmax KL over raw values cannot encode which member of an exact
tie wins the stable expert-ID rule. Membership ranking remains necessary, but
this is not evidence of corrupt capture data.

## Fixed validation anchors

The frozen HARP candidate scores have request-macro validation Recall@8:

| Horizon | Frozen base | c64 oracle coverage |
|---:|---:|---:|
| H1 | 0.82168 | 0.98927 |
| H2 | 0.80249 | 0.98119 |
| H3 | 0.78607 | 0.96965 |
| H4 | 0.76217 | 0.95627 |
| Mean H1--H4 | **0.79319** | **0.97410** |

Reaching 0.90 requires an additional:

```text
(0.900000 - 0.793195) * 8 = 0.85444
```

correct expert slots per token-layer, while the c64 pool contains about:

```text
(0.974096 - 0.793195) * 8 = 1.44721
```

recoverable slots beyond the frozen top-8. The desired ranker therefore has to
recover roughly 59% of the currently recoverable errors, not merely rescale
scores while preserving their order.

## Matched v1 results available at audit time

### Raw-residual rank-512

| Epoch | Mean H1--H4 R@8 | Gain over frozen | H1 | H2 | H3 | H4 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.793195 | 0 | 0.82168 | 0.80249 | 0.78607 | 0.76217 |
| 1 | 0.794453 | +0.001258 | 0.82365 | 0.80379 | 0.78665 | 0.76373 |
| 2 | 0.794742 | +0.001547 | 0.82397 | 0.80418 | 0.78709 | 0.76372 |

Epoch 2 adds only 0.000289 over epoch 1. The run was stopped after a complete
epoch 2 with a resumable state. This is an early plateau warning, not proof of
the five-epoch asymptote.

### J-Lens rank-512

The matched J run's first completed epoch reached 0.794358, versus 0.794453 for
raw residual at the same epoch. The difference is only -0.000095. J-space had
therefore not produced a measurable advantage by epoch 1. Its epoch-2 result
must be used for the final matched attribution.

## Evidence-ranked findings

### 1. High confidence: the loss rewards score sharpening without top-8 repair

The v1 prediction is:

```text
final_score = frozen_harp_score + learned_delta
```

The default loss is:

```text
1.00 * boundary
+ 0.50 * balanced_bce
+ 0.50 * listwise
+ 0.25 * restricted_kl
```

Boundary, BCE, and listwise supervision include many positives and negatives
that the frozen model already orders correctly. Their easiest joint solution
is to enlarge the existing separation. None explicitly isolates a frozen
top-8 false positive and forces it below a missing true member.

From raw epoch 1 to epoch 2:

| Training component | Epoch 1 | Epoch 2 | Change |
|---|---:|---:|---:|
| boundary | 0.13895 | 0.13135 | -0.00760 |
| balanced BCE | 0.24901 | 0.23977 | -0.00924 |
| listwise | 2.62379 | 2.59428 | -0.02951 |
| restricted KL | 0.48288 | 0.50958 | **+0.02670** |
| weighted total | 1.69606 | 1.67576 | -0.02030 |

The surrogate improves by 1.20%, but Recall@8 improves by only 0.029 percentage
points. Most optimization progress therefore does not cross the top-8 cutoff.

An eight-row held-out diagnostic of the epoch-2 checkpoint found:

| Delta statistic | Value |
|---|---:|
| RMS of delta | 0.8019 |
| RMS after removing per-row candidate mean | 0.6249 |
| frozen-score centered RMS | 0.6722 |
| centered delta/frozen-score correlation | **0.8810** |
| common-mode delta variance fraction | 0.3926 |

The correction has substantial magnitude, but much of it is common-mode or
aligned with the frozen ordering. Both are largely irrelevant to top-8.

### 2. High confidence: too few candidate swaps are attempted

The exact global validation gain at epoch 2 is:

```text
0.001547 * 8 = 0.01238 net correct slots per layer
```

A separate 16-row all-horizons-valid checkpoint sample produced:

| Split | Mean slots changed from frozen top-8 | Beneficial layer-rows | Harmful layer-rows | Neutral layer-rows |
|---|---:|---:|---:|---:|
| train | 0.2641 | 7.50% | 5.39% | 87.11% |
| validation | 0.2539 | 6.13% | 5.51% | 88.36% |

These sample statistics are diagnostic rather than the official complete-split
metric, but they explain the global result: the model changes roughly one slot
in every four layer-rows and only slightly more changes help than hurt. A
0.90-mean solution needs about 0.85 net additional hits per layer.

### 3. High confidence from code: v1 discards the strongest frozen latent

The c64 pools contain:

```text
generator_context.f16 [row, horizon, layer, 384]
```

but all v1 training and evaluation calls use `include_context=False`.
Candidate features retain base score, source gate, copy gate, current-route
features, and route membership, but not the latent that generated the frozen
forecast. The 33-million-parameter reranker must reconstruct that information
from raw/J history and MTP inputs using only about 1,229 training requests.

The v2 implementation makes generator context mandatory through a
function-preserving input path.

### 4. High confidence from code: MTP and router geometry are insufficiently layer-specific

V1 constructs one MTP context per horizon, `[B,H,D]`, then broadcasts it to
all 40 target layers. It cannot directly choose a different depth mixture for
each target layer.

V1 also emits router queries through one shared linear map, although the 40
router SVD coordinate systems are independently oriented. Layer embeddings and
shared nonlinear layers can approximate conditioning, but this is an
unnecessarily difficult route to 40 distinct dynamic maps.

V2 uses horizon-by-layer MTP queries and layer-specific low-rank router heads.

### 5. Medium confidence: restricted KL is unhelpful for this candidate-cutoff objective

On the deterministic 2,048-row audit, weighted score-gradient norms at frozen
initialization were:

| Component | Weighted gradient norm |
|---|---:|
| boundary | 2.089e-4 |
| listwise | 2.374e-4 |
| balanced BCE | 6.720e-5 |
| restricted KL | 2.590e-5 |

KL is not the dominant gradient. Its cosine with boundary was -0.0116, with
BCE -0.0155, and with listwise +0.0051: approximately orthogonal overall. It
also worsened materially during the second full epoch. Exact BF16 cutoff ties
make raw-distribution matching incapable of representing the authoritative
stable tie decision by itself.

Removing KL is therefore justified as a named accuracy ablation, but cannot by
itself explain or repair the roughly 10.5-point gap.

### 6. Medium confidence: two epochs are insufficient to declare an architectural asymptote

The raw run completed 2,536 optimizer steps for a 33.0-million-parameter
randomly initialized reranker. The loss was still decreasing. Zero-initialized
final output weights block upstream gradients only on the first optimizer
step; gradients flow normally thereafter, so this is not a sustained dead
path. Nevertheless, stopping at epoch 2 means the current result is an early
training diagnostic rather than a converged capacity estimate.

Complete matched comparisons should use the same epoch budget or a documented
validation early-stop. More epochs on the unchanged v1 objective are unlikely
to bridge the whole gap because the learned delta already exhibits the
score-sharpening shortcut.

## Tiny real-row overfit

The audit trained a deliberately small v1 model on four real train rows:

```text
rows = [0, 13516, 27036, 40553]
seed = 1234
steps = 60
AdamW learning rate = 3e-3
weight decay = 0
dropout = 0
model width = 32
heads = 4
FFN width = 64
temporal/axial/MTP/set blocks = 1 each
inducing points = 4
candidate coverage = 0.99355
```

| Objective | Final train R@8 | Mean swaps/layer |
|---|---:|---:|
| preregistered v1 | 0.97109 | 1.0844 |
| v1 without restricted KL | 0.97422 | 1.1234 |
| membership ranking | **0.98008** | **1.1250** |

The winning exact loss configuration was:

```python
JSpaceRerankerLossConfig(
    boundary=1.0,
    balanced_bce=0.0,
    listwise=1.0,
    restricted_kl=0.0,
    temperature=2.0,
    hard_negative_count=24,
    horizon_weights=(1.0, 1.0, 1.25, 1.5),
)
```

This establishes that candidate ordering, masking, target membership, residual
addition, and gradients permit near-ceiling fitting. It does not establish
generalization from four rows.

## Recommended implementation sequence

### Immediate, low-risk

1. Preserve the preregistered default.
2. Add an explicit CLI profile:

   ```text
   --loss-profile preregistered_v1
   --loss-profile membership_ranking_v1
   ```

3. Resolve `membership_ranking_v1` to the exact weights above and write both
   the profile name and full resolved config into checkpoints/manifests.
4. Run a short paired v2 pilot with both profiles on the same order and seed.
5. Select only on request-macro validation Recall@8; do not use surrogate loss
   as the model-selection metric.

### Next objective: explicit corrective-swap loss

Add a term focused exactly on frozen/current top-8 mistakes. For each
token-horizon-layer, define:

```text
FN = true candidate members absent from detached predicted top-8
FP = nonmembers present in detached predicted top-8
```

Then minimize:

```text
mean softplus(score_fp - score_fn + margin), fp in FP, fn in FN
```

Rows already at their c64 ceiling should contribute zero. Log:

- mean swaps from the frozen top-8;
- beneficial, harmful, and neutral swap counts;
- recovered fraction of the c64 gap;
- recall conditional on frozen error count;
- delta/base centered correlation.

This objective directly rewards crossing the cutoff instead of increasing
confidence on already-correct pairs. It should be introduced as another named
experimental profile and tested against membership ranking, not substituted
silently.

### Architecture priority

Run v2 next because it fixes the clearest information bottlenecks while
preserving epoch-zero frozen scores. If v2 remains far below 0.90, pivot to the
documented layer-specific full-256 router forecaster. A c64-only ranker cannot
repair missing candidates, and a full-router auxiliary head supplies much
richer supervision than eight membership bits.

For the later full-router model:

1. predict all 256 future expert scores separately at H1--H8;
2. use horizon-by-layer MTP hidden/router queries;
3. include frozen HARP context and dense J/raw controls;
4. train exact membership/cutoff ranking plus centered-score supervision;
5. form a candidate union from the full-router head and frozen HARP;
6. apply a small cutoff reranker only after candidate generation;
7. add causal MTP draft-token confidence and multiple branches for H2--H4
   mismatch rows.

## Reproducible audit command and artifacts

Local diagnostic and tests:

```text
harp8/audit_jspace_plateau.py
tests/test_harp8_jspace_plateau_audit.py
```

Focused test command:

```bash
pytest -q tests/test_harp8_jspace_plateau_audit.py tests/test_harp8_jspace_reranker.py
```

Remote score/tie audit command:

```bash
PYTHONPATH=/root/j_harp_c64/source_prefetch_v2 \
python -m harp8.audit_jspace_plateau \
  --capture-dir /root/j_harp_c64/inputs/compact \
  --pool-root /root/harp8_accuracy_expansion/pools/temp1_epoch5_c64_v3_validation \
  --sample-rows 2048 \
  --seed 42 \
  --active-horizons 4 \
  --output /root/j_harp_c64/audits/v1_plateau_20260807.json
```

Durable audit artifact:

```text
/workspace/LLM_prefetch_study/artifacts/harp8/j_harp_c64_runs/
  preflight_20260807/audits/v1_plateau_tie_gradient_20260807.json
  preflight_20260807/audits/v1_plateau_full_20260807.json
```

SHA-256 of the first tie/gradient audit artifact:

```text
f4a1b9aa2797f149cd2459a7d5886fb8765cbb6d1477b6c1cc2282da94b28ccf
```

SHA-256 of the expanded artifact containing the exact tiny overfit:

```text
9448a8c41c66d748c63d6e2175b38b9fa352dcfa39ebf6541d1bd77daedae197
```

The expanded diagnostic also supports a reproducible tiny overfit with:

```text
--tiny-overfit-pool
--mtp-dir
--target-features
--target-feature-rms
--router-keys
--tiny-overfit-rows 4
--tiny-overfit-steps 60
```

## Acceptance decision

Do not interpret v1's +0.15 percentage-point result as evidence that J-space
contains no useful routing signal. V1 does not give J/MTP evidence the
layer-specific or frozen-context paths needed for a clean test. It does show
that a large residual ranker plus broad confidence losses is not sufficient.

The next meaningful gate is:

```text
v2 + membership_ranking_v1
```

against the same raw/J features, seed, candidate pool, and request split. A
material J-space claim still requires the matched J-minus-raw paired bootstrap;
an architecture accuracy claim should be judged against the frozen HARP base
and c64 oracle gap.
