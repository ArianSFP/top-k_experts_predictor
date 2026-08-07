# J-HARP / J-space safe-stop handoff — 2026-08-07

## Stop state

The accuracy campaign is paused at a clean experiment boundary.

- The last active trainer completed all 12 epochs, restored its best
  validation checkpoint, wrote the complete validation tables, and finished
  a 5,000-replicate paired request bootstrap.
- The trainer exited with status 0. There are no GPU compute processes.
- No matched OOF J-Lens, dual-stream, residual-scale sweep, rank-1024 feature
  build, corrected-MTP ablation, or sealed-test evaluation was launched after
  the stop request.
- The final run, exact source snapshot, log, and checksums are preserved on
  network storage. Compact non-request-level evidence is also copied into this
  repository.

The campaign did **not** reach the approximately 0.90 mean H1--H4 Recall@8
goal. Its best valid OOF level-2 improvement is statistically resolved but
operationally negligible.

## Final OOF result

The stopped endpoint is the raw-residual rank-512, candidate-64
JSpaceV2CandidateReranker. It has 1,808,425 trainable parameters and uses the
membership_ranking_v1 objective: boundary weight 1, listwise weight 1,
membership BCE and restricted KL disabled, 24 hard negatives, and horizon
weights [1, 1, 1.25, 1.5] for H1--H4. H5--H8 are exact frozen-base
passthroughs.

| Horizon | OOF frozen base | Best model | Gain | c64 coverage ceiling |
|---:|---:|---:|---:|---:|
| H1 | 0.81362181 | 0.81500554 | +0.00138373 | 0.98750508 |
| H2 | 0.79380312 | 0.79524366 | +0.00144055 | 0.97964963 |
| H3 | 0.77988543 | 0.78128909 | +0.00140367 | 0.96925649 |
| H4 | 0.75734807 | 0.75873298 | +0.00138491 | 0.95712398 |
| **Mean H1--H4** | **0.78616461** | **0.78756782** | **+0.00140321** | **0.97338379** |

The paired request-macro gain is:

~~~text
+0.0014032140
95% CI [+0.0011840897, +0.0016335211]
410 complete validation requests, 5,000 bootstrap replicates
~~~

The best checkpoint is epoch 11. Exact-set@8 is 0.32269, 0.30958, 0.30688,
and 0.29216 at H1--H4. The result is positive in a statistical sense, but it
adds only 0.14 percentage points to the selection mean and remains 11.24
percentage points below 0.90.

The c64 candidate pool is not the limiting ceiling: its mean coverage is
0.97338. The model recovers only about 80.9% of the positives present in that
pool, whereas a 0.90 endpoint would require about 92.5%. Candidate ranking,
not candidate count, is the immediate bottleneck.

## Experiment ledger

The table below distinguishes valid comparisons from diagnostics that cannot
support promotion.

| Experiment | Result | Status and finding |
|---|---:|---|
| Historical HARP endpoint | 0.788451 mean H1--H4 | Initial full-split anchor. |
| Warm-start + direct skips | 0.790497 | Small, not practically significant gain. |
| H3/H4-weighted generator | 0.792675 | Best completed generator continuation before the temperature run. |
| Temperature-1 generator, epoch 5 | 0.793101 | Best full-split frozen generator used by the original c64 pools. |
| Original c32 reranker, epoch 2 | 0.794187 | Partial improvement; stopped early. |
| Original v1 raw rank-512 reranker, epoch 2 | 0.794742 | Small gain on an invalid in-sample/out-of-sample level-2 stack. |
| Original v1 dense-J rank-512 reranker, epoch 2 | 0.794606 | J-minus-raw = -0.00013573, 95% CI [-0.00016826, -0.00010374]. Dense J was slightly worse. |
| Full256 raw rank-512/rank-192 | 0.788114 at epoch 4 vs 0.793195 epoch 0 | Regressed because level-2 train scores were in-sample while validation scores were out-of-sample. |
| Full256 dense-J rank-512/rank-192 | 0.788107 at epoch 4 vs 0.793195 epoch 0 | Essentially identical regression to raw; representation choice was not the cause. |
| Original Full256 residual-scale diagnostic | 0.795441 at alpha 0.5 | Validation-oracle diagnostic only. It showed a weak useful correction direction and excessive learned magnitude, not a selectable result. |
| OOF Full256 raw512/output-rank16 | 0.786406 vs 0.786165 | Valid meta-train-selected alpha 1.0; +0.00024163, 95% CI [+0.00020192, +0.00027930]. Practically negligible. |
| **OOF c64 JSpaceV2 raw512, epoch 11** | **0.787568 vs 0.786165** | **Final stopped endpoint; +0.00140321, still negligible relative to the goal.** |

The OOF results use 246 inner-heldout requests for level-2 fitting and 410
outer-validation requests for model selection, with zero request overlap and
zero overlap between level-2 meta-train requests and the 983 requests used to
fit the frozen base predictor. They remain labelled
transductive_preprocessing_oof_proxy: target/MTP preprocessing was fitted
before the 983/246 split. A final scientific stack would require fold-local
preprocessing and complete request-group cross-fitting.

## What the J-space investigation established

### Literal sparse J-space was weaker than residual controls

The earlier frozen BF16 probe found:

| Input | H1 Recall@8 | H2 Recall@8 |
|---|---:|---:|
| Route history + sparse J | 0.5450 | 0.4927 |
| Route history + residual PCA | 0.6155 | 0.5497 |

Residual PCA beat sparse J in all 720/720 matched target cells at both
horizons. Literal sparse J-space is therefore a negative prior for this task.

### Dense J transport aligns layers but is not a better standalone predictor

Rank-512 PCA captures 93.23% of dense-J variance versus 81.07% of raw-residual
variance, so the lens clearly makes the shared across-layer distribution more
compressible. That did not translate into better candidate ranking: the
matched v1 dense-J arm was slightly worse than raw.

A separate train-fitted, validation-evaluated linear audit did find
complementary errors:

| Representation | Mean H1--H4 Recall@8 |
|---|---:|
| Dense J rank 512 | 0.60293 |
| Raw residual rank 512 | 0.62135 |
| Raw + dense J rank 512 | 0.62821 |

Dual minus raw was +0.00686, 95% CI [+0.00598, +0.00775]. This supports J as
an auxiliary coordinate system, not as a replacement for raw residuals. The
matched OOF J-only and dual nonlinear runs were intentionally not launched
after the safe-stop request, so no end-to-end J benefit is claimed.

## Why the models plateaued

1. **Most learned corrections sharpen the frozen ordering.** In the v1
   plateau audit, centered correction and centered base score had correlation
   0.881. Only about 0.25 of eight selected slots changed per layer-row, and
   beneficial changes barely exceeded harmful ones.
2. **The first level-2 corpus was invalid for stacking.** Its frozen predictor
   scored 0.84948 in-sample on level-2 train but only 0.79319 out-of-sample on
   validation. Both large Full256 arms learned that mismatch and regressed.
3. **OOF stacking fixed the gross distribution mismatch, not the information
   deficit.** The OOF base was 0.78462 on meta-train and 0.78616 on validation;
   corrections stopped immediately regressing, but the valid gains remained
   tiny.
4. **A single greedy MTP branch fails on the hard rows.** At H4, base Recall@8
   was about 0.8019 when the draft prefix matched the committed continuation
   and 0.5913 when it did not. About 18.5% of H4 validation rows were prefix
   mismatches. Better ranking on easy matched rows alone cannot reach 0.90.
5. **More candidates do not solve selection.** OOF c64 coverage is 0.97338,
   yet the final ranker reaches only 0.78757. Moving to c128 would mostly raise
   a ceiling that is already well above the target.
6. **Dense J cannot create information.** It is a frozen linear transport of
   a residual state. Its plausible benefit is conditioning/alignment; the
   evidence does not support treating it as privileged ground truth.

## Completed implementation and unrun work

The following code or design work is saved but was not promoted by a run:

- strict harp8_candidate_pool_v2 OOF lineage and request-overlap checks;
- exact legacy-HARP compatibility for the complete zero-initialized direct
  extension only;
- Full256 trust controls: base KL, centered-delta penalty, relative regret,
  and optional output-bias freezing;
- MTP path corrections: local diagonal, null-only-when-missing, and explicit
  horizon-by-depth attention bias;
- a v2 candidate residual-scale evaluator that selects scale only on
  level-2 meta-train and labels validation sweeps as oracle diagnostics;
- a fully verified matched J-Lens launch preflight, including the provisional
  FP16 precision-audit limitation;
- a raw/J rank-1024 feature-build preflight; and
- a dual-stream design using independent raw/J encoders plus a zero-initialized
  residual fusion. The dual source changes were **designed but not
  implemented** because the required patch sandbox failed, and work was
  stopped rather than using an unsafe workaround.

No residual-scale grid, matched OOF J arm, dual arm, rank-1024 build, or
corrected-MTP accuracy run was started.

## Data-access audit

The trainers and evaluators opened only explicit train and validation pools;
no sealed-test candidate tensor, target route, label, prediction, or metric
was used.

During a separate schema preflight, an unfiltered command accidentally printed
two shared request-catalog rows. Exactly one printed row was tagged test and
included catalog metadata/token IDs. No test tensor arrays, routes, labels,
metrics, or candidate pools were opened, and nothing in that record informed
architecture, hyperparameter, checkpoint, or model-selection decisions. All
subsequent checks used explicit train/validation allowlists. Accordingly, the
precise claim is that **sealed-test model evidence remains unopened**, while
one test catalog metadata/token record was incidentally displayed.

## Durable artifacts

Canonical network archive:

~~~text
/workspace/LLM_prefetch_study/artifacts/harp8/j_harp_c64_runs/
  oof_proxy/jspace_v2_c64_raw512_w64_rq16_seed42_20260807a
~~~

It contains best.pt, last.pt, complete training history, validation
horizon/layer/domain/position/request tables, paired bootstrap, manifest,
training log, exact source snapshot, and SHA256SUMS. It is read-only and all
entries passed checksum verification.

Key hashes:

~~~text
best.pt                         eb3cc889f0cd6b0ed8728ea107c32924001b1b036a1ba739ce4193e486f85763
last.pt                         7f5b082dce9ce7b189c9b61e50736ccce191422a7969cf74ec98aab9eaa54efe
manifest.json                   48b5e88980956c200f5f09c3169e240ce711dd6f75404b95e961e6bc13b93fad
validation_metrics.json         05200adb9dec84c96adbc3f1881facf53c804b3af2b221a973622636af8ad162
paired bootstrap JSON           baa2502c9a5721ac21d5cfe764f40868efd4fa43cd674a6823e6a1c16020dd0e
relative-path source tree       981f67f59a1d25032535dbaa5368e128e823be2a13acf73db6d0c5da7055b87f
~~~

A second independently verified backup exists at:

~~~text
/workspace/LLM_prefetch_study/artifacts/harp8/j_harp_c64_runs/
  oof_v2_raw512_c64_stop_20260807
~~~

Compact local evidence is under:

~~~text
artifacts/harp8/j_harp_c64_runs/oof_stop_20260807/raw512_c64_v2
~~~

Related records:

- J_HARP_FULL256_EXPERIMENT_LEDGER_20260807.md
- J_HARP_C64_V1_PLATEAU_AUDIT_20260807.md
- J_HARP_JLENS_COMPLEMENTARITY_AUDIT_20260807.md
- J_HARP_OOF_JLENS_PREFLIGHT_AND_ACCESS_AUDIT_20260807.md
- J_HARP_V2_CANDIDATE_RESIDUAL_SCALE_DIAGNOSTIC_20260807.md
- J_HARP_RANK1024_FEATURE_PREFLIGHT_20260807.md
- HARP8_LEVEL2_OOF_CANDIDATE_POOL_PROTOCOL_20260807.md

## Recommendation if work resumes

Do not resume by merely widening the same ranker. The evidence now says that
capacity and candidate width are not the main problem. The next study should
first test whether new causal information changes the attainable ceiling:

1. capture or construct multiple plausible MTP branches, especially for H3
   and H4 prefix-mismatch rows;
2. expose causal draft-token identity and chosen-token confidence summaries;
3. train a direct multi-horizon full-router model, with raw residual as the
   primary stream and dense J only as an independently normalized auxiliary;
4. use fold-local preprocessing and multi-fold request-group cross-fitting;
5. require a material validation gate before any further rank or parameter
   sweep; and
6. keep the sealed target-route test tensors unopened until the architecture,
   losses, and checkpoint selection are frozen.

The most defensible conclusion is a negative one: under the current
single-greedy-branch data and tested OOF residual-ranking architectures,
J-space did not yield a material improvement toward 90% Recall@8.
