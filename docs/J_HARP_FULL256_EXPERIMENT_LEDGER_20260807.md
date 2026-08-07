# J-HARP Full256 experiment ledger — 2026-08-07

## Scope

This ledger records every Full256/J-space experiment used to develop an
H1--H8 expert-router forecaster, with validation selection focused on mean
H1--H4 Recall@8. The sealed test partition is not accessible to the trainer
and has not been opened.

The scientific target is approximately 0.90 validation request-macro mean
H1--H4 Recall@8. This is a goal, not a claimed result.

## Development data and level-2 split warning

- Original training pool: `temp1_epoch5_c64_v3_train`, 40,557 rows.
- Original validation pool: `temp1_epoch5_c64_v3_validation`, 13,530 rows.
- Target capture: 2,048 complete requests, 40 layers, 256 routed experts.
- Native MTP chain: six depth-aligned states and router vectors per source
  position.
- Dense target representation: rank-512 PCA of a full residual transported
  through the averaged J-Lens, plus log RMS, unless an arm explicitly states
  otherwise. This dense transported state is not the literal sparse,
  nonnegative J-space decomposition evaluated in `J_ROUTE_0_REPORT_20260805.md`.
- Decision profile: `token_end_informational_upper_bound`; the current MTP
  evidence is informationally causal but was not ready by the old measured
  runtime cache-decision deadlines.

The original Full256 training pool is now known to be unsuitable for a
scientific level-2 stacking result: its HARP scores and generator contexts
were produced on requests used to fit HARP. The validation pool was
out-of-sample. This creates a base-output distribution mismatch between
level-2 training and validation. Those original runs remain useful controls,
but model promotion now requires request-disjoint out-of-fold (OOF) base
predictions. The first OOF proxy and its limitations are recorded below.

## Fixed baseline and matched representation result

The frozen HARP endpoint has approximately 0.79319 validation request-macro
mean H1--H4 Recall@8. On the identical candidate-only epoch-2 evaluation,
rank-512 J-Lens scored 0.79460631 and rank-512 raw residual scored
0.79474204. The paired J-minus-raw difference was -0.00013573 with a 95%
request bootstrap interval of [-0.00016826, -0.00010374]. Dense J-Lens is
therefore not a demonstrated standalone improvement in that restricted
ranker.

## Literal sparse J-space versus dense transported J-Lens

Two distinct representations must not be conflated:

- `J_ROUTE_0_REPORT_20260805.md` evaluated literal sparse J-space: a greedy
  nonnegative decomposition over token-indexed J-Lens vectors. That study was
  negative for J-specific prediction. Route + J reached 0.5450 at H1 and
  0.4927 at H2, while route + residual PCA reached 0.6155 and 0.5497. Residual
  PCA beat sparse J in all 720/720 matched target cells at both horizons.
- The present Full256 branch uses a dense transported state, computed as the
  full post-layer residual multiplied by the per-layer averaged J-Lens, then
  represented with train-only PCA and log RMS. This is a learned-coordinate
  conditioning experiment, not a replication of the sparse J-space
  intervention.

Because the dense transport is linear, it cannot create information absent
from the raw residual. Its possible value is alignment and conditioning across
layers. Raw, dense-J, and dual-stream arms are therefore required controls.

## Validation-only representation complementarity audit

A frozen linear probe on 4,096 deterministic train source rows was evaluated
on all 12,300 H1--H4-eligible validation rows from 410 complete requests:

| Representation | H1 | H2 | H3 | H4 | Mean H1--H4 |
|---|---:|---:|---:|---:|---:|
| Dense J rank 512 | 0.65921 | 0.60807 | 0.58738 | 0.55707 | 0.60293 |
| Raw residual rank 512 | 0.68045 | 0.62731 | 0.60441 | 0.57324 | 0.62135 |
| Dual rank 512 | 0.68754 | 0.63356 | 0.61132 | 0.58041 | 0.62821 |

Dual minus raw was +0.00685855 request-macro mean Recall@8, with paired 95%
CI [+0.00597648, +0.00774619], and was positive at every priority horizon.
This justifies a strictly matched dual-stream Full256 arm, but does not show
that dense J alone is better than raw. The audit report and machine-readable
summary are preserved in `J_HARP_JLENS_COMPLEMENTARITY_AUDIT_20260807.md` and
`artifacts/harp8/j_harp_c64_runs/jlens_complementarity_audit_20260807/audit_summary.json`.
The reproducer is now preserved as
`harp8/audit_jlens_complementarity.py`, with regression coverage in
`tests/test_harp8_jlens_complementarity_audit.py`.

## Full256 implementation and CUDA pilot

Full256 predicts `[B,8,40,256]` scores and adds a learned correction to an
exact epoch-zero expansion of the frozen HARP c64 scores. The implementation
uses authoritative native top-8 IDs for labels, stable score-descending and
expert-ID-ascending tie handling, a causal input allowlist, validation-only
checkpoint selection, and no test-pool CLI.

The hardened CUDA pilot `pilot_j512_hardened_20260807c` passed:

- 13,418,235 trainable parameters at output rank 64;
- epoch-zero validation smoke Recall@8 mean H1--H4 0.88178712;
- epoch-one validation smoke Recall@8 mean H1--H4 0.88061525;
- peak CUDA allocated 410,528,256 bytes;
- peak CUDA reserved 459,276,288 bytes.

The 16-row validation prefix is one request and is only a functional smoke
test, not an accuracy estimate.

The production loader/loss path has since been hardened for exact semantics
and lower host overhead. The complete HARP test suite passed 169 tests with
one pre-existing benign warning. A standalone validation-only checkpoint
evaluator is now implemented in
`harp8/evaluate_jspace_router_checkpoint.py`; it refuses the sealed test split,
checks checkpoint provenance and target-stream contracts, and can export
prediction-only top-8 rows for paired validation analysis.

## Learnability and loss diagnosis

### Original mixed objective, 256 rows

`overfit256_j512_rank64_noKL_20260807a` lowered its scalar loss but failed the
training-recall gate:

- frozen-base training mean H1--H4 Recall@8: 0.83837603;
- final epoch-10 training Recall@8: 0.82231620;
- epoch-zero validation-prefix Recall@8: 0.77303609;
- final validation-prefix Recall@8: 0.72940351.

The failure identified three objective problems: an absolute-zero membership
BCE despite arbitrary score offsets, dilution by H5--H8, and hundreds of easy
pairwise margins overwhelming the few swaps that determine top-8 recall.

### Four-row capacity controls

The boundary-only rank-64 run `overfit4_predboundary_j512_20260807a` reached
1.0000 training Recall@8 at epoch 58 and remained 1.0000 at epoch 100. This
rules out a broken target path or a fundamental inability to represent four
real examples.

An opt-in `top8_swap` loss was then implemented. It compares only missing
authoritative positives with false experts occupying predicted top-8 slots,
is invariant to a row-wise score offset, uses evaluation-consistent stable
ties, and is exactly zero for a correct set. The swap-only four-row run
`overfit4_swaponly_j512_rank64_20260807a` reached 0.9971 by epoch 32 and
1.0000 by epoch 58.

### Output-rank gate, 256 rows

With the corrected H1--H4 cutoff profile over 16 epochs:

| Output rank | Final diagnostic-train mean H1--H4 Recall@8 |
|---:|---:|
| 64 | 0.88843351 |
| 192 | 0.94259069 |

This 5.42-point difference is a material in-sample capacity result. Rank 64
must not be treated as saturated for the accuracy-first architecture.

The rank-192 swap-only 256-row gate completed at epoch 32 with diagnostic-train
mean H1--H4 Recall@8 **0.9945502817**. Its horizon values were H1 0.9936279349,
H2 0.9950301279, H3 0.9949380199, and H4 0.9946542623. It therefore passed
the predeclared 0.99 mean gate and the 0.98 per-priority-horizon gate. This is
an in-sample learnability result, not validation evidence.

## Full-split matched representation arms

The first complete matched arm,
`fullsplit_j512_rank192_swapguard_seed42_20260807a`, used dense transported
J rank 512 and the complete 40,557-row training / 13,530-row validation pools.
Its validation request-macro H1--H4 Recall@8 was:

| Epoch | Validation mean H1--H4 Recall@8 | Diagnostic-train mean H1--H4 Recall@8 |
|---:|---:|---:|
| 0 | **0.7931948095** | -- |
| 1 | 0.7900848673 | 0.84044215 |
| 2 | 0.7890525062 | 0.84142800 |
| 3 | 0.7881699211 | 0.84268918 |
| 4 | 0.7881068831 | 0.84361414 |

The epoch-zero frozen HARP expansion remained best. Training-subset recall
rose while validation recall fell, so this is a negative dense-J-only
generalization result under the tested rank-192 swap-guard objective. The run
stopped at its validation gate; it must not be summarized as evidence that the
representation contains no signal.

The exactly matched raw-residual rank-512 arm
`fullsplit_raw512_rank192_swapguard_seed42_20260807a` also completed:

| Epoch | Raw validation mean H1--H4 Recall@8 | Raw diagnostic-train mean H1--H4 Recall@8 |
|---:|---:|---:|
| 0 | **0.7931948095** | -- |
| 1 | 0.7901231611 | 0.84048942 |
| 2 | 0.7890785305 | 0.84160017 |
| 3 | 0.7882163258 | 0.84269900 |
| 4 | 0.7881144687 | 0.84353031 |

Raw and dense-J behavior was effectively identical under this matched arm:
at epoch 4 raw exceeded J by only 0.0000075855. Both learned corrections
improved their diagnostic training subsets while damaging validation at full
residual scale. This rules out J-space versus raw residual representation as
the primary cause of this particular failure.

The separately normalized, learned-gate dual J/raw stream is implemented and
backward-compatible with single-stream checkpoints. Its Full256 experiment
was deliberately deferred after the stacking audit below; running a larger
dual model on the same invalid in-sample level-2 pool would not resolve the
identified mismatch.

## Train-versus-validation stacking diagnosis

The frozen HARP base itself has a large level-2 role gap on the original
pools:

| Base-output role | Mean H1--H4 Recall@8 |
|---|---:|
| In-sample HARP training outputs | 0.84948215 |
| Out-of-sample outer validation outputs | 0.79319481 |

The per-horizon in-sample values are 0.84906050, 0.84876774, 0.85212506,
and 0.84797532 for H1--H4. The corresponding validation values are
0.82155651, 0.80244904, 0.78588293, and 0.76289076. The gap therefore grows
with horizon, from about 2.75 points at H1 to 8.51 points at H4.

This is not explained by a gross change in route persistence or expert
frequency: train/validation persistence was 0.27898/0.27919 and the
train-versus-validation expert-frequency correlation was 0.9992. The direct
failure mode is instead the classic level-2 stacking mismatch: Full256 saw
optimistic, in-sample HARP scores and contexts during fitting, then had to
correct genuinely out-of-sample HARP predictions at validation. The
rank-192 head also places 27.61 million of its 31.77 million parameters
(86.9%) in 320 independent layer-by-horizon output heads, making it easy to
fit this mismatch.

Consequently, no dual-stream or rank-1024 arm is promoted from the original
pool. The scientific path is OOF base generation, then a smaller level-2
control.

## Validation-oracle residual-scale diagnostics

The last epoch-4 checkpoints were evaluated as

`base_router_scores + alpha * learned_delta`

for alpha in `{0,.01,.02,.05,.1,.2,.35,.5,.75,1}`. Both representations
peaked at alpha 0.5:

| Checkpoint | Alpha | H1 | H2 | H3 | H4 | Mean H1--H4 | Mean H1--H8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense J rank 512, last | 0.5 | 0.82493442 | 0.80486210 | 0.78768441 | 0.76427466 | 0.795438897 | 0.748805588 |
| Raw residual rank 512, last | 0.5 | 0.82492541 | 0.80484876 | 0.78770679 | 0.76428253 | **0.795440873** | **0.748807681** |

Relative to the common alpha-zero base, the raw alpha-0.5 H1--H4 gain is
0.0022460634 and the dense-J gain is 0.0022440874. The nearly identical
curves show that the learned correction direction contains weak validation
signal but its unconstrained magnitude overshoots. This motivated explicit
base-KL, centered-delta, relative-regret, and output-bias-freezing controls.

These are **validation-oracle diagnostics**, not deployable selected models:
alpha 0.5 was selected on the same validation sweep being reported. Paired
request bootstrap confidence intervals for these oracle choices are pending.

## Leakage-safe OOF candidate-pool protocol

`harp8_candidate_pool_v2` separates three facts that the old `split` field
overloaded: source-row selection (`source_data_split`), immutable outer role
(`source_offline_split`), and the downstream level-2 role (`level2_split`).
Every level-2 training pool must prove `base_fit_excluded=true`, carry a
stable base-fold ID and reproducible base-train request IDs, and recompute a
zero base-fit overlap. The Full256 trainer also rejects train/validation
request overlap. Historical v1 validation pools remain readable; v1 training
pools fail closed without an explicit legacy API override. Details are in
`HARP8_LEVEL2_OOF_CANDIDATE_POOL_PROTOCOL_20260807.md`.

Two real compatibility failures were found and corrected while producing the
proxy:

1. The epoch-30 HARP checkpoint predates later direct-skip modules, so strict
   loading initially failed on missing keys. A named
   `zero_initialized_direct_extension` profile now adds only zero-initialized
   extension parameters, preserving the historical checkpoint function.
2. The original strict-lineage check required one common split-manifest
   identity. That cannot describe an inner-heldout level-2 training pool and
   immutable outer validation simultaneously. Source-row and base-fit
   manifests are now separate; the trainer verifies base-fold lineage and
   request exclusion rather than falsely requiring the two source manifests
   to be identical.

The complete HARP suite passed 169 tests after the OOF, trust-region, MTP-path,
and evaluation additions.

## One-fold OOF proxy result

The first proxy uses the epoch-30 HARP checkpoint trained on 983 inner-train
requests. Its 246 held-out inner requests (8,118 rows) become level-2 training
data; the immutable 410-request outer validation (13,530 rows) remains the
level-2 validation set. Both exported pools report zero overlap with the
base-fit request IDs and zero train/validation request overlap.

This is explicitly a `transductive_preprocessing_oof_proxy`, not the final
cross-fit result: target/MTP preprocessing was fit before the 983/246 inner
split. A strict result still requires fold-local preprocessing and complete
request-group cross-fitting.

The OOF base distributions are well matched:

| Level-2 role | Requests | Rows | Base mean H1--H4 Recall@8 |
|---|---:|---:|---:|
| Inner-heldout meta-train | 246 | 8,118 | 0.784618 |
| Outer validation | 410 | 13,530 | 0.7861646170 |

Outer-validation base Recall@8 across H1--H8 is 0.8136218244, 0.7938031271,
0.7798854366, 0.7573480800, 0.7326421995, 0.7139990851, 0.6872482034, and
0.6596063068.

A compute-minimal raw-rank-512, output-rank-16 Full256 proxy completed three
epochs:

| Epoch | Validation mean H1--H4 Recall@8 | H2 Recall@8 | Mean H1--H8 Recall@8 |
|---:|---:|---:|---:|
| 0 | 0.7861646170 | 0.7938031271 | 0.7422692829 |
| 1 | **0.7864062495** | **0.7940255929** | 0.7424323996 |
| 2 | 0.7863982333 | 0.7940017743 | **0.7424371556** |
| 3 | 0.7856808802 | 0.7932700670 | 0.7419694937 |

Epoch 1 is the current H1--H4-selected best, a point gain of only
0.0002416324 over epoch zero. Its H1--H4 recalls are 0.8140013051,
0.7940255929, 0.7801445828, and 0.7574535171. The gain is positive at each
priority horizon but is very small in practical terms.

The only valid scale-selection set is the 246-request level-2 meta-train
pool. Its scale grid selected alpha 1.0: mean H1--H4 rose from 0.784617647 to
0.784909306, a paired request-macro gain of 0.000291659 with 95% bootstrap CI
[+0.000244798, +0.000339816] (5,000 replicates).

That frozen alpha 1.0 was then applied once to the 410-request outer
validation set. Mean H1--H4 rose from 0.7861646170 to 0.7864062495, a paired
gain of +0.0002416324 with 95% request-bootstrap CI
[+0.0002019221, +0.0002793009] (5,000 replicates). Alpha 1.0 also happened to
be the best point on the diagnostic outer-validation grid, but that fact was
not used for the valid selection: alpha was already fixed by meta-train.

| Horizon | Fixed-alpha gain | Paired 95% request CI |
|---:|---:|---:|
| H1 | +0.0003794807 | [+0.0003143417, +0.0004432280] |
| H2 | +0.0002224658 | [+0.0001617282, +0.0002827268] |
| H3 | +0.0002591462 | [+0.0002035735, +0.0003157025] |
| H4 | +0.0001054371 | [+0.0000454713, +0.0001636246] |
| H5 | +0.0000688605 | [+0.0000194495, +0.0001172204] |
| H6 | +0.0001244010 | [+0.0000732250, +0.0001763934] |
| H7 | +0.0000982385 | [+0.0000468608, +0.0001476398] |
| H8 | +0.0000469042 | [-0.0000076220, +0.0001017311] |

The aggregate H1--H4 gain is therefore statistically supported on this
proxy, and H1--H7 are individually positive at the reported intervals; H8 is
not resolved. The effect is nevertheless practically negligible--about 0.024
percentage points in the selection mean--and fails the accuracy-first
promotion criterion. It does not justify a larger dual or rank-1024 run.

The proxy nevertheless supports the stacking diagnosis: once base outputs
have matched OOF status, a small head no longer produces the 0.3--0.5 point
immediate degradation seen in the original rank-192 runs. It does not yet
support a claim that Full256 materially improves HARP.

## MTP alignment and source-path audit

The MTP path audit found three avoidable ambiguities in the legacy Full256
path:

- the nominal horizon-matched diagonal consumed depth-contextualized memory,
  so a nonmatching draft depth could influence (for example) the H1 diagonal;
- the learned null memory competed with real MTP nodes even when at least one
  real node was present;
- all-depth attention had content attention but no explicit per-horizon,
  per-native-depth positional prior.

Opt-in corrections now provide a strictly node-local diagonal before
cross-depth encoding, expose the null memory only for all-missing rows, and
add separate zero-initialized per-head horizon-by-depth biases for hidden and
router attention. H1 maps to native depth 1 through H6/depth 6; H7 and H8 use
explicit learned-missing values and never alias depth 6. Future labels and
acceptance outcomes remain excluded from the forward input allowlist.

All corrections default off so historical config serialization, parameter
names, state-dict layout, and forward semantics remain loadable. The audit,
contracts, and regression tests are documented in
`J_HARP_FULL256_MTP_PATH_CORRECTIONS_20260807.md`. No accuracy result is
claimed for the corrected MTP path yet.

## Throughput and bottleneck audit

The original full-split path took approximately 99 seconds per benchmarked
step group at batch 64 and 77 seconds at batch 128, with peak allocated CUDA
memory about 2.50 GiB and 4.50 GiB respectively. Batch 128 was selected for
the matched arms.

The process used roughly 368% CPU and about 29 GiB resident memory while the
GPU was frequently under-occupied. Storage reads were page-cached
(`read_bytes` remained zero), so the dominant bottleneck was CPU-side batch
materialization, ranking work, and synchronization rather than network or
NVMe throughput or VRAM capacity. Exact-semantics optimizations now avoid
redundant candidate reads and full-namespace sorts, conditionally skip
zero-weight loss branches, share stable predicted rankings, defer training
metric transfers, and reuse persisted best-validation metrics. These changes
passed the full HARP test suite, but no post-optimization full-split timing is
claimed yet.

## Next gated experiments

1. Treat the one-fold result as a protocol pilot. Promote Full256 only if a
   strict multi-fold, request-group cross-fit with fold-local preprocessing
   produces a materially larger validation gain.
2. If that gate passes, run a matched dual raw/J arm and trust-region
   controls. Do not spend rank-1024 or dense-head compute on in-sample base
   pools.
3. Then ablate the corrected MTP diagonal, null masking, all-depth attention,
   and horizon-depth bias on the winning target representation.
4. Freeze architecture and checkpoint-selection rules before any sealed-test
   access. The sealed test remains unopened.

## External method interpretation

The Jacobian lens is an average linear transport from a layer's residual
stream into the final-layer basis. It was designed to expose representations
that are disposed to affect future verbal output. That makes it a plausible
cross-layer alignment prior, but not a guarantee of retaining layer-local MoE
routing information. Raw and J streams are therefore compared separately and
jointly. The prior literal sparse J-space study was negative, and the present
dense transport is a different conditioning experiment; J is not privileged
as ground truth.
