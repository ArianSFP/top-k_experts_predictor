# J-HARP Full256 v1 implementation protocol — 2026-08-07

## Status

This document specifies the implemented experimental successor to the
candidate-only J-HARP rankers. The complete HARP suite passes 136 tests with
one pre-existing benign warning. A hardened Full256 CUDA pilot, cutoff-aligned
learnability gates, and one complete full-split dense-J arm have now completed.
The rank-192 swap-only 256-row gate passed, while the full-split dense-J arm
regressed from its epoch-zero baseline and therefore selected epoch zero. An
exactly matched raw rank-512 arm is running; the dual raw/J stream is
implemented but has not yet run. The sealed test split remains untouched.

The model is deliberately accuracy-first. It emits direct predictions for all
256 experts at every target layer and every horizon H1--H8. Checkpoint
selection is validation-only and prioritizes request-macro Recall@8 averaged
over H1--H4, with H2 Recall@8 as the tie-breaker.

Implementation files:

- `harp8/jspace_router_data.py`
- `harp8/jspace_router_forecaster.py`
- `harp8/jspace_router_metrics.py`
- `harp8/train_jspace_router_forecaster.py`
- `harp8/evaluate_jspace_router_checkpoint.py`
- `tests/test_harp8_jspace_router_data.py`
- `tests/test_harp8_jspace_router_forecaster.py`
- `tests/test_harp8_jspace_router_metrics.py`
- `tests/test_harp8_jspace_router_training.py`

CLI entry point:

```text
harp8-train-jspace-router-forecaster
```

## Implemented and tested correctness gates

As of the latest 2026-08-07 optimization update, the complete HARP suite
passes **136 tests** with one pre-existing benign warning. The focused Full256
files include:

```text
tests/test_harp8_jspace_router_data.py
tests/test_harp8_jspace_router_forecaster.py
tests/test_harp8_jspace_router_metrics.py
tests/test_harp8_jspace_router_training.py
tests/test_harp8_evaluate_jspace_router_checkpoint.py
```

The table distinguishes code/test completion from GPU execution:

| Gate | Implemented behavior | Verification status | Full256 CUDA status |
|---|---|---|---|
| Stable tie mining | Authoritative positives are masked first; teacher and predicted negatives use stable score-descending, expert-ID-ascending selection. | Unit-tested on exact ties and authoritative top-8 disagreement with tied raw logits. | The CUDA runs exercised the same miner; exact-tie semantics remain established by the focused unit tests. |
| Selectable epoch-zero baseline | The complete configured validation view is evaluated before training. Epoch zero is saved as resumable `best.pt` and `last.pt`, and remains best if later validation regresses. | Deliberate-regression and checkpoint-load tests pass. | Passed in the hardened pilot: epoch zero scored 0.8817871232 and remained selectable after epoch one regressed to 0.8806152474. |
| Versioned baseline expansion | The complete c64-to-256 rule is serialized as `harp8_full_router_baseline_expansion_v1` in provenance, checkpoint, inference contract, and manifest. Resume rejects a changed floor margin or contract. | Round-trip and changed-contract rejection tests pass. | Exercised by the CUDA pilot and diagnostic artifacts. |
| One-transfer evaluation | Recall@8/16, base recall, exact-set, KL, row means, and validity are packed for one `.cpu().numpy()` transfer per batch, then grouped with ordered NumPy accumulation. | Exact legacy-reference parity plus request/layer/domain order/value tests pass; the evaluator has one CPU-transfer site. | Completed in the CUDA pilot and diagnostics without a metric-integrity failure. |
| Loader integrity | Every causal numeric source and expanded baseline must be finite; candidate IDs and authoritative top-8 IDs must be in range and unique; future masks must be binary and request-bounded. | Parameterized NaN/Inf, duplicate/out-of-range top-8, censoring, and boundary tests pass. | The checked CUDA runs passed loader admission. |
| Ordered accumulation-group prefetch | One complete accumulation group is read in order, staged in compact pinned CPU tensors on CUDA, copied nonblocking once, and sliced into unchanged microbatches. | Full256 CPU integration runs in the tiny training test. The shared primitive already has value/order/dropout/gradient/resume equivalence tests. | General Full256 CUDA execution now passes; no dedicated Full256 legacy-versus-prefetch timing or equivalence result is inferred from that pilot. |
| Finite optimization | Non-finite scalar loss raises before backward; gradient clipping uses `error_if_nonfinite=True`; validation selection rejects non-finite metrics. | Normal finite loss/gradient behavior and CPU training round trip pass. | Finite CUDA forward, backward, optimizer, validation, and checkpoint execution completed. |
| v2 checkpoint contracts | Training state schema is `harp8_jspace_full_router_training_v2`; final manifest schema is `harp8_jspace_full_router_training_manifest_v2`. Load/resume reject incompatible schemas and inconsistent embedded expansion contracts. | CPU checkpoint, inference round-trip, epoch-zero-best, and resume-contract tests pass. | CUDA pilot and diagnostics wrote the versioned artifacts successfully. |

These remain implementation/correctness results, not model-accuracy evidence.
The Full256 CUDA pilot, diagnostic overfits, and first full-split dense-J arm
described below are genuine Full256 results. The matched raw arm is running,
the dual arm is implemented but unrun, and no sealed-test evaluation has
occurred.

The standalone evaluator in `harp8/evaluate_jspace_router_checkpoint.py`
scores saved `best.pt` or `last.pt` checkpoints on validation independently of
training. It refuses a test split before opening data, validates checkpoint
lineage and single/dual target-stream contracts, and can write prediction-only
top-8 rows for paired complete-request analysis.

## Completed Full256 CUDA evidence

The hardened pilot uses 13,418,235 trainable parameters. The purposes and
evidentiary limits of the completed diagnostic runs differ:

| Run | Purpose | Result | Interpretation |
|---|---|---|---|
| `pilot_j512_hardened_20260807c` | Hardened CUDA functional and memory pilot | Epoch-zero validation mean H1--H4 Recall@8 0.8817871232; epoch one 0.8806152474; peak allocated 410,528,256 bytes and peak reserved 459,276,288 bytes. | Forward, backward, optimizer, validation, checkpoint selection, and memory gates passed. The small pilot is not a generalization estimate. |
| `overfit4_predboundary_j512_20260807a` | Four-row boundary-only capacity check | First perfect training mean H1--H4 Recall@8 at epoch 58; final training value 1.0 at epoch 100. | Proves that this narrow boundary objective and model path can memorize four rows. Its four-row validation view is only a smoke check and must not be reported as validation performance. |
| `overfit256_j512_rank64_noKL_20260807a` | 256-row objective diagnostic without KL | Frozen-base training mean 0.838376; final training mean 0.822316. Epoch-zero validation mean 0.773036; final validation mean 0.729404, despite falling loss. | Negative diagnostic: the tested composite objective was misaligned with Recall@8 and failed the required real-row overfit gate. |
| rank-192 swap-only 256-row gate | Corrected cutoff-aligned learnability gate | Epoch-32 diagnostic-train mean H1--H4 Recall@8 0.9945502817; H1 0.9936279349, H2 0.9950301279, H3 0.9949380199, H4 0.9946542623. | Passed the 0.99 mean and 0.98 per-horizon in-sample gates. It is not validation evidence. |

Durable artifacts are stored at:

```text
/workspace/LLM_prefetch_study/artifacts/harp8/j_harp_c64_runs/
    pilots/full256_pilot_j512_hardened_20260807c/
    diagnostics/overfit4_predboundary_j512_20260807a/
    diagnostics/overfit256_j512_rank64_noKL_20260807a/
```

The four-row result did not rescue the original failed 256-row mixed-objective
gate. The subsequent rank-192 swap-only gate did pass, establishing that the
current output head and cutoff-aligned loss can fit 256 real rows. That result
authorized a full-split screen, but it did not predict generalization.

## Dense transported J-Lens is not literal sparse J-space

`J_ROUTE_0_REPORT_20260805.md` evaluated genuine sparse J-space: a greedy
nonnegative decomposition over token-indexed J-Lens directions. That prior
study was negative for J-specific prediction. Route + J scored 0.5450 at H1
and 0.4927 at H2; route + residual PCA scored 0.6155 and 0.5497, and residual
PCA beat sparse J in all 720/720 matched cells at each horizon.

Full256 instead receives a dense target stream formed from the complete
post-layer residual transported through the averaged J-Lens, followed by
train-only PCA and log RMS. This linear transport cannot create information;
the hypothesis is that its shared final-layer coordinate system may improve
conditioning. It is therefore compared against raw-residual PCA and a
separately projected dual stream. Results from this branch must be called
"dense transported J-Lens," not literal sparse J-space.

## Full-split status

The complete dense-J rank-512 arm
`fullsplit_j512_rank192_swapguard_seed42_20260807a` used 40,557 training rows,
13,530 validation rows, output rank 192, batch/evaluation batch 128, seed 42,
and priority weights 1.0 for H1--H4 and 0.1 for H5--H8. Validation
request-macro mean H1--H4 Recall@8 was:

| Epoch | Validation | Diagnostic train |
|---:|---:|---:|
| 0 | **0.7931948095** | -- |
| 1 | 0.7900848673 | 0.84044215 |
| 2 | 0.7890525062 | 0.84142800 |
| 3 | 0.7881699211 | 0.84268918 |
| 4 | 0.7881068831 | 0.84361414 |

Epoch zero remained the selected checkpoint. The opposite movement of the
diagnostic training subset and validation shows overfit/objective
generalization failure under this configuration; it is not evidence that
dense-J features are empty.

The exactly matched raw rank-512 arm
`fullsplit_raw512_rank192_swapguard_seed42_20260807a` is running at this
snapshot. No raw outcome is claimed. The dual implementation uses independent
normalization/projection for J and raw streams and a learned per-cell softmax
gate. Legacy single-stream checkpoint loading and epoch-zero passthrough remain
exact. The dual full-split arm has not yet run.

A separate validation-only linear complementarity audit found mean H1--H4
Recall@8 0.60293267 for dense J512, 0.62135093 for raw512, and 0.62820948 for
dual512. Dual minus raw was +0.00685855 with paired request-bootstrap 95% CI
[+0.00597648, +0.00774619], positive at H1, H2, H3, and H4. This supports
testing learned dual fusion, while preserving raw as the stronger individual
stream. Its report and JSON are persisted, but its one-off execution code was
ephemeral and was not retained; it is not presently a runnable audit command.

## Full-split throughput diagnosis

The pre-optimization benchmark was approximately 99 seconds at batch 64 and
77 seconds at batch 128, with about 2.50 GiB and 4.50 GiB peak allocated CUDA
memory respectively. Batch 128 was chosen for the matched arms. The process
used roughly 368% CPU and 29 GiB RSS while the GPU was often under-occupied.
The source data were page-cached (`read_bytes=0`), making CPU materialization,
ranking, and synchronization the observed bottleneck rather than storage or
VRAM.

Exact-semantics optimizations now remove redundant candidate rereads and
full-namespace sorts, skip zero-weight loss branches, share the stable
predicted ranking between losses, defer training metric transfers, and reuse
persisted best-validation metrics. The optimized code passes the full HARP
suite. No new throughput number is claimed until the same benchmark is rerun.

## Why this branch exists

The frozen HARP endpoint obtains approximately 0.793 mean H1--H4 Recall@8.
The c64 oracle ceiling is approximately 0.974, but the candidate-only J-HARP
rankers have so far made only very small gains. Their main limitations are:

1. they can reorder only the fixed c64 candidate set;
2. they do not directly supervise a complete future router distribution;
3. they use future-router geometry only as a candidate scalar; and
4. their output path cannot discover an expert absent from the frozen pool.

Full256 v1 removes those restrictions. The fixed pool remains only a causal
epoch-zero baseline and a source of the frozen generator context.

## Exact causal input contract

Only these fields cross `model.forward`:

```text
base_router_scores      [B,8,L,E]
route_history           [B,T,L,E]
route_mask              [B,T,L]
j_states                [B,T,L,D_J]
j_mask                  [B,T,L]
secondary_target_states [B,T,L,D_2] optional
secondary_target_mask   [B,T,L] optional
generator_context       [B,8,L,D_G]
mtp_states              [B,N,S,D_M] or [B,N,D_M]
mtp_router_logits       [B,N,E]
mtp_mask                [B,N]
within_request          [B]
```

For the current production geometry:

```text
L=40, E=256, T=3, N=6, D_J=513 with log-RMS, D_G=384, D_M=2048.
```

In a dual arm, `D_2` is the independently prepared width of the secondary
stream, including its own log-RMS feature. Both stream identities and lineage
are serialized; the fields are absent for legacy single-stream runs.

The loader and model independently apply an exact allowlist. Future router
scores, authoritative future top-8 IDs and future-validity masks remain in the
outer loss/evaluation batch only.

Explicitly forbidden inputs include:

- actual or prefix acceptance;
- first rejection depth;
- future committed token IDs;
- `mtp_draft_target_token_ids`;
- `mtp_draft_target_logprobs`;
- future cache/transfer outcomes; and
- any derived prefix-match flag.

The MTP inputs are informationally causal from the committed prefix, but the
existing capture did not make them ready by the measured runtime decision
deadlines. Therefore this remains a
`token_end_informational_upper_bound`, with `mtp_timing_causal=false`. No
deployment-latency claim is permitted.

## Label construction and cutoff-tie semantics

For source capture row `r` and horizon `h`, supervision is read from row
`r+h` only when that row remains inside the same complete request. The loader
uses:

```text
raw_router_logits.npy   -> teacher_router_scores
top8_expert_ids.npy     -> authoritative target_top8
```

The BF16-expanded router logits contain exact cutoff ties. Consequently:

1. native `top8_expert_ids.npy` is authoritative;
2. target membership is never reconstructed with `argpartition` or raw-logit
   `topk`;
3. hard negatives are selected only after masking all eight authoritative
   positives;
4. prediction metrics use stable score-descending, expert-ID-ascending
   tie-breaking; and
5. full-router KL is auxiliary because a tied float vector cannot encode the
   discrete native cutoff identity by itself.

Membership and boundary objectives, not KL alone, supervise tied top-8
identity.

## Frozen HARP full-namespace baseline

The immutable c64 pool stores frozen HARP candidate IDs and scores. For every
horizon/layer row, Full256 v1 creates a finite 256-vector by:

1. assigning each retained candidate its frozen HARP score; and
2. assigning every absent expert `min(candidate_score) - 1.0`.

Candidate IDs must be unique. A stable tie audit proves that expanding the
namespace preserves the frozen candidate top-8 exactly. The output head is
zero initialized, so epoch-zero model scores are bit-identical to this
baseline.

This construction gives the experiment two useful properties:

- it starts at the known HARP endpoint rather than relearning it; and
- every one of the 256 experts can be promoted by the learned residual.

## Architecture

### Causal route/target-state history

Each lag/layer cell independently projects:

- one dense target-state stream (raw residual PCA or dense transported J-Lens
  PCA), with an optional independently normalized second stream; and
- the centered native target-router vector.

When both raw and J target streams are present, independent RMSNorm/projection
paths feed a learned per-cell softmax gate; their unrelated PCA coordinates
are never added directly. The fused target-state representation and centered
route stream are then combined. Content-dependent temporal weights pool the
three causal lags, after which two axial Transformer blocks mix all 40 target
layers.

### Frozen generator context

The context already stored in `generator_context.f16` is projected separately
for every horizon and layer. It preserves information learned by the frozen
HARP generator instead of forcing the new forecaster to reconstruct that
latent solely from J/MTP features.

### Native MTP memories

MTP hidden states and MTP router logits are encoded as separate memories. They
do not meet until the final source-fusion stage. Each horizon/layer query reads
both memories independently.

The model also exposes a direct depth-diagonal source:

```text
H1 <- captured MTP depth 1
...
H6 <- captured MTP depth 6
H7/H8 <- explicit learned missing source
```

H7/H8 never silently alias depth 6. A learned attention read over all six
depths remains available at every horizon.

### Direct full-router head

Every `(horizon, target layer)` has its own configurable low-rank residual map.
The original default is rank 64:

```text
context[192] -> rank[64] -> expert scores[256]
```

This avoids pretending that the 40 independently oriented router coordinate
systems share one output map. The correction is added to the expanded frozen
HARP baseline.

The learnability gate and matched full-split screens override this to output
rank 192. The rank-64 parameter totals below describe the original default,
not those rank-192 experiments.

Production-default parameter count:

```text
total trainable parameters: 13,418,235
output heads:                9,256,960
route/J history encoder:     1,122,042
native MTP encoder:          1,339,264
```

The default microbatch is four complete token records with eight-way gradient
accumulation. BF16 autocast, TF32 matrix multiplication, fused AdamW and FP32
loss reductions are enabled on CUDA. The profile is intentionally below 20M
parameters and is designed to fit comfortably on a 24 GiB RTX 3090.

## Objective

The loss is a weighted combination of:

1. authoritative top-8 versus high-scoring true-router negatives;
2. authoritative top-8 versus dynamically mined predicted false positives;
3. missing authoritative positives versus false experts occupying predicted
   top-8 slots (`top8_swap`);
4. balanced full-namespace membership loss;
5. forward full-router KL; and
6. centered-logit Smooth-L1.

Default coefficients:

| Component | Weight |
|---|---:|
| true-router boundary | 1.00 |
| predicted false-positive boundary | 1.00 |
| top-8 swap | 0.00 |
| full top-8 membership | 0.25 |
| forward router KL | 0.25 |
| centered score | 0.05 |

H1--H4 receive weight 1.0 each and H5--H8 receive weight 0.5 each. All eight
horizons are directly supervised; the weighting reflects the current priority
without making later outputs passthroughs.

The matched rank-192 swap-guard screen is a deliberate override, not the
default table above. It uses top-8 swap 1.0, predicted-boundary 0.10,
true-router boundary 0.05, router KL 0.02, centered score 0.01, membership
0.0, and horizon weights 1.0 for H1--H4 and 0.1 for H5--H8. Raw, dense-J, and
dual representation arms must keep this profile fixed for attribution.

## Metrics and selection

Validation reports:

- Recall@8 and Recall@16 for H1--H8;
- frozen-HARP baseline Recall@8 and paired point gain;
- exact-set@8;
- router KL;
- request-macro and micro aggregation;
- per-layer and per-domain rows;
- mean H1--H4 Recall@8;
- H2 Recall@8; and
- mean H1--H8 Recall@8.

Checkpoint ordering is lexicographic:

```text
(validation request-macro mean H1--H4 Recall@8,
 validation request-macro H2 Recall@8)
```

The trainer exposes only `--train-pool` and `--validation-pool`. There is no
test-pool argument. Both the checkpoint and final manifest attest
`sealed_test_accessed=false`.

## Required experiment sequence

1. Run the focused local test suite.
2. Run a 32-row train / 16-row validation CUDA pilot.
3. Verify epoch-zero exact baseline equality and finite gradients.
4. Overfit 128--256 real training rows; training Recall@8 should approach the
   attainable full-namespace target. Failure is an implementation/objective
   warning, not evidence that more data is needed.
5. Run matched raw-residual and J-Lens versions on the same immutable split.
6. Select only on untouched validation requests.
7. Compare paired complete-request gains against the frozen HARP baseline.

Current status: steps 1--4 are complete. The original mixed-objective 256-row
diagnostic failed, but the corrected rank-192 swap-only gate passed. The
dense-J full-split arm in step 5 completed negatively with epoch zero selected;
the exactly matched raw arm is running, and the dual arm remains unrun. Steps
6--7 are not complete.

Do not open the sealed test partition until architecture, loss and checkpoint
selection are frozen.

## Example local-NVMe pilot

Adapt paths to the staged run root. A representative command is:

```bash
python -m harp8.train_jspace_router_forecaster \
  --enable-experimental-full-router \
  --train-pool /root/j_harp_c64/inputs/train_pool_c64 \
  --validation-pool /root/j_harp_c64/inputs/validation_pool_c64 \
  --capture-dir /root/j_harp_c64/inputs/capture \
  --mtp-dir /root/j_harp_c64/inputs/mtp \
  --target-features /root/j_harp_c64/features_controls_r128_256_512_v1/j_lens/rank512/features_normalized.npy \
  --target-feature-rms /root/j_harp_c64/features_controls_r128_256_512_v1/j_lens/rank512/log_rms.npy \
  --precision-audit /root/j_harp_c64/precision_audit_n256/audit_fp16_roundtrip_v1/precision_audit.json \
  --allow-provisional-j-without-probe \
  --output /root/j_harp_c64/runs_full_router_v1/pilot_j_rank512_seed42 \
  --device cuda:0 \
  --epochs 1 --minimum-epochs 1 --patience 1 \
  --microbatch-size 2 --evaluation-batch-size 2 \
  --gradient-accumulation 2 \
  --max-train-rows 32 --max-validation-rows 16
```

The hardened pilot confirms ample memory headroom, and the corrected real-row
overfit gate has passed. The command remains a small functional example rather
than the matched full-split profile. If the staged paths differ, use the
immutable manifests rather than copying values into a new untracked dataset.

## Acceptance gates

The implementation is admitted to a production validation run only when:

- all focused tests pass;
- epoch-zero full scores equal the frozen baseline exactly;
- authoritative target top-8 is never reconstructed from raw tied logits;
- the exact causal input allowlist rejects acceptance/future-token fields;
- a tiny checkpoint round-trip reproduces scores exactly;
- train and validation request IDs are disjoint;
- the trainer has no sealed-test interface;
- a real-row overfit demonstrates large top-8 recovery; and
- a CUDA pilot records peak VRAM below 24 GiB.

The CUDA functional and memory gates pass. The original mixed-objective
256-row diagnostic failed, but the corrected rank-192 swap-only gate passed
with 0.9945502817 mean H1--H4 training Recall@8, so matched full-split
validation screens were admitted. The dense-J arm completed with epoch zero
best; raw is running and dual is pending. The sealed test partition has not
been accessed.

Approaching 0.90 mean H1--H4 Recall@8 remains an empirical goal, not an
acceptance assertion. If this direct model still plateaus materially below the
goal, the next interventions are predicted MTP-prefix reliability, causal
draft-token embeddings, and multi-branch MTP capture—not further widening of
the old c64-only reranker.
