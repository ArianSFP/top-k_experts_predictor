# J-HARP-C64 implementation and experiment protocol — 2026-08-07

## 1. Purpose and status

This protocol governs the first accuracy-first use of the target-model
J-Lens for expert-route forecasting.  The metric used for model selection is
complete-request macro **Recall@8**, with the denominator fixed at the native
eight routed experts, averaged over direct horizons `t+1` through `t+4`.
The model continues to emit `t+1` through `t+8`; horizons 5--8 are auxiliary
and are not used to select the first endpoint.

The immediate target is 0.90 mean H1--H4 Recall@8.  This is a stretch target,
not an assumption about the result.  A 0.85 endpoint is the first practical
promotion threshold.  The sealed 409-request test partition remains closed
until preprocessing, architecture, loss, and checkpoint selection are frozen.

Implemented entry points:

- `python -m harp8.prepare_jspace_features`
- `python -m harp8.router_geometry`
- `python -m harp8.train_jspace_reranker`
- `runpod/run_j_harp_c64.sh`

The trainer has deliberately no test-pool argument.  It accepts only a pool
whose immutable manifest declares `train` and one declaring `validation`.

## 2. Evidence and hypothesis

### 2.1 Sparse J-space is a negative prior

The previous literal sparse J-space study was useful but negative as a main
representation.  It used non-negative pursuit over token/vocabulary
directions, not dense transported states.  On the 256-request matched probe:

| Input | H1 Recall@8 | H2 Recall@8 |
|---|---:|---:|
| route history | 0.5079 | 0.4630 |
| route + sparse J | 0.5450 | 0.4927 |
| route + raw residual | 0.5696 | 0.5110 |
| route + residual PCA | **0.6155** | **0.5497** |

Residual PCA beat sparse J in all 720/720 matched cells at both horizons.
In the longer source-layer-38 probe, adding sparse J to native MTP did not
beat MTP plus residual PCA.  These results are documented in
`J_ROUTE_0_REPORT_20260805.md`.  Sparse pursuit is therefore only a later,
bounded diagnostic and is not part of the production launcher.

### 2.2 Dense J-Lens is a distinct, untested hypothesis

For target post-block residual `xplus[t,l]`, the dense feature is

```text
j[t,l] = xplus[t,l] @ J[l].T,  l = 0..38
j[t,39] = xplus[t,39]          (identity final-layer endpoint)
```

The multiplication is accumulated in FP32.  Each resulting vector is stored
as unit-RMS features plus its separate natural-log RMS magnitude.  A single
train-only PCA basis is shared across all layers.

Dense J transport is a frozen linear change of coordinates and cannot create
information absent from the residual.  Its possible benefit is better
conditioning and a shared final-layer coordinate system for cross-layer
attention.  A J-specific claim therefore requires it to beat both a matched
raw-residual control and deterministic random-orthogonal controls.  Raw
residual and random controls use identical PCA ranks, train requests, model
capacity, optimizer, seed, and candidate pool.

The target J-Lens applies only at the target post-block boundary.  It must not
be applied to MTP hidden states, MTP fused states, router inputs, or
post-attention residuals.  Native MTP tensors use a separate encoder.

### 2.3 Current forecasting headroom

The best observed HARP generator checkpoint has validation request-macro:

| Horizon | Recall@8 |
|---:|---:|
| H1 | 0.821683 |
| H2 | 0.802488 |
| H3 | 0.786069 |
| H4 | 0.762165 |
| Mean H1--H4 | **0.793101** |

The corresponding candidate-64 pool has mean H1--H4 oracle coverage
**0.9740958**, with H4 coverage **0.9562749**.  Reaching 0.90 with this fixed
pool means recovering approximately `0.90 / 0.9740958 = 92.4%` of the
positives available to the reranker.  The present generator recovers about
81.4%.  Candidate ranking, rather than candidate coverage, is consequently
the first bottleneck.

## 3. Immutable inputs

All paths below are on the persistent network volume.  The production script
copies them to local NVMe before preprocessing or training, then verifies the
same hashes again locally.

### 3.1 Target residuals and aligned route capture

| Artifact | Network path | SHA-256 |
|---|---|---|
| all-layer post-block residuals, `[69632,40,2048]` FP16 | `/workspace/LLM_prefetch_study/artifacts/j_route_0/captures/all_post_layer_residuals_n2048_20260805a/post_layer_residuals.npy` | `5fccdbcba2192202909edcca6e7226406ffbfbf2e49d1155a1dad614d21440a6` |
| requests, 2,048 complete requests | `/workspace/LLM_prefetch_study/artifacts/j_route_0/captures/bf16_large_compact_n2048_20260805a/requests.jsonl` | `c24249a072426bc20fcc50f63d36a2e26dab156b08345af2cf8f3105838c66e4` |
| raw router logits, `[69632,40,256]` FP32 | same compact-capture directory, `raw_router_logits.npy` | `5b2438523e1dacfb6f5a826a8261e7ef4b19e8c1ff8bf3204dd8ef9ce519f655` |
| native top-8 IDs, `[69632,40,8]` uint16 | same compact-capture directory, `top8_expert_ids.npy` | `3a0cfd451eb4693d021a6f95a213c4d4ee3b1498cbfe3e99b21bda63290b1386` |

The residual manifest SHA-256 is
`e0d632de6cf40bb66803e53ba6fb3a480d0270bd3238151ab53030ee7e64b5f3`.
There are exactly 34 route rows per request.  Offline splits are 1,229 train,
410 validation, and 409 sealed test requests.

### 3.2 J-Lens and BF16 router geometry

| Artifact | Network path | Size | SHA-256 |
|---|---|---:|---|
| primary 1,000-prompt J-Lens | `/workspace/LLM_prefetch_study/artifacts/j_route_0/hf/qwen3.6-35B-A3B-jlens/lens.pt` | 327,166,510 B | `2fdf5128203b0ff8cfafa782baa2f3e180e1dbb6535843cdc7cccbe5de9953d1` |
| BF16 Transformers router artifact | `/workspace/LLM_prefetch_study/artifacts/gcrp2/captures/gcrp2r_transformers_bf16_corpus_v1/segments/gcrp2r_tf_bf16_seg_000000_000000/router_artifacts.safetensors` | 86,033,176 B | `39bb674489f47b7542e98c22f55ca3a292cc0f2dbf5ee79daf7d54a31ab5e10f` |

The router artifact contains 40 FP32 effective numeric matrices named
`target_router_weight.layer_00` through `.layer_39`, each `[256,2048]`.
The launcher exports full-rank `[40,256,256]` SVD keys and checks their
manifest.  It explicitly rejects the MXFP4 pinned router artifact and its
SHA-256 `998e0c1fa2f7d4aeaf15516e7c8dce54e85836acdab1c4172a1a88438773ea97`.

### 3.3 Native MTP depth chain

Directory:

```text
/workspace/LLM_prefetch_study/artifacts/j_route_0/captures/mtp_depths_bf16_large_n2048_20260805a
```

| Tensor | Shape/dtype | SHA-256 |
|---|---|---|
| `mtp_hidden_depths.npy` | `[69632,6,2048]`, FP16 | `98db81275f26dac6e7868504fe2b1ca1573334f85d5fc4f7391c24888fe00047` |
| `mtp_router_logits_depths.npy` | `[69632,6,256]`, FP32 | `d588fc81fd1e6590e6ccf43872cf4323a09399b29d3ba388a3ba8a6d167e2b6f` |

The six nodes are native autoregressive MTP depths.  Future committed target
tokens are labels only and were not used to construct these features.

**Causality boundary:** all six MTP nodes are informationally causal functions
of the committed prefix, but the capture audit found 0 of 56,872 structurally
valid MTP sources ready by either the recorded early router deadline or the
post-expert retention deadline.  This J-HARP experiment is consequently an
accuracy/representation upper-bound unless a future deployment schedule makes
the required nodes ready.  It must not be described as timing-causal or
deployment-ready.  Draft acceptance, prefix match, and rejection depth are
label-only audit metadata and are never training inputs.

The declared profile for this study is
`token_end_informational_upper_bound`: MTP nodes are aligned to the committed
prefix at token end and may be used to measure information content, while no
claim is made that the captured schedule delivered them before a layer-level
cache decision.  A later systems experiment must establish a different,
timestamp-validated profile before these gains can be credited as realizable
prefetch or retention gains.

### 3.4 Candidate-64 pools

Durable backup root, verified as approximately 35 GiB:

```text
/workspace/LLM_prefetch_study/artifacts/harp8_accuracy_expansion/remote_ephemeral_backup_20260807
```

Pool directories:

```text
pools/temp1_epoch5_c64_v3_train
pools/temp1_epoch5_c64_v3_validation
```

| File | SHA-256 |
|---|---|
| train `manifest.json` | `88559ef9e8333f839016f172aeb42bda8bc0300c8ab0c641c6f7a026fd602b59` |
| train `metadata.json` | `b4f3a18b84238320c8160f9524662b2b374f7873c8e764f7c3549872a5161254` |
| validation `manifest.json` | `5d1dc2c388caab84e26b8f7879955ae3de99bcb42a010be88875b9753b26ec77` |
| validation `metadata.json` | `65ba08ead2c0cdc26635cca183a3b8c9c5a244781c8bbc8cc4905e448558e990` |

The pool manifests contain hashes for every dense array.  The trainer checks
all declared arrays before admitting a run.

## 4. Feature/control matrix

The orchestration script prepares these immutable representations with one
common procedure:

| Condition | Transform before shared PCA | Purpose |
|---|---|---|
| `raw_residual` | identity | strongest non-J information control |
| `random_orthogonal_shared` | one signed permutation shared by layers | invertible coordinate-change control |
| `random_orthogonal_per_layer` | independent signed permutation per layer | cross-layer coordinate-misalignment control |
| `j_lens` | `xplus @ J[l].T`, layer 39 identity | dense J hypothesis |

PCA ranks are 128, 256, 512, and 1,024 in the full run.  PCA is fit once at
the maximum requested rank on at most 30,000 vectors, stratified by domain,
request, and target layer; smaller ranks are prefixes of the same basis.  The
fit split is exactly `train`.  Validation and test are rejected as fit splits.

Rank 512 is the preregistered matched-control comparison.  After J rank 512
passes the attribution gate, full-size J runs at ranks 128, 256, and 1,024
measure the information/capacity curve.  A full 2,048-dimensional projection,
raw+J dual-stream model, future-J auxiliary loss, route-tuned J LoRA, and c96
union pool are follow-up extensions; the current single-stream trainer must
not be presented as having implemented them.

## 5. Leakage and correctness gates

The run fails closed on any of the following:

1. A pinned input hash differs before or after the local copy.
2. Either pool path or manifest declares a test split.
3. Train and validation request IDs overlap.
4. PCA fit split is not `train` or includes validation/test requests.
5. Feature rows do not align by immutable request ID and within-request
   position.
6. Route, MTP, target-feature, or candidate geometry differs.
7. Candidate-64 validation coverage is below 0.97 mean H1--H4 or 0.95 at H4.
8. Router keys are not `[40,256,256]`, are non-finite, or disagree with their
   immutable manifest.
9. Epoch-zero reranker scores differ by even one FP32 value from the frozen
   HARP candidate scores, or any zero-initialized correction is nonzero.
10. NaN/Inf, an incomplete checkpoint, or a CUDA OOM occurs in the pilot.

The available full residual capture stores runtime BF16 block outputs in an
FP16 array.  It has no same-request higher-precision full-corpus counterpart;
the older FP32 256-request capture has zero request-ID overlap with the 2,048
request corpus and must not be falsely presented as a paired audit of the
large capture.  Instead, the launcher performs a representative storage
sensitivity audit: it deterministically round-trips the FP32 n256 residuals
through FP16 and compares those downcast values with the original values on
the exact same n256 rows.  This isolates FP16 storage-format sensitivity; it
does not prove equality between two separate executions.  An optional exact,
same-request full-corpus reference may also be supplied.  Every representative
or exact audit must report the transport gates:

- mean transported-state cosine at least 0.9995;
- minimum transported-state cosine at least 0.995;

When a matched downstream probe is run, its Recall@8 difference must be no
greater than 0.001.  A null probe delta is allowed for the representative
storage-format audit, but must be disclosed and leaves downstream metric
invariance provisional.  A controls or full J-attribution run is forbidden
unless at least the representative transport audit or an exact same-request
audit passes.

## 6. Model contract

`JSpaceCandidateReranker` consumes:

```text
candidate_scores          [B,8,40,64]
candidate_ids             [B,8,40,64]
target/J history          [B,3,40,R+1]
native MTP states         [B,6,2048]
native MTP router logits  [B,6,256]
full-rank router keys     [40,256,256] (frozen)
candidate scalars         [B,8,40,64,13]
```

The target-history channel contains current, t-1, and t-2 states with explicit
availability masks.  The extra scalar is log RMS.  Candidate scalars contain
current score/rank, copy gate, three generator source gates, lagged router
scores and memberships, and normalized request position.  History masks are
separate model inputs; they are not part of the thirteen candidate scalars.

Production model defaults:

```text
model width             384
attention heads           8
feed-forward width      1536
temporal blocks            2
cross-layer axial blocks   4
MTP blocks                 2
induced-set blocks          2
inducing points            16
expert residual width     128
dropout                   0.05
```

The 64 candidates are an unordered set.  Candidate processing is
permutation-equivariant.  The frozen router-row SVD key supplies candidate
geometry; a trainable layer-expert residual embedding permits information not
captured by that geometry.  The final output is a zero-initialized correction
to frozen HARP scores.

## 7. Loss and training contract

The fixed ranking loss is:

```text
1.00 * hard-boundary pairwise loss (24 hardest negatives)
0.50 * balanced candidate-membership BCE
0.50 * multi-positive listwise loss
0.25 * candidate-restricted teacher-router KL, temperature 2
```

Horizon weights for H1--H8 are:

```text
[1.0, 1.0, 1.25, 1.5, 0.25, 0.25, 0.25, 0.25]
```

Production optimizer settings:

```text
AdamW                    fused when CUDA supports it
learning rate            2e-4
minimum learning rate    2e-5
betas                    (0.9, 0.95)
epsilon                  1e-8
weight decay             0.01
warm-up                  3%
decay                    cosine
gradient clip            1.0
microbatch               1 complete token record
gradient accumulation    32
maximum epochs           30
minimum epochs           10
early-stop patience      5
seed                     42
```

CUDA execution uses BF16 autocast, FP32 loss reductions, TF32 matrix
multiplication, and FP32 optimizer states.  Norm gains, biases, and embeddings
receive no weight decay.  Checkpoints are selected lexicographically by:

1. validation request-macro mean H1--H4 Recall@8;
2. minimum individual H1--H4 Recall@8.

The exact environment used by the current 3090 pilot is Python 3.12,
PyTorch `2.8.0+cu128`, NumPy `2.1.2`, and safetensors `0.6.2`.

## 8. Execution order and failure thresholds

1. Copy and hash all pinned network inputs to local NVMe.
2. Run the focused J-HARP test suite.
3. Export full-rank BF16 router keys.
4. Prepare raw, J, shared-random, and per-layer-random feature stores.
5. Run a one-epoch, reduced-capacity, 32-row/16-row real-data pilot.
6. Train rank-512 controls in this order:
   `raw_residual`, `random_orthogonal_shared`,
   `random_orthogonal_per_layer`, `j_lens`.
7. Compute a paired complete-request bootstrap for J versus raw.
8. Continue the J rank sweep only when:
   - paired 95% CI lower bound is greater than zero; and
   - J gain over raw is at least 0.003 absolute.
9. If that gate passes, train J ranks 128, 256, and 1,024.
10. Freeze the best representation/rank before any sealed-test command is
    separately implemented and reviewed.

Milestones are reported rather than silently tuned around:

| Gate | Requirement |
|---|---:|
| candidate pool mean H1--H4 coverage | >= 0.97 |
| candidate pool H4 coverage | >= 0.95 |
| J scientific attribution | paired CI lower bound > 0 against raw |
| J practical attribution | J minus raw >= 0.003 |
| first endpoint promotion | mean H1--H4 >= 0.85 |
| stretch endpoint | mean H1--H4 >= 0.90 |

If training conditional recovery is below 0.95, ranking capacity/objective is
the diagnosed bottleneck.  If training recovery is high but validation is
below 0.85, collect more complete requests rather than repeatedly widening
the model on the same validation split.

## 9. Evaluation artifacts

Every completed condition emits:

```text
best.pt
last.pt
training_history.csv
training_history.json
validation_metrics.json
validation_horizon_metrics.csv
validation_request_metrics.csv
validation_layer_metrics.csv
validation_domain_metrics.csv
validation_position_metrics.csv
validation_h1_h4_paired_bootstrap.json
manifest.json
```

Metrics include request-macro and micro Recall@8, exact-set@8, candidate
coverage, conditional recovery, and per-horizon/layer/domain/position rows.
The bootstrap unit is a complete request with 2,000 paired replicates.

The launcher writes locally under:

```text
/root/j_harp_c64_runs/$RUN_ID/
```

and continuously mirrors immutable inputs' provenance, logs, epoch
checkpoints, and reports to:

```text
/workspace/LLM_prefetch_study/artifacts/harp8/j_harp_c64_runs/$RUN_ID/
```

It never deletes or reuses either directory.  `.partial` checkpoint files are
not mirrored.  Local input copies are never mirrored back to their source
network volume.  Growing feature-store files are excluded from periodic
mirrors; after the complete preparation command and manifest succeed, the
feature root is copied under an `.incomplete` staging name and atomically
renamed to `outputs/features`.  A failed run remains preserved with a nonzero
exit-status marker and must be resumed manually or superseded by a new run ID.

## 10. Commands

Syntax check and focused tests:

```bash
bash -n runpod/run_j_harp_c64.sh
python -m pytest -q \
  tests/test_harp8.py \
  tests/test_harp8_reranker.py \
  tests/test_harp8_jspace_features.py \
  tests/test_harp8_jspace_data.py \
  tests/test_harp8_jspace_reranker.py \
  tests/test_harp8_jspace_metrics.py \
  tests/test_harp8_router_geometry.py \
  tests/test_harp8_jspace_training.py \
  tests/test_harp8_prepare_jspace_features.py
```

Default safe pilot:

```bash
RUN_ID=j_harp_c64_20260807a \
MODE=pilot \
bash runpod/run_j_harp_c64.sh
```

Full matched controls and gated J rank sweep:

```bash
RUN_ID=j_harp_c64_20260807b \
MODE=all \
bash runpod/run_j_harp_c64.sh
```

On a shared GPU the script refuses to start unless no compute process is
present.  `ALLOW_BUSY_GPU=1` exists for deliberate scheduling only; it is not
recommended.  `DRY_RUN=1` prints resolved paths and exits before creating
anything.  See the script header for all environment overrides.

## 11. Audited implementation milestones and preliminary c64 log

The following milestones were completed on 2026-08-07 before a production J
comparison was launched:

- The focused feature/data/model/metric/geometry/training/CLI suite passed
  **30/30** tests.  The launcher additionally runs the two core HARP regression
  files, making the fail-closed preflight suite **46 tests** in total.
- A real two-request CUDA feature pilot succeeded at
  `/root/j_harp_c64/pilot_feature_v1` (log `prepare.log`), producing aligned
  J-Lens and raw-residual rank-16 stores.  Its source residual SHA-256 was
  `fe7805a92d1a06b652a045879d1833d37d73ce4d3b91262865cbf8cea0c143ad`.
- Full-rank BF16 router keys were exported at
  `/root/j_harp_c64/router_keys_r256/router_svd_keys.npy`; SHA-256
  `324c92258dba0fc0ba8299b2bd2de85400f492c67e7d0c95cf70714964198bd9`.
  The maximum and mean relative Gram errors were respectively
  `2.0822340047743637e-06` and `6.610074414936662e-07`.
- The representative FP32-to-FP16 storage audit is at
  `/root/j_harp_c64/precision_audit_n256/audit_fp16_roundtrip_v1/precision_audit.json`.
  It compared 256 identical rows at seven layers (1,792 transported vectors),
  was accepted, and measured mean cosine 1.0, minimum cosine
  0.9999999999999998, RMSE 0.0, and maximum absolute error 0.0.  Candidate and
  reference SHAs were respectively
  `2979d8b39a4add78c5dbc31abfd8ef95fe7c3cf31e201baa8c74f0b00956c897`
  and `84bff466f96f8814f84b365d9fb497ab723a20f5fe701bdeedfdd29299dc9cec`.
  Exactness occurs because the representative BF16 source values are exactly
  representable in FP16 over their observed range.  Probe Recall@8 was not
  measured (`probe_recall_delta = null`), and this remains a storage-format
  audit rather than a paired n2048 execution audit.

The c64 entries below use the earlier `harp8.reranker`, not J-HARP.  They are
operational and baseline evidence rather than a dense-J result.

- `reranker_temp1_epoch5_c64_v4_baseline`: output
  `/root/harp8_accuracy_expansion/reranker_temp1_epoch5_c64_v4_baseline`, log
  at the same stem with `.log`.  PID 141707 used batch 32, LR `3e-4`, 20
  epochs, seed 42.  It exited before epoch 1 with CUDA OOM in a Transformer
  feed-forward allocation request of 2.50 GiB while the process used about
  21.10 GiB and only 2.45 GiB was free.
- `reranker_temp1_epoch5_c64_v5_baseline_b8`: output
  `/root/harp8_accuracy_expansion/reranker_temp1_epoch5_c64_v5_baseline_b8`,
  log at the same stem with `.log`.  PID 141936 used batch 8, LR `1e-4`, 20
  epochs, seed 42.  It ran healthily at approximately 98% GPU utilization and
  8.56 GiB VRAM.  Epochs 1 and 2 reached validation request-macro mean H1--H4
  Recall@8 **0.7937605125233492** and **0.7939002648566016**, respectively;
  training loss fell from `0.3249714015150917` to `0.32280298905847576`.
  The process was deliberately stopped after the epoch-2 `best.pt` was saved,
  because the second-epoch gain was only 0.00014 and the GPU was needed for
  the new matched controls.  This is an explicitly partial, non-converged
  control, preserved under the durable `controls/legacy_c64_v5_partial_epoch2`
  artifact with `sealed_test_accessed=false`.

The v4 OOM and v5 memory profile justify the J-HARP production default of
microbatch 1 with accumulation 32.  The partial v5 score is a useful frozen
baseline checkpoint, but it does not constitute a converged or J-space
comparison.
