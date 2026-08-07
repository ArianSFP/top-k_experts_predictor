# J-HARP-C64 J-Lens/J-space experiment log — 2026-08-07

## 1. Snapshot and scientific status

This is the evidence log for the dense J-Lens/J-space HARP experiment. It is
separate from the implementation protocol in
`J_HARP_C64_IMPLEMENTATION_PROTOCOL_20260807.md`: the protocol says what must
be done, while this file records what has actually completed.

Snapshot time: **2026-08-07 14:05 UTC**.

At this snapshot:

- dense J, raw-residual, shared-random, and per-layer-random feature stores at
  PCA ranks 128, 256, and 512 are complete;
- the strict real-data functional pilots and full-capacity memory pilots have
  completed without NaN, Inf, CUDA OOM, or lineage failure;
- the full matched **raw-residual rank-512** validation run was stopped safely
  after epoch 2 with a complete resumable state;
- raw rank-512 epoch 2 improved mean H1--H4 Recall@8 by 0.0015472 over its
  frozen-HARP baseline, but remains a partial, non-converged control;
- the corrected ordered-group-prefetch pipeline passed matched remote
  equivalence and timing pilots;
- the full matched **J-Lens rank-512** run was also stopped safely after epoch
  2 with a complete resumable state;
- independent strict epoch-2 evaluators reproduced both saved training-history
  results over all 13,530 usable validation rows and all 410 requests;
- J-Lens reached 0.7946063 mean H1--H4 Recall@8 versus 0.7947420 for raw;
- the 2,000-resample request-paired J-minus-raw difference is
  **-0.0001357290**, 95% CI **[-0.0001682591, -0.0001037424]**;
- this first dense-J candidate-reranker result therefore fails the
  preregistered J-benefit gate and is a small but statistically resolved
  disadvantage at the matched epoch;
- the sealed 409-request test split has not been accessed.

Nothing in this log should be read as a claim that J-space improves route
forecasting. J transport makes the shared-PCA spectrum substantially more
concentrated, but that concentration did not improve Recall@8 in this matched
rank-512, candidate-only v1 comparison. This does not establish that J-space
is universally harmful; it establishes that this specific encoder/reranker did
not convert the representation into a predictive advantage.

## 2. Objective and frozen evaluation contract

The model predicts the native top-eight routed experts at horizons `t+1`
through `t+8`. The first endpoint reranks horizons H1--H4 and returns the
frozen HARP candidate scores unchanged at H5--H8.

The primary selection metric is complete-request macro Recall@8, with the
denominator fixed at the native `K=8`, averaged over H1--H4. The checkpoint
tie-breaker is the minimum individual H1--H4 request-macro Recall@8. The first
promotion threshold is 0.85 mean H1--H4 and the stretch target is 0.90.

The frozen generator endpoint has:

| Horizon | Validation request-macro Recall@8 |
|---:|---:|
| H1 | 0.821683 |
| H2 | 0.802488 |
| H3 | 0.786069 |
| H4 | 0.762165 |
| **Mean H1--H4** | **0.793101** |

The fixed candidate-64 validation pool has:

| Horizon | Request-macro oracle coverage |
|---:|---:|
| H1 | 0.9892671378 |
| H2 | 0.9811933117 |
| H3 | 0.9696479150 |
| H4 | 0.9562748984 |
| **Mean H1--H4** | **0.9740958157** |

Thus 0.90 Recall@8 requires recovering about 92.4% of positives already
present in the fixed pool. This experiment tests candidate ranking, not a new
candidate generator.

## 3. Hypothesis and required controls

For target post-block residual `xplus[t,l]`, dense J transport is

```text
j[t,l] = xplus[t,l] @ J[l].T,  l = 0..38
j[t,39] = xplus[t,39]          (identity endpoint)
```

The transform is accumulated in FP32 with TF32 disabled. Each transported
vector is decomposed into a unit-RMS direction and a separate natural-log RMS
magnitude. One train-only PCA coordinate system is shared across all layers.

Dense J transport is a frozen linear coordinate change; it cannot manufacture
information that was absent from `xplus`. The hypothesis is narrower: the
learned transport may align all source layers with a common final-layer
coordinate system and improve the conditioning of cross-layer forecasting.
This requires three matched controls:

1. `raw_residual`: no transport before shared PCA;
2. `random_orthogonal_shared`: one deterministic signed permutation shared by
   every layer;
3. `random_orthogonal_per_layer`: a different deterministic signed
   permutation at each layer.

All four conditions use identical request splits, PCA sample pairs and rank,
candidate pools, router keys, model capacity, optimizer, seed, losses, and
validation code. A J-specific claim requires J to beat raw residual and to be
interpretable against both random controls.

The previous literal sparse J-space study is not evidence for this dense
representation. It was a negative prior: on the matched 256-request probe,
route plus sparse J reached H1/H2 0.5450/0.4927, while route plus residual PCA
reached 0.6155/0.5497. Dense transport is deliberately being tested as a
different hypothesis.

## 4. Data, split, and causality profile

The aligned corpus contains 2,048 complete requests and 69,632 token rows,
exactly 34 rows per request. Offline request-level splits are:

| Split | Requests | Rows |
|---|---:|---:|
| Train | 1,229 | 41,786 |
| Validation | 410 | 13,940 |
| Sealed test | 409 | 13,906 |

PCA fitting uses only the 1,229 training requests. Model selection uses only
validation. The scientific trainer does not expose a test-pool argument.

The six MTP nodes are native autoregressive depth-chain states derived from a
committed prefix, so they are informationally causal. They are **not**
timing-causal under the recorded capture schedule: the audit found 56,872
structurally valid MTP sources and zero ready by either the early-router or
post-expert-retention decision deadline. The declared profile is therefore:

```text
decision_profile = token_end_informational_upper_bound
mtp_timing_causal = false
acceptance_labels_used = false
```

This permits a representation/accuracy upper-bound but not a claim of
deployable prefetch lead time. Draft acceptance, prefix match, and rejection
depth remain label-only audit metadata and are not model inputs.

## 5. Implemented model

`JSpaceCandidateReranker` consumes:

```text
candidate scores/IDs       [B,8,40,64]
target/J route history     [B,3,40,R+1]
native MTP hidden states   [B,6,2048]
native MTP router logits   [B,6,256]
frozen BF16 router keys    [40,256,256]
candidate scalars          [B,8,40,64,13]
```

The three target lags are current, `t-1`, and `t-2`, with explicit history
masks. The thirteenth target channel is log RMS. The 13 candidate scalars are
current score and rank, copy gate, three generator source gates, lagged router
scores and memberships, and normalized request position.

The MTP state and MTP router streams have their own encoder; the target J-Lens
is never applied to MTP tensors. Candidate expert geometry comes from frozen
full-rank router SVD keys plus learned layer-expert residual embeddings.
Target-history temporal mixing, cross-layer axial mixing, MTP mixing, and
permutation-equivariant induced-set attention produce a candidate correction.
That correction is zero-initialized, so epoch-zero H1--H4 scores are exactly
the frozen HARP scores.

The production-size rank-512 configuration is:

| Component | Value |
|---|---:|
| Model width | 384 |
| Attention heads | 8 |
| Feed-forward width | 1,536 |
| Temporal blocks | 2 |
| Cross-layer axial blocks | 4 |
| MTP blocks | 2 |
| Induced-set blocks | 2 |
| Inducing points | 16 |
| Expert residual width | 128 |
| Trainable parameters | 33,011,624 |

The loss is fixed before the matched comparison:

```text
1.00 * hard-boundary pairwise loss (24 hardest negatives)
0.50 * balanced candidate-membership BCE
0.50 * multi-positive listwise loss
0.25 * candidate-restricted forward KL (temperature 2)
```

Horizon weights for active H1--H4 are `[1.0, 1.0, 1.25, 1.5]`. H5--H8 are
an exact frozen-score passthrough in `CompositeJSpaceInference`; they are not
silently dropped from the public output contract.

## 6. Feature preparation results

All four representations were fit from the exact same 30,000 stratified
training `(row, layer)` pairs, seed 42, six randomized-PCA iterations. The fit
pair hash is:

```text
847d333bb801f24c39a618f28b6ab690f7f4392925ea96ce36b14ece64bbf6bc
```

The root feature manifest is:

```text
local NVMe: /root/j_harp_c64/features_controls_r128_256_512_v1/manifest.json
durable:    /workspace/LLM_prefetch_study/artifacts/harp8/j_harp_c64_runs/
            preflight_20260807/features_controls_r128_256_512_v1/manifest.json
SHA-256:    47bf689c100f2fc59d5e1bb637232ccce66525d21bc4c385f3778374b86e2116
```

Captured train variance fractions are:

| Representation | Rank 128 | Rank 256 | Rank 512 |
|---|---:|---:|---:|
| J-Lens | **0.78864475** | **0.86955854** | **0.93227618** |
| Raw residual | 0.61563491 | 0.71304188 | 0.81066632 |
| Shared random orthogonal | 0.61563504 | 0.71304190 | 0.81061792 |
| Per-layer random orthogonal | 0.55443748 | 0.64446205 | 0.75431559 |

The raw and shared-random spectra are essentially the same, as expected for a
shared invertible signed permutation. Per-layer random rotations make one
shared PCA basis less effective. J-Lens is much more compressible in the
shared coordinate system. This is strong evidence that the transform aligns
the layer distributions, but **not yet evidence that it preserves more
route-predictive information**.

Feature-store SHA-256 values (`features_normalized.npy`) are:

| Representation | Rank 128 | Rank 256 | Rank 512 |
|---|---|---|---|
| J-Lens | `26e06c5407e6c354fd97de97aa3700f4da93defcb8b2775d57d2c223f47bb2ca` | `49acac681e9802b96756229f94b9fb3945c3e0b10bc4a78fcbad94c460040621` | `79d28d2b4deb3a47220a554111c6742dbb394b2d3aefa00cabb70c6666e2e4a2` |
| Raw residual | `eaa535995a383c3094caa8438ea1565b22da2bf5bbc27ccd363750a5f0fe2df0` | `3edcf1d656d4ad0002d9b600023357bbaab2fa7c2492059c190a56dcf9d49da3` | `dbe6bb47060089b846e6d6393d2e1ad084800fb5a822331da0885d809e214d47` |
| Shared random | `f9d1b1f2ba21ecb4a9a810a1d2d08859f02be931b2b4e5e2f23a57537227302e` | `6060257c3b9afdf2db5e82437be2657a14cf7a9bf2de77d6a9685300e73b43b1` | `442a5fc5be8528d9aefb846dfaca7f04492d559f1bfe6ed7f5d7a6794446c3c7` |
| Per-layer random | `02e4db1ef0d796767c07836c67fc3c312a4f6d9a483fd2c933c5aac8d1923fb9` | `97a408c84835c05b6fbac5b1efc9476f668f144059e2fc139cd84617b922be07` | `0cedd44ef3a8d3f3d99e10bc013b74720adb53b88d7e5698dd11e0a53b0c3e8f` |

Rank-512 store manifests are:

| Representation | Manifest SHA-256 |
|---|---|
| J-Lens | `27c409bf49e753cfdefa725efdc271da301c5db0c2a9e92b491591ce7afbe620` |
| Raw residual | `f17c1a70a39547a85c6b61e4d49ebec3162cd2531a67d89acd64f66b7e0341c8` |
| Shared random | `536a34885ac8a1ebc645dfb7df3e90c87ac61f9fdc9aadeb2188555b7cb19ca7` |
| Per-layer random | `05412cfda89ccb6c395d14b3a83784f5128d04ab41f2ad436977cb2b956ef8fb` |

Rank 1,024 and full-rank 2,048 feature stores have not been produced. They are
follow-up work and must not be described as completed.

## 7. Precision audit and limitation

The available full 2,048-request capture contains runtime BF16 residuals
stored in an FP16 array. There is no exact same-request higher-precision
full-corpus reference. The completed audit is therefore a representative
storage-format sensitivity check, not a paired execution audit.

Artifact:

```text
/root/j_harp_c64/precision_audit_n256/audit_fp16_roundtrip_v1/
    precision_audit.json
SHA-256: c0834f39cb4eb73b3852101b494dc709b9e65a48155d97b4827452380525ab13
```

It round-tripped 256 identical FP32 rows through FP16 at seven layers: 1,792
transported vectors. Results were:

| Metric | Result |
|---|---:|
| Accepted | true |
| Mean cosine | 1.0 |
| Minimum cosine | 0.9999999999999998 |
| RMSE | 0.0 |
| Maximum absolute error | 0.0 |
| Downstream Recall@8 probe delta | not measured (`null`) |

The candidate/reference source SHAs were respectively
`2979d8b39a4add78c5dbc31abfd8ef95fe7c3cf31e201baa8c74f0b00956c897`
and
`84bff466f96f8814f84b365d9fb497ab723a20f5fe701bdeedfdd29299dc9cec`.
Exactness occurred because the representative BF16 values were exactly
representable in FP16 over their observed range.

The large-corpus J attribution therefore remains explicitly **provisional**:
the transport sensitivity gate passed, but downstream metric invariance has
not been measured on matched requests. The scientific J command must record
this audit and use the explicit provisional override; it must not imply that
an exact same-request n2048 audit was performed.

## 8. Router geometry and immutable artifact ledger

Full-rank SVD router keys were exported from the BF16 Transformers router
artifact. The keys are `[40,256,256]` FP32 and frozen during reranker training.

```text
router artifact:
  /root/j_harp_c64/inputs/router/router_artifacts.safetensors
  SHA-256 39bb674489f47b7542e98c22f55ca3a292cc0f2dbf5ee79daf7d54a31ab5e10f

router keys:
  /root/j_harp_c64/router_keys_r256/router_svd_keys.npy
  SHA-256 324c92258dba0fc0ba8299b2bd2de85400f492c67e7d0c95cf70714964198bd9

key manifest:
  /root/j_harp_c64/router_keys_r256/manifest.json
  SHA-256 55a29b61a681ad5063f4d6d413f542c6800e0674af1c6bd8764571e334235a18
```

All retained singular-value energy is 1.0. Relative Gram reconstruction error
was at most `2.0822340e-6`, mean `6.6100744e-7`.

Other key immutable inputs are:

| Artifact | Path | SHA-256 |
|---|---|---|
| Post-layer residuals `[69632,40,2048]` FP16 | `/root/j_harp_c64/inputs/residual/post_layer_residuals.npy` | `5fccdbcba2192202909edcca6e7226406ffbfbf2e49d1155a1dad614d21440a6` |
| J-Lens | `/root/j_harp_c64/inputs/lens/lens.pt` | `2fdf5128203b0ff8cfafa782baa2f3e180e1dbb6535843cdc7cccbe5de9953d1` |
| Requests | `/root/j_harp_c64/inputs/compact/requests.jsonl` | `c24249a072426bc20fcc50f63d36a2e26dab156b08345af2cf8f3105838c66e4` |
| Raw router logits | `/root/j_harp_c64/inputs/compact/raw_router_logits.npy` | `5b2438523e1dacfb6f5a826a8261e7ef4b19e8c1ff8bf3204dd8ef9ce519f655` |
| Native top-8 IDs | `/root/j_harp_c64/inputs/compact/top8_expert_ids.npy` | `3a0cfd451eb4693d021a6f95a213c4d4ee3b1498cbfe3e99b21bda63290b1386` |
| Compact-capture manifest | `/root/j_harp_c64/inputs/compact/manifest.json` | `2500305735fabd9341f2b310aba3767e711abf41804712e759495ceeadfa9ce1` |
| Strict capture audit | `/root/j_harp_c64/inputs/compact/audit.strict.json` | `1e786fb4270ff880221f29abf67bf562a825e5dac58126a28ccf6ae580c162b5` |
| MTP hidden depths | `/root/j_harp_c64/inputs/mtp/mtp_hidden_depths.npy` | `98db81275f26dac6e7868504fe2b1ca1573334f85d5fc4f7391c24888fe00047` |
| MTP router logits | `/root/j_harp_c64/inputs/mtp/mtp_router_logits_depths.npy` | `d588fc81fd1e6590e6ccf43872cf4323a09399b29d3ba388a3ba8a6d167e2b6f` |
| MTP manifest | `/root/j_harp_c64/inputs/mtp/manifest.json` | `a3af6775f75bb6a3d9105ac291366004eeb7854ae7eabc5372ee500766499653` |
| Train pool manifest | `/root/harp8_accuracy_expansion/pools/temp1_epoch5_c64_v3_train/manifest.json` | `88559ef9e8333f839016f172aeb42bda8bc0300c8ab0c641c6f7a026fd602b59` |
| Validation pool manifest | `/root/harp8_accuracy_expansion/pools/temp1_epoch5_c64_v3_validation/manifest.json` | `5d1dc2c388caab84e26b8f7879955ae3de99bcb42a010be88875b9753b26ec77` |

The pool manifests embed hashes for every dense array and the frozen generator
checkpoint SHA
`410ced95e6082f6a9bfa962d082926ee1ff5c29d203a3869f713a5ee7e58a09d`.

The preflight feature stores and strict source snapshot are durable at:

```text
/workspace/LLM_prefetch_study/artifacts/harp8/j_harp_c64_runs/
    preflight_20260807/
```

The raw control imported from `/root/j_harp_c64/source_strict_v1`, not the
mutable repository working tree. Its principal snapshot hashes are:

| File | SHA-256 |
|---|---|
| `harp8/train_jspace_reranker.py` | `6ddda29595ce5b2b90bcf1593666f8c768dd1baca59615938a5bad797bdad784` |
| `harp8/jspace_reranker.py` | `9bca5f546880dec6a35417586dfaf1f551356fe2cc44da91e4c75686d031d323` |
| `harp8/jspace_data.py` | `023803d534e49def25f8c1c256f70065f01e18449b427e5602b94e992406393f` |
| `harp8/jspace_features.py` | `0f9c90146a6db3f823200ab96afc39bd38ec7f7ea856ec6ab643f234404f5a19` |
| `harp8/jspace_metrics.py` | `17fe474c3f33b2d34bf92bca3c1c2d7a3816220fa134474701e6f05286ea45a6` |
| `runpod/run_j_harp_c64.sh` | `e23e6a67d02fa62b8f401958d23f37992e7d8e73c268bdfabacd9e43a21b98d1` |

## 9. Correctness and strict-pilot results

The focused J-HARP suite initially passed 30/30 tests. With core HARP
regressions, the strict launcher preflight passed 46 tests. The initial
ordered-group-prefetch path raised this to 49, and the expanded-view regression
raised the current complete local suite to **50 tests**.
The tests cover dense transport, PCA split discipline, feature alignment,
router geometry, set equivariance, zero-init identity, loss direction,
candidate masking, checkpoint composition, and
training CLI constraints.

A two-request CUDA feature pilot at
`/root/j_harp_c64/pilot_feature_v1` produced aligned J and raw rank-16 stores.
The source residual SHA was
`fe7805a92d1a06b652a045879d1833d37d73ce4d3b91262865cbf8cea0c143ad`.

The strict training pilots all completed a real forward/backward/optimizer
step sequence, validation, checkpoint round trip, and manifest write. All
recorded finite-value and candidate-coverage gates passed; all had exact
epoch-zero frozen-score equality and `sealed_test_accessed=false`.

| Pilot | Train/val rows | Params | Epoch-0 mean H1--H4 | Epoch-1 mean H1--H4 | Manifest SHA-256 |
|---|---:|---:|---:|---:|---|
| raw rank128, reduced | 32/16 | 1,421,736 | 0.8817871 | 0.8809570 | `c108c37e677f4fb7b35d595aafc90a5b0c3d846c90cdcac1a4180344e8c922d8` |
| J rank128, reduced | 32/16 | 1,421,736 | 0.8817871 | 0.8809570 | `98ad323adcaebc8246499f223bf00b8f3c294a99c9ab55cb49852210c66cb5de` |
| raw rank512, full capacity b1 | 8/4 | 33,011,624 | 0.8830078 | 0.8828125 | `70d4aba52fd428432c307c5b5b909ea5a5003d656f161ce821d2583c6fc42be6` |
| J rank512, full capacity b2 | 64/16 | 33,011,624 | 0.8817871 | 0.8800293 | `01e34bf584be608745e99d5d6aa268fb35cb789b94c2661038e2e72a65097bfa` |
| J rank512, full capacity b4 | 256/16 | 33,011,624 | 0.8817871 | 0.8801758 | `afbf5d6c7db9bacc2acf2d4ff3ed6ac49f6b35ae808c54c25441dc2e9ef1b20a` |
| J rank512, full capacity b16 | 1,024/64 | 33,011,624 | 0.7730361 | 0.7725687 | `e738e00c504280ed2c736689e20fce3eacebcfc6314f75a596985ecf71359256` |

These validation values are **not model-comparison results**. The pilots use
different, tiny prefixes of validation and only one epoch; the epoch-zero
means differ because the evaluated request subsets differ. Their purpose is
to demonstrate functional correctness and safe execution.

### 9.1 Wall-time and memory probes

| Configuration | Train/val rows | End-to-end wall time | Observed VRAM | Interpretation |
|---|---:|---:|---:|---|
| Full capacity, b1 | 8/4 | 43.504 s | not recorded | functional floor; dominated by startup/audits |
| Full capacity, b2 | 64/16 | 45.648 s | not recorded | functional pilot |
| Full capacity, b4 | 256/16 | 50.462 s | 5,032 MiB snapshot | observed value, not peak |
| Full capacity, b16 | 1,024/64 | 71.061 s | 18,422 MiB snapshot | safe-sizing evidence, not instrumented peak |

Wall times include strict lineage hashing, model construction, training,
checkpointing, validation, bootstrap setup, and artifact writing. They are not
pure training throughput and must not be divided into rows to claim model
rows/s. The b16 snapshot demonstrated adequate headroom on the 24 GiB RTX
3090 and justified using microbatch 16 for the matched run, but it is not a
formal peak-memory measurement.

For context, an older non-J c64 reranker at batch 32 OOMed after using about
21.10 GiB and requesting another 2.50 GiB with 2.45 GiB free. Its batch-8
restart reached 0.7937605 and 0.7939003 after epochs 1 and 2, then was
deliberately stopped and preserved as a partial, non-converged baseline. It is
not a dense-J result.

### 9.2 Opt-in ordered-group-prefetch input path

An input-side optimization has been implemented. Its first remote RTX 3090
benchmark was attempted only after the raw control was stopped safely and
failed before a scientific result. The corrected retry has now passed matched
equivalence and timing pilots. It is not used as evidence for model accuracy.

The optimized path gathers one complete gradient-accumulation group from the
memory-mapped stores in a vectorized operation, maintains one ordered
background CPU group read-ahead, retains only tensors consumed by the model,
stages them in pinned CPU memory, performs one nonblocking group transfer to
the GPU, and then takes the same ordered microbatch slices on device. It is
opt-in so that the original loader remains an explicit control.

For rank 512, candidate 64, microbatch 16, and accumulation 2, one compact
group is analytically **31.706 MiB**. The former full one-microbatch input was
about 27 MiB, so this adds only about 4--5 MiB of peak GPU input storage while
using approximately 64 MiB for double-buffered pinned host staging.

After adding the expanded-view contiguity regression, the complete local suite
passed **50 tests** and the focused remote suite passed **8 tests**. They prove:

- bit-identical gathered tensor values and identical row ordering;
- identical microbatch boundaries;
- identical dropout random-number progression;
- identical losses and gradients;
- identical resume behavior.

The first remote attempt had exposed an expanded zero-stride tensor reaching
`pin_memory()` and raising `RuntimeError`. It failed before a scientific
result, data write, or checkpoint update; no artifact was damaged. The fix
materializes the expanded view contiguously before pinned staging.

The corrected matched end-to-end pilot measured:

| Input path | Configuration | Wall time |
|---|---|---:|
| Legacy loader | J rank512, b16/acc2, train n1024 | 71.046 s |
| Ordered group prefetch | J rank512, b16/acc2, train n1024 | 69.287 s |
| **Difference** | identical scientific inputs | **-1.759 s (-2.475%)** |

Both pilots used the same 64-row validation prefix. Their
`training_history.csv` files are byte-identical with SHA-256
`613f83a474b884ddb8a3c6b68fd6515140171864a1aca451960aedefe7066f18`,
and their `validation_metrics.json` files are byte-identical with SHA-256
`fd720f7a789a152f81ab7cc77261c052128a46901c398597a33c48ce17b4175d`.
The corrected and legacy manifest SHAs are respectively
`67164c3d6af9ebaf4bb51f3bb17818fa3e83a1e8c95b614caca4328d32012a67`
and `59a1f7790dacb38b41381c290aa22738fd7322955d7b8bf48fba5d414618e21e`.
The corrected manifest records `harp8_ordered_group_prefetch_v1`, one CPU
read-ahead group, 32 rows per materialization, compact model-only transfer,
and pinned nonblocking CUDA transfer.

These are matched whole-pilot wall times, including validation and artifact
work, rather than pure steady-state training throughput. The approximately
2.475% reduction is real for this pilot; no full-epoch speedup is inferred.

## 10. Raw rank-512 control: safe stop after epoch 2

The first matched scientific condition was launched at approximately
**2026-08-07 12:06:47 UTC** in tmux session `jharp_raw512`:

```text
source snapshot:
  /root/j_harp_c64/source_strict_v1
output:
  /root/j_harp_c64/runs_matched_v1/raw_rank512_seed42
representation/rank:
  raw_residual / 512
training rows:
  41,786 (all 1,229 train requests)
validation rows:
  13,940 (all 410 validation requests)
epochs/minimum epochs/patience:
  5 / 5 / 5
microbatch/eval batch/gradient accumulation:
  16 / 16 / 2
optimizer:
  AdamW, LR 2e-4 to 2e-5 cosine, seed 42
bootstrap:
  2,000 complete-request paired replicates
decision profile:
  token_end_informational_upper_bound
```

The training child stopped safely after completing epoch 2. The tmux pane is
dead and no Python trainer remains active. The two completed validation passes
were:

| Epoch | Global step | Learning rate | Frozen base mean | Model mean H1--H4 | Gain vs base |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,268 | 0.0001866955463 | 0.7931947980 | 0.7944532695 | +0.0012584715 |
| 2 | 2,536 | 0.0001427461467 | 0.7931947980 | **0.7947420419** | **+0.0015472439** |

The epoch-1-to-epoch-2 gain was only **+0.0002887724**. Training loss continued
to fall, but validation Recall@8 was already nearly flat, so the control was
stopped at a complete checkpoint boundary. This is a partial, non-converged
control and remains far below the 0.85 promotion threshold.

Epoch-2 horizon results are:

| Metric | Epoch-2 result |
|---|---:|
| H1 request-macro Recall@8 | 0.8239694199 |
| H2 request-macro Recall@8 | 0.8041813549 |
| H3 request-macro Recall@8 | 0.7870940696 |
| H4 request-macro Recall@8 | 0.7637233233 |
| **Mean H1--H4** | **0.7947420419** |
| Minimum H1--H4 | 0.7637233233 |

The complete epoch-2 checkpoint state and history are mirrored to:

```text
/workspace/LLM_prefetch_study/artifacts/harp8/j_harp_c64_runs/
    matched_v1/raw_rank512_seed42/
```

`last.pt` records `completed_epoch=2` and `global_step=2536`.

| Artifact | Current SHA-256 |
|---|---|
| `best.pt` | `54bc4c364653c84beda2ba82579209a3dee56f185e9439d0daad7841ad77cd86` |
| `last.pt` | `bddb8ba7238ac4360eef9a0fc24e673a3b65a70aa5a012c84df3f61e0f30c3dd` |
| `training_history.csv` | `6440c35c7e29f48f1bb3a81f33fadee7da18bc8f4a73d7034f9b738500120db5` |

Local and network hashes agree. The manual stop leaves final aggregate reports
and the completed-condition manifest unfinished; no missing result is inferred.

## 11. Full matched J-Lens rank-512 run

The full J-Lens condition was launched at approximately
**2026-08-07 12:58:20 UTC**:

```text
tmux session:   jharp_j512
source:         /root/j_harp_c64/source_prefetch_v2
output:         /root/j_harp_c64/runs_matched_v1/j_rank512_seed42
representation: j_lens / rank 512
precision:      representative audit; provisional without downstream probe
input path:     harp8_ordered_group_prefetch_v1
```

All scientific arguments match the raw run except the preregistered
representation, required J precision-attribution flags, and validated input
pipeline. The run used microbatch 16, accumulation 2, five configured epochs,
seed 42, the same train/validation pools, and the same 2,000-replicate
complete-request bootstrap contract. It was stopped at the same completed
epoch-2 boundary used for the raw comparison.

The training source snapshot includes:

| File | SHA-256 |
|---|---|
| `harp8/train_jspace_reranker.py` | `dbe6e9dca78f829486adf19e8c6dabbd98eaee819c1a2452f9fc96788a8aaf56` |
| `harp8/jspace_data.py` | `c9ea2244a1735a7a810c901f8816fc5d4dfec57f5260e7cfebccc953d559033d` |

The completed validation passes were:

| Epoch | Global step | Frozen base mean | J mean H1--H4 | Gain vs base |
|---:|---:|---:|---:|---:|
| 1 | 1,268 | 0.7931947980 | 0.7943584503 | +0.0011636523 |
| 2 | 2,536 | 0.7931947980 | **0.7946063129** | **+0.0014115149** |

Epoch-2 horizon results are H1 0.8238867331, H2 0.8040655966,
H3 0.7869327793, and H4 0.7635401425. The resumable `last.pt` has SHA-256
`cb0496af80eef46d23bbefa7da074a43e00adf1fbc2a9b9c06ec4a82d38f13d0`,
records `completed_epoch=2` and `global_step=2536`, and is mirrored under
`matched_v1/j_rank512_seed42/`.

### 11.1 Independent strict epoch-2 evaluation

Both epoch-2 checkpoints were independently reloaded and evaluated over all
13,530 usable validation rows. Each evaluator verified strict lineage, used
410 complete requests, constructed no optimizer, invoked no training resume,
and recorded `split=validation` and `sealed_test_accessed=false`. Both results
exactly reproduce their saved epoch-2 training histories.

| Condition | Mean H1--H4 R@8 | Gain vs frozen base | Paired-vs-base 95% CI |
|---|---:|---:|---|
| Raw rank-512 | 0.7947420419 | +0.0015472439 | [0.0014046387, 0.0016963144] |
| J rank-512 | 0.7946063129 | +0.0014115149 | [0.0012738143, 0.0015540542] |

The separately written request CSVs contain the same 3,280
`(request_id, horizon)` keys. Averaging H1--H4 within each request and
resampling the 410 paired request differences 2,000 times with seed 42 gives:

```text
J minus raw point difference: -0.0001357290436
paired 95% CI:                [-0.0001682590697, -0.0001037424372]
```

The per-horizon J-minus-raw differences are:

| Horizon | J R@8 | Raw R@8 | J - raw | Paired 95% CI |
|---:|---:|---:|---:|---|
| H1 | 0.8238867331 | 0.8239694199 | -0.0000826868 | [-0.0001392797, -0.0000274796] |
| H2 | 0.8040655966 | 0.8041813549 | -0.0001157583 | [-0.0001757931, -0.0000562057] |
| H3 | 0.7869327793 | 0.7870940696 | -0.0001612903 | [-0.0002215469, -0.0001057240] |
| H4 | 0.7635401425 | 0.7637233233 | -0.0001831807 | [-0.0002375570, -0.0001254888] |

H5--H8 are exactly identical because v1 passes the frozen HARP scores through
at those horizons. The negative interval is statistically resolved but very
small in practical magnitude. It fails both parts of the preregistered J
benefit gate; it is evidence against a J benefit in this candidate-reranker,
not a general claim about every possible J-space forecaster.

All evaluation outputs are durable at:

```text
/workspace/LLM_prefetch_study/artifacts/harp8/j_harp_c64_runs/
    matched_v1/evaluations/v1_j_epoch2/
    matched_v1/evaluations/v1_raw_epoch2/
    matched_v1/evaluations/v1_j_minus_raw_epoch2/
```

Key hashes are:

| Artifact | SHA-256 |
|---|---|
| J strict-evaluation manifest | `6fbcfc2fb9cb51d0280cb2aecf3679bf40a18d68ada8120549ca969ba96ceecf` |
| Raw strict-evaluation manifest | `f287c4c5efb0ade8e9aa37ad31be18e38928cd798fb1b5a4b90df387adc621d7` |
| Paired-comparison manifest | `e29f473408b560e138d5f7de072646a3c56da79871c020813369cf508dd9b02c` |
| J request metrics CSV | `77ccc5e8f9f5f9ab15a806a83c29ad09f2b185d6af3e1e819d40f9fe7deea965` |
| Raw request metrics CSV | `afb6b9ddea641788772c58bd7017e7572bd7b6c599f3ed3c39b4db3948c226eb` |
| Paired JSON | `fe6f778790a3fdce84b8cadae6279ac909efe3ec1264af8438b248dfe127fd3f` |

The paired directory also contains `COMMANDS.txt`, the per-horizon CSV, a
compact README, and a self-verifying manifest. Local and network copies were
compared file-by-file by byte count and SHA-256.

## 12. Interpretation and next steps

The matched J-versus-raw epoch-2 comparison is complete. The next scientific
steps are:

1. Preserve both epoch-2 states as partial, resumable, matched controls; do not
   describe either as converged.
2. Treat the negative J-minus-raw interval as a failed v1 J-benefit gate.
3. Run shared-random and per-layer-random rank-512 controls only if attribution
   of the coordinate effect remains decision-relevant.
4. Do not promote rank 1,024 solely on the strength of J compressibility; no
   rank-1,024 feature store exists and rank 512 did not beat raw.
5. Prioritize the direct Full256 H1--H8 forecaster, which can discover experts
   outside c64 and directly supervises the complete router distribution.
6. Treat the Full256 CPU correctness gates as complete, then run its separately
   recorded CUDA functional/overfit/memory pilots before any production run.
7. Freeze preprocessing, architecture, loss, and checkpoint selection before a
   separately reviewed sealed-test command.

The completed comparison supports a narrow conclusion: shared-PCA spectral
concentration alone was not sufficient to improve this v1 candidate ranker.
The random controls can still separate learned-J geometry from generic
coordinate effects, but they cannot turn the observed J-minus-raw result into
a positive J result.

## 13. Results table template

Completed matched results and pending controls are:

| Condition | Rank | Params | Best epoch | H1 R@8 | H2 R@8 | H3 R@8 | H4 R@8 | Mean H1--H4 | Min H1--H4 | Pool coverage | Conditional recovery | Delta vs raw | Paired 95% CI | Precision status | Artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Raw residual (partial) | 512 | 33,011,624 | 2 | 0.8239694 | 0.8041814 | 0.7870941 | 0.7637233 | 0.7947420 | 0.7637233 | 0.9740958 | 0.8158767 | reference | n/a | n/a | `matched_v1/raw_rank512_seed42` |
| Shared random | 512 | 33,011,624 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 0.9740958 | TBD | TBD | TBD | n/a | TBD |
| Per-layer random | 512 | 33,011,624 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 0.9740958 | TBD | TBD | TBD | n/a | TBD |
| J-Lens (partial) | 512 | 33,011,624 | 2 | 0.8238867 | 0.8040656 | 0.7869328 | 0.7635401 | 0.7946063 | 0.7635401 | 0.9740958 | 0.8157373 | -0.0001357 | [-0.0001683, -0.0001037] | provisional until paired precision probe | `matched_v1/j_rank512_seed42` |

For every row, additionally publish micro Recall@8, exact-set@8, per-layer,
per-domain, per-position, and complete-request metrics. Keep H5--H8 in a
separate table because they are frozen HARP passthrough values in this first
reranker:

| Condition | H5 R@8 | H6 R@8 | H7 R@8 | H8 R@8 | Passthrough exact? |
|---|---:|---:|---:|---:|---|
| Any v1 J-HARP condition | frozen base | frozen base | frozen base | frozen base | yes |

## 14. Claims that are currently justified

It is justified to state that:

- dense J-space preprocessing, its matched controls, and a strict scientific
  trainer now exist;
- all representations use identical train-only PCA sampling and immutable
  alignment;
- J-space has markedly higher shared-PCA captured variance at matched rank;
- full-capacity rank-512 training is feasible on this RTX 3090 at microbatch
  16 in the observed pilot;
- candidate coverage is sufficient for a 0.90 H1--H4 endpoint in principle;
- MTP inputs are informationally causal but not timing-causal in this capture;
- test remains sealed;
- raw rank-512 epoch 2 produced a +0.0015472 partial validation gain over its
  frozen-score baseline, with a durable resumable checkpoint at step 2,536;
- J rank-512 epoch 2 produced a +0.0014115 partial validation gain over the
  same frozen baseline, with a durable resumable checkpoint at step 2,536;
- the strict request-paired comparison found J below raw by 0.0001357, with
  95% CI [-0.0001683, -0.0001037], so the preregistered J-benefit gate failed;
- the corrected ordered-group-prefetch loader passed 50 local and 8 remote
  tests, reproduced byte-identical training/evaluation artifacts, and reduced
  matched whole-pilot wall time by approximately 2.475%; and
- both strict evaluation bundles and their paired comparison are mirrored and
  hash-verified on durable network storage.

It is not yet justified to state that:

- J-space improves Recall@8 over raw residuals or random controls;
- the current full raw run has converged or established a final improvement;
- any MTP-enabled gain is realizable before a cache-decision deadline;
- FP16 storage is downstream-metric invariant on the full matched corpus;
- the 0.85 or 0.90 endpoint has been reached.
