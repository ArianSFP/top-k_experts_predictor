# Matched OOF J-Lens JSpaceV2 preflight and access audit

Date: 2026-08-07
Status: preflight passed; **not launched**
Backend: RTX 3090 RunPod, local-overlay inputs
Scientific scope: train/validation model development only

## Decision

The matched J-Lens rank-512 arm is ready to launch after the active raw-residual rank-512 arm exits and completes its final validation artifacts. The proposed arm uses the same OOF candidate pools, frozen candidate-generator checkpoint, router keys, model architecture, optimizer, seed, objective, batching, and validation-selection policy as the raw arm.

Only the target-state representation and the precision-attribution arguments differ. No model or trainer source was modified during this preflight, and no job was launched.

## Allowlisted inputs

Only the explicit level-2 train and validation pools were preflighted:

| Role | Path | Requests | Rows | Manifest SHA-256 |
|---|---|---:|---:|---|
| Meta-train | `/root/j_harp_c64/oof_proxy/fold0_meta_train_c64` | 246 | 8,118 | `a447f398655f6f04f87813c313a2f9214e797fc8f13cb9c251b68853901a79e0` |
| Outer validation | `/root/j_harp_c64/oof_proxy/outer_validation_c64` | 410 | 13,530 | `aa95ac9b835810190d7f77a12c45008e8d89568a4d24221070b5ad16e986aeab` |

The two request sets have zero overlap. The meta-train set also has zero overlap with the 983 requests used to fit the frozen base HARP checkpoint. Both pools name the same base checkpoint:

`4f17a6025f789c3df6105e2e8af45e5d3f8f4b8cbd25df776061ab7f47fe94b8`

Shared lineage manifests:

| Artifact | SHA-256 |
|---|---|
| Compact target capture manifest | `2500305735fabd9341f2b310aba3767e711abf41804712e759495ceeadfa9ce1` |
| MTP capture manifest | `a3af6775f75bb6a3d9105ac291366004eeb7854ae7eabc5372ee500766499653` |
| Full-rank router-key manifest | `55a29b61a681ad5063f4d6d413f542c6800e0674af1c6bd8764571e334235a18` |
| Shared feature-preparation manifest | `47bf689c100f2fc59d5e1bb637232ccce66525d21bc4c385f3778374b86e2116` |

## J-Lens feature verification

The actual files were hashed read-only and agree with their immutable manifests.

| Artifact | Shape / dtype | SHA-256 |
|---|---|---|
| J-Lens rank-512 features | `[69632, 40, 512]`, FP16 | `79d28d2b4deb3a47220a554111c6742dbb394b2d3aefa00cabb70c6666e2e4a2` |
| J-Lens log-RMS side channel | `[69632, 40]`, FP32 | `315819648cf637c137b1808ce65fb8aac4425c375913f2b52d4b2bb3c5862f49` |
| J-Lens shared rank-512 PCA file | artifact | `c9d8770c7c3e08b662b6f2fe2ec0dc8fad284fb47dc7a100d55bb7c3ac2439cf` |
| Frozen J-Lens checkpoint | 39 transported layers, width 2,048 | `2fdf5128203b0ff8cfafa782baa2f3e180e1dbb6535843cdc7cccbe5de9953d1` |
| Precision-audit JSON | audit record | `c0834f39cb4eb73b3852101b494dc709b9e65a48155d97b4827452380525ab13` |

The J and raw feature stores have identical row/layer/rank geometry, normalization contract, source-residual SHA, PCA sample identities, fit-vector count, and seed. Their expected representation-specific variance fractions differ:

- J-Lens rank 512: `0.9322761848`
- Raw-residual rank 512: `0.8106663157`

This is a representation difference, not a matching failure.

## Exact matched-arm differences

Relative to the active raw512 arm, change only:

1. `--target-features` to the J-Lens rank-512 feature array.
2. `--target-feature-rms` to the matching J-Lens log-RMS array.
3. Add the matching `--precision-audit` record.
4. Add `--allow-provisional-j-without-probe`.
5. Use distinct J-Lens output, log, and tmux-session names.

All numerical training arguments remain identical.

## Launch-ready command

Run only after the raw512 process has exited:

```bash
tmux new-session -d -s oof_jv2_j512_20260807a "env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/root/j_harp_c64/source_oof_candidate_v3_20260807a python -u -m harp8.train_jspace_v2_reranker --enable-experimental-v2 --train-pool /root/j_harp_c64/oof_proxy/fold0_meta_train_c64 --validation-pool /root/j_harp_c64/oof_proxy/outer_validation_c64 --capture-dir /root/j_harp_c64/inputs/compact --mtp-dir /root/j_harp_c64/inputs/mtp --target-features /root/j_harp_c64/features_controls_r128_256_512_v1/j_lens/rank512/features_normalized.npy --target-feature-rms /root/j_harp_c64/features_controls_r128_256_512_v1/j_lens/rank512/log_rms.npy --precision-audit /root/j_harp_c64/precision_audit_n256/audit_fp16_roundtrip_v1/precision_audit.json --allow-provisional-j-without-probe --router-keys /root/j_harp_c64/router_keys_r256 --output-dir /root/j_harp_c64/oof_proxy/jspace_v2_c64_j512_w64_rq16_seed42_20260807a --device cuda:0 --rows-per-request 34 --history 3 --epochs 12 --minimum-epochs 2 --patience 2 --microbatch-size 32 --evaluation-batch-size 32 --gradient-accumulation 2 --learning-rate 1e-4 --minimum-learning-rate 1e-5 --warmup-fraction 0.05 --weight-decay 0.05 --gradient-clip 1.0 --seed 42 --bootstrap-replicates 5000 --ordered-group-prefetch --model-width 64 --attention-heads 4 --feedforward-width 128 --expert-embedding-width 32 --router-query-rank 16 --temporal-blocks 1 --axial-blocks 1 --mtp-hidden-blocks 1 --mtp-router-blocks 1 --set-blocks 1 --inducing-points 8 --dropout 0.10 --loss-profile membership_ranking_v1 2>&1 | tee /root/j_harp_c64/logs/jspace_v2_c64_j512_w64_rq16_seed42_20260807a.log"
```

At preflight time, the proposed session, output directory, and log path did not exist.

## Required limitations in result interpretation

### Provisional precision attribution

The precision audit is finite and accepted, with its lens hash matching the feature store, but `probe_recall_delta` is null. The explicit provisional override is therefore required by the fail-closed trainer. The resulting run must be reported as **provisional precision attribution**, not as probe-validated BF16/FP16 equivalence.

### Transductive preprocessing proxy

The frozen PCA preprocessing was fitted on the original 1,229 offline-train requests. Those include the 246 level-2 meta-train requests, although the frozen base predictor itself excluded those 246 requests. This makes both matched raw and J arms a `transductive_preprocessing_oof_proxy`, not the final fold-specific preprocessing crossfit. It does not invalidate the matched raw-versus-J comparison, but it prevents presenting the result as the final leakage-minimized estimate.

## Runtime and storage estimate

The matched raw arm used approximately 4.74 GiB of GPU memory. Its first epoch appeared after 12 minutes 22 seconds, while subsequent warm epochs took about 1 minute 58 seconds. The long initial silence includes full provenance hashing and a cold first data pass.

For the J arm, budget:

- first log: approximately 8–13 minutes;
- complete 12-epoch run plus final validation/bootstrap: approximately 35–42 minutes;
- steady output storage: approximately 48–55 MiB;
- peak output storage during atomic checkpoint replacement: less than 85 MiB.

The J inputs already exist, and the local overlay had approximately 134 GiB free at preflight time.

## Incidental access audit

During schema inspection, one command intended to inspect the shared request catalog printed two catalog rows. Exactly one printed row was tagged as belonging to the sealed test split and included catalog metadata/token IDs.

- No test tensor arrays were opened.
- No test router states or routes were opened.
- No test labels or evaluation metrics were opened.
- No test candidate pool was opened.
- No test result informed architecture, hyperparameter, checkpoint, or model-selection decisions.
- The record contents are intentionally not reproduced here.
- All subsequent checks used explicit train/validation allowlists only.

This was an incidental catalog-record exposure in the diagnostic shell, not test-set evaluation by either the active raw trainer or the proposed J-Lens arm. It is recorded here for complete auditability and must not be silently omitted from provenance notes.
