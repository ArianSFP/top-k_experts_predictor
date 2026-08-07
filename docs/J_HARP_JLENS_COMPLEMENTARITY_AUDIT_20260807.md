# J-Lens / raw-residual complementarity audit

Date: 2026-08-07
Status: validation-only; sealed test untouched

## Decision

Use a dual J-Lens plus raw-residual stream in the next accuracy-first model. A frozen linear probe over all 40 layers improves mean H1-H4 validation Recall@8 from **0.62135** with raw rank-512 alone to **0.62821** with both streams. The paired request-macro gain is **+0.00686**, 95% CI **[+0.00598, +0.00775]**.

For the next dimension expansion, prioritize **raw rank-1024 + J rank-512** before symmetric dual rank-1024. Raw PCA has considerably more variance left after rank 512, while J-space is closer to saturation.

Machine-readable evidence: [audit_summary.json](artifacts/harp8/j_harp_c64_runs/jlens_complementarity_audit_20260807/audit_summary.json).

## Leakage controls and protocol

- PCA was fitted on 30,000 train-only `(row, layer)` pairs, stratified by domain, request, and target layer.
- The diagnostic probe fitted on 4,096 deterministic train source rows.
- Evaluation used all 12,300 H1-H4-eligible source rows from 410 complete validation requests and all 40 layers.
- The test split was not indexed. Its count was read only from request metadata.
- Each layer received an independent ridge head from current post-layer features to four future 256-router-logit vectors. Ridge strength was frozen at `0.01 * n_train`.
- Top-8 used descending predicted score with stable ascending expert-ID tie-breaking.
- Confidence intervals use 5,000 paired bootstrap replicates over complete validation requests.

This is a representation diagnostic, not a replacement for a trained HARP/Full256 comparison. It excludes route history, MTP, and nonlinear fusion deliberately.

## Main predictive result

| Input | H1 R@8 | H2 R@8 | H3 R@8 | H4 R@8 | Mean H1-H4 |
|---|---:|---:|---:|---:|---:|
| J rank-512 | 0.65921 | 0.60807 | 0.58738 | 0.55707 | 0.60293 |
| Raw rank-512 | 0.68045 | 0.62731 | 0.60441 | 0.57324 | 0.62135 |
| Dual rank-512 | **0.68754** | **0.63356** | **0.61132** | **0.58041** | **0.62821** |

| Horizon | Dual minus raw | Paired 95% CI |
|---|---:|---:|
| H1 | +0.00709 | [+0.00630, +0.00792] |
| H2 | +0.00626 | [+0.00535, +0.00713] |
| H3 | +0.00691 | [+0.00589, +0.00793] |
| H4 | +0.00717 | [+0.00608, +0.00825] |
| Mean H1-H4 | **+0.00686** | **[+0.00598, +0.00775]** |

The improvement is positive at every horizon. Raw alone is stronger than J alone by 0.01842 mean Recall@8, so J should be a complementary residual source rather than a replacement for raw residuals.

The two single-stream top-8 lists have mean Jaccard overlap 0.6659 and mean union size 9.87 experts. That union covers 0.66890 of true route slots. This is not an R@8 result because its budget is almost ten, but it confirms usefully different errors and motivates learned fusion or reranking.

## Geometry and information overlap

| Representation | Rank 128 | Rank 256 | Rank 512 |
|---|---:|---:|---:|
| J-Lens | 0.78864 | 0.86956 | **0.93228** |
| Raw residual | 0.61563 | 0.71304 | **0.81067** |

J transport makes the shared across-layer distribution more compressible. It is not automatically more route-predictive: raw rank-512 wins the linear future-router task.

For every layer, the J PCA span was pulled back into raw coordinates as `J_l.T @ C_J` and compared with `C_raw` using canonical angles. Across all 40 layers, mean squared canonical correlation is **0.4190**, equivalent to **214.6 shared dimensions out of 512** under this overlap measure. Layer means range from 0.3564 to 0.4853. The two truncations therefore select substantially different raw-coordinate subspaces.

A held-out layer-specific ridge reconstruction on layers 4, 12, 20, 28, 36, and 39 explains 81.72% of raw rank-512 variance from J rank-512, and 78.23% in the reverse direction. The streams overlap heavily, but each retains meaningful non-reconstructible information.

## Is rank 1024 likely useful?

Yes for raw residuals; plausibly but less urgently for J-space.

Because the PCA spectrum is decreasing and 1,536 dimensions remain after rank 512, the next 512 PCs contain at least one third of remaining tail energy and no more energy than PCs 257-512. This gives conservative bounds:

| Representation | Rank-512 captured | Rank-1024 captured bound | Potential added fraction |
|---|---:|---:|---:|
| J-Lens | 0.93228 | [0.95485, 0.99499] | +0.02257 to +0.06272 |
| Raw residual | 0.81067 | [0.87378, 0.90829] | +0.06311 to +0.09762 |

These are mathematical spectrum bounds, not measured rank-1024 artifacts. Predictive utility need not track variance, so matched training is required.

## Accuracy-first rank-1024 plan

1. Re-run immutable train-only PCA preparation with the same source hashes, 30,000 fit pairs, seed 42, six PCA iterations, and `--pca-ranks 512,1024`, exporting only `raw_residual,j_lens` to a new directory. Do not modify the rank-512 artifact.
2. Verify fit-pair SHA `847d333bb801f24c39a618f28b6ab690f7f4392925ea96ce36b14ece64bbf6bc`, source/lens hashes, finite values, FP16 round-trip, and row alignment.
3. Run this matched validation-only factorial: raw512, J512, dual512, **raw1024+J512**, raw1024, J1024, and only then symmetric dual1024.
4. Keep the Full256 route head, initialization, sample order, optimizer steps, and loss profile identical. Select on complete-request validation Recall@8 H1-H4, while reporting H1-H8 and per-layer results.
5. Use independent source normalization and a learned per-layer source gate. Do not add J and raw coordinates directly because their PCA bases differ.
6. If raw1024+J512 wins, test raw2048+J512 before J1024. The evidence favors preserving more raw tail first.

Each FP16 `[69632,40,1024]` stream is about 5.31 GiB. The pod had about 149 GiB local free during this audit.

## Interpretation

1. J-space is not a superior substitute for raw residual PCA here.
2. It is a statistically supported complementary stream, worth about 0.7 Recall@8 points in a simple linear fusion.
3. Rank expansion should be asymmetric: recover raw residual tail first, retain J512 as auxiliary semantic transport, then test whether J1024 adds anything.

This does not imply the combined model is near the 0.90 target. It identifies a reliable next direction for the nonlinear Full256/HARP experiment.
