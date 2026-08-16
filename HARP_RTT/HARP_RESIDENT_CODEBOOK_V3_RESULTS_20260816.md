# HARP Resident Functional Codebook v3 Results

Date: 2026-08-16

## Outcome

The resident functional codebook failed its predeclared four-phase component
gate and is stopped before all-layer fitting or exact-cache evaluation.

The experiment asked whether a missing selected target expert could be replaced
by one resident expert function, or by a convex mixture of two resident expert
functions, while preserving the target router's original execution weight. On
layers 0, 6, 20, and 32, both variants were worse than executing no substitute
for the missing expert.

No optimizer was constructed. Formal validation, calibration, and sealed test
data remained closed. No target or MTP recapture was performed.

## Frozen experiment

- Source commit: \`34d773c1a1ef1428176a6a500c8de86b1f0e33c6\`.
- Parent: sealed Resident-Only v2, 3,850 group-64 INT4 resident cells,
  5.992392 GiB.
- Fitting split: 224 requests / 3,584 source rows.
- Tuning split: 32 requests / 512 source rows, request-disjoint from fitting.
- Representative target layers: 0, 6, 20, and 32.
- Expert fingerprints: request-balanced sampled activations from fitting only.
- Top-1 codebook: one resident proxy per missing expert.
- Top-2 codebook: a non-negative, sum-to-one mixture of two resident proxies.
- Execution weights were not renormalized.
- The runtime implementation never requests or loads a target expert.

The compact fitting corpus was copied to local NVMe before execution. Each
layer artifact is immutable, hash-bound to its parent and source commit, and
records \`training_started=false\` and \`optimizer_constructed=false\`.

## Results

Request-macro next-router Recall@8 on the untouched tuning requests:

| Layer | Resident-only parent | Nearest one | Nearest two | Exact missing-tail ceiling |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.951965 | 0.942078 | 0.947754 | 0.987061 |
| 6 | 0.939209 | 0.927063 | 0.933777 | 0.983887 |
| 20 | 0.948242 | 0.933044 | 0.937500 | 0.981445 |
| 32 | 0.951843 | 0.941956 | 0.948120 | 0.983521 |
| **Mean** | **0.947815** | **0.936035** | **0.941788** | **0.983978** |

Relative to Resident-Only v2:

- top-1 proxy substitution regressed by 0.011780 absolute;
- top-2 proxy substitution regressed by 0.006027 absolute;
- neither layer nor phase improved.

Mean normalized residual MSE also worsened:

| Condition | Mean normalized residual MSE |
| --- | ---: |
| Resident-only parent | **0.205314** |
| Nearest one | 0.315234 |
| Nearest two | 0.254158 |
| Exact missing tail | 0.000000 |

The codebook therefore fails both the route metric and the underlying residual
metric. It misses the required 0.978 next-router gate, the 80% local-gap
recovery gate, and the 60% residual-error-reduction gate by a wide margin.

## Interpretation

The useful compact mechanism is not interchangeable expert identity. The
5.99-GiB parent succeeds because it executes complete nonlinear functions for
the resident target experts and omits the rest. Adding a different complete
expert function for a missing slot is worse than leaving that contribution at
zero, even when the proxy was selected from sampled functional responses.

This rules out:

- fitting all forty codebook layers;
- using top-2 or wider convex resident mixtures as a capacity sweep;
- spending RTX PRO 6000 time on the codebook candidate.

The implementation remains useful as a reproducible negative control. The
selected compact candidate remains the sealed Resident-Only v2 bundle.

## Artifacts

Persistent mirror:

\`/workspace/LLM_prefetch_study/artifacts/harp_rtt/resident_codebook_v3_20260816_34d773c\`

It contains the complete fits for layers 0, 6, 20, and 32, including fit
diagnostics, immutable manifests, tensor tables, stage results, and verified
SHA-256 inventories.

## Next experiment

Proceed with the already-declared exact-cache evaluation of Resident-Only v2:

1. run the 32-request budget-16 Shadow-LM screen on an RTX PRO 6000 96GB;
2. require mean H1-H4 Recall@8 at least 0.88 and H4 at least 0.85;
3. run the full 128-request diagnostic partition only if the screen passes;
4. promote the compact model only at mean H1-H4 Recall@8 at least 0.90.

No further small substitute head should be added before this closed-loop test.
The remaining local exact-tail gap is real, but the negative fallback,
correction, miniature-expert, basis, and codebook experiments show that it is
not recoverable by another low-capacity static approximation.
