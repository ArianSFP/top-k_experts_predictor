# HARP Resident-Shadow v1: Compact ShadowRoute Handoff

Date: 2026-08-15

## Outcome and current decision

PR #7 established the first system above the project target: strict mean
H1--H4 Recall@8 was 0.9166, with H1/H2/H3/H4 of
0.9445/0.9301/0.9090/0.8827. Its full target-expert INT4 shadow bundle was
about 15.94 GiB, however, so it is an accuracy teacher rather than the final
low-VRAM prefetcher.

The compact successor is **Resident-Shadow v1**. It keeps a globally budgeted
subset of target expert functions permanently resident in group-64 INT4 and
uses a width-512 shared draft only for selected execution-weight mass whose
expert is absent. It never loads, calls or holds a reference to an offloaded
target expert. The native MTP model is outside the user-declared size limit.

The frozen budget is 3,680 resident layer/expert cells plus forty BF16 shared
fallbacks:

| Payload | Size |
| --- | ---: |
| Resident target expert cells, group-64 INT4 | 5.7275 GiB |
| Width-512 BF16 shared fallbacks | 0.2344 GiB |
| **Total shadow routed model** | **5.9619 GiB** |

Exact non-expert target weights, token embedding, LM head and committed hybrid
cache are shared with the serving target and are not duplicated. Native MTP is
also excluded. Dynamic branch state is reported separately.

## What made the large ShadowRoute model successful

The causal gains came from execution semantics, not from INT4 as a format:

1. The official target normalisation, 30 GDN/linear mixers, ten full-attention
   layers, shared experts, routers and committed-prefix cache remained exact.
2. Future branch tokens were executed closed-loop through all forty layers.
3. Expert identity and the complete expert-specific nonlinear function were
   preserved. Static set heads and compressed router-coordinate transitions
   did not recreate the hidden trajectory.
4. Native selected IDs and original execution weights were propagated without
   truncation-weight renormalisation.
5. Causal Shadow-LM path likelihood added about 5.6 points on the Stage-A
   factual mixture and nearly saturated the target-posterior control on the
   full probe.
6. H1 remained an exact committed-token root rather than a branch mixture.

The cleanest causal ladder was:

| Condition | Branch Recall@8 |
| --- | ---: |
| Static direct translator | 0.5572 |
| Exact target top-1 + width-512 draft | 0.7985 |
| Exact target top-4 | 0.9146 |
| Full target functions, group-64 INT4 top-8 | 0.9552 |

Magnitude-selected miniature expert neurons reached only 0.679--0.715. The
required object is therefore the expert-conditioned nonlinear residual inside
an exact closed-loop backbone.

## Compact architecture

For target layer `l`, let `R_l` be the frozen train-selected resident expert
namespace. The exact router supplies all eight IDs and the original weights.
For resident selections the packed target function is executed. Missing mass
uses the layer's shared learned routed-residual model:

```text
resident(e) = 1[e in R_l]
missing_mass = sum_i w_i * (1 - resident(e_i))

delta_hat = missing_mass * SharedDraft_l(a_l)
          + sum_{i: resident(e_i)} w_i * INT4TargetExpert[l,e_i](a_l)
```

No weights are renormalised. When a route-conditioned BasisDraft fallback is
used, resident outputs replace the corresponding basis approximation rather
than being added twice.

The per-layer namespaces are chosen once from outer-train routes only. With
equal expert-cell storage cost, the planner gives every layer a floor of 64
residents, then assigns each remaining cell to the largest available marginal
train-slot coverage, up to 128 per layer. This is the exact global optimum for
the declared slot-coverage objective. The plan is hash-bound to the partition,
reuse split, companion and source commit.

For the 3,680-cell plan, counts range from 73 to 115 residents per layer.
Examples are L0=115, L6=102, L20=81 and L32=103. Mean train selected-slot
coverage is 0.6957. The allocation substantially repairs the high-entropy
early layers without increasing the bundle size.

## No-recapture experiments completed

All results below use only request-disjoint outer-train data. Formal
validation, calibration and sealed test remain unopened.

### Rejected compact readouts

The matched layer-20 width-512 shared baseline reached development next-router
Recall@8 0.7064. The following variants were stopped under the user's large-
gain rule:

| Variant | Result |
| --- | --- |
| Scalar route coefficients | negligible gain |
| Per-neuron basis coefficients | about +0.5 point tune |
| Expert residual width 2 | about +1 point |
| Expert residual width 16, 25.3M trainable parameters/layer | 0.6823 to 0.6934 tune after two epochs; stopped |

This rules out spending further time on larger independent miniature adapters
with the current fitting corpus.

### Resident hybrid

Uniform resident-80 produced the first large component result: layer 20 rose
from 0.7064 to 0.8170 (+11.06 points). Uniform resident-92 fits under 6 GiB and
raised the representative three-layer mean by +9.57 points.

The frozen coverage-aware allocation improves the weakest layers at the same
size. Development results are:

| Layer | Shared | Resident count | Resident hybrid | Gain |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7770 | 115 | 0.8506 | +0.0736 |
| 6 | 0.7016 | 102 | 0.7983 | +0.0967 |
| 20 | 0.7064 | 81 | 0.8185 | +0.1121 |
| 32 | 0.7564 | 103 | 0.8403 | +0.0839 |

The four-phase mean gain is +9.16 points. This is a large architectural gain
and justifies constructing the complete bundle. It is not yet an H1--H4
closed-loop result.

### Complete forty-layer bundle

The complete source-bound bundle was constructed and sealed on the 3090. It
contains exactly forty resident shards and exactly 3,680 resident
layer/expert cells. The serialized deployment payload is 6,401,871,784 bytes,
or **5.9622 GiB**. This is the actual checkpoint footprint, not only a formula.

Thirty-two layers have an authoritative teacher-forced next-router metric.
Every one of those layers improved over its matched width-512 shared fallback:

| Metric over 32 auditable layers | Shared fallback | Resident hybrid | Gain |
| --- | ---: | ---: | ---: |
| Mean next-router Recall@8 | 0.7247 | 0.8172 | **+0.0926** |

The smallest per-layer gain was +5.67 points and the largest was +12.27
points. The eight layers whose immutable next-router reconstruction audit did
not pass (7, 16, 30, 34, 35, 37, 38 and 39) were trained and evaluated only on
the authoritative routed-residual target. No tolerance was relaxed and no
router metric was fabricated. Their resident tier consistently improved the
residual reconstruction; for example layer 34 normalized development RMSE
fell from 1.1475 to 0.4773 and layer 39 from 0.6625 to 0.3826.

The deployment manifest binds:

```text
training/export source = 6d119b3173f207e7082be21ebad5d09d1051d176
target checkpoint index = 41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83
allocation plan = 8f8e7d31522557c42669778de26834e751fefa10cd2170249115588ca471ba15
source bundle result = 3b18f17d347e309c32ff00e1d065a944daf97728026875e334c2c8f6b6b3f98a
```

The immutable persistent mirror is:

```text
/workspace/LLM_prefetch_study/artifacts/harp_rtt/
  resident_shadow_v1_20260815_6d119b3/
    scientific_bundle/
    deployment_bundle/
```

All 687 scientific files and all 42 deployment files passed checksum
verification after mirroring. `deployment_bundle/` contains only the forty
runtime checkpoints, frozen allocation plan, deployment manifest and
checksums. It excludes training predictions and shared-fit diagnostics.

## Execution contract

- Packed resident tensors are part of the predictor bundle; there is no native
  target-expert module or offload callback in the runtime graph.
- The complete native top-eight route is required. Missing execution-weight
  mass is never dropped or renormalised.
- Exact committed-prefix replay retains the target experts only in the 96-GB
  scientific evaluator. Production receives the cache already constructed by
  normal target decoding.
- Runtime tree execution is ancestor-closed budget 16; it does not execute the
  unused adaptive-32 nodes.
- Shadow-LM probabilities remain causal and captured-path mass remains
  unnormalised; OTHER completes residual probability with the frozen anchor.
- Every packed layer shard records resident IDs, plan/checkpoint hashes,
  source/data lineage, optimizer absence and sealed-split flags.

## Remaining gate

The forty-layer compact bundle is complete. The next required machine is an
RTX PRO 6000 96 GB, H100 NVL 94 GB, or H200 141 GB for:

1. one-tree native-prefix/cache parity;
2. a 32-request budget-16 closed-loop screen with Shadow-LM;
3. the full 128-request probe only if the screen retains branch Recall@8 at
   least 0.90 and mean H1--H4 at least 0.88.

Final compact promotion remains mean H1--H4 at least 0.90 and H4 at least
0.85. The absolute continuation floor is 0.80. A final C64 ranker stays closed;
the compact route execution must first demonstrate that the candidate/route
information survives compression.
