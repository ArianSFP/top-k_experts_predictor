# HARP RouteQuant v1 execution ledger

Date: 2026-08-16

## Objective

Find the smallest resident representation of the Qwen target experts that can
retain the authoritative HARP-ShadowRoute v1 result while preserving the
mechanism that produced it.

The accuracy target is the frozen request-disjoint 128-request / 2,048-position
ShadowRoute result:

| Metric | Recall@8 |
| --- | ---: |
| H1 | 0.944464 |
| H2 | 0.930142 |
| H3 | 0.908997 |
| H4 | 0.882690 |
| Mean H1--H4 | **0.916573** |
| Branch H2--H4 | 0.951183 |

The full group-64 INT4 expert representation used 15.9375 GiB. The native MTP
model and target-owned non-expert backbone are excluded from expert-payload
size, as requested.

No formal validation, calibration, or sealed-test labels were opened. All
selection utilities use fitting requests only; the 32 request-disjoint tuning
requests measure the component frontier; diagnostic development is loaded only
for lineage-disjointness checks.

## Scientific diagnosis

ShadowRoute succeeded because it retained:

1. the exact target non-expert backbone and authoritative hybrid prefix cache;
2. closed-loop target-state evolution through all forty layers;
3. native router IDs, execution weights, and tie semantics;
4. each target expert's complete identity-specific nonlinear SwiGLU function;
5. causal Shadow-LM branch likelihood and explicit OTHER-to-anchor completion.

The compact failures establish the converse. Generic residual experts,
BasisDraft, magnitude-selected miniature experts, low-rank expert adapters, and
resident functional proxies did not retain the route trajectory. The largest
causal gain came from restoring complete expert-specific nonlinear execution.

RouteQuant therefore compresses the functions rather than replacing them. It
stores every one of the 10,240 layer--expert cells at 1--4 bits, always executes
all native top-eight IDs with their original weights, never renormalizes those
weights, and never initiates a target expert load.

The design is consistent with published evidence that MoE expert weights can be
more low-bit robust than dense FFNs (MoQE), while the extreme-quantization
literature (AQLM and QMoE) motivates learned/mixed precision and warns that
storage format and decoder kernels must be evaluated separately.

## Implementation

Branch: `agent/harp-routequant-v1`

Implemented and tested:

- real packed 1-, 2-, 3-, and 4-bit full-width expert matrices;
- symmetric group-64 MSE quantization;
- deterministic fixed LOG8 group scales;
- expert-identity-preserving mixed bit schedules;
- train-only route/residual marginal-utility allocation;
- unseen-expert base-precision fallback;
- globally allocated, hash-bound 40x256 schedules;
- separate gate/up and down precision controls;
- strict exporter, bundle loader, exact-backbone installer, and exact-cache
  evaluator integration;
- reference dequantization cache with unchanged arithmetic;
- safe recovery/sealing of post-metric interrupted screens;
- fail-closed split, lineage, target, schedule, checksum, and privacy contracts.

Implementation commits through this checkpoint:

- `8879607`: identity-preserving RouteQuant;
- `7ed0a06`: exact reference dequantization cache;
- `15efada`: train-only route-aware allocation;
- `4c6e68c`: fixed LOG8 scale storage;
- `0289774`: exporter/bundle/evaluator integration;
- `e4aa157`: interruption-safe finalization;
- `de122cc`: global train-utility schedule compiler;
- `46c57f6`: asymmetric projection screen;
- `5f38fbf`: asymmetric runtime/export support.
- `7eed340`: globally normalized cross-layer utility allocation.

The current complete RouteQuant/ShadowRoute test slice passes 111 tests.

## 3090 component frontier

All values below are request-macro teacher-forced **one-layer next-router**
Recall@8 on layers 0, 6, 20, and 32. They are not end-to-end H1--H4 results.

### Uniform LOG8 precision

| Representation | Projected expert bytes | Mean next-router R@8 | Residual NRMSE |
| --- | ---: | ---: | ---: |
| INT1 | 4.2189 GiB | 0.876984 | 0.824352 |
| INT2 | 7.9689 GiB | 0.918976 | 0.561988 |
| INT3 | 11.7189 GiB | 0.958054 | 0.267910 |
| INT4 | 15.4689 GiB | **0.976273** | **0.128585** |

LOG8 scales save approximately 0.469 GiB over BF16 scales. FP8 E4M3 scales
were rejected because their range/underflow behavior materially degraded the
4-bit expert weights.

### Train-utility mixed adjacent bits

| Base -> upgrade | Upgraded cells | Size | Mean next-router R@8 | Residual NRMSE |
| --- | ---: | ---: | ---: | ---: |
| 1 -> 2 | 12.5% | 4.6876 GiB | 0.896896 | 0.710789 |
| 1 -> 2 | 25% | 5.1564 GiB | 0.904587 | 0.663446 |
| 1 -> 2 | 50% | 6.0939 GiB | 0.912323 | 0.612404 |
| 1 -> 2 | 75% | 7.0314 GiB | 0.915909 | 0.582626 |
| 2 -> 3 | 12.5% | 8.4376 GiB | 0.937637 | 0.446040 |
| 2 -> 3 | 25% | 8.9064 GiB | 0.944565 | 0.393483 |
| 2 -> 3 | 50% | 9.8439 GiB | 0.950699 | 0.334608 |
| 2 -> 3 | 75% | 10.7814 GiB | 0.953781 | 0.304722 |
| 3 -> 4 | 12.5% | 12.1876 GiB | 0.965881 | 0.216271 |
| 3 -> 4 | 25% | 12.6564 GiB | 0.968750 | 0.193734 |
| 3 -> 4 | 50% | 13.5939 GiB | 0.971512 | 0.166822 |
| 3 -> 4 | 75% | 14.5314 GiB | 0.972839 | 0.153646 |

The under-6-GiB tier is now measured rather than assumed. Its component route
accuracy is useful, but its gap from INT4 is too large to claim retention of the
0.916573 end-to-end result without exact-cache evidence.

### Asymmetric projection precision

| Gate/up bits | Down bits | Size | Mean next-router R@8 | Residual NRMSE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 5.4689 GiB | 0.891037 | 0.745434 |
| 2 | 1 | 6.7189 GiB | 0.896637 | 0.703293 |
| 2 | 3 | 9.2189 GiB | 0.931976 | 0.462396 |
| 3 | 2 | 10.4689 GiB | 0.939636 | 0.429718 |
| 2 | 4 | 10.4689 GiB | 0.935196 | 0.436613 |
| 4 | 2 | 12.9689 GiB | 0.945389 | 0.398484 |
| 3 | 4 | 12.9689 GiB | 0.964828 | 0.210662 |
| 4 | 3 | 14.2189 GiB | 0.967346 | 0.211010 |

Projection sensitivity is real and varies by layer, but every tested asymmetric
point is Pareto-dominated by train-utility complete-cell mixing. Asymmetric
precision is retained as a reproducible negative/control path and excluded from
the first exact-cache shortlist.

## All-layer component result

The optimizer-free train-utility pass completed over all 39 router transitions.
Its request-macro tuning aggregates are:

| Representation | Projected size | Mean next-router R@8 | Worst layer gap from exact | Residual NRMSE |
| --- | ---: | ---: | ---: | ---: |
| Uniform INT3 | 11.7189 GiB | 0.957056 | 0.036316 | 0.286870 |
| Global 50% INT3/INT4 | 13.5939 GiB | 0.968933 | 0.020203 | 0.187171 |
| Uniform INT4 | 15.4689 GiB | **0.973719** | **0.014221** | **0.139497** |

The globally normalized allocator freezes 25%, 50%, and 75% INT4 schedules.
Layer 39 is intentionally all INT4 because it controls Shadow-LM likelihood and
has no next-router transition for the same utility estimator. Across layers
0--38, the 50% schedule assigns 83--179 INT4 experts per layer; the most
capacity goes to layers 0--2 and 32--38, while layers 20--26 receive the least.
This confirms that global allocation preserves meaningful cross-layer
sensitivity which a fixed per-layer quota would discard.

The screen opened no development metrics, formal validation, calibration, or
sealed-test labels, constructed no optimizer, and started no training. Its
sealed checksums pass.

## Frozen closed-loop shortlist

The first exact-cache shortlist will be chosen from:

- uniform INT4 LOG8: 15.4689 GiB;
- globally mixed INT3/INT4 at 75%, 50%, and 25%;
- uniform INT3 LOG8: 11.7189 GiB.

Each candidate must first run on the same 32 development requests. Only
candidates within 0.5 point of the full-INT4 control, with no horizon losing
more than one point, proceed to the full 128 requests. The final retention gate
is mean H1--H4 >= 0.911573 with paired 95% lower delta >= -0.005 versus the
0.916573 reference and no horizon worse by more than 0.01. Among passing
representations, serialized bytes decide; measured kernel latency is reported
separately.

## Sealed artifacts

All five reference bundles were exported locally, each passed all 42 checksum
entries, and the persistent copies passed a second complete checksum readback.
They remain fail-closed with `closed_loop_authorized=false`:

| Bundle | Serialized bytes | GiB |
| --- | ---: | ---: |
| Uniform INT3 | 12,583,775,440 | 11.7196 |
| Global 25% INT4 | 13,590,423,504 | 12.6571 |
| Global 50% INT4 | 14,597,056,464 | 13.5946 |
| Global 75% INT4 | 15,603,689,552 | 14.5321 |
| Uniform INT4 | 16,610,312,400 | 15.4696 |

Persistent root:

`/workspace/LLM_prefetch_study/artifacts/harp_rtt/routequant_v1_20260816_7eed340`

The root is 69 GiB and contains the all-layer screen, three frozen schedules,
five bundles, and `ARTIFACT_INVENTORY.json`. The inventory SHA-256 is
`1271eb2de1efd986807bc4e28ad5a86a3811b41f97ca6ab5af7e5d91574b738f`.
The inventory binds every schedule and bundle result/checksum manifest to
implementation commit `7eed34031715a80041bcf95775de39941dca297f`.

## Hardware boundary and pause

The 3090 work is complete. Exact-cache closed-loop comparison requires one RTX
PRO 6000 96GB (or H100 NVL 94GB / H200 141GB), at least 192GB host RAM, and at
least 1TB local NVMe. No new target or MTP capture is required. Per the
user-requested pause, no RTX PRO 6000 work was started. The 3090 pod is stopped
after the GitHub and persistent-artifact handoff is verified.
