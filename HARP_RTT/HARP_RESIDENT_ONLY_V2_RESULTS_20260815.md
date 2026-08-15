# HARP Resident-Only v2 3090 Results

Date: 2026-08-15

## Outcome

The 3090 information and component gate passed. The compact successor is now
**Resident-Only ShadowRoute**, not the v1 resident-plus-shared-fallback model.

The decisive result is that the old width-512 fallback was actively damaging
the next-router state. Removing it and reallocating its storage to more complete
INT4 expert functions gives a large, consistent proxy gain while remaining
below the exact six-GiB limit.

No formal validation, calibration, or sealed-test data were opened. The native
MTP model is excluded from the size accounting, as required.

## What made ShadowRoute successful

The successful mechanism from PR #7 is preserved:

1. exact target non-expert backbone and committed-prefix hybrid cache;
2. exact frozen target routers, native top-eight IDs, execution weights, and tie
   semantics;
3. closed-loop hidden-state evolution through all forty layers;
4. complete expert-specific nonlinear functions for resident experts;
5. causal Shadow-LM branch likelihood plus explicit OTHER-to-anchor mass.

Full INT4 worked because it retained expert identity and nonlinear function.
Static route heads, generic residual heads, magnitude-selected miniature
experts, and tiny expert adapters did not.

## Experiments

### 1. Tail/control headroom audit

The first audit compared the sealed v1 hybrid with:

- resident-only execution;
- exact missing-tail execution;
- rank-32/64/128 router-rowspace correction oracles.

On representative layers 0, 6, 20, and 32:

| Condition | Mean next-router Recall@8 |
| --- | ---: |
| v1 resident + generic fallback | 0.815033 |
| v1 residents only | 0.939667 |
| exact missing tail | 0.983978 |
| rank-128 correction from v1 | 0.859543 |

The generic fallback therefore caused most of the observed error.

### 2. Tail retraining

Two single-seed layer-0 diagnostics were run:

- retrain from the sealed fallback initializer;
- zero the fallback output before tail-specific retraining.

Both selected epoch zero. Optimizer steps caused immediate and sustained
regression. The zero-output epoch-zero checkpoint reached 0.928650 on tune and
0.930191 on diagnostic development, while trained epochs fell to about
0.806-0.813 on tune. Tail optimization was stopped after the predeclared
patience window.

This rules out spending all-layer compute on the current tail objective.

### 3. Fallback-free reallocation

The 240 MiB fallback was removed. Its bytes were reassigned to complete
group-64 INT4 resident expert cells. A training-only execution-weight-mass
planner allocated 3,850 cells under per-layer bounds 64-128.

Training coverage:

| Metric | Value |
| --- | ---: |
| Selected-slot coverage | 0.707438 |
| Selected execution-weight mass | 0.746339 |
| Resident cells | 3,850 |

The planner used only the 224 training requests / 3,584 rows from the frozen B2
reuse split. Tune selected no parameters or namespace. Diagnostic development
did not influence allocation.

### 4. All-auditable-layer proxy

Thirty-one target transitions passed the authoritative next-router
reconstruction check. Transitions 7, 16, 30, 34, 35, 36, 37, and 38 were
excluded rather than assigned fabricated metrics.

| Condition | Mean Recall@8 |
| --- | ---: |
| Resident-only | **0.946807** |
| Exact missing tail | 0.984042 |
| Rank-32 correction oracle | 0.956226 |
| Rank-64 correction oracle | 0.962117 |
| Rank-128 correction oracle | 0.971656 |

Complete-request bootstrap for resident-only:

- point estimate: 0.946807;
- 95% interval: [0.943143, 0.950434];
- 1,000 replicates, seed 42, 32 complete requests.

By horizon:

| Horizon | Recall@8 |
| --- | ---: |
| H1 | 0.947872 |
| H2 | 0.944612 |
| H3 | 0.946486 |
| H4 | 0.948258 |

On the matched four representative layers, Resident-Only improves over the v1
fallback hybrid by **+0.132706**, with paired request-bootstrap interval
[+0.120056, +0.146011]. This passes the requested large-gain gate.

The rank-128 correction has only +0.024849 mean oracle lift after the stronger
resident allocation, so it is rejected before training.

## Size and safety

The immutable forty-layer bundle contains:

- 3,850 complete resident expert cells;
- group-64 signed INT4 weights and BF16 scales;
- no fallback weights;
- no target expert references;
- no permission to initiate target expert loads;
- no optimizer state.

Exact serialized checkpoint size:

- 6,434,281,576 bytes;
- 5.992392 GiB;
- 8,169,368 bytes below 6 GiB.

The native MTP is not included. The target non-expert backbone, embedding,
LM head, and committed cache are shared with the running target and are not
duplicated by this bundle.

## Artifact inventory

Local NVMe:

- plan:
  `/root/shadow_v2_nvme/runs/resident_only_plan_3850_8c0cde1`
- sealed bundle:
  `/root/shadow_v2_nvme/runs/resident_only_export_8c0cde1`
- proxy aggregate:
  `/root/shadow_v2_nvme/runs/resident_only_proxy_aggregate_25317a5`
- failed initialized-tail run:
  `/root/shadow_v2_nvme/runs/tail_l00_seed42_9c76024`
- failed zero-tail run:
  `/root/shadow_v2_nvme/runs/tail_zero_l00_seed42_5ff3d20`

Key hashes:

- bundle source commit: `8c0cde1c272cb1eca82cf90e500a9c73367b06df`;
- allocation plan SHA-256:
  `06a42a5c1014b852fe703e583c45d21539b7488d46141d4a714c3ecd09038033`;
- target checkpoint index SHA-256:
  `41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83`.

Every layer has its own checksum inventory. The sealed root has a complete
`BUNDLE_SHA256SUMS`.

## Next experiment

The next required hardware is one **RTX PRO 6000 96GB** (or H100 NVL 94GB /
H200 141GB equivalent), with at least 192 GB host RAM and 250 GB local NVMe.

Run the exact-cache closed-loop evaluation in this order:

1. 32-request screen with budget-16 runtime node masking and Shadow-LM;
2. compare Resident-Only v2 against the sealed v1 resident hybrid on the exact
   same requests;
3. proceed to all 128 requests only if v2 preserves at least 0.88 mean H1-H4,
   reaches at least 0.85 H4, and does not regress materially against v1;
4. promote compact v2 at mean H1-H4 >= 0.90.

The PRO run is essential because the 3090 metric is teacher-forced one-layer
agreement. It strongly predicts a large gain, but only exact hybrid-cache
closed-loop execution can measure compounding trajectory error.
