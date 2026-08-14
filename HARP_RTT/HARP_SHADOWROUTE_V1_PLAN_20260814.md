# HARP-ShadowRoute v1

Date: 2026-08-14  
Lineage: agent/harp-shadowroute-v1 from 9543d1cca19ef6a87975749c605a3e7ddaf6afd5  
Objective: strict Recall@8 at H1-H4, with large-gain experiments only

## Decision

DeltaRoute v4 found the decisive bottleneck. Authoritative adaptive-tree routes
combined with the raw causal MTP prior reach 0.8646 H2-H4 Recall@8, but learned
branch-route translation remains near 0.56. More tree width, selector capacity,
or C64 ranking cannot recover that missing 30-point route gap.

ShadowRoute replaces route translation with approximate target execution:

1. restore the exact committed-prefix target hybrid cache;
2. execute the exact frozen target token mixer, norms, router, and shared expert;
3. replace only the 60 GiB routed-expert pool with learned resident shadow
   experts;
4. feed the resulting hidden state through the next target layer;
5. read each layer's exact frozen target router.

The route predictor therefore learns the missing nonlinear routed residual,
rather than independently classifying 160 expert sets.

## Hardware class

The first exact-backbone experiment requires one RTX PRO 6000 Blackwell 96 GB
or equivalent single GPU with at least 94 GB VRAM, at least 192 GB host RAM,
and at least 1 TB local NVMe. H100 80 GB is not accepted because the exact BF16
target is about 67 GiB before cache, hooks, and workspaces. H100 NVL 94 GB or
H200 141 GB are valid substitutes.

Layer-local S0/S1/S2 fitting uses one native expert layer at a time and fits on
a 24 GB 3090/4090. S2 is always layer-sharded. A monolithic target plus a live
one-billion-parameter S2 Adam optimizer is not authorized on a 96 GB GPU.

The run order writes to local NVMe, audits and checksums locally, then mirrors
immutably to persistent storage. No pod may be started automatically. The pod
must be stopped, not terminated, after the experiment and documentation are
complete.

## Implemented architecture

### S0: mechanism oracle

Each target layer keeps the exact native top-one expert contribution and adds
one learned width-512 SwiGLU draft expert. Only the draft is trainable.

This is privileged because the native routed pool remains resident. Its purpose
is to test whether exact backbone execution plus an approximate missing routed
residual transfers the DraftExpert mechanism to the 256-expert/top-eight Qwen
target.

### S1: deployable shared residual

One width-128 layer-local SwiGLU predicts the complete routed residual. It is
31.46 million parameters across forty layers and is the mandatory memory and
latency control.

### S2: expert-indexed sparse shadows

Each target layer and expert receives a width-16 SwiGLU shadow. The complete
pool is 1,006,632,960 parameters, 1.875 GiB in BF16. Only the selected eight
shadows execute. Each layer is initialized from the sixteen native target
neurons with greatest deterministic gate/up/down norm product.

Experts below the frozen minimum sample count never emit random output. Any
route containing such an expert falls back for that complete layer to S1.
Promotion requires at least 99 percent of held-out selected weight mass on
trained expert/layer cells.

### Exact-prefix branch rollout

The evaluator restores native routed experts while replaying the authoritative
committed prefix. It then freezes that hybrid cache, restores the chosen shadow
pool, executes exact H1 once, and visits each adaptive-tree node once in
parent-before-child order. Every sibling receives a deep-cloned parent cache.

The reference cache contains thirty GatedDeltaNet recurrent/conv states and ten
full-attention KV states. Calling it a KV-only cache is incorrect.

### Raw-prior mixture

For H2-H4, exact-k shadow route marginals are weighted by unnormalized causal
MTP path mass. Uncaptured probability is assigned to OTHER, which reproduces
the frozen HARP anchor. Captured mass is never renormalized over only the tree.

The raw MTP prior remains frozen initially because native routes score 0.8646
with it, versus 0.8551 with the learned posterior.

## Existing-data stage

The layer-local trainer uses only the already-open, request-disjoint outer-train
reuse profile:

- fitting: 224 requests, 3,584 source positions;
- tuning: 32 requests, 512 positions;
- diagnostic development: 128 requests, 2,048 positions.

Each factual position supplies four target tokens, forty pre-MoE router inputs,
native top-eight IDs and weights, and the full aggregate routed residual. The
trainer loads only one native expert layer from the sharded checkpoint, exactly
reconstructs selected expert outputs, and trains one immutable layer shard.
Formal validation, calibration, and sealed test stay closed.

The local component gate is deliberately weak and is not an accuracy claim:
at least 50 percent residual-MSE reduction over zero and cosine at least 0.70.
Its sole purpose is to prevent an obviously broken layer from entering the
expensive closed-loop run.

## Large-gain ladder

### A. Thirty-two-position engineering proof

Use S0, exact native prefix replay, and the existing adaptive trees and native
counterfactual companion.

Required:

- exact target config and all eighty native routed tensor names match;
- full native prefix plus native future replay reproduces every stored BF16
  top-eight ID bit-for-bit;
- H1 is processed once and siblings are cache-isolated;
- layer shards have one source/model/split lineage;
- target labels never enter the causal model API;
- optimizer is absent;
- maximum reserved VRAM is at most 90 GiB.

Any native parity mismatch stops the run.

### B. S0 large-gain experiment

Train seed 42 only. Small diagnostic variants and routine seed sweeps are
forbidden.

- 100k-token checkpoint: continue only with branch H2-H4 Recall@8 at least
  0.70, a gain of at least 0.10 over the 0.55717 DeltaRoute translator.
- 500k-token checkpoint: require H2-H4 at least 0.85 and H4 at least 0.80.
- Run 1M only if the 500k result is between 0.80 and 0.85 and its measured
  slope projects a crossing.

Failure ends ShadowRoute without new branch capture.

### C. S1

Run only after S0 passes. Require at least half of S0's lift over the matched
no-shadow baseline, branch H2-H4 at least 0.75, and no H4 regression. If S1
meets the final learned candidate gates, S2 is skipped.

### D. S2

Train one layer at a time from exact selected-expert effects. Require
teacher-token branch route recall at least 0.92, fully closed-loop recall at
least 0.88, and retention of at least 80 percent of the S0 lift. No width-32
sweep is authorized.

### E. Existing adaptive tree

Use the deployed budget-16 mask and raw causal MTP prior. The strict interim
accuracy gate is:

- H2-H4 Recall@8 at least 0.80;
- H4 Recall@8 at least 0.75;
- learned C64 mean H2-H4 at least 0.98;
- H4 C64 at least 0.97;
- H4 prefix-mismatch C64 at least 0.93.

This is an intermediate gate toward the final 0.90 H1-H4 objective, not the
final claim. A learned selector, axial corrector, or final C64 ranker remains
closed until this gate passes.

## Capture policy

No new request capture is authorized for S0 engineering or the first
closed-loop result. Existing factual full states and counterfactual route labels
are sufficient.

Only after S0 demonstrates a large closed-loop gain may target replay add a
label-only ShadowRoute sidecar. The compact contract stores:

- post-attention/pre-MoE state, aggregate routed residual, and post-layer state;
- native router logits, IDs, and execution weights;
- each selected expert effect projected into the next router basis;
- full 2,048-dimensional individual expert outputs only on a deterministic
  audit subset.

The sidecar is outer-train-only, runtime-unavailable, hash-bound to its tree,
model, geometry, split, and source commit, and inaccessible to inference.
All-256 expert outputs are never materialized or stored.

If the existing-data mechanism passes, the request-diverse successor uses at
least 2,000 new requests split by complete request and lineage group, with
8k, 20k, then 32k source-position gates. The full 32k capture is not launched
from an unpromoted 100k diagnostic.

## Artifact contract

Every stage uses a fresh output directory and writes:

- immutable run_manifest.json before work;
- MEMORY_AUTOTUNE.json or memory preflight;
- a separate OPTIMIZER_START.json when an optimizer is constructed;
- fsynced metrics;
- request-level diagnostic rows;
- an immutable best checkpoint;
- STAGE_RESULT.json;
- SHA256SUMS.

Every file records source commit, target checkpoint index hash, partition and
companion hashes, exact trainable names, mode/layer, seed, split seals, and
whether training or an optimizer started.

## Current implementation map

- harp_rtt/shadow_expert.py: S0/S1/S2 routed substitutes and exact layer-local
  native expert replay.
- harp_rtt/shadow_checkpoint.py: exact Qwen contract and selective sharded
  checkpoint access.
- harp_rtt/shadow_backbone.py: official-model replacement seam and exact-prefix
  native/shadow switching.
- harp_rtt/shadow_cache.py: causal hybrid-cache tree traversal.
- harp_rtt/shadow_rollout.py: exact-prefix future shadow execution.
- harp_rtt/shadow_bundle.py: fail-closed forty-layer bundle loading.
- harp_rtt/shadow_route.py: raw-MTP-prior exact-cardinality mixture plus OTHER.
- harp_rtt/shadow_capture.py: label-only compact capture contract.
- harp_rtt/shadow_training.py: residual, hidden, router, set, boundary, LM and
  factual objectives plus large-gain gates.
- runpod/train_shadow_experts_local.py: layer-sharded no-recapture fitting.
- runpod/evaluate_shadow_route_top8.py: native parity and strict closed-loop
  route evaluation.

## Stop conditions

Do not respond to failure by increasing tree width, candidate width, selector
capacity, ranker capacity, or shadow width. Stop when:

- native cache parity fails;
- S0 misses its 100k large-gain gate;
- exact backbone execution remains weak despite accurate local residuals;
- any split, lineage, or label-leakage audit fails.

Only a successful large-gain S0 result authorizes S1/S2 and eventual
request-diverse capture.
