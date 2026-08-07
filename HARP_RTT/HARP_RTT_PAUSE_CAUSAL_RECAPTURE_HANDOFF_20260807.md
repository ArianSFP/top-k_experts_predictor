# HARP8 / HARP-RTT pause and causal-recapture handoff

**Decision time:** 2026-08-07 14:16 UTC

**Primary metric:** request-macro Recall@8 at H1--H4 (`t+1` through `t+4`)

**Decision:** pause same-input C64 optimization; pivot to branch-rich causal recapture
**Test policy:** the sealed outer test remains unopened

## Executive decision

The unchanged C64 baseline was stopped at the first clean boundary after the
pivot request: completed epoch 8. Its validation mean H1--H4 Recall@8 is
`0.7948380510`. The epoch-8 checkpoint is preserved read-only in local-overlay
and persistent storage with SHA-256:

```text
0f39a9922f460ab8460e952f70a38b83bb972b32a839afde30556d92accff438
```

This is a valid converged-direction measurement, but it is not close to the
available candidate ceiling. The fixed C64 pool contains mean H1--H4 oracle
coverage `0.9740958157`. Epochs 1--8 gained only about `0.00164` over the
frozen score baseline near `0.79320`, while reaching `0.90` requires about
`0.85` additional correct selections per layer-row. The trained ranker was
mostly sharpening the incumbent order instead of learning the missing swaps.

The prefix-stratified ceiling identifies the more important failure. When the
single native MTP draft prefix is wrong, H2--H4 base Recall@8 drops to roughly
`0.58--0.59`, and even the existing C64 oracle drops to roughly `0.85--0.87`.
More epochs on the same single-chain representation cannot recover experts
that are absent from the pool and cannot infer conditional routes for token
hypotheses that were never represented.

The next experiment is therefore a recapture, not another ranker run:

1. capture an exact-H1-rooted adaptive native-MTP tree before target H1;
2. preserve multiple H2--H4 token hypotheses and all causal branch state;
3. on a deterministic train-only subset, teacher-force the frozen target on
   selected speculative paths and store the resulting route semantics in a
   separate label-only companion;
4. gate scale-up on wrong-prefix H3/H4 candidate and route-reconstruction lift.

No HARP-RTT Phase 2--5 training is authorized from the legacy index. The
trainer already fails closed on that condition.

## Scientific evidence for the pivot

### Current achieved and oracle values

| Result | H1 | H2 | H3 | H4 | Mean H1--H4 |
|---|---:|---:|---:|---:|---:|
| Historical HARP endpoint | .815660 | .799970 | .781214 | .756959 | .788451 |
| Temperature-1 HARP, epoch 5 | .821683 | .802488 | .786069 | .762165 | .793101 |
| Partial C32 reranker, epoch 2 | -- | -- | -- | -- | .794187 |
| Unchanged C64 baseline, epoch 8 | -- | -- | -- | -- | **.794838** |
| Fixed C64 oracle | .989267 | .981193 | .969648 | .956275 | **.974096** |

The C64 epoch-8 run logged:

| Epoch | Train loss | Validation mean H1--H4 Recall@8 |
|---:|---:|---:|
| 1 | .32579985 | .79378818 |
| 2 | .32283706 | .79392172 |
| 3 | .32220849 | .79414472 |
| 4 | .32161627 | .79426548 |
| 5 | .32113063 | .79445089 |
| 6 | .32062962 | .79466517 |
| 7 | .32008596 | .79471569 |
| 8 | .31956349 | **.79483805** |

The loss continued to fall, but validation gains were small and did not change
the scientific regime. Stopping does not assert epoch 8 is the mathematical
convergence point; it records that further same-input convergence is a lower
value experiment than acquiring the missing conditional information.

### Prefix-stratified ceiling

| Horizon | Greedy-prefix match rate | Base, match | Base, mismatch | C64 oracle, match | C64 oracle, mismatch |
|---:|---:|---:|---:|---:|---:|
| H2 | 96.13% | .8112 | .5857 | .9856 | .8718 |
| H3 | 89.37% | .8103 | .5807 | .9830 | .8577 |
| H4 | 81.46% | .8019 | .5913 | .9797 | .8532 |

At H4, `18.54%` of positions are in the mismatched-prefix stratum. The single
draft chain is not a reliable sufficient statistic there. An adaptive tree is
needed to retain alternatives; counterfactual target labels are needed to
teach what target routes those alternatives imply.

### Plateau diagnostics already completed

- A four-row real-data overfit reached about 98% Recall@8. The label plumbing,
  candidate membership, and residual-ranker mechanism can learn corrective
  swaps in a memorization setting.
- On validation, centered learned corrections had about `.881` correlation
  with frozen scores, so the ranker mostly reinforced the incumbent ranking.
- Only about `0.25` of eight selected experts changed per layer-row.
- The observed improvement corresponds to about `0.0124` net additional
  correct experts per layer, versus about `0.85` required for mean `.90`.
- Wider MTP PCA ranks, raw hidden features, direct additive skips, temperature,
  static frequency, and loss/capacity variations produced sub-point gains or
  negative controls; none changed the wrong-prefix information limit.

Detailed chronological evidence remains in:

- `HARP8_ACCURACY_EXPANSION_LEDGER_20260807.md`
- `HARP8_ACCURACY_EXPANSION_STOP_HANDOFF_20260807.md`
- `J_HARP_C64_EXPERIMENT_LOG_20260807.md` (internal evidence log; not included
  in this public branch)
- `J_HARP_C64_V1_PLATEAU_AUDIT_20260807.md` (internal evidence log; not
  included in this public branch)

## Work completed for HARP-RTT

The formal architecture is specified in
`HARP_RTT/HARP_RTT_90_formal_architecture.md`. The following implementation
exists and is tested.

### Data, indexing, and leakage controls

- Rich event index over full target route/state roles, history, exact-token
  metadata, MTP nodes, readiness, and request lineage.
- Lazy, typed dataset that preserves native BF16 bit patterns and explicit
  availability masks.
- Request/lineage-safe split handling; token rows from one request are not
  split across partitions.
- Sealed-test fail-closed checks in capture, indexing, dataset, smoke, probes,
  and training paths.
- Formal inventory gate: Phase 2--5 refuses any index that is not wholly
  adaptive and does not contain a complete six-depth anchor spine.
- Immutable-input checksum inventory and phase/checkpoint lineage.

### Exact objectives and diagnostics

- Exact-cardinality `k=8` log-space dynamic program and exact marginals.
- Exact-set NLL, soft Recall@8, hard top-8 boundary mining, router KL, direct
  query supervision, branch/acceptance loss, and two-round trajectory loss.
- H1--H4 request-macro Recall-first checkpoint selection.
- Auxiliary/shared gradient-ratio auditing and adaptive auxiliary scaling.
- Phase-0 primitives for true-future-state reconstruction, two-layer future
  state ceilings, and prefix-conditional Bayes ceilings.

### Model

- Eight-token, 40-layer route-grid encoder.
- Independent-gate target control/content encoder using exact router-visible
  coordinates plus router-blind and residual channels.
- Parent/sibling-aware adaptive MTP tree encoder.
- Direct H1--H4 axial endpoint decoder with dedicated short/long adapters.
- Frozen router-geometry plus free residual scoring head.
- Exact-8 branch mixture preserving path uncertainty.
- Two rounds of route-trajectory refinement.
- C64 candidate union and rich permutation-equivariant axial set reranker.
- Zero-initialized residual joins that preserve the temperature-1 HARP anchor
  exactly at epoch zero.
- Legacy H5--H8 public-output compatibility.

### Training driver

- Staged Phase 2--5 ownership and learning-rate controls.
- Effective batch size 32 with stable request ordering and deterministic RNG.
- CUDA microbatch autotuning across 8/4/2/1 with explicit AdamW reservation.
- TF32 disabled; router geometry, exact-cardinality math, and legacy PCA
  reconstruction use explicit IEEE-FP32 islands.
- Router-numerics audit is checksum-bound before training.
- Fresh-output and non-overwrite rules for every run.

### Adaptive capture already implemented

The current source can build a runtime-causal adaptive graph with:

- exact committed/argmax H1 root selected after `t` and before target H1;
- maximum-32 parent-before-child H2--H4 native-MTP nodes;
- deep greedy spine plus entropy/confidence-aware best-first width;
- isolated sibling execution without shared mutable KV state;
- node/parent/branch/path IDs, target-position mapping, token IDs, prefix hashes;
- local/path probabilities, entropy, margins, conditioning class, and readiness;
- token embedding, hidden, fused, router input, full MTP router logits,
  selected expert IDs/native BF16 execution weights, and top-64 vocabulary;
- acceptance emitted only after factual execution and marked label-only;
- a separate exact local-top1 H1--H6 spine used solely by the frozen legacy
  HARP bridge. It does not consume adaptive nodes or enter adaptive inputs.

The blocking auditor accepts valid EOS-shortened trees, rejects non-EOS gaps,
requires exact H1/prefix coherence, verifies native BF16 router execution
weights, and verifies the independent H1--H6 anchor spine.

### Production dtype defect found and closed

The first real compatibility smoke from v1.2 failed at
`harp_rtt/model/target.py` because a native BF16 capture tensor reached an FP32
`Linear` while CPU diagnostics had disabled autocast. That failed artifact is
preserved at:

```text
artifact ID: harp_rtt_phase2_legacy_compat_cpu_b1_20260807T140305Z
stderr SHA-256: d94282ae79c3916d8729dd67b2537840330bddc02ec8feeebbf7e4014debd6ca
```

V1.3 makes CPU diagnostics use the same BF16 learned-network autocast policy
as production CUDA. A second latent join was then found and fixed: FP32 exact
base scores and BF16 learned candidate corrections now cross an explicit
dtype boundary before `scatter`.

Verification:

```text
local full HARP-RTT suite:  146 passed, 3 CUDA-only skipped
RunPod snapshot suite:      146 passed, 3 CUDA-only skipped
```

The final real validation-only compatibility smoke passed:

```text
run artifact ID: harp_rtt_phase2_legacy_compat_cpu_b1_20260807T141556Z
persistent artifact ID: harp_rtt/smokes/
                        harp_rtt_phase2_legacy_compat_cpu_b1_20260807T141556Z
report SHA-256: ba667a1bbec49269e567bcb692195f80aee59efa6628c80b382f68112930d1b5
stderr: empty
```

It proved:

- exact H1--H4 epoch-zero equality to the anchor with `torch.equal` and
  maximum absolute error `0.0`;
- matching FP32 shape `[1,4,40,256]` on CPU;
- finite gradients for all `518` Phase-2 trainable parameter tensors and all
  `74,031,501` trainable elements;
- finite effective objective `31.45209885`;
- no optimizer, optimizer step, checkpoint write, source write, promotion, or
  test access;
- total CPU time `92.41 s` (forward `4.91 s`, backward `60.83 s`).

This was a legacy-compatibility smoke only. Its tree has zero adaptive nodes,
and it cannot authorize formal training.

## Frozen artifacts at the pause

### HARP-RTT source v1.3

```text
source snapshot ID: harp_rtt_v1_3_source_20260807T141040Z
persistent artifact ID: harp_rtt/source_snapshots/
                        harp_rtt_v1_3_source_20260807T141040Z
archive SHA-256: a47626dc353a3ccf1fdf8f3a3bfa02b3f35e3a533433cdf3c77f5b473b68a5e0
SOURCE_SHA256SUMS SHA-256: 1f44e085f2f3a8498f6f82d8aada9dcbbcacbbb799f52005da29bc27f033f72f
```

Both inventories pass `sha256sum -c`; the trees and archives are read-only.

### Paused C64 baseline

```text
run artifact ID: harp8_c64_baseline_v1_seed42_b16_lr1e4_e20_20260807T1200Z
persistent artifact ID: harp8_c64_baselines/runs/
                        harp8_c64_baseline_v1_seed42_b16_lr1e4_e20_20260807T1200Z
checkpoint: output/best_through_epoch08.pt
checkpoint SHA-256: 0f39a9922f460ab8460e952f70a38b83bb972b32a839afde30556d92accff438
final launcher log SHA-256: bea92df223d01fa8b3628fa407297481371f5294bca4d2db20899f8c5f3e700e
```

The checkpoint and persistent log are read-only. The Python process, pipeline,
and launcher heartbeat shells are stopped. No GPU compute process remains.

### Static/anchor inputs

```text
target revision: 995ad96eacd98c81ed38be0c5b274b04031597b0
static target artifact ID: static_target_qwen36_35b_a3b_rev995ad96_v1
router geometry SHA-256: a6d1daa092ee028d567d8bebb61c735135b92472ae4785460ae7c7c31f1ad8e0
router audit SHA-256: 0b66ef3662522c107f7a4cb242088741ce7ccc2a4f793b46c8f629a16cc5e405
HARP anchor SHA-256: 410ced95e6082f6a9bfa962d082926ee1ff5c29d203a3869f713a5ee7e58a09d
```

## Current data status

The existing rich index is legacy single-chain data:

```text
index artifact ID: harp_rtt/indexes/rich_v1_20260807T1200Z
39 segments
609 captured sequences
153,506 generated positions
6,140,240 target-layer rows
174,270 six-depth legacy MTP nodes
```

The earlier HARP experiment split used 1,229 training requests, 410 validation
requests, and 409 sealed-test requests from a 2,048-request corpus. These counts
must not be conflated with the 609 sequences currently represented by the rich
HARP-RTT index.

There is no real adaptive HARP-RTT artifact yet. The persistent adaptive capture
directory was empty at the pause. No counterfactual target branch record has
been captured.

## Missing causal information

The adaptive source code already captures multiple causal MTP hypotheses, but
the following information is absent from real data and/or supervision:

1. **No real adaptive graph.** All indexed MTP nodes are from one greedy chain.
2. **No counterfactual target execution.** For an unaccepted branch, there is
   no target router coordinate, target router logit vector, selected target
   experts, target execution weights, or target token likelihood.
3. **Underidentified branch semantics.** The current branch loss can learn
   factual acceptance, but wrong branches are told only that they were not
   realized—not what conditional target route they imply.
4. **No direct branch-prefix CE.** Formal token/path supervision is not yet
   wired into the objective.
5. **Captured vocabulary evidence is underused.** Top-64 tokens and several
   entropy/margin fields are stored by adaptive code but not yet exposed to
   the model adapter.
6. **Timing limitation.** The legacy MTP sources are informational token-end
   upper bounds; their old schedule showed no node ready by the intended early
   prefetch deadline. The new capture must retain truthful readiness events.

## Next experiment: causal-information recapture

### Two-channel contract

Keep runtime inputs and offline teacher labels in separate immutable artifacts.

#### A. Runtime-causal adaptive tree

For every selected source position, use the existing adaptive-v2 contract:

- authoritative prefix through `t` only;
- exact committed H1 root;
- H2--H4 native-MTP graph, default node budget 32;
- selection from local/path probability, entropy, and causal state only;
- no factual H2--H4 token, acceptance, or future target route in expansion;
- every required causal node persisted and marked ready before target H1;
- separate H1--H6 local-top1 anchor spine for exact legacy initialization.

#### B. Train-only counterfactual target-supervision companion

For a deterministic subset of training prefixes, select at most four deepest
paths with distinct H2 children:

1. the highest-path-probability branch;
2. the best descendant under each of the next three highest-probability H2
   alternatives.

Selection is recomputable from a sanitized causal allowlist. It must never use
acceptance, realized H2--H4 tokens, factual future routes, or eventual sequence
outcomes.

Teacher-force the frozen target on:

```text
authoritative_prefix_through_t + selected_branch_path
```

Use isolated full-prefix recomputation. Do not reuse authoritative KV state
beyond `t`, and do not share mutable sibling caches.

Minimum label-only payload per selected node/edge:

- source-capture SHA, request/lineage/sequence/tree/node/parent/path IDs;
- depth/horizon, branch token path, authoritative and conditioning prefix hashes;
- selector version/hash, selection rank/reason, causal-allowlist hash;
- target revision/config/tokenizer hashes;
- target router-visible coordinate `q` `[40,255]` FP32;
- raw target router logits `[40,256]` with native BF16 semantics;
- selected target expert IDs `[40,8]` and BF16 execution weights `[40,8]`;
- forced-edge and cumulative target path log probability;
- target top-64 vocabulary IDs/log-probabilities where feasible;
- `label_only=true`, `adaptive_model_input=false`,
  `runtime_available=false`, `selection_uses_realized_future=false`, and
  `selection_uses_acceptance=false`.

Store full normalized target router input `[40,2048]` only on a small audit
sample. The rank-255 coordinate preserves centered-router information while
controlling corpus size.

### Readiness and leakage rules

- Causal tree readiness and teacher-label readiness are different event types.
- Teacher records never enter `ready_mtp_nodes` and never reuse `source_ready`.
- Counterfactual labels are generated only for preassigned training
  requests/lineages. Validation receives causal trees but no teacher labels.
- The companion is reachable only under dataset `targets`, never `inputs`.
- The auditor recomputes selected path IDs from the causal allowlist and rejects
  any dependence on acceptance/factual-future fields.
- Exact prefix hashes must match full-prefix teacher recomputation.
- Selected IDs must be bit-exact top-8 of captured raw target logits under the
  deployed tie policy.
- Test remains sealed and has no capture authorization switch.

### Pilot ladder

#### Stage A: 32--64 source positions

Purpose: engineering correctness, not a metric claim.

Required pass conditions:

- adaptive blocking auditor passes every tree;
- exact H1 and parent-prefix coherence;
- native BF16 MTP expert-weight agreement;
- complete independent H1--H6 anchor spine;
- deterministic counterfactual path selection reproduces exactly;
- counterfactual top-8 agrees exactly with its raw target logits;
- zero counterfactual fields are reachable through model inputs;
- rich index builds and one dataset batch loads;
- source artifacts remain non-overwritten and checksum-inventoried.

#### Stage B: 2,048 train source positions

Use identical source positions for these controls:

1. six-depth single greedy chain;
2. adaptive node budget 16;
3. adaptive node budget 32;
4. adaptive-32 plus oracle counterfactual route union;
5. adaptive-32 plus a learned MTP-to-target branch translator.

Stratify every result by old greedy-prefix match/mismatch and report H2, H3,
and H4 separately:

- realized token-path coverage;
- true expert coverage of single-chain and adaptive C64 unions;
- oracle counterfactual-route C64 coverage;
- branch `q` reconstruction and target top-8 agreement;
- branch calibration versus target path probability;
- capture throughput, CPU/GPU memory, and ready-time distribution.

Promotion requires a material H3/H4 lift concentrated in the mismatch stratum,
not merely a gain on already-correct prefixes. Before full scaling, require the
adaptive/counterfactual union to trend toward the formal candidate gate:

```text
mean H1--H4 C64 coverage >= 0.985
H4 C64 coverage >= 0.970
```

If the oracle companion cannot lift the wrong-prefix ceiling, change the tree
width/path selector before collecting a large corpus. If oracle labels lift the
ceiling but the learned translator does not, improve branch alignment/losses
rather than adding ranker depth.

#### Stage C: scale only after the information gate

Recommended production corpus:

- at least 10,000 independent lineage-deduplicated requests;
- at least 250,000 usable committed-token source positions;
- multiple domains, languages, context lengths, and decoding strata;
- request/lineage splits plus dedicated calibration and sealed test;
- adaptive tree for all positions;
- deterministic counterfactual supervision on a storage-budgeted train subset.

Only after the recapture passes the information gate should Phase 0 probes and
Phase 2--5 training resume.

## Remaining architecture work after recapture

- Index/dataset support for the label-only counterfactual companion.
- Direct branch-prefix cross-entropy and counterfactual route/query losses.
- Expose the already captured vocabulary entropy/margin evidence through the
  adaptive tree adapter.
- Phase-4 oracle-candidate retention warm-up and anneal to predicted candidates.
- Source-temperature calibration and the remaining formal reranker features.
- Phase-6 three-seed ensemble/distillation and formal ablation runner.
- Real Phase-0 execution on adaptive data.
- Latency validation after the teacher is accurate enough to justify runtime
  distillation.

## Resume rule

Do not resume the epoch-8 C64 run by default. It is a frozen measurement
baseline. Resume it only if a later controlled question specifically requires
the same-input convergence curve.

The active path is:

```text
counterfactual contract -> 32--64-position engineering pilot
-> 2,048-position information probe -> scale gate -> architecture training
```
