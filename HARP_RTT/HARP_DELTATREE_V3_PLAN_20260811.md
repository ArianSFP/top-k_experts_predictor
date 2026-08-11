# HARP-DeltaTree v3: anchor-protected direct-set architecture

**Status:** B3.1 completed on the frozen 2,048-row probe. A no-recapture
DeltaTree semantic pilot on the existing B2 fitting/tuning artifacts is the
next execution stage; the 20k capture remains conditional on that pilot.

**Lineage:** branch `agent/harp-deltatree-v3`, current diagnostic/training
contract commit `6b7ed83c4770d36c65b436af2f868b854b2d09f7` plus the documented
no-recapture reuse runner changes.

**Primary objective:** maximize request-macro SlotRecall@8 for H1--H4, with long-generation positions as the primary development protocol and source positions 0--32 as a no-regression guard.

## 1. What looked like a regression, and what actually regressed

The old approximately `0.795` result and the B3 result `0.536401` were not measured on the same position distribution:

- the historical endpoint was selected on the legacy first-33-position protocol;
- the B3 diagnostic probe uniformly sampled long generations;
- on that same long probe, the frozen HARP anchor scored only `0.470046` mean H1--H4 Recall@8;
- B3 scored `0.536401`, an absolute gain of about `0.06636` over its matched baseline.

Therefore B3 did not erase 26 Recall points on a matched evaluation. It exposed a severe position-distribution weakness in the incumbent and improved it modestly.

The genuine B3 failure was information retention. The accepted request-disjoint B1.5 branch oracle reached approximately:

| Metric | B1.5 accepted oracle | B3 learned generator |
| --- | ---: | ---: |
| Mean H2--H4 C64 | 0.982895 | 0.905447 |
| H4 C64 | 0.979030 | 0.885579 |
| H4 mismatch C64 | 0.962117 | 0.823856 |

The tree contained useful route information, but the learned route/posterior/candidate interface discarded too much of it.

## 2. Concrete lessons from v2/B3

The implementation audit found six actionable causes.

1. B2 promoted `geometry + free` branch-semantic selected sets, but B3 candidate generation did not expose the learned free semantic source. It used anchor, geometry, mixture, and trajectory sources. The path that passed semantic supervision was therefore not the path deciding C64 membership.
2. Heterogeneous raw score sources were combined through maxima and fixed quotas without a shared calibrated inclusion-probability meaning.
3. Hard C64 construction received no direct coverage gradient. The factual loss optimized dense scores, while the gate evaluated a discontinuous candidate namespace.
4. Approximately 75.3 million trainable residual-teacher parameters were fit on only 224 fitting requests. Counterfactual semantic retention was downweighted by `0.1` inside factual generator training, making forgetting unsurprising.
5. The factual generator could overwrite a useful incumbent across all experts even when branch evidence was weak.
6. The legacy position feature was trained only on source positions 0--32. The compatibility bridge now clamps that frozen channel, but the replacement needs its own bounded causal position representation rather than extrapolating `p/33` quadratically.

These findings supersede another wide generator or unconstrained reranker run.

## 3. Architectural decision

HARP-DeltaTree is a small direct-set proposer and selective-swap model around the frozen HARP anchor. It has about 4.4 million trainable parameters in the production configuration and is capped at 25 million by test.

### 3.1 Causal inputs

The learned adapter consumes:

- frozen HARP H1--H4 scores;
- eight-token target route history;
- current target router-visible coordinates;
- exact committed H1 token embedding and reconstructed final hidden state;
- all 32 adaptive causal MTP nodes: vocabulary hidden, fused state, router input, router logits, conditioning-token embedding, and reusable structural metadata;
- weighted top-64 vocabulary embeddings plus retained/omitted mass, entropy, top-1 probability, top-1/top-2 margin, and top-8 mass;
- bounded log/bucket/Fourier source-position features with no clamp at position 32.

Counterfactual target routes, target path probabilities, factual prefix hashes, acceptance, and future selected sets remain labels only.

### 3.2 Ancestor-masked adaptive tree

Three parent/ancestor-only attention blocks encode the adaptive-32 tree. A node can attend only to itself and its causal ancestors. Sample-local branch hashes are not embedded; the model receives stable structure such as depth, child rank, first-divergence depth, cumulative path rank, sibling count, path probability, and readiness.

### 3.3 Direct selected-set semantics

For each horizon, target layer, and branch, the model predicts a router query and a layer-conditioned low-rank free residual:

\[
s_{h,l,b}=K_l\hat q_{h,l,b}+b_l+U_l c_{h,l,b}.
\]

Exact-cardinality marginals are computed from these scores. Native target selected IDs remain the primary counterfactual label; router-query regression is auxiliary.

H1 is handled by a separate exact-root path using the committed token and current target state. It is no longer diluted into the H2--H4 alternative-branch problem.

### 3.4 Factual branch posterior and `OTHER`

The branch posterior begins at causal MTP path probabilities plus zero-initialized learned corrections. `OTHER` receives residual uncaptured mass. Its route evidence is the untouched HARP anchor. Prefix-hash equality produces the factual branch label; an absent factual prefix maps to `OTHER`.

### 3.5 Candidate generation

Candidate expansion operates on calibrated exact-set marginal lift over the anchor, never raw cross-source maxima.

- available anchor quotas are 64, 48, 40, and 32;
- epoch zero selects anchor top-64 exactly;
- only experts with strictly positive calibrated lift may displace an anchor fallback;
- if evidence is absent, the policy returns anchor top-64 exactly;
- quota supervision selects the highest-coverage quota, breaking ties toward more anchor protection.

### 3.6 Selective ranker

The ranker is candidate-conditioned and cannot rewrite all 256 experts. It starts at zero correction and zero swap logit, so epoch-zero output is anchor top-8 bit-for-bit. At serving time it may make at most four swaps, and only when:

- the proposed expert is in C64;
- learned confidence exceeds `0.65`;
- its corrected score exceeds the displaced incumbent by the frozen margin.

The exact-set ranker objective masks experts outside C64, matching the deployed decision namespace.

## 4. B3.1 diagnostic before new capture

`runpod/export_harp_rtt_b31_factor_bundle.py` replays the frozen B3 checkpoint
on the existing diagnostic probe and writes a compact, hash-bound bundle of
native/precomputed selected IDs. `runpod/evaluate_harp_rtt_b31_factorial.py`
then evaluates that bundle using selected-set inclusion mass:

\[
m_e=\sum_b p(b)\mathbf1[e\in S_b],
\]

not router softmax. Captured weights remain absolute when `OTHER` is explicit. It reports each horizon for:

- semantic, geometry, free, and other exported route sources;
- learned and target branch posteriors;
- corrected 48/16, 40/24, 32/32, and global calibrated C64 policies.

The evaluator refuses non-outer-train provenance and any bundle that does not explicitly keep validation, calibration, and test sealed. It does not train.

### 4.1 Completed result

The completed bundle contains 2,048 rows and seven route sources and is bound
by SHA-256 `a361fa6683c4fa249793d07ade9e0c6b6ac5954aff2967421cb8e4767662fc55`.
The decisive conditions were:

| Route sets | Posterior | C64 policy | Mean H2--H4 | H4 | Aggregate prefix-mismatch |
| --- | --- | --- | ---: | ---: | ---: |
| Native target IDs | learned | 32/32 | 0.985307 | 0.982077 | 0.965950 |
| Native target IDs | target | 32/32 | 0.985615 | 0.982959 | 0.967397 |
| Learned semantic | learned | 32/32 | 0.890235 | 0.878183 | 0.874615 |
| Learned semantic | target | 40/24 | 0.890557 | 0.879042 | 0.874613 |
| Learned geometry | learned | 40/24 | 0.878359 | 0.861150 | 0.858158 |
| B3 learned branch | learned | 32/32 | 0.828879 | 0.792990 | 0.730067 |

The learned posterior loses only `0.000308` mean H2--H4 versus the target
posterior when both receive authoritative native route sets. Therefore the
captured tree, the learned posterior, and the quota union are sufficient. The
dominant failure is translating causal branch state into the target selected
set, followed by B3 exposing an even weaker learned-branch interface to its
candidate generator. New training must promote on direct selected-set
translation and metric-aligned C64, not posterior loss or raw score loss.

The replay also exposed and fixed two input-contract bugs before new training:

- frozen B3 diagnostics now execute under the same BF16 autocast contract as
  training;
- adaptive dataset reconstruction no longer counts the observed exact H1
  root's historical MTP rank as a branch divergence. Frozen B3 evaluation
  explicitly reconstructs the old feature so its diagnostic remains faithful.

## 5. No-recapture semantic pilot

Before authorizing the 20k capture, reuse the already audited outer-train
artifacts:

- 224 requests / 3,584 positions for fitting;
- 32 request-disjoint requests / 512 positions for tuning;
- 128 request-disjoint requests / 2,048 positions for diagnostic development.

This profile is named `b2_reuse_4096`. It is diagnostic reuse, not a fresh
promotion holdout. The runner requires the original B2 fitting partition and
seed-42 request split, verifies all three request sets are disjoint, requires
the completed B3.1 native-route information gate, and records every binding.
No new target or MTP execution is performed.

Semantic checkpoint selection uses factual quota-32 C64 computed from direct
exact-set branch marginals, not total training loss. Reports additionally
include:

- direct counterfactual route Recall@8 for H2--H4;
- direct factual H1 root Recall@8;
- learned factual branch/`OTHER` accuracy;
- semantic quota-32 C64 for each horizon and mean H2--H4.

Proceed to candidate training only if the learned semantic path retains a
material fraction of the native-route B3.1 ceiling. If it does not, stop and
redesign the direct route translator; do not capture 20k more examples of an
architecture that cannot use the existing labels.

The first microbatch-4 attempt was stopped before completing epoch 1 after it
exposed a mechanical throughput defect: the budget-16 semantic mask was passed
as zero weights to exact-k, but the dynamic program still evaluated all 32
nodes independently at H2, H3, and H4. The runner now gathers only active
supervised node/layer rows before exact-k. A regression test verifies identical
loss values and gradients, including zero gradients for inactive rows. The
stopped attempt produced no selectable checkpoint and is not an experimental
result.

A subsequent full-supervision microbatch-8 throughput trial at source revision
`d8b58f473aa7a88433008ab339841b2762f93686` completed one train/tune epoch on
the same frozen split before being stopped at the start of epoch 2. Its tuning
result was semantic quota-32 C64 `0.828005` over H2--H4 (H2 `0.853412`, H3
`0.829761`, H4 `0.800842`), counterfactual route Recall@8 `0.238343`, and
factual path accuracy `0.265137`. This is an initialization/learning-curve
diagnostic, not a selectable checkpoint. The epoch required about 55 minutes
of training plus 8 minutes of tuning because the audited exact-cardinality
dynamic program launches sequential expert/order kernels for every
microbatch. The immutable artifact is
`/root/harp_delta_v3/reuse/runs/semantic_seed42_d8b58f4_mb8` on the execution
pod and contains its epoch-zero audit, run manifest, and one metrics row.

Revision `681642d5cb247373ba84422d57782a19dc784051` therefore permits microbatches
16 and 32, both exact divisors of the unchanged effective batch 32. The
microbatch-32 retry changes neither examples, objective, optimizer step batch,
nor promotion metric; it only batches the exact-set dynamic program more
efficiently. It produced a directly comparable epoch-1 H2--H4 semantic C64 of
`0.827842`, route Recall@8 `0.238371`, and path accuracy `0.264648`. Those
values match the microbatch-8 diagnostic closely, but training still required
about 48 minutes because the larger tensors did not remove the nested
expert/order launch sequence. The run was stopped at the start of epoch 2 and
is retained at
`/root/harp_delta_v3/reuse/runs/semantic_seed42_681642d_mb32` without a
selectable checkpoint.

Revision `e97e7e100adb32f6bd30bcce9e71eedffc28e876` removes that launch
bottleneck without changing the objective: within each expert recurrence, all
eight independent cardinality orders are updated in one tensor operation.
Strict tests show bit-identical log partitions and inclusion marginals against
both scalar audited references on FP32 and BF16 inputs. Its epoch-1 training
loss and every tune metric were bit-identical to the old-kernel microbatch-32
run, including H2--H4 C64 `0.827842` and route Recall@8 `0.238371`. Training
fell to about 45 minutes, showing that frozen downstream inference had become
the dominant cost. It was stopped at epoch 2 and retained at
`/root/harp_delta_v3/reuse/runs/semantic_seed42_e97e7e1_mb32` without a
selectable checkpoint.

Revision `cb7963a67e77dcefff42e637d5dad0f9e46a7cef` adds a semantic-only
training forward. It shares the exact adapter/tree/root/node/path computation
with full inference but skips frozen exact marginals, candidate construction,
and C64 ranker execution. Tuning, development, epoch-zero protection, and all
later stages still use the complete model. Strict tests assert equality of all
shared semantic tensors, and the complete repository suite passes. This is
the first retry authorized to train through patience and emit a selectable
checkpoint; it uses the same microbatch/effective-batch 32, split, optimizer,
seed, labels, and metric-aligned full-model tune evaluation.

The completed source audit also makes the boundary between semantic and
candidate learning explicit. The candidate subsystem contains only a scalar
marginal-lift calibration and a four-way anchor-quota policy. It cannot alter
the ordering of experts within a predicted branch route, and therefore cannot
repair an underfit branch translator. Candidate optimization is not a valid
substitute for passing the semantic information test.

The immutable all-node companions already contain supervised target routes for
every adaptive-32 node, but the first semantic run uses only the nested
budget-16 mask for both route loss and target-posterior distillation. The
remaining nodes are runtime inputs, not negative examples; folding them into
`OTHER` during distillation can teach a posterior inconsistent with the
all-node runtime mixture. A no-recapture refinement is therefore predeclared:

1. allow the budget-16 semantic run to finish and select on its request-grouped
   tuning set;
2. evaluate its selected checkpoint once on the untouched diagnostic
   development requests;
3. if semantic translation remains the limiting factor, initialize a new,
   immutable semantic run from that checkpoint;
4. supervise exact sets and the target posterior with the nested `all` mask;
5. compare budget-16 and all-node checkpoints on exactly the same development
   data before constructing any candidate optimizer.

`--resume-same-stage --counterfactual-budget all` implements this refinement.
It accepts only a completed semantic checkpoint with the identical
architecture, profile, and partition hash. The original default remains
budget 16. The companion stores nested `4/8/16` masks and represents `all`
with its authoritative `node_mask`; the adapter rejects any other geometry.
Unit tests bind that mapping and verify that all-node probability mass is no
longer incorrectly assigned to `OTHER`.

The same revision removes an evaluation-only throughput artifact. Quota C64
previously accumulated every dense score tensor on the CPU and replayed the
candidate insertion policy rank by rank after each tune pass. The policy is
now expressed as three exact lexicographic groups—guaranteed anchor prefix,
strictly positive non-anchor branch evidence, then unused-anchor fallback—and
semantic membership is counted per GPU microbatch. Randomized parity tests
cover stable ties, zero/negative evidence, overlap, fallback, uniqueness, and
ordering for identical inputs. Unlike the old reporting path, the new path no
longer rounds dense branch and anchor evidence through CPU BF16 before applying
the policy; it retains FP32 evidence. Candidate IDs may therefore differ only
at BF16 rounding boundaries, and this metric-precision boundary is recorded in
the source lineage.

The first `cb7963a` train-through attempt completed 16 logged epochs before an
operational loader-lifecycle fault. Its best tune point was epoch 15: semantic
quota-32 C64 `0.915445` overall, `0.906771` over H2--H4, H4 `0.886487`, and
direct counterfactual route Recall@8 `0.526960`. Epoch 16 was lower on the
selection metric (`0.914032`). Because the driver created a new
`persistent_workers=True` loader for every train and tune pass, abandoned
worker queues accumulated 701 file descriptors; the epoch-17 tune pin-memory
thread hit the soft limit and exited. The driver correctly emitted no
checkpoint before final development evaluation, so this run is retained as a
learning-curve diagnostic and must not initialize another stage.

Revision `ad09fe2e14c971da8ddb0c669f10a51a9af0c580` closes workers with every
epoch-scoped loader and adds a regression test for that lifecycle. The complete
CPU suite passes 271 tests (three CUDA-only skips), and all 40 focused tests,
including the CUDA precision cases, pass on the execution pod. A deterministic
budget-16 recovery uses a fresh output directory, the same seed/data/order,
and unchanged model/optimizer math. Its first epoch reproduced train loss,
tune loss, route recall, and path accuracy bit-for-bit. FP32 C64 was
`0.84613495` versus the old BF16-staged `0.84613953`, an absolute difference of
`4.6e-6`; this confirms training reproduction while making the reporting
precision change explicit.

The next immutable refinement source also uses a semantic-evaluation forward.
It materializes the same anchor, H1-root, node, and posterior-mixture exact-k
marginals as the production forward, but does not execute the frozen candidate
calibrator or C64 ranker. During a semantic stage those downstream modules are
still at their epoch-zero identity, so ordinary candidate Recall/Coverage is
computed directly from stable anchor top-8/top-64. Strict tests show every
shared marginal tensor is bit-identical to the full forward. Full inference is
still required for epoch-zero protection and for candidate/ranker stages.

## 6. Conditional new 20k pilot

`prepare_harp_delta_20k_partition.py` freezes 1,250 group-disjoint outer-train requests with 16 complete pre-EOS H1--H4 positions each:

- 1,000 requests / 16,000 positions for fitting;
- 125 requests / 2,000 positions for tuning;
- 125 requests / 2,000 positions for final development evaluation.

Each request selects a deterministic mixture of:

- four positions from 0--32;
- four log-spaced later positions;
- four causal MTP entropy/margin strata;
- four global-uniform positions;
- deterministic causal backfill if a category overlaps or is unavailable.

The partition builder requires a declared split-manifest SHA, prior request/group exclusions, complete sequence start/end metadata, and a causal scout. It fails if fewer than 1,250 disjoint eligible outer-train requests exist. This means the old 452-request pilot split cannot silently be reused as the new 20k corpus.

## 7. Training ladder

`runpod/train_harp_delta_v3.py` executes exactly one immutable stage per output directory.

| Stage | Trainable subsystem | Primary objective | Initializer |
| --- | --- | --- | --- |
| semantic | raw adapter, tree, direct-set head, posterior | H1 factual exact-set + balanced budget-16 counterfactual exact-set + factual branch/OTHER CE; query and target-posterior auxiliaries | frozen HARP only |
| candidate | marginal-lift gain and adaptive quota | dense exact-set, C64 quota classification, missing-true vs false-anchor pair loss | semantic checkpoint |
| ranker | candidate ranker only | C64-conditioned exact-set, swap pair loss, outsider confidence | candidate checkpoint |
| calibration | candidate gain/quota and swap confidence only | candidate plus ranker calibration | ranker checkpoint |

Every stage uses effective batch 32, AdamW, weight decay `0.01`, gradient clipping `1.0`, request-grouped tune selection, patience 5, and a separate development evaluation. The frozen anchor, token embedding, router geometry, and inactive subsystems cannot receive gradients.

Before the semantic optimizer can be constructed, the runner requires:

- final top-8 equals frozen anchor top-8;
- C64 equals frozen anchor top-64;
- every anchor quota equals 64;
- train/tune/development contain exactly 16,000/2,000/2,000 rows and 16 rows per request;
- every companion joins one-to-one and reports sealed-test access false.

## 8. Evaluation and promotion

Two evaluation columns are mandatory.

1. **Primary long-generation protocol:** the frozen 20k mixed-position development split, reported request-macro by H1--H4 and by position bucket.
2. **Legacy guard:** source positions 0--32 under the historical protocol, evaluated for matched no-regression only after the architecture and checkpoint are frozen.

No result may compare the old first-33 number directly with the long-generation number as if they were the same distribution.

Candidate promotion uses the user-accepted B1.5 operating point rather than retroactively calling `0.982895` a failure:

- mean H2--H4 C64 target at least `0.98` and statistically consistent with the accepted oracle operating point;
- H4 C64 at least `0.97`;
- H4 greedy-prefix-mismatch C64 at least `0.93`;
- no material H1 or short-position regression versus the frozen anchor.

Ranker promotion additionally requires a positive request-bootstrap improvement over the learned generator, bounded swap behavior, and no negative short-position interval. Formal validation remains unopened until the architecture, data selector, checkpoint, and thresholds are frozen. Calibration and sealed test remain closed.

## 9. Execution order

1. Freeze and publish this source lineage.
2. Restore the existing B3 epoch-10 artifact and complete B3.1 without recapture. **Done.**
3. Run only the DeltaTree semantic stage on `b2_reuse_4096`; no candidate or ranker optimizer exists yet.
4. If semantic translation passes, run the candidate stage on the same reuse profile and assess retained native-route lift.
5. Only if reuse is information-limited rather than architecture-limited, build the new outer-train manifest and authorize the 20k capture.
6. Capture to local NVMe first, then audit/checksum and mirror immutably.
7. Run semantic, candidate, ranker, and calibration without skipping initializer boundaries.
8. Evaluate the frozen development set and matched legacy guard.
9. When experimentation and artifact mirroring finish, stop—not terminate—the supplied pod with `runpodctl pod stop` and record the returned desired status.

No pod is started by repository code or by this plan.

## 10. Implemented files

- `harp_rtt/delta.py`: production architecture.
- `harp_rtt/delta_batch.py`: rich-capture adapter and factual prefix/`OTHER` labels.
- `harp_rtt/delta_training.py`: strict ownership and staged losses.
- `harp_rtt/b31.py`: metric-aligned source-factorial primitives.
- `runpod/evaluate_harp_rtt_b31_factorial.py`: fail-closed B3.1 evaluator.
- `runpod/export_harp_rtt_b31_factor_bundle.py`: no-recapture frozen-checkpoint exporter.
- `runpod/transformers_mtp_bridge/prepare_harp_delta_20k_partition.py`: deterministic 20k partition.
- `runpod/train_harp_delta_v3.py`: staged training driver.

The associated focused tests cover anchor equality, positive-lift fallback, maximum swaps, parameter cap, long-position features, ancestor-safe geometry, factual-prefix matching, counterfactual leakage, probability mass with `OTHER`, stage ownership, loss gradients, evaluator provenance, and deterministic mixed-position selection.
