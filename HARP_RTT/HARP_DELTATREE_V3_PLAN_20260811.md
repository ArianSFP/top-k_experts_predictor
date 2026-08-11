# HARP-DeltaTree v3: anchor-protected direct-set architecture

**Status:** implemented locally; real B3.1 replay, 20k capture, and optimization have not started because no running GPU pod was supplied for this stage.

**Lineage:** branch `agent/harp-deltatree-v3`, based on `cdb403dd858faa35981f2c088718abbfcea3a577`.

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

## 5. New 20k pilot

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

## 6. Training ladder

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

## 7. Evaluation and promotion

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

## 8. Execution order

1. Freeze and publish this source lineage.
2. On a user-supplied running pod, restore the existing B3 epoch-10 artifact and export/run B3.1; do not recapture for this diagnostic.
3. Build the new outer-train candidate manifest and causal scout, then freeze the 20k partition.
4. Capture adaptive-32 base data and node-indexed counterfactual labels to local NVMe first; audit and checksum before immutable network mirroring.
5. Run the staged trainer in semantic, candidate, ranker, calibration order. Never skip an initializer boundary.
6. Evaluate the frozen development set and then the matched legacy guard.
7. Scale only if learned semantic and candidate paths retain the accepted branch-information lift.
8. When experimentation and artifact mirroring finish, stop—not terminate—the supplied pod with `runpodctl pod stop` and record the returned desired status.

No pod is started by repository code or by this plan.

## 9. Implemented files

- `harp_rtt/delta.py`: production architecture.
- `harp_rtt/delta_batch.py`: rich-capture adapter and factual prefix/`OTHER` labels.
- `harp_rtt/delta_training.py`: strict ownership and staged losses.
- `harp_rtt/b31.py`: metric-aligned source-factorial primitives.
- `runpod/evaluate_harp_rtt_b31_factorial.py`: fail-closed B3.1 evaluator.
- `runpod/export_harp_rtt_b31_factor_bundle.py`: no-recapture frozen-checkpoint exporter.
- `runpod/transformers_mtp_bridge/prepare_harp_delta_20k_partition.py`: deterministic 20k partition.
- `runpod/train_harp_delta_v3.py`: staged training driver.

The associated focused tests cover anchor equality, positive-lift fallback, maximum swaps, parameter cap, long-position features, ancestor-safe geometry, factual-prefix matching, counterfactual leakage, probability mass with `OTHER`, stage ownership, loss gradients, evaluator provenance, and deterministic mixed-position selection.
