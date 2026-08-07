# J-HARP-C64 v2 layer-aware reranker — implementation protocol

Date: 2026-08-07
Status: implemented locally; no remote job launched; sealed test data untouched

## Purpose

This experimental v2 ranker addresses the concrete information-path
bottlenecks found in the v1 implementation while preserving v1 for the matched
J/raw/random attribution study.

The trainable interface remains a fixed c64 candidate reranker for H1--H4.
The public composite interface still returns predictions through H8, with the
frozen HARP scores copied exactly at H5--H8.

## Differences from v1

1. `generator_context.f16` is mandatory. It is read from each immutable HARP
   candidate pool at `[row,horizon,target_layer,model_width]`. A pool built
   with `store_context=false` is rejected.
2. MTP hidden states and MTP router logits have separate projections,
   scale/statistic side channels, depth embeddings, Transformer stacks,
   missing-source tokens, and cross-attention modules. They first meet at the
   explicit three-source fusion (`local`, `MTP hidden`, `MTP router`).
3. Each MTP query is `[horizon,target_layer]` specific. It contains local
   target J-history context, the corresponding frozen HARP generator context,
   horizon identity, and target-layer identity. No horizon-only MTP vector is
   broadcast to all 40 layers.
4. Router-coordinate queries use layer-specific low-rank maps:

   ```text
   q[b,h,l] = SiLU(c[b,h,l] @ A[l]) @ B[l] + bias[l]
   ```

   This respects the 40 independently oriented router-SVD coordinate systems.
   Frozen router keys remain non-trainable. Candidate-key-to-hidden adapters
   are layer-specific as well.
5. The final candidate correction remains residual and exactly zero at
   initialization. Epoch-zero candidate scores equal the frozen HARP scores
   bit-for-bit.

## Causal and leakage contract

The decision profile remains:

```text
token_end_informational_upper_bound
```

The current MTP evidence is informationally causal from the committed prefix,
but it was not ready by the captured deployment deadlines. Accordingly:

```text
mtp_timing_causal = false
acceptance_labels_used = false
```

The loader and model explicitly reject known label-only fields, including
prefix-match/acceptance labels, future target token IDs, first rejection depth,
future router labels, and the stored log probability selected at the future
committed target token. `target_membership` and teacher candidate scores remain
in the training batch for loss calculation but are never accessed by the model
forward path.

## Artifact and schema contract

No capture or candidate-pool format changed. v2 reuses:

- `harp8_candidate_pool_v1`, with `store_context=true` and a declared,
  hash-verified `generator_context.f16` array;
- the v1 aligned request-ID join for J/target/MTP evidence;
- the strict target/MTP/feature/PCA/precision lineage audit;
- train/validation request-disjointness and sealed-test prohibition; and
- the existing c64 validation coverage gate.

New implementation schemas are:

```text
harp8_jspace_v2_aligned_candidate_data_v1
harp8_jspace_v2_candidate_reranker_v1
harp8_jspace_v2_reranker_training_v1
harp8_jspace_v2_reranker_manifest_v1
```

The checkpoint is self-contained: it stores the model configuration, frozen
router-key buffer, trainable state, exact input provenance, and composite
inference contract.

## CLI

Entry point:

```text
harp8-train-jspace-v2-reranker
```

The CLI deliberately requires:

```text
--enable-experimental-v2
```

This prevents an unmatched v2 run from being mistaken for a continuation of
the preregistered v1 attribution controls. Other scientific gates are the same
as v1:

- strict train/validation manifests;
- no test pool interface;
- full router rank unless `--allow-router-rank-ablation` is explicit;
- a J-Lens precision audit for J features;
- mandatory context-enabled pools;
- exactly four active horizons; and
- explicit provisional attribution when the representative precision audit
  lacks a Recall@8 probe.

New capacity flags are:

```text
--router-query-rank
--mtp-hidden-blocks
--mtp-router-blocks
```

The remaining optimizer, I/O, PCA, router-key, and resume flags mirror v1.

## Files

```text
harp8/jspace_v2_data.py
harp8/jspace_v2_reranker.py
harp8/train_jspace_v2_reranker.py
tests/test_harp8_jspace_v2_data.py
tests/test_harp8_jspace_v2_reranker.py
tests/test_harp8_jspace_v2_training.py
```

## Acceptance tests

Focused tests cover:

- mandatory generator-context ingestion and H1--H4 prefix reads;
- exact epoch-zero base-score equality;
- candidate permutation equivariance after a nonzero correction is enabled;
- separate MTP hidden/router memories and all-MTP-missing behavior;
- layer-specific router-query isolation;
- output and input shape failures;
- label-only input rejection;
- exact H5--H8 frozen composite passthrough;
- data-derived configuration and explicit CLI gate; and
- a tiny end-to-end training/checkpoint/composite-inference round trip.

This implementation is a candidate-ranker experiment, not yet the full
256-expert router forecaster recommended as the next production architecture.
It cannot recover experts absent from c64, whose measured mean H1--H4 oracle
ceiling is approximately 0.9741.
