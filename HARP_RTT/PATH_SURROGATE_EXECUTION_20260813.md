# HARP-DeltaRoute v4: Large-Corpus Token-Path Route Trajectory

Date: 2026-08-13  
Lineage: `agent/harp-deltaroute-v4`

## Decision

The adaptive-32 tree, learned posterior, and anchor/branch candidate policy are
not the dominant remaining bottlenecks. Native counterfactual routes with the
learned posterior already give approximately 0.985 H2-H4 C64 coverage. The
budget-16 learned translator gives only 0.914467 C64 and 0.557170 branch
Recall@8. Small architectural variants trained on the same 3,584 rows moved
route recall only into the 0.55-0.56 range.

The largest untested lever is supervised route diversity. The audited existing
outer-train corpus contains 113,289 factual source positions over 452 usable
requests, with complete future token IDs and forty-layer target router labels.
This stage reuses those artifacts; it performs no target replay and no capture.

## Data contract

A source-lineage-bound seed-42 pilot samples at most 64 positions from each
eligible outer-train request. The 128 source requests recaptured for adaptive
development are excluded, even though their recapture IDs differ. The exact 32
prior translator-tuning source requests remain the holdout. This leaves 292
training requests (18,688 rows), 32 tuning requests (2,048 rows), and 20,736
rows total. It is written to local NVMe before training.

Each row contains only:

- four current target residual streams projected through the frozen 128-rank
  target PCA;
- the exact current forty-layer router coordinates;
- eight causal route-history steps with native IDs and execution weights;
- the factual H1-H4 token path as a training-only teacher input;
- factual H1-H4 centered router logits and native selected IDs as labels.

At serving and adaptive-tree evaluation, the token path is reconstructed only
from the causal MTP tree. Counterfactual labels, future target states, formal
validation, calibration, and sealed test are never model inputs.

The cache auditor verifies array geometry/dtypes, finite values, expert/token
ranges, request-disjoint row splits, unique request-position pairs, native top-8
agreement from stored logits, source hashes, and sealed-split flags. Training
fails closed without this audit.

## Architecture

The model has approximately ten million trainable parameters and preserves the
high-value inputs until layer/horizon conditioning:

1. Each of four current-state PCA channels has its own normalization and
   projection, with a learned per-layer mixture.
2. Native execution-weighted route history is embedded by target layer with a
   learned horizon-specific lag mixture.
3. Frozen target token embeddings are scanned recurrently over the proposed
   H1-H4 path.
4. A causally horizon-masked axial encoder integrates the 160 horizon/layer
   cells. Later proposed tokens cannot affect earlier-horizon forecasts.
5. A free rollout scans the forty target layers. At each layer it predicts a
   frozen-geometry router query plus a low-rank residual score, selects exact
   top eight, constructs an execution-style expert-effect message, and feeds
   that message and the predicted query to the next layer.
6. The route message uses a hard selected-expert forward pass and a dense
   straight-through gradient. Training therefore exposes the transition to its
   own route decisions while retaining useful gradient flow.

The factual objective is exact-set NLL plus centered-logit Huber and a top-8
boundary loss. Horizon weights `[1.0, 1.0, 1.25, 1.5]` emphasize the known
H3/H4 bottleneck without dropping H1/H2.

## Execution and decision ladder

1. Build and audit the 20,736-row source-lineage-disjoint local-NVMe cache.
2. Run one seed (42), with effective batch 64, AdamW at `3e-4`, BF16 transforms,
   FP32 router geometry, maximum ten epochs, and patience three. Small
   diagnostics deliberately do not consume multiple seeds.
3. Select by request-disjoint tuning H2-H4 native route Recall@8 and retain all
   per-horizon results.
4. Evaluate the frozen checkpoint on the already-open 2,048-row adaptive-tree
   development probe. Reconstruct each node prefix solely from node token ID,
   parent, depth, and mask. Report both frozen budget-16 deployment and all-node
   diagnostic C64, H4, H4-prefix-mismatch, and counterfactual route recall.
5. Only a large zero-shot tree gain justifies counterfactual fine-tuning. The
   final ranker remains closed until learned C64 reaches H2-H4 >= 0.98, H4 >=
   0.97, and H4 mismatch >= 0.93.

## Implemented invariants before optimization

- later path tokens cannot change earlier-horizon scores;
- changing a layer's expert-effect table cannot change that layer but changes
  downstream layers;
- the expert-effect feedback path receives gradient;
- serving model APIs expose no target-label tensor;
- tree prefixes are deterministic, ancestor ordered, depth consistent, and
  parent-before-child;
- local cache split is request-disjoint and training refuses an unaudited
  cache;
- no persistent DataLoader workers retain segment file descriptors;
- a separate optimizer-free epoch-zero artifact is written before AdamW is
  constructed.

## Result

Completed and rejected. The 20,736-row cache passed its audit and seed 42
completed ten epochs. Best tuning H2-H4 route Recall@8 was `0.561812`.

On the untouched adaptive-tree development probe, budget-16 H2-H4 C64 reached
`0.928656` versus `0.914519` for the selected parent, a gain of only `0.014136`.
H4 reached `0.911301` and H4 prefix-mismatch reached `0.869460`; all failed the
`0.98/0.97/0.93` candidate gates. Native route recall on the tree was lower than
the parent (`0.515305` versus `0.560171`). No counterfactual fine-tuning or
ranker training was authorized.
