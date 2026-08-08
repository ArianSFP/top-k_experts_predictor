# HARP-RTT B1.5: exact-set branch oracle and adaptive-32 path-budget ceiling

**Decision date:** 2026-08-08
**Lineage:** `agent/harp-rtt-b1.5-exact-set-oracle` from `ac7cd5e18a5fb5bd9468440da03281e2ce863320`
**Scope:** optimizer-free B1.1/B1.5 information experiment; no B2/B3 or H1 training
**Frozen GPU execution commit:** `127558a1511c5d3c03cfdba4e1abfdf84d8fa90a`

**Execution status:** the first exact-set B1.1 report is preserved but its
quota-policy result is superseded pending a zero-capture rerun with strict
positive branch-mass filling. See
[HARP_RTT_B1_5_RESULTS_20260808.md](HARP_RTT_B1_5_RESULTS_20260808.md).

## Decision

PR #3 established that adaptive causal capture and target-label replay work, but its four-path oracle missed every promotion gate. Before training a translator, determine whether the failure is the old router-softmax metric, the four-path label budget, the candidate quota, or adaptive-32 itself.

The blocking sequence is:

1. recompute the immutable four-path B1 artifact using native selected-set inclusion mass and the corrected positive-mass quota fallback;
2. capture every unique adaptive-32 H2--H4 node once;
3. evaluate all-node first and stop immediately if it fails; only after an all-node pass evaluate budgets 16, 8, and the backward-compatible budget 4;
4. freeze the smallest passing selector/candidate policy on the existing 128-request probe;
5. confirm it on the 64 never-selected outer-train requests at 32 positions each;
6. stop before all optimization, regardless of outcome.

If adaptive-32 fails, document an H4-biased adaptive-64 design but do not capture it in this lineage.

## Correct oracle

Router softmax is not selected-expert inclusion probability. For unique captured nodes at horizon/layer:

\[
m_e=\sum_b p(b)\mathbf1[e\in S_b].
\]

Assign uncaptured mass to exact-k anchor marginals:

\[
m^{full}_e=m_e+\left(1-\sum_b p(b)\right)\mu^{anchor}_e.
\]

Use native BF16 target selected IDs as authoritative. Report:

- captured-factual ceiling;
- target-posterior inclusion oracle;
- causal-MTP-prior inclusion oracle;
- global top-64 of full inclusion mass;
- anchor/branch quotas `(48,16)`, `(40,24)`, and `(32,32)`.

The fixed policy preference is 48/16, 40/24, 32/32, then global top-64. H1 remains frozen-anchor-only and is excluded from branch promotion.

The quota constructor inserts anchor top-`K_A`, then distinct experts with
strictly positive branch inclusion mass, and finally fills unused capacity
from the next anchor experts. An all-zero branch source must reproduce anchor
top-64 exactly; zero-mass stable-tie IDs may never displace anchor candidates.

## Node-indexed contract

The v4 companion stores one target record per unique parent-before-child tree node:

```text
node_mask / node_local_indices / parent_local_indices   [B,32]
depth / first_divergence_depth                          [B,32]
budget_node_masks / budget_endpoint_masks               [B,3,32]
budget_realized                                         [B,3]
budget_category_counts                                  [B,3,4]
source/target edge/path logp                             [B,32]
target next-token argmax ID/logp/valid                   [B,32]
query_coordinates                                       [B,32,40,255]
router_logits                                           [B,32,40,256]
selected_ids / selected_weights                         [B,32,40,8]
valid                                                    [B,32,40]
```

H1 and padding remain target-masked. Records are label-only, outer-train-only, hash-bound to the base capture/model/geometry/split/selector/source commit, and inaccessible to inference, validation, calibration, or test loaders. The immutable four-path v2 schema remains readable.

Reference execution independently replays each full prefix. Optimized execution walks a sibling-isolated cloned-parent-cache tree while retaining only the active ancestor stack. Promotion requires bit-identical native selected IDs and the frozen router geometry tolerances.

Target next-token argmax IDs and log probabilities are label-only diagnostics
captured from logits already produced by replay. They enable a
`captured_target_greedy_oracle` that distinguishes missing target-greedy paths
from posterior averaging or candidate construction. They are never runtime inputs.

## Causal path budgets

- Budget 4: the existing selector, bit-for-bit, including missing masked categories.
- Budget 8: `1 greedy + 2 H2 + 2 H3 + 3 H4`.
- Budget 16: `1 + 3 + 5 + 7`.
- All-node: every valid adaptive-32 H2--H4 node.

For 8/16, fill frozen divergence quotas first. Backfill unused slots only with endpoints adding a new H2--H4 node, ordered by deeper endpoint, cumulative MTP probability, token path, then local index. Selection may not read factual continuation, acceptance, target probabilities, or target routes.

Selections are constructed incrementally and must satisfy endpoint and
ancestor-closed node nesting:

\[
E_4\subseteq E_8\subseteq E_{16}\subseteq E_{all},\qquad
N_4\subseteq N_8\subseteq N_{16}\subseteq N_{all}.
\]

## Gates and data use

The target-posterior oracle with the selected policy must satisfy:

\[
\operatorname{Mean}(C_{64,H2:H4})\ge0.985,
\quad C_{64,H4}\ge0.970,
\quad C_{64,H4,\mathrm{mismatch}}\ge0.930.
\]

Report 1,000 complete-request bootstrap replicates with seed 42, but keep the point gates fixed. H1 has a separate future contract and gate `C64_H1 >= 0.98`; B1.5 implements its data/loss/evaluator contract but constructs no H1 optimizer.

Development uses the existing 128-request outer-train probe only. Confirmation uses request-order slice `[388:452]`: all 64 untouched outer-train requests, 32 uniform complete pre-EOS H1--H4 positions each. The 256 translator-fitting requests remain untouched.

## Pre-registered all-node interpretation

An all-node target-posterior failure does not automatically justify a wider tree.

| All-node result | Interpretation | Next action |
| --- | --- | --- |
| Factual ceiling fails | Adaptive-32 lacks enough realized route support | Specify an H4-biased adaptive-64 successor; do not execute it in this PR |
| Factual ceiling passes but captured target mass is low | The tree often contains the factual branch but misses too much probability mass | A wider or differently allocated tree is justified |
| Factual ceiling and path mass are high but target-posterior inclusion fails | Posterior mixing is misaligned with the greedy realized-route objective, or candidate construction is lossy | Inspect target-greedy occurrence, quota loss, and `OTHER`; do not automatically widen |
| Target-posterior passes but MTP-prior fails | Route information exists but causal branch weighting is weak | B2 posterior learning is justified in a separate PR |
| Target-posterior and MTP-prior both pass | Adaptive-32 is sufficient | Find the smallest passing offline label budget, then confirm it disjointly |

## Execution and preservation

The supplied RTX 3090 may run tests and B1.1 because the immutable artifact mirror and frozen anchor fit; it must not load the 67 GiB target. All target replay requires an RTX PRO 6000 supplied and started by the user.

Write every output to a fresh directory. Audit, checksum, and mirror it under a new persistent B1.5 artifact root. Publish only code and small reports. Do not place model/capture payloads in Git. Once B1.5 is documented and mirrored, stop—but never terminate—the experiment pod using `runpodctl`.

Before capture, estimate development, confirmation, index/dataflow,
reference/optimized, and mirror-staging bytes. Require:

\[
B_{free}\ge\max(100\ \mathrm{GiB},\,1.25B_{estimated\ total}).
\]
