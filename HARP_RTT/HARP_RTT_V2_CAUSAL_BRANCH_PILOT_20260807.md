# HARP-RTT v2 Branch-Rich Causal Information Pilot

**Status:** controlling execution plan  
**Lineage:** `agent/harp-rtt-causal-branches-v2` from `c972450`  
**Decision:** do not resume the legacy C64 run. The pause evidence supersedes
the v1.3 resume instruction: candidate generation and within-pool ranking are
separate bottlenecks, and the next test is counterfactual causal branch evidence.

## Execution record

Preflight completed on 2026-08-07 on the supplied
`root@213.192.2.104:40168` pod:

- GPU: RTX PRO 6000 Blackwell Server Edition, 97,887 MiB total and about
  97,252 MiB free at inspection; host RAM 1.5 TiB.
- Frozen BF16 target:
  `/workspace/LLM_prefetch_study/artifacts/j_route_0/hf/Qwen3.6-35B-A3B`.
- Frozen static router geometry SHA-256:
  `a6d1daa092ee028d567d8bebb61c735135b92472ae4785460ae7c7c31f1ad8e0`.
- Frozen split manifest SHA-256:
  `be03308606e3c6ae524f0268bbd8a8b3590a76332f42ae165c203758bafc970b`.
- No live capture or optimizer process was found. The historical C64
  `RUNNING` marker was preserved and renamed
  `STALE_RUNNING_RECONCILED_20260807T160000Z`.
- The repository baseline suite passed (165 tests). A real one-source CUDA
  adaptive capture at
  `/root/harp-rtt-v2-runs/preflight/adaptive_cuda_smoke_20260807T1602Z`
  loaded the full target, captured all 40 router layers and passed
  `CAPTURE_AUDIT_ADAPTIVE.json`. It did not start training or open sealed
  data.

Implementation now includes the deterministic partition freezer; uniform
per-request source offsets; adaptive/fixed controls; divergence-depth selector;
hash-bound label companion; isolated-reference and cloned-prefix-cache replay;
companion/parity auditors; train-only companion loader; layer-conditioned
translator; explicit OTHER; vocabulary pooling; semantic branch features;
anytime masking; coverage-preserving candidate curriculum; counterfactual,
posterior, factual acceptance and swap losses. Stage A remains optimizer-free
and begins only from a committed, test-clean source tree.

## Frozen inputs and inner partitions

The frozen HARP anchor remains reusable; v1.3 teacher checkpoints are incompatible.
Execution requires the 67 GiB BF16 model, frozen geometry, audited corpus and
persistent artifact volume. Use outer-train requests only and freeze three
request/lineage-disjoint partitions from split manifest SHA-256
`be03308606e3c6ae524f0268bbd8a8b3590a76332f42ae165c203758bafc970b`:

- Engineering: 4 requests x 8 positions = 32.
- Diagnostic probe: 128 requests x 16 uniformly selected positions = 2,048.
- Translator fitting: 256 requests x 16 positions = 4,096, half uniform and
  half balanced across causal MTP entropy/margin strata.

Validation, calibration and test remain sealed. The probe is outer-train data
used only for architecture promotion.

## Capture contract

Capture one canonical adaptive-32 tree. Adaptive-16 is its first 16
parent-before-child nodes and must equal an independently built 16-node tree.
Controls are greedy spine; fixed beam-16 (root + five prefixes at each H2--H4);
and fixed beam-32 (root + 11 H2, ten H3 and ten H4 prefixes). Rank beam prefixes
by cumulative MTP probability with stable token/path tie-breaking.

Select four counterfactual paths causally:

1. greedy;
2. best path first diverging at H2;
3. best path matching greedy H2 then first diverging at H3;
4. best path matching greedy H2--H3 then first diverging at H4.

Missing categories remain masked, never backfilled. Selection must not inspect
acceptance, factual continuation or target labels.

Write labels to a separate hash-bound companion. At selected H2--H4 nodes store:

- router coordinate `q [40,255]` FP32;
- raw router logits `z [40,256]` with native BF16 semantics;
- selected IDs `S [40,8]` and execution weights `alpha [40,8]`;
- target edge and cumulative path log probabilities;
- full normalized router input `[40,2048]` only for a deterministic 5% audit.

Bind the companion to the base capture audit, source manifest, target/model and
geometry hashes, split manifest, selector hash and source commit. Mark all fields
label-only, runtime-unavailable and forbidden from inputs. Isolated full-prefix
replay is the Stage-A reference. Promote cloned authoritative-prefix-cache
execution only when Stage-A top-8 IDs are identical and numerical differences
satisfy the frozen router-audit tolerance.

Expose only `targets.counterfactual`:

```text
path_mask                 [B,4]
path_depths               [B,4]
node_local_indices        [B,4,4]
source/target_path_logp   [B,4,4]
query_coordinates         [B,4,4,40,255]
router_logits             [B,4,4,40,256]
selected_ids/weights      [B,4,4,40,8]
valid                     [B,4,4,40]
```

H1 and padding are masked. Sealed-split and ordinary inference loaders cannot
request these tensors.

## Architecture and training contract

Use a layer/horizon-conditioned translator, width 384 and adapter rank 16:

```text
u = Wu(endpoint) + horizon_embedding + layer_embedding
b = Wb(tree_node)
c = RMSNorm(u + b + Wx(u * b))
q = Q0(c) + B_layer(A_layer(c))
```

Preserve FP32 `Kq+b`, layer rank masks and zero-gated anchor attachment. Add
OTHER at index `max_tree_nodes`: observed priors are MTP path probabilities,
OTHER receives residual uncaptured mass, and posterior logits are log-priors plus
zero-initialized corrections. OTHER uses the unchanged anchor score and a
branch-free query. Posterior and acceptance logits are separate.

Wire top-64 vocabulary IDs/log-probabilities through a weighted embedding-bag
pool plus retained/omitted mass, entropy, top-1 probability, top-1/top-2 margin
and top-8 mass. Remove learned sample-local path-ID embeddings. Encode child
rank, depth, first-divergence depth, cumulative path rank, sibling count, parent
confidence and path probability; hashes are audit keys only.

Set adaptive depth exactly four, H1 root at depth one. The H1--H6 compatibility
spine remains separate. Use ancestor-closed anytime prefixes `{1,4,8,16,32}`;
randomly truncate during training and mask nodes before encoding.

C64 is coverage-preserving: epoch zero and the first 10% of generator training
are exactly anchor top-64; guaranteed anchor quota anneals to 48 by 50% and then
freezes. Fill 16 slots from calibrated active branch/geometry/trajectory sources.
Unopened/random sources cannot contribute. Anchor quota 32 is a declared
training-only ablation.

Losses:

- path/depth-balanced counterfactual exact-set NLL + 0.2 query + 0.1 router KL;
- posterior CE against target captured-path probabilities plus OTHER, weight 0.1;
- separate factual branch/OTHER CE and acceptance BCE;
- stable-base-top-8 swap loss, margin 0.125 and weight 0.1; log outside-C64
  misses separately.

Translator pretraining: effective batch 32; BF16 transforms and FP32 geometry;
AdamW `3e-4`, weight decay `0.01`, clip `1.0`, at most 30 epochs, patience five,
seeds 42/43/44. Freeze anchor and reranker. Unfreeze upper endpoint/generator
only after the translator gate.

## Ladder and blocking gates

| Stage | Work | Blocking result |
| --- | --- | --- |
| Preflight | Verify RTX PRO 6000/model/static hashes/processes/source/dependencies; one CUDA source-position smoke; reconcile stale historical marker. | Model fits; full suite and source capture/audit pass. |
| A | Capture 32 outer-train positions with adaptive-16/32 and counterfactual labels; reference/cache, index and one dataset batch; no optimizer. | Exact topology/prefix/tensor/leakage/checksum/cache parity, storage and throughput pass. |
| B1 | Capture 2,048-position probe; greedy/fixed/adaptive path coverage and adaptive-32 oracle routes. | Mean oracle C64 >= .985, H4 >= .970, H4 prefix-mismatch >= .930. |
| B2 | Only after B1, capture 4,096 fit positions, train three seeds, evaluate untouched probe. | Mean-seed recovery G >= .50 on H3/H4 mismatch, paired request-bootstrap lower bound > 0, no negative seed. |
| B3 | Only after B2, train zero-gated factual generator, freeze predicted C64, then rich ranker. | Learned candidate gates match B1; formal-validation mean Recall@8 > .85 before capacity grows. |

Report path occurrence, C64 coverage, Recall/Coverage, query Huber/cosine, router
KL, counterfactual top-8 recall and first-divergence distribution. Stratify by
prefix match/divergence H2--H4, entropy/margin, horizon, layer and block phase.
Verify nesting. Use 1,000 paired complete-request bootstraps, seed 42.

If B1 fails, change tree allocation or test 64 nodes, not a larger ranker. If B1
passes but B2 is noisy, grow fitting data to 8k then 20k before 250k. Open no
formal-validation counterfactual labels and no sealed test. Scale toward 10,000
requests/250,000 positions only after all gates pass.

## Tests and acceptance

Unit tests cover divergence-depth selection, fixed/adaptive determinism,
layer-rotated bases, OTHER mass, vocabulary pooling invariance, depth-four
enforcement, semantic branch identity, candidate schedule, visibility leakage and
swap gradients. Synthetic writer -> auditor -> index -> dataset -> loss tests
include malicious leakage and sealed-split rejection.

Stage A must reconstruct `Kq+b` within frozen tolerance, reproduce authoritative
top-8 tie semantics, contain complete H2--H4 x 40-layer records and return
bit-identical selected IDs for reference and optimized execution. Preserve exact
epoch-zero HARP score/top-8 equality. Before publication run the full repository
suite, compilation, shell checks, wheel build and privacy scan.

Write captures to fresh local directories, audit/checksum them and mirror them
immutably to persistent storage. Speculative-decoding trees are controls because
they optimize token acceptance; HARP promotes policies on route coverage and
keeps direct exact-set prediction primary.

