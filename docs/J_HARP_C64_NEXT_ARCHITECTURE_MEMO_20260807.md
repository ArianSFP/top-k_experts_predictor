# J-HARP-C64 next-architecture memo — 2026-08-07

## Purpose and scope

This memo records the highest-leverage next experiments if the matched dense
J-Lens rank-512 reranker does not materially beat the frozen HARP endpoint or
approach the 0.90 mean H1--H4 Recall@8 objective.

The review was read-only. No training was launched, no test partition was
opened, and no production code or trace artifact was changed. The conclusions
come from the current HARP/J-HARP implementation, immutable manifests, prior
validation reports, and a new validation-only stratification of the existing
c64 pool by MTP greedy-prefix agreement.

## Executive conclusion

A J-only rank-512 candidate reranker is unlikely, by itself, to bridge the
roughly 0.793 to 0.90 gap. Dense J transport cannot create information that is
absent from the residual; its plausible benefit is better cross-layer
conditioning. The current implementation also leaves several more direct
signals underused:

1. it discards the frozen HARP generator context even though the candidate
   pools store it;
2. it pools MTP evidence with horizon-only queries and broadcasts the same MTP
   summary to all 40 target layers;
3. it uses one shared router-query map across 40 independently oriented SVD
   coordinate systems;
4. it gives the geometric router score no direct full-router supervision;
5. it does not ingest the causal MTP draft-token IDs already present in the
   capture; and
6. it has no explicit MTP-prefix reliability head or branch model.

The recommended pivot, after finishing the preregistered J/raw controls, is a
layer-specific full-router forecaster. It should map depth-aligned MTP states,
raw/J target context, and the frozen HARP context directly to all 256 future
router scores. A candidate union and set reranker should follow that full-router
prediction, rather than serving as the only learned future-route head.

## Fixed empirical anchors

The best frozen HARP checkpoint used for the current candidate pool has
approximately the following validation request-macro Recall@8:

| Horizon | Recall@8 |
|---:|---:|
| H1 | 0.8217 |
| H2 | 0.8025 |
| H3 | 0.7861 |
| H4 | 0.7622 |
| Mean H1--H4 | **0.7931** |

The c64 pool oracle coverage is:

| Horizon | c64 oracle coverage |
|---:|---:|
| H1 | 0.9893 |
| H2 | 0.9812 |
| H3 | 0.9696 |
| H4 | 0.9563 |
| Mean H1--H4 | **0.9741** |

Consequently, 0.90 mean H1--H4 Recall@8 requires recovering approximately:

```text
0.90 / 0.9741 = 92.4%
```

of the positives available inside the c64 pool. H4 alone requires:

```text
0.90 / 0.9563 = 94.1%
```

conditional recovery. The current approximate recovery is only 81.4% averaged
over H1--H4 and 79.8% at H4. Candidate ranking is therefore the immediate
bottleneck, although MTP branch errors also reduce candidate coverage on the
hardest rows.

Increasing candidate width alone cannot bridge the full gap. Even replacing
c64 by an oracle-complete pool would add at most 2.6 percentage points to the
mean ceiling and 4.4 points at H4, whereas the desired mean gain from the
current endpoint is about 10.7 points.

## New MTP-prefix discriminator

### Result

The existing validation c64 pool was stratified by whether the single greedy
MTP branch matched the committed continuation through the depth required to
construct the corresponding future-state estimate.

| Horizon | Greedy prefix match | Base R@8, match | Base R@8, mismatch | c64 ceiling, match | c64 ceiling, mismatch |
|---:|---:|---:|---:|---:|---:|
| H1 | 100.00% by construction | 0.8216 | -- | 0.9893 | -- |
| H2 | 96.13% | 0.8112 | 0.5857 | 0.9856 | 0.8718 |
| H3 | 89.37% | 0.8103 | 0.5807 | 0.9830 | 0.8577 |
| H4 | 81.46% | 0.8019 | 0.5913 | 0.9797 | 0.8532 |

The complete row counts were:

| Horizon | Valid rows | Match rows | Mismatch rows |
|---:|---:|---:|---:|
| H1 | 13,530 | 13,530 | 0 |
| H2 | 13,120 | 12,612 | 508 |
| H3 | 12,710 | 11,359 | 1,351 |
| H4 | 12,300 | 10,019 | 2,281 |

The same full-corpus MTP arrays give greedy token accuracy / cumulative prefix
accuracy for drafted positions offset by two through seven tokens:

| Draft predicts offset | Valid rows | Token top-1 accuracy | Prefix accuracy through draft |
|---:|---:|---:|---:|
| +2 | 67,584 | 0.9599 | 0.9599 |
| +3 | 65,536 | 0.9013 | 0.8920 |
| +4 | 63,488 | 0.8346 | 0.8106 |
| +5 | 61,440 | 0.7795 | 0.7372 |
| +6 | 59,392 | 0.7231 | 0.6646 |
| +7 | 57,344 | 0.6425 | 0.5634 |

The H2--H4 validation rates differ slightly from the full-corpus rates because
they use only valid rows in the frozen validation split.

### Interpretation

The single-branch MTP representation is highly informative when its drafted
prefix is correct, but both ranking and candidate coverage collapse on prefix
mismatches. This is especially important at H4, where 18.54% of validation
rows are mismatches.

If mismatch-row H4 recall remained at its current 0.5913, match rows would need
approximately:

```text
(0.90 - 0.1854 * 0.5913) / 0.8146 = 0.970
```

Recall@8 to reach 0.90 overall. That is nearly the entire 0.9797 match-row c64
oracle. If mismatch rows instead approached their existing 0.8532 c64 oracle,
match rows would need approximately 0.911. Thus the target requires progress
on both correct-branch ranking and branch-mismatch handling; optimizing only
the already-correct MTP rows is not sufficient.

### Exact provenance and method

This was a read-only, validation-only calculation using the staged local-NVMe
copies on the RunPod:

```text
/root/harp8_accuracy_expansion/pools/temp1_epoch5_c64_v3_validation
/root/j_harp_c64/inputs/mtp/mtp_draft_token_ids.npy
/root/j_harp_c64/inputs/mtp/mtp_draft_target_token_ids.npy
/root/j_harp_c64/inputs/compact/requests.jsonl
```

The validation pool was aligned to the capture by:

1. reading `request_id` and `within` from the pool `metadata.json`;
2. mapping `request_id` to its immutable position in `requests.jsonl`; and
3. calculating `capture_row = request_number * 34 + within`.

Only rows enabled by the pool's `valid_future.u1` mask were included. For
horizon `h > 1`, prefix match required equality between MTP draft IDs and
committed target IDs for draft depths `0..h-2`. H1 requires no speculative
draft prefix because its MTP state consumes the already-sampled exact next
token.

The candidate pool is stored in descending frozen-generator score order.
Base Recall@8 was therefore calculated by summing `target_membership` over the
first eight candidates and dividing by `8 * 40`. c64 coverage summed membership
over all 64 candidates with the same denominator. All requests have the same
number of valid rows at a given horizon, so the unstratified row mean and
complete-request macro mean coincide; the match/mismatch rows are reported as
descriptive conditional strata.

No future token identity, prefix-match flag, or target log probability was
presented to a model. These values were used only after prediction as audit
labels.

## Leakage-safe feature contract

### Causal under `token_end_informational_upper_bound`

At the declared token-end decision point, the following are informationally
causal:

- current and prior committed target routes and states;
- the exact next-token ID already sampled from the current target logits;
- MTP hidden states and MTP router logits produced from that committed prefix;
- `mtp_draft_token_ids.npy`, because these are the MTP head's own greedy
  choices;
- a chosen draft token's own normalized log probability;
- causal vocabulary entropy, top-1/top-2 margin, top-8 probability mass, and
  cumulative draft-path log probability;
- branch IDs, parentage, and branch probabilities; and
- frozen HARP candidate scores, source gates, copy gates, and generator
  context.

The current depth-chain artifact does not store the chosen draft token's own
log probability or the full causal vocabulary confidence summaries. Those
must be added in a future capture if reliability conditioning is promoted.

### Label/evaluation only

The following must never enter the predictor as observed features:

- `mtp_draft_target_token_ids.npy`;
- actual MTP-prefix equality or engine acceptance;
- first-rejection depth;
- future committed target routes or token IDs beyond the already-sampled H1
  token;
- future cache outcomes; and
- `mtp_draft_target_logprobs.npy` as currently stored.

The final item is important: the stored value is the MTP distribution's log
probability at the actual future committed target ID. Selecting that vocabulary
coordinate requires knowing the future label. It is valid for evaluation or
auxiliary supervision, not as an input. By contrast, the log probability of
the MTP-chosen top-1 token is causal, but it was not preserved in this artifact.

Prefix match and acceptance may supervise an auxiliary reliability head. The
head's *predicted* reliability is causal at inference; the observed label is
not.

This remains an informational upper-bound study. The earlier timing audit
found that none of the structurally valid MTP sources were ready by the
recorded early-router or post-expert deadlines. Any deployment claim requires
a new schedule and direct ready-before-decision validation.

## Current implementation bottlenecks

### 1. Frozen generator context is available but dropped

Both c64 pool manifests declare `store_context=true` and contain:

```text
generator_context.f16 [rows, 8, 40, 384]
```

`CandidatePool` can load it, but `train_jspace_reranker.py` calls aligned-data
batches with `include_context=False` for the epoch-zero assertion and every
training microbatch. `JSpaceCandidateReranker` consequently receives frozen
base scores, source/copy gates, and candidate features, but not the HARP latent
that generated those scores.

The legacy c64 reranker did consume generator context and improved only about
0.0008 after two partial epochs, so context alone is not expected to solve the
problem. It should nevertheless be included as a zero-gated skip so the new
model can learn corrections conditional on the frozen generator's richer
route/MTP state instead of reconstructing it.

### 2. MTP attention is horizon-specific but not target-layer-specific

The MTP encoder constructs one token per draft depth. Cross-attention queries
are made only from horizon embeddings, producing `[B,H,D]`. This value is then
broadcast across 40 target layers before adding J context and a layer
embedding.

The model can transform the broadcast summary differently after the layer
embedding is added, but it cannot choose different MTP depths or source
mixtures for different target layers. The original HARP generator did use a
horizon-by-layer query. The next model should query MTP tokens using the
current target-layer context plus both horizon and layer identity.

### 3. MTP hidden and router evidence are merged too early

Projected MTP hidden state and projected MTP router logits are added before the
node fusion block. This permits destructive interference and prevents clean
source ablations. Keep them as separate source tokens, or concatenate them
with explicit source embeddings and a learned gate. Preserve hidden-state RMS
and router scale/margin statistics as side channels rather than removing them
through normalization.

Depth-specific low-rank adapters are also justified: recurrent depths 1--6 do
not have identical error distributions, and additive depth embeddings cannot
fully replace a depth-conditioned linear map.

### 4. The full-rank router coordinates are independently oriented by layer

For layer `l`, the router factorization is:

```text
W_l = U_l Sigma_l V_l.T
K_l = U_l Sigma_l
q_l = V_l.T a_l
```

Each `V_l` is a distinct basis. The current model uses one shared linear
`router_query` to emit a 256-dimensional query for every target layer. A layer
embedding and shared nonlinear block can approximate layer conditioning, but
they are an inefficient way to learn 40 independent coordinate maps.

Use one of:

- a dense layer-specific query head;
- a shared query plus layer-specific low-rank adapters or FiLM;
- direct prediction in the common 2048-dimensional hidden coordinate using
  the raw router rows; or
- an unconstrained layer-specific 256-logit head as a mandatory control.

At accuracy-first scale, a layer-specific mapping is preferable even if it
adds tens of millions of parameters.

### 5. Router geometry receives no direct future-router loss

The current router-key dot product is embedded as one candidate feature. It is
not directly added to the final score and is not required to reproduce future
router logits. The network can ignore it completely.

The capture already supplies future full router logits as offline labels. Add
a direct auxiliary loss on all 256 experts:

- forward router KL;
- top-8 versus ranks 9--16 boundary ranking; and
- optionally smooth-L1 on centered logits or layer-standardized router
  coordinates.

The predicted full-router score should also participate directly in candidate
ranking and candidate-union generation.

### 6. Causal token and reliability information is unused

The MTP depth capture contains causal greedy draft-token IDs, but
`AlignedJCandidateData` loads only MTP hidden states and MTP router logits. A
frozen Qwen token embedding followed by a learned projection is the preferred
representation; a learned embedding is a larger fallback.

For H1, the exact next token is already sampled and is causal at token end.
For H2--H4, the corresponding MTP path token IDs are causal hypotheses. Their
explicit embeddings make branch identity and semantic divergence easier to
model than relying on hidden states alone.

An auxiliary prefix-reliability head should predict whether the draft remains
correct through each horizon. Its labels may use stored draft/target equality,
but only its predictions may gate MTP sources at inference.

### 7. J-only preprocessing can suppress route-relevant directions

Dense J is a frozen layer-specific linear map. It can improve cross-layer
conditioning, but it can also attenuate information that is predictive of a
local router and irrelevant to final-token semantics. Shared rank-512 PCA is
unsupervised: retaining 93.2% of transported variance does not prove that it
retains 93.2% of future-route information.

The existing J-Lens correspondence audit primarily demonstrates agreement
between the 100-prompt and 1,000-prompt lenses. It does not establish that
transported states are optimal sufficient statistics for future routing.

Use separate raw and J streams. A gated dual-stream model can exploit J's
cross-layer alignment without discarding raw residual information. A
supervised low-rank projection or route-tuned LoRA on J is a later experiment,
after the frozen-J attribution result is known.

### 8. The ranking loss may spend gradient on calibration rather than swaps

The legacy c64 loss declined from approximately 0.3250 to 0.3228 while mean
Recall@8 improved by only 0.00014 between epochs 1 and 2. Although that run was
partial, the pattern warns that absolute BCE/KL improvement need not change
the top-eight boundary.

Track at least:

- within-row delta standard deviation;
- number of candidate rank swaps;
- false-positive and false-negative swaps at the rank-8 boundary;
- training versus validation conditional recovery; and
- Recall@8 by base margin and MTP-prefix agreement.

Candidate-centered deltas, temperature-1 KL, and a rank-8 structured loss are
more directly aligned than an absolute balanced BCE alone.

## Prioritized next-model sequence

### Stage 0 — finish the matched J attribution

Complete rank-512 runs for:

1. raw residual;
2. one shared signed-permutation control;
3. independent per-layer signed-permutation control; and
4. dense J-Lens.

Raw and shared-orthogonal preprocessing contain the same information and
preserve cross-layer orientation up to one global orthogonal transform. They
should be statistically equivalent. A material difference is evidence of
optimizer variance, inadequate convergence, or a preprocessing error.

Promote a J-specific claim only if the paired complete-request bootstrap
against raw has a positive lower confidence bound and the gain is practically
material. If J is tied with or below raw, keep J as an optional dual stream
rather than the primary representation.

### Stage 1 — run cheap bottleneck probes

Before another large Transformer, perform these leakage-safe probes on the
same split and pool:

1. **Real-row overfit:** overfit 128--256 aligned rows. Training Recall@8
   should approach the c64 oracle. Failure diagnoses implementation, objective,
   or optimization rather than generalization.
2. **Base-only calibration control:** candidate scalar correction without
   J/MTP.
3. **Frozen generator-context control:** add only `generator_context`.
4. **Depth-diagonal MTP probe:** for H1--H4, map the matching MTP hidden depth
   directly to future router logits.
5. **Shared versus layer-specific heads:** hold every input fixed and vary only
   the output map.
6. **Raw versus J local state:** same-layer current state only.
7. **All-layer raw versus J:** introduce axial context.
8. **MTP state versus MTP router versus both:** retain source separation.
9. **Draft-token embedding:** add only causal chosen-token IDs.

A request-split ridge/logistic or shallow residual head is sufficient. These
experiments answer whether information exists before confounding it with a
33-million-parameter set model.

Expected discriminators:

- a clear layer-specific-head gain identifies the shared SVD-coordinate map
  as a bottleneck;
- a strong depth-diagonal MTP gain concentrated on prefix-match rows identifies
  fusion/alignment as the bottleneck;
- high training recovery but weak validation means more independent requests
  are needed;
- low training recovery means widening the dataset will not repair the current
  model/loss; and
- no conditional J gain over raw means J should not remain the principal
  research direction.

### Stage 2 — direct full-router MTP/J forecaster

Build the next production candidate as a residual full-router predictor.

For each horizon and target layer:

1. encode current and prior target routes;
2. encode raw and J target-state grids in separate streams;
3. load frozen HARP generator context;
4. keep MTP hidden, router, token, and reliability evidence as separate
   sources;
5. construct a horizon-by-layer query;
6. attend over MTP depth/source tokens with explicit depth-to-horizon bias;
7. predict a layer-specific future router query or direct 256-logit vector;
8. add the correction residually to frozen HARP full-router scores; and
9. supervise the complete future distribution and top-8 boundary.

For the first high-signal control, use only the matching diagonal MTP depth:

```text
H1 <- MTP state depth 1
H2 <- MTP state depth 2
H3 <- MTP state depth 3
H4 <- MTP state depth 4
```

Then compare learned attention over all six depths. Direct, non-recursive
H1--H8 heads remain preferable to rolling a predicted intermediate route.

This model produces all 256 expert scores, so it can discover candidates
absent from the frozen HARP c64 list. It also better matches the user's desired
interface: top-eight predictions at each horizon that feed both cache retention
and prefetch scheduling.

### Stage 3 — candidate union and conditional reranker

Create the candidate namespace as the union of:

- frozen HARP top candidates;
- direct MTP/router-forecaster top candidates;
- current-route persistence candidates; and
- optional transition-prior candidates.

Deduplicate by layer-specific expert object. Score the union with predicted
full-router logits, frozen HARP scores, transition/session evidence, raw/J
context, and reliability. A permutation-equivariant set head remains useful at
this stage, but it should refine explicit full-router predictions rather than
serve as their substitute.

Keep c64 for the first matched experiment. Move to c96/c128 only after either:

- c64 conditional recovery approaches 0.95 and the remaining error is visibly
  coverage-bound; or
- a new multi-branch forecaster supplies meaningful new candidates outside the
  original pool.

### Stage 4 — reliability-gated and multi-branch MTP

Using existing data, train a causal auxiliary prefix-reliability head from:

- MTP hidden and router features;
- draft token IDs;
- depth; and
- causal target-route/J context.

Supervise with prefix equality as a label, and gate or mix the MTP expert with
a route/J fallback using only predicted reliability.

For a new capture, retain one primary greedy chain plus a sampled or production
beam of two to four branches through H4. Store chosen-token log probabilities,
vocabulary confidence summaries, branch probability, and parentage. Predict a
full target-router distribution per branch, marginalize or gate with branch
probabilities, and union their candidate sets.

The prefix audit makes this the most credible path for improving H3/H4. It
directly attacks the rows where current c64 coverage is only about 0.85.

### Stage 5 — metric-aligned loss and optimization

Once the information path is corrected, run a bounded objective study:

- teacher KL temperature 1 versus 2;
- remove or reduce balanced BCE;
- pair true top-8 against true ranks 9--16;
- add dynamically mined predicted false positives;
- compare pairwise boundary loss with LambdaRank or another differentiable
  top-k surrogate;
- subtract each row's mean candidate delta so common calibration shifts cannot
  dominate;
- warm up the layer-specific router/output heads at a higher learning rate;
- then unfreeze large encoders at a lower learning rate; and
- use separate learning rates for pretrained/frozen-context adapters and new
  heads.

Do not interpret a falling NLL/KL as success without actual rank-8 swaps and
Recall@8 improvement.

## Recommended decision tree

1. **J beats raw with a meaningful paired gain:** retain J, then test raw+J.
2. **J ties raw but both improve the base:** use raw as the simpler primary
   stream; keep J only if dual-stream fusion adds signal.
3. **Neither improves validation, but real-row overfit reaches the oracle:**
   collect more diverse complete requests and regularize.
4. **Real-row overfit cannot approach the oracle:** replace the shared router
   query and candidate-only loss before collecting more data.
5. **MTP diagonal probe is strong only on prefix matches:** prioritize
   reliability gating and multi-branch capture.
6. **Layer-specific full-router heads beat shared heads:** promote them even if
   their parameter count is substantially larger; accuracy is currently the
   objective.
7. **c64 conditional recovery exceeds roughly 0.95:** then widen or union the
   candidate pool to remove the final coverage ceiling.

## Expected route toward 0.90

No single modification should be assumed to supply a 10-point gain. The most
plausible sequence is cumulative:

1. recover conditional signal already learned by HARP through its frozen
   context;
2. directly align MTP depth and target layer;
3. replace the shared SVD-coordinate query with layer-specific full-router
   heads;
4. use raw+J rather than forcing J to be sufficient;
5. use draft-token and predicted reliability information;
6. add branch-aware candidates for H3/H4; and
7. apply a top-8-aligned objective and reranker.

If these changes yield high training conditional recovery but validation still
stalls below 0.85, the current 1,229-request training corpus is the next
bottleneck. Repeatedly widening the same model against the same validation set
would not be a sound route to 0.90; a larger, domain-balanced, request-split
capture would then be required.

## Code locations reviewed

The implementation observations above refer to:

- `harp8/jspace_reranker.py` — J grid, MTP encoder, shared router query,
  candidate set head, and loss;
- `harp8/jspace_data.py` — aligned histories, candidate features, and current
  omission of draft-token/confidence features;
- `harp8/train_jspace_reranker.py` — explicit `include_context=False`, active
  H1--H4 training, optimizer, evaluation, and lineage gates;
- `harp8/reranker.py` — candidate-pool context storage and the legacy reranker;
- `harp8/jspace_features.py` and `harp8/prepare_jspace_features.py` — dense J,
  raw, random-control, and shared-PCA preprocessing;
- `harp8/router_geometry.py` — layer-specific SVD router keys;
- `harp8/model.py` — the frozen HARP generator and its horizon-by-layer MTP
  query; and
- `jroute0/capture_mtp_depths_bf16.py` — exact MTP depth-chain and token-label
  semantics.

The existing sparse-J negative evidence and HARP endpoint history are recorded
in `J_ROUTE_0_REPORT_20260805.md`,
`HARP8_ACCURACY_EXPANSION_LEDGER_20260807.md`, and
`J_HARP_C64_IMPLEMENTATION_PROTOCOL_20260807.md`.
