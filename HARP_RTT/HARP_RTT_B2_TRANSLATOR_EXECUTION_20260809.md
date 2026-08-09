# HARP-RTT B2 Branch-Semantic Translator Execution

Date: 2026-08-09
Branch: `agent/harp-rtt-b1.5-exact-set-oracle`

## Promotion status

The immutable B1.5 confirmation remains a numerical threshold failure:

| Gate | Measured | Threshold | Original result |
| --- | ---: | ---: | --- |
| Mean H2-H4 C64 | 0.9828948975 | 0.985 | Fail |
| H4 C64 | 0.9790298462 | 0.970 | Pass |
| H4 greedy-prefix-mismatch C64 | 0.9621174046 | 0.930 | Pass |

The user explicitly accepted this result as an operational gate pass and
authorized continuation with the original plan. The measured report, threshold,
and `passed=false` value are not changed. The machine-readable authorization is
`B15_USER_OPERATIONAL_OVERRIDE_20260809.json`. It preserves budget 16 and the
frozen 40/24 candidate policy; no confirmation fallback was selected.

## B2 question

Can the causal adaptive-32 tree state recover at least half of the budget-16
target-oracle lift on held-out H3/H4 greedy-prefix-mismatch rows?

B2 trains only:

- the adaptive tree encoder and categorical branch/OTHER posterior;
- the layer/horizon-conditioned branch translator;
- the geometry and free branch route heads.

The HARP anchor, frozen token embedding, endpoint/generator modules, trajectory,
candidate generator, and rich reranker remain frozen. H1 and B3 training remain
blocked until the three-seed B2 gate is evaluated.

## Frozen data protocol

- Fitting: 256 disjoint outer-train requests, 16 positions/request, 4,096 total.
- Per request: eight pre-EOS uniform positions plus eight positions balanced
  across global causal MTP vocabulary-entropy and top1/top2-margin quadrants.
- The train-only scout reads only MTP metadata, causal event order, vocabulary
  entropy, and margin. It never reads acceptance, factual continuation, target
  routes, validation, calibration, or test.
- Fitting/tuning: 224/32 complete requests, fixed by request hash and shared by
  seeds.
- Evaluation: the untouched 128-request, 2,048-position diagnostic probe from
  B1.5. It is never used for early stopping.
- Counterfactual semantic supervision uses the ancestor-closed budget-16 mask.
  The encoder still consumes the full 32-node runtime tree.
- Unselected runtime nodes are aggregated into OTHER in posterior CE rather
  than treated as known-negative branches.

## Optimization contract

- Seeds: 42, 43, 44.
- Effective batch: 32.
- BF16 learned transforms with explicit FP32 router geometry.
- AdamW: learning rate 3e-4, weight decay 0.01, betas (0.9, 0.95).
- Gradient clipping: 1.0.
- Maximum 30 epochs; patience 5 on request-grouped fitting-tuning loss.
- Semantic objective: exact-set NLL + 0.2 query loss + 0.1 router KL.
- Target branch/OTHER posterior CE weight: 0.1.
- Random ancestor-closed anytime budgets: 1, 4, 8, 16, 32.

## Execution order

1. Freeze and checksum the fitting position manifest.
2. Commit and test the B2 implementation; bind capture to that source commit.
3. Capture adaptive-32 fitting inputs to local NVMe.
4. Replay all unique counterfactual nodes once, audit native IDs/geometry, and
   build the label-only dataflow.
5. Mirror only audited immutable capture products to persistent storage.
6. Train the three seeds independently and evaluate each on the untouched probe.
7. Aggregate with 1,000 complete-request bootstrap replicates, seed 42.
8. Start neither H1 nor B3 unless the B2 gate passes.
9. Document and publish all outcomes, then stop (not terminate) the supplied pod.

## B2 promotion gate

For both H3 and H4 greedy-prefix-mismatch rows:

- mean-seed oracle-lift recovery `G >= 0.50`;
- paired complete-request bootstrap lower bound for learned-minus-anchor
  candidate coverage is above zero;
- no seed has negative recovery.

A failure stops before B3. A noisy positive result expands fitting data to 8k
and then 20k positions before any 250k-position capture.
