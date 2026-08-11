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

## Execution evidence

The frozen training implementation is
`f61230d53f5f78992d86113f1e47241ce6c0b112`. Capture and fitting used local
NVMe first and were mirrored only after audit.

- The fitting partition contains 256 outer-train requests and 4,096 positions:
  eight uniform and eight causal entropy/margin-stratified positions per
  request. Its selector SHA-256 is
  `0a6deca40720c8e6c385469f7b92859ed0d8b485fa77bb8cb3169d2530a846f3`.
- The adaptive-32 base capture contains 131,072 parent-before-child nodes.
  Exact capture-device BF16 replay matches all 131,072 native MTP weight rows.
  The earlier CPU-arithmetic failure is preserved separately; it consisted of
  seven nonzero rows and one BF16 ULP boundary rather than a capture mismatch.
- The all-node companion contains 4,096 label-only records and 5,079,040 valid
  layer rows. Native selected IDs match every authoritative row. The maximum
  centered-logit error is 0.0341653 (tolerance 0.0625), and the maximum router
  coordinate error is 1.049e-5.
- Writer, auditor, index, dataset, and privacy checks pass. Counterfactual
  labels remain reachable only under `targets.counterfactual`; validation,
  calibration, test, and inference access are rejected.

Seed 42 ran all 30 fixed epochs on an RTX PRO 6000. Its best fitting-tuning
checkpoint was epoch 29 with total loss 16.1871076003. The original automatic
probe evaluation failed before scoring because the relocated probe corpus
still contained an ephemeral
`/root/harp-rtt-v2-runs/.../adaptive32_controls_probe_v3_7a5b456` symlink.
The checkpoint had already been written and remained unchanged.

Recovery commit `c6f41475d1d1b2e70c9968fda2ed9f2f539924eb` adds:

- a pre-training corpus relocation check bound to the companion's base
  manifest, checksum-ledger, and capture-audit hashes;
- an evaluation-only command that verifies the checkpoint/run-manifest
  binding, constructs no optimizer, and refuses to overwrite any result;
- regression coverage for valid and broken relocation links.

The three recovered base hashes exactly equal the companion bindings:

| Artifact | SHA-256 |
| --- | --- |
| Base run manifest | `4050dad290d784804437bc334efe7a1814cd0ba28762b6997166d602b13a048d` |
| Base checksum ledger | `08410efb49a44b5473422579fa11b9b1fd6949d539bb23e366c1f6cf99d46157` |
| Base adaptive audit | `7f1896de00079c50be70ed6dba7e7a7352dc006a46d95efca58c67a972e13391` |

The recovered seed-42 diagnostic result is:

| Stratum | Anchor C64 | Learned C64 | Oracle C64 | Recovery G |
| --- | ---: | ---: | ---: | ---: |
| H2 prefix mismatch | 0.756136 | 0.884167 | 0.966004 | 0.610053 |
| H3 prefix mismatch | 0.734171 | 0.880402 | 0.969669 | 0.620945 |
| H4 prefix mismatch | 0.701704 | 0.869463 | 0.964669 | 0.637952 |
| Mean H2-H4, all rows | 0.823061 | 0.889922 | 0.985477 | n/a |

Seed 42 is checksum-complete both on local NVMe and at
`artifacts/harp_rtt/b2_translator_20260809_f61230d/training/seed42`.
The RTX PRO 6000 pod was stopped, not terminated, after mirror verification.
Seeds 43 and 44 were then started independently on the supplied RTX 4090 from
the unchanged training commit and the hash-verified repaired probe view.

## User-authorized operational promotion

After seed 42 completed and the early seed-43/44 learning curves tracked the
same direction, the user authorized stopping the remaining seeds and moving to
B3. Seed 43 was stopped after three complete epochs and seed 44 after two. Both
were interrupted with `SIGINT` at an epoch boundary; neither wrote a checkpoint
or opened the diagnostic probe. Their metrics, run manifests, logs, stop
records, and checksums are retained as partial evidence.

The machine-readable decision is
`B2_USER_OPERATIONAL_OVERRIDE_20260809.json`. This decision does not change the
thresholds below and does not convert a one-seed result into a three-seed
statistical claim. The preregistered aggregate and paired bootstrap are marked
not completed. B3 inherits the unchanged seed-42 checkpoint and its exact hash.

## Original B2 promotion gate (not completed)

For both H3 and H4 greedy-prefix-mismatch rows:

- mean-seed oracle-lift recovery `G >= 0.50`;
- paired complete-request bootstrap lower bound for learned-minus-anchor
  candidate coverage is above zero;
- no seed has negative recovery.

A failure would have stopped before B3. A noisy positive result would have
expanded fitting data to 8k and then 20k positions before any 250k-position
capture. The user-authorized operational promotion supersedes the execution
block, but not the recorded scientific status of this gate.
