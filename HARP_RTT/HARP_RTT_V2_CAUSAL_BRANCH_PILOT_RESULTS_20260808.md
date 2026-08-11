# HARP-RTT v2 causal-branch pilot: execution results and handoff

**Decision time:** 2026-08-08 UTC  
**Lineage:** `agent/harp-rtt-causal-branches-v2` from `c972450`  
**Final evaluated source commit:** `869024e278daaf55af9fd12cb7a5e45d7fe351c7`  
**Status:** Stage A passed; Stage B1 completed and failed all promotion gates; B2/B3 were not run.

> **Superseded metric protocol:** B1.5 retains these values as the legacy router-softmax baseline. Promotion now uses native selected-set inclusion mass, explicit OTHER anchor marginals, H2--H4-only branch gates, and an all-node adaptive-32 ceiling. See [the B1.5 plan](HARP_RTT_B1_5_EXACT_SET_ORACLE_PLAN_20260808.md).

## Outcome

The branch-rich capture stack is working and the counterfactual target records are numerically and causally auditable. Adaptive branching substantially improves the target-route oracle over the frozen anchor, but the four-path adaptive-32 information source is not sufficient for promotion.

| B1 gate | Required | Observed | Decision |
| --- | ---: | ---: | --- |
| Mean H1--H4 oracle C64 | 0.985 | 0.958824 | Fail |
| H4 oracle C64 | 0.970 | 0.962311 | Fail |
| H4 prefix-mismatch oracle C64 | 0.930 | 0.888714 | Fail |

The evaluator's declared decision is `stop_before_B2_change_tree_policy`. No translator fitting, generator training, reranker training, optimizer step, formal-validation label access, calibration-label access or sealed-test access occurred.

The positive result is real: adaptive oracle coverage improves over the frozen anchor by 0.116592 mean H1--H4, with a 1,000-replicate paired complete-request bootstrap 95% interval of `[0.112006, 0.121375]`. On H4 prefix-mismatch rows the improvement is 0.178690, with interval `[0.161340, 0.195849]`. The captured branch information is useful, but not yet sufficient.

## Implemented architecture and contracts

The pilot branch implements the planned v2 data and model interfaces:

- deterministic outer-train engineering, diagnostic and fitting partitions;
- exact-budget parent-before-child adaptive trees and matched greedy/fixed/adaptive controls;
- causal first-divergence path selection with no acceptance, factual-continuation or target-label reads;
- separate hash-bound counterfactual target companion records;
- isolated full-prefix reference execution and sibling-isolated cloned-prefix-cache execution;
- fail-closed companion, parity, index, dataset and leakage auditors;
- a layer/horizon-conditioned branch translator with rank-16 layer adapters;
- explicit OTHER posterior mass and separate categorical posterior versus acceptance heads;
- top-64 vocabulary evidence pooling and stable structural branch features without sample-local semantic IDs;
- ancestor-closed anytime views `{1,4,8,16,32}`;
- anchor-preserving C64 quota scheduling and the counterfactual, posterior, factual, acceptance and swap losses;
- exact depth-four enforcement, FP32 `Kq+b`, native BF16 selected-ID labels and strict JSON reporting.

These interfaces are implemented and unit/synthetic tested, but the new trainable modules were not optimized because B1 did not pass.

## Frozen inputs and safety

Execution used the supplied RTX PRO 6000 pod and the frozen Qwen3.6-35B-A3B BF16 target. Important bindings are:

- split manifest SHA-256: `be03308606e3c6ae524f0268bbd8a8b3590a76332f42ae165c203758bafc970b`;
- v3 partition manifest SHA-256: `f6baebb80244f21b3f15563d394b00543cd819b9c60f437b69ed812067405a6a`;
- router geometry SHA-256: `a6d1daa092ee028d567d8bebb61c735135b92472ae4785460ae7c7c31f1ad8e0`;
- frozen HARP anchor SHA-256: `410ced95e6082f6a9bfa962d082926ee1ff5c29d203a3869f713a5ee7e58a09d`;
- sequence-start SHA-256: `261b8a249c17636bcdc6989a6dc028ab7180e0c076aec5df8f3bf05d32acad7d`;
- sequence-end SHA-256: `fe225a7e8fef5639ae2d80f2fb4d5b7c4e50234c772b1dcfabff78827d4a5423`.

All experiment rows are outer-train rows. Counterfactual tensors are label-only and are exposed only as `targets.counterfactual`. Validation, calibration, test and ordinary-inference loaders reject them. Every definitive capture and companion has a stop marker and checksum inventory.

## Capture corrections and the earlier 102 requests

The earlier partial B1 capture was useful but not promotion-salvageable. At the time it was questioned, 102 requests had completed; 103 completed before request 104 exposed the terminal-path failure. Those records are mechanically useful for debugging and throughput evidence, and they remain preserved at `adaptive32_controls_probe_4f687d3`, but they have no final manifest/checksum inventory and belong to the superseded partition semantics.

Two fail-closed corrections changed the canonical dataset contract:

1. an MTP branch may terminate before H4, so the diagnostic greedy control must follow the actual coherent rank-zero parent chain rather than assume the first four adaptive nodes form a path;
2. source positions must have complete factual H1--H4 tokens before the first EOS; H4 may equal the first EOS, but no post-EOS position is eligible.

Because the corrected v3 partition and capture hashes differ, mixing the partial rows with a fresh suffix would have produced a non-canonical probe. The clean recapture authorized by the user was therefore the correct scientific choice. The partial rows are salvaged as failure evidence only, not as B1 gates or training data.

## Definitive Stage A

The fresh v3 Stage-A base capture is:

`/root/harp-rtt-v2-runs/stage_a/adaptive32_controls_v3_7a5b456`

It contains 32 trees, exactly 1,024 adaptive nodes and 192 separate H1--H6 anchor-spine nodes. Adaptive depths H1/H2/H3/H4 are `32/103/275/614`. Both blocking base/control audits pass.

The corrected H1-conditioned companion pair is:

- `counterfactual_reference_v3_h1cond_7a5b456`;
- `counterfactual_cloned_v3_h1cond_7a5b456`.

Each contains 32 records and 14,360 valid H2--H4 layer rows. The optimized/reference parity report is `COUNTERFACTUAL_CACHE_PARITY_V3_H1COND_7a5b456.json`: topology and native selected IDs are identical, and query coordinates, logits and weights have exactly zero difference. The optimized companion audit reports maximum query-coordinate error `6.914139e-6` and maximum centered-logit error `0.032786`, below the frozen `0.0625` tolerance. Native selected IDs match on all 14,360 rows.

The dataflow artifact `dataflow_v3_h1cond_7a5b456` passes one-to-one writer -> companion -> index -> dataset checks on all 32 rows. Counterfactual data exists only under targets, all prohibited loader modes reject access, and no optimizer was constructed.

## Definitive Stage B1 capture

The clean v3 B1 base capture is:

`/root/harp-rtt-v2-runs/stage_b1/adaptive32_controls_probe_v3_7a5b456`

It contains 128 requests, 2,048 source trees, exactly 65,536 adaptive nodes and 12,288 separate anchor-spine nodes. Adaptive depths H1/H2/H3/H4 are `2,048/6,533/18,269/38,686`. Base and independently reconstructed matched-control audits pass.

Factual path occurrence is:

| View | H1 | H2 | H3 | H4 |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 1.000000 | 0.919434 | 0.805664 | 0.691895 |
| Fixed-16 | 1.000000 | 0.991699 | 0.949707 | 0.873535 |
| Adaptive-16 | 1.000000 | 0.980957 | 0.945801 | 0.904785 |
| Fixed-32 | 1.000000 | 0.996094 | 0.970703 | 0.911621 |
| Adaptive-32 | 1.000000 | 0.987793 | 0.964844 | 0.929688 |

Adaptive-32 is slightly behind fixed-32 at H2/H3 but ahead by 0.018067 at H4. This supports adaptive allocation for later horizons, although the selected counterfactual path budget remains insufficient.

Prefix nesting was verified with zero failures. First divergence counts over 2,048 rows are H2 `165`, H3 `233`, H4 `233`, and fully matched through H4 `1,417`.

## Definitive counterfactual companion and geometry

The B1 label artifact is:

`/root/harp-rtt-v2-runs/stage_b1/counterfactual_probe_v3_h1cond_7a5b456`

It contains 2,048 sealed label-only records, 909,840 valid H2--H4 layer rows and 44,960 full-router-input audit rows. Its audit passes:

- native selected IDs match on all 909,840 rows;
- maximum query-coordinate error is `1.049042e-5`;
- maximum centered-logit error is `0.033642`, below `0.0625`;
- mean reconstructed centered-logit cosine is `0.999940`;
- mean router KL is `3.655035e-5`;
- FP32-geometry top-8 recall against native BF16 selected IDs is `0.983716`, with boundary differences retained as audited numerical cases.

The dataflow artifact `dataflow_probe_v3_h1cond_7a5b456` passes on all 2,048 records and rejects validation, calibration, test and inference access.

Mean captured target path mass decreases from `0.947810` at H2 to `0.870575` at H3 and `0.790134` at H4. This is a direct sign that four selected counterfactual paths do not represent enough late branch probability.

## B1 oracle result

The authoritative strict-JSON report is:

`/root/harp-rtt-v2-runs/stage_b1/oracle_gate_v3_final_869024e/B1_ORACLE_GATE_REPORT.json`

Its SHA-256 is `16944fe564658346799b1e9689cbbc42d1c612812d36c2d7fec940e12b7e0eb3`; request metrics SHA-256 is `f04ea1c390c7f2066fccdda22a70fbea3b5e77bbf9e3d6b838c4abecc20ebfd2`.

| Metric | H1 | H2 | H3 | H4 | Mean H1--H4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen-anchor C64 | 0.899718 | 0.860300 | 0.825047 | 0.783864 | 0.842232 |
| Oracle anchor48 + branch16 C64 | 0.899718 | 0.993034 | 0.980232 | 0.962311 | 0.958824 |
| Frozen-anchor Recall@8 | 0.540720 | 0.482368 | 0.445512 | 0.411584 | 0.470046 |
| Oracle route Recall@8 | 0.540720 | 0.923410 | 0.882030 | 0.840509 | 0.796667 |

The H2--H4 oracle C64 mean is `0.978526`, still below 0.985 even if H1 is excluded. H4 fully greedy-prefix-matched oracle coverage is `0.999120`, while H4 prefix-mismatch coverage is `0.888714`. The failure is sharply localized to branches needed after a wrong greedy prefix.

## Frozen-anchor compatibility finding

The diagnostic probe uniformly spans long generations, while the legacy HARP anchor was trained on 34 rows per request and only source positions 0--32. Passing raw positions hundreds of tokens long into its `[p/33, (p/33)^2]` features was unsupported extrapolation. Commit `0a9df8fa8f43dbef1998e6c33defb7c58819f265` causally clamps only the frozen compatibility channel to `[0,32]`; the v2 inputs retain the raw position. Focused and full regression suites pass.

The corrected sealed evaluator changes the B1 values only in the fourth decimal place. The remaining anchor gap is therefore distribution/content shift, not merely position arithmetic. These anchor numbers must not be compared directly with the old first-33-row formal-validation result. There is also no new H1 branch source in this pilot: H1 remains anchor top-64, so the overall 0.985 gate cannot pass on this probe when H1 coverage is 0.899718. Future gates must report H1 anchor health separately from the H2--H4 branch-information gate or add a causal H1 source.

## Scientific decision

Do not run B2 translator training or B3 generator/ranker training on this information source. Do not enlarge the ranker, resume the old C64 run or treat the 0.958824 aggregate as a near pass. All three predeclared B1 gates failed, and the H4 mismatch gap is 0.041286.

The current result separates three effects:

1. adaptive-32 contains the factual H4 prefix 92.97% of the time, so the base tree is substantially richer than the greedy chain;
2. four causally selected paths retain only 79.01% mean target path mass at H4;
3. when the greedy prefix matches, route candidates are essentially complete, but mismatch rows remain far below gate.

This points first to counterfactual path selection/execution budget, then to tree-node allocation, rather than translator capacity.

## Recommended next experiment

Run a companion-only causal path-budget information sweep before recapturing the expensive base trees:

1. Keep the existing immutable adaptive-32 B1 base capture unchanged.
2. On Stage A, capture matched counterfactual companions for path budgets 4, 8, 16 and all unique adaptive-32 nodes. Use causal probability/divergence-depth selection only. A reasonable frozen allocation is greedy plus H2/H3/H4 alternatives in ratios `1+2+2+3` for budget 8 and `1+3+5+7` for budget 16.
3. Measure target path mass and oracle C64 with the same candidate contract. This tells whether the 32-node tree already contains sufficient route information that the four-path companion discarded.
4. Use the current B1 probe only for hypothesis generation. Freeze the new selector and confirm promotion on a new request-disjoint outer-train diagnostic probe; do not tune and promote on the same 128 requests.
5. If the all-node adaptive-32 oracle passes H4 mismatch >= 0.93, scale the counterfactual execution budget or learn a causal selector before B2.
6. If the all-node adaptive-32 oracle still fails, recapture adaptive-64 with H4-biased frontier allocation and matched fixed-64 control. Do not increase ranker size.
7. Evaluate legacy-anchor health on a separate request-disjoint first-33-position control, while keeping the long-generation branch gate focused on H2--H4. Do not open formal validation or sealed test.

This experiment is cheaper and more diagnostic than immediately repeating all target/base capture at 64 nodes. It directly distinguishes a four-path supervision bottleneck from a 32-node tree-information bottleneck.

## Artifact preservation

The complete local run tree, including failed/superseded evidence and definitive Stage A/B1 artifacts, is under `/root/harp-rtt-v2-runs`. Definitive apparent sizes include:

- Stage-A base: `1,505,498,534` bytes;
- Stage-A reference companion: `45,580,910` bytes;
- Stage-A optimized companion: `45,580,909` bytes;
- B1 base: `45,169,767,499` apparent bytes;
- B1 companion: `2,345,208,186` bytes;
- B1 dataflow/index: `292,888,007` bytes;
- final B1 report: `2,899,416` bytes.

The entire sparse-aware run tree will be mirrored to:

`/workspace/LLM_prefetch_study/artifacts/harp_rtt/v2_causal_branch_pilot_b1_final_20260808`

The mirror is complete only when its inventory and embedded definitive checksums pass. The pod must be stopped, never terminated, only after that verification and GitHub publication.

## Verification status

The complete local repository suite passed with 193 tests, three expected CUDA-only skips and two warnings. Python compilation, shell syntax checks and the wheel build passed. The final evaluator report passes a strict JSON parser and its checksum inventory. The tracked-file privacy scan found no private-key, token or ephemeral-endpoint match, and the largest tracked file is 62,873 bytes, so no capture/model payload is included in Git.
