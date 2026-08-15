# HARP Resident-Shadow v2 PRO 6000 Results

Date: 2026-08-15

## Outcome

Resident-Only v2 is a major same-size improvement over the sealed v1 hybrid,
but it does not pass the compact closed-loop promotion screen.

On the frozen 32-request / 512-position outer-train diagnostic partition:

| Condition | Branch H2--H4 Recall@8 | Shadow-LM H1--H4 | Shadow-LM H2--H4 | H4 |
| --- | ---: | ---: | ---: | ---: |
| v1 resident + generic fallback | 0.258521 | 0.393674 | 0.420449 | 0.383875 |
| v2 resident-only | **0.867058** | **0.828593** | **0.813814** | **0.784229** |
| Absolute v2 gain | **+0.608537** | **+0.434920** | **+0.393365** | **+0.400354** |

The old route-agnostic width-512 fallback is therefore decisively harmful in
closed loop. Removing it and assigning the same six-GiB budget to complete
resident INT4 expert functions was the correct architectural decision.

The preregistered compact screen required H1--H4 at least 0.88 and H4 at least
0.85. Resident-Only v2 misses both gates, so the 128-request evaluation was not
opened. No formal validation, calibration, or sealed-test data were opened.

## Exact screen contract

The screen used:

- source bundle commit `8c0cde1c272cb1eca82cf90e500a9c73367b06df`;
- evaluator commit `880acbad44160512d5f384f6dd314f5b9bfc729b`;
- 3,850 complete group-64 INT4 resident layer-expert cells;
- exact serialized bundle size 5.992392 GiB;
- no fallback, no target expert loads, and no optimizer;
- exact target non-expert backbone and committed-prefix hybrid cache;
- budget-16 ancestor-closed runtime node masks;
- causal Shadow-LM branch likelihood and OTHER-to-anchor completion;
- 1,000 complete-request bootstrap replicates with seed 42.

The v2 screen executed 14,105 visible tree nodes. Native exact-prefix parity was
checked over the complete screen and produced zero selected-ID mismatches. Peak
reserved VRAM was 86.166 GiB on the RTX PRO 6000.

The evaluator initially failed closed because its parity comparison included
valid all-node labels for nodes intentionally omitted by the reduced runtime
budget. Masked nodes carry sentinel outputs by design. The audit now intersects
label validity with the runtime node mask. A regression test covers the reduced
budget case; model execution is unchanged.

## v2 detail

Strict factual Recall@8:

| Horizon | Recall@8 |
| --- | ---: |
| H1 | 0.872931 |
| H2 | 0.842950 |
| H3 | 0.814264 |
| H4 | 0.784229 |
| Mean H1--H4 | 0.828593 |

The complete-request 95% interval for H1--H4 is
`[0.806001, 0.847644]`. The H2--H4 interval is
`[0.788404, 0.836836]`.

The branch-route score is much stronger than the final mixture:

| Horizon | Branch Recall@8 |
| --- | ---: |
| H2 | 0.869635 |
| H3 | 0.867609 |
| H4 | 0.863932 |
| Mean H2--H4 | 0.867058 |

Shadow-LM remains the best causal selector in this compact condition. Its
H1--H4 result is 0.828593, compared with 0.803523 for the raw MTP prior and
0.799115 for the old learned posterior. Perfect target branch weighting reaches
only 0.841187 and realised factual-branch selection reaches 0.844730. The next
large gain must therefore improve route execution, not posterior capacity.

## Proxy failure and diagnosis

The 3090 teacher-forced proxy predicted 0.946807 mean next-router Recall@8 for
Resident-Only v2. Exact closed-loop branch recall is 0.867058, a 0.079748 drop.
Errors from zeroing missing routed residuals compound through the exact frozen
backbone. One-layer teacher forcing is useful for rejecting clearly bad local
modules, but is not a reliable promotion metric for omission-heavy rollouts.

The resident namespace covers about 70% of native branch slots and 73% of
native branch execution weight on this screen. The successful full INT4 model
showed that preserving expert-specific nonlinear computation is essential.
The compact successor must restore useful nonlinear computation for the
remaining route mass without returning to a generic average residual.

## Rejected opportunistic cache extension

A label-only audit measured the exact experts already resident from the current
committed token. They add only the following execution-weight coverage beyond
the static v2 residents:

| Horizon | Static v2 | Incremental current-token cache | Combined |
| --- | ---: | ---: | ---: |
| H2 | 0.729488 | 0.052803 | 0.782291 |
| H3 | 0.734767 | 0.048700 | 0.783467 |
| H4 | 0.732354 | 0.047115 | 0.779470 |

This is useful as a future opportunistic deployment feature, but the incremental
mass is too small to justify another expensive closed-loop run under the current
large-gain-only rule. No cache-aware execution code was added.

## Next large-gain experiment

The next no-recapture experiment is a resident functional codebook.

For every missing selected expert, execute one or a small mixture of complete
resident nonlinear experts chosen from train-only activation behavior. Preserve
the native selected weight and do not renormalize. This uses the resident expert
functions already stored in the 5.992-GiB bundle, so the mapping and mixture
coefficients cost only KiB to low MiB and no target expert load is initiated.

The immutable ladder is:

1. On a 24-GiB 3090/4090, fit nearest-one and nearest-two resident mappings per
   layer using train-only factual activations and exact target expert outputs.
2. Evaluate deployed routed-residual error and authoritative next-router Recall@8
   on the 32 tune requests; do not use diagnostic-development labels to select.
3. Continue only if the functional codebook recovers at least half of the
   resident-only-to-exact-tail gap and predicts a closed-loop gain of roughly
   eight to ten points with no late-layer regression.
4. Freeze one mapping and run a 32-request exact-cache PRO screen.
5. Open the 128-request partition only at H1--H4 at least 0.88 and H4 at least
   0.85; compact promotion remains H1--H4 at least 0.90.

If nearest resident functions do not recover the tail gap, the six-GiB limit is
binding for this static-resident architecture. The next honest trade-off is a
larger resident bundle or on-demand expert I/O, not another static route head.
