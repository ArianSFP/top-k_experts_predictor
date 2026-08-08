# HARP-RTT B1.5 execution results

**Source commit:** `b1ce3c41c114b12db88c102d5d0cd18e22b99632`
**Branch:** `agent/harp-rtt-b1.5-exact-set-oracle`
**Status:** B1.1 complete; Stage A/B1.5 target replay pending an RTX PRO 6000; no optimizer started

## B1.1 corrected four-path oracle

The immutable 2,048-position, 128-request diagnostic probe was re-evaluated
without recapture. Native BF16 selected IDs, rather than router-softmax values,
define branch inclusion mass. Uncaptured target probability is assigned to
exact-k frozen-anchor marginals. H1 is forced to the frozen-anchor top-64 and is
excluded from branch promotion.

The result is a clean information-gate failure. The strongest declared policy
is anchor-48 plus branch-16:

| Metric | Gate | Result | Decision |
| --- | ---: | ---: | --- |
| Mean C64, H2--H4 | 0.985 | 0.970386 | Fail |
| C64, H4 | 0.970 | 0.956250 | Fail |
| C64, H4 greedy-prefix mismatch | 0.930 | 0.890411 | Fail |

Horizon coverage for that policy is H2 `0.983257`, H3 `0.971651`, and H4
`0.956250`. The fixed quota outperforms global top-64 inclusion mass
(`0.966668` mean H2--H4 and `0.947519` H4), so quota choice matters, but it
cannot repair the missing information.

The lift over the frozen anchor is real. The paired complete-request bootstrap
delta is `0.143553` over H2--H4 with 95% interval
`[0.137734, 0.149485]`; on H4 mismatch rows it is `0.156662` with interval
`[0.138305, 0.174420]`. The failure is nevertheless localized and decisive:
H4 matched-prefix C64 is `0.989953`, whereas H4 mismatch C64 is `0.890411`.

The four selected paths retain mean target path mass H2/H3/H4 of
`0.947810/0.870576/0.790134` and contain the factual prefix at
`0.969238/0.901367/0.828125`. The underlying adaptive-32 tree contains the
factual H4 prefix at `0.929688`. This is direct evidence that useful adaptive-32
nodes were omitted by the four-path companion and motivates all-node replay.

Greedy-prefix nesting passed with no failures: 165 positions first diverge at
H2, 233 at H3, 233 at H4, and 1,417 remain fully matched through H4. The
independent frozen-anchor H1 C64 is `0.899750`, below its future `0.98` gate;
it is nonblocking for B1.5 and no H1 optimizer was constructed.

## Preservation and safety

The checksummed authoritative artifact is mirrored at:

```text
/workspace/LLM_prefetch_study/artifacts/harp_rtt/b15_exact_set_oracle_20260808/
  b11_budget4_exact_selected_set_b1ce3c4/
```

It records `training_started=false`, `optimizer_started=false`, and no access to
validation, calibration, or sealed test data. The complete repository suite
passes with 207 tests and three expected CUDA-only skips; compilation, shell
syntax, wheel build, diff checks, and the publication privacy scan also pass.

## Blocking next step

Load the 67 GiB target only on a user-supplied RTX PRO 6000. On the immutable
32-position Stage-A adaptive trees, capture both full-prefix reference and
sibling-isolated cloned-parent-cache all-node companions, then require geometry,
native selected-ID, cache-parity, checksum, and label-isolation audits to pass.
Only after Stage A passes may the existing 128-request probe be replayed all-node
and evaluated in the order all, 16, 8, 4. B2 translator and B3 ranker training
remain prohibited.
