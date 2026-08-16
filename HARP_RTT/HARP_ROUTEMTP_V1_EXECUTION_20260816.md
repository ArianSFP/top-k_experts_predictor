# HARP RouteMTP v1 Execution

Date: 2026-08-16

Implementation commit: `fec4166fce10c25169f06d89f04080f98d0f5c28`

Hydration/replay commit: `1f53e65429d07771833b89d30a4cb5a503a037d2`

## Objective

RouteMTP v1 tests whether routing-relevant computation can be learned into a
separately recurrent copy of the native MTP layer while the original MTP
remains solely responsible for token generation, tree topology and its causal
prior. The first invasive stage opens LoRA only on MTP fusion, attention Q and
attention O; K/V and the committed-prefix cache remain adapter-off and
coherent.

The design retains five MTP channels until target-layer conditioning, predicts
one 40-layer route trajectory for each node's actual depth, combines exact
target-router geometry with a direct low-rank score residual, and separates
pre-execution edge correction from post-execution branch evidence. Factual
training uses an exact mixture likelihood over branch exact-set distributions
plus OTHER/anchor. Budgets 1/4/8/16 receive independently calibrated OTHER
mass.

## Frozen source and tests

- Branch: `agent/harp-routemtp-v1`
- Implementation commit: `fec4166fce10c25169f06d89f04080f98d0f5c28`
- Source archive SHA256:
  `f66d63553118a004900636d071babd50f2323378312ce5cbae80f710deee3400`
- Full repository suite: 510 passed, 3 skipped.
- Python compilation: passed.
- Wheel build: passed.
- Adapter-off Stage-A replay uses the earlier hydration source commit because
  that commit produced the authoritative cache records. The trainer accepts it
  only through the separate `hydration_source_commit` binding; trainable code
  remains fixed to the implementation commit above.

## RTX PRO 6000 phase

The PRO-only phase performed deterministic prefix hydration and parity. It did
not construct an optimizer or train any parameter.

### Stage A

- 2 complete outer-train requests, 32 source positions, 1,024 tree nodes.
- Depth counts: H1 32, H2 91, H3 282, H4 619.
- Selected-ID mismatches: 0.
- Maximum fused/router-input/post-MoE/vocabulary-head/router-logit error: 0.
- Maximum selected-weight error: 0.
- Peak reserved VRAM: 66.296875 GiB, below the 90-GiB gate.
- Hydration manifest SHA256:
  `f81fed2a348c5fd24ad6dc6e4566c75582058387853413a166c515e421be38e0`
- Parity report SHA256:
  `ec7ced72358185c169cb0fffb6ff47d4fb5d903674e2a21ce7be942fc0efe1e5`
- Hydration audit passed every checksum, geometry and causal-offset check.

Cached append was not promoted: it produced 300/1,024 native-ID mismatches due
to a BF16 matrix-shape arithmetic difference. Isolated full-prefix replay is
the authoritative bit-exact implementation.

### Fitting and official-tune hydration

- 256 request-disjoint outer-train requests.
- 4,096 source positions.
- Hydration manifest SHA256:
  `e68080464d2beffbb4a99d40fcc791fe8b0b2dd067451715ab6489067310afc8`
- Record bytes: 608,460,040.
- Cache-length range: 269--1,127.
- Peak reserved VRAM: 67.197265625 GiB.
- All 256 record checksums, tensor geometries, causal offsets and provenance
  bindings passed.

The protected split is reconstructed from the frozen B2 request hash. The 32
official tuning requests remain untouched during architecture selection. Only
the 224 fitting requests are divided into 192 internal-fit and 32
internal-development requests.

### Diagnostic-development hydration

- 128 request-, lineage- and dedup-group-disjoint outer-train requests.
- 2,048 source positions.
- Hydration manifest SHA256:
  `3bfb008c6ba764b6f3fcf83f2c6f0653e5ac02df9442b77b35dcc0ec874e057c`
- Record bytes: 307,607,464.
- Cache-length range: 269--958.
- Peak reserved VRAM: 67.349609375 GiB.
- All 128 record checksums, tensor geometries, causal offsets and provenance
  bindings passed.

The diagnostic partition remains unopened by training. Formal validation,
calibration and sealed test were never opened.

## Training preflight

The no-optimizer `f0_captured` preflight passed from the frozen implementation
commit. It verified:

- Stage-A hydration is an exact record-level subset of full hydration;
- source, model, MTP, split, companion, geometry and anchor hashes;
- independent native-MTP token generation and RouteMTP route-only behavior;
- fusion/Q/O adapter contract with immutable base K/V;
- protected internal-fit/internal-development and official-tune partitions;
- the 1-GiB added-weight ceiling;
- one CUDA batch and the sealed-data guards.

Preflight result:

- added deployed BF16 bytes: 11,684,402;
- trainable BF16 bytes at F0: 6,956,846;
- selected metric: branch Recall@8;
- `training_started=false`;
- `optimizer_constructed=false`;
- `official_tune_opened=false`;
- `diagnostic_development_opened=false`.

Two earlier preflight invocations failed closed before model work because the
index file and segment directory were supplied instead of the index root.
They created no optimizer and changed no capture. The successful immutable
preflight is `preflight/` in the mirror.

## Persistent artifact

The authoritative mirror is:

`/workspace/LLM_prefetch_study/artifacts/harp_rtt/routemtp_v1_20260816_fec4166`

It contains the frozen source archive, Stage-A inputs/hydration/parity/audit,
fitting inputs/hydration/audit, diagnostic inputs/hydration/audit, successful
preflight and a recursively generated `SHA256SUMS` inventory.

## Next phase

No further RTX PRO 6000 work is required for the pilot. Training proceeds on
one 24-GB RTX 3090 or 4090 with at least 64 GB host RAM and 250 GB local NVMe.
The immutable ladder is F0-Captured -> F0-Replay -> F1-Replay -> F1-P ->
R2-QO -> R2-P -> R2-Joint, opening nonlinear R3 stages only after the declared
representation gate.

The pod stop is an out-of-band control-plane action after mirror verification.
This document is sealed before that action so it is present on persistent
storage; the controlling handoff records the verified terminal pod state.
