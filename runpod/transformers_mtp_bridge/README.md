# GCRP-2R v1.3 Transformers joint pilot

Status: **blocking audit passed; stop before production capture or training**.

## RouteMTP v1

`hydrate_routemtp_prefix_cache.py` performs the one-time, outer-train-only
adapter-off prefix hydration required by the independent recurrent RouteMTP
runner. It writes one checkpoint-derived K/V cache per request and a
hash-bound source-position table. Consumers must use
`harp_rtt.routemtp_cache.load_causal_cache_slice`; direct unsliced
full-request cache access is outside the model contract. The blocking next
command is `runpod/audit_harp_routemtp_replay.py`, which audits exactly 32
positions and must pass before `runpod/train_harp_routemtp_v1.py` may
construct an optimizer.

This immutable pilot proves that the BF16 Transformers backend can emit an
event-aligned, same-execution target-plus-MTP trace for GCRP-2R v1.3.  It is a
schema/instrumentation pilot, not a statistically useful training corpus.

## Captured data

For each of 98 authoritative token positions and all 40 target MoE layers:

- post-attention/pre-MoE residual `u`;
- normalized target router/expert input `a`;
- post-MoE/post-layer residual `x+`;
- raw router logits, full FP32 probabilities, ordered top-8 IDs and weights;
- routed residual, gated shared residual, shared gate, and complete MoE delta;
- eight aligned unweighted and weighted candidate expert-output vectors;
- all GCRP-2R v1.3 late-view scalar reductions.

Six speculative cycles contain one native checkpoint-MTP branch through depths
1--6 (36 nodes).  Every node stores fused state, normalized router input,
post-FFN state, vocabulary-head input, raw router logits, full router
probabilities, top-8 IDs/weights, top-64 vocabulary log probabilities, full
vocabulary logits for this tiny audit, explicit state/input/vocabulary
positions, branch provenance, and a separate label-only acceptance record.

## Position contract

For a cycle rooted at authoritative target router-state position `p`:

- MTP depth 1 router/input state is at `p+1`;
- its vocabulary prediction and draft token are at `p+2`;
- depth `d` maps to GCRP target position `p+d` and raw GCRP horizon `d`.

The production GCRP-2R model consumes source slots 1 and 2.  Depths 3--6 are
retained for future work.

## Audit outcome

`CAPTURE_AUDIT.json` passed:

- 3,920/3,920 target-layer rows complete;
- 98/98 positions contain all 40 target layers;
- 36/36 MTP nodes have tensor-complete and label-complete records;
- all six depths occur exactly six times;
- no positive-gap native top-8 mismatch;
- target router probability maximum absolute error: 1.19e-7;
- MTP router probability maximum absolute error: 1.49e-8;
- target residual identity maximum relative RMS error: 0.00419;
- complete MoE decomposition maximum relative RMS error: 0.00816;
- at least 18 generated positions have valid t+2 truth.

The small BF16 SGLang run remains the native-MTP semantic oracle. Cross-engine
BF16 kernels are not route-identical near cutoff boundaries, so the
Transformers trace is authoritative for Transformers experiments. The bridge
validated token/position semantics and acceptance: one oracle prompt had an
exact six-token Transformers MTP tree, while the other showed prompt-sensitive
argmax divergence under small numerical perturbations but identical first-token
acceptance. This is recorded, not hidden.

## Reproduction

Use the files under `code/`. The capture writer refuses to overwrite an
existing directory. Run the auditor before any downstream view construction.

No split manifest, flattened examples, production corpus, J-space corpus, cache
simulation, predictor weights, optimizer state, or training result exists in
this artifact.

## HARP-RTT adaptive-tree capture

The original driver above is intentionally unchanged and remains the immutable
single-chain v1 reference.  The HARP-RTT extension is a separate schema and
driver:

- `adaptive_mtp_tree.py`: frozen deterministic expansion policy;
- `native_mtp_branch.py`: isolated native checkpoint-MTP path execution;
- `capture_transformers_adaptive_segment.py`: exact-H1, adaptive H2--H4 writer;
- `audit_adaptive_tree_capture.py`: blocking schema/leakage/parent/budget audit;
- `run_adaptive_tree_pilot.sh`: one-source-position, one-request pilot only.

The exact target-selected `x[t+1]` is node 0. A greedy spine is materialized
through H4 first. Remaining budget is confidence-adaptive: confident nodes have
one or two children, uncertain nodes expose four or eight, and best-first path
probability plus a depth bonus allocates the rest. The first-teacher capture
hard-caps the complete tree, including its root, at 32 nodes.

The current Transformers cache API is mutable and does not expose a safe native
fork primitive. The adaptive driver therefore evaluates each path with an
isolated full-prefix call, never sharing a mutable KV cache across siblings.
This is true native checkpoint-MTP branching, but intentionally capture-time
and slower than a production tree kernel. Each child receives only target final
hidden rows through `t` and MTP head-input rows from its own ancestors.

Acceptance is emitted later in separate `mtp_acceptance_label` events. Causal
node rows carry no realized H2--H4 IDs, future target routes/states, or eventual
acceptance. Prompt rows marked `test`, `sealed_test`, `outer_test`, or external
evaluation fail closed.

CPU-only deterministic policy dry run (no model or corpus access):

```bash
python runpod/transformers_mtp_bridge/adaptive_mtp_tree.py \
  --dry-run --scenario uncertain --max-nodes 32
```

Small BF16 pilot on a pod (one prompt, one generated-token source-state warmup,
one source position, three label-lookahead tokens, then the blocking auditor):

```bash
runpod/transformers_mtp_bridge/run_adaptive_tree_pilot.sh \
  <model-directory> \
  <new-output-directory>
```

CUDA remains the default. On a large-RAM CPU pod, select CPU explicitly; both
the frozen target and native MTP remain BF16 with eager attention/experts:

```bash
runpod/transformers_mtp_bridge/run_adaptive_tree_pilot.sh \
  <model-directory> \
  <new-output-directory> \
  --device cpu
```

`HARP_RTT_CAPTURE_DEVICE=cpu` is the equivalent launcher environment override.
There is no automatic CUDA-to-CPU fallback: this keeps numerical provenance
explicit. CPU mode needs enough RAM for the full BF16 checkpoint and is expected
to be substantially slower than CUDA. The run manifest records the actual
execution device, CPU topology, PyTorch CUDA build separately from active CUDA,
and clock semantics without claiming CUDA synchronization for a CPU run.

The output path must not exist. The driver writes versioned manifests and
checksums, refuses overwrite, leaves `training_started=false`, and ends at
`STOP_BEFORE_TRAINING.json`. Do not start a broad recapture until the pilot's
adaptive blocking audit passes.
