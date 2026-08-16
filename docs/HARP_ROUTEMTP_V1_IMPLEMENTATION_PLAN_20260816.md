# HARP-RouteMTP v1 implementation and execution protocol

## Decision

RouteMTP is a second, independent execution of the frozen native MTP layer.
Native MTP remains solely responsible for token IDs, adaptive-tree topology,
and the initial causal edge probabilities. RouteMTP consumes those fixed token
IDs with its own recurrent hidden state and cache, and predicts target expert
routes plus a corrected posterior over branches already present in the tree.

This tests a hypothesis not covered by the earlier translators: whether the
MTP computation itself can be adapted to retain target-routing information.
It does not train another head over detached frozen MTP snapshots.

## Frozen serving contract

- Native MTP base tensors may be shared immutably; native and RouteMTP
  recurrent states are independent.
- Native token/tree generation is always adapter-off.
- RouteMTP initially adapts fusion `fc` and attention Q/O with rank-32 LoRA.
  K/V remain frozen. Every adapter is masked to speculative H1-H4 rows, so
  the committed prefix stays exactly adapter-off.
- RouteMTP predicts one `[40,256]` route trajectory for each node's actual
  depth. It never predicts four horizons from one node.
- The five MTP channels stay separate until target-layer conditioning:
  fusion output, router input, post-MoE hidden, vocabulary-head input, and MTP
  router/selected-route evidence.
- The causal current-token target bank uses a static DFlare-style source-layer
  mixture with a low-rank horizon bias. Dynamic cross-attention is not part of
  the first pilot.
- Target scores are frozen-router geometry `K_l q + b_l` plus a rank-16
  direct residual.
- The full vocabulary head is not executed by RouteMTP at serving time.
- Added serialized BF16 weights are hard-limited to 1 GiB; native MTP is
  explicitly excluded. The implemented R2 predictor is 5,121,305 parameters,
  or 9.77 MiB BF16, before 1.125 MiB of fusion/Q/O LoRA. Frozen router
  geometry is loaded from the bound static artifact and is not duplicated in
  compact checkpoints.

## Causal posterior and factual objective

The branch posterior is split into two causal signals:

1. A pre-execution correction uses the adapted parent state, proposed child
   token, causal target context, and structural metadata. It can order tree
   expansion.
2. A post-execution correction uses the child's adapted state and predicted
   route uncertainty. It is available only after that node runs.

At every parent, visible captured children are normalized together with an
explicit OTHER outcome. At zero initialization this exactly recovers the raw
native edge probabilities and assigns all masked/unseen mass to OTHER.

The factual training loss is the exact mixture of exact-eight set
distributions:

```text
-log [ pi_OTHER P_anchor(S*) + sum_n pi_n P_n(S*) ]
```

Inference mixes exact-k inclusion marginals, repairs only numerical
cardinality drift, and applies stable top eight. It never treats
`logit(mixture_marginal)` as the mixture likelihood.

Training samples budgets `{1,4,8,16}`. Each view applies its deployed
ancestor-closed mask and receives its own calibrated OTHER mass. All-node
counterfactual labels still supervise branch semantics.

## Data and split discipline

- Existing factual H1-H4 labels and all-node H2-H4 counterfactual labels are
  reused. No target-route recapture or new request generation is authorized.
- A one-time deterministic target-prefix hydration pass creates adapter-off
  MTP K/V caches and exact final-normalized target hidden rows for existing
  requests. One record is stored per request; the public loader can return
  only a hash-bound causal source-position slice.
- Hydration is bound to source, target/MTP checkpoint, split, base capture,
  and counterfactual companion hashes.
- The 224 fitting requests are deterministically split into 192 internal-fit
  and 32 internal-development requests for architecture/epoch decisions.
- The official 32-request tuning partition is reserved for one final
  calibration after the architecture is frozen.
- The 128-request diagnostic partition is opened once. Formal validation,
  calibration, and sealed test remain closed.

## Immutable experiment ladder

1. `F0-Captured`: historical detached-state reference.
2. `F0-Replay`: zero-adapter recurrent replay using the same runner/head.
3. `F1-Replay`: add the static target-layer bank and route history.
4. `F1-P`: freeze routes and train the zero-initialized pre/post posterior.
5. `R2-QO`: train fusion/Q/O LoRA under the raw native path prior.
6. `R2-P`: freeze R2 representation, reset/retrain the posterior.
7. `R2-Joint`: low-rate joint tuning; posterior gradients into the recurrent
   representation are scaled to 0.1.
8. `R3-Residual`: add a zero-gated width-256 recurrent SwiGLU residual.
9. `R3-Expert`: rank-8 LoRA on MTP expert gate/up/down functions while router
   weights stay frozen and IDs are recomputed from the adapted state.
10. `R3-Router`: route-only MTP router residual. It is never supervised with
    target expert IDs. Training anneals temperature 1.0 to 0.2, uses a sparse
    top-16 warm-up followed by hard top eight, and retains native-router
    trust/load regularization. Epoch-zero and inference remain exact top eight.

The first failed nonlinear addition ends the ladder. Seed 42 screens all
variants. Seed 43 is run only for the winning information-positive R2 model.

## Gates

- Adapter-off replay: exact native selected IDs; dense MTP states/router
  logits within frozen BF16 tolerances; sibling isolation; no optimizer.
- Information-positive: R2 beats F1-Replay by at least 0.03 H2-H4 branch
  Recall@8, including H4, with positive paired request-bootstrap lower bound.
- Breakthrough: at least +0.10 branch Recall@8.
- Scale-worthy: branch H2-H4 Recall@8 at least 0.75 and improves with more
  request diversity.
- Standalone useful: branch at least 0.85, factual H2-H4 at least 0.80, H4 at
  least 0.75, and C64 at least 0.97.
- Final candidate: branch approximately 0.935, factual H1-H4 at least 0.90,
  H4 approximately 0.87, with the learned causal posterior.

All frozen evaluations write complete-request rows and deterministic 1,000
replicate, seed-42 request-bootstrap intervals. They also record wall time and
milliseconds per source position for each anytime budget. Layer/deadline-aware
timely recall is computed only after real hardware timings are available.

## Hardware sequence

1. Hydration and the blocking 32-position parity proof require one RTX PRO
   6000-class GPU with 96 GB VRAM, at least 192 GB host RAM, and at least 1 TB
   local NVMe. The full BF16 target is approximately 67 GiB. The run has a
   predeclared 90-GiB maximum-reserved gate.
2. After hydration the target is unloaded. F0/F1/R2/R3 training uses a 24 GB
   RTX 3090/4090 with at least 64 GB RAM and 250 GB local NVMe.
3. A 96 GB GPU is requested again only after the information-positive gate for
   final exact-cache timing/evaluation.

All artifacts are written to fresh local NVMe, audited and checksummed, then
mirrored immutably. Each pod is stopped—not terminated—after its assigned
stage and mirror verification complete.

## Stage-A execution result (2026-08-16)

The first optimized cached-append replay was correctly rejected. Although it
used the same frozen K/V projections, changing the MTP fusion/attention GEMM
from the capture's full-prefix matrix shape to one appended row changed BF16
rounding at expert boundaries and produced recursive route divergence. Its
32-position diagnostic had 300 selected-ID mismatches over 1,024 nodes. No
tolerance was relaxed.

The authoritative runner now reproduces the capture's isolated full-prefix
execution and applies every trainable adapter only to the speculative tail.
The frozen committed-prefix rows remain adapter-off. On the blocking
32-position audit it achieved:

- 1,024/1,024 nodes audited across H1-H4;
- zero maximum error for fused, router-input, post-MoE and vocabulary-head
  state channels;
- zero router-logit and execution-weight error;
- zero selected-ID mismatches;
- 66.30 GiB hydration peak, below the frozen 90-GiB gate;
- no optimizer and no formal-validation, calibration or sealed-test access.

Cached append remains implemented only as an unpromoted performance research
path. It cannot replace the full-prefix reference unless a future audit is
bit-identical under the same immutable inputs.

## Implemented files

```text
harp_rtt/routemtp.py
harp_rtt/routemtp_adapters.py
harp_rtt/routemtp_batch.py
harp_rtt/routemtp_cache.py
harp_rtt/routemtp_loss.py
harp_rtt/routemtp_replay.py
runpod/transformers_mtp_bridge/hydrate_routemtp_prefix_cache.py
runpod/prepare_routemtp_hydration_inputs.py
runpod/audit_harp_routemtp_replay.py
runpod/train_harp_routemtp_v1.py
runpod/evaluate_harp_routemtp_v1.py
tests/test_harp_routemtp.py
tests/test_harp_routemtp_cache.py
tests/test_harp_routemtp_loss.py
```

The hydration/parity stage must pass before any optimizer is constructed.
