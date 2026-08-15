# HARP Route-Conditioned BasisDraft v1

Date: 2026-08-15

> **Status:** superseded after execution. Scalar/per-neuron BasisDraft and
> width-2/width-16 expert residuals produced only 0--1.1 point gains. The
> successful compact successor is the globally budgeted Resident-Shadow v1
> documented in `HARP_RESIDENT_SHADOW_V1_HANDOFF_20260815.md`.

## Objective

Compress the successful HARP-ShadowRoute execution teacher into a resident
predictor that preserves strict Recall@8 while never loading or executing an
offloaded target routed expert. The native MTP model is explicitly outside the
predictor-size budget. The first deployable target is:

- at least 0.80 mean H1--H4 Recall@8;
- preserve as much of ShadowRoute's 0.9166 result as possible;
- no target routed-expert dependency in the runtime module graph;
- approximately 6 GB or less total live predictor state, excluding native MTP
  and state already owned and shared by the serving target;
- materially less work than one target decode step.

## What made ShadowRoute successful

The large gain did not come from INT4 itself or from a wider classifier. It
came from retaining the computation that determines later router inputs:

1. exact frozen target normalisation, hybrid GDN/attention backbone, routers,
   shared experts, committed-prefix state and native top-8 weight semantics;
2. closed-loop hidden-state execution through all 40 layers;
3. expert-specific nonlinear routed effects rather than a static set readout;
4. causal Shadow-LM branch likelihood and explicit OTHER-to-anchor mass;
5. exact H1 root treatment.

The evidence is sharp. A static direct translator reached about 0.557 branch
Recall@8. Exact target top-1 plus one width-512 draft residual reached 0.7985.
Exact top-4 reached 0.9146, and full INT4 top-8 reached 0.9552 branch recall.
Shadow-LM then raised the complete H1--H4 result to 0.9166. Conversely,
expert-indexed magnitude-selected miniature networks reached only 0.679--0.715.
The indispensable property is therefore the route-conditioned nonlinear
function, not expert identity alone.

## Architecture

Every target layer owns `J=16` shared SwiGLU basis functions of width 32 and a
small neuron-coefficient table `c[e,j,r]` for its 256 target experts. For the exact
router-selected IDs and native execution weights `(e_i,w_i)`, all eight target
expert effects are approximated without target-expert I/O:

```text
alpha[j,r] = sum_i w_i * c[e_i,j,r]
delta_routed = sum_j Down[j](alpha[j] * BasisActivation[j](router_input))
```

The shared basis pool contains 125,829,120 weights and the neuron-coefficient
tables contain 5,242,880 weights. The first scalar/neuron-only diagnostics did
not produce a material routing gain, so v1 also includes a zero-initialised
expert-specific width-2 residual, adding 125,829,120 weights. The complete
student is 256,901,120 parameters: approximately 490 MiB BF16, 245 MiB INT8,
or about 130 MiB group-64 INT4. Coefficients remain BF16.

The initializer splits each successful width-512 S0 expert into sixteen
contiguous width-32 blocks and sets every expert coefficient to one. This is
function-identical to S0 when native top-8 weights sum to one, while subsequent
training can specialise each target expert through shared functional bases.

Runtime uses the frozen ancestor-closed budget-16 tree. Unselected nodes are
not executed. Removed captured probability is assigned to OTHER rather than
renormalised. The evaluator defaults to budget 16 and retains nested 4/8/16
controls.

## Memory contract

The deployable selective loader instantiates the official target text backbone
on `meta`, replaces every native routed-expert pool, and only then materialises
checkpoint tensors. It records every omitted native expert tensor and rejects
any unresolved native expert dependency.

Approximate BF16 inventory excluding native MTP and all target routed experts:

| Component | Size |
| --- | ---: |
| exact target non-routed text core | 2.67 GiB |
| token embedding | 0.95 GiB |
| Shadow-LM head | 0.95 GiB |
| BasisDraft | 0.24 GiB |
| anchor and geometry | 0.10--0.13 GiB |
| bounded live branch state (budget 16) | about 0.43 GiB |

This is approximately 5.4--5.8 GiB. In integrated serving, target-owned core,
embedding, LM head and committed cache are shared, making incremental predictor
weights far smaller. Predictor execution must never initiate an expert load;
later opportunistic use of an already-resident target expert is a separate
ablation, not the default.

## No-recapture experiment ladder

1. Train a layer-local width-512 shared baseline on existing factual teacher
   states with aggregate routed-residual and next-router losses.
2. Initialise BasisDraft exactly from that checkpoint and train expert-specific
   effects, aggregate residual and next-router boundaries.
3. Require a large component gain: at least ten absolute points of next-router
   Recall@8 above the matched shared baseline on representative layers, without
   worse residual reconstruction. Small gains do not justify all-layer work.
4. If the gate passes, train all 40 layer shards, retaining best and latest
   only and binding every shard to source, target and data hashes.
5. Run a 32-request budget-16 closed-loop screen with the exact non-expert
   backbone and Shadow-LM. Request a 96 GB GPU only for this stage. Promote to
   all 128 requests only if branch Recall@8 is at least 0.90 and mean H1--H4 is
   at least 0.88.
6. Compact promotion requires mean H1--H4 at least 0.90, H4 at least 0.85 and
   BF16 BasisDraft no larger than 300 MiB. The absolute continuation floor is
   0.80 H1--H4.
7. Quantise INT8, then group-64 QAT INT4, only after BF16 passes; accept at most
   one point absolute loss.

Only one engineering seed is used for diagnostic variants. A second seed is
reserved for a result close enough to change the promotion decision.

## Training and leakage contract

- Existing request-disjoint outer-train factual states provide router inputs,
  IDs, execution weights and aggregate routed residuals.
- Individual selected-expert teacher effects are computed layer-locally from
  one frozen target expert layer; they are never stored as runtime inputs.
- Formal validation, calibration and sealed test remain unopened.
- The frozen target has no gradient. Runtime `forward` accepts only hidden
  state, causal selected IDs and native weights.
- BasisDraft requires all eight selected routes and never silently truncates or
  renormalises them.
- Every run writes an immutable manifest, optimizer-start record, epoch-zero
  audit, fsynced metrics, best checkpoint, result and checksums.

## RunPod classes

- RTX 3090/4090 24 GB: layer-local training and unit/integration tests.
- RTX PRO 6000 96 GB (or H100 NVL/H200): exact-backbone closed-loop evaluation
  only after the component gate passes.
- No new target or MTP capture is authorised by this stage.
