# HARP-DeltaRoute v4: branch-conditioned router-state trajectories

**Status (2026-08-12):** implementation complete locally; GPU execution has
not started. The first required GPU action is the optimizer-free Stage-0
export/evaluation on the existing 2,048-position outer-train diagnostic
probe. No target or MTP recapture is authorized by this plan.

**Lineage:** `agent/harp-deltaroute-v4`, based on DeltaTree v3 source
`63c0035ee22020610d654b1e9df1b3c4c502e310`.

**Selected frozen parent:** budget-16 v3 semantic checkpoint SHA-256
`218a26d9bac6e598ca4fa1640a2d3efb17ecd0f18e282f47e001fabce5e34d77`.
The rejected all-node checkpoint is not an initializer.

**Primary objective:** long-generation request-macro SlotRecall@8 for H1--H4.
The final target is at least `0.90`, with the working allocation
`(0.93, 0.91, 0.89, 0.87)`. H1 remains an independent exact-root problem;
this document's branch-information and candidate gates apply to H2--H4.

## 1. Why v4 supersedes a larger DeltaTree head

The existing probe establishes:

| Condition | Mean H2--H4 C64 | H4 C64 | Interpretation |
| --- | ---: | ---: | --- |
| Native counterfactual routes + learned posterior | `0.985307` | `0.982077` | Tree/posterior information is sufficient |
| Selected budget-16 route translator | `0.914467` | `0.898477` | Branch-to-target route translation is the bottleneck |
| Rejected all-node translator | `0.913429` | `0.895567` | More independently decoded nodes do not fix factual aggregation |
| Native routes, target posterior | about `+0.0003` over learned posterior | — | Posterior capacity is not the main loss |

The v3 head independently decodes every target layer from nearly the same
static node representation. It does not make the layer-27 route part of the
state used to predict layer 28, and its semantic objective does not train the
posterior-weighted H2--H4 factual mixture used to construct C64.

DeltaRoute v4 replaces that inductive bias with a causal 40-layer route scan
and trains the deployed factual mixture directly. The rich C64 ranker remains
closed until learned candidate information is at least:

- mean H2--H4 C64 `>= 0.98`;
- H4 C64 `>= 0.97`;
- H4 greedy-prefix-mismatch C64 `>= 0.93`.

## 2. No-recapture data contract

The existing immutable data are sufficient for the next ladder:

- fitting: 224 requests / 3,584 positions;
- tuning: 32 request-disjoint requests / 512 positions;
- diagnostic development: 128 request-disjoint requests / 2,048 positions;
- all 32 adaptive nodes remain causal runtime inputs;
- the nested budget-16 mask remains the deployed factual-supervision mask;
- every unique node has 40-layer target query coordinates, raw BF16 router
  logits, authoritative native selected IDs, and selected execution weights;
- eight factual history trajectories retain logits, IDs, and weights;
- factual H1--H4 router inputs, logits, IDs, and execution weights already
  exist in sidecars. `dataset.py` now exposes future execution weights only
  under `targets`.

Formal validation, calibration, and sealed test remain unopened. A target,
future route, counterfactual route, acceptance value, or factual prefix hash
may never enter the causal trajectory initializer.

## 3. Architecture

### 3.1 Layer-conditioned multi-channel tree state

Every adaptive node retains seven separate channel tokens until the target
layer asks its question:

1. MTP fused state;
2. MTP post-FFN hidden state;
3. MTP router input;
4. weighted top-64 vocabulary state;
5. branch token embedding;
6. MTP router logits;
7. stable structural/reliability metadata.

A learned query for `(horizon, target layer)` pools these channels. A
parent-before-child GRU path state supplies reusable prefix structure. This
removes the v3 early bottleneck that had to preserve every useful direction
for all 40 target layers in one node vector.

### 3.2 Router-control scan

For node `b` and target layer `l`, v4 predicts a router coordinate and native
route distribution, then uses the weighted predicted expert effects in the
next-layer transition:

\[
\widehat s_{b,l}=K_l\widehat q_{b,l}+b_l,
\qquad
(\widehat S_{b,l},\widehat w_{b,l})=
\operatorname{Top8Gate}(\widehat s_{b,l}),
\]

\[
g_{b,l}=\sum_e \widehat w_{b,l,e}
\left(d_{l,e}+r_{l,e}\odot C\widehat q_{b,l}\right),
\]

\[
\widehat q_{b,l+1}=A_l\widehat q_{b,l}
+G_\theta(\widehat q_{b,l},g_{b,l},\widetilde u_{b,l},l).
\]

The transition core is shared across branches and horizons, with rank-8
layer adapters. The initial configuration is latent width 256, expert-effect
width 64, transition width 512, and a zero-gated low-rank free score residual.
The entire architecture remains below 25 million trainable parameters.

Predicted Top8 execution uses the native stable tie policy in the forward
pass and a dense straight-through route message in the backward pass.
Execution weights—not merely selected IDs—affect the next-layer state.

### 3.3 Parent-preserving residualization

R2 and joint training are residualized around the frozen budget-16 v3 node
queries and scores. At epoch zero:

- every node query is bit-identical to the selected parent;
- every node score is bit-identical to the selected parent, including its
  learned free residual;
- C64 is bit-identical to the parent candidate contract;
- `OTHER` remains the anchor.

The runtime, factual mixture, and promotion C64 consume all 32 causal nodes.
All-node labels pretrain shared transition dynamics; the frozen budget-16 mask
remains the branch-specific counterfactual semantic-supervision mask during
joint fine-tuning. It is not allowed to silently prune the deployed runtime
mixture.

### 3.4 Expert-conditioned factual aggregation

The joint model retains branch identity. For every branch, layer, and expert,
it predicts a reliability correction conditioned on tree state, target-layer
context, expert router key, branch marginal, anchor marginal, path prior, and
query uncertainty. A zero `tau` reproduces the learned global posterior
mixture exactly; `OTHER` remains explicit.

The deployed H2--H4 objective contains exact-cardinality factual mixture NLL:

\[
\mathcal L_{\rm factual}
=-\log\left[
\pi_{\rm OTHER}P_A(S^T)
+\sum_{b\in\mathcal B_{16}}\pi_bP_b(S^T)
\right].
\]

Counterfactual exact-set, router-induced logit Huber, and top-8/top-9 boundary
losses remain strong stabilizers. Low-probability branches are depth-balanced
for semantic learning rather than suppressed by their path probability.

## 4. Immutable experiment ladder

### Stage 0 — no optimizer

`export_harp_deltaroute_v4_ceiling_bundle.py` performs one frozen budget-16
forward over the 2,048-row diagnostic probe. It exports compact native routes,
posteriors, candidate evidence, and per-true-expert support ranks. It does not
store multi-gigabyte dense node scores.

`evaluate_harp_deltaroute_v4_ceiling.py` reports:

1. native factual-branch Top8 with anchor fallback;
2. native target-posterior Top8;
3. native learned-posterior Top8;
4. native causal-MTP-prior Top8;
5. C64 swap-cap ceilings for 1, 2, 4, 6, and 8 swaps;
6. anchor, best-branch, learned-mixture, MTP-mixture, factual-branch, and
   raw-geometry ranks for every true expert missing from C64.

The native factual H2--H4 Top8 ceiling must be at least `0.89`. Failure means
the realized adaptive-32 routes are insufficient and stops dynamics work.

### Stage 1 — cheap factual aligners

Run M0 and M1 from the same frozen parent for seeds 42 and 43.

- M0: summary-feature per-expert residual MLP.
- M1: expert-conditioned branch attention retaining branch identity.

Both start with dense and C64 outputs bit-identical to the parent. They train
H2--H4 factual exact-set NLL plus `0.1` missing-true/false-anchor pair loss;
M1 adds `0.01` posterior-coherence KL. AdamW uses LR `3e-4`, weight decay
`0.01`, effective batch 32, clipping 1, at most 30 epochs, and patience 5.

Promotion requires both seeds to improve H2--H4, the mean-seed paired
1,000-request-bootstrap lower bound to exceed zero, and neither H4 nor H4
mismatch to regress. M0/M1 failure does not exhaust the no-recapture path.

### R0 — teacher-forced one-step dynamics

For every all-node H2--H4 trajectory, use true `q_l`, true selected IDs and
execution weights, and causal layer-conditioned branch context to predict
`q_{l+1}` and the next selected set. Report a learned layer-affine query-only
control from the same model.

With rejected all-node direct translation recall `0.565437`, define:

\[
G_{route}=\frac{R_{R0}-0.565437}{1-0.565437}.
\]

R0 passes only if `G_route >= 0.50` (`R_R0 >= 0.7827185`), the paired
request-bootstrap lower bound versus the frozen parent is positive, and every
H2/H3/H4 point gain is nonnegative. Failure justifies a new router-blind
content capture; it does not justify widening the ranker.

### R1 — true-seed closed loop

Seed layer 0 with the true target query. Route forcing decays linearly from
1 to 0 over the first 60% of optimizer steps; the final 40% are fully closed
loop. R1 must retain at least 80% of R0's lift and lose no more than 0.05 on
any horizon. A failure triggers one preregistered noise/self-conditioning
remediation before model width changes.

### R2 — causal MTP-seeded rollout

Seed only from causal multi-channel branch inputs. No target tensor enters the
initializer or serving rollout. Initializer/channels/adapters train first;
the shared transition core opens at epoch 6 at one-tenth the head LR.

R2 may enter joint factual training only if:

- route recall exceeds `0.557170` with positive paired lower bound;
- factual H2--H4 C64 is at least `0.914467`;
- factual H4 C64 is at least `0.898477`.

### Joint factual training

Train exact factual mixture, expert-conditioned aggregation, branch
initializer/adapters, and then the shared core at one-tenth LR. All-node labels
remain transition supervision and all 32 causal nodes remain in the deployed
mixture. Budget 16 limits branch-specific semantic labels, not runtime
evidence.

Only a joint pass of `0.98 / 0.97 / 0.93` opens the final C64 axial ranker.
At C64 `0.98`, an H2--H4 Recall@8 allocation around `0.89` still requires
about `90.8%` recovery of contained true slots, so final ordering remains a
substantial separate problem.

## 5. Artifact and operational contract

Every stage uses a fresh exclusive directory and writes:

- immutable `run_manifest.json` before work, with `optimizer_constructed=false`;
- `MEMORY_AUTOTUNE.json` before optimizer construction;
- `EPOCH_ZERO_AUDIT.json`;
- separate `OPTIMIZER_START.json` when applicable;
- fsynced epoch metrics;
- best checkpoint;
- per-request prediction rows;
- `STAGE_RESULT.json`;
- `SHA256SUMS`.

Inputs and output staging run on local NVMe. Every input hash is verified after
copy, all training writes stay local, and finalized checksum-bound artifacts
are mirrored immutably to persistent storage. Loaders use
`persistent_workers=False`.

A 24-GB RTX 3090 is expected to be sufficient. Each trainer performs a real
forward/backward microbatch search and accepts the largest divisor whose peak
reserved memory is no more than 21 GiB. Dynamics tries `8,4,2,1`; aligners try
`32,16,8,4,2,1`. Ask for an RTX PRO 6000 only if microbatch 1 exceeds the cap
after checkpointing, or if a later gate explicitly authorizes target replay.

## 6. Stop conditions and ranker policy

Do not train the final ranker when:

- native factual Top8 is below 0.89;
- R0 cannot learn true-state one-step dynamics;
- R1 collapses in closed loop after the declared remediation;
- R2 fails to beat the selected parent route/C64 values;
- joint learned C64 fails `0.98 / 0.97 / 0.93`.

If R0 fails, the next capture is narrowly specified: a low-rank
router-invisible content sketch plus projected MoE and attention transition
deltas. No broad recapture is pre-authorized.

## 7. Implementation inventory

- `harp_rtt/route_ceiling.py`
- `harp_rtt/factual_mixture.py`
- `harp_rtt/factual_branch_attention.py`
- `harp_rtt/route_dynamics.py`
- `harp_rtt/deltaroute_batch.py`
- `harp_rtt/deltaroute_training.py`
- `harp_rtt/deltaroute_metrics.py`
- `runpod/export_harp_deltaroute_v4_ceiling_bundle.py`
- `runpod/evaluate_harp_deltaroute_v4_ceiling.py`
- `runpod/train_harp_deltaroute_v4.py`
- `runpod/aggregate_harp_deltaroute_v4_aligners.py`
- `runpod/train_harp_deltaroute_v4_dynamics.py`

The first GPU result belongs in a separate execution ledger. This architecture
document records no unrun result as evidence.
