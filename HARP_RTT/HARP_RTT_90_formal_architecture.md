# HARP-RTT-90 v1.0

> **Active information gate (2026-08-08):** the four-path v2 B1 result failed
> all promotion gates. The active plan is
> [HARP_RTT_B1_5_EXACT_SET_ORACLE_PLAN_20260808.md](HARP_RTT_B1_5_EXACT_SET_ORACLE_PLAN_20260808.md):
> correct the selected-set oracle, measure the all-node adaptive-32 ceiling, and
> confirm the minimum sufficient path budget before any translator or ranker training.
>
> **Superseded execution instruction (2026-08-07):** the later pause evidence
> and the v2 causal-information pilot supersede Phase 1's instruction to resume
> the existing C64 run. That run is a frozen measurement baseline. Its same-input
> reranker plateaued while the C64 oracle remained high, and prefix-mismatch rows
> exposed an additional candidate-generation failure. The active execution plan
> is [HARP_RTT_V2_CAUSAL_BRANCH_PILOT_20260807.md](HARP_RTT_V2_CAUSAL_BRANCH_PILOT_20260807.md),
> which treats candidate identification and within-pool ranking as separate
> bottlenecks and gates branch-rich counterfactual capture before further ranker
> training.

## Branch-Aware Router-Query Trajectory Teacher with Exact-\(k\) Axial Reranking

**Status:** proposed next accuracy-first architecture after the HARP-8T approximately 0.79 H1--H4 endpoint

**Primary target:** Qwen3.6-35B-A3B, 40 routed-MoE layers, 256 experts per layer, top-8 routing

**Primary metric:** request-macro SlotRecall@8 averaged over horizons H1--H4

**Target:** mean H1--H4 SlotRecall@8 at least 0.90, with H4 at least 0.87
**Deployment contract:** the native target router remains authoritative; this model only ranks and calibrates future expert-use probabilities for a separate scheduler

---

## 1. Executive architectural decision

The next model should not be a wider version of the current HARP generator and should not be only a wider fixed-candidate reranker. The repository evidence indicates that the current system has already found a high-recall candidate set but cannot identify the true eight accurately enough within that set.

The proposed model is therefore a three-stage residual extension of the validated HARP checkpoint:

1. **Branch-aware future-token encoder.** Use the exact committed token at H1 and an adaptive native-MTP token tree for H2--H4, preserving branch identity, path probability, hidden/fused state, router input, and router logits.
2. **Router-query trajectory generator.** Predict the future router-visible query at every horizon and target layer, score all 256 experts through frozen centered-router geometry plus an unconstrained dense residual, and refine the 40-layer route trajectory with two self-conditioned iterations.
3. **Rich C64 axial set reranker.** Rerank a 64-expert candidate union using expert-specific route history, branch evidence, router geometry, transition support, generator context, within-layer set attention, cross-layer route summaries, and cross-horizon dependence. Corrections are residual and zero-initialized over the current generator.

The model emits all 256 final scores and exact-cardinality inclusion marginals. The scheduler remains responsible for deciding how many experts to retain or prefetch.

---

## 2. Exact objective and success criterion

For committed source token position \(t\), horizon \(h\in\{1,2,3,4\}\), target layer \(l\in\{0,\ldots,39\}\), and expert \(e\in\{0,\ldots,255\}\), the model emits

\[
 s_{t,h,l,e}\in\mathbb R.
\]

The predicted target set is

\[
 \widehat S_{t,h,l}=\operatorname{Top8}_e(s_{t,h,l,e}).
\]

The primary metric is

\[
 R_h=\mathbb E_{t,l}\left[\frac{|\widehat S_{t,h,l}\cap S^T_{t+h,l}|}{8}\right],
\qquad
 \bar R_{1:4}=\frac14\sum_{h=1}^{4}R_h.
\]

The acceptance target is

\[
 \boxed{\bar R_{1:4}\ge 0.90}
\]

with the anti-gaming constraint

\[
 \boxed{R_4\ge 0.87.}
\]

A useful engineering allocation is

\[
 (R_1,R_2,R_3,R_4)\approx(0.93,0.91,0.89,0.87),
\]

which averages exactly 0.90. These are design budgets, not promised outcomes.

This document interprets “90% accuracy” as SlotRecall@8. A 90% exact-set match rate would be a materially different and much harder target.

---

## 3. Quantitative diagnosis of the current endpoint

At the published pause point, the best generator is approximately

\[
 (0.8217,0.8025,0.7861,0.7622),
\qquad \bar R_{1:4}=0.7931.
\]

The partial C32 reranker is approximately 0.7942. The completed C64 pool has oracle candidate coverage

\[
 C_{64}=0.9741
\]

averaged over H1--H4, with H4 coverage approximately 0.9563.

Consequently:

\[
 \frac{0.7942}{0.9741}\approx0.8153
\]

of candidate-contained true slots are currently recovered, whereas the 0.90 target requires

\[
 \frac{0.90}{0.9741}\approx0.9239.
\]

Equivalently, the new model must remove about 58.8% of the current within-pool ordering error. The hard ceiling is not presently candidate recall; it is conditional ranking quality.

This conclusion determines the architecture. Increasing C from 64 to 128 cannot by itself deliver the target. The candidate representation and conditional ranker must become much more informative.

---

## 4. Why the current model is likely to plateau

### 4.1 Missing exact next-token information

At a token-end decision point, the target has already selected the committed token \(x_{t+1}\). This token and its frozen target embedding are causal inputs for H1. The current compact HARP data/model interface does not include token IDs or embeddings.

The new model MUST define a token-end profile in which

\[
 e_{t+1}=\operatorname{Embedding}(x_{t+1})
\]

is an explicit input. An earlier pre-sampling profile is a separate model and must use a token distribution rather than the realized token.

### 4.2 One greedy MTP chain collapses future-token uncertainty

The current compact corpus exposes six depth-indexed states from one native MTP chain. It does not model alternative token branches. A wrong token at depth one contaminates all deeper states and is a plausible cause of H3/H4 decay.

The next capture MUST preserve an adaptive MTP tree with branch and parent identity. A fixed node budget should be allocated deep-and-narrow in confident contexts and shallow-and-wide in uncertain contexts.

### 4.3 Critical state channels are compressed or absent

The current compact model receives one rank-128 target-state channel and one rank-128 MTP-state channel. It does not separately receive the exact target normalized router input, router-visible coordinates, router-blind content, MTP fused state, MTP router input, or branch token embeddings.

A generic PCA basis is not guaranteed to preserve the low-dimensional directions that control top-8 boundaries. The new corpus should store exact centered-router coordinates and a separate content sketch.

### 4.4 Source softmax forces complementary evidence to compete

The current model forms a softmax over route, target-state, and MTP sources. The source weights sum to one. This is unnecessarily restrictive when route history, exact-token evidence, target state, and MTP branches can all be simultaneously useful.

The next model uses an anchored additive fusion with independent sigmoid/vector gates. No optional source is required to reduce another source’s weight merely by becoming useful.

### 4.5 The output is dense but not geometry-aware

The current dense per-horizon/per-layer output head is expressive and should be retained as a residual anchor. However, it does not explicitly exploit the frozen target router’s expert geometry. The next model predicts a future router query and scores experts through the exact centered-router row space, while retaining a free dense residual so geometry is not a hard bottleneck.

### 4.6 The current reranker is information-poor

The current candidate reranker receives generator scores, current scores/ranks, copy gates, three source gates, expert IDs, and optionally one global generator context. It does not receive candidate-specific MTP evidence, eight-token route histories, branch probabilities, router-row geometry, token embeddings, transition support, or cross-horizon/cross-layer structure. It also reranks each \((h,l)\) set independently.

A C64 pool with 0.974 coverage needs a substantially richer conditional scorer, not simply more epochs on the same ten scalar features.

### 4.7 The objective is not the exact target metric

The current loss combines router KL, pairwise boundary loss, balanced inclusion BCE, centered score regression, and a latent auxiliary. It does not use exact-cardinality set likelihood or a differentiable approximation to SlotRecall@8. Its sigmoid “inclusion probabilities” do not satisfy the required sum of eight.

The next model makes exact-\(k\) set likelihood and soft Recall@8 the primary objectives. Router-score distillation remains auxiliary.

---

## 5. Causal input contract

### 5.1 Token-end timing

The primary HARP-RTT-90 profile executes after target token \(t\) has completed and token \(x_{t+1}\) has been selected, but before target token \(t+1\) begins its routed-expert work.

Allowed inputs are:

1. all target routes and target state summaries through committed token \(t\);
2. the exact committed next-token ID \(x_{t+1}\) and its frozen embedding;
3. MTP tree nodes produced from the authoritative prefix through \(t\) and, where the runtime supports it, rooted on the exact \(x_{t+1}\);
4. request-local causal route and transition statistics;
5. frozen target router matrices and derived descriptors.

Forbidden inputs are:

1. any target route, hidden state, router input, or expert output from \(t+1\) onward;
2. the realized H2--H4 committed token IDs;
3. eventual branch acceptance as an input;
4. future cache or transfer outcomes.

### 5.2 Required target-route history

Store eight committed-token route grids:

\[
 Z_t\in\mathbb R^{40\times 8\times256},
\]

with centered full logits, selected IDs, selected execution weights, entropy, top-8/top-9 gap, top-1/top-2 gap, and availability masks.

### 5.3 Required target-state channels

For every current token/layer, capture the authoritative tensors needed to derive:

- pre-attention or layer-input residual;
- post-attention/pre-MoE residual;
- exact normalized router input \(a^T_{t,l}\);
- post-MoE layer residual;
- routed residual and shared residual separately.

For the centered target router

\[
 \bar W_l=C_EW_l,
 \qquad
 C_E=I-\frac1{256}\mathbf1\mathbf1^\top,
\]

compute an SVD

\[
 \bar W_l=U_l\Sigma_lV_l^\top.
\]

Because \(\operatorname{rank}(\bar W_l)\le255\), store exact router-visible coordinates

\[
 q^T_{t,l}=V_l^\top a^T_{t,l}
\]

at the retained numerical rank. These coordinates preserve all centered-router information. Separately store a 128- or 256-dimensional training-split content sketch of

\[
 a^{\perp,T}_{t,l}=(I-V_lV_l^\top)a^T_{t,l}.
\]

Future \(q^T_{t+h,l}\) values are labels only.

### 5.4 Required exact-token source

Store:

- committed token ID \(x_{t+1}\);
- frozen target embedding \(e_{t+1}\), or enough metadata to recover it exactly;
- final target hidden state that produced \(x_{t+1}\);
- sampling/greedy mode and sampling parameters.

### 5.5 Required MTP tree

For every MTP tree node, store:

- node ID, parent ID, branch ID, depth, target-position mapping;
- token ID and frozen target token embedding;
- path log probability and local token probability;
- MTP hidden state;
- MTP fused state;
- MTP router input;
- complete MTP router logits;
- top vocabulary probabilities or a calibrated compact vocabulary representation;
- source-ready timestamp;
- structural validity, conditioning class, and exact-prefix hash;
- acceptance label as a label only.

The tree root for H1 is the exact committed \(x_{t+1}\). H2--H4 branches are produced by the native MTP path. Use a maximum of 32 nodes for the first teacher and ablate 16 and 64.

### 5.6 Counterfactual branch supervision

For a subset of training prefixes, teacher-force the frozen target model on the highest-probability MTP branches and capture their target router inputs and routes. This creates valid branch-conditioned supervision rather than training branch states only through the one realized continuation.

Counterfactual branches are training data only and are weighted by their MTP path probability or a clipped importance weight.

---

## 6. Static expert and router descriptors

For target layer \(l\), define the fixed router-derived expert key

\[
 K_l=U_{l,r}\Sigma_{l,r}\in\mathbb R^{256\times r}.
\]

The row \(K_{l,e}\) is the fixed key for expert object \((l,e)\). Augment it with:

- log router-row norm;
- a learned 32-dimensional residual identity embedding \(\epsilon_{l,e}\), initialized to zero;
- optional fixed descriptors of the expert input-side weights, after an explicit ablation.

Expert IDs are layer-specific objects. Numerical expert ID \(e\) at two different layers is not treated as one shared object.

---

## 7. Module A: route-grid encoder

### 7.1 Cell encoding

For token lag \(\tau\) and layer \(l\), construct

\[
 r_{\tau,l}=\operatorname{CellMLP}
\left[
 P_z\bar z^T_{t-\tau,l};
 \operatorname{SetPool}(S^T_{t-\tau,l},\alpha^T,K_l);
 \xi_{t-\tau,l};
 e_\tau;e_l;e_{\tau(l)}
\right]
\in\mathbb R^{384}.
\]

The complete 256 logits are retained through a learned projection, while an explicit selected-set key pool gives the model direct expert-object information.

### 7.2 Axial history modeling

Apply:

1. two temporal Transformer blocks over the eight lags independently for each layer;
2. two layer Transformer blocks over the resulting 40 layer states;
3. one local convolution/attention block over neighboring layers to retain short-range layer phase.

Output

\[
 R_l\in\mathbb R^{384}.
\]

Do not discard candidate-specific history. The later candidate scorer also gathers the raw history logits/ranks for each candidate expert.

---

## 8. Module B: target-state control/content encoder

For every target layer, separately project:

- current router-visible coordinate;
- router-blind content sketch;
- post-attention/pre-MoE summary;
- post-MoE/routed residual summary;
- shared residual summary.

Each source receives its own RMSNorm and projection to width 256. Fuse them with independent vector gates:

\[
 X_l=X_l^{\mathrm{anchor}}
+g_l^{\mathrm{ctrl}}\odot P_{\mathrm{ctrl}}q^T_{t,l}
+g_l^{\mathrm{blind}}\odot P_{\mathrm{blind}}b^T_{t,l}
+g_l^{\mathrm{post}}\odot P_{\mathrm{post}}p^T_{t,l}.
\]

Apply two cross-layer Transformer blocks. Output

\[
 X_l\in\mathbb R^{384}.
\]

No source softmax is used.

---

## 9. Module C: adaptive token/MTP tree encoder

### 9.1 Node representation

For tree node \(n\), construct

\[
 b_n^0=\operatorname{MLP}
[
 P_hh_n^M;
 P_ff_n^M;
 P_aa_n^M;
 P_rr_n^M;
 P_ee(x_n);
 \eta_n;
 e_{d_n};e_{\mathrm{branch}(n)}
]
\in\mathbb R^{512}.
\]

Here \(h^M,f^M,a^M,r^M\) denote hidden, fused, router-input, and centered-router-logit channels. \(\eta_n\) contains path probability, local probability, entropy, depth, readiness, and validity features.

### 9.2 Tree attention

Use four width-512 Transformer blocks with an ancestor/parent attention bias:

- every node attends to its ancestors;
- siblings may attend to one another through a separate branch-comparison head;
- horizon queries attend most strongly to nodes whose mapped target position equals the horizon, without hard exclusion of neighboring depths.

### 9.3 Adaptive expansion

Use a fixed node budget. Expand high-confidence branches deeper and high-entropy branches wider. The expansion policy is outside the predictor’s learned router output and is frozen before test evaluation.

Output all branch node states \(B_n\), branch posterior logits \(\rho_n\), and horizon-conditioned branch sets \(\mathcal B_h\).

---

## 10. Module D: direct horizon/layer endpoint decoder

Initialize one query per horizon and target layer:

\[
 u^0_{h,l}=W_RR_l+W_XX_l+e_h+e_l
+\operatorname{CrossAttn}(q^0_{h,l},\{B_n\})
+W_ee_{t+1}.
\]

The exact next-token term is used for all horizons but receives a separate horizon gate. H1 has a dedicated short-horizon adapter.

Use six alternating axial blocks:

1. layer attention over \(l=0\ldots39\) independently for each horizon;
2. causal horizon attention over H1--H4 independently for each layer;
3. cross-attention to the MTP tree;
4. gated SwiGLU feed-forward.

The horizon attention is causal in prediction order but every horizon retains a direct source skip, so H3/H4 do not depend exclusively on earlier predicted routes.

Use separate decoder adapters for:

- short horizons H1--H2;
- long horizons H3--H4.

H5--H8 are not optimized in the primary 90% model. They may be attached later with the main encoders frozen.

---

## 11. Module E: hybrid router-geometry and free score head

### 11.1 Router-query head

Predict a router-visible query coordinate

\[
 \hat q_{b,h,l}=Q_{h,l}(u_{b,h,l})\in\mathbb R^{r_l}
\]

for each relevant branch.

The geometry score is

\[
 s^{\mathrm{geo}}_{b,h,l}=K_l\hat q_{b,h,l}\in\mathbb R^{256}.
\]

### 11.2 Unconstrained direct residual

Retain the validated dense HARP score as an anchor and add a free nonlinear residual:

\[
 s^{\mathrm{free}}_{b,h,l}
=W^{(2)}_{h,l}\operatorname{SwiGLU}(W^{(1)}_hu_{b,h,l}).
\]

### 11.3 Persistence and transition anchors

Compute separately:

- current-route persistence score;
- same-layer token-transition support;
- previous-layer route-transition support;
- request-local streak/frequency support.

These are features or gated residuals, not fixed priors that override the neural score.

### 11.4 Branch-conditioned score

\[
 s^{(0)}_{b,h,l}
=s^{\mathrm{HARP}}_{h,l}
+\gamma^{\mathrm{geo}}_{h,l}s^{\mathrm{geo}}_{b,h,l}
+\gamma^{\mathrm{free}}_{h,l}s^{\mathrm{free}}_{b,h,l}
+\gamma^{\mathrm{trans}}_{h,l}s^{\mathrm{trans}}_{b,h,l}.
\]

All new final residual projections and gates are zero-initialized, making the architecture exactly equal to the loaded HARP checkpoint at step zero.

The router-query loss is auxiliary. The direct exact-set score remains primary, avoiding a hard feature-regression bottleneck.

---

## 12. Module F: branch mixture

For each branch, compute an exact-8 subset distribution from its score vector and exact marginal inclusion probabilities

\[
 \mu_{b,h,l,e}=P(e\in S\mid b,h,l).
\]

Let normalized branch posterior weights be

\[
 \pi_{b,h}=\operatorname{softmax}_{b\in\mathcal B_h}(\rho_{b,h}).
\]

The branch-mixture marginal is

\[
 \boxed{
 \bar\mu_{h,l,e}=\sum_{b\in\mathcal B_h}\pi_{b,h}\mu_{b,h,l,e}.
 }
\]

Because every branch distribution selects exactly eight experts,

\[
 \sum_e\bar\mu_{h,l,e}=8.
\]

For expected SlotRecall@8, the Bayes action is to rank experts by these conditional inclusion probabilities. The branch mixture therefore preserves token-path uncertainty instead of collapsing it to the single greedy MTP path.

---

## 13. Module G: two-round route-trajectory refinement

The current model predicts each endpoint largely independently. HARP-RTT-90 adds a bounded dependent residual.

At refinement round \(r\), derive a soft selected-set embedding

\[
 y^{(r)}_{h,l}=\frac18\sum_e\mu^{(r)}_{h,l,e}K_{l,e}.
\]

Form route-summary tokens

\[
 g^{(r)}_{h,l}=\operatorname{MLP}[u_{h,l};y^{(r)}_{h,l};R_l;X_l].
\]

Apply:

1. a layer-path Transformer over \(g_{h,0:39}\);
2. a causal horizon Transformer over \(g_{1:4,l}\);
3. explicit train-only transition messages from predicted previous-layer and previous-horizon marginals.

Produce a correction

\[
 s^{(r+1)}_{h,l}=s^{(0)}_{h,l}+\Delta^{(r)}_{h,l}.
\]

Use exactly two rounds. Every correction is residual to the direct score \(s^{(0)}\), so rollout errors cannot erase the direct endpoint prediction.

During training, feed the model’s own predicted soft sets from the first pass. A teacher-forced set may be used during initial warm-up, then annealed away. This is the route analogue of training-time self-conditioning.

---

## 14. Module H: C64 candidate union

Build a 64-object candidate pool for each \((h,l)\) from the union of:

1. current HARP dense score;
2. geometry/control score;
3. branch-mixture marginal;
4. trajectory-refined score;
5. current-route persistence;
6. conditional transition support.

Normalize each source by a frozen training-split temperature before union. Deduplicate by layer-specific expert object. Fill remaining positions from the strongest aggregate score.

The target gate is:

\[
 C_{64}^{1:4}\ge0.985,
\qquad
 C_{64}^{H4}\ge0.97.
\]

The current 0.974/0.956 pool is already sufficient for 0.90 mean recall, but the higher gate supplies margin for ranker errors.

---

## 15. Module I: rich axial candidate reranker

### 15.1 Candidate token

For candidate object \((h,l,e)\), construct a 384-dimensional token from:

**Static object features**

- fixed router key \(K_{l,e}\);
- router-row norm;
- zero-initialized learned object residual \(\epsilon_{l,e}\);
- layer, phase, block-type, and horizon embeddings.

**Generator evidence**

- HARP score, rank, and margins to ranks 8, 16, 32, and 64;
- geometry score and free residual score;
- branch-mixture marginal and entropy;
- trajectory correction;
- copy/persistence gate.

**Expert-specific causal history**

- centered router logit and rank for this object over the preceding eight committed tokens;
- membership, execution weight, streak, and recent count;
- same-layer token-transition support;
- previous-layer route-transition support.

**Branch-specific evidence**

- mapped MTP score for this target object from every relevant branch/depth;
- max, log-sum-exp, mean, variance, and path-probability-weighted support;
- probability mass of branches supporting the candidate;
- branch acceptance/confidence features, detached from acceptance labels.

**Context interactions**

- dot product of candidate key with predicted router query;
- dot product with current control and content projections;
- exact next-token compatibility for H1;
- global endpoint query \(u_{h,l}\).

Raw MTP expert ID is not assumed to equal a target-layer expert ID. MTP evidence is mapped through branch hidden/router-query translators before it becomes a candidate-specific target score.

### 15.2 Efficient axial structure

A full attention operation over every \((h,l,e)\) node is unnecessary. Use:

1. **two permutation-equivariant Set-Transformer blocks** over the C64 candidates independently for each \((h,l)\);
2. pool one route-summary token from each set;
3. **two axial summary blocks** across the 40 layers and four horizons;
4. broadcast the refined summary back to each candidate;
5. one cross-horizon block for the same layer-specific object when it appears at multiple horizons;
6. one zero-initialized scalar residual head.

No candidate-rank positional embedding is used. Rank is supplied only as a scalar feature. Candidate permutation must only permute outputs.

### 15.3 All-256 output preservation

The reranker produces corrections only for C64 objects:

\[
 s'_{h,l,e}=s^{(2)}_{h,l,e}+\mathbb1[e\in\mathcal C_{64}]\Delta s^{\mathrm{rank}}_{h,l,e}.
\]

All 256 scores remain in the final exact-8 normalizer. This avoids treating the candidate pool as the complete expert universe and keeps a calibrated outside-pool probability mass.

---

## 16. Exact-8 probability model

For score vector \(s\in\mathbb R^{256}\), define

\[
 P(S\mid s)=
 \mathbb1[|S|=8]
 \frac{\exp(\sum_{e\in S}s_e)}
 {Z_8(s)},
\]

where

\[
 Z_8(s)=\operatorname{ESP}_8(\exp s).
\]

Compute \(\log Z_8\) and exact marginals in FP32 with an \(O(Ek)\) log-space dynamic program. The final marginal vector satisfies

\[
 0\le\mu_e\le1,
\qquad
 \sum_e\mu_e=8.
\]

The final ranking is by \(\mu_e\), not by independent sigmoid outputs. Positive score-temperature calibration does not change ranking but supplies scheduler-facing confidence.

---

## 17. Training objective

### 17.1 Primary exact-set likelihood

\[
 \mathcal L_{\mathrm{set}}
 =-\sum_{e\in S^*}s_e+\log Z_8(s).
\]

This is the primary proper likelihood and directly enforces exact cardinality.

### 17.2 Differentiable SlotRecall@8

Use a differentiable soft top-8 operator \(m_\tau(s)\), such as LapSum, and define

\[
 \mathcal L_{\mathrm{recall}}
 =1-\frac18\sum_{e\in S^*}m_{\tau,e}(s).
\]

Start this coefficient at zero, introduce it after the exact-set head is stable, and anneal \(\tau\) gradually. It is an auxiliary metric-alignment loss, not the only objective.

### 17.3 Boundary loss

Concentrate on the weakest true expert and strongest false experts:

\[
 \mathcal L_{\mathrm{boundary}}
 =\operatorname{softplus}
 \left(
 m+\operatorname{LSE}_{j\notin S^*}^{\mathrm{hard}}s_j
 -\operatorname{SoftMin}_{e\in S^*}s_e
 \right).
\]

Mine hard negatives from model ranks 6--64 and target ranks 9--64.

### 17.4 Router-score distillation

Use forward KL at temperature 1 as an auxiliary:

\[
 \mathcal L_{\mathrm{router}}
 =\operatorname{KL}
 \left(
 \operatorname{softmax}(z^*/1)
 \Vert
 \operatorname{softmax}(s/1)
 \right).
\]

### 17.5 Router-query auxiliary

\[
 \mathcal L_{\mathrm{query}}
 =\operatorname{Huber}(\hat q,q^*)
 +\lambda_{\cos}(1-\cos(\hat q,q^*)).
\]

It is deliberately low-weight and cannot become the sole route path.

### 17.6 Branch and token losses

Use direct token/path supervision for the adaptive tree:

- branch-prefix cross entropy;
- calibrated path/acceptance BCE;
- optional counterfactual branch route loss.

### 17.7 Trajectory consistency

Supervise each refinement round and penalize destructive corrections:

\[
 \mathcal L_{\mathrm{traj}}
 =\sum_{r=1}^{2}\mathcal L_{\mathrm{set}}(s^{(r)},S^*)
 +\lambda_\Delta\|\Delta^{(r)}\|_2^2.
\]

### 17.8 Default combined objective

After warm-up:

\[
\boxed{
\begin{aligned}
\mathcal L={}&
1.00\mathcal L_{\mathrm{set}}
+0.50\mathcal L_{\mathrm{recall}}
+0.25\mathcal L_{\mathrm{boundary}}
+0.15\mathcal L_{\mathrm{router}}\\
&+0.10\mathcal L_{\mathrm{query}}
+0.10\mathcal L_{\mathrm{branch}}
+0.05\mathcal L_{\mathrm{traj}}.
\end{aligned}
}
\]

These are initial engineering coefficients. Measure component gradient norms. The sum of all auxiliary gradients on shared parameters must not exceed half the primary exact-set gradient for 500 consecutive steps; reduce coefficients when it does.

### 17.9 Horizon balance

Use equal H1--H4 primary weights plus a worst-horizon term:

\[
 \mathcal L_H=\frac14\sum_{h=1}^{4}\mathcal L_h
 +0.25\max_h\mathcal L_h.
\]

This discourages an average-score improvement obtained by sacrificing H4. H5--H8 have zero primary weight during the 90% program.

---

## 18. Training schedule

### Phase 0: oracle and probe ladder

Before expensive training, measure:

1. current C32/C64/C128 candidate coverage;
2. exact-next-token embedding probe;
3. single-chain versus adaptive-tree branch coverage;
4. true future-router-input reconstruction sanity check;
5. two-layer predictor from true future pre-attention state;
6. prefix-conditional Bayes ceiling under the deployed decoding mode;
7. C64 ranker overfit on 128 requests.

If a model with true future router input does not reconstruct top-8 essentially exactly, the capture or indexing is wrong.

### Phase 1: preserve the completed C64 baseline (superseded)

Do not resume the already exported C64 reranker. The later pause audit completed
the measurement needed from this baseline and found a same-input plateau. Preserve
its artifacts unchanged and follow the v2 branch-rich causal-information pilot.

### Phase 2: function-preserving residual generator

1. load the best temperature-1 HARP checkpoint;
2. freeze every existing parameter;
3. attach exact-token, tree, control/geometry, independent-gate, and pointwise residual modules with zero final projections;
4. train only new modules;
5. retain epoch zero as the incumbent checkpoint.

### Phase 3: low-learning-rate joint generator

Unfreeze the upper route/state encoders and dense score heads with a learning rate at least 5--10 times lower than the new modules. Train direct, branch, and two-round trajectory outputs jointly.

### Phase 4: C64 rich ranker

Freeze the generator and train the axial C64 ranker. During early training, ensure every true candidate already present in C64 remains in the local ranker set. Anneal to purely predicted candidate sets before validation.

### Phase 5: joint ranker refinement

Unfreeze only the generator’s final query/score adapters and the reranker. Candidate selection remains stop-gradient. Select by H1--H4 Recall@8, not candidate coverage.

### Phase 6: seed ensemble and distillation

Train three independent seeds. Average calibrated inclusion logits or marginals. If the ensemble reaches the target, distill it into a smaller runtime student using exact-set, marginal, and top-8 boundary distillation.

---

## 19. Data scale

The current 1,229-request training split is adequate for architecture development but small for a 10-point absolute gain across four future horizons.

Recommended minimum next corpus:

- at least 10,000 independent, lineage-deduplicated requests;
- at least 250,000 usable committed-token source positions;
- approximately 10 million target-layer examples per horizon;
- multiple domains, languages, context lengths, and deterministic/stochastic decoding strata;
- branch tree and exact-token capture from the same execution;
- counterfactual branch supervision on a deterministic subset.

Use request/lineage splits. Never split token rows from one request across train and validation/test. Keep a dedicated calibration partition and a sealed outer test set.

---

## 20. Model-selection protocol

Use this ordered selection tuple:

1. mean H1--H4 request-macro SlotRecall@8;
2. minimum H1--H4 SlotRecall@8;
3. H4 SlotRecall@8;
4. exact-set NLL;
5. C64 coverage;
6. router KL.

Candidate coverage is a gate, not the primary checkpoint criterion. The previous candidate-first rule can discard a genuine Recall@8 improvement.

Report paired complete-request bootstrap intervals over at least 1,000 replicates. Do not open the test set until the architecture, seed policy, calibration, and candidate width are frozen.

---

## 21. Mandatory ablations

Run in this order:

1. unchanged C64 reranker to convergence;
2. exact-8 + soft Recall loss versus current BCE/pairwise/KL;
3. exact committed H1 token embedding versus masked token;
4. adaptive branch tree versus one greedy MTP chain;
5. counterfactual branch supervision versus realized branch only;
6. router-control coordinates versus generic PCA target state;
7. independent source gates versus source softmax;
8. geometry head versus dense head versus hybrid;
9. direct endpoints versus direct plus two-round route refinement;
10. local set reranker versus local + layer/horizon axial summaries;
11. C32 versus C64;
12. H1--H4-only training versus shared H1--H8 training;
13. 2k, 5k, and 10k request scaling;
14. single model versus three-seed ensemble;
15. teacher versus distilled runtime student.

Do not spend another broad sweep on MTP PCA rank alone before testing exact token, tree branches, and router-control coordinates; the existing rank sweep did not yield a material gain.

---

## 22. Kill gates and interpretation

1. **Candidate gate:** if the new C64 union cannot exceed 0.985 mean and 0.97 H4 coverage, improve the generator before enlarging the reranker.
2. **Ranker gate:** if the rich ranker cannot exceed 0.85 on validation despite C64 coverage above 0.97, the current causal features are insufficient or misaligned; do not merely increase ranker depth.
3. **Branch gate:** if an adaptive tree does not improve H3/H4 token-path and expert candidate coverage, remove it and investigate MTP alignment.
4. **Control gate:** if exact router-control coordinates do not beat generic PCA after a matched warm start, retain them only as an audit/auxiliary channel.
5. **Sampling ceiling:** under stochastic decoding, estimate the prefix-conditional Bayes ceiling. If it is below 0.90, optimize expected marginal recall rather than claiming a deterministic 0.90 is reachable.
6. **Latency gate:** after the teacher reaches the target, distill or prune any module whose delay removes the scheduler’s prefetch window.

---

## 23. Expected result and uncertainty

The current C64 ceiling shows that a 0.90 H1--H4 average is not excluded by candidate generation. It does not guarantee that the present compact inputs contain enough information to identify the correct eight.

The architecture is designed around the four missing ingredients most likely to change the regime rather than add sub-point gains:

1. the exact known H1 token;
2. adaptive branch-aware H2--H4 token futures;
3. exact router-control coordinates and frozen expert geometry;
4. a rich cross-horizon/cross-layer C64 set ranker trained directly for exact top-8 retrieval.

A wider MLP, a larger PCA rank, temperature tuning, static frequency blending, or a wider candidate set alone should be treated as controls, not as the main path to 0.90.

---

## 24. Literature provenance

The design borrows principles, not claimed results, from:

- **Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models**, arXiv:2511.10676: near-router same-layer states, simple layer-specific predictors, and ranking-aware supervision.
- **EAGLE-2**, arXiv:2406.16858; **EAGLE-3**, arXiv:2503.01840; **TALON**, arXiv:2601.07353; **Hydra**, arXiv:2402.05109; **ReDrafter**, arXiv:2403.09919: branch trees, context-dependent expansion, sequential dependence, multi-layer feature fusion, and training on self-generated states.
- **Polysemantic Experts, Monosemantic Paths**, arXiv:2604.17837: router-visible control, router-blind content, and route trajectories as a structured unit.
- **Routers Learn the Geometry of Their Experts**, arXiv:2605.12476: router/expert geometric coupling and expert keys.
- **ProbMoE**, arXiv:2606.01509: exact-cardinality subset distributions and exact marginals.
- **LapSum**, ICML 2025: efficient differentiable top-k selection.
- **A Spatio-Temporal Expert Prefetching Framework**, arXiv:2606.15453, and **Orders in Chaos**, arXiv:2510.05497: temporal and cross-layer route structure.
- **DraftExpert**, arXiv:2607.24434: multi-signal residual/token/router distillation and expansion-aware expert prefetching.
- **SpecPrefetch**, arXiv:2607.24787: preserving the native router while using predictions only for asynchronous movement.

These papers do not establish that HARP-RTT-90 will reach 0.90 on Qwen3.6. They justify the individual architectural principles and the experiment order.
