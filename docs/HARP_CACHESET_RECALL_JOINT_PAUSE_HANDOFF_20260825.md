# HARP Resident-Shadow v2: CacheSet/Recall Joint-Improvement Pause Handoff

Date: 2026-08-25
Status: **paused at the user's request**
Correct target checkpoint-index SHA256: `41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83`

No experiment, capture, training run, evaluation, or model mutation remained active
when this handoff was written. The active audit agents were stopped before the
documentation pass.

## 1. Executive status

The requested deployable outcome has **not** yet been achieved.

The joint target is:

- mean H1--H4 CacheSet recall at least `0.95`;
- mean H1--H4 factual Recall@8 at least `0.90`;
- no serving-efficiency regression relative to the efficient ShadowRoute path;
- no increase in resident expert footprint;
- exactly 3,850 resident layer/expert cells at the current 37.59765625% residency;
- every layer must contain its 64 most frequently selected experts. This last
  invariant is non-negotiable.

What exists at pause:

1. The CacheSet statistic is implemented and audited on retained scores.
2. The old resident-only v2 diagnostic baseline is `0.896260` mean CacheSet and
   `0.828593` mean factual Recall@8.
3. A diagnostic CacheTrajectory candidate reached `0.959037` blind CacheSet,
   but factual Recall@8 collapsed to `0.702573`; it therefore failed.
4. A smaller, faster factorized joint ranker reached `0.950411` CacheSet on its
   four-request calibration split, but only `0.777380` factual Recall@8; it also
   failed and did not open the reused holdout.
5. A correct, immutable, top-64-compliant 3,850-cell resident bundle has been
   extracted and sealed. It has not yet undergone closed-loop evaluation.
6. J-space and expert-transition priors carry measurable structure, but the
   retained request-disjoint evidence shows no useful factual-ranking gain.
7. No new capture was used in this joint-improvement effort.

Consequently, there is currently no honest basis for claiming either a formal
95% CacheSet model or a formal 90% factual model under the final resident
constraint. The 95.9037% CacheSet number is diagnostic and belongs to a model
whose factual score is unacceptable.

## 2. Metric definitions

Let `S_t(l)` be the target model's true top-8 expert set at token `t`, layer `l`,
and let `P_(t+h)(l)` be the predicted top-8 expert set for horizon `h`.

Factual Recall@8 is the ordinary future-set recall:

```text
|S_(t+h)(l) intersect P_(t+h)(l)| / 8
```

CacheSet recall asks only whether the experts that survive from the current set
into the future set are retained by the prediction:

```text
|S_t(l) intersect S_(t+h)(l) intersect P_(t+h)(l)|
---------------------------------------------------
        |S_t(l) intersect S_(t+h)(l)|
```

The reported target statistic is the request-macro mean over valid token/layer
cells. The audit also records pooled numerators and denominators.

These metrics are complementary. CacheSet rewards preservation of survivors;
factual Recall@8 also requires finding future experts not present in `S_t`.
Forcing current-set experts into the prediction can therefore raise CacheSet
while sharply lowering factual Recall@8. That is exactly what happened in the
first 95%-CacheSet candidate.

## 3. Authoritative old diagnostic baseline

The numbers quoted in
`docs/HARP_RESIDENT_SHADOW_V2_PRO_RESULTS_20260815.md` were produced by the
resident-only ShadowRoute v2 architecture at 3,850/10,240 resident cells, or
37.59765625% residency. Rounded, this is the requested 37.5% condition.

The exact retained score audit is:

```text
/workspace/LLM_prefetch_study/artifacts/harp_rtt/
  resident_shadow_v2_cacheset_dev_20260825/evaluations/
  v2_screen_512_score_audit_v2/
```

Its retained score tensor is `cache_set_audit.pt`, SHA256:

```text
587ed9df000fc57117a1f393311b6001988c758f11ae5f1413e0be776e6bc3de
```

### 3.1 Factual and CacheSet scores

| Horizon | Factual Recall@8 | CacheSet request-macro recall | Mean true intersection experts | True intersection total |
| --- | ---: | ---: | ---: | ---: |
| H1 | 0.872931 | 0.929291 | 2.629541 | 53,853 |
| H2 | 0.842950 | 0.901839 | 1.852490 | 37,939 |
| H3 | 0.814264 | 0.885789 | 1.844629 | 37,778 |
| H4 | 0.784229 | 0.868121 | 1.683838 | 34,485 |
| **Mean H1--H4** | **0.828593** | **0.896260** | -- | -- |

The pooled CacheSet recalls were `0.929456`, `0.899338`, `0.881968`, and
`0.865217`; the request-macro values above are the primary comparison metric.

### 3.2 Important validity limitation

This old diagnostic plan has 3,850 cells, but it misses 14 cells that belong to
the mandatory per-layer frequency top-64 sets. It is therefore useful for
architecture diagnosis only and cannot be promoted under the user's final
constraint.

The retained 32-request score audit is also no longer a formal blind set. Its
16 odd/request holdout requests were opened by CacheTrajectory95. Subsequent
use is explicitly marked adaptive and non-promotable.

## 4. Correct resident plan and sealed bundle

The corrected allocation is:

```text
/workspace/LLM_prefetch_study/artifacts/harp_rtt/
  resident_damage_no_capture_20260825/
  core_mass_hitcap_3850_89d6b5f/resident_allocation.json
```

Key lineage:

| Item | Value |
| --- | --- |
| Allocation-file SHA256 | `8b25bdb5a39f08089f8b175ae5f8befa0fe2c3995d5d16bffcc28d2d8cd96536` |
| Membership SHA prefix | `89d6b5f4` |
| Core/frequency SHA prefix | `c090aef6` |
| Source commit | `8c0cde1c272cb1eca82cf90e500a9c73367b06df` |
| Target checkpoint-index SHA256 | `41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83` |
| Mandatory core | 64 experts/layer, 2,560 cells total |
| Optional cells | 1,290 |
| Total cells | 3,850 |
| Residency | 37.59765625% |

Every layer's frequency top-64 is present. This is a hard validation, not a
soft allocation preference.

The corresponding sealed checkpoint bundle is:

```text
/workspace/LLM_prefetch_study/artifacts/harp_rtt/
  resident_damage_no_capture_20260825/
  compliant_resident_bundle_3850_89d6b5f/
```

`BUNDLE_RESULT.json` records:

| Property | Result |
| --- | ---: |
| Validation | passed |
| Layers | 40 |
| Resident cells | 3,850 |
| Logical packed INT4 bytes | 6,433,996,800 |
| Serialized checkpoint bytes | 6,434,309,736 |
| Serialized GiB | 5.992418 |
| Fallback bytes | 0 |
| Target expert loads permitted | false |
| Optimizer constructed | false |
| Closed-loop evaluation started | false |
| Formal validation opened | false |

The bundle was obtained by exact tensor selection from the immutable full-INT4
teacher; there was no optimization, requantization, or approximation. Byte
identity and all mandatory residents were checked during extraction.

## 5. Bottlenecks and measured ceilings

### 5.1 Error decomposition

On the old diagnostic baseline:

- future target slots covered by a resident expert: `68.9424%`;
- factual error caused by target experts being nonresident: `7.7519` points;
- factual error despite the target expert being resident: `9.3887` points;
- the latter is about `54.8%` of all factual errors.

This means static resident coverage is not the sole bottleneck. Omission changes
the closed-loop hidden trajectory, so later routers miss experts even when
those future experts are resident.

Future targets split as follows:

| Target type | Share of target slots | Recall | Contribution to total factual error |
| --- | ---: | ---: | ---: |
| Novel relative to current top-8 | 74.97% | 0.80540 | 14.589 points |
| Survivor from current top-8 | 25.03% | 0.89805 | 2.552 points |

Novel experts account for about `85.1%` of the factual error. Their error grows
with horizon: `10.388`, `13.374`, `15.852`, and `18.740` points for H1--H4.
CacheSet is therefore the easier objective; robust novel-expert ranking is the
main obstacle to 90% factual Recall@8.

The most damaging layers are broadly 23--28 at H1/H2 and 35--39 at H3/H4.

### 5.2 Candidate-set ceiling

The retained 32-candidate pool is not the primary ceiling:

| Condition | Mean factual Recall@8 | Mean CacheSet |
| --- | ---: | ---: |
| Deployed old baseline | 0.828593 | 0.896260 |
| Perfect selection from C32 candidates | 0.935219 | 1.000000 |

The factual candidate oracle crosses 90% already at C19 (`0.901662`). The task
is selecting the right candidates, especially novel ones, rather than widening
the candidate set or adding more speculative nodes.

### 5.3 Full-expert closed-loop ceiling

Executing every expert through the same group-64 INT4 closed loop gives about
`0.916573` mean H1--H4 factual Recall@8, with approximately:

```text
H1 0.944464
H2 0.930142
H3 0.908997
H4 0.882690
```

This violates the footprint constraint, but proves that the ShadowRoute tree
and router machinery can exceed the 90% mean target if omission damage is
removed. Full-expert INT4 itself loses only about `0.882` point relative to its
higher-precision control; layer-11 INT4 costs about `0.830` point. Quantization
is secondary to omission and trajectory drift.

## 6. Approaches executed in this joint-improvement effort

### 6.1 CacheSet instrumentation and retained-score audit

The evaluator was extended to retain exact dense reranking scores and labels,
verify that saved scores reproduce deployed top-8 sets, and report both
request-macro and pooled CacheSet statistics. The retained tensor contract is:

```text
scores       [512, 4, 40, 256]
current_ids  [512, 40, 8]
target_ids   [512, 4, 40, 8]
valid        [512, 4, 40]
```

It does not retain node hidden states, pre-gate router inputs, vocabulary
logits, explicit route history, or per-node posterior components. This limits
faithful no-capture training of changes inside the closed loop.

Outcome: the requested statistic and true intersection counts are available;
the baseline is reproducible, but its old resident allocation is not compliant.

### 6.2 CacheTrajectory95: quota-based joint C32 ranker

Artifact:

```text
/workspace/cachetrajectory_c32_joint_v1/STAGE_RESULT.json
```

Architecture/contract:

- 84,714 parameters;
- 169,428 deployment BF16 bytes;
- 79,872 linear MACs per token/layer cell;
- trained on 12 requests, calibrated on 4, evaluated on 16 retained blind
  requests;
- no new capture;
- joint candidate reranking with a policy over residual scale, current-set bias,
  and a fixed survivor quota.

Blind result:

| Horizon | CacheSet | Factual Recall@8 |
| --- | ---: | ---: |
| H1 | 0.975750 | 0.802966 |
| H2 | 0.950019 | 0.722070 |
| H3 | 0.955558 | 0.661084 |
| H4 | 0.954821 | 0.624170 |
| **Mean** | **0.959037** | **0.702573** |

Outcome: `passed=false`. It demonstrated that 95% CacheSet is easy to obtain by
over-preserving current experts, but this destroys novel-expert recall. It also
opened the old 16-request holdout, making that retained bundle diagnostic only.

### 6.3 DeepSets C32 context proposal

A permutation-invariant candidate-set model was prepared to add cross-expert
context. Before training, matched active-path timing showed:

| Path | Median | p95 |
| --- | ---: | ---: |
| Existing factorized + quota-2 | 3.622 ms | 3.817 ms |
| DeepSets + continuous policy | 3.890 ms | 4.073 ms |
| Regression | +7.4% | +6.7% |

Outcome: rejected before training because it violated the non-regression
constraint. No accuracy claim was made.

### 6.4 Factorized continuous joint set ranker

Local implementation:

```text
.harp-resident-shadow-v2/harp_rtt/cachetrajectory_c32_setrank.py
.harp-resident-shadow-v2/runpod/train_cachetrajectory_c32_setrank.py
.harp-resident-shadow-v2/tests/test_cachetrajectory_c32_setrank.py
```

SHA256 values:

```text
module  248e9d9685e6abefe255cde73efafc1e35a502de5d1cc76181a265bca74edcaf
driver  c24e1daffe56370f268d06f7cd015d4a211b0f3fc1dc9317c1730b3155413331
tests   903b0529da97d4bba5078584a34e8dd8e1655304e2988a4c579170227aaca4f0
```

Remote result:

```text
/workspace/cachetrajectory_c32_setrank_factorized_w20_20260825_v1/
```

Architecture:

- factorized per-candidate topology;
- state width 20, object embedding 8, horizon embedding 4, layer embedding 4;
- factual residual applied only to novel candidates;
- separate survivor head;
- continuous policy with no mandatory fixed survivor quota;
- zero factual residual at initialization;
- 83,514 parameters;
- 167,028 BF16 parameter bytes;
- 42,240 linear MACs/cell.

The BF16 input/LayerNorm dtype mismatch discovered in preflight was fixed. Three
focused tests pass, including BF16 forward from float inputs and mean-CacheSet
policy feasibility when individual horizons are below 95%.

Matched efficiency measurements all favored the new factorized path:

| Measurement | Old | New | Change |
| --- | ---: | ---: | ---: |
| Model median | 2.324 ms | 2.206 ms | -5.08% |
| Model p95 | 2.481 ms | 2.342 ms | -5.61% |
| Policy-inclusive median | 2.918 ms | 2.662 ms | -8.77% |
| Policy-inclusive p95 | 3.036 ms | 2.774 ms | -8.63% |
| Full preprocessing median, 640 cells | 10.549 ms | 9.262 ms | -12.20% |
| Full preprocessing p95, 640 cells | 10.895 ms | 9.429 ms | -13.45% |

Two seeds (`701`, `702`) were run for at most 12 epochs with patience 2. Seed
701 was selected. The old holdout was not opened because calibration missed the
predeclared 88% factual promotion gate.

Calibration BF16 result:

| Horizon | CacheSet | Factual Recall@8 |
| --- | ---: | ---: |
| H1 | 0.994748 | 0.807080 |
| H2 | 0.935453 | 0.853467 |
| H3 | 0.921407 | 0.752832 |
| H4 | 0.950034 | 0.696143 |
| **Mean** | **0.950411** | **0.777380** |

The selected policy used no survivor head at any horizon. H1 also used no novel
residual; H2--H4 used novel residual scale 1.0. The independent scorer therefore
failed to find complementary joint signal.

Checkpoint-only failure analysis showed why:

- survivor base AUC on calibration: `0.9615`;
- base + learned survivor AUC: `0.9646`;
- nevertheless, learned pair accuracy was worse than base at every horizon;
- the head repaired 21--31% of rare base mistakes but broke 0.6--1.25% of the
  much larger correct-pair population;
- novel base AUC on calibration: `0.9413`;
- base + residual AUC: `0.9461`;
- actual novel top-8 factual contribution fell `0.64725 -> 0.64495`;
- candidate-covered novel recall fell `0.93650 -> 0.93312`;
- the learned residual alone was inverted after H1, with calibration AUCs
  `[0.9217, 0.0589, 0.0934, 0.0929]`.

Optimistic split-conditioned controls under mean CacheSet at least 95%:

| Configuration | Training factual / CacheSet | Calibration factual / CacheSet |
| --- | ---: | ---: |
| Learned survivor + oracle novel | 0.9120 / 0.9491, infeasible | 0.9147 / 0.9528 |
| Oracle survivor + learned novel | 0.8435 / 0.9993 | 0.8599 / 0.9988 |
| Learned survivor only | 0.6958 / 0.9503 | 0.7337 / 0.9503 |
| Learned novel only | 0.7396 / 0.9510 | 0.7774 / 0.9504 |

Outcome: hard stop. Novel ordering is the larger bottleneck, while the learned
survivor head is mostly redundant with the base scores. Better pooled AUC is not
sufficient; the architecture must improve within-cell top-k decisions.

### 6.5 J-space and expert-transition investigation

Artifacts:

```text
/tmp/transition_prior_exact512_cv_v1.json
  SHA256 10604e3099e0bd6b4b64ef1ad3362bb6e4c66a79bbaf5d6f8588b04e58688b16
/tmp/transition_prior_b2_factors_v1.pt
  SHA256 2d1772b0589bad466862491fe94f4dc1e74b08a6a4bc185548c1c9f388383c19
```

The B2 corpus supplies 224 training requests, 55,721 transitions, and
570,583,040 expert-pair events. Exact512 used 16 design requests with 4-fold
12/4 request-disjoint cross-validation.

Results relative to base C32 (`0.848108` factual, `0.905556` CacheSet):

| Transition score | Factual delta | CacheSet delta |
| --- | ---: | ---: |
| Shared directed rank 4 | -0.1270 point | +0.5137 point |
| Shared directed rank 8 | -0.1456 point | +0.6182 point |
| Full-rank table | -0.1096 point | +0.6273 point |

Even full-rank in-sample fitting produced only `+0.0345` factual point and
`+0.5478` CacheSet point, then reversed on held-out requests. Base
survivor/novel AUC was `0.9609/0.9499`; the full transition table achieved only
`0.6801/0.7609`. Shuffled controls were approximately `0.49--0.50`, and the
best cross-validated factual delta came from a shuffled target (`-0.0394`
point). This is strong negative evidence against the tested transition prior.

The only valid all-expert J table is at layer 11. Its omission metrics were
non-positive for every held-out novel slice and negative for survivors. Prior
activation-space J correction at layer 11 was `-0.2808` point; a rank-192
row-space projection produced only `+0.1953` point. Earlier evidence of roughly
`+0.7` point from a direct J-space lens is real but far below the required gap
and would require a dense serving lens to realize directly.

Outcome: J/transition branch stopped. J-based reallocation was not applied.
The correct resident plan remains 2,560 locked top-64 cells plus 1,290 optional
cells. A different permutation-invariant current-set context is not
mathematically ruled out, but the dense version already failed latency and the
tested low-rank transition forms supplied no factual signal.

### 6.6 Resident allocation and omitted-expert controls

The following retained-data or small-screen controls were tried:

| Approach | Best measured result | Decision |
| --- | --- | --- |
| Missing-weight renormalization | `0` to `+0.055` factual point | Too small; stop |
| Static missing-ID/direct-router table | about `+0.21` to `+0.23` point | Too small; stop |
| Folded shared-state compensation | best `+0.116` point; full directional A regressed | Too small; stop |
| Direct-router residual at layer 8 | learned variants regressed from zero residual | Stop |
| Rank-8 expert-down fold after INT4 | approximately zero gain; one run collapsed routing | Stop |
| Mixed Q3 allocation | fixed-call mass proxy about `+1.10` points, but no production kernel and no compliant efficiency path | Not promotable |
| Core64 + lightweight J-tail | faster, but factual Recall regressed `1.3855` points | Stop |
| J-tail convex mixture | plateaued below gate | Early stop |
| Damage/reallocation surrogate | predicted as much as `+3.421` points; actual layer-11 gain only `+0.1404` point | Surrogate invalid for global allocation |

The central lesson is that execution-frequency or gate-mass allocation is a
useful baseline, but local allocation proxies do not predict closed-loop route
damage accurately enough. J-space did not fix that. No allocation was allowed
to remove a mandatory frequency top-64 resident.

### 6.7 Quantization and precision changes

Full all-expert group-64 INT4 loses about 0.882 factual point versus its higher-
precision control, and layer 11 loses about 0.830 point. These losses are much
smaller than the approximately 8.8-point gap between resident-only and
all-expert INT4 closed loop. More precision cannot recover the target without
reducing expert coverage under the fixed byte budget, so this branch was not
pursued.

## 7. Exact score path and why a final ranker is not naturally foldable

There is no standalone learned ShadowRoute score head in the retained resident
checkpoint.

For each speculative tree:

1. `run_shadow_tree` restores the exact target prefix and executes resident-
   INT4-only future nodes.
2. `ShadowRouteHooks` captures the frozen native router logits, selected IDs and
   weights, post-layer hidden state, and LM output needed during that run.
3. H1 audit scores are the root node's exact native router logits.
4. H2--H4 scores are `shadow_inclusion` marginals:
   LM-head edge log-probabilities become cumulative path probabilities; captured
   path mass is scatter-added onto each node's selected expert IDs; OTHER mass
   is completed with frozen HARP anchor marginals; stable top-8 is then taken.

Therefore, a learned residual after `shadow_inclusion` is an additional serving
operation. There is no existing final affine layer into which it can simply be
folded.

Two algebraic same-shape folds were identified but **not executed** before the
pause:

1. Train a low-rank router update `delta_W = A B`, then materialize it into the
   existing bias-free `W_router[256,2048]`. This preserves router GEMM shape,
   bytes, calls, and MACs and is the only zero-extra-cost fold that can change
   candidate generation. However, the same router is used for exact-prefix and
   shadow-future passes, so an unconstrained update can corrupt authoritative
   prefix routing. It also requires damaged-shadow pre-gate inputs, which the
   retained score audit does not save.
2. Fold a low-rank update into the existing LM head to alter branch/path
   posterior. This also preserves the GEMM shape, but can only reweight experts
   already present in tree nodes. Existing target-posterior/factual-branch
   controls move the old mean only to about 0.8412/0.8447, so it cannot plausibly
   reach 0.90 by itself. It must also be proven isolated from production token
   generation.

Explicit current-route IDs and eight-step history are not direct inputs to the
native router. Adding embeddings for them would add state, parameters, and
work; they cannot be folded into the existing linear router unless their signal
is already represented in its 2,048-dimensional hidden input.

## 8. Fixed 32-node tree and branch-allocation evidence

The old DeltaRoute native counterfactual audit showed a large oracle but poor
learnability:

| Condition | H2--H4 factual Recall@8 |
| --- | ---: |
| Learned branch posterior | 0.855138 |
| Causal MTP prior | 0.864570 |
| Target-forced posterior oracle | 0.929717 |
| Factual branch with anchor fallback | 0.940647 |

Candidate coverage was already `0.985358` mean H2--H4 and `0.982202` at H4.
Thus tree width is not the main ceiling. The gap lies in path posterior and
realized-branch selection.

On the current retained old resident-only audit, however, factual-branch
selection gives only:

```text
H1-H4 mean  0.844730
H2-H4 mean  0.835329
H2          0.855597
H3          0.842450
H4          0.807941
```

This is above the `0.828593` deployed shadow-LM mean but far below 0.90. Earlier
branch translators and selectors did not generalize from the 256-request
fitting corpus. The prior DeltaRoute handoff recommended at least 2,000 diverse
training requests/32,000 positions if this family is revisited.

At pause, the fixed-node branch-allocation audit had only reached this retained
evidence review. No new branch policy was trained, no CacheSet score was
computed for factual-branch selection, and no capture was launched.

## 9. Relevant earlier architecture evidence reviewed and not repeated

These pre-existing results materially constrain what should be tried next.
They were reviewed in this effort; they were not rerun.

### 9.1 Generic shared fallback

The old resident-plus-width-512 fallback was actively harmful. On representative
teacher-forced layers, resident-only next-router recall was `0.939667` versus
`0.815033` with the generic fallback. In closed loop, resident-only v2 improved
mean H1--H4 factual Recall from `0.393674` to `0.828593`. Tail retraining from
both the old initializer and a zero initializer failed to recover the gap.

### 9.2 Resident proxy/codebook substitution

Top-1 and top-2 resident proxies for a missing expert both regressed versus
omitting the tail:

| Condition | Mean representative-layer next-router Recall@8 | Normalized residual MSE |
| --- | ---: | ---: |
| Resident-only | 0.947815 | 0.205314 |
| Nearest one | 0.936035 | 0.315234 |
| Nearest two | 0.941788 | 0.254158 |
| Exact missing tail | 0.983978 | 0.000000 |

The functional codebook failed its component gates before a 40-layer fit.

### 9.3 Compact basis, correction, and miniature-expert readouts

Earlier scalar route coefficients were negligible; per-neuron basis
coefficients added about 0.5 point; width-2 expert residuals added about one
point; a 25.3M-parameter-per-layer width-16 residual moved only `0.6823` to
`0.6934` after two epochs. Rank-32/64/128 correction oracles became too small
after the stronger resident allocation; even the rank-128 oracle offered only
about `+2.485` points locally. These families do not supply the required
closed-loop gain at the serving budget.

### 9.4 Opportunistic current-token cache extension

Label-only coverage from current-token experts contributed only about five
additional points of static coverage:

| Horizon | Static resident | Incremental current-token cache | Combined |
| --- | ---: | ---: | ---: |
| H2 | 0.729488 | 0.052803 | 0.782291 |
| H3 | 0.734767 | 0.048700 | 0.783467 |
| H4 | 0.732354 | 0.047115 | 0.779470 |

This was insufficient and does not address trajectory drift.

### 9.5 Route translators and RouteMTP

Prior DeltaRoute raw-channel, high-rank, recurrent, axial, post-MoE, and
tree-global variants all failed large-gain gates. The best early branch gains
were only roughly 1--2 points, and token-to-route lookup reached only `0.348842`
H2--H4 Top-8 despite high token repetition. RouteMTP screening peaked around
`0.142412` branch Recall@8 and `0.150488` factual H2--H4 Recall@8 on its replay
screen. These results reinforce that a small route head without richer,
request-diverse causal supervision is not the solution.

## 10. Data, leakage, and promotion status

The old 512-tree audit consists of 32 requests. CacheTrajectory used 12 for
training, 4 for calibration, and opened the remaining 16 as blind diagnostic
requests. No further result from this audit can be formal or sealed.

The factorized successor deliberately did not open those 16 requests because
its four-request calibration factual mean was only `0.777380`, below its 0.88
promotion gate.

The B2 route corpus contains 256 requests (224 train, 32 tune) and exact target
router inputs/logits/IDs/weights, but appears to contain normalized exact-target
router inputs rather than omission-damaged shadow inputs. Previous alignment
found no B2-to-Exact512 row joins. A router-weight fold trained only on exact
inputs would retune the target router, not learn omission compensation.

No fresh request-disjoint formal validation has been run on the compliant
resident bundle. `sealed_test_opened=false` and `formal_validation_opened=false`
remain true in its manifest.

## 11. Efficiency status

No promoted candidate increased resident experts or resident bytes. The only
sealed compliant bundle remains exactly 3,850 cells and 5.992418 serialized
GiB, with no fallback and no target expert loads.

The first CacheTrajectory head was compact but factually unacceptable. The
factorized successor was both smaller and faster than it. DeepSets was stopped
before training because its measured active-path latency regressed. Dense
J-space lenses and post-inclusion corrections were rejected because they would
add serving work. Mixed-Q3 was not promoted because it lacked a production
kernel/verified fixed-call path.

No claim has yet been established that a successful 95/90 architecture meets
the complete end-to-end ShadowRoute latency contract, because no successful
joint architecture exists.

## 12. Current bottleneck statement

The strongest supported interpretation at pause is:

1. CacheSet survivor preservation is not the hard part.
2. Novel future experts dominate the factual error.
3. The C32 candidate pool contains enough correct experts to exceed 90%.
4. Independent candidate scores and static transition priors cannot select them
   reliably.
5. Resident omission changes the hidden trajectory and native router outputs;
   more than half of factual misses occur even when the target expert is
   resident.
6. J-space contains weak complementary geometry, but not enough deployable
   signal in the tested forms.
7. The final H2--H4 score is a path-marginal computation, not a trainable dense
   head. Any genuinely stronger correction must either improve the hidden/router
   trajectory, improve same-budget branch selection, or be proven foldable into
   an existing GEMM.

## 13. Work explicitly not completed at pause

- Closed-loop baseline evaluation of the new top-64-compliant bundle.
- A formal, fresh request-disjoint capture/evaluation.
- A CacheSet audit of alternative same-32-node branch policies.
- Search for a preserved omission-damaged 2,048-D pre-gate corpus.
- Low-rank router-weight folding with exact-prefix parity constraints.
- Proof that an LM-head fold can be isolated from serving token-generation
  semantics.
- Any new J-conditioned resident reallocation.
- Any architecture achieving both targets.

## 14. Recommended restart gates

These are recommendations only; work is paused.

1. First run one closed-loop diagnostic baseline on the sealed compliant bundle.
   This is necessary to establish the actual starting point under the mandatory
   top-64 constraint.
2. Before any capture, determine whether omission-damaged pre-gate hidden inputs
   already exist. If they do, test only rank-4 then rank-8 router folds on one
   high-damage layer, with exact-prefix top-8 parity/KL anchoring.
3. Stop the router-fold family after one epoch if either factual Recall or
   CacheSet regresses, if joint gain is below roughly one point, or after two
   validations with less than 0.25-point improvement.
4. Promote to a matched closed-loop screen only after a multi-point
   teacher-forced gain. Small local gains are not credible because the measured
   teacher-forced-to-closed-loop gap is about eight points.
5. If no damaged-hidden corpus exists, the only currently measured large oracle
   is request-diverse factual branch selection. A new capture is justified only
   for that explicit hypothesis, with request-disjoint supervision at the scale
   recommended by the DeltaRoute handoff. Do not capture merely to train another
   independent score head.
6. Preserve exactly 3,850 residents and all original frequency top-64 experts
   per layer throughout. If any experiment changes router weights, explicitly
   define whether the frequency core is bound to the original frozen router;
   absent a new user decision, retain the current original-router top-64 core.
7. Any final claim requires a new sealed request-disjoint evaluation and matched
   end-to-end latency measurement. The old 32-request audit cannot be reused for
   promotion.

## 15. Do-not-repeat list

Unless new evidence changes a measured ceiling, do not spend further cycles on:

- fixed survivor quotas as the main CacheSet mechanism;
- independent per-candidate residual scoring without set/trajectory context;
- dense DeepSets at the measured latency;
- static expert-transition/J tables;
- J-tail convex mixtures;
- generic shared fallbacks;
- one- or two-resident functional codebooks;
- post-INT4 low-rank expert-down folds;
- simple missing-weight renormalization;
- static missing-ID/router-logit tables;
- allocation changes justified only by local damage surrogates;
- additional epochs on the failed factorized checkpoint;
- a new capture for another small head with no measured multi-point ceiling.

This handoff is deliberately conservative about claims: it preserves the one
valid 95%-CacheSet diagnostic result, the 90% candidate/full-expert ceilings,
and the correct compliant resident bundle, while clearly separating them from
a deployable joint success—which has not yet been obtained.
