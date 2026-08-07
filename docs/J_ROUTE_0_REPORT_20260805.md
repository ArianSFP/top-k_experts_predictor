# J-Route-0: frozen BF16 characterisation report

Date: 2026-08-05

## Scope and decision

This study tests whether genuine sparse J-space in the frozen BF16
`Qwen3.6-35B-A3B` model helps predict that same model's selected experts at
future tokens. It does not make an MXFP4/GGUF deployment claim.

The main result is:

> Sparse J-space contains held-out future-routing information, but it is not a
> privileged representation of that information. Raw/residual-PCA features and
> the checkpoint's native MTP state are consistently stronger. Literal J-space
> does not pass the predeclared gate for causal intervention, cache replay, or a
> deployment predictor branch.

The recommended future predictor backbone is therefore route history + native
MTP + compressed residual state, with direct multi-horizon heads. A nonlinear
J interaction can remain a bounded diagnostic ablation, but is not a dependency
for the caching work.

## Frozen artifacts and audits

### BF16 target trace

- Capture: `artifacts/j_route_0/captures/bf16_screen_n256_20260805a`
- Local audit: `audit.local.strict.json` (`accepted: true`)
- Requests: 256, balanced across eight domains
- Routed rows: 8,704 (34 per request)
- Rows with both `t+1` and `t+2` labels: 8,192
- Arrays: all 40 post-block residuals, all 40×256 router logits and
  probabilities, ordered top-8 IDs, and execution weights
- Frozen split: 154 train / 51 validation / 51 test requests

The trace is mirrored locally and is documented in
`captures/bf16_screen_n256_20260805a/LOCAL_ANALYSIS_README.md`.

### Lens correspondence

- Lens artifact hash: `2fdf5128...`
- Held-out correspondence gate: accepted
- Rows/requests: 128/4
- Every tested source layer improved the median target-token rank over the
  logit lens
- Mean 100-prompt/1,000-prompt transported-direction cosine: 0.99064

The third-party lens is treated as a screening artifact. Sparse extraction uses
an independently implemented greedy nonnegative pursuit because the reference
repository does not ship the paper's sparse-decomposition solver.

### Sparse extraction

- Remote canonical extraction: `jspace_bf16_n256_k16_20260805a`
- Strict audit: accepted, including hashes, geometry, support, reconstruction,
  and every per-K pursuit trajectory
- Source layers: 4, 12, 20, 28, 36, 38
- Profiles: pre-router and post-expert
- Frames: J, logit lens, and matched random
- Maximum support: K=16

Adaptive J occupancy is small: median 1–3 directions depending on layer and
profile. Direction identity has little adjacent-token persistence in early and
middle layers, rising modestly late (pre-router layer 38: mean Jaccard 0.116 and
coefficient cosine 0.174 at `t+1`; 0.029/0.045 at `t+2`).

### Native MTP trace

- Capture: `artifacts/j_route_0/captures/mtp_bf16_n256_20260805a`
- Local audit: `audit.local.strict.json` (`accepted: true`)
- Rows: 8,704, aligned one-to-one with the BF16 target trace
- All 19 native `mtp.*` checkpoint tensors loaded explicitly
- Router hashes/geometry/weights: accepted

The MTP shift is independently validated with the frozen LM head:

| Candidate target | Valid rows | Mean NLL | Top-1 accuracy |
| --- | ---: | ---: | ---: |
| `t+1` | 8,704 | 19.108 | 0.17% |
| `t+2` (expected) | 8,448 | 0.236 | 93.04% |
| `t+3` | 8,192 | 14.992 | 0.61% |

The MTP MoE is a distinct learned layer. Its expert IDs are not direct target
model expert predictions; its hidden state and router logits are used as
features.

## J-space association with selected experts

The descriptive atlas tests active J directions against `S[t,l]`, `S[t+1,l]`,
and `S[t+2,l]` across six source layers, two profiles, fixed/adaptive support,
and all 40 target routers.

- Atlas: `artifacts/j_route_0/analyses/association_bf16_n256_k16_t_t1_t2_20260805a`
- Reported associations: 14,400 (top five per cell)
- Direction support records: 19,748
- Bootstrap unit/replicates: complete request / 1,000

The selected top-five pairs have high descriptive enrichment, but their
identity is not stable across horizons:

| Horizon pair | Cells sharing ≥1 top-five pair | Cells | Mean shared pairs (of 5) |
| --- | ---: | ---: | ---: |
| `t` vs `t+1` | 122 | 960 | 0.143 |
| `t+1` vs `t+2` | 105 | 960 | 0.139 |
| `t` vs `t+2` | 49 | 960 | 0.072 |

Only 23 direction/expert pairs recur in the top five at all three horizons.
Because associations were selected and bootstrapped on the same corpus, these
figures are descriptive and selection-biased. They are not used for the
forecasting decision.

## Held-out future-router probes

All probes predict centred 256-way BF16 router logits, select ridge
regularisation on validation requests, evaluate the test requests once, and
use 1,000 paired request-bootstrap replicates. Feature ranks and incremental
head sizes are matched.

### Complete source/profile matrix at `t+1` and `t+2`

The complete matrix has 18 source/profile cells × 40 target layers × two
horizons. Rankings averaged across all 720 cells per horizon are:

| Condition | `t+1` top-8 recall | `t+2` top-8 recall |
| --- | ---: | ---: |
| Route history + residual PCA | 61.55% | 54.97% |
| Route history + raw projected residual | 56.96% | 51.10% |
| Route history + non-J remainder | 56.56% | 50.85% |
| Route history + J | 54.50% | 49.27% |
| Route history | 50.79% | 46.30% |

J versus matched controls, counted over 720 target cells:

| Reference | `t+1` significant J wins/losses | `t+2` wins/losses |
| --- | ---: | ---: |
| Route history | 720 / 0 | 712 / 0 |
| Shuffled J | 720 / 0 | 720 / 0 |
| Random frame | 678 / 0 | 603 / 0 |
| Logit-lens frame | 266 / 48 | 202 / 20 |
| Non-J remainder | 1 / 705 | 0 / 707 |
| Raw residual | 0 / 718 | 0 / 716 |
| Residual PCA | 0 / 720 | 0 / 720 |

Thus J contains real information, but the broader residual state contains more.
The predeclared J-specific gate has no passing source/profile/horizon region.

### Native MTP and longer horizons

The strongest late source (post-expert layer 38) was tested against all 40
target routers at direct horizons 1, 2, 4, 6, 8, and 16. The strongest
configuration is route history + native MTP + residual PCA:

| Horizon | Top-8 recall | Route-only | Route + J |
| ---: | ---: | ---: | ---: |
| 1 | 69.15% | 54.68% | 58.86% |
| 2 | 63.00% | 48.56% | 52.09% |
| 4 | 53.32% | 44.24% | 46.05% |
| 6 | 48.61% | 41.10% | 42.39% |
| 8 | 45.68% | 38.56% | 39.62% |
| 16 | 38.22% | 30.56% | 32.54% |

At horizon 16, MTP+non-J has marginally lower mean router cross-entropy than
MTP+PCA, while MTP+PCA retains the best mean top-8 recall.

MTP+J never significantly beats the capacity-matched MTP+PCA condition in any
of the 240 horizon/target-layer cells. It often underperforms MTP alone at
horizons 1–6 and has no significant advantage at horizons 8 or 16.

## Go/no-go decisions

1. **Ordinary sparse J-space as a deployment feature: no-go.** It fails raw,
   PCA, and non-J controls.
2. **J-specific causal interventions: not run.** The predeclared positive probe
   gate failed.
3. **J-driven cache replay: not run.** The same gate failed; replaying a weaker
   predictor would not establish J-specific actionability.
4. **Literal nonlinear J branch: not justified on this screen.** It can be an
   optional compute-matched diagnostic on a future larger corpus.
5. **Residual/MTP route predictor: go.** This is the strongest evidence-backed
   route-forecast direction.

## Recommended next predictor

Use the supplied architecture idea, but make the fast branch primary:

1. Inputs available after token `t` completes:
   - compressed current/prior route history;
   - train-only residual PCA or learned low-rank residual projection;
   - native MTP hidden state and centred MTP router logits;
   - causal availability and horizon masks.
2. Encoder:
   - two small gated residual/SwiGLU blocks;
   - one rank-32 or rank-64 factorised bilinear interaction;
   - direct learned horizon and target-layer embeddings.
3. Heads:
   - continuous future router logits or a rank-64 target-router geometry code;
   - exact-step top-8 use probability;
   - first-use hazard;
   - expected horizon workload.
4. Losses:
   - router-distribution KL;
   - hard-negative top-8/top-9 boundary loss;
   - expert-use and hazard losses;
   - calibration/Brier loss.
5. Policy:
   - keep residency, in-flight transfers, byte cost, and deadlines in a
     separate cache controller.

Do not train this nonlinear model seriously on only 256 requests. First capture
a larger compact corpus (preferably at least 2,048–4,096 independent requests)
that retains selected residual layers, native MTP features, complete router
labels, and horizons through at least 16. Split by request and source item.

## Format-5 isolation

The study made no writes to the frozen Format-5 worktree. The audited checkpoint
remains tagged at `e17563b5a3573d1a9ae413f255f2f43ef0667bfe`. During this
parallel investigation, the other workstream advanced that worktree to
`b56e5dfc8b317c3c64e695b5a4766a9701e7863c` with two descendant commits and
left pre-existing backup/status entries. J-Route-0 did not consume those
changes; its primary trace is direct BF16 PyTorch capture.

## Principal local result paths

- Base capture: `artifacts/j_route_0/captures/bf16_screen_n256_20260805a`
- Native MTP: `artifacts/j_route_0/captures/mtp_bf16_n256_20260805a`
- Occupancy: `artifacts/j_route_0/analyses/occupancy_bf16_n256_k16_20260805a`
- Association atlas: `artifacts/j_route_0/analyses/association_bf16_n256_k16_t_t1_t2_20260805a`
- Association summary: `artifacts/j_route_0/analyses/association_summary_bf16_n256_20260805a`
- Full matched matrix: `artifacts/j_route_0/probes/bf16_matrix_n256_20260805b`
- Full matrix summary/heatmaps: `artifacts/j_route_0/analyses/probe_summary_bf16_n256_20260805a`
- Same-layer `t/t+1/t+2` probes: `artifacts/j_route_0/probes/diagonal_t_t1_t2_20260805a`
- MTP/long-horizon probe: `artifacts/j_route_0/probes/mtp_j_post_expert_source38_h1_2_4_6_8_16_20260805a`
- MTP summary/heatmaps: `artifacts/j_route_0/analyses/mtp_probe_summary_bf16_n256_20260805a`

## References

- Jacobian lens and J-space: https://transformer-circuits.pub/2026/workspace/index.html
- Reference Jacobian-lens implementation: https://github.com/anthropics/jacobian-lens
- Qwen checkpoint/model card: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- Qwen3.5 MTP implementation used to cross-check the reconstructed contract:
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py
