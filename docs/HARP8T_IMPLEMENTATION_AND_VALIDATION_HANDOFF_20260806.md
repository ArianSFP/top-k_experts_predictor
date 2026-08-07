# HARP-8T implementation and validation handoff

**Date:** 2026-08-06
**Status:** implementation and planned validation controls complete; test unopened by the HARP pipeline

## 1. Outcome

HARP-8T implements the revised task contract proposed after the GCRP-2R v1.3
diagnosis: one model directly predicts the complete 256-expert router score vector
for every layer and every endpoint from `t+1` through `t+8`. The eight heads are
direct and non-recursive. A later scheduler can request top-8 predictions or wider
top-12/16/24/32 candidate lists without coupling the predictor to one cache policy.

The primary two-stage validation curriculum reached the immediate design target:

| Checkpoint | H2 request-macro SlotRecall@8 | 95% request-bootstrap CI | Test opened |
|---|---:|---:|---:|
| H2 bridge, epoch 30 | 0.802562 | [0.796641, 0.808592] | No |
| Unified H1-H8, epoch 30 | 0.799970 | [0.793990, 0.805906] | No |

The unified checkpoint is the first result to use for downstream development. The
H2-only checkpoint is a curriculum/diagnostic artifact, not a separate production
model. The unified model's complete validation endpoint curve is:

| Horizon | Request-macro SlotRecall@8 |
|---:|---:|
| 1 | 0.815660 |
| 2 | 0.799970 |
| 3 | 0.781214 |
| 4 | 0.756959 |
| 5 | 0.735197 |
| 6 | 0.715189 |
| 7 | 0.688649 |
| 8 | 0.660282 |

The H1-H8 mean is 0.744140. These are validation results on 410 held-out requests.
They are not claims about the sealed 409-request test partition.

For scheduler-facing wider candidate budgets, the same checkpoint provides:

| Horizon | Top-8 | Top-12 | Top-16 | Top-24 | Top-32 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8157 | 0.9009 | 0.9309 | 0.9585 | 0.9711 |
| 2 | 0.8000 | 0.8852 | 0.9170 | 0.9467 | 0.9607 |
| 3 | 0.7812 | 0.8663 | 0.8991 | 0.9309 | 0.9462 |
| 4 | 0.7570 | 0.8426 | 0.8773 | 0.9120 | 0.9292 |
| 5 | 0.7352 | 0.8215 | 0.8579 | 0.8947 | 0.9135 |
| 6 | 0.7152 | 0.8008 | 0.8382 | 0.8771 | 0.8973 |
| 7 | 0.6886 | 0.7744 | 0.8139 | 0.8562 | 0.8786 |
| 8 | 0.6603 | 0.7459 | 0.7869 | 0.8319 | 0.8564 |

Every entry is coverage of the native top-8 set and therefore retains denominator
8 even when the predictor exposes more than eight candidates.

## 2. Implemented package

The implementation lives in `harp8/`:

| File | Role |
|---|---|
| `harp8/config.py` | serializable model, loss and optimizer configuration |
| `harp8/data.py` | request-grouped, fail-closed compact-corpus loader |
| `harp8/model.py` | route, residual, MTP and direct H1-H8 prediction model |
| `harp8/losses.py` | KL, boundary, inclusion, score and latent objectives |
| `harp8/metrics.py` | request/micro/domain/layer metrics and bootstrap CI |
| `harp8/train.py` | curriculum, gradient calibration, resume and controls |
| `harp8/evaluate.py` | explicit one-time sealed-test evaluation entry point |
| `harp8/overfit.py` | real-trace overfit correctness diagnostic |
| `tests/test_harp8.py` | contracts, causality, invariance and checkpoint tests |

`pyproject.toml` exports the training, evaluation and overfit commands.

## 3. Architecture actually trained

### 3.1 Inputs

The compact compatibility corpus supplies:

- eight committed-token router-logit history positions for all 40 layers;
- one rank-128 PCA channel of the current same-layer post-block target state;
- one rank-128 PCA MTP hidden channel at each captured MTP node;
- complete 256-way MTP router logits and four metadata values per node;
- six available native MTP depths; configured depths 7 and 8 are zero-filled and
  explicitly unavailable.

Future target routes and router scores are loaded only into label tensors.

### 3.2 Route encoder

For every layer and lag, the model centers the 256 router logits and appends six
statistics: normalized entropy, top-1/top-2 margin, top-8/top-9 margin, top-8
probability mass, logit standard deviation and mean absolute logit. A 256-wide
cell encoder is followed by two temporal Transformer blocks and two cross-layer
Transformer blocks.

### 3.3 Target-state encoder

The current token's per-layer rank-128 post-block PCA state is projected to width
256 and processed by a cross-layer Transformer block. This is a token-end source;
it is not presented as an early-layer causal feature.

### 3.4 MTP encoder

Each MTP node combines projected hidden state, centered 256-way MTP router logits,
metadata and depth embedding. Two horizon-aware cross-attention blocks condition
each `(horizon, layer)` query on available nodes. A learned relative depth-to-
horizon attention bias lets the model distinguish aligned and off-horizon nodes.

### 3.5 Source fusion and outputs

Route, target-state and MTP representations are fused with a learned availability-
masked source gate, source dropout and two gated residual blocks. Every endpoint
and layer then has its own direct dense `384 x 256` output map. A learned
expert-wise copy gate provides a persistence path from the current router logits.

The principal tensor contract is:

```text
future_router_scores: [batch, 8 horizons, 40 layers, 256 experts]
```

No predicted horizon is fed into another predicted horizon.

The dense compact model has 42,022,288 trainable parameters. The rank-128 output
control has 21,443,984 parameters.

## 4. Objective and optimization

The loss is the sum of:

1. forward KL from the native future router distribution at temperature 2;
2. pairwise top-8 versus native ranks 9-32 cutoff loss;
3. balanced positive/hard-negative inclusion loss;
4. centered router-score Smooth-L1 loss;
5. future latent Smooth-L1 loss.

Horizon 2 receives twice the weight of each other active endpoint. Before normal
training, component gradient norms are measured on the shared trunk and the loss
coefficients are frozen to preregistered relative gradient targets. The primary
run used 32 calibration batches.

Training uses AdamW, BF16 autocast, FP32 loss reductions, learning rate `2e-4`,
cosine decay to `2e-5`, 3% warmup, weight decay 0.01, gradient clipping 1.0,
batch size 64, seed 42, at most 30 epochs, at least 10 epochs and patience 5.

The accuracy-first curriculum is:

1. train all parameters using only the H2 objective;
2. initialize the same single model from the best H2 checkpoint;
3. train all direct H1-H8 heads jointly, with H2 still double-weighted.

## 5. Validation and controls

### 5.1 Primary checkpoint

Artifact directory:

```text
artifacts/harp8/training/multihorizon_current_n2048_seed42
```

Best checkpoint SHA-256:

```text
4317d6d03db98c72c2c2581a85e14bf664884c252ebe044db8e778c8bad06399
```

H2 bridge SHA-256:

```text
4997a99189e23e6cbacc0ca834e9d302ba63295f1d940869f3e0909e6451a4c6
```

### 5.2 Matched H2 source controls

All H2 bridge controls use the same split, seed, objective, optimizer, batch sizes,
30-epoch limit and validation selector. Inactive source encoders remain present in
the parameter count but are availability-masked, so the comparison holds dense
output capacity fixed.

| H2 bridge condition | Recall@8 | 95% request-bootstrap CI | Gain over route-only |
|---|---:|---:|---:|
| Route history only | 0.760018 | [0.751941, 0.768008] | -- |
| Route plus target residual | 0.764151 | [0.756198, 0.771898] | +0.004133 |
| Route plus MTP depth 1 | 0.787914 | [0.781237, 0.794628] | +0.027896 |
| Full residual plus MTP depths 1-6 | 0.802562 | [0.796641, 0.808592] | +0.042543 |

Paired 2,000-replicate request-bootstrap intervals for those gains are:

| Paired H2 difference | Gain | 95% CI |
|---|---:|---:|
| Residual minus route-only | +0.004133 | [0.003302, 0.004952] |
| MTP1 minus route-only | +0.027896 | [0.026064, 0.029619] |
| Full minus route-only | +0.042543 | [0.039854, 0.044941] |
| MTP1 minus residual | +0.023764 | [0.022127, 0.025393] |
| Full minus MTP1 | +0.014647 | [0.013515, 0.015852] |

The target residual adds a small but reproducible increment over the stronger
route-only bridge. MTP depth 1 supplies the larger standalone gain. The remaining
full-source gain shows that the target residual and/or MTP depths 2-6 still contain
information not captured by depth 1 alone.

For reference, the earlier route-only all-horizon run without an H2 bridge reached
H2 0.748446 and H1-H8 mean 0.700836. It remains a useful unified-curve diagnostic,
but the table above is the curriculum-matched H2 attribution.

### 5.3 Dense versus rank-128 output control

On the matched H2 bridge, rank-128 reached 0.786119 with 95% CI
`[0.779694, 0.792583]`, versus 0.802562 for the dense head. The dense head gains
1.644 points with a paired 95% CI of `[1.558, 1.728]` points, at the cost of
approximately 20.58 million additional parameters.

The completed unified rank-128 checkpoint reached H2 0.782872 with 95% CI
`[0.776503, 0.789336]` and H1-H8 mean 0.726193. Its complete H1-H8 curve is
`[0.796242, 0.782872, 0.762536, 0.739208, 0.716226, 0.697411, 0.671476,
0.643571]`. Dense improves unified H2 by 1.710 points with paired 95% CI
`[1.614, 1.801]` points, and improves the H1-H8 mean by 1.795 points.

### 5.4 Real-trace overfit check

On 256 real training rows, H2 Recall@8 increased from 0.2179 to 0.9522 in 350
steps. This passes the intended implementation/supervision-path diagnostic and
argues against a hard wiring, alignment or loss-direction fault.

## 6. Metric contract

Evaluation reports:

- native-denominator SlotRecall@8 for every horizon;
- candidate coverage at 12, 16, 24 and 32, always divided by native `k=8`;
- exact top-8 set rate and forward router KL;
- request-macro and micro aggregates;
- all 40 layer rows and per-domain rows;
- a paired/request-grouped 2,000-replicate H2 bootstrap interval.

Model selection order is fixed as:

1. validation H2 request-macro SlotRecall@8;
2. validation mean H1-H8 request-macro SlotRecall@8;
3. lower validation mean router KL.

## 7. Split and leakage protection

The compact corpus contains:

| Split | Requests | Source rows |
|---|---:|---:|
| Train | 1,229 | 40,557 |
| Validation | 410 | 13,530 |
| Test | 409 | 13,497 |

The loader splits by request, not by token. Normal training can access only train
and validation. Test loading requires an explicit `allow_test` path, and the
evaluation CLI additionally requires the exact confirmation string
`OPEN_HARP8_SEALED_TEST_ONCE`. None of the runs documented here used it.

The same 409-request partition was evaluated historically by earlier J-route
experiments, so it is not globally pristine. The precise claim here is narrower:
the HARP training, checkpoint selection and controls did not load the test split.


## 8. Current-corpus limitations

This result validates the architecture direction, but the present corpus is a
compatibility pilot rather than the desired final capture:

- only 2,048 total requests are represented;
- MTP depths 1-6 exist; depths 7-8 do not;
- the target source is one PCA post-block channel rather than raw same-execution
  `u`, normalized router input `a`, post-MoE `x+`, and derived MoE delta;
- the MTP source is one PCA hidden channel plus router logits, without separately
  retained fused state, router input and compact vocabulary distribution;
- no calibrated cache/transfer simulator is attached to this predictor result.

Consequently, the next scientific capture should retain the already specified
same-execution native-dtype target tensors, MTP fused/router/vocabulary sources,
explicit positional semantics and causal readiness fields. HARP-8T should then be
retrained and compared with this compact checkpoint rather than silently mixing
the two data regimes.

## 9. Reproduction commands

The following is the logical command sequence; paths can be relocated as long as
the manifest hashes and split grouping remain unchanged.

```bash
python -m harp8.overfit \
  --capture-dir <harp8-data-root>/capture \
  --mtp-dir <harp8-data-root>/mtp \
  --target-state-features <harp8-data-root>/features/local_post_layer_pca.npy \
  --mtp-state-features <harp8-data-root>/features/mtp_hidden_pca.npy \
  --output-dir <harp8-runs-root>/overfit256_seed42

python -m harp8.train \
  --capture-dir <harp8-data-root>/capture \
  --mtp-dir <harp8-data-root>/mtp \
  --target-state-features <harp8-data-root>/features/local_post_layer_pca.npy \
  --mtp-state-features <harp8-data-root>/features/mtp_hidden_pca.npy \
  --output-dir <harp8-runs-root>/h2_bridge_current_n2048_seed42 \
  --epochs 30 --minimum-epochs 10 --patience 5 \
  --batch-size 64 --evaluation-batch-size 256 \
  --learning-rate 0.0002 --seed 42 --h2-only \
  --gradient-calibration-steps 32 --condition primary

python -m harp8.train \
  --capture-dir <harp8-data-root>/capture \
  --mtp-dir <harp8-data-root>/mtp \
  --target-state-features <harp8-data-root>/features/local_post_layer_pca.npy \
  --mtp-state-features <harp8-data-root>/features/mtp_hidden_pca.npy \
  --output-dir <harp8-runs-root>/multihorizon_current_n2048_seed42 \
  --initialize-from <harp8-runs-root>/h2_bridge_current_n2048_seed42/best.pt \
  --epochs 30 --minimum-epochs 10 --patience 5 \
  --batch-size 64 --evaluation-batch-size 256 \
  --learning-rate 0.0002 --seed 42 \
  --gradient-calibration-steps 32 --condition primary
```

## 10. Acceptance status

- [x] Direct H1-H8 full-router outputs in one model.
- [x] Non-recursive horizon heads.
- [x] H2-focused curriculum.
- [x] Validation H2 approximately 80% with a grouped confidence interval.
- [x] Route-only and output-rank diagnostics.
- [x] Real-trace overfit check.
- [x] Full-resume optimizer/scheduler/sampler/RNG checkpoint state.
- [x] Test loader and evaluator fail closed.
- [x] Finish and freeze the matched H2 source controls.
- [ ] Repeat on the richer same-execution capture.
- [ ] Open the sealed test exactly once after the architecture and decision rule
  are frozen.
- [ ] Feed candidate lists into a separately calibrated cache/transfer simulator.
