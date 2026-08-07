# HARP8 accuracy-expansion stop handoff — 2026-08-07

## Stop status

All RunPod work for this experiment is paused at a safe boundary. The active c64 reranker was stopped before its first epoch; no c128 export, c128 reranker, or high-dimensional MTP feature build started. The RunPod files remain intact under:

```
<harp8-expansion-root>/
```

The test partition was never opened. All reported values below are validation-only unless explicitly labelled as an inner-split diagnostic.

The detailed chronological ledger is:

- `HARP8_ACCURACY_EXPANSION_LEDGER_20260807.md`

## Scientific objective

The objective was a material improvement over the previous HARP endpoint in top-8 expert Recall@8 for direct horizons t+1 through t+4, while retaining the t+5--t+8 outputs and preserving a candidate pool for a future cache/prefetch scheduler.

The historical endpoint anchor is:

`artifacts/harp8/training/multihorizon_current_n2048_seed42`

It used the original full split (1,229 train requests / 410 validation requests / 409 sealed test requests). Its validation request-macro Recall@8 was:

| Horizon | Historical endpoint |
|---:|---:|
| t+1 | 0.8156604 |
| t+2 | 0.7999698 |
| t+3 | 0.7812144 |
| t+4 | 0.7569591 |
| Mean t+1--t+4 | **0.7884509** |

## Split and leakage controls

- Primary comparisons use the historical full validation split: 410 requests.
- The 409-request outer test set remains sealed.
- The nested inner split (983/246 requests) was used only as a leakage-aware diagnostic; it is not used for the endpoint claim.
- PCA-width work queued for later uses `fit_split=full_train`, which means outer validation and test requests are excluded from preprocessing fit.
- No random token-level split was introduced.

The matched inner evaluation of the old endpoint (mean 0.8297711) is useful for diagnosing split effects but is not an endpoint improvement claim because those requests were part of the old model's training corpus.

## Completed generator experiments

| Run | H1 | H2 | H3 | H4 | Mean H1--H4 | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Historical endpoint | .815660 | .799970 | .781214 | .756959 | **.788451** | Baseline anchor |
| Full-split warm-start + direct skips, best epoch 30 | .818177 | .801273 | .783164 | .759373 | **.790497** | +.002046; not significant |
| Frozen direct-adapter final state | .818834 | .801458 | .783333 | .759501 | **.790782** | Final state improved, but candidate-coverage selection retained the warm-start checkpoint |
| H3/H4 weighted continuation, selected epoch 27 | .820415 | .801945 | .785771 | .762570 | **.792675** | Best completed generator; +.004224 over historical |
| Temperature-1 exploratory peak, epoch 5 | .821683 | .802488 | .786069 | .762165 | **.793101** | Best observed generator; +.004650 over historical |
| Temperature-1 epoch 6 | .821596 | .802281 | .785499 | .761915 | **.792823** | Regressed; stopped |

The temperature-1 run was intentionally stopped after epoch 6. Its epoch-5 model is preserved as `recall_best_epoch5.pt`; this is a real but modest gain, not yet the requested significant improvement.

The H3/H4 selected checkpoint has mean candidate-16 coverage 0.9088318 and minimum candidate-16 coverage 0.8803072. The temperature-1 epoch-5 point has candidate-16 mean coverage 0.9085463.

## Candidate-pool/reranker experiments

The scheduler-facing experiment separates candidate recall from final top-8 ranking.

### Candidate-32

The corrected candidate-32 pools contain 40,557 train rows and 13,530 validation rows. Their validation candidate coverage ceiling is:

- mean H1--H4: **0.9524322**
- H4: **0.9293521**
- fixed-16 gate field: `passes=false` because the H4 threshold is narrowly below 0.93

A permutation-equivariant c32 reranker was trained for two epochs before the stop request:

| Reranker state | Validation mean H1--H4 Recall@8 |
|---|---:|
| Generator (temperature-1 epoch 5) | 0.7931011 |
| c32 reranker epoch 1 | 0.7939985 |
| c32 reranker epoch 2 (saved best) | **0.7941871** |

This is an improvement, but only +0.005736 over the historical endpoint. The reranker has no final manifest because it was stopped early; its epoch-2 `best.pt` is preserved.

### Candidate-64

The complete c64 pools were exported successfully:

- 40,557 train rows
- 13,530 validation rows
- validation oracle coverage mean H1--H4: **0.9740958**
- H4 oracle coverage: **0.9562749**
- coverage gate: `passes=true`

The c64 reranker had not completed its first epoch when the pause was requested. Its output directory contains only a stop marker; no c64 ranking result is claimed.

### Candidate-128

No c128 pool or reranker was started. Its waiting wrapper was stopped.

## Important implementation correction

The first arbitrary-width candidate export failed because the generator exposes `copy_gate` for all 256 experts while the candidate pool stores only `candidate_count` experts. The exporter attempted to write [B,H,L,256] into a [B,H,L,C] memmap.

The fix gathers:

```python
outputs["copy_gate"].gather(-1, candidate_ids)
```

before writing the candidate pool. The failed partial directories are retained, and clean versioned `v2/v3` outputs were used afterward.

Focused tests after the fix:

```
pytest -q tests/test_harp8.py tests/test_harp8_reranker.py
16 passed
```

## Higher-dimensional MTP screen

A full-training PCA feature build was queued at maximum rank 1024, with raw-standardized output disabled to save space. It was stopped before construction. Sequential MTP widths 256, 512 and 1024 were therefore **not trained**.

The intended future protocol is:

- fit PCA on full outer-training requests only;
- train the same route/target architecture from the same seed;
- compare H1--H4 against the historical endpoint and temperature-1 epoch-5 checkpoint;
- keep the test partition sealed.

## Local archive

Completed local artifacts include:

- [H3/H4 selected and resumable checkpoints](artifacts/harp8_accuracy_expansion/h34_continue_recall/)
- [Temperature-1 partial run and epoch-5 peak checkpoint](artifacts/harp8_accuracy_expansion/temp1_continue_partial/)
- [Partial c32 reranker](artifacts/harp8_accuracy_expansion/reranker_temp1_epoch5_c32_v2/)
- [Stopped c64 reranker log/marker](artifacts/harp8_accuracy_expansion/reranker_temp1_epoch5_c64_v3/)
- [c64 pool provenance manifests](artifacts/harp8_accuracy_expansion/pool_manifests/)
- [Chronological experiment ledger](HARP8_ACCURACY_EXPANSION_LEDGER_20260807.md)

The large c32/c64 memmap pools remain on the RunPod under:

```
<harp8-expansion-root>/pools/
```

They were not deleted or overwritten.

## Resume plan

1. Resume c64 reranking from the complete `temp1_epoch5_c64_v3_train` and `..._validation` pools. Use a new output directory (the stopped c64 directory is intentionally not reused).
2. Evaluate the c64 reranker per horizon and compare against the c64 oracle ceiling.
3. Export and rerank c128 only if c64 does not provide the desired gain; its larger candidate set is a scheduler diagnostic, not a fixed-16 claim.
4. Run the full-training PCA width screen (256, 512, 1024) only after the candidate results are archived.
5. If c64/c128 reranking remains below the practical target, revise the reranker objective toward direct top-8/listwise optimization rather than adding more generator complexity.
6. Compute request-level paired bootstrap intervals for the selected generator and reranker before declaring a statistically supported improvement.

## Bottom line

The best completed generator is temperature-1 epoch 5 at mean H1--H4 Recall@8 **0.7931011**, and the best completed scheduler-facing result is the partial c32 reranker at **0.7941871**. Both are safely above the historical .7884509 anchor but neither is yet a significant practical improvement or the desired approximately .80+ scheduler result.

The most informative next experiment is the already-exported c64 pool: it has a 0.9741 oracle coverage ceiling and is ready to resume without recollecting data or reopening the test partition.
