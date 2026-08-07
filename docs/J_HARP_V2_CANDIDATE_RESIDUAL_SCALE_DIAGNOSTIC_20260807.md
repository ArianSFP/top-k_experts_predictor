# J-HARP v2 candidate residual-scale diagnostic

`harp8.evaluate_jspace_checkpoint` can audit whether a saved v2 candidate
reranker learned a useful correction whose magnitude is miscalibrated.  It
does not train, mutate, or create a checkpoint.

For each candidate, the diagnostic evaluates

```text
H1-H4: frozen_candidate_score + alpha * saved_active_delta
H5-H8: frozen_candidate_score (bit-exact passthrough)
```

The explicit scale grid must include both scientific anchors: `alpha=0` is
the frozen HARP candidate ranking and `alpha=1` is the exact saved v2
checkpoint.  Every scale shares one checkpoint forward per input batch.

Scale selection is leakage-safe only on the level-2 meta-training split:

```text
--split train       level2_meta_train_scale_selection; selectable
--split validation  post_hoc_validation_only_oracle_diagnostic; not selectable
```

There is no test-split CLI or API path.  The evaluator rejects any split other
than `train` or `validation` before opening a checkpoint or candidate pool.
The normal evaluator remains unchanged when `--residual-scale-grid` is absent,
including v1 checkpoint support and its existing output schema.

Example:

```bash
python -m harp8.evaluate_jspace_checkpoint \
  --checkpoint RUN/best.pt \
  --output-dir RUN/train_scale_audit \
  --split train \
  --residual-scale-grid 0,0.05,0.1,0.2,0.35,0.5,0.75,1
```

The default uncertainty calculation is a deterministic 5,000-replicate paired
percentile bootstrap over requests.  It reports H1-H4 mean Recall@8 gain and
each individual H1-H4 gain against `alpha=0`.  Outputs are:

```text
residual_scale_diagnostic.json
residual_scale_summary.csv
residual_scale_horizon_metrics.csv
```

A validation sweep is an oracle sensitivity analysis only.  Its winning scale
must not be promoted to a model hyperparameter; choose the scale on an OOF
level-2 meta-training pool, freeze it, and evaluate that fixed choice once on
validation.
