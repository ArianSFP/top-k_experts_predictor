# Top-k Experts Predictor (HARP8)

This repository contains the source release of the HARP8 multi-horizon expert-route predictor used in the Qwen3.6/MoE caching study. HARP8 predicts the target router top-8 experts for horizons t+1 through t+8 and exposes candidate pools for a separate scheduler or reranker. The expanded research package also contains J-Lens/J-space candidate rerankers and the experimental Full256 future-router forecaster.

The model is a research prototype and is router-preserving: predictions do not replace native routing or change model outputs.

## Included

- `harp8/`: model, causal data adapters, feature preparation, losses, training, evaluation, candidate export, J-space rerankers, and Full256 future-router forecasting.
- `tests/`: generator, OOF candidate-pool, J-space, Full256, causal masking, and metric contract tests.
- `docs/`: implementation protocols, experiment ledgers, MTP-path notes, leakage controls, and the OOF stacking protocol.
- `results/`: small aggregate validation summaries and machine-readable audits.

Raw traces, PCA arrays, checkpoints, candidate memmaps, RunPod scripts, credentials, and private infrastructure details are intentionally excluded.

## Install and test

Python 3.10+ and PyTorch 2.4+ are required.

~~~text
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest -q tests/test_harp8*.py
~~~

GPU training is not required for the focused test suite.

## Data contract

The loader expects an audited request-grouped corpus supplied outside this repository.

The capture directory contains:

~~~text
raw_router_logits.npy      [request_count * rows_per_request, 40, 256]
top8_expert_ids.npy        [request_count * rows_per_request, 40, 8]
requests.jsonl             one record per request
~~~

Each request record includes request_id, offline_split, and domain. The default format has 34 rows per request; the final row is not a source example because it has no future label.

The MTP directory contains:

~~~text
mtp_router_logits_depths.npy    [rows, captured_depths, 256]
~~~

Two feature arrays are required:

~~~text
target-state features    [rows, 40, target_state_width]
MTP-state features       [rows, captured_depths, mtp_state_width]
~~~

Feature preparation must fit PCA or related transforms on training requests only. The data adapter fails closed on geometry mismatches and keeps the outer test split unavailable unless explicitly opened by the evaluation command after model selection.

## Typical workflow

Prepare features:

~~~text
python -m harp8.features --capture-dir <capture-directory> --mtp-dir <mtp-directory> --output-dir <feature-output> --maximum-rank 1024 --fit-split full_train --device cuda:0
~~~

Run the real-trace overfit diagnostic:

~~~text
python -m harp8.overfit --capture-dir <capture-directory> --mtp-dir <mtp-directory> --target-state-features <feature-output>/local_post_layer_pca.npy --mtp-state-features <feature-output>/mtp_hidden_pca.npy --output-dir <overfit-output> --device cuda:0
~~~

Train the generator:

~~~text
python -m harp8.train --capture-dir <capture-directory> --mtp-dir <mtp-directory> --target-state-features <feature-output>/local_post_layer_pca.npy --mtp-state-features <feature-output>/mtp_hidden_pca.npy --output-dir <training-output> --epochs 30 --seed 42 --device cuda:0
~~~

Export an OOF-safe scheduler-facing candidate pool. A level-2 training pool must come from requests excluded from the fitted base model:

~~~text
python -m harp8.candidates \
  --capture-dir <capture-directory> \
  --mtp-dir <mtp-directory> \
  --target-state-features <feature-output>/local_post_layer_pca.npy \
  --mtp-state-features <feature-output>/mtp_hidden_pca.npy \
  --checkpoint <fold-training-output>/best.pt \
  --output-dir <heldout-candidate-pool> \
  --split-manifest <fold-split-manifest.json> \
  --candidate-count 64 \
  --source-data-split validation \
  --source-offline-split train \
  --level2-split train \
  --base-fold-id <fold-id> \
  --base-fit-excluded \
  --base-fit-split-manifest <fold-split-manifest.json> \
  --device cuda:0
~~~

Train the set reranker only after measuring candidate-pool oracle coverage:

~~~text
python -m harp8.reranker --train-pool <train-pool> --validation-pool <validation-pool> --output-dir <reranker-output> --candidate-count 32 --device cuda:0
~~~

J-space and Full256 entry points are also available:

~~~text
harp8-prepare-jspace --help
harp8-train-jspace-reranker --help
harp8-train-full-router --help
harp8-evaluate-full-router --help
~~~

The Full256 path is experimental and deliberately opt-in. Read `docs/HARP8_LEVEL2_OOF_CANDIDATE_POOL_PROTOCOL_20260807.md` before training any level-2 model; candidate-pool v2 rejects in-sample base predictions and request overlap by default.

## Reproducibility and status

The loader splits by complete request rather than individual token. The historical full validation anchor was 410 requests with request-macro Recall@8 H1 0.8156604, H2 0.7999698, H3 0.7812144, H4 0.7569591, mean H1-H4 0.7884509.

The best completed expansion generator reached mean H1-H4 0.7931011. The historical c32 reranker reached 0.7941871. Later Full256 work identified an in-sample stacking mismatch and added a fail-closed OOF pool schema. The final OOF raw-rank-512 c64 JSpaceV2 control reached 0.7875678 versus its 0.7861646 base: +0.0014032, 95% CI [+0.0011841, +0.0016335]. This is statistically positive but practically negligible. Test route tensors, labels, predictions, and metrics remained unopened; the stop handoff records one incidental shared-catalog metadata/token-row exposure during schema inspection.

Aggregate evidence is published under `results/audits/`. It contains no request-level rows, checkpoints, or tensors. The completed safe-stop OOF candidate-ranker metrics and access audit are included.

Full provenance and limitations are in docs/. This source release does not claim that the current prototype meets the eventual accuracy target or is a deployable cache controller.

## Contributing

Please include the exact data schema, complete-request split, command line, seed, device, PyTorch version, checkpoint provenance, per-horizon request-macro Recall@8, candidate coverage, and calibration metrics in any result report. Confirm that the outer test partition was not used for model selection.

Do not commit raw traces, checkpoints, credentials, or private RunPod paths. Add a focused regression test for data-contract, causal-mask, candidate-index, or test-sealing changes.

## License

No license file is included in this initial source publication. Add an explicit project license before redistributing or incorporating the code elsewhere.
