# Top-k Experts Predictor (HARP8)

This repository contains the source release of the HARP8 multi-horizon expert-route predictor used in the Qwen3.6/MoE caching study. HARP8 predicts the target router top-8 experts for horizons t+1 through t+8 and exposes candidate pools for a separate scheduler or reranker.

The model is a research prototype and is router-preserving: predictions do not replace native routing or change model outputs.

## Included

- harp8/: model, causal data adapter, feature preparation, losses, training, evaluation, candidate export, and set reranker.
- tests/: focused generator and reranker contract tests.
- docs/: implementation/validation handoff and chronological accuracy-expansion ledger.
- results/: small validation metric summaries.

Raw traces, PCA arrays, checkpoints, candidate memmaps, RunPod scripts, credentials, and private infrastructure details are intentionally excluded.

## Install and test

Python 3.10+ and PyTorch 2.4+ are required.

~~~text
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest -q tests/test_harp8.py tests/test_harp8_reranker.py
~~~

The verified source release passes 16 focused tests. GPU training is not required for those tests.

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

Export a scheduler-facing candidate pool:

~~~text
python -m harp8.candidates --capture-dir <capture-directory> --mtp-dir <mtp-directory> --target-state-features <feature-output>/local_post_layer_pca.npy --mtp-state-features <feature-output>/mtp_hidden_pca.npy --checkpoint <training-output>/best.pt --output-dir <candidate-pool> --candidate-count 32 --split validation --device cuda:0
~~~

Train the set reranker only after measuring candidate-pool oracle coverage:

~~~text
python -m harp8.reranker --train-pool <train-pool> --validation-pool <validation-pool> --output-dir <reranker-output> --candidate-count 32 --device cuda:0
~~~

Use python -m harp8.train --help for the full set of controls and ablations.

## Reproducibility and status

The loader splits by complete request rather than individual token. The historical full validation anchor was 410 requests with request-macro Recall@8 H1 0.8156604, H2 0.7999698, H3 0.7812144, H4 0.7569591, mean H1-H4 0.7884509.

The best completed expansion generator reached mean H1-H4 0.7931011. The paused c32 reranker reached 0.7941871. The c64 candidate pool had a 0.9740958 oracle coverage ceiling but was not reranked before the pause. These are validation-only figures; the outer test partition remained sealed.

Full provenance and limitations are in docs/. This source release does not claim that the current prototype meets the eventual accuracy target or is a deployable cache controller.

## Contributing

Please include the exact data schema, complete-request split, command line, seed, device, PyTorch version, checkpoint provenance, per-horizon request-macro Recall@8, candidate coverage, and calibration metrics in any result report. Confirm that the outer test partition was not used for model selection.

Do not commit raw traces, checkpoints, credentials, or private RunPod paths. Add a focused regression test for data-contract, causal-mask, candidate-index, or test-sealing changes.

## License

No license file is included in this initial source publication. Add an explicit project license before redistributing or incorporating the code elsewhere.
