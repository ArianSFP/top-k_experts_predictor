# Contributing

HARP8 is experimental research code for multi-horizon top-k expert prediction.

Before opening a pull request:

1. Run pytest -q tests/test_harp8.py tests/test_harp8_reranker.py.
2. Record Python, PyTorch, CUDA, and GPU versions for any training result.
3. State the complete-request split and confirm that the outer test set was not used for selection.
4. Report request-macro Recall@8 separately for every horizon and include candidate-pool coverage when using the reranker.
5. Keep checkpoints, raw traces, credentials, and private infrastructure paths out of commits.

Changes to the data contract, causal masking, candidate indexing, or test sealing should include a regression test.
