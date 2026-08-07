# Validation summaries

These CSV files are small, source-controlled summaries only. They contain validation metrics and no model weights or raw token traces.

- historical_endpoint_validation_metrics.csv: original HARP8 full-split endpoint.
- h34_continue_recall_validation_metrics.csv: selected H3/H4 continuation checkpoint.
- temp1_continue_partial_latest_validation_metrics.csv: latest validation state from the paused temperature-1 continuation.

All values are validation-only. The outer test partition was not opened for the expansion experiments.

The `audits/` directory contains compact aggregate JSON evidence for the original validation controls, J-Lens/raw-residual complementarity, the OOF Full256 proxy, and the final stopped raw-rank-512 c64 JSpaceV2 control. It also includes small non-request-level horizon, layer, domain, position, training-history, and provenance tables for that final run. No checkpoints, raw traces, model inputs, or request-level metrics are included.
