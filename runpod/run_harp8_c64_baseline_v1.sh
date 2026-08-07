#!/usr/bin/env bash
# Launch and continuously mirror the validation-only unchanged C64 baseline.

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

: "${SNAPSHOT_ROOT:?set SNAPSHOT_ROOT to the immutable source snapshot}"
: "${TRAIN_POOL:?set TRAIN_POOL to the overlay train pool}"
: "${VALIDATION_POOL:?set VALIDATION_POOL to the overlay validation pool}"
: "${RUN_ROOT:?set RUN_ROOT to a new local run directory}"
: "${PERSIST_ROOT:?set PERSIST_ROOT to a new persistent mirror directory}"

readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly DEVICE="${DEVICE:-cuda:0}"
readonly EPOCHS="${EPOCHS:-20}"
readonly BATCH_SIZE="${BATCH_SIZE:-16}"
readonly LEARNING_RATE="${LEARNING_RATE:-1e-4}"
readonly SEED="${SEED:-42}"
readonly BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"
readonly MIRROR_INTERVAL="${MIRROR_INTERVAL:-60}"
readonly OUTPUT_DIR="$RUN_ROOT/output"
readonly LOG_DIR="$RUN_ROOT/logs"

[[ ! -e "$RUN_ROOT" ]] || {
    echo "fatal: refusing to reuse local run root $RUN_ROOT" >&2
    exit 1
}
[[ ! -e "$PERSIST_ROOT" ]] || {
    echo "fatal: refusing to reuse persistent run root $PERSIST_ROOT" >&2
    exit 1
}
[[ -f "$SNAPSHOT_ROOT/runpod/harp8_c64_baseline_v1.py" ]] || {
    echo "fatal: baseline runner missing from snapshot" >&2
    exit 1
}
[[ -f "$TRAIN_POOL/manifest.json" && -f "$VALIDATION_POOL/manifest.json" ]] || {
    echo "fatal: overlay candidate pools are incomplete" >&2
    exit 1
}

mkdir -p -- "$LOG_DIR" "$PERSIST_ROOT"
readonly RUNNING_MARKER="$RUN_ROOT/RUNNING"
readonly HEARTBEAT="$RUN_ROOT/HEARTBEAT"
date -u +%Y-%m-%dT%H:%M:%SZ >"$RUNNING_MARKER"

mirror_loop() {
    while [[ -f "$RUNNING_MARKER" ]]; do
        date -u +%Y-%m-%dT%H:%M:%SZ >"$HEARTBEAT"
        rsync --archive --no-owner --no-group --partial -- \
            "$RUN_ROOT/" "$PERSIST_ROOT/"
        sleep "$MIRROR_INTERVAL"
    done
}

mirror_loop &
readonly MIRROR_PID=$!
trap 'kill "$MIRROR_PID" 2>/dev/null || true' EXIT

cd -- "$SNAPSHOT_ROOT"
set +e
PYTHONPATH="$SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" runpod/harp8_c64_baseline_v1.py \
    --train-pool "$TRAIN_POOL" \
    --validation-pool "$VALIDATION_POOL" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --seed "$SEED" \
    --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
    2>&1 | tee "$LOG_DIR/training.log"
readonly TRAIN_STATUS=${PIPESTATUS[0]}
set -e

mv -- "$RUNNING_MARKER" "$RUN_ROOT/FINALIZING"
wait "$MIRROR_PID" 2>/dev/null || true
if [[ "$TRAIN_STATUS" -eq 0 ]]; then
    mv -- "$RUN_ROOT/FINALIZING" "$RUN_ROOT/COMPLETE"
else
    mv -- "$RUN_ROOT/FINALIZING" "$RUN_ROOT/FAILED"
fi
date -u +%Y-%m-%dT%H:%M:%SZ >"$HEARTBEAT"
rsync --archive --no-owner --no-group --partial -- \
    "$RUN_ROOT/" "$PERSIST_ROOT/"
exit "$TRAIN_STATUS"
