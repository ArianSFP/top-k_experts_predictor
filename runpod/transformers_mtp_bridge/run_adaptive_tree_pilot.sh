#!/usr/bin/env bash
set -euo pipefail

# One-request, one-source-position schema pilot.  This intentionally uses the
# full 32-node first-teacher budget and stops after the blocking auditor.

usage() {
    echo "usage: run_adaptive_tree_pilot.sh MODEL_DIR NEW_OUTPUT_DIR [PROMPT_MANIFEST_JSONL] [--device cpu|cuda|cuda:N]" >&2
}

[[ $# -ge 2 ]] || { usage; exit 2; }

readonly MODEL_DIR="$1"
readonly OUTPUT_DIR="$2"
shift 2
PROMPT_MANIFEST=""
CAPTURE_DEVICE="${HARP_RTT_CAPTURE_DEVICE:-cuda}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            CAPTURE_DEVICE="$2"
            shift 2
            ;;
        --device=*)
            CAPTURE_DEVICE="${1#--device=}"
            shift
            ;;
        -*)
            usage
            exit 2
            ;;
        *)
            [[ -z "$PROMPT_MANIFEST" ]] || { usage; exit 2; }
            PROMPT_MANIFEST="$1"
            shift
            ;;
    esac
done
readonly PROMPT_MANIFEST
readonly CAPTURE_DEVICE
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PYTHON_BIN="${PYTHON_BIN:-python3}"
readonly RUN_ID="${HARP_RTT_RUN_ID:-harp_rtt_adaptive_mtp_tree_pilot_v1}"

test -d "$MODEL_DIR"
test -f "$MODEL_DIR/model.safetensors.index.json"
test ! -e "$OUTPUT_DIR"
if [[ -n "$PROMPT_MANIFEST" ]]; then
    test -f "$PROMPT_MANIFEST"
fi

command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/capture_transformers_adaptive_segment.py"
    --model "$MODEL_DIR"
    --output "$OUTPUT_DIR"
    --run-id "$RUN_ID"
    --source-positions 1
    --label-lookahead 3
    --max-tree-nodes 32
    --limit 1
    --seed 42
    --device "$CAPTURE_DEVICE"
)
if [[ -n "$PROMPT_MANIFEST" ]]; then
    command+=(--prompt-manifest "$PROMPT_MANIFEST")
fi

"${command[@]}"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_adaptive_tree_capture.py" \
    --capture "$OUTPUT_DIR" \
    --authoritative-native-weight-device "$CAPTURE_DEVICE"

echo "adaptive capture pilot audited; training remains stopped: $OUTPUT_DIR"
