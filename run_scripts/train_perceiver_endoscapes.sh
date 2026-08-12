#!/usr/bin/env bash
# train_perceiver_endoscapes.sh
#
# Stage 2: trains the gated temporal Perceiver on Endoscapes-CVS201
# pre-extracted features. Reproduces weights/perceiver_endoscapes
# (test mAP macro 0.6927).
#
# Usage:
#   bash run_scripts/train_perceiver_endoscapes.sh [extra python args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python main_scripts/main_temporal_perceiver_concat_gate.py \
    --dataset endoscapes \
    --config configs/perceiver_endoscapes.yaml "$@"
