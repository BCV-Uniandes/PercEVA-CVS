#!/usr/bin/env bash
# train_perceiver_sages.sh
#
# Stage 2: trains the gated temporal Perceiver on SAGES_2024 pre-extracted
# features (see feature_extractor/extract_ft.py). Reproduces weights/best_perceiver
# (originally trained under the run name classic-sweep-767, test mAP macro 0.6604
# with all inference_temporal.py fixes applied — see CLAUDE.md).
#
# Usage:
#   bash run_scripts/train_perceiver_sages.sh [extra python args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python main_scripts/main_temporal_perceiver_concat_gate.py \
    --dataset sages \
    --config configs/perceiver_sages.yaml "$@"
