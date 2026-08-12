#!/usr/bin/env bash
# train_encoder_endoscapes.sh
#
# Stage 1: trains the EVA-02 image encoder on Endoscapes-CVS201.
# Reproduces weights/eva02_enc_endoscapes (test mAP 0.6726).
#
# NOTE: this is intentionally not named after weights/best_eva02_enc_endoscapes.pt
# in the sibling MaskCVS repo — that file underperforms badly (mAP 0.26, near
# random) per MaskCVS/outputs/encoder_EVA02_Endoscapes/test_metrics.json. See
# configs/eva02_endoscapes.yaml for details.
#
# Usage:
#   bash run_scripts/train_encoder_endoscapes.sh [extra python args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python main_scripts/main_image_encoder.py --dataset endoscapes --config configs/eva02_endoscapes.yaml "$@"
