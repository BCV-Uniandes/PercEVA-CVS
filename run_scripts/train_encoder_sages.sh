#!/usr/bin/env bash
# train_encoder_sages.sh
#
# Stage 1: trains the EVA-02 image encoder on SAGES_2024.
# Reproduces weights/best_eva02_enc_cvs.pt.
#
# Usage:
#   bash run_scripts/train_encoder_sages.sh [extra python args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python main_scripts/main_image_encoder.py --dataset sages --config configs/eva02.yaml "$@"
