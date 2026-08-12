#!/usr/bin/env bash
# train_e2e_sages.sh
#
# End-to-end training (encoder + perceiver jointly) on SAGES_2024.
# Reproduces the paper's PercEVA-CVS E2E result: test mAP 0.6536, test BAcc 0.6481.
#
# Single GPU by default. For multi-GPU DDP, launch with torchrun instead —
# see the "multi-GPU DDP" example in README's Training section.
#
# Usage:
#   bash run_scripts/train_e2e_sages.sh [extra python args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python main_scripts/main_e2e.py --dataset sages --config configs/end2end_sages.yaml "$@"
