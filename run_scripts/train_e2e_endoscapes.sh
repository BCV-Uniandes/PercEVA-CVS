#!/usr/bin/env bash
# train_e2e_endoscapes.sh
#
# End-to-end training (encoder + perceiver jointly) on Endoscapes-CVS201.
# Methodology-matched to the SAGES E2E recipe (see configs/end2end_endoscapes.yaml
# for caveats — not pinned to one specific documented "best run" number the
# way the SAGES config is).
#
# Single GPU by default. For multi-GPU DDP, launch with torchrun instead —
# see the "multi-GPU DDP" example in README's Training section.
#
# Usage:
#   bash run_scripts/train_e2e_endoscapes.sh [extra python args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python main_scripts/main_e2e.py --dataset endoscapes --config configs/end2end_endoscapes.yaml "$@"
