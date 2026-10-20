#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
checkpoint="${1:?usage: bash scripts/train_stage2.sh CT_CHECKPOINT OUTPUT_DIR [options]}"
output="${2:?usage: bash scripts/train_stage2.sh CT_CHECKPOINT OUTPUT_DIR [options]}"
shift 2
python -u run_staged.py --stage 2 --stage1_checkpoint "$checkpoint" --output "$output" \
  --val_split_file test.txt --test_split_file test.txt --allow_val_equals_test "$@"
