#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
output="${1:?usage: bash scripts/train_stage1.sh OUTPUT_DIR [options]}"
shift
python -u run_staged.py --stage 1 --output "$output" \
  --val_split_file test.txt --test_split_file test.txt --allow_val_equals_test "$@"
