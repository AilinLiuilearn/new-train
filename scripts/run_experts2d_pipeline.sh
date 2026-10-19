#!/usr/bin/env bash
# Sequential complete pipeline; stops immediately if any stage fails.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
run_dir="${1:?usage: run_experts2d_pipeline.sh OUTPUT_DIR [common options]}"
shift
bash scripts/train_experts2d.sh ct --output "$run_dir/ct" "$@"
bash scripts/train_experts2d.sh pet --output "$run_dir/pet" "$@"
bash scripts/train_experts2d.sh joint --output "$run_dir/joint" \
  --ct_checkpoint "$run_dir/ct/best.pt" --pet_checkpoint "$run_dir/pet/best.pt" "$@"
