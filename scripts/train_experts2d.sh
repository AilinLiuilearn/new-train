#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mode="${1:?usage: train_experts2d.sh ct|pet|joint [options]}"
shift
python -u run_experts.py --mode "$mode" \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split train_original.txt --val_split test.txt --test_split test.txt \
  --allow_val_equals_test --image_size 512 --batch_size 4 --num_workers 4 \
  --epochs 60 --learning_rate 8e-5 --norm zscore --log_interval 50 \
  --output "checkpoints_new/EXPERTS2D/${mode}-$(date +%Y%m%d-%H%M%S)" \
  "$@"
