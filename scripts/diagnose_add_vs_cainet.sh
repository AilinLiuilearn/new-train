#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python diagnose_add_vs_cainet.py \
  --add_checkpoint /root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/full-add-baseline \
  --fusion_checkpoint /root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/full-cainet-s1s4-both-imagenet-ema-s2023 \
  --root /root/autodl-tmp/data/PCLT20K \
  --split_file test.txt \
  --batch_size 1 --num_workers 4 \
  --output_dir "diagnostics/add-vs-cainet-$(date +%Y%m%d-%H%M%S)" \
  "$@"
