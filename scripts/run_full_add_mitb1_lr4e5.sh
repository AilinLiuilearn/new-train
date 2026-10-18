#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python run_full_add_baseline.py \
  --hash "full-add-mitb1-lr4e5-60ep-ema-s2023-$(date +%Y%m%d-%H%M%S)" \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt --val_split_file test.txt --test_split_file test.txt \
  --allow_val_equals_test true --image_size_2d 512 --batch_size 16 \
  --num_workers 4 --pin_memory true --aug_mode cipa --norm_mode imagenet --hd95_mode cipa_official \
  --ct_backbone mit_b1 --pretrained true \
  --ct_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1 \
  --epochs 60 --learning_rate 4e-5 --weight_decay 1e-4 --optimizer adamw \
  --cosine_warmup 3 --cosine_min_lr 1e-6 --lr_flat_ratio 0.3 \
  --mixed_precision true --ema_enabled true --ema_decay 0.999 \
  --ema_start_epoch 3 --ema_decay_warmup true --grad_clip 5 \
  --early_stop_patience 60 --random_state 2023 \
  --use_deep_supervision false --deep_supervision false --accumulation_steps 1 \
  --train_batch_mode full --pet_backbone mit_b1 \
  --pet_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1 \
  "$@"
