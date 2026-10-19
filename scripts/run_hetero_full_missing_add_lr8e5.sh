#!/usr/bin/env bash
# Fixed clean heterogeneous Full/Missing rerun. Path overrides only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN_HASH="${RUN_HASH:-hetero-full-missing-add-lr8e5-ema-s2023-$(date +%Y%m%d-%H%M%S)}"
exec "${PYTHON_BIN:-python}" run_full_missing_baseline.py \
  --hash "$RUN_HASH" \
  --checkpoint_root "${CHECKPOINT_ROOT:-./checkpoints_new}" \
  --root "${DATA_ROOT:-/root/autodl-tmp/data/PCLT20K}" \
  --train_split_file train_original.txt \
  --val_split_file test.txt --test_split_file test.txt \
  --allow_val_equals_test true \
  --image_size_2d 512 --batch_size 16 --num_workers 4 \
  --aug_mode cipa --norm_mode imagenet \
  --ct_backbone convnextv2_nano --pet_backbone mit_b1 \
  --ct_pretrained_path "${CT_WEIGHTS:-/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano}" \
  --pet_pretrained_path "${PET_WEIGHTS:-/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1}" \
  --pretrained true --decoder_channels 512 256 128 64 \
  --use_deep_supervision false --deep_supervision false \
  --train_batch_mode mixed --accumulation_steps 1 \
  --epochs 60 --early_stop_patience 10 \
  --optimizer adamw --learning_rate 8e-5 --weight_decay 1e-4 \
  --cosine_warmup 3 --cosine_min_lr 1e-6 --lr_flat_ratio 0.3 \
  --mixed_precision true --grad_clip 5.0 \
  --ema_enabled true --ema_decay 0.999 \
  --ema_start_epoch 3 --ema_decay_warmup true \
  --hd95_mode cipa_official --random_state 2023 \
  --final_test_missing_rates 0 0.25 0.5 0.75 1
