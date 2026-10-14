#!/usr/bin/env bash
# Fresh experiment. Extra arguments override defaults for matched ablations.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u run_decoder_fg_bg_prototype_add.py \
  --hash add-fgbg-per-scale-ds020-adaptive-k4-s2023 \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt \
  --val_split_file test.txt \
  --test_split_file test.txt \
  --allow_val_equals_test true \
  --image_size_2d 512 \
  --batch_size 16 \
  --num_workers 4 \
  --pin_memory true \
  --aug_mode cipa \
  --norm_mode imagenet \
  --hd95_mode cipa_official \
  --ct_backbone convnextv2_nano \
  --pet_backbone mit_b1 \
  --ct_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano \
  --pet_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1 \
  --pretrained true \
  --decoder_channels 512 256 128 64 \
  --use_deep_supervision true \
  --deep_supervision false \
  --train_batch_mode mixed \
  --epochs 60 \
  --accumulation_steps 1 \
  --optimizer adamw \
  --learning_rate 8e-5 \
  --weight_decay 1e-4 \
  --cosine_warmup 3 \
  --cosine_min_lr 1e-6 \
  --lr_flat_ratio 0.3 \
  --mixed_precision true \
  --grad_clip 5.0 \
  --early_stop_patience 10 \
  --random_state 2023 \
  --ema_enabled false \
  --prototype_enabled true \
  --prototype_gain_mode per_scale \
  --prototype_deep_supervision_weight 0.2 \
  --prototype_adaptive_sampling true \
  --prototype_sampling_decay 0.99 \
  --prototype_max_patches 16 \
  --prototype_logit_scale 2.0 \
  --prototype_correction_alpha 0.1 \
  --prototype_dim 32 \
  --prototype_clusters 4 \
  --prototype_capacity 256 \
  --prototype_min_candidates 24 \
  --prototype_decay 0.999 \
  --prototype_refresh_steps 20 \
  --prototype_warmup_steps 100 \
  --prototype_min_gain 0.01 \
  --prototype_ct_aux_weight 0.1 \
  --log_interval 100 \
  --checkpoint_root /root/autodl-tmp/mkd-main/new-train/checkpoints_new \
  --final_test_missing_rates 0 0.25 0.5 0.75 1 \
  "$@"
