#!/usr/bin/env bash
# Fresh experiment. Extra arguments override defaults for matched ablations.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u run_ct_full_matrix_add.py \
  --hash add-ctfull-kv-lowrank-r8-fullinit-s2023 \
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
  --use_deep_supervision false \
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
  --matrix_value_mode lowrank \
  --matrix_value_rank 8 \
  --matrix_value_init full \
  --matrix_enabled true \
  --matrix_clusters 4 \
  --matrix_capacity 256 \
  --matrix_min_candidates 24 \
  --matrix_min_cluster_samples 2 \
  --matrix_samples_per_class 8 \
  --matrix_ridge 0.01 \
  --matrix_decay 0.99 \
  --matrix_temperature 0.5 \
  --matrix_gate_init 0.1 \
  --matrix_outlier_filter false \
  --matrix_warmup_steps 100 \
  --matrix_refresh_steps 20 \
  --log_interval 100 \
  --checkpoint_root /root/autodl-tmp/mkd-main/new-train/checkpoints_new \
  --final_test_missing_rates 0 0.25 0.5 0.75 1 \
  "$@"
