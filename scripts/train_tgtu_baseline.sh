#!/usr/bin/env bash
# TGTU Baseline+CPPI+AffineCalibration from-scratch training (myenv)
set -euo pipefail

# ---- activate myenv first ----
# source /root/miniconda3/etc/profile.d/conda.sh && conda activate myenv

cd /root/autodl-tmp/mkd-main/new-train

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

HASH="${HASH:-e1-api-masked-baseline-CPPI-k6-c4-affinecalib-tgtu-fromscratch}"

python run_mdt_seg.py \
  --gpus 0 \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt \
  --val_split_file test.txt \
  --test_split_file test.txt \
  --image_size_2d 512 \
  --batch_size 16 \
  --epochs 60 \
  --learning_rate 8e-5 \
  --decoder_lr 8e-5 \
  --weight_decay 1e-4 \
  --cosine_warmup 3 \
  --cosine_min_lr 1e-6 \
  --lr_flat_ratio 0.3 \
  --mixed_precision True \
  --grad_clip 5.0 \
  --early_stop_patience 10 \
  --random_state 2023 \
  --ct_backbone convnextv2_nano \
  --pet_backbone mit_b1 \
  --ct_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano \
  --pet_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1 \
  --cppi_num_clusters 6 \
  --cppi_build_stage 4 \
  --use_tgtu_fusion True \
  --tgtu_use_text True \
  --tgtu_use_turr_loss True \
  --tgtu_turr_interval 5 \
  --eval_full_pet True \
  --eval_fixed_missing_pet True \
  --eval_random_missing_pet False \
  --vis_every_epoch False \
  --enable_wandb False \
  --checkpoint_root /root/autodl-tmp/mkd-main/new-train/checkpoints_new \
  --hash "${HASH}"
