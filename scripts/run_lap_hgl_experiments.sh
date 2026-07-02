#!/usr/bin/env bash
# LapHGL PET prior experiments on Full PET (B0 / P0 / P1)
set -euo pipefail
cd /root/autodl-tmp/mkd-main/new-train

COMMON=(
  --ct_backbone convnext_tiny
  --ct_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/convnext_tiny
  --use_deep_supervision true
  --batch_size 16
  --epochs 60
  --learning_rate 8e-5
  --train_pet_drop_prob 0.0
  --eval_full_pet true
  --aug_mode cipa
  --norm_mode cipa
  --use_aligned_loader true
)

echo "========== B0: CT-only (pet_prior_type=none) =========="
python run_mdt_seg_lap_hgl.py \
  "${COMMON[@]}" \
  --hash lap_hgl_b0_none \
  --pet_prior_type none

echo "========== P0: PET intensity prior (sanity check) =========="
python run_mdt_seg_lap_hgl.py \
  "${COMMON[@]}" \
  --hash lap_hgl_p0_intensity \
  --pet_prior_type intensity

echo "========== P1: Laplacian HGL prior (lite) =========="
python run_mdt_seg_lap_hgl.py \
  "${COMMON[@]}" \
  --hash lap_hgl_p1_lite \
  --pet_prior_type lap_hgl \
  --pet_prior_size lite
