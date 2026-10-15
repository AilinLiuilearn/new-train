#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
arm="${1:-full}"
if [[ $# -gt 0 ]]; then shift; fi
fusion=true; correction=true; guidance=true; aux=true
case "$arm" in
  add) fusion=false; correction=false; guidance=false; aux=false ;;
  aux) correction=false; guidance=false ;;
  residual) guidance=false; aux=false ;;
  full) ;;
  no_aux) aux=false ;;
  aux_no_guidance) guidance=false ;;
  *) echo 'Usage: bash scripts/run_full_lesion_guided_ablation.sh add|aux|residual|full|no_aux|aux_no_guidance [overrides]' >&2; exit 2 ;;
esac
python run_full_lesion_guided_fusion.py \
  --hash "full-add-lesion-guided-${arm}-imagenet-ema-s2023" \
  --lesion_fusion_enabled "$fusion" --lesion_correction_enabled "$correction" \
  --lesion_guidance_enabled "$guidance" --lesion_aux_loss_enabled "$aux" \
  --lesion_aux_loss_weight 0.1 --lesion_widths 16 32 64 64 \
  --require_cuda true --retry_failed_start true \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt --val_split_file test.txt --test_split_file test.txt \
  --allow_val_equals_test true --image_size_2d 512 --batch_size 16 \
  --num_workers 4 --pin_memory true --aug_mode cipa --norm_mode imagenet --hd95_mode cipa_official \
  --ct_backbone convnextv2_nano --pet_backbone mit_b1 \
  --ct_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano \
  --pet_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1 \
  --pretrained true --epochs 60 --learning_rate 8e-5 --weight_decay 1e-4 \
  --cosine_warmup 3 --cosine_min_lr 1e-6 --lr_flat_ratio 0.3 \
  --mixed_precision true --ema_enabled true --ema_decay 0.999 \
  --ema_start_epoch 3 --ema_decay_warmup true --grad_clip 5 \
  --early_stop_patience 10 --random_state 2023 "$@"
