#!/usr/bin/env bash
# Identical training protocol; only CACR/GCM switches and run id differ.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
arm="${1:-both}"
if [[ $# -gt 0 ]]; then shift; fi
case "$arm" in
  add)  cacr=false; gcm=false ;;
  cacr) cacr=true;  gcm=false ;;
  gcm)  cacr=false; gcm=true ;;
  both) cacr=true;  gcm=true ;;
  *) echo 'Usage: bash scripts/run_full_cainet_ablation.sh add|cacr|gcm|both [shared overrides]' >&2; exit 2 ;;
esac
python run_full_cainet_fusion.py \
  --hash "full-cainet-s1s4-${arm}-imagenet-ema-s2023" \
  --cacr_enabled "$cacr" --gcm_enabled "$gcm" \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt \
  --val_split_file test.txt --test_split_file test.txt \
  --allow_val_equals_test true \
  --image_size_2d 512 --batch_size 16 --num_workers 4 --pin_memory true \
  --aug_mode cipa --norm_mode imagenet --hd95_mode cipa_official \
  --ct_backbone convnextv2_nano --pet_backbone mit_b1 \
  --ct_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano \
  --pet_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1 \
  --pretrained true --epochs 60 --learning_rate 8e-5 --weight_decay 1e-4 \
  --cosine_warmup 3 --cosine_min_lr 1e-6 --lr_flat_ratio 0.3 \
  --mixed_precision true --ema_enabled true --ema_decay 0.999 \
  --ema_start_epoch 3 --ema_decay_warmup true \
  --grad_clip 5 --early_stop_patience 10 --random_state 2023 "$@"
