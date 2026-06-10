#!/bin/bash
# 教师模型优化训练脚本
# 使用改进的配置来提升教师模型性能

set -e

echo "=========================================="
echo "教师模型性能提升训练"
echo "=========================================="

# 配置选择
CONFIG_TYPE=${1:-"conservative"}  # conservative / aggressive / large

case $CONFIG_TYPE in
  "conservative")
    echo "使用保守优化方案 (推荐首选)"
    echo "预期提升: Dice 0.75 → 0.77-0.78"
    TASK_NAME="MDT_Teacher_Conservative"
    BACKBONE="convnext_nano"
    PRETRAINED="./pretrained/convnext_nano"
    LR=8e-5
    DECODER_LR=1e-4
    BATCH_SIZE=16
    ACCUM_STEPS=1
    EPOCHS=100
    ;;
    
  "aggressive")
    echo "使用激进优化方案 (需要验证)"
    echo "预期提升: Dice 0.75 → 0.78-0.80"
    TASK_NAME="MDT_Teacher_Aggressive"
    BACKBONE="convnext_nano"
    PRETRAINED="./pretrained/convnext_nano"
    LR=1.5e-4
    DECODER_LR=2.0e-4
    BATCH_SIZE=16
    ACCUM_STEPS=1
    EPOCHS=120
    ;;
    
  "large")
    echo "使用大模型方案 (需要更多资源)"
    echo "预期提升: Dice 0.75 → 0.80-0.82"
    TASK_NAME="MDT_Teacher_Large"
    BACKBONE="convnext_small"
    PRETRAINED="./pretrained/convnext_small"
    LR=6e-5
    DECODER_LR=8e-5
    BATCH_SIZE=12
    ACCUM_STEPS=2
    EPOCHS=120
    
    # 检查预训练权重
    if [ ! -d "$PRETRAINED" ]; then
      echo "警告: 未找到 convnext_small 预训练权重"
      echo "请先下载到 $PRETRAINED"
      exit 1
    fi
    ;;
    
  *)
    echo "错误: 未知配置类型 '$CONFIG_TYPE'"
    echo "用法: $0 [conservative|aggressive|large]"
    exit 1
    ;;
esac

echo ""
echo "配置参数:"
echo "  - Task: $TASK_NAME"
echo "  - Backbone: $BACKBONE"
echo "  - Learning Rate: $LR"
echo "  - Decoder LR: $DECODER_LR"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Accumulation: $ACCUM_STEPS"
echo "  - Epochs: $EPOCHS"
echo ""

# 确认执行
read -p "是否开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 开始训练
python run_mdt_seg.py \
  --_task "$TASK_NAME" \
  --backbone "$BACKBONE" \
  --pretrained_path "$PRETRAINED" \
  --use_tcpm True \
  --learning_rate $LR \
  --decoder_lr $DECODER_LR \
  --weight_decay 1e-3 \
  --grad_clip 0.5 \
  --ema_decay 0.9998 \
  --ema_warmup_epochs 8 \
  --cudm_tumor_weight 1.0 \
  --cudm_bg_weight 0.5 \
  --cudm_orth_weight 0.3 \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --accumulation_steps $ACCUM_STEPS \
  --cosine_warmup 5 \
  --lr_flat_ratio 0.15 \
  --early_stop_patience 25 \
  --bce_weight 0.8 \
  --dice_weight 1.2

echo ""
echo "=========================================="
echo "训练完成!"
echo "检查点保存在: ./checkpoints_new/MDT/$TASK_NAME/"
echo "=========================================="
