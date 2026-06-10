# 教师模型性能提升方案

## 当前性能分析 (Epoch 5-8)
- Dice: 0.75左右
- IoU: 0.60左右  
- HD95: 19-20
- 问题: Epoch 6出现性能下降,说明训练不够稳定

---

## 一、模型架构优化方案

### 1.1 升级Backbone (最重要)
**当前**: convnext_nano (80, 160, 320, 640)
**建议**: 升级到更强的backbone

```python
# 优先级从高到低:
1. convnext_small (96, 192, 384, 768) - 推荐首选
2. pvt_v2_b2 (64, 128, 320, 512) - 平衡性能
3. swin_tiny_patch4_window7_224 - Transformer架构
```

**预期提升**: Dice +2-4%

### 1.2 增强解码器容量
**当前**: decoder_channels=(512, 256, 128, 64)
**建议**: 增加到 (768, 384, 192, 96)

### 1.3 启用TCPM融合模块
**当前**: use_tcpm=True (已启用)
**优化**: 调整CUDM权重参数

```python
cudm_tumor_weight: 0.0 → 1.0  # 启用肿瘤区域对比学习
cudm_bg_weight: 0.0 → 0.5     # 背景区域对比
cudm_orth_weight: 0.0 → 0.3   # 正交约束
```

---

## 二、训练策略优化

### 2.1 学习率调整
**当前**: lr=1.2e-4, decoder_lr=1.2e-4
**问题**: 学习率可能偏高,导致训练不稳定

**建议方案A (保守)**:
```python
learning_rate: 1.2e-4 → 8e-5
decoder_lr: 1.2e-4 → 1.0e-4
cosine_warmup: 3 → 5
lr_flat_ratio: 0.1 → 0.15
```

**建议方案B (激进 - 大模型)**:
```python
learning_rate: 1.5e-4
decoder_lr: 2.0e-4
cosine_warmup: 8
lr_flat_ratio: 0.2
```

### 2.2 正则化增强
**当前**: weight_decay=5e-4
**建议**: 
```python
weight_decay: 5e-4 → 1e-3
grad_clip: 1.0 → 0.5
ema_decay: 0.9995 → 0.9998
ema_warmup_epochs: 5 → 8
```

### 2.3 训练周期延长
**当前**: epochs=80
**建议**: epochs=120-150

### 2.4 Batch Size优化
**当前**: batch_size=16, accumulation_steps=1
**建议**: 
```python
batch_size: 12
accumulation_steps: 2
```

---

## 三、拉大教师-学生差距的关键策略

### 3.1 信息优势最大化
- 强化PET特征提取: 使用更深的PET编码器
- 跨模态注意力: 增强CT-PET交互
- 多尺度融合: 在多个层级融合双模态

### 3.2 容量优势
- 更大的模型: 学生用pico,教师用small/base
- 更深的解码器: 教师使用更多解码层
- 集成学习: 教师可以是多模型集成

### 3.3 训练优势
- 更长训练: 教师训练150 epochs vs 学生80 epochs
- 更强数据增强: 教师使用更复杂的增强策略

---

## 四、推荐实施方案

### 阶段1: 快速验证 (优先级最高)
**目标**: 在当前架构下快速提升2-3%

**配置修改**:
```bash
--backbone convnext_nano
--learning_rate 8e-5
--decoder_lr 1.0e-4
--weight_decay 1e-3
--grad_clip 0.5
--ema_decay 0.9998
--cudm_tumor_weight 1.0
--cudm_bg_weight 0.5
--cudm_orth_weight 0.3
--epochs 100
--cosine_warmup 5
--lr_flat_ratio 0.15
```

**预期**: Dice 0.75 → 0.77-0.78

### 阶段2: 架构升级 (中期目标)
**目标**: 通过更强backbone提升到0.80+

**配置修改**:
```bash
--backbone convnext_small
--pretrained_path ./pretrained/convnext_small
--learning_rate 6e-5
--decoder_lr 8e-5
--epochs 120
--batch_size 12
--accumulation_steps 2
```

**预期**: Dice 0.78 → 0.80-0.82

### 阶段3: 极致优化 (长期目标)
**目标**: 冲击0.85+

**配置修改**:
```bash
--backbone swin_base_patch4_window7_224
--learning_rate 5e-5
--epochs 150
```

**预期**: Dice 0.82 → 0.85+

---

## 五、快速启动命令

### 方案A: 保守优化 (推荐先试)
```bash
cd /root/autodl-tmp/mkd-main/new-train
python run_mdt_seg.py \
  --_task MDT_Teacher_v2 \
  --backbone convnext_nano \
  --learning_rate 8e-5 \
  --decoder_lr 1e-4 \
  --weight_decay 1e-3 \
  --grad_clip 0.5 \
  --ema_decay 0.9998 \
  --cudm_tumor_weight 1.0 \
  --cudm_bg_weight 0.5 \
  --cudm_orth_weight 0.3 \
  --epochs 100 \
  --cosine_warmup 5 \
  --lr_flat_ratio 0.15
```

### 方案B: 激进升级 (需要更多资源)
```bash
python run_mdt_seg.py \
  --_task MDT_Teacher_Large \
  --backbone convnext_small \
  --pretrained_path ./pretrained/convnext_small \
  --learning_rate 6e-5 \
  --decoder_lr 8e-5 \
  --epochs 120 \
  --batch_size 12 \
  --accumulation_steps 2 \
  --weight_decay 1e-3 \
  --cudm_tumor_weight 1.0 \
  --cudm_bg_weight 0.5
```
