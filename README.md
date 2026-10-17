# PET-CT 缺失 PET 分割：空间调制库

本分支只保留当前统一空间调制库及其训练、验证、数据和网络依赖。历史双原型、前景/背景原型、解析矩阵与低秩矩阵实验在原分支中保留。

## 当前模型

CT：ConvNeXtV2-Nano；PET：MiT-B1。四尺度对齐后，Full 正常相加融合并进入共享解码器。
旁路根据 detach 的 CT/Full 配对特征学习局部 gamma/beta，以归一化 L1 监督。
各尺度统一库保存 CT Key 和配对 gamma/beta；所有 Full 位置参与流式累计，首轮不检索，第二轮起读取并同步 EMA 更新。
Missing 仅使用自身 CT 查询并应用调制，不调用 PET 编码器。

## 运行

```bash
cd /root/autodl-tmp/mkd-main/new-train
bash scripts/train_spatial_modulation.sh
```

默认每尺度 K=8、额外损失权重0.1；每尺度 K=4 对照：

```bash
bash scripts/train_spatial_modulation.sh \
  --matrix_clusters 4 \
  --hash add-spatial-modulation-unified-k4-lmod010-s2023
```

关闭额外损失：

```bash
bash scripts/train_spatial_modulation.sh \
  --matrix_mod_loss_enabled false \
  --hash add-spatial-modulation-k8-nomodloss-s2023
```

同一 checkpoint 开关补偿验证：

```bash
python -u eval_spatial_modulation_add.py \
  --checkpoint checkpoints_new/MDT/add-spatial-modulation-unified-k8-lmod010-s2023/ckpt.best_joint.pth.tar \
  --root /root/autodl-tmp/data/PCLT20K \
  --use_ema false --compare_on_off true --missing_rates 0 1 \
  --output_dir checkpoints_new/MDT/add-spatial-modulation-unified-k8-lmod010-s2023/diagnosis_on_off
```

## 文件职责

| 文件 | 职责 |
|---|---|
| `models/spatial_modulation_bank.py` | 调制生成器、归一化L1、统一流式库和检索 |
| `models/spatial_modulation_add_model.py` | Full/Missing 分流与解码器前接入 |
| `models/dual_shared_add_baseline.py`、`models/components/` | 共享骨干、对齐、相加融合和解码器 |
| `tasks/spatial_modulation_seg.py` | 混合训练、成功更新后入库、恢复 |
| `tasks/mdt_seg.py` | 复用的最终分割损失、指标、优化器和模型EMA |
| `utils/spatial_modulation_runtime.py` | 数据加载和checkpoint路径 |
| `utils/spatial_modulation_diagnostics.py` | 槽位和训练诊断 |
| `utils/spatial_modulation_regions.py` | 验证集GT区域校正强度统计 |
| `run_spatial_modulation_add.py` | 训练入口 |
| `eval_spatial_modulation_add.py` | 评估和开关诊断入口 |
| `tests/` | 当前模块、恢复、训练/评估和区域统计测试 |

为兼容已有权重，`model.matrix`的state_dict键名及`--matrix_*`参数保持不变；它们指当前仿射调制库，不是旧矩阵Value模型。
本次清理不改变训练公式、超参数、读取方式或选模规则。历史版本不需复制回本分支，可切换对应旧分支查看。
数据、预训练权重和本地实验输出不纳入版本控制。

## 检查

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests -q
PYTHONPATH=. python tests/smoke_spatial_modulation.py --size 64 --backward
PYTHONPATH=. python tests/smoke_spatial_modulation.py --size 512
```

依赖：PyTorch及匹配torchvision、timm、transformers、numpy、scipy、Pillow、opencv-python-headless、medpy；测试另需pytest。
完整机制、更新规则及边界见 [SPATIAL_MODULATION_BANK_ZH.md](SPATIAL_MODULATION_BANK_ZH.md)。
