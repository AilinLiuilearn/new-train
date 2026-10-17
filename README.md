# PET-CT 缺失 PET 分割：空间调制库

本分支只保留当前统一空间调制库及其训练、验证、数据和网络依赖。历史双原型、前景/背景原型、解析矩阵与低秩矩阵实验在原分支中保留。

## 当前模型

CT：ConvNeXtV2-Nano；PET：MiT-B1。四尺度对齐后，Full 正常相加融合并进入共享解码器。
四尺度各自维护统一CT Key（buffer，聚类/EMA更新）与可学习gamma/beta（Parameter，梯度更新）。
所有Full CT位置参与；首轮只积累Key并保持零Value，第二轮起逐位置读取和仿射补偿。
Full的detach CT执行同样检索，以detach融合特征监督检索后的输出；辅助损失只更新Value。
Missing只使用自身CT，最终分割损失同时训练Value与CT/解码器，不调用PET编码器。
本分支 `exp/ct-key-learnable-affine-value` 基于清理分支 `exp/ct-full-spatial-modulation-bank@e697de2`。

## 运行

```bash
cd /root/autodl-tmp/mkd-main/new-train
bash scripts/train_spatial_modulation.sh
```

默认每尺度 K=8、额外损失权重0.1；每尺度 K=4 对照：

```bash
bash scripts/train_spatial_modulation.sh \
  --matrix_clusters 4 \
  --hash add-learnable-affine-k4-lret010-s2023
```

关闭额外损失（Value仍通过Missing分割损失学习）：

```bash
bash scripts/train_spatial_modulation.sh \
  --matrix_mod_loss_enabled false \
  --hash add-learnable-affine-k8-no-lret-s2023
```

同一 checkpoint 开关补偿验证：

```bash
python -u eval_spatial_modulation_add.py \
  --checkpoint checkpoints_new/MDT/add-learnable-affine-k8-lret010-s2023/ckpt.best_joint.pth.tar \
  --root /root/autodl-tmp/data/PCLT20K \
  --use_ema false --compare_on_off true --missing_rates 0 1 \
  --output_dir checkpoints_new/MDT/add-learnable-affine-k8-lret010-s2023/diagnosis_on_off
```

## 文件职责

| 文件 | 职责 |
|---|---|
| `models/spatial_modulation_bank.py` | EMA CT Key、可学习仿射Value、检索输出归一化L1 |
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

`model.matrix`及`--matrix_*`命令行名称保留以减少接口改动，表示本仿射模块。
新checkpoint采用独立experiment_type，不能直接续训旧统计库或矩阵权重；原版保留在原分支。
数据、预训练权重和本地实验输出不纳入版本控制。

## 检查

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests -q
PYTHONPATH=. python tests/smoke_spatial_modulation.py --size 64 --backward
PYTHONPATH=. python tests/smoke_spatial_modulation.py --size 512
```

依赖：PyTorch及匹配torchvision、timm、transformers、numpy、scipy、Pillow、opencv-python-headless、medpy；测试另需pytest。
完整机制、更新规则及边界见 [SPATIAL_MODULATION_BANK_ZH.md](SPATIAL_MODULATION_BANK_ZH.md)。
