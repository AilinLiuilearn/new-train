# 干净异构编码器 Full/Missing 相加基线

分支：`exp/clean-full-missing-hetero-lr4e5`。
基于 `exp/mixed-mitb1-add-baseline@c64a8c6`，将 CT 编码器恢复为 ConvNeXtV2-Nano，PET 保持 MiT-B1，峰值学习率为 **4e-5**。
本分支只保留混合基线的训练、评估、预检、数据、模型与相关测试。
实验融合、原型/仿射/重建/蒸馏模块、Full-only/CT-only 入口、旧诊断工具及历史运行日志均移出当前分支；原实验分支保留。

## 结构

- CT：ConvNeXtV2-Nano；PET：MiT-B1，独立权重，分别加载对应本地预训练。
- 保留两侧四尺度 `1×1 Conv + BatchNorm + ReLU` 对齐，目标通道 `64/128/320/512`。512 输入对应空间尺寸 `128/64/32/16`。
- Full：`aligned_CT + aligned_PET → GroupNorm UNet 解码器`。
- Missing：`aligned_CT + 0 → 同一个解码器`。Missing 行不调用 PET 编码器或 PET 对齐层，推理允许 `pet=None`。
- batch=16，每个 batch 精确 **8 Full + 8 Missing**，按 seed 和 step 可复现地打乱状态。每个样本只走一种状态，不复制成两份；不保证每名患者整个训练历史恰好缺失一半。
- CT 一次处理 16 行，PET 只处理 8 个 Full 行，共享解码器一次处理 16 行。
- 损失：`0.5 × (Full BCE + Dice) + 0.5 × (Missing BCE + Dice)`；一次反向，一次 AdamW 更新。
- 只监督最终输出，不使用补偿、重建、蒸馏、深度监督或额外分割头。

## 默认设置

| 设置 | 默认值 |
|---|---|
| CT / PET 预训练 | `pretrained/convnextv2_nano` / `pretrained/mit-b1` |
| 学习率 / 优化器 / weight decay | `4e-5` / AdamW / `1e-4` |
| 训练预算 / patience | 60 / 60 |
| warmup / min LR / flat ratio | 3 轮 / `1e-6` / 0.3 |
| batch / 分辨率 / seed | 16 / 512 / 2023 |
| 增强 / 归一化 / HD95 | cipa / imagenet / cipa_official（pixel） |
| AMP / 梯度裁剪 | 开 / 5 |
| 模型 EMA | 开，0.999，前三轮不启用，第 4 轮同步后开始更新 |
| 数据划分 | train_original.txt / test.txt / test.txt，显式允许 val=test |

除 CT 编码器及其预训练来源外，设置与最近双 MiT-B1 混合基线一致。
更早 EMA 关闭、patience=10 的实验与本配置并非只差学习率；比较时应记录这些差异。
沿用已有 val=test 开发协议，按该集合选模后的最终结果不是独立测试指标。
相同随机种子不保证 CUDA 下逐位复现。

## 运行

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch exp/clean-full-missing-hetero-lr4e5
git pull --ff-only origin exp/clean-full-missing-hetero-lr4e5
bash scripts/train_full_missing_hetero_lr4e5.sh
```

脚本设置 OMP/MKL 线程为 1，并自动生成时间戳 hash。
输出：`checkpoints_new/MDT/mixed-add-hetero-lr4e5-60ep-ema-s2023-时间戳/`。
这是重新训练：只加载编码器预训练，不加载旧分割 checkpoint，不冻结网络。
发现已有 checkpoint 或训练日志会拒绝覆盖；当前不提供断点续训参数。
可在脚本末尾覆盖参数，例如匹配已有的无 EMA 实验：

```bash
bash scripts/train_full_missing_hetero_lr4e5.sh --ema_enabled false --hash mixed-add-hetero-lr4e5-noema-s2023-rerun1
```

## 选模和复评

每轮分别对全部验证样本运行 Full 和 Missing，按 `Joint Dice = 0.5×Full Dice + 0.5×Missing Dice` 保存同一个 `ckpt.best_joint.pth.tar`，同时保存 `ckpt.last.pth.tar`。
日志包含 Full/Missing 损失、样本数、Dice、HD95、两侧编码器/对齐层/解码器梯度及跳过更新数。
最终使用最佳 Joint checkpoint 记录的 raw/EMA 权重，输出缺失率 `0/0.25/0.5/0.75/1` 的指标和患者级缺失分配。
中间缺失率按患者分配，与训练 batch 内缺失不是同一口径。
独立复评自动遵循 checkpoint 选模权重，不重新加载编码器预训练：

```bash
python -u eval_full_missing_baseline.py --checkpoint checkpoints_new/MDT/你的实验目录/ckpt.best_joint.pth.tar
```

## 预检和测试

```bash
# AutoDL：真实预训练、CUDA、两步混合前反传；batch2 不代表 batch16 显存足够。
python scripts/check_full_missing_baseline.py --device cuda --image_size 512 --batch_size 2

# 无真实数据/权重的 CPU 结构检查
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/check_full_missing_baseline.py --device cpu --image_size 64 --batch_size 2 --pretrained false

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests -q
```

预检使用真实异构网络，检查两侧对齐、主要组件梯度及 Missing 禁止调用 PET。
预训练缺失或不兼容会报错，不静默替换随机网络；`--pretrained false` 用于显式结构预检。
基础依赖：PyTorch、torchvision、timm、transformers、safetensors、numpy、scipy、Pillow、opencv-python、medpy；测试另需 pytest。
本地验证使用合成数据与随机初始化，不代表已完成 AutoDL 真实数据训练或验证了分割收益。

## 本次本地验证

- 43 项测试通过，覆盖真实异构模型、精确一半分流、PET 泄漏防护、梯度、EMA、保存重载、最终 raw/EMA 选模及独立复评一致性。
- 512×512、batch=2、CPU、随机初始化：两步混合前向/反向/AdamW 更新通过；PET 每步只处理 1 个 Full 样本，纯 Missing 不调用 PET。
- 模型参数量：32,442,641。未执行 AutoDL 真实数据训练、未验证真实预训练文件或 CUDA batch=16 显存。
- 核心编码器、对齐、相加、解码器、数据增强、损失、指标、EMA 和调度实现沿用来源基线。
