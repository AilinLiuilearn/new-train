# MiT-B1＋MiT-B1：batch内50% Full / 50% Missing相加基线

分支：`exp/mixed-mitb1-add-baseline`。
来源：`exp/full-add-local-reassembly@81ff8cf`，对齐到其中的 `scripts/run_full_add_mitb1_lr4e5.sh` 同构Full-only实验设置。
本实验复用其原始Add基线，不调用LocalReassemblyFusion、补偿库、仿射、蒸馏、内部相加保护损失或额外分割头。
原有模型、任务、数据、Full-only/CT-only训练入口和脚本均不修改；新增独立混合入口与启动/预检脚本。

## 网络和训练

CT和PET各一个独立MiT-B1：结构相同、参数不共享。初始化分别读取同一份MiT-B1编码器预训练，之后各自学习。
保留两侧全部四尺度Conv1×1+BN+ReLU对齐层，即使MiT原通道已为64/128/320/512也不删除。
512输入时尺度为128/64/32/16，解码器输入通道64/128/320/512，原共享GN UNet解码器通道512/256/128/64。

每个batch16，精确8个Full、8个Missing，step/seed决定可复现的位置分配。一次batch每个样本仅使用一种状态，不复制成16 Full＋16 Missing，不保证每个患者整个训练历史恰好一半缺失。
- Full：aligned_CT + aligned_PET → 共享解码器。
- Missing：aligned_CT + 0 → 同一解码器，不编码Missing行的PET。训练PET编码器实际只处理8行。
- CT编码器处理16行；PET对齐层只处理Full行；解码器处理16行。
- L = .5×(Full子集BCE+Dice) + .5×(Missing子集BCE+Dice)，一次backward、一次AdamW更新。
没有重建、知识库或其他辅助损失；仅最终分割监督。

| 配置 | 默认值 |
|---|---|
| CT / PET编码器 | mit_b1 / mit_b1，独立权重 |
| 预训练路径，两侧相同 | /root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1 |
| 峰值学习率 | 4e-5（warmup期间及余弦衰减时低于该值） |
| 优化器 / weight_decay | AdamW / 1e-4 |
| 训练预算 / patience | 60 / 60，与参考同构Full-only脚本一致 |
| warmup / min_lr / flat_ratio | 3轮 / 1e-6 / .3 |
| 输入 / batch / workers | 512 / 16 / 4 |
| 增强 / 归一化 / HD95 | cipa / imagenet / cipa_official |
| AMP / clip / seed | 开 / 5 / 2023 |
| 模型EMA | .999，前三轮不启用，第4轮同步后启用，decay_warmup开启 |
| 深度监督 / 梯度累积 | 关 / 1 |

这是一项新的完整训练：只加载编码器预训练，重新初始化对齐层和解码器，不加载之前训练好的分割checkpoint，不冻结编码器。
默认EMA和60轮预算来自参考分支的同构Full-only脚本；与旧补偿实验的EMA关闭/patience10配置不同，比较时应记录该差异。
这个实验可检验新编码器/学习率组合的混合基线；单次结果不能把差异全部归因于同构，也不能证明原框架的理论上限。

## 正式运行

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch exp/mixed-mitb1-add-baseline
git pull --ff-only origin exp/mixed-mitb1-add-baseline

# 可选：真实预训练、CUDA、batch2的两步混合前反传检查
python scripts/check_mixed_mitb1_baseline.py --device cuda --image_size 512 --batch_size 2

# 正式实验：batch16，每batch8 Full＋8 Missing
bash scripts/run_mixed_add_mitb1_lr4e5.sh
```

batch2预检通过不代表batch16显存足够。正式脚本自动设置OMP_NUM_THREADS=1和MKL_NUM_THREADS=1。
输出目录为 `checkpoints_new/MDT/mixed-add-mitb1-lr4e5-60ep-ema-s2023-时间戳/`。
可在命令末尾覆盖 `--hash 自定义实验名`；发现已有checkpoint或train_log时拒绝覆盖。该独立入口当前不提供断点续训参数。
编码器预训练路径缺失/不兼容时按原严格加载器报错。`--pretrained false`用于本地无权重预检，正式实验默认true。

如果需要与EMA关闭的旧实验做额外控制，可明确运行：
```bash
bash scripts/run_mixed_add_mitb1_lr4e5.sh --ema_enabled false --hash mixed-add-mitb1-lr4e5-noema-s2023
```
这不是默认设置；先完成与参考Full-only设置一致的主实验。

## 日志、选模和最终评价

每轮在整个验证集分别执行Full和Missing：Joint Dice=.5×Full Dice+.5×Missing Dice，按Joint保存 `ckpt.best_joint.pth.tar`，另存 `ckpt.last.pth.tar`。
两种状态最终来自同一个best_joint checkpoint，而不是各自挑最佳轮。模型EMA存在不等于被用于选模：最终严格遵循checkpoint记录的eval_weights（raw/ema）。
训练CSV记录Full/Missing样本数与损失、Full/Missing/Joint Dice及HD95、两侧编码器和对齐层/共享解码器梯度、跳过更新等；默认每100batch打印进度。
最终自动输出0/.25/.5/.75/1缺失率指标CSV/JSON及患者缺失分配；中间缺失率沿用原患者级固定随机分配，不等于训练batch级缺失比例。
数据采用train_original.txt / test.txt / test.txt并显式allow_val_equals_test，延续已有开发实验协议；验证和最终测试不是独立患者集合。

## 本地验证命令

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests/test_mixed_mitb1_baseline.py -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/check_mixed_mitb1_baseline.py --device cpu --image_size 64 --batch_size 2 --pretrained false
```

预检使用真实MiT-B1结构、实际对齐/解码器/分割损失，检查精确比例、两侧参数独立、PET只编码Full行、所有主要组件梯度有限且非零、纯Missing不调用PET。

## 本次验证结果

新增4项测试通过：实际脚本参数、真实MiT双分支/两侧对齐/精确一半路由与梯度、raw最佳权重最终评价、EMA最佳权重最终评价、存档严格重载及拒绝覆盖。
原基线、混合训练、模型EMA、MiT预训练加载和Full-only入口共59项回归测试通过；新版PyTorch环境下旧测试的torch.load通过TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1读取本地生成的可信测试checkpoint，新入口沿用显式weights_only=False的加载器。
512×512、batch2、真实双MiT-B1、CPU、无预训练，两步前向/反向/AdamW更新通过；每步PET实际编码1个Full样本，Missing不调用PET。模型参数量30,541,889。
本地没有AutoDL真实数据/真实预训练权重或CUDA；未执行正式训练，未验证batch16显存或分割收益。
