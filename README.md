# 干净异构 PET–CT baseline 与非对称融合

本分支 `exp/clean-hetero-asym-dssa-ega` 仅保留两个 Full-only 实验：

| 模型 | CT 编码器 | PET 编码器 | 融合 |
|---|---|---|---|
| baseline | ConvNeXtV2-Nano | MiT-B1 | 对齐后四尺度相加 |
| asymmetric | 同上 | 同上 | PET-DSSA 区域上下文 + CT-EGA 细节处理 + 双模态残差 |

两者使用同一个数据加载、训练、EMA、选模和评估流程。原分支保留原实验；本分支移除了其他融合、检索、原型、蒸馏、保护损失、混合训练入口及历史日志。不需要 baseline 检查点，也不支持续训。仅在启动时加载两个编码器各自的本地预训练权重。`--pretrained false` 仅供随机初始化 smoke test。

## 模块

输入为对齐后的四组 `C_s, P_s`，通道为 `[64,128,320,512]`；512 输入对应分辨率 `[128,64,32,16]`。额外输入是同一次增强/归一化后的 CT 图像。

1. 分别将 CT/PET 特征降至 `[16,16,32,32]` 通道。
2. PET 副分支以不超过 32×32 的网格执行 DSSA：区域 top-k → 像素 top-k → 聚合。原分辨率 PET 投影与上采样上下文拼接，产生区域 logits 和 PET 修正。
3. CT 图像逆归一化后构建一次拉普拉斯金字塔，按当前特征尺度取细节。EGA 用 PET 区域 logits、它的预测边缘、CT 细节来处理 CT 投影。
4. 输出 `F_s = C_s + P_s + delta_CT_s + delta_PET_s`。只将最后两组修正投影零初始化，所以初始化严格等于 Add。修正允许正负值。

全尺度启用，参数不共享。注意力与 EGA 核心使用 FP32，其余网络使用原 AMP 策略。新增融合参数 **233,602**，完整模型 **32,676,243**；Add 模型 **32,442,641**。PET 区域提示未被默认监督，不能直接解释为校准的病灶概率图。高频也不是肿瘤边界标签。

## 服务器运行

在已有仓库中切换到此分支后运行；不需要重新克隆数据或权重。

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch exp/clean-hetero-asym-dssa-ega
git pull --ff-only

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# CUDA + 真实编码器预训练预检，默认要求本地权重存在且可加载。
python scripts/check_asymmetric.py \
  --device cuda --image_size 512 --batch_size 2 \
&& bash scripts/run_full_asymmetric_lr4e5.sh
```

同协议异构相加 baseline：

```bash
bash scripts/run_full_add_hetero_lr4e5.sh
```

脚本的共同设置：512、batch16、60 epochs、AdamW、学习率 **4e-5**、weight decay1e-4、warmup3、cosine_min1e-6、flat_ratio0.3、CIPA 增强、ImageNet 归一化、GN 解码器、AMP、clip5、seed2023、EMA0.999。EMA 在 epoch4 开始时硬同步，再按成功更新跟踪；前3轮使用 raw 权重评估。

与现有实验一致，运行脚本使用 `train_original.txt / test.txt / test.txt`，显式允许验证复用测试。该结果是开发集选模结果，不是独立测试估计；可通过参数替换为独立 `val.txt`。

默认权重路径：

- CT：`pretrained/convnextv2_nano`
- PET：`pretrained/mit-b1`

默认 loss 只有 **BCE+Dice**。默认没有额外区域监督、教师、学生、保护、重建或检索损失。EMA 是评估权重平均，不提供训练监督。模型从编码器预训练开始联合训练，融合和解码器随机初始化。

## 开关与消融

```bash
# 关闭整个融合：完全绕过所有新分支，输出为严格相加。
bash scripts/run_full_asymmetric_lr4e5.sh --fusion_enabled false

# 可选区域监督，默认0；只在单独消融时启用。
bash scripts/run_full_asymmetric_lr4e5.sh --region_loss_weight 0.05

# 独立验证划分（两组实验都需使用相同覆盖参数）。
bash scripts/run_full_asymmetric_lr4e5.sh \
  --val_split_file val.txt --allow_val_equals_test false
```

区域监督开启时：`L=L_main+weight*mean_s(L_BCEDice(region_s,maxpool_s(GT)))`。平均四尺度而不累加四倍权重；下采样目标代表网格是否含前景。该粗区域目标不能用于边界评价。关闭时不计算此损失，日志区域损失为0。

`--fusion add` 构建真正不含融合参数的 baseline。用于论文对照时优先运行独立 baseline 脚本，关闭融合开关仅用于功能消融。训练脚本生成时间戳目录；重复 `--hash` 遇到已有结果会报错，避免覆盖。

检查点保存 `config_args.json`、`train_log.csv`、`ckpt.best_full.pth.tar`、`ckpt.last.pth.tar` 和 `final_test_full.json`。最终评估严格按保存的模型配置和 raw/EMA 标记重建，不再次加载编码器预训练。日志包含两路编码器/对齐/解码/融合梯度、跳过更新次数、主/区域损失，以及每尺度修正与相加的范数比。梯度均值仅统计有限值，非有限更新反映在 `skipped_updates`；非 AMP 出现非有限梯度立即停止。

```bash
python eval_full.py \
  --checkpoint checkpoints_new/MDT/<run>/ckpt.best_full.pth.tar \
  --device cuda
```

## 验证范围

本地通过：模块合同测试、真实异构编码器的模型测试、两种训练入口的一轮合成数据训练/选模/重载/最终评估，以及 512×512、batch2 的两步 CPU 前反向预检。原始 DSSA/EGA 核心另做匹配权重数值核对，见源码说明。

本地没有 CUDA、实际 PCLT20K 数据及服务器编码器权重，因此未宣称 GPU AMP、batch16 显存、真实权重加载或分割提升已验证。服务器预检通过后再启动完整训练。

```bash
python -m pytest tests -q
# 不依赖数据/外部编码器权重的本地预检
python scripts/check_asymmetric.py \
  --device cpu --image_size 512 --batch_size 2 --pretrained false
```

依赖见 `requirements.txt`，源码归属、固定版本及适配差异见 [ASYMMETRIC_SOURCE.md](ASYMMETRIC_SOURCE.md)。
