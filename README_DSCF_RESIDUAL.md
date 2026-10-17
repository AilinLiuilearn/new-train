# Full Add + DSCF residual fusion

实验分支：`exp/full-add-dscf-residual`，从 `exp/full-cmx-ffm` 的 `c2708c1` 派生。

这是 Stage-1 完整 CT+真实 PET 实验。四尺度输出均为：

```text
F_s = C_s + P_s + U_s(DSCF(D_C_s(C_s), D_P_s(P_s)))
```

仅 `U_s` 的权重和 bias 置零，初始输出严格等于相加；内部投影、偏移预测与注意力保留普通初始化。
第一个反向步骤仅输出头获得非零修正分支梯度，更新之后梯度才能进入内部模块，这是预期行为。
最终投影没有 ReLU，修正可以为正或为负；没有输出乘以 2，也没有额外的标量零门控。

## 来源与修改边界

借鉴 CVPR 2025 **Keep the Balance** 的 DSCF，官方仓库：https://github.com/imcjx/KTB 。
固定 commit：`75370dd584fb9d8571bced1aabade198dfbdc80f`。
原始源码和许可证保存在 `third_party/KTB/`，详细修改说明见其中 README。

原生核心执行：两个模态独立低维投影 → 各自预测采样偏移 → 在双方位置采样两个模态
及联合 query → 模态加权 → 带相对位置偏置的注意力 → 恢复通道。
`dscf_source.py` 的四个类体保持源码一致，并通过 AST、数值和梯度对照测试。
外部相加残差、输出头零初始化、尺寸与 rank 配置和 FP32 包装是本项目适配，
不能称为完整 KTB 的原样复现；未引入其 MAPA、PEFT 或自教学损失。

| 尺度 | 输入 CT/PET 与输出形状（512 输入） | 低维通道 | 采样步长 | 每个模态的参考网格 |
|---|---|---|---|---|
| S1 | B×64×128×128 | 16 | 16 | 8×8 |
| S2 | B×128×64×64 | 16 | 8 | 8×8 |
| S3 | B×320×32×32 | 40 | 4 | 8×8 |
| S4 | B×512×16×16 | 64 | 2 | 8×8 |

两个网格合成 128 个采样位置。原始高分辨率 `CT+PET` 路径没有被这些网格降采样。
粗网格仍可能漏掉小病灶；保留相加可以保留原特征，但不能保证修正分支改善性能。
新增融合参数 **487,200**；模型总参数 **32,929,841**（默认异构编码器、GN 解码器）。

CT/PET 编码器、对齐器、GN 解码器、主 BCE+Dice、优化器、调度、EMA 和数据处理沿用当前基线。
学习率 `4e-5`，batch 16，512 输入，60 epochs，patience 60，EMA 0.999、epoch 3 后开始。
缺失路径继承当前基线：不编码缺失 PET，不调用 DSCF，直接使用 CT 特征；
`auto` 只对 Full 行调用融合。本实验没有接入 Stage-2 检索或建库。

## 获取正确分支并运行

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch exp/full-add-dscf-residual
git pull --ff-only

python -m pip install -r requirements_dscf.txt
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 预检通过才启动正式训练
python scripts/check_dscf.py --device cuda --image_size 512 --batch_size 2 \
  && bash scripts/run_full_dscf_residual.sh
```

`git switch` 若提示本地修改冲突，应保留修改并先处理冲突，不要强制覆盖。
默认输出：`checkpoints_new/MDT/full-add-dscf-residual-lr4e5-60ep-ema-s2023/`。
已有非空目录会拒绝覆盖；重复实验请给新 hash：

```bash
bash scripts/run_full_dscf_residual.sh --hash full-add-dscf-residual-lr4e5-s2024 --random_state 2024
```

同入口的纯相加对照（无融合参数，保持共享权重初始化与后续 RNG 一致）：

```bash
bash scripts/run_full_dscf_residual.sh \
  --dscf_enabled false --hash full-add-dscf-control-lr4e5-s2023
```

预检使用随机初始化的真实编码器和合成数据，不是数据集性能评估，也不验证服务器预训练文件。
CUDA 预检包含真实模型 AMP 前后向；修正核心强制 FP32，编码器与解码器沿用 AMP。
为区分 AMP 支持与初始 GradScaler 校准，预检初始 loss scale=128；正式训练保持基线默认值。
可另用 `--batch_size 16` 预检正式 batch 的 GPU 显存，不应由 batch=2 结果推断 batch=16 一定可运行。

## 日志与检查点

`train_log.csv` 保留原有指标和 `grad_pet_align`、`grad_fusion`，新增：

- `dscf_delta_ratio_s1` 到 `dscf_delta_ratio_s4`：每样本 `||delta||₂ / max(||C+P||₂,1e-6)` 的 epoch 均值。
- 初始为零属于预期；之后若长期接近零，应与输出头/采样分支梯度结合分析。
- 该比率衡量修正幅度，不代表病灶定位正确或对分割有益。

checkpoint 同时保存 raw/EMA、源码 pin、四尺度尺寸/rank/stride、精度和输出形式。
每次 final test 严格重建同一种模型。独立评估也会校验架构和源码元数据：

```bash
python eval_full_dscf.py \
  --checkpoint checkpoints_new/MDT/full-add-dscf-residual-lr4e5-60ep-ema-s2023/ckpt.best_full.pth.tar \
  --output_dir diagnostics/dscf-ema --use_ema true

python eval_full_dscf.py \
  --checkpoint checkpoints_new/MDT/full-add-dscf-residual-lr4e5-60ep-ema-s2023/ckpt.best_full.pth.tar \
  --output_dir diagnostics/dscf-raw --use_ema false
```

重载不需要原始预训练权重路径。评估目录须为空或不存在。
禁用融合的 checkpoint 也可使用同一评估入口，架构开关从 checkpoint 恢复。

## 集成 API

```python
from models.components.dscf_residual import DSCFResidualFusion

fusion = DSCFResidualFusion(enabled=True, image_size=512)
# ct_feats / pet_feats: S1..S4 的 BCHW 张量列表；通道 64/128/320/512。
fused_feats = fusion(ct_feats, pet_feats)
stats = fusion.pop_diagnostics()  # 训练模式下累积；返回后清空。
```

输入需同 dtype/device、同 batch/channel。PET 空间尺寸不同时沿用基线的 bilinear resize。
要求 FP32 master weights，使用 autocast 控制混合精度；不支持将整个模块 `.half()` 或 `.double()`。
`image_size` 必须 >=64 且为 32 的整数倍；实际特征必须产生至少 2×2 采样网格，否则明确报错。

## 验证与实验判据

```bash
python -m pytest -q tests/test_dscf_residual.py
python scripts/check_dscf.py --device cpu --image_size 512 --batch_size 2
```

覆盖：固定源码/许可 hash、提取类体一致、四个真实特征尺寸的源码输出和输入/参数梯度一致；
初始/关闭输出严格等于 Add；输出正负修正；两次更新后偏移与投影梯度非零；
低精度输入/FP32 核心；Missing/auto PET 隔离；独立 runner 两步训练、EMA 更新、final test 和严格重载；
主损失不变、目录防覆盖及 CUDA AMP 测试（无 GPU 时跳过）。

在相同 seed/数据/训练预算下与相加比较；先看是否有稳定的分病例改善，再关注小病灶召回、
假阳性与 HD95。单次最优 Dice 的小幅波动不能证明融合有效。
若多 seed 无稳定增益，或平均 Dice 提高却小病灶/边界持续恶化，应回到 Add，否定该修正假设。
本地 CPU 冒烟不证明 CUDA 可用、batch 16 显存足够或真实数据 Dice 提升。

本地验证记录（2026-10-17，PyTorch 2.5.1 CPU）：新模块测试 25 passed、1 CUDA skip；
回归共 169 passed、6 CUDA skipped、1 deselected。未运行的 1 项为旧的 CT-only 数据集测试，
原因是本地 OpenCV 原生依赖限制。512×512、batch=2 的真实双编码器预检通过，
两步合成数据主损失为 1.749383、1.581104；仅用来确认数值与梯度，不代表真实数据收益。
服务器 PyTorch 2.1/CUDA、实际预训练文件、真实数据和 batch 16 显存仍需服务器验证。
