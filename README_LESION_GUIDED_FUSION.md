# Full-only Add + 病灶引导双模态残差融合

分支：`exp/full-add-lesion-guided`。
父基线：`exp/aligned-add-baseline` / `663f9786122365d8b2774e4497df92bc06b715d5`。
本分支只增加文件，原 baseline 模型、训练脚本、数据、损失、优化器、解码器和 EMA 文件保持不变。
本次接通 Stage-1 Full（CT+真实 PET），不改变现有 Module-1、建库/检索或 Missing 训练范式。

## 已实现结构

每个尺度：

```
X_C = Conv1x1/GN/GELU(C); X_P = Conv1x1/GN/GELU(P)
U_m = X_m + Conv1x1/GN/GELU(cat(DW3(X_m), DW5(X_m), DW7(X_m)))
Z = Conv1x1/GN/GELU(cat(U_C,U_P))
a = Conv3x3/GN/GELU -> Conv1x1(Z); m = sigmoid(a)
[g_C,g_P] = sigmoid(Conv3x3/GN/GELU -> Conv1x1(cat(Z,m)))
F = (C+P) + W_C(g_C*U_C) + W_P(g_P*U_P)
```

S1..S4 通道 `64/128/320/512`，512 输入空间尺寸 `128/64/32/16`；
旁路宽度 `16/32/64/64`。四尺度结构相同、参数独立，所有卷积 stride=1。
联合门的输出为 `2*d` 通道，两路独立 sigmoid，不要求和为 1。
`W_C/W_P` 是不带后续激活的线性 1x1 卷积，weight/bias 零初始化。
初始输出为 Add。没有双重残差，也没有用病灶图直接遮住 Add 主路径。
PET 空间尺寸按原 Add 的 bilinear / align_corners=False 方式对齐。

新增参数 **535,588**，逐尺度 **14,561 / 51,649 / 209,921 / 259,457**；
默认完整模型 **32,978,229**。参数量是实现测量值，不是性能保证。

## 损失和独立开关

主损失沿用 baseline BCE+Dice。
辅助损失在 FP32 中将每尺度 logits 上采样到原始 GT 尺寸，分别计算相同约定的 BCE+Dice，再平均四尺度：

```
L_total = L_main + lesion_aux_loss_weight * mean(L_aux_S1,...,L_aux_S4)
```

默认权重 0.1。不会把 GT 缩小到 16x16，以免直接在监督标签中丢掉小前景；
这也不代表粗尺度能恢复其网格无法表达的细节。

| 参数 | 默认 | 行为 |
| --- | --- | --- |
| `--lesion_fusion_enabled` | true | false 时无融合参数，直接四尺度 Add；需同时将辅助损失开关设为 false |
| `--lesion_correction_enabled` | true | false 时输出 Add，但训练中仍可计算辅助预测/监督 |
| `--lesion_guidance_enabled` | true | false 时门控的病灶概率输入置零；保留联合特征条件及相同门控参数形状 |
| `--lesion_aux_loss_enabled` | true | false 时完全不计算/反传辅助损失，记录的 aux 项为 0 |
| `--lesion_aux_loss_weight` | 0.1 | 有限非负数；0 也完全关闭辅助损失 |
| `--lesion_widths` | 16 32 64 64 | 四尺度旁路宽度；保存进 checkpoint，评估按记录重建 |

**关闭辅助损失不等于关闭病灶图引导。** `no_aux` 仍由特征预测病灶图，用主损失训练修正路径；
其预测图不再接受直接 GT 监督，不能把它当作经监督验证的语义概率。
若同时关闭 loss 和 guidance，辅助头不执行。辅助头保留在 state_dict 中以维持各消融配置的拓扑一致性；
不会用零乘损失伪造梯度。aux-only 中不执行门控和残差计算。

推理不需要 GT。完整版推理仍执行病灶头来生成门控条件，仅不返回辅助图/计算辅助损失；不能随意删掉该头。
验证/测试 loss 只计算主分割损失。训练日志新增：`train_loss_main`、`train_aux_raw`、
`train_aux_weighted`、`train_aux_s1..s4`、`grad_pet_align`、`grad_fusion`。
比较 train/val loss 时请用 `train_loss_main`，不要把含辅助项的 train_loss 当成同一口径。

## 服务器运行

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch exp/full-add-lesion-guided
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 无数据/无预训练权重预检：真实骨干，初始化等价，两个 AMP 更新
python scripts/check_lesion_guided.py --device cuda --image_size 512 --batch_size 2

# 预检通过后训练完整方案
bash scripts/run_full_lesion_guided_ablation.sh full

# 保留结构，只关闭辅助损失
bash scripts/run_full_lesion_guided_ablation.sh no_aux
```

启动脚本沿用当前对照协议：512、batch16、workers4、seed2023、ImageNet 归一化、
CIPA 增强、lr8e-5、AdamW、warmup3、60epoch、patience10、EMA0.999/start3、CIPA official HD95。
沿用 `train_original.txt/test.txt/test.txt` 与显式 allow_val_equals_test；该协议含 test 上模型选择，
不等于独立留出测试估计。此次不静默改变对照划分。

启动前检查 CUDA；CPU smoke 必须 `--require_cuda false`。只有启动配置与空日志的失败目录
会整体改名备份重试；含 epoch、checkpoint 或未知文件则拒绝覆盖。未实现 resume。
每种 arm 独立 hash，从 ImageNet 骨干初始化开始，不加载旧 Add 分割 checkpoint。

### 消融入口

| arm | 融合模块 | 修正 | 病灶图引导 | 辅助损失 |
| --- | --- | --- | --- | --- |
| `add` | 关 | 关 | 关 | 关 |
| `aux` | 开 | 关 | 关 | 开 |
| `residual` | 开 | 开 | 关 | 关 |
| `full` | 开 | 开 | 开 | 开 |
| `no_aux` | 开 | 开 | 开 | 关 |
| `aux_no_guidance` | 开 | 开 | 关 | 开 |

`residual` 与完整版保留相同模块参数结构，但未使用的辅助头不执行、不更新。
这样可比较额外的局部双模态修正、辅助监督及显式病灶图引导；不能只比较 full 与 Add 就把收益全归因于门控。
示例：

```bash
bash scripts/run_full_lesion_guided_ablation.sh full \
  --lesion_aux_loss_weight 0.05 --hash full-add-lesion-guided-full-w005-s2023

bash scripts/run_full_lesion_guided_ablation.sh full \
  --lesion_aux_loss_enabled false --hash full-add-lesion-guided-custom-noaux-s2023
```

同配置重跑只换 `--hash`。不要复用已含训练结果的目录。

## 独立评估

```bash
python eval_full_lesion_guided_fusion.py \
  --checkpoint checkpoints_new/MDT/full-add-lesion-guided-full-imagenet-ema-s2023/ckpt.best_full.pth.tar \
  --output_dir checkpoints_new/MDT/full-add-lesion-guided-full-imagenet-ema-s2023/eval_full
```

从 checkpoint 保存的开关、widths、尺度重建模型，strict=True 加载，不依赖原 ImageNet 文件路径。
默认使用 checkpoint 记录的 raw/EMA；可用 `--use_ema true/false` 明确覆盖。
训练结束最终评估也重建同一个模型，结果写入 `final_test_full.json`。

## 独立模块 API

```python
from models.components.lesion_guided_fusion import LesionGuidedFusion, auxiliary_lesion_loss
fusion = LesionGuidedFusion(enabled=True, correction_enabled=True, guidance_enabled=True)
fused, aux_logits = fusion.forward_with_aux(ct_features, pet_features, return_aux=True)
loss_aux, stats = auxiliary_lesion_loss(aux_logits, original_gt, enabled=True, weight=0.1)
# inference: no GT, no graph cached inside fusion
fused = fusion(ct_features, pet_features)
```

完整模型位于 `models/full_lesion_guided_fusion.py`，核心模块文件仅依赖 torch，可独立导入。
Full-only wrapper 在任何编码器执行前拒绝 Missing/mixed；模块本身不调用编码器、不访问真实 PET 图像或原型库。
后续若检索输出是对应 PET 特征，可按同形状接口另行接入；若检索输出为融合知识，不应强行当作 PET 输入。
当前不声称已经完成 Stage-2 Missing 适配。

## 验证与回退

测试：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests`。
CPU 512 预检：`python scripts/check_lesion_guided.py --device cpu --image_size 512 --batch_size 2`。
详细已执行结果见 `LESION_GUIDED_VERIFICATION.md`；文献来源见 `LESION_GUIDED_SOURCE.md`。
CUDA/AMP 和 batch16 显存必须由用户服务器预检；当前开发环境无 GPU。

成功判断：同协议固定阈值比较全局/病例 Dice、远处 FP、边界 F1/HD95、小中面积组和完全漏检切片。
目标是保住前景覆盖并减少背景误响应；若再次出现新增 FP 抵消收益、小中面积组系统退化，拒绝有效性结论。
初始化等价、能训练和有梯度都不等于真实数据有效。新颖性也未由此实现证明。

回退直接运行原 `run_full_add_baseline.py` 或本分支 `add` arm；不需撤销公共文件改动。
