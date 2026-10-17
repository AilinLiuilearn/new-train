# Full PET–CT：可拒绝的双向局部重组融合

分支 `exp/full-add-local-reassembly`；从 `exp/full-add-dscf-residual`
的 `e5693414c4c91b3d9f56e6fcee2aefb81f13d409` 派生。旧 baseline、
DSCF、数据、主损失、EMA 与解码器文件均未改动。新入口不启用 DSCF。

## 待验证假设

历史诊断显示已有融合的修复和新增错误相互抵消。本实验检验：
在保留相加路径的基础上，允许双向局部内容选择和拒绝候选，
再用相加参照约束新增退化，能否得到更稳定的净收益。
不预设空间错位是主要病因，不承诺提升，也不将约 0.78 视为架构上限。

## 精确实现

每尺度输入已对齐的 `C,P: B×D×H×W`；D=[64,128,320,512]。
四尺度参数独立，低维 r=[16,16,40,64]，两个注意力头。

1. 独立 1×1 压缩得到 `c,p`。Q/K 使用逐位置 LayerNorm 后的表达，
   value 使用压缩后的原表达。不是对整张图池化。
2. CT 生成 query，PET 邻域生成 key/value；反方向独立执行。
3. 每个 query 的候选为其 3×3 邻域（含中心）和一个空候选。
   空候选的 key 可学习、value 固定为零。越界邻居在 softmax 前屏蔽。
   不是所有 query 共享一张 8×8 采样网格，也不执行全局 N×N 注意力。
4. `delta_P=U_P(transport(P|C)-p)`；`delta_C=U_C(transport(C|P)-c)`。
5. `F=(C+P)+(delta_C+delta_P)`。不额外乘 2、不做末端 ReLU。
   仅 U_C/U_P 的权重置零，无输出偏置，其余正常初始化。
   初始严格等于 Add；第一个反传输出头有梯度，内部梯度为零，
   输出头更新后内部 Q/K、投影与空候选 key 才获得非零梯度。

核心计算固定 FP32，输出残差转换为 Add 的 dtype；支持 CUDA AMP、
FP16/BF16 特征，不支持将整个模块 `.half()`/`.double()`。
不依赖 Triton、einops 或自定义 CUDA 扩展。
采用移位视图，避免展开完整九邻域 value；实际显存仍需服务器预检。

空候选和负残差只提供抑制候选内容的能力，不保证消除假阳性。
注意力相似度不等于医学可靠性，空候选权重也不是经过校准的置信度。

## 相加参照与训练损失

默认从一个原始 Full Add checkpoint 的所选 raw/EMA 权重初始化学生，
并建立同权重的冻结 Add 参照。两者输入同一个已经增强的训练 batch。
参照固定 eval、无梯度、FP32，不跟随学生更新 BN、权重或 EMA。

只在参照按 0.5 阈值预测正确的位置，计算
`relu(BCE(student,GT)-BCE(reference,GT))`。
正确前景、正确背景分别平均，然后对非空类别平均；空集合返回图连接的零。
错误位置不受保护，主 BCE+Dice 仍可以纠正它们。

`L = main_BCE_Dice + protect_weight * relative_protection`

默认 `protect_weight=0.1` 是未验证的起点，不是实验最优值。
`--protect_enabled false` 完全取消参照模型的构建和额外前向，保留主损失。
没有四尺度辅助分割头、边界标签头、原型库或重建损失。
该约束不会保证每个像素或最终 Dice 都不退化。
学生处于 train、参照处于 eval，且学生外部可能启用 AMP，
因此即使融合头初始为零，首步保护损失也不必为零。
它约束整个学生的预测变化，并非只衡量融合分支的因果损害；
必须用上述相同微调预算和损失开关的对照分离这些影响。

所有学生参数共同用 AdamW、lr=4e-5 更新；warm-start 时也重建
optimizer/scheduler/EMA，从新 epoch 1 开始，不恢复旧优化器状态。
保留当前 60 epoch、warmup=3、EMA=0.999（epoch3后激活）、clip=5。
这属于从 Add 出发的受控微调；不能直接当作从头训练对照。

参照 checkpoint 必须声明原始 `full_add_baseline`/`AddFusion`，
且图像大小、归一化、编码器、解码通道一致；共享 state keys/shapes 严格一致。
按 `--baseline_weights auto|raw|ema` 选择；auto 使用该 checkpoint 的 eval_weights。
记录文件 SHA256、epoch、所选权重、来源学习率和 EMA decay。
默认脚本沿用历史 `full-add-baseline` 路径：**如果当前 lr4e-5 baseline
保存在另一个目录，必须传入其实际路径，不要把旧 lr8e-5 的文件误作新 baseline。**

参照放在训练 task 外部属性，不注册到学生，不参与 optimizer/EMA，
不写入模型 state_dict。训练 checkpoint 保存学生 raw/EMA 与配置。
独立评估不重新打开参照文件，也不读取编码器预训练目录。
配置包含架构和损失 schema；不支持断点续训，同名非空输出目录会拒绝覆盖。

## Full / Missing 生命周期

- Full：CT+真实 PET，四尺度融合后交给原 GN 解码器。
- Missing：继承 baseline 的 CT-only 路径，完全绕过 PET encoder 和融合，
  PET=None 或无效 PET 内容都不得影响 Missing 输出。
- auto：只给 Full 行编码 PET/调用融合，再还原 batch 顺序。
- 本入口仅训练 Full；没有实现 Stage-2 的检索 PET 路由。
  未来接入检索 PET 需单独适配和训练，不能默认零样本迁移有效。

## 运行

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch exp/full-add-local-reassembly
git pull --ff-only
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 不需要下载新论文权重；使用现有 PyTorch/timm/transformers 环境。
# 可先检查结构、梯度与冻结参照，随机权重，不读真实数据：
python scripts/check_local_reassembly.py --device cuda --image_size 512 --batch_size 2

# 正式运行；默认源文件需确实是你希望比较的相加 baseline：
bash scripts/run_full_local_reassembly.sh

# 源文件在其他位置时，在命令末尾覆盖默认路径：
bash scripts/run_full_local_reassembly.sh \
  --baseline_checkpoint /absolute/path/to/your/current_add/ckpt.best_full.pth.tar
```

脚本使用历史数据拆分 train_original.txt / test.txt / test.txt，并显式开启
allow_val_equals_test，与现有开发诊断一致；它不是独立保留测试评估。
输出：`checkpoints_new/MDT/full-add-local-reassembly-protect-lr4e5-s2023/`。
预检成功只能验证代码与该 batch 的 CUDA 支持，不能证明 batch16 显存或训练效果。
需要时可以用 `--batch_size 16` 做同配置显存预检。

四组相同 warm-start 和训练预算的消融（均可末尾覆盖源路径）：

```bash
bash scripts/run_full_local_reassembly.sh --local_enabled false --protect_enabled false --hash add-continue-lr4e5-s2023
bash scripts/run_full_local_reassembly.sh --local_enabled false --protect_enabled true --hash add-continue-protect-lr4e5-s2023
bash scripts/run_full_local_reassembly.sh --local_enabled true --protect_enabled false --hash local-no-protect-lr4e5-s2023
bash scripts/run_full_local_reassembly.sh --local_enabled true --protect_enabled true --hash local-protect-lr4e5-s2023
```

从头训练结构消融必须同时取消参照路径与损失：
`bash scripts/run_full_local_reassembly.sh --baseline_checkpoint '' --protect_enabled false --hash local-from-scratch-s2023`。
此时 `pretrained=true` 仍读取现有两个编码器预训练目录；CPU smoke 再加 `--pretrained false --require_cuda false`。

评估：

```bash
python eval_full_local_reassembly.py \
  --checkpoint checkpoints_new/MDT/full-add-local-reassembly-protect-lr4e5-s2023/ckpt.best_full.pth.tar \
  --output_dir eval_local_protect --device cuda
# --use_ema false 可单独评估 raw；默认使用 checkpoint 记录的选择。
```

## 日志与验证标准

保留原指标/梯度列，额外包含：
- `train_loss_main`、`train_protect_loss`、`train_protect_weighted`；
- `train_protect_correct_fg/bg`：本 epoch 参照预测正确的训练像素累计次数；
- 每尺度 `local_delta_ratio_s*`：每样本修正范数 / max(Add范数,1e-6) 的均值；
- `local_null_mass_s*`、`local_center_mass_s*`：两个方向、所有 head/位置/样本的平均注意力质量。

至少同时检查全量 Dice、病例均值、小前景组、边界 F1、FP/FN、
修复与新增错误计数，配对种子确认方向。当前入口沿用常规分割指标；
分组/错误转移分析使用既有诊断流程，未声称训练 CSV 已自动包含这些分析。
若收益只来自保护损失，应归于训练策略；若结构持续没有净收益，关闭结构。

## 来源边界

这是我们根据已批准设计编写的新模块，不是直接复制某一论文模块。

- GeminiFusion，ICML 2024：借鉴逐位置、双向跨模态交互的思路。
  论文 https://proceedings.mlr.press/v235/jia24b.html
  官方实现 https://github.com/JiaDingCN/GeminiFusion/blob/main/models/mix_transformer.py
  已检查 Attention 中的两个 cross-attention、relation_judger 与 noise embedding。
  原方法嵌入 MiT/Swin；本实现放在异构编码器对齐之后，改用局部邻域、
  固定零值空候选与中心差分，不声称复现原版或复制其源码。
- PNDC，TMI 2025，DOI 10.1109/TMI.2025.3526818：借鉴区分正负影响的训练思想。
  官方实现 https://github.com/yansheng-qiu/PNDC/blob/main/train_PNDC.py
  已检查 findPNmap 熵划分及正负区域处理。
  本实现不移植其四模态 BraTS 熵筛选/Reverse Audit/Forward Checksum，
  改用训练 GT 与冻结相加预测定义正确区域。

单独的注意力、残差、空候选或该损失均不宣称原创；研究新意与有效性待查重和消融。

## 独立调用与本地验证

```python
from models.components.local_reassembly import LocalReassemblyFusion

fusion = LocalReassemblyFusion(enabled=True)
# 两个列表均为 S1..S4 的 BCHW 特征，通道 64/128/320/512。
fused_features = fusion(ct_features, pet_features)
diagnostics = fusion.pop_diagnostics()  # 仅训练模式积累，读取后清空。
```

新融合参数 220,208；默认完整学生模型参数 32,662,849。
冻结参照为另一个 baseline 模型：增加训练前向与常驻模型显存，推理时移除。

```bash
python -m pytest tests/test_local_reassembly.py -q
python scripts/check_local_reassembly.py --device cpu --image_size 512 --batch_size 2
```

本地 PyTorch 2.5.1 CPU：新测试 24 passed、1 CUDA skipped。
覆盖真实四尺度尺寸、空候选/邻域选择/越界 mask、两个方向两步梯度、
FP16/BF16 特征、严格 Add 初始化、Missing/auto 禁止调用、参照冻结、
四开关组合的训练/EMA/final test/raw及EMA严格重载、源文件移除后独立推理、
源架构/配置防错和目录保护。
回归套件通过 192 项、跳过 7 项 CUDA 测试；旧的
`test_ct_only_dataset_has_no_pet` 因本地 OpenCV 原生依赖问题未运行（1 deselected）。
随后补充编码器预训练文件隔离检查，新模块的 24 项测试再次通过。
512×512、batch=2 的真实双编码器 CPU 两步预检通过：
主损失 1.749383→1.580462；冻结参照的参数和所有 buffer 完全未变。
这些是合成数据上的数值验证，不是性能提升证据。
未验证 AutoDL 的 CUDA/AMP、batch16 显存、真实预训练文件和真实数据效果。
