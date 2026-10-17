# Full PET–CT：可拒绝的双向局部重组融合

分支 `exp/full-add-local-reassembly`；从 `exp/full-add-dscf-residual`
的 `e5693414c4c91b3d9f56e6fcee2aefb81f13d409` 派生。旧 baseline、
DSCF、数据、主损失、EMA 与解码器文件均未改动。新入口不启用 DSCF。

**当前主实验：与相加 baseline 相同的初始训练协议。只加载相同来源的两个编码器
预训练权重，重新初始化解码器、对齐器和融合模块；不加载任何分割 checkpoint，
不建立冻结参照，保护损失默认关闭。** “从头训练”指重新训练分割任务，
并不是把 baseline 使用的编码器预训练也取消。

## 待验证假设

历史诊断显示已有融合的修复和新增错误相互抵消。本实验检验：
在保留相加路径的基础上，允许双向局部内容选择和拒绝候选，
在相同初始权重、数据、主损失和训练预算下，能否比纯相加得到更稳定的净收益。
主实验只改变融合结构。相加参照与保护损失保留为显式选择的后续实验。
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

## 主实验：重新训练分割任务

默认 `baseline_checkpoint=None/空字符串`、`protect_enabled=false`、`pretrained=true`。
CT、PET 编码器加载与 baseline 相同的本地预训练文件；对齐器、解码器、融合模块
重新初始化。旧的最佳分割权重、旧 optimizer、scheduler 和 EMA 均不载入。
所有参数从第 1 epoch 联合训练；优化器、scheduler、EMA 都重新建立。

`L = main_BCE_Dice`

维持 lr=4e-5、AdamW、60 epoch、batch16、warmup3、EMA0.999（epoch3后激活）、clip5。
新融合的输出头为零，所以初始特征严格等于相加；不代表继承了相加模型学到的权重。
同 seed 构建时，新模块的初始化不改变共享参数或后续 RNG；可用同入口关闭融合进行配对对照。
这里的最佳 baseline 仅用于最终结果比较，不参与初始化、训练损失或推理。
主协议下 `train_protect_loss/weighted` 和正确区域计数应为零，参照模型不构建。

## 可选实验：相加权重微调与保护损失

仅显式传 `--baseline_checkpoint PATH` 才加载已训练的原始 Full Add 权重；
仅显式开启 `--protect_enabled true` 才建立冻结相加参照和额外前向。
这属于另一种训练协议，不能与主实验混用来证明结构有效性。

相加参照固定 eval、无梯度、FP32，两者输入相同的已增强训练 batch。
只在参照按 0.5 阈值预测正确的位置，计算
`relu(BCE(student,GT)-BCE(reference,GT))`。
正确前景、正确背景分别平均，再对非空类别平均；空集合返回图连接的零。
`L = main_BCE_Dice + protect_weight * relative_protection`。
默认可选权重0.1未经调优；错误位置不受保护，主损失可以纠正它们。
学生 train 与参照 eval/FP32 不同，首步保护损失不必为零；该项反映整个学生变化，
不能解释为只由融合导致的因果损害，也不保证 Dice 不退化。

可选参照必须声明 `full_add_baseline`/`AddFusion`，且尺寸、归一化、编码器、
解码通道和共享 state keys/shapes 一致；auto 按 checkpoint 的 eval_weights 选 raw/EMA。
记录 SHA256、epoch、所选权重、来源学习率和 EMA decay。
即使微调也重建 optimizer/scheduler/EMA，不恢复旧优化器状态。
参照在 task 外部属性，不注册到学生、不参与优化或 EMA，不写入学生 state_dict。
独立评估不需要源分割权重或编码器预训练目录。
原有 warm-start checkpoint 仍可严格加载，架构未改变。
不支持断点续训，同名非空输出目录仍会拒绝覆盖。

## Full / Missing 生命周期

- Full：CT+真实 PET，四尺度融合后交给原 GN 解码器。
- Missing：继承 baseline 的 CT-only 路径，完全绕过 PET encoder 和融合，
  PET=None 或无效 PET 内容都不得影响 Missing 输出。
- auto：只给 Full 行编码 PET/调用融合，再还原 batch 顺序。
- 本入口仅训练 Full；没有实现 Stage-2 的检索 PET 路由。
  未来接入检索 PET 需单独适配和训练，不能默认零样本迁移有效。

## 运行主实验

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch exp/full-add-local-reassembly
git pull --ff-only
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 首次运行可做 CUDA 预检；使用随机真实编码器，不读真实数据。
python scripts/check_local_reassembly.py --device cuda --image_size 512 --batch_size 2

# 主实验：不加载任何已训练的分割 checkpoint。
bash scripts/run_full_local_reassembly.sh

# 同入口纯相加对照，初始权重、损失、学习率、预算相同。
bash scripts/run_full_local_reassembly.sh \
  --local_enabled false --hash full-add-local-control-lr4e5-60ep-ema-s2023
```

主训练读取已有 `pretrained/convnextv2_nano` 和 `pretrained/mit-b1`，
无需下载新的论文权重。环境变量 BASELINE_CKPT 不会自动传入脚本。
启动日志应为 `mode=from_scratch_or_encoder_pretrained`、`baseline_sha256=None`、`protect=False`，
不应出现加载原始 Full Add checkpoint 的 `[REFERENCE]` 日志。
要重跑同配置，用新的 `--hash`；无需删除已有实验。

默认输出：`checkpoints_new/MDT/full-local-reassembly-lr4e5-60ep-ema-s2023/`。
脚本沿用 train_original.txt / test.txt / test.txt，显式开启 allow_val_equals_test，
与现有开发诊断一致；它不是独立保留测试评估。
batch2 CUDA 预检不能证明 batch16 显存或真实效果，需要时可用 `--batch_size 16` 预检。
CPU smoke 可末尾加 `--pretrained false --require_cuda false`，但不用于正式性能比较。

评估主实验：

```bash
python eval_full_local_reassembly.py \
  --checkpoint checkpoints_new/MDT/full-local-reassembly-lr4e5-60ep-ema-s2023/ckpt.best_full.pth.tar \
  --output_dir eval_local_fresh --device cuda
# --use_ema false 可评估 raw；默认按 checkpoint 记录的选择。
```

可选旧协议必须显式传入两个参数，单独命名输出：

```bash
bash scripts/run_full_local_reassembly.sh \
  --baseline_checkpoint /absolute/path/to/add/ckpt.best_full.pth.tar \
  --protect_enabled true --protect_weight 0.1 --hash local-warmstart-protect-s2023
```

若要研究该可选策略，需相同源 checkpoint/微调预算下分别比较
Add、Add+保护、Local、Local+保护；不能将微调结果直接归因于结构。

## 日志与验证标准

保留原指标/梯度列，额外包含：
- `train_loss_main`、`train_protect_loss`、`train_protect_weighted`；
- `train_protect_correct_fg/bg`：本 epoch 参照预测正确的训练像素累计次数；
- 每尺度 `local_delta_ratio_s*`：每样本修正范数 / max(Add范数,1e-6) 的均值；
- `local_null_mass_s*`、`local_center_mass_s*`：两个方向、所有 head/位置/样本的平均注意力质量。

至少同时检查全量 Dice、病例均值、小前景组、边界 F1、FP/FN、
修复与新增错误计数，配对种子确认方向。当前入口沿用常规分割指标；
分组/错误转移分析使用既有诊断流程，未声称训练 CSV 已自动包含这些分析。
先用不加载分割权重、不加保护损失的 Add/Local 配对实验判断结构有效性。
只有主实验显示正向迹象后再单独研究可选训练策略；若结构持续没有净收益，关闭结构。

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
主实验不建立冻结参照。仅可选保护模式会增加一个 baseline 的训练前向和常驻显存。

```bash
python -m pytest tests/test_local_reassembly.py -q
python scripts/check_local_reassembly.py --device cpu --image_size 512 --batch_size 2
```

本地 PyTorch 2.5.1 CPU：当前新模块测试 26 passed、1 CUDA skipped；
连同原 Full Add runner 与 EMA 回归测试，共 48 passed、1 CUDA skipped。
覆盖真实四尺度尺寸、空候选/邻域选择/越界 mask、两个方向两步梯度、
FP16/BF16 特征、严格 Add 初始化、Missing/auto 禁止调用、参照冻结、
四开关组合的训练/EMA/final test/raw及EMA严格重载、源文件移除后独立推理、
源架构/配置防错和目录保护。
新增测试确认默认无分割 checkpoint、无参照模型、总损失严格等于原 BCE+Dice，
并覆盖融合开/关两种从头训练的两步更新、EMA、final test 与严格 checkpoint 加载。
512×512、batch=2 的真实双编码器 CPU 两步预检通过：
主损失 1.749383→1.580462；冻结参照的参数和所有 buffer 完全未变。
这些是合成数据上的数值验证，不是性能提升证据。
未验证 AutoDL 的 CUDA/AMP、batch16 显存、真实预训练文件和真实数据效果。
