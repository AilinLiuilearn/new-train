# CT 预训练 → PET/CT 双解码器干净基线

分支：`exp/ct-pretrain-dual-decoder-add-lr8e5`。
只保留两个阶段，不含 nnU-Net 专家、蒸馏、原型库、检索、收益损失、补偿、复杂融合或深度监督。

## 固定合同

| 项目 | Stage 1 | Stage 2 |
|---|---|---|
| CT 编码器 | ConvNeXtV2-Nano | 完整继承 Stage 1，继续更新 |
| CT 对齐 | 1×1 Conv + **BN** + ReLU，64/128/320/512 | 完整继承，继续更新 |
| CT 解码器 | 原有 **GN** 解码器，512/256/128/64 | 完整继承，继续更新 |
| PET 编码器 | 不创建，不读 PET | MiT-B1，本地预训练，继续更新 |
| PET 对齐 | 无 | 同结构，常规初始化，继续更新 |
| Full 融合 | 无 | 四尺度对齐后直接 CT + PET |
| Full 解码器 | 无 | 复制 Stage 1 CT 解码器初值；之后参数独立、继续更新 |
| 冻结 | 无 | **无**，不另加冻结适应阶段 |
| 选模 | 验证 CT Dice | 验证 Joint=(Full Dice+Missing Dice)/2 |

Stage 1：`CT → E_CT → A_CT → D_CT → z_CT`。

Stage 2 Missing：`CT → E_CT → A_CT → D_CT → z_M`。

Stage 2 Full：`CT/PET → 各自编码和对齐 → 相加 → D_Full → z_F`；同一 Full 样本的 CT 同时走 D_CT，得到参考预测。CT 参考与 Missing 是同一条路径。

训练 batch 精确一半 Full，一半 Missing（batch 必须为偶数）。**全部行**计算 CT 分割，**仅 Full 行**读取和编码 PET、计算融合分割。两个子损失各自取平均：

`loss = 0.5 * BCE_Dice(CT_logits_all, GT_all) + 0.5 * BCE_Dice(Full_logits_half, GT_half)`。

相邻两轮使用同一 shuffle 排列并交换 Full/Missing 状态，每个保留样本在这两轮中各为 Full/Missing 一次；下两轮重新 shuffle。沿用 drop_last，日志明确打印每轮丢弃的尾部样本数。Full/CT 的预测差异目前**不进入额外损失**。

推理只需一套模型权重：Full 使用 D_Full，Missing 使用 D_CT；Missing 不调用 PET 编码器、PET 对齐、相加或 Full 解码器。评价 Full/Missing 时一次 CT 编码/解码复用，避免重复跑两遍 CT。

## 环境

可继续使用已经安装过依赖的环境，无需为了本分支重装 PyTorch。先确认当前环境的 torch/torchvision 是匹配的 CUDA 版本：

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

如果需要新环境，**先创建并激活**，再按服务器驱动安装配套的 CUDA torch/torchvision，最后安装本分支依赖：

```bash
conda create -n petct-staged python=3.10 -y
conda activate petct-staged
# 在这里安装与你服务器匹配的 CUDA torch/torchvision；不要误装 CPU wheel。
python -m pip install -r requirements.txt
python -m pip check
```

源码检查环境为 torch 2.5.1 CPU / torchvision 0.20.1 / timm 1.0.15 / transformers 4.49.0。timm、transformers 明确固定版本以便重建同样的编码器。无远程预训练下载。

## 数据和固定参数

沿用原 PNG 目录，不转换成 nnU-Net 格式：

```
PCLT20K/
  train_original.txt
  val.txt
  test.txt
  case001/
    case001_001_CT.png
    case001_001_PET.png
    case001_001_mask.png
```

split 每行是无扩展名的 image_id；case_id 沿用原实现取第一个下划线之前的部分。检查患者级交叉、重复切片和文件存在性。Stage 1 只要求 CT/mask，Missing 训练行也不读取 PET；Stage 2 配对训练集整体仍需要每个样本的 PET，因为下一轮会交换状态。

保留原 `cipa` 空间增强（包括原 mask 插值后阈值化）、`imagenet` 三通道归一化、BCE+Dice 和 CIPA 全局计数指标；不是逐病例平均 Dice。修复两项数据完整性边界：0/1 mask 在增强前转换为 0/255，CT/PET/mask 原图尺寸必须一致，不允许静默错位。

默认：512²，batch=16，workers=4；每阶段 60 epoch；AdamW，lr=8e-5，wd=1e-4；原调度实现 warmup=3、flat_ratio=0.3、min_lr=1e-6；AMP、clip=5、accumulation=1；EMA=0.999，epoch>3 激活，decay warmup；seed=2023；无提前结束，跑完固定预算；HD95=cipa_official，阈值固定0.5，距离按像素计。

Python 入口默认 train_original/val/test，两个 shell 启动脚本**显式沿用你当前 train_original/test/test 开发协议**，启动时打印警告。该协议不构成独立最终测试；不会擅自更改你的划分。

## 已有 CT-only 权重：直接开始 Stage 2

把已有文件复制到数据盘下的 `checkpoints/ct_stage1/`；无需复制旧代码或重训 Stage 1。来源须是先前干净 CT-only 基线（同编码器、同对齐、同 GN 解码器和数据协议）。原文件可继续叫 `ckpt.best.pth.tar`：

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch --track origin/exp/ct-pretrain-dual-decoder-add-lr8e5
# 本地已存在同名分支时，使用 git switch exp/ct-pretrain-dual-decoder-add-lr8e5
# 如果 switch 提示本地改动冲突，先保留改动；不要强制覆盖。

# 已在可用环境中就继续使用；只安装缺失依赖。
python -m pip install -r requirements.txt

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 先做 GPU + 旧权重迁移检查：包含真实模型、两个解码器、优化器和 EMA。
python scripts/check_staged.py --device cuda --stage 2 --batch_size 16 \
  --stage1_checkpoint checkpoints/ct_stage1/ckpt.best.pth.tar

# 检查通过后启动。新目录不可覆盖旧实验；仅初始化失败、只留下审计文件时，会自动归档失败目录后重试。
bash scripts/train_stage2.sh \
  checkpoints/ct_stage1/ckpt.best.pth.tar \
  checkpoints_new/STAGED/dual-add-s2023
```

如果显存不足，在正式运行前固定一个可运行的**偶数** batch，并在后续所有消融保持一致；例如预检和训练均加 `--batch_size 8`。

### 严格迁移规则

- 自动读取源 checkpoint 的 `eval_weights`，加载实际被选中的 raw 或 EMA；不混用编码器和解码器来源。
- 支持旧 clean baseline 的 `model` / `model_ema`，新 Stage 1 的 best.pt，以及明确选择 raw 的纯 state_dict。统一 `module.` 前缀可去除。
- **完整匹配 CT 编码器、对齐、解码器的所有键和形状**；不是 `strict=False` 部分迁移。Full checkpoint 或其他结构直接拒绝。
- 验证源文件可用的编码器、解码器通道、输入尺寸、增强、归一化、split 名称等元数据；有 split 哈希时也严格比较。
- 旧 checkpoint 若没有完整数据元信息，只能验证结构及已有字段，无法证明原训练数据完全一致；在 `transfer.json` 明确列出未知项。
- 源文件没有 `eval_weights` 时不猜测，需显式加 `--stage1_weights raw` 或 `ema`。
- 第二阶段重新建立 optimizer、scheduler、GradScaler 和整个模型的 EMA；不继承 Stage 1 动量、调度或 EMA 计数。所有参数加入优化器。
- 输出 `transfer.json`：文件绝对路径、SHA256、epoch、raw/EMA、加载/复制张量数、已核对/未知配置。

真实用户 checkpoint 不在当前工作区，**尚未实际加载你的文件**；请先运行上面的预检，不会静默忽略不兼容。

## 没有 Stage 1 权重时

```bash
bash scripts/train_stage1.sh checkpoints_new/STAGED/ct-s2023
bash scripts/train_stage2.sh \
  checkpoints_new/STAGED/ct-s2023/best.pt \
  checkpoints_new/STAGED/dual-add-s2023
```

Stage 1 不创建任何 PET 模块，不读取 PET 文件。每轮只评价 CT，不额外评价 Full。

## 恢复和重新评价

```bash
python -u run_staged.py --resume checkpoints_new/STAGED/dual-add-s2023/last.pt
python -u run_staged.py --evaluate checkpoints_new/STAGED/dual-add-s2023/best.pt
# 仅评价 Missing：数据加载也不再要求 PET 文件
python -u run_staged.py --evaluate checkpoints_new/STAGED/dual-add-s2023/best.pt --eval_mode missing
```

恢复使用保存的原配置；不接受悄悄改 lr/batch/epochs。支持 epoch 边界恢复（中断中的 epoch 重跑），保存 raw/EMA、优化器、调度器、scaler、随机数和 DataLoader generator。保留原实验目录内的 best.pt；不允许从 best.pt 伪装成 last.pt 继续。GPU/cuDNN 和不同版本的数值一致性不保证位级复现。

best.pt 和 last.pt 分开保存，每轮记录 train.csv / train.jsonl、CT/Full/Missing 样本数、首次 batch 各模块梯度、Full/Missing/Joint、EMA 状态与跳过更新数；第一批及每50批打印 loss/耗时/ETA，评价也打印进度。config.json / environment.json / splits.json / split_summary.json 保存协议和环境。最终重载最佳检查点所记录的 raw/EMA 权重生成 final_test.json。

Stage 2 始终按同一个检查点的 Full、Missing 平均 Dice 选模，不拼接不同 epoch/权重的最佳指标。

## 审查与测试

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_staged.py -q
python scripts/check_staged.py --device cpu --image_size 64 --batch_size 2
```

测试使用真实 timm/HuggingFace 编码器，不用玩具网络替代：严格旧权重/EMA迁移、Missing 初始预测等价、两解码器参数独立、全部训练部件的梯度和更新、Missing 禁止调用 PET/融合/Full 解码器、恰好一半 PET 编码、跨轮状态互换、无 PET 文件的 Missing 数据读取、0/1 mask、实际 CIPA 增强、合成 PNG 两阶段训练/保存/恢复/EMA/重新评价。

CPU 合成流程通过只能证明实现合同，不证明真实分割收益，也不代表已验证 3090 的 batch=16。Full 每轮只看到约一半训练样本；跨方法比较需同时记录训练预算与样本曝光量。后续收益筛选/检索只在这版基础上做独立增量。

实际审查记录：

- 新增合同测试覆盖上述完整流程，并额外检查 512² 的官方 CIPA HD95 与配置边界。
- 与旧干净分支 `5aadf6e` 的 Stage 1 对照：state_dict 键/形状完全匹配，复制权重后预测逐值相同。
- 与旧数据实现对照：CT-only/配对两种模式、8组随机种子的 CIPA 增强、ImageNet归一化和0/255标签输出逐值相同。
- 实际默认双解码器模型参数35,907,538；CPU完成512²、batch=2的前向、反向、AdamW和EMA更新及Missing隔离检查。该batch包含1 Full + 1 Missing，耗时约11秒只是CPU预检，不能推算3090训练吞吐。
- 未取得你的真实Stage 1 checkpoint、未访问AutoDL真实数据、未测GPU显存；对应检查须在服务器执行。

AMP 兼容修复：训练和预检根据 API 可用性选择 `torch.amp` 或旧版 `torch.cuda.amp`，继续使用现有 myenv，无需升级 torch。若启动在创建 optimizer/EMA 前失败，且目录仅含 split_summary.json / transfer.json，会将失败目录完整改名保存，再用原输出路径重试；含 config、日志或 checkpoint 的目录仍拒绝覆盖。
