# 全样本 Full/Missing 双路径干净基线

分支：`exp/ct-pretrain-dual-decoder-add-lr8e5`。
当前实验直接联合训练：**不加载CT-only任务权重**，CT/PET各自加载本地通用预训练；每个样本同时接受Full和CT-only监督。保留旧CT预训练迁移入口用于对照。不含nnU-Net专家、蒸馏、原型库、检索、排序、补偿、复杂融合或深度监督。

## 当前入口：直接联合训练，不需要第一阶段文件

| 项目 | 固定处理 |
|---|---|
| CT/PET编码器 | ConvNeXtV2-Nano / MiT-B1，各自加载原有本地通用预训练，全部更新 |
| CT/PET对齐层 | 原有BN对齐，常规随机初始化，全部更新 |
| 两个GN解码器 | 相同随机初值，参数独立，全部更新；不继承CT-only分割权重 |
| 融合 | 原有四尺度CT+PET相加 |
| 每个训练样本 | 同时计算Full和CT-only预测，共享一次CT编码 |
| 损失 | `0.5 * CT_BCE_Dice + 0.5 * Full_BCE_Dice`，各自在整个batch取平均 |
| 训练配置 | 512²、batch=16、60epoch、lr=8e-5，优化器/调度/EMA/增强/划分/评价不变 |
| 选模 | 同一检查点的Joint=(Full Dice+Missing Dice)/2 |

继续使用已激活的myenv，不需安装依赖或新建环境。更新代码后，预检通过才启动：

```bash
cd /root/autodl-tmp/mkd-main/new-train &&
git switch exp/ct-pretrain-dual-decoder-add-lr8e5 &&
git pull --ff-only origin exp/ct-pretrain-dual-decoder-add-lr8e5 &&
python scripts/check_staged.py --device cuda --stage 2 --batch_size 16 \
  --joint_from_pretrained &&
bash scripts/train_joint.sh checkpoints_new/STAGED/dual-add-joint-nocttask-s2023
```

默认通用预训练目录仍为：

- `/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano`
- `/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1`

如果位置不同，预检和训练都传入同样的 `--ct_pretrained_path` / `--pet_pretrained_path`。正常运行不使用 `--no_pretrained`；它仅供合成数据随机权重检查。文件不存在或编码器权重不完整会报错，不静默退回随机初始化。

启动会打印两侧 `[PRETRAIN]` 加载报告、`[INIT] ... backbone_pretrained ... no CT-only task weights loaded`，以及 `[PAIRED] each batch: Full=16 CT-only=16`。`initialization.json`和checkpoint的source记录两侧实际权重文件SHA256及随机对齐/解码器初始化，不生成CT任务迁移的transfer.json。

预检会打印每次尝试的loss和AMP缩放值。若反向出现实际Inf/NaN梯度，允许GradScaler跳过该次更新、降低scale并重试（最多8次）；只有一次真实成功的参数更新及EMA更新才算通过。最终JSON记录attempts、skipped_updates和amp_scale。权重/前向异常、FP32梯度异常、仅总范数溢出或持续AMP溢出仍然报错，严格裁剪检查不会被关闭。该修复仅影响预检，不改正式训练配置。

若新版预检仍失败，可只做FP32诊断（不会自动改正式训练精度）：

```bash
python scripts/check_staged.py --device cuda --stage 2 --batch_size 16 \
  --joint_from_pretrained --no-mixed_precision
```

`--joint_from_pretrained`与`--stage1_checkpoint`互斥。内部 `--stage 2` 仅选择双路径结构；此入口只执行一次60epoch联合训练，不先执行CT-only训练。使用新输出目录，不能在旧训练目录上覆盖。

需要继续或评价时，仍使用原入口，不重新读取通用预训练文件：

```bash
python -u run_staged.py --resume checkpoints_new/STAGED/dual-add-joint-nocttask-s2023/last.pt
python -u run_staged.py --evaluate checkpoints_new/STAGED/dual-add-joint-nocttask-s2023/best.pt
python -u run_staged.py --evaluate checkpoints_new/STAGED/dual-add-joint-nocttask-s2023/best.pt --eval_mode missing
```

比较本版和CT任务初始化版时，保持联合训练预算和配置一致，并分别记录额外CT预训练预算。取消预训练不保证Full/Missing提高，结果须由真实训练验证。

## 可选对照：原CT预训练 → 双路径训练

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

第二阶段每个配对样本都计算 Full 和 CT-only 两个预测，**不再按 50% 分配样本**。一个 batch 的全部行都读取 PET 并参与融合分割，同时这些样本的 CT 特征独立进入 CT 解码器。CT 编码和对齐只计算一次，两个解码器各计算一次；不是将 CT 编码器完整运行两遍。两个子损失各自取平均：

`loss = 0.5 * BCE_Dice(CT_logits_all, GT_all) + 0.5 * BCE_Dice(Full_logits_all, GT_all)`。

沿用原基线 DataLoader 的逐轮 shuffle 和 drop_last，日志明确打印每轮丢弃的尾部样本数。Full/CT 使用同一次空间增强后的同样本、同标签监督。Full/CT 的预测差异目前**不进入额外损失**，不加入排序、蒸馏或补偿；PET 也不新增单模态任务预训练。全部参数继续更新。

例如 batch=16 时，CT 编码器输入16、PET编码器输入16、两个解码器各输入16。每轮 `ct_samples=full_samples=missing_samples=实际保留样本数`，后两项是同一批样本的两条路径计数，不能相加当成独立训练样本数。若训练集为17416条、batch=16，三项均为17408，尾部8条丢弃。

推理只需一套模型权重：Full 使用 D_Full，Missing 使用 D_CT；Missing 不调用 PET 编码器、PET 对齐、相加或 Full 解码器。评价 Full/Missing 时一次 CT 编码/解码复用，避免重复跑两遍 CT。

## 环境

继续使用你当前的 **myenv**，本次修改没有新增依赖，不需要创建环境、安装 requirements 或升级 PyTorch。如需查看当前环境：

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

本次源码检查环境为Python 3.12 / torch 2.5.1 CPU / torchvision 0.20.1 / timm 1.0.15 / transformers 4.49.0。服务器继续使用你现有myenv。timm、transformers明确固定版本以便重建同样的编码器。无远程预训练下载。

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

split 每行是无扩展名的 image_id；case_id 沿用原实现取第一个下划线之前的部分。检查患者级交叉、重复切片和文件存在性。Stage 1 只要求 CT/mask；Stage 2 每个训练样本都需要配对 PET。Missing-only 推理和评价不要求 PET 文件，也不会调用 PET 模块。

保留原 `cipa` 空间增强（包括原 mask 插值后阈值化）、`imagenet` 三通道归一化、BCE+Dice 和 CIPA 全局计数指标；不是逐病例平均 Dice。修复两项数据完整性边界：0/1 mask 在增强前转换为 0/255，CT/PET/mask 原图尺寸必须一致，不允许静默错位。

默认：512²，batch=16，workers=4；每阶段 60 epoch；AdamW，lr=8e-5，wd=1e-4；原调度实现 warmup=3、flat_ratio=0.3、min_lr=1e-6；AMP、clip=5、accumulation=1；EMA=0.999，epoch>3 激活，decay warmup；seed=2023；无提前结束，跑完固定预算；HD95=cipa_official，阈值固定0.5，距离按像素计。

Python入口默认train_original/val/test，三个shell启动脚本显式沿用你当前train_original/test/test开发协议，启动时打印警告。该协议不构成独立最终测试；不会擅自更改你的划分。

## 已有 CT-only 权重：直接开始 Stage 2

把已有文件复制到数据盘下的 `checkpoints/ct_stage1/`；无需复制旧代码或重训 Stage 1。来源须是先前干净 CT-only 基线（同编码器、同对齐、同 GN 解码器和数据协议）。原文件可继续叫 `ckpt.best.pth.tar`：

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch exp/ct-pretrain-dual-decoder-add-lr8e5
git pull --ff-only origin exp/ct-pretrain-dual-decoder-add-lr8e5
# 首次获取、本地尚无该分支时，使用 git switch --track origin/exp/ct-pretrain-dual-decoder-add-lr8e5
# 如果 switch 提示本地改动冲突，先保留改动；不要强制覆盖。

# 继续使用已激活的 myenv，无需安装新依赖。
CT_CKPT="/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/ct_only-hetero-lr8e5-60ep-ema-s2023-20261019-160946/ckpt.best.pth.tar"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 先做 GPU + 旧权重迁移检查：包含真实模型、两个解码器、优化器和 EMA。
python scripts/check_staged.py --device cuda --stage 2 --batch_size 16 \
  --stage1_checkpoint "$CT_CKPT"

# 检查通过后启动。新目录不可覆盖旧实验；仅初始化失败、只留下审计文件时，会自动归档失败目录后重试。
bash scripts/train_stage2.sh \
  "$CT_CKPT" \
  checkpoints_new/STAGED/dual-add-paired-s2023
```

本次 PET/融合解码器由半批改为全批，显存和计算量可能增加，不能沿用旧预检推断 batch=16 一定可用。先运行上面的新版预检。若显存不足，在正式运行前固定可运行的 batch，并在后续所有消融保持一致；例如预检和训练均加 `--batch_size 8`。不再要求 batch 为偶数。

启动应打印 `protocol=paired_full_missing` 和 `[PAIRED] each batch: Full=16 CT-only=16`。使用新输出目录，不覆盖旧 `dual-add-baseline-s2023` 实验。预检不是训练，正式启动仍从你的 CT-only 文件初始化。

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

## 可选：重新训练CT任务初始化对照

当前直接联合训练使用开头的train_joint.sh，不需要下面这一步。只有要重建旧CT任务初始化对照时才运行：

```bash
bash scripts/train_stage1.sh checkpoints_new/STAGED/ct-s2023
bash scripts/train_stage2.sh \
  checkpoints_new/STAGED/ct-s2023/best.pt \
  checkpoints_new/STAGED/dual-add-paired-s2023
```

Stage 1 不创建任何 PET 模块，不读取 PET 文件。每轮只评价 CT，不额外评价 Full。

## 恢复和重新评价

```bash
python -u run_staged.py --resume checkpoints_new/STAGED/dual-add-paired-s2023/last.pt
python -u run_staged.py --evaluate checkpoints_new/STAGED/dual-add-paired-s2023/best.pt
# 仅评价 Missing：数据加载也不再要求 PET 文件
python -u run_staged.py --evaluate checkpoints_new/STAGED/dual-add-paired-s2023/best.pt --eval_mode missing
```

恢复使用保存的原配置；不接受悄悄改 lr/batch/epochs。支持 epoch 边界恢复（中断中的 epoch 重跑），保存 raw/EMA、优化器、调度器、scaler、随机数和 DataLoader generator。保留原实验目录内的 best.pt；不允许从 best.pt 伪装成 last.pt 继续。GPU/cuDNN 和不同版本的数值一致性不保证位级复现。

新版checkpoint格式为`ct_pretrain_dual_decoder_v2_paired`。初始化方式保存在config/source中，恢复时沿用原配置，不能通过resume更换初始化。未包含新初始化字段的旧checkpoint默认按原CT任务初始化方式兼容。旧v1的CT-only文件仍可严格迁移，旧Stage 1可恢复，旧Stage 2可评价；拒绝将旧50%路由Stage 2直接恢复成配对训练。若需继续原50%路由运行，请在另一干净checkout使用原提交`c0ba6fb`。

best.pt 和 last.pt 分开保存，每轮记录 train.csv / train.jsonl、CT/Full/Missing 样本数、首次 batch 各模块梯度、Full/Missing/Joint、EMA 状态与跳过更新数；第一批及每50批打印 loss/耗时/ETA，评价也打印进度。config.json / environment.json / splits.json / split_summary.json 保存协议和环境。最终重载最佳检查点所记录的 raw/EMA 权重生成 final_test.json。

Stage 2 始终按同一个检查点的 Full、Missing 平均 Dice 选模，不拼接不同 epoch/权重的最佳指标。

## 审查与测试

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_staged.py tests/test_preflight_amp.py -q
python scripts/check_staged.py --device cpu --image_size 64 --batch_size 2
```

测试使用真实 timm/HuggingFace 编码器，不用玩具网络替代：严格旧权重/EMA迁移、Missing 初始预测等价、两解码器参数独立、全部训练部件的梯度和更新、CT/PET各一次全批编码、两个输出按同序标签监督、PET变化不改变CT-only输出、Missing禁止调用PET/融合/Full解码器、全配对加载及奇数batch、无PET文件的Missing评价、0/1 mask、实际CIPA增强、合成PNG两阶段训练/保存/恢复/EMA/重新评价、旧协议断点保护。

CPU 合成流程通过只能证明实现合同，不证明真实分割收益，也不代表已验证3090的batch=16。新版Full和CT-only每轮各看到全部保留样本；与旧50%方案比较需同时记录训练预算、样本曝光量和计算量。后续排序、融合、补偿分别做独立增量。

实际审查记录：

- AMP预检修复后10项测试全部通过（CPU约35.2秒）。使用真实CPU GradScaler注入Inf，验证跳过错误更新、缩放回退后仅一次optimizer/EMA更新；持续溢出、FP32异常、非有限权重及有限元素的总范数溢出均拒绝通过。这不等于已在用户GPU复现或排除了其全部数值问题。
- 本次7项合同测试全部通过（CPU复测约30.3秒）。
- 直接联合训练新增合同测试：本地测试权重完整加载到CT/PET编码器，对齐/解码器保持随机初值，两解码器参数独立；无需任何CT-only文件即可完成训练；删除通用预训练文件后仍可恢复且raw/EMA逐值一致，并可重新评价。
- 直接联合入口的CPU预检已通过512²、batch=2的完整前向/反向、AdamW、EMA及Missing隔离；此预检显式使用随机权重，通用预训练加载另由上面的本地权重测试覆盖。未测AutoDL真实权重、真实数据性能或GPU显存。
- 上一版6项合同测试通过（CPU约32.7秒），包含真实两阶段PNG流程、新协议恢复逐值一致、旧Stage-1日志兼容与旧Stage-2恢复拒绝；本次仍全部通过。
- 新增合同测试覆盖上述完整流程，并额外检查 512² 的官方 CIPA HD95 与配置边界。
- 与旧干净分支 `5aadf6e` 的 Stage 1 对照：state_dict 键/形状完全匹配，复制权重后预测逐值相同。
- 与旧数据实现对照：CT-only/配对两种模式、8组随机种子的 CIPA 增强、ImageNet归一化和0/255标签输出逐值相同。
- 实际默认双解码器模型参数35,907,538。旧版CPU预检（1 Full + 1 Missing）的耗时不适用于新版；请使用新版预检报告的全批Full/CT-only覆盖和显存。
- 新版CPU预检已完成512²、batch=2的全批Full/CT-only前向、反向、AdamW和EMA更新；CT/PET各编码一次且均为2个样本，六个训练模块均有非零梯度，Missing隔离检查通过。CPU耗时约11.4秒，不代表GPU吞吐或显存。
- 未取得你的真实Stage 1 checkpoint、未访问AutoDL真实数据、未测GPU显存；对应检查须在服务器执行。

AMP兼容修复：训练和预检根据API可用性选择torch.amp或旧版torch.cuda.amp，继续使用现有myenv，无需升级torch。若启动在创建optimizer/EMA前失败，且目录仅含split_summary.json / transfer.json / initialization.json，会将失败目录完整改名保存，再用原输出路径重试；含config、日志或checkpoint的目录仍拒绝覆盖。
