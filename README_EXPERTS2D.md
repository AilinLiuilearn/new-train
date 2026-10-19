# 2D CT/PET 双专家：先独立学习，再合作训练

分支 `exp/simmlm-2d-ct-pet-experts`。本分支新增独立入口，不修改旧异构基线入口。

## 实际实现及来源

- 两个独立的完整 2D PlainConvUNet，直接调用官方 `dynamic-network-architectures==0.3.1`，非自行重写。
- 来源：https://github.com/MIC-DKFZ/dynamic-network-architectures ，`architectures/unet.py`、`initialization/weight_init.py`。
- nnU-Net 使用该库构建网络：https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/get_network_from_plans.py 。
- SimMLM 官方源码 https://github.com/LezJ/SimMLM ，核对版本 `c9be74ba913db08013b8e573ce6cbda25da9ea1a`，参考 `models/dmome.py` 的输出层专家融合和分阶段训练。
- 这是 **nnU-Net 官方网络实现 + 自定义数据/训练流程**，没有使用完整 nnU-Net 自动规划、预处理、交叉验证及默认训练协议。也不是 SimMLM 的完整复现。
- 采用 6 层，通道 32/64/128/256/512/512，InstanceNorm2d、LeakyReLU、官方 He 初始化，输出单通道 binary logits；无深度监督。512 输入最深特征16²。
- 门控：两侧 bottleneck GAP 描述拼接 → Linear(1024,64) → ReLU → Linear(64,2) → softmax；末层零初始化，起始权重恰好各0.5。不同于原论文输入图像 CNN router。
- Full 为 logits 加权，不对 sigmoid 概率进行融合。Missing 直接运行 CT 专家，不调用 PET/门控；状态显式传入，不用全零图像推断缺失。
- 本版不加入 MoFe、TG-ECNet、原型、蒸馏、收益图或额外纠错模块。

## 数据

保持当前目录及列表方式：

```
PCLT20K/
  train_original.txt
  val.txt
  test.txt
  Case001/
    Case001_0001_CT.png
    Case001_0001_PET.png
    Case001_0001_mask.png
```

列表每行是 `Case001_0001`（不含扩展名/模态后缀）。患者ID沿用原项目：取下划线前第一段。若真实ID不满足该规则，必须先修改解析再训练。

- 检查文件存在、切片重复、患者级泄漏、原图与mask尺寸一致。
- mask 支持0/1或0/255二值标签，拒绝其他类别值。标签缩放与几何变换用最近邻。
- CT-only 入口不读取 PET；PET专家入口使用配对数据索引，但前向仅输入PET。
- 原生单通道灰度，不复制为RGB。默认逐图 zscore（全图均值/标准差），可选 unit [0,1]；**不是原异构基线的 ImageNet 归一化，也不是原始HU/SUV**。
- 共同增强：水平翻转、随机±20°旋转、0.9–1.1缩放及±10%平移，CT/PET/GT共享几何参数；不是旧cipa增强。
- 新框架的两个专家及对照必须使用这一套共同规则。旧基线成绩不能用来单独归因于框架变化。

## 安装

先激活已有带CUDA的PyTorch环境，使用 `python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'` 核对。运行需要 PyTorch>=2.3。

```bash
python -m pip install -r requirements-experts.txt
```

本项目不依赖 timm、transformers 或 ImageNet 预训练文件来运行新入口。不下载任何模型权重。

## 推荐的完整顺序（单卡串行）

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch --track origin/exp/simmlm-2d-ct-pet-experts

# 为本轮选择一个全新的目录；任一步出错立即停止。
bash scripts/run_experts2d_pipeline.sh \
  checkpoints_new/EXPERTS2D/run01
```

顺序：CT 60 epoch → PET 60 epoch → 加载两者 best.pt，合作60 epoch。
默认：512²、batch4、AdamW lr8e-5、weight_decay1e-4、warmup3 epoch、cosine min_lr1e-6、flat_ratio0.3、AMP、梯度裁剪5、seed2023。
EMA=.999，epoch>3 时同步并启动，启用decay warmup。无早停，完整执行配置的epoch预算。
batch4是双完整网络的保守起点，尚未确认3090显存；修改batch需在所有阶段保持一致。60轮是可运行的初始预算，不宣称从零nnU-Net在该预算必然收敛。

脚本沿用用户现有 `train_original.txt/test.txt/test.txt`，明确传入 allow_val_equals_test 并打印警告。正式独立评价请使用患者互斥的val.txt：

```bash
# 直接Python入口默认 val.txt，不允许val=test。
python -u run_experts.py --mode ct --output checkpoints_new/EXPERTS2D/formal_ct
```

全流程脚本可以覆盖 --val_split val.txt，但同时存在显式allow标志；患者检查仍会拒绝非同名split的病例交叉。

## 分别运行

```bash
bash scripts/train_experts2d.sh ct --output checkpoints_new/EXPERTS2D/run01/ct
bash scripts/train_experts2d.sh pet --output checkpoints_new/EXPERTS2D/run01/pet
bash scripts/train_experts2d.sh joint \
  --ct_checkpoint checkpoints_new/EXPERTS2D/run01/ct/best.pt \
  --pet_checkpoint checkpoints_new/EXPERTS2D/run01/pet/best.pt \
  --output checkpoints_new/EXPERTS2D/run01/joint
```

合作阶段每个batch CT/PET各前向一次，同时计算 `0.5*L_full+0.5*L_ct`，L为原项目BCE+Dice。两个专家和门控均更新。验证同样复用CT logits，无重复Missing前向。
联合导入严格检查模式、通道结构、图像大小、归一化、train/val列表内容。加载checkpoint记录的raw/EMA选模权重，严格加载网络；不兼容会报错，不随机回退。

## 断点恢复及复评

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u run_experts.py \
  --resume checkpoints_new/EXPERTS2D/run01/joint/last.pt

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u run_experts.py \
  --evaluate checkpoints_new/EXPERTS2D/run01/joint/best.pt
```

恢复原run配置、模型raw/EMA、优化器、调度器、AMP、随机状态和训练loader随机生成器；原目录和best.pt必须保留。恢复到上个完整epoch结束，下一个未保存epoch重新执行，不是batch内恢复。不用重新指定stage1文件。
复评采用保存的raw/EMA，允许显式 --root 覆盖数据位置及 --device cpu/cuda；不加载预训练。
多worker/CUDA跨版本不承诺逐位复现，CPU workers0的中断恢复已做逐参数一致性测试。

## 输出和选模

每个run保存 config.json、environment.json、split_summary.json、train.csv、train.jsonl、best.pt、last.pt、final_test.json。
专家分别按自身验证Dice选模；合作按 `(Full Dice + Missing Dice)/2` 选同一个checkpoint。日志含各状态Dice/IoU/HD95及漏检计数、损失、CT门控均值、LR、跳过更新、EMA次数、epoch耗时。
每轮训练与验证首batch、每50batch、末batch打印进度。正式测试仅在训练结束加载best后执行。
指标沿用原项目 `SegmentationMetricsCIPA`，含其slice级Dice/HD95空集口径，HD95单位pixel，不是3D病例级mm。
旧val=test协议不是独立测试证据。后续需要独立验证/测试、匹配额外训练预算的CT-only对照。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_experts2d.py -q
```

4项新测试通过，包括真实官方网络前后向、门控初始相加等价、CT/PET/门控梯度、Missing禁止调用PET/门控、二值PNG读取、三个阶段训练、EMA、checkpoint导入、epoch恢复逐参数一致、独立复评一致性。
默认41,301,900参数模型已在CPU完成512×512、batch1的Full前后向。真实PCLT20K数据、AutoDL GPU、实际分割收益尚未验证。

服务器先做GPU预检（可将batch改为4验证训练显存）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/check_experts2d.py --device cuda --batch_size 4
```
