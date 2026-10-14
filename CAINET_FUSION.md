# Full CT–PET：CACR / GCM 四组消融

实验分支：`exp/full-cainet-cacr-gcm`。直接基于干净的
`exp/aligned-add-baseline` / `663f9786122365d8b2774e4497df92bc06b715d5`，
不包含之前的 AMDANet 实验。

## 源码与实验边界

- 论文：Lv, Liu, Li, *Context-Aware Interaction Network for RGB-T Semantic
  Segmentation*, IEEE TMM 26 (2024), 6348–6360，DOI `10.1109/TMM.2023.3349072`。
- [作者仓库](https://github.com/YingLv1106/CAINet)。固定提交：
  `4fff413041f1776ce1e588eb4763fe46fca2677a`。
- [作者源文件](https://github.com/YingLv1106/CAINet/blob/4fff413041f1776ce1e588eb4763fe46fca2677a/toolbox/models/cainet.py)。
  本地 `models/components/cainet_source.py` 原样复制 `Interv2`（CACR）和
  `CRRM`（GCM）两个类。包括原作者的 reshape、矩阵乘法、残差、softmax 和
  固定 128 通道的 GCM 输出；没有重写成近似注意力块。
- 源码对应与类体校验由 `tests/test_cainet_fusion.py` 记录。保留作者署名，
  本实验不为上游源码另行声明许可证。

这是**作者关系模块在当前 PET–CT baseline 上的移植实验**。原网络使用双
MobileNetV2、相同空间分辨率的三个高层特征，并配有 ARLM 辅助监督和 DA 解码。
当前网络是异构编码器，三个高层特征的大小和通道各不相同，已有自己的解码器。
因此外围需要以下明确的接口适配；不能称为整篇 CAINet 的原样复现。

## 接入位置与张量约定

CT ConvNeXtV2-Nano、PET MiT-B1 独立编码 → 原有各自 Conv1×1+BN+ReLU
对齐层 → `CAINetFusion` → 原有 GroupNorm U-Net 解码器。

输入为两个长度为 4 的 BCHW 特征列表，相同尺度的 CT/PET 形状一致，
所有特征 batch、device、浮点 dtype 一致且数值有限。输出仍为 4 个原形状特征。

| 尺度 | 每模态通道 | 512 输入对应尺寸 | CACR 开启时 |
|---|---:|---:|---|
| S1 | 64 | 128×128 | Interv2(128) |
| S2 | 128 | 64×64 | Interv2(256) |
| S3 | 320 | 32×32 | Interv2(640) |
| S4 | 512 | 16×16 | Interv2(1024) |

四个尺度均使用同样的融合逻辑：

```python
joint = cat([ct_s, pet_s], dim=1)
if cacr_enabled:
    joint = Interv2(joint)       # 原作者的 x+y 残差已在类内
ct_updated, pet_updated = joint.chunk(2, dim=1)
fused_s = ct_updated + pet_updated
```

没有在输出上再叠加一次 CT+PET，避免重复计算作者已有的输入残差。

GCM 开启时，取 S2/S3/S4 的 `joint`（CACR 关闭时是原始 CT/PET 拼接），
统一到 S3 网格，再按 S4、S3、S2 顺序拼接为 1920 通道，送入原始 `CRRM(1920)`。
保留作者整网 `cat([fuse5, fuse4, fuse3])` 的高到低层顺序；
相关投影均从头学习，不加载 CAINet 预训练参数。
固定 128 通道的 GCM 输出经过外部 1×1 Conv（128→512），缩放到 S4 后残差相加。
S1–S3 的融合输出不会被 GCM 直接改写，后续仍由原解码器传播上下文。

外部 1×1 Conv **零初始化**，避免一开始向 baseline 注入随机 GCM 修正。
所以 GCM 开启配置在初始时刻与对应的 GCM 关闭配置输出严格一致；第一步先学习
该投影，GCM 原始类参数第一步梯度为零，投影更新后即可收到非零梯度。
CACR 的初始化完全保留作者默认行为，开启 CACR 的初始输出并不等同 Add。

关系模块的矩阵计算在 AMP 环境内显式使用 FP32，再转回输入特征 dtype；
源类体不改公式、不加新的归一化、门控或除数。其余网络继续使用原 AMP 策略。

## 开关、参数量和训练协议

两个布尔开关默认都为 true；关闭的模块不实例化、不加入优化器。

| 脚本参数 | `--cacr_enabled` | `--gcm_enabled` | 融合新增参数 | 总参数 |
|---|---|---|---:|---:|
| add | false | false | 0 | 32,442,641 |
| cacr | true | false | 2,406,400 | 34,849,041 |
| gcm | false | true | 835,584 | 33,278,225 |
| both | true | true | 3,241,984 | 35,684,625 |

GCM 参数量包含外部投影。两个开关均关闭时直接调用原 `AddFusion`；相同种子下，
整个模型的公共权重、state_dict 键和预测与原 baseline 严格一致。
所有配置先完成原 baseline 初始化，再实例化融合模块。

当前版本默认 CACR 覆盖 S1–S4；S1 的独立 `Interv2(128)` 新增 25,600 参数。
这把作者主要用于高层的关系模块扩展到当前任务的浅层，类体保持原样，但扩展效果
需要消融实验验证。GCM 的输入范围仍为 S2–S4，不把浅层细节强行并入全局上下文。
新训练记录 `cacr_stages=[1,2,3,4]`，脚本 hash 加入 `s1s4`，区分此前三尺度版本。
独立评估根据检查点保存的 `cacr_stages` 恢复；旧 `[2,3,4]` 检查点仍按原三尺度
结构严格加载，不会被静默转换成四尺度。新四尺度实验应重新训练，不能直接当作旧模型续跑。

训练仍是 Full-only，所有样本必须有真实 CT 和 PET；Missing 路径会在编码前报错。
复用原 `run_full_add_baseline.py` 的训练/验证/最终测试循环，仅增加可注入的
配置、模型构造器、类型检查和结果标记，默认参数仍运行原 Add baseline。
编码器、对齐层、解码器、BCE+Dice、AdamW、调度、数据、指标、EMA 均沿用 baseline。
不引入 CAINet 辅助损失、DA、原型、文本、缺失模拟或历史 AMDANet 模块。

## 完整训练命令

```bash
git fetch origin
git switch --track origin/exp/full-cainet-cacr-gcm

# 四组可分别运行；其余设置完全相同
bash scripts/run_full_cainet_ablation.sh add
bash scripts/run_full_cainet_ablation.sh cacr
bash scripts/run_full_cainet_ablation.sh gcm
bash scripts/run_full_cainet_ablation.sh both
```

已经有本地分支时，使用 `git switch exp/full-cainet-cacr-gcm` 后
`git pull --ff-only origin exp/full-cainet-cacr-gcm`。

脚本默认：真实数据路径 `/root/autodl-tmp/data/PCLT20K`、512、batch 16、
ImageNet 归一化、CIPA 增强/HD95、60 epochs、LR 8e-5、weight decay 1e-4、
warmup 3、min LR 1e-6、flat ratio 0.3、AMP、EMA .999（start epoch 3）、
grad clip 5、patience 10、seed 2023。预训练权重位置在脚本内显式列出。

脚本沿用已有比较命令的 `val=test.txt` 和 `allow_val_equals_test=true`。
这会在测试集上选择检查点，不是独立留出的测试结果；正式评估需要四组一致地
传入互不重叠的验证集。数据与权重路径不同可追加覆盖，例如：

```bash
bash scripts/run_full_cainet_ablation.sh both \
  --root /your/PCLT20K \
  --val_split_file val_disjoint.txt --allow_val_equals_test false \
  --ct_pretrained_path /your/convnextv2_nano \
  --pet_pretrained_path /your/mit-b1 \
  --hash full-cainet-s1s4-both-disjoint-s2023
```

也可直接运行 `python run_full_cainet_fusion.py` 并传入上述数据、权重、训练参数，
用 `--cacr_enabled true/false --gcm_enabled true/false` 控制模块。
每次重跑请使用新的 `--hash`；本入口拒绝覆盖已有 `config_args.json` 的目录，
没有训练续跑功能。修改 batch 等设置时四组应一致修改，旧实验只在协议完全相同时可直接比较。

## 结果与评估

结果在 `checkpoints_new/MDT/<hash>/`（可用 `--checkpoint_root` 更改）：

- `config_args.json`：开关、源/基线提交、接口位置、初始化方式及训练参数。
- `train_log.csv`：每轮 Full Dice、IoU、HD95 等。
- `ckpt.best_full.pth.tar`：验证 Full Dice 最佳；`ckpt.last.pth.tar`：最后一轮。
- `final_test_full.json`：从 best_full 严格重建、使用记录的 raw/EMA 权重得到的指标与开关。

```bash
python eval_full_cainet_fusion.py \
  --checkpoint checkpoints_new/MDT/full-cainet-s1s4-both-imagenet-ema-s2023/ckpt.best_full.pth.tar \
  --output_dir eval-cainet-s1s4-both-s2023
```

评估从检查点读取两个开关及 CACR 尺度并严格加载权重，不允许静默恢复成其他组合。
默认沿用训练归一化、划分、HD95 模式和检查点选优时的 raw/EMA 选择；不再需要编码器
预训练文件。可显式覆盖 `--root`、`--batch_size`、`--num_workers`、`--hd95_mode`、
`--use_ema`。输出目录不能已有同名评估结果。此入口不接受 Add 原入口或 Missing 检查点。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/test_cainet_fusion.py -q -p no:cacheprovider

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests -q -p no:cacheprovider

# 无数据集、无预训练文件的真实编码器 512 前向与反向
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/smoke_full_cainet.py --device cpu --image_size 512 --batch_size 1

# 训练 GPU 上验证实际 AMP / batch-16 显存
python tools/smoke_full_cainet.py --device cuda --amp --image_size 512 --batch_size 16
```

全套测试的 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` 只用于兼容继承的 EMA 测试：该测试
用旧式 `torch.load` 读取自己刚生成的可信检查点，PyTorch 2.6+ 的默认加载策略
会拒绝其中的 NumPy RNG 状态。新模块测试不需要这个环境变量。生产加载策略不改动，
训练脚本也不设置它。

专项验证覆盖：原始类体校验、四组开关前后向、模块关闭后不存在、输出维度、
GCM 零初始化等价与第二步非零梯度、原始模块组合计算、与 Add 的全模型严格等价、
真实编码器/优化器接入、检查点开关恢复、Full-only 约束，以及四组各自完整的
合成训练→选优→保存→重建→测试流程。合成测试分数不代表真实数据集性能。

本次本地验证：全套 **84 项通过**（31 项融合专项、53 项继承）。
覆盖 S1 模块实际执行、非零梯度及优化器更新，旧三尺度检查点原布局恢复，
以及 CPU bfloat16 autocast 下的关系块 FP32
执行、激活后的 EMA 检查点选择及显式 raw 覆盖、PET 内容敏感性和目录覆盖保护。
新增 S1 后，CACR-only 和 CACR+GCM 配置均完成 CPU / batch 1 / 512×512 的
真实双编码器前后向，输出都是
`[1,1,512,512]`，loss 与所有可训练参数梯度有限。GCM 内层的第一步零梯度符合上述初始化设计。

GPU AMP/batch-16 显存和真实 PET–CT Dice 提升需要在训练机器上验证。
切回原 baseline：`git switch exp/aligned-add-baseline`。
