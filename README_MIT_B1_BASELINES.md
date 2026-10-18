# MiT-B1 编码器诊断实验（lr=4e-5）

当前分支：`exp/full-add-local-reassembly`。新增两个独立启动脚本，复用原相加/CT-only训练器。
不调用 LocalReassemblyFusion，不使用内部相加修正损失，不加载训练好的分割 baseline。

## 两组设置

| 实验 | CT 编码器 | PET 编码器 | 融合 | 主损失 |
|---|---|---|---|---|
| Full 同构 | MiT-B1 | 独立参数的 MiT-B1 | 原对齐层后逐尺度相加 | BCE+Dice |
| CT-only | MiT-B1 | 无 | 无 | BCE+Dice |

同构不等于共享权重：两个编码器分别读取同一份 MiT-B1 预训练，初始化数值相同，
之后各自更新。CT-only 不创建 PET 编码器，不读 PET 图像，不是把双模态模型 PET 置零。
两个实验均保留原四尺度 `Conv1x1+BN+ReLU` 对齐层和 GN 解码器。
MiT-B1 四尺度通道已经是64/128/320/512，仍保留对齐层，以免同时改变其他结构。

共同设置：lr4e-5、AdamW、weight_decay1e-4、512、batch16、60epoch、warmup3、
cosine_min_lr1e-6、flat_ratio0.3、AMP、EMA0.999/start_epoch3/warmup、clip5、seed2023、
cipa增强、imagenet归一化、cipa_official HD95、无深度监督。
60轮是预算，尚不能断言MiT-B1已经训练到最佳。学习率保持一致用于控制比较，不能因此宣称对所有编码器都最优。

预训练来源：`/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1`。
不要把ConvNeXt预训练目录传给CT的MiT-B1。缺失/不兼容来源会报错。
对齐层和解码器重新初始化，所有模型参数参与训练，优化器/调度/EMA从本次实验建立。

## 运行

```bash
cd /root/autodl-tmp/mkd-main/new-train
git switch exp/full-add-local-reassembly
git pull --ff-only
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 双模态和CT-only的CUDA两步预检（依次执行，不同时保留两个模型）
python scripts/check_mit_b1_baselines.py \
  --device cuda --image_size 512 --batch_size 2

# 实验1：同构双MiT-B1，相加
bash scripts/run_full_add_mitb1_lr4e5.sh

# 实验2：MiT-B1 CT-only；建议等上一组结束后运行
bash scripts/run_ct_only_mitb1_lr4e5.sh
```

单独预检可用 `--mode full` 或 `--mode ct_only`。
预检默认读取真实编码器预训练；本地随机初始化执行检查可加 `--device cpu --pretrained false`。
更换预训练路径：预检使用 `--mit_pretrained_path`；双模态训练分别传
`--ct_pretrained_path` 和 `--pet_pretrained_path`；CT-only只传前者。
预检batch2通过不代表训练batch16的显存足够。

两个训练脚本默认自动加时间戳，输出目录分别以：

- `checkpoints_new/MDT/full-add-mitb1-lr4e5-60ep-ema-s2023-...`
- `checkpoints_new/MDT/ct-only-mitb1-lr4e5-60ep-ema-s2023-...`

开头。可以用 `--hash 新实验名` 覆盖，请勿复用已有实验目录。参数写在脚本最后可覆盖默认值。
双模态选择 `ckpt.best_full.pth.tar`，输出 `final_test_full.json`；
CT-only选择 `ckpt.best_ct.pth.tar`，输出 `final_test_ct.json`，每轮日志均为 `train_log.csv`。
修正了旧CT-only最终评价误传PET参数的问题，改成纯CT评价一次；最终遵循checkpoint记录的raw/EMA选择。
默认ConvNeXt CT-only仍可运行，本次只是增加MiT-B1支持。

数据仍为train_original.txt / test.txt / test.txt，保持旧开发实验协议。val和test相同，
两者不是独立评价，最终结果仍需保留独立患者测试集核验。

## 这些实验能回答什么

1. 旧的Nano(CT)+MiT(PET)相加 vs 新的MiT(CT)+MiT(PET)相加：
   替换CT编码器以后，在同一训练协议下是否更好。差异包含编码器结构、预训练和参数容量变化，
   不能把增益全部归因于“同构”。
2. MiT CT-only vs MiT+MiT相加：相同CT编码器下，加入真实PET的双分支模型是否带来增益。
   两者为独立训练模型，双分支容量更多，差值并不只度量PET信息本身。
3. 即便两组都停留在相近水平，也不能据此证明框架/数据的理论上限或排除解码器因素。

更完整的控制矩阵是“CT编码器Nano/MiT” × “CT-only/CT+PET”，PET编码器始终MiT。
如果没有同设置的Nano CT-only，可在第二个脚本上覆盖：

```bash
bash scripts/run_ct_only_mitb1_lr4e5.sh \
  --ct_backbone convnextv2_nano \
  --ct_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano \
  --hash ct-only-nano-lr4e5-control-s2023
```

先做现有seed2023的筛查；出现值得跟进的差异后再用配对多seed复核。
同时比较病例平均、小前景组、FP/FN、边界和HD95，以及固定预算下的训练曲线。
不能只比较某轮峰值，不能将训练后置零PET的结果当作独立CT-only训练。

## 本次验证

新增测试6项、原Full Add/EMA/混合训练回归37项，均通过。测试覆盖两份启动脚本的实际参数、
兼容MiT预训练文件读取、双编码器存储独立、CT-only无PET、两组一轮训练/保存/严格重载/最终评价，
并验证EMA未激活时最终使用raw。
双组512×512、batch2的CPU两步前反传预检通过；参数量Full=30,541,889，CT-only=17,003,393。
CPU smoke使用随机初始化，权重读取测试使用本地生成的兼容MiT权重。
服务器的真实预训练文件、CUDA/AMP和真实数据结果尚未在本地验证，请运行CUDA预检。
