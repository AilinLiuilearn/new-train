# 单模型 PET–CT 局部重组融合与内部修正损失

当前分支：`exp/full-add-local-reassembly`。当前配置/损失 schema=2。

## 当前训练形式

只有一个可训练的分割网络：
CT/PET → 独立编码器 → 四尺度对齐 → 局部重组融合 → GN 解码器 → 分割。

- **所有组件均随机初始化，包括 CT、PET 编码器；不读取任何预训练或分割权重。**
- 没有学生/教师、冻结 baseline、外部参照模型、蒸馏或原型库。
- 不再提供 baseline_checkpoint、reference_checkpoint、pretrained 或预训练路径参数。
- 优化器、学习率调度和 EMA 从第1轮建立。EMA 是普通评价权重平均，不提供损失参照。
- 默认开启融合和修正损失；两者均有独立开关。

若之前相加 baseline 使用了编码器预训练，则当前全随机模型不能与它单独比较来归因于融合。
结构消融须使用本入口关闭融合，保证初始化、损失、学习率和预算一致。

## 融合模块

S1..S4 通道 `[64,128,320,512]`，低维通道 `[16,16,40,64]`，两个注意力头，各尺度独立参数。
CT query 选择 PET 的3×3邻域内容，反方向同样执行；每个位置还有一个可学习 key、固定零 value
的空候选，越界邻居在 softmax 前屏蔽。通道压缩、逐位置 LayerNorm 后生成 Q/K，value保留原低维表达。

`delta_P = U_P(transport(P|C)-project(P))`
`delta_C = U_C(transport(C|P)-project(C))`
`F = C+P+delta_C+delta_P`

仅两个输出投影置零，无输出偏置，无末端ReLU，不乘2，初始严格等于相加。
第一步反传内部梯度为零、输出头有梯度；输出头更新后内部投影、Q/K和空候选获得梯度。
核心 FP32，输出转换为 Add 的 dtype；支持 AMP，不支持整个模块 `.half()`/`.double()`。
不使用全局 N×N 注意力、Triton 或自定义 CUDA 扩展。融合新增参数220,208，默认网络总参数32,662,849。
空候选提供拒绝邻域内容的能力，不代表经过验证的医学可靠性。

## 修正损失：同一个模型的内部相加比较

CT/PET只编码一次，得到四尺度 C/P。开启修正损失时：

1. 常规路径：`z_fuse = Decoder(F)`，正常反向传播。
2. 对照计算：在 `torch.no_grad()` 下，用**同一个解码器**计算 `z_add = Decoder(C+P)`。
   复用已经产生的特征，不再调用编码器，不复制解码器，不加载任何权重。
3. 用训练GT确定当前相加输出按0.5阈值预测正确的位置。
4. 正确前景/背景分别平均 `relu(BCE(z_fuse,GT)-BCE(z_add,GT))`，再对非空类别平均。

`L = BCE_Dice(z_fuse,GT) + 0.1 * L_internal_add`

0.1是未验证的起点，不承诺最优。参照预测和筛选 mask 均停止梯度；GT仅用于训练损失。
对照分支不产生自己的梯度，主路径和修正项共同更新整个网络。
现有解码器只包含 Conv/GroupNorm/ReLU，没有BN或Dropout，第二次解码不额外更新运行统计。
初始融合等于相加，所以两条输出相同，修正损失初始为零（相同精度下）。

**这已经不是“冻结最佳 baseline 的保护损失”。** 内部相加对照随当前网络变化，
只检验当前融合是否比当前相加更差，不能保护历史 baseline 的能力，也不保证最终Dice提升。
对照路径未单独接受分割监督，可能较弱；需通过有/无损失消融判断是否有价值。
它不是PNDC原版或其他论文的直接复现，不将该组合宣称为已验证创新。

关闭 `--protect_enabled false` 后只计算原 BCE+Dice，解码器也只前向一次。
验证/推理默认只走融合预测，不计算相加对照或修正损失，不需要GT。
继承原Missing路径：缺失PET时不调用PET编码器或融合，直接CT-only；auto只编码Full行PET。
目前仍是Full训练，未接入Stage-2检索补偿。

## 运行

先停止旧训练，再在正确分支更新：

```bash
cd /root/autodl-tmp/mkd-main/new-train
git switch exp/full-add-local-reassembly
git pull --ff-only
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/check_local_reassembly.py --device cuda --image_size 512 --batch_size 2
bash scripts/run_full_local_reassembly.sh
```

默认输出 `checkpoints_new/MDT/full-local-single-random-protect-lr4e5-s2023/`。
同名非空目录会拒绝覆盖；重新运行用新 `--hash`。
启动日志应为 `mode=all_random pretrained=False external_weights=none`，
修正方式应为 `same_model_detached_Add`，不会出现参照模型权重来源信息。
不要再传旧的 baseline_checkpoint/pretrained 参数。

保留当前 lr4e-5、AdamW、batch16、512输入、60epoch、warmup3、EMA0.999、clip5、seed2023。
数据仍为 train_original.txt / test.txt / test.txt，并显式允许val与test相同，
与原开发诊断一致，不是独立保留测试评估。
原主任务类的历史名字含Teacher，但新入口只将它作为普通分割训练器使用，不创建教师网络。

四组消融：

```bash
bash scripts/run_full_local_reassembly.sh --local_enabled false --protect_enabled false --hash add-random-s2023
bash scripts/run_full_local_reassembly.sh --local_enabled false --protect_enabled true  --hash add-random-correction-s2023
bash scripts/run_full_local_reassembly.sh --local_enabled true  --protect_enabled false --hash local-random-no-loss-s2023
bash scripts/run_full_local_reassembly.sh --local_enabled true  --protect_enabled true  --hash local-random-correction-s2023
```

关闭融合时两条路径相同，内部修正项应为零，是一致性检查。
评价全量Dice、病例平均、小前景组、边界F1、FP/FN和修复/新增错误；不能只凭单次峰值判断有效。

## 日志与 checkpoint

保留原指标、梯度，额外记录主损失、修正原值及加权值、有效正确前景/背景计数，
四尺度残差范数比、空候选/中心注意力质量。后两项不代表贡献正确或临床可靠。
保存普通模型raw、EMA和优化器等训练状态；独立评估严格检查schema和参数。
架构张量相同，但旧的冻结参照损失schema=1与当前协议不同，不混用其训练配置。

```bash
python eval_full_local_reassembly.py \
  --checkpoint checkpoints_new/MDT/full-local-single-random-protect-lr4e5-s2023/ckpt.best_full.pth.tar \
  --output_dir eval_local_single --device cuda
```

## 实现来源与验证

双向逐位置交互借鉴 GeminiFusion（ICML2024）的思路，原代码：
https://github.com/JiaDingCN/GeminiFusion/blob/main/models/mix_transformer.py 。
正负影响校准思想参考PNDC（TMI2025），官方代码：https://github.com/yansheng-qiu/PNDC 。
本实现的局部候选、零value候选、中心残差与内部Add损失均是任务适配，不声称原版复现。

单元验证覆盖：禁止初始化读权重、相加初始化、空候选与边界mask、两步梯度、
编码器只运行一次/解码器复用、Add输出无梯度、Missing隔离、损失/融合四开关、
训练/EMA/严格checkpoint重载和旧损失schema拒绝。
本地CPU测试不证明CUDA可用、batch16显存足够或真实数据有效。

本地验证（PyTorch2.5.1 CPU）：新入口23项通过、1项CUDA跳过；原Full Add/EMA/混合训练
回归37项通过。512×512、batch2、真实双编码器两步预检通过，主损失1.749383→1.581113，
初始内部修正损失0，第二步约1.67e-6；调用计数为CT/PET各2次、同一解码器4次。
这些只验证数值/生命周期，不代表真实数据性能。新单模型路径尚需在服务器重新做CUDA预检。
