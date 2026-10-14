# 双原型校正实验：实现与审查说明

本分支保留 `exp/aligned-add-baseline` 的 `663f978` 对齐基线。
目标是只验证原型知识设计：每个 batch 严格 50% Full、50% Missing；
Full 保持 CT+PET 简单相加。没有引入此前讨论的复杂融合模块。

## 代码位置

| 文件 | 职责 |
| --- | --- |
| `models/decoder_dual_prototype.py` | 收益筛选、空间/通道候选、欧氏聚类、可选离群过滤、中心 EMA、交叉注意力 |
| `models/prototype_add_model.py` | Add 基线扩展，在解码器 D4/D3/D2/D1 接入 Missing 校正 |
| `tasks/prototype_seg.py` | 精确半批损失、CT 对照监督、成功更新后建库、模型 EMA 同步和恢复 |
| `run_decoder_prototype_add.py` | 独立混合训练入口、日志、最佳 Joint 选择、最终测试 |
| `eval_decoder_prototype_add.py` | 仅使用已保存原型进行评估 |
| `tests/test_decoder_prototypes.py` | 原型、数据隔离、梯度、恢复和完整入口的合同测试 |
| `tests/smoke_decoder_prototype.py` | 真实双编码器的无数据合成检查 |

已合入该基线的双模态对齐和官方 HD95 实现，并统一默认 ImageNet 输入归一化。
GroupNorm 解码器和简单相加融合保持原样。原有两个测试中的
checkpoint 读取改为调用仓库已有的加载函数，以兼容新版 PyTorch 的加载默认值。

## 已固定的实现细节

1. 原型从 **Full 行的解码表征**提取，按解码层独立维护；默认通道从深到浅为
   512/256/128/64。Missing 当前解码表征生成 Q，原型生成 K/V。
   Full 的 CT 和 PET 均独立经过 Conv1x1 + BatchNorm + ReLU 后再相加；
   Missing 不运行 PET 编码器或 PET 对齐层。
2. 空间库存储收益区域的局部 Full 描述；通道库存储相同收益区域内有符号的
   `Gram(Full)-Gram(CT对照)`。没有增加单独的 CT 检索键。
3. 压缩采用每尺度固定、可复现、随 checkpoint 保存的正交基，默认 32 维。
   注意力投影与修正头可学习。避免另设一个未得到监督、坐标持续漂移的建库投影。
4. 输出修正头零初始化；空库直接旁路。模型前向不更新原型，验证/测试只读库。
5. PET 仅编码 Full 行；Missing 支持 `pet=None`。Full 绕过原型校正。
6. 建库候选无梯度；优化器成功更新后才写入。AMP 溢出跳步丢弃候选。
7. 默认前 100 个成功优化步骤不收集；每库至少 24 个候选、6 个不同中心后激活。
   缓冲区 256 条，初始化及每 20 步刷新。中心 EMA 默认 0.999，可配置。
   特征距离离群过滤默认关闭：`--prototype_outlier_filter false`，每簇全部成员
   参与求均值。收益筛选、非有限值检查和空簇检查仍保留。后续消融可显式设为
   `true`。日志记录实际开关。旧 checkpoint 未保存此参数时按历史开启解释，
   恢复训练时改变该策略会明确报错，避免静默改变实验。
8. 模型 EMA 对网络参数正常平均，对已执行中心 EMA 的原型 buffer 直接复制，
   避免二次平均。原型、队列、有效标记和计数都能恢复。
9. epoch 边界恢复包括优化器、调度器、AMP、混合分配步数、早停状态，以及
   Python/NumPy/Torch/CUDA/DataLoader 的随机状态。
10. 最终测试严格使用最佳 Joint 当时记录的 raw/EMA 权重，避免选中早期 raw
    checkpoint 却错误评估尚未启用的 EMA。

## 一个必须保持公平的实验设置

收益比较需要训练过的 CT 对照预测，因此默认总损失是：

`0.5 × Full损失 + 0.5 × Missing损失 + 0.1 × CT对照损失`。

CT 对照只对 Full 子集额外解码，不增加 CT 编码次数。这个额外监督本身也
可能改善结果，必须做以下对照，不能把所有提高都归因于原型：

| 实验 | 参数 |
| --- | --- |
| 原始混合 Add 目标 | `--prototype_enabled false --prototype_ct_aux_weight 0` |
| 相同额外 CT 监督、无原型 | `--prototype_enabled false --prototype_ct_aux_weight 0.1` |
| 仅空间原型 | `--prototype_spatial true --prototype_channel false` |
| 仅通道原型 | `--prototype_spatial false --prototype_channel true` |
| 双原型 | 默认参数 |

默认 2:1 的前景/背景系数影响收益候选选择和混合区域的组成，不能解释为
所有前景原型都会被放大两倍，因为归一化平均中的统一系数会抵消。

## 运行

沿用原基线环境和预训练权重。下面使用已确认的验证/测试共用 `test.txt` 设置。

```bash
python run_decoder_prototype_add.py \
  --hash add-mixed-dualproto-k4-diag-s2023 \
  --root /root/autodl-tmp/data/PCLT20K \
  --train_split_file train_original.txt \
  --val_split_file test.txt --test_split_file test.txt \
  --allow_val_equals_test true \
  --image_size_2d 512 --batch_size 16 --epochs 60 \
  --ct_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano \
  --pet_pretrained_path /root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1 \
  --prototype_clusters 4 --prototype_outlier_filter false --log_interval 100 \
  --aug_mode cipa --norm_mode imagenet --hd95_mode cipa_official --random_state 2023
```

学习率、EMA、增强和划分应与对照组保持一致；来源分支默认关闭模型 EMA，
本入口保留这个默认值，需要时显式加入 `--ema_enabled true`。

三个基线约束已经固定：默认 ImageNet 归一化、双路对齐后相加、
验证与最终测试都使用同一 `cipa_official` HD95 口径。数据增强仍是 `cipa`，
它与输入归一化是不同设置。请安装 `MedPy`；官方 HD95 要求 512×512 掩码。
Dice/IoU 沿用基线的 sigmoid 0.5 阈值和全数据像素计数；HD95 的中心点补救
只作用于内部副本，不改变 Dice 或漏检计数。旧版缺少 PET 对齐层的 checkpoint
与修正后结构不兼容，正式实验应重新训练。小尺寸合成入口测试显式使用 legacy，
不代表正式训练改变默认评价口径。

同样参数、同一 `--hash` 下，额外加入以下参数恢复：

```bash
--resume checkpoints_new/MDT/add-mixed-dualproto-k4-diag-s2023/ckpt.last.pth.tar
```

单独评估：

```bash
python eval_decoder_prototype_add.py \
  --checkpoint checkpoints_new/MDT/add-mixed-dualproto-k4-diag-s2023/ckpt.best_joint.pth.tar
```

训练时检查 `prototype_log.jsonl` 的逐尺度有效标记、候选数和更新次数，
确认两类库实际激活，而不是仅凭最终 Dice 判断原型已经生效。

终端默认每 **100 个完成的 batch** 打印一次训练/验证进度，保留每轮开始、
结束和 checkpoint 阶段提示，移除首批/首次前向调试提示。每 100 个训练 batch
以及每轮训练结束写一次诊断，未满 100 批的一轮也会有记录。

新增三个 CSV，原 `train_log.csv` 的表头保持兼容：

| 文件 | 用途 | 关键字段 |
| --- | --- | --- |
| `prototype_gain.csv` | 判断是否筛出了足够有用区域，以及前景是否稀缺 | `useful_image_fraction`、`fg_selected_fraction`、`bg_selected_fraction`、`fg_mean_selected_gain`、Full/CT-reference 的 FN/FP 及比率、成功/跳过/warmup batch 数 |
| `prototype_banks.csv` | 按 D4–D1、空间/通道各库观察建库与更新状态 | `status`、`pool_size`、`accepted_total`、`generated_epoch`、`written_epoch`、更新失败原因、`occupied_clusters`、`assignment_entropy_normalized`、`max_cluster_fraction`、中心间距、EMA 前后移动量、读库注意力熵及校正 RMS 比 |
| `prototype_clusters.csv` | 观察每个原型槽的样本分布和使用情况 | `pool_count`、`pool_fraction`、`mean_member_distance`、`center_norm`、`last_success_count`、`read_attention_mean`、`read_top1_fraction` |

读这些记录时需明确以下口径：

1. 仍然是每尺度两个库、每库 K 个无监督聚类；没有新增病灶/背景子库。
   空间候选里的 FG patch 指含有正前景收益质量的窗口，可能混有背景。
2. `pool_size` 是当前 FIFO 占用，最多 256；`accepted_total` 是历史累计写入，
   可以超过 256。`generated_epoch` 包括 warmup 和优化器跳过批次产生的候选；
   `written_epoch` 仅统计成功更新且结束 warmup 后真正写入的候选。
3. 每个快照的收益计数和 `_epoch` 计数都是**本轮累计**，不能把相邻行相加；
   `event` 区分间隔快照与轮末快照，轮末刚好是第 100 批时会保留两种事件。
   无前景等情况下的比率、建库前的簇分布会留空，避免把未定义误记为 0。
4. `pool_count` 是当前缓冲区对当前 **EMA 中心**的最近中心分配；每库各簇
   计数之和等于 `pool_size`。`last_success_count` 是最近一次成功 K-means
   在 EMA/可选离群裁剪之前的分簇数，已匹配到同样的槽编号。两者可以不同。
   归一化分布熵 1 表示均匀、0 表示全部落到一簇；中心编号不是病灶语义标签。
5. `last_attempt_status` 是最近尝试的状态；`last_success_*` 和中心移动量
   来自最近一次成功刷新。首次建库直接复制中心，EMA 移动量留空。之后对比
   `last_raw_center_drift` 与 `last_applied_center_drift`，能看出 0.999 是否
   使实际更新很慢。不同维度/尺度的距离不能直接当作同一量纲比较。
6. 收益和 FN/FP 只比较 **Full 行**的 Full/CT-reference 输出，并不是验证集
   指标，也不是主训练 Missing 半批的表现。原始增益均值不含前景/背景权重；
   加权质量占比才含权重。可靠但低于阈值的像素计入 `below_gain`。
7. 注意力统计只采样打印批次和最后一批的 **Missing 行**，在优化器更新前
   读库；建库快照则在本批成功提交后计算。检查 `read_sample_global_step`
   和 `read_available`：本批刚初始化的库，在本批前向时可能尚不可读。
   `read_attention_mean` 是各中心平均注意力，`read_top1_fraction` 是成为
   最大注意力中心的查询占比，两者分别在各簇上求和为 1。
8. 校正 RMS / 输入 RMS 比可以观察模块是否输出了实际调整；零初始化时为 0。
   注意力熵、门控大小或非零校正均不能单独证明调整有益，仍要结合 Missing
   验证指标与消融对照。空间查询数按解码器位置计，通道查询数按 Missing 图像计。
9. 前 100 个**成功优化器步**是既有 warmup，这时库为空、状态 `warmup` 正常；
   第 101 步开始写入，满足候选数/唯一向量要求后建立库。优化器溢出跳过会
   推迟成功步计数。`successful_steps` 是原型成功提交数，关闭模块时不增长。

所有诊断都 detach，不进入损失、检索或候选筛选；不保存患者标识或特征张量，
也不增加模型前向。池与中心的距离仅在快照时于 CPU 计算，复杂度为 N×K×D，
不会构造 N×N 的样本矩阵；校正强度计算也避免额外构造完整通道残差特征图。

新 checkpoint 单独保存刷新诊断元数据，没有改变模型 tensor state_dict。
旧 checkpoint 可以严格加载，缺失的尝试/失败/最近刷新元数据从恢复后重新累计，
历史 accepted/update tensor 计数仍保留；无法从 checkpoint 重建之前缺失的 CSV。
恢复时清理 checkpoint 之后未完成轮次的诊断行，防止重复计算中断的半轮。
已有进程不会热加载更改，需下次启动/恢复后生效。

## 已完成的验证与边界

本次完整测试：80 项通过。新增覆盖手工收益/簇分布计数、EMA 实际移动量、
诊断只读与不消耗随机数、注意力统计开启前后的输出/梯度逐元素一致、
不同日志频率下的连续/断点恢复一致、warmup/溢出跳过的写入统计、
旧检查点兼容、CSV 快照与中断半轮清理、默认每 100 批的打印行为。

- 原型合同测试包含零初始化基线等价、Full 旁路、PET 隔离、行顺序、
  收益正确性、无收益拒绝、有符号矩阵、两分支梯度、CPU bfloat16、
  原型恢复、模型 EMA、读库有效性和连续/中断恢复一致性。
- 真实 ConvNeXtV2-Nano + MiT-B1：64×64 合成输入完成前向、反向、参数更新；
  512×512 合成输入完成启用非零校正的前向和 Missing PET 隔离检查。
- 默认新增可学习参数 546,756；模型总参数 32,989,397。
- 未提供患者数据或本地预训练权重，未运行真实训练；环境无 CUDA，未验证
  RTX 3090 的 batch-16 显存和 CUDA AMP。CPU bfloat16 通过不能代替 CUDA 验证。

这里验证的是实现合同和执行流程，不是性能结论，也不能保证任意环境和
未覆盖输入都没有缺陷。空间/通道两分支是职责分解，尚未证明严格解耦；
收益筛选也不是 PET 因果贡献识别。应通过上述对照判断实际有效性。

更完整的数学约定、状态生命周期和参考来源见 `DECODER_PROTOTYPES.md`。
