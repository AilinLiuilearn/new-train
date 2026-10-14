# 逐尺度监督与前景／背景收益原型实验

基点：`exp/decoder-dual-prototype-add@038fc5b`。
实验分支：`exp/decoder-fg-bg-scale-prototype-add`。
所有实现使用新文件，原模型、入口和原型文件保持原样。

## 已实现结构

1. CT=ConvNeXtV2-Nano，PET=MiT-B1；双路独立 Conv1x1+BN+ReLU 对齐，
   Full 四尺度相加，Missing 不调用 PET 编码器/对齐层。共享 GroupNorm 解码器。
2. D4/D3/D2 复用解码器已有 1x1 辅助头，D1 复用最终头。同尺度三路共用
   同一组权重。Full 和 CT-reference 不接受原型；Missing 在每层解码块后校正。
3. 辅助 logits 先用 bilinear（align_corners=False）上采样至原始输入尺寸，
   再计算 BCE+Dice 和本尺度收益。D1 仅算一次主损失。默认损失为：

   `L = L_D1 + 0.2 * (L_D4 + L_D3 + L_D2)/3`

   `L_Dl = 0.5*L_Full,l + 0.5*L_Missing,l + 0.1*L_CTreference,l`。

4. 每尺度独立比较同一张 Full 图的 `p_Full,l` 与 `p_CTreference,l`：
   `gain = (p_CT-y)^2 - (p_Full-y)^2`。
   Full 必须阈值正确，且 gain>=0.01。没有前景2/背景1的手工系数。
5. 先计算 `gain_selected*y` 与 `gain_selected*(1-y)`，再 area 降采样。
   固定正交压缩到32维，沿用局部中心/3x3均值/局部变化的96维描述。
   Full 与 CT-reference 投影特征使用共同 RMS 尺度；每类窗口分别收益加权平均。
   特征描述包含邻域上下文，不宣称是纯病灶或纯背景内容。
6. D4/D3/D2/D1 各有前景、背景两个独立库；每库 K=4、capacity=256、
   min_candidates=24，共32个原型中心。没有通道 Gram 差分库或 PET 重建路径。
7. 每尺度局部查询将所有有效类别中心联合做一次注意力，Q/K 在投影后 L2
   归一化，固定逆温度 s=2，没有 sqrt(d) 除法。没有 CT 二值预测硬选库。
   只建立一个类别库时只读该库；两个都空时严格旁路。
8. 检索结果与当前 Missing 特征拼接，经3x3卷积生成单通道 sigmoid 门控。
   1x1输出头零初始化。原始残差按每图C/H/W的RMS平滑限制：

   `delta = raw * alpha*RMS(x)/(alpha*RMS(x)+RMS(raw))`，alpha=0.1。

   控制系数 detach；零头初始恒等且仍有梯度，零输入输出为零。
   这是每层**联合校正**上限；不同于旧版每个分支各0.1。不能保证最终HD95。
   比较新旧模块是在比较整体检索/注入设计，不能单独归因到分库存储。

## 比例自适应采样的确定规则

收益本身作候选内加权。类别比例只影响候选配额，不直接放大特征或校正。
每尺度统计 Full 行在原分辨率的有效前景/背景收益像素数量。
维护 `sampling_mass[2]`，首次非空成功观察直接复制，后续使用 decay=0.99 的
平滑均值；两类都无收益时不更新。统计仅在成功优化器步后提交，并随模型保存。

配额优先级 `a_c = 1/sqrt(sampling_mass_c + 1)`，归一化得到前景采样比例。
首次还没有历史时用本批收益像素数。不增加前景/背景增益系数。
每张 Full 图、每尺度总预算16个候选：

- 两类都有候选：前景配额为round(16*p_F)，限制到1至15；背景使用剩余配额。
- 各类内部按窗口本类收益平均值降序选择，不依赖全局随机采样。
- 某类候选不足：将未用配额分给另一个类的可用独立候选；不重复填充。
- 只有一类有收益：最多取该类16个候选。无收益则不产生候选。
- `--prototype_adaptive_sampling false` 改为两类各8个名额，使用同样不足重分配。

这会改善稀缺类别的采样机会，但如果稀缺类根本没有足够候选，自适应与固定
模式可能产生相同结果；日志记录实际生成量，不能仅凭开关宣称起效。
两库独立容量，背景不能覆盖前景。GT比例和采样历史不参与推理注意力。

## 生命周期和推理

候选提取、聚类、EMA均无梯度；固定投影矩阵不训练，查询投影运算仍向CT反传。
三路主/辅助BCE+Dice和空间求和均使用float32，避免早期512x512半精度求和溢出；
编码器、解码器和预测头仍可使用AMP。前100个成功优化器步训练辅助头，
同时可累计采样统计，但不写入原型候选。第101步开始写入，满足各库的候选/
唯一向量/非空聚类要求后激活。欧氏K-means、中心槽匹配、中心EMA=0.999及
每20成功步刷新沿用现实现。距离离群裁剪关闭。
候选来自本批优化器更新前的前向，成功更新后提交；AMP跳步全部丢弃，
包括采样历史。推理前向不接受GT、不计算CT-reference、不计算辅助头、不改库。
模型EMA若开启，所有原型和采样buffer直接复制，避免中心重复平均。

新检查点类型为 `decoder_fg_bg_scale_prototype_add`。严格拒绝旧双原型检查点
作为resume；新模式必须从头训练。每轮边界恢复网络、库、采样历史、优化器、
调度器、AMP、最佳Joint、原始/EMA选权重、所有随机状态及DataLoader生成器。
改变收益模式、采样/深度监督等实验策略的resume会明确拒绝。

## 入口和文件

| 文件 | 职责 |
| --- | --- |
| models/decoder_fg_bg_prototype.py | 类别候选、采样状态、独立库、联合检索与限幅 |
| models/fg_bg_prototype_add_model.py | 三路共享头、四尺度插入、独立收益 |
| tasks/fg_bg_prototype_seg.py | 三路分割/深度监督、成功提交、恢复 |
| run_decoder_fg_bg_prototype_add.py | 完整混合训练、最佳Joint、最终测试 |
| eval_decoder_fg_bg_prototype_add.py | 保存检查点的只读推理 |
| utils/fg_bg_prototype_diagnostics.py | 每100批及轮末写逐尺度诊断 |
| scripts/train_fg_bg_scale_prototypes.sh | 用户原始训练设置＋新模块的完整命令 |
| tests/test_fg_bg_prototypes.py | 新合同、入口/恢复/评估测试 |
| tests/smoke_fg_bg_prototype.py | 真实双编码器的无数据检查 |

CSV：train_log保留总体指标，新增深度监督损失、有限梯度计数/非有限梯度计数、
Full和Missing各自HD95。`scale_train.csv` 每快照4行记录三路本尺度BCE+Dice。
`prototype_gain.csv` 每快照4行记录本尺度收益、FN/FP、筛选量。
`prototype_banks.csv` 每快照8行记录FG/BG状态、采样统计、簇分布、中心漂移和
读库行为；`prototype_clusters.csv` 每快照32行记录各槽。均为本轮累计快照。

注意力是联合softmax：同一尺度所有有效前景+背景槽的注意力均值之和为1。
一个库内部的槽权重和是该类别注意力质量，**不要求单库内部之和为1**。
top1比例也在两个库联合统计；联合熵/总校正强度在两库行重复记录，不可相加。
收益/FN/FP来自训练Full行，不是验证指标。采样统计用于模型行为，不能当作
可丢弃的日志；其他日志detach且不影响输出/梯度，不保存病人标识或特征图。

## 运行和最少对照

```bash
git fetch origin
git switch exp/decoder-fg-bg-scale-prototype-add
git pull --ff-only origin exp/decoder-fg-bg-scale-prototype-add
bash scripts/train_fg_bg_scale_prototypes.sh
```

沿用用户授权的test.txt验证/测试共用、ImageNet归一化、CIPA增强与官方HD95。
这是已有实验口径，不将最终测试解释为独立于验证选模的无偏评估。
旧空间/通道开关、前景/背景权重和离群开关不属于新入口，不能照搬旧命令。
模型参数EMA默认关闭；中心EMA仍0.999；采样统计平滑系数另为0.99。

在同一脚本后追加参数即可覆盖默认值，每组必须使用不同hash：

| 实验 | 追加参数 |
| --- | --- |
| 新方案 | 默认per_scale＋DS0.2＋adaptive |
| 相同DS与双库，最终收益对照 | --hash add-fgbg-final-ds020-adaptive-k4-s2023 --prototype_gain_mode final |
| 相同DS与双库，固定采样对照 | --hash add-fgbg-per-scale-ds020-fixed-k4-s2023 --prototype_adaptive_sampling false |
| 相同DS和CT辅助监督，无原型 | --hash add-fgbg-ds020-no-prototype-s2023 --prototype_enabled false |

这些组分别检验收益来源、采样和模块总体贡献。旧分支仍可独立运行空间/通道
设计，但不是只改变一个变量的对照。建议先各跑一个相同seed，再根据走势
做必要复核；不能从一轮最好指标认定有效。

```bash
python eval_decoder_fg_bg_prototype_add.py \
  --checkpoint checkpoints_new/MDT/add-fgbg-per-scale-ds020-adaptive-k4-s2023/ckpt.best_joint.pth.tar
```

仅继续**本方案同一实验**时，在训练脚本后加 `--resume <本方案ckpt.last路径>`。

## 验证边界

完整回归 **111项通过**（原有90项＋新合同21项）。真实ConvNeXtV2-Nano/MiT-B1
完成64x64 CPU合成前向/反向/更新，以及512x512非零校正前向/PET隔离。
训练脚本通过bash语法检查、实际--help入口和全部参数解析。
额外可训练参数：原型模块78916个，三个辅助头899个，共79815个；
库和采样历史都是buffer，不包含在可训练参数数量中。

合同测试覆盖：零头基线恒等、辅助头共享与梯度、逐尺度收益差异、类别分离、
先乘后降采样、自适应/固定配额、不重复填充、部分库激活、RMS上限、零输入、
PET隔离、Full旁路、读库只读、CPU bfloat16、失败步/预热、采样状态恢复、
模型EMA的库buffer复制、日志输出/梯度/RNG一致、完整入口断点恢复和检查点评估。
真实数据与RTX3090训练仍须用户运行，不能由CPU合成检查证明Dice/HD95提升。

核心模块可独立导入，空库时直接返回输入；训练时用完整模态候选提交激活库：

```python
from models.decoder_fg_bg_prototype import ForegroundBackgroundScale
scale = ForegroundBackgroundScale(128, dim=32, clusters=4)
# x_missing: [B,128,H,W]，不接受GT/PET；零输出头初始恒等。
x_corrected = scale(x_missing)
# 候选提取仅训练调用：Full/reference特征匹配；gain/GT为原分辨率[B,1,h,w]。
fg_candidates, bg_candidates, counts = scale.candidates(full, ct_reference, gain, gt)
# 正式模型使用ForegroundBackgroundMemory.commit处理warmup和成功步，
# 不在Missing前向中add/refresh。
```
