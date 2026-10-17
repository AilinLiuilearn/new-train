# CT Key＋可学习仿射 Value：实验设计与运行说明

分支：`exp/ct-key-learnable-affine-value`。
基础：已清理的 `exp/ct-full-spatial-modulation-bank@e697de2`；编码器、对齐、AddFusion、共享解码器、数据、分割损失、选模和指标实现保持不变。旧统计调制库保留在原分支。

## 独立四尺度与数据流

对齐通道为64/128/320/512，512输入时空间为128/64/32/16。
Full：CT C与真实PET P正常相加F=C+P，进入共享解码器。Full主分割不读取库。
Missing：仅编码CT，查询本尺度独立库，生成逐位置gamma/beta，校正CT后进入同一个解码器；不调用PET编码器、不使用GT或Full特征。

每尺度默认K=8；Key [K,C]为buffer；gamma、beta [K,C]为nn.Parameter。四尺度Value总参数16384。没有额外生成器、分割头、深度监督、前景/背景分库、门控或RMS限幅。

## 建Key与更新

Key描述为CT的3×3邻域均值，L2归一化后按余弦相似度硬分配。所有Full CT位置参与，包括有限零向量；不做收益筛选、256候选缓存、随机抽样。chunk_size=8192仅控制统计分块，不丢样本。
首次成功batch采用确定性最远方向初始化中心。epoch1按每批当前中心分配所有位置，累计归一化Key和及计数，归一化累计均值；这是流式聚类，非全epoch离线K-means。
epoch>=2对有成员槽位使用本批Key均值做EMA：normalize(.99*K+.01*batch_mean)。空槽保留原中心，不重排槽号。
所有Key更新都在本次backward和成功optimizer step之后。失败batch丢弃pending Key统计。GT仅记录槽位病灶占比，不影响分配或更新权重。

## 读取与空间调制

对每个Missing位置i，使用同样的CT局部描述查询：
`a_ij = softmax_j(cosine(describe(C)_i, K_j)/temperature)`，默认temperature=.5。
`gamma_hat_i = sum_j a_ij*gamma_j`，`beta_hat_i = sum_j a_ij*beta_j`。
`C_hat_i = (1+gamma_hat_i)*C_i + beta_hat_i`。
查询描述归一化；被调制的CT保留原始幅值。各位置权重不同，得到[B,C,H,W]调制图；库不存整张二维图或绝对坐标，不直接搬运邻居内容。

## Value学习与辅助监督

gamma、beta零初始化，仅由优化器更新，绝不被Key统计/EMA覆盖。
Full的旁路只用detach(C)执行与Missing完全相同的读取和仿射，目标是detach(F)。Full信息通过损失梯度进入Value，而不是作为查询/生成器输入。
每样本每尺度：`mean(abs(retrieve_and_modulate(detach(C))-detach(F)))/(RMS(detach(F))+1e-6)`，平均Full样本与四尺度，得到Lret。
`L=.5*Lseg_full+.5*Lseg_missing+matrix_mod_loss_weight*Lret`；默认权重.1，待实验验证。
Lret只更新Value；Missing分割更新Value、CT编码器/对齐和共享解码器；Full分割更新CT、PET及共享解码器。Key不反传；Missing查询对CT保持可微。

epoch1仅积累Key，gamma/beta保持0，训练和验证均不读库，Lret=0；epoch2开始读库并训练Value。验证/测试库全部冻结。

## 训练命令与开关

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch exp/ct-key-learnable-affine-value
git pull --ff-only origin exp/ct-key-learnable-affine-value
bash scripts/train_spatial_modulation.sh
```

默认hash：`add-learnable-affine-k8-lret010-s2023`，其余正式训练配置沿用清理分支（seed2023、512、batch16、mixed各半、预训练、AdamW8e-5、AMP、60轮、patience10、模型EMA关闭）。Key EMA始终独立存在。

关闭Full特征辅助损失，Value仍通过Missing最终分割学习：
```bash
bash scripts/train_spatial_modulation.sh --matrix_mod_loss_enabled false --hash add-learnable-affine-k8-no-lret-s2023
```

关闭整个模块（同时关闭Key写入和辅助损失）：
```bash
bash scripts/train_spatial_modulation.sh --matrix_enabled false --hash add-learnable-affine-off-s2023
```

调整辅助权重：`--matrix_mod_loss_weight 0.05`，同时使用新的hash。值为0或开关false均不向辅助路径提供有效梯度；不能把关闭辅助损失解释为关闭Value训练。

恢复同一实验：原命令加 `--resume <ckpt.last.pth.tar>`。配置必须一致。
`experiment_type=ct_key_learnable_affine_value`；旧统计库/矩阵checkpoint无法直接resume或用新评估器读取，避免参数语义及优化器状态混用。新实验从头加载backbone预训练权重训练。
`model.matrix`及`matrix_*`CLI名保留以减少接口改动；它们表示本仿射模块。

## 日志与验证

`modulation_slots.csv`记录每槽count/foreground_fraction/key_norm/gamma_rms/beta_rms、Value梯度范数、读取注意力均值、逐查询最大权重均值和归一化熵、Missing校正强度。
新增retrieval_error和identity_error比较同一Full训练batch中检索后特征和原CT相对Full的误差，并分别记录GT前景/背景误差。epoch1这些误差为空，辅助损失为0。
主CSV的modulation_loss沿用字段名，现指实际检索输出Lret的epoch平均；拟合/读取/梯度统计为采样批次，不能当作全数据集指标。梯度记录在unscale/clip之后；失败步骤仍可能显示非有限梯度，结合skipped_updates判断。
count是累计硬分配数，Value按软读取梯度更新，两种计数不能混为一谈。

```bash
python -u eval_spatial_modulation_add.py \
  --checkpoint checkpoints_new/MDT/add-learnable-affine-k8-lret010-s2023/ckpt.best_joint.pth.tar \
  --root /root/autodl-tmp/data/PCLT20K \
  --use_ema false --compare_on_off true --missing_rates 0 1 \
  --output_dir checkpoints_new/MDT/add-learnable-affine-k8-lret010-s2023/diagnosis_on_off
```

该命令评估同checkpoint读取开关并生成GT区域校正强度；它用于解释推理作用，不能替代从头训练的模块消融，也不是整体方法唯一的通过条件。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python tests/smoke_spatial_modulation.py --size 64 --backward
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python tests/smoke_spatial_modulation.py --size 512
```

假设与风险：CT相似不一定意味着PET作用相似；8个仿射槽位表达有限；背景可能主导全图拟合；Key EMA漂移可能暂时改变Value的使用分布；特征拟合与分割目标可能冲突。此实现验证监督检索输出及可学习Value的作用，不宣称已获得分割收益。

## 本次本地验证

8项测试通过，覆盖全位置Key统计及GT不参与路由、Key写入不覆盖Value、首轮恒等与无Value梯度、辅助损失只训练Value且可降低检索误差、关闭辅助损失后Missing仍能更新Value、位置相关读取及查询梯度、Full与干净Add基线严格一致、零Value的Missing与基线严格一致、Missing禁止调用PET、评估禁止写库、失败步骤不写库、参数及优化器状态恢复、旧版本拒绝恢复、两轮训练/存档/评估与整个模块关闭。
真实ConvNeXtV2-Nano/MiT-B1、合成数据、无预训练：CPU 64输入两轮前向/反向/优化器更新通过；CPU 512输入K8 Full/Missing前向和评估状态不变检查通过。
Shell语法和git diff空白检查通过。未连接AutoDL、未使用真实数据/预训练checkpoint、未验证CUDA AMP或正式分割收益。
