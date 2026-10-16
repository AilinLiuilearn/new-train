# 融合停滞诊断：先定位证据，再修改模型

当前分支 exp/full-cmx-ffm 新增独立脚本 scripts/diagnose_fusion_plateau.py。它支持本仓库的 Full Add baseline 与 Full CMX/SKFF checkpoint，不改变模型、训练入口或已有结果。

**实现状态：本次工作环境连接失败，无法运行 Python。因此下面的代码、自测和真实模型链路尚未在本次会话执行；不能沿用以前 SKFF 测试通过的数字来声称此脚本已通过测试。** 已核对当前模型、checkpoint、数据集和指标接口。请先执行数据无关自测，再在服务器运行诊断。自测失败时不要继续解释诊断结果。

## 先运行自测

~~~bash
cd /root/autodl-tmp/mkd-main/new-train
git pull --ff-only origin exp/full-cmx-ffm
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m py_compile scripts/diagnose_fusion_plateau.py
python scripts/diagnose_fusion_plateau.py --self_test
# 可选：完整的合成单元测试（需要 pytest）
python -m pytest -q tests/test_fusion_plateau_diagnosis.py
~~~

自测检查混淆计数、GT oracle、患者 bootstrap、跨患者 PET 配对、固定子集、2D 连通分量、非有限日志条目、小样本拟合副本不影响原模型，以及合成模型的缓存特征评估和报告生成。它不验证真实骨干/CUDA/checkpoint 接入。

## 第一轮：少量患者的完整对照

~~~bash
python scripts/diagnose_fusion_plateau.py \
  --baseline_checkpoint checkpoints_new/MDT/full-add-baseline/ckpt.best_full.pth.tar \
  --fusion_checkpoint checkpoints_new/MDT/full-skff-imagenet-ema-s2023/ckpt.best_full.pth.tar \
  --root /root/autodl-tmp/data/PCLT20K \
  --output_dir diagnostics/skff-vs-add-quick \
  --device cuda --batch_size 2 --num_workers 4 \
  --max_cases 20 --slices_per_case 8
~~~

请将两个 checkpoint 路径换成实际的文件。CMX 也可以直接作为 fusion_checkpoint；当前脚本不支持 CAiNet、原型检索分支或其他未核对架构。支持的是 independently trained Full Add reference；它不是从融合 checkpoint 临时改出的 Add。

默认固定随机种子，最多选 20 名评估患者、每人 8 张切片，以及独立的最多 20 名训练患者、每人 8 张切片。选择不使用 GT，但抽样患者数/切片数会影响病例组成。manifest.json 记录确切样本、PET donor、checkpoint SHA256、raw/EMA 标签及完整配置。该子集得分不等于全划分得分。

如果缺少 MedPy，官方 HD95 模式会明确报错；安装训练环境所需依赖后再跑。也可显式加 --hd95_mode surface 做不同口径的诊断，但不能和原日志 HD95 混比。原有 val=test 的设定会被记录，报告不会把它当作独立测试验证。

## 第二轮：检查能否拟合少量训练样本

~~~bash
python scripts/diagnose_fusion_plateau.py \
  --baseline_checkpoint checkpoints_new/MDT/full-add-baseline/ckpt.best_full.pth.tar \
  --fusion_checkpoint checkpoints_new/MDT/full-skff-imagenet-ema-s2023/ckpt.best_full.pth.tar \
  --root /root/autodl-tmp/data/PCLT20K \
  --output_dir diagnostics/skff-vs-add-tinyfit \
  --device cuda --batch_size 2 --num_workers 4 \
  --max_cases 20 --slices_per_case 8 \
  --overfit_steps 200 --overfit_samples 8 --overfit_lr 1e-4
~~~

小样本探针复制融合模型，仅在内存副本上训练全部权重，固定训练样本，无增强，FP32、AdamW、weight decay=0，BN/dropout 保持 eval 状态，不保存或覆盖 checkpoint。每 25 步记录固定样本上的损失、全局/平均/最小单图 Dice。默认门槛是每张样本 Dice 都达到 0.95；这是预设的记忆能力诊断门槛，不是论文评价标准。

能拟合只说明可以记住这些样本。不能拟合也可能与学习率、步数、冻结 BN、样本难度或优化有关，不能单凭失败宣布网络到达容量上限。探针报告正样本数，避免把全背景样本的高分误当病灶学习能力。

## 扩大评估与权重选择

完整所选验证/测试划分：--max_cases 0 --slices_per_case 0。默认 split=val，也可指定 --split test。完整训练评估子集：--train_cases 0 --train_slices_per_case 0。所有组合都会额外运行多种推理压力测试，完整评估耗时明显更长。

默认 --weights recorded 分别使用每个 checkpoint 保存的评估权重标签。也可以统一 --weights raw 或 --weights ema；请求的权重不存在时直接报错。两个模型的实际标签记录在 manifest 中；raw/EMA 不同时，不要把差异完全归因于融合结构。

脚本拒绝输入尺寸、归一化、骨干、解码器通道、HD95 设定或 split 文件名不同/缺失的 checkpoint。其他训练设置仍需结合 manifest 检查，不能把不同训练种子或训练预算的差异当作纯模块效应。

## 检查内容

- 两个独立训练模型：固定阈值 0.5 的全局 Dice、患者平均 Dice、HD95、FP/FN；患者配对重采样区间。
- 改善/损害：按患者、切片统计，记录纠正与破坏的像素。
- PET 敏感性：保留自身网络，分别把编码后的 PET 特征置零、或使用不同患者的 PET；配对关系固定并保存。错配不是严格排列，供体可重复。
- 尺度敏感性：同一融合模型仅在推理时把单尺度或全部融合结果换成 Add，不迁移两模型的参数。
- 特征变化：相对 Add 的原始差异、消除最佳标量缩放后的差异、GT 前景/背景残差强度；SKFF 通道选择权重及熵。GT 使用面积权重映射，不用最近邻丢弃小前景。
- 错误分层：GT 边界带 FP/FN、2D GT 8 连通分量的预测覆盖率；面积 <=256 输入像素视为小分量，覆盖 >=10% 视为检测到。此定义不是标准 3D 病灶匹配。
- 阈值敏感性：同一集合的事后扫描，用来发现校准影响，不作为新的独立测试成绩。
- 训练状态：关闭增强、eval 模式下的训练子集评估；原日志的最佳轮、末尾窗口变化、非有限梯度条目与跳过更新比例。
- 可选小样本拟合探针。

缓存特征预测会在首批和原始 forward 结果做一致性断言。评估不使用 AMP，避免把混合精度异常掺入主对照；FP32 仍出现非有限输出会中止并记录 error.json。

## 输出文件

- report.md：中文判读说明与主要指标、下降最多的患者。
- summary.json：完整汇总、患者 bootstrap、oracle、训练子集及可选探针。
- per_case.csv / per_slice.csv：患者/切片对照。
- per_lesion.csv：各模式的 2D GT 连通分量覆盖。
- feature_scales.csv：特征变化和可用的 SKFF 选择统计。
- threshold_sweep.csv：诊断性阈值扫描。
- training_logs.json：原训练日志摘要。
- manifest.json：所有输入、权重标签、协议和样本映射。
- tiny_fit.json：仅在启用探针时产生。

输出目录必须不存在；原 checkpoint 和训练日志只读。

## 能回答什么，不能回答什么

它能够显示“两个模型是否在同一批患者上稳定不同”“改进是否被新增 FP/FN 抵消”“当前预测对 PET/各尺度是否敏感”“无增强训练拟合与评估之间是否有差距”“固定小样本是否能够拟合”。

**理论模型上限无法由本脚本确认。** 推理时零特征、跨患者 PET 和临时 Add 都改变分布，只是压力测试，不能代替重训消融或独立训练的 CT-only。GT pixel oracle 只衡量两个固定二值预测之间的互补空间，不能解释为可达精度或模型上限。患者 bootstrap 不覆盖训练随机种子不确定性。

若两模型在同一区域犯错，也可能与共同训练偏差、损失、数据标注或表达方式有关，并不证明输入中已经没有可学信息。根据诊断结果决定后续受控实验，而不是根据单条验证曲线宣告上限。
