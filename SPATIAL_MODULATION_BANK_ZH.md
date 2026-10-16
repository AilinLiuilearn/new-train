# 解码器前统一空间调制库

新分支 `exp/ct-full-spatial-modulation-bank`，基于可学习矩阵分支；原矩阵模型、任务、训练脚本保持原样。

## 数据流和损失

Full：通道对齐后的干净CT C、PET P 正常相加得到F，F直接进入共享解码器，不读取调制库。
旁路：detach(C)、detach(F)拼接，深度3×3卷积→GELU→1×1降到32通道→GELU→1×1输出2C通道，分成gamma/beta。末层零初始化。
生成的参数为[B,C,H,W]局部调制图，Ctilde=(1+gamma)*detach(C)+beta。
每样本每尺度计算 mean(abs(Ctilde-detach(F)))/(RMS(detach(F))+1e-6)，对Full样本、四尺度平均。
总损失=.5*Full最终BCE+Dice+.5*Missing最终BCE+Dice+mod_loss_weight*Lmod。
不增加分割头/深度监督；旁路损失只更新生成器。bank写入detach的参数后，Missing损失不能直接反传到生成器。

## 所有位置参与、首轮不读

四尺度独立，每尺度一个统一库、默认8槽位，不分前景/背景。
Key是干净CT的3×3平均邻域描述，L2方向用于路由；Value是配对的原始gamma/beta通道向量。
首个成功batch用确定性最远方向初始化临时中心；首轮每个成功batch所有Full位置按当前中心归属，累计key方向和gamma/beta之和与计数，以累计均值更新。
这是在线球面聚类的初始化和统计，不是保存整个epoch后做精确离线K-means；归属会随中心演化，但每个位置均计入一次。首批可能决定初始分组；空槽位保留，不用GT强行补齐。
没有256缓存、随机位置采样或候选FIFO。chunk_size=8192只用于分块计算，不会丢弃位置。所有有限CT位置（包括零向量）参与，异常非有限值报错。
整个epoch1训练和验证均不检索；begin_epoch(2)启用已累计库。第一轮之后每个成功batch按当前Key分配所有位置，对有成员槽位的Key、gamma和beta一起以decay=.99做EMA；Key再归一化。
EMA单位为成功batch，不是每20步；没有成员的槽位保留。EMA不是全部历史样本的严格平均，首轮才是当前在线归属下的累计均值。
梯度非有限导致跳过optimizer step时，整个batch不写库；所有数据更新发生在backward结束之后。
GT只用于分割监督与日志前景占比/拟合误差，不参与聚类、检索、候选筛选。

## Missing

相同3×3 CT描述查询全部8个Key，cosine/temperature(.5)后softmax。
按权重分别读取gamma/beta，在当前Missing病例各位置重建调制图，输出(1+gamma_hat)*CT+beta_hat，再进入解码器。
不调用PET编码器、不需要GT、没有新增门控/RMS限幅。仅空间上下文决定逐位置通道仿射，不搬运其他病例绝对坐标。
验证/测试不更新库；Full开关检索应完全一致。参数、累计和/计数、epoch、optimizer、RNG和库状态均支持严格恢复；不能直接续训旧矩阵checkpoint。

## 运行

`bash scripts/train_spatial_modulation.sh`

默认hash `add-spatial-modulation-unified-k8-lmod010-s2023`，默认额外损失开启、权重0.1（起始实验值，未验证最优）。
关闭额外损失：
`bash scripts/train_spatial_modulation.sh --matrix_mod_loss_enabled false --hash add-spatial-modulation-k8-nomodloss-s2023`
修改权重：
`bash scripts/train_spatial_modulation.sh --matrix_mod_loss_weight 0.05 --hash add-spatial-modulation-k8-lmod005-s2023`
关闭整个补偿分支：
`bash scripts/train_spatial_modulation.sh --matrix_enabled false --hash add-spatial-modulation-off-s2023`
零初始化下关闭额外损失会使生成器没有训练信号，库保持零调制，预期是身份补偿；这正是消融含义。
断点恢复使用原命令加 `--resume <ckpt.last.pth.tar>`，开关/权重等配置必须一致。

## 日志与评估

训练主CSV新增modulation_loss、modulation_initialized_scales；modulation_slots.csv记录全部槽位计数、病灶占比、Key范数、gamma/beta幅值、检索权重、采样校正强度和前景/背景拟合误差。
计数为累计所有写入位置；GT占比是诊断而非类权重；拟合误差为最新训练batch，modulation_loss为本epoch平均。
使用 `eval_spatial_modulation_add.py --checkpoint <best> --compare_on_off true --output_dir <diagnosis> --missing_rates 0 1` 做同checkpoint开关评估及GT区域强度统计。
不调用旧矩阵专用audit脚本分析这个新模型。

设计风险：背景主导、beta退化为特征偏移、库平均后失去个体信息、编码器/调制生成器变化导致知识陈旧；需通过日志及开关对照验证。该实现不宣称实验收益。
