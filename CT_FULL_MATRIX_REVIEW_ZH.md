# CT条件 Key–Value 校正矩阵补偿

本实验独立于旧版解码器内FG/BG原型。原文件不改动，不能加载旧实验权重续训。

## 张量与路径

- CT ConvNeXtV2-Nano、PET MiT-B1及原有通道对齐和GN解码器保持一致。
- S1..S4对齐通道64/128/320/512，512输入对应128²/64²/32²/16²。
- Full: CT+PET相加后直接解码，不读取矩阵；Missing: 不编码PET，四尺度干净CT各自查询、校正后解码。
- 每尺度前景/背景两库，每库默认4个槽位；绝不共享最深尺度聚类成员。
- 训练GT仅用于候选类别；推理forward不接收GT，不更新库。

## 参考来源与我们的扩展

参考 https://arxiv.org/abs/2603.04887 第3.5节与官方
https://github.com/ccarliu/FedMEPD ，核对参考版本620f820b588bd0d3a51994901065a6b8e26b0d41：
`utils/fl_utils.py` 的 `getClsPrototypes`、`cluster_Fs`、`EMA_Fs`。
借鉴GT类别组织、多锚点与EMA维护原则。FedMEPD原本保存融合特征锚点；本实验的独立像素配对FIFO、
球面CT聚类、Key–矩阵Value和正则化回归是我们自己的适配，不是原文复现，也不继承SAR-Net的Gram/重建。

## 候选与矩阵

1. Full训练病例提供同位置原始CT/融合配对。GT面积降采样得到占比q；前景q>0且权重q；背景仅q=0。
   占比避免小病灶直接丢失，但混合单元仍是混合特征，不能当纯病灶。
2. 每图、每尺度、每类最多8对，均匀无放回；不足不重复。私有种子由成功步数/尺度确定，不扰动编码器RNG。
3. 类别FIFO各256对（CT、Full、权重成对），未补偿CT的单位方向用于球面K-means，独立尺度独立运行。
4. 每簇使用原始幅值计算Scc=sum(w*c*cT)/sum(w)、Sfc=sum(w*f*cT)/sum(w)。
5. lambda=eta*(trace(Scc)/C+1e-6)，默认eta=.01。
   T=(Sfc+lambda*I)*(Scc+lambda*I)^(-1)，实际使用FP32线性求解，禁止显式逆和AMP低精度拟合。
   目标为配对拟合误差+lambda*||T-I||²；不是额外网络损失，不调用backward。
6. 初次直接初始化；以后CT Key按余弦距离一对一匹配新旧槽位，Key与T一起EMA(.99)，Key重新单位化。
   支持不足/空簇/求解失败不覆盖旧槽位；数值异常不标记有效。
7. 前100成功步收集但不启用；其后无效库随支持情况初始化，有效库每20成功步刷新。
   默认最少24个候选、每簇至少2个有效候选。此为数值有效条件，绝非统计可靠保证；尤其C=512时256对依然依赖正则。
8. 默认关闭离群过滤。显式打开时按每簇CT角距离保留最近90%（至少满足最小支持），原始配对一起过滤。

## 读取、门控与监督

Missing每位置单位CT与同尺度有效Key做cosine/temperature(.5)和softmax；没有可训练Q/K投影。
对原始CT计算read=sum_j attention_j*(T_j*c)，实现不生成每像素C×C动态矩阵。
门控为Conv1x1([CT,read])→sigmoid，单通道，初始采用程度.1。
output=CT+gate*(read-CT)。这里是插值运算，不是将Full-CT差值存入库。
无有效库/关闭补偿时严格旁路，Full始终不变。额外可训练参数只有4个门控，共2052个。

只保留最终损失：0.5*(BCE+Dice)_Full+0.5*(BCE+Dice)_Missing。
没有深度监督、CT参考辅助损失、收益筛选、PET重建损失。原始两个深度监督开关开启会明确报错。
每次forward只准备候选，优化成功后才提交；失败步不改池/矩阵/计数。模型EMA启用时矩阵缓冲区硬拷贝，避免双EMA。
断点保存全部配对池、Key、T、支持计数、优化器/调度器/AMP/RNG/DataLoader状态；恢复拒绝改变矩阵策略。

## 训练与消融

```bash
bash scripts/train_ct_full_matrix.sh
bash scripts/train_ct_full_matrix.sh --hash add-ctfull-kv-matrix-off-s2023 --matrix_enabled false
```

原训练数据、预训练路径、60epoch、batch16、mixed、seed2023、AdamW8e-5等保持脚本设置。
默认模型EMA关闭。候选/矩阵更新策略可用`--matrix_*`参数覆盖；其他设置也可通过脚本尾部参数覆盖。

```bash
python eval_ct_full_matrix_add.py --checkpoint /path/ckpt.best_joint.pth.tar --read_mode normal
python eval_ct_full_matrix_add.py --checkpoint /path/ckpt.best_joint.pth.tar --read_mode off
python eval_ct_full_matrix_add.py --checkpoint /path/ckpt.best_joint.pth.tar --read_mode identity
python eval_ct_full_matrix_add.py --checkpoint /path/ckpt.best_joint.pth.tar --read_mode shuffle
```

默认评估缺失率0、1。各模式输出到独立matrix_eval_*目录，不覆盖原始结果。
shuffle只循环置换有效Value，Key保持不变；identity是恒等矩阵的精确旁路；Full在所有模式相同。

## 诊断与限制

- train_log.csv：Full/Missing/Joint与两路HD95、有效类别库数、非有限梯度与跳过更新。
- matrix_banks.csv：候选量、刷新失败/状态、采样门控、校正/变换RMS与注意力类别占比。
- matrix_slots.csv：每槽支持、Key/矩阵范数、拟合误差、恒等误差、正则量、更新漂移、读取占比。
- 所有read统计仅对应采样训练batch，不是验证集或整轮平均；同尺度联合门控重复出现在FG/BG行，不相加。
- fit_mse_in_pool是本次新拟合矩阵在本池中的误差（EMA之前），不是泛化误差，不可冒充独立验证。
- 当前版本未加入独立留出配对误差验证；需结合单独诊断数据验证矩阵迁移。
- 类别正确/拟合良好不保证Full更有用。矩阵无法恢复CT中不可推断的个体PET信息。
- 单位方向聚类可能拒绝方向单一的候选；失败状态可见。门控可能旁路，必须做开关/恒等/打乱及从头训练关闭的对照。
- 保存矩阵和配对池约16MB FP32（不含临时拟合/读取张量），计算与CUDA峰值显存仍需目标3090实测。

## 验证命令

```bash
python -m pytest tests/test_ct_full_matrix.py -q
PYTHONPATH=. python tests/smoke_ct_full_matrix.py --size 64 --backward
PYTHONPATH=. python tests/smoke_ct_full_matrix.py --size 512
```

冒烟测试使用随机初始化真实编码器与合成输入，不代表患者数据训练有效或收敛。

本次验证：新合同11项通过，旧基线/旧原型回归111项通过，共122项；真实编码器64输入
前向/反向/优化提交通过，512输入有效矩阵前向、Full旁路和Missing的PET隔离通过。
训练脚本bash语法、实际入口--help、109个参数token解析通过。未运行真实患者数据或CUDA训练；
RTX3090的batch16峰值显存、速度、收敛及性能收益必须实测。

## 同一 checkpoint 开启／关闭补偿诊断

```bash
python -u eval_ct_full_matrix_add.py \
  --checkpoint checkpoints_new/MDT/add-ctfull-kv-matrix-k4-finalonly-s2023/ckpt.best_joint.pth.tar \
  --compare_on_off true --missing_rates 0 1 \
  --output_dir checkpoints_new/MDT/add-ctfull-kv-matrix-k4-finalonly-s2023/diagnosis_on_off
```

`normal/`、`off/`分别保存相同病例分配下的评估结果，`comparison.json`汇总开启减关闭的 Dice/HD95 差值。
`missing_region_correction.csv`另一次全 Missing、FP32 诊断遍历记录四尺度前景／背景校正强度。
GT 仅在 forward hook 中用于统计；面积下采样占比 q 对前景加权、1-q 对背景加权，混合单元按比例计入两类。
跨整个测试集累积通道平均平方能量后计算 RMS 比值，不平均逐批比值。无该类时输出空值。
该比值是特征变化幅度，不代表分割收益；与训练日志的采样批次统计不可直接等同。
不会更新网络、候选缓存或矩阵库。Full 开关结果应一致，Missing Dice 差值正向、HD95 差值负向表示改善。
