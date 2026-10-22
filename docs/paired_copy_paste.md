# 双状态引导的配对上下文 Copy-Paste：首轮及有界条件迭代

这是新的Copy-Paste实验，区别于此前只改变裁剪位置/区域概率的crop-v1/v2。
当前分支 `exp/ct-pretrain-dual-decoder-add-lr8e5`，HEAD `3c1c5d4`。
开始前已记录工作区状态、diff及相关源文件快照于
`checkpoints_new/AUGMENTATION/copypaste-v1-review/before/`，未覆盖已有未提交改动。

## 训练协议

保持35,907,538参数的原双路径模型、BN对齐/GN解码器、四尺度相加和两路等权BCE+Dice。
每个合成样本的CT/PET同时进入Full，完全相同的合成CT进入Missing；两路监督同一合成GT。
从原通用编码器预训练重新初始化，两个解码器独立，不继承任何baseline任务checkpoint。
原best.pt只用于已有的train-only离线地图，本轮复用同一份地图，没有增加teacher训练。

启动器仍从地图manifest提取源baseline配置并验证通用预训练SHA256：512²、batch16、lr8e-5、
seed2023、60轮上限及对应原调度、原AMP/EMA/增强/归一化/优化器/梯度裁剪/drop_last。
按用户最新要求，所有新run连续5轮Joint无严格提升早停，只保存best.pt，不保存last.pt。
train_original/test/test及val=test警告保留；这些是开发结果，不是独立最终测试。

## 供体与标签

只扫描训练GT和已有差异地图。以真实2D前景连通区域为单位：

- D=1且GT=1：Missing漏检、Full正确的目标部分。
- D=2且GT=1：Full漏检、Missing正确的目标部分。
- 背景误报不作为肿瘤供体；D=3只进入普通总体错误对照。

首轮overlap策略：目标含任意D=1像素即进入类别1供体池，含D=2即进入类别2池；
两个池都有时各半，再均匀选择完整目标。两池可以重叠，这一点明确记录，不当成两类独立样本。
索引共13,501个可用目标，hard=12,640，类别1=3,874，类别2=3,646，
交集2,084，占差异供体并集5,436的38.34%。

取目标bbox，周围上下文为max(4,round(0.1*最大边长))；最大patch边长不超过图像边长0.4。
若上下文切到另一个GT连通目标则拒绝该候选，避免复制另一个目标的碎片。
供体和受体来自不同训练case。复制同一个供体的CT/PET/GT，使用完全相同的矩形、平移、
尺度1.0；无独立PET变换、无跨模态错配、无pseudo-label、无强度混合。
矩形内受体CT/PET/GT整体替换，外部不变，不覆盖受体原有GT前景。

Copy-Paste在原CIPA几何增强之前执行，之后仍走原旋转、翻转、随机裁剪和归一化，
包括原GT插值与最终二值化。它不与差异引导裁剪叠加，避免把两种改动混在一起。
每个原训练样本仍遍历一次，batch/optimizer steps不增加；粘贴会增加目标曝光，三组需匹配并记录。

## 落点规则及限制

现有PNG没有器官标签/HU/体素间距，不能可靠进行器官配准。首版使用透明的保守启发式：

1. CT灰度>8的最大连通区域填洞，作为粗人体掩码，而非器官分割。
2. 将供体目标中心映射到受体人体bbox相对坐标，位置抖动±0.08人体bbox边长。
3. patch至少95%位于粗人体区域，不能覆盖受体GT前景，不能越界。
4. patch外圈CT灰度均值与标准差各自相差不超过32（8-bit灰度单位）。
5. 最多尝试12个位置，失败回退原样本；不悄悄放宽约束。

该约束不能保证解剖/生理合理；人工拼图中存在可见的局部拼接边缘，可能形成捷径。
不能把“配对与标签正确”当成“解剖合理已验证”。单图、局部放大及独立真实图像验证必须共同检查。
本轮不引入器官网络、形变配准或复杂融合；若结果不支持，不能用堆模块掩盖。

## 匹配对照及预先限定的迭代

首轮三组都采用相同prob=0.3、上下文、位置规则和原始训练预算：

|组|供体池|
|---|---|
|dual_state|类别1/2各半的差异目标|
|hard|含任意D!=0前景错误的完整目标|
|random|任意可用完整GT目标|

hard/random先依据相同的双状态参考目标面积分布抽floor(log2(area))区间，再在各自池内抽目标，
尽量匹配目标大小。区间匹配不等于逐像素面积完全一致，成功落点后的分布仍可能不同。
每轮记录尝试/成功/回退、选择与成功类别、面积区间、粘贴前景像素、patch大小及CT背景差异。
不把回退算成粘贴，不依赖worker私有全局计数；donor图像LRU每worker最多8个，完整图不常驻全量内存。

若首轮A3没有同时超过B0及两个对照，只允许一次预先声明的规则变化：
dominance策略按该目标内D=1和D=2的像素数大小，分别归到Missing更弱/Full更弱的池；
相等则不进入差异池。它等价于在GT目标内部比较两路漏检像素差，避免一个目标同时代表两种状态。
使用同一索引、同一地图、同一空间操作，只改目标归组。对应hard对照重新按新参考面积分布匹配。
该条件迭代已做单元测试及512张审查：类别1=2,701，类别2=2,424，池不重叠。

controller最多运行首轮3组＋条件迭代2组，共5个run，单种子，未扫描位置/概率/尺度参数。
每个run完毕自动生成comparison/interpretation；每个版本结束生成同Joint选中epoch的分析JSON/Markdown。
若首轮已有优势则不继续预设第二版；第二版结束后停止，不自动搜索第三版。
结果只能是单种子开发证据，不能据此声称新颖性已证明、因果贡献或SOTA。

## 现场验证

- 训练集17,416张索引耗时179.1秒，全部来源与原地图身份一致，未扫描val/test图像或GT。
- 固定随机512张，强制尝试一次：dual 381/512、hard 384/512、random 387/512成功；
  对应训练prob0.3的估计实际覆盖约22.32%/22.50%/22.68%。
- 成功样本的平均粘贴前景面积分别423.6/406.2/401.2像素，已做面积区间匹配仍有残余差异。
- dominance固定512张：377次成功（训练覆盖估计22.09%），成功类别186/191。
- 保存原受体、原供体、混合CT/PET/GT及边界外扩放大图；程序逐像素验证供体patch/混合patch配对、
  原有GT未覆盖。人工查看小目标0042_012和大目标0570_215等；记录明显接缝风险，不声称临床真实性。
- 真实PNG batch16 CUDA预检，13个样本发生粘贴，一次实际AdamW+EMA更新成功，无AMP跳过；
  CT/Full各16样本，6个训练模块梯度非零，参数量不变。GPU RTX3090，peak allocated
  13,510,347,264字节（约12.58GiB），有效更新1.66秒；这不是完整epoch吞吐。
- 新Copy-Paste测试5项通过（8.53秒），含配对/标签、跨case、位置边界、空候选、prob0无额外RNG、
  文件指纹、身份、worker统计、真实模型训练保存、无bank/PET的Missing评价以及条件归组规则。
- 原回归22项通过，另1项因系统盘写满导致torch.save失败；清理仅本会话pytest临时权重后，
  原失败的通用预训练/恢复测试单独重跑通过（19.02秒）。没有删除真实实验权重或降低断言。

## 命令与输出

```bash
cd /root/autodl-tmp/mkd-main/new-train
source /root/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MAP_DIR=checkpoints_new/AUGMENTATION/dual-state-maps-s2023-v1
INDEX=checkpoints_new/AUGMENTATION/paired-copy-paste-v1/index.json

# 已完成；拒绝覆盖现有索引/审查目录。
python -u scripts/build_paste_index.py --map_dir "$MAP_DIR" --output "$INDEX"
python -u scripts/audit_paired_copy_paste.py --map_dir "$MAP_DIR" --paste_index "$INDEX" \
  --output checkpoints_new/AUGMENTATION/paired-copy-paste-v1/audit-new --mode dual_state
python -m pytest -q tests/test_paired_copy_paste.py
python -u scripts/check_paired_copy_paste.py --map_dir "$MAP_DIR" --paste_index "$INDEX" \
  --output checkpoints_new/AUGMENTATION/paired-copy-paste-v1/preflight-new.json

# 已启动时不要重复运行；队列controller自带有界条件迭代。
python -u scripts/run_copy_paste_iterations.py --map_dir "$MAP_DIR" --paste_index "$INDEX" \
  --output checkpoints_new/AUGMENTATION/copy-paste-iterations-s2023

# 单独启动的等价接口（需另用新输出目录）。
bash scripts/train_joint_aug.sh "$MAP_DIR" checkpoints_new/STAGED/paste-new \
  --crop_guidance none --copy_paste dual_state --paste_index "$INDEX" --early_stopping_patience 5
```

run目录为`checkpoints_new/STAGED/dual-add-joint-{dualstatepaste,hardpaste,randompaste}-cp-s2023-v1`；
条件第二版后缀v2。只写best.pt、原train日志及copy_paste.jsonl，不生成last.pt。
controller状态在`checkpoints_new/AUGMENTATION/copy-paste-iterations-s2023/status.json`，
版本子目录保存各run日志和解释；版本结束会写v1/v2-analysis.json及.md。
关闭copy_paste（默认none）恢复原baseline数值路径，不读donor bank；历史crop实验保持暂停。
