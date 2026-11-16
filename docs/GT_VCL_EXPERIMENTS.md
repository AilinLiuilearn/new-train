# GT-VCL：E1 → E2 → E4 固定实验

用户本次明确授权实现并顺序运行三组。原M1、双编码器、原对齐与GN融合解码器保持不变，不恢复已放弃的LPI-BI，不使用BCSI前置模块。三组都是独立从头训练的受控配方，不是加载前一组best继续训练。

|组|三路BCE+Dice，各权重1|CT参照校准|文本区域约束|
|---|---|---|---|
|E1|Full、CT、PET|关闭|关闭|
|E2|相同|权重0.1，容差0.05|关闭|
|E4|相同|相同|权重0.1，间隔0.2|

## 网络与初始化

原ConvNeXtV2-Nano / MiT-B1与原双路对齐输出四尺度。原M1和GN融合解码器产生Full预测。新增一套GN解码器decoder_sep，分别处理原始CT和PET对齐特征，两次调用共享全部参数；与decoder_full同构但参数、初始化及优化独立。没有教师、冻结视觉分支或额外纠错阶段。

仅编码器加载通用预训练，原任务模块从头；辅助解码器私有随机种子2133、文本投影私有种子2183，不消耗公共数据随机流。三组公共参数初值和模型构造后的CPU RNG严格一致；初始融合输出与同权重M1严格一致。初始化恒等不保证训练轨迹或收益。

## 校准定义

CT当前输出与GT确定正确像素，使用停止梯度的真实类别置信度作为参考及权重。对Full的真实类别置信度下降超过0.05进行hinge惩罚；仅该项不向CT预测反向传播，Full损失和CT自身监督仍正常更新编码器。

按GT前景与背景分别以权重之和归一化，再固定各乘0.5合并；空区域该项为0。没有新增置信阈值、温度或额外教师。该项是GT筛选的在线CT参照保持，不是概率可靠性校准，也不证明PET的因果贡献。

## 文本与精确区域计算

- CT：`A CT image showing the structure, shape, and boundary of a lung tumor.`
- PET：`A PET image showing the location and metabolic activity of a lung tumor.`

`scripts/prepare_gt_vcl_text.py`只在CPU离线运行。严格加载本地BioMedCLIP训练权重中的200个text张量（包括官方MLP全局投影），使用匹配30522词表配置和原tokenizer。处理旧checkpoint持久化position_ids与当前Transformers的差异，不随机补参数。重新计算旧缓存最大误差为0；新语句向量保存为`resources/gt_vcl_text.json`，不修改旧角色缓存。文字编码器不进入训练或推理。

E4仅在共享单模态解码器最后一级64通道特征上做各自独立64→512线性投影；逐像素归一化，使用AreaResize GT占比对前景/背景分别池化，再归一化区域向量，与对应冻结文本计算余弦，要求前景比分背景高0.2。每个模态按有效样本平均后两路相加；无前景或无背景样本跳过。E4不是文本语义作用已成立的证据，随机向量和可学习锚点对照未授权本轮运行。

原共享GN解码器使用无随机重计算的激活checkpoint；语义投影按1024个空间位置分块checkpoint，保持“先逐像素投影归一化、再GT池化”的原公式，不改成先池化再投影。它们只改变内存使用，不改变网络数学。编码器、对齐和M1中的BN不在checkpoint内，不重复更新BN统计。

## 验证、推理与保存

每轮一次验证遍历，编码器各运行一次，同批产生三路输出。只有Full的原CIPA Dice选best；HD95、Precision、Sensitivity从该轮一起记录。CT/PET记录同批Dice/Precision/Sensitivity与前景/背景分开的八种正确性组合；Full记录3像素容差微平均Boundary F1与固定≤256像素二维小分量覆盖。后者不是临床体积/患者级病灶检测。

验证objective显式接收当前选中的raw或EMA模型；E4不会错误使用raw投影计算EMA模型的文本loss。部署加载器严格移除decoder_sep和semantic，只有编码器→M1→融合解码器；导出后的融合输出已与所选模型核对严格一致。GT不进入模型forward或部署接口。

沿用myenv、seed2023、512、batch16、AdamW8e-5/1e-4、原增强/调度/EMA、最多60轮、连续3轮Full Dice无提升早停；best-only，无optimizer/RNG/last断点包。每5轮或用户询问、异常时检查，不实时轮询。

## 代码、来源和证据边界

`variants/gt_vcl/`为独立候选；编码器、M1、GN解码器、原BCE+Dice、调度、EMA与CIPA评估直接复用。GN解码器仅增加可选return_features，不改变默认运算。训练engine增加可选model-aware objective，使EMA与训练模块绑定正确；旧调用不受影响。

参考PNDC的正负影响分析问题意识（TMI2025，PMID40031068）及TIRNet的GT区域语义监督思想（MICCAI2026，Paper1662）；本文共享解码器、概率间隔约束和简化区域loss是本候选自己的适配，不称原论文完整复现。BioMedCLIP调用已安装官方OpenCLIP文本塔代码，离线导出严格使用实际训练权重。

现有同协议M1参考78.1551%/16.4411；旧78.8112%与78.4970%保留为不同历史实验，不能混成同协议基线。所有结果仍是单seed、反复使用开发集的探索结果。

## 预检与顺序启动

`docs/GT_VCL_GPU_PREFLIGHT.json`记录三组真实512/batch16的三个有限AMP更新、完整初始化核对、剪裁前后有限梯度、导出推理一致性。临时更新全部丢弃。三组各有一次初始AMP溢出后自动降低scale，随后完成三次有限更新；这不是长期稳定性或性能结论。初始CT尚未找对前景时，校准前景项可为空；合成GT梯度测试另行覆盖这一分支。

```bash
bash scripts/myenv.sh scripts/run_gt_vcl.py --campaign gt_vcl_joint_cal_text_20261116 --gpu 0
```

提交并推送协同分支后，在固定提交的独立worktree顺序运行。E1自然完成后运行E2，之后E4，均不读前组best。任一异常或campaign STOP取消后续；三组结束写比较表，不自动调参、加E3或其他消融。完整实验结束不等于研究主张成立或指标已达标。

## 核验结果记录

44项测试通过，覆盖前景/背景校准平衡与CT参考stop-gradient、空掩码、文本分块/稠密计算及梯度等价、共享GN重计算等价、公共初始化/RNG一致、辅助解码器共享与融合参数独立、GT分支梯度隔离、EMA模型感知loss和部署裁剪预测一致。独立上下文代码审查未发现阻断问题；不是创新或性能验收。

预检参数量E1/E2为46,977,746，E4为47,044,306。E1/E2峰值分配约15.33GiB，E4约15.52GiB；包含优化器、EMA和临时部署模型。三组临时训练均丢弃，不用于正式启动。
