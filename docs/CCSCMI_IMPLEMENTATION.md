# CC-SCMI V1 完整模型试验

结构：异构编码器→原双路对齐→CC-SCMI（四尺度）→原M1→原GN单解码器。所有视觉任务参数更新；仅编码器加载通用预训练，任务模块从头。没有使用旧分割权重、V3.1专家或纠错器。

本次用户明确要求只运行两组：先无文本，再有文本。两组均使用逐维空间权重。未自动增加M1或共享权重对照，因此结果不能单独证明逐维机制，也不能独立证明医学文本语义有效。

## 首版固定设置

- 四尺度通道64/128/320/512，各选50%，保留原通道编号；硬前向、Sigmoid软梯度，温度1。
- 每尺度潜在宽度32，8×8格子池化Query；两个模态各64个有界Offset位置，CT在前/PET在后拼接。
- 两方向独立Q/K/V，LayerNorm后的公共空间分数＋tanh(beta)逐维乘积；beta初值0，评分/Softmax/聚合明确FP32。
- 原始跳连不缩放；双向输出投影置零，接收端掩码回写，再进入原M1。
- 文本版仅在评分器加入固定BioMedCLIP缓存向量与当前GAP描述的条件分数；条件末层零初始化。共同视觉参数和模型构造后的RNG与无文本版一致。
- seed2023、512、batch16、AdamW8e-5/1e-4、原BCE+Dice/调度/EMA、最多60轮、早停3、best-only。每轮完整模型验证一次，不增加单向或模块关闭验证。

## 源码来源与自定义部分

1. KTB：Cai等，CVPR2025，https://github.com/imcjx/KTB ，固定版本75370dd584fb9d8571bced1aabade198dfbdc80f。
   - `third_party/ktb_sampling.py`适配其`semseg/models/backbones/swin.py`中的双网格grid_sample与拼接模式。
   - 保留Apache-2.0完整许可`third_party/KTB_LICENSE.txt`。改为统一(x,y)、无分组输入、格子中心参考点、align_corners=False和border填充。
   - 不是DSCF完整复现：本模块使用纯跨模态双向K/V，而非原文模态加权混合K/V；不使用MAPA/PEFT或MMST训练。
2. BCSI：Huang等，2026，https://arxiv.org/abs/2601.05855 ，作者代码https://github.com/taozh2017/BCSI ，参考版本3bbba85042cfc2bbe2a6f5ffedc5356c8946803d。
   - 借鉴选择通道再交互的思路，不复制其模型代码或特征队列。原文是共享编码器的有标签/无标签流；本实验是配对异构PET/CT。
3. 自定义：稳定原编号ST掩码、边界内Offset、逐维评分、双输出残差接口、文本条件评分、诊断和对照接口。可运行不代表这些构件已验证有创新或性能收益。

## 检查

CPU测试包括采样坐标、零初始化恒等、掩码数量和未选中通道保留、六次真实参数更新后的路由/Offset/beta任务梯度、共同参数及随机流一致性。

`scripts/ccscmi_preflight.py`对真实512图像、batch16执行完整模型检查：严格共同参数匹配、初始输出与原M1完全一致、两版AMP前后向及JSON诊断正常。预检参数只用于临时检查，之后全部丢弃；正式训练重新初始化，不把验证样本的临时梯度用于训练模型。

预检所见峰值约15.4GiB分配、16.7GiB保留（不含训练EMA复制的额外常驻量）。这不是性能评估，也不证明目标指标已达到。

## 运行

```bash
bash scripts/myenv.sh -m variants.ccscmi.train --run ccscmi_visual_s2023 --gpu 0
bash scripts/myenv.sh -m variants.ccscmi.train --text --run ccscmi_text_s2023 --gpu 0

# 本机固定提交上依次运行两组；GPU预检源文件SHA必须匹配
bash scripts/myenv.sh scripts/run_ccscmi.py --campaign ccscmi_full_20261115 --gpu 0
```

队列输出`STATUS.json`、`RESULTS.md`，每组独立best.pt与日志。无文本组自然完成后按用户授权接有文本组，无论前组是否超过历史分数，均如实报告。遇到异常停止等待分析；STOP取消当前组和后续；两组结束不自动调参或安排其他试验。

记录每尺度特征与实际增量、注意力跨维差异/熵、beta、Offset与坐标、选中和未选中评分梯度、文本条件梯度，以及原训练/验证指标。不得以可视化或注意力差异代替分割收益。
