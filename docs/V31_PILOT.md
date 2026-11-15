# V3.1完整候选试跑

本次明确授权：等待现有三组重跑全部自然完成，然后运行GPU预检、第二阶段联合训练、第三阶段冻结纠错。当前三组代码与配置不变。没有安排匹配继续训练对照、E2—E5消融或自动调参，因此本轮不能独立证明专家、门控或保护各自有效。

## 模型与初始化

- 原ConvNeXtV2-Nano＋MiT-B1，原双路对齐、M1和GN融合解码器。
- 第二阶段加载历史78.8112% M1第5轮EMA，严格匹配所有共享参数；新CT/PET GN解码器独立随机初始化，独立RNG种子2133。
- CT专家：M1前CT特征→有符号距离，tanh限制范围。PET专家：M1前PET特征→候选Logit。三路联合训练。
- 第三阶段加载第二阶段Fusion Dice最佳那一轮的完整选中权重，冻结整个网络与所有运行统计。只训练32通道专家联合表示、错误门控和零末层初始化的残差分支。
- 纠错主要在128网格计算，Logit上采样回512。推理保留三个解码器。门控输入专家表示detach，残差输入保持梯度。训练标签不进入推理接口。

第二阶段使用原分割损失＋0.1 PET BCE/Dice＋0.1加权CT SmoothL1。PET目标为当前增强GT圆盘扩张3像素。CT距离按16像素截断归一化，距边界3像素以内权重5、其余1。空/满GT显式处理。

第三阶段使用原分割损失＋0.1平衡错误BCE＋0.1保护损失。错误掩码由冻结网络对当前增强图像重新预测产生，阈值严格沿用现有CIPA评估的Sigmoid>0.5。保护正确前景及GT外3像素正确背景，两类每图独立归一化，空类跳过。不依赖学习中的共识生成标签。

共享设置：myenv、seed2023、512、batch16、AdamW lr8e-5/decay1e-4、原调度与EMA、最多60轮、连续3轮无提升早停。专家GN解码器可做activation checkpointing以节省显存，不对带BN的编码器/M1做重计算；不改变有效batch或损失。显存不够时停止报告，不静默调整训练设定。

## 运行

```bash
# 各阶段也可单独运行；机器路径请按实际填写
bash scripts/myenv.sh -m variants.v31.train --phase joint --run v31_joint --init /path/to/historical_M1_best.pt --gpu 0
bash scripts/myenv.sh -m variants.v31.train --phase correction --run v31_correct --init /path/to/v31_joint/best.pt --gpu 0

# 等待既有三组全部完成后，依次预检和运行两个阶段
bash scripts/myenv.sh scripts/run_v31.py --wait-for /path/to/three_run_campaign --init /path/to/historical_M1_best.pt --run v31_full_20261115 --gpu 0

# 最终模型跨机器验证，需要同时传输joint和correction的best.pt
bash scripts/myenv.sh scripts/evaluate.py --variant v31 --checkpoint /path/to/correction/best.pt --base-checkpoint /path/to/joint/best.pt --output /path/to/evaluation.json
```

使用独立的固定Git提交worktree执行；每阶段记录代码、环境、配置和输入权重SHA。第二阶段只保存完整最佳选中模型；第三阶段只保存纠错器最佳权重及基座路径/SHA，避免重复复制基座。不保存last.pt、优化器、Scaler、调度器或RNG续训包。

## 验证与监控

- 自动GPU预检在前一队列完成后才执行：真实512样本，共享M1初始输出等价、两个专家梯度、零残差等价、冻结基座更新前后校验。
- 每轮仅一次完整验证。第三阶段用同次前向得到的原预测计算纠错前后指标，不重复运行编码器。
- 记录主/辅助损失、有效区域数量、逐尺度特征统计、AMP安全梯度、专家候选精确率/召回率/面积、CT边界距离误差及容差3边界F1。
- 记录2048桶近似门控PR-AUC（不是精确全像素排序AUC）、32桶门控分布、残差和实际改变量、前景/背景/近边界/远背景的修复与破坏。
- 记录总/可训练参数及验证batch下CUDA前向毫秒/图；不是batch1部署延迟，完整FLOPs核算仍需单独完成。
- 训练期间不做LLM逐轮轮询。前一队列失败或被停止则取消后续；本流水线STOP文件取消当前阶段和后续阶段。失败不会静默调参重启。

本轮目标是完成完整候选的有效性试跑。没有预先保证达到某个指标。阶段二即便不超过历史基座，也会按本轮授权试跑阶段三；结论须分别报告继续训练后的基座变化及冻结纠错的真实增量。
