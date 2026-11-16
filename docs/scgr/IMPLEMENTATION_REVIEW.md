# SCGR 设计与实现独立核验

- 审查范围：FINAL_PROPOSAL.md、EXPERIMENT_PLAN.md、variants/scgr、三份配置、test_scgr.py、复用的训练引擎。requested model: gpt-6.1-terra, medium；actual reported identity: unknown；review_independence: same-family fresh context；acceptance_status: provisional。
- 结论：设计适合作为有条件的探索性 pilot，前提是最终代码版本通过 GPU 资源/AMP 预检与计划中的回归检查；不据此认定论文贡献、语义有效性或实际实验成功。用户授权的 V→T→S 是检验顺序，不是三组结果即可证明创新的保证。

## 核对证据

1. `module.py` 的 value 仅接收 local 与上采样 context；门控从同尺度共享 support 头的两路 sigmoid 输出计算，差值只用于 capture，未进入内容或门控；上下文的评分先在原网格 sigmoid 再插值。分数是条件，不是可靠性或错误概率。
2. `losses.py` 对每尺度的两个原网格 logit 分别用 area 缩放的软 GT 做 BCE，八项等权平均；仅 S 将其加到主任务。V/T 会计算诊断 BCE，但它不进入 loss 的计算图。空前景仍作为背景监督。
3. 三组结构一致；V/T 唯一有效计算差别是零/固定向量条件，S 在 T 上增加辅助评分损失。`fork_rng` 固定独立 refiner 种子，避免改动公共 RNG；对应 CPU 测试覆盖共同状态和初始恒等。模型从通用预训练编码器、随机任务模块开始；不得以这一代码审查取代实际资源检查。
4. `Model.forward(ct, pet)` 不接收 GT，`inference.py` 从 checkpoint 恢复缓存向量与权重，无需文本塔。常量文本可被训练参数吸收，当前实验不能将结果归因于医学语义。
5. 引擎每轮调用一次完整融合验证；仅在融合 Dice 改进时原子写 `best.pt`，保存选择时的 raw/EMA 权重与指标，不写完整续训状态。训练和验证 batch 检查 STOP；停止整个队列、异常处理及预检还需要结合实际队列入口检查。

## 已修复：空前景误差分区

`metrics.py` 对空 GT 的 `distance_transform_edt(~y)` 直接计距离。全 True 数组没有真实 GT 参考点，SciPy 的边界行为会将该切片的 FP 虚假地分配至距 GT 3px 内、3–12px 或 >12px；这污染错误位置分析，但不影响训练目标、Dice 或 HD95。现已将空 GT 的 FP 记为 fp_empty_gt，从三个距离分区排除；测试验证 104 个 FP 单列且三项距离分区均为零。距离分区 fraction 的分母仍为全部 FP，包含空 GT 的 FP，解读时需明确这一口径；不能将这些 FP 描述为距 GT 某个距离。

## 科学边界与运行门

- M1 相加造成错误、深尺度能修复错误均未得到因果证实；模块可能只是多参数上下文残差或抑制幅度。须同时看 FN 与 FP，匹配容量残差、去门控和等范数非语义常量对照仍缺。
- 开发集与单种子只能用于探索；独立患者测试、种子重复、匹配对照未完成。固定文本无独立语义证明。
- 首次零初始化使内部若干梯度在第一步为零；按计划观察至少三次成功更新，不把首步零梯度当成错误，也不能用梯度非零替代任务收益。
- 正式启动前仍需验证实际 512/batch16、AMP overflow 跳步、EMA/optimizer 显存、checkpoint 重载，以及队列 STOP 和异常不续跑。三组每次从头训练，所选指标同一权重汇报。

## CPU 检查

`bash scripts/myenv.sh -m pytest tests/test_scgr.py -q`：5 passed，21.30 秒。现有测试覆盖初始状态、native-grid 软标签、空掩码的评分损失、共享头、梯度和非空 GT 区域诊断；该次运行先于空 GT 距离分区修复；修复后由主任务报告测试覆盖。CPU 测试不等同于 GPU 预检或实验有效性。

## 补验：2026-11-16

- 实查空 GT 修复与测试：104 个预测 FP 单列为 fp_empty_gt，三个有定义的距离分区为零，原缺陷已修正。FP 距离比例使用全部 FP 为分母。
- `run_scgr.py` 只依次启动 V/T/S；campaign STOP 会传播到当前训练输出并阻止后继，子进程失败或未完整结算标 needs_analysis，成功后不启动额外实验。队列测试使用模拟子进程，不能替代实际 GPU 队列演练。推理模型只加载权重和缓存向量；训练依旧单次完整验证、只保存最佳权重。
- 主任务报告修复后的 CPU 测试 11 项及队列测试 3 项通过。第一轮真实 GPU 预检三组各三次 batch16/512 AMP 成功，无溢出，初值/RNG/初始恒等及 checkpoint 推理一致，每组参数 43,822,873，峰值 allocated 约 14.95 GiB。该预检早于 metrics 修复，最终源哈希对应的 GPU 预检仍需完成。
- 队列最终比较此前缺 fp_outer3_12 与 fp_empty_gt，现已在比较列表和队列测试数据中补齐；各组原始 counts 均保存。
- 通过最终预检及核对后可进行探索性 pilot；三组仍缺容量匹配、去门控、非语义向量对照及独立测试和重复，论文机制、创新与文本语义证据不足。

## 最终增量复核

- refiner 构造在 fork_rng 中只重设 CPU 默认生成器，避免此前 torch.manual_seed 同时改变 CUDA RNG；GPU 预检现有逐元素 CUDA RNG 断言。最终版本预检须以当前源码哈希通过。
- 每组最佳权重选定后，仅对四张预先固定历史样例执行单次推理并画图；GT 只参与绘图和统计，未传入模型。使用同一 GPU 锁；导图失败则停止后继。其模拟队列测试不等于真实导图的成功证明，主任务正在进行导图 smoke。
- 目前审阅的训练、推理和队列路径未发现其他明确运行阻碍；此结论以最终 CPU 回归、真实导图 smoke、当前源码 GPU 预检成功和干净提交为条件。该条件不代表预测效果或论文主张成立。
