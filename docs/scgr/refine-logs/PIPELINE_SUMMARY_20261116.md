# 流程摘要
问题：M1融合后的错误位置细化。候选：分离视觉内容与评分接纳的四尺度后置残差，文本只调制通道。
状态：REVISE_FOR_PAPER / READY_FOR_PILOT；不保证顶刊创新或性能。
运行：V(无文本无score损失) -> T(只加文本) -> S(只加score损失)，三组从头。
证据：方案FINAL_PROPOSAL.md；计划EXPERIMENT_PLAN.md；审查REVIEW_SUMMARY.md；执行EXPERIMENT_TRACKER.md。
主要风险：粗分数与错误无关、额外监督干扰、语义常量可被吸收、参数容量混杂、开发集反复使用。
