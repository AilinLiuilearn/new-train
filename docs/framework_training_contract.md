# 固定 CT、可缺失 PET：框架与训练协议

状态更新：用户已停止BN对照并要求总结。BN完成9轮验证，第10轮中断，未达到早停条件。按已观察到的Joint最佳结果保留GN作为工程基线；所有训练和自动等待流程已停止。最终模型/训练参数见 `checkpoints_new/AUGMENTATION/framework-finalization-s2023-v1/frozen_framework.md` 和 `frozen_training_config.json`，明确记录提前停止限制；不是BN充分收敛或结构普遍优劣的证明。

本轮仅审查和诊断，所有正式训练及自动迭代继续暂停。框架作为后续实验的固定 baseline；下述诊断不修改其训练实现、损失或参数。

## 问题与部署路径

研究对象是 CT 始终可用、PET 可能整幅缺失的配对肿瘤分割。PET 可用时运行 Full，缺失时显式选择 Missing。当前不覆盖部分 PET 切片缺失、错配、低质量 PET 或 CT 缺失；也没有自动识别缺失状态的模块。

CT: ConvNeXtV2-Nano，PET: MiT-B1。四尺度通道 `(64,128,320,512)`，两路对齐层为 BN，两个独立 GN 解码器。CT 编码及对齐特征同时用于两路；Full 逐尺度 CT+PET 相加后经 Full 解码器；Missing 直接经 CT 解码器。Missing 不调用 PET encoder/align、fusion 或 Full decoder。参数量 35,907,538，模型文件本轮不改。

## 固定训练策略

用户在完成raw/EMA对照后明确选择保留EMA。后续baseline与创新实验默认沿用相同EMA策略，不因单个模块提升小而临时关闭或改系数。额外raw诊断不能偷换正式选模规则。详见 `ema_comparison.md`。

直接从相同通用预训练 CT/PET 编码器开始联合训练；对齐层和解码器沿用现有初始化。两个解码器初始值相同，之后参数独立。不是必须先训练 CT 再训练融合的两阶段方案。

每个配对训练样本都产生 Full 与 Missing 预测，CT 仅编码一次，目标为相同真实 GT。训练中始终存在可部署的 CT-only 监督路径；无需依靠随机将 PET 置零来让这个路径见到缺失情况。PET 训练数据参与 Full 路径，不输入 Missing 路径。

当前没有 Full→Missing 的 logits 蒸馏或原型传递；跨状态联系来自共享 CT 表征的联合优化。PET 可以在训练期通过 Full 损失影响共享 CT 参数，这与 Missing 推理期读取 PET 是不同的机制。

```
L_M = BCE(logits_M, Y) + DiceLoss(logits_M, Y)
L_F = BCE(logits_F, Y) + DiceLoss(logits_F, Y)
L   = 0.5 * L_M + 0.5 * L_F
```

CT encoder/align 的梯度来自两路平均；PET encoder/align 和 Full decoder 只受 Full 损失驱动；CT decoder 只受 Missing 损失驱动。等权损失不保证梯度等强，也不等于实际部署缺失率为 50%。

512 输入，batch16，lr8e-5，AdamW，weight_decay1e-4；60轮上限，原60轮调度保持，warmup3；AMP、EMA decay0.999/start epoch3、clip5、seed2023、CIPA配对增强、ImageNet归一化及 drop_last 保持 baseline。未来新实验 early-stopping patience5，按 Joint 严格提升选同一个 best.pt；只存 best，不存 last。预算与选模规则不得因某个方案暂时落后而临时修改。

Joint=(Full Dice+Missing Dice)/2。在固定 Joint 选择下报告同epoch的两路指标、HD95、漏检与误报。以 Missing 为首要部署目标并不等于应立即改选模规则；若将来改变，需要另立协议并重新建立可比 baseline。

## 已知 baseline 与科学边界

当前 B0 重跑最佳 epoch5：Full0.7786667593，Missing0.7687558183，Joint0.7737112888。该 run 被暂停，不是完整收敛结论。当前 val=test.txt 且显式允许复用，这些是 pilot 开发结果，不是独立最终测试证据。正式结论需要冻结设计后建立独立患者级验证/测试，不能把同一组复用指标当独立测试。

双编码器、双解码器与相加融合足以构成合理 baseline，本身不应宣称新颖。共享不必然导致负迁移。Full 超过 Missing 也不证明 PET 对每个区域都有因果收益；两个解码器的参数、优化差异会影响比较。CT 完全不可辨认的信息，不能保证靠联合训练或蒸馏恢复。

还没有同预算独立 CT-only 训练对照，因此不能声称 Full 分支已经改善或损害了 CT-only 能力。这个对照是检验负迁移/正迁移的直接证据，梯度快照无法替代。

## 梯度诊断方法与局限

脚本 `scripts/diagnose_shared_ct_gradients.py` 只读取 checkpoint 和 train split。固定 seed2023，抽取256张训练切片，32 batch×8，原分辨率、归一化和 BCE/Dice。逐 batch 同一次 Full 前向分别求两路对 CT encoder/align 的梯度；报告整体、stem、四stage、align分组。

检查 `cos(g_M,g_F)`、范数比、合并梯度抵消，以及 `g_M·(g_M+g_F)/2` 和 `g_F·(g_M+g_F)/2`。负 cosine 是方向冲突；后两项为负才表示平均梯度在无限小普通 SGD 步下使对应损失局部上升。这些量不是实际 AdamW、AMP、裁剪后参数更新的反事实结果。

主诊断使用 raw 权重、train模式与原 CIPA 增强。train-mode BN缓冲每batch恢复，不更新权重、不构造优化器。eval raw/EMA的无增强诊断单独报告；它们是固定部署状态的敏感性，不应冒充历史训练梯度。梯度计算使用 FP32，禁用 TF32，不等同于原 AMP batch16 的所有数值细节。

验证了 strict checkpoint加载、参数和缓冲完全不变、无 `.grad` 累积、平均梯度与等权目标的数值一致性，以及 Missing 禁止调用 PET/Full 模块。首次实现中的逐参数近零误差检查受到 TF32 数值影响；显式禁用 TF32 后重新执行。另修复 backbone 的 `model` 包装层造成 stage 分组未拆分的诊断问题，分组增加回归测试并重新测量；不涉及训练代码。

最佳checkpoint附近的32个batch不能代表整个训练轨迹，也不是患者级独立统计证据。没有早期历史权重时，不能倒推早期没有冲突；今后可在正常 baseline 训练中按预定时点做小规模诊断，不需要保存每轮大权重。

负梯度 cosine 的定义和边界参考 [Gradient Surgery for Multi-Task Learning, NeurIPS 2020](https://arxiv.org/abs/2001.06782)。该论文也讨论方向、梯度量级和曲率的共同作用；不能看到一个负 cosine 就自动采用 PCGrad。本轮不引入任何梯度处理。

## 后续设计的门槛

保持这套骨架与训练协议；先为新模块提出可检验的问题，明确它改变哪个路径、梯度或信息来源。一次仅改一个因素，与固定 B0 同预算比较。如要主张解决负迁移，必须增加独立 CT-only 对照；如要主张差异引导有益，必须超过普通困难区域增强。当前全部增强实验保持暂停，不把不成功的增强等同于整个骨架失败。

## 本轮实测结论

| checkpoint | train-mode整体平均cosine | 负cosine批次 | Full/Missing梯度范数比中位数 |
|---|---:|---:|---:|
| 原baseline最佳epoch6 | 0.6545 | 0/32 | 0.8377 |
| B0重跑最佳epoch5 | 0.6956 | 0/32 | 1.0474 |

逐层存在少量负cosine：原baseline stem/stage2/stage3各1/32，B0 stage2为1/32、stage3为3/32。所有测量模式及分组均未出现等权合并梯度对任一路的负下降内积。原baseline的eval-raw整体存在2/32负cosine，不能表述为“所有模式从无冲突”。

结论是：没有发现这两个训练快照被严重方向冲突或整体梯度强度失衡主导的证据；不能证明整个训练从无冲突，也不能由此证明联合训练优于独立CT-only。无需仅因增强失败而改变共享编码器，或立即加入PCGrad/动态损失。

两次完整实测分别约61.8/60.5秒；RTX3090，FP32 batch8峰值allocated9.41GiB，不代表AMP batch16正式训练显存。3项新增单元测试通过。结果保存在 `checkpoints_new/AUGMENTATION/framework-review-s2023/analysis.md` 和 `{original,b0}-final/report.json`。
