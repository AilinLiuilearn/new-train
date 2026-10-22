# 框架判断、归一化敏感性与PET缺失率

用户决定继续固定EMA。正式训练和自动迭代保持暂停；本轮没有优化器更新或模型结构修改。

## 可以固定骨架，并不等于证明每项细节最优

当前baseline能够完成两路分割、有效梯度更新与PET隔离；已测CT共享梯度整体多数同向，raw/EMA完整重评可复现。不应因融合或增强提升小就推翻骨架。也不能据此保证GN、对齐和训练预算在所有新模块下均最优。

EMA不直接改变raw更新；它平滑时间上的参数和浮点buffer，并影响选模/早停。尚无证据证明EMA系统性隐藏了有效融合模块，但新模块初始残差、学习速度与EMA/早停的交互不能凭现有结果排除。今后可在预定诊断时点同时查看raw/EMA，保持正式EMA选模不变，不为某次结果临时改规则。

当前对齐为每尺度1×1 Conv+BN+ReLU，功能是通道投影与数值变换，不保证不同编码器通道语义一一对应，也不是图像空间配准算法。尚未发现当前代码错接/维度错误或整路断梯度；新模块是否适合在这些表征上工作仍需验证。

## GN确实可能抵消某些融合设计的变化

当前Full解码器每尺度入口为无bias的1×1 Conv→GN→ReLU。若融合只把整个尺度的特征乘以正标量a，则Conv后整体也乘以a，GN的均值/标准差归一化会近似消除它。epsilon与有限精度使其不是严格相等。这个推论不适用于任意空间/通道注意力或改变CT/PET相对贡献的运算。

使用B0最佳EMA checkpoint，在固定seed2023抽取的8张train图上，eval、无增强、不改权重：

| 仅用于敏感性测试的干预 | 两批logits相对L2变化 |
|---|---|
| 全部融合特征乘2 | 5.22e-6 / 4.81e-6 |
| 各尺度分别乘0.5/2/0.5/2 | 6.69e-6 / 5.98e-6 |
| CT+PET 改为 CT+0.5PET | 0.01081 / 0.01157 |

因此不能把“门控值变了”直接当作“解码器获得了新信息”。但这不是GN导致过去全部模块失败的因果证明：没有逐一复现所有旧模块，也没有受控GN/BN重训对照。无需为了进入创新而先遍历所有归一化方案；新模块至少应检查有效梯度、融合实际改变量和到logits的传播，再进入同预算训练。

仓库旧 `diagnostics/add-vs-cainet-full/report.md` 另有Full-only模型诊断：CAINet全局Dice约0.77810、Add约0.77728，但病例平均Dice/HD95等不一致改善；部分深层背景改变量明显大于前景。这是旧MDT模型的描述证据，不能直接当作当前双解码器的因果实验或“GN无问题”的证明。

原始敏感性结果：`checkpoints_new/AUGMENTATION/framework-review-s2023/decoder-normalization-probe.json`。脚本 `scripts/probe_decoder_normalization.py`，参数/buffer未变断言通过。未使用这些8张图比较性能或选择融合超参数。

## 不同缺失率的定义与现有能力

当前模型支持CT始终在、PET整幅存在或不存在的两种推理路径。现有 `run_staged.py` 的evaluate接口直接输出两端Full/Missing；它并未直接提供中间缺失率评估。新增脚本使用已验证的同样本两路预测计数，实现病例级随机缺失的路由汇总。同一患者全部切片走同一路；每次随机排列内各缺失率使用嵌套缺失集合。

这是MCAR病例分配下的缓存预测模拟，不是新的无PET数据读取实测。它依赖当前模型Full返回的CT预测与独立Missing推理相等；此前模型隔离测试已验证。若未来改了CT路径的信息依赖，必须重新验证这个前提。

在B0 EMA、4514切片/121病例上，20次固定种子分配（不是20次训练）：

| 目标病例缺失率 | 实际病例缺失率 | 混合全局Dice均值 | 分配间最小/最大 |
|---|---:|---:|---|
| 0% | 0% | 0.778667 | 相同 |
| 25% | 24.79% | 0.776566 | 0.773654 / 0.780519 |
| 50% | 50.41% | 0.773717 | 0.768715 / 0.776723 |
| 75% | 75.21% | 0.771634 | 0.767884 / 0.774641 |
| 100% | 100% | 0.768756 | 相同 |

121个病例无法精确分到每个目标比例；每病例切片数不同，实际切片缺失率也不同，JSON全部报告。按选中路径汇总TP/FP/FN再计算全局Dice，不是Full/Missing Dice的线性插值。原有Joint算术平均选模规则完全不变，中间缺失率的混合Dice不是新的Joint定义。

缺失率只改变两路被调用的比例，不会让同一CT-only模型随缺失率自动变强。它不模拟图像内局部PET缺失、低质量/错位PET，也不代表临床非随机缺失。任一新融合若只改善Full而不改变Missing，收益通常随缺失率上升而减弱；100%PET缺失端不会通过测试时不存在的融合模块直接获益。训练期共享CT参数变化仍可能间接改变Missing，必须实测，不能保证。

结果在 `checkpoints_new/AUGMENTATION/framework-review-s2023/b0-missing-rates.json`。新增病例一致路由/端点/固定种子/嵌套集合/非法参数测试1项通过。复用了现有pilot预测，不将其声称为独立测试或新推理性能证据。

```bash
python scripts/evaluate_missing_rates_from_counts.py \
  --samples checkpoints_new/AUGMENTATION/ema-review-s2023-v1/00-dual-add-joint-cropcontrol-s2023-v1/ema-samples.jsonl \
  --output checkpoints_new/AUGMENTATION/framework-review-s2023/b0-missing-rates.json
```

已有输出会拒绝覆盖，重跑选新路径。所有原权重和数据保留。

参考：[Group Normalization, ECCV2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.php)。GN不依赖batch维度统计；本仓库的近似标量缩放不变性同时依据实际Conv/GN结构与上述实测，不据此推断任意模块均失效。
