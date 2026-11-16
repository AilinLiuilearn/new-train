# LPI-BI V1：完整模型首轮实验

用户本轮授权实现并运行完整无文本模型一组。只保留编码器通用预训练；原双路对齐、LPI-BI、M1、GN单解码器全部从头、全部更新。不是加载78.49或78.81权重微调。

## 代码与来源

- `variants/lpi_bi/front.py`：监督定位头、峰值响应混合池化、条件通道MLP、有效3×3邻域、原始V和双向零出口残差。按定稿自行实现。
- `variants/lpi_bi/losses.py`：八头软占比BCE及平方分母Dice；空图只计算BCE，固定八头分母。没有调用原分割loss的二值化函数处理辅助GT。
- 原有 `src/models/components/local_region_residual_fusion.py`、GN解码器、编码器、对齐、数据增强、AdamW/调度/EMA、CIPA指标均直接复用。
- `variants/tsbci/train.py`仅新增可选objective/protocol注入点；原调用不传入时保持原BCE+Dice行为。新loss在训练和同一完整验证中都调用。
- ZePT作者源码：<https://github.com/Yankai96/ZePT>，核验提交`32d3dcb023372b9fc34ee9bb59afbe339002f4ca`。查看`models/ZePT.py`、`models/OSTR.py`、`utils/loss.py`：其预测图生成注意力mask、查询细化与本版“空间条件影响检索”的组织思想相关，但其3D硬mask/查询对齐不同。本版没有复制其网络、硬阈值或训练体系；所核验公开树未发现许可证文件，因此不vendor该代码。
- TMFormer只作为论文层面的区域优先思想来源，本轮未取得并复制作者源码，不虚称源码复现。

## 固定协议

完整E5：四尺度64/128/320/512，latent32、4头、每头8维、滑动3×3；定位占比监督权重0.1；通道MLP末层及残差出口置零；beta初值0.1、有界0至1；epsilon总质量1e-6。原C/P和V不受空间图或通道门裁剪，Q/K接受通道调制。

seed2023、512、batch16、AdamW8e-5/1e-4、原增强/调度/EMA、最长60轮、连续3轮Dice不提升早停、myenv、best-only。每轮只进行一次完整融合验证，不增加开启/关闭或单向验证。每5轮再人工查看，异常或用户询问除外。

结果选择验证Dice最佳轮，同轮记录HD95/灵敏度/精确率；自动比较同协议M1（78.1551%、16.4411），不混入较早不同协议的78.8112或78.4970作直接差值。它们仍保留为历史结果。当前目标78.8%以上、争取78.9%以上且同一权重HD95<17，不保证达到，也不凭本次一组证明机制成立。

## 核验

- CPU：独立坐标循环对照局部注意力，soft占比与空图loss，源内容不裁剪，零出口恒等，六次真实更新后纯任务loss到所有局部参数的梯度；新旧训练入口的objective在训练和验证均接通。
- GPU：见`LPI_BI_GPU_PREFLIGHT.json`。真实512/batch16三次有限AMP更新；公共初始化和后续CPU RNG与原M1一致；初始FP32完整输出严格相同；模型参数44,167,457，新模块654,608（以实际预检JSON为准）；包括优化器和EMA时峰值分配约16.9GiB、保留约17.9GiB；必要权重导出/加载严格一致。
- GPU梯度快照来自完整总损失，不能单独证明定位头接收到任务梯度；纯任务梯度证据来自CPU多步局部单元核验，完整实际参数连接另经只读源码审计。预检临时更新全部丢弃，不作为训练初始化或性能结果。
- 日志：主/辅助loss、各尺度定位loss、门控/响应/注意力、真实增量RMS、解缩放梯度及AMP异常。只保存best.pt实际选中权重，无优化器/RNG续训包。loss和梯度快照不能替代任务增益。

## 运行

```bash
bash scripts/myenv.sh scripts/run_lpi_bi.py --campaign lpi_bi_full_20261116 --gpu 0
```

先提交代码、推送协同分支，再在固定提交的独立worktree运行。该脚本核验GPU预检对应源码SHA，只有一个full组；自然结束写入同轮结果和`FINAL_ANALYSIS.md`，没有自动后续调参队列。停止时在campaign目录创建`STOP`。真实数据、通用预训练、输出路径由各机不提交Git的`paths.local.json`提供。
