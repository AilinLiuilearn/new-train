# 三组从头重训：连续3轮未提升早停

三组只使用编码器通用预训练，任务模块从头初始化，固定seed2023、512输入、batch16、AdamW学习率8e-5、原BCE+Dice、EMA、GN解码器、原划分及评估。最长仍60轮；连续3轮验证Dice不高于历史最佳即停止。每轮仅完整融合验证一次，best-only。

|名称|结构|对齐|
|---|---|---|
|m1|原M1＋GN，不引入TSBCI|原CT、PET双路对齐均保留|
|concat_single|CT对齐＋原始PET→拼接→原ConcatFusion的1×1压缩→原GN解码器|拼接前仅CT有对齐层；PET无独立对齐层|
|tsbci|原78.4970%架构：50%发送/接收筛选、文本条件、双向关系及Null＋M1＋GN|原双路对齐保留|

选择CT一侧对齐，是因为MiT-B1本来输出64/128/320/512四个宽度。拼接后的Conv1×1是原ConcatFusion已有的融合压缩，输出仍64/128/320/512，不新增BN/ReLU或改变解码器。压缩层沿用原[I,I]初始化，从两路相加开始学习，不有意破坏初始化。

CT/PET编码器、原CT对齐和GN解码器在固定种子下与原M1共享参数初始化一致；移除PET对齐和M1不改变随后数据迭代的主RNG状态。所有实际参数可训练。

拼接对照同时改变了PET对齐和融合方式，因此与m1的差值不能独立归因于M1。不能以压低baseline为研究目标；所有结果均保留。tsbci与m1是保持双路对齐、原M1和GN不变的前置模块比较。

原近期纯M1历史参考78.3644%/17.4866、TSBCI历史参考78.4970%/16.9834。更早78.8112%/17.3627继续作为独立历史参考保留，不能承诺此次重训复现该数字。早停由5改3、运行环境为myenv，也使本次不等同于旧日志的逐步续接。

```bash
# 单组：可在另一张卡拉取同一提交并独立运行
bash scripts/myenv.sh -m variants.rerun.train --variant m1 --run m1_p3_s2023 --gpu 0
bash scripts/myenv.sh -m variants.rerun.train --variant concat_single --run concat_ctalign_p3_s2023 --gpu 0
bash scripts/myenv.sh -m variants.rerun.train --variant tsbci --run tsbci_p3_s2023 --gpu 0

# 本机只有一张卡时，严格按 m1→concat_single→tsbci 顺序运行
bash scripts/myenv.sh scripts/run_three.py --campaign rerun_three_p3_20261115 --gpu 0
```

后台管理输出位于本机output_root/campaign名称；STATUS.json记录当前组，RESULTS.md汇总自然结束的同轮最佳Dice/HD95。向该目录放置STOP将终止当前组并取消后续；遇到异常停止队列等待分析，不静默改训练参数。三组结束不自动调参或安排其他实验。
