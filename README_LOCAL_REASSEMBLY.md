# PET–CT 局部重组融合：编码器预训练 + 原分割损失

分支：`exp/full-add-local-reassembly`，协议 schema=3。

## 当前实验协议

本入口只在原相加 baseline 的四尺度融合位置替换模块：
CT/PET → 原独立编码器 → 原对齐层 → 局部重组融合 → 原 GN 解码器。

- **CT/PET 编码器加载与原 baseline 相同的本地预训练权重，并参与联合训练。**
- 对齐层、融合模块、解码器重新初始化。融合的两个输出投影置零，初始严格等于相加。
- 不加载训练好的分割 baseline，不继承其优化器、调度器或 EMA 状态。
- 单个任务模型，无冻结参照、教师/学生、蒸馏或相加对照分支。
- 损失只有原 BCE+Dice，内部相加修正损失及其参数、代码、额外解码已删除。
- 常规 EMA 仍按原 baseline 配置用于评价，不提供损失参照。

“编码器预训练”与“从最佳分割 baseline 继续微调”是两回事。这里使用前者，
从新的分割任务训练开始；不是所有组件全随机初始化，也不是从训练好的分割模型续训。
旧的随机初始化/内部相加损失实验不是本版本的对照。schema1/2 的配置不能混用。

## 融合模块

四尺度 S1..S4 通道 `[64,128,320,512]`、低维通道 `[16,16,40,64]`、两个注意力头。
CT query 选择 PET 的3×3局部候选；反方向同样执行。额外的可学习 key、固定零 value
为空候选，越界邻居在 softmax 前屏蔽。压缩特征逐位置 LayerNorm 后生成 Q/K，value 保留原低维内容。

```text
delta_P = U_P(transport(P|C) - project(P))
delta_C = U_C(transport(C|P) - project(C))
F = C + P + delta_C + delta_P
```

仅两个无偏置输出投影置零，无末端 ReLU、不乘2。第一步输出头有梯度，
内部梯度为零；头更新后内部投影、Q/K 和空候选获得梯度。
核心 FP32、输出转换为 Add 的 dtype，支持 AMP，不支持整个模块 half/double。
不依赖全局 N×N 注意力、Triton 或自定义 CUDA。新增参数220,208，总参数32,662,849。
空候选不代表已验证的医学可靠性；模块是否有效需要实际消融。

每个训练 batch，CT/PET 编码器各前向一次、解码器前向一次：
`L = BCE_Dice(z_fused, GT)`。验证/推理也只有正常融合预测。
Missing 路径仍绕过 PET 编码器和融合；auto 只编码 Full 行的 PET。
本入口仍为 Full 训练，未接入 Stage-2 检索补偿。

## 运行

更新后启动全新实验，不要复用旧训练进程或旧输出目录：

```bash
cd /root/autodl-tmp/mkd-main/new-train
git switch exp/full-add-local-reassembly
git pull --ff-only
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/check_local_reassembly.py \
  --device cuda --image_size 512 --batch_size 2 \
&& bash scripts/run_full_local_reassembly.sh \
  --hash "local-encoder-pretrained-$(date +%Y%m%d-%H%M%S)"
```

默认编码器来源与原 baseline 一致：

- CT：`/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano`
- PET：`/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1`

预检和训练均默认加载这些来源。可以分别向两条命令传入 `--ct_pretrained_path`、
`--pet_pretrained_path` 更改路径。开启预训练时缺失/不兼容会报错，不能静默降级为随机初始化。
启动应看到 `[PRETRAIN] CT_Encoder/PET_Encoder` 加载统计，以及：

```text
[INIT] mode=encoder_pretrained_fresh_segmentation pretrained=True segmentation_checkpoint=none loss=BCE+Dice extra_loss=none
```

不要再传 `--baseline_checkpoint`、`--reference_checkpoint`、`--protect_enabled`、`--protect_weight`。
非空实验目录拒绝覆盖，重复运行请换 `--hash`。

lr4e-5、AdamW、batch16、512输入、60epoch、warmup3、EMA0.999、clip5、seed2023 保持现有配置。
数据仍是 train_original.txt / test.txt / test.txt，允许val/test相同；这是开发期选模，
不是独立保留测试评价。历史训练器名称含Teacher，但此入口仅把它作为普通分割训练器使用。

## 成对消融

只切换融合，保持编码器来源、随机种子、主损失、数据和训练预算一致：

```bash
bash scripts/run_full_local_reassembly.sh --local_enabled false --hash add-encoder-pretrained-s2023
bash scripts/run_full_local_reassembly.sh --local_enabled true  --hash local-encoder-pretrained-s2023
```

关闭融合时不注册融合参数。相同编码器权重和种子下，共享参数初始化和初始化输出等于 Add。
评价全量Dice、病例平均、小前景组、边界F1、FP/FN及修复/新增错误，不能只凭一次峰值判断有效。
日志保留原指标、梯度和主损失，新增四尺度残差比、空候选/中心注意力质量，无修正损失列。

## 保存与评价

保存 raw、EMA、优化器等本次训练状态。最终/独立评价直接严格加载完整分割 checkpoint，
重建时不再读取原编码器预训练文件；保存的训练初始化信息保持不变。

```bash
python eval_full_local_reassembly.py \
  --checkpoint checkpoints_new/MDT/你的实验名/ckpt.best_full.pth.tar \
  --output_dir eval_local_pretrained --device cuda
```

## 实现来源和验证边界

双向交互借鉴 GeminiFusion（ICML2024）：
https://github.com/JiaDingCN/GeminiFusion/blob/main/models/mix_transformer.py 。
局部候选、零 value 候选、中心残差和零初始化为本任务适配，非原版直接复现。
当前没有 PNDC 损失或内部相加修正损失，不将该组合宣称为已验证创新。

测试覆盖真实本地编码器权重读取、来源失败报错、相同 Add 初始化/RNG、四尺度传输、
两步梯度、单次解码、Missing 隔离、融合开关、原 BCE+Dice 等价、raw/EMA 重载且无需编码器文件。
本地 CPU smoke 可用 `--pretrained false`；这只是执行检查，不是正式训练设置。
服务器真实预训练文件、CUDA和真实数据效果仍需用上面的预检及训练确认。

本次本地验证：新协议22项通过、1项CUDA跳过；原Full Add/EMA/混合训练回归37项通过。
512×512、batch2的CPU两步预检通过，CT/PET/解码器各累计调用2次。
编码器读取测试使用本地生成的兼容权重文件；不代表已读取服务器的真实预训练文件。
