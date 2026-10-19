# 双 ConvNeXtV2-Nano 相加对照（lr=4e-5）

在当前 `exp/full-add-local-reassembly` 分支增加独立运行入口，不新建分支。
本实验检验：在与双 MiT-B1 相同训练协议下，换成双 ConvNeXtV2-Nano 后的表现。
不根据一次结果断言同构/异构优劣、模型理论上限或融合模块是否有效。

## 模型与保持项

- CT、PET 分别创建一个 ConvNeXtV2-Nano，参数独立，不共享权重。
- 两个编码器分别读取同一份本地 Nano 预训练，训练时各自更新。
  只读取编码器预训练，不加载训练好的分割 baseline。
- 原生通道 `80/160/320/640`，两路均经过原 `Conv1x1+BN+ReLU`
  对齐到 `64/128/320/512`，逐尺度相加后进入原 GN U-Net 解码器。
- 仅 Full：每个样本使用真实 CT 和 PET；主损失仍为 BCE+Dice。
- 与 `scripts/run_full_add_mitb1_lr4e5.sh` 保持相同训练器、数据划分、增强、
  归一化、指标、512输入、batch16、AdamW、lr4e-5、weight_decay1e-4、
  60epoch、warmup3、cosine_min_lr1e-6、flat_ratio0.3、AMP、clip5、
  EMA0.999/start_epoch3/warmup、seed2023、无深度监督。
- 新目录以 `full-add-convnextnano-lr4e5-60ep-ema-s2023-时间戳` 命名。
  不调用 LocalReassembly、不创建额外参考/学生网络、不增加保护或辅助损失。

## AutoDL 运行

```bash
cd /root/autodl-tmp/mkd-main/new-train
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 编码器预训练加载、两路梯度、对齐尺寸和两步 AMP 前反传
python scripts/check_convnext_nano_baseline.py \
  --device cuda --image_size 512 --batch_size 2

# 正式训练
bash scripts/run_full_add_convnextnano_lr4e5.sh
```

预检默认要求真实预训练存在，来源为：
`/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano`。
不要使用 ConvNeXt V1 或 MiT 权重替代；加载失败应先修正路径或权重。
修改路径时，预检传 `--nano_pretrained_path`，训练脚本传
`--ct_pretrained_path` 和 `--pet_pretrained_path`。
两个路径必须同时修改才能保持同源初始化。

本地随机初始化预检：

```bash
python scripts/check_convnext_nano_baseline.py \
  --device cpu --image_size 64 --batch_size 2 --pretrained false
```

预检batch2通过不代表训练batch16的显存足够。如果必须降低batch，
应为比较组采用相同batch重新核验，避免将batch影响归为编码器收益。
所有额外命令行参数会在脚本末尾覆盖默认值。

## 对比解释

| 对照 | 变化 | 可以回答 |
|---|---|---|
| 双 MiT-B1 → 双 Nano | 两个编码器及其预训练、原生通道和参数量 | 哪套完整配置更适合当前训练协议 |
| Nano(CT)+MiT(PET) → 双 Nano | PET 编码器及其预训练、PET对齐输入通道 | 在 Nano CT 固定时，替换 PET 分支是否有帮助 |

相同学习率不等于各编码器都达到各自最优；60轮是固定预算。
同时比较验证 Dice、HD95、病例级变化及训练成本，不能只看单轮峰值。
继续保留原 `train_original.txt / test.txt / test.txt` 开发协议；val=test
不构成独立最终测试，不能把最终JSON当作未参与选模的测试结果。

输出沿用 `train_log.csv`、`config_args.json`、`ckpt.best_full.pth.tar`、
`ckpt.last.pth.tar`、`final_test_full.json`，最终评价遵循 checkpoint 记录的 raw/EMA 权重。

## 本地验证范围

- 新增两项测试及 MiT、Full Add、EMA、混合训练回归：共45项通过。
- 测试读取两份启动脚本实际参数，确认仅编码器和对应权重路径不同；
  用本地生成的兼容 Nano 权重验证两套真实编码器的加载一致、存储独立。
- 512×512、batch2、CPU随机初始化的两步前反传通过：两路编码器、对齐层、
  解码器梯度非零且有限，对齐后四尺度尺寸正确。参数量34,343,393。
- 本地无CUDA及AutoDL真实预训练/数据；尚未验证CUDA/AMP、batch16显存或真实分割效果。
