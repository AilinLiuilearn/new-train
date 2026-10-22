# 当前 baseline 的 EMA 行为与验证

正式训练和自动迭代保持暂停。本轮只重评两个已有 baseline 的同轮 raw / EMA 完整权重，不生成新训练权重。

## 实际机制

`utils/ema.py` 维护整个模型的独立副本，各参数按训练时间平均；不是对 Full/Missing 分支互相平均，也不是对预测概率做集成。副本处于 eval 状态、关闭梯度，不进入训练 loss 或 AdamW。

现有 runner 在 epoch>3 时 reset 并启用 EMA。之后只在成功的 optimizer step 后更新：

```
d_t = min(0.999, (1+t)/(10+t))
EMA_t = d_t * EMA_(t-1) + (1-d_t) * raw_t
```

浮点 buffer 也参与平均，包括对齐层 BN 的 running_mean / running_var；整数计数器直接复制。评估不额外重算 BN 统计。使用 raw 或 EMA 推理时均只加载一套相同架构的模型，参数数量与前向路径不变。

B0 epoch5 的 EMA 更新2175次，末次 decay约0.995881；原baseline epoch6 更新3264次，末次约0.997251。0.999 是 warmup 的上限，不是这两个checkpoint从头到尾的实际常数。局部平滑尺度可作解释，不能当作完整历史的固定窗口。

EMA维护不直接改变raw梯度和optimizer更新；当前代码用EMA指标选模、早停，因而会影响最终选择与训练停止时点。新增测试用相同小网络及AdamW验证了EMA维护前后raw状态逐张量一致；这项机制测试不应冒充真实编码器关闭EMA的重训练结果。

## 评估协议

固定两个保存epoch，严格分别加载其中的model / model_ema，FP32、512输入、batch16、同一阈值与现有BCE/Dice/HD95。4514张切片、121个case，shuffle=False / drop_last=False；完整保留尾批。全程eval、inference_mode，不更新参数和buffer。

输出Full/Missing/Joint Dice、HD95、敏感度、precision、FP/FN像素，以及阳性GT但预测全空的切片计数。后者不是全部病灶漏检数。HD95沿用cipa_official，单位像素；不变更空集协议。

每张切片保存TP/FP/FN/TN，用患者case_id做配对cluster bootstrap（seed2023，5000次）。区间条件于现有checkpoint及当前患者集，不包含训练种子不确定性，也不能消除EMA选模偏差。

两个best原本均按EMA选出。因此本轮能回答“同一保存轮次使用哪套权重的差别”，不能把raw叫作“完全关闭EMA实验的最佳checkpoint”。要比较各自最佳，应在预定训练预算内逐epoch同时记录raw/EMA，各按同一指标规则选模，并事先定义停止协议。

当前val/test复用test.txt，这些是pilot，不是独立最终测试；两个checkpoint同用这些患者，也不能算两份独立测试证据。

## 命令与结果

```bash
cd /root/autodl-tmp/mkd-main/new-train
conda activate myenv
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/compare_raw_ema.py \
  --checkpoint checkpoints_new/STAGED/dual-add-joint-cropcontrol-s2023-v1/best.pt \
  --checkpoint checkpoints_new/STAGED/dual-add-joint-nocttask-s2023/best.pt \
  --output checkpoints_new/AUGMENTATION/ema-review-s2023-v1 \
  --batch_size 16 --num_workers 4
python scripts/analyze_raw_ema.py checkpoints_new/AUGMENTATION/ema-review-s2023-v1
python -m pytest -q tests/test_raw_ema_comparison.py
```

输出目录拒绝覆盖，重新运行须选新目录。结果在 `checkpoints_new/AUGMENTATION/ema-review-s2023-v1/analysis.md`，原始数据在各子目录 `raw.json`、`ema.json`、`*-samples.jsonl`、`report.json`、`paired-bootstrap.json`。未覆盖任何原权重或原训练日志。

EMA属于固定的训练/选模基础设置，本身不作为本项目创新。若同轮Dice改善但敏感度退化，应报告取舍，不凭单一最高Dice声称全面更好；不根据当前复用测试集调EMA系数或阈值。

参考：[PyTorch AveragedModel文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html)。具体实现以本仓库为准。
