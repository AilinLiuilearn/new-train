# Full-only Add + LSConv 实验

分支：`exp/full-add-lsconv`。
基线：`exp/aligned-add-baseline`，提交 `663f9786122365d8b2774e4497df92bc06b715d5`。
源码与适配说明见 [LSNET_SOURCE.md](LSNET_SOURCE.md)。
已运行检查与限制见 [LSCONV_VERIFICATION.md](LSCONV_VERIFICATION.md)。

## 实际接入

每个尺度独立计算：

`B_s = aligned_CT_s + aligned_PET_s`

`F_s = B_s + BN_s(SKA_s(B_s, LKP_s(B_s)))`

S1/S2/S3/S4 通道为 64/128/320/512，512 输入时空间尺寸为
128/64/32/16。四个尺度结构与 7/3/8 核配置一致，参数独立。
新增融合参数 **539,008**，完整模型 **32,981,649**。
最后 BN 零初始化，初始融合特征和完整模型 logits 与 Add 精确一致。
第一步最后 BN 获得梯度；更新后 LKP 获得非零梯度，已验证两步训练。
LSConv 自带残差，外层没有再加一次 B_s。

这次只验证 Stage-1 完整 CT+真实 PET。Missing/mixed 状态在编码前报错，
不会暗中当作已完成的缺失模态模型。CT/PET 编码器、两个通道对齐器、
GroupNorm 解码器、BCE+Dice、AdamW、调度和 EMA 均沿用基线。

## 运行

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch --track origin/exp/full-add-lsconv

# 无数据预检：官方 CUDA 算子与参考实现的值/梯度；真实编码器两步 AMP 训练
python scripts/check_lsconv.py --device cuda --backend triton --image_size 512 --batch_size 2

# 四尺度融合训练
bash scripts/run_full_lsconv_ablation.sh lsconv

# 相同入口、关闭模块的 Add 对照（独立 hash，参数键与 baseline 一致）
bash scripts/run_full_lsconv_ablation.sh add
```

训练脚本配置：512×512、batch 16、workers 4、seed 2023、CIPA 增强、
ImageNet 归一化、60 epochs、lr 8e-5、weight decay 1e-4、warmup 3、
EMA 0.999/start 3、CIPA 官方 HD95。保持当前比较采用的
`train_original.txt/test.txt/test.txt` 与显式 `allow_val_equals_test=true`；
因此该结果属于沿用既有协议的模型选择结果，不是独立留出的测试集估计。
脚本尾部支持覆盖相同的公共参数，例如：

```bash
bash scripts/run_full_lsconv_ablation.sh lsconv --random_state 2024 --hash full-add-lsconv-imagenet-ema-s2024
```

`lsconv` 默认 hash：`full-add-lsconv-imagenet-ema-s2023`。
`add` 默认 hash：`full-add-lsconv-control-imagenet-ema-s2023`。
已有相同 hash 的 config 会报错，避免覆盖之前实验。未实现恢复训练。
这是从 ImageNet 编码器初始化开始的新实验，不是加载旧 Add 分割 checkpoint 微调。

## 单独评估

```bash
python eval_full_lsconv_fusion.py \
  --checkpoint checkpoints_new/MDT/full-add-lsconv-imagenet-ema-s2023/ckpt.best_full.pth.tar \
  --output_dir checkpoints_new/MDT/full-add-lsconv-imagenet-ema-s2023/eval_full
```

从 checkpoint 的 config 重建融合开关和结构，以 `strict=True` 加载参数。
默认遵循 checkpoint 记录的 raw/EMA 选择；可用 `--use_ema true/false` 覆盖。
不依赖原 ImageNet 文件路径；CPU 评估自动使用无参数差异的 torch 后端。
评估指标输出到 `eval_full_lsconv.json`。训练结束也会用同一个模型构建器
重建最佳模型，生成 `final_test_full.json`。

日志在原字段外增加 `grad_pet_align`、`grad_fusion`；`config_args.json`
记录源码提交、四尺度通道、核配置、零初始化和后端。

## 核验与判断

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
python scripts/check_lsconv.py --device cpu --backend torch --image_size 512 --batch_size 2
```

实现环境：PyTorch 2.5.1 CPU。已验证四尺度实际尺寸、初始化等价、
参考算子的独立 unfold 值/双输入梯度对照、真实骨干两步训练、优化器成员、
raw/EMA 严格重载，以及一轮训练/验证/最终测试闭环。CUDA FP32/FP16 对照
测试已编写，但本环境没有 CUDA，尚未执行；实际 3090 batch16 显存、耗时、
真实数据 Dice 也未验证。服务器应先运行 CUDA 预检，通过后再训练。

最小可证伪假设：相加后按上下文生成局部聚合核，能比纯 Add 更好地组织
局部响应；对照需要同时观察全局/病例 Dice、小目标、边界、远处假阳性和
完全失去重叠的切片。若新增假阳性抵消收益或重复种子无一致收益，应拒绝
有效性结论；不能仅凭一次最佳 epoch 宣称突破上限。

回退：运行原 `run_full_add_baseline.py`，或使用上述 `add` 对照；原模型文件、
解码器、数据和损失文件未改。唯一公共文件调整是让原 Full runner 接受显式
模型构建器/融合类型及额外梯度字段；默认调用仍是原 Add 路径。

设计决定：**Proceed，作为受控实验实现；性能改善尚待真实数据验证。**
