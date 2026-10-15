# 实现核验记录

日期：2026-10-15。
父基线：`663f9786122365d8b2774e4497df92bc06b715d5`。
环境：Python 3.12、torch 2.5.1+cpu、timm 1.0.11、transformers 4.44.2；无 GPU。

## 已执行

测试分三组执行，合计 **86 passed, 1 skipped**（87 项）：

1. `python -m pytest -q tests/test_lesion_guided_fusion.py --tb=short`
   - **14 passed, 1 skipped**，14.13 秒；跳过的是 CUDA AMP 前向/两步反向。
2. `python -m pytest -q tests/test_lesion_guided_integration.py --tb=short`
   - **19 passed**，228.78 秒。
3. `python -m pytest -q tests/test_baseline_contract.py tests/test_ema.py tests/test_full_add_runner.py tests/test_mixed_batch_training.py --tb=short`
   - **53 passed**，222.74 秒。

以上执行时使用 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。

覆盖内容：

- 四尺度实际特征形状；train/eval 初始化精确等于 Add。
- 完整模型的共享参数初始化与 Add 相同，构建新模块不偏移后续 CPU RNG。
- 六种消融配置的真实 ConvNeXtV2-Nano/MiT-B1 两步训练。
- 关闭辅助损失时总损失精确等于主损失；loss weight=0 的贡献为零且无辅助图构建要求。
- 四尺度辅助损失与上采样 logits 后调用 baseline BCE+Dice 的独立对照一致。
- 全背景、全前景和小前景辅助监督；损失、梯度有限。
- 辅助监督关闭但引导开启时，预测头仍在前向中按设计使用；引导关闭时改变辅助头不影响修正输出。
- 输出投影先学习，更新后四尺度门控和两路局部上下文获得非零梯度。
- 优化器包含全部模型参数；不用的分支无伪造的零损失梯度。
- add/aux/full/no_aux 四种配置的一轮训练→验证→EMA→最佳 checkpoint 重建→最终评估。
- raw/EMA 独立严格重载，保存的损失/引导/修正开关和宽度恢复正确。
- Missing/mixed/PET 空缺在任何编码器运行前拒绝，未暗中实现其他 Missing 路径。
- 无 CUDA 不创建实验目录；启动失败目录整体备份，已有 checkpoint 拒绝覆盖。
- 原 CT-only、Add、Full/Missing、EMA 回归测试通过。

## 512 输入 CPU 预检

命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/check_lesion_guided.py \
  --device cpu --image_size 512 --batch_size 2
```

结果：passed。真实骨干，随机初始化，不加载真实数据/预训练文件。
完整 logits 初始精确等于 Add，两个训练步骤均通过有限梯度及四尺度学习检查。

| 步骤 | 总损失 | 主损失 | 辅助损失原值 | 加权辅助损失 |
| --- | ---:| ---:| ---:| ---:|
| 1 | 1.909346 | 1.749383 | 1.599622 | 0.159962 |
| 2 | 1.692580 | 1.536395 | 1.561855 | 0.156186 |

参数：融合 535,588；完整模型 32,978,229。
这些是合成数据执行检查，不是性能实验。

## 其他检查

- Python compileall、训练入口 `--help`、bash 语法、git diff whitespace 检查通过。
- 通过命令替身检查启动脚本六种 arm 的实际参数、独立 hash、固定 seed2023 和 CUDA 要求；非法 arm 返回 2。
- 相对父基线只有新文件，没有修改、删除或重命名任何已跟踪的 baseline 文件。
- 文献源码核对与适配差异记录在 `LESION_GUIDED_SOURCE.md`。

## 尚未验证

- 用户服务器 torch 2.1.2+cu118 / RTX3090 上的 CUDA AMP 前向、反向、batch16 显存和速度。
- 用户预训练文件与真实数据训练，以及真实 checkpoint 的全局/病例 Dice、远处 FP、小面积组和边界收益。
- Stage-2 Missing、检索库接口或完整训练重启续跑。

服务器先运行 CUDA 预检，通过后再训练。实现有效性、新颖性和临床有效性不由上述执行检查证明。
