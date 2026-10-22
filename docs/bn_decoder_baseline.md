# GN → BN 解码器 baseline 对照

状态更新：用户已要求停止。BN完成9轮验证、第10轮中断，最佳epoch7 Joint0.766512；GN最佳epoch5 Joint0.773711。保留GN为后续工程基线，未宣称BN充分收敛或达到5轮早停。训练及等待流程均已停止，最终记录见 `checkpoints_new/AUGMENTATION/framework-finalization-s2023-v1/`。以下启动与运行说明保留作历史审查，不表示仍在训练。

用户明确要求在相同baseline设置下，把两个GN解码器换成BN并重新训练。本轮已实现、完成检查并启动单个BN实验；旧增强/Copy-Paste自动迭代继续暂停。不把BN必然改善融合当作先验结论。

## 实际改动

两个解码器的10个 `Conv→GN→ReLU` 块分别改为 `Conv→BatchNorm2d→ReLU`。BN使用PyTorch默认eps1e-5、momentum0.1、affine=True、track_running_stats=True；两个解码器独立。对齐层原有BN、编码器、融合相加、损失及数据处理不变。

新增 `--decoder_norm {GN,BN}`。旧checkpoint缺少字段默认GN，加载/恢复使用保存配置且禁止临时覆盖归一化类型。新BN模型严格加载BN权重，不能用GN任务checkpoint初始化。stage1移交归一化不符时明确拒绝。

实际训练从同一通用CT/PET预训练重新开始，不加载GN任务best.pt，也不继承其优化器、EMA或调度器。两个编码器文件SHA与参考run初始化记录一致。固定种子验证GN/BN共有初始状态逐张量相同；参数量同为35,907,538，BN只新增运行统计buffer。

共享解码器组件保留默认GN，以兼容原模型；本轮BN启动脚本显式指定BN。诊断/EMA重评入口按checkpoint保存的归一化建模。当前离线建图schema仍限定GN，不能把BN任务checkpoint当GN加载；本轮完全不用增强地图，也未重建地图。

## 固定协议与差异记录

配置来源 `checkpoints_new/STAGED/dual-add-joint-cropcontrol-s2023-v1/config.json`，仅读取配置、split元数据和通用预训练SHA，不打开任务checkpoint。来源及可复制命令见审查目录launch.json。

保持：512、batch16、lr8e-5、60轮上限及原调度、warmup3、AdamW/weight_decay1e-4、AMP、EMA decay0.999/start epoch3及原warmup、clip5、seed2023、ImageNet归一化、CIPA随机配对增强、原划分/drop_last。每图同时Full/Missing、CT一次编码、loss=0.5CT_loss+0.5Full_loss；Joint等权均值选模。crop_guidance/copy_paste均none。

参考B0配置早停曾为10，本轮遵循用户后续明确的统一政策：连续5轮Joint无严格提升停止，只存best.pt，不存last.pt。此差异明确记录在launch.json，不声称保存配置完全相同。参考GN B0实际已在第10轮后暂停、最佳第5轮。

val=test.txt且显式允许复用，保留原警告；这是pilot对照，不是独立最终测试，不自动根据此结果调BN/EMA/阈值。

## 实测检查

- GN修改前真实实现作为tests/oracles/group_unet_decoder_before_bn.py保存；默认GN状态及CPU输出exact相等。
- BN训练缓冲更新、独立解码器、严格state兼容/拒绝错误类型、保存→evaluate→resume、保存配置权威与Missing禁止PET/Full调用检查通过。
- tests/test_decoder_norm.py + tests/test_staged.py + tests/test_preflight_amp.py：11 passed，3 skipped（torch2.1.2无CPU torch.amp.GradScaler；未伪称通过）。
- RTX3090上真实PNG、原通用预训练、512/batch16/AMP：一次有效AdamW+EMA更新，首次尝试成功、零AMP跳步；六个训练模块梯度均非零，Missing隔离通过，峰值allocated11.7248GiB。它是独立预检查进程，权重未用于正式训练。

原始检查结果与日志在 `checkpoints_new/AUGMENTATION/bn-baseline-review-s2023-v1/`。完成测试后仅清理了该目录pytest-tmp内的合成checkpoint以回收约7GB，保留测试日志及原用户权重。

## 当前运行与交付

输出：`checkpoints_new/STAGED/dual-add-joint-bndecoder-s2023-v1/`

日志：`checkpoints_new/AUGMENTATION/bn-baseline-review-s2023-v1/training.console.log`

状态：同目录status.json、process.json。启动PID51946（独立进程组）；以实际进程状态和日志为准。训练结束后脚本自动写comparison.json，对比同一个Joint选出的GN/BN各自epoch的Full/Missing/Joint，不自动启动其他实验。训练期间尚无最终性能结论。

```bash
cd /root/autodl-tmp/mkd-main/new-train
conda activate myenv
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/train_bn_baseline.py \
  --source_run checkpoints_new/STAGED/dual-add-joint-cropcontrol-s2023-v1 \
  --output checkpoints_new/STAGED/dual-add-joint-bndecoder-s2023-v1 \
  --review checkpoints_new/AUGMENTATION/bn-baseline-review-s2023-v1
```

该命令已在后台运行，不要重复启动；新实验需要新output和review路径。GN历史权重可按旧config继续评估，关闭BN选项不会将已有BN权重变回GN。无用户数据删除、无切换/推送分支、无环境升级。
