# 两个独立的干净基线

分支：`exp/clean-ct-full-add-hetero-lr8e5`。
来源：`exp/clean-full-missing-hetero-lr4e5`，保留其数据、对齐、GN 解码器、BCE+Dice、EMA 和调度实现。

| 训练 | 结构 | 选模 |
|---|---|---|
| CT-only | ConvNeXtV2-Nano → CT 对齐 → GN UNet 解码器 | CT-only Dice |
| Full Add | CT ConvNeXtV2-Nano / PET MiT-B1 → 各自四尺度对齐 → 逐元素相加 → GN UNet 解码器 | Full Dice |

CT-only 不创建 PET 编码器、PET 对齐层或融合层，也不读取 PET 文件。
Full Add 每个训练样本均使用真实 PET，不混入 Missing 样本。
对齐仍为原基线的 1×1 Conv+BN+ReLU，目标通道 64/128/320/512；解码器使用 GN。
无原型库、补偿、门控、注意力、蒸馏、混合模态训练或深度监督。

共同设置：峰值学习率 **8e-5**，AdamW，weight decay=1e-4，60 epoch，patience=60，batch=16，512²，seed=2023；warmup=3，min_lr=1e-6，flat_ratio=0.3；EMA=0.999，epoch>3 开始，decay warmup 开；AMP 开，clip=5；cipa 增强、imagenet 归一化、cipa_official HD95。
沿用 train_original.txt / test.txt / test.txt 和显式 allow_val_equals_test，结果属于沿用该集合选模的开发评价，并非独立测试结果。

```bash
cd /root/autodl-tmp/mkd-main/new-train
git fetch origin
git switch exp/clean-ct-full-add-hetero-lr8e5
bash scripts/train_ct_only_lr8e5.sh
# 上一项完成后运行
bash scripts/train_full_add_lr8e5.sh
```

分别生成独立的时间戳目录。仅加载对应编码器预训练，从头训练分割模型，无冻结。
已存在 config_args.json 时拒绝覆盖。预训练缺失或不兼容直接报错。

每轮保存 ckpt.last.pth.tar，以及按自身训练场景 Dice 选择的 ckpt.best.pth.tar。
Full Add 仅执行 Full 验证与最终评价；CT-only 仅评价 CT。每轮各验证一次，不执行 Missing/Joint 诊断。
训练结束加载最佳 checkpoint 中记录的 raw/EMA 权重，写入 final_test.json。

```bash
python eval_baseline.py --checkpoint checkpoints_new/MDT/你的目录/ckpt.best.pth.tar
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests -q
```

依赖：PyTorch、torchvision、timm、transformers、safetensors、numpy、scipy、Pillow、opencv-python、medpy、pytest。
当前编写环境没有 PyTorch，已做 Python 语法及 shell 检查；尚未执行模型测试、真实数据训练、预训练加载和 CUDA 显存检查。
