# ConTriNet MFM：Full-only PET-CT 融合实验

基线：`exp/aligned-add-baseline`，commit `663f9786122365d8b2774e4497df92bc06b715d5`。
实验分支：`exp/full-contrinet-mfm`。

## 官方源码来源

- 论文：*Divide-and-Conquer: Confluent Triple-Flow Network for RGB-T Salient Object Detection*，IEEE TPAMI 2025，47(3):1958–1974。
- 官方仓库：https://github.com/CSer-Tang-hao/ConTriNet_RGBT-SOD
- 固定版本：`9af315d4a93d897cf0b9abb11e194d1ed1072ff9`。
- 原文件：`models/ConTriNet.py`，完整存档在 `third_party/contrinet/ConTriNet.py`。
- `models/components/contrinet_source.py` 原样抽取 `MFM_0`、`MFM`，类定义及计算未修改。
- MIT 许可证保留在 `third_party/contrinet/LICENSE`；来源、行号及 SHA256 在 `SOURCE.json`。
- 测试同时检查归档哈希、AST 一致性和两个类的逐字符一致性。

本实验仅迁移 MFM，不是完整 ConTriNet 的复现。未引入 Res2Net、三路解码器、MDAM、额外损失或原训练策略。

## 模型与四尺度接口

CT ConvNeXtV2-Nano → 原 `ct_align`；PET MiT-B1 → 原 `pet_align`。
两路均为 `[64,128,320,512]` 通道，512 输入下空间大小为 `[128,64,32,16]`。
原 GroupNorm 解码器、损失、优化器、数据处理、EMA 和评估协议保持原设置。

1. S1：作者 `MFM_0(64,64)`。
2. S2：作者 `MFM(128,64)`，第三输入为池化后的 S1 **内部**融合特征。
3. S3：作者 `MFM(320,64)`，第三输入为池化后的 S2 内部融合特征。
4. S4：作者 `MFM(512,64)`，第三输入为池化后的 S3 内部融合特征。

所有 MFM 内部融合输出均保持作者的 64 通道，通过外部 1×1 卷积适配回解码器需要的各尺度通道。
标准二倍金字塔使用与作者相同的 MaxPool2d(2,2)；奇数尺寸使用 adaptive max pooling 对齐。
跨尺度传播不 detach，深层损失可回传浅层 MFM。这里的浅到深发生在融合路径，不修改编码器内部。

**严格按源码计算：**源码中的通道瓶颈门控没有 GAP，不将其替换成教科书式 SE；不把卷积调制称作全局自注意力。
保留官方 BatchNorm、Sigmoid、卷积顺序及双向调制。解码器仍仅使用 GroupNorm。

## 输出模式与消融

| arm | 设置 | 解码器输入 |
|---|---|---|
| `add` | MFM 关闭，无新增融合参数 | `C_l + P_l` |
| `local` | MFM 开启，上一层信息置零 | `C_l + P_l + Proj_l(MFM_l)` |
| `cross`（默认） | MFM 开启，保留浅到深路径 | `C_l + P_l + Proj_l(MFM_l)` |
| `replace` | MFM 开启，保留浅到深路径 | `Proj_l(MFM_l)` |

`local` 保留同样的 MFM 参数化，仅令第三输入为零，隔离跨尺度信息传递的影响；不等同于作者另一个原生模块。
`cross`/`local` 的 CT+PET 外层直连是本任务适配，并非作者原式。
残差输出投影零初始化，初始输出严格等于 Add；第一步投影获得梯度，后续 MFM 获得非零梯度。
`replace` 使用 PyTorch 默认投影初始化，S1 采用 Identity；它也只是原始 MFM 的四尺度接口迁移，并非整篇论文复现。
关闭 MFM 时 state_dict 与 Add baseline 严格兼容；启用 MFM 时不得用 Add checkpoint 进行 strict 恢复。

## 运行

推荐的默认实验：

```bash
bash scripts/run_full_contrinet_ablation.sh cross
```

其他对照：

```bash
bash scripts/run_full_contrinet_ablation.sh add
bash scripts/run_full_contrinet_ablation.sh local
bash scripts/run_full_contrinet_ablation.sh replace
```

脚本沿用当前 CAINet 对照参数：ImageNet 归一化、CIPA 增强、512 输入、batch 16、AdamW 8e-5、EMA 0.999、前三轮 EMA 等待、种子 2023。
脚本同现有对照使用 `test.txt` 选模并显式允许 val=test，因此这些数字属于该共同实验协议，不能称为独立留出测试结果。
若使用独立验证划分，在所有对照一致修改：

```bash
bash scripts/run_full_contrinet_ablation.sh cross \
  --hash full-contrinet-cross-val-s2023 \
  --val_split_file val.txt --allow_val_equals_test false
```

公共配置的默认归一化仍为原来的 cipa；正式对照建议用上述脚本确保与现有 CAINet 一致。
直接入口也可使用 `python run_full_contrinet_fusion.py --help` 查看全部参数。
已有 `config_args.json` 的运行目录会被拒绝覆盖，需换 `--hash`；本入口不支持 resume。

仅训练和评估 Full：传入 Missing 状态或缺失 PET 会在编码前报错。
原来的 CT-only 和 Full/Missing Add 入口不受此实验影响。后续缺失模态补偿接入应独立实施与验证。

## checkpoint 与评估

按验证 Full Dice 选择 `ckpt.best_full.pth.tar`，保留 `ckpt.last.pth.tar`。
训练末尾以相同 MFM 开关重建模型并 strict 加载选中的 raw/EMA 权重进行 Full 测试。
`config_args.json`、checkpoint 和最终测试 JSON 保存三个 MFM 开关；CSV 增加 `grad_fusion` 和 `grad_pet_align`。

独立评估不需要预训练文件，自动读取保存的开关与 raw/EMA 选择：

```bash
python eval_full_contrinet_fusion.py \
  --checkpoint checkpoints_new/MDT/full-contrinet-cross-imagenet-ema-s2023/ckpt.best_full.pth.tar \
  --output_dir eval_results/contrinet-cross-s2023
```

可用 `--use_ema false` 显式改评 raw 权重。输出目录已有同名结果时会拒绝覆盖。
不要使用 `eval_joint_baseline.py` 评估 MFM checkpoint，该入口仅重建原始 Add/CT 模型。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_contrinet_fusion.py tests/test_contrinet_integration.py \
  tests/test_baseline_contract.py tests/test_full_add_runner.py \
  tests/test_ema.py tests/test_mixed_batch_training.py
```

需要原 baseline 的 PyTorch、timm、transformers、numpy、scipy，以及测试依赖 pytest；没有新增训练依赖。
验证包含官方类一致性、512 对应四尺度、奇数尺寸、前后向、跨尺度因果性、零初始化后学习、CPU bfloat16 autocast、真实编码器前后向、优化器成员、关闭严格等价、raw/EMA checkpoint、四组训练末尾重建，以及原 baseline 回归测试。
真实 PCLT20K 数据训练、GPU FP16 显存与最终精度需在用户训练环境验证。

## 回退

使用 `add` 运行关闭全部 MFM 参数；或切回 `exp/aligned-add-baseline`。
本分支对已有代码仅参数化 `run_full_add_baseline.py` 的模型构建回调及可选日志字段，原入口默认仍为 Add。
