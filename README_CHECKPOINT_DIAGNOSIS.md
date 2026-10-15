# Add / CAINet checkpoint 对照诊断

只增加离线推理、成对误差统计和报告，不修改训练、融合模块或 checkpoint。
服务器 checkpoint 和真实数据必须在运行机器上存在。本地合成测试不代表真实实验结果。

## 运行

在 `exp/full-cainet-cacr-gcm` 分支仓库根目录运行：

```bash
bash scripts/diagnose_add_vs_cainet.sh
```

默认使用：

- Add：`/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/full-add-baseline`
- CAINet：`/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/full-cainet-s1s4-both-imagenet-ema-s2023`
- 数据：`/root/autodl-tmp/data/PCLT20K`，共同评估 `test.txt`。
- 文件夹内只自动选择 `ckpt.best_full.pth.tar`。不回退到 last、Joint 或任意权重；其他名称请显式传入完整文件路径。
- 两边分别遵循 checkpoint 的 `eval_weights` 选择 raw / EMA，使用各自保存的归一化。
- 不需要原始 ImageNet 预训练文件，整个模型权重严格从 checkpoint 恢复。
- 默认 float32 推理，batch=1；没有 AMP 数值差异。阈值固定为 `sigmoid(logits) > 0.5`。

快速检查少量切片（不代表完整指标）：

```bash
bash scripts/diagnose_add_vs_cainet.sh --max_samples 8 --num_workers 0 \
  --output_dir diagnostics/add-vs-cainet-smoke
```

显式指定文件、输出目录：

```bash
python diagnose_add_vs_cainet.py \
  --add_checkpoint /root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/full-add-baseline/ckpt.best_full.pth.tar \
  --fusion_checkpoint /root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/full-cainet-s1s4-both-imagenet-ema-s2023/ckpt.best_full.pth.tar \
  --root /root/autodl-tmp/data/PCLT20K --split_file test.txt \
  --output_dir diagnostics/add-vs-cainet-full
```

输出目录必须不存在或为空。默认不保存全部预测；`--save_predictions` 可额外保存压缩二值 NPZ。
如需检查 raw/EMA 影响，可用相同命令加 `--add_weights raw --fusion_weights raw` 并指定新输出目录。
这属于另一种权重对比，不替代按原记录选择的主对比。

## 主要输出

| 文件 | 内容 |
|---|---|
| `report.md` | 中文汇总、指标约定、配置混杂因素、诊断阅读方法 |
| `summary.json` | 全局/病例/切片指标、面积分组、融合特征统计及运行配置 |
| `per_slice.csv` | 切片 Dice 差值、概率差、修复/新增 FN/FP、边界及远处错误 |
| `per_case.csv` | 按病例汇总像素计数的 Dice；不是逐切片平均 |
| `per_lesion.csv` | 二维 GT 连通域覆盖率与检出，包含同权重临时相加对照 |
| `feature_scales.csv` | 每张切片 S1–S4 的融合相对变化，分别统计前景/背景 |
| `top_improved/`, `top_harmed/` | 改善/退化最多切片的 CT、PET、GT、预测与差异图 |
| `metadata.json` | 选用权重、配置、split 内容哈希、代码版本和可比性说明 |

可视化差异颜色：绿=修复漏检，青=修复假阳性，红=新增漏检，紫=新增假阳性。

## 解释边界

- 主指标沿用仓库全局 Dice：汇总 TP/FP/FN 再计算。切片和病例平均单独列出。
- 2D 连通域使用四邻接；面积按像素分箱：small<=64、medium<=256、large<=1024，其余 very_large。
  可通过 `--area_bins` 修改。检出定义为覆盖 GT 连通域至少 10%，可通过 `--lesion_overlap` 修改。
  该检出率不等于临床病灶级敏感度，也没有衡量每个病灶的完整分割质量。
- 边界 F1 默认 2 像素容差。GT 膨胀/腐蚀带宽默认 2 次四邻接操作；远处 FP 为 GT 膨胀区域外的预测。
- `pet_hot_fp` 使用 PET 图像灰度严格高于第95百分位的区域作为描述代理；不表示自动识别病理热点。
- “同权重临时相加”：CAINet 的编码器/对齐/解码器不变，仅融合输出改为相加。
  无需重新编码；存在特征分布变化，不能替代正式重训消融或证明训练因果。
- 旧 Add 若没有 `pet_align.*`，恢复 Identity，保持旧版 PET 路径；其余参数仍 strict=True。
  若两个 checkpoint 的 PET 对齐、归一化、训练划分等不同，会在报告中标注，不能把差异全部归因于融合。
- 默认 CIPA official HD95（需 MedPy，512×512）。空 GT 且非空预测使用图像对角线作为单独标记的扩展惩罚；
  若数据存在此情况，HD95 不声称完全复现原 CIPA。Dice 不受该扩展影响。
- 默认持平区间是切片/病例 Dice 差值绝对值 <=0.001（0.1 个百分点）；可通过 `--delta_epsilon` 修改。
- 不自动判断框架上限。先确认主指标复现，再根据逐病例、病灶大小、边界及错误抵消情况提出下一版可验证假设。

## 本地检查

```bash
python -m pytest tests/test_checkpoint_diagnosis.py -q
```
