# LSConv 接入核验记录

核验日期：2026-10-15。
基线：`663f9786122365d8b2774e4497df92bc06b715d5`。
作者源码：`cbe737c92b7c43ecf02d08545a07f03f1010177c`。
环境：Python 3.12、PyTorch 2.5.1 CPU；无 CUDA。

## 启动与旧 AMP 接口兼容修复

同日复测全套测试：**90 passed, 2 skipped**（51.06 秒）。
使用 PyTorch 2.5.1+cpu、timm 1.0.11、transformers 4.44.2。
新增检查覆盖 CUDA 不可用时不创建实验目录、失败启动文件完整备份、
拒绝覆盖已有训练产物，以及新 AMP 导入与模拟旧 CUDA AMP 签名的兼容性。
实际选中的 AMP 装饰器通过 CPU autograd 检查。

`lsconv_ska_triton.py` 现在仅将一行 AMP 导入指向本地兼容层；还原这一行
后文件仍匹配作者 blob，Triton 核和反向实现没有改变。旧 PyTorch 的真实
CUDA AMP/Triton 执行仍需用户服务器预检，不能用模拟接口测试替代。
两项 CUDA FP32/FP16 对照仍因无 GPU 跳过。
以下为首次接入时的历史核验记录。

## 已运行

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests`

结果：**76 passed, 2 skipped**（58.13 秒）。跳过项仅为官方 Triton 算子
FP32/FP16 与 torch 参考实现的 CUDA 值/反向梯度对照。随后对跨 Python 版本
的源码 AST 摘要兼容性调整单独运行测试，1 passed。

- 作者 `Conv2d_BN`、`LKP`、`LSConv` 三个类 AST 一致；首次接入时 Triton 文件 Git blob SHA 一致。
- torch SKA 与独立 unfold 计算的数值、输入梯度、动态权重梯度一致。
- S1–S4 实际尺寸：64@128²、128@64²、320@32²、512@16²。
- train/eval 下初始融合输出与 Add 精确一致；完整模型初始 logits 精确一致。
- 共用编码器、对齐器、解码器初始化一致，模块初始化不偏移后续 CPU RNG 状态。
- 第一步最后 BN 获得梯度；更新后四尺度 LKP 均获得非零梯度。
- 编码器、对齐器、解码器和融合参数都被同一优化器覆盖，梯度有限。
- 关闭模块时参数键与 Add baseline 一致，可严格互载。
- raw/EMA checkpoint 严格重载，评估跟随已记录的权重选择。
- 合成数据的一轮训练→验证→最佳模型重建→最终评估闭环，融合开关正确保存/重建。
- 原 Add、CT-only、Full/Missing baseline 和 EMA 回归测试通过。
- Missing/mixed 状态在实验模型调用编码器前拒绝。
- 脚本 bash 语法、Python 编译和训练入口 help 检查通过。

`python scripts/check_lsconv.py --device cpu --backend torch --image_size 512 --batch_size 2`

结果：**passed**。真实 ConvNeXtV2-Nano/MiT-B1，随机初始化，不加载数据/预训练权重；
初始模型 logits 精确等于 Add，两个优化步骤损失为 0.701573、0.700321，所有
梯度检查通过。这是执行检查，不是分割性能实验。

```json
{
  "device": "cpu",
  "backend": "torch",
  "image_size": 512,
  "batch_size": 2,
  "params_fusion": 539008,
  "params_total": 32981649
}
```

## 尚未验证

- 官方 Triton 算子在用户 CUDA/PyTorch/Triton 组合中的编译、FP16 数值和反向。
- 3090 上 batch16 显存与耗时。
- 用户 ImageNet 权重文件、真实数据训练、真实 checkpoint 的 Dice/假阳性/边界收益。

正式训练前执行：

```bash
python scripts/check_lsconv.py --device cuda --backend triton --image_size 512 --batch_size 2
```
