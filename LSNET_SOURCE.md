# LSNet 源码来源与适配边界

- 论文：Wang et al., **LSNet: See Large, Focus Small**, CVPR 2025, pp. 9718–9729。
- 正式论文：https://openaccess.thecvf.com/content/CVPR2025/html/Wang_LSNet_See_Large_Focus_Small_CVPR_2025_paper.html
- 作者仓库：https://github.com/THU-MIG/lsnet
- 固定提交：`cbe737c92b7c43ecf02d08545a07f03f1010177c`。

| 本仓库文件 | 对应作者文件 | 核验 |
| --- | --- | --- |
| `models/components/lsconv_source.py` | `model/lsnet.py`，blob `1c46507d7b8a84d7fc2631840a2ad421a2663799` | 提取 `Conv2d_BN`、`LKP`、`LSConv`，三个类的 AST 与作者代码一致；只调整导入和空白 |
| `models/components/lsconv_ska_triton.py` | `model/ska.py`，blob `ee0ef7ac261ec8821827f781c712431a68ae7b46` | 仅将 AMP 导入改为本地兼容层；还原这一行后 Git blob SHA 与作者完全一致 |

来源链接：

- https://github.com/THU-MIG/lsnet/blob/cbe737c92b7c43ecf02d08545a07f03f1010177c/model/lsnet.py
- https://github.com/THU-MIG/lsnet/blob/cbe737c92b7c43ecf02d08545a07f03f1010177c/model/ska.py

## 保留的源码机制

LKP：1×1 Conv/BN/ReLU → 7×7 DWConv/BN → 1×1 Conv/BN/ReLU →
1×1 Conv → GroupNorm → `[B,C/8,9,H,W]`。
SKA：按位置生成的 3×3 核聚合原输入，零填充，不加 Softmax。
作者 CUDA 源码用 `ci % (C/8)` 映射共享核，不能替换成连续通道的
`repeat_interleave`。输出：`x + BN(SKA(x, LKP(x)))`。

## 本项目明确增加的适配

1. 先复用原 baseline 的 `AddFusion`，再将结果输入 LSConv。
2. S1–S4 均使用同一个 LSConv 结构，独立参数，通道 64/128/320/512。
   作者完整 LSNet 在最深阶段还使用 Attention；这里并未声称复制完整 LSNet。
3. 在融合包装器中将 LSConv **最后一个 BN** 的 weight/bias 设为零，
   初始输出严格等于 Add。LKP 内的 BN 与 GN 保留作者初始化。
4. 新模块初始化放在 CPU RNG fork 内，避免额外随机初始化改变 baseline
   之后的数据/Dropout 随机数序列；不保证训练后的随机轨迹相同。
5. `lsconv_ska.py` 增加无参数后端选择器，以及数学等价的 PyTorch 流式参考实现。
   正式训练脚本显式选择 `triton`，调用作者 `SkaFn`；不会静默切换算子。
6. `lsconv_amp.py` 优先使用 `torch.amp.custom_fwd/custom_bwd`；缺少该接口时，
   使用旧版 `torch.cuda.amp` 并移除其不支持的 `device_type` 参数。
   Triton 文件仅改变这一行导入，算子、装饰器调用和梯度实现保持原样。

后端选择不改变 state_dict 的参数键。`torch` 参考实现保留全部核心计算，
不是用固定卷积代替动态卷积。CUDA 与 PyTorch 实现的浮点误差必须通过
服务器预检验证。作者 Triton 文件依赖 `triton`；本项目兼容新旧 CUDA AMP 接口；
预检会暴露安装、编译或反向传播问题。

未搬入作者分类头、下采样、RepVGG/SE、Attention、FFN，也没有新增辅助预测头、
辅助损失、跨尺度传播或手工边缘规则。源码归属原作者；本文件不添加或替代作者授权。
