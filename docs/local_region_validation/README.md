# Local / region residual fusion

Full-only结构：`F=(C+out_ct(z_ct))+(P+out_pet(z_pet))`。
C/P为原有对齐特征，不原地修改。两路末端投影weight/bias零初始化，初始等价C+P。

| 来源或步骤 | 保留与修改 |
| --- | --- |
| CIPA CT/PET stems | 保留原宽度、3x3卷积拓扑、偏置、BN、ReLU和stride |
| CIPA区域／局部分工 | 保留4x4区域与局部分工；明确修正矩形坐标恢复 |
| Mamba建模 | 替换为独立的4-head pre-norm SDPA、ratio2 GELU FFN，各一层 |
| CT高斯差分 | 自定义三组signed DoG，全特征图reflect可分离卷积，固定buffer核 |
| PET区域读出 | 全局区域自注意力，精确repeat_interleave广播至4x4 |
| 跨模态作用 | 仅在最终相加发生；没有联合token attention或门控 |

CIPA本地核验commit `5c90d86ba1331e8043a92b8e732aa36234c548dc`；
`region_mamba.py` blob `0bd6e0a347bdeaf9433a34ee1023b05041ebb0d6`与设计记录一致。
原CIPA Full存在`out_rgb + out_x + DCIM_output`外部残差。
源：https://github.com/mj129/CIPA；stem适配MIT许可见`CIPA_LICENSE.txt`。
HDNet仅提供多尺度局部对比动机，本实现不是MAC复现。
Gaussian DoG不是RAC-Net RASFE原实现，无深层特征HU解释。

v1明确拒绝H/W不能被4整除的特征。Gaussian reflect要求空间尺寸大于最大核半径4。
标准512的尺度128/64/32/16均满足。融合只服务Full；CT decoder仍接收原CT特征。

## 已完成实验

相同Joint选模规则，第5轮EMA最佳，完成10轮后早停。

| 指标 | GN add baseline | Fusion | 差值 |
| --- | ---: | ---: | ---: |
| Full Dice | 0.77866676 | 0.78438079 | +0.5714个百分点 |
| Missing Dice | 0.76875582 | 0.76652556 | -0.2230个百分点 |
| Joint | 0.77371129 | 0.77545318 | +0.1742个百分点 |
| Full HD95↓ | 17.804063 | 17.037423 | -0.766641 |
| Missing HD95↓ | 22.554602 | 22.975225 | +0.420623 |

原始汇总、逐轮CSV、实际训练配置、checkpoint位置／SHA256见`results/`。
保留此设计作为候选，baseline仍固定GN add。单种子、val/test复用，不宣称显著性、
独立泛化、小病灶改善或高斯机制已被证实。需要配对多种子和去高斯对照。

## 验证与性能

初版21项专项／回归测试通过；完整清理后的验证见`../cleanup_verification.json`。
矩形20x28坐标、精确广播、高斯常量／impulse、零初始化等价、内部梯度、Missing隔离、
optimizer/EMA覆盖、strict加载和真实resume/evaluate均有测试。

`performance.json`记录RTX3090、torch2.1.2+cu118、512/batch16的实际测量：
总参数46,977,746，融合新增11,070,208；AMP训练3次有效更新，峰值allocated15.235GiB、
reserved16.258GiB；期间2次GradScaler跳步单独记录，不计为有效更新。
初始FP32/AMP融合特征最大误差均为0。模型和融合核心在本次清理中未改变。

| Scale | 输入特征（batch16） | PET区域tokens | 融合参数 |
| --- | --- | ---: | ---: |
| 1 | 16x64x128x128 | 1024 | 119,524 |
| 2 | 16x128x64x64 | 256 | 473,864 |
| 3 | 16x320x32x32 | 64 | 2,945,780 |
| 4 | 16x512x16x16 | 16 | 7,531,040 |

运行方式以仓库根README为准。性能脚本`profile_local_region.py`保留初版测量配置；
其预训练默认路径可在脚本构造cfg处修改，也可直接用可传路径的`check_staged.py`预检。
