# 源码来源与适配边界

这是两项机制的目标任务适配，不是任一论文全模型复现，也没有验证新颖性或 PET–CT 性能提升。

## PET：MedFormer / DSSA

- 论文：*MedFormer: Hierarchical Medical Vision Transformer with Content-Aware Dual Sparse Selection Attention*。
- 正式出处：Physics in Medicine & Biology，2025，70(19):195005；不是 TMI。
- DOI：https://doi.org/10.1088/1361-6560/ae07a1
- 论文：https://arxiv.org/abs/2507.02488
- 作者仓库：https://github.com/XiaZunhui/MedFormer
- 核验 commit：`3ede41568bd2e6fffb718da9267cf508e01665f0`
- 机制文件：`models/dssa.py`、`models/dsa.py`。

保留：QKV 投影；detach 的区域平均 Q/K 路由；区域 top-k；每查询像素 top-k；softmax 权重聚合；局部深度卷积；输出投影。默认像素保留比例1/8与该版本源码一致，QK缩放保持 `dim**-0.5`。

适配：统一 BCHW（原文件注释称 NCHW，但 forward 实际先按 NHWC 转置）；处理小尺寸；区域均值排除填充；屏蔽填充 key；最少保留一个像素；上下文网格上限32；目标任务增加 PET 区域提示头和两路零初始化残差投影。注意力筛选不等同于 foreground/top-intensity 筛选，也不意味着定位准确。

## CT：MEGANet / EGA

- 论文：*MEGANet: Multi-Scale Edge-Guided Attention Network for Weak Boundary Polyp Segmentation*。
- 出处：WACV 2024，pp.7985–7994。
- 论文：https://openaccess.thecvf.com/content/WACV2024/html/Bui_MEGANet_Multi-Scale_Edge-Guided_Attention_Network_for_Weak_Boundary_Polyp_Segmentation_WACV_2024_paper.html
- 作者仓库：https://github.com/UARK-AICV/MEGANet
- 核验 commit：`8ae378def060071c3bca5a3f85f22f58dd26559e`
- 机制文件：`MEGANet-ResNet/lib/modules.py`，以及调用方式 `lib/EGANet.py`。

保留：高斯滤波/零插值重建的拉普拉斯金字塔；`1-sigmoid(pred)` 反向分支；预测拉普拉斯分支；图像细节分支；三路拼接卷积、空间门控、残差和 CBAM。保留 EGA 的 BN，不替换 baseline 解码器的 GN。

适配：粗预测来自 PET 区域头；图像细节来自同步增强的 CT；CT 先逆转现有 ImageNet/CIPA 归一化；金字塔按特征步长4/8/16/32匹配层级（原模型各EGA复用一个边缘层）；移除 `.cuda()` 硬编码；极小图像使用 replicate padding；低通道 CBAM 保留至少一个隐藏通道；外部 `U_CT(EGA(c)-c)` 零初始化后接回 `C+P`。

## 数值核对与边界

在 CPU、FP32、16通道、batch2、尺寸8/16/32的有效整除网格中，将官方与适配模块置为eval并映射相同权重，分别比较 DSSA 核心与 EGA 核心。MEGANet原实现的高斯核默认CUDA开关仅在核对进程内改为CPU；未改运算公式。六组比较的最大绝对误差均为0（容差 `atol=2e-6, rtol=1e-5`）。该核对不覆盖新增的上下文池化、区域头、零初始化封装、非整除填充或完整论文模型。

核验时上述两个仓库均未发现明确的 LICENSE 文件。这里保留来源与作者归属，不将来源代码擅自标为 MIT/Apache 等许可证；仓库公开可读不等于许可条款已明确。

## 融合接口

```python
from models.components.asymmetric_fusion import AsymmetricFusion

fusion = AsymmetricFusion(norm_mode='imagenet')
# ct_feats, pet_feats: aligned lists, channels [64,128,320,512]
fused, region_logits = fusion(ct_feats, pet_feats, augmented_normalized_ct)
# Missing：不调用fusion/PET编码器，沿用CT特征解码。
```

整个模块不依赖 GT、不读取原型库/病例统计、不使用训练期专属信息推理。默认区域头只通过主分割训练，不能把其 sigmoid 直接解释成已校准的肿瘤概率。边缘分支同样只提供细节线索，不能保证区分肿瘤、血管或噪声。
