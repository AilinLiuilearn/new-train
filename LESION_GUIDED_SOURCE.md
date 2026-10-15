# 文献机制与本项目适配边界

本实现对应讨论确认的“病灶证据引导的双模态残差融合”，不是 CMX/TFDet 原模块的完整复现。
没有搬入第三方完整类，也不新增自定义 CUDA/Triton 算子。代码根据已批准的设计独立编写。

## 已核对的作者来源

1. CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers，IEEE T-ITS 2023。
   - 论文：https://arxiv.org/abs/2203.04838
   - 官方代码：https://github.com/huaaaliu/RGBX_Semantic_Segmentation/blob/main/models/net_utils.py
   - 核对时文件 blob：`ebe09316222c871a2335adb804576a472b5a1013`。
   - 核对 `ChannelWeights`、`SpatialWeights`、`FeatureRectifyModule`：双模态共同生成权重，双向残差修正。
2. TFDet: Target-Aware Fusion for RGB-T Pedestrian Detection，IEEE TNNLS，DOI `10.1109/TNNLS.2024.3443455`。
   - 论文：https://arxiv.org/html/2305.16580v4
   - 官方代码：https://github.com/XueZ-phd/TFDet/blob/main/mmdetection/mmdet/models/detectors/my_faster_rcnn_mask_supervised_spatialAttention_v5.py
   - 核对时文件 blob：`d23cfcee415df790288894d4b811a9cadecd1d77`。
   - 核对 `TargetAwareFusion`、mask 预测、Dice+BCE 与余弦通道相关性精炼。

## 保留与改变

- 借鉴 CMX“联合生成修正权重”，本项目改为低维局部空间-通道门控；没有原 CMX 通道注意力/FFM 全局交互，也不向下一个骨干阶段回注特征。
- 借鉴 TFDet“目标监督服务前景/背景辨别”，本项目使用已有肿瘤 GT 监督联合预测头；没有框 mask、DCNv2、Conv3d 配对、全局余弦相关性或相关性最大化损失。
- 本项目独立设计：原 Add 直接路径、分开的 CT/PET 3/5/7 局部上下文、联合病灶图条件门、两个带符号的零初始化输出投影。
- 两个 sigmoid 门独立，不做模态 softmax 竞争、不把一致性等同于正确性、不把门解释为已校准可靠性。
- 输出投影初始化只保证初始输出等于 Add；辅助损失从第一步就会更新上游特征，因此训练轨迹不是 Add。
- 理由来自 Add/CAINet 既有成对诊断：补回前景伴随新增假阳性，小中面积组受损。不能把这些定位结论直接套到未完成逐样本诊断的 LSConv。

源论文不证明本适配一定改善 PET-CT。性能收益、新颖性和 Full/Missing 扩展均需独立实验。
