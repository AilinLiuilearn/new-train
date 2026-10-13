# RoadFormer+ GFRM provenance

Paper: Huang et al., RoadFormer+: Delivering RGB-X Scene Parsing Through
Scale-Aware Information Decoupling and Advanced Heterogeneous Feature Fusion,
IEEE Transactions on Intelligent Vehicles 10(5):3156–3165, 2025.
Online publication 2024. DOI: 10.1109/TIV.2024.3448251.
Paper: https://labsun.org/pub/TIV2025_roadformer.pdf
Official repository: https://github.com/LiJiahang617/Road-Former
Inspected commit: f675a3467cb168ebc727648390c304279bbcb079.
Original file: mmseg_custom/models/necks/roadformer2neck.py
Original blob: 4564f04617a9e71e5ee149a33d5519cf58d0a468.
Reference configuration: configs/roadformer+_mfnet/roadformer2_convnext-b_mfnet-480x640.py
(GN, num_groups=32).

This experiment adapts the **GFRM submodule**, implemented by `GFFM`, `Scale`
and `FFRM` in the author repository. It is not a complete RoadFormer+ reproduction.
Global/local encoders, LFFM, FEIM, shared backbone, Mask2Former and author losses
are not part of this submodule transplant.

Preserved: raw channel Q/K/V, two directional D×D attention matrices,
unscaled dot products, separate row softmaxes, cross-modal values, independent
zero-initialized residual scales, spatial LayerNorm with affine HW parameters,
global average pooling, 1×1 bias-free convolution, GN32, ReLU, sigmoid,
and multiplicative residual recalibration. The source ReLU means the FFRM
sigmoid weights are in [0.5,1), an enhancement rather than a free suppression gate.
FFRM convolution initialization follows the author's `caffe2_xavier_init`
(Kaiming uniform, a=1, fan_in).

Explicit adaptations:
1. Replace OpenMMLab BaseModule/ConvModule dependencies with identical PyTorch
   operators (Conv2d -> GroupNorm -> ReLU), no operator omitted.
2. CT and PET features are supplied by the unchanged existing heterogeneous
   encoders and alignment layers. Four stages share the structure, not weights.
3. Construct spatial LayerNorm shapes from input size and strides 4/8/16/32;
   reject mismatched resolution instead of silently replacing the normalization.
4. Compute bmm/softmax in FP32 under AMP; keep the same mathematical equations.
5. Author GFRM returns 2D concatenated channels. Split and sum those two halves
   to keep the unchanged decoder's D-channel contract, with no learned extra
   projection and no second raw CT/PET skip.
6. Full-only experimental wrapper, optional detached diagnostics/maps.

Do not claim exact Add initialization: GFFM gamma=0 only disables cross messages;
LayerNorm and FFRM remain active. `--gfrm_enabled False` builds genuine AddFusion.
Do not label channel RMS maps as tumor localization or confidence estimates.

No LICENSE file was found in the inspected author repository tree. This notice
records attribution and does not assign a new license to the author's material.
Retain this provenance when sharing the adapted code.
