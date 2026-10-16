# CMX source provenance

This experiment vendors the author's implementation rather than reimplementing the paper equations.

| Item | Pinned source |
| --- | --- |
| Repository | https://github.com/huaaaliu/RGBX_Semantic_Segmentation |
| Commit | `e251d860aebc2f583a6c4919877e6bebe7f1aff3` |
| Original file | `models/net_utils.py` |
| Local file | `models/components/cmx_source.py` |
| Original/local Git blob | `ebe09316222c871a2335adb804576a472b5a1013` |
| License | MIT, Copyright 2022 Huayao Liu |
| Local license | `third_party/CMX_LICENSE` |
| Original/local license blob | `4c65cea1d1a536da33bd79a1f1e7a86988a63203` |
| Paper | [CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers](https://arxiv.org/abs/2203.04838) |

The complete original file and license are byte-identical, including original whitespace and the source file's lack of a final newline. Verify with:

```bash
git hash-object models/components/cmx_source.py third_party/CMX_LICENSE
```

`models/encoders/dual_segformer.py` at the same upstream commit was inspected for the MiT-B1 configuration: channels `[64,128,320,512]`, heads `[1,2,5,8]`, reduction 1 and BatchNorm2d in FFM. That file is not copied into this project.

## Faithful modules, project-specific integration

`CMXFusion` constructs the unchanged `FeatureFusionModule` at four aligned CT/PET scales. Its CrossAttention, CrossPath, ChannelEmbed, internal residuals, LayerNorm, BatchNorm2d and constructor initialization are retained. There is no additional CT+PET residual around FFM, zero initialization, normalization replacement, precision rewrite, or auxiliary loss. Native FFM initialization does not reproduce Add output.

The optional `FeatureRectifyModule` is also unchanged: reduction 1, lambda_c=lambda_s=0.5, and its native standalone constructor behavior. The wrapper does not add an initialization pass to FRM.

This is an adaptation of official CMX modules to the existing heterogeneous PET/CT model, **not a reproduction of the complete CMX network**. CT ConvNeXtV2-Nano, PET MiT-B1, existing projection aligners and the existing decoder remain. In original CMX, rectified features are fed into subsequent encoder stages. Here optional FRM operates after the existing aligners, immediately before FFM or Add; it does not feed back into either encoder. Default experiments disable FRM.

The only project-specific fusion behavior is four-scale validation, baseline-compatible PET interpolation when spatial shapes differ, optional routing, and finite-output checks. Full real CT+PET is required. Missing/mixed states are rejected before either encoder runs.

Baseline commit: `663f9786122365d8b2774e4497df92bc06b715d5`. All files tracked by that baseline remain unchanged on this branch.
