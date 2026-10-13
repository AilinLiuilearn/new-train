# DRWKV reference and adaptation

DRWKV: Focusing on Object Edges for Low-Light Image Enhancement,
Xuecheng Bai and coauthors, WACV 2026 accepted paper.

Paper: https://arxiv.org/abs/2507.18594
Author repository: https://github.com/Bai-Xuecheng/DRWKV
Inspected Block.py blob: c61504d8b410dd33b740f96c9117a2df58844f6d
Original license: Apache License 2.0, retained in LICENSE.txt in this directory.

models/structure_response_fusion.py adapts CrossAttention, IEL and Scharr from
that source. It preserves their core channel-attention, complete IEL/SAE
expansion/split/double depthwise/tanh-residual/product/projection, and SEE
Scharr/L1/BatchNorm/16-channel bottleneck operations. Changes are explicitly
marked: actual fixed Scharr buffers, FP32 attention/normalization arithmetic,
half-width CT/PET stems, two independent interaction/enhancement directions,
PET local contrast, raw dual residual paths, zero output heads and diagnostics.
It does not copy the full CFEM or RWKV network, use its CUDA extension, import
its original module, or claim an exact reproduction of the low-light task.

PET LRCE is an experimental target-task adaptation, not a DRWKV component.
