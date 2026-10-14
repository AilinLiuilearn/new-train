# -*- coding: utf-8 -*-
"""Four-scale, Full-only AMDANet FDAM with the requested CT+PET bypass.

At every scale: delta = Global_Eliminating(*Local_Eliminating(CT, PET));
output = CT + PET + delta. The FDAM itself is unmodified upstream code.
The outer CT+PET bypass is our sole architectural adaptation. It is not an
original AMDANet equation and is not zero initialized or attenuated.
"""
from typing import Sequence

import torch
from torch import Tensor, nn

from models.components.amdanet_fdam import Local_Eliminating, Global_Eliminating


class AMDANetFusion(nn.Module):
    """Accept/return four BCHW tensors in S1 -> S4 order.

    Default input contract for 512x512 images:
    (64,128,128), (128,64,64), (320,32,32), (512,16,16).
    All tensors must have matching device, dtype, batch and paired shape.
    No spatial interpolation or new normalization is performed here.
    """

    def __init__(self, channels=(64, 128, 320, 512),
                 num_heads=(1, 2, 5, 8), sr_ratios=(8, 4, 2, 1)):
        super().__init__()
        if not (len(channels) == len(num_heads) == len(sr_ratios) == 4):
            raise ValueError('AMDANetFusion requires exactly four scales')
        if any(c <= 0 or h <= 0 or r <= 0 or c % h
               for c, h, r in zip(channels, num_heads, sr_ratios)):
            raise ValueError('channels/heads/ratios must be positive; channels divisible by heads')
        self.channels = tuple(channels)
        self.sr_ratios = tuple(sr_ratios)
        self.local = nn.ModuleList([Local_Eliminating(dim=c, reduction=4)
                                    for c in channels])
        self.global_ = nn.ModuleList([
            Global_Eliminating(dim=c, num_heads=h, qkv_bias=True,
                              attn_drop=0., proj_drop=0., sr_ratio=r)
            for c, h, r in zip(channels, num_heads, sr_ratios)
        ])

    def forward(self, ct_feats: Sequence[Tensor],
                pet_feats: Sequence[Tensor]) -> list[Tensor]:
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('AMDANetFusion expects four CT and four PET feature maps')
        outputs = []
        first = ct_feats[0]
        if not isinstance(first, Tensor) or first.ndim != 4:
            raise ValueError('CT S1 must be a BCHW tensor')
        for i, (c, p, dim, ratio) in enumerate(zip(
                ct_feats, pet_feats, self.channels, self.sr_ratios)):
            if not isinstance(c, Tensor) or not isinstance(p, Tensor):
                raise ValueError(f'S{i+1}: CT and PET must be tensors')
            if c.ndim != 4 or p.ndim != 4 or c.shape != p.shape:
                raise ValueError(f'S{i+1}: paired BCHW shapes must match exactly')
            if c.shape[0] < 1 or c.shape[0] != first.shape[0] or c.shape[1] != dim:
                raise ValueError(f'S{i+1}: incorrect batch/channel count (expected channels={dim})')
            if min(c.shape[-2:]) < ratio:
                raise ValueError(f'S{i+1}: spatial dimensions must be >= K/V reduction {ratio}')
            if c.device != p.device or c.device != first.device:
                raise ValueError(f'S{i+1}: all CT/PET features must share a device')
            if c.dtype != p.dtype or c.dtype != first.dtype or not c.is_floating_point():
                raise ValueError(f'S{i+1}: all CT/PET features must share a floating dtype')
            if not torch.isfinite(c).all() or not torch.isfinite(p).all():
                raise RuntimeError(f'[NaN/Inf] AMDANet input S{i+1}')
            local_c, local_p = self.local[i](c, p)
            delta = self.global_[i](local_c, local_p)
            fused = c + p + delta
            if not torch.isfinite(fused).all():
                raise RuntimeError(f'[NaN/Inf] AMDANet output S{i+1}')
            outputs.append(fused)
        return outputs
