# -*- coding: utf-8 -*-
"""Official MAFI followed by the baseline's four-scale elementwise addition.

Source: masurq/EG-CMFNet, commit eed0ad865eea8b0dc0e18db1da11afc062a36b52,
models/new_model/modules.py (MAFI, CrossChannelAttention, CrossSpatialAttention).
Paper: EG-CMFNet, TGRS 2026, DOI 10.1109/TGRS.2026.3680719.

Core operations, sequential updates, sample-standard-deviation convention,
and four zero-initialized unconstrained gamma parameters are retained.
Unused surrounding network/profiling imports are omitted; timm's initializer
name is provided by PyTorch. The wrapper applies the author's Conv2d
initializer, called through the parent network's self.apply in the source.
It aligns spatial sizes as AddFusion does and restores each input dtype
before addition. No Q/K/V, extra loss or normalization is introduced.
Missing routing belongs to the baseline model and bypasses this module.
"""
import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.add_fusion import AddFusion

trunc_normal_ = nn.init.trunc_normal_
MAFI_SOURCE_COMMIT = 'eed0ad865eea8b0dc0e18db1da11afc062a36b52'

class MAFI(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(MAFI, self).__init__()

        self.CCA1 = CrossChannelAttention(in_channels, ratio=ratio)
        self.CCA2 = CrossChannelAttention(in_channels, ratio=ratio)
        self.CSA1 = CrossSpatialAttention(kernel_size=kernel_size)
        self.CSA2 = CrossSpatialAttention(kernel_size=kernel_size)
        self.gamma1 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma3 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma4 = nn.Parameter(torch.zeros(1), requires_grad=True)
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x1,x2):

        x1 = self.gamma1 *(self.CCA1(x2) * x1) + x1
        x2 = self.gamma2 *(self.CCA2(x1) * x2) + x2
        x1 = self.gamma3 *(self.CSA1(x2) * x1) + x1
        x2 = self.gamma4 *(self.CSA2(x1) * x2) + x2

        return x1,x2
class CrossChannelAttention(nn.Module):

    def __init__(self, in_channels, ratio=16):
        super(CrossChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            std_out = torch.std(x, dim=(2, 3), keepdim=True)
            std_out = self.fc(std_out)
            out = avg_out + max_out + std_out
            out = self.sigmoid(out)
            return out
class CrossSpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(CrossSpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        std_out = torch.std(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out, std_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out


class MAFIAddFusion(nn.Module):
    """Four BxCxHxW feature pairs -> four enhanced sums, same decoder API.

    All scales use ratio=16 and kernel=7, with independent parameters.
    At initialization this is exactly AddFusion for finite supported inputs.
    """
    def __init__(self, channels: Sequence[int] = (64, 128, 320, 512)):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        if len(self.channels) != 4 or any(c < 16 for c in self.channels):
            raise ValueError('MAFIAddFusion requires four channel counts >= 16')
        self.blocks = nn.ModuleList([MAFI(c, ratio=16, kernel_size=7)
                                     for c in self.channels])
        self.blocks.apply(MAFI._init_weights)
        self.add = AddFusion()

    def forward(self, ct_feats: Sequence[torch.Tensor],
                pet_feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('MAFIAddFusion expects four CT and four PET scales')
        ct_enhanced, pet_enhanced = [], []
        for i, (c, p, block) in enumerate(zip(ct_feats, pet_feats, self.blocks)):
            if c.ndim != 4 or p.ndim != 4:
                raise ValueError(f'MAFI scale {i}: tensors must be BxCxHxW')
            if c.shape[0] != p.shape[0] or c.shape[1] != self.channels[i] \
                    or p.shape[1] != self.channels[i]:
                raise ValueError(f'MAFI scale {i}: batch/channel mismatch '
                                 f'CT={tuple(c.shape)} PET={tuple(p.shape)}')
            if c.device != p.device or c.dtype != p.dtype:
                raise ValueError(f'MAFI scale {i}: device/dtype mismatch')
            if not c.is_floating_point() or not p.is_floating_point():
                raise ValueError(f'MAFI scale {i}: floating-point features required')
            if p.shape[-2:] != c.shape[-2:]:
                p = F.interpolate(p, size=c.shape[-2:], mode='bilinear',
                                  align_corners=False)
            if c.shape[-2] * c.shape[-1] <= 1:
                raise ValueError(f'MAFI scale {i}: official torch.std requires '
                                 'more than one spatial element')
            ce, pe = block(c, p)
            ct_enhanced.append(ce.to(dtype=c.dtype))
            pet_enhanced.append(pe.to(dtype=p.dtype))
        return self.add(ct_enhanced, pet_enhanced)

