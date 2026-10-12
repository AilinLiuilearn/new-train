# -*- coding: utf-8 -*-
"""Unconstrained four-scale CT/PET weighted addition.

F_s = weights[s,0] * CT_s + weights[s,1] * PET_s.
Eight trainable scalar coefficients start at one, reproducing AddFusion.
No sigmoid, softmax, clipping or sign/range constraint. These are global
per-scale parameters, not per-sample/per-pixel confidence estimates.
Missing samples bypass this module in the baseline model.
"""
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedAddFusion(nn.Module):
    def __init__(self, channels: Sequence[int] = (64, 128, 320, 512)):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        if len(self.channels) != 4 or any(c <= 0 for c in self.channels):
            raise ValueError('WeightedAddFusion requires four positive channel counts')
        self.weights = nn.Parameter(torch.ones(4, 2))

    def forward(self, ct_feats: Sequence[torch.Tensor],
                pet_feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('WeightedAddFusion expects four CT and four PET scales')
        fused = []
        for i, (c, p) in enumerate(zip(ct_feats, pet_feats)):
            if c.ndim != 4 or p.ndim != 4:
                raise ValueError(f'WeightedAdd scale {i}: expected BxCxHxW')
            if c.shape[0] != p.shape[0] or c.shape[1] != self.channels[i] \
                    or p.shape[1] != self.channels[i]:
                raise ValueError(f'WeightedAdd scale {i}: batch/channel mismatch')
            if c.device != p.device or c.dtype != p.dtype:
                raise ValueError(f'WeightedAdd scale {i}: device/dtype mismatch')
            if not c.is_floating_point() or not p.is_floating_point():
                raise ValueError(f'WeightedAdd scale {i}: floating-point features required')
            if p.shape[-2:] != c.shape[-2:]:
                p = F.interpolate(p, size=c.shape[-2:], mode='bilinear',
                                  align_corners=False)
            # Scalar parameters participate in autograd; no detach or item here.
            out = self.weights[i, 0] * c + self.weights[i, 1] * p
            if not torch.isfinite(out).all():
                raise RuntimeError(f'[NaN/Inf] WeightedAdd output scale {i}')
            fused.append(out)
        return fused

