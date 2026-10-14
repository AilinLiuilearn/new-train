# -*- coding: utf-8 -*-
"""Switchable source CACR/GCM for aligned Full CT-PET features.

CACR: native Interv2(cat(CT,PET)) at S1/S2/S3/S4, then split and sum.
GCM: native CRRM on spatially aligned S2/S3/S4 joint features; project its
128-channel output and add ONLY to S4. The necessary outer projection is
zero initialized. Both off is exactly the Add baseline.
"""
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.components.add_fusion import AddFusion
from models.components.cainet_source import Interv2, CRRM


class CAINetFusion(nn.Module):
    def __init__(self, cacr_enabled=True, gcm_enabled=True,
                 cacr_stages=(1, 2, 3, 4)):
        super().__init__()
        if not isinstance(cacr_enabled, bool) or not isinstance(gcm_enabled, bool):
            raise ValueError('CACR/GCM flags must be bool values')
        self.cacr_enabled = cacr_enabled
        self.gcm_enabled = gcm_enabled
        # Legacy three-scale layout is only needed to restore saved v1 runs.
        self.cacr_stages = tuple(cacr_stages)
        if self.cacr_stages not in ((1, 2, 3, 4), (2, 3, 4)):
            raise ValueError('CACR stages must be (1,2,3,4), or legacy (2,3,4)')
        self.channels = (64, 128, 320, 512)
        self.add = AddFusion()
        if cacr_enabled:
            self.cacr = nn.ModuleList([
                Interv2(2*self.channels[s-1]) for s in self.cacr_stages])
        if gcm_enabled:
            self.gcm = CRRM(2 * sum(self.channels[1:]))
            # Interface adapter only: retain upstream CRRM's fixed 128 output.
            self.gcm_to_s4 = nn.Conv2d(128, self.channels[3], 1, bias=False)
            nn.init.zeros_(self.gcm_to_s4.weight)

    @staticmethod
    def _relation(module: nn.Module, x: Tensor) -> Tensor:
        # Keep the source equations and parameter dtype unchanged. Accumulate
        # their global products in fp32 under AMP, then restore feature dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = module(x.float())
        return output.to(dtype=x.dtype)

    def forward(self, ct_feats: Sequence[Tensor],
                pet_feats: Sequence[Tensor]) -> list[Tensor]:
        if not self.cacr_enabled and not self.gcm_enabled:
            return self.add(ct_feats, pet_feats)
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('CAINetFusion expects four CT and four PET scales')
        first = ct_feats[0]
        if not isinstance(first, Tensor) or first.ndim != 4:
            raise ValueError('S1 must be a BCHW tensor')
        for i, (c, p, channels) in enumerate(zip(ct_feats, pet_feats, self.channels)):
            if not isinstance(c, Tensor) or not isinstance(p, Tensor):
                raise ValueError(f'S{i+1}: CT/PET must be tensors')
            if c.ndim != 4 or p.ndim != 4 or c.shape != p.shape:
                raise ValueError(f'S{i+1}: paired BCHW shapes must match')
            if c.shape[0] < 1 or c.shape[0] != first.shape[0] or c.shape[1] != channels:
                raise ValueError(f'S{i+1}: batch/channel count mismatch')
            if c.device != p.device or c.device != first.device:
                raise ValueError('All CT/PET features must share a device')
            if c.dtype != p.dtype or c.dtype != first.dtype or not c.is_floating_point():
                raise ValueError('All CT/PET features must share a floating dtype')
            if not torch.isfinite(c).all() or not torch.isfinite(p).all():
                raise RuntimeError(f'[NaN/Inf] CAINet input S{i+1}')
        fused = []
        joint = []
        for i in range(4):
            pair = torch.cat((ct_feats[i], pet_feats[i]), dim=1)
            if self.cacr_enabled and i+1 in self.cacr_stages:
                pair = self._relation(self.cacr[self.cacr_stages.index(i+1)], pair)
            joint.append(pair)
            c, p = pair.chunk(2, dim=1)
            fused.append(c + p)
        if self.gcm_enabled:
            # S3 grid: 32x32 for 512x512 images. No change to original streams.
            size = joint[2].shape[-2:]
            context = torch.cat([
                x if x.shape[-2:] == size else F.interpolate(
                    x, size=size, mode='bilinear', align_corners=False)
                for x in reversed(joint[1:])  # upstream fuse5/fuse4/fuse3 order
            ], dim=1)
            global_context = self._relation(self.gcm, context)
            delta = self.gcm_to_s4(global_context)
            delta = F.interpolate(delta, size=fused[3].shape[-2:],
                                  mode='bilinear', align_corners=False)
            fused[3] = fused[3] + delta
        if any(not torch.isfinite(x).all() for x in fused):
            raise RuntimeError('[NaN/Inf] CAINet fused features')
        return fused
