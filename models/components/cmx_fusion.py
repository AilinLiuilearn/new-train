"""Four-scale seam for unchanged official CMX modules (see CMX_SOURCE.md).

Default: F_s = official_FFM_s(aligned_CT_s, aligned_PET_s).
Optional FRM is local to this seam; it does not feed the next encoder stage.
FFM already contains its own residuals. Never add CT+PET outside its output.
Only the both-disabled arm equals Add at initialization (and at all times).
"""
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cmx_source import FeatureFusionModule, FeatureRectifyModule

STAGE_CHANNELS = (64, 128, 320, 512)
STAGE_HEADS = (1, 2, 5, 8)
UPSTREAM_COMMIT = 'e251d860aebc2f583a6c4919877e6bebe7f1aff3'
UPSTREAM_BLOB = 'ebe09316222c871a2335adb804576a472b5a1013'


class CMXFusion(nn.Module):
    def __init__(self, ffm_enabled: bool = True, frm_enabled: bool = False):
        super().__init__()
        if not isinstance(ffm_enabled, bool) or not isinstance(frm_enabled, bool):
            raise TypeError('cmx_ffm_enabled and cmx_frm_enabled must be bool')
        self.ffm_enabled, self.frm_enabled = ffm_enabled, frm_enabled
        # Construct all FFMs first: toggling FRM must not shift shared FFM init.
        self.ffms = nn.ModuleList(FeatureFusionModule(dim=c, reduction=1,
            num_heads=h, norm_layer=nn.BatchNorm2d) for c, h in
            zip(STAGE_CHANNELS, STAGE_HEADS)) if ffm_enabled else nn.ModuleList()
        self.frms = nn.ModuleList(FeatureRectifyModule(dim=c, reduction=1,
            lambda_c=.5, lambda_s=.5) for c in STAGE_CHANNELS) if frm_enabled else nn.ModuleList()
        # No additional initialization, normalization, cast, residual or loss.

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor]) -> list[Tensor]:
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('CMXFusion requires CT/PET in S1..S4 order')
        outputs = []
        for i, (ct, pet, channels) in enumerate(zip(ct_feats, pet_feats, STAGE_CHANNELS)):
            if (ct.ndim != 4 or pet.ndim != 4 or ct.shape[:2] != pet.shape[:2]
                    or ct.shape[1] != channels or ct.shape[0] < 1
                    or ct.device != pet.device or ct.dtype != pet.dtype
                    or not ct.is_floating_point()
                    or min(*ct.shape[-2:], *pet.shape[-2:]) < 1):
                raise ValueError(f'S{i + 1}: incompatible CT/PET BCHW tensors, channels={channels}')
            if pet.shape[-2:] != ct.shape[-2:]:
                pet = F.interpolate(pet, size=ct.shape[-2:], mode='bilinear', align_corners=False)
            if self.frm_enabled:
                ct, pet = self.frms[i](ct, pet)
            fused = self.ffms[i](ct, pet) if self.ffm_enabled else ct + pet
            if not torch.isfinite(fused).all():
                raise RuntimeError(f'[NaN/Inf] CMXFusion S{i + 1}')
            outputs.append(fused)
        return outputs
