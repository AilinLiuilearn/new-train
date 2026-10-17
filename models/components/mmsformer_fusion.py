"""Four-scale adapter around the author's unchanged MMSFormer FusionBlock.

Only integration validation, baseline spatial alignment and an Add control
are added. Native linear/multiscale-convolution/SE computation, BN and
constructor initialization are retained without an external residual/scale.
Missing routing is owned by the parent baseline, before this Full seam.
"""
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mmsformer_source import FusionBlock

STAGE_CHANNELS = (64, 128, 320, 512)
UPSTREAM_COMMIT = 'b5287df092c39e95dd46a06c6ba842d501285619'
UPSTREAM_BLOB = '47a8a834d7275414553c64951e311e0965fcc319'
LICENSE_BLOB = 'ac4f24556b2e04cf30cc4c6fa294b2b0d77c7bf2'


class MMSFormerFusion(nn.Module):
    """Input/output: S1..S4 floating BCHW pairs, channels 64/128/320/512.

    Enabled: native FusionBlock([aligned_CT, aligned_PET], stage_index).
    Disabled: CT+PET, no fusion parameters. Output retains CT spatial shape.
    """

    def __init__(self, enabled: bool = True):
        super().__init__()
        if not isinstance(enabled, bool):
            raise ValueError('mmsformer_enabled must be a bool')
        self.enabled = enabled
        self.block = FusionBlock(list(STAGE_CHANNELS), reduction=16, num_modals=2) if enabled else None

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor]) -> list[Tensor]:
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('MMSFormerFusion requires four CT/PET scales in S1..S4 order')
        outputs = []
        for i, (ct, pet, channels) in enumerate(zip(ct_feats, pet_feats, STAGE_CHANNELS)):
            if (not isinstance(ct, Tensor) or not isinstance(pet, Tensor)
                    or ct.ndim != 4 or pet.ndim != 4
                    or ct.shape[:2] != pet.shape[:2] or ct.shape[1] != channels
                    or ct.device != pet.device or ct.dtype != pet.dtype
                    or not ct.is_floating_point() or ct.shape[0] < 1
                    or min(*ct.shape[-2:], *pet.shape[-2:]) < 1):
                raise ValueError(f'S{i+1}: incompatible CT/PET BCHW features; expected channels={channels}')
            if not torch.isfinite(ct).all() or not torch.isfinite(pet).all():
                raise RuntimeError(f'[NaN/Inf] MMSFormerFusion input S{i+1}')
            if pet.shape[-2:] != ct.shape[-2:]:
                pet = F.interpolate(pet, size=ct.shape[-2:], mode='bilinear', align_corners=False)
            fused = self.block([ct, pet], layer_idx=i) if self.enabled else ct + pet
            if fused.shape != ct.shape or not torch.isfinite(fused).all():
                raise RuntimeError(f'Invalid MMSFormerFusion output S{i+1}')
            outputs.append(fused)
        return outputs
