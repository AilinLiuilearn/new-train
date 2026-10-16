"""Four-scale adapter around unchanged official MIRNet-v2 SKFF.

Each stage takes two floating BCHW tensors (aligned CT, aligned PET),
channels 64/128/320/512, and returns the original weighted sum. No x2,
external residual, extra normalization or reinitialization is applied.
State/missingness is handled before encoding by FullCMXPETCT.
"""
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .skff_source import SKFF

STAGE_CHANNELS = (64, 128, 320, 512)
UPSTREAM_COMMIT = '9014049f051ad714eed8b816f23a571e986a46c7'
UPSTREAM_BLOB = '637424c6073422745d906eb632918460d5209181'


class SKFFFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.skffs = nn.ModuleList(
            SKFF(c, height=2, reduction=8, bias=False) for c in STAGE_CHANNELS)

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor]) -> list[Tensor]:
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('SKFFFusion requires four CT/PET scales in S1..S4 order')
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
            fused = self.skffs[i]([ct, pet])
            if not torch.isfinite(fused).all():
                raise RuntimeError(f'[NaN/Inf] SKFFFusion S{i + 1}')
            outputs.append(fused)
        return outputs
