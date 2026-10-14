"""Four-stage adapter around ConTriNet's *unchanged* official MFM classes.

Author width stays 64. The author features, not the external residual output,
propagate shallow -> deep. No detach, triple decoder, or extra loss is added.
"""
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.add_fusion import AddFusion
from models.components.contrinet_source import MFM, MFM_0

STAGE_CHANNELS = (64, 128, 320, 512)
MFM_WIDTH = 64
OUTPUT_MODES = ('residual', 'replace')


class ConTriNetMFMFusion(nn.Module):
    """Return four decoder-ready tensors, matching aligned CT feature shapes.

    residual (default): C + P + zero-initialized Projection(author_MFM).
    replace: Projection(author_MFM), without an external C+P skip.
    cross_scale=False feeds zeros as MFM's third input: same parameterization,
    with the shallow-to-deep information path removed for causal ablation.
    enabled=False constructs no parameters and delegates to original AddFusion.
    """

    def __init__(self, enabled: bool = True, cross_scale: bool = True,
                 output_mode: str = 'residual') -> None:
        super().__init__()
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f'output_mode must be one of {OUTPUT_MODES}')
        self.enabled = bool(enabled)
        self.cross_scale = bool(cross_scale)
        self.output_mode = output_mode
        self.add = AddFusion()
        if self.enabled:
            # MFM_0's raw skip requires in_dim == out_dim. S1 is already 64.
            self.first = MFM_0(MFM_WIDTH, MFM_WIDTH)
            self.stages = nn.ModuleList(MFM(c, MFM_WIDTH)
                                        for c in STAGE_CHANNELS[1:])
            self.output_proj = nn.ModuleList(
                nn.Conv2d(MFM_WIDTH, c, 1, bias=False)
                if output_mode == 'residual' or c != MFM_WIDTH else nn.Identity()
                for c in STAGE_CHANNELS)
            if output_mode == 'residual':
                for projection in self.output_proj:
                    nn.init.zeros_(projection.weight)

    @staticmethod
    def _validate(ct_feats: Sequence[torch.Tensor],
                  pet_feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('ConTriNet MFM expects exactly four CT and PET scales')
        aligned_pet = []
        for i, (ct, pet, channels) in enumerate(zip(ct_feats, pet_feats, STAGE_CHANNELS)):
            if ct.ndim != 4 or pet.ndim != 4:
                raise ValueError(f'S{i+1}: expected BCHW tensors')
            if ct.shape[0] != pet.shape[0] or ct.shape[0] != ct_feats[0].shape[0]:
                raise ValueError(f'S{i+1}: inconsistent batch size')
            if ct.shape[1] != channels or pet.shape[1] != channels:
                raise ValueError(f'S{i+1}: aligned channel count must be {channels}')
            if ct.device != pet.device or ct.dtype != pet.dtype:
                raise ValueError(f'S{i+1}: CT/PET device and dtype must match')
            if ct.device != ct_feats[0].device or ct.dtype != ct_feats[0].dtype:
                raise ValueError('All four scales must share device and dtype')
            if not ct.is_floating_point() or min(*ct.shape, *pet.shape) <= 0:
                raise ValueError(f'S{i+1}: expected nonempty floating-point tensors')
            if i and any(a > b for a, b in zip(ct.shape[-2:], ct_feats[i-1].shape[-2:])):
                raise ValueError('Feature order must be shallow-to-deep (S1 to S4)')
            if not torch.isfinite(ct).all() or not torch.isfinite(pet).all():
                raise RuntimeError(f'[NaN/Inf] ConTriNet MFM input S{i+1}')
            if pet.shape[-2:] != ct.shape[-2:]:
                pet = F.interpolate(pet, size=ct.shape[-2:], mode='bilinear',
                                    align_corners=False)
            aligned_pet.append(pet)
        return aligned_pet

    def forward(self, ct_feats: Sequence[torch.Tensor],
                pet_feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if not self.enabled:
            return self.add(ct_feats, pet_feats)
        pet_feats = self._validate(ct_feats, pet_feats)
        result = []
        previous = None
        for i, (ct, pet) in enumerate(zip(ct_feats, pet_feats)):
            if i == 0:
                author_feature = self.first(ct, pet)
            else:
                # Exact source maxpool(2,2) for dyadic pyramid sizes; adaptive
                # max pooling only handles odd-sized encoder outputs safely.
                size = ct.shape[-2:]
                if tuple(previous.shape[-2:]) == tuple(2 * n for n in size):
                    context = F.max_pool2d(previous, kernel_size=2, stride=2)
                elif previous.shape[-2:] == size:
                    context = previous
                else:
                    context = F.adaptive_max_pool2d(previous, size)
                if not self.cross_scale:
                    context = torch.zeros_like(context)
                author_feature = self.stages[i-1](ct, pet, context)
            previous = author_feature
            increment = self.output_proj[i](author_feature).to(dtype=ct.dtype)
            fused = ct + pet + increment if self.output_mode == 'residual' else increment
            if not torch.isfinite(fused).all():
                raise RuntimeError(f'[NaN/Inf] ConTriNet MFM output S{i+1}')
            result.append(fused)
        return result
