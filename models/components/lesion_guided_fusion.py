"""Four-scale lesion-guided dual-modal residual fusion (ordinary PyTorch only).

Input/output: CT/PET lists in S1..S4 order, BCHW, C=(64,128,320,512).
Spatial PET alignment is identical to AddFusion. Forward returns only features;
forward_with_aux optionally returns native-resolution joint lesion logits.
No targets, encoders, retrieval banks, state caches or implicit Missing route.
Inspired by CMX joint rectification and TFDet target supervision; this is an
adaptation, not their original module. See LESION_GUIDED_SOURCE.md.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

STAGE_CHANNELS = (64, 128, 320, 512)
DEFAULT_WIDTHS = (16, 32, 64, 64)
LOCAL_KERNELS = (3, 5, 7)


def _conv_norm_act(cin: int, cout: int, kernel: int = 1) -> nn.Sequential:
    # <=8 groups and >=2 channels/group, including at 1x1 spatial resolution.
    groups = min(8, max(1, cout // 2))
    while cout % groups:
        groups -= 1
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2,
                                  bias=False), nn.GroupNorm(groups, cout), nn.GELU())


class LocalContext(nn.Module):
    def __init__(self, channels: int, width: int):
        super().__init__()
        self.project = _conv_norm_act(channels, width)
        self.kernels = nn.ModuleList(nn.Conv2d(width, width, k, padding=k // 2,
                                              groups=width) for k in LOCAL_KERNELS)
        self.mix = _conv_norm_act(3 * width, width)

    def forward(self, x: Tensor) -> Tensor:
        x = self.project(x)
        return x + self.mix(torch.cat([conv(x) for conv in self.kernels], dim=1))


class LesionGuidedBlock(nn.Module):
    def __init__(self, channels: int, width: int):
        super().__init__()
        self.ct_context = LocalContext(channels, width)
        self.pet_context = LocalContext(channels, width)
        self.joint = _conv_norm_act(2 * width, width)
        self.mask_head = nn.Sequential(_conv_norm_act(width, width, 3),
                                       nn.Conv2d(width, 1, 1))
        # Independent sigmoid gates: not a two-modality softmax competition.
        self.gate = nn.Sequential(_conv_norm_act(width + 1, width, 3),
                                  nn.Conv2d(width, 2 * width, 1))
        self.out_ct = nn.Conv2d(width, channels, 1)
        self.out_pet = nn.Conv2d(width, channels, 1)
        # Only the final signed projections are zero initialized.
        for projection in (self.out_ct, self.out_pet):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, ct: Tensor, pet: Tensor, *, correction_enabled: bool,
                guidance_enabled: bool, return_aux: bool) -> tuple[Tensor, Tensor | None]:
        base = ct + pet
        if not correction_enabled and not return_aux:
            return base, None
        uc, up = self.ct_context(ct), self.pet_context(pet)
        joint = self.joint(torch.cat((uc, up), dim=1))
        logits = self.mask_head(joint) if (return_aux or (correction_enabled and guidance_enabled)) else None
        if correction_enabled:
            probability = (logits.sigmoid() if guidance_enabled else
                           joint.new_zeros((joint.shape[0], 1, *joint.shape[-2:])))
            gc, gp = self.gate(torch.cat((joint, probability), dim=1)).sigmoid().chunk(2, dim=1)
            base = base + self.out_ct(gc * uc) + self.out_pet(gp * up)
        return base, logits if return_aux else None


class LesionGuidedFusion(nn.Module):
    def __init__(self, enabled: bool = True, correction_enabled: bool = True,
                 guidance_enabled: bool = True, widths: Sequence[int] = DEFAULT_WIDTHS):
        super().__init__()
        for name, value in (('enabled', enabled), ('correction_enabled', correction_enabled),
                            ('guidance_enabled', guidance_enabled)):
            if not isinstance(value, bool):
                raise TypeError(f'{name} must be bool')
        if len(widths) != 4 or any(isinstance(w, bool) or not isinstance(w, int) or w < 2 for w in widths):
            raise ValueError('widths must be four integers >=2 in S1..S4 order')
        self.enabled, self.correction_enabled = enabled, correction_enabled
        self.guidance_enabled, self.widths = guidance_enabled, tuple(widths)
        self.blocks = nn.ModuleList(LesionGuidedBlock(c, w) for c, w in
                                    zip(STAGE_CHANNELS, self.widths)) if enabled else nn.ModuleList()

    def forward_with_aux(self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor],
                         *, return_aux: bool = False) -> tuple[list[Tensor], list[Tensor]]:
        if not isinstance(return_aux, bool):
            raise TypeError('return_aux must be bool')
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('Expected four CT/PET scales in S1..S4 order')
        outputs, auxiliary = [], []
        for i, (ct, pet, channels) in enumerate(zip(ct_feats, pet_feats, STAGE_CHANNELS)):
            if (ct.ndim != 4 or pet.ndim != 4 or ct.shape[:2] != pet.shape[:2]
                    or ct.shape[1] != channels or ct.device != pet.device
                    or ct.dtype != pet.dtype or not ct.is_floating_point()
                    or min(*ct.shape[-2:], *pet.shape[-2:]) < 1):
                raise ValueError(f'S{i + 1}: incompatible BCHW CT/PET tensors; C={channels}')
            if pet.shape[-2:] != ct.shape[-2:]:
                pet = F.interpolate(pet, size=ct.shape[-2:], mode='bilinear', align_corners=False)
            if self.enabled:
                fused, logits = self.blocks[i](ct, pet, correction_enabled=self.correction_enabled,
                                               guidance_enabled=self.guidance_enabled, return_aux=return_aux)
                if logits is not None:
                    auxiliary.append(logits)
            else:
                fused = ct + pet
            if not torch.isfinite(fused).all():
                raise RuntimeError(f'[NaN/Inf] LesionGuidedFusion S{i + 1}')
            outputs.append(fused)
        return outputs, auxiliary

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor]) -> list[Tensor]:
        return self.forward_with_aux(ct_feats, pet_feats)[0]


def auxiliary_lesion_loss(logits: Sequence[Tensor], target: Tensor, *,
                           enabled: bool = True, weight: float = .1) -> tuple[Tensor, dict[str, Tensor]]:
    """FP32 BCE+Dice, upsample each logits map to original GT, average 4 scales.

    The GT is NEVER downsampled to coarse feature grids. Disabled/zero weight
    returns an exact zero without looking at the logits or constructing a graph.
    Uses the baseline BCE+Dice convention (per-sample Dice, smooth=1).
    """
    if not isinstance(enabled, bool):
        raise TypeError('enabled must be bool')
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Auxiliary loss weight must be finite and nonnegative')
    zero = target.new_zeros((), dtype=torch.float32)
    stats = {'aux_raw': zero, 'aux_weighted': zero,
             **{f'aux_s{i + 1}': zero for i in range(4)}}
    if not enabled or weight == 0:
        return zero, stats
    if len(logits) != 4:
        raise ValueError('Auxiliary supervision requires exactly four scale logits')
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError('Expected target [B,1,H,W] or [B,H,W]')
    losses = []
    with torch.autocast(device_type=target.device.type, enabled=False):
        mask = (target.float() >= .5).float()
        for x in logits:
            if (x.ndim != 4 or x.shape[:2] != mask.shape[:2] or x.device != target.device
                    or not x.is_floating_point() or not torch.isfinite(x).all()):
                raise ValueError('Invalid auxiliary logits shape/device/dtype/values')
            resized = F.interpolate(x.float(), size=mask.shape[-2:], mode='bilinear', align_corners=False)
            bce = F.binary_cross_entropy_with_logits(resized, mask)
            probability = resized.sigmoid()
            intersection = (probability * mask).sum((1, 2, 3))
            dice = (1 - (2 * intersection + 1) /
                    (probability.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)).mean()
            losses.append(bce + dice)
        raw = torch.stack(losses).mean()
        total = raw * weight
    stats.update(aux_raw=raw.detach(), aux_weighted=total.detach(),
                 **{f'aux_s{i + 1}': x.detach() for i, x in enumerate(losses)})
    return total, stats
