"""Four-scale Add-preserving correction using the official KTB DSCF core.

Source: imcjx/KTB, CVPR 2025, Apache-2.0. Native extracted class bodies
are in dscf_source.py; the exact source and license are archived under
third_party/KTB. This file contains our explicit integration changes:
per-scale geometry/rank, zero output head, FP32 correction and CT+PET skip.
No claim of reproducing the complete KTB PEFT/prompt/self-teaching method.
"""
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .dscf_source import DAttentionMM, DeformMPGBlock

STAGE_CHANNELS = (64, 128, 320, 512)
STAGE_RANKS = (16, 16, 40, 64)  # max(16, channels // 8), always divisible by two heads
UPSTREAM_COMMIT = '75370dd584fb9d8571bced1aabade198dfbdc80f'
UPSTREAM_BLOB = 'e333b8f4f881b1f4846682d4470f305ac0158291'
LICENSE_BLOB = 'ac4f24556b2e04cf30cc4c6fa294b2b0d77c7bf2'
DIAGNOSTIC_KEYS = tuple(f'dscf_delta_ratio_s{i+1}' for i in range(4))


def stage_geometry(image_size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """At 512px: query sizes 128/64/32/16; strides 16/8/4/2.

    Each modality predicts an 8x8 reference sampling grid at 512px. Small
    smoke inputs use stride>=1 and at least two points per spatial axis.
    """
    if type(image_size) is not int or image_size < 64 or image_size % 32:
        raise ValueError('DSCF image_size must be an integer >=64 divisible by 32')
    sizes = tuple(image_size // factor for factor in (4, 8, 16, 32))
    return sizes, tuple(max(1, size // 8) for size in sizes)


class SparseCorrection(DeformMPGBlock):
    """Native DeformMPGBlock forward with integration-specific construction.

    Uses the exact source low-rank projections, two learned sampling grids,
    six grid_sample operations, modality sample weighting, attention with
    relative-position bias and the native internal identity/deform weights.
    Only U_fc1 is zero-initialized. Both input projections and the entire
    attention core keep ordinary source initialization.
    """

    def __init__(self, channels: int, rank: int, size: int, stride: int, stage: int):
        nn.Module.__init__(self)
        self.stage = stage
        self.stride = stride
        self.rank = rank
        self.D_fc1 = nn.Linear(channels, rank)
        self.D_fc2 = nn.Linear(channels, rank)
        self.U_fc1 = nn.Linear(rank, channels)
        self.act = nn.GELU()  # Native constructor field; source forward does not call it.
        self.deform_atten = DAttentionMM(
            dims=rank, q_size=(size, size), stride=stride,
            n_heads=2, n_groups=1, dpr=0., level=stage)
        nn.init.zeros_(self.U_fc1.weight)
        nn.init.zeros_(self.U_fc1.bias)

    def forward(self, ct: Tensor, pet: Tensor) -> Tensor:
        batch, channels, height, width = ct.shape
        # Source grid normalization divides by Hk-1 and Wk-1. Reject a
        # degenerate grid explicitly instead of silently propagating NaNs.
        if min(height, width) < 2 or min((height-1)//self.stride+1, (width-1)//self.stride+1) < 2:
            raise ValueError(f'S{self.stage+1}: spatial size too small for DSCF stride={self.stride}')
        # PyTorch 2.1 CUDA grid_sample/attention precision must not depend on
        # modern torch.amp decorators. Master parameters remain float32.
        with torch.autocast(device_type=ct.device.type, enabled=False):
            correction = super().forward(
                ct.float().flatten(2).transpose(1, 2),
                pet.float().flatten(2).transpose(1, 2), height, width, self.stage)
        return correction.transpose(1, 2).reshape(batch, channels, height, width)


class DSCFResidualFusion(nn.Module):
    """Full seam: four pairs of floating BCHW -> four BCHW CT-shaped outputs.

    F_s = CT_s + aligned_PET_s + DSCF_correction_s.
    Enabled initialization equals Add exactly. Disabled has no parameters.
    Parent model owns Missing/auto routing and never calls this seam for
    absent-PET rows. Tensor geometry is configured from image_size_2d.
    Use AMP with FP32 master weights; model.half()/model.double() unsupported.
    """

    def __init__(self, enabled: bool = True, image_size: int = 512):
        super().__init__()
        if type(enabled) is not bool:
            raise ValueError('dscf_enabled must be a bool')
        self.enabled = enabled
        self.image_size = image_size
        self.stage_sizes, self.strides = stage_geometry(image_size)
        self.blocks = nn.ModuleList([
            SparseCorrection(c, r, size, stride, i)
            for i, (c, r, size, stride) in enumerate(zip(
                STAGE_CHANNELS, STAGE_RANKS, self.stage_sizes, self.strides))
        ]) if enabled else nn.ModuleList()
        self.reset_diagnostics()

    def reset_diagnostics(self) -> None:
        self._diagnostic_sums = None
        self._diagnostic_samples = 0

    def pop_diagnostics(self) -> dict[str, float]:
        """Detached per-sample mean ||delta||2 / max(||CT+PET||2, 1e-6).

        Accumulated only during train mode; neither state_dict nor EMA
        contains these counters. Calling this also clears the accumulator.
        """
        values = ([0.] * 4 if self._diagnostic_sums is None else
                  (self._diagnostic_sums / self._diagnostic_samples).cpu().tolist())
        self.reset_diagnostics()
        return dict(zip(DIAGNOSTIC_KEYS, values))

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor]) -> list[Tensor]:
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('DSCFResidualFusion requires four CT/PET scales in S1..S4 order')
        if self.enabled and self.blocks[0].D_fc1.weight.dtype != torch.float32:
            raise ValueError('DSCF requires FP32 master weights; use autocast, not model.half()/double()')
        outputs, ratios = [], []
        for i, (ct, pet, channels) in enumerate(zip(ct_feats, pet_feats, STAGE_CHANNELS)):
            if (not isinstance(ct, Tensor) or not isinstance(pet, Tensor)
                    or ct.ndim != 4 or pet.ndim != 4
                    or ct.shape[:2] != pet.shape[:2] or ct.shape[1] != channels
                    or ct.device != pet.device or ct.dtype != pet.dtype
                    or ct.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                    or ct.shape[0] < 1 or min(*ct.shape[-2:], *pet.shape[-2:]) < 1):
                raise ValueError(f'S{i+1}: incompatible floating CT/PET BCHW features; channels={channels}')
            if not torch.isfinite(ct).all() or not torch.isfinite(pet).all():
                raise RuntimeError(f'[NaN/Inf] DSCFResidualFusion input S{i+1}')
            if pet.shape[-2:] != ct.shape[-2:]:
                pet = F.interpolate(pet, size=ct.shape[-2:], mode='bilinear', align_corners=False)
            base = ct + pet
            delta = self.blocks[i](ct, pet).to(dtype=base.dtype) if self.enabled else None
            fused = base + delta if self.enabled else base
            if fused.shape != ct.shape or not torch.isfinite(fused).all():
                raise RuntimeError(f'Invalid DSCFResidualFusion output S{i+1}')
            if self.training and self.enabled:
                with torch.no_grad():
                    numerator = delta.detach().float().flatten(1).norm(dim=1)
                    denominator = base.detach().float().flatten(1).norm(dim=1).clamp_min(1e-6)
                    ratios.append((numerator / denominator).sum())
            outputs.append(fused)
        if ratios:
            sums = torch.stack(ratios)
            self._diagnostic_sums = sums if self._diagnostic_sums is None else self._diagnostic_sums + sums
            self._diagnostic_samples += ct_feats[0].shape[0]
        return outputs
