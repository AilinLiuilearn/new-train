"""MMTM-inspired channel calibration followed by baseline addition.

Adapted from Amirreza Shaban's MIT-licensed MMTM implementation; the exact
source and license are preserved under third_party/MMTM. This is NOT an
unchanged MMTM reproduction: independent sigmoid scaling becomes bounded,
zero-initialized residual scaling, with a smaller hidden layer and Add output.

Contract: four floating BCHW CT/PET feature pairs, channels 64/128/320/512.
PET is spatially aligned exactly as in AddFusion. Output has CT shape/dtype.
Missing routing belongs to the parent model and must bypass this module.
"""
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

STAGE_CHANNELS = (64, 128, 320, 512)
UPSTREAM_COMMIT = '1c81cfefad5532cfb39193b8af3840ac3346e897'
UPSTREAM_BLOB = '513b906450288beb9fef21f4d97b3d0e4c368e21'
LICENSE_BLOB = '164301f9e9fe36407082079d91a8ec9071c1ab75'


def validate_settings(reduction=8, strength=0.25):
    if isinstance(reduction, bool) or not isinstance(reduction, int) or reduction < 1:
        raise ValueError('channel reduction must be a positive integer')
    if isinstance(strength, bool) or not math.isfinite(float(strength)) or not 0 < strength < 1:
        raise ValueError('channel strength must be finite and strictly between 0 and 1')


class JointChannelCalibration(nn.Module):
    """Separate modality descriptors, shared squeeze, independent excitation."""

    def __init__(self, channels: int, reduction: int = 8, strength: float = 0.25):
        super().__init__()
        validate_settings(reduction, strength)
        self.channels = channels
        self.strength = float(strength)
        hidden = max(channels // reduction, 16)
        # Borrowed MMTM mechanism: mean each modality, concatenate, shared
        # squeeze + ReLU, two modality-specific linear excitation heads.
        self.fc_squeeze = nn.Linear(2 * channels, hidden)
        self.relu = nn.ReLU()
        self.fc_ct = nn.Linear(hidden, channels)
        self.fc_pet = nn.Linear(hidden, channels)
        for head in (self.fc_ct, self.fc_pet):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def channel_weights(self, ct: Tensor, pet: Tensor) -> tuple[Tensor, Tensor]:
        # Accumulate large feature maps in FP32 under AMP; retain FP64 when
        # explicitly using a double-precision module. No detach on this path.
        pool_dtype = torch.float32 if ct.dtype in (torch.float16, torch.bfloat16) else ct.dtype
        descriptors = [x.mean(dim=(-2, -1), dtype=pool_dtype) for x in (ct, pet)]
        hidden = self.relu(self.fc_squeeze(torch.cat(descriptors, dim=1)))
        weights = []
        for head in (self.fc_ct, self.fc_pet):
            logits = head(hidden)
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            weight = 1.0 + self.strength * torch.tanh(logits)
            weights.append(weight.to(dtype=ct.dtype).view(ct.shape[0], self.channels, 1, 1))
        return weights[0], weights[1]

    def forward(self, ct: Tensor, pet: Tensor) -> Tensor:
        wc, wp = self.channel_weights(ct, pet)
        return wc * ct + wp * pet


class ChannelCalibrationFusion(nn.Module):
    """All four scales, both modalities. Disabled mode has zero parameters."""

    def __init__(self, enabled: bool = True, reduction: int = 8, strength: float = 0.25):
        super().__init__()
        validate_settings(reduction, strength)
        if not isinstance(enabled, bool):
            raise ValueError('enabled must be a bool')
        self.enabled = enabled
        self.blocks = nn.ModuleList(
            JointChannelCalibration(c, reduction, strength) for c in STAGE_CHANNELS
        ) if enabled else nn.ModuleList()

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Sequence[Tensor]) -> list[Tensor]:
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('ChannelCalibrationFusion requires S1..S4 for CT and PET')
        outputs = []
        for i, (ct, pet, channels) in enumerate(zip(ct_feats, pet_feats, STAGE_CHANNELS)):
            if (not isinstance(ct, Tensor) or not isinstance(pet, Tensor)
                    or ct.ndim != 4 or pet.ndim != 4
                    or ct.shape[:2] != pet.shape[:2] or ct.shape[1] != channels
                    or ct.device != pet.device or ct.dtype != pet.dtype
                    or not ct.is_floating_point() or ct.shape[0] < 1
                    or min(*ct.shape[-2:], *pet.shape[-2:]) < 1):
                raise ValueError(f'S{i + 1}: incompatible CT/PET BCHW tensors; expected channels={channels}')
            if not torch.isfinite(ct).all() or not torch.isfinite(pet).all():
                raise RuntimeError(f'[NaN/Inf] ChannelCalibrationFusion input S{i + 1}')
            if pet.shape[-2:] != ct.shape[-2:]:
                pet = F.interpolate(pet, size=ct.shape[-2:], mode='bilinear', align_corners=False)
            out = self.blocks[i](ct, pet) if self.enabled else ct + pet
            if not torch.isfinite(out).all():
                raise RuntimeError(f'[NaN/Inf] ChannelCalibrationFusion output S{i + 1}')
            outputs.append(out)
        return outputs
