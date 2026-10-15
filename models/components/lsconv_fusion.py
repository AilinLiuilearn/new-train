"""All-scale Add -> official LSConv, with zero-initialized final BN.

Inputs: S1..S4 aligned CT/PET lists, channels (64,128,320,512).
Outputs: four tensors with exactly the CT feature shapes.
The final residual is already in LSConv: never add CT+PET again outside it.
"""
import torch
from torch import nn

from .add_fusion import AddFusion
from .lsconv_ska import BACKENDS, SKA
from .lsconv_source import LSConv

STAGE_CHANNELS = (64, 128, 320, 512)


class LSConvFusion(nn.Module):
    def __init__(self, enabled=True, backend='auto'):
        super().__init__()
        if not isinstance(enabled, bool):
            raise TypeError('enabled must be a bool')
        if backend not in BACKENDS:
            raise ValueError(f'Unsupported SKA backend {backend!r}')
        self.enabled = enabled
        self.backend = backend
        self.add = AddFusion()
        self.blocks = nn.ModuleList()
        if enabled:
            for channels in STAGE_CHANNELS:
                block = LSConv(channels)
                block.ska = SKA(backend)
                nn.init.zeros_(block.bn.weight)
                nn.init.zeros_(block.bn.bias)
                self.blocks.append(block)

    def forward(self, ct_feats, pet_feats):
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('LSConvFusion requires S1..S4 for both modalities')
        for i, (ct, pet, channels) in enumerate(zip(ct_feats, pet_feats, STAGE_CHANNELS)):
            if ct.ndim != 4 or pet.ndim != 4:
                raise ValueError(f'S{i + 1}: expected BCHW tensors')
            if (ct.shape[:2] != pet.shape[:2] or ct.shape[1] != channels
                    or ct.device != pet.device or ct.dtype != pet.dtype):
                raise ValueError(f'S{i + 1}: CT/PET batch, channels, device and dtype must match; C={channels}')
        added = self.add(ct_feats, pet_feats)
        if not self.enabled:
            return added
        out = [block(x) for block, x in zip(self.blocks, added)]
        for i, x in enumerate(out):
            if not torch.isfinite(x).all():
                raise RuntimeError(f'[NaN/Inf] LSConvFusion S{i + 1}')
        return out
