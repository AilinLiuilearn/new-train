"""RoadFormer+ GFRM submodule transplant for aligned 2D CT/PET features.

Author source: LiJiahang617/Road-Former, commit
f675a3467cb168ebc727648390c304279bbcb079,
mmseg_custom/models/necks/roadformer2neck.py (GFFM, Scale, FFRM).
See third_party/RoadFormer/NOTICE.md for provenance and explicit adaptations.

GFRM = GFFM -> FFRM. Source outputs concatenated 2D channels. The only
segmentation-interface bridge here splits that output and adds its two halves.
No extra raw C/P skip, spatial sampling, reconstruction, or auxiliary loss.
Zero cross scales do NOT make enabled GFRM equal to raw addition: spatial
LayerNorm and FFRM remain active. Disabled mode is the exact Add control.
"""
from contextlib import nullcontext
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Scale(nn.Module):
    def __init__(self, scale: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class GFFM(nn.Module):
    """Author's bidirectional channel interaction, with spatial LayerNorm.

    Attention is B x D x D, not B x HW x HW. Unscaled logits, separate
    directional softmaxes, raw Q/K/V, zero Scale parameters are retained.
    The bmm/softmax run in float32 under AMP to avoid half overflow; no
    temperature, cosine normalization, or clipping changes the equation.
    """
    def __init__(self, feat_scale: tuple[int, int]):
        super().__init__()
        self.feat_scale = tuple(int(x) for x in feat_scale)
        self.gammax = Scale(0.0)
        self.gammay = Scale(0.0)
        self.norm = nn.LayerNorm(self.feat_scale[0] * self.feat_scale[1])

    def forward(self, x: Tensor, return_details: bool = False):
        c, p = x.chunk(2, dim=1)
        b, d, h, w = c.shape
        if (h, w) != self.feat_scale:
            raise ValueError(f'GFFM spatial LN expects {self.feat_scale}, got {(h,w)}')
        # Float64 remains available for source parity tests; AMP uses FP32.
        dtype = torch.float64 if c.dtype == torch.float64 else torch.float32
        context = (torch.autocast(device_type=c.device.type, enabled=False)
                   if c.device.type in ('cpu', 'cuda') else nullcontext())
        with context:
            cf = c.flatten(2).to(dtype)
            pf = p.flatten(2).to(dtype)
            ac = F.softmax(torch.bmm(cf, pf.transpose(1, 2)), dim=-1)
            ap = F.softmax(torch.bmm(pf, cf.transpose(1, 2)), dim=-1)
            dc = self.gammax(torch.bmm(ac, pf)).reshape(b, d, h, w).to(c.dtype)
            dp = self.gammay(torch.bmm(ap, cf)).reshape(b, d, h, w).to(p.dtype)
        ce, pe = c + dc, p + dp
        out = self.norm(torch.cat((ce.flatten(2), pe.flatten(2)), dim=1))
        out = out.reshape(b, 2*d, h, w)
        if return_details:
            return out, {'delta_ct': dc, 'delta_pet': dp,
                         'attention_ct': ac, 'attention_pet': ap}
        return out


class FFRM(nn.Module):
    """Source ConvModule(1x1, bias=False, GN32, ReLU) + sigmoid + skip."""
    def __init__(self, channels: int, norm_groups: int = 32):
        super().__init__()
        if channels % norm_groups:
            raise ValueError('FFRM channels must be divisible by GN groups')
        self.conv_atten = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(norm_groups, channels),
            nn.ReLU(inplace=True),
        )
        # Author FFRM.init_weights: caffe2_xavier_init == Kaiming uniform a=1.
        nn.init.kaiming_uniform_(self.conv_atten[0].weight, a=1,
                                mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: Tensor, return_weights: bool = False):
        weights = torch.sigmoid(self.conv_atten(F.avg_pool2d(x, x.shape[-2:])))
        out = x + x * weights
        return (out, weights) if return_weights else out


class RoadFormerGFRMStage(nn.Module):
    def __init__(self, channels: int, feature_size: tuple[int, int]):
        super().__init__()
        self.channels = int(channels)
        self.cross = GFFM(feature_size)
        self.recalibrate = FFRM(2 * self.channels)

    def forward(self, ct: Tensor, pet: Tensor, return_details: bool = False):
        if ct.ndim != 4 or ct.shape != pet.shape or ct.shape[1] != self.channels:
            raise ValueError(f'Expected matching BCHW features with C={self.channels}')
        if ct.device != pet.device or ct.dtype != pet.dtype:
            raise ValueError('CT and PET must share device and dtype')
        if return_details:
            z, details = self.cross(torch.cat((ct, pet), 1), True)
            z, weights = self.recalibrate(z, True)
        else:
            z = self.recalibrate(self.cross(torch.cat((ct, pet), 1)))
        enhanced_ct, enhanced_pet = z.chunk(2, dim=1)
        fused = enhanced_ct + enhanced_pet
        if return_details:
            details.update(ct=ct, pet=pet, enhanced_ct=enhanced_ct,
                           enhanced_pet=enhanced_pet, channel_weights=weights,
                           fused=fused, add=ct+pet)
            return fused, details
        return fused


class RoadFormerGFRMFusion(nn.Module):
    """Four identical structures, independent parameters; shallow -> deep.

    Inputs: two four-element lists BCHW, D=(64,128,320,512), matching
    spatial sizes specified at construction. Outputs: four BCHW features.
    Missing is deliberately handled outside this Full-only experiment.
    Diagnostics are opt-in, detached, and never persisted by normal forward.
    """
    def __init__(self, channels: Sequence[int] = (64,128,320,512),
                 feature_sizes: Sequence[tuple[int,int]] = ((128,128),(64,64),(32,32),(16,16))):
        super().__init__()
        if len(channels) != 4 or len(feature_sizes) != 4:
            raise ValueError('GFRM expects exactly four scales')
        self.stages = nn.ModuleList([RoadFormerGFRMStage(c, hw)
                                    for c, hw in zip(channels, feature_sizes)])

    def forward(self, ct: Sequence[Tensor], pet: Sequence[Tensor],
                return_diagnostics: bool = False, return_maps: bool = False):
        if len(ct) != 4 or len(pet) != 4:
            raise ValueError('Expected four CT and four PET features')
        fused, diagnostics = [], {}
        for i, (block, c, p) in enumerate(zip(self.stages, ct, pet), 1):
            if not (return_diagnostics or return_maps):
                fused.append(block(c, p))
                continue
            out, detail = block(c, p, True)
            fused.append(out)
            with torch.no_grad():
                rms = lambda x: x.detach().float().square().mean().sqrt()
                entropy = lambda a: (-(a * a.clamp_min(1e-12).log()).sum(-1).mean()
                                     / torch.log(a.new_tensor(float(a.shape[-1]))))
                stats = {
                    'gamma_ct': block.cross.gammax.scale.detach().item(),
                    'gamma_pet': block.cross.gammay.scale.detach().item(),
                    'delta_ct_ratio': (rms(detail['delta_ct']) / rms(c).clamp_min(1e-8)).item(),
                    'delta_pet_ratio': (rms(detail['delta_pet']) / rms(p).clamp_min(1e-8)).item(),
                    'attention_ct_entropy': entropy(detail['attention_ct']).item(),
                    'attention_pet_entropy': entropy(detail['attention_pet']).item(),
                    'channel_weight_mean': detail['channel_weights'].float().mean().item(),
                    'fused_vs_add_ratio': (rms(out-(c+p)) / rms(c+p).clamp_min(1e-8)).item(),
                }
                if return_maps:
                    stats['maps'] = {name: detail[name].detach().float().cpu()
                        for name in ('ct','pet','delta_ct','delta_pet','enhanced_ct',
                                     'enhanced_pet','fused','add','channel_weights')}
                diagnostics[f's{i}'] = stats
        return (fused, diagnostics) if (return_diagnostics or return_maps) else fused
