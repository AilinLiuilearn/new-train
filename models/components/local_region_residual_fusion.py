"""Full-only residual fusion: independent CT local / PET regional processing.

CIPA convolution stems are retained; attention/FFN replace Mamba. Gaussian
feature contrast is a custom hypothesis, not RAC-Net RASFE or HU modeling.
Only the final sum mixes modalities. Non-multiple-of-four grids are explicitly
rejected in v1 (no implicit cropping, resizing, or padding of valid features).
"""
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_FUSION_CONFIG = {
    'channels': [64, 128, 320, 512], 'window_size': 4, 'heads': 4,
    'ffn_ratio': 2, 'use_gaussian': True,
    'gaussian_pairs': [[[3, .6], [5, 1.0]], [[5, 1.0], [7, 1.4]],
                       [[7, 1.4], [9, 2.0]]],
}


def window_partition(x, r=4):
    b, c, h, w = x.shape
    if h % r or w % r:
        raise ValueError('v1 requires H and W divisible by window_size=4')
    return x.reshape(b, c, h//r, r, w//r, r).permute(0, 2, 4, 3, 5, 1).reshape(-1, r*r, c)


def window_reverse(tokens, layout, r=4):
    b, h, w = layout
    c = tokens.shape[-1]
    return tokens.reshape(b, h//r, w//r, r, r, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)


def broadcast_regions(region_map, r=4):
    return region_map.repeat_interleave(r, -2).repeat_interleave(r, -1)


class CIPAConvStem(nn.Module):
    def __init__(self, channels):
        super().__init__()
        def conv(a, b, stride=1, inplace=True):
            return [nn.Conv2d(a, b, 3, stride=stride, padding=1, bias=True),
                    nn.BatchNorm2d(b), nn.ReLU(inplace=inplace)]
        c = channels
        self.inner_convs = nn.Sequential(*conv(c, c//4), *conv(c//4, c, inplace=False))
        self.outer_convs = nn.Sequential(*conv(c, c//16), *conv(c//16, c//8, 2),
                                        *conv(c//8, c//4, 2), *conv(c//4, c, inplace=False))

    def forward(self, C, P):
        return self.inner_convs(C), self.outer_convs(P), (C.shape[0], *C.shape[-2:])


class GaussianScaleContrast(nn.Module):
    def __init__(self, channels, pairs=None):
        super().__init__()
        pairs = DEFAULT_FUSION_CONFIG['gaussian_pairs'] if pairs is None else pairs
        if len(pairs) != 3 or any(len(pair) != 2 for pair in pairs):
            raise ValueError('exactly three Gaussian small/large pairs required')
        self.pairs = []
        kernels = {}
        for pair in pairs:
            names = []
            for size, sigma in pair:
                if int(size) != size or size < 1 or size % 2 != 1 or sigma <= 0:
                    raise ValueError('Gaussian kernel must be odd with positive sigma')
                key = (int(size), float(sigma))
                if key not in kernels:
                    name = f'kernel_{len(kernels)}'
                    coord = torch.arange(size, dtype=torch.float32) - size//2
                    kernel = torch.exp(-coord.square()/(2*sigma*sigma))
                    self.register_buffer(name, kernel/kernel.sum())
                    kernels[key] = name
                names.append(kernels[key])
            self.pairs.append(tuple(names))
        self.project = nn.Conv2d(3*channels, channels, 1)

    def blur(self, x, name):
        kernel = getattr(self, name).to(dtype=x.dtype)
        radius = kernel.numel()//2
        if min(x.shape[-2:]) <= radius:
            raise ValueError('Gaussian reflect padding requires each spatial size > kernel radius')
        c = x.shape[1]
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode='reflect'),
                     kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        return F.conv2d(F.pad(x, (0, 0, radius, radius), mode='reflect'),
                        kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)

    def differences(self, x):
        blurred = {name: self.blur(x, name) for name, _ in self.named_buffers()}
        return [blurred[a]-blurred[b] for a, b in self.pairs]

    def forward(self, x):
        return self.project(torch.cat(self.differences(x), dim=1))


class AttentionBlock(nn.Module):
    """Pre-norm residual self-attention; never requests attention matrices."""
    def __init__(self, channels, heads=4, ffn_ratio=2):
        super().__init__()
        if channels % heads:
            raise ValueError('channels must be divisible by heads')
        self.heads = heads
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3*channels)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, int(channels*ffn_ratio)), nn.GELU(),
                                 nn.Linear(int(channels*ffn_ratio), channels))

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, c//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        # Internal block residuals are separate from the external C/P residuals.
        x = x + self.proj(attn.transpose(1, 2).reshape(b, n, c))
        return x + self.ffn(self.norm2(x))


class LocalAttentionBlock(AttentionBlock):
    def forward(self, c, h, layout):
        u = window_partition(c) + window_partition(h)
        return window_reverse(super().forward(u), layout)


class RegionAttentionBlock(AttentionBlock):
    def forward(self, p, layout):
        b, c, h, w = p.shape
        q = super().forward(p.flatten(2).transpose(1, 2))
        return broadcast_regions(q.transpose(1, 2).reshape(b, c, h, w))


class LocalRegionResidualFusion(nn.Module):
    def __init__(self, channels, window_size=4, heads=4, ffn_ratio=2,
                 use_gaussian=True, gaussian_pairs=None):
        super().__init__()
        if window_size != 4 or channels < 16:
            raise ValueError('v1 retains CIPA width and fixed window_size=4; channels >=16')
        self.channels = channels
        self.use_gaussian = use_gaussian
        self.stem = CIPAConvStem(channels)
        # Keep names/state identical for the no-Gaussian controlled comparison.
        self.contrast = GaussianScaleContrast(channels, gaussian_pairs)
        self.local_block = LocalAttentionBlock(channels, heads, ffn_ratio)
        self.region_block = RegionAttentionBlock(channels, heads, ffn_ratio)
        self.out_ct = nn.Conv2d(channels, channels, 1)
        self.out_pet = nn.Conv2d(channels, channels, 1)
        for layer in (self.out_ct, self.out_pet):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, C, P):
        if C.ndim != 4 or P.ndim != 4 or C.shape != P.shape or C.shape[1] != self.channels:
            raise ValueError('single-scale fusion requires aligned B,C,H,W tensors')
        if C.device != P.device or C.dtype != P.dtype:
            raise ValueError('CT/PET dtype and device must match')
        if C.shape[-2] % 4 or C.shape[-1] % 4:
            raise ValueError('v1 requires H and W divisible by window_size=4')
        c, p, layout = self.stem(C, P)
        h = self.contrast(c) if self.use_gaussian else torch.zeros_like(c)
        z_ct = self.local_block(c, h, layout)
        z_pet = self.region_block(p, layout)
        return (C + self.out_ct(z_ct)) + (P + self.out_pet(z_pet))


class MultiScaleLocalRegionResidualFusion(nn.Module):
    def __init__(self, channels=(64, 128, 320, 512), window_size=4, heads=4,
                 ffn_ratio=2, use_gaussian=True, gaussian_pairs=None):
        super().__init__()
        if len(channels) != 4:
            raise ValueError('exactly four scales required')
        self.blocks = nn.ModuleList([LocalRegionResidualFusion(c, window_size, heads,
            ffn_ratio, use_gaussian, gaussian_pairs) for c in channels])

    def forward(self, ct_feats, pet_feats):
        if len(ct_feats) != 4 or len(pet_feats) != 4:
            raise ValueError('exactly four CT and PET scales required')
        outputs = []
        for block, C, P in zip(self.blocks, ct_feats, pet_feats):
            if C.ndim != 4 or P.ndim != 4 or C.shape[:2] != P.shape[:2]:
                raise ValueError('CT/PET batch and channel must match')
            if C.device != P.device or C.dtype != P.dtype:
                raise ValueError('CT/PET dtype and device must match')
            if C.shape[-2:] != P.shape[-2:]:
                P = F.interpolate(P, size=C.shape[-2:], mode='bilinear', align_corners=False)
            out = block(C, P)
            if not torch.isfinite(out).all():
                raise RuntimeError('nonfinite residual fusion output')
            outputs.append(out)
        return outputs
