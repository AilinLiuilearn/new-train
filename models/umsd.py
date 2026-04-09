# -*- coding: utf-8 -*-
"""UMSD blocks and losses for teacher/student."""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_dwt2d(x: torch.Tensor):
    """
    一级 Haar DWT（固定、无学习参数），内部 float32 以兼容 AMP。
    LL/LH/HL/HH 空间尺寸为 ceil(H/2)×ceil(W/2)。
    """
    orig_dtype = x.dtype
    ctx = torch.amp.autocast("cuda", enabled=False) if x.is_cuda else contextlib.nullcontext()
    with ctx:
        x = x.float()
        b, c, h, w = x.shape
        x = F.pad(x, (0, w % 2, 0, h % 2))
        h2, w2 = x.shape[2], x.shape[3]
        x = x.reshape(b, c, h2 // 2, 2, w2 // 2, 2)
        x00 = x[..., 0, 0]
        x01 = x[..., 0, 1]
        x10 = x[..., 1, 0]
        x11 = x[..., 1, 1]
        s = 0.5
        ll = (x00 + x01 + x10 + x11) * s
        lh = (x00 + x01 - x10 - x11) * s
        hl = (x00 - x01 + x10 - x11) * s
        hh = (x00 - x01 - x10 + x11) * s
    ll, lh, hl, hh = [t.to(orig_dtype) for t in (ll, lh, hl, hh)]
    return ll, lh, hl, hh


class WaveletParallelFusion(nn.Module):
    """
    并行小波支路：用 LH/HL/HH 高频子带经瓶颈 1×1 压成与骨干同通道的残差，再与原始特征相加。
    动机：小目标能量常落在高频子带，与 FFT 低通/高通路径互补；复杂度 O(C^2) 经瓶颈限制。
    """

    def __init__(self, in_ch: int, bottleneck_ratio: int = 4):
        super().__init__()
        mid = max(16, in_ch // bottleneck_ratio)
        self.merge = nn.Sequential(
            nn.Conv2d(in_ch * 3, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, lh, hl, hh = haar_dwt2d(x)
        h, w = x.shape[2], x.shape[3]
        lh = F.interpolate(lh, size=(h, w), mode="bilinear", align_corners=False)
        hl = F.interpolate(hl, size=(h, w), mode="bilinear", align_corners=False)
        hh = F.interpolate(hh, size=(h, w), mode="bilinear", align_corners=False)
        d = torch.cat([lh, hl, hh], dim=1)
        return x + self.scale * self.merge(d)


def _fft_lowpass(x: torch.Tensor, ratio: float = 0.25):
    """Return low/high frequency parts."""
    orig_dtype = x.dtype
    ctx = torch.amp.autocast("cuda", enabled=False) if x.is_cuda else contextlib.nullcontext()
    with ctx:
        x32 = x.float()
        b, c, h, w = x32.shape
        xf = torch.fft.rfft2(x32, norm="ortho")
        h_cut = max(1, int(h * ratio / 2))
        w_cut = max(1, int((w // 2 + 1) * ratio))
        mask = torch.zeros_like(xf)
        mask[..., :h_cut, :w_cut] = 1.0
        if h_cut > 1:
            mask[..., -h_cut + 1 :, :w_cut] = 1.0
        x_low = torch.fft.irfft2(xf * mask, s=(h, w), norm="ortho")
    x_low = x_low.to(orig_dtype)
    x_high = x - x_low
    return x_low, x_high


def hsic_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """HSIC independence loss with adaptive RBF bandwidth."""
    b = x.shape[0]
    if b < 2:
        return x.new_zeros(1).squeeze()
    x = x.reshape(b, -1).float()
    y = y.reshape(b, -1).float()
    dxx = torch.cdist(x, x)
    dyy = torch.cdist(y, y)
    # Median heuristic prevents kernel collapse to near-identity.
    sx = dxx.detach().median().clamp(min=1e-6)
    sy = dyy.detach().median().clamp(min=1e-6)
    if sigma is not None and sigma > 0:
        sx = sx * float(sigma)
        sy = sy * float(sigma)
    kx = torch.exp(-(dxx ** 2) / (2 * sx * sx))
    ky = torch.exp(-(dyy ** 2) / (2 * sy * sy))
    h = torch.eye(b, device=x.device) - 1.0 / b
    val = torch.trace(kx @ h @ ky @ h) / ((b - 1) ** 2)
    return val.to(x.dtype)


def cross_scale_consistency_loss(z_shared_list):
    """Anatomy consistency over adjacent scales."""
    if len(z_shared_list) < 2:
        return z_shared_list[0].new_zeros(1).squeeze()
    total = z_shared_list[0].new_zeros(1).squeeze()
    n = 0
    for i in range(len(z_shared_list) - 1):
        a = z_shared_list[i]
        b = F.interpolate(z_shared_list[i + 1], size=a.shape[-2:], mode="bilinear", align_corners=False)
        # Channel alignment for adjacent scales (e.g. 32 vs 64).
        if a.shape[1] != b.shape[1]:
            target_c = min(a.shape[1], b.shape[1])
            a = F.adaptive_avg_pool1d(a.flatten(2).transpose(1, 2), target_c).transpose(1, 2).view(
                a.shape[0], target_c, a.shape[2], a.shape[3]
            )
            b = F.adaptive_avg_pool1d(b.flatten(2).transpose(1, 2), target_c).transpose(1, 2).view(
                b.shape[0], target_c, b.shape[2], b.shape[3]
            )
        va = F.normalize(F.adaptive_avg_pool2d(a, 1).flatten(1), dim=1)
        vb = F.normalize(F.adaptive_avg_pool2d(b, 1).flatten(1), dim=1)
        total = total + (1.0 - (va * vb).sum(dim=1).mean())
        n += 1
    return total / max(n, 1)


class UMSDBlock(nn.Module):
    """Teacher UMSD block: CT+PET -> shared/specific -> fused."""

    def __init__(self, in_ch: int, low_freq_ratio: float = 0.25, use_wavelet_parallel: bool = True):
        super().__init__()
        self.low_freq_ratio = low_freq_ratio
        self.use_wavelet_parallel = use_wavelet_parallel
        self.wave_ct = WaveletParallelFusion(in_ch) if use_wavelet_parallel else None
        self.wave_pet = WaveletParallelFusion(in_ch) if use_wavelet_parallel else None
        shared_ch = max(1, in_ch // 2)
        specific_ch = max(1, in_ch // 4)
        self.shared_enc = nn.Sequential(
            nn.Conv2d(in_ch * 2, shared_ch, 1, bias=False),
            nn.BatchNorm2d(shared_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_ch, shared_ch, 3, padding=1, groups=shared_ch, bias=False),
            nn.BatchNorm2d(shared_ch),
            nn.ReLU(inplace=True),
        )
        self.ct_enc = nn.Sequential(
            nn.Conv2d(in_ch, specific_ch, 1, bias=False),
            nn.BatchNorm2d(specific_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(specific_ch, specific_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(specific_ch),
            nn.ReLU(inplace=True),
        )
        self.pet_enc = nn.Sequential(
            nn.Conv2d(in_ch, specific_ch, 1, bias=False),
            nn.BatchNorm2d(specific_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(specific_ch, specific_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(specific_ch),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(shared_ch + specific_ch * 2, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_ct, feat_pet):
        if self.wave_ct is not None:
            feat_ct = self.wave_ct(feat_ct)
        if self.wave_pet is not None:
            feat_pet = self.wave_pet(feat_pet)
        ct_low, ct_high = _fft_lowpass(feat_ct, self.low_freq_ratio)
        pet_low, pet_high = _fft_lowpass(feat_pet, self.low_freq_ratio)
        z_shared = self.shared_enc(torch.cat([ct_low, pet_low], dim=1))
        z_ct = self.ct_enc(ct_high)
        z_pet = self.pet_enc(pet_high)
        fused = self.fusion(torch.cat([z_shared, z_ct, z_pet], dim=1))
        return fused, z_shared, z_ct, z_pet


class StudentUMSDBlock(nn.Module):
    """Student UMSD block: CT -> general/specific -> fused."""

    def __init__(self, in_ch: int, low_freq_ratio: float = 0.25, use_wavelet_parallel: bool = True):
        super().__init__()
        self.low_freq_ratio = low_freq_ratio
        self.wave = WaveletParallelFusion(in_ch) if use_wavelet_parallel else None
        half = max(1, in_ch // 2)
        self.general_enc = nn.Sequential(
            nn.Conv2d(in_ch, half, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, 3, padding=1, groups=half, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )
        self.specific_enc = nn.Sequential(
            nn.Conv2d(in_ch, half, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, 3, padding=1, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(half * 2, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_ct):
        if self.wave is not None:
            feat_ct = self.wave(feat_ct)
        ct_low, ct_high = _fft_lowpass(feat_ct, self.low_freq_ratio)
        z_general = self.general_enc(ct_low)
        z_specific = self.specific_enc(ct_high)
        fused = self.fusion(torch.cat([z_general, z_specific], dim=1))
        return fused, z_general, z_specific