# -*- coding: utf-8 -*-
"""Anisotropic Directional Contrast MAC for MiT/SegFormer feature stages."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_directional_kernels(size=5):
    """Generate four zero-sum directional center-surround kernels: H/V/D/AD."""
    if size < 3 or size % 2 == 0:
        raise ValueError('Directional kernel size must be odd and >= 3')
    kernels = {}
    center = size // 2

    k = np.full((size, size), -1.0 / (size * size - size), dtype=np.float32)
    k[center, :] = 1.0 / size
    kernels['H'] = k

    k = np.full((size, size), -1.0 / (size * size - size), dtype=np.float32)
    k[:, center] = 1.0 / size
    kernels['V'] = k

    k = np.full((size, size), -1.0 / (size * size - size), dtype=np.float32)
    for i in range(size):
        k[i, i] = 1.0 / size
    kernels['D'] = k

    k = np.full((size, size), -1.0 / (size * size - size), dtype=np.float32)
    for i in range(size):
        k[i, size - 1 - i] = 1.0 / size
    kernels['AD'] = k
    return kernels


def generate_laplacian_kernel(size=3):
    if size < 3 or size % 2 == 0:
        raise ValueError('Laplacian kernel size must be odd and >= 3')
    k = np.full((size, size), -1.0 / (size * size - 1), dtype=np.float32)
    k[size // 2, size // 2] = 1.0
    return k


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, max(1, in_planes // 16), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(max(1, in_planes // 16), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SurroundSuppression(nn.Module):
    """Fixed Difference-of-Gaussians center-surround suppression."""

    def __init__(self, channels, ksize=7, center_sigma=1.0, surround_sigma=3.0):
        super().__init__()
        if ksize < 3 or ksize % 2 == 0:
            raise ValueError('Surround suppression kernel size must be odd and >= 3')
        self.channels = int(channels)
        self.ksize = int(ksize)
        center = self._gaussian_2d(ksize, center_sigma)
        surround = self._gaussian_2d(ksize, surround_sigma)
        dog = center - surround
        dog = dog - dog.mean()
        dog = dog / dog.abs().sum().clamp_min(1e-6)
        self.register_buffer('dog', dog.view(1, 1, ksize, ksize))

    @staticmethod
    def _gaussian_2d(ksize, sigma):
        coords = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * float(sigma) ** 2))
        return kernel / kernel.sum().clamp_min(1e-6)

    def forward(self, x):
        pad = self.ksize // 2
        kernel = self.dog.to(dtype=x.dtype).repeat(x.shape[1], 1, 1, 1)
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        return F.conv2d(x_pad, kernel, groups=x.shape[1])


class ADCMAC(nn.Module):
    """Anisotropic Directional Contrast MAC with residual strength gamma."""

    def __init__(self, inplanes, outplanes=None, one=3, two=5, three=3, scales=4, temperature=0.5, init_gamma=0.1):
        super().__init__()
        outplanes = inplanes if outplanes is None else outplanes
        if outplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        self.scales = scales
        self.spx = outplanes // scales
        self.temperature = temperature
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))
        self.cache_direction_weights = False
        self._last_direction_weights = None
        self.relu = nn.ReLU(inplace=True)

        self.inconv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
        )
        self.conv_h = self._direction_conv(one, dilation=1, direction='H')
        self.conv_v = self._direction_conv(one, dilation=1, direction='V')
        self.conv_d = self._direction_conv(two, dilation=2, direction='D')
        self.conv_ad = self._direction_conv(three, dilation=1, direction='AD', with_bn=False)
        self.conv_ad_edge = nn.Conv2d(
            self.spx, self.spx, three, 1, (three // 2) * 2,
            groups=self.spx, dilation=2, bias=False,
        )
        self._init_depthwise_kernel(self.conv_ad_edge, generate_laplacian_kernel(three))
        self.bn_ad = nn.BatchNorm2d(self.spx)

        hidden = max(outplanes // 4, 16)
        self.dir_agg = nn.Sequential(
            nn.Conv2d(outplanes, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 4, 1, bias=False),
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
        )
        self.suppress = SurroundSuppression(outplanes, ksize=7)
        self.ca = ChannelAttention(outplanes)
        self.sa = SpatialAttention()

    def _direction_conv(self, ksize, dilation, direction, with_bn=True):
        padding = (ksize // 2) * dilation
        conv = nn.Conv2d(self.spx, self.spx, ksize, 1, padding, groups=self.spx, dilation=dilation, bias=False)
        kernel = generate_directional_kernels(ksize)[direction]
        self._init_depthwise_kernel(conv, kernel)
        if with_bn:
            return nn.Sequential(conv, nn.BatchNorm2d(self.spx))
        return conv

    def _init_depthwise_kernel(self, conv, kernel):
        weight = torch.from_numpy(kernel).float().view(1, 1, kernel.shape[0], kernel.shape[1])
        with torch.no_grad():
            conv.weight.copy_(weight.repeat(self.spx, 1, 1, 1))
        conv.weight.requires_grad = False

    def forward(self, x):
        x = self.inconv(x)
        residual = x
        xs = torch.chunk(x, self.scales, dim=1)

        h = self.relu(self.conv_h(xs[0]))
        v = self.relu(self.conv_v(xs[1]))
        d = self.relu(self.conv_d(xs[2]))
        ad = self.relu(self.bn_ad(self.conv_ad(xs[3]) + self.conv_ad_edge(xs[3])))

        directional = torch.cat([h, v, d, ad], dim=1)
        weights = F.softmax(self.dir_agg(directional) / self.temperature, dim=1)
        if self.cache_direction_weights:
            self._last_direction_weights = weights.detach().cpu()
        y = torch.cat([
            h * weights[:, 0:1],
            v * weights[:, 1:2],
            d * weights[:, 2:3],
            ad * weights[:, 3:4],
        ], dim=1)
        y = self.outconv(y)
        y = self.suppress(y)
        y = self.ca(y) * y
        y = self.sa(y) * y
        return self.relu(residual + self.gamma * y)
