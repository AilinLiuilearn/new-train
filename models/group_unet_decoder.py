# -*- coding: utf-8 -*-
"""GroupNorm UNet-style decoder (Conv + GroupNorm + ReLU, unchanged)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_layer(channels):
    # Largest power-of-two group count (>=2) dividing the channels.
    groups = 8
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def _decoder_block(in_channels, out_channels, kernel_size=1, stride=1, dilation=1):
    """Conv + GroupNorm + ReLU."""
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        _norm_layer(out_channels),
        nn.ReLU(inplace=True),
    )


class UNetStyleDecoder(nn.Module):
    """Shared decoder for both baselines. Expects exactly four scales."""

    def __init__(self, encoder_channels=(64, 128, 320, 512),
                 decoder_channels=(512, 256, 128, 64),
                 out_channels=1, use_deep_supervision=False):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels
        self.use_deep_supervision = bool(use_deep_supervision)
        self.proj4 = _decoder_block(c4, d4, kernel_size=1)
        self.proj3 = _decoder_block(c3, d3, kernel_size=1)
        self.proj2 = _decoder_block(c2, d2, kernel_size=1)
        self.proj1 = _decoder_block(c1, d1, kernel_size=1)
        self.fuse3 = nn.Sequential(_decoder_block(d4 + d3, d3, kernel_size=3),
                                   _decoder_block(d3, d3, kernel_size=3))
        self.fuse2 = nn.Sequential(_decoder_block(d3 + d2, d2, kernel_size=3),
                                   _decoder_block(d2, d2, kernel_size=3))
        self.fuse1 = nn.Sequential(_decoder_block(d2 + d1, d1, kernel_size=3),
                                   _decoder_block(d1, d1, kernel_size=3))
        self.seg_head = nn.Conv2d(d1, out_channels, kernel_size=1)
        if self.use_deep_supervision:
            self.aux_head_d2 = nn.Conv2d(d2, out_channels, kernel_size=1)
            self.aux_head_d3 = nn.Conv2d(d3, out_channels, kernel_size=1)
            self.aux_head_d4 = nn.Conv2d(d4, out_channels, kernel_size=1)

    def forward(self, features, target_size):
        if len(features) != 4:
            raise ValueError(f'UNetStyleDecoder expects 4 scales, got {len(features)}')
        x1, x2, x3, x4 = features
        d4 = self.proj4(x4)
        s3 = self.proj3(x3)
        d3 = self.fuse3(torch.cat([F.interpolate(
            d4, size=s3.shape[-2:], mode='bilinear', align_corners=False), s3], dim=1))
        s2 = self.proj2(x2)
        d2 = self.fuse2(torch.cat([F.interpolate(
            d3, size=s2.shape[-2:], mode='bilinear', align_corners=False), s2], dim=1))
        s1 = self.proj1(x1)
        d1 = self.fuse1(torch.cat([F.interpolate(
            d2, size=s1.shape[-2:], mode='bilinear', align_corners=False), s1], dim=1))
        logits = self.seg_head(d1)
        final_logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        if not torch.isfinite(final_logits).all():
            raise RuntimeError('[NaN/Inf] decoder logits contain invalid values')
        if not self.use_deep_supervision:
            return {'logits': final_logits}
        return {'logits': final_logits,
                'aux_logits': [self.aux_head_d2(d2), self.aux_head_d3(d3), self.aux_head_d4(d4)]}