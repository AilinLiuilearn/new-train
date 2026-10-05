# -*- coding: utf-8 -*-
"""Channel alignment: Conv1x1 + BatchNorm + ReLU per scale (unchanged)."""
import torch.nn as nn


class StageChannelAlign(nn.Module):
    """Align each encoder stage to a fixed decoder-input channel count.

    Structure is deliberately kept as-is (Conv1x1 + BatchNorm + ReLU); this
    refactor does not change normalization.
    """

    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        if len(in_channels_list) != len(out_channels_list):
            raise ValueError('in/out channel lists must have the same length')
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ) for c_in, c_out in zip(in_channels_list, out_channels_list)
        ])

    def forward(self, feats):
        if len(feats) != len(self.proj):
            raise ValueError(
                f'StageChannelAlign expected {len(self.proj)} scales, got {len(feats)}')
        return [proj(feat) for proj, feat in zip(self.proj, feats)]