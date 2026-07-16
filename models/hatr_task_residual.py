from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.build_mdt_seg import ConvBNAct


def _valid_group_count(channels):
    for g in range(min(8, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class StageCTProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_valid_group_count(in_channels), in_channels, affine=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TaskResidualTransitionBlock(nn.Module):
    def __init__(self, channels, parent_channels=None):
        super().__init__()
        self.channels = int(channels)
        self.parent_proj = None if parent_channels is None else nn.Conv2d(parent_channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_valid_group_count(channels), channels, affine=True)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.act = nn.GELU()
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')
        if self.parent_proj is not None:
            nn.init.kaiming_normal_(self.parent_proj.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, current_ct_state, parent_hidden=None):
        z = current_ct_state
        if parent_hidden is not None:
            parent = parent_hidden
            if parent.shape[-2:] != z.shape[-2:]:
                parent = F.interpolate(parent, size=z.shape[-2:], mode='bilinear', align_corners=False)
            if self.parent_proj is not None:
                parent = self.parent_proj(parent)
            z = z + parent
        update = self.pwconv(self.act(self.dwconv(self.norm(z))))
        hidden = z + update
        return hidden


class TaskResidualHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


class HierarchicalTaskResidualRecovery(nn.Module):
    def __init__(self, decoder_channels, ct_channels):
        super().__init__()
        d4, d3, d2, d1 = map(int, decoder_channels)
        c1, c2, c3, c4 = map(int, ct_channels)
        self.stage_proj4 = StageCTProjection(c4, d4)
        self.stage_proj3 = StageCTProjection(c3, d3)
        self.stage_proj2 = StageCTProjection(c2, d2)
        self.stage_proj1 = StageCTProjection(c1, d1)
        self.trtb4 = TaskResidualTransitionBlock(d4)
        self.trtb3 = TaskResidualTransitionBlock(d3, parent_channels=d4)
        self.trtb2 = TaskResidualTransitionBlock(d2, parent_channels=d3)
        self.trtb1 = TaskResidualTransitionBlock(d1, parent_channels=d2)
        self.head4 = TaskResidualHead(d4)
        self.head3 = TaskResidualHead(d3)
        self.head2 = TaskResidualHead(d2)
        self.head1 = TaskResidualHead(d1)

    def forward(self, ct_feats):
        c1, c2, c3, c4 = ct_feats
        e4 = self.stage_proj4(c4)
        h4 = self.trtb4(e4)
        r4 = self.head4(h4)
        e3 = self.stage_proj3(c3)
        h3 = self.trtb3(e3, h4)
        r3 = self.head3(h3)
        e2 = self.stage_proj2(c2)
        h2 = self.trtb2(e2, h3)
        r2 = self.head2(h2)
        e1 = self.stage_proj1(c1)
        h1 = self.trtb1(e1, h2)
        r1 = self.head1(h1)
        return [r1, r2, r3, r4], [h1, h2, h3, h4]


class CorrectionAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)
