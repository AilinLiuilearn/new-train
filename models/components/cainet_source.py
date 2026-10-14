# -*- coding: utf-8 -*-
"""Unmodified CAINet relation blocks from the authors' official source.

Paper: Lv, Liu, Li, Context-Aware Interaction Network for RGB-T Semantic
Segmentation, IEEE TMM 26 (2024), 6348-6360.
Source: https://github.com/YingLv1106/CAINet/blob/4fff413041f1776ce1e588eb4763fe46fca2677a/toolbox/models/cainet.py
Commit: 4fff413041f1776ce1e588eb4763fe46fca2677a
Interv2 is the released CACR block; CRRM is the released GCM block.
Class definitions are copied verbatim; adaptation lives in cainet_fusion.py.
"""
import torch
from torch import nn


class Interv2(nn.Module):
    def __init__(self, in_channels):
        super(Interv2, self).__init__()
        self.N = in_channels // 4
        self.S = in_channels // 2

        self.theta = nn.Conv2d(in_channels, self.N, 1, 1, 0, bias=False)
        self.phi = nn.Conv2d(in_channels, self.S, 1, 1, 0, bias=False)

        self.relu = nn.ReLU()

        self.node_conv = nn.Conv1d(self.N, self.N, 1, 1, 0, bias=False)
        self.channel_conv = nn.Conv1d(self.S, self.S, 1, 1, 0, bias=False)

        self.conv_2 = nn.Conv2d(self.S, in_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        batch, C, H, W = x.size()
        L = H * W

        B = self.theta(x).view(-1, self.N, L) #N*L

        phi = self.phi(x).view(-1, self.S, L)  #S*L
        phi = torch.transpose(phi, 1, 2)       #L*S

        V = torch.bmm(B, phi) / L  #  #N*S

        AV = self.relu(self.node_conv(V))  #N*S
        IV = V+AV
        IAVW = self.relu(self.channel_conv(torch.transpose(IV, 1, 2)))  #S*N

        y = torch.bmm(torch.transpose(B, 1, 2), torch.transpose(IAVW, 1, 2))
        y = y.view(-1, self.S, H, W)
        y = self.conv_2(y)

        return x + y


class CRRM(nn.Module):
    def __init__(self, in_channels):
        super(CRRM, self).__init__()
        self.C = 128
        self.omg = nn.Conv2d(in_channels, self.C, 1, 1, 0, bias=False)  #C*H*W
        self.theta = nn.Conv2d(in_channels, self.C, 1, 1, 0, bias=False)  #C*H*W
        self.ori = nn.Conv2d(in_channels, self.C, 1, 1, 0, bias=False)   #C*H*W

        self.relu = nn.ReLU()

        self.node_conv = nn.Conv1d(self.C, self.C, 1, 1, 0, bias=False)
        self.channel_conv = nn.Conv1d(self.C, self.C, 1, 1, 0, bias=False)


    def forward(self, x):
        batch, C, H, W = x.size()
        L = H * W
        o = self.omg(x)
        omg = o.view(-1, self.C, L) #C*L
        theta = self.theta(x).view(-1, self.C, L) #C*L
        ori = self.ori(x).view(-1, self.C, L)  #C*L

        B = torch.transpose(ori, 1, 2)       #L*C

        V = torch.bmm(theta, B)  #
        A = torch.nn.functional.softmax(V, dim=-1) #CC
        AV = self.node_conv(A)  #C*C
        IV = A+AV
        IAV = self.relu(self.channel_conv(torch.transpose(IV, 1, 2)))  #C*C

        y = torch.bmm(IAV, omg)
        y = y.view(-1, self.C, H, W)

        return o + y

