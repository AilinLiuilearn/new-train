# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_tf_


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    if act == 'relu6':
        return nn.ReLU6(inplace)
    if act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    if act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    if act == 'gelu':
        return nn.GELU()
    if act == 'hswish':
        return nn.Hardswish(inplace)
    raise NotImplementedError(f'activation {act} not supported')


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k, stride, k // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True),
            )
            for k in kernel_sizes
        ])
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(kernel_sizes)
        self.use_skip_connection = stride == 1

        self.ex_channels = int(in_channels * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(activation, inplace=True),
        )
        self.msdc = MSDC(self.ex_channels, kernel_sizes, stride, activation, dw_parallel=dw_parallel)
        self.combined_channels = self.ex_channels if add else self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if self.use_skip_connection and in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, math.gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        return out


def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
    blocks = [MSCB(in_channels, out_channels, stride, kernel_sizes, expansion_factor, dw_parallel, add, activation)]
    for _ in range(1, n):
        blocks.append(MSCB(out_channels, out_channels, 1, kernel_sizes, expansion_factor, dw_parallel, add, activation))
    return nn.Sequential(*blocks)


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            act_layer(activation, inplace=True),
        )
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.activation = act_layer(activation, inplace=True)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class CAB(nn.Module):
    def __init__(self, in_channels, ratio=16, activation='relu'):
        super().__init__()
        if in_channels < ratio:
            ratio = in_channels
        reduced_channels = in_channels // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool)))
        max_out = self.fc2(self.activation(self.fc1(max_pool)))
        return self.sigmoid(avg_out + max_out)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7, 11)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class MSCAM(nn.Module):
    def __init__(self, channels, kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True, activation='relu6', lgag_ks=3):
        super().__init__()
        self.cab = CAB(channels, activation=activation)
        self.sab = SAB(kernel_size=lgag_ks)
        self.mscb = MSCBLayer(channels, channels, n=1, stride=1, kernel_sizes=kernel_sizes,
                              expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

    def forward(self, x):
        x = self.cab(x) * x
        x = self.sab(x) * x
        x = self.mscb(x)
        return x


class EMCADDecoder(nn.Module):
    def __init__(self, channels=(256, 160, 64, 32), kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super().__init__()
        eucb_ks = 3
        c4, c3, c2, c1 = channels

        self.mscam4 = MSCAM(c4, kernel_sizes, expansion_factor, dw_parallel, add, activation, lgag_ks)
        self.eucb3 = EUCB(in_channels=c4, out_channels=c3, kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag3 = LGAG(F_g=c3, F_l=c3, F_int=c3 // 2, kernel_size=lgag_ks, groups=c3 // 2)
        self.mscam3 = MSCAM(c3, kernel_sizes, expansion_factor, dw_parallel, add, activation, lgag_ks)

        self.eucb2 = EUCB(in_channels=c3, out_channels=c2, kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag2 = LGAG(F_g=c2, F_l=c2, F_int=c2 // 2, kernel_size=lgag_ks, groups=c2 // 2)
        self.mscam2 = MSCAM(c2, kernel_sizes, expansion_factor, dw_parallel, add, activation, lgag_ks)

        self.eucb1 = EUCB(in_channels=c2, out_channels=c1, kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag1 = LGAG(F_g=c1, F_l=c1, F_int=max(1, c1 // 2), kernel_size=lgag_ks, groups=max(1, c1 // 2))
        self.mscam1 = MSCAM(c1, kernel_sizes, expansion_factor, dw_parallel, add, activation, lgag_ks)

    def forward(self, x4, skips):
        x3, x2, x1 = skips

        d4 = self.mscam4(x4)
        d3 = self.eucb3(d4)
        d3 = d3 + self.lgag3(g=d3, x=x3)
        d3 = self.mscam3(d3)

        d2 = self.eucb2(d3)
        d2 = d2 + self.lgag2(g=d2, x=x2)
        d2 = self.mscam2(d2)

        d1 = self.eucb1(d2)
        d1 = d1 + self.lgag1(g=d1, x=x1)
        d1 = self.mscam1(d1)

        return [d4, d3, d2, d1]
