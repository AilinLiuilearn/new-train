# -*- coding: utf-8 -*-
"""CT-dominant PET-guided global LL cross-attention wavelet fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    def forward(self, x):
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh, (h, w)


class HaarIDWT2D(nn.Module):
    def forward(self, ll, lh, hl, hh, output_size=None):
        b, c, h, w = ll.shape
        x = ll.new_zeros(b, c, h * 2, w * 2)
        x[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        x[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
        x[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
        x[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
        if output_size is not None:
            out_h, out_w = output_size
            x = x[:, :, :out_h, :out_w]
        return x


class ConvBNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class PETContextRefiner(nn.Module):
    def __init__(self, channels, reduction=16, init_scale=0.1):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, pet):
        refined = pet * self.channel(pet)
        refined = refined * self.spatial(refined)
        refined = self.mix(refined)
        return pet + self.scale * refined


class PETGlobalKVReducer(nn.Module):
    def __init__(self, channels, sr_ratio=4):
        super().__init__()
        self.sr_ratio = max(1, int(sr_ratio))
        if self.sr_ratio > 1:
            self.reduce = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=self.sr_ratio,
                    stride=self.sr_ratio,
                    groups=channels,
                    bias=False,
                ),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            )
        else:
            self.reduce = nn.Identity()

    def forward(self, pet):
        return self.reduce(pet)


class LLLightGlobalCrossAttention(nn.Module):
    def __init__(self, channels, num_heads=4, sr_ratio=4, attn_ratio=0.25, init_scale=0.1):
        super().__init__()
        attn_channels = max(num_heads, int(channels * attn_ratio))
        attn_channels = max(num_heads, (attn_channels // num_heads) * num_heads)
        self.channels = channels
        self.attn_channels = attn_channels
        self.num_heads = num_heads
        self.head_dim = attn_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.pet_reduce = PETGlobalKVReducer(channels, sr_ratio=sr_ratio)
        self.ll_norm = nn.LayerNorm(channels)
        self.pet_norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, attn_channels, bias=True)
        self.k = nn.Linear(channels, attn_channels, bias=True)
        self.v = nn.Linear(channels, attn_channels, bias=True)
        self.proj = nn.Sequential(
            nn.Linear(attn_channels, channels, bias=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, ll, pet):
        b, c, h, w = ll.shape
        pet = F.interpolate(pet, size=(h, w), mode='bilinear', align_corners=False)
        pet_kv = self.pet_reduce(pet)

        q_tokens = ll.flatten(2).transpose(1, 2)
        kv_tokens = pet_kv.flatten(2).transpose(1, 2)
        q = self.q(self.ll_norm(q_tokens))
        k = self.k(self.pet_norm(kv_tokens))
        v = self.v(self.pet_norm(kv_tokens))

        q = q.view(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_conf = attn.max(dim=-1).values.mean(dim=1).view(b, 1, h, w)
        delta = attn @ v
        delta = delta.transpose(1, 2).contiguous().view(b, h * w, self.attn_channels)
        delta = self.proj(delta).transpose(1, 2).contiguous().view(b, c, h, w)
        delta = self.out(delta)
        self.last_visuals = {
            'attn_conf': attn_conf.detach(),
            'll_delta': delta.detach(),
        }
        return ll + self.gamma * delta


class CTResidualWaveAdapter(nn.Module):
    def __init__(self, channels, init_scale=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.beta = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, ct, wave):
        delta = self.adapter(torch.cat([ct, wave], dim=1))
        return ct + self.beta * delta


class PETGuidedLLGlobalFusionStage(nn.Module):
    def __init__(self, channels, window_size=8, num_heads=4, attn_ratio=0.25, wavelet_ratio=0.25, sr_ratio=4):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()
        self.pet_refine = PETContextRefiner(channels, init_scale=0.1)
        self.ll_cross_attn = LLLightGlobalCrossAttention(
            channels,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            attn_ratio=attn_ratio,
            init_scale=0.1,
        )
        self.ct_adapter = CTResidualWaveAdapter(channels, init_scale=0.1)

    def forward(self, ct, pet):
        ll, lh, hl, hh, output_size = self.dwt(ct)
        pet_refined = self.pet_refine(pet)
        ll_enhanced = self.ll_cross_attn(ll, pet_refined)
        wave = self.idwt(ll_enhanced, lh, hl, hh, output_size=output_size)
        out = self.ct_adapter(ct, wave)
        attn_visuals = getattr(self.ll_cross_attn, 'last_visuals', {})
        self.last_visuals = {
            'ct': ct.detach(),
            'pet': pet.detach(),
            'll': ll.detach(),
            'pet_refined': pet_refined.detach(),
            'll_enhanced': ll_enhanced.detach(),
            'wave': wave.detach(),
            'out': out.detach(),
        }
        self.last_visuals.update(attn_visuals)
        return out


class MultiStagePETWindowWaveletFusion(nn.Module):
    def __init__(
        self,
        encoder_channels,
        window_sizes=(8, 8, 4, 4),
        heads_per_stage=(1, 2, 4, 8),
        attn_ratio=0.25,
        wavelet_ratio=0.25,
        sr_ratios=(4, 4, 2, 1),
    ):
        super().__init__()
        self.stages = nn.ModuleList([
            PETGuidedLLGlobalFusionStage(
                ch,
                window_size=win,
                num_heads=head,
                attn_ratio=attn_ratio,
                wavelet_ratio=wavelet_ratio,
                sr_ratio=sr,
            )
            for ch, win, head, sr in zip(encoder_channels, window_sizes, heads_per_stage, sr_ratios)
        ])

    def forward(self, ct_feats, pet_feats):
        return [stage(ct_f, pet_f) for stage, ct_f, pet_f in zip(self.stages, ct_feats, pet_feats)]

    def get_fusion_visuals(self):
        visuals = {}
        for idx, stage in enumerate(self.stages, start=1):
            stage_visuals = getattr(stage, 'last_visuals', None)
            if stage_visuals:
                visuals[f'fuse{idx}'] = stage_visuals
        return visuals
