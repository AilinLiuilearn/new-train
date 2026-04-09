# -*- coding: utf-8 -*-
"""
PET/CT 深层融合：对齐 FDMF-Net (Chen et al., IEEE TGRS 2025) 的 Stage4 流水线，并做医学任务适配。

参考官方实现 https://github.com/fy-sun/FDMF-Net ：
  - ASD: FrequencyModule — 3×3 空域 + fft2 + shift + 可学习中心低频掩膜 + ifft，|·| 得实部特征
  - ME: FrequencySpaticalFusion — 空间注意力强化高低频分支 + 门控混合（剔除冗余、强化互补）
  - LFGF: Fusion — concat 双模态低频 → Self-Attn → 分别以低频为 Q 对 PET/CT 高频 Cross-Attn → concat + 3×3/1×1

医学适配与自有创新（区别于「整段照搬」）：
  - 双流 backbone 在网外：本模块仅接 h_pet / h_ct（各 C 维，默认 256）。
  - SACA（Spatial CT-Anchored Shared Low）：蒸馏支路专用，空间自适应 w·ct_low + (1-w)·pet_low；
    LFGF 仍严格使用 FDMF 的 pet_le/ct_le 双路输入，便于写「对齐 FDMF + 面向 CT-only 蒸馏的结构性扩展」。
  - 辅助损失：模态内 CLUB + learning_loss；跨模态 MINE 低频，与官方 CE + α·MI − β·low_MI 同号约定。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


# ------------------------- ASD: FrequencyModule (aligned with FDMF) -------------------------


class FDMFFrequencyModule(nn.Module):
    """幅度谱域解耦：与 FDMF frequency_modules.FrequencyModule 一致。"""

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        rdim = self._reduction_dim(dim)
        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, rdim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(rdim, 2, 1, bias=False),
        )
        self.alpha_h = nn.Parameter(torch.tensor(0.5))
        self.alpha_w = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _reduction_dim(dim):
        if dim < 8:
            return max(2, dim)
        log_dim = math.log2(dim)
        return max(2, int(dim // log_dim))

    @staticmethod
    def _shift(x):
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    @staticmethod
    def _unshift(x):
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def forward(self, x):
        x = self.conv1(x)
        b, c, h, w = x.shape
        mask = torch.zeros(b, c, h, w, device=x.device, dtype=x.dtype)
        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold).sigmoid()
        bh = self.alpha_h * threshold[:, 0, :, :] + (1 - self.alpha_h) * threshold[:, 1, :, :]
        bw = self.alpha_w * threshold[:, 0, :, :] + (1 - self.alpha_w) * threshold[:, 1, :, :]

        for i in range(b):
            h_ = int((h // 2 * bh[i]).round().item())
            w_ = int((w // 2 * bw[i]).round().item())
            h_ = max(1, min(h // 2, h_))
            w_ = max(1, min(w // 2, w_))
            mask[i, :, h // 2 - h_ : h // 2 + h_, w // 2 - w_ : w // 2 + w_] = 1.0

        # FFT 用 float32 更稳（混合精度下）
        x32 = x.float()
        fft = torch.fft.fft2(x32, norm='forward', dim=(-2, -1))
        fft = self._shift(fft)
        fft_high = fft * (1 - mask.float())
        high = self._unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2, -1))
        high = torch.abs(high).to(dtype=x.dtype)

        fft_low = fft * mask.float()
        low = self._unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2, -1))
        low = torch.abs(low).to(dtype=x.dtype)

        mask_vis = mask.mean(dim=1, keepdim=True)
        return high, low, mask_vis


# ------------------------- ME: FrequencySpaticalFusion -------------------------


class FDMFSpatialAttention(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(4, 4 * reduction, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 1, kernel_size),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        x1_max, _ = torch.max(x1, dim=1, keepdim=True)
        x1_mean = torch.mean(x1, dim=1, keepdim=True)
        x2_max, _ = torch.max(x2, dim=1, keepdim=True)
        x2_mean = torch.mean(x2, dim=1, keepdim=True)
        x_cat = torch.cat((x1_mean, x1_max, x2_mean, x2_max), dim=1)
        return self.mlp(x_cat)


class FDMEFrequencySpatialFusion(nn.Module):
    """模态增强 ME：与 FDMF FrequencySpaticalFusion 一致。"""

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.high_sa = FDMFSpatialAttention(reduction=reduction)
        self.low_sa = FDMFSpatialAttention(reduction=reduction)
        self.spatial_gate = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, x, x_high, x_low):
        high_w = self.high_sa(x, x_high)
        low_w = self.low_sa(x, x_low)
        x_high = x + high_w * x_high
        x_low = x + low_w * x_low
        gate = torch.sigmoid(self.spatial_gate(x))
        out = gate * x_high + (1 - gate) * x_low
        return x_high, x_low, out


# ------------------------- LFGF: attention blocks (SegFormer-style, FDMF) -------------------------


class _DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, bias=True, groups=dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        return x


class _MlpDW(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = _DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _SelfAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class _SelfAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _SelfAttn(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _MlpDW(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class _CrossAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_y = norm_layer(dim)
        self.attn = _CrossAttn(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _MlpDW(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x, y, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm_y(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class FDMFLowGuidedFusion(nn.Module):
    """LFGF：与 encoder_agg.Fusion 一致拓扑。"""

    def __init__(self, model_dim=256, num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.concat_fusion = nn.Sequential(
            nn.Conv2d(model_dim * 2, model_dim, 1, stride=1, bias=True),
        )
        self.low_self = _SelfAttnBlock(
            model_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer
        )
        self.cross_x = _CrossAttnBlock(
            model_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer
        )
        self.cross_y = _CrossAttnBlock(
            model_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer
        )
        self.conv = nn.Sequential(
            nn.Conv2d(model_dim * 2, model_dim, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(model_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_dim, model_dim, 1, stride=1, bias=True),
        )

    def forward(self, x_high, x_low, y_high, y_low):
        b, c, h, w = x_high.shape
        low = torch.cat((x_low, y_low), dim=1)
        low = self.concat_fusion(low)
        low = low.flatten(2).transpose(1, 2)
        xh = x_high.flatten(2).transpose(1, 2)
        yh = y_high.flatten(2).transpose(1, 2)

        low = self.low_self(low, h, w)
        xh = self.cross_x(low, xh, h, w)
        yh = self.cross_y(low, yh, h, w)

        xh = xh.transpose(1, 2).reshape(b, c, h, w)
        yh = yh.transpose(1, 2).reshape(b, c, h, w)
        fused = self.conv(torch.cat((xh, yh), dim=1))
        return fused, xh, yh


# ------------------------- CLUB / MINE (official-style) -------------------------


class FDMFCLUBMean(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=None):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hs = hidden_size or max(256, x_dim)
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, int(hs)),
            nn.ReLU(),
            nn.Linear(int(hs), y_dim),
        )

    def _vec(self, t):
        return self.pool(t).flatten(1)

    def forward(self, x_samples, y_samples):
        x_vec, y_vec = self._vec(x_samples), self._vec(y_samples)
        mu = self.p_mu(x_vec)
        positive = -((mu - y_vec) ** 2) / 2.0
        pred = mu.unsqueeze(1)
        y0 = y_vec.unsqueeze(0)
        negative = -((y0 - pred) ** 2).mean(dim=1) / 2.0
        positive = positive.sum(dim=-1)
        negative = negative.sum(dim=-1)
        return (positive - negative).mean()

    def loglikeli(self, x_samples, y_samples):
        x_vec, y_vec = self._vec(x_samples), self._vec(y_samples)
        mu = self.p_mu(x_vec)
        return (-(mu - y_vec) ** 2).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)


class FDMFMINEMean(nn.Module):
    def __init__(self, dim, hidden_size=None):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hs = hidden_size or max(dim, 256)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hs),
            nn.ReLU(),
            nn.Linear(hs, 1),
        )

    def _vec(self, t):
        return self.pool(t).flatten(1)

    def forward(self, x_samples, y_samples):
        x = self._vec(x_samples)
        y = self._vec(y_samples)
        joint = torch.cat([x, y], dim=1)
        t1 = self.net(joint)
        y_perm = y[torch.randperm(y.size(0), device=y.device)]
        marginal = torch.cat([x, y_perm], dim=1)
        t2 = self.net(marginal)
        log_mean_exp = torch.logsumexp(t2, dim=0) - math.log(t2.numel())
        return torch.mean(t1) - log_mean_exp


# ------------------------- 原创：面向 CT-only 部署的蒸馏低频构造（非 FDMF 原文）-------------------------


class SpatialCTAnchoredSharedLow(nn.Module):
    """
    SACA：空间自适应「解剖锚定」共享低频（我们自己的模块）

    动机（PET/CT + 知识蒸馏）：
    - 部署时仅 CT，学生应对齐的「共享解剖语义」在可见模态上应更锚定 CT；
    - 肿瘤随访中 PET 低频仍含功能信息，但在结构与 CT 一致的区域不必强行 50/50 平均。
    做法：用 concat(pet_l, ct_l) 预测逐像素权重 w∈(0,1)，z = w*ct_l + (1-w)*pet_l，再送入蒸馏投影。
    与 FDMF 的「简单 concat 低频进 LFGF」并行：LFGF 仍用原始 pet_le/ct_le，仅蒸馏支路用 z。

    训练期 mask 引导（可选）：GT 肿瘤区域 foreground=1 时，对 logits 施加 −γ·m，使 w 降低、
    (1−w)·pet_l 增大，即**在病灶处更强调 PET 低频（代谢/功能）**；背景仍以网络预测 w 为主。
    推理不传 mask，与训练期无标注时行为一致。
    """

    def __init__(self, dim):
        super().__init__()
        hidden = max(dim // 2, 32)
        self.body = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden, 1, 1, bias=True)
        nn.init.zeros_(self.body[0].weight)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.25)
        # 正值：mask=1 处降低 w → 增大 PET 低频权重
        self.mask_gamma = nn.Parameter(torch.tensor(0.35))

    def forward(self, pet_l, ct_l, mask=None):
        h = self.body(torch.cat([pet_l, ct_l], dim=1))
        logits = self.head(h)
        if mask is not None:
            m = mask.float()
            if m.dim() == 3:
                m = m.unsqueeze(1)
            if m.shape[1] > 1:
                m = m[:, :1]
            m = F.interpolate(m, size=logits.shape[-2:], mode='nearest')
            logits = logits - self.mask_gamma * m
        w = torch.sigmoid(logits)
        return w * ct_l + (1.0 - w) * pet_l


# ------------------------- Top-level: PET/CT Stage4 -------------------------


class MedicalFDMFPETCTFusion(nn.Module):
    """
    PET/CT 深层：FDMF 对齐主干（ASD+ME+LFGF+CLUB/MINE）+ SACA 蒸馏支路（原创）。
    """

    def __init__(
        self,
        dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path=0.05,
        distill_dim=None,
        use_spatial_ct_anchor_low=True,
        use_mask_guide_saca=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim must divide num_heads'
        self.dim = dim
        self.distill_dim = distill_dim or (dim // 2)
        self.use_spatial_ct_anchor_low = use_spatial_ct_anchor_low
        self.use_mask_guide_saca = use_mask_guide_saca

        self.freq_pet = FDMFFrequencyModule(dim)
        self.freq_ct = FDMFFrequencyModule(dim)
        self.me_pet = FDMEFrequencySpatialFusion(dim)
        self.me_ct = FDMEFrequencySpatialFusion(dim)
        self.lfgf = FDMFLowGuidedFusion(
            model_dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop=0.0, attn_drop=0.0, drop_path=drop_path,
        )
        self.low_distill = nn.Conv2d(dim, self.distill_dim, 1, bias=False)
        self.high_distill = nn.Conv2d(dim, self.distill_dim, 1, bias=False)
        self.saca_low = SpatialCTAnchoredSharedLow(dim) if use_spatial_ct_anchor_low else None
        self.res_scale = nn.Parameter(torch.ones(1))

        self.club_pet = FDMFCLUBMean(dim, dim)
        self.club_ct = FDMFCLUBMean(dim, dim)
        self.mine_low = FDMFMINEMean(dim, hidden_size=max(dim, 256))

        self.fdmf_mi_loss = None
        self.fdmf_low_mi_lb = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, h_pet, h_ct, return_separated=False, mask=None):
        pet_h, pet_l, pet_mask = self.freq_pet(h_pet)
        ct_h, ct_l, ct_mask = self.freq_ct(h_ct)

        pet_he, pet_le, _ = self.me_pet(h_pet, pet_h, pet_l)
        ct_he, ct_le, _ = self.me_ct(h_ct, ct_h, ct_l)

        mi = (
            self.club_pet(pet_he, pet_le)
            + self.club_pet.learning_loss(pet_he, pet_le)
            + self.club_ct(ct_he, ct_le)
            + self.club_ct.learning_loss(ct_he, ct_le)
        )
        low_lb = self.mine_low(pet_le, ct_le)
        self.fdmf_mi_loss = mi
        self.fdmf_low_mi_lb = low_lb

        fused, xh, yh = self.lfgf(pet_he, pet_le, ct_he, ct_le)
        fused = (h_pet + h_ct) * 0.5 + self.res_scale * fused

        freq_mask = (pet_mask + ct_mask) * 0.5
        if return_separated:
            if self.saca_low is not None:
                m = mask if (mask is not None and self.use_mask_guide_saca) else None
                low_for_distill = self.saca_low(pet_le, ct_le, mask=m)
            else:
                low_for_distill = (pet_le + ct_le) * 0.5
            low_s = self.low_distill(low_for_distill)
            high_s = self.high_distill((xh + yh) * 0.5)
            return fused, low_s, high_s, pet_mask, ct_mask
        return fused


