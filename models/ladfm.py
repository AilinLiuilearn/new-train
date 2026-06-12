from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cmrm import CMRM


class CrossAttention(nn.Module):
    """KTB CrossAttention used inside CrossPath."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}.')
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        b, n, c = x1.shape
        q1 = x1.reshape(b, -1, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(b, -1, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(b, n, c).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(b, n, c).contiguous()
        return x1, x2


class CrossPath(nn.Module):
    """KTB CrossPath for bidirectional cross-modal interaction."""

    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        if num_heads is None:
            num_heads = 1
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


class ChannelEmbed(nn.Module):
    """KTB ChannelEmbed maps token fusion back to feature map."""

    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3,
                      stride=1, padding=1, bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels),
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, h, w):
        b, _, c = x.shape
        x = x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        return self.norm(residual + x)


class FactorAttConvRelPosEnc(nn.Module):
    """ATFuse factorized attention core used by MHCABlock."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}.')
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, minus=False):
        b, n, c = q.shape
        q = self.q(q).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        k_softmax = k.softmax(dim=2)
        k_softmax_t_dot_v = torch.einsum('b h n k, b h n v -> b h k v', k_softmax, v)
        factor_att = torch.einsum('b h n k, b h k v -> b h n v', q, k_softmax_t_dot_v)
        x = v - factor_att if minus else factor_att
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ATFuse MLP used by MHCABlock."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MHCABlock(nn.Module):
    """ATFuse MHCABlock for factorized long-range enhancement."""

    def __init__(self, dim, num_heads=8, mlp_ratio=3, qkv_bias=True, qk_scale=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.factoratt_crpe = FactorAttConvRelPosEnc(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.norm2 = norm_layer(dim)

    def forward(self, q, k, v, minus=False):
        b, c, h, w = q.shape
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        x = q + self.factoratt_crpe(q, k, v, minus)
        cur = self.norm2(x)
        x = x + self.mlp(cur)
        return x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


class LADFM(nn.Module):
    """
    Lesion-Aware Dynamic Fusion Module.

    The module follows KTB's PredictorLG style for token reliability, KTB's
    CrossPath/ChannelEmbed for cross-modal fusion, and ATFuse MHCABlock for
    factorized contextual enhancement.
    """

    def __init__(self, dim, num_heads=8, reduction=1, use_mhca=True):
        super().__init__()
        self.dim = dim
        self.use_mhca = use_mhca
        gate_hidden = max(dim // 4, 8)
        self.ct_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        self.pet_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction)
        self.mhca = MHCABlock(dim, num_heads=num_heads, mlp_ratio=2) if use_mhca else nn.Identity()
        self.res_weight = nn.Parameter(torch.tensor(0.15))
        self._last_aux = None

    def forward(self, ct_r, pet_r):
        b, c, h, w = ct_r.shape
        ct_tokens = ct_r.flatten(2).transpose(1, 2)
        pet_tokens = pet_r.flatten(2).transpose(1, 2)

        g_ct = self.ct_gate(ct_tokens)
        g_pet = self.pet_gate(pet_tokens)
        g_weights = F.softmax(torch.cat([g_ct, g_pet], dim=-1), dim=-1)
        w_ct = g_weights[:, :, 0:1].transpose(1, 2).reshape(b, 1, h, w)
        w_pet = g_weights[:, :, 1:2].transpose(1, 2).reshape(b, 1, h, w)

        ct_gated = ct_r * w_ct
        pet_gated = pet_r * w_pet
        ct_t = ct_gated.flatten(2).transpose(1, 2)
        pet_t = pet_gated.flatten(2).transpose(1, 2)
        ct_cross, pet_cross = self.cross(ct_t, pet_t)
        fused = self.channel_emb(torch.cat([ct_cross, pet_cross], dim=-1), h, w)
        enhanced = self.mhca(fused, pet_r, ct_r, minus=False) if self.use_mhca else fused
        out = enhanced + torch.sigmoid(self.res_weight) * (ct_r + pet_r)

        self._last_aux = {
            'w_ct': w_ct.detach(),
            'w_pet': w_pet.detach(),
            'gate_mean_ct': w_ct.detach().mean(),
            'gate_mean_pet': w_pet.detach().mean(),
        }
        return out


class PETCTFusionPipeline(nn.Module):
    """Four-stage CMRM + LADFM fusion pipeline for light U-Net decoder."""

    def __init__(self, ct_channels, pet_channels, out_channels,
                 num_heads=(1, 2, 4, 8), reduction=1, lambda_c=0.5, lambda_s=0.5,
                 use_mhca=(False, False, True, True)):
        super().__init__()
        if not (len(ct_channels) == len(pet_channels) == len(out_channels) == 4):
            raise ValueError('PETCTFusionPipeline expects 4 feature stages.')
        self.cmrm = nn.ModuleList([
            CMRM(ct_ch, pet_ch, out_ch, reduction=reduction, lambda_c=lambda_c, lambda_s=lambda_s)
            for ct_ch, pet_ch, out_ch in zip(ct_channels, pet_channels, out_channels)
        ])
        self.ladfm = nn.ModuleList([
            LADFM(out_ch, num_heads=num_heads[i], reduction=reduction, use_mhca=use_mhca[i])
            for i, out_ch in enumerate(out_channels)
        ])
        self._last_aux = {}

    def forward(self, ct_feats, pet_feats):
        fused = []
        aux = {}
        for idx, (cmrm, ladfm, ct_feat, pet_feat) in enumerate(zip(self.cmrm, self.ladfm, ct_feats, pet_feats), start=1):
            ct_r, pet_r = cmrm(ct_feat, pet_feat)
            fused.append(ladfm(ct_r, pet_r))
            aux[f'stage{idx}'] = ladfm._last_aux
        self._last_aux = aux
        return fused

    def get_fusion_visuals(self):
        if not self._last_aux:
            return {}
        visuals = {}
        for stage, info in self._last_aux.items():
            if not info:
                continue
            visuals[f'{stage}_w_ct'] = info['w_ct']
            visuals[f'{stage}_w_pet'] = info['w_pet']
            visuals[f'{stage}_gate_mean_ct'] = info['gate_mean_ct']
            visuals[f'{stage}_gate_mean_pet'] = info['gate_mean_pet']
        return visuals
