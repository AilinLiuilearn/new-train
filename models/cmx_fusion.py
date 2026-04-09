# -*- coding: utf-8 -*-
"""
跨模态与解耦相关模块（精简版，对齐当前训练路径）：
- 浅层：FreqSpatialFusionBlock（FSF）
- CT-PET 专用浅层融合：CMFRMAsymmetricCTAnchor + LightweightFFMPETtoCT；可选 ShallowFusionHLInspiredCIM（HLMamba 启发 CIM 桥 + 选择性空间混合）
- ShaSpec / OPD：见下文类

CMX 原版大块已精简；此处为面向 CT–PET 的改进版，非逐行复刻 RGB–X。
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


class FreqSpatialFusionBlock(nn.Module):
    """
    FSF：频域-空间轻量融合 (Frequency-Spatial Fusion)。
    频域：FFT → 可学习掩码混合 CT/PET 谱 → IFFT；空间：PET 提示加权 CT。
    out = x_ct + x_pet + alpha * (x_freq + x_hint)，alpha 初值 0 等价于相加基线。
    return_attn=True 时额外返回 spatial_hint 通道均值 (B,1,H,W)，供 CAGD 蒸馏。
    """

    def __init__(self, dim):
        super().__init__()
        self.freq_mask_conv = nn.Conv2d(dim * 2, dim, 1, bias=True)
        nn.init.zeros_(self.freq_mask_conv.weight)
        nn.init.constant_(self.freq_mask_conv.bias, 0.5)
        self.spatial_conv = nn.Conv2d(dim, dim, 1, bias=True)
        nn.init.zeros_(self.spatial_conv.weight)
        nn.init.zeros_(self.spatial_conv.bias)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x_ct, x_pet, return_attn=False):
        orig_dtype = x_ct.dtype
        amp_ctx = (
            torch.amp.autocast('cuda', enabled=False)
            if x_ct.is_cuda
            else contextlib.nullcontext()
        )
        with amp_ctx:
            xc = x_ct.float()
            xp = x_pet.float()
            F_ct = torch.fft.fft2(xc, norm='ortho')
            F_pet = torch.fft.fft2(xp, norm='ortho')
            amp_ct = F_ct.abs()
            amp_pet = F_pet.abs()
            mask = torch.sigmoid(
                self.freq_mask_conv(torch.cat([amp_ct, amp_pet], dim=1))
            )
            F_mix = mask * F_ct + (1.0 - mask) * F_pet
            x_freq = torch.fft.ifft2(F_mix, norm='ortho').real
        x_freq = x_freq.to(orig_dtype)

        spatial_hint = torch.sigmoid(self.spatial_conv(x_pet))
        x_hint = x_ct * spatial_hint
        x_enhanced = x_freq + x_hint
        out = x_ct + x_pet + self.alpha * x_enhanced

        if return_attn:
            attn_map = spatial_hint.mean(dim=1, keepdim=True)
            return out, attn_map
        return out


# ============ CT–PET：非对称 CM-FRM + 轻量 FFM（仅 PET→CT）============


class CMFRMAsymmetricCTAnchor(nn.Module):
    """
    非对称跨模态特征校正（受 CMX CM-FRM 启发，面向 CT–PET）：
    - 通道：全局 concat(avg/max CT, avg/max PET) 后，CT 与 PET 各用独立 MLP 产生门控；
      交叉调制：F_ct ← F_ct + σ(w_pet)⊙F_pet，F_pet ← F_pet + σ(w_ct)⊙F_ct。
    - 空间：concat(F_ct,F_pet) 后 CT / PET 各用独立 1×1 产生空间门控，再交叉残差。
    - 解剖锚定：输出前对 CT 支路加强（可学习 β∈(0.5,1) 与 CT 融合）。
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 8)
        self.mlp_ct = nn.Sequential(
            nn.Linear(4 * dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim),
        )
        self.mlp_pet = nn.Sequential(
            nn.Linear(4 * dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim),
        )
        self.spatial_conv_ct = nn.Conv2d(dim * 2, dim, 1, bias=True)
        self.spatial_conv_pet = nn.Conv2d(dim * 2, dim, 1, bias=True)
        self.ct_anchor = nn.Parameter(torch.tensor(0.85))

    def forward(self, f_ct: torch.Tensor, f_pet: torch.Tensor):
        b, c, _, _ = f_ct.shape
        pool = torch.cat(
            [
                F.adaptive_avg_pool2d(f_ct, 1).flatten(1),
                F.adaptive_max_pool2d(f_ct, 1).flatten(1),
                F.adaptive_avg_pool2d(f_pet, 1).flatten(1),
                F.adaptive_max_pool2d(f_pet, 1).flatten(1),
            ],
            dim=1,
        )
        w_ct = torch.sigmoid(self.mlp_ct(pool)).view(b, c, 1, 1)
        w_pet = torch.sigmoid(self.mlp_pet(pool)).view(b, c, 1, 1)
        f_ct_r = f_ct + w_pet * f_pet
        f_pet_r = f_pet + w_ct * f_ct
        cat = torch.cat([f_ct_r, f_pet_r], dim=1)
        s_ct = torch.sigmoid(self.spatial_conv_ct(cat))
        s_pet = torch.sigmoid(self.spatial_conv_pet(cat))
        f_ct_r = f_ct_r + s_pet * f_pet_r
        f_pet_r = f_pet_r + s_ct * f_ct_r
        beta = torch.sigmoid(self.ct_anchor) * 0.5 + 0.5
        f_ct_out = beta * f_ct_r + (1.0 - beta) * (0.5 * (f_ct_r + f_pet_r))
        return f_ct_out, f_pet_r


class LightweightFFMPETtoCT(nn.Module):
    """
    轻量 FFM：仅 PET→CT。用 concat 门控 + PET 经瓶颈投影后的残差注入 CT 流，输出单张融合图（与解码器通道一致）。
    """

    def __init__(self, dim: int, bottleneck_ratio: int = 4):
        super().__init__()
        mid = max(dim // bottleneck_ratio, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=True),
        )
        self.pet_inject = nn.Sequential(
            nn.Conv2d(dim, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.out_norm = nn.BatchNorm2d(dim)

    def forward(self, f_ct: torch.Tensor, f_pet: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([f_ct, f_pet], dim=1)))
        delta = self.pet_inject(f_pet)
        return self.out_norm(f_ct + g * delta)


class ShallowFusionCTAnchorPETtoCT(nn.Module):
    """浅层一条线：非对称 CM-FRM → 仅 PET→CT 的轻量 FFM → 单尺度融合特征。"""

    def __init__(self, dim: int):
        super().__init__()
        self.crm = CMFRMAsymmetricCTAnchor(dim)
        self.ffm = LightweightFFMPETtoCT(dim)

    def forward(self, f_ct: torch.Tensor, f_pet: torch.Tensor, return_aux: bool = False):
        fc, fp = self.crm(f_ct, f_pet)
        out = self.ffm(fc, fp)
        if return_aux:
            return out, {"fused": out, "f_ct": fc, "f_pet": fp}
        return out


# ============ HLMamba 启发：轻量 CIM 式跨模态桥 + 选择性空间混合（非官方复现）============


class SelectiveSpatialMix2D(nn.Module):
    """
    轻量「选择性」空间混合：行向 + 列向深度可分卷积近似双向扫描，不依赖 mamba-ssm。
    受状态空间/选择性扫描在长程依赖上的动机启发，参数量 O(C·k)，适合密集预测。
    """

    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.dw_row = nn.Conv2d(
            dim, dim, kernel_size=(1, kernel_size), padding=(0, pad), groups=dim, bias=True
        )
        self.dw_col = nn.Conv2d(
            dim, dim, kernel_size=(kernel_size, 1), padding=(pad, 0), groups=dim, bias=True
        )
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.gamma = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw_row(x) + self.dw_col(x)
        y = self.pw(y)
        y = self.bn(y)
        return x + torch.tanh(self.gamma) * F.relu(y)


class CrossModalBridgeCIMLite(nn.Module):
    """
    CIM 思想：在双模态拼接流形上做瓶颈投影再残差回 CT 主导支路（医学场景保留解剖锚定后的 CT 流）。
    与遥感 HLMamba 中「跨模态交互」目的相同，结构为原创轻量头（1×1 瓶颈），非仓库逐行移植。
    """

    def __init__(self, dim: int, bottleneck_ratio: int = 4):
        super().__init__()
        mid = max(dim // bottleneck_ratio, 16)
        self.net = nn.Sequential(
            nn.Conv2d(2 * dim, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, f_ct: torch.Tensor, f_pet: torch.Tensor) -> torch.Tensor:
        z = self.net(torch.cat([f_ct, f_pet], dim=1))
        alpha = torch.sigmoid(self.scale)
        return alpha * z


class ShallowFusionHLInspiredCIM(nn.Module):
    """
    浅层融合（HLMamba / CIM 启发 + 本任务先验）：
    1) CMFRMAsymmetricCTAnchor：非对称校正 + CT-anchor（保留原有医学约束）；
    2) CrossModalBridgeCIMLite：concat 瓶颈残差（CIM 式跨模态混合，仅注入 CT 支路，PET→CT 语义）；
    3) LightweightFFMPETtoCT：仅 PET→CT 注入；
    4) SelectiveSpatialMix2D：轻量选择性长程混合。

    说明：HLMamba 原文面向 RGB-红外与 Mamba 块；此处为 CT-PET 适配的原创轻量堆叠，不依赖 mamba_ssm，
    亦不构成对 GitHub 源码的完整复现。若需完整 Mamba 算子，需单独安装 mamba-ssm 并替换 SelectiveSpatialMix2D。

    蒸馏接口：forward(..., return_aux=True) 返回 (fused, aux)，aux 含中间张量便于特征对齐损失扩展。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.crm = CMFRMAsymmetricCTAnchor(dim)
        self.cim_bridge = CrossModalBridgeCIMLite(dim)
        self.ffm = LightweightFFMPETtoCT(dim)
        self.post_mix = SelectiveSpatialMix2D(dim)

    def forward(
        self,
        f_ct: torch.Tensor,
        f_pet: torch.Tensor,
        return_aux: bool = False,
    ):
        fc, fp = self.crm(f_ct, f_pet)
        bridge = self.cim_bridge(fc, fp)
        fc = fc + bridge
        fused = self.ffm(fc, fp)
        fused = self.post_mix(fused)
        if return_aux:
            aux = {
                "fused": fused,
                "f_ct": fc,
                "f_pet": fp,
                "cim_residual": bridge,
            }
            return fused, aux
        return fused


# ============ ShaSpec / DAO ============


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)


class FusionGate(nn.Module):
    def __init__(self, in_channels, init_bias=5.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, init_bias)

    def forward(self, x):
        gate = torch.sigmoid(self.conv(x))
        return gate * x


class ResidualFusionProj(nn.Module):
    def __init__(self, dim_r, dim_s, mid=None):
        super().__init__()
        mid = mid or max((dim_r + dim_s) // 2, 8)
        self.proj = nn.Sequential(
            nn.Conv2d(dim_r + dim_s, mid, 1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim_r, 1),
        )

    def forward(self, r, s):
        x = torch.cat([r, s], dim=1)
        return self.proj(x) + r


class DomainClassifier(nn.Module):
    def __init__(self, in_channels, num_domains=2, mid=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, num_domains),
        )

    def forward(self, x):
        return self.fc(self.pool(x))


# ============ OPD / 学生解耦 ============


def opd_orthogonal_loss(Fs, Fp):
    B, C, H, W = Fs.shape
    Fs_flat = Fs.reshape(B, C, -1)
    Fp_flat = Fp.reshape(B, C, -1)
    gram = torch.bmm(Fs_flat, Fp_flat.transpose(1, 2))
    return gram.norm(p='fro', dim=(1, 2)).mean()


class OrthogonalProjectionDisentangler(nn.Module):
    def __init__(self, in_ch, shared_ch):
        super().__init__()
        self.Ws = nn.Conv2d(in_ch, shared_ch, 1, bias=False)
        self.W_lift = nn.Conv2d(shared_ch, in_ch, 1, bias=False)
        self.W_priv = nn.Conv2d(in_ch, shared_ch, 1, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, normalize=True):
        S = self.Ws(x)
        R = x - self.W_lift(S)
        P = self.W_priv(R)
        if normalize:
            S = F.normalize(S, dim=1, eps=1e-6)
            P = F.normalize(P, dim=1, eps=1e-6)
        return S, P


class ResidualDisentangleEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = max(in_ch, out_ch)
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.act(self.main(x) + self.shortcut(x))


class GatedReconDecoder(nn.Module):
    def __init__(self, half_ch, out_ch):
        super().__init__()
        self.gate = nn.Conv2d(half_ch * 2, half_ch * 2, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.proj = nn.Sequential(
            nn.Conv2d(half_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z_general, z_specific):
        merged = torch.cat([z_general, z_specific], dim=1)
        gate = torch.sigmoid(self.gate(merged))
        return self.proj(gate * merged)


class ResidualAttnFusion(nn.Module):
    def __init__(self, fusion_dim):
        super().__init__()
        mid = max(fusion_dim // 4, 32)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(fusion_dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, fusion_dim),
            nn.Sigmoid(),
        )
        self.bn = nn.BatchNorm2d(fusion_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        gate = self.se(x).view(x.shape[0], -1, 1, 1)
        return self.bn(x + x * gate)
