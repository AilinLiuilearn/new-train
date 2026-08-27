"""
Dual-Origin PET Rectification and CT-Anchored Complementary Fusion
==================================================================

Standalone Module-2 for four-scale PET-CT segmentation features.

Research role
-------------
This module is designed to sit AFTER an upstream PET evidence recovery/calibration
module and BEFORE the segmentation decoder.

Expected upstream interface at each scale s:
    Full:
        C_s       : real CT feature
        P_s       : calibrated real-PET feature
    Missing:
        C_s       : real CT feature
        P_s       : calibrated proxy/recovered-PET feature

The module explicitly models two origin-dependent properties while keeping the
overall processing skeleton matched across Full and Missing states:

1) PET semantic selection is PET-driven for BOTH origins.
   A channel-domain sparse semantic projector is used so PET evidence itself
   determines the PET semantic scaffold.

2) The provenance of patient-specific local detail differs:
   - Real PET: local detail is extracted from real PET itself.
   - Proxy PET: local detail is extracted from CT, but it can enter the PET
     representation only through the proxy-PET-derived semantic projector.

After origin-specific rectification, both states share exactly the same
CT-anchored fusion:
    F_s = C_s + PET_message(C_s, P*_s)

The CT identity path is never compressed or reconstructed.

Source / implementation provenance
----------------------------------
A. Sparse channel-domain attention and GDFN are structurally adapted from:
   "Adaptive Sparse Self-Attention for Efficient Image Super-resolution
    and Beyond" (ASSANet), official implementation:
   https://github.com/sunny2109/ASSANet
   Repository license: Apache-2.0.
   The relevant motifs retained here are:
       - 2D LayerNorm
       - normalized channel-domain correlation
       - ReLU sparse attention
       - gated depthwise-convolution FFN (GDFN)

   We DO NOT copy ASSANet's CuPy IDynamicDWConv/TLC code because those are
   implementation-specific to the restoration network and would introduce a
   custom CUDA/CuPy dependency into the PET-CT segmentation project.

B. The shared fusion's "global attention + local convolution" decomposition is
   conceptually inspired by the ACFM/CAFMAttention design in CAF-YOLO:
   "CAF-YOLO: A Robust Framework for Multi-Scale Lesion Detection in
    Biomedical Imagery"
   https://github.com/xiaochen925/CAF-YOLO

   CAF-YOLO's repository is AGPL-3.0. To avoid copying AGPL-covered source into
   another project, the local/global fusion below is an original PyTorch
   re-implementation of the paper-level computational idea rather than copied
   source code.

Design constraints
------------------
- No MoE / router / prompt.
- No reliability estimator or upstream side signal.
- No hand-crafted |C-P| or C*P feature concatenations.
- No HW x HW attention.
- No replacement of CT by a bottlenecked latent representation.
- Full/Missing have independent PET rectifier parameters but the SAME topology.
- Shared fusion parameters are identical for Full and Missing.
- Only standard PyTorch operators; no CuPy/custom CUDA dependency.
- Default channels match the current PET-CT pipeline:
      S1=64, S2=128, S3=320, S4=512.

Main public class
-----------------
    DualOriginPETRectificationFusion

Forward API
-----------
    result = module(
        ct_feats,
        pet_feats_cal,
        route="full" | "missing" | "auto",
        pet_available=None,   # required only for route="auto"
    )

Returns
-------
    MultiScaleFusionOutput
        .features : list[Tensor], same four shapes as CT inputs
        .stats    : detached diagnostics only

This file is intentionally standalone and can be smoke-tested directly:
    python dual_origin_pet_rectification_fusion.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _check_feature_list(name: str, xs: Sequence[torch.Tensor], channels: Sequence[int]) -> None:
    if len(xs) != len(channels):
        raise ValueError(f"{name}: expected {len(channels)} feature scales, got {len(xs)}")
    for i, (x, c) in enumerate(zip(xs, channels), start=1):
        if x.ndim != 4:
            raise ValueError(f"{name}[S{i}] must be BCHW, got {tuple(x.shape)}")
        if x.shape[1] != c:
            raise ValueError(f"{name}[S{i}] expected {c} channels, got {x.shape[1]}")
        if not torch.isfinite(x).all():
            raise RuntimeError(f"{name}[S{i}] contains NaN/Inf")


def _check_pair_shapes(ct_feats: Sequence[torch.Tensor], pet_feats: Sequence[torch.Tensor]) -> None:
    for i, (ct, pet) in enumerate(zip(ct_feats, pet_feats), start=1):
        if ct.shape != pet.shape:
            raise ValueError(f"S{i} CT/PET shape mismatch: {tuple(ct.shape)} vs {tuple(pet.shape)}")


def _valid_num_heads(dim: int, requested_heads: int) -> int:
    for h in range(min(dim, requested_heads), 0, -1):
        if dim % h == 0:
            return h
    return 1


def _latent_dim(channels: int, cap: int) -> int:
    """Correction-space width only; identity features keep full channels."""
    d = min(int(channels), int(cap))
    if d >= 8:
        d = max(8, (d // 8) * 8)
    return d


# -----------------------------------------------------------------------------
# ASSANet-style 2D LayerNorm
# -----------------------------------------------------------------------------


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def _to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) * torch.rsqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    """ASSANet/Restormer-style LayerNorm over channels for BCHW tensors."""

    def __init__(self, dim: int, bias: bool = True) -> None:
        super().__init__()
        self.body = WithBiasLayerNorm(dim) if bias else BiasFreeLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


# -----------------------------------------------------------------------------
# ASSANet-style GDFN
# -----------------------------------------------------------------------------


class GatedDconvFeedForward(nn.Module):
    """
    Gated depthwise-convolution feed-forward network.

    Core topology follows the FeedForward/GDFN used by ASSANet:
        1x1 projection -> depthwise 3x3 -> split ->
        GELU(branch1) * branch2 -> 1x1 projection.
    """

    def __init__(self, dim: int, expansion_factor: float = 2.0, bias: bool = False) -> None:
        super().__init__()
        hidden = int(round(dim * expansion_factor))
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# -----------------------------------------------------------------------------
# Origin-aware sparse semantic projector
# -----------------------------------------------------------------------------


class OriginSparseSemanticAttention(nn.Module):
    """
    PET-driven sparse channel-domain semantic projector.

    Compared with ASSANet SparseSelfAttention:
      - keeps normalized channel-domain QK^T correlation;
      - keeps ReLU sparsification;
      - adds a learnable non-negative per-head threshold so Real/Proxy
        rectifiers can learn distinct sparsity policies;
      - accepts an optional detail tensor that is injected into the VALUE
        stream only. Therefore detail cannot redefine the PET-derived semantic
        Q/K scaffold.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        bias: bool = False,
        threshold_init: float = 0.02,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = _valid_num_heads(self.dim, int(num_heads))
        if self.dim % self.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.qkv = nn.Conv2d(self.dim, self.dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            self.dim * 3, self.dim * 3, kernel_size=3, stride=1, padding=1,
            groups=self.dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=bias)

        threshold_init = max(float(threshold_init), 1e-6)
        raw_init = math.log(math.expm1(threshold_init))
        self.raw_sparse_threshold = nn.Parameter(
            torch.full((self.num_heads, 1, 1), raw_init)
        )

    @property
    def sparse_threshold(self) -> torch.Tensor:
        return F.softplus(self.raw_sparse_threshold)

    def forward(
        self,
        x: torch.Tensor,
        detail_value: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {c}")
        if detail_value is not None and detail_value.shape != x.shape:
            raise ValueError(
                "detail_value must match semantic feature shape: "
                f"{tuple(detail_value.shape)} vs {tuple(x.shape)}"
            )

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = c // self.num_heads
        n = h * w

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, self.num_heads, head_dim, n)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        q32 = F.normalize(q.float(), dim=-1)
        k32 = F.normalize(k.float(), dim=-1)
        logits = torch.matmul(q32, k32.transpose(-2, -1))
        logits = logits * self.temperature.float().unsqueeze(0)

        threshold = self.sparse_threshold.float().unsqueeze(0)
        attn = F.relu(logits - threshold)

        if detail_value is not None:
            d = reshape_heads(detail_value)
            v = v + d

        out = torch.matmul(attn.to(dtype=v.dtype), v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)

        stats: Dict[str, torch.Tensor] = {}
        if return_stats:
            with torch.no_grad():
                eps = 1e-8
                attn_det = attn.detach()
                active = (attn_det > eps).float()
                stats = {
                    "semantic_sparsity": (1.0 - active.mean()).float(),
                    "semantic_active_mean": (
                        attn_det.sum() / active.sum().clamp_min(1.0)
                    ).float(),
                    "sparse_threshold_mean": self.sparse_threshold.detach().float().mean(),
                    "temperature_mean": self.temperature.detach().float().mean(),
                }
        return out, stats


# -----------------------------------------------------------------------------
# Local detail provenance operator
# -----------------------------------------------------------------------------


class LocalDetailExtractor(nn.Module):
    """
    Shared-topology local detail extractor.

    Only the SOURCE differs:
        Real  -> source = PET
        Proxy -> source = CT

    The output lives in the same latent space as the PET semantic VALUE stream.
    """

    def __init__(self, in_channels: int, latent_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.norm = LayerNorm2d(in_channels, bias=True)
        self.project_in = nn.Conv2d(in_channels, latent_dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            latent_dim * 2, latent_dim * 2, kernel_size=3, stride=1, padding=1,
            groups=latent_dim * 2, bias=bias
        )
        self.project_out = nn.Conv2d(latent_dim, latent_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# -----------------------------------------------------------------------------
# One origin-specific PET rectifier
# -----------------------------------------------------------------------------


@dataclass
class RectifierOutput:
    feature: torch.Tensor
    stats: Dict[str, torch.Tensor]


class OriginPETRectifier(nn.Module):
    """
    Matched-topology PET rectifier.

    The topology is identical for both origins:
        PET -> latent semantic stream
        detail_source -> latent local-detail stream
        PET-derived sparse Q/K + (PET V + local detail)
        -> GDFN
        -> residual rectification of original PET
    """

    def __init__(
        self,
        channels: int,
        latent_dim: int,
        num_heads: int = 4,
        ffn_expansion: float = 2.0,
        bias: bool = False,
        layer_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.latent_dim = int(latent_dim)

        self.pet_in = nn.Conv2d(self.channels, self.latent_dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm2d(self.latent_dim, bias=True)
        self.detail = LocalDetailExtractor(self.channels, self.latent_dim, bias=bias)
        self.semantic = OriginSparseSemanticAttention(
            self.latent_dim, num_heads=num_heads, bias=bias
        )
        self.norm2 = LayerNorm2d(self.latent_dim, bias=True)
        self.ffn = GatedDconvFeedForward(
            self.latent_dim, expansion_factor=ffn_expansion, bias=bias
        )
        self.pet_out = nn.Conv2d(self.latent_dim, self.channels, kernel_size=1, bias=True)

        # Near-identity at initialization WITHOUT zeroing gradients to internal branches.
        self.gamma = nn.Parameter(
            torch.full((1, self.channels, 1, 1), float(layer_scale_init))
        )

    def forward(self, pet: torch.Tensor, detail_source: torch.Tensor) -> RectifierOutput:
        if pet.shape != detail_source.shape:
            raise ValueError(
                "PET/detail source must have identical BCHW shape; got "
                f"{tuple(pet.shape)} vs {tuple(detail_source.shape)}"
            )

        x = self.pet_in(pet)
        d = self.detail(detail_source)
        attn_out, stats = self.semantic(
            self.norm1(x), detail_value=d, return_stats=True
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        correction = self.pet_out(x).to(dtype=pet.dtype)
        out = pet + self.gamma.to(dtype=pet.dtype) * correction

        with torch.no_grad():
            pet_abs = pet.detach().float().abs().mean().clamp_min(1e-8)
            corr_abs = (
                self.gamma.detach().float() * correction.detach().float()
            ).abs().mean()
            stats.update({
                "detail_abs_mean": d.detach().float().abs().mean(),
                "rectification_ratio": (corr_abs / pet_abs).float(),
                "layer_scale_abs_mean": self.gamma.detach().float().abs().mean(),
            })
        return RectifierOutput(feature=out, stats=stats)


# -----------------------------------------------------------------------------
# Shared CT-anchored complementary fusion
# -----------------------------------------------------------------------------


class CrossChannelAttention(nn.Module):
    """Efficient PET->CT channel-domain cross attention; Q=CT, K/V=PET."""

    def __init__(self, dim: int, num_heads: int = 4, bias: bool = False) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = _valid_num_heads(self.dim, int(num_heads))
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ct: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        if ct.shape != pet.shape:
            raise ValueError(
                f"Cross attention CT/PET mismatch: {tuple(ct.shape)} vs {tuple(pet.shape)}"
            )
        b, c, h, w = ct.shape
        n = h * w
        hd = c // self.num_heads

        q = self.q_dwconv(self.q(ct))
        k, v = self.kv_dwconv(self.kv(pet)).chunk(2, dim=1)
        q = q.reshape(b, self.num_heads, hd, n)
        k = k.reshape(b, self.num_heads, hd, n)
        v = v.reshape(b, self.num_heads, hd, n)

        q32 = F.normalize(q.float(), dim=-1)
        k32 = F.normalize(k.float(), dim=-1)
        attn = torch.matmul(q32, k32.transpose(-2, -1))
        attn = attn * self.temperature.float().unsqueeze(0)
        attn = F.softmax(attn, dim=-1).to(dtype=v.dtype)

        out = torch.matmul(attn, v).reshape(b, c, h, w)
        return self.project_out(out)


class SharedCTAnchoredFusion(nn.Module):
    """
    Shared fusion used identically for Full and Missing.

    ACFM-inspired decomposition:
        global branch : PET->CT cross-channel attention
        local branch  : local CT-PET convolutional interaction

    Final output:
        F = CT + gamma * Message
    """

    def __init__(
        self,
        channels: int,
        latent_dim: int,
        num_heads: int = 4,
        ffn_expansion: float = 2.0,
        bias: bool = False,
        layer_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.ct_proj = nn.Conv2d(channels, latent_dim, kernel_size=1, bias=bias)
        self.pet_proj = nn.Conv2d(channels, latent_dim, kernel_size=1, bias=bias)
        self.ct_norm = LayerNorm2d(latent_dim, bias=True)
        self.pet_norm = LayerNorm2d(latent_dim, bias=True)

        self.global_cross = CrossChannelAttention(
            latent_dim, num_heads=num_heads, bias=bias
        )
        self.local_fuse = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim * 2, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(
                latent_dim * 2, latent_dim * 2, kernel_size=3, stride=1, padding=1,
                groups=latent_dim * 2, bias=bias
            ),
            nn.GELU(),
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=1, bias=bias),
        )

        self.refine_norm = LayerNorm2d(latent_dim, bias=True)
        self.refine = GatedDconvFeedForward(
            latent_dim, expansion_factor=ffn_expansion, bias=bias
        )
        self.out_proj = nn.Conv2d(latent_dim, channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(
            torch.full((1, channels, 1, 1), float(layer_scale_init))
        )

    def forward(self, ct: torch.Tensor, pet: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if ct.shape != pet.shape:
            raise ValueError(
                f"Shared fusion CT/PET mismatch: {tuple(ct.shape)} vs {tuple(pet.shape)}"
            )

        c = self.ct_norm(self.ct_proj(ct))
        p = self.pet_norm(self.pet_proj(pet))
        global_msg = self.global_cross(c, p)
        local_msg = self.local_fuse(torch.cat([c, p], dim=1))
        msg = global_msg + local_msg
        msg = msg + self.refine(self.refine_norm(msg))
        msg = self.out_proj(msg).to(dtype=ct.dtype)

        scaled_msg = self.gamma.to(dtype=ct.dtype) * msg
        out = ct + scaled_msg

        with torch.no_grad():
            ct_abs = ct.detach().float().abs().mean().clamp_min(1e-8)
            stats = {
                "fusion_message_ratio": (
                    scaled_msg.detach().float().abs().mean() / ct_abs
                ).float(),
                "global_abs_mean": global_msg.detach().float().abs().mean(),
                "local_abs_mean": local_msg.detach().float().abs().mean(),
                "fusion_layer_scale_abs_mean": self.gamma.detach().float().abs().mean(),
            }
        return out, stats


# -----------------------------------------------------------------------------
# One scale: dual-origin rectification -> shared fusion
# -----------------------------------------------------------------------------


@dataclass
class ScaleFusionOutput:
    feature: torch.Tensor
    rectified_pet: torch.Tensor
    stats: Dict[str, torch.Tensor]


class DualOriginScaleUnit(nn.Module):
    """One scale with same-topology Real/Proxy rectifiers + one shared fusion."""

    def __init__(
        self,
        channels: int,
        latent_cap: int = 128,
        num_heads: int = 4,
        ffn_expansion: float = 2.0,
        bias: bool = False,
        layer_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.latent_dim = _latent_dim(self.channels, latent_cap)

        self.real_rectifier = OriginPETRectifier(
            self.channels, self.latent_dim, num_heads, ffn_expansion, bias, layer_scale_init
        )
        self.proxy_rectifier = OriginPETRectifier(
            self.channels, self.latent_dim, num_heads, ffn_expansion, bias, layer_scale_init
        )
        self.shared_fusion = SharedCTAnchoredFusion(
            self.channels, self.latent_dim, num_heads, ffn_expansion, bias, layer_scale_init
        )

    def _forward_single_route(self, ct: torch.Tensor, pet: torch.Tensor, route: str) -> ScaleFusionOutput:
        route = str(route).strip().lower()
        if route == "full":
            rect = self.real_rectifier(pet=pet, detail_source=pet)
            prefix = "real"
        elif route == "missing":
            rect = self.proxy_rectifier(pet=pet, detail_source=ct)
            prefix = "proxy"
        else:
            raise ValueError(f"Unsupported fixed route={route!r}")

        fused, fusion_stats = self.shared_fusion(ct=ct, pet=rect.feature)
        stats: Dict[str, torch.Tensor] = {
            **{f"{prefix}_{k}": v for k, v in rect.stats.items()},
            **fusion_stats,
        }
        return ScaleFusionOutput(fused, rect.feature, stats)

    def forward(
        self,
        ct: torch.Tensor,
        pet: torch.Tensor,
        route: str,
        pet_available: Optional[torch.Tensor] = None,
    ) -> ScaleFusionOutput:
        route = str(route).strip().lower()
        if route in {"full", "missing"}:
            return self._forward_single_route(ct, pet, route)

        if route != "auto":
            raise ValueError("route must be 'full', 'missing', or 'auto'")
        if pet_available is None:
            raise ValueError("route='auto' requires pet_available [B]")

        availability = pet_available.to(device=ct.device).long().view(-1)
        if availability.numel() != ct.shape[0]:
            raise ValueError("pet_available must have one entry per sample")
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError("pet_available must contain only 0/1")

        out = torch.empty_like(ct)
        rectified = torch.empty_like(pet)
        stats: Dict[str, torch.Tensor] = {}

        full_idx = torch.nonzero(availability == 1, as_tuple=False).flatten()
        miss_idx = torch.nonzero(availability == 0, as_tuple=False).flatten()

        if full_idx.numel() > 0:
            r = self._forward_single_route(
                ct.index_select(0, full_idx), pet.index_select(0, full_idx), "full"
            )
            out.index_copy_(0, full_idx, r.feature)
            rectified.index_copy_(0, full_idx, r.rectified_pet)
            for k, v in r.stats.items():
                stats[f"auto_full_{k}"] = v

        if miss_idx.numel() > 0:
            r = self._forward_single_route(
                ct.index_select(0, miss_idx), pet.index_select(0, miss_idx), "missing"
            )
            out.index_copy_(0, miss_idx, r.feature)
            rectified.index_copy_(0, miss_idx, r.rectified_pet)
            for k, v in r.stats.items():
                stats[f"auto_missing_{k}"] = v

        stats["auto_full_fraction"] = availability.float().mean().detach()
        stats["auto_missing_fraction"] = (1.0 - availability.float().mean()).detach()
        return ScaleFusionOutput(out, rectified, stats)


# -----------------------------------------------------------------------------
# Four-scale public module
# -----------------------------------------------------------------------------


@dataclass
class MultiScaleFusionOutput:
    features: List[torch.Tensor]
    stats: Dict[str, torch.Tensor]
    rectified_pet_features: Optional[List[torch.Tensor]] = None


class DualOriginPETRectificationFusion(nn.Module):
    """
    Four-scale standalone Module-2.

    The forward signature intentionally mirrors the current SPRE fusion module:
        forward(ct_feats, pet_feats_cal, route, pet_available=None)
    """

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        latent_cap: int = 128,
        num_heads: int = 4,
        ffn_expansion: float = 2.0,
        bias: bool = False,
        layer_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        if len(self.channels) != 4:
            raise ValueError("Current PET-CT decoder integration expects four scales")

        self.scale_units = nn.ModuleList([
            DualOriginScaleUnit(
                c, latent_cap, num_heads, ffn_expansion, bias, layer_scale_init
            ) for c in self.channels
        ])

    def forward(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_feats_cal: Sequence[torch.Tensor],
        route: str,
        pet_available: Optional[torch.Tensor] = None,
        return_rectified_pet: bool = False,
    ) -> MultiScaleFusionOutput:
        _check_feature_list("ct_feats", ct_feats, self.channels)
        _check_feature_list("pet_feats_cal", pet_feats_cal, self.channels)
        _check_pair_shapes(ct_feats, pet_feats_cal)

        route = str(route).strip().lower()
        if route not in {"full", "missing", "auto"}:
            raise ValueError(f"Unsupported route={route!r}; expected full/missing/auto")

        features: List[torch.Tensor] = []
        rectified: List[torch.Tensor] = []
        stats: Dict[str, torch.Tensor] = {}

        for scale_idx, (ct, pet, unit) in enumerate(
            zip(ct_feats, pet_feats_cal, self.scale_units), start=1
        ):
            result = unit(ct, pet, route, pet_available=pet_available)
            features.append(result.feature)
            rectified.append(result.rectified_pet)
            for key, value in result.stats.items():
                stats[f"s{scale_idx}_{key}"] = value

        stats["route_full"] = torch.tensor(
            1.0 if route == "full" else 0.0, device=features[0].device
        )
        stats["route_missing"] = torch.tensor(
            1.0 if route == "missing" else 0.0, device=features[0].device
        )

        return MultiScaleFusionOutput(
            features=features,
            stats=stats,
            rectified_pet_features=rectified if return_rectified_pet else None,
        )


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------


def _grad_abs_sum(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().abs().sum().item())
    return total


def _smoke_test() -> None:
    torch.manual_seed(2023)
    channels = (64, 128, 320, 512)
    spatial = ((8, 8), (4, 4), (2, 2), (1, 1))
    b = 2

    model = DualOriginPETRectificationFusion(
        channels=channels,
        latent_cap=128,
        num_heads=4,
        ffn_expansion=2.0,
    )
    model.train()

    ct = [
        torch.randn(b, c, h, w, requires_grad=True)
        for c, (h, w) in zip(channels, spatial)
    ]
    pet = [
        torch.randn(b, c, h, w, requires_grad=True)
        for c, (h, w) in zip(channels, spatial)
    ]

    # Full route
    out_full = model(ct, pet, route="full", return_rectified_pet=True)
    for x, y in zip(out_full.features, ct):
        assert x.shape == y.shape and torch.isfinite(x).all()
    sum(x.square().mean() for x in out_full.features).backward()

    full_real_grad = sum(_grad_abs_sum(u.real_rectifier) for u in model.scale_units)
    full_proxy_grad = sum(_grad_abs_sum(u.proxy_rectifier) for u in model.scale_units)
    full_shared_grad = sum(_grad_abs_sum(u.shared_fusion) for u in model.scale_units)
    assert full_real_grad > 0.0
    assert full_proxy_grad == 0.0
    assert full_shared_grad > 0.0

    model.zero_grad(set_to_none=True)

    # Missing route
    out_missing = model(
        [x.detach().requires_grad_(True) for x in ct],
        [x.detach().requires_grad_(True) for x in pet],
        route="missing",
        return_rectified_pet=True,
    )
    sum(x.square().mean() for x in out_missing.features).backward()

    miss_real_grad = sum(_grad_abs_sum(u.real_rectifier) for u in model.scale_units)
    miss_proxy_grad = sum(_grad_abs_sum(u.proxy_rectifier) for u in model.scale_units)
    miss_shared_grad = sum(_grad_abs_sum(u.shared_fusion) for u in model.scale_units)
    assert miss_real_grad == 0.0
    assert miss_proxy_grad > 0.0
    assert miss_shared_grad > 0.0

    model.zero_grad(set_to_none=True)

    # Auto route
    availability = torch.tensor([1, 0], dtype=torch.long)
    out_auto = model(
        [x.detach() for x in ct],
        [x.detach() for x in pet],
        route="auto",
        pet_available=availability,
    )
    for x, y in zip(out_auto.features, ct):
        assert x.shape == y.shape and torch.isfinite(x).all()

    n_params = sum(p.numel() for p in model.parameters())
    print("[SMOKE] DualOriginPETRectificationFusion PASS")
    print(f"[SMOKE] parameters: {n_params / 1e6:.3f} M")
    print(
        "[SMOKE] full grads "
        f"real={full_real_grad:.3e} proxy={full_proxy_grad:.3e} shared={full_shared_grad:.3e}"
    )
    print(
        "[SMOKE] missing grads "
        f"real={miss_real_grad:.3e} proxy={miss_proxy_grad:.3e} shared={miss_shared_grad:.3e}"
    )


if __name__ == "__main__":
    _smoke_test()
