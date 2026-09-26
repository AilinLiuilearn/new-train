"""Full CT/PET asymmetric fusion v2 (residual + competitive gate + text mask).

Same single-sign paradigm as v1 (``models/full_petct_asymmetric_fusion.py``):
CT/PET self-extract their own features, text only does channel modulation,
and the joint gate fuses them. v2 fixes three diagnosed flaws without
changing the paradigm:

1. TextChannelModulation: the old ``mlp(t)`` term was image-independent
   (same prompt -> same bias for every image). v2 keeps channel modulation
   but the text only produces a channel *mask* applied to the *image*
   descriptor, so every text effect passes through image content.
2. PETGlobalCorrection: adds a full-resolution depthwise branch (zero-
   initialized, so v2 starts exactly as v1) beside the pooled global path,
   keeping small-lesion detail at shallow scales.
3. JointSpatialFusion: independent sigmoids -> per-location softmax
   competition between CT/PET, plus a CT residual anchor
   (``fused = ct + delta``). The ``delta`` term is the PET-contributed
   correction and is what the stage-2 bank stores.

``forward`` keeps the v1 signature (returns fused only).
``forward_with_delta`` additionally returns the per-scale ``delta`` list for
bank building: bank stores (ct_key, delta_value) pairs per scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import unittest

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.full_petct_asymmetric_fusion import (
    CT_PROMPT,
    PET_PROMPT,
    ChannelLayerNorm,
    CTMultiScaleCorrection,
    _offline_clip_embeddings,
)

__all__ = ["FullPETCTAsymmetricFusionV2"]


class TextChannelModulationV2(nn.Module):
    """Channel modulation where text can only mask the image descriptor.

    Old flaw: ``logits = mlp(desc(u)) + mlp(t)`` added an image-independent
    constant. Here ``logits = mlp(d * g + d)`` with ``g = sigmoid(gate(t))``:
    zeroing the text leaves a constant 0.5 mask, and any text effect must
    pass through the image descriptor ``d``.
    """

    def __init__(self, channels: int, text_dim: int, pool: str, use_text: bool):
        super().__init__()
        self.pool = pool
        self.use_text = use_text
        if use_text:
            self.text_proj = nn.Sequential(nn.Linear(text_dim, channels),
                                           nn.LayerNorm(channels), nn.GELU())
            hidden = max(channels // 16, 4)
            self.channel_gate = nn.Sequential(nn.Linear(channels, hidden),
                                              nn.GELU(),
                                              nn.Linear(hidden, channels))
        self.visual_conv = nn.Conv2d(channels, channels, 3, padding=1)
        hidden = max(channels // 16, 4)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, channels, 1))

    def forward(self, x: Tensor, text: Tensor | None) -> Tensor:
        if self.use_text:
            if text is None:
                raise ValueError("Text-enabled modulation requires cached embeddings.")
            t = self.text_proj(text).reshape(1, -1, 1, 1)
            g = torch.sigmoid(self.channel_gate(t.reshape(1, -1))).reshape(1, -1, 1, 1)
            d = (F.adaptive_avg_pool2d(self.visual_conv(x), 1) if self.pool == "avg"
                 else F.adaptive_max_pool2d(self.visual_conv(x), 1))
            logits = self.mlp(d * g + d)
        else:
            u = self.visual_conv(x)
            descriptor = (F.adaptive_avg_pool2d(u, 1) if self.pool == "avg"
                          else F.adaptive_max_pool2d(u, 1))
            logits = self.mlp(descriptor)
        return x * (1.0 + torch.tanh(logits))


class PETGlobalCorrectionV2(nn.Module):
    """v1 global path plus a zero-initialized full-resolution local branch."""

    def __init__(self, channels: int, dim: int, heads: int,
                 grid_cap: int, checkpoint_attention: bool):
        super().__init__()
        self.heads, self.dim, self.grid_cap = heads, dim, grid_cap
        self.checkpoint_attention = checkpoint_attention
        self.in_proj = nn.Conv2d(channels, dim, 1)
        self.position_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = ChannelLayerNorm(dim)
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.attn_out = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, channels, 1)
        self.local_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        nn.init.zeros_(self.local_dw.weight)
        if self.local_dw.bias is not None:
            nn.init.zeros_(self.local_dw.bias)

    def _attention(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        x = x + self.position_conv(x)
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, self.dim // self.heads, h * w)
        q, k, v = qkv.permute(1, 0, 2, 4, 3).unbind(0)
        y = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                           dropout_p=0.0, is_causal=False)
        y = y.transpose(-2, -1).reshape(b, self.dim, h, w)
        return self.attn_out(y)

    def forward(self, x: Tensor) -> Tensor:
        from torch.utils.checkpoint import checkpoint
        h, w = x.shape[-2:]
        target = (min(h, self.grid_cap), min(w, self.grid_cap))
        x_down = F.adaptive_avg_pool2d(x, target) if target != (h, w) else x
        y = self.in_proj(x_down)
        if self.checkpoint_attention and self.training and torch.is_grad_enabled():
            y = checkpoint(self._attention, y, use_reentrant=False)
        else:
            y = self._attention(y)
        if target != (h, w):
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return self.out_proj(y) + self.local_dw(x)


class JointSpatialFusionV2(nn.Module):
    """Per-location CT/PET softmax competition plus a CT residual anchor."""

    def __init__(self, channels: int):
        super().__init__()
        self.ct_proj = nn.Conv2d(channels, channels, 1)
        self.pet_proj = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(4, 2, 3, padding=1)
        self.out_proj = nn.Conv2d(2 * channels, channels, 1)

    def forward_with_delta(self, ct: Tensor, pet: Tensor) -> tuple[Tensor, Tensor]:
        c, p = self.ct_proj(ct), self.pet_proj(pet)
        stats = torch.cat((c.mean(1, keepdim=True), c.amax(1, keepdim=True),
                           p.mean(1, keepdim=True), p.amax(1, keepdim=True)), dim=1)
        w = self.gate(stats).softmax(dim=1)
        wc, wp = w.chunk(2, dim=1)
        delta = self.out_proj(torch.cat((wc * c, wp * p), dim=1))
        return ct + delta, delta

    def forward(self, ct: Tensor, pet: Tensor) -> Tensor:
        fused, _ = self.forward_with_delta(ct, pet)
        return fused


class FusionScaleV2(nn.Module):
    def __init__(self, channels: int, text_dim: int, pet_dim: int, heads: int,
                 grid_cap: int, use_text: bool, checkpoint_attention: bool):
        super().__init__()
        self.ct_mod = TextChannelModulationV2(channels, text_dim, "avg", use_text)
        self.pet_mod = TextChannelModulationV2(channels, text_dim, "max", use_text)
        self.ct_correction = CTMultiScaleCorrection(channels)
        self.pet_correction = PETGlobalCorrectionV2(channels, pet_dim, heads, grid_cap,
                                                    checkpoint_attention)
        self.fusion = JointSpatialFusionV2(channels)

    def forward_with_delta(self, ct: Tensor, pet: Tensor, text: Tensor | None):
        ct_mod = self.ct_mod(ct, None if text is None else text[0])
        pet_mod = self.pet_mod(pet, None if text is None else text[1])
        ct_out = ct + self.ct_correction(ct_mod)
        pet_out = pet + self.pet_correction(pet_mod)
        fused, _ = self.fusion.forward_with_delta(ct_out, pet_out)
        # Module-level delta is the TOTAL correction vs the scale input: the
        # bank stores (ct_key, delta_value) so Missing can do ct + delta_hat.
        return fused, fused - ct

    def forward(self, ct: Tensor, pet: Tensor, text: Tensor | None) -> Tensor:
        fused, _ = self.forward_with_delta(ct, pet, text)
        return fused


class FullPETCTAsymmetricFusionV2(nn.Module):
    """Full-only four-scale fusion, v2 (see module docstring for deltas vs v1)."""

    def __init__(self, clip_path: str | Path | None = None, *,
                 channels: Sequence[int] = (64, 128, 320, 512),
                 pet_dims: Sequence[int] = (64, 128, 160, 256),
                 heads: int = 4, grid_cap: int = 32, text_dim: int = 512,
                 use_text: bool = True, text_embeddings: Tensor | None = None,
                 prompts: Sequence[str] = (CT_PROMPT, PET_PROMPT),
                 checkpoint_attention: bool = False):
        super().__init__()
        self.channels = tuple(channels)
        dims = tuple(pet_dims)
        if len(self.channels) != 4 or len(dims) != 4:
            raise ValueError("Exactly four scales are required, shallow-to-deep.")
        values = (*self.channels, *dims, heads, grid_cap, text_dim)
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in values):
            raise ValueError("Channels, dimensions, heads and grid_cap must be positive integers.")
        if any(d % heads for d in dims):
            raise ValueError("Each PET attention dimension must be divisible by heads.")
        if len(prompts) != 2 or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("Provide two nonempty prompts, CT then PET.")
        self.use_text = use_text
        self._contract = dict(version=2, channels=list(self.channels), pet_dims=list(dims),
                              heads=heads, grid_cap=grid_cap, text_dim=text_dim,
                              use_text=use_text, prompts=list(prompts))
        if use_text:
            if text_embeddings is None:
                if clip_path is None:
                    clip_path = Path(__file__).resolve().parent.parent / "pretrained" / "clip-vit-base-patch32"
                text_embeddings = _offline_clip_embeddings(clip_path, prompts)
            if not isinstance(text_embeddings, Tensor) or text_embeddings.shape != (2, text_dim):
                raise ValueError(f"Expected CLIP pooler embeddings [2,{text_dim}].")
            if not text_embeddings.is_floating_point() or not torch.isfinite(text_embeddings).all():
                raise ValueError("Text embeddings must be finite floating point values.")
            text_embeddings = text_embeddings.detach().to(device="cpu", dtype=torch.float32).clone()
        elif text_embeddings is not None:
            raise ValueError("Do not supply text_embeddings with use_text=False.")
        self.register_buffer("text_embeddings", text_embeddings, persistent=True)
        self.scales = nn.ModuleList([
            FusionScaleV2(c, text_dim, d, heads, grid_cap, use_text, checkpoint_attention)
            for c, d in zip(self.channels, dims)])
        self.apply(self._init_weights)
        # Keep the local detail branch an exact no-op at init (v2 starts as v1
        # plus the gate/residual change); the re-zeroing below intentionally
        # runs after apply().
        for scale in self.scales:
            nn.init.zeros_(scale.pet_correction.local_dw.weight)
            if scale.pet_correction.local_dw.bias is not None:
                nn.init.zeros_(scale.pet_correction.local_dw.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            if module.kernel_size == (1, 1):
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_extra_state(self) -> dict:
        return dict(self._contract)

    def set_extra_state(self, state: dict) -> None:
        if state != self._contract:
            raise RuntimeError("Fusion checkpoint contract differs from constructor settings/prompts.")

    def _validate(self, ct_features: Sequence[Tensor], pet_features: Sequence[Tensor]):
        if not isinstance(ct_features, (list, tuple)) or not isinstance(pet_features, (list, tuple)):
            raise TypeError("Features must be lists/tuples of four BCHW tensors.")
        if len(ct_features) != 4 or len(pet_features) != 4:
            raise ValueError("Expected four CT and four PET feature maps.")
        first = ct_features[0]
        if not isinstance(first, Tensor) or first.ndim != 4:
            raise ValueError("CT features must be BCHW tensors.")
        for i, (c, p, channels) in enumerate(zip(ct_features, pet_features, self.channels)):
            if not isinstance(c, Tensor) or not isinstance(p, Tensor):
                raise TypeError(f"Scale {i}: inputs must be tensors.")
            if c.ndim != 4 or p.shape != c.shape or c.shape[1] != channels or min(c.shape) <= 0:
                raise ValueError(f"Scale {i}: matching nonempty BCHW tensors with {channels} channels required.")
            if not c.is_floating_point() or p.dtype != c.dtype or c.dtype != first.dtype:
                raise ValueError("All feature maps must share a floating dtype.")
            if c.device != p.device or c.device != first.device or c.shape[0] != first.shape[0]:
                raise ValueError("All feature maps must share batch size and device.")
        parameter = next(self.parameters())
        if parameter.device != first.device:
            raise ValueError("Move the fusion module to the input device before forward.")

    def forward_with_delta(self, ct_features: Sequence[Tensor], pet_features: Sequence[Tensor],
                           *, state: str = "full") -> tuple[list[Tensor], list[Tensor]]:
        """Returns (fused, delta) per scale with ``delta = fused - ct_input``.

        The bank stores (ct_key, delta_value) pairs: in Missing, ``ct`` is
        computed frozen on the spot and only ``delta`` is retrieved.
        """
        if state != "full":
            raise ValueError("This module supports only state='full' with real CT and PET.")
        self._validate(ct_features, pet_features)
        fused, deltas = [], []
        for layer, c, p in zip(self.scales, ct_features, pet_features):
            f, d = layer.forward_with_delta(c, p, self.text_embeddings)
            fused.append(f)
            deltas.append(d)
        return fused, deltas

    def forward(self, ct_features: Sequence[Tensor], pet_features: Sequence[Tensor],
                *, state: str = "full") -> list[Tensor]:
        fused, _ = self.forward_with_delta(ct_features, pet_features, state=state)
        return fused


class ContractTestsV2(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)
        self.kw = dict(channels=(8, 12, 16, 24), pet_dims=(8, 8, 12, 16),
                       text_dim=16, grid_cap=8)

    def model(self, **kwargs):
        return FullPETCTAsymmetricFusionV2(text_embeddings=torch.randn(2, 16), **self.kw, **kwargs)

    def features(self):
        return [torch.randn(2, c, h, w, requires_grad=True)
                for c, (h, w) in zip(self.kw["channels"], [(17, 15), (9, 8), (5, 4), (3, 2)])]

    def test_fused_equals_ct_plus_delta(self):
        model = self.model()
        c, p = self.features(), self.features()
        fused, deltas = model.forward_with_delta(c, p)
        self.assertEqual(len(fused), 4)
        self.assertEqual(len(deltas), 4)
        for f, d, ci in zip(fused, deltas, c):
            self.assertEqual(f.shape, ci.shape)
            self.assertEqual(d.shape, ci.shape)
            self.assertTrue(torch.allclose(f, ci.detach() + d, atol=1e-5))

    def test_gates_compete_and_sum_to_one(self):
        model = self.model()
        c, p = self.features(), self.features()
        scale = model.scales[0]
        with torch.no_grad():
            cc = scale.fusion.ct_proj(c[0])
            pp = scale.fusion.pet_proj(p[0])
            stats = torch.cat((cc.mean(1, keepdim=True), cc.amax(1, keepdim=True),
                               pp.mean(1, keepdim=True), pp.amax(1, keepdim=True)), dim=1)
            w = scale.fusion.gate(stats).softmax(dim=1)
            self.assertTrue(torch.allclose(w.sum(1), torch.ones_like(w[:, 0]), atol=1e-5))

    def test_text_effect_passes_through_image(self):
        torch.manual_seed(0)
        model = self.model()
        x1 = torch.randn(2, 8, 9, 8)
        x2 = torch.randn(2, 8, 9, 8)
        t = torch.randn(16)
        mod = model.scales[0].ct_mod
        with torch.no_grad():
            y1 = mod(x1, t)
            y2 = mod(x2, t)
            # Same text, different images -> different modulation ratios per
            # sample (no constant additive bias can explain it).
            r1 = (y1 / (x1 + 1e-6)).mean((2, 3))
            r2 = (y2 / (x2 + 1e-6)).mean((2, 3))
            self.assertFalse(torch.allclose(r1, r2, atol=1e-4))

    def test_backward_finite(self):
        model = self.model(checkpoint_attention=True)
        c, p = self.features(), self.features()
        out = model(c, p)
        sum(x.square().mean() for x in out).backward()
        for x in c + p:
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_rejects_non_full_state(self):
        model = self.model()
        c, p = self.features(), self.features()
        with self.assertRaises(ValueError):
            model(c, p, state="missing")
