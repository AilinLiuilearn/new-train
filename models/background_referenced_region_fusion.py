# -*- coding: utf-8 -*-
"""Background-referenced region-to-local PET/CT fusion (research candidate).

Destination: new-train/models/background_referenced_region_fusion.py
Dependencies: Python >= 3.10, PyTorch >= 2.1. Matplotlib only for visualization.

This is an ORIGINAL implementation of the design discussed on 2026-10-10,
not a reproduction of CHAL/DCIM and not a validated medical segmentation method.
Source mechanisms inspected:
  CIPA / DCIM, CVPR 2025, https://arxiv.org/abs/2503.17261
  https://github.com/mj129/CIPA/blob/main/models/encoders/local_vmamba/region_mamba.py
  (PET regional context -> CT local tokens; original uses Mamba and broadcast.)
  CHAL, CVPR 2026, https://github.com/UESTC-nnLab/CHAL
  nets/neural_background_field.py (background prediction -> anomaly evidence;
  original uses video, NOT the spatial blind-region model implemented here).
No source Mamba/temporal/causal implementation is copied or silently replaced.

Implemented equations, separately at every scale:
  a = (PET_raw - mu) / sigma, detached from the segmentation graph
  wA = softmax(a / tauA); wB = softmax(-abs(a) / tauB)
  uCA/uCB = leave-one-position-out weighted CT contexts
  d = cos(qCT,uCA) - cos(qCT,uCB)
  pPET = sigmoid(zPET); pCTPET = sigmoid(zPET + eta*d)
  delta = pCTPET - pPET
  message = delta * (weighted_PET_A - weighted_PET_B)
  fused = CT + PET + zero_initialized_projection(overlap_average(message))

Contracts:
* Features are already aligned, shallow -> deep; default channels 64/128/320/512.
  Native ConvNeXt channels are NOT accepted in place of aligned features.
* Raw PET is [B,1,H,W], float in [0,1], AFTER the SAME spatial augmentation,
  BEFORE ImageNet/cipa normalization. Helpers below can invert those EXACT
  normalization modes for the baseline's duplicated grayscale input.
* background_train_mask is [B,1,H,W], bool or 0/1, True ONLY where a background
  target is valid. It is consumed by the loss only, NEVER by prediction/routing.
  make_background_train_mask excludes dilated foreground; pass an independent
  valid tissue/ROI mask when available to avoid black canvas dominating fitting.
* Background fitting is necessary: total_loss = original_seg_loss +
  lambda_bg * result.background_loss. lambda_bg is an experiment hyperparameter,
  NOT silently set here. A randomly initialized/frozen background predictor is
  not a meaningful normality model. Normal background parameters receive only
  background NLL gradients; segmentation cannot rewrite the background target.
* Full: requires real PET features AND raw PET. Missing: aligned CT unchanged;
  never calls the background model/fusion blocks, never reads missing PET rows.
  Mixed batch accepts PET tensors in full-batch or compact Full-row order.
* This file does NOT instantiate encoders/decoder or edit a training entry point.
  It does not implement Stage-2 retrieval. Add an independent runner on integration.
* Keep module parameters FP32; use torch.autocast for AMP, not model.half().
  Scores, leave-one-out subtraction, overlap accumulation and NLL use FP32.
* Gaussian mean/sigma and background-centered weighting are hypotheses, not
  calibrated lesion probabilities. CT leave-one-out removes direct self-membership,
  NOT receptive-field overlap. Region aggregation is lossy, not evidence recovery.

Explicit implementation choices (not paper defaults):
* Background: raw-image masked convolutions with 9/17 kernels, a common 5x5 hole,
  then ONLY pointwise layers. Zero padding, never reflection (which could copy a
  center near image borders). Gaussian sigma constrained to [0.01,0.5] for [0,1].
* Evidence mapping: signed_peak keeps the sign of the largest absolute deviation
  in each feature cell. It can amplify noise. area is an explicit pooling ablation.
* Regions: 8x8 feature cells, stride 4, same structure/independent weights on all
  scales. Padding is masked out of weights and queries; overlaps are averaged.
* No hard foreground threshold, lesion ROI crop, pretrained background bank,
  global all-pairs attention, boundary loss, or feature reconstruction loss.

Minimal Full use (must be registered in model.__init__, then optimizer sees it):
  fusion = BackgroundReferencedRegionFusion()
  raw = pet_to_unit_interval(batch['pet'], mode='imagenet')
  bgmask = make_background_train_mask(batch['mask'], valid_mask=body_mask)
  result = fusion(ct_aligned, pet_aligned, raw,
                  background_train_mask=bgmask, return_diagnostics=True)
  out = original_decoder(result.fused, target_size=raw.shape[-2:])
  loss = original_seg_loss(out, gt) + lambda_bg * result.background_loss
  # Visualize occasionally (not every training batch):
  save_fusion_diagnostics(result, 'fusion_debug.png')

Checks (synthetic contract checks, NOT a dataset training runner):
  python background_referenced_region_fusion.py --self-test
  python background_referenced_region_fusion.py --smoke-512
  python background_referenced_region_fusion.py --demo visualization.png
"""
from __future__ import annotations

import argparse
import io
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class FusionResult(NamedTuple):
    fused: List[Tensor]
    background_loss: Optional[Tensor]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class FusionConfig:
    channels: Tuple[int, ...] = (64, 128, 320, 512)
    descriptor_dim: int = 32
    region_size: int = 8
    region_stride: int = 4
    window_chunk: int = 256
    evidence_pool: str = 'signed_peak'
    bg_kernels: Tuple[int, ...] = (9, 17)
    bg_hole: int = 5
    bg_ring_channels: int = 4
    bg_hidden: int = 16
    sigma_min: float = 0.01
    sigma_max: float = 0.5
    temperature_min: float = 0.1
    temperature_max: float = 10.0
    eta_max: float = 4.0
    prior_logit_limit: float = 6.0
    loo_min_mass: float = 1e-4
    checkpoint_windows: bool = True
    checkpoint_background: bool = True
    validate_values: bool = True

    def __post_init__(self):
        if not self.channels or any(c < 2 for c in self.channels):
            raise ValueError('channels must be nonempty and >=2')
        if self.descriptor_dim < 2:
            raise ValueError('descriptor_dim must be >=2')
        if not 1 <= self.region_stride <= self.region_size or self.region_size < 2:
            raise ValueError('require 1 <= stride <= region_size, region_size >=2')
        if self.window_chunk < 1 or self.bg_ring_channels < 1 or self.bg_hidden < 1:
            raise ValueError('chunk and background dimensions must be positive')
        if self.evidence_pool not in ('signed_peak', 'area'):
            raise ValueError('evidence_pool must be signed_peak or area')
        if self.bg_hole < 1 or self.bg_hole % 2 == 0 or not self.bg_kernels:
            raise ValueError('bg_hole must be positive odd; need >=1 kernel')
        if any(k % 2 == 0 or k <= self.bg_hole for k in self.bg_kernels):
            raise ValueError('background kernels must be odd and larger than hole')
        if not 0 < self.sigma_min < self.sigma_max:
            raise ValueError('require 0 < sigma_min < sigma_max')
        if not 0 < self.temperature_min < 1.0 < self.temperature_max:
            raise ValueError('temperature bounds must enclose initial temperature 1')
        if not self.eta_max > 1 or not self.prior_logit_limit > 0:
            raise ValueError('eta_max must exceed 1, prior limit must be positive')
        if not 0 < self.loo_min_mass < 1:
            raise ValueError('invalid loo_min_mass')


def _binary_mask(x: Tensor, name: str) -> Tensor:
    if x.dtype == torch.bool:
        return x.detach()
    if not torch.all((x == 0) | (x == 1)):
        raise ValueError(f'{name} must contain only 0/1 (not 0/255 or soft labels)')
    return x.detach().bool()


def make_background_train_mask(foreground: Tensor, valid_mask: Optional[Tensor] = None,
                               exclusion_radius: int = 2) -> Tensor:
    """Training-loss selection only. valid_mask may be body/lung ROI or valid canvas.

    Without valid_mask, all non-foreground pixels (including canvas) are eligible.
    That fallback is explicit and may bias fitting; it is not a tissue extractor.
    """
    if foreground.ndim != 4 or foreground.shape[1] != 1 or exclusion_radius < 0:
        raise ValueError('foreground must be [B,1,H,W], radius >=0')
    fg = _binary_mask(foreground, 'foreground')
    if exclusion_radius:
        k = 2 * exclusion_radius + 1
        fg = F.max_pool2d(fg.float(), k, 1, exclusion_radius).bool()
    eligible = ~fg
    if valid_mask is not None:
        if valid_mask.shape != foreground.shape or valid_mask.device != foreground.device:
            raise ValueError('valid_mask must match foreground shape/device')
        eligible = eligible & _binary_mask(valid_mask, 'valid_mask')
    return eligible


def pet_to_unit_interval(pet: Tensor, mode: str = 'imagenet') -> Tensor:
    """Invert the inspected baseline normalization, WITHOUT silently clipping.

    For ImageNet expect its duplicated grayscale 3-channel tensor. Arbitrary color
    PET cannot be converted by this helper. Returning one channel preserves the
    grayscale value and the dataset's synchronized spatial augmentation.
    """
    if pet.ndim != 4 or not pet.is_floating_point():
        raise ValueError('PET must be floating BCHW')
    x = pet.detach().float()
    if mode == 'imagenet':
        if x.shape[1] != 3:
            raise ValueError('imagenet inversion requires three normalized channels')
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        x = x * std + mean
    elif mode == 'cipa':
        x = (x + 1.6) / 3.2
    elif mode != 'unit':
        raise ValueError('mode must be imagenet, cipa, or unit')
    if x.shape[1] not in (1, 3):
        raise ValueError('expect one or duplicated three grayscale channels')
    if x.shape[1] == 3 and not torch.allclose(x, x[:, :1].expand_as(x), atol=2e-5, rtol=0):
        raise ValueError('inverted channels differ: not duplicated grayscale PET')
    x = x[:, :1]
    if not torch.isfinite(x).all() or x.min() < -2e-5 or x.max() > 1 + 2e-5:
        raise ValueError('inverted PET outside [0,1]: check actual preprocessing')
    return x.clamp(0, 1)  # only removes the verified floating-point inversion error


class _RingConv(nn.Module):
    """Exactly one spatial mixing operation with a permanently excluded center."""
    def __init__(self, channels: int, kernel: int, hole: int):
        super().__init__()
        self.radius = kernel // 2
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(channels))
        mask = torch.ones(1, 1, kernel, kernel)
        lo = (kernel - hole) // 2
        mask[..., lo:lo + hole, lo:lo + hole] = 0
        self.register_buffer('support_mask', mask)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, raw: Tensor) -> Tensor:
        return F.conv2d(raw, self.weight * self.support_mask, self.bias,
                        padding=self.radius)


class SpatialBackgroundReference(nn.Module):
    """Non-target conditional Gaussian. No BN/GN/global pool/spatial layer after rings.

    This strict graph restriction prevents the center being leaked back through
    adjacent activations. Normal parameters stay FP32 under outer AMP.
    """
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.rings = nn.ModuleList([
            _RingConv(config.bg_ring_channels, k, config.bg_hole)
            for k in config.bg_kernels])
        self.pointwise = nn.Sequential(
            nn.Conv2d(len(config.bg_kernels) * config.bg_ring_channels, config.bg_hidden, 1),
            nn.GELU(), nn.Conv2d(config.bg_hidden, 2, 1))
        # Initial mean 0.1 and sigma 0.1 are starting values, NOT fitted statistics.
        with torch.no_grad():
            self.pointwise[-1].bias[0] = math.log(0.1 / 0.9)
            fraction = (min(max(0.1, config.sigma_min + 1e-5), config.sigma_max - 1e-5)
                        - config.sigma_min) / (config.sigma_max - config.sigma_min)
            self.pointwise[-1].bias[1] = math.log(fraction / (1 - fraction))

    def _predict(self, raw: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.autocast(device_type=raw.device.type, enabled=False):
            h = torch.cat([F.gelu(layer(raw.float())) for layer in self.rings], dim=1)
            mu_logit, sigma_logit = self.pointwise(h).chunk(2, dim=1)
            mu = mu_logit.sigmoid()
            sigma = self.config.sigma_min + (
                self.config.sigma_max - self.config.sigma_min) * sigma_logit.sigmoid()
            return mu, sigma

    def forward(self, raw: Tensor) -> Tuple[Tensor, Tensor]:
        raw = raw.detach().float()
        if self.config.checkpoint_background and self.training and torch.is_grad_enabled():
            return checkpoint(self._predict, raw, use_reentrant=False, preserve_rng_state=False)
        return self._predict(raw)

    @staticmethod
    def loss(raw: Tensor, mu: Tensor, sigma: Tensor, mask: Tensor) -> Tensor:
        """Gaussian NLL, averaged within valid images, then across valid images.

        The constant is omitted, so NLL may be negative; do NOT clamp to zero.
        An empty mask returns graph-connected zero, never NaN.
        """
        eligible = _binary_mask(mask, 'background_train_mask')
        if eligible.shape != raw.shape or eligible.device != raw.device:
            raise ValueError('background mask must match raw PET shape/device')
        with torch.autocast(device_type=raw.device.type, enabled=False):
            nll = 0.5 * ((raw.detach().float() - mu.float()) / sigma.float()).square()
            nll = nll + sigma.float().log()
            n = eligible.flatten(1).sum(1)
            sums = torch.where(eligible, nll, torch.zeros_like(nll)).flatten(1).sum(1)
            per_image = sums / n.clamp_min(1)
            return (per_image * (n > 0)).sum() / (n > 0).sum().clamp_min(1)


class _PixelChannelNorm(nn.Module):
    """Per-position channel LN; no batch statistics or spatial mixing."""
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            centered = xf - xf.mean(1, keepdim=True)
            return centered * torch.rsqrt(centered.square().mean(1, keepdim=True) + 1e-5) * self.weight + self.bias


def _map_evidence(a: Tensor, size: Tuple[int, int], mode: str) -> Tensor:
    """No trainable mapper, no label input, no interpolation upsampling of evidence."""
    if size[0] > a.shape[-2] or size[1] > a.shape[-1]:
        raise ValueError('feature grid cannot exceed raw PET grid')
    if mode == 'area':
        return F.adaptive_avg_pool2d(a, size)
    hi = F.adaptive_max_pool2d(a, size)
    lo = -F.adaptive_max_pool2d(-a, size)
    return torch.where(hi.abs() >= lo.abs(), hi, lo)


class RegionLocalCorrection(nn.Module):
    """Single-scale signed correction; shares structure, not weights, across scales."""
    def __init__(self, channels: int, config: FusionConfig):
        super().__init__()
        self.config = config
        d = config.descriptor_dim
        self.ct_descriptor = nn.Sequential(
            _PixelChannelNorm(channels), nn.Conv2d(channels, d, 1, bias=False),
            nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False), nn.GELU())
        # V acts on the original aligned PET, not its normalized/anomaly map.
        self.pet_value = nn.Conv2d(channels, d, 1, bias=False)
        self.out_projection = nn.Conv2d(d, channels, 1, bias=False)
        nn.init.zeros_(self.out_projection.weight)
        t0 = (1.0 - config.temperature_min) / (config.temperature_max - config.temperature_min)
        self.temperature_logits = nn.Parameter(torch.full((3,), math.log(t0 / (1 - t0))))
        self.eta_logit = nn.Parameter(torch.tensor(math.log(1.0 / (config.eta_max - 1.0))))
        self.prior_bias = nn.Parameter(torch.tensor(-1.0))

    def _window_message(self, e: Tensor, v: Tensor, a: Tensor,
                        valid: Tensor) -> Tuple[Tensor, Tensor]:
        # e,v: [windows,d,r*r], a,valid: [windows,r*r]. FP32 cancellation safety.
        cfg = self.config
        with torch.autocast(device_type=e.device.type, enabled=False):
            e, v, a = e.float(), v.float(), a.float()
            temperatures = cfg.temperature_min + (
                cfg.temperature_max - cfg.temperature_min) * self.temperature_logits.sigmoid()
            wa = (a / temperatures[0]).masked_fill(~valid, -torch.inf).softmax(-1)
            wb = (-a.abs() / temperatures[1]).masked_fill(~valid, -torch.inf).softmax(-1)
            total_a = (e * wa[:, None]).sum(-1, keepdim=True)
            total_b = (e * wb[:, None]).sum(-1, keepdim=True)
            mass_a, mass_b = 1.0 - wa, 1.0 - wb
            ca = (total_a - e * wa[:, None]) / mass_a[:, None].clamp_min(cfg.loo_min_mass)
            cb = (total_b - e * wb[:, None]) / mass_b[:, None].clamp_min(cfg.loo_min_mass)
            usable = (valid & (mass_a > cfg.loo_min_mass) & (mass_b > cfg.loo_min_mass)
                      & (ca.square().sum(1) > 1e-12) & (cb.square().sum(1) > 1e-12)
                      & (e.square().sum(1) > 1e-12))
            comparison = (F.normalize(e, dim=1, eps=1e-6) * (
                F.normalize(ca, dim=1, eps=1e-6) - F.normalize(cb, dim=1, eps=1e-6))).sum(1)
            comparison = torch.where(usable, comparison, torch.zeros_like(comparison))
            z = cfg.prior_logit_limit * torch.tanh(
                (a / temperatures[2] + self.prior_bias) / cfg.prior_logit_limit)
            prior = z.sigmoid()
            eta = cfg.eta_max * self.eta_logit.sigmoid()
            corrected = (z + eta * comparison).sigmoid()
            delta = (corrected - prior) * valid
            pet_contrast = (v * (wa - wb)[:, None]).sum(-1, keepdim=True)
            message = delta[:, None] * pet_contrast
            # These maps are routing descriptors, not foreground probabilities.
            diagnostic = torch.stack((prior * valid, corrected * valid, delta,
                                      comparison, usable.float()), dim=1)
            return message, diagnostic

    def forward(self, ct: Tensor, pet: Tensor, evidence: Tensor,
                return_diagnostics: bool = False) -> Tuple[Tensor, Dict[str, Tensor]]:
        cfg = self.config
        b, _, h, w = ct.shape
        r, s, d = cfg.region_size, cfg.region_stride, cfg.descriptor_dim
        nh = max(1, math.ceil((h - r) / s) + 1)
        nw = max(1, math.ceil((w - r) / s) + 1)
        hp, wp = (nh - 1) * s + r, (nw - 1) * s + r
        e = self.ct_descriptor(ct).float()
        v = self.pet_value(pet).float()
        a = _map_evidence(evidence.detach().float(), (h, w), cfg.evidence_pool)
        padding = (0, wp - w, 0, hp - h)
        # unfold is a VIEW; only the current indexed chunk is materialized.
        def windows(x: Tensor) -> Tensor:
            return F.pad(x, padding).unfold(2, r, s).unfold(3, r, s).permute(0, 2, 3, 1, 4, 5)
        ev, vv, av = windows(e), windows(v), windows(a)
        valid_view = windows(torch.ones_like(a, dtype=torch.bool))
        accum = e.new_zeros(b * hp * wp, d)
        diag_accum = e.new_zeros(b * hp * wp, 5) if return_diagnostics else None
        offset_y = torch.arange(r, device=ct.device).repeat_interleave(r)
        offset_x = torch.arange(r, device=ct.device).repeat(r)
        for start in range(0, b * nh * nw, cfg.window_chunk):
            ids = torch.arange(start, min(start + cfg.window_chunk, b * nh * nw), device=ct.device)
            ib = ids // (nh * nw)
            iy, ix = (ids % (nh * nw)) // nw, ids % nw
            ew = ev[ib, iy, ix].flatten(-2)
            vw = vv[ib, iy, ix].flatten(-2)
            aw = av[ib, iy, ix].flatten(1)
            valid = valid_view[ib, iy, ix].flatten(1)
            if cfg.checkpoint_windows and self.training and torch.is_grad_enabled():
                msg, maps = checkpoint(self._window_message, ew, vw, aw, valid,
                                       use_reentrant=False, preserve_rng_state=False)
            else:
                msg, maps = self._window_message(ew, vw, aw, valid)
            index = (ib[:, None] * hp * wp + (iy[:, None] * s + offset_y) * wp
                     + ix[:, None] * s + offset_x).reshape(-1)
            accum.index_add_(0, index, msg.transpose(1, 2).reshape(-1, d))
            if diag_accum is not None:
                diag_accum.index_add_(0, index, maps.detach().transpose(1, 2).reshape(-1, 5))
        count = F.fold(e.new_ones(1, r * r, nh * nw), (hp, wp), r, stride=s)
        message = accum.view(b, hp, wp, d).permute(0, 3, 1, 2) / count
        message = message[..., :h, :w]
        correction = self.out_projection(message)
        base = ct + pet
        fused = base + correction.to(base.dtype)
        diagnostics: Dict[str, Tensor] = {}
        if diag_accum is not None:
            maps = diag_accum.view(b, hp, wp, 5).permute(0, 3, 1, 2) / count
            names = ('pet_prior', 'ct_corrected', 'signed_delta', 'ct_comparison', 'usable_fraction')
            diagnostics = {name: maps[:, i:i + 1, :h, :w].detach() for i, name in enumerate(names)}
            diagnostics['evidence'] = a.detach()
            diagnostics['correction_rms'] = correction.detach().float().square().mean(1, keepdim=True).sqrt()
        return fused, diagnostics


class BackgroundReferencedRegionFusion(nn.Module):
    """Four-scale standalone module; all labels stay out of the fusion forward graph.

    pet_available: bool/0-1 integer [B]. None means all Full. PET features/raw/mask
    may be [B,...] or [N_full,...] in ascending full_idx order. Only Full rows are
    evaluated. If all Missing, PET arguments may be absent or otherwise invalid;
    no PET validation or branch computation occurs.
    """
    def __init__(self, **kwargs: Any):
        super().__init__()
        for key in ('channels', 'bg_kernels'):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        self.config = FusionConfig(**kwargs)
        self.background = SpatialBackgroundReference(self.config)
        self.scales = nn.ModuleList([RegionLocalCorrection(c, self.config) for c in self.config.channels])

    def get_config(self) -> Dict[str, Any]:
        return asdict(self.config)

    @staticmethod
    def _select(t: Tensor, idx: Tensor, batch: int, name: str) -> Tensor:
        if t.shape[0] == batch:
            return t.index_select(0, idx)
        if t.shape[0] == idx.numel():
            return t
        raise ValueError(f'{name} batch must be B or N_full')

    def forward(self, ct_features: Sequence[Tensor], pet_features: Optional[Sequence[Tensor]] = None,
                pet_raw: Optional[Tensor] = None, *, pet_available: Optional[Tensor] = None,
                background_train_mask: Optional[Tensor] = None,
                return_diagnostics: bool = False) -> FusionResult:
        cfg = self.config
        if len(ct_features) != len(cfg.channels):
            raise ValueError('wrong number of CT scales')
        first = ct_features[0]
        if first.ndim != 4 or first.shape[0] < 1:
            raise ValueError('CT features must be nonempty BCHW')
        batch, device = first.shape[0], first.device
        if next(self.parameters()).device != device:
            raise ValueError('move module to the feature device before forward')
        if next(self.parameters()).dtype != torch.float32:
            raise ValueError('keep fusion parameters FP32; use autocast instead of .half()')
        previous_size = None
        for c, channels in zip(ct_features, cfg.channels):
            if c.ndim != 4 or c.shape[:2] != (batch, channels) or c.device != device or not c.is_floating_point():
                raise ValueError('CT features must match configured channels/batch/device')
            if min(c.shape[-2:]) < 1:
                raise ValueError('empty spatial dimension')
            if previous_size and any(a > b for a, b in zip(c.shape[-2:], previous_size)):
                raise ValueError('features must be shallow -> deep')
            previous_size = c.shape[-2:]
            if cfg.validate_values and not torch.isfinite(c).all():
                raise ValueError('CT contains nonfinite values')
        if pet_available is None:
            state = torch.ones(batch, dtype=torch.bool, device=device)
        else:
            state = torch.as_tensor(pet_available, device=device)
            if state.ndim != 1 or state.numel() != batch or state.is_floating_point():
                raise ValueError('pet_available must be bool/0-1 integer [B]')
            state = _binary_mask(state, 'pet_available')
        idx = state.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return FusionResult(list(ct_features), None, {'full_indices': idx.detach()} if return_diagnostics else {})
        if pet_features is None or pet_raw is None or len(pet_features) != len(cfg.channels):
            raise ValueError('Full requires all PET feature scales AND pet_raw')
        if pet_raw.ndim != 4 or pet_raw.shape[1] != 1 or pet_raw.device != device or not pet_raw.is_floating_point():
            raise ValueError('pet_raw must be floating [B or N_full,1,H,W] on feature device')
        raw = self._select(pet_raw, idx, batch, 'pet_raw').detach().float()
        if cfg.validate_values and (not torch.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1):
            raise ValueError('Full raw PET must be finite in [0,1]')
        pfs = []
        for c, p in zip(ct_features, pet_features):
            if p.ndim != 4 or p.device != device or not p.is_floating_point():
                raise ValueError('invalid PET feature tensor')
            p = self._select(p, idx, batch, 'PET features')
            if p.shape[1:] != c.shape[1:] or p.dtype != c.dtype:
                raise ValueError('aligned PET and CT shapes/dtypes must match per scale')
            if cfg.validate_values and not torch.isfinite(p).all():
                raise ValueError('Full PET features contain nonfinite values')
            pfs.append(p)
        # No background autograd graph at inference, or when only segmentation is
        # requested. Background parameters are trained solely by the returned NLL.
        need_bg_grad = self.training and torch.is_grad_enabled() and background_train_mask is not None
        with torch.set_grad_enabled(need_bg_grad):
            mu, sigma = self.background(raw)
            bg_loss = None
            if background_train_mask is not None:
                if background_train_mask.device != device:
                    raise ValueError('background_train_mask must be on feature device')
                mask = self._select(background_train_mask, idx, batch, 'background_train_mask')
                bg_loss = self.background.loss(raw, mu, sigma, mask)
        evidence = ((raw - mu.detach()) / sigma.detach()).detach()
        fused, scale_diags = [], []
        for c, p, block in zip(ct_features, pfs, self.scales):
            full, maps = block(c.index_select(0, idx), p, evidence, return_diagnostics)
            fused.append(c.index_copy(0, idx, full.to(c.dtype)))
            if return_diagnostics:
                scale_diags.append(maps)
        diag: Dict[str, Any] = {}
        if return_diagnostics:
            diag = {'full_indices': idx.detach(), 'pet_raw': raw.detach(), 'background_mean': mu.detach(),
                    'background_sigma': sigma.detach(), 'raw_evidence': evidence, 'scales': scale_diags}
        return FusionResult(fused, bg_loss, diag)


def save_fusion_diagnostics(result: FusionResult, path: str, full_row: int = 0) -> None:
    """Plot detached Full-row maps; no CT/PET tumor probability claim. Optional matplotlib.

    full_row indexes the COMPACT Full batch; original batch indices are in
    result.diagnostics['full_indices']. Save infrequently to avoid training I/O.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    diag = result.diagnostics
    if 'scales' not in diag:
        raise ValueError('need a Full result with return_diagnostics=True')
    if not 0 <= full_row < len(diag['full_indices']):
        raise ValueError('full_row outside compact Full batch')
    stages = diag['scales']
    fig, axes = plt.subplots(1 + len(stages), 5, figsize=(17, 3.2 * (1 + len(stages))))
    raw_names = ['pet_raw', 'background_mean', 'background_sigma', 'raw_evidence']
    stage_names = ['evidence', 'pet_prior', 'ct_corrected', 'signed_delta', 'correction_rms']
    def draw(ax: Any, tensor: Tensor, title: str, signed: bool = False):
        data = tensor[full_row, 0].detach().float().cpu().numpy()
        if signed:
            bound = max(float(abs(data).max()), 1e-6)
            im = ax.imshow(data, cmap='coolwarm', vmin=-bound, vmax=bound)
        else:
            im = ax.imshow(data, cmap='viridis')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.045)
    for i, name in enumerate(raw_names):
        draw(axes[0, i], diag[name], name, name == 'raw_evidence')
    axes[0, 4].axis('off')
    original = int(diag['full_indices'][full_row].cpu())
    axes[0, 4].text(0, 0.5, f'Original batch row: {original}\nRouting maps are NOT lesion probabilities.\nZero-init: correction_rms = 0.', fontsize=10)
    for s, maps in enumerate(stages):
        for j, name in enumerate(stage_names):
            draw(axes[s + 1, j], maps[name], f's{s + 1}: {name}', name in ('evidence', 'signed_delta'))
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _self_test() -> None:
    """Fast independent contract checks. No real data, checkpoints, or trainer."""
    from unittest.mock import patch
    torch.manual_seed(7)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    model = BackgroundReferencedRegionFusion(channels=(8, 12, 16, 20), descriptor_dim=6,
        region_size=4, region_stride=2, window_chunk=13,
        bg_kernels=(5, 7), bg_hole=3, bg_ring_channels=2, bg_hidden=6)
    sizes = [(13, 11), (7, 6), (4, 3), (2, 1)]
    ct = [torch.randn(2, c, h, w, requires_grad=True) for c, (h, w) in zip(model.config.channels, sizes)]
    pet = [torch.randn_like(c, requires_grad=True) for c in ct]
    raw = torch.rand(2, 1, 26, 22, requires_grad=True)
    mask = torch.ones_like(raw, dtype=torch.bool)
    model.train()
    result = model(ct, pet, raw, background_train_mask=mask, return_diagnostics=True)
    for out, c, p in zip(result.fused, ct, pet):
        torch.testing.assert_close(out, c + p, rtol=0, atol=0)
    assert result.background_loss is not None and torch.isfinite(result.background_loss)
    sum(x.square().mean() for x in result.fused).backward()
    assert all(p.grad is None for p in model.background.parameters()), 'segmentation leaked into background'
    assert raw.grad is None
    assert all(c.grad is not None and torch.isfinite(c.grad).all() for c in ct + pet)
    assert model.scales[0].out_projection.weight.grad.abs().sum() > 0
    result.background_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.background.parameters())
    assert raw.grad is None
    for ring in model.background.rings:
        assert torch.count_nonzero(ring.weight.grad * (1 - ring.support_mask)) == 0

    # Strict raw-center exclusion, including after multiple pointwise nonlinearities.
    bg = model.background
    bg.eval()
    x1 = torch.rand(1, 1, 17, 17)
    x2 = x1.clone()
    x2[..., 7:10, 7:10] = 1 - x2[..., 7:10, 7:10]
    with torch.no_grad():
        mu1, sd1 = bg(x1)
        mu2, sd2 = bg(x2)
    torch.testing.assert_close(mu1[..., 8, 8], mu2[..., 8, 8], rtol=0, atol=0)
    torch.testing.assert_close(sd1[..., 8, 8], sd2[..., 8, 8], rtol=0, atol=0)

    # Missing must not call ANY PET-dependent module, even with junk PET objects.
    with patch.object(model.background, 'forward', side_effect=AssertionError('PET branch called')):
        missing = model(ct, None, None, pet_available=torch.zeros(2, dtype=torch.bool))
        assert all(a is b for a, b in zip(missing.fused, ct))
        assert missing.background_loss is None
    with torch.no_grad():
        for stage in model.scales:
            nn.init.normal_(stage.out_projection.weight, std=0.02)
    model.eval()
    p_mixed = [p.detach().clone() for p in pet]
    raw_mixed = raw.detach().clone()
    for p in p_mixed:
        p[1] = float('nan')
    raw_mixed[1] = float('nan')
    state = torch.tensor([True, False])
    with torch.no_grad():
        mixed = model(ct, p_mixed, raw_mixed, pet_available=state)
        compact = model(ct, [p[:1] for p in p_mixed], raw_mixed[:1], pet_available=state)
        solo = model([c[:1] for c in ct], [p[:1] for p in pet], raw[:1])
        for i in range(4):
            torch.testing.assert_close(mixed.fused[i][0], solo.fused[i][0])
            torch.testing.assert_close(mixed.fused[i][1], ct[i][1], rtol=0, atol=0)
            torch.testing.assert_close(compact.fused[i], mixed.fused[i], rtol=0, atol=0)

    # Graph routes once projections become nonzero; bg loss must not train encoders.
    model.train()
    model.zero_grad(set_to_none=True)
    ct2 = [c.detach().clone().requires_grad_() for c in ct]
    pet2 = [p.detach().clone().requires_grad_() for p in pet]
    out = model(ct2, pet2, raw.detach(), background_train_mask=mask)
    out.background_loss.backward()
    assert all(x.grad is None for x in ct2 + pet2)
    model.zero_grad(set_to_none=True)
    sum(x.square().mean() for x in out.fused).backward()
    assert all(p.grad is None for p in model.background.parameters())
    assert model.scales[0].ct_descriptor[1].weight.grad.abs().sum() > 0
    assert model.scales[0].pet_value.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    # Label/mask leakage: only background loss may change; outputs must be identical.
    model.eval()
    with torch.no_grad():
        yes = model(ct, pet, raw, background_train_mask=mask)
        no = model(ct, pet, raw, background_train_mask=torch.zeros_like(mask))
        for x, y in zip(yes.fused, no.fused):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        assert no.background_loss == 0

    # Window operator versus explicit leave-one-out reference, including masking.
    block = model.scales[0]
    e, v = torch.randn(2, 6, 5), torch.randn(2, 6, 5)
    a = torch.tensor([[0., 1., 2., -1., 0.2], [0., 0., 0., 0., 0.]])
    valid = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    message, maps = block._window_message(e, v, a, valid)
    taus = model.config.temperature_min + (model.config.temperature_max - model.config.temperature_min) * block.temperature_logits.sigmoid()
    wa = (a / taus[0]).masked_fill(~valid, -torch.inf).softmax(-1)
    wb = (-a.abs() / taus[1]).masked_fill(~valid, -torch.inf).softmax(-1)
    for i in range(4):
        keep = valid[0].clone()
        keep[i] = False
        ca = (e[0, :, keep] * wa[0, keep]).sum(-1) / wa[0, keep].sum()
        cb = (e[0, :, keep] * wb[0, keep]).sum(-1) / wb[0, keep].sum()
        expected = F.cosine_similarity(e[0, :, i], ca, dim=0) - F.cosine_similarity(e[0, :, i], cb, dim=0)
        torch.testing.assert_close(maps[0, 3, i], expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(message[1], torch.zeros_like(message[1]), atol=0, rtol=0)
    assert message[0, :, 4].abs().sum() == 0
    # Both signs are realizable; not a positive-only attention implementation.
    e3 = torch.tensor([[[1., 0.9, 0., 0.1], [0., 0.1, 1., 0.9]]])
    e3 = F.pad(e3, (0, 0, 0, 4))
    _, ms = block._window_message(e3, torch.randn_like(e3), torch.tensor([[3., 2., 0., 0.]]), torch.ones(1, 4, dtype=torch.bool))
    assert ms[:, 2].max() > 0 and ms[:, 2].min() < 0

    # Exact baseline normalization round trip.
    unit = torch.rand(2, 1, 7, 9)
    mean = unit.new_tensor([.485, .456, .406])[None, :, None, None]
    std = unit.new_tensor([.229, .224, .225])[None, :, None, None]
    torch.testing.assert_close(pet_to_unit_interval((unit - mean) / std), unit)
    torch.testing.assert_close(pet_to_unit_interval(unit * 3.2 - 1.6, 'cipa'), unit)
    fg = torch.zeros_like(unit)
    fg[..., 3, 4] = 1
    assert not make_background_train_mask(fg, exclusion_radius=1)[..., 2:5, 3:6].any()

    # Checkpoint round-trip with explicit serializable constructor configuration.
    stream = io.BytesIO()
    torch.save({'fusion_config': model.get_config(), 'fusion_state': model.state_dict()}, stream)
    stream.seek(0)
    saved = torch.load(stream, map_location='cpu', weights_only=True)
    restored = BackgroundReferencedRegionFusion(**saved['fusion_config']).eval()
    restored.load_state_dict(saved['fusion_state'], strict=True)
    with torch.no_grad():
        reloaded = restored(ct, pet, raw)
        for x, y in zip(reloaded.fused, yes.fused):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    print('PASS: four scales, zero-init, signed/LOO math, odd/small grids, Full/Missing/mixed,')
    print('      no label/PET leakage, blind region, gradient separation, checkpoint reload.')


def _smoke_512(demo_path: Optional[str] = None) -> None:
    torch.manual_seed(11)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    model = BackgroundReferencedRegionFusion().eval()
    ct = [torch.randn(1, c, n, n) for c, n in zip(model.config.channels, (128, 64, 32, 16))]
    pet = [torch.randn_like(c) for c in ct]
    raw = torch.rand(1, 1, 512, 512) * 0.03
    raw[..., 220:238, 260:278] += 0.5
    start = time.perf_counter()
    with torch.inference_mode():
        result = model(ct, pet, raw, return_diagnostics=demo_path is not None)
    for f, c, p in zip(result.fused, ct, pet):
        torch.testing.assert_close(f, c + p, rtol=0, atol=0)
    print(f'CPU B=1 512 smoke: {time.perf_counter() - start:.3f}s; '
          f'parameters={sum(p.numel() for p in model.parameters()):,}')
    if demo_path:
        save_fusion_diagnostics(result, demo_path)
        print(f'Saved SYNTHETIC, UNTRAINED diagnostic example: {demo_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--smoke-512', action='store_true')
    parser.add_argument('--demo', type=str, help='synthetic, untrained visualization output path')
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    if args.smoke_512 or args.demo:
        _smoke_512(args.demo)
    if not (args.self_test or args.smoke_512 or args.demo):
        parser.print_help()
