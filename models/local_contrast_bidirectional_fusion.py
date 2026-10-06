"""Standalone local-contrast-guided bidirectional PET/CT feature fusion.

Destination: models/local_contrast_bidirectional_fusion.py
Requires Python >=3.10, PyTorch >=2.1. No other dependencies.

This is the agreed task adaptation, NOT an HDNet/FCENet reproduction:
* HDNet MAC inspiration: trainable center-surround initialized depthwise
  3x3 convolutions at dilation 1 and 2; GN and residual descriptor mixing.
  Kernel initialization matches the 3x3 center-surround construction in
  https://github.com/iLearn-Lab/TGRS25-HDNet/blob/main/model/MAC_Kernel.py
  Original MAC's grouped cascade/BN/full 3x3 mixing is deliberately replaced
  by the previously agreed lightweight descriptor architecture.
* FCENet (CVPR 2025): preserve modality-specific content in addition to
  exchanged information. Its Fourier FDSM/CRM/DRM are NOT implemented here.
  https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Complementary_Advantages_Exploiting_Cross-Field_Frequency_Correlation_for_NIR-Assisted_Image_Denoising_CVPR_2025_paper.html

All four scales use identical architecture/configuration, independent weights.
Full: fused=(ct+delta_ct)+(pet+delta_pet). Final projections are zero-initialized.
Missing: fused=ct; no cross-modal module is executed. No PET synthesis.
Default pyramid forward returns a list compatible with AddFusion's seam.
forward_features returns (fused, ct_enhanced, pet_enhanced) lists.
Feature masks cannot stop an external PET encoder: caller must route image
samples BEFORE PET encoding (as the clean baseline already does).

Local attention is exact 5x5 (including center), with learned displacement
bias, separate directions and masked out-of-image candidates. Row chunking
and optional activation checkpointing change storage, not the operation.
Global attention is full d-by-d channel attention over ALL spatial locations,
not full spatial attention, pooling, sparse selection, or Fourier attention.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ['LocalContrastBidirectionalFusion', 'LocalContrastFusionPyramid']


def _groups(channels: int) -> int:
    return math.gcd(channels, 8)


def _pair(ct: Tensor, pet: Tensor | None, channels: int) -> None:
    if not isinstance(ct, Tensor) or ct.ndim != 4:
        raise ValueError('CT must be a BCHW tensor')
    if ct.shape[1] != channels or min(ct.shape[0], *ct.shape[2:]) < 1:
        raise ValueError(f'Expected nonempty CT with {channels} channels')
    if not ct.is_floating_point():
        raise TypeError('Features must be floating point')
    if pet is not None:
        if not isinstance(pet, Tensor) or pet.shape != ct.shape:
            raise ValueError('Full CT/PET must have identical BCHW shapes')
        if pet.device != ct.device or pet.dtype != ct.dtype:
            raise ValueError('CT/PET must have identical devices and dtypes')


def _fp32_context(x: Tensor):
    # Explicit FP32 logits and softmax even inside mixed-precision training.
    return torch.autocast(device_type=x.device.type, enabled=False) if x.device.type in ('cpu', 'cuda') else nullcontext()


class _ContrastDescriptor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.near = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.far = nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False)
        self.norm = nn.GroupNorm(_groups(dim), dim)
        self.mix = nn.Conv2d(dim, dim, 1)
        with torch.no_grad():
            kernel = torch.full((3, 3), -1.0 / 8.0)
            kernel[1, 1] = 1.0
            for conv in (self.near, self.far):
                conv.weight.copy_(kernel.view(1, 1, 3, 3).expand_as(conv.weight))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mix(F.gelu(self.norm(self.near(x) + self.far(x))))


class _LocalExchange(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, chunk_rows: int,
                 checkpoint_chunks: bool):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.window, self.radius = window, window // 2
        self.chunk_rows, self.checkpoint_chunks = chunk_rows, checkpoint_chunks
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.relative_bias = nn.Parameter(torch.zeros(heads, window * window))

    def _chunk(self, query: Tensor, key: Tensor, value: Tensor,
               valid: Tensor, bias: Tensor) -> Tensor:
        b, d, rows, w = query.shape
        n, dh, k = rows * w, d // self.heads, self.window ** 2
        q = query.reshape(b, self.heads, dh, n)
        # Stripe already has the full halo; no additional unfold padding.
        keys = F.unfold(key, self.window).reshape(b, self.heads, dh, k, n)
        values = F.unfold(value, self.window).reshape(b, self.heads, dh, k, n)
        allowed = F.unfold(valid, self.window).reshape(1, 1, k, n).bool()
        with _fp32_context(query):
            scores = torch.einsum('bhdn,bhdkn->bhkn', q.float(), keys.float()) / math.sqrt(dh)
            scores = scores + bias.float()[None, :, :, None]
            scores = scores.masked_fill(~allowed, -torch.inf)
            attn = scores.softmax(dim=2)
            out = torch.einsum('bhkn,bhdkn->bhdn', attn, values.float())
        return out.reshape(b, d, rows, w).to(query.dtype)

    def forward(self, query_descriptor: Tensor, source_descriptor: Tensor,
                source_content: Tensor) -> Tensor:
        q = self.query(query_descriptor)
        k, v = self.key(source_descriptor), self.value(source_content)
        b, _, h, w = q.shape
        r = self.radius
        k, v = F.pad(k, (r, r, r, r)), F.pad(v, (r, r, r, r))
        valid = F.pad(q.new_ones(1, 1, h, w), (r, r, r, r))
        chunks = []
        for start in range(0, h, self.chunk_rows):
            end = min(start + self.chunk_rows, h)
            args = (q[:, :, start:end], k[:, :, start:end + 2*r],
                    v[:, :, start:end + 2*r], valid[:, :, start:end + 2*r],
                    self.relative_bias)
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                out = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                out = self._chunk(*args)
            chunks.append(out)
        return torch.cat(chunks, dim=2)


class _GlobalChannelExchange(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        # Positive temperature, initial tau=1. Clamp protects numerical scale.
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, target: Tensor, source: Tensor) -> Tensor:
        q, k, v = self.query(target), self.key(source), self.value(source)
        b, d, h, w = q.shape
        with _fp32_context(q):
            qf = F.normalize(q.float().flatten(2), dim=-1, eps=1e-6)
            kf = F.normalize(k.float().flatten(2), dim=-1, eps=1e-6)
            tau = self.log_temperature.float().clamp(-6., 6.).exp()
            weights = (torch.bmm(qf, kf.transpose(1, 2)) / tau).softmax(dim=-1)
            message = torch.bmm(weights, v.float().flatten(2))
        return message.reshape(b, d, h, w).to(v.dtype)


class _Update(nn.Module):
    def __init__(self, dim: int, channels: int):
        super().__init__()
        self.local = nn.Conv2d(dim, dim, 1)
        self.global_ = nn.Conv2d(dim, dim, 1)
        self.refine = nn.Sequential(nn.GroupNorm(_groups(dim), dim), nn.GELU(),
                                    nn.Conv2d(dim, dim, 3, padding=1, groups=dim))
        self.out = nn.Conv2d(dim, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, local: Tensor, global_: Tensor) -> Tensor:
        return self.out(self.refine(self.local(local) + self.global_(global_)))


class LocalContrastBidirectionalFusion(nn.Module):
    """One scale. forward returns (fused, ct_enhanced, pet_enhanced).

    Missing returns (ct, ct, zeros_like(ct)); the zero PET output is only an
    absent-feature placeholder, never processed by attention. Full requires PET.
    """
    def __init__(self, channels: int, dim: int = 32, heads: int = 4,
                 window: int = 5, chunk_rows: int = 16,
                 checkpoint_chunks: bool = True):
        super().__init__()
        if channels < 1 or dim < 8 or heads < 1 or dim % heads:
            raise ValueError('channels>0, dim>=8 and dim divisible by heads required')
        if window < 1 or window % 2 != 1 or chunk_rows < 1:
            raise ValueError('window must be positive odd; chunk_rows must be positive')
        self.channels = channels
        self.ct_project = nn.Sequential(nn.GroupNorm(_groups(channels), channels), nn.Conv2d(channels, dim, 1))
        self.pet_project = nn.Sequential(nn.GroupNorm(_groups(channels), channels), nn.Conv2d(channels, dim, 1))
        self.ct_contrast, self.pet_contrast = _ContrastDescriptor(dim), _ContrastDescriptor(dim)
        args = (dim, heads, window, chunk_rows, checkpoint_chunks)
        self.ct_local, self.pet_local = _LocalExchange(*args), _LocalExchange(*args)
        self.ct_local_state, self.pet_local_state = nn.Conv2d(dim, dim, 1), nn.Conv2d(dim, dim, 1)
        self.ct_global, self.pet_global = _GlobalChannelExchange(dim), _GlobalChannelExchange(dim)
        self.ct_update, self.pet_update = _Update(dim, channels), _Update(dim, channels)

    def forward(self, ct: Tensor, pet: Tensor | None = None,
                mode: str = 'full') -> tuple[Tensor, Tensor, Tensor]:
        if mode not in ('full', 'missing'):
            raise ValueError("mode must be 'full' or 'missing'")
        _pair(ct, pet if mode == 'full' else None, self.channels)
        if mode == 'missing':
            return ct, ct, torch.zeros_like(ct)
        if pet is None:
            raise ValueError('Full fusion requires real PET features')
        xc, xp = self.ct_project(ct), self.pet_project(pet)
        ec, ep = self.ct_contrast(xc), self.pet_contrast(xp)
        lc = self.ct_local(ec, ep, xp)
        lp = self.pet_local(ep, ec, xc)
        # Simultaneous updates: neither direction uses the other's updated state.
        zc, zp = xc + self.ct_local_state(lc), xp + self.pet_local_state(lp)
        gc, gp = self.ct_global(zc, zp), self.pet_global(zp, zc)
        dc = self.ct_update(lc, gc).to(ct.dtype)
        dp = self.pet_update(lp, gp).to(pet.dtype)
        ce, pe = ct + dc, pet + dp
        return ce + pe, ce, pe


class LocalContrastFusionPyramid(nn.Module):
    """Four-scale AddFusion-compatible wrapper (fine-to-coarse order).

    Default: fused_list = module(ct_list, pet_list)
    Detailed: fused, ct_plus, pet_plus = module.forward_features(ct_list, pet_list)
    Optional auto mask: bool/0-1 integer vector (B,), True means PET available.
    In auto, PET lists must contain B rows, and only Full rows are selected.
    All-Missing permits pet=None and never executes scale fusion blocks.
    """
    def __init__(self, channels: Sequence[int] = (64, 128, 320, 512),
                 dim: int = 32, heads: int = 4, window: int = 5,
                 chunk_rows: int = 16, checkpoint_chunks: bool = True):
        super().__init__()
        self.channels = tuple(channels)
        if len(self.channels) != 4:
            raise ValueError('Exactly four scales required')
        self.scales = nn.ModuleList([
            LocalContrastBidirectionalFusion(c, dim, heads, window, chunk_rows, checkpoint_chunks)
            for c in self.channels])

    def _validate_ct(self, ct: Sequence[Tensor]) -> None:
        if not isinstance(ct, (list, tuple)) or len(ct) != 4:
            raise ValueError('CT must be a list/tuple of four BCHW feature tensors')
        for c, channels in zip(ct, self.channels):
            _pair(c, None, channels)
        for i in range(1, 4):
            if ct[i].shape[0] != ct[0].shape[0] or ct[i].device != ct[0].device or ct[i].dtype != ct[0].dtype:
                raise ValueError('CT scales must share batch/device/dtype')
            if any(a > b for a, b in zip(ct[i].shape[2:], ct[i-1].shape[2:])):
                raise ValueError('Expected fine-to-coarse scale ordering')

    def forward_features(self, ct: Sequence[Tensor], pet: Sequence[Tensor] | None = None,
                         mode: str = 'full', pet_available: Tensor | Sequence[int] | None = None
                         ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        self._validate_ct(ct)
        if mode not in ('full', 'missing', 'auto'):
            raise ValueError("mode must be 'full', 'missing' or 'auto'")
        b = ct[0].shape[0]
        if mode == 'missing':
            return list(ct), list(ct), [torch.zeros_like(c) for c in ct]
        if mode == 'auto':
            if pet_available is None:
                raise ValueError('auto requires explicit pet_available')
            mask = torch.as_tensor(pet_available, device=ct[0].device)
            ints = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            if mask.ndim != 1 or mask.shape[0] != b or (mask.dtype != torch.bool and mask.dtype not in ints):
                raise ValueError('pet_available must be a length-B bool or integer vector')
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError('pet_available values must be 0 or 1')
            idx = mask.bool().nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                return list(ct), list(ct), [torch.zeros_like(c) for c in ct]
        else:
            idx = None
        if not isinstance(pet, (list, tuple)) or len(pet) != 4:
            raise ValueError('Full rows require four PET feature tensors')
        if any(p.dtype != c.dtype for p, c in zip(pet, ct)):
            # AMP autocast can leave CT (BatchNorm tail, fp32) and PET (fp16)
            # in different dtypes. Align PET to the CT dtype once here; this
            # changes representation only, not the fusion computation (all
            # attention math already runs in fp32 internally).
            pet = [p.to(dtype=c.dtype) if p.dtype != c.dtype else p
                   for p, c in zip(pet, ct)]
        outputs = [[], [], []]
        for block, c, p, channels in zip(self.scales, ct, pet, self.channels):
            _pair(c, p, channels)
            if idx is None:
                result = block(c, p)
            else:
                ff, cc, pp = block(c.index_select(0, idx), p.index_select(0, idx))
                result = (c.index_copy(0, idx, ff), c.index_copy(0, idx, cc),
                          torch.zeros_like(c).index_copy(0, idx, pp))
            for output, feature in zip(outputs, result):
                output.append(feature)
        return outputs[0], outputs[1], outputs[2]

    def forward(self, ct: Sequence[Tensor], pet: Sequence[Tensor] | None = None,
                mode: str = 'full', pet_available: Tensor | Sequence[int] | None = None
                ) -> list[Tensor]:
        return self.forward_features(ct, pet, mode, pet_available)[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Four-scale forward smoke test; no training data required')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--full-size', action='store_true', help='Use 128/64/32/16 feature sizes')
    parser.add_argument('--backward', action='store_true')
    opts = parser.parse_args()
    torch.manual_seed(2026)
    model = LocalContrastFusionPyramid().to(opts.device)
    sizes = (128, 64, 32, 16) if opts.full_size else (16, 8, 4, 2)
    ct = [torch.randn(opts.batch_size, c, n, n, device=opts.device, requires_grad=opts.backward)
          for c, n in zip(model.channels, sizes)]
    pet = [torch.randn_like(c) for c in ct]
    with torch.set_grad_enabled(opts.backward):
        fused = model(ct, pet)
        assert all(torch.equal(f, c+p) for f, c, p in zip(fused, ct, pet))
        if opts.backward:
            sum(f.square().mean() for f in fused).backward()
    print('PASS: four-scale shapes, finiteness and exact step-0 addition')
    print('parameters:', sum(p.numel() for p in model.parameters()))
    print('output shapes:', [tuple(f.shape) for f in fused])
