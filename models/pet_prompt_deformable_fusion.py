"""PET Gaussian prompt -> CT deformable reading -> prompt correction -> PET reading.

Standalone PyTorch >= 2.1 implementation of the approved PET-CT adaptation.
Fixed Gaussian filters follow RAC-Net Eq. 6-9 (sigma=1,1.5,2; dilation=1,1,2).
Bounded offsets/bilinear sampling follow the mechanism of Stable-SAM Sec. 4.1;
its inspected public repository has no implementation, so this is NOT copied
author code. Prompt correction and the multimodal organization are our proposal.
No HU inversion, hard ROI, central differences, global attention or extra loss.
Four scales use identical architecture and settings, independently parameterized.
Raw C/P appear once: (C+delta_C)+(P+delta_P). Step zero equals C+P exactly.
Full only is the current experiment. Missing bypass is CT, NOT PET completion.
Diagnostics are explicit detached outputs; no persistent graph/capture buffer.
"""
from __future__ import annotations

import argparse
import math
import struct
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ['FixedGaussianPrompt', 'PETPromptDeformableFusion',
           'MultiScalePETPromptDeformableFusion', 'export_soft_prompts']


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(c, 8), c)


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _state(value, b, device):
    s = torch.as_tensor(value, device=device)
    if s.ndim != 1 or s.numel() != b:
        raise ValueError('pet_available must be a length-B vector')
    if s.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16,
                        torch.int32, torch.int64):
        raise ValueError('pet_available must be bool or 0/1 integers')
    if not bool(((s == 0) | (s == 1)).all()):
        raise ValueError('pet_available values must be 0 or 1')
    return s.bool()


class FixedGaussianPrompt(nn.Module):
    """Full-channel identity + three fixed depthwise Gaussian responses + merge."""
    def __init__(self, channels=32):
        super().__init__()
        self.channels = _positive('channels', channels)
        y, x = torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), indexing='ij')
        kernels = []
        for sigma in (1., 1.5, 2.):
            g = torch.exp(-(x.square()+y.square()).float()/(2*sigma*sigma))
            kernels.append((g/g.sum())[None, None].repeat(channels, 1, 1, 1))
        self.register_buffer('gaussian_kernels', torch.stack(kernels))
        self.merge = nn.Sequential(nn.Conv2d(3*channels, channels, 1), _gn(channels),
                                   nn.GELU(), nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        responses = [F.conv2d(F.pad(x, (d, d, d, d), mode='replicate'),
                              self.gaussian_kernels[i].to(dtype=x.dtype),
                              dilation=d, groups=self.channels)
                     for i, d in enumerate((1, 1, 2))]
        return x + self.merge(torch.cat(responses, dim=1))


class PETPromptDeformableFusion(nn.Module):
    """One scale. forward -> (fused, ct_enhanced, pet_enhanced).

    CT: heads x 9 bounded movable points, bilinear interpolation, weighted sum.
    PET: heads x 9 fixed points, corrected candidate logits + relative bias,
    explicitly masked at image edges. Values never pass through Gaussian filters.
    Both row chunking and optional recomputation compute every point, exactly.
    Flags are structure ablations; defaults execute the complete approved design.
    """
    def __init__(self, channels: int, inner_channels: int = 32, heads: int = 4,
                 offset_radius: float = 2., chunk_rows: int = 16,
                 checkpoint_chunks: bool = True, use_gaussian: bool = True,
                 use_pet_prompt: bool = True, use_deformable: bool = True,
                 use_ct_correction: bool = True, check_finite: bool = True):
        super().__init__()
        self.channels = _positive('channels', channels)
        self.inner_channels = _positive('inner_channels', inner_channels)
        self.heads = _positive('heads', heads)
        self.chunk_rows = _positive('chunk_rows', chunk_rows)
        if inner_channels % heads:
            raise ValueError('inner_channels must be divisible by heads')
        if not math.isfinite(offset_radius) or offset_radius <= 0:
            raise ValueError('offset_radius must be finite and >0')
        for name, flag in [('checkpoint_chunks', checkpoint_chunks), ('use_gaussian', use_gaussian),
                           ('use_pet_prompt', use_pet_prompt), ('use_deformable', use_deformable),
                           ('use_ct_correction', use_ct_correction), ('check_finite', check_finite)]:
            if not isinstance(flag, bool):
                raise TypeError(f'{name} must be bool')
            setattr(self, name, flag)
        self.offset_radius = float(offset_radius)
        self.head_dim = inner_channels//heads
        self.ct_project = nn.Sequential(_gn(channels), nn.Conv2d(channels, inner_channels, 1))
        self.pet_project = nn.Sequential(_gn(channels), nn.Conv2d(channels, inner_channels, 1))
        self.pet_descriptor = FixedGaussianPrompt(inner_channels)
        self.pet_location = nn.Conv2d(inner_channels, 1, 1)
        self.prompt_embed = nn.Conv2d(1, inner_channels, 1)
        self.sampling_stem = nn.Sequential(nn.Conv2d(inner_channels, inner_channels, 1),
            nn.Conv2d(inner_channels, inner_channels, 5, padding=2,
                      padding_mode='replicate', groups=inner_channels),
            _gn(inner_channels), nn.GELU())
        self.offset_head = nn.Conv2d(inner_channels, heads*9*2, 1)
        self.weight_head = nn.Conv2d(inner_channels, heads*9, 1)
        self.ct_value = nn.Conv2d(inner_channels, inner_channels, 1)
        self.pet_value = nn.Conv2d(inner_channels, inner_channels, 1)
        self.correction = nn.Sequential(nn.Conv2d(2*inner_channels, inner_channels, 1),
            _gn(inner_channels), nn.GELU(),
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1,
                      padding_mode='replicate', groups=inner_channels),
            _gn(inner_channels), nn.GELU(), nn.Conv2d(inner_channels, 1, 1))
        self.pet_relative_bias = nn.Parameter(torch.zeros(heads, 9))
        self.ct_out = nn.Conv2d(inner_channels, channels, 1)
        self.pet_out = nn.Conv2d(inner_channels, channels, 1)
        yy, xx = torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), indexing='ij')
        self.register_buffer('base_offsets_xy', torch.stack((xx.flatten(), yy.flatten()), -1))
        self.register_buffer('contract_signature', torch.tensor([
            1, channels, inner_channels, heads, int(use_gaussian), int(use_pet_prompt),
            int(use_deformable), int(use_ct_correction)], dtype=torch.int64))
        # Integer bits are invariant under EMA and module half/float conversions.
        radius_bits = struct.unpack('q', struct.pack('d', self.offset_radius))[0]
        self.register_buffer('recorded_radius_bits', torch.tensor(radius_bits, dtype=torch.int64))
        for layer in (self.offset_head, self.correction[-1], self.ct_out, self.pet_out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        sig = state_dict.get(prefix+'contract_signature')
        radius = state_dict.get(prefix+'recorded_radius_bits')
        if sig is not None and not torch.equal(sig.cpu(), self.contract_signature.cpu()):
            raise RuntimeError('Fusion contract differs; rebuild using checkpoint config')
        if radius is not None and not torch.equal(radius.cpu(), self.recorded_radius_bits.cpu()):
            raise RuntimeError('offset_radius differs from checkpoint config')
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _validate(self, x, name):
        if not isinstance(x, Tensor) or x.ndim != 4 or not x.is_floating_point():
            raise ValueError(f'{name} must be a floating BCHW Tensor')
        if x.shape[1] != self.channels or min(x.shape[0], *x.shape[2:]) < 1:
            raise ValueError(f'{name}: wrong channels or empty input')
        if self.check_finite and not bool(torch.isfinite(x).all()):
            raise ValueError(f'{name} contains NaN/Inf')

    def _ct_chunk(self, value, offsets, weights, start, end):
        b, _, h, w = value.shape
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=value.device.type, enabled=False):
            v = value.to(dtype).reshape(b*self.heads, self.head_dim, h, w)
            off = offsets[..., start:end, :].to(dtype).permute(0, 1, 4, 5, 2, 3)
            y, x = torch.meshgrid(torch.arange(start, end, device=value.device, dtype=dtype),
                                 torch.arange(w, device=value.device, dtype=dtype), indexing='ij')
            base = torch.stack((x, y), -1)[None, None, :, :, None]
            points = base + self.base_offsets_xy.to(dtype)[None, None, None, None] + off
            # Border replication for CT samples, align_corners=False pixel-center convention.
            px = points[..., 0].clamp(0, w-1)
            py = points[..., 1].clamp(0, h-1)
            grid = torch.stack((2*(px+.5)/w-1, 2*(py+.5)/h-1), -1)
            grid = grid.reshape(b*self.heads, end-start, w*9, 2)
            sampled = F.grid_sample(v, grid, mode='bilinear', padding_mode='border',
                                    align_corners=False)
            sampled = sampled.reshape(b, self.heads, self.head_dim, end-start, w, 9)
            alpha = weights[..., start:end, :].to(dtype).permute(0, 1, 3, 4, 2)
            result = (sampled*alpha.unsqueeze(2)).sum(-1)
            return result.reshape(b, self.inner_channels, end-start, w).to(value.dtype)

    def _pet_chunk(self, value, logits, start, end):
        b, _, h, w = value.shape
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=value.device.type, enabled=False):
            def unfold(x):
                stripe = F.pad(x.to(dtype), (1,1,1,1))[..., start:end+2, :]
                return F.unfold(stripe, 3).reshape(b, x.shape[1], 9, end-start, w)
            candidates = unfold(logits)[:, 0]
            dx, dy = self.base_offsets_xy[:, 0], self.base_offsets_xy[:, 1]
            y = torch.arange(start, end, device=value.device)[None, :, None]+dy[:, None, None]
            x = torch.arange(w, device=value.device)[None, None, :]+dx[:, None, None]
            valid = (y >= 0) & (y < h) & (x >= 0) & (x < w)
            scores = candidates[:, None] + self.pet_relative_bias.to(dtype)[None, :, :, None, None]
            scores = scores.masked_fill(~valid[None, None], -torch.inf)
            alpha = scores.softmax(dim=2)
            neighbors = unfold(value).reshape(b, self.heads, self.head_dim, 9, end-start, w)
            return (neighbors*alpha.unsqueeze(2)).sum(3).reshape(
                b, self.inner_channels, end-start, w).to(value.dtype)

    def _chunks(self, fn, *xs):
        h = xs[0].shape[-2]
        parts = []
        for start in range(0, h, self.chunk_rows):
            call = partial(fn, start=start, end=min(h, start+self.chunk_rows))
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                parts.append(checkpoint(call, *xs, use_reentrant=False))
            else:
                parts.append(call(*xs))
        return torch.cat(parts, -2)

    def _full(self, ct, pet, diagnostics=False):
        self._validate(ct, 'ct'); self._validate(pet, 'pet')
        if ct.shape != pet.shape or ct.device != pet.device:
            raise ValueError('CT/PET must have the same shape and device; no implicit resize')
        xc, xp = self.ct_project(ct), self.pet_project(pet)
        descriptor = self.pet_descriptor(xp) if self.use_gaussian else xp
        a0 = self.pet_location(descriptor)
        g0 = a0.sigmoid()
        q = xc + self.prompt_embed(g0) if self.use_pet_prompt else xc
        stem = self.sampling_stem(q)
        b, _, h, w = xc.shape
        offsets = self.offset_radius*self.offset_head(stem).tanh()
        offsets = offsets.reshape(b, self.heads, 9, 2, h, w)
        if not self.use_deformable:
            offsets = offsets*0
        weight_logits = self.weight_head(stem).reshape(b, self.heads, 9, h, w)
        dtype = torch.float64 if xc.dtype == torch.float64 else torch.float32
        weights = weight_logits.to(dtype).softmax(dim=2)
        mc = self._chunks(self._ct_chunk, self.ct_value(xc), offsets, weights)
        correction = self.correction(torch.cat((xc, mc), 1)) if self.use_ct_correction else torch.zeros_like(a0)
        a1 = a0 + correction
        mp = self._chunks(self._pet_chunk, self.pet_value(xp), a1)
        dc, dp = self.ct_out(mc).to(ct.dtype), self.pet_out(mp).to(pet.dtype)
        ce, pe = ct + dc, pet + dp
        fused = ce + pe
        if self.check_finite and not bool(torch.isfinite(fused).all()):
            raise RuntimeError('Fusion output contains NaN/Inf')
        result = (fused, ce, pe)
        if not diagnostics:
            return result
        maps = {'coarse_logits': a0.detach(), 'coarse_prompt': g0.detach(),
                'corrected_logits': a1.detach(), 'corrected_prompt': a1.sigmoid().detach(),
                'ct_logit_correction': correction.detach(),
                'prompt_change': (a1.sigmoid()-g0).detach()}
        return result, maps

    def forward(self, ct, pet=None, forward_mode='full', pet_available=None):
        self._validate(ct, 'ct')
        if forward_mode == 'missing':
            return ct, ct, torch.zeros_like(ct)
        if forward_mode == 'full':
            return self._full(ct, pet)
        if forward_mode != 'auto' or pet_available is None:
            raise ValueError('Use full/missing, or auto with pet_available')
        state = _state(pet_available, ct.shape[0], ct.device)
        idx = state.nonzero(as_tuple=True)[0]
        if not idx.numel():
            return ct, ct, torch.zeros_like(ct)
        if not isinstance(pet, Tensor) or pet.shape != ct.shape or pet.device != ct.device:
            raise ValueError('auto with Full rows requires a matching PET tensor')
        full = self._full(ct.index_select(0, idx), pet.index_select(0, idx))
        bases = (ct, ct, torch.zeros_like(ct))
        return tuple(base.index_copy(0, idx, item.to(base.dtype)) for base, item in zip(bases, full))

    def forward_with_diagnostics(self, ct, pet):
        """Explicit Full-only diagnostics, detached; regular training should call forward."""
        return self._full(ct, pet, diagnostics=True)


class MultiScalePETPromptDeformableFusion(nn.Module):
    def __init__(self, channels: Sequence[int] = (64,128,320,512), **kwargs):
        super().__init__()
        self.channels = tuple(channels)
        if len(self.channels) != 4:
            raise ValueError('Exactly four fine-to-coarse scales are required')
        self.scales = nn.ModuleList([PETPromptDeformableFusion(c, **kwargs) for c in self.channels])

    def _check(self, ct, pet, require_pet):
        if not isinstance(ct, (list, tuple)) or len(ct) != 4:
            raise ValueError('ct_feats must contain exactly four tensors')
        if require_pet and (not isinstance(pet, (list, tuple)) or len(pet) != 4):
            raise ValueError('pet_feats must contain exactly four tensors')
        for x in ct:
            if not isinstance(x, Tensor) or x.ndim != 4 or x.shape[0] != ct[0].shape[0] or x.device != ct[0].device:
                raise ValueError('CT scales must be BCHW with matching batch/device')

    def forward_with_features(self, ct_feats, pet_feats=None, forward_mode='full', pet_available=None):
        self._check(ct_feats, None, False)
        need_pet = forward_mode == 'full'
        if forward_mode == 'auto':
            need_pet = bool(_state(pet_available, ct_feats[0].shape[0], ct_feats[0].device).any())
        self._check(ct_feats, pet_feats, need_pet)
        results = [layer(c, pet_feats[i] if need_pet else None, forward_mode, pet_available)
                   for i, (layer, c) in enumerate(zip(self.scales, ct_feats))]
        return tuple([r[i] for r in results] for i in range(3))

    def forward(self, ct_feats, pet_feats=None, forward_mode='full', pet_available=None):
        return self.forward_with_features(ct_feats, pet_feats, forward_mode, pet_available)[0]

    def forward_with_diagnostics(self, ct_feats, pet_feats):
        self._check(ct_feats, pet_feats, True)
        outputs, maps = [], []
        for layer, c, p in zip(self.scales, ct_feats, pet_feats):
            triple, diag = layer.forward_with_diagnostics(c, p)
            outputs.append(triple[0]); maps.append(diag)
        return outputs, maps


def export_soft_prompts(diagnostics, output_dir, prefix='sample', sample_index=0,
                        ct_image=None, pet_image=None, mask=None, scale_indices=None):
    """Save exact maps in NPZ and fixed-range PNG panels. NumPy/Pillow only on export.
    Images are display-normalized; prompts ALWAYS use [0,1], changes [-1,1].
    Optional mask is display-only and NEVER used to compute prompts/predictions.
    Caller should obtain diagnostics under eval + no_grad from raw or EMA model.
    scale_indices optionally labels each entry with its true scale number
    (e.g. exporting a single scale [2]); default keeps the legacy 1..N labels.
    """
    import json
    import numpy as np
    from PIL import Image, ImageDraw
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    if not prefix or Path(prefix).name != prefix:
        raise ValueError('prefix must be a plain filename component')
    diagnostics = [diagnostics] if isinstance(diagnostics, dict) else diagnostics
    if scale_indices is None:
        scale_indices = list(range(1, len(diagnostics) + 1))
    if len(scale_indices) != len(diagnostics):
        raise ValueError('scale_indices must match the number of diagnostics entries')
    saved = []
    for scale, maps in zip(scale_indices, diagnostics):
        data = {k: v[sample_index].detach().float().cpu().numpy() for k, v in maps.items()}
        npz = directory/f'{prefix}_s{scale}.npz'
        np.savez_compressed(npz, **data)
        panels = []
        for label, value in [('CT (display)', ct_image), ('PET (display)', pet_image), ('GT (display only)', mask)]:
            if value is None: continue
            a = value.detach().float().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
            if a.ndim == 4: a = a[sample_index]
            if a.ndim == 3: a = a[0]
            if a.ndim != 2 or not np.isfinite(a).all(): raise ValueError('Display image must be finite HW/CHW/BCHW')
            a = (a-a.min())/max(float(a.max()-a.min()), 1e-8)
            rgb = np.repeat((a[..., None]*255).astype('uint8'), 3, -1)
            panels.append((label, Image.fromarray(rgb)))
        for key, label in [('coarse_prompt','PET prompt [0,1]'), ('corrected_prompt','CT-corrected [0,1]'),
                           ('prompt_change','Change [-1,1]')]:
            a = data[key][0]
            if key == 'prompt_change':
                a = np.clip(a, -1, 1)
                rgb = np.stack((np.maximum(a,0), np.zeros_like(a), np.maximum(-a,0)), -1)
            else:
                a = np.clip(a,0,1)
                rgb = np.stack((a, a*.6, 1-a), -1)
            panels.append((label, Image.fromarray((rgb*255).astype('uint8'))))
        canvas = Image.new('RGB', (256*len(panels), 284), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (label, picture) in enumerate(panels):
            draw.text((256*i+5,5), label, fill='black')
            canvas.paste(picture.resize((256,256), resample=Image.Resampling.BILINEAR), (256*i,28))
        png = directory/f'{prefix}_s{scale}.png'; canvas.save(png)
        meta = {'scale':scale, 'prompt_range':[0,1], 'change_range':[-1,1],
                'note':'Internal learned prompts, not calibrated lesion or boundary probabilities.',
                'statistics':{k:{'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean())} for k,v in data.items()}}
        (directory/f'{prefix}_s{scale}.json').write_text(json.dumps(meta, indent=2))
        saved.extend([str(png), str(npz)])
    return saved


def _smoke(device, visualize_dir):
    torch.manual_seed(2023); torch.set_num_threads(2)
    model = MultiScalePETPromptDeformableFusion().to(device).eval()
    shapes = [(1,c,s,s) for c,s in zip((64,128,320,512),(128,64,32,16))]
    ct, pet = [[torch.randn(shape, device=device) for shape in shapes] for _ in range(2)]
    with torch.no_grad():
        out, maps = model.forward_with_diagnostics(ct, pet)
        assert all(torch.equal(f,c+p) for f,c,p in zip(out,ct,pet))
    if visualize_dir:
        export_soft_prompts(maps, visualize_dir, prefix='untrained_synthetic', ct_image=ct[0], pet_image=pet[0])
    layer = PETPromptDeformableFusion(8, inner_channels=8, heads=2).to(device).train()
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    c,p = [torch.randn(2,8,9,11,device=device) for _ in range(2)]
    for _ in range(2):
        opt.zero_grad(); f,_,_ = layer(c,p); (f.square().mean()).backward()
        assert all(t.grad is None or torch.isfinite(t.grad).all() for t in layer.parameters())
        opt.step()
    print({'shapes':[list(x.shape) for x in out], 'params':sum(p.numel() for p in model.parameters()),
           'step_zero':'exact C+P', 'two_steps':'finite', 'device':str(device)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--visualize_dir', default=None)
    args = parser.parse_args()
    if not args.smoke: parser.error('Use --smoke or import the module in the repository')
    _smoke(args.device, args.visualize_dir)
