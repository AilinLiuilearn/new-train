"""Spatial bidirectional PET-CT fusion; standalone, PyTorch-only implementation.

Place this file at models/spatial_bidirectional_fusion.py.
Inputs are *already channel-aligned* NCHW encoder features, not images.
Scale order: shallow -> deep, default channels (64, 128, 320, 512).

Full: local reciprocal exchange -> horizontal exchange -> vertical exchange
      -> sum of the two residual streams. At initialization exactly CT + PET.
Missing: CT passthrough; no normalization, projection, or attention on that row.
No encoders, decoder, memory bank, extra loss, text, or PET hallucination.

This is a new adaptation, NOT a reproduction of the referenced papers:
GeminiFusion (ICML 2024): fine-grained multimodal exchange / own-stream retention.
  https://proceedings.mlr.press/v235/jia24b.html
Keep the Balance (CVPR 2025): balanced treatment of the two modalities.
  https://github.com/imcjx/KTB (DAttentionMM inspected, not copied).
DFormerv2 (CVPR 2025): spatially structured, axial attention computation.
  https://github.com/VCIP-RGBD/DFormer (Decomposed_GSA inspected, not copied).
Shared row/column normalization is the adopted design, not attributed to KTB's
DSCF. No concatenated relation predictor, deformation, depth prior or hard Top-K.

Requires Python >= 3.9 and PyTorch >= 2.1. Training checkpoints are stateless
and contain no memory-bank data. Float32 score/softmax under AMP for stability.
Run: python spatial_bidirectional_fusion.py --smoke
     python spatial_bidirectional_fusion.py --smoke --native-sizes --device cuda
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

State = Optional[Union[bool, Tensor, Sequence[int]]]
Streams = Tuple[Tensor, Tensor]


def _fp32_context(x: Tensor):
    return torch.autocast(device_type=x.device.type, enabled=False) if x.device.type in ("cpu", "cuda") else nullcontext()


def _state(state: State, x: Tensor) -> Tensor:
    """True=real PET present. Reject float states and non-binary integers."""
    if state is None:
        return torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    if isinstance(state, bool):
        return torch.full((x.shape[0],), state, device=x.device, dtype=torch.bool)
    raw = torch.as_tensor(state, device=x.device)
    if raw.ndim != 1 or raw.shape[0] != x.shape[0]:
        raise ValueError("pet_available must be a length-B vector or a bool")
    if raw.dtype == torch.bool:
        return raw
    if raw.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("pet_available must contain bools or 0/1 integers")
    if not bool(((raw == 0) | (raw == 1)).all()):
        raise ValueError("pet_available integer values must be 0 or 1")
    return raw.bool()


def _check_ct(c: Tensor, channels: int) -> None:
    if not isinstance(c, Tensor) or c.ndim != 4:
        raise ValueError("CT features must be a NCHW tensor")
    if not c.is_floating_point() or c.shape[1] != channels or min(c.shape) < 1:
        raise ValueError(f"Expected floating nonempty CT features with {channels} channels")


def _check_pair(c: Tensor, p: Tensor) -> None:
    if not isinstance(p, Tensor) or p.ndim != 4 or not p.is_floating_point():
        raise ValueError("PET features must be a floating NCHW tensor")
    if c.shape != p.shape:
        raise ValueError(f"CT/PET shape mismatch: {tuple(c.shape)} vs {tuple(p.shape)}")
    if c.device != p.device or c.dtype != p.dtype:
        raise ValueError("CT/PET must have the same device and dtype")


def _neighbor(x: Tensor, dy: int, dx: int, fill: float = 0.0) -> Tensor:
    """At (y,x), return input (y+dy,x+dx); never wrap around an image edge."""
    h, w = x.shape[-2:]
    y = F.pad(x, (max(-dx, 0), max(dx, 0), max(-dy, 0), max(dy, 0)), value=fill)
    return y[..., max(dy, 0):max(dy, 0) + h, max(dx, 0):max(dx, 0) + w]


class _Projections(nn.Module):
    """Per-pixel channel LayerNorm: independent of other batch samples."""
    def __init__(self, channels: int, dim: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.norm_c = nn.LayerNorm(channels)
        self.norm_p = nn.LayerNorm(channels)
        self.embed_c = nn.Conv2d(channels, dim, 1, bias=False)
        self.embed_p = nn.Conv2d(channels, dim, 1, bias=False)
        self.value_c = nn.Conv2d(channels, dim, 1, bias=False)
        self.value_p = nn.Conv2d(channels, dim, 1, bias=False)
        self.out_c = nn.Conv2d(dim, channels, 1, bias=False)
        self.out_p = nn.Conv2d(dim, channels, 1, bias=False)
        nn.init.zeros_(self.out_c.weight)
        nn.init.zeros_(self.out_p.weight)

    def project(self, c: Tensor, p: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        c = self.norm_c(c.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        p = self.norm_p(p.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        b, _, h, w = c.shape
        return tuple(layer(x).reshape(b, self.heads, self.head_dim, h, w)
                     for layer, x in ((self.embed_c, c), (self.embed_p, p),
                                      (self.value_c, c), (self.value_p, p)))

    def update(self, c: Tensor, p: Tensor, mc: Tensor, mp: Tensor) -> Streams:
        # Explicit cast avoids promoting the original encoder feature dtype
        # when attention was evaluated in float32 under mixed precision.
        dc = self.out_c(mc.flatten(1, 2)).to(c.dtype)
        dp = self.out_p(mp.flatten(1, 2)).to(p.dtype)
        return c + dc, p + dp


class _LocalExchange(_Projections):
    def __init__(self, channels: int, dim: int, heads: int, kernel_size: int):
        super().__init__(channels, dim, heads)
        r = kernel_size // 2
        self.offsets = tuple((dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1))
        self.inverse = tuple(self.offsets.index((-dy, -dx)) for dy, dx in self.offsets)
        self.position_bias = nn.Parameter(torch.zeros(heads, len(self.offsets)))

    def _scores(self, ec: Tensor, ep: Tensor) -> Tensor:
        """Sparse edges: score[b,head,offset,y,x] = S[CT(y,x), PET(y+dy,x+dx)]."""
        valid = torch.ones_like(ec[:, :1, 0], dtype=torch.bool)
        with _fp32_context(ec):
            ec, ep = ec.float(), ep.float()
            scores = []
            for k, (dy, dx) in enumerate(self.offsets):
                s = (ec * _neighbor(ep, dy, dx)).sum(2) * self.head_dim ** -0.5
                s = s + self.position_bias[:, k].float()[None, :, None, None]
                scores.append(s.masked_fill(~_neighbor(valid, dy, dx), -torch.inf))
            return torch.stack(scores, dim=2)

    def _weights(self, ec: Tensor, ep: Tensor) -> Streams:
        s = self._scores(ec, ep)
        # Exact sparse transpose: at PET(j), candidate CT(j+offset) uses the
        # original score at CT(j+offset), with the opposite offset. This also
        # reuses the original directed position bias rather than predicting it.
        rev = torch.stack([_neighbor(s[:, :, self.inverse[k]], dy, dx, -torch.inf)
                           for k, (dy, dx) in enumerate(self.offsets)], dim=2)
        return s.softmax(dim=2), rev.softmax(dim=2)

    def forward(self, c: Tensor, p: Tensor) -> Streams:
        ec, ep, vc, vp = self.project(c, p)
        ac, ap = self._weights(ec, ep)
        with _fp32_context(c):
            vc, vp = vc.float(), vp.float()
            mc, mp = torch.zeros_like(vp), torch.zeros_like(vc)
            for k, (dy, dx) in enumerate(self.offsets):
                mc = mc + ac[:, :, k].unsqueeze(2) * _neighbor(vp, dy, dx)
                mp = mp + ap[:, :, k].unsqueeze(2) * _neighbor(vc, dy, dx)
        return self.update(c, p, mc.to(ec.dtype), mp.to(ep.dtype))


class _AxialExchange(_Projections):
    def __init__(self, channels: int, dim: int, heads: int, axis: str,
                 max_axis_length: int, chunk_size: int):
        super().__init__(channels, dim, heads)
        self.axis, self.max_axis_length, self.chunk_size = axis, max_axis_length, chunk_size
        self.position_bias = nn.Parameter(torch.zeros(heads, 2 * max_axis_length - 1))

    def _pack(self, x: Tensor) -> Tensor:
        # B,heads,d,H,W -> B*H,heads,W,d or B*W,heads,H,d
        b, heads, d, h, w = x.shape
        if self.axis == "horizontal":
            return x.permute(0, 3, 1, 4, 2).reshape(b * h, heads, w, d)
        return x.permute(0, 4, 1, 3, 2).reshape(b * w, heads, h, d)

    def _unpack(self, x: Tensor, shape: torch.Size) -> Tensor:
        b, heads, d, h, w = shape
        if self.axis == "horizontal":
            return x.reshape(b, h, heads, w, d).permute(0, 2, 4, 1, 3).contiguous()
        return x.reshape(b, w, heads, h, d).permute(0, 2, 4, 3, 1).contiguous()

    def forward(self, c: Tensor, p: Tensor) -> Streams:
        length = c.shape[-1] if self.axis == "horizontal" else c.shape[-2]
        if length > self.max_axis_length:
            raise ValueError(f"{self.axis} length {length} exceeds max_axis_length={self.max_axis_length}")
        ec, ep, vc, vp = self.project(c, p)
        shape = ec.shape
        ec, ep, vc, vp = (self._pack(x) for x in (ec, ep, vc, vp))
        positions = torch.arange(length, device=c.device)
        # delta = PET position - CT position; reverse uses S.transpose exactly.
        index = positions[None, :] - positions[:, None] + self.max_axis_length - 1
        mc, mp = [], []
        with _fp32_context(c):
            bias = self.position_bias[:, index].float().unsqueeze(0)
            for start in range(0, ec.shape[0], self.chunk_size):
                stop = start + self.chunk_size
                s = ec[start:stop].float() @ ep[start:stop].float().transpose(-1, -2)
                s = s * self.head_dim ** -0.5 + bias
                mc.append(s.softmax(-1) @ vp[start:stop].float())
                mp.append(s.transpose(-1, -2).softmax(-1) @ vc[start:stop].float())
        mc = self._unpack(torch.cat(mc).to(ec.dtype), shape)
        mp = self._unpack(torch.cat(mp).to(ep.dtype), shape)
        return self.update(c, p, mc, mp)


class SpatialBidirectionalFusion(nn.Module):
    """One-scale module; forward returns (fused, ct_enhanced, pet_enhanced).

    `pet_available=None` means all Full, False means all Missing, or pass a
    length-B bool/0-1 integer vector. PET must have full batch layout when any
    rows are Full. Missing rows are ignored, including non-finite PET values.
    pet_enhanced is zeros on Missing rows (placeholder, NOT reconstructed PET).

    Normalization/projections are independent per step and per modality.
    mode: full | local | axial | add; use one mode consistently at all scales.
    max_axis_length=128 supports the specified 512x512 input feature pyramid.
    checkpointing reduces saved training intermediates, not attention FLOPs.
    """
    def __init__(self, channels: int, attention_dim: int = 64, num_heads: int = 4,
                 local_kernel_size: int = 5, max_axis_length: int = 128,
                 axis_chunk_size: int = 32, mode: str = "full",
                 use_checkpoint: bool = True, check_finite: bool = True):
        super().__init__()
        positive = (channels, attention_dim, num_heads, local_kernel_size, max_axis_length, axis_chunk_size)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in positive):
            raise ValueError("Channel, dimension, head, kernel, length and chunk arguments must be positive integers")
        if attention_dim % num_heads or local_kernel_size % 2 == 0:
            raise ValueError("attention_dim must divide into heads; local kernel must be odd")
        if mode not in ("full", "local", "axial", "add"):
            raise ValueError("mode must be full, local, axial or add")
        self.channels, self.mode = channels, mode
        self.use_checkpoint, self.check_finite = use_checkpoint, check_finite
        steps = []
        if mode in ("full", "local"):
            steps.append(_LocalExchange(channels, attention_dim, num_heads, local_kernel_size))
        if mode in ("full", "axial"):
            steps.extend(_AxialExchange(channels, attention_dim, num_heads, axis,
                                        max_axis_length, axis_chunk_size)
                         for axis in ("horizontal", "vertical"))
        self.steps = nn.ModuleList(steps)

    def _full(self, c: Tensor, p: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.check_finite and (not bool(torch.isfinite(c).all()) or not bool(torch.isfinite(p).all())):
            raise RuntimeError("Non-finite CT/PET features in Full rows")
        for step in self.steps:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                c, p = checkpoint(step, c, p, use_reentrant=False)
            else:
                c, p = step(c, p)
        fused = c + p
        if self.check_finite and not bool(torch.isfinite(fused).all()):
            raise RuntimeError("Non-finite fusion output")
        return fused, c, p

    def forward(self, ct_feature: Tensor, pet_feature: Optional[Tensor] = None,
                pet_available: State = None) -> Tuple[Tensor, Tensor, Tensor]:
        _check_ct(ct_feature, self.channels)
        mask = _state(pet_available, ct_feature)
        if self.check_finite and not bool(torch.isfinite(ct_feature).all()):
            raise RuntimeError("Non-finite CT features")
        index = mask.nonzero(as_tuple=True)[0]
        if index.numel() == 0:
            return ct_feature, ct_feature, torch.zeros_like(ct_feature)
        if pet_feature is None:
            raise ValueError("Full rows require real PET features")
        _check_pair(ct_feature, pet_feature)
        if index.numel() == ct_feature.shape[0]:
            return self._full(ct_feature, pet_feature)
        fused, c, p = self._full(ct_feature.index_select(0, index), pet_feature.index_select(0, index))
        return (ct_feature.index_copy(0, index, fused),
                ct_feature.index_copy(0, index, c),
                torch.zeros_like(ct_feature).index_copy(0, index, p))


class MultiScaleSpatialBidirectionalFusion(nn.Module):
    """Four independent, structurally identical fusion blocks.

    Drop-in Full-path interface for baseline AddFusion:
        fused_features = module(ct_features, pet_features)
    Optional future interface:
        fused, ct_enhanced, pet_enhanced = module(..., return_features=True)

    Spatial resizing of PET follows inspected AddFusion: bilinear with
    align_corners=False. Channel alignment remains the model's responsibility.
    """
    def __init__(self, channels: Sequence[int] = (64, 128, 320, 512),
                 attention_dim: int = 64, num_heads: int = 4,
                 local_kernel_size: int = 5, max_axis_length: int = 128,
                 axis_chunk_size: int = 32, mode: str = "full",
                 use_checkpoint: bool = True, check_finite: bool = True,
                 resize_pet: bool = True):
        super().__init__()
        self.channels = tuple(channels)
        if len(self.channels) != 4:
            raise ValueError("Expected exactly four shallow-to-deep channel counts")
        self.resize_pet = resize_pet
        self.blocks = nn.ModuleList(SpatialBidirectionalFusion(
            c, attention_dim, num_heads, local_kernel_size, max_axis_length,
            axis_chunk_size, mode, use_checkpoint, check_finite) for c in self.channels)

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Optional[Sequence[Tensor]] = None,
                pet_available: State = None, return_features: bool = False):
        if len(ct_feats) != 4 or (pet_feats is not None and len(pet_feats) != 4):
            raise ValueError("CT and PET pyramids must each have exactly four scales")
        for c, ch in zip(ct_feats, self.channels):
            _check_ct(c, ch)
        if any(c.shape[0] != ct_feats[0].shape[0] or c.device != ct_feats[0].device
               or c.dtype != ct_feats[0].dtype for c in ct_feats):
            raise ValueError("CT scales must share batch size, device and dtype")
        mask = _state(pet_available, ct_feats[0])
        # All-Missing must not even inspect/resize supplied PET tensors.
        has_full = bool(mask.any())
        if has_full and pet_feats is None:
            raise ValueError("Full rows require a four-scale PET pyramid")
        fused, ct_out, pet_out = [], [], []
        for i, block in enumerate(self.blocks):
            c = ct_feats[i]
            p = pet_feats[i] if has_full else None
            if p is not None:
                if not isinstance(p, Tensor) or p.ndim != 4 or not p.is_floating_point():
                    raise ValueError(f"Invalid PET tensor at scale {i}")
                if p.shape[:2] != c.shape[:2] or p.device != c.device or p.dtype != c.dtype:
                    raise ValueError(f"PET batch/channels/device/dtype mismatch at scale {i}")
                if min(p.shape) < 1:
                    raise ValueError(f"Empty PET tensor at scale {i}")
                if p.shape[-2:] != c.shape[-2:]:
                    if not self.resize_pet:
                        raise ValueError(f"PET spatial mismatch at scale {i}")
                    # Interpolation never mixes different samples; Missing PET
                    # rows cannot influence Full rows and will be discarded.
                    p = F.interpolate(p, size=c.shape[-2:], mode="bilinear", align_corners=False)
            f, co, po = block(c, p, mask)
            fused.append(f)
            ct_out.append(co)
            pet_out.append(po)
        return (fused, ct_out, pet_out) if return_features else fused


def _smoke(device: str, native_sizes: bool) -> None:
    torch.manual_seed(2023)
    model = MultiScaleSpatialBidirectionalFusion().to(device)
    sizes = (128, 64, 32, 16) if native_sizes else (16, 8, 4, 2)
    ct = [torch.randn(2, ch, s, s, device=device, requires_grad=True)
          for ch, s in zip(model.channels, sizes)]
    pet = [torch.randn_like(c, requires_grad=True) for c in ct]
    state = torch.tensor([1, 0], device=device)
    out = model(ct, pet, pet_available=state)
    for c, p, f in zip(ct, pet, out):
        assert torch.equal(f[0], c[0] + p[0])
        assert torch.equal(f[1], c[1])
    sum(f.square().mean() for f in out).backward()
    assert all(p.grad is not None and bool((p.grad[1] == 0).all()) for p in pet)
    print("PASS: mixed forward/backward, exact additive initialization, Missing isolation")
    print("shapes:", [tuple(f.shape) for f in out])
    print("parameters:", sum(p.numel() for p in model.parameters()))
    if device.startswith("cuda"):
        print("peak_allocated_MiB:", torch.cuda.max_memory_allocated() / 2**20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--native-sizes", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.smoke:
        _smoke(args.device, args.native_sizes)
    else:
        parser.print_help()
