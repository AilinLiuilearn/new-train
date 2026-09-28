"""CT base context + PET-conditioned increment fusion (standalone PyTorch).

Proposed research module, NOT a reproduction of KTB or a validated method.
Only the bounded sampling idea is related to deformable attention; this file
is an original implementation with no mmcv/timm/text encoder dependency.

Four aligned BCHW CT/PET feature maps -> four fused maps of the same shapes.
Default channels: (64,128,320,512); internal widths: (64,128,160,256).
F = Conv1x1(cat(C, read(H_C), read(gate*(H_CP-H_C)))). NO external +C.

Modes:
  full: PET required. Optional aux supplies detached bank_key/delta pairs.
  missing: PET MUST be None; supply either retrieved_deltas or a retriever.
  ct_only: explicit zero-increment ablation; NOT identity CT and NOT retrieval.

Retriever signature: retriever(scale_index, normalized_ct_keys) -> B,N,d.
It is called with live keys, so differentiable soft retrieval can train keys.
The returned value must already represent the POST-GATE Full increment.
Bank maintenance, clustering, labels, encoders, loss, optimizer and EMA are
external. There are no hidden caches or updates to a bank during forward.
Input PET/CT maps must be spatially aligned before entering this module.

Requires PyTorch >= 2.0 (SDPA and non-reentrant activation checkpointing).
Float32 master weights + autocast recommended, rather than model.half().
Example:
  model = PETCTIncrementFusion()
  fused, aux = model(ct_features, pet_features, return_aux=True)
  # aux[i]['bank_key'], aux[i]['delta'] are detached by default.
  missing = model(ct_features, state='missing', retriever=my_retriever)
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ['PETCTIncrementFusion']
Retriever = Callable[[int, Tensor], Tensor]
Grid = Union[int, Tuple[int, int]]


def _grid_pair(grid: Grid) -> Tuple[int, int]:
    values = (grid, grid) if isinstance(grid, int) else tuple(grid)
    if len(values) != 2 or any(not isinstance(v, int) or v < 1 for v in values):
        raise ValueError('Each region grid must be a positive int or (height,width).')
    return values


def _check_feature(x: Tensor, channels: int, name: str, check_finite: bool) -> None:
    if not isinstance(x, Tensor) or x.ndim != 4 or x.shape[1] != channels:
        raise ValueError(f'{name} must be BCHW with channels={channels}.')
    if min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError(f'{name} must be a nonempty floating-point tensor.')
    if check_finite and not bool(torch.isfinite(x).all()):
        raise ValueError(f'{name} contains NaN/Inf; values are not sanitized.')


class _Projection(nn.Sequential):
    def __init__(self, in_channels: int, width: int, local_ct: bool):
        layers = [nn.Conv2d(in_channels, width, 1, bias=False),
                  nn.GroupNorm(1, width), nn.GELU()]
        if local_ct:
            layers.append(nn.Conv2d(width, width, 3, padding=1,
                                    groups=width, bias=False))
        super().__init__(*layers)


class _ScaleFusion(nn.Module):
    def __init__(self, channels: int, width: int, heads: int,
                 grid: Tuple[int, int], points: int, offset_radius: float,
                 query_chunk_size: int, checkpoint_attention: bool,
                 check_finite: bool):
        super().__init__()
        self.channels, self.width, self.heads = channels, width, heads
        self.grid, self.points = grid, points
        self.offset_radius = offset_radius
        self.query_chunk_size = query_chunk_size
        self.checkpoint_attention = checkpoint_attention
        self.check_finite = check_finite
        self.ct_proj = _Projection(channels, width, local_ct=True)
        self.pet_proj = _Projection(channels, width, local_ct=False)
        self.local_norm = nn.LayerNorm(width)
        self.offset = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                    nn.GELU(), nn.Linear(width, 2 * points))
        # One set of Q/K/V projections for CT-only and CT+PET local reading.
        self.local_q = nn.Linear(width, width, bias=False)
        self.local_k = nn.Linear(width, width, bias=False)
        self.local_v = nn.Linear(width, width, bias=False)
        # No output bias: a shared bias cancels in the difference anyway.
        self.local_out = nn.Linear(width, width, bias=False)
        self.gate = nn.Sequential(nn.LayerNorm(2 * width),
                                  nn.Linear(2 * width, width), nn.GELU(),
                                  nn.Linear(width, width))
        self.read_query = nn.Linear(width, width, bias=False)
        self.read_key = nn.Linear(width, width, bias=False)
        self.out_proj = nn.Conv2d(channels + 2 * width, channels, 1, bias=True)

        # Regular points within each regional bin: 4 -> its four quadrants.
        side = math.isqrt(points)
        a = (torch.arange(side, dtype=torch.float32) + 0.5) / side - 0.5
        yy, xx = torch.meshgrid(a, a, indexing='ij')
        self.register_buffer('base_offsets', torch.stack([xx, yy], -1).reshape(points, 2),
                             persistent=False)
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        # Ordinary learned sigmoid gate, without standalone gamma parameters.
        nn.init.zeros_(self.gate[-1].bias)
        # All three output blocks are active at step 0. No CT-identity claim.
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def sampling_grid(self, a: Tensor, height: int, width: int,
                      gh: int, gw: int) -> Tensor:
        """Normalized (x,y) coordinates for align_corners=False, B,N,S,2.

        Pool-bin bounds match adaptive pooling even for nondivisible shapes.
        Offsets are bounded in units of the bin width/height, not pixels.
        """
        offsets = self.offset(a).reshape(a.shape[0], gh * gw, self.points, 2)
        with torch.autocast(device_type=a.device.type, enabled=False):
            iy = torch.arange(gh, device=a.device, dtype=torch.float32)
            ix = torch.arange(gw, device=a.device, dtype=torch.float32)
            y0, y1 = (iy * height / gh).floor(), ((iy + 1) * height / gh).ceil()
            x0, x1 = (ix * width / gw).floor(), ((ix + 1) * width / gw).ceil()
            cy, cx = (y0 + y1) / height - 1, (x0 + x1) / width - 1
            sy, sx = 2 * (y1 - y0) / height, 2 * (x1 - x0) / width
            yy, xx = torch.meshgrid(cy, cx, indexing='ij')
            hy, wx = torch.meshgrid(sy, sx, indexing='ij')
            center = torch.stack([xx, yy], -1).reshape(1, gh * gw, 1, 2)
            extent = torch.stack([wx, hy], -1).reshape(1, gh * gw, 1, 2)
            shift = self.base_offsets.float()[None, None] + self.offset_radius * offsets.float().tanh()
            pos = center + extent * shift
            # Bound to pixel centers, so border samples never read zero padding.
            lo = pos.new_tensor([-1 + 1 / width, -1 + 1 / height])
            hi = pos.new_tensor([1 - 1 / width, 1 - 1 / height])
            return torch.maximum(torch.minimum(pos, hi), lo)

    @staticmethod
    def sample(x: Tensor, pos: Tensor) -> Tensor:
        # AMP-safe grid sampling; backward flows to both features and offsets.
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = F.grid_sample(x.float(), pos.float(), mode='bilinear',
                              padding_mode='border', align_corners=False)
        return y.permute(0, 2, 3, 1).to(dtype=x.dtype)  # B,N,S,d

    def local_read(self, a: Tensor, samples: Tensor) -> Tensor:
        b, n, s, d = samples.shape
        h, dh = self.heads, d // self.heads
        q = self.local_q(self.local_norm(a)).reshape(b, n, h, dh).unsqueeze(-2)
        tokens = self.local_norm(samples)
        k = self.local_k(tokens).reshape(b, n, s, h, dh).transpose(2, 3)
        v = self.local_v(tokens).reshape(b, n, s, h, dh).transpose(2, 3)
        # Only S/2S candidates per region; float32 softmax avoids fp16 overflow.
        with torch.autocast(device_type=a.device.type, enabled=False):
            weight = (q.float() @ k.float().transpose(-2, -1)) * (dh ** -0.5)
            z = weight.softmax(-1) @ v.float()
        return self.local_out(z.squeeze(-2).reshape(b, n, d).to(dtype=a.dtype))

    @staticmethod
    def read_pair(q: Tensor, k: Tensor, hc: Tensor, delta: Tensor) -> Tuple[Tensor, Tensor]:
        # Same Q,K, scale and zero dropout => same mathematical attention W.
        # Two SDPA calls keep head dimensions equal, allowing fused CUDA kernels.
        # Never construct/store a dense HW x HW matrix or return attention maps.
        return (F.scaled_dot_product_attention(q, k, hc, dropout_p=0.0),
                F.scaled_dot_product_attention(q, k, delta, dropout_p=0.0))

    def write_back(self, xc: Tensor, key: Tensor, hc: Tensor,
                   delta: Tensor) -> Tuple[Tensor, Tensor]:
        b, d, height, width = xc.shape
        h, dh = self.heads, d // self.heads
        q = self.read_query(xc.flatten(2).transpose(1, 2))
        q = q.reshape(b, height * width, h, dh).transpose(1, 2)
        k = key.reshape(b, -1, h, dh).transpose(1, 2).to(dtype=q.dtype)
        base = hc.reshape(b, -1, h, dh).transpose(1, 2).to(dtype=q.dtype)
        inc = delta.reshape(b, -1, h, dh).transpose(1, 2).to(dtype=q.dtype)
        rc, rd = [], []
        use_cp = self.checkpoint_attention and self.training and torch.is_grad_enabled()
        for start in range(0, height * width, self.query_chunk_size):
            args = (q[:, :, start:start + self.query_chunk_size], k, base, inc)
            if use_cp:
                x, y = checkpoint(self.read_pair, *args, use_reentrant=False)
            else:
                x, y = self.read_pair(*args)
            rc.append(x)
            rd.append(y)
        def spatial(parts: List[Tensor]) -> Tensor:
            return torch.cat(parts, dim=2).transpose(1, 2).reshape(b, height, width, d).permute(0, 3, 1, 2)
        return spatial(rc), spatial(rd)

    def forward(self, ct: Tensor, pet: Optional[Tensor], state: str,
                delta: Optional[Tensor], retriever: Optional[Retriever],
                scale: int, return_aux: bool, detach_aux: bool):
        xc = self.ct_proj(ct)
        height, width = xc.shape[-2:]
        # Explicit grid cap permits smaller smoke-test inputs without up-pooling.
        gh, gw = min(self.grid[0], height), min(self.grid[1], width)
        a = F.adaptive_avg_pool2d(xc, (gh, gw)).flatten(2).transpose(1, 2)
        pos = self.sampling_grid(a, height, width, gh, gw)
        c_samples = self.sample(xc, pos)
        hc = self.local_read(a, c_samples)
        key = self.read_key(a)
        # Bank keys are normalized; dense read uses the raw projected keys.
        # Values remain signed/un-normalized to preserve increment magnitude.
        with torch.autocast(device_type=ct.device.type, enabled=False):
            bank_key = F.normalize(key.float(), dim=-1, eps=1e-6)
        gate = raw_delta = None
        if state == 'full':
            xp = self.pet_proj(pet.to(dtype=ct.dtype))
            p_samples = self.sample(xp, pos)
            hcp = self.local_read(a, torch.cat([c_samples, p_samples], dim=2))
            raw_delta = hcp - hc
            gate = self.gate(torch.cat([a, raw_delta], dim=-1)).sigmoid()
            delta = gate * raw_delta
        elif state == 'ct_only':
            delta = torch.zeros_like(hc)
        else:
            if retriever is not None:
                delta = retriever(scale, bank_key)
            if not isinstance(delta, Tensor) or tuple(delta.shape) != tuple(hc.shape):
                raise ValueError(f'retrieved delta at scale {scale} must have shape {tuple(hc.shape)}.')
            if delta.device != hc.device or not delta.is_floating_point():
                raise ValueError('Retrieved deltas must be floating tensors on the CT device.')
            if self.check_finite and not bool(torch.isfinite(delta).all()):
                raise ValueError('Retrieved delta contains NaN/Inf.')
            delta = delta.to(dtype=hc.dtype)  # differentiable; do not detach here
        rc, rd = self.write_back(xc, key, hc, delta)
        fused = self.out_proj(torch.cat([ct.to(dtype=rc.dtype), rc, rd], dim=1))
        if self.check_finite and not bool(torch.isfinite(fused).all()):
            raise RuntimeError(f'Fusion output at scale {scale} contains NaN/Inf.')
        if not return_aux:
            return fused, None
        aux = {'bank_key': bank_key, 'delta': delta,
               'ct_context': hc, 'sample_grid': pos}
        if state == 'full':
            aux.update({'raw_delta': raw_delta, 'gate': gate})
        if detach_aux:
            aux = {name: value.detach() for name, value in aux.items()}
        return fused, aux


class PETCTIncrementFusion(nn.Module):
    """Independent multi-scale fusion; instantiate AFTER baseline decoder.

    Full outputs are complete features, not residual corrections.
    state is a homogeneous string. For mixed batches the parent model must
    split rows BEFORE calling the PET encoder and this module, then scatter
    fused features and decode the whole batch once.

    No exact AddFusion/CT identity initialization is promised. Output blocks
    use Xavier initialization so every branch can receive gradients immediately.
    With a fixed torch seed initialization is reproducible on the same backend.

    region_grids entries are capped to input H,W (reported via aux shapes).
    num_points must be a square, e.g. 1,4,9; default=4, not four attention heads.
    query_chunk_size changes computation scheduling, not the receptive field.
    """
    def __init__(self, channels: Sequence[int] = (64, 128, 320, 512),
                 inner_channels: Sequence[int] = (64, 128, 160, 256),
                 num_heads: int = 4, region_grids: Sequence[Grid] = (16, 16, 16, 16),
                 num_points: int = 4, offset_radius: float = 0.5,
                 query_chunk_size: int = 512, checkpoint_attention: bool = True,
                 check_finite: bool = True):
        super().__init__()
        self.channels, self.inner_channels = tuple(channels), tuple(inner_channels)
        self.region_grids = tuple(_grid_pair(g) for g in region_grids)
        self.num_heads, self.num_points = num_heads, num_points
        self.offset_radius = float(offset_radius)
        if not self.channels or len(self.channels) != len(self.inner_channels) or len(self.channels) != len(self.region_grids):
            raise ValueError('channels, inner_channels and region_grids must have equal nonzero lengths.')
        if any(not isinstance(v, int) or v < 1 for v in self.channels + self.inner_channels):
            raise ValueError('Channel counts must be positive integers.')
        if not isinstance(num_heads, int) or num_heads < 1 or any(d % num_heads for d in self.inner_channels):
            raise ValueError('Every inner width must be divisible by num_heads.')
        if not isinstance(num_points, int) or num_points < 1 or math.isqrt(num_points) ** 2 != num_points:
            raise ValueError('num_points must be a positive square: 1,4,9,...')
        if not math.isfinite(self.offset_radius) or self.offset_radius < 0:
            raise ValueError('offset_radius must be finite and nonnegative.')
        if not isinstance(query_chunk_size, int) or query_chunk_size < 1:
            raise ValueError('query_chunk_size must be a positive integer.')
        self.check_finite = bool(check_finite)
        self.stages = nn.ModuleList([
            _ScaleFusion(c, d, num_heads, grid, num_points, self.offset_radius,
                         query_chunk_size, bool(checkpoint_attention), self.check_finite)
            for c, d, grid in zip(self.channels, self.inner_channels, self.region_grids)])

    def get_extra_state(self) -> Dict:
        # Catch semantic checkpoint mismatch even when parameter shapes match.
        return {'format_version': 1, 'channels': self.channels,
                'inner_channels': self.inner_channels, 'region_grids': self.region_grids,
                'num_heads': self.num_heads, 'num_points': self.num_points,
                'offset_radius': self.offset_radius}

    def set_extra_state(self, state: Dict) -> None:
        if state != self.get_extra_state():
            raise RuntimeError(f'Fusion checkpoint contract mismatch: saved={state}, current={self.get_extra_state()}')

    def forward(self, ct_feats: Sequence[Tensor], pet_feats: Optional[Sequence[Tensor]] = None,
                state: str = 'full', *, retrieved_deltas: Optional[Sequence[Tensor]] = None,
                retriever: Optional[Retriever] = None, return_aux: bool = False,
                detach_aux: bool = True):
        if not isinstance(state, str) or state not in ('full', 'missing', 'ct_only'):
            raise ValueError("state must be 'full', 'missing', or 'ct_only'; split mixed batches upstream.")
        if not isinstance(ct_feats, (list, tuple)) or len(ct_feats) != len(self.stages):
            raise ValueError(f'Expected {len(self.stages)} CT scales in S1->S4 order.')
        if state == 'full':
            if not isinstance(pet_feats, (list, tuple)) or len(pet_feats) != len(self.stages):
                raise ValueError('Full requires one real PET feature tensor per scale.')
            if retrieved_deltas is not None or retriever is not None:
                raise ValueError('Full does not accept retrieved deltas or a retriever.')
        elif pet_feats is not None:
            raise ValueError('Missing/ct_only require pet_feats=None; real PET must not enter this path.')
        if state == 'missing':
            if (retrieved_deltas is None) == (retriever is None):
                raise ValueError('Missing requires exactly one of retrieved_deltas or retriever.')
            if retriever is not None and not callable(retriever):
                raise ValueError('retriever must be callable.')
            if retrieved_deltas is not None and (not isinstance(retrieved_deltas, (list, tuple)) or len(retrieved_deltas) != len(self.stages)):
                raise ValueError('retrieved_deltas must contain one B,N,d tensor per scale.')
        if state == 'ct_only' and (retriever is not None or retrieved_deltas is not None):
            raise ValueError('ct_only is explicit zero-increment mode; do not supply a bank.')
        batch = device = None
        for i, (c, channels) in enumerate(zip(ct_feats, self.channels)):
            _check_feature(c, channels, f'CT scale {i}', self.check_finite)
            if i == 0:
                batch, device = c.shape[0], c.device
            if c.shape[0] != batch or c.device != device:
                raise ValueError('All scales must share the CT batch size and device.')
            if state == 'full':
                p = pet_feats[i]
                _check_feature(p, channels, f'PET scale {i}', self.check_finite)
                if p.shape != c.shape or p.device != c.device:
                    raise ValueError('CT/PET shapes and devices must match; no automatic spatial alignment.')
        fused, aux = [], []
        for i, (stage, c) in enumerate(zip(self.stages, ct_feats)):
            y, info = stage(c, pet_feats[i] if state == 'full' else None,
                            state, retrieved_deltas[i] if retrieved_deltas is not None else None,
                            retriever, i, return_aux, detach_aux)
            fused.append(y)
            if return_aux:
                aux.append(info)
        return (fused, aux) if return_aux else fused


if __name__ == '__main__':
    torch.manual_seed(2023)
    torch.set_num_threads(2)
    module = PETCTIncrementFusion(channels=(8, 16, 24, 32),
                                  inner_channels=(8, 16, 24, 32),
                                  region_grids=(4, 4, 2, 2))
    ct = [torch.randn(1, c, s, s, requires_grad=True)
          for c, s in zip(module.channels, (16, 8, 4, 2))]
    pet = [torch.randn_like(c, requires_grad=True) for c in ct]
    result, aux = module(ct, pet, return_aux=True)
    sum(x.square().mean() for x in result).backward()
    module.eval()
    with torch.no_grad():
        full, aux = module(ct, pet, return_aux=True)
        replay = module(ct, state='missing', retrieved_deltas=[a['delta'] for a in aux])
        for x, y in zip(full, replay):
            torch.testing.assert_close(x, y)
    print('CPU smoke passed:', [tuple(x.shape) for x in full])
    print('Default parameters:', sum(p.numel() for p in PETCTIncrementFusion().parameters()))
