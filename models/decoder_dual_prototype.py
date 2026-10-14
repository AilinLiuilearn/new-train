"""Task-selected decoder memories. All collection/update methods are no-grad.

Spatial memory: Euclidean centers of local Full descriptors in useful regions.
Channel memory: centers of signed, weighted Gram(Full)-Gram(CT-reference).
Queries always come from the current Missing decoder feature. No paired CT
keys, PET reconstruction, trainable memory parameters, or inference updates.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class EMACentroidBank(nn.Module):
    """Bounded checkpointed FIFO + deterministic Euclidean K-means + EMA.

No per-vector L2 normalization: clustering is ordinary Euclidean, not
spherical K-means. New centers are matched one-to-one to old slots before
EMA, using deterministic greedy minimum-distance matching.
"""

    def __init__(self, clusters: int, dimension: int, capacity: int = 256,
                 min_candidates: int = 24, decay: float = .999,
                 iterations: int = 15, outlier_mad: float = 3.,
                 outlier_filter: bool = False):
        super().__init__()
        if not (1 < clusters <= min_candidates <= capacity) or dimension < 1:
            raise ValueError('require 1 < clusters <= min_candidates <= capacity and dimension > 0')
        if not 0 <= decay < 1 or iterations < 1 or outlier_mad <= 0:
            raise ValueError('invalid centroid EMA/K-means configuration')
        self.clusters, self.dimension = int(clusters), int(dimension)
        self.capacity, self.min_candidates = int(capacity), int(min_candidates)
        self.decay, self.iterations, self.outlier_mad = float(decay), int(iterations), float(outlier_mad)
        self.outlier_filter = bool(outlier_filter)
        self.register_buffer('centers', torch.zeros(clusters, dimension))
        self.register_buffer('pool', torch.zeros(capacity, dimension))
        self.register_buffer('size', torch.zeros((), dtype=torch.long))
        self.register_buffer('cursor', torch.zeros((), dtype=torch.long))
        self.register_buffer('valid', torch.zeros((), dtype=torch.bool))
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('accepted', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def add(self, values: Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f'expected [N,{self.dimension}] candidates')
        if not torch.isfinite(values).all():
            raise ValueError('nonfinite memory candidate')
        values = values.detach().to(device=self.pool.device, dtype=self.pool.dtype)
        if values.shape[0] == 0:
            return
        self.accepted.add_(values.shape[0])
        values = values[-self.capacity:]
        n, start = values.shape[0], int(self.cursor)
        first = min(n, self.capacity - start)
        self.pool[start:start + first].copy_(values[:first])
        if first < n:
            self.pool[:n - first].copy_(values[first:])
        self.cursor.fill_((start + n) % self.capacity)
        self.size.fill_(min(self.capacity, int(self.size) + n))

    @staticmethod
    def _distance(x: Tensor, centers: Tensor) -> Tensor:
        return (x.square().sum(1, keepdim=True) + centers.square().sum(1)[None]
                - 2 * x @ centers.T).clamp_min_(0)

    @torch.no_grad()
    def refresh(self) -> bool:
        if int(self.size) < self.min_candidates:
            return False
        # Disable ambient AMP: Gram/K-means/EMA stay float32 in formal AMP runs.
        with torch.autocast(device_type=self.pool.device.type, enabled=False):
            x = self.pool[:int(self.size)].float()
            if torch.unique(x, dim=0).shape[0] < self.clusters:
                return False
            if bool(self.valid):
                centers = self.centers.float().clone()
            else:
                # Deterministic farthest-point seeds, independent of global RNG.
                first = ((x - x.mean(0)).square().sum(1)).argmax()
                seeds = [x[first]]
                for _ in range(1, self.clusters):
                    seeds.append(x[self._distance(x, torch.stack(seeds)).min(1).values.argmax()])
                centers = torch.stack(seeds)
            previous = None
            for _ in range(self.iterations):
                distances = self._distance(x, centers)
                assignment = distances.argmin(1)
                if previous is not None and torch.equal(previous, assignment):
                    break
                previous = assignment
                updated = centers.clone()
                for k in range(self.clusters):
                    members = x[assignment == k]
                    if members.numel():
                        updated[k] = members.mean(0)
                    else:
                        updated[k] = x[distances.min(1).values.argmax()]
                centers = updated
            assignment = self._distance(x, centers).argmin(1)
            # Default: every cluster member contributes to its center.
            # Optional MAD trimming is retained only for explicit ablations.
            for k in range(self.clusters):
                members = x[assignment == k]
                if members.shape[0] == 0:
                    # Do not mark a collapsed bank as a valid K-center library.
                    return False
                if self.outlier_filter:
                    distance = (members - centers[k]).square().sum(1).sqrt()
                    median = distance.median()
                    mad = (distance - median).abs().median()
                    keep = distance <= median + self.outlier_mad * 1.4826 * mad + 1e-6
                    members = members[keep]
                centers[k] = members.mean(0)
            if bool(self.valid):
                distances = self._distance(self.centers.float(), centers)
                ordered = torch.empty_like(centers)
                for _ in range(self.clusters):
                    flat = int(distances.argmin())
                    old, new = divmod(flat, self.clusters)
                    ordered[old] = centers[new]
                    distances[old, :] = float('inf')
                    distances[:, new] = float('inf')
                self.centers.mul_(self.decay).add_(ordered, alpha=1 - self.decay)
            else:
                self.centers.copy_(centers)
                self.valid.fill_(True)
            self.updates.add_(1)
        return True


def gain_weights(full_logits: Tensor, reference_logits: Tensor, target: Tensor,
                 foreground_weight: float = 2., background_weight: float = 1.,
                 min_gain: float = .01) -> Tensor:
    """Detached soft error improvement; Full must also be threshold-correct.

Foreground/background weights influence candidate selection/composition;
they do not change the baseline BCE+Dice objective. An absolute gain floor
rejects tiny confidence fluctuations. This is an operational benefit score,
not a causal estimate of PET contribution.
"""
    with torch.no_grad(), torch.autocast(device_type=full_logits.device.type, enabled=False):
        pf, p0 = full_logits.detach().float().sigmoid(), reference_logits.detach().float().sigmoid()
        if target.ndim == 3:
            target = target[:, None]
        y = F.interpolate(target.float(), size=pf.shape[-2:], mode='nearest').ge(.5).float()
        improvement = (p0 - y).square() - (pf - y).square()
        reliable = pf.ge(.5).eq(y.bool())
        priority = foreground_weight * y + background_weight * (1 - y)
        return improvement.clamp_min(0) * reliable * (improvement >= min_gain) * priority


class DecoderPrototypeScale(nn.Module):
    """One decoder level; memory source and queries share a fixed basis.

Fixed seeded orthogonal compression avoids an independently trained
collector whose moving coordinate basis would invalidate stored matrices.
Learnable attention projections and output heads are optimized by Missing
segmentation loss. Shapes: feature [B,C,H,W], spatial centers [K,3d],
channel centers [K,d*d]. Both final heads start at exactly zero.
"""

    def __init__(self, channels: int, dim: int = 32, clusters: int = 6,
                 capacity: int = 256, min_candidates: int = 24,
                 decay: float = .999, refresh_steps: int = 20,
                 spatial: bool = True, channel: bool = True,
                 patch_size: int = 5, max_patches: int = 16, seed: int = 2023,
                 outlier_filter: bool = False):
        super().__init__()
        if not spatial and not channel:
            raise ValueError('at least one prototype branch must be enabled')
        if channels < 1 or dim < 1 or refresh_steps < 1 or patch_size < 1 or max_patches < 1:
            raise ValueError('invalid prototype dimensions or collection cadence')
        self.channels, self.dim = int(channels), min(int(dim), int(channels))
        self.spatial_enabled, self.channel_enabled = bool(spatial), bool(channel)
        self.refresh_steps, self.patch_size, self.max_patches = int(refresh_steps), int(patch_size), int(max_patches)
        gen = torch.Generator().manual_seed(int(seed))
        basis = torch.linalg.qr(torch.randn(channels, self.dim, generator=gen), mode='reduced').Q.T
        self.register_buffer('basis', basis.contiguous())
        self.spatial_bank = EMACentroidBank(clusters, 3 * self.dim, capacity, min_candidates,
                                            decay, outlier_filter=outlier_filter)
        self.channel_bank = EMACentroidBank(clusters, self.dim**2, capacity, min_candidates,
                                            decay, outlier_filter=outlier_filter)
        d = self.dim
        self.spatial_norm = nn.LayerNorm(3 * d)
        self.spatial_q = nn.Linear(3 * d, d, bias=False)
        self.spatial_k = nn.Linear(3 * d, d, bias=False)
        self.spatial_v = nn.Linear(3 * d, d, bias=False)
        self.spatial_gate = nn.Conv2d(channels + d, 1, 1)
        self.spatial_out = nn.Conv2d(d, channels, 1, bias=False)
        self.channel_norm_q = nn.LayerNorm(d + d*d)
        self.channel_norm_k = nn.LayerNorm(d*d)
        self.channel_q = nn.Linear(d + d*d, d, bias=False)
        self.channel_k = nn.Linear(d*d, d, bias=False)
        self.channel_v = nn.Linear(d*d, d, bias=False)
        self.channel_out = nn.Linear(d, 2 * channels)
        nn.init.zeros_(self.spatial_out.weight)
        nn.init.zeros_(self.channel_out.weight)
        nn.init.zeros_(self.channel_out.bias)

    def _project(self, x: Tensor) -> Tensor:
        # Fixed compression; RMS scaling is handled symmetrically below.
        return torch.einsum('dc,bchw->bdhw', self.basis.float(), x.float())

    @staticmethod
    def _local(z: Tensor) -> Tensor:
        mean = F.avg_pool2d(z, 3, stride=1, padding=1, count_include_pad=False)
        deviation = F.avg_pool2d((z - mean).abs(), 3, stride=1,
                                 padding=1, count_include_pad=False)
        return torch.cat([z, mean, deviation], 1)

    @staticmethod
    def _gram(z: Tensor, w: Tensor) -> Tensor:
        z, w = z.flatten(2), w.flatten(2)
        return (z * w) @ z.transpose(1, 2) / w.sum(2, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def candidates(self, full: Tensor, reference: Tensor, weight: Tensor,
                   foreground: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if full.shape != reference.shape or full.ndim != 4 or full.shape[1] != self.channels:
            raise ValueError('Full/reference decoder features must match [B,C,H,W]')
        with torch.autocast(device_type=full.device.type, enabled=False):
            # Symmetric common scaling preserves sign under Full/reference swap.
            zf, z0 = self._project(full), self._project(reference)
            rms = ((zf.square().mean((1, 2, 3), keepdim=True)
                    + z0.square().mean((1, 2, 3), keepdim=True)) * .5 + 1e-6).sqrt()
            zf, z0 = zf / rms, z0 / rms
            w = F.interpolate(weight.detach().float(), size=full.shape[-2:], mode='area')
            mass = w.sum((1, 2, 3))
            useful = mass > 1e-6
            relations = (self._gram(zf, w) - self._gram(z0, w))[useful].flatten(1)
            if not self.channel_enabled:
                relations = relations[:0]
            spatial = []
            if self.spatial_enabled:
                local = self._local(zf)
                p = min(self.patch_size, *full.shape[-2:])
                wm = F.avg_pool2d(w, p, stride=p, ceil_mode=True, count_include_pad=False)
                pooled = F.avg_pool2d(local * w, p, stride=p, ceil_mode=True,
                                      count_include_pad=False) / wm.clamp_min(1e-8)
                fg = (F.interpolate(foreground.float(), size=full.shape[-2:], mode='area')
                      if foreground is not None else torch.ones_like(w))
                foreground_mass = F.avg_pool2d(fg * w, p, stride=p, ceil_mode=True,
                                              count_include_pad=False)
                for b in range(full.shape[0]):
                    score = wm[b, 0].flatten()
                    valid = score > 1e-6
                    is_fg = foreground_mass[b, 0].flatten() > 1e-8
                    # Reserve foreground capacity so easy background cannot dominate.
                    fg_idx = (valid & is_fg).nonzero(as_tuple=True)[0]
                    bg_idx = (valid & ~is_fg).nonzero(as_tuple=True)[0]
                    fg_idx = fg_idx[score[fg_idx].argsort(descending=True)][:self.max_patches]
                    bg_budget = min(self.max_patches // 2, self.max_patches - fg_idx.numel())
                    bg_idx = bg_idx[score[bg_idx].argsort(descending=True)][:bg_budget]
                    idx = torch.cat([fg_idx, bg_idx])
                    if idx.numel():
                        spatial.append(pooled[b].flatten(1).T[idx])
            s = torch.cat(spatial) if spatial else full.new_empty((0, 3 * self.dim), dtype=torch.float32)
            return s.detach(), relations.detach()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'expected decoder [B,{self.channels},H,W] feature')
        use_s = self.spatial_enabled and bool(self.spatial_bank.valid)
        use_c = self.channel_enabled and bool(self.channel_bank.valid)
        if not use_s and not use_c:
            return x
        # Statistics and softmax in float32; final residual returns input dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self._project(x)
            z = z / (z.square().mean((1, 2, 3), keepdim=True) + 1e-6).sqrt()
            residual = torch.zeros_like(x, dtype=torch.float32)
            if use_s:
                local = self._local(z).flatten(2).transpose(1, 2)
                q = self.spatial_q(self.spatial_norm(local))
                memory = self.spatial_norm(self.spatial_bank.centers.detach())
                k, v = self.spatial_k(memory), self.spatial_v(memory)
                attn = (q @ k.T / math.sqrt(self.dim)).softmax(-1)
                read = (attn @ v).transpose(1, 2).reshape(x.shape[0], self.dim, *x.shape[-2:])
                gate = self.spatial_gate(torch.cat([x.float(), read], 1)).sigmoid()
                residual = residual + gate * self.spatial_out(read)
            if use_c:
                stat = torch.cat([z.mean((2, 3)), self._gram(z, torch.ones_like(z[:, :1])).flatten(1)], 1)
                q = self.channel_q(self.channel_norm_q(stat))
                memory = self.channel_norm_k(self.channel_bank.centers.detach())
                k, v = self.channel_k(memory), self.channel_v(memory)
                read = (q @ k.T / math.sqrt(self.dim)).softmax(-1) @ v
                # The correction head receives retrieved knowledge only;
                # no CT-only adapter shortcut can bypass the memory read.
                a, b = self.channel_out(read).chunk(2, 1)
                # Bounded additive affine residual; not a new PET feature.
                residual = residual + .25 * a.tanh()[:, :, None, None] * x.float()
                residual = residual + .25 * b.tanh()[:, :, None, None]
            result = x + residual.to(x.dtype)
            if not torch.isfinite(result).all():
                raise RuntimeError('nonfinite decoder prototype correction')
            return result


class DecoderDualPrototype(nn.Module):
    """Four independent decoder-scale libraries ordered [D4,D3,D2,D1]."""

    def __init__(self, channels: Sequence[int], dim=32, clusters=6, capacity=256,
                 min_candidates=24, decay=.999, refresh_steps=20, warmup_steps=100,
                 spatial=True, channel=True, outlier_filter=False):
        super().__init__()
        if len(channels) != 4 or warmup_steps < 0:
            raise ValueError('require four decoder levels and nonnegative warmup')
        self.warmup_steps = int(warmup_steps)
        self.scales = nn.ModuleList(DecoderPrototypeScale(
            c, dim, clusters, capacity, min_candidates, decay, refresh_steps,
            spatial, channel, seed=2023 + i, outlier_filter=outlier_filter)
            for i, c in enumerate(channels))
        self.register_buffer('update_steps', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def commit(self, candidates: Sequence[tuple[Tensor, Tensor]]) -> None:
        if not self.training:
            raise RuntimeError('memory writes are forbidden in eval mode')
        if len(candidates) != len(self.scales):
            raise ValueError('expected four scales of candidates')
        self.update_steps.add_(1)
        if int(self.update_steps) <= self.warmup_steps:
            return
        for scale, (s, c) in zip(self.scales, candidates):
            for bank, values, enabled in ((scale.spatial_bank, s, scale.spatial_enabled),
                                          (scale.channel_bank, c, scale.channel_enabled)):
                if not enabled or values.numel() == 0:
                    continue
                bank.add(values)
                if not bool(bank.valid) or int(self.update_steps) % scale.refresh_steps == 0:
                    bank.refresh()

    def diagnostics(self) -> dict:
        stats = {'successful_steps': int(self.update_steps),
                 'outlier_filter_enabled': self.scales[0].spatial_bank.outlier_filter}
        for i, scale in enumerate(self.scales):
            for label, bank in (('spatial', scale.spatial_bank), ('channel', scale.channel_bank)):
                stats[f'd{4-i}_{label}_valid'] = bool(bank.valid)
                stats[f'd{4-i}_{label}_candidates'] = int(bank.accepted)
                stats[f'd{4-i}_{label}_updates'] = int(bank.updates)
        return stats
