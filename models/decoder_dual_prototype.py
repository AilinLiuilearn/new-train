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
        # Logging metadata only; excluded from tensor state_dict so old weights
        # load strictly. The task checkpoints this separately when available.
        self.refresh_stats = dict(attempts=0, insufficient_pool=0, insufficient_unique=0,
                                  empty_cluster=0, last_status='never_attempted')

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
        self.refresh_stats['attempts'] += 1
        if int(self.size) < self.min_candidates:
            self.refresh_stats['insufficient_pool'] += 1
            self.refresh_stats['last_status'] = 'insufficient_pool'
            return False
        # Disable ambient AMP: Gram/K-means/EMA stay float32 in formal AMP runs.
        with torch.autocast(device_type=self.pool.device.type, enabled=False):
            x = self.pool[:int(self.size)].float()
            if torch.unique(x, dim=0).shape[0] < self.clusters:
                self.refresh_stats['insufficient_unique'] += 1
                self.refresh_stats['last_status'] = 'insufficient_unique'
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
            iterations_run = 0
            for _ in range(self.iterations):
                iterations_run += 1
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
            retained = 0
            for k in range(self.clusters):
                members = x[assignment == k]
                if members.shape[0] == 0:
                    # Do not mark a collapsed bank as a valid K-center library.
                    self.refresh_stats['empty_cluster'] += 1
                    self.refresh_stats['last_status'] = 'empty_cluster'
                    return False
                if self.outlier_filter:
                    distance = (members - centers[k]).square().sum(1).sqrt()
                    median = distance.median()
                    mad = (distance - median).abs().median()
                    keep = distance <= median + self.outlier_mad * 1.4826 * mad + 1e-6
                    members = members[keep]
                retained += members.shape[0]
                centers[k] = members.mean(0)
            counts = torch.bincount(assignment, minlength=self.clusters)
            initialized = not bool(self.valid)
            raw_drift = applied_drift = None
            if bool(self.valid):
                distances = self._distance(self.centers.float(), centers)
                ordered = torch.empty_like(centers)
                ordered_counts = torch.empty_like(counts)
                for _ in range(self.clusters):
                    flat = int(distances.argmin())
                    old, new = divmod(flat, self.clusters)
                    ordered[old] = centers[new]
                    ordered_counts[old] = counts[new]
                    distances[old, :] = float('inf')
                    distances[:, new] = float('inf')
                raw_drift = float((ordered - self.centers.float()).norm(dim=1).mean())
                old_centers = self.centers.clone()
                self.centers.mul_(self.decay).add_(ordered, alpha=1 - self.decay)
                applied_drift = float((self.centers - old_centers).norm(dim=1).mean())
                counts = ordered_counts
            else:
                self.centers.copy_(centers)
                self.valid.fill_(True)
            self.updates.add_(1)
            self.refresh_stats.update(last_status='initialized' if initialized else 'updated',
                                      last_iterations=iterations_run,
                                      last_pool_size=int(x.shape[0]),
                                      last_retained=retained,
                                      last_cluster_counts=counts.cpu().tolist(),
                                      last_raw_drift=raw_drift,
                                      last_applied_drift=applied_drift)
        return True


def gain_weights(full_logits: Tensor, reference_logits: Tensor, target: Tensor,
                 foreground_weight: float = 2., background_weight: float = 1.,
                 min_gain: float = .01, stats: dict | None = None) -> Tensor:
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
        weight = improvement.clamp_min(0) * reliable * (improvement >= min_gain) * priority
        if stats is not None:
            # Additive detached reductions, transferred together by the task.
            selected = weight > 0
            stats['full_images'] = pf.shape[0]
            stats['useful_full_images'] = selected.flatten(1).any(1).sum()
            for label, region in (('fg', y.bool()), ('bg', ~y.bool())):
                passed = selected & region
                stats[f'{label}_pixels'] = region.sum()
                stats[f'{label}_selected'] = passed.sum()
                stats[f'{label}_full_wrong'] = (region & ~reliable).sum()
                stats[f'{label}_below_gain'] = (region & reliable & (improvement < min_gain)).sum()
                stats[f'{label}_gain_sum'] = (improvement * passed).sum()
                stats[f'{label}_weight_sum'] = (weight * passed).sum()
            for label, probability in (('full', pf), ('reference', p0)):
                prediction = probability >= .5
                stats[f'{label}_fn'] = (~prediction & y.bool()).sum()
                stats[f'{label}_fp'] = (prediction & ~y.bool()).sum()
        return weight


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
                 outlier_filter: bool = False, channel_qk_norm: bool = True,
                 channel_logit_scale: float = 2., local_correction: bool = True,
                 correction_alpha: float = .1):
        super().__init__()
        if not spatial and not channel:
            raise ValueError('at least one prototype branch must be enabled')
        if channels < 1 or dim < 1 or refresh_steps < 1 or patch_size < 1 or max_patches < 1:
            raise ValueError('invalid prototype dimensions or collection cadence')
        self.channels, self.dim = int(channels), min(int(dim), int(channels))
        if not math.isfinite(channel_logit_scale) or channel_logit_scale <= 0:
            raise ValueError('channel_logit_scale must be finite and positive')
        if not math.isfinite(correction_alpha) or not 0 < correction_alpha <= 1:
            raise ValueError('correction_alpha must be in (0,1]')
        self.channel_qk_norm = bool(channel_qk_norm)
        self.channel_logit_scale = float(channel_logit_scale)
        self.local_correction = bool(local_correction)
        self.correction_alpha = float(correction_alpha)
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
        # Only the new mode adds parameters; legacy checkpoints load strictly
        # with local_correction=False. The head still depends on memory alone.
        if self.local_correction:
            self.channel_gate = nn.Conv2d(channels + d, 1, 3, padding=1)
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
                   foreground: Tensor | None = None, stats: dict | None = None) -> tuple[Tensor, Tensor]:
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
            fg_count = bg_count = 0
            if self.spatial_enabled:
                local = self._local(zf)
                p = min(self.patch_size, *full.shape[-2:])
                wm = F.avg_pool2d(w, p, stride=p, ceil_mode=True, count_include_pad=False)
                pooled = F.avg_pool2d(local * w, p, stride=p, ceil_mode=True,
                                      count_include_pad=False) / wm.clamp_min(1e-8)
                # Multiply at the original gain-map resolution FIRST. Separate
                # downsampling can turn adjacent background gain into FG gain.
                if foreground is not None:
                    fg = foreground.detach().float()
                    if fg.ndim == 3:
                        fg = fg[:, None]
                    if fg.shape != weight.shape:
                        raise ValueError('foreground must match the original gain-map shape')
                    fg_gain = F.interpolate(fg.ge(.5).float() * weight.detach().float(),
                                            size=full.shape[-2:], mode='area')
                else:
                    fg_gain = w
                foreground_mass = F.avg_pool2d(fg_gain, p, stride=p, ceil_mode=True,
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
                    fg_count += fg_idx.numel()
                    bg_count += bg_idx.numel()
                    if idx.numel():
                        spatial.append(pooled[b].flatten(1).T[idx])
            s = torch.cat(spatial) if spatial else full.new_empty((0, 3 * self.dim), dtype=torch.float32)
            if stats is not None:
                stats.update(spatial_generated=s.shape[0], channel_generated=relations.shape[0],
                             spatial_fg_patches=fg_count, spatial_bg_patches=bg_count)
            return s.detach(), relations.detach()

    def _channel_attention(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        if self.channel_qk_norm:
            # Normalize AFTER learned projections; fixed scale cannot grow
            # with Q/K norms. No additional sqrt(d) division in cosine mode.
            logits = self.channel_logit_scale * (F.normalize(q, dim=-1, eps=1e-8)
                                                @ F.normalize(k, dim=-1, eps=1e-8).T)
        else:
            logits = q @ k.T / math.sqrt(self.dim)
        return logits.softmax(-1), logits

    @staticmethod
    def _rms(x: Tensor) -> Tensor:
        return x.float().square().mean((1, 2, 3), keepdim=True).sqrt()

    def _limit_spatial(self, residual: Tensor, x: Tensor) -> Tensor:
        # Smooth per-image bound, preserving initial head gradient. Detach the
        # magnitude controller so the head cannot exploit its denominator.
        budget = self.correction_alpha * self._rms(x).detach()
        magnitude = self._rms(residual).detach()
        factor = budget / (budget + magnitude).clamp_min(1e-12)
        return residual * factor

    @staticmethod
    @torch.no_grad()
    def _attention_diagnostics(q: Tensor, k: Tensor, logits: Tensor) -> dict:
        return dict(q_norm_mean=q.detach().norm(dim=-1).mean(),
                    k_norm_mean=k.detach().norm(dim=-1).mean(),
                    logit_min=logits.detach().min(), logit_max=logits.detach().max(),
                    logit_span_mean=(logits.detach().max(-1).values
                                     - logits.detach().min(-1).values).mean())

    @staticmethod
    @torch.no_grad()
    def _read_diagnostics(attention, correction, x, shift=None):
        probabilities = attention.detach().reshape(-1, attention.shape[-1])
        correction = correction.detach()
        squared_norm = correction.norm().square()
        if shift is not None:
            # ||scale*x + shift|| without materializing another full feature
            # map (important for the high-resolution decoder on small GPUs).
            shift = shift.detach()
            squared_norm = (squared_norm + 2 * (correction.sum((2, 3), keepdim=True) * shift).sum()
                            + correction.shape[-2] * correction.shape[-1] * shift.norm().square())
        return dict(query_count=probabilities.shape[0],
                    attention_entropy=(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
                                       / math.log(probabilities.shape[-1])).mean(),
                    attention_mean=probabilities.mean(0),
                    top1_fraction=torch.bincount(probabilities.argmax(1), minlength=probabilities.shape[-1])
                                  / probabilities.shape[0],
                    correction_rms_ratio=squared_norm.clamp_min(0).sqrt()
                                         / x.detach().float().norm().clamp_min(1e-8))

    def forward(self, x: Tensor, diagnostics: dict | None = None) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'expected decoder [B,{self.channels},H,W] feature')
        use_s = self.spatial_enabled and bool(self.spatial_bank.valid)
        use_c = self.channel_enabled and bool(self.channel_bank.valid)
        if diagnostics is not None:
            diagnostics.update(spatial=dict(available=use_s), channel=dict(available=use_c))
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
                spatial_residual = gate * self.spatial_out(read)
                if self.local_correction:
                    spatial_residual = self._limit_spatial(spatial_residual, x)
                residual = residual + spatial_residual
                if diagnostics is not None:
                    diagnostics['spatial'].update(self._read_diagnostics(attn, spatial_residual, x))
                    diagnostics['spatial']['gate_mean'] = gate.detach().mean()
            if use_c:
                stat = torch.cat([z.mean((2, 3)), self._gram(z, torch.ones_like(z[:, :1])).flatten(1)], 1)
                q = self.channel_q(self.channel_norm_q(stat))
                memory = self.channel_norm_k(self.channel_bank.centers.detach())
                k, v = self.channel_k(memory), self.channel_v(memory)
                attn, logits = self._channel_attention(q, k)
                read = attn @ v
                # The correction head receives retrieved knowledge only;
                # no CT-only adapter shortcut can bypass the memory read.
                a, b = self.channel_out(read).chunk(2, 1)
                if self.local_correction:
                    context = read[:, :, None, None].expand(-1, -1, *x.shape[-2:])
                    gate = self.channel_gate(torch.cat([x.float(), context], 1)).sigmoid()
                    # Triangle inequality gives per-image RMS(delta)/RMS(x)
                    # <= alpha; neither a global absolute shift nor hard mask.
                    amplitude = self._rms(x).detach()
                    channel_residual = (.5 * self.correction_alpha * gate *
                                        (a.tanh()[:, :, None, None] * x.float()
                                         + amplitude * b.tanh()[:, :, None, None]))
                    residual = residual + channel_residual
                else:
                    # Exact previous affine design for controlled comparisons.
                    scale_residual = .25 * a.tanh()[:, :, None, None] * x.float()
                    shift_residual = .25 * b.tanh()[:, :, None, None]
                    residual = residual + scale_residual + shift_residual
                if diagnostics is not None:
                    if self.local_correction:
                        diagnostics['channel'].update(self._read_diagnostics(attn, channel_residual, x))
                        diagnostics['channel']['gate_mean'] = gate.detach().mean()
                    else:
                        diagnostics['channel'].update(self._read_diagnostics(attn, scale_residual, x, shift_residual))
                    diagnostics['channel'].update(self._attention_diagnostics(q, k, logits))
            if diagnostics is not None:
                diagnostics['total_correction_rms_ratio'] = (residual.detach().norm()
                                                            / x.detach().float().norm().clamp_min(1e-8))
            result = x + residual.to(x.dtype)
            if not torch.isfinite(result).all():
                raise RuntimeError('nonfinite decoder prototype correction')
            return result


class DecoderDualPrototype(nn.Module):
    """Four independent decoder-scale libraries ordered [D4,D3,D2,D1]."""

    def __init__(self, channels: Sequence[int], dim=32, clusters=6, capacity=256,
                 min_candidates=24, decay=.999, refresh_steps=20, warmup_steps=100,
                 spatial=True, channel=True, outlier_filter=False,
                 channel_qk_norm=True, channel_logit_scale=2., local_correction=True,
                 correction_alpha=.1):
        super().__init__()
        if len(channels) != 4 or warmup_steps < 0:
            raise ValueError('require four decoder levels and nonnegative warmup')
        self.warmup_steps = int(warmup_steps)
        self.scales = nn.ModuleList(DecoderPrototypeScale(
            c, dim, clusters, capacity, min_candidates, decay, refresh_steps,
            spatial, channel, seed=2023 + i, outlier_filter=outlier_filter,
            channel_qk_norm=channel_qk_norm, channel_logit_scale=channel_logit_scale,
            local_correction=local_correction, correction_alpha=correction_alpha)
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
