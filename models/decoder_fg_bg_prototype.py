"""GT-selected FG/BG local Full memories, queried only by Missing features.

Sources and queries use the same fixed orthogonal basis. Each scale owns two
Euclidean centroid FIFOs; no Gram/channel memory, trainable centers, PET
reconstruction or label-dependent inference. Candidate payloads are
(foreground [N,3d], background [M,3d], selected_pixel_counts [2]).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.decoder_dual_prototype import EMACentroidBank, DecoderPrototypeScale


class ForegroundBackgroundScale(nn.Module):
    def __init__(self, channels: int, dim: int = 32, clusters: int = 4,
                 capacity: int = 256, min_candidates: int = 24,
                 decay: float = .999, refresh_steps: int = 20,
                 patch_size: int = 5, max_patches: int = 16, seed: int = 2023,
                 adaptive_sampling: bool = True, sampling_decay: float = .99,
                 logit_scale: float = 2., correction_alpha: float = .1):
        super().__init__()
        if channels < 1 or dim < 1 or refresh_steps < 1 or patch_size < 1 or max_patches < 2:
            raise ValueError('invalid FG/BG dimensions, cadence or patch budget')
        if not math.isfinite(logit_scale) or not 0 < logit_scale <= 100:
            raise ValueError('logit_scale must be in (0,100]')
        if not math.isfinite(correction_alpha) or not 0 < correction_alpha <= 1:
            raise ValueError('correction_alpha must be in (0,1]')
        if not 0 <= sampling_decay < 1:
            raise ValueError('sampling_decay must be in [0,1)')
        self.channels, self.dim = int(channels), min(int(dim), int(channels))
        self.refresh_steps, self.patch_size, self.max_patches = int(refresh_steps), int(patch_size), int(max_patches)
        self.adaptive_sampling, self.sampling_decay = bool(adaptive_sampling), float(sampling_decay)
        self.logit_scale, self.correction_alpha = float(logit_scale), float(correction_alpha)
        gen = torch.Generator().manual_seed(seed)
        basis = torch.linalg.qr(torch.randn(channels, self.dim, generator=gen), mode='reduced').Q.T
        self.register_buffer('basis', basis.contiguous())
        self.foreground_bank = EMACentroidBank(clusters, 3 * self.dim, capacity, min_candidates, decay)
        self.background_bank = EMACentroidBank(clusters, 3 * self.dim, capacity, min_candidates, decay)
        # Sampling history is model state, NOT logging metadata. Only successful
        # commits change it; checkpoints/EMA copy it with the other buffers.
        self.register_buffer('sampling_mass', torch.zeros(2))
        self.register_buffer('sampling_updates', torch.zeros((), dtype=torch.long))
        self.query_norm = nn.LayerNorm(3 * self.dim)
        self.memory_norm = nn.LayerNorm(3 * self.dim)
        self.q = nn.Linear(3 * self.dim, self.dim, bias=False)
        self.k = nn.Linear(3 * self.dim, self.dim, bias=False)
        self.v = nn.Linear(3 * self.dim, self.dim, bias=False)
        self.gate = nn.Conv2d(channels + self.dim, 1, 3, padding=1)
        self.out = nn.Conv2d(self.dim, channels, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    @property
    def banks(self):
        return self.foreground_bank, self.background_bank

    def _project(self, x: Tensor) -> Tensor:
        return torch.einsum('dc,bchw->bdhw', self.basis.float(), x.float())

    _local = staticmethod(DecoderPrototypeScale._local)

    def sampling_probability(self, fallback: Tensor | None = None) -> Tensor:
        mass = self.sampling_mass if int(self.sampling_updates) or fallback is None else fallback
        # +1 pixel prevents singular weights, not a class-specific gain factor.
        priority = (mass.float() + 1.).rsqrt() if self.adaptive_sampling else torch.ones_like(mass)
        return priority / priority.sum()

    @torch.no_grad()
    def candidates(self, full: Tensor, reference: Tensor, weight: Tensor, target: Tensor,
                   stats: dict | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if full.shape != reference.shape or full.ndim != 4 or full.shape[1] != self.channels:
            raise ValueError('Full/reference features must match [B,C,H,W]')
        if target.ndim == 3:
            target = target[:, None]
        if target.shape != weight.shape or weight.ndim != 4 or weight.shape[1] != 1:
            raise ValueError('GT and gain map must match [B,1,H,W]')
        if weight.shape[0] != full.shape[0] or not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError('invalid gain map')
        with torch.autocast(device_type=full.device.type, enabled=False):
            y, w = target.detach().ge(.5), weight.detach().float()
            selected = w > 0
            counts = torch.stack([(selected & y).sum(), (selected & ~y).sum()]).float()
            probability = self.sampling_probability(counts)
            quota_fg = max(1, min(self.max_patches - 1,
                                 int(torch.round(self.max_patches * probability[0]))))
            zf, z0 = self._project(full), self._project(reference)
            common_rms = ((zf.square().mean((1, 2, 3), keepdim=True)
                           + z0.square().mean((1, 2, 3), keepdim=True)) * .5 + 1e-6).sqrt()
            local = self._local(zf / common_rms)
            p = min(self.patch_size, *full.shape[-2:])
            scores, descriptors = [], []
            # Split BEFORE area reduction, so adjacent BG gain cannot enter FG.
            for region in (y, ~y):
                class_weight = F.interpolate(w * region.float(), size=full.shape[-2:], mode='area')
                score = F.avg_pool2d(class_weight, p, stride=p, ceil_mode=True, count_include_pad=False)
                desc = F.avg_pool2d(local * class_weight, p, stride=p, ceil_mode=True,
                                    count_include_pad=False) / score.clamp_min(1e-8)
                scores.append(score.flatten(1))
                descriptors.append(desc.flatten(2).transpose(1, 2))
            result = [[], []]
            budget_totals = [0, 0]
            available_totals = [0, 0]
            for b in range(full.shape[0]):
                indices = []
                for score in scores:
                    valid = (score[b] > 1e-6).nonzero(as_tuple=True)[0]
                    indices.append(valid[score[b, valid].argsort(descending=True)])
                available = [idx.numel() for idx in indices]
                if not available[0] or not available[1]:
                    take = [min(n, self.max_patches) for n in available]
                else:
                    take = [min(available[0], quota_fg), min(available[1], self.max_patches - quota_fg)]
                    remaining = self.max_patches - sum(take)
                    # Reallocate only to available unique candidates, never
                    # duplicate a rare candidate to satisfy a nominal quota.
                    for c in (0, 1):
                        extra = min(remaining, available[c] - take[c])
                        take[c] += extra; remaining -= extra
                for c in (0, 1):
                    budget_totals[c] += take[c]
                    available_totals[c] += available[c]
                    if take[c]: result[c].append(descriptors[c][b, indices[c][:take[c]]])
            values = [torch.cat(parts).detach() if parts else full.new_empty((0, 3 * self.dim), dtype=torch.float32)
                      for parts in result]
            if stats is not None:
                for c, label in enumerate(('foreground', 'background')):
                    stats[f'{label}_generated'] = values[c].shape[0]
                    stats[f'{label}_available_patches'] = available_totals[c]
                    stats[f'{label}_allocated_patches'] = budget_totals[c]
                stats['sampling_foreground_probability'] = probability[0].detach()
            return values[0], values[1], counts.detach()

    @torch.no_grad()
    def observe(self, counts: Tensor):
        if counts.shape != (2,) or not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError('selected-pixel counts must be finite nonnegative [2]')
        counts = counts.to(self.sampling_mass)
        if not bool((counts > 0).any()):
            return
        if not int(self.sampling_updates):
            self.sampling_mass.copy_(counts)
        else:
            self.sampling_mass.mul_(self.sampling_decay).add_(counts, alpha=1 - self.sampling_decay)
        self.sampling_updates.add_(1)

    def forward(self, x: Tensor, diagnostics: dict | None = None) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'expected [B,{self.channels},H,W]')
        ready = [bool(bank.valid) for bank in self.banks]
        if diagnostics is not None:
            diagnostics.update({label: dict(available=valid) for label, valid in
                                zip(('foreground', 'background'), ready)})
        if not any(ready): return x
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self._project(x)
            z = z / (z.square().mean((1, 2, 3), keepdim=True) + 1e-6).sqrt()
            q = self.q(self.query_norm(self._local(z).flatten(2).transpose(1, 2)))
            centers = torch.cat([bank.centers.detach() for bank, valid in zip(self.banks, ready) if valid])
            memory = self.memory_norm(centers)
            k, v = self.k(memory), self.v(memory)
            logits = self.logit_scale * (F.normalize(q, dim=-1, eps=1e-8)
                                        @ F.normalize(k, dim=-1, eps=1e-8).T)
            attention = logits.softmax(-1)
            read = (attention @ v).transpose(1, 2).reshape(x.shape[0], self.dim, *x.shape[-2:])
            gate = self.gate(torch.cat([x.float(), read], 1)).sigmoid()
            raw = gate * self.out(read)
            budget = self.correction_alpha * x.float().square().mean((1, 2, 3), keepdim=True).sqrt().detach()
            magnitude = raw.square().mean((1, 2, 3), keepdim=True).sqrt().detach()
            correction = raw * (budget / (budget + magnitude).clamp_min(1e-12))
            if diagnostics is not None:
                joint = DecoderPrototypeScale._read_diagnostics(attention, correction, x)
                joint.update(DecoderPrototypeScale._attention_diagnostics(q, k, logits))
                joint['gate_mean'] = gate.detach().mean()
                diagnostics['joint'] = joint
                start = 0
                for label, bank, valid in zip(('foreground', 'background'), self.banks, ready):
                    if valid:
                        stop = start + bank.clusters
                        mean, top1 = joint['attention_mean'][start:stop], joint['top1_fraction'][start:stop]
                        diagnostics[label].update(attention_mean=mean, top1_fraction=top1,
                                                  class_mass=mean.sum())
                        start = stop
            result = x + correction.to(x.dtype)
            if not torch.isfinite(result).all(): raise RuntimeError('nonfinite FG/BG correction')
            return result


class ForegroundBackgroundMemory(nn.Module):
    def __init__(self, channels: Sequence[int], warmup_steps: int = 100, **scale_kwargs):
        super().__init__()
        if len(channels) != 4 or warmup_steps < 0:
            raise ValueError('require four decoder levels and nonnegative warmup')
        self.warmup_steps = int(warmup_steps)
        self.scales = nn.ModuleList(ForegroundBackgroundScale(c, seed=2023 + i, **scale_kwargs)
                                    for i, c in enumerate(channels))
        self.register_buffer('update_steps', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def commit(self, candidates: Sequence[tuple[Tensor, Tensor, Tensor]]):
        if not self.training: raise RuntimeError('memory writes forbidden in eval')
        if len(candidates) != 4: raise ValueError('expected four candidate payloads')
        self.update_steps.add_(1)
        for scale, (fg, bg, counts) in zip(self.scales, candidates):
            scale.observe(counts)
            if int(self.update_steps) <= self.warmup_steps: continue
            for bank, values in zip(scale.banks, (fg, bg)):
                if not values.numel(): continue
                bank.add(values)
                if not bool(bank.valid) or int(self.update_steps) % scale.refresh_steps == 0:
                    bank.refresh()

    def diagnostics(self):
        result = dict(successful_steps=int(self.update_steps))
        for i, scale in enumerate(self.scales):
            for label, bank in zip(('foreground', 'background'), scale.banks):
                result[f'd{4-i}_{label}_valid'] = bool(bank.valid)
                result[f'd{4-i}_{label}_candidates'] = int(bank.accepted)
                result[f'd{4-i}_{label}_updates'] = int(bank.updates)
        return result
