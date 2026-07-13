# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["StagePETGroundedMetabolicTokenRetrieval", "PETGroundedMetabolicTokenRetrieval"]


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    for g in range(min(channels, max_groups), 0, -1):
        if channels % g == 0:
            return g
    return 1


def _rms(x):
    return x.float().pow(2).mean().add(1e-12).sqrt()


def _entropy(a):
    p = a.float().clamp_min(1e-8)
    return (-(p * p.log()).sum(dim=1).mean()) / math.log(float(a.shape[1]))


def _peak(a):
    return a.float().max(dim=1).values.mean()


def _token_cos(tokens):
    if tokens.shape[0] <= 1:
        return tokens.new_tensor(0.0)
    t = F.normalize(tokens.float(), dim=-1, eps=1e-6)
    sim = t @ t.t()
    return (sim.sum() - sim.diag().sum()) / float(tokens.shape[0] * (tokens.shape[0] - 1))


def _sample_balanced(loss_map: torch.Tensor, mask: torch.Tensor):
    mask = mask.float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    fg = mask > 0.5
    bg = ~fg
    eps = 1e-6
    B = loss_map.shape[0]
    flat = loss_map.flatten(1)
    fg_flat = fg.flatten(1)
    bg_flat = bg.flatten(1)
    fg_sum = (flat * fg_flat.float()).sum(1)
    bg_sum = (flat * bg_flat.float()).sum(1)
    fg_den = fg_flat.float().sum(1).clamp_min(eps)
    bg_den = bg_flat.float().sum(1).clamp_min(eps)
    fg_mean = fg_sum / fg_den
    bg_mean = bg_sum / bg_den
    has_fg = fg_flat.any(1)
    has_bg = bg_flat.any(1)
    sample = torch.where(has_fg & has_bg, 0.5 * (fg_mean + bg_mean), torch.where(has_fg, fg_mean, bg_mean))
    return sample.mean(), fg_mean.mean(), bg_mean.mean(), has_fg.float().mean(), mask.mean()


class StagePETGroundedMetabolicTokenRetrieval(nn.Module):
    def __init__(self, in_channels, num_tokens=8, latent_dim=None, temperature=0.07):
        super().__init__(); self.in_channels=int(in_channels); self.num_tokens=int(num_tokens); self.temperature=float(temperature)
        self.latent_dim=int(latent_dim or min(max(self.in_channels//4,32),128))
        self.ct_query_proj=nn.Sequential(nn.GroupNorm(_valid_group_count(self.in_channels), self.in_channels), nn.Conv2d(self.in_channels, self.latent_dim, 1, bias=False))
        for m in self.ct_query_proj.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x): return self.ct_query_proj(x)


class PETGroundedMetabolicTokenRetrieval(nn.Module):
    def __init__(self, channels_list: Sequence[int], num_tokens=8, temperature=0.07, stage_mode='all'):
        super().__init__()
        ch = [int(c) for c in channels_list]
        self.channels_list = ch; self.num_tokens=int(num_tokens); self.temperature=float(temperature); self.stage_mode=str(stage_mode)
        modes = {'s4': (4,), 's34': (3,4), 'deep': (3,4), 's234': (2,3,4), 'all': (1,2,3,4)}
        if self.stage_mode not in modes: raise ValueError(stage_mode)
        self.active_stage_numbers = modes[self.stage_mode]
        self.active_stage_indices = tuple(i-1 for i in self.active_stage_numbers)
        self.writer_stage = max(self.active_stage_numbers)
        self.latent_dim = min(max(ch[self.writer_stage-1]//4, 32), 128)
        self.stage_modules = nn.ModuleDict({str(s): StagePETGroundedMetabolicTokenRetrieval(ch[s-1], num_tokens=self.num_tokens, latent_dim=self.latent_dim, temperature=self.temperature) for s in self.active_stage_numbers})
        self.shared_memory_tokens = nn.Parameter(torch.empty(self.num_tokens, self.latent_dim))
        self.shared_token_key = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.shared_token_value = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.pet_query_proj_writer = nn.Sequential(nn.GroupNorm(_valid_group_count(ch[self.writer_stage-1]), ch[self.writer_stage-1]), nn.Conv2d(ch[self.writer_stage-1], self.latent_dim, 1, bias=False))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.shared_memory_tokens, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.shared_token_key.weight); nn.init.xavier_uniform_(self.shared_token_value.weight)
        for m in self.pet_query_proj_writer.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')

    def _kv(self, detach=False):
        k = self.shared_token_key(self.shared_memory_tokens); v = self.shared_token_value(self.shared_memory_tokens)
        return (k.detach(), v.detach()) if detach else (k, v)

    def _assign(self, q, k):
        q = F.normalize(q.float(), dim=1, eps=1e-6); k = F.normalize(k.float(), dim=-1, eps=1e-6)
        return torch.softmax(torch.einsum('bdhw,kd->bkhw', q, k) / self.temperature, dim=1)

    def _read(self, a, v, dtype):
        return torch.einsum('bkhw,kd->bdhw', a.float(), v.float()).to(dtype=dtype)

    def _balanced_loss(self, loss_map, mask):
        return _sample_balanced(loss_map, mask)

    def _full_diag(self, writer, ct_a, pet_a, pet_mem):
        return {
            f'pg_mtr_s{writer}_ct_route_entropy': _entropy(ct_a).detach(),
            f'pg_mtr_s{writer}_ct_route_peak': _peak(ct_a).detach(),
            f'pg_mtr_s{writer}_pet_route_entropy': _entropy(pet_a).detach(),
            f'pg_mtr_s{writer}_pet_route_peak': _peak(pet_a).detach(),
            f'pg_mtr_s{writer}_pet_memory_rms': _rms(pet_mem).detach(),
            'pg_mtr_writer_stage': torch.tensor(float(writer), device=pet_mem.device),
            'pg_mtr_shared_token_key_cosine': _token_cos(self.shared_token_key(self.shared_memory_tokens)).detach(),
            'pg_mtr_shared_token_value_cosine': _token_cos(self.shared_token_value(self.shared_memory_tokens)).detach(),
        }

    def forward_full(self, ct_feats, pet_feats, mask=None):
        writer = self.writer_stage; wi = writer - 1
        ct_writer_feat = ct_feats[wi].detach(); pet_writer_feat = pet_feats[wi].detach()
        ct_q = self.stage_modules[str(writer)](ct_writer_feat)
        pet_q = self.pet_query_proj_writer(pet_writer_feat)
        k, v = self._kv(detach=False)
        ct_a = self._assign(ct_q, k); pet_a = self._assign(pet_q, k); pet_mem = self._read(pet_a, v, pet_q.dtype)
        if mask is None:
            return None, {'pg_mtr_route_loss': ct_q.new_tensor(0.0), 'pg_mtr_mem_loss': ct_q.new_tensor(0.0)}, self._full_diag(writer, ct_a, pet_a, pet_mem)
        if mask.ndim == 3: mask = mask.unsqueeze(1)
        mask = (F.adaptive_max_pool2d(mask.float(), ct_a.shape[-2:]) > 0.5).float()
        eps = 1e-8
        route_map = (pet_a.float().clamp_min(eps) * (pet_a.float().clamp_min(eps).log() - ct_a.float().clamp_min(eps).log())).sum(1, keepdim=True)
        mem_map = 1.0 - (F.normalize(pet_mem.float(), dim=1, eps=1e-6) * F.normalize(pet_q.float(), dim=1, eps=1e-6)).sum(1, keepdim=True)
        route_loss, fg_route, bg_route, fg_ratio, fg_pix = _sample_balanced(route_map, mask)
        mem_loss, fg_mem, bg_mem, _, _ = _sample_balanced(mem_map, mask)
        diag = self._full_diag(writer, ct_a, pet_a, pet_mem)
        diag.update({'writer_route_loss_fg': fg_route, 'writer_route_loss_bg': bg_route, 'writer_mem_loss_fg': fg_mem, 'writer_mem_loss_bg': bg_mem, 'writer_fg_valid_ratio': fg_ratio, 'writer_fg_pixel_ratio': fg_pix})
        return None, {'pg_mtr_route_loss': route_loss, 'pg_mtr_mem_loss': mem_loss}, diag

    def forward_missing(self, ct_feats, detach_bank=True):
        k, v = self._kv(detach=detach_bank)
        ret, diag = {}, {}
        for s in self.active_stage_numbers:
            i = s-1; q = self.stage_modules[str(s)](ct_feats[i])
            a = self._assign(q, k); mem = self._read(a, v, ct_feats[i].dtype)
            ret[s] = mem
            diag.update({f'pg_mtr_s{s}_ct_route_entropy': _entropy(a).detach(), f'pg_mtr_s{s}_ct_route_peak': _peak(a).detach(), f'pg_mtr_s{s}_retrieved_memory_rms': _rms(mem).detach()})
        diag['pg_mtr_writer_stage'] = torch.tensor(float(self.writer_stage), device=ct_feats[0].device)
        diag['pg_mtr_shared_token_key_cosine'] = _token_cos(k).detach(); diag['pg_mtr_shared_token_value_cosine'] = _token_cos(v).detach()
        return ret, {}, diag

    def forward(self, ct_feats, pet_feats=None, mode='missing', mask=None, detach_bank=True):
        mode = str(mode)
        if mode == 'full':
            if pet_feats is None: raise ValueError('mode full requires pet_feats')
            return self.forward_full(ct_feats, pet_feats, mask=mask)
        if mode == 'missing': return self.forward_missing(ct_feats, detach_bank=detach_bank)
        raise ValueError(mode)
