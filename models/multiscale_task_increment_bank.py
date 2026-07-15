# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pg_mtr import _valid_group_count

__all__ = [
    'StageTaskIncrementQuery',
    'SharedMultiScaleTaskIncrementBank',
]


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean().add(1e-12).sqrt()


def _entropy(a: torch.Tensor) -> torch.Tensor:
    p = a.float().clamp_min(1e-8)
    return (-(p * p.log()).sum(dim=1).mean()) / math.log(float(a.shape[1]))


def _peak(a: torch.Tensor) -> torch.Tensor:
    return a.float().max(dim=1).values.mean()


def _pairwise_cosine(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.shape[0] <= 1:
        return tokens.new_tensor(0.0)
    t = F.normalize(tokens.float(), dim=-1, eps=1e-6)
    sim = t @ t.t()
    return (sim.sum() - sim.diag().sum()) / float(tokens.shape[0] * (tokens.shape[0] - 1))


class StageTaskIncrementQuery(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent_dim = int(latent_dim)
        self.proj = nn.Sequential(
            nn.GroupNorm(_valid_group_count(self.in_channels), self.in_channels, affine=True),
            nn.Conv2d(self.in_channels, self.latent_dim, kernel_size=1, bias=False),
        )
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SharedMultiScaleTaskIncrementBank(nn.Module):
    def __init__(
        self,
        channels_list: Sequence[int],
        num_tokens: int = 8,
        temperature: float = 0.07,
        stage_mode: str = 'all',
    ):
        super().__init__()
        self.channels_list = [int(c) for c in channels_list]
        self.num_tokens = int(num_tokens)
        self.temperature = float(temperature)
        self.stage_mode = str(stage_mode)
        modes = {'s4': (4,), 's34': (3, 4), 'deep': (3, 4), 's234': (2, 3, 4), 'all': (1, 2, 3, 4)}
        if self.stage_mode not in modes:
            raise ValueError(f'Unsupported mtib_stages={stage_mode!r}')
        self.active_stage_numbers = modes[self.stage_mode]
        deepest = self.active_stage_numbers[-1] - 1
        self.latent_dim = min(max(self.channels_list[deepest] // 4, 32), 128)

        self.full_queries = nn.ModuleDict({str(s): StageTaskIncrementQuery(self.channels_list[s - 1], self.latent_dim) for s in self.active_stage_numbers})
        self.ct_queries = nn.ModuleDict({str(s): StageTaskIncrementQuery(self.channels_list[s - 1], self.latent_dim) for s in self.active_stage_numbers})
        self.shared_bank_tokens = nn.Parameter(torch.empty(self.num_tokens, self.latent_dim))
        self.bank_key = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.bank_value = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

        self.retrieval_adapters = nn.ModuleDict()
        self.gamma = nn.ParameterDict()
        for s in self.active_stage_numbers:
            ch = self.channels_list[s - 1]
            adapter = nn.Sequential(
                nn.Conv2d(self.latent_dim, ch, kernel_size=1, bias=False),
                nn.GroupNorm(_valid_group_count(ch), ch, affine=False),
            )
            nn.init.kaiming_normal_(adapter[0].weight, mode='fan_out', nonlinearity='linear')
            self.retrieval_adapters[str(s)] = adapter
            self.gamma[str(s)] = nn.Parameter(torch.tensor(0.01).log())

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.shared_bank_tokens, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.bank_key.weight)
        nn.init.xavier_uniform_(self.bank_value.weight)

    def _kv(self, detach_bank: bool = False):
        k = self.bank_key(self.shared_bank_tokens)
        v = self.bank_value(self.shared_bank_tokens)
        if detach_bank:
            k = k.detach()
            v = v.detach()
        return k, v

    def _assign(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q.float(), dim=1, eps=1e-6)
        k = F.normalize(k.float(), dim=-1, eps=1e-6)
        return torch.softmax(torch.einsum('bdhw,kd->bkhw', q, k) / self.temperature, dim=1)

    def _read(self, assignment: torch.Tensor, value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        retrieved = torch.einsum('bkhw,kd->bdhw', assignment.float(), value.float())
        return retrieved.to(dtype=dtype)

    def _route(self, q: torch.Tensor, detach_bank: bool = False):
        k, v = self._kv(detach_bank=detach_bank)
        assignment = self._assign(q, k)
        retrieved = self._read(assignment, v, q.dtype)
        return assignment, retrieved, k, v

    def _apply_adapter(self, stage: int, mem: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.gamma[str(stage)])
        return alpha * self.retrieval_adapters[str(stage)](mem)

    def forward_full(self, full_feats: Sequence[torch.Tensor], true_increment: Dict[int, torch.Tensor]):
        out = {}
        diag = {}
        losses = []
        for s in self.active_stage_numbers:
            i = s - 1
            q = self.full_queries[str(s)](full_feats[i].detach())
            assignment, mem, k, v = self._route(q, detach_bank=False)
            delta = self._apply_adapter(s, mem)
            target = true_increment[s].detach()
            loss = F.smooth_l1_loss(delta, target)
            losses.append(loss)
            diag.update({
                f'mtib_s{s}_full_route_entropy': _entropy(assignment).detach(),
                f'mtib_s{s}_full_route_peak': _peak(assignment).detach(),
                f'mtib_s{s}_full_retrieved_rms': _rms(mem).detach(),
                f'mtib_s{s}_bank_recon_rms': _rms(delta).detach(),
                f'mtib_s{s}_bank_recon_target_ratio': _rms(delta).detach() / (_rms(target) + 1e-6),
                f'mtib_s{s}_bank_loss': loss.detach(),
                f'mtib_s{s}_gamma': F.softplus(self.gamma[str(s)]).detach(),
            })
            out[s] = {'assignment': assignment, 'retrieved': mem, 'delta': delta, 'target': target}
        diag['mtib_token_rms'] = _rms(self.shared_bank_tokens).detach()
        diag['mtib_key_pairwise_cosine'] = _pairwise_cosine(self.bank_key(self.shared_bank_tokens)).detach()
        diag['mtib_value_pairwise_cosine'] = _pairwise_cosine(self.bank_value(self.shared_bank_tokens)).detach()
        return out, torch.stack(losses).mean(), diag

    def forward_ct_comp(self, ct_feats: Sequence[torch.Tensor], true_increment: Dict[int, torch.Tensor]):
        out = {}
        diag = {}
        losses = []
        for s in self.active_stage_numbers:
            i = s - 1
            q = self.ct_queries[str(s)](ct_feats[i].detach())
            assignment, mem, _, _ = self._route(q, detach_bank=True)
            delta = self._apply_adapter(s, mem)
            target = true_increment[s].detach()
            loss = F.smooth_l1_loss(delta, target)
            losses.append(loss)
            diag.update({
                f'mtib_s{s}_ct_route_entropy': _entropy(assignment).detach(),
                f'mtib_s{s}_ct_route_peak': _peak(assignment).detach(),
                f'mtib_s{s}_ct_retrieved_rms': _rms(mem).detach(),
                f'mtib_s{s}_ct_comp_rms': _rms(delta).detach(),
                f'mtib_s{s}_ct_comp_target_ratio': _rms(delta).detach() / (_rms(target) + 1e-6),
                f'mtib_s{s}_comp_loss': loss.detach(),
                f'mtib_s{s}_gamma': F.softplus(self.gamma[str(s)]).detach(),
            })
            out[s] = {'assignment': assignment, 'retrieved': mem, 'delta': delta, 'target': target}
        diag['mtib_token_rms'] = _rms(self.shared_bank_tokens).detach()
        diag['mtib_key_pairwise_cosine'] = _pairwise_cosine(self.bank_key(self.shared_bank_tokens)).detach()
        diag['mtib_value_pairwise_cosine'] = _pairwise_cosine(self.bank_value(self.shared_bank_tokens)).detach()
        return out, torch.stack(losses).mean(), diag

    def forward_missing(self, ct_feats: Sequence[torch.Tensor]):
        out = {}
        diag = {}
        for s in self.active_stage_numbers:
            i = s - 1
            q = self.ct_queries[str(s)](ct_feats[i].detach())
            assignment, mem, _, _ = self._route(q, detach_bank=True)
            delta = self._apply_adapter(s, mem)
            out[s] = delta
            diag.update({
                f'mtib_s{s}_missing_route_entropy': _entropy(assignment).detach(),
                f'mtib_s{s}_missing_route_peak': _peak(assignment).detach(),
                f'mtib_s{s}_missing_retrieved_rms': _rms(mem).detach(),
                f'mtib_s{s}_missing_injection_rms': _rms(delta).detach(),
                f'mtib_s{s}_gamma': F.softplus(self.gamma[str(s)]).detach(),
            })
        diag['mtib_token_rms'] = _rms(self.shared_bank_tokens).detach()
        diag['mtib_key_pairwise_cosine'] = _pairwise_cosine(self.bank_key(self.shared_bank_tokens)).detach()
        diag['mtib_value_pairwise_cosine'] = _pairwise_cosine(self.bank_value(self.shared_bank_tokens)).detach()
        return out, diag
