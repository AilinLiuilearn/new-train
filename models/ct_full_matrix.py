"""Independent CT-key / transform-value memories, before the shared decoder.

Contract: scales are S1..S4, each feature [B,C,H,W]. Training collection takes
paired raw CT/Full features and GT; inference takes CT only. Spherical routing
uses unit CT directions, while ridge fitting and application use RAW features.
Pools/keys/counters are checkpointed buffers. Analytic mode uses buffer values;
lowrank mode uses trainable I+AB values initialized once from paired Full data.
Missing segmentation loss trains the adoption gates and lowrank factors.

Borrowed principle: FedMEPD (arXiv:2603.04887), sec.3.5: GT class organization,
multi-anchor maintenance and EMA. Independent pixel-pair pools, spherical CT
clustering and identity-regularized transform values are OUR adaptation, not
FedMEPD's feature anchors. No cross-scale assignments, FAISS or reconstruction.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MatrixBank(nn.Module):
    def __init__(self, channels: int, clusters: int = 4, capacity: int = 256,
                 min_candidates: int = 24, min_cluster_samples: int = 2,
                 ridge: float = .01, decay: float = .99, iterations: int = 15,
                 outlier_filter: bool = False):
        super().__init__()
        if not (channels > 0 and clusters > 1 and min_cluster_samples >= 2
                and clusters * min_cluster_samples <= min_candidates <= capacity):
            raise ValueError('require C>0, K>1 and K*min_cluster_samples <= min_candidates <= capacity')
        if not math.isfinite(ridge) or ridge <= 0 or not 0 <= decay < 1 or iterations < 1:
            raise ValueError('ridge must be positive; decay in [0,1); iterations >= 1')
        self.channels, self.clusters, self.capacity = channels, clusters, capacity
        self.min_candidates, self.min_cluster_samples = min_candidates, min_cluster_samples
        self.ridge, self.decay, self.iterations = float(ridge), float(decay), int(iterations)
        self.outlier_filter = bool(outlier_filter)
        for name in ('pool_ct', 'pool_full'):
            self.register_buffer(name, torch.zeros(capacity, channels))
        self.register_buffer('pool_weight', torch.zeros(capacity))
        self.register_buffer('keys', torch.zeros(clusters, channels))
        self.register_buffer('transforms', torch.eye(channels).repeat(clusters, 1, 1))
        self.register_buffer('support', torch.zeros(clusters, dtype=torch.long))
        self.register_buffer('fit_error', torch.zeros(clusters))
        self.register_buffer('identity_error', torch.zeros(clusters))
        self.register_buffer('ridge_lambda', torch.zeros(clusters))
        self.register_buffer('matrix_drift', torch.zeros(clusters))
        for name in ('size', 'cursor', 'accepted', 'updates', 'refresh_attempts', 'refresh_failures'):
            self.register_buffer(name, torch.zeros((), dtype=torch.long))
        self.register_buffer('valid', torch.zeros((), dtype=torch.bool))
        self.refresh_stats = {'status': 'never_attempted'}

    @torch.no_grad()
    def add(self, ct: Tensor, full: Tensor, weight: Tensor):
        if ct.ndim != 2 or ct.shape != full.shape or ct.shape[1] != self.channels or weight.shape != ct.shape[:1]:
            raise ValueError('expected paired [N,C] CT/Full and [N] weights')
        if not all(bool(torch.isfinite(v).all()) for v in (ct, full, weight)) or bool((weight <= 0).any()):
            raise ValueError('candidates must be finite with strictly positive weights')
        keep = ct.float().norm(dim=1) > 1e-8
        ct, full, weight = ct[keep], full[keep], weight[keep]
        self.accepted.add_(len(ct))
        ct, full, weight = ct[-self.capacity:], full[-self.capacity:], weight[-self.capacity:]
        n, start = len(ct), int(self.cursor)
        if not n: return
        first = min(n, self.capacity - start)
        for dst, src in ((self.pool_ct, ct), (self.pool_full, full), (self.pool_weight, weight)):
            src = src.detach().to(dst)
            dst[start:start+first].copy_(src[:first])
            if first < n: dst[:n-first].copy_(src[first:])
        self.cursor.fill_((start+n) % self.capacity)
        self.size.fill_(min(self.capacity, int(self.size)+n))

    def effective_transforms(self):
        return self.transforms

    def apply_value(self, query, slot):
        return query @ self.transforms[slot].detach().T

    @staticmethod
    @torch.no_grad()
    def fit_transform(x: Tensor, y: Tensor, w: Tensor, ridge: float):
        """Return the analytic RAW-feature transform, relative lambda and fit MSE.

        min_T sum_i w_i ||T x_i-y_i||^2/sum(w) + lambda ||T-I||_F^2.
        No backward/optimizer is involved. The error is in-pool, not validation.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x, y, w = x.float(), y.float(), w.float()
            cov = (x.T * w) @ x / w.sum()
            cross = (y.T * w) @ x / w.sum()
            lam = float(ridge) * (cov.trace()/x.shape[1] + 1e-6)
            eye = torch.eye(x.shape[1], device=x.device)
            # Right inverse: T A = B -> A^T T^T = B^T. No explicit inverse.
            t = torch.linalg.solve((cov+lam*eye).T, (cross+lam*eye).T).T
            err = ((x @ t.T-y).square().mean(1)*w).sum()/w.sum()
            return t, lam, err

    @torch.no_grad()
    def refresh(self) -> bool:
        self.refresh_attempts.add_(1)
        def fail(status):
            self.refresh_failures.add_(1); self.refresh_stats = {'status': status}
            return False
        n = int(self.size)
        if n < self.min_candidates: return fail('insufficient_candidates')
        with torch.autocast(device_type=self.keys.device.type, enabled=False):
            x, y, w = self.pool_ct[:n].float(), self.pool_full[:n].float(), self.pool_weight[:n].float()
            unit = F.normalize(x, dim=1, eps=1e-8)
            if len(torch.unique(unit, dim=0)) < self.clusters: return fail('insufficient_directions')
            if bool(self.valid): centers = self.keys.float().clone()
            else:
                # Deterministic farthest-direction initialization; no global RNG.
                first = (unit-F.normalize(unit.mean(0), dim=0)).square().sum(1).argmax()
                chosen = [unit[first]]
                for _ in range(1, self.clusters):
                    chosen.append(unit[(unit @ torch.stack(chosen).T).max(1).values.argmin()])
                centers = torch.stack(chosen)
            previous = None
            for _ in range(self.iterations):
                assignments = (unit @ centers.T).argmax(1)
                if previous is not None and torch.equal(assignments, previous): break
                previous = assignments.clone()
                for j in range(self.clusters):
                    mask = assignments == j
                    if not bool(mask.any()): return fail('empty_cluster')
                    v = (unit[mask]*w[mask, None]).sum(0)
                    if float(v.norm()) <= 1e-8: return fail('zero_direction_center')
                    centers[j] = F.normalize(v, dim=0)
            assignments = (unit @ centers.T).argmax(1)
            transforms, support, errors, identity, lambdas = [], [], [], [], []
            for j in range(self.clusters):
                ids = (assignments == j).nonzero(as_tuple=True)[0]
                if self.outlier_filter and len(ids) >= 10:
                    # Explicit ablation: remove farthest angular 10%, keeping raw pairs aligned.
                    count = max(self.min_cluster_samples, math.ceil(.9*len(ids)))
                    ids = ids[(unit[ids] @ centers[j]).argsort(descending=True)[:count]]
                if len(ids) < self.min_cluster_samples: return fail('insufficient_cluster_support')
                centers[j] = F.normalize((unit[ids]*w[ids, None]).sum(0), dim=0)
                try: t, lam, err = self.fit_transform(x[ids], y[ids], w[ids], self.ridge)
                except RuntimeError: return fail('solve_failed')
                if not bool(torch.isfinite(t).all()) or not bool(torch.isfinite(err)):
                    return fail('nonfinite_transform')
                transforms.append(t); support.append(len(ids)); errors.append(err); lambdas.append(lam)
                identity.append(((x[ids]-y[ids]).square().mean(1)*w[ids]).sum()/w[ids].sum())
            transforms = torch.stack(transforms)
            support = torch.tensor(support, device=x.device)
            errors, identity, lambdas = map(torch.stack, (errors, identity, lambdas))
            if bool(self.valid):
                distance = 1-self.keys @ centers.T
                order = torch.empty(self.clusters, dtype=torch.long, device=x.device)
                for _ in range(self.clusters):
                    old, new = divmod(int(distance.argmin()), self.clusters)
                    order[old] = new
                    distance[old, :] = float('inf'); distance[:, new] = float('inf')
                centers, transforms = centers[order], transforms[order]
                support, errors, identity, lambdas = support[order], errors[order], identity[order], lambdas[order]
                self.matrix_drift.copy_((transforms-self.transforms).flatten(1).norm(dim=1)*(1-self.decay))
                mixed = self.decay*self.keys+(1-self.decay)*centers
                self.keys.copy_(F.normalize(mixed, dim=1, eps=1e-8))
                self.transforms.mul_(self.decay).add_(transforms, alpha=1-self.decay)
            else:
                self.keys.copy_(centers); self.transforms.copy_(transforms); self.valid.fill_(True)
            self.support.copy_(support); self.fit_error.copy_(errors)
            self.identity_error.copy_(identity); self.ridge_lambda.copy_(lambdas)
            self.updates.add_(1); self.refresh_stats = {'status': 'updated'}
            return True


class MatrixScale(nn.Module):
    def __init__(self, channels: int, samples_per_class: int = 8,
                 temperature: float = .5, gate_init: float = .1, **bank_kwargs):
        super().__init__()
        if samples_per_class < 1 or not math.isfinite(temperature) or temperature <= 0 or not 0 < gate_init < 1:
            raise ValueError('invalid sampling budget, temperature or gate initialization')
        self.channels, self.samples_per_class = channels, samples_per_class
        self.temperature, self.gate_init = float(temperature), float(gate_init)
        value_mode = bank_kwargs.pop('value_mode', 'analytic')
        rank = bank_kwargs.pop('value_rank', 8)
        initialization = bank_kwargs.pop('value_init', 'full')
        if value_mode == 'lowrank':
            from models.ct_full_lowrank_value import LowRankValueBank
            def make_bank(): return LowRankValueBank(channels, value_rank=rank, value_init=initialization, **bank_kwargs)
        elif value_mode == 'analytic':
            def make_bank(): return MatrixBank(channels, **bank_kwargs)
        else: raise ValueError('value_mode must be analytic/lowrank')
        self.foreground_bank = make_bank()
        self.background_bank = make_bank()
        self.gate = nn.Conv2d(2*channels, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(gate_init/(1-gate_init)))

    @property
    def banks(self): return self.foreground_bank, self.background_bank

    @torch.no_grad()
    def candidates(self, ct: Tensor, full: Tensor, target: Tensor, seed: int = 2023):
        if ct.shape != full.shape or ct.ndim != 4 or ct.shape[1] != self.channels:
            raise ValueError('paired CT/Full features must match [B,C,H,W]')
        if target.ndim == 3: target = target[:, None]
        if target.ndim != 4 or target.shape[:2] != (ct.shape[0], 1):
            raise ValueError('GT must be [B,1,H,W]')
        if not all(bool(torch.isfinite(v).all()) for v in (ct, full, target)):
            raise ValueError('nonfinite collection inputs')
        with torch.autocast(device_type=ct.device.type, enabled=False):
            # Private CPU generator: collection cannot perturb backbone/dropout RNG.
            generator = torch.Generator().manual_seed(int(seed))
            occupancy = F.interpolate(target.ge(.5).float(), size=ct.shape[-2:], mode='area').flatten(1)
            xx, yy = ct.float().flatten(2).transpose(1,2), full.float().flatten(2).transpose(1,2)
            nonzero = xx.norm(dim=-1) > 1e-8
            classes = []
            for foreground in (True, False):
                rows = [[], [], []]
                for b in range(len(ct)):
                    mask = ((occupancy[b] > 0) if foreground else (occupancy[b] == 0)) & nonzero[b]
                    ids = mask.nonzero(as_tuple=True)[0]
                    if len(ids) > self.samples_per_class:
                        chosen = torch.randperm(len(ids), generator=generator)[:self.samples_per_class]
                        ids = ids[chosen.to(ids.device)]
                    rows[0].append(xx[b, ids]); rows[1].append(yy[b, ids])
                    rows[2].append(occupancy[b, ids] if foreground else torch.ones(len(ids), device=ct.device))
                classes.append(tuple(torch.cat(parts).detach() for parts in rows))
            return tuple(classes)

    def forward(self, x: Tensor, diagnostics: dict | None = None, read_mode: str = 'normal'):
        if read_mode not in ('normal', 'off', 'identity', 'shuffle'):
            raise ValueError('read_mode must be normal/off/identity/shuffle')
        if x.ndim != 4 or x.shape[1] != self.channels: raise ValueError('expected [B,C,H,W]')
        ready = [b for b in self.banks if bool(b.valid)]
        if diagnostics is not None: diagnostics['available'] = bool(ready)
        if not ready or read_mode in ('off', 'identity'): return x
        with torch.autocast(device_type=x.device.type, enabled=False):
            keys = torch.cat([b.keys.detach() for b in ready])
            slots = [(b,j) for b in ready for j in range(b.clusters)]
            if read_mode == 'shuffle': slots = slots[-1:]+slots[:-1]
            query = x.float().flatten(2).transpose(1,2)
            attention = (F.normalize(query, dim=-1, eps=1e-8) @ keys.T / self.temperature).softmax(-1)
            read = torch.zeros_like(query)
            # Do not allocate [B,H*W,K,C,C] dynamic matrices.
            for j, (bank, slot) in enumerate(slots):
                read = read + bank.apply_value(query, slot)*attention[..., j:j+1]
            transformed = read.transpose(1,2).reshape_as(x)
            gate = self.gate(torch.cat([x.float(), transformed], dim=1)).sigmoid()
            corrected = x.float()+gate*(transformed-x.float())
            if not bool(torch.isfinite(corrected).all()): raise RuntimeError('nonfinite matrix correction')
            if diagnostics is not None:
                rms = x.float().square().mean().sqrt().clamp_min(1e-8)
                diagnostics.update(gate_mean=float(gate.detach().mean()),
                    correction_rms_ratio=float((corrected-x.float()).detach().square().mean().sqrt()/rms.detach()),
                    transform_rms_ratio=float(transformed.detach().square().mean().sqrt()/rms.detach()),
                    attention_mean=attention.detach().mean((0,1)).cpu().tolist(),
                    top1_fraction=F.one_hot(attention.detach().argmax(-1), len(keys)).float().mean((0,1)).cpu().tolist(),
                    attention_entropy=float((-(attention.detach()*attention.detach().clamp_min(1e-12).log()).sum(-1)).mean()))
            return corrected.to(x.dtype)


class CTFullMatrixMemory(nn.Module):
    def __init__(self, channels: Sequence[int], warmup_steps: int = 100,
                 refresh_steps: int = 20, sampling_seed: int = 2023, **scale_kwargs):
        super().__init__()
        if len(channels) != 4 or warmup_steps < 0 or refresh_steps < 1:
            raise ValueError('require four independent scales and valid update cadence')
        self.warmup_steps, self.refresh_steps = warmup_steps, refresh_steps
        self.sampling_seed = int(sampling_seed)
        self.scales = nn.ModuleList(MatrixScale(c, **scale_kwargs) for c in channels)
        self.register_buffer('update_steps', torch.zeros((), dtype=torch.long))

    def forward(self, features, diagnostics=None, read_mode='normal'):
        if len(features) != 4: raise ValueError('expected four scales')
        return [s(x, diagnostics[i] if diagnostics is not None else None, read_mode)
                for i, (s, x) in enumerate(zip(self.scales, features))]

    @torch.no_grad()
    def candidates(self, ct, full, target):
        if not self.training: raise RuntimeError('GT collection is training-only')
        if len(ct) != 4 or len(full) != 4: raise ValueError('expected four paired scales')
        return [s.candidates(x, y, target, self.sampling_seed+1000003*int(self.update_steps)+i*7919)
                for i, (s, x, y) in enumerate(zip(self.scales, ct, full))]

    @torch.no_grad()
    def commit(self, pending):
        if not self.training: raise RuntimeError('matrix writes forbidden in eval')
        if len(pending) != 4: raise ValueError('expected four candidate payloads')
        self.update_steps.add_(1)
        for s, payload in zip(self.scales, pending):
            for b, values in zip(s.banks, payload):
                b.add(*values)
                if int(self.update_steps) <= self.warmup_steps: continue
                if not bool(b.valid) or int(self.update_steps) % self.refresh_steps == 0:
                    b.refresh()

    def diagnostics(self):
        result = {'successful_steps': int(self.update_steps)}
        for i, s in enumerate(self.scales, 1):
            for label, b in zip(('foreground', 'background'), s.banks):
                result[f's{i}_{label}_valid'] = bool(b.valid)
                result[f's{i}_{label}_updates'] = int(b.updates)
        return result
