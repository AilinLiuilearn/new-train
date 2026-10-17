"""Unified streaming CT-key / affine-value banks; no pixel candidate truncation.

First batch seeds online spherical clustering; all first-epoch positions then
contribute to count-weighted sufficient statistics. Later batches update paired
keys/values with EMA. This is streaming clustering, not exact epoch-wide kmeans.
"""

import torch
from torch import nn
import torch.nn.functional as F


class UnifiedModulationBank(nn.Module):
    def __init__(self, channels, clusters=8, decay=0.99, chunk_size=8192):
        super().__init__()
        if channels < 1 or clusters < 2 or not 0 <= decay < 1 or chunk_size < 1:
            raise ValueError("invalid bank settings")
        self.channels, self.clusters, self.decay, self.chunk_size = (
            channels,
            clusters,
            decay,
            chunk_size,
        )
        for name, shape in [
            ("keys", (clusters, channels)),
            ("gamma", (clusters, channels)),
            ("beta", (clusters, channels)),
        ]:
            self.register_buffer(name, torch.zeros(shape))
        for name, shape in [
            ("key_sum", (clusters, channels)),
            ("gamma_sum", (clusters, channels)),
            ("beta_sum", (clusters, channels)),
            ("count", (clusters,)),
            ("fg_mass", (clusters,)),
        ]:
            self.register_buffer(name, torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("accepted", torch.tensor(0, dtype=torch.long))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))
        self.refresh_stats = {"status": "accumulating"}

    @torch.no_grad()
    def add_all(self, key, gamma, beta, foreground, first_epoch):
        if key.shape != gamma.shape or key.shape != beta.shape or key.ndim != 2:
            raise ValueError("paired [N,C] required")
        if not all(torch.isfinite(x).all() for x in (key, gamma, beta, foreground)):
            raise ValueError("nonfinite candidates")
        n = len(key)
        if not n:
            return
        # Include zero CT vectors as well: all finite positions participate.
        if not bool(self.initialized):
            if n < self.clusters:
                raise ValueError("initial batch needs at least K positions")
            unit = F.normalize(key.float(), dim=1)
            chosen = [int((unit - unit.mean(0)).square().sum(1).argmax())]
            for _ in range(1, self.clusters):
                score = (unit @ unit[chosen].T).max(1).values
                score[chosen] = float("inf")
                chosen.append(int(score.argmin()))
            self.keys.copy_(unit[chosen])
            self.initialized.fill_(True)
        sums = [torch.zeros_like(self.key_sum) for _ in range(3)]
        count = torch.zeros_like(self.count)
        fg = torch.zeros_like(self.fg_mass)
        for start in range(0, n, self.chunk_size):
            k = F.normalize(key[start : start + self.chunk_size].float(), dim=1)
            ids = (k @ self.keys.T).argmax(1)
            count.index_add_(
                0, ids, torch.ones(len(ids), device=ids.device, dtype=torch.float64)
            )
            fg.index_add_(0, ids, foreground[start : start + self.chunk_size].double())
            for total, values in zip(
                sums,
                (
                    k,
                    gamma[start : start + self.chunk_size],
                    beta[start : start + self.chunk_size],
                ),
            ):
                total.index_add_(0, ids, values.double())
        if first_epoch:
            for total, add in zip((self.key_sum, self.gamma_sum, self.beta_sum), sums):
                total.add_(add)
            self.count.add_(count)
            self.fg_mass.add_(fg)
            source = (self.key_sum, self.gamma_sum, self.beta_sum)
            den = self.count
        else:
            source = sums
            den = count
            self.count.add_(count)
            self.fg_mass.add_(fg)
        present = den > 0
        for target, total in zip((self.keys, self.gamma, self.beta), source):
            mean = (total[present] / den[present, None]).float()
            updated = (
                mean
                if first_epoch
                else self.decay * target[present] + (1 - self.decay) * mean
            )
            target[present] = (
                F.normalize(updated, dim=1) if target is self.keys else updated
            )
        self.accepted.add_(n)
        self.updates.add_(1)
        self.refresh_stats = {
            "status": "first_epoch_cumulative" if first_epoch else "paired_ema"
        }


class SpatialModulationScale(nn.Module):
    def __init__(
        self, channels, clusters=8, decay=0.99, temperature=0.5, chunk_size=8192
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.channels, self.temperature = channels, temperature
        self.bank = UnifiedModulationBank(channels, clusters, decay, chunk_size)
        self.generator = nn.Sequential(
            nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels),
            nn.GELU(),
            nn.Conv2d(2 * channels, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 2 * channels, 1),
        )
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)
        self.register_buffer("epoch", torch.tensor(1, dtype=torch.long))

    def describe(self, x):
        return F.avg_pool2d(x.float(), 3, stride=1, padding=1, count_include_pad=False)

    def supervise(self, ct, full, mask, loss_enabled, eps=1e-6):
        if ct.shape != full.shape or ct.ndim != 4 or ct.shape[1] != self.channels:
            raise ValueError("matching [B,C,H,W] required")
        if mask.ndim == 3:
            mask = mask[:, None]
        with torch.autocast(device_type=ct.device.type, enabled=False):
            c, f = ct.detach().float(), full.detach().float()
            gamma, beta = self.generator(torch.cat([c, f], 1)).chunk(2, dim=1)
            dims = (1, 2, 3)
            reference = f.square().mean(dims).sqrt() + eps
            loss = (
                ((c * (1 + gamma) + beta - f).abs().mean(dims) / reference).mean()
                if loss_enabled
                else c.new_zeros(())
            )
            q = F.interpolate(mask.detach().float(), size=c.shape[-2:], mode="area")
            with torch.no_grad():
                error = (
                    (c * (1 + gamma.detach()) + beta.detach() - f)
                    .abs()
                    .mean(1, keepdim=True)
                )
                error = error / reference[:, None, None, None]
                self.last_supervision = {}
                for name, weight in [("foreground", q), ("background", 1 - q)]:
                    self.last_supervision[name + "_error"] = (
                        float((error * weight).sum() / weight.sum())
                        if weight.sum() > 0
                        else None
                    )
            flat = (
                lambda x: x.flatten(2).transpose(1, 2).reshape(-1, x.shape[1]).detach()
            )
            pending = (
                flat(self.describe(c)),
                flat(gamma),
                flat(beta),
                q.flatten().detach(),
            )
            return loss, pending

    def forward(self, x, diagnostics=None, read_mode="normal"):
        if read_mode not in ("normal", "off", "identity", "shuffle"):
            raise ValueError("invalid read mode")
        ready = bool(self.bank.initialized) and int(self.epoch) >= 2
        if diagnostics is not None:
            diagnostics["available"] = ready
        if not ready or read_mode in ("off", "identity"):
            return x
        with torch.autocast(device_type=x.device.type, enabled=False):
            query = self.describe(x).flatten(2).transpose(1, 2)
            attention = (
                F.normalize(query, dim=-1) @ self.bank.keys.T / self.temperature
            ).softmax(-1)
            gamma, beta = self.bank.gamma, self.bank.beta
            if read_mode == "shuffle":
                gamma, beta = gamma.roll(1, 0), beta.roll(1, 0)
            g = (attention @ gamma).transpose(1, 2).reshape_as(x)
            b = (attention @ beta).transpose(1, 2).reshape_as(x)
            result = x.float() * (1 + g) + b
            if not torch.isfinite(result).all():
                raise RuntimeError("nonfinite affine output")
            if diagnostics is not None:
                rms = x.float().square().mean().sqrt().clamp_min(1e-6)
                diagnostics.update(
                    correction_rms_ratio=float(
                        ((result - x).square().mean().sqrt() / rms).detach()
                    ),
                    gamma_effect_rms=float((g * x).detach().square().mean().sqrt()),
                    beta_rms=float(b.detach().square().mean().sqrt()),
                    attention_mean=attention.mean((0, 1)).detach().cpu().tolist(),
                )
            return result.to(x.dtype)


class SpatialModulationMemory(nn.Module):
    def __init__(
        self,
        channels=(64, 128, 320, 512),
        clusters=8,
        decay=0.99,
        temperature=0.5,
        chunk_size=8192,
    ):
        super().__init__()
        self.scales = nn.ModuleList(
            SpatialModulationScale(c, clusters, decay, temperature, chunk_size)
            for c in channels
        )
        self.register_buffer("update_steps", torch.tensor(0, dtype=torch.long))

    def set_epoch(self, epoch):
        for s in self.scales:
            s.epoch.fill_(epoch)

    def forward(self, features, diagnostics=None, read_mode="normal"):
        if len(features) != 4:
            raise ValueError("four scales required")
        return [
            s(x, None if diagnostics is None else diagnostics[i], read_mode)
            for i, (s, x) in enumerate(zip(self.scales, features))
        ]

    @torch.no_grad()
    def commit(self, pending):
        if not self.training:
            raise RuntimeError("evaluation bank writes forbidden")
        if len(pending) != 4:
            raise ValueError("four scales required")
        for s, p in zip(self.scales, pending):
            s.bank.add_all(*p, first_epoch=int(s.epoch) == 1)
        self.update_steps.add_(1)

    def diagnostics(self):
        return {
            "successful_steps": int(self.update_steps),
            "scales": [
                dict(
                    accepted=int(s.bank.accepted),
                    initialized=bool(s.bank.initialized),
                    read_enabled=bool(s.bank.initialized) and int(s.epoch) >= 2,
                    count=s.bank.count.cpu().tolist(),
                    fg_mass=s.bank.fg_mass.cpu().tolist(),
                )
                for s in self.scales
            ],
        }
