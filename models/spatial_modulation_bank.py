"""Per-scale CT keys (streaming/EMA buffers) and learnable affine Values.

Inputs/outputs are matching [B,C,H,W] tensors. Full CT positions update only
keys after a successful optimizer step. Gamma/beta are Parameters, trained by
Missing segmentation and optional detached-CT -> detached-Full retrieval L1.
Epoch 1 is key collection only; epoch >=2 enables positional soft retrieval.
This module has no PET encoder, segmentation head, GT routing or spatial gate.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

EXPERIMENT_TYPE = "ct_key_learnable_affine_value"


class UnifiedModulationBank(nn.Module):
    def __init__(
        self,
        channels: int,
        clusters: int = 8,
        decay: float = 0.99,
        chunk_size: int = 8192,
    ):
        super().__init__()
        if channels < 1 or clusters < 2 or not 0 <= decay < 1 or chunk_size < 1:
            raise ValueError("invalid bank settings")
        self.channels, self.clusters = channels, clusters
        self.decay, self.chunk_size = decay, chunk_size
        self.register_buffer("keys", torch.zeros(clusters, channels))
        self.gamma = nn.Parameter(torch.zeros(clusters, channels))
        self.beta = nn.Parameter(torch.zeros(clusters, channels))
        self.register_buffer(
            "key_sum", torch.zeros(clusters, channels, dtype=torch.float64)
        )
        self.register_buffer("count", torch.zeros(clusters, dtype=torch.float64))
        self.register_buffer("fg_mass", torch.zeros(clusters, dtype=torch.float64))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("accepted", torch.tensor(0, dtype=torch.long))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))
        self.refresh_stats = {"status": "accumulating_keys"}

    @torch.no_grad()
    def add_all(
        self, key: torch.Tensor, foreground: torch.Tensor, first_epoch: bool
    ) -> None:
        """Consume every finite Full CT position; NEVER write gamma or beta.

        foreground is an occupancy statistic only and cannot affect assignment.
        chunk_size limits temporary assignment matrices, not sample count.
        """
        if not self.training:
            raise RuntimeError("evaluation bank writes forbidden")
        if key.ndim != 2 or key.shape[1] != self.channels:
            raise ValueError("CT keys must have shape [N,C]")
        if foreground.shape != (len(key),):
            raise ValueError("foreground statistics must have shape [N]")
        if not torch.isfinite(key).all() or not torch.isfinite(foreground).all():
            raise ValueError("nonfinite CT keys or occupancy statistics")
        n = len(key)
        if not n:
            return
        # Include zero CT vectors: all finite positions participate.
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
        sums = torch.zeros_like(self.key_sum)
        count = torch.zeros_like(self.count)
        fg = torch.zeros_like(self.fg_mass)
        for start in range(0, n, self.chunk_size):
            k = F.normalize(key[start : start + self.chunk_size].float(), dim=1)
            ids = (k @ self.keys.T).argmax(1)
            count.index_add_(
                0, ids, torch.ones(len(ids), device=ids.device, dtype=torch.float64)
            )
            sums.index_add_(0, ids, k.double())
            fg.index_add_(0, ids, foreground[start : start + self.chunk_size].double())
        self.count.add_(count)
        self.fg_mass.add_(fg)
        if first_epoch:
            self.key_sum.add_(sums)
            source, den = self.key_sum, self.count
        else:
            source, den = sums, count
        present = den > 0
        mean = (source[present] / den[present, None]).float()
        updated = (
            mean
            if first_epoch
            else self.decay * self.keys[present] + (1 - self.decay) * mean
        )
        self.keys[present] = F.normalize(updated, dim=1)
        self.accepted.add_(n)
        self.updates.add_(1)
        self.refresh_stats = {
            "status": "first_epoch_cumulative_keys" if first_epoch else "key_ema"
        }


class SpatialModulationScale(nn.Module):
    def __init__(
        self,
        channels: int,
        clusters: int = 8,
        decay: float = 0.99,
        temperature: float = 0.5,
        chunk_size: int = 8192,
    ):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.channels, self.temperature = channels, temperature
        self.bank = UnifiedModulationBank(channels, clusters, decay, chunk_size)
        self.register_buffer("epoch", torch.tensor(1, dtype=torch.long))
        self.last_supervision = {}

    @property
    def read_enabled(self) -> bool:
        return bool(self.bank.initialized) and int(self.epoch) >= 2

    def describe(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x.float(), 3, stride=1, padding=1, count_include_pad=False)

    def forward(
        self,
        x: torch.Tensor,
        diagnostics: dict | None = None,
        read_mode: str = "normal",
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("CT features must have shape [B,C,H,W]")
        if read_mode not in ("normal", "off", "identity", "shuffle"):
            raise ValueError("invalid read mode")
        if diagnostics is not None:
            diagnostics["available"] = self.read_enabled
        if not self.read_enabled or read_mode in ("off", "identity"):
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
                with torch.no_grad():
                    rms = x.float().square().mean().sqrt().clamp_min(1e-6)
                    entropy = -(attention * attention.clamp_min(1e-12).log()).sum(-1)
                    diagnostics.update(
                        correction_rms_ratio=float(
                            (result - x).square().mean().sqrt() / rms
                        ),
                        gamma_effect_rms=float((g * x).square().mean().sqrt()),
                        beta_rms=float(b.square().mean().sqrt()),
                        attention_mean=attention.mean((0, 1)).cpu().tolist(),
                        attention_top1_mean=float(attention.max(-1).values.mean()),
                        attention_entropy_normalized=float(
                            entropy.mean() / math.log(self.bank.clusters)
                        ),
                    )
            return result.to(x.dtype)

    def supervise(
        self,
        ct: torch.Tensor,
        full: torch.Tensor,
        mask: torch.Tensor,
        loss_enabled: bool,
        eps: float = 1e-6,
    ):
        """Supervise the SAME retrieval used by Missing, with detached references.

        Return (loss, pending_CT_key_statistics). Epoch 1 has no Value graph.
        With loss disabled, Missing segmentation still trains Values normally.
        """
        if ct.shape != full.shape or ct.ndim != 4 or ct.shape[1] != self.channels:
            raise ValueError("matching [B,C,H,W] references required")
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.ndim != 4 or mask.shape[:2] != (ct.shape[0], 1):
            raise ValueError("diagnostic mask must have shape [B,1,H,W]")
        with torch.autocast(device_type=ct.device.type, enabled=False):
            c, f = ct.detach().float(), full.detach().float()
            q = F.interpolate(mask.detach().float(), size=c.shape[-2:], mode="area")
            pending = (
                self.describe(c)
                .flatten(2)
                .transpose(1, 2)
                .reshape(-1, self.channels)
                .detach(),
                q.flatten().detach(),
            )
            self.last_supervision = {}
            if not self.read_enabled:
                return c.new_zeros(()), pending
            with torch.set_grad_enabled(torch.is_grad_enabled() and loss_enabled):
                # Do not detach the bank Values or put this read under no_grad
                # when the auxiliary objective is enabled.
                prediction = self(c)
                reference = f.square().mean((1, 2, 3)).sqrt() + eps
                per_image = (prediction - f).abs().mean((1, 2, 3)) / reference
                loss = per_image.mean() if loss_enabled else c.new_zeros(())
            with torch.no_grad():
                error = (prediction - f).abs().mean(1, keepdim=True) / reference[
                    :, None, None, None
                ]
                identity = (c - f).abs().mean(1, keepdim=True) / reference[
                    :, None, None, None
                ]
                self.last_supervision = dict(
                    retrieval_error=float(per_image.mean()),
                    identity_error=float(identity.mean()),
                )
                for name, weight in (("foreground", q), ("background", 1 - q)):
                    mass = weight.sum()
                    self.last_supervision[name + "_error"] = (
                        float((error * weight).sum() / mass) if mass > 0 else None
                    )
                    self.last_supervision[name + "_identity_error"] = (
                        float((identity * weight).sum() / mass) if mass > 0 else None
                    )
            return loss, pending


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
        if len(channels) != 4:
            raise ValueError("four scales required")
        self.scales = nn.ModuleList(
            SpatialModulationScale(c, clusters, decay, temperature, chunk_size)
            for c in channels
        )
        self.register_buffer("update_steps", torch.tensor(0, dtype=torch.long))

    def set_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("epoch must be >= 1")
        for s in self.scales:
            s.epoch.fill_(epoch)
            s.last_supervision = {}

    def forward(self, features, diagnostics=None, read_mode="normal"):
        if len(features) != 4:
            raise ValueError("four scales required")
        return [
            s(x, None if diagnostics is None else diagnostics[i], read_mode)
            for i, (s, x) in enumerate(zip(self.scales, features))
        ]

    @torch.no_grad()
    def commit(self, pending) -> None:
        if not self.training:
            raise RuntimeError("evaluation bank writes forbidden")
        if len(pending) != 4:
            raise ValueError("four scales required")
        for s, p in zip(self.scales, pending):
            s.bank.add_all(*p, first_epoch=int(s.epoch) == 1)
        self.update_steps.add_(1)

    def diagnostics(self):
        return dict(
            successful_steps=int(self.update_steps),
            scales=[
                dict(
                    accepted=int(s.bank.accepted),
                    initialized=bool(s.bank.initialized),
                    read_enabled=s.read_enabled,
                    count=s.bank.count.cpu().tolist(),
                    fg_mass=s.bank.fg_mass.cpu().tolist(),
                )
                for s in self.scales
            ],
        )
