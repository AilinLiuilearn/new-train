# -*- coding: utf-8 -*-
"""CT-conditioned direct affine compensation for the retrieved PET prior.

Strict contract
---------------
- Inputs: four-scale aligned CT features and four-scale retrieved PET priors,
  each pair with identical shapes [B_m, C_l, H_l, W_l] (B_m = Missing rows).
- The affine parameters (gamma, beta) are generated from CT ONLY
  (ct_feature.detach()). Real PET, PET prior, PET statistics, labels, or
  Full-branch predictions must NEVER be used to generate gamma/beta.
- pet_prior is only the modulated object: pet_comp = gamma * pet_prior + beta.
- No AdaIN, no prior normalization/whitening, no extra prior skip, no
  (1+gamma)*prior formulation, no sigmoid gate, no scalar alpha or post-scale.
- No BatchNorm, no dropout, no batch-dependent statistics inside generators.
- Identity init: gamma_head weight=0 bias=1, beta_head weight=0 bias=0,
  so that pet_comp == P_prior exactly before training.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


def _affine_finite_or_raise(scale_idx: int, name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        bad = int((~torch.isfinite(x)).sum().item())
        raise RuntimeError(f"[CTAffine][S{scale_idx + 1}] {name} contains {bad} NaN/Inf values")


class _PerScaleAffineGenerator(nn.Module):
    """Single-scale CT-only affine generator.

    trunk = Conv1x1(C, h) -> GELU -> DWConv3x3(groups=h) -> GELU
    gamma_head = Conv1x1(h, C), beta_head = Conv1x1(h, C)
    """

    def __init__(self, channels: int):
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels!r}")
        hidden = max(8, channels // 4)
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=True),
            nn.GELU(),
        )
        self.gamma_head = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.beta_head = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        # Identity initialization: direct affine starts as pet_comp = P_prior.
        with torch.no_grad():
            nn.init.zeros_(self.gamma_head.weight)
            nn.init.ones_(self.gamma_head.bias)
            nn.init.zeros_(self.beta_head.weight)
            nn.init.zeros_(self.beta_head.bias)

    def forward(self, ct_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # CT-only input; caller detaches before entering.
        z = self.trunk(ct_feature)
        gamma = self.gamma_head(z)
        beta = self.beta_head(z)
        return gamma, beta


class CTConditionedPETAffine(nn.Module):
    """Four-scale CT-conditioned direct affine: pet_comp = gamma(CT)*prior + beta(CT)."""

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        channels = [int(c) for c in channels]
        if len(channels) != 4:
            raise ValueError(f"CTConditionedPETAffine requires 4 scales, got {len(channels)}")
        if any(c <= 0 for c in channels):
            raise ValueError(f"channels must be positive, got {channels!r}")
        self.in_channels = list(channels)
        self.generators = nn.ModuleList([_PerScaleAffineGenerator(c) for c in channels])

    def forward_ct_params(
        self, ct_feats: Sequence[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate (gamma, beta) from CT ONLY. pet_prior must not be passed here."""
        ct_feats = list(ct_feats)
        if len(ct_feats) != 4:
            raise ValueError(f"expected 4 CT scales, got {len(ct_feats)}")
        gammas: List[torch.Tensor] = []
        betas: List[torch.Tensor] = []
        for s, ct in enumerate(ct_feats):
            if ct.ndim != 4:
                raise ValueError(
                    f"[CTAffine][S{s + 1}] ct_feature must be [B,C,H,W], got {tuple(ct.shape)}"
                )
            if int(ct.shape[1]) != int(self.in_channels[s]):
                raise ValueError(
                    f"[CTAffine][S{s + 1}] channel mismatch: ct={int(ct.shape[1])} "
                    f"expected={int(self.in_channels[s])}"
                )
            _affine_finite_or_raise(s, "ct_feature", ct)
            gamma, beta = self.generators[s](ct.detach())
            _affine_finite_or_raise(s, "gamma", gamma)
            _affine_finite_or_raise(s, "beta", beta)
            gammas.append(gamma)
            betas.append(beta)
        return gammas, betas

    def forward(
        self,
        ct_feats: Sequence[torch.Tensor],
        pet_prior: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Apply direct affine: pet_comp = gamma(CT) * pet_prior + beta(CT)."""
        ct_feats = list(ct_feats)
        pet_prior = list(pet_prior)
        if len(ct_feats) != 4 or len(pet_prior) != 4:
            raise ValueError(
                f"expected 4 scales each, got ct={len(ct_feats)} prior={len(pet_prior)}"
            )
        gammas, betas = self.forward_ct_params(ct_feats)
        compensated: List[torch.Tensor] = []
        for s, (ct, prior, gamma, beta) in enumerate(zip(ct_feats, pet_prior, gammas, betas)):
            if prior.ndim != 4:
                raise ValueError(
                    f"[CTAffine][S{s + 1}] pet_prior must be [B,C,H,W], got {tuple(prior.shape)}"
                )
            if tuple(prior.shape) != tuple(gamma.shape) or tuple(prior.shape) != tuple(ct.shape):
                raise ValueError(
                    f"[CTAffine][S{s + 1}] shape mismatch: ct={tuple(ct.shape)} "
                    f"prior={tuple(prior.shape)} gamma={tuple(gamma.shape)} beta={tuple(beta.shape)}"
                )
            if prior.device != ct.device:
                raise ValueError(
                    f"[CTAffine][S{s + 1}] device mismatch: ct={ct.device} prior={prior.device}"
                )
            _affine_finite_or_raise(s, "pet_prior", prior)
            pet_comp = gamma * prior + beta
            _affine_finite_or_raise(s, "pet_comp", pet_comp)
            compensated.append(pet_comp)
        return compensated, gammas, betas
