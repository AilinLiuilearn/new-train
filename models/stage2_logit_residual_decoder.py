"""Stage-2 independent logit residual decoder.

Frozen Stage-1 fused features S1–S4 feed a lightweight FPN-style decoder that
predicts a bounded logit residual ΔZ. The frozen Stage-1 decoder produces the
stable anchor Z_stage1; the final prediction is:

    Z_final = Z_stage1 + M * tanh(raw_ΔZ / M)

Zero-init of delta_head guarantees step-0 identity without an extra outer
gamma that would kill first-step gradients.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int) -> nn.GroupNorm:
    """Pick a GroupNorm group count that divides num_channels."""
    for g in (8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


def _resolve_availability(
    batch_size: int,
    route: Optional[str],
    pet_available: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Per-sample availability: 1=full, 0=missing."""
    route_l = None if route is None else str(route).strip().lower()
    if route_l == 'full':
        return torch.ones(batch_size, device=device, dtype=torch.float32)
    if route_l == 'missing':
        return torch.zeros(batch_size, device=device, dtype=torch.float32)
    if route_l == 'auto':
        if pet_available is None:
            raise ValueError("route='auto' requires pet_available")
        availability = pet_available.to(device=device).float().view(-1)
        if availability.numel() != batch_size:
            raise ValueError('pet_available must contain one state per sample')
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError('pet_available values must be 0 or 1')
        return availability
    raise ValueError(
        f"stage2 residual decoder requires route in {{full, missing, auto}}, got {route!r}"
    )


class _LateralBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            _group_norm(hidden_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _RefineBlock(nn.Module):
    """Depthwise-separable 3x3 refine: DWConv -> GN -> GELU -> PWConv -> GN -> GELU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            _group_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _group_norm(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Stage2LogitResidualDecoder(nn.Module):
    """Lightweight multi-scale residual decoder that outputs bounded Δlogits."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        hidden_channels: int = 64,
        out_channels: int = 1,
        state_conditioned: bool = False,
        delta_logit_max: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = tuple(int(c) for c in encoder_channels)
        if len(channels) != 4:
            raise ValueError(f'Expected 4 encoder scales, got {len(channels)}')
        if int(hidden_channels) <= 0:
            raise ValueError(f'hidden_channels must be > 0, got {hidden_channels}')
        if float(delta_logit_max) <= 0:
            raise ValueError(f'delta_logit_max must be > 0, got {delta_logit_max}')
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f'dropout must be in [0, 1), got {dropout}')

        self.encoder_channels = channels
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.state_conditioned = bool(state_conditioned)
        self.delta_logit_max = float(delta_logit_max)
        self.dropout_p = float(dropout)

        self.laterals = nn.ModuleList(
            [_LateralBlock(c, self.hidden_channels) for c in self.encoder_channels]
        )
        # refine for S3, S2, S1 fusion nodes (not needed at S4 tip).
        self.refine3 = _RefineBlock(self.hidden_channels)
        self.refine2 = _RefineBlock(self.hidden_channels)
        self.refine1 = _RefineBlock(self.hidden_channels)

        self.drop = nn.Dropout2d(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.delta_head = nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        self.state_film = None
        if self.state_conditioned:
            self.state_film = nn.Sequential(
                nn.Linear(2, self.hidden_channels),
                nn.GELU(),
                nn.Linear(self.hidden_channels, 2 * self.hidden_channels),
            )
            nn.init.zeros_(self.state_film[-1].weight)
            nn.init.zeros_(self.state_film[-1].bias)

    def _apply_state_film(
        self,
        x: torch.Tensor,
        route: Optional[str],
        pet_available: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.state_conditioned or self.state_film is None:
            return x
        b = x.shape[0]
        availability = _resolve_availability(b, route, pet_available, device=x.device)
        state_vec = torch.stack([availability, 1.0 - availability], dim=1)  # [B,2]
        gamma_beta = self.state_film(state_vec.float())
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.to(dtype=x.dtype).view(b, self.hidden_channels, 1, 1)
        beta = beta.to(dtype=x.dtype).view(b, self.hidden_channels, 1, 1)
        return x * (1.0 + gamma) + beta

    def forward(
        self,
        features: Sequence[torch.Tensor],
        target_size,
        route: Optional[str] = None,
        pet_available: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if len(features) != 4:
            raise ValueError(f'Expected 4 features, got {len(features)}')
        for i, (feat, expected_c) in enumerate(zip(features, self.encoder_channels), start=1):
            if feat.ndim != 4:
                raise ValueError(f'S{i} must be BCHW, got {tuple(feat.shape)}')
            if feat.shape[1] != expected_c:
                raise ValueError(
                    f'S{i} channel mismatch: expected {expected_c}, got {feat.shape[1]}'
                )

        s1, s2, s3, s4 = features
        l1 = self.laterals[0](s1)
        l2 = self.laterals[1](s2)
        l3 = self.laterals[2](s3)
        l4 = self.laterals[3](s4)

        p4 = l4
        p3 = self.refine3(
            l3 + F.interpolate(p4, size=l3.shape[-2:], mode='bilinear', align_corners=False)
        )
        p2 = self.refine2(
            l2 + F.interpolate(p3, size=l2.shape[-2:], mode='bilinear', align_corners=False)
        )
        p1 = self.refine1(
            l1 + F.interpolate(p2, size=l1.shape[-2:], mode='bilinear', align_corners=False)
        )
        p1 = self._apply_state_film(p1, route=route, pet_available=pet_available)
        p1 = self.drop(p1)

        raw = self.delta_head(p1)
        raw = F.interpolate(raw, size=target_size, mode='bilinear', align_corners=False)
        m = self.delta_logit_max
        delta = m * torch.tanh(raw / m)

        abs_delta = delta.detach().float().abs()
        stats = {
            'delta_logit_abs_mean': abs_delta.mean(),
            'delta_logit_abs_max': abs_delta.amax(),
            'delta_logit_l2': torch.linalg.vector_norm(
                delta.detach().float().reshape(delta.shape[0], -1), dim=1
            ).mean(),
        }
        return {
            'delta_logits': delta,
            'raw_delta_logits': raw,
            'stats': stats,
        }


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _smoke_test() -> None:
    torch.manual_seed(0)
    channels = (64, 128, 320, 512)
    feats = [
        torch.randn(2, 64, 72, 72),
        torch.randn(2, 128, 36, 36),
        torch.randn(2, 320, 18, 18),
        torch.randn(2, 512, 9, 9),
    ]
    target = (288, 288)

    for state_cond in (False, True):
        m = Stage2LogitResidualDecoder(
            encoder_channels=channels,
            hidden_channels=64,
            out_channels=1,
            state_conditioned=state_cond,
            delta_logit_max=2.0,
        )
        m.train()
        out_full = m(feats, target, route='full')
        out_miss = m(feats, target, route='missing')
        assert out_full['delta_logits'].shape == (2, 1, 288, 288)
        assert float(out_full['delta_logits'].abs().max()) == 0.0
        assert float(out_miss['delta_logits'].abs().max()) == 0.0
        assert torch.isfinite(out_full['delta_logits']).all()

        pet_available = torch.tensor([1, 0], dtype=torch.long)
        out_auto = m(feats, target, route='auto', pet_available=pet_available)
        assert out_auto['delta_logits'].shape == (2, 1, 288, 288)
        assert float(out_auto['delta_logits'].abs().max()) == 0.0

        loss = out_full['delta_logits'].sum() + torch.randn_like(out_full['delta_logits']).sum() * 0
        # Force a non-zero loss path through delta_head via an external target.
        target_t = torch.randn_like(out_full['delta_logits'])
        # Temporarily break zero-init identity by a synthetic objective on raw path:
        # use (delta + target).mean() so grads flow once weights leave zero after step.
        # At step-0 delta=0, grads to delta_head still exist via tanh Jacobian.
        ((out_full['delta_logits'] - target_t) ** 2).mean().backward()
        assert m.delta_head.weight.grad is not None
        assert float(m.delta_head.weight.grad.abs().sum()) > 0
        assert torch.isfinite(m.delta_head.weight.grad).all()

    print('Stage2LogitResidualDecoder smoke test: PASS')
    print(f'params(no-state)={count_trainable_parameters(Stage2LogitResidualDecoder(channels)):}')
    print(
        f'params(state)={count_trainable_parameters(Stage2LogitResidualDecoder(channels, state_conditioned=True)):}'
    )


if __name__ == '__main__':
    _smoke_test()
