"""Stage-2 decoder interface adapter (post-hoc residual on frozen d1).

Keeps the pretrained UNetStyleDecoder frozen and applies a zero-start residual
refinement after fuse1 / before seg_head:

    d1_adapted = d1_old + DecoderAdapter(d1_old, role_context)
    logits = frozen_seg_head(d1_adapted)

role_context is sample-level (state + private-gate statistics) and modulates
the bottleneck via FiLM. No attention.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class Stage2DecoderAdapter(nn.Module):
    """Lightweight residual adapter on decoder d1 features."""

    def __init__(
        self,
        channels: int = 64,
        bottleneck: int = 16,
        role_context_dim: int = 64,
        level: str = 'd1',
    ) -> None:
        super().__init__()
        level = str(level).strip().lower()
        if level != 'd1':
            raise ValueError(
                f'stage2_decoder_adapter_level only supports d1 in v1, got {level!r}'
            )
        self.channels = int(channels)
        self.bottleneck = int(bottleneck)
        self.role_context_dim = int(role_context_dim)
        self.level = level

        self.down = nn.Conv2d(self.channels, self.bottleneck, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=min(8, self.bottleneck), num_channels=self.bottleneck)
        self.act1 = nn.GELU()
        self.dw = nn.Conv2d(
            self.bottleneck,
            self.bottleneck,
            kernel_size=3,
            padding=1,
            groups=self.bottleneck,
            bias=True,
        )
        self.act2 = nn.GELU()
        self.up = nn.Conv2d(self.bottleneck, self.channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        # FiLM: produce per-channel scale/shift for bottleneck from role_context.
        self.film = nn.Linear(self.role_context_dim, 2 * self.bottleneck)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(
        self,
        d1: torch.Tensor,
        role_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if d1.ndim != 4:
            raise ValueError(f'd1 must be BCHW, got {tuple(d1.shape)}')
        if d1.shape[1] != self.channels:
            raise ValueError(
                f'd1 channels mismatch: expected {self.channels}, got {d1.shape[1]}'
            )
        b = d1.shape[0]
        x = self.down(d1)
        x = self.norm(x)
        x = self.act1(x)

        if role_context is None:
            role_context = d1.new_zeros(b, self.role_context_dim)
        else:
            if role_context.ndim != 2:
                raise ValueError(
                    f'role_context must be [B,C], got {tuple(role_context.shape)}'
                )
            if role_context.shape[0] != b:
                raise ValueError('role_context batch mismatch')
            if role_context.shape[1] != self.role_context_dim:
                # Allow broadcasting / projection mismatch to fail loudly.
                raise ValueError(
                    f'role_context dim mismatch: expected {self.role_context_dim}, '
                    f'got {role_context.shape[1]}'
                )

        gamma_beta = self.film(role_context.float())
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.to(dtype=x.dtype).view(b, self.bottleneck, 1, 1)
        beta = beta.to(dtype=x.dtype).view(b, self.bottleneck, 1, 1)
        # Identity at init: gamma=0, beta=0 => (1+0)*x + 0 = x
        x = x * (1.0 + gamma) + beta

        x = self.dw(x)
        x = self.act2(x)
        delta = self.up(x)
        return delta


def _smoke_test() -> None:
    torch.manual_seed(0)
    adapter = Stage2DecoderAdapter(channels=64, bottleneck=16, role_context_dim=64)
    d1 = torch.randn(2, 64, 32, 32)
    ctx = torch.randn(2, 64)
    delta = adapter(d1, ctx)
    assert delta.shape == d1.shape
    assert float(delta.abs().max().item()) == 0.0
    print('Stage2DecoderAdapter smoke test: PASS')


if __name__ == '__main__':
    _smoke_test()
