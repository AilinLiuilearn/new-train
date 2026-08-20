# -*- coding: utf-8 -*-
"""
TaskMoE Stage-4 Refiner
=======================

Standalone adaptation of the Task-aware gating + multi-expert collaboration
idea from TG-ECNet (ICML 2025), specialized for a fused stage-4 feature map.

Reference:
    Yiming Sun et al., "Task-Gated Multi-Expert Collaboration Network for
    Degraded Multi-Modal Image Fusion", ICML 2025.
    https://github.com/LeeX54946/TG-ECNet

Intended input in the PET-CT project:
    F4_base = C4 + alpha_route * P4_cal
    shape   = [B, C, H, W], typically [B, 512, 16, 16]

Kept core ideas:
    fused feature -> feature-derived task prompt -> noisy Top-K routing
    -> MLP experts -> weighted collaboration -> residual refinement

This module does NOT own CPPI, calibration, alpha, CT/PET encoders, or decoder.

Residual modes:
    paper:      y = x + delta
    zero_start: y = x + beta * delta, beta starts from 0 (recommended for
                inserting into a frozen Stage-1 best checkpoint).

Recommended first experiment:
    channels=512, num_experts=6, top_k=2, prompt_atoms=32,
    prompt_dim=256, mlp_ratio=2.0, residual_mode="zero_start".

Run directly:
    python taskmoe_s4_refiner.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _cv_squared(x: Tensor, eps: float = 1e-10) -> Tensor:
    """Coefficient-of-variation squared used for expert-load balancing."""
    if x.numel() <= 1:
        return x.new_zeros(())
    xf = x.float()
    return xf.var(unbiased=False) / (xf.mean().square() + eps)


@dataclass
class TaskMoEDebug:
    importance: Tensor
    load: Tensor
    topk_indices: Tensor
    topk_gates: Tensor
    residual_scale: Tensor


class FeatureTaskPrompt(nn.Module):
    """Feature-derived sample-level task prompt, following TG-ECNet's pattern."""

    def __init__(
        self,
        in_channels: int,
        num_atoms: int = 32,
        prompt_dim: int = 256,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.condition_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 32, 1),
        )
        self.atom_logits = nn.Linear(32, num_atoms)
        self.dictionary = nn.Parameter(torch.randn(num_atoms, prompt_dim))
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")
        if x.shape[-2] < 9 or x.shape[-1] < 9:
            raise ValueError("TaskPrompt expects spatial size >= 9x9")
        z = self.condition_net(x)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        weights = F.softmax(self.atom_logits(z), dim=-1)
        return self.act(weights @ self.dictionary)


class ExpertMLP(nn.Module):
    """One expert: C -> rC -> C."""

    def __init__(self, channels: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(1, int(round(channels * mlp_ratio)))
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, channels)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class NoisyTopKGate(nn.Module):
    """Task-aware noisy Top-K sparse router over spatial tokens."""

    def __init__(
        self,
        channels: int,
        num_experts: int = 6,
        top_k: int = 2,
        noisy_gating: bool = True,
        noise_epsilon: float = 1e-2,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")

        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.noise_epsilon = float(noise_epsilon)

        gate_in = 2 * channels
        self.w_gate = nn.Parameter(torch.empty(gate_in, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(gate_in, num_experts))
        self.softplus = nn.Softplus()
        nn.init.normal_(self.w_gate, mean=0.0, std=1.0 / math.sqrt(gate_in))

    @staticmethod
    def _normal_cdf(v: Tensor) -> Tensor:
        return 0.5 * (1.0 + torch.erf(v / math.sqrt(2.0)))

    def _prob_in_top_k(
        self,
        clean_logits: Tensor,
        noisy_logits: Tensor,
        noise_std: Tensor,
        top_values: Tensor,
    ) -> Tensor:
        n = clean_logits.shape[0]
        m = top_values.shape[1]
        flat = top_values.reshape(-1)

        pos_in = torch.arange(n, device=clean_logits.device) * m + self.top_k
        th_in = flat.gather(0, pos_in).unsqueeze(1)
        pos_out = pos_in - 1
        th_out = flat.gather(0, pos_out).unsqueeze(1)

        is_in = noisy_logits > th_in
        std = noise_std.clamp_min(1e-6)
        p_in = self._normal_cdf((clean_logits - th_in) / std)
        p_out = self._normal_cdf((clean_logits - th_out) / std)
        return torch.where(is_in, p_in, p_out)

    def forward(
        self,
        token_features: Tensor,
        token_prompts: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if token_features.shape != token_prompts.shape:
            raise ValueError("token_features/token_prompts shape mismatch")

        router_in = torch.cat([token_features, token_prompts], dim=-1)
        clean_logits = router_in @ self.w_gate

        noise_std: Optional[Tensor] = None
        if self.noisy_gating and self.training:
            noise_std = self.softplus(router_in @ self.w_noise) + self.noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
            logits = noisy_logits
        else:
            noisy_logits = clean_logits
            logits = clean_logits

        request_k = (
            min(self.top_k + 1, self.num_experts)
            if self.noisy_gating and self.training and self.top_k < self.num_experts
            else self.top_k
        )
        top_values, top_indices = logits.topk(request_k, dim=-1)
        topk_logits = top_values[:, : self.top_k]
        topk_indices = top_indices[:, : self.top_k]
        topk_gates = F.softmax(topk_logits, dim=-1)

        gates = torch.zeros_like(logits)
        gates.scatter_(1, topk_indices, topk_gates)

        importance = gates.sum(dim=0)
        if (
            self.noisy_gating
            and self.training
            and self.top_k < self.num_experts
            and noise_std is not None
        ):
            load = self._prob_in_top_k(
                clean_logits, noisy_logits, noise_std, top_values
            ).sum(dim=0)
        else:
            load = (gates > 0).sum(dim=0).to(gates.dtype)

        return gates, load, importance, topk_indices, topk_gates


class SparseExpertCollaboration(nn.Module):
    """Dispatch selected tokens to experts and aggregate weighted outputs."""

    def __init__(
        self,
        channels: int,
        num_experts: int = 6,
        top_k: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        noisy_gating: bool = True,
        noise_epsilon: float = 1e-2,
    ) -> None:
        super().__init__()
        self.router = NoisyTopKGate(
            channels, num_experts, top_k, noisy_gating, noise_epsilon
        )
        self.experts = nn.ModuleList(
            [ExpertMLP(channels, mlp_ratio, dropout) for _ in range(num_experts)]
        )

    def forward(
        self,
        tokens: Tensor,
        token_prompts: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        gates, load, importance, topk_indices, topk_gates = self.router(
            tokens, token_prompts
        )

        delta = torch.zeros_like(tokens)
        for eid, expert in enumerate(self.experts):
            token_ids = torch.nonzero(gates[:, eid] > 0, as_tuple=False).flatten()
            if token_ids.numel() == 0:
                continue
            expert_in = tokens.index_select(0, token_ids)
            expert_out = expert(expert_in)
            weight = gates.index_select(0, token_ids)[:, eid].unsqueeze(-1)
            delta.index_add_(0, token_ids, expert_out * weight)

        balance_loss = _cv_squared(importance) + _cv_squared(load)
        routing = {
            "importance": importance,
            "load": load,
            "topk_indices": topk_indices,
            "topk_gates": topk_gates,
        }
        return delta, balance_loss, routing


class TaskMoEStage4Refiner(nn.Module):
    """
    Standalone Stage-4 TaskMoE refiner.

    Integration:
        f4_base = ct4 + alpha_route * pet4_cal
        f4_out, aux_loss = taskmoe_s4(f4_base)

    residual_mode:
        "paper"      -> out = x + delta
        "zero_start" -> out = x + beta * delta, beta learnable from 0
    """

    def __init__(
        self,
        channels: int = 512,
        num_experts: int = 6,
        top_k: int = 2,
        prompt_atoms: int = 32,
        prompt_dim: int = 256,
        prompt_hidden_channels: int = 64,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        noisy_gating: bool = True,
        noise_epsilon: float = 1e-2,
        balance_loss_weight: float = 0.1,
        residual_mode: str = "zero_start",
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        if residual_mode not in {"paper", "zero_start"}:
            raise ValueError("residual_mode must be 'paper' or 'zero_start'")

        self.channels = channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_weight = float(balance_loss_weight)
        self.residual_mode = residual_mode

        self.task_prompt = FeatureTaskPrompt(
            in_channels=channels,
            num_atoms=prompt_atoms,
            prompt_dim=prompt_dim,
            hidden_channels=prompt_hidden_channels,
        )
        self.prompt_projection = nn.Linear(prompt_dim, channels)
        self.collaboration = SparseExpertCollaboration(
            channels=channels,
            num_experts=num_experts,
            top_k=top_k,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            noisy_gating=noisy_gating,
            noise_epsilon=noise_epsilon,
        )

        if residual_mode == "zero_start":
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        else:
            self.register_buffer("residual_scale", torch.tensor(1.0), persistent=False)

    def forward(self, x: Tensor, return_debug: bool = False):
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW input, got {tuple(x.shape)}")
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"Configured C={self.channels}, input C={c}")

        prompt = self.prompt_projection(self.task_prompt(x))  # [B,C]
        tokens = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        token_prompts = (
            prompt[:, None, None, :]
            .expand(b, h, w, c)
            .reshape(b * h * w, c)
        )

        delta_tokens, raw_balance, routing = self.collaboration(tokens, token_prompts)
        delta = (
            delta_tokens.reshape(b, h, w, c)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        out = x + self.residual_scale.to(dtype=x.dtype) * delta
        aux_loss = raw_balance * self.balance_loss_weight

        if not return_debug:
            return out, aux_loss

        debug = TaskMoEDebug(
            importance=routing["importance"].detach(),
            load=routing["load"].detach(),
            topk_indices=routing["topk_indices"].detach(),
            topk_gates=routing["topk_gates"].detach(),
            residual_scale=self.residual_scale.detach().clone(),
        )
        return out, aux_loss, debug


def refine_only_stage4(
    fused_features: List[Tensor],
    taskmoe_s4: TaskMoEStage4Refiner,
) -> Tuple[List[Tensor], Tensor]:
    """Keep S1-S3 unchanged and refine only S4."""
    if len(fused_features) != 4:
        raise ValueError("Expected [F1,F2,F3,F4]")
    out = list(fused_features)
    out[3], aux_loss = taskmoe_s4(out[3])
    return out, aux_loss


def _self_test() -> None:
    torch.manual_seed(2026)
    print("TaskMoEStage4Refiner self-test")

    # small CPU test
    x = torch.randn(2, 64, 16, 16, requires_grad=True)
    m = TaskMoEStage4Refiner(
        channels=64,
        num_experts=6,
        top_k=2,
        prompt_dim=64,
        mlp_ratio=2.0,
        residual_mode="zero_start",
        residual_scale_init=0.0,
    )
    y, aux, dbg = m(x, return_debug=True)
    diff = (y - x).abs().max().item()
    print("zero_start shape:", tuple(y.shape))
    print("zero_start max |out-in|:", diff)
    print("balance loss:", float(aux))
    print("beta:", float(dbg.residual_scale))
    assert y.shape == x.shape
    assert diff == 0.0

    (y.square().mean() + aux).backward()
    assert m.residual_scale.grad is not None
    print("beta grad:", float(m.residual_scale.grad))

    x2 = torch.randn(2, 64, 16, 16, requires_grad=True)
    p = TaskMoEStage4Refiner(
        channels=64,
        num_experts=6,
        top_k=2,
        prompt_dim=64,
        residual_mode="paper",
    )
    y2, aux2, dbg2 = p(x2, return_debug=True)
    print("paper mean |delta|:", float((y2 - x2).abs().mean()))
    print("paper routing top-k shape:", tuple(dbg2.topk_indices.shape))
    (y2.mean() + aux2).backward()
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()