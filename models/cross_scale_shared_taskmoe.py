"""Cross-scale Shared TaskMoE for 4-stage PET-CT segmentation.

Design goal
-----------
Keep the already-validated all-scale Stage-1 fused features on the main path,
and refine all four scales with one *shared expert bank*. Each scale keeps:
    1) TG-ECNet-style TaskPrompt generator,
    2) input/output projection,
    3) task-aware Noisy Top-K router,
    4) residual combination controlled by residual_mode.

The expert bank itself is shared across S1/S2/S3/S4.

Expected Stage-1 feature shapes:
    S1: [B,  64, 128, 128]
    S2: [B, 128,  64,  64]
    S3: [B, 320,  32,  32]
    S4: [B, 512,  16,  16]

This file intentionally preserves the previously validated TaskMoE choices
(num_experts=6, top_k=2, mlp_ratio=2.0) so that the *only major structural
change* is: four independent expert banks -> one cross-scale shared expert bank.

Residual modes:
    - zero_start:
      F_out = F_base + beta_s * DeltaF,
      beta_s initialized from 0 (learnable).
    - paper:
      F_out = F_base + DeltaF,
      no external learnable residual scaling.
      Router + Expert themselves learn residual correction magnitude.

Reference idea:
    TG-ECNet / TaskMoE (ICML 2025)
    - CondNet + GAP -> softmax prompt dictionary
    - task-aware Noisy Top-K routing
    - sparse MLP experts
    - importance/load balance loss

PET-CT-specific adaptation:
    - all-scale fused features are Stage-1 outputs
    - scale-specific routers, shared expert bank
    - low-dimensional residual refinement branch
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fixed_medical_text_prior import FixedMedicalTextExpertPrior


# -----------------------------------------------------------------------------
# 1. TG-ECNet-style task prompt
# -----------------------------------------------------------------------------

class TaskPromptGenerator(nn.Module):
    """TG-ECNet-style CondNet + GAP + prompt dictionary.

    Input:
        x: [B, C, H, W]

    Output:
        prompt:       [B, atom_dim]
        atom_weights: [B, atom_num]
    """

    def __init__(
        self,
        in_channels: int,
        atom_num: int = 32,
        atom_dim: int = 256,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()

        # Matches the official TG-ECNet Taskprompt CondNet structure.
        self.cond_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 32, kernel_size=1),
        )

        self.atom_logits = nn.Linear(32, atom_num)
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.cond_net(x)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)  # [B, 32]

        atom_weights = F.softmax(self.atom_logits(z), dim=-1)  # [B, atom_num]
        prompt = atom_weights @ self.dictionary                # [B, atom_dim]
        prompt = self.act(prompt)
        return prompt, atom_weights


# -----------------------------------------------------------------------------
# 2. Sparse experts
# -----------------------------------------------------------------------------

class ExpertMLP(nn.Module):
    """One MLP expert: D -> ratio*D -> D."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(round(dim * mlp_ratio))
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SharedExpertBank(nn.Module):
    """One expert bank shared by all scales.

    Sparse dispatch is token-wise. Each token only executes experts selected by
    its Top-K gate rather than evaluating all experts densely.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 6,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [ExpertMLP(dim, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, tokens: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [N, D]
            gates:  [N, E], sparse after Top-K (rows sum to 1)
        Returns:
            out:    [N, D]
        """
        n_tokens, dim = tokens.shape
        out = tokens.new_zeros((n_tokens, dim))

        for expert_id, expert in enumerate(self.experts):
            token_idx = torch.nonzero(gates[:, expert_id] > 0, as_tuple=False).flatten()
            if token_idx.numel() == 0:
                continue

            expert_in = tokens.index_select(0, token_idx)
            expert_out = expert(expert_in)

            weight = gates.index_select(0, token_idx)[:, expert_id].unsqueeze(1)
            weighted_out = expert_out * weight.to(expert_out.dtype)
            out = out.index_add(0, token_idx, weighted_out)

        return out


# -----------------------------------------------------------------------------
# 3. Stable Noisy Top-K router
# -----------------------------------------------------------------------------

def _cv_squared(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Squared coefficient of variation used by the MoE balance loss."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e6, neginf=0.0)
    if x.numel() <= 1:
        return x.new_zeros(())
    mean = x.mean()
    var = x.var(unbiased=False)
    return var / (mean.square() + eps)


def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
    x = x.clamp(-10.0, 10.0)
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class NoisyTopKRouter(nn.Module):
    """Scale-specific task-aware router, TG-ECNet style.

    Router input is [token || projected_task_prompt], hence 2*D.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 6,
        top_k: int = 2,
        noisy_gating: bool = True,
        noise_epsilon: float = 1e-2,
    ) -> None:
        super().__init__()
        if not (1 <= top_k <= num_experts):
            raise ValueError(f"top_k must be in [1, num_experts], got {top_k}/{num_experts}")

        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon

        self.w_gate = nn.Parameter(torch.empty(input_dim, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(input_dim, num_experts))
        nn.init.normal_(self.w_gate, mean=0.0, std=0.02)

    def _prob_in_top_k(
        self,
        clean_logits: torch.Tensor,
        noisy_logits: torch.Tensor,
        noise_std: torch.Tensor,
        noisy_top_values: torch.Tensor,
    ) -> torch.Tensor:
        """Expected expert load under noisy Top-K, following sparse MoE practice."""
        batch = clean_logits.size(0)
        m = noisy_top_values.size(1)  # k+1 when k < num_experts
        flat_top = noisy_top_values.reshape(-1)

        threshold_pos_if_in = torch.arange(batch, device=clean_logits.device) * m + self.top_k
        threshold_if_in = flat_top.index_select(0, threshold_pos_if_in).unsqueeze(1)

        is_in = noisy_logits > threshold_if_in

        threshold_pos_if_out = threshold_pos_if_in - 1
        threshold_if_out = flat_top.index_select(0, threshold_pos_if_out).unsqueeze(1)

        safe_std = torch.nan_to_num(noise_std.float(), nan=1.0, posinf=10.0, neginf=1e-3)
        safe_std = safe_std.clamp(1e-3, 10.0)

        z_in = (clean_logits.float() - threshold_if_in.float()) / safe_std
        z_out = (clean_logits.float() - threshold_if_out.float()) / safe_std

        prob_if_in = _standard_normal_cdf(z_in)
        prob_if_out = _standard_normal_cdf(z_out)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def forward(
        self,
        router_input: torch.Tensor,
        expert_logit_prior: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Route in FP32 for numerical stability. Gradients still flow to input/projections.
        x = torch.nan_to_num(router_input.float(), nan=0.0, posinf=20.0, neginf=-20.0)

        clean_logits_visual = x @ self.w_gate.float()
        clean_logits_visual = torch.nan_to_num(
            clean_logits_visual, nan=0.0, posinf=20.0, neginf=-20.0
        )
        clean_logits_visual = clean_logits_visual.clamp(-20.0, 20.0)

        if expert_logit_prior is None:
            clean_logits = clean_logits_visual
        else:
            # Additive text expert prior with scale matching; no learnable scalar.
            # expert_logit_prior: [N, num_experts]
            if expert_logit_prior.ndim != 2:
                raise ValueError(
                    f'expert_logit_prior must be [N,E], got shape={tuple(expert_logit_prior.shape)}'
                )
            if expert_logit_prior.shape != clean_logits_visual.shape:
                raise ValueError(
                    'expert_logit_prior shape mismatch: '
                    f'prior={tuple(expert_logit_prior.shape)} '
                    f'visual={tuple(clean_logits_visual.shape)}'
                )
            text = expert_logit_prior.float()
            text = torch.nan_to_num(text, nan=0.0, posinf=20.0, neginf=-20.0)
            text_centered = text - text.mean(dim=-1, keepdim=True)
            text_std = text_centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            text_normalized = text_centered / text_std
            visual_std = clean_logits_visual.detach().std(
                dim=-1, keepdim=True, unbiased=False
            ).clamp_min(1e-4)
            matched_text_prior = text_normalized * visual_std
            matched_text_prior = torch.nan_to_num(
                matched_text_prior, nan=0.0, posinf=20.0, neginf=-20.0
            )
            if not torch.isfinite(matched_text_prior).all():
                raise RuntimeError('matched text expert prior contains NaN/Inf')
            clean_logits = clean_logits_visual + matched_text_prior
            clean_logits = torch.nan_to_num(clean_logits, nan=0.0, posinf=20.0, neginf=-20.0)
            clean_logits = clean_logits.clamp(-20.0, 20.0)

        if self.training and self.noisy_gating:
            raw_noise = x @ self.w_noise.float()
            raw_noise = torch.nan_to_num(raw_noise, nan=0.0, posinf=20.0, neginf=-20.0)
            raw_noise = raw_noise.clamp(-20.0, 20.0)

            noise_std = F.softplus(raw_noise) + self.noise_epsilon
            noise_std = torch.nan_to_num(noise_std, nan=1.0, posinf=10.0, neginf=1e-3)
            noise_std = noise_std.clamp(1e-3, 10.0)

            logits = clean_logits + torch.randn_like(clean_logits) * noise_std
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            logits = logits.clamp(-20.0, 20.0)
        else:
            noise_std = None
            logits = clean_logits

        # Keep k+1 values during training so expected-load estimation matches
        # the classic noisy Top-K formulation.
        k_plus_one = min(self.top_k + 1, self.num_experts)
        top_values, top_indices = logits.topk(k_plus_one, dim=1)

        top_k_values = top_values[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = F.softmax(top_k_values, dim=1)

        gates = torch.zeros_like(logits)
        gates = gates.scatter(1, top_k_indices, top_k_gates)

        importance = gates.sum(dim=0)

        if (
            self.training
            and self.noisy_gating
            and self.top_k < self.num_experts
            and noise_std is not None
        ):
            load = self._prob_in_top_k(clean_logits, logits, noise_std, top_values).sum(dim=0)
        else:
            load = (gates > 0).sum(dim=0).float()

        importance = torch.nan_to_num(importance.float(), nan=0.0, posinf=1e6, neginf=0.0)
        load = torch.nan_to_num(load.float(), nan=0.0, posinf=1e6, neginf=0.0)

        balance_loss = _cv_squared(importance) + _cv_squared(load)
        balance_loss = torch.nan_to_num(balance_loss, nan=0.0, posinf=1e3, neginf=0.0)

        stats = {
            "importance": importance.detach(),
            "load": load.detach(),
        }
        return gates, balance_loss, stats


# -----------------------------------------------------------------------------
# 4. One scale adapter + cross-scale shared expert bank
# -----------------------------------------------------------------------------

class ScaleAdapter(nn.Module):
    """Scale-specific projection, prompt generation, and router.

    The actual expert MLPs are NOT here; they live in one SharedExpertBank.
    """

    def __init__(
        self,
        in_channels: int,
        expert_dim: int,
        atom_num: int,
        atom_dim: int,
        prompt_hidden_channels: int,
        num_experts: int,
        top_k: int,
        noisy_gating: bool,
        noise_epsilon: float,
    ) -> None:
        super().__init__()

        # Only the residual correction branch is compressed to expert_dim.
        # The original Stage-1 feature remains untouched on the skip path.
        self.in_proj = nn.Conv2d(in_channels, expert_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(expert_dim, in_channels, kernel_size=1, bias=False)

        self.prompt = TaskPromptGenerator(
            in_channels=in_channels,
            atom_num=atom_num,
            atom_dim=atom_dim,
            hidden_channels=prompt_hidden_channels,
        )
        self.prompt_proj = nn.Linear(atom_dim, expert_dim)

        self.router = NoisyTopKRouter(
            input_dim=2 * expert_dim,
            num_experts=num_experts,
            top_k=top_k,
            noisy_gating=noisy_gating,
            noise_epsilon=noise_epsilon,
        )


@dataclass
class CrossScaleTaskMoEOutput:
    features: List[torch.Tensor]
    balance_loss: torch.Tensor
    stats: Dict[str, torch.Tensor]


class CrossScaleSharedTaskMoE(nn.Module):
    """All-scale TaskMoE with one shared expert bank.

    Forward path for each scale s:
        F_s^base
          -> scale-specific 1x1 projection -> X_s in shared expert space
          -> scale-specific TaskPrompt(F_s^base)
          -> scale-specific router([token, prompt])
          -> SAME shared expert bank for all scales
          -> scale-specific out projection -> Delta F_s
          -> residual:
               zero_start: F_s^out = F_s^base + beta_s * Delta F_s
               paper:      F_s^out = F_s^base + Delta F_s

    This keeps the all-stage paradigm while replacing four independent expert
    banks with one shared bank.
    """

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        expert_dim: int = 128,
        num_experts: int = 6,
        top_k: int = 2,
        atom_num: int = 32,
        atom_dim: int = 256,
        prompt_hidden_channels: int = 64,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        noisy_gating: bool = True,
        noise_epsilon: float = 1e-2,
        balance_loss_weight: float = 0.1,
        residual_mode: str = "zero_start",
        zero_start: bool | None = None,
        use_text_prior: bool = False,
        text_model_path: Optional[str] = None,
        text_tower_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Backward-compatible alias: zero_start=True/False -> residual_mode.
        if zero_start is not None:
            residual_mode = "zero_start" if bool(zero_start) else "paper"
        residual_mode = str(residual_mode).strip().lower()
        if residual_mode not in {"zero_start", "paper"}:
            raise ValueError(
                f"residual_mode must be 'zero_start' or 'paper', got {residual_mode!r}"
            )

        self.channels = tuple(channels)
        self.num_scales = len(self.channels)
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_weight = float(balance_loss_weight)
        self.residual_mode = residual_mode
        self.use_text_prior = bool(use_text_prior)

        self.scale_adapters = nn.ModuleList(
            [
                ScaleAdapter(
                    in_channels=c,
                    expert_dim=expert_dim,
                    atom_num=atom_num,
                    atom_dim=atom_dim,
                    prompt_hidden_channels=prompt_hidden_channels,
                    num_experts=num_experts,
                    top_k=top_k,
                    noisy_gating=noisy_gating,
                    noise_epsilon=noise_epsilon,
                )
                for c in self.channels
            ]
        )

        # The core change: exactly ONE expert bank is shared by S1/S2/S3/S4.
        self.shared_expert_bank = SharedExpertBank(
            dim=expert_dim,
            num_experts=num_experts,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        if self.residual_mode == "zero_start":
            self.beta = nn.Parameter(torch.zeros(self.num_scales))
        else:
            # paper: no learnable residual scale; F_out = F_base + DeltaF.
            self.register_parameter("beta", None)

        self.text_prior = None
        if self.use_text_prior:
            self.text_prior = FixedMedicalTextExpertPrior(
                num_experts=num_experts,
                text_model_path=text_model_path,
                text_tower_path=text_tower_path,
            )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        route: Optional[str] = None,
        pet_available: Optional[torch.Tensor] = None,
    ) -> CrossScaleTaskMoEOutput:
        if len(features) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} scales, got {len(features)}. "
                f"Expected channels={self.channels}."
            )

        out_features: List[torch.Tensor] = []
        total_balance = features[0].new_zeros((), dtype=torch.float32)
        stats: Dict[str, torch.Tensor] = {}

        text_expert_logits = None
        if self.use_text_prior:
            if self.text_prior is None:
                raise RuntimeError('use_text_prior=True but text_prior module is missing')
            if route is None:
                raise ValueError(
                    'use_text_prior=True requires route in {full, missing, auto}'
                )
            text_expert_logits, text_stats = self.text_prior(
                batch_size=int(features[0].shape[0]),
                route=route,
                pet_available=pet_available,
                device=features[0].device,
            )
            for k, v in text_stats.items():
                stats[k] = v
        else:
            stats['text_prior_enabled'] = torch.tensor(0.0, device=features[0].device)

        for scale_idx, (feat, adapter, expected_c) in enumerate(
            zip(features, self.scale_adapters, self.channels), start=1
        ):
            if feat.ndim != 4:
                raise ValueError(f"S{scale_idx} must be BCHW, got shape={tuple(feat.shape)}")
            if feat.shape[1] != expected_c:
                raise ValueError(
                    f"S{scale_idx} channel mismatch: expected {expected_c}, got {feat.shape[1]}"
                )

            b, _, h, w = feat.shape
            feat_dtype = feat.dtype

            # Residual branch only: project into a common expert space.
            x_shared = adapter.in_proj(feat)  # [B, D, H, W]
            d = x_shared.shape[1]

            # TG-ECNet-style prompt is generated from the original scale feature.
            prompt, atom_weights = adapter.prompt(feat)       # [B, atom_dim]
            prompt = adapter.prompt_proj(prompt).float()      # [B, D]

            # Tokenize shared-space feature.
            tokens = x_shared.permute(0, 2, 3, 1).reshape(-1, d)  # [BHW, D]

            # Repeat one sample-level task prompt for all tokens of that sample.
            prompt_tokens = (
                prompt[:, None, None, :]
                .expand(b, h, w, d)
                .reshape(-1, d)
            )

            router_input = torch.cat([tokens.float(), prompt_tokens], dim=1)  # [BHW, 2D]

            expert_logit_prior = None
            if text_expert_logits is not None:
                # Sample-level [B,6] -> token-level [BHW,6], shared across scales.
                expert_logit_prior = (
                    text_expert_logits[:, None, None, :]
                    .expand(b, h, w, self.num_experts)
                    .reshape(-1, self.num_experts)
                    .contiguous()
                )

            gates, raw_balance, router_stats = adapter.router(
                router_input,
                expert_logit_prior=expert_logit_prior,
            )

            # Sparse shared experts. Run expert arithmetic in FP32 for stability,
            # then cast the residual correction back to the feature dtype.
            expert_tokens = self.shared_expert_bank(tokens.float(), gates.float())
            expert_map = expert_tokens.reshape(b, h, w, d).permute(0, 3, 1, 2).contiguous()
            expert_map = expert_map.to(dtype=adapter.out_proj.weight.dtype)

            delta = adapter.out_proj(expert_map)
            delta = delta.to(feat_dtype)

            if self.residual_mode == "zero_start":
                beta_s = self.beta[scale_idx - 1].to(dtype=feat_dtype)
                out = feat + beta_s * delta
            else:
                # paper: no external residual scaling.
                out = feat + delta
            out_features.append(out)

            # Keep the same per-scale balance regularization strength as the
            # previous independent all-scale TaskMoE experiment:
            # total = 0.1 * L_s1 + ... + 0.1 * L_s4.
            total_balance = total_balance + self.balance_loss_weight * raw_balance.float()

            prefix = f"s{scale_idx}"
            stats[f"{prefix}_balance_raw"] = raw_balance.detach().float()
            if self.residual_mode == "zero_start":
                stats[f"{prefix}_beta"] = self.beta[scale_idx - 1].detach().float()
            # Residual strength diagnostics (detached; never used in loss).
            delta_f = delta.detach().float()
            feat_f = feat.detach().float()
            delta_abs_mean = delta_f.abs().mean()
            feat_abs_mean = feat_f.abs().mean()
            delta_l2 = torch.linalg.vector_norm(
                delta_f.reshape(delta_f.shape[0], -1), dim=1
            ).mean()
            feat_l2 = torch.linalg.vector_norm(
                feat_f.reshape(feat_f.shape[0], -1), dim=1
            ).mean()
            stats[f"{prefix}_delta_abs_mean"] = delta_abs_mean
            stats[f"{prefix}_delta_feat_ratio"] = delta_abs_mean / (feat_abs_mean + 1e-8)
            stats[f"{prefix}_delta_feat_l2_ratio"] = delta_l2 / (feat_l2 + 1e-8)
            stats[f"{prefix}_importance"] = router_stats["importance"]
            stats[f"{prefix}_load"] = router_stats["load"]
            stats[f"{prefix}_atom_entropy"] = (
                -(atom_weights.float().clamp_min(1e-8) * atom_weights.float().clamp_min(1e-8).log())
                .sum(dim=-1)
                .mean()
                .detach()
            )

        stats["balance_loss"] = total_balance.detach().float()
        stats["residual_mode"] = torch.tensor(
            0 if self.residual_mode == "zero_start" else 1,
            device=total_balance.device,
        )
        return CrossScaleTaskMoEOutput(
            features=out_features,
            balance_loss=total_balance,
            stats=stats,
        )


# -----------------------------------------------------------------------------
# 5. Helper / smoke test
# -----------------------------------------------------------------------------

def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _smoke_test() -> None:
    """CPU smoke test with reduced spatial sizes; validates both residual modes."""
    torch.manual_seed(0)

    # S4 must remain >= 9x9 because the original CondNet contains two
    # kernel=3,stride=3 convolutions without padding.
    feats = [
        torch.randn(1, 64, 72, 72),
        torch.randn(1, 128, 36, 36),
        torch.randn(1, 320, 18, 18),
        torch.randn(1, 512, 9, 9),
    ]

    zs = CrossScaleSharedTaskMoE(
        channels=(64, 128, 320, 512),
        residual_mode="zero_start",
    )
    zs.train()
    out_zs = zs(feats)
    for i, (x, y) in enumerate(zip(feats, out_zs.features), start=1):
        assert x.shape == y.shape
        assert (x - y).abs().max().item() == 0.0
    assert zs.beta is not None

    paper = CrossScaleSharedTaskMoE(
        channels=(64, 128, 320, 512),
        residual_mode="paper",
    )
    paper.train()
    out_p = paper([f.clone() for f in feats])
    for i, (x, y) in enumerate(zip(feats, out_p.features), start=1):
        assert x.shape == y.shape
        assert torch.isfinite(y).all()
        assert torch.isfinite(out_p.stats[f"s{i}_delta_feat_ratio"])
    assert paper.beta is None
    assert torch.isfinite(out_p.balance_loss)
    loss = sum(y.mean() for y in out_p.features) + out_p.balance_loss
    loss.backward()
    for n, p in paper.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n

    print("CrossScaleSharedTaskMoE smoke test: PASS")
    print(f"zero_start params: {count_trainable_parameters(zs):,}")
    print(f"paper params: {count_trainable_parameters(paper):,}")


if __name__ == "__main__":
    _smoke_test()