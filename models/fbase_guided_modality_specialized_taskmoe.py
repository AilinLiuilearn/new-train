"""FBase-Guided Cross-Scale Modality-Specialized TaskMoE.

A standalone Stage-2 module for 4-scale PET-CT segmentation under Full/Missing PET.

Research anchor
---------------
This module intentionally keeps the core routing pattern of TG-ECNet / TaskMoE:
  1) TG-ECNet-style task prompt: CondNet -> GAP -> softmax prompt dictionary.
  2) Router input = [local fused-state token || projected task prompt].
  3) Noisy Top-K sparse routing.
  4) Optional importance/load balancing loss.
  5) Sparse MLP experts.

TG-ECNet paper/code:
  Paper: https://openreview.net/pdf?id=OcFsPBXREI
  Code : https://github.com/LeeX54946/TG-ECNet

PET-CT-specific innovation
--------------------------
Stage-1 is assumed frozen and supplies, at every scale s:
  - C_s      : real CT feature.
  - P_s      : calibrated real PET feature on Full route, OR calibrated proxy PET
               feature on Missing route.
  - Fbase_s  : frozen Stage-1 base-fusion feature.

The three inputs have deliberately different roles:
  - Fbase_s is Controller + Anchor, NOT expert content.
      * generates the task prompt;
      * provides the local token used by the router;
      * is the residual anchor in Fout = Fbase + beta * DeltaF.
  - C_s / P_s are Expert Content.
      * CT experts process only projected CT tokens;
      * Real-PET experts process only projected real-PET tokens;
      * Proxy-PET experts process only projected compensated-PET tokens.

The expert bank is shared across S1/S2/S3/S4. Each scale owns its projections,
prompt generator, router, output projection, and residual beta.

Route-specific candidate masking
--------------------------------
For num_experts=6 (2 experts/group):
  Full    candidates = 2 CT + 2 Real-PET; 2 Proxy-PET experts are masked.
  Missing candidates = 2 CT + 2 Proxy-PET; 2 Real-PET experts are masked.

For num_experts=9 (3 experts/group):
  Full    candidates = 3 CT + 3 Real-PET.
  Missing candidates = 3 CT + 3 Proxy-PET.

`num_experts` must be divisible by 3. Therefore 6 -> 2/group and 9 -> 3/group
require no code changes.

Important integration assumption
--------------------------------
This file DOES NOT freeze Stage-1 itself. The caller must load the best Stage-1
checkpoint, freeze Stage-1 parameters and CPPI state, and pass detached/frozen
Stage-1 features into this module. This file only implements the independent
Stage-2 expert adapter.

Default Stage-1 feature channels:
  S1: [B,  64, 128, 128]
  S2: [B, 128,  64,  64]
  S3: [B, 320,  32,  32]
  S4: [B, 512,  16,  16]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. TG-ECNet-style task prompt
# -----------------------------------------------------------------------------


class TaskPromptGenerator(nn.Module):
    """TG-ECNet-style CondNet + GAP + learnable prompt dictionary.

    The prompt is generated ONLY from Fbase, because Fbase is the frozen
    Stage-1 fused-state representation used to condition routing.

    Args:
        in_channels: Fbase channels for this scale.
        atom_num: number of prompt atoms.
        atom_dim: dimension of each prompt atom.
        hidden_channels: CondNet hidden width.

    Input:
        fbase: [B, C, H, W]

    Returns:
        prompt: [B, atom_dim]
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

    def forward(self, fbase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if fbase.ndim != 4:
            raise ValueError(f"TaskPromptGenerator expects BCHW, got {tuple(fbase.shape)}")
        # The original TG-ECNet CondNet has two unpadded 3x3/stride-3 layers.
        # A spatial side < 9 cannot safely pass both layers.
        if fbase.shape[-2] < 9 or fbase.shape[-1] < 9:
            raise ValueError(
                "TaskPromptGenerator requires H,W >= 9 because it preserves the "
                "TG-ECNet two-layer 3x3/stride-3 CondNet. "
                f"Got HxW={fbase.shape[-2:]}"
            )

        z = self.cond_net(fbase)
        z = F.adaptive_avg_pool2d(z, output_size=1).flatten(1)  # [B, 32]
        atom_weights = F.softmax(self.atom_logits(z), dim=-1)   # [B, atom_num]
        prompt = atom_weights @ self.dictionary                 # [B, atom_dim]
        prompt = self.act(prompt)
        return prompt, atom_weights


# -----------------------------------------------------------------------------
# 2. Expert definitions
# -----------------------------------------------------------------------------


class ExpertMLP(nn.Module):
    """One lightweight expert: D -> ratio*D -> D."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(round(dim * mlp_ratio)))
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


class ModalitySpecializedSharedExpertBank(nn.Module):
    """Cross-scale shared experts with explicit modality-source specialization.

    Expert index layout for G experts/group:
        [0, G)       : CT experts
        [G, 2G)      : Real-PET experts
        [2G, 3G)     : Proxy/Missing-PET experts

    All scales call this SAME module instance.
    """

    GROUP_CT = "ct"
    GROUP_REAL = "real_pet"
    GROUP_PROXY = "proxy_pet"

    def __init__(
        self,
        dim: int,
        num_experts: int = 6,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_experts < 3 or num_experts % 3 != 0:
            raise ValueError(
                "num_experts must be divisible by 3 so CT / Real-PET / Proxy-PET "
                f"have equal capacity. Got num_experts={num_experts}."
            )
        self.dim = int(dim)
        self.num_experts = int(num_experts)
        self.experts_per_group = self.num_experts // 3

        self.experts = nn.ModuleList(
            [
                ExpertMLP(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(self.num_experts)
            ]
        )

    @property
    def ct_slice(self) -> slice:
        g = self.experts_per_group
        return slice(0, g)

    @property
    def real_pet_slice(self) -> slice:
        g = self.experts_per_group
        return slice(g, 2 * g)

    @property
    def proxy_pet_slice(self) -> slice:
        g = self.experts_per_group
        return slice(2 * g, 3 * g)

    def expert_group(self, expert_id: int) -> str:
        g = self.experts_per_group
        if 0 <= expert_id < g:
            return self.GROUP_CT
        if g <= expert_id < 2 * g:
            return self.GROUP_REAL
        if 2 * g <= expert_id < 3 * g:
            return self.GROUP_PROXY
        raise IndexError(f"Invalid expert_id={expert_id} for E={self.num_experts}")

    def active_indices(self, route: str, device: torch.device) -> torch.Tensor:
        """Return route-valid expert IDs.

        Full    -> CT + Real-PET
        Missing -> CT + Proxy-PET
        """
        route = normalize_route(route)
        g = self.experts_per_group
        ct = torch.arange(0, g, device=device, dtype=torch.long)
        if route == "full":
            pet = torch.arange(g, 2 * g, device=device, dtype=torch.long)
        else:
            pet = torch.arange(2 * g, 3 * g, device=device, dtype=torch.long)
        return torch.cat([ct, pet], dim=0)

    def forward(
        self,
        ct_tokens: torch.Tensor,
        pet_tokens: torch.Tensor,
        gates: torch.Tensor,
        route: str,
    ) -> torch.Tensor:
        """Sparse modality-aware dispatch.

        Args:
            ct_tokens:  [N, D]
            pet_tokens: [N, D] -- real PET on Full, proxy PET on Missing.
            gates:      [N, E] sparse route-masked Top-K gates.
            route:      'full' or 'missing'.

        Returns:
            out: [N, D]
        """
        route = normalize_route(route)
        if ct_tokens.ndim != 2 or pet_tokens.ndim != 2:
            raise ValueError("ct_tokens and pet_tokens must be [N,D]")
        if ct_tokens.shape != pet_tokens.shape:
            raise ValueError(
                f"CT/PET token shape mismatch: {tuple(ct_tokens.shape)} vs {tuple(pet_tokens.shape)}"
            )
        if gates.shape != (ct_tokens.shape[0], self.num_experts):
            raise ValueError(
                f"gates must be [N,{self.num_experts}], got {tuple(gates.shape)}"
            )

        n_tokens, dim = ct_tokens.shape
        out = ct_tokens.new_zeros((n_tokens, dim))

        for expert_id, expert in enumerate(self.experts):
            token_idx = torch.nonzero(gates[:, expert_id] > 0, as_tuple=False).flatten()
            if token_idx.numel() == 0:
                continue

            group = self.expert_group(expert_id)
            if group == self.GROUP_CT:
                source = ct_tokens
            elif group == self.GROUP_REAL:
                if route != "full":
                    raise RuntimeError(
                        "A Real-PET expert received nonzero gates on Missing route. "
                        "Route masking is broken."
                    )
                source = pet_tokens
            else:
                if route != "missing":
                    raise RuntimeError(
                        "A Proxy-PET expert received nonzero gates on Full route. "
                        "Route masking is broken."
                    )
                source = pet_tokens

            expert_in = source.index_select(0, token_idx)
            expert_out = expert(expert_in)
            weight = gates.index_select(0, token_idx)[:, expert_id].unsqueeze(1)
            weighted = expert_out * weight.to(expert_out.dtype)
            out.index_add_(0, token_idx, weighted)

        return out


# -----------------------------------------------------------------------------
# 3. Route-masked TG-ECNet-style Noisy Top-K router
# -----------------------------------------------------------------------------


def _cv_squared(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e6, neginf=0.0)
    if x.numel() <= 1:
        return x.new_zeros(())
    mean = x.mean()
    var = x.var(unbiased=False)
    return var / (mean.square() + eps)


def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=10.0, neginf=-10.0)
    x = x.clamp(-10.0, 10.0)
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class RouteMaskedNoisyTopKRouter(nn.Module):
    """Fbase-guided route-masked Noisy Top-K router.

    The router sees only Fbase-derived information:
        router_input = [local Fbase token || projected Fbase task prompt].

    It scores ALL experts, then restricts Top-K to route-valid candidates:
        Full    : CT + Real-PET experts.
        Missing : CT + Proxy-PET experts.

    Balance loss, when enabled, is computed ONLY over the active candidate set.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        experts_per_group: int,
        top_k: int = 2,
        noisy_gating: bool = True,
        noise_epsilon: float = 1e-2,
        enable_balance_loss: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_experts = int(num_experts)
        self.experts_per_group = int(experts_per_group)
        self.top_k = int(top_k)
        self.noisy_gating = bool(noisy_gating)
        self.noise_epsilon = float(noise_epsilon)
        self.enable_balance_loss = bool(enable_balance_loss)

        active_count = 2 * self.experts_per_group
        if not (1 <= self.top_k <= active_count):
            raise ValueError(
                f"top_k must be in [1,{active_count}] because each route exposes "
                f"2 expert groups, got top_k={self.top_k}."
            )

        self.w_gate = nn.Parameter(torch.empty(self.input_dim, self.num_experts))
        self.w_noise = nn.Parameter(torch.zeros(self.input_dim, self.num_experts))
        nn.init.normal_(self.w_gate, mean=0.0, std=0.02)

    def _active_indices(self, route: str, device: torch.device) -> torch.Tensor:
        route = normalize_route(route)
        g = self.experts_per_group
        ct = torch.arange(0, g, device=device, dtype=torch.long)
        if route == "full":
            pet = torch.arange(g, 2 * g, device=device, dtype=torch.long)
        else:
            pet = torch.arange(2 * g, 3 * g, device=device, dtype=torch.long)
        return torch.cat([ct, pet], dim=0)

    def _prob_in_top_k(
        self,
        clean_logits: torch.Tensor,
        noisy_logits: torch.Tensor,
        noise_std: torch.Tensor,
        noisy_top_values: torch.Tensor,
    ) -> torch.Tensor:
        """Expected active-expert load, following the Noisy Top-K formulation."""
        batch = clean_logits.size(0)
        m = noisy_top_values.size(1)  # k+1 when k < active expert count
        flat_top = noisy_top_values.reshape(-1)

        threshold_pos_if_in = (
            torch.arange(batch, device=clean_logits.device, dtype=torch.long) * m + self.top_k
        )
        threshold_if_in = flat_top.index_select(0, threshold_pos_if_in).unsqueeze(1)
        is_in = noisy_logits > threshold_if_in

        threshold_pos_if_out = threshold_pos_if_in - 1
        threshold_if_out = flat_top.index_select(0, threshold_pos_if_out).unsqueeze(1)

        safe_std = torch.nan_to_num(
            noise_std.float(), nan=1.0, posinf=10.0, neginf=1e-3
        ).clamp(1e-3, 10.0)

        z_in = (clean_logits.float() - threshold_if_in.float()) / safe_std
        z_out = (clean_logits.float() - threshold_if_out.float()) / safe_std
        prob_if_in = _standard_normal_cdf(z_in)
        prob_if_out = _standard_normal_cdf(z_out)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def forward(
        self,
        router_input: torch.Tensor,
        route: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        route = normalize_route(route)
        if router_input.ndim != 2 or router_input.shape[1] != self.input_dim:
            raise ValueError(
                f"router_input must be [N,{self.input_dim}], got {tuple(router_input.shape)}"
            )

        # Keep routing arithmetic in FP32 for AMP stability.
        x = torch.nan_to_num(
            router_input.float(), nan=0.0, posinf=20.0, neginf=-20.0
        )
        clean_full = x @ self.w_gate.float()  # [N,E]
        clean_full = torch.nan_to_num(
            clean_full, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)

        active_idx = self._active_indices(route, x.device)
        clean = clean_full.index_select(1, active_idx)  # [N,E_active]
        active_count = clean.shape[1]

        if self.training and self.noisy_gating:
            raw_noise_full = x @ self.w_noise.float()
            raw_noise_full = torch.nan_to_num(
                raw_noise_full, nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(-20.0, 20.0)
            raw_noise = raw_noise_full.index_select(1, active_idx)
            noise_std = F.softplus(raw_noise) + self.noise_epsilon
            noise_std = torch.nan_to_num(
                noise_std, nan=1.0, posinf=10.0, neginf=1e-3
            ).clamp(1e-3, 10.0)
            noisy = clean + torch.randn_like(clean) * noise_std
            logits = torch.nan_to_num(
                noisy, nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(-20.0, 20.0)
        else:
            noise_std = None
            logits = clean

        # k+1 is kept only when it exists, matching classical Noisy Top-K load estimation.
        k_plus_one = min(self.top_k + 1, active_count)
        top_values, top_local_idx = logits.topk(k_plus_one, dim=1)
        top_k_values = top_values[:, : self.top_k]
        top_k_local_idx = top_local_idx[:, : self.top_k]
        top_k_gates = F.softmax(top_k_values, dim=1)

        # Convert local active-set IDs back to global expert IDs.
        top_k_global_idx = active_idx[top_k_local_idx]
        gates = torch.zeros(
            (x.shape[0], self.num_experts), device=x.device, dtype=top_k_gates.dtype
        )
        gates.scatter_(1, top_k_global_idx, top_k_gates)

        # Active-set statistics. Inactive groups are explicitly zero.
        importance_active = top_k_gates.new_zeros(active_count)
        importance_active.scatter_add_(
            0,
            top_k_local_idx.reshape(-1),
            top_k_gates.reshape(-1),
        )

        if (
            self.training
            and self.noisy_gating
            and self.top_k < active_count
            and noise_std is not None
        ):
            load_active = self._prob_in_top_k(
                clean, logits, noise_std, top_values
            ).sum(dim=0)
        else:
            local_sparse = torch.zeros_like(logits)
            local_sparse.scatter_(1, top_k_local_idx, top_k_gates)
            load_active = (local_sparse > 0).sum(dim=0).float()

        importance_active = torch.nan_to_num(
            importance_active.float(), nan=0.0, posinf=1e6, neginf=0.0
        )
        load_active = torch.nan_to_num(
            load_active.float(), nan=0.0, posinf=1e6, neginf=0.0
        )

        if self.enable_balance_loss:
            balance_loss = _cv_squared(importance_active) + _cv_squared(load_active)
            balance_loss = torch.nan_to_num(
                balance_loss, nan=0.0, posinf=1e3, neginf=0.0
            )
        else:
            balance_loss = x.new_zeros(())

        importance_full = x.new_zeros(self.num_experts)
        load_full = x.new_zeros(self.num_experts)
        importance_full.index_copy_(0, active_idx, importance_active.to(x.dtype))
        load_full.index_copy_(0, active_idx, load_active.to(x.dtype))

        # Entropy of the actual sparse Top-K gates. Lower means more decisive routing.
        sparse_entropy = -(
            top_k_gates.float().clamp_min(1e-8)
            * top_k_gates.float().clamp_min(1e-8).log()
        ).sum(dim=1).mean()

        stats = {
            "importance": importance_full.detach(),
            "load": load_full.detach(),
            "active_indices": active_idx.detach(),
            "routing_entropy": sparse_entropy.detach(),
        }
        return gates, balance_loss, stats


# -----------------------------------------------------------------------------
# 4. Scale-specific controller/content projections
# -----------------------------------------------------------------------------


class ScaleControllerAdapter(nn.Module):
    """Scale-specific projections + Fbase prompt/router.

    Fbase is Controller only. CT/PET projections produce Expert Content.
    The experts themselves live in ONE shared expert bank outside this class.
    """

    def __init__(
        self,
        in_channels: int,
        expert_dim: int,
        atom_num: int,
        atom_dim: int,
        prompt_hidden_channels: int,
        num_experts: int,
        experts_per_group: int,
        top_k: int,
        noisy_gating: bool,
        noise_epsilon: float,
        enable_balance_loss: bool,
    ) -> None:
        super().__init__()
        self.ct_proj = nn.Conv2d(in_channels, expert_dim, kernel_size=1, bias=False)
        self.pet_proj = nn.Conv2d(in_channels, expert_dim, kernel_size=1, bias=False)
        self.fbase_proj = nn.Conv2d(in_channels, expert_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(expert_dim, in_channels, kernel_size=1, bias=False)

        self.prompt = TaskPromptGenerator(
            in_channels=in_channels,
            atom_num=atom_num,
            atom_dim=atom_dim,
            hidden_channels=prompt_hidden_channels,
        )
        self.prompt_proj = nn.Linear(atom_dim, expert_dim)

        self.router = RouteMaskedNoisyTopKRouter(
            input_dim=2 * expert_dim,
            num_experts=num_experts,
            experts_per_group=experts_per_group,
            top_k=top_k,
            noisy_gating=noisy_gating,
            noise_epsilon=noise_epsilon,
            enable_balance_loss=enable_balance_loss,
        )


# -----------------------------------------------------------------------------
# 5. Main module
# -----------------------------------------------------------------------------


@dataclass
class ModalitySpecializedTaskMoEOutput:
    """Output container.

    features:
        Refined multi-scale features, same shapes as fbase_features.
    balance_loss:
        Already multiplied by balance_loss_weight and summed across scales.
        Exact zero when enable_balance_loss=False.
    stats:
        Detached diagnostics for analysis/logging.
    """

    features: List[torch.Tensor]
    balance_loss: torch.Tensor
    stats: Dict[str, torch.Tensor]


class FBaseGuidedCrossScaleModalitySpecializedTaskMoE(nn.Module):
    """Cross-scale shared modality-specialized Stage-2 TaskMoE.

    Args:
        channels:
            Per-scale channels. Default matches the PET-CT project.
        expert_dim:
            Common expert-space width.
        num_experts:
            Total experts, divisible by 3. Use 6 -> 2/group or 9 -> 3/group.
        top_k:
            Number of experts selected per token among route-valid candidates.
        atom_num/atom_dim:
            TG-ECNet-style prompt dictionary dimensions.
        mlp_ratio:
            Expert MLP expansion ratio.
        enable_balance_loss:
            Master switch for MoE importance/load balancing loss.
        balance_loss_weight:
            Per-scale coefficient applied before summing scale losses.
        residual_mode:
            'zero_start': Fout = Fbase + beta_s*DeltaF, beta initialized to 0.
            'paper'     : Fout = Fbase + DeltaF.

    Forward:
        ct_features:    list[S1..S4] of real CT features.
        pet_features:   list[S1..S4] of calibrated real PET (Full) OR calibrated
                        proxy PET (Missing).
        fbase_features: list[S1..S4] of frozen Stage-1 fused features.
        route:          'full' or 'missing'. Batch-level route is assumed.
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
        enable_balance_loss: bool = True,
        balance_loss_weight: float = 0.1,
        residual_mode: str = "zero_start",
    ) -> None:
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels cannot be empty")
        if num_experts < 3 or num_experts % 3 != 0:
            raise ValueError(
                f"num_experts must be divisible by 3; got {num_experts}. "
                "Examples: 6 (2/group), 9 (3/group)."
            )
        experts_per_group = num_experts // 3
        active_experts_per_route = 2 * experts_per_group
        if not (1 <= top_k <= active_experts_per_route):
            raise ValueError(
                f"top_k={top_k} is invalid. With num_experts={num_experts}, each "
                f"route has {active_experts_per_route} active candidates."
            )

        residual_mode = str(residual_mode).strip().lower()
        if residual_mode not in {"zero_start", "paper"}:
            raise ValueError(
                f"residual_mode must be 'zero_start' or 'paper', got {residual_mode!r}"
            )

        self.channels = tuple(int(c) for c in channels)
        self.num_scales = len(self.channels)
        self.expert_dim = int(expert_dim)
        self.num_experts = int(num_experts)
        self.experts_per_group = int(experts_per_group)
        self.top_k = int(top_k)
        self.enable_balance_loss = bool(enable_balance_loss)
        self.balance_loss_weight = float(balance_loss_weight)
        self.residual_mode = residual_mode

        self.scale_adapters = nn.ModuleList(
            [
                ScaleControllerAdapter(
                    in_channels=c,
                    expert_dim=self.expert_dim,
                    atom_num=atom_num,
                    atom_dim=atom_dim,
                    prompt_hidden_channels=prompt_hidden_channels,
                    num_experts=self.num_experts,
                    experts_per_group=self.experts_per_group,
                    top_k=self.top_k,
                    noisy_gating=noisy_gating,
                    noise_epsilon=noise_epsilon,
                    enable_balance_loss=self.enable_balance_loss,
                )
                for c in self.channels
            ]
        )

        # Exactly ONE modality-specialized expert bank is shared by every scale.
        self.shared_expert_bank = ModalitySpecializedSharedExpertBank(
            dim=self.expert_dim,
            num_experts=self.num_experts,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        if self.residual_mode == "zero_start":
            self.beta = nn.Parameter(torch.zeros(self.num_scales))
        else:
            self.register_parameter("beta", None)

    def _validate_inputs(
        self,
        ct_features: Sequence[torch.Tensor],
        pet_features: Sequence[torch.Tensor],
        fbase_features: Sequence[torch.Tensor],
    ) -> None:
        if not (
            len(ct_features) == len(pet_features) == len(fbase_features) == self.num_scales
        ):
            raise ValueError(
                "Expected the same number of CT/PET/Fbase scales: "
                f"{self.num_scales}. Got CT={len(ct_features)}, PET={len(pet_features)}, "
                f"Fbase={len(fbase_features)}."
            )

        for idx, (ct, pet, fbase, expected_c) in enumerate(
            zip(ct_features, pet_features, fbase_features, self.channels), start=1
        ):
            for name, x in (("CT", ct), ("PET", pet), ("Fbase", fbase)):
                if x.ndim != 4:
                    raise ValueError(f"S{idx} {name} must be BCHW, got {tuple(x.shape)}")
                if x.shape[1] != expected_c:
                    raise ValueError(
                        f"S{idx} {name} channel mismatch: expected {expected_c}, "
                        f"got {x.shape[1]}"
                    )
            if ct.shape != pet.shape or ct.shape != fbase.shape:
                raise ValueError(
                    f"S{idx} CT/PET/Fbase must be aligned and have equal shape. "
                    f"Got CT={tuple(ct.shape)}, PET={tuple(pet.shape)}, "
                    f"Fbase={tuple(fbase.shape)}"
                )

    def forward(
        self,
        ct_features: Sequence[torch.Tensor],
        pet_features: Sequence[torch.Tensor],
        fbase_features: Sequence[torch.Tensor],
        route: str,
    ) -> ModalitySpecializedTaskMoEOutput:
        route = normalize_route(route)
        self._validate_inputs(ct_features, pet_features, fbase_features)

        out_features: List[torch.Tensor] = []
        stats: Dict[str, torch.Tensor] = {}
        total_balance = fbase_features[0].new_zeros((), dtype=torch.float32)

        for scale_idx, (ct, pet, fbase, adapter) in enumerate(
            zip(ct_features, pet_features, fbase_features, self.scale_adapters), start=1
        ):
            b, _, h, w = fbase.shape
            input_dtype = fbase.dtype

            # Content projections: only CT/PET will enter the experts.
            ct_map = adapter.ct_proj(ct)       # [B,D,H,W]
            pet_map = adapter.pet_proj(pet)    # [B,D,H,W]

            # Controller projection: Fbase is used for routing, not expert content.
            fbase_ctrl = adapter.fbase_proj(fbase)  # [B,D,H,W]
            d = fbase_ctrl.shape[1]

            ct_tokens = ct_map.permute(0, 2, 3, 1).reshape(-1, d)
            pet_tokens = pet_map.permute(0, 2, 3, 1).reshape(-1, d)
            fbase_tokens = fbase_ctrl.permute(0, 2, 3, 1).reshape(-1, d)

            # TG-ECNet-style sample/scale-level prompt from ORIGINAL Fbase.
            task_prompt, atom_weights = adapter.prompt(fbase)  # [B,atom_dim]
            task_prompt = adapter.prompt_proj(task_prompt).float()  # [B,D]
            prompt_tokens = (
                task_prompt[:, None, None, :]
                .expand(b, h, w, d)
                .reshape(-1, d)
            )

            # Fbase Controller: [local fused-state token || global task prompt].
            router_input = torch.cat(
                [fbase_tokens.float(), prompt_tokens], dim=1
            )  # [BHW,2D]

            gates, raw_balance, router_stats = adapter.router(
                router_input=router_input,
                route=route,
            )

            # Sparse modality-specialized experts. Arithmetic in FP32 improves AMP safety.
            expert_tokens = self.shared_expert_bank(
                ct_tokens=ct_tokens.float(),
                pet_tokens=pet_tokens.float(),
                gates=gates.float(),
                route=route,
            )
            expert_map = (
                expert_tokens.reshape(b, h, w, d)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            expert_map = expert_map.to(dtype=adapter.out_proj.weight.dtype)
            delta = adapter.out_proj(expert_map).to(dtype=input_dtype)

            if self.residual_mode == "zero_start":
                beta_s = self.beta[scale_idx - 1].to(dtype=input_dtype)
                out = fbase + beta_s * delta
                effective_delta = beta_s * delta
            else:
                out = fbase + delta
                effective_delta = delta
            out_features.append(out)

            # balance_loss returned by the module is already weighted per scale.
            if self.enable_balance_loss:
                total_balance = (
                    total_balance
                    + self.balance_loss_weight * raw_balance.float()
                )

            prefix = f"s{scale_idx}"
            stats[f"{prefix}_balance_raw"] = raw_balance.detach().float()
            stats[f"{prefix}_importance"] = router_stats["importance"]
            stats[f"{prefix}_load"] = router_stats["load"]
            stats[f"{prefix}_routing_entropy"] = router_stats["routing_entropy"]
            stats[f"{prefix}_active_indices"] = router_stats["active_indices"]

            # Prompt specialization diagnostic.
            atom_w = atom_weights.float().clamp_min(1e-8)
            stats[f"{prefix}_atom_entropy"] = (
                -(atom_w * atom_w.log()).sum(dim=-1).mean().detach()
            )

            # Residual diagnostics.
            delta_f = delta.detach().float()
            effective_f = effective_delta.detach().float()
            base_f = fbase.detach().float()
            raw_l2 = torch.linalg.vector_norm(
                delta_f.reshape(delta_f.shape[0], -1), dim=1
            ).mean()
            effective_l2 = torch.linalg.vector_norm(
                effective_f.reshape(effective_f.shape[0], -1), dim=1
            ).mean()
            base_l2 = torch.linalg.vector_norm(
                base_f.reshape(base_f.shape[0], -1), dim=1
            ).mean()
            stats[f"{prefix}_raw_delta_l2_ratio"] = raw_l2 / (base_l2 + 1e-8)
            stats[f"{prefix}_effective_delta_l2_ratio"] = effective_l2 / (base_l2 + 1e-8)
            if self.beta is not None:
                stats[f"{prefix}_beta"] = self.beta[scale_idx - 1].detach().float()

            # Group-level importance for Full/Missing interpretability.
            imp = router_stats["importance"]
            g = self.experts_per_group
            stats[f"{prefix}_importance_ct"] = imp[0:g].sum().detach()
            stats[f"{prefix}_importance_real_pet"] = imp[g : 2 * g].sum().detach()
            stats[f"{prefix}_importance_proxy_pet"] = imp[2 * g : 3 * g].sum().detach()

        if not self.enable_balance_loss:
            total_balance = fbase_features[0].new_zeros((), dtype=torch.float32)

        stats["balance_loss"] = total_balance.detach().float()
        stats["balance_enabled"] = torch.tensor(
            1.0 if self.enable_balance_loss else 0.0,
            device=total_balance.device,
        )
        stats["num_experts"] = torch.tensor(
            float(self.num_experts), device=total_balance.device
        )
        stats["experts_per_group"] = torch.tensor(
            float(self.experts_per_group), device=total_balance.device
        )
        stats["route_is_full"] = torch.tensor(
            1.0 if route == "full" else 0.0,
            device=total_balance.device,
        )

        return ModalitySpecializedTaskMoEOutput(
            features=out_features,
            balance_loss=total_balance,
            stats=stats,
        )

    def expert_layout(self) -> Dict[str, List[int]]:
        """Human-readable expert IDs for logging/config checks."""
        g = self.experts_per_group
        return {
            "ct": list(range(0, g)),
            "real_pet": list(range(g, 2 * g)),
            "proxy_pet": list(range(2 * g, 3 * g)),
        }


# -----------------------------------------------------------------------------
# 6. Utilities / smoke tests
# -----------------------------------------------------------------------------


def normalize_route(route: str) -> str:
    if not isinstance(route, str):
        raise TypeError(f"route must be str, got {type(route)}")
    route_n = route.strip().lower().replace("-", "_")
    aliases = {
        "full": "full",
        "complete": "full",
        "real": "full",
        "missing": "missing",
        "miss": "missing",
        "proxy": "missing",
        "compensated": "missing",
    }
    if route_n not in aliases:
        raise ValueError(
            f"Unsupported route={route!r}. Use 'full' or 'missing'."
        )
    return aliases[route_n]


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _make_smoke_features(
    batch: int = 1,
    channels: Sequence[int] = (64, 128, 320, 512),
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    # Every size remains >= 9 for the TG-ECNet-style prompt CondNet.
    sizes = (24, 18, 15, 12)
    ct = [torch.randn(batch, c, s, s) for c, s in zip(channels, sizes)]
    pet = [torch.randn(batch, c, s, s) for c, s in zip(channels, sizes)]
    fbase = [c + 0.2 * p for c, p in zip(ct, pet)]
    return ct, pet, fbase


def _smoke_test_one(num_experts: int, enable_balance_loss: bool) -> None:
    torch.manual_seed(7)
    channels = (64, 128, 320, 512)
    ct, pet, fbase = _make_smoke_features(channels=channels)

    model = FBaseGuidedCrossScaleModalitySpecializedTaskMoE(
        channels=channels,
        expert_dim=64,  # reduced smoke-test width; production default is 128
        num_experts=num_experts,
        top_k=2,
        enable_balance_loss=enable_balance_loss,
        balance_loss_weight=0.1,
        residual_mode="zero_start",
    )
    model.train()

    for route in ("full", "missing"):
        out = model(ct, pet, fbase, route=route)
        assert len(out.features) == 4
        for x, y in zip(fbase, out.features):
            assert x.shape == y.shape
            # beta=0 -> exact Stage-1 identity at initialization.
            assert torch.equal(x, y)

        if enable_balance_loss:
            assert torch.isfinite(out.balance_loss)
        else:
            assert out.balance_loss.item() == 0.0

        layout = model.expert_layout()
        for scale_idx in range(1, 5):
            imp = out.stats[f"s{scale_idx}_importance"]
            if route == "full":
                # Proxy experts must be exactly inactive.
                assert imp[layout["proxy_pet"]].abs().sum().item() == 0.0
            else:
                # Real-PET experts must be exactly inactive.
                assert imp[layout["real_pet"]].abs().sum().item() == 0.0

        # With beta=0, expert/router gradients are expected to be zero on the first
        # segmentation-only step; beta itself must receive finite gradients.
        loss = sum(y.mean() for y in out.features) + out.balance_loss
        model.zero_grad(set_to_none=True)
        loss.backward()
        if model.beta is not None:
            assert model.beta.grad is not None
            assert torch.isfinite(model.beta.grad).all()

    print(
        f"PASS: E={num_experts}, group={num_experts // 3}, "
        f"balance={'on' if enable_balance_loss else 'off'}, "
        f"params={count_trainable_parameters(model):,}"
    )


def smoke_test() -> None:
    """Validate the required 6->9 expert expansion and balance-loss switch."""
    _smoke_test_one(num_experts=6, enable_balance_loss=True)
    _smoke_test_one(num_experts=6, enable_balance_loss=False)
    _smoke_test_one(num_experts=9, enable_balance_loss=True)
    _smoke_test_one(num_experts=9, enable_balance_loss=False)
    print("All modality-specialized TaskMoE smoke tests passed.")


if __name__ == "__main__":
    smoke_test()
