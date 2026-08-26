"""State–Scale Factorized Shared–Private TaskMoE (SSF-SP TaskMoE).

Combines one always-on shared full MLP expert with structurally eligible
scale-private and state-private low-rank residual experts:

    y = E_shared(x) + a_scale * E_scale_s(x) + a_state * E_state_r(x)

No global Noisy Top-K and no load-balancing loss. Specialization is enforced
by data eligibility (scale / Full-Missing route), not by forcing equal traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cross_scale_shared_taskmoe import ExpertMLP, TaskPromptGenerator


class LowRankResidualExpert(nn.Module):
    """Low-rank residual expert: D -> rank -> D with zero-init output."""

    def __init__(self, dim: int, rank: int = 16) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f'private rank must be >= 1, got {rank}')
        self.fc1 = nn.Linear(dim, rank)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(rank, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class RoleMixer(nn.Module):
    """Predict private gates [a_scale, a_state] in [0, 1] via sigmoid."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(gate_bias_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FP32 mixer arithmetic for numerical stability.
        logits = self.net(x.float())
        gates = torch.sigmoid(logits)
        return gates.clamp(0.0, 1.0)


class FactorizedScaleAdapter(nn.Module):
    """Scale-specific projection + TaskPrompt (experts live outside)."""

    def __init__(
        self,
        in_channels: int,
        expert_dim: int,
        atom_num: int,
        atom_dim: int,
        prompt_hidden_channels: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, expert_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(expert_dim, in_channels, kernel_size=1, bias=False)
        self.prompt = TaskPromptGenerator(
            in_channels=in_channels,
            atom_num=atom_num,
            atom_dim=atom_dim,
            hidden_channels=prompt_hidden_channels,
        )
        self.prompt_proj = nn.Linear(atom_dim, expert_dim)


@dataclass
class StateScaleFactorizedTaskMoEOutput:
    features: List[torch.Tensor]
    aux_loss: torch.Tensor
    stats: Dict[str, torch.Tensor]
    role_context: torch.Tensor
    z_shared_maps: List[torch.Tensor]


def _resolve_state_ids(
    batch_size: int,
    route: Optional[str],
    pet_available: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Return per-sample state ids: 1=full, 0=missing."""
    route_l = None if route is None else str(route).strip().lower()
    if route_l == 'full':
        return torch.ones(batch_size, device=device, dtype=torch.long)
    if route_l == 'missing':
        return torch.zeros(batch_size, device=device, dtype=torch.long)
    if route_l == 'auto':
        if pet_available is None:
            raise ValueError("route='auto' requires pet_available")
        availability = pet_available.to(device=device).long().view(-1)
        if availability.numel() != batch_size:
            raise ValueError('pet_available must contain one state per sample')
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError('pet_available values must be 0 or 1')
        return availability
    raise ValueError(
        f"state_scale_factorized requires route in {{full, missing, auto}}, got {route!r}"
    )


class StateScaleFactorizedTaskMoE(nn.Module):
    """SSF-SP TaskMoE over S1–S4 fused Stage-1 features."""

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        expert_dim: int = 128,
        private_rank: int = 16,
        atom_num: int = 32,
        atom_dim: int = 256,
        prompt_hidden_channels: int = 64,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        beta_max: float = 1.0,
        gate_bias_init: float = -2.0,
        role_context_dim: int = 64,
        role_loss_weight: float = 0.02,
        fers_mode: str = 'both',
    ) -> None:
        super().__init__()
        if int(private_rank) not in (8, 16, 32):
            raise ValueError(
                f'taskmoe_private_rank must be in {{8,16,32}}, got {private_rank}'
            )
        fers_mode = str(fers_mode or 'both').strip().lower()
        if fers_mode not in ('both', 'scale', 'state', 'none'):
            raise ValueError(
                f'taskmoe_fers_mode must be both|scale|state|none, got {fers_mode!r}'
            )
        self.channels = tuple(channels)
        self.num_scales = len(self.channels)
        if self.num_scales != 4:
            raise ValueError(f'SSF-SP TaskMoE expects 4 scales, got {self.num_scales}')
        self.expert_dim = int(expert_dim)
        self.private_rank = int(private_rank)
        self.beta_max = float(beta_max)
        if self.beta_max <= 0:
            raise ValueError(f'beta_max must be > 0, got {self.beta_max}')
        self.role_loss_weight = float(role_loss_weight)
        self.fers_mode = fers_mode
        self.role_context_dim = int(role_context_dim)

        self.scale_adapters = nn.ModuleList(
            [
                FactorizedScaleAdapter(
                    in_channels=c,
                    expert_dim=self.expert_dim,
                    atom_num=atom_num,
                    atom_dim=atom_dim,
                    prompt_hidden_channels=prompt_hidden_channels,
                )
                for c in self.channels
            ]
        )

        # One always-on shared full MLP.
        self.shared_expert = ExpertMLP(
            self.expert_dim, mlp_ratio=mlp_ratio, dropout=dropout
        )
        # Four scale-private low-rank experts.
        self.scale_experts = nn.ModuleList(
            [
                LowRankResidualExpert(self.expert_dim, rank=self.private_rank)
                for _ in range(self.num_scales)
            ]
        )
        # Two state-private low-rank experts: index 0=missing, 1=full.
        self.state_experts = nn.ModuleList(
            [
                LowRankResidualExpert(self.expert_dim, rank=self.private_rank),
                LowRankResidualExpert(self.expert_dim, rank=self.private_rank),
            ]
        )

        self.scale_embeddings = nn.Embedding(self.num_scales, self.expert_dim)
        self.state_embeddings = nn.Embedding(2, self.expert_dim)
        mixer_in = 4 * self.expert_dim  # x || prompt || state_emb || scale_emb
        self.role_mixer = RoleMixer(
            input_dim=mixer_in,
            hidden_dim=self.expert_dim,
            gate_bias_init=gate_bias_init,
        )

        # Bounded residual: beta_s = beta_max * tanh(raw_beta_s), raw init 0.
        self.raw_beta = nn.Parameter(torch.zeros(self.num_scales))

        # FERS: tiny shared role classifiers (train-only supervision; unused at inference).
        self.scale_role_head = nn.Linear(self.expert_dim, self.num_scales)
        self.state_role_head = nn.Linear(self.expert_dim, 2)

        # Sample-level role context for optional decoder adapter (FiLM).
        self.role_context_proj = nn.Sequential(
            nn.Linear(2 + 2 + self.expert_dim, self.role_context_dim),
            nn.GELU(),
            nn.Linear(self.role_context_dim, self.role_context_dim),
        )

    def effective_beta(self) -> torch.Tensor:
        return self.beta_max * torch.tanh(self.raw_beta)

    def _apply_state_experts(
        self,
        tokens: torch.Tensor,
        state_ids: torch.Tensor,
    ) -> torch.Tensor:
        """tokens: [B,H,W,D], state_ids: [B] with 0/1."""
        b, h, w, d = tokens.shape
        flat = tokens.reshape(b, h * w, d)
        out = tokens.new_zeros(b, h * w, d)
        for state_id in (0, 1):
            mask = state_ids == state_id
            if not bool(mask.any()):
                continue
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            selected = flat.index_select(0, idx).reshape(-1, d)
            expert_out = self.state_experts[state_id](selected.float()).to(dtype=tokens.dtype)
            expert_out = expert_out.reshape(idx.numel(), h * w, d)
            out.index_copy_(0, idx, expert_out)
        return out.reshape(b, h, w, d)

    def _forward_scale(
        self,
        feat: torch.Tensor,
        adapter: FactorizedScaleAdapter,
        scale_idx: int,
        state_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b, _, h, w = feat.shape
        feat_dtype = feat.dtype
        d = self.expert_dim

        x_shared = adapter.in_proj(feat)  # [B,D,H,W]
        prompt, _atom_weights = adapter.prompt(feat)
        prompt = adapter.prompt_proj(prompt).float()  # [B,D]

        tokens = x_shared.permute(0, 2, 3, 1).contiguous()  # [B,H,W,D]
        tokens_f = tokens.float()
        prompt_tokens = (
            prompt[:, None, None, :]
            .expand(b, h, w, d)
            .contiguous()
        )
        scale_emb = (
            self.scale_embeddings.weight[scale_idx]
            .view(1, 1, 1, d)
            .expand(b, h, w, d)
            .float()
        )
        state_emb = self.state_embeddings(state_ids).float()  # [B,D]
        state_tokens = (
            state_emb[:, None, None, :]
            .expand(b, h, w, d)
            .contiguous()
        )

        mixer_in = torch.cat(
            [tokens_f, prompt_tokens, state_tokens, scale_emb],
            dim=-1,
        ).reshape(-1, 4 * d)
        gates = self.role_mixer(mixer_in).reshape(b, h, w, 2)
        a_scale = gates[..., 0:1]
        a_state = gates[..., 1:2]

        flat_tokens = tokens_f.reshape(-1, d)
        z_shared = self.shared_expert(flat_tokens).reshape(b, h, w, d)
        z_scale = self.scale_experts[scale_idx](flat_tokens).reshape(b, h, w, d)
        z_state = self._apply_state_experts(tokens_f, state_ids)

        y = z_shared + a_scale * z_scale + a_state * z_state
        expert_map = y.permute(0, 3, 1, 2).contiguous().to(dtype=adapter.out_proj.weight.dtype)
        delta = adapter.out_proj(expert_map).to(feat_dtype)

        beta_s = self.effective_beta()[scale_idx].to(dtype=feat_dtype)
        out = feat + beta_s * delta

        # Diagnostics (detached).
        delta_f = delta.detach().float()
        feat_f = feat.detach().float()
        eff_delta = (beta_s.detach().float() * delta_f)
        delta_l2 = torch.linalg.vector_norm(delta_f.reshape(b, -1), dim=1).mean()
        feat_l2 = torch.linalg.vector_norm(feat_f.reshape(b, -1), dim=1).mean()
        eff_l2 = torch.linalg.vector_norm(eff_delta.reshape(b, -1), dim=1).mean()
        z_shared_map = z_shared.permute(0, 3, 1, 2).contiguous()

        # Sample-level gate means for role_context / logging.
        scale_gate_mean = a_scale.detach().float().mean(dim=(1, 2, 3))  # [B]
        state_gate_mean = a_state.detach().float().mean(dim=(1, 2, 3))  # [B]

        stats = {
            f's{scale_idx + 1}_beta': beta_s.detach().float(),
            f's{scale_idx + 1}_raw_beta': self.raw_beta[scale_idx].detach().float(),
            f's{scale_idx + 1}_shared_output_norm': z_shared.detach().float().norm(dim=-1).mean(),
            f's{scale_idx + 1}_scale_private_output_norm': z_scale.detach().float().norm(dim=-1).mean(),
            f's{scale_idx + 1}_state_private_output_norm': z_state.detach().float().norm(dim=-1).mean(),
            f's{scale_idx + 1}_scale_gate_mean': scale_gate_mean.mean(),
            f's{scale_idx + 1}_state_gate_mean': state_gate_mean.mean(),
            f's{scale_idx + 1}_raw_delta_ratio': delta_l2 / (feat_l2 + 1e-8),
            f's{scale_idx + 1}_effective_delta_ratio': eff_l2 / (feat_l2 + 1e-8),
            # Keep legacy key aliases used by run_mdt_seg CSV logging.
            f's{scale_idx + 1}_delta_feat_ratio': (delta_f.abs().mean() / (feat_f.abs().mean() + 1e-8)),
            f's{scale_idx + 1}_delta_feat_l2_ratio': delta_l2 / (feat_l2 + 1e-8),
            f's{scale_idx + 1}_scale_gate_per_sample': scale_gate_mean,
            f's{scale_idx + 1}_state_gate_per_sample': state_gate_mean,
        }
        return out, z_shared_map, z_scale, z_state, stats

    def build_role_context(
        self,
        state_ids: torch.Tensor,
        stats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Sample-level context for Stage2 decoder adapter FiLM."""
        b = state_ids.shape[0]
        device = state_ids.device
        scale_gate = torch.stack(
            [stats[f's{i}_scale_gate_per_sample'] for i in range(1, 5)],
            dim=1,
        ).mean(dim=1)  # [B]
        state_gate = torch.stack(
            [stats[f's{i}_state_gate_per_sample'] for i in range(1, 5)],
            dim=1,
        ).mean(dim=1)  # [B]
        state_emb = self.state_embeddings(state_ids).float()
        # Aggregate: [scale_gate, state_gate, frac_full, frac_missing, state_emb]
        frac_full = state_ids.float()
        frac_missing = 1.0 - frac_full
        raw = torch.cat(
            [
                scale_gate.unsqueeze(1),
                state_gate.unsqueeze(1),
                frac_full.unsqueeze(1),
                frac_missing.unsqueeze(1),
                state_emb,
            ],
            dim=1,
        )
        ctx = self.role_context_proj(raw)
        if ctx.shape[0] != b:
            raise RuntimeError('role_context batch mismatch')
        return ctx.to(device=device)

    def _fers_from_private(
        self,
        z_scale: torch.Tensor,
        z_state: torch.Tensor,
        scale_idx: int,
        state_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-scale role CE from already-computed private expert maps.

        z_*: [B,H,W,D]; returns (scale_ce, state_ce, scale_acc, state_acc).
        """
        v_scale = z_scale.float().mean(dim=(1, 2))  # GAP over HW -> [B,D]
        v_state = z_state.float().mean(dim=(1, 2))
        scale_logits = self.scale_role_head(v_scale)
        state_logits = self.state_role_head(v_state)
        scale_target = torch.full(
            (v_scale.shape[0],),
            int(scale_idx),
            device=v_scale.device,
            dtype=torch.long,
        )
        scale_ce = F.cross_entropy(scale_logits, scale_target)
        state_ce = F.cross_entropy(state_logits, state_ids.long())
        scale_acc = (scale_logits.argmax(dim=-1) == scale_target).float().mean().detach()
        state_acc = (state_logits.argmax(dim=-1) == state_ids.long()).float().mean().detach()
        return scale_ce, state_ce, scale_acc, state_acc

    def forward(
        self,
        features: Sequence[torch.Tensor],
        route: Optional[str] = None,
        pet_available: Optional[torch.Tensor] = None,
    ) -> StateScaleFactorizedTaskMoEOutput:
        if len(features) != self.num_scales:
            raise ValueError(
                f'Expected {self.num_scales} scales, got {len(features)}'
            )
        b = int(features[0].shape[0])
        state_ids = _resolve_state_ids(
            b, route, pet_available, device=features[0].device
        )

        out_features: List[torch.Tensor] = []
        z_shared_maps: List[torch.Tensor] = []
        stats: Dict[str, torch.Tensor] = {}
        scale_ces: List[torch.Tensor] = []
        state_ces: List[torch.Tensor] = []
        scale_accs: List[torch.Tensor] = []
        state_accs: List[torch.Tensor] = []
        z_scale_pooled: List[torch.Tensor] = []
        z_state_pooled: List[torch.Tensor] = []

        for scale_idx, (feat, adapter, expected_c) in enumerate(
            zip(features, self.scale_adapters, self.channels)
        ):
            if feat.ndim != 4:
                raise ValueError(f'S{scale_idx + 1} must be BCHW, got {tuple(feat.shape)}')
            if feat.shape[1] != expected_c:
                raise ValueError(
                    f'S{scale_idx + 1} channel mismatch: expected {expected_c}, got {feat.shape[1]}'
                )
            out, z_shared_map, z_scale, z_state, scale_stats = self._forward_scale(
                feat, adapter, scale_idx, state_ids
            )
            out_features.append(out)
            z_shared_maps.append(z_shared_map)
            stats.update(scale_stats)

            # FERS uses private outputs already produced by this forward (no extra Stage1/route).
            use_fers_grad = (
                self.training
                and self.role_loss_weight > 0
                and self.fers_mode != 'none'
            )
            if use_fers_grad:
                scale_ce, state_ce, scale_acc, state_acc = self._fers_from_private(
                    z_scale, z_state, scale_idx, state_ids
                )
            else:
                with torch.no_grad():
                    scale_ce, state_ce, scale_acc, state_acc = self._fers_from_private(
                        z_scale, z_state, scale_idx, state_ids
                    )
            scale_ces.append(scale_ce)
            state_ces.append(state_ce)
            scale_accs.append(scale_acc)
            state_accs.append(state_acc)
            z_scale_pooled.append(F.normalize(z_scale.detach().float().mean(dim=(1, 2)), dim=-1))
            z_state_pooled.append(F.normalize(z_state.detach().float().mean(dim=(1, 2)), dim=-1))

        role_context = self.build_role_context(state_ids, stats)
        # Drop bulky per-sample tensors from exportable diagnostics.
        for i in range(1, 5):
            stats.pop(f's{i}_scale_gate_per_sample', None)
            stats.pop(f's{i}_state_gate_per_sample', None)
        stats['role_context'] = role_context.detach()

        # L_scale-role = mean over 4 scales of CE (equiv. 1/(4B) sum_s sum_b)
        l_scale = torch.stack(scale_ces).mean()
        l_state = torch.stack(state_ces).mean()
        if self.fers_mode == 'scale':
            l_fers = l_scale
        elif self.fers_mode == 'state':
            l_fers = l_state
        elif self.fers_mode == 'none':
            l_fers = features[0].new_zeros((), dtype=torch.float32)
        else:
            l_fers = 0.5 * (l_scale + l_state)

        if self.training and self.role_loss_weight > 0 and self.fers_mode != 'none':
            aux_loss = (self.role_loss_weight * l_fers).float()
        else:
            aux_loss = features[0].new_zeros((), dtype=torch.float32)

        stats['balance_loss'] = aux_loss.detach()  # legacy slot: now carries weighted FERS
        stats['fers_loss'] = l_fers.detach().float()
        stats['fers_scale_loss'] = l_scale.detach().float()
        stats['fers_state_loss'] = l_state.detach().float()
        stats['scale_role_acc'] = torch.stack(scale_accs).mean()
        stats['state_role_acc'] = torch.stack(state_accs).mean()
        stats['role_loss_weight'] = torch.tensor(
            self.role_loss_weight, device=features[0].device, dtype=torch.float32
        )

        # Offline-style similarity diagnostics (detached).
        # Mean pairwise cosine among 4 scale-private pooled vectors (batch-mean).
        scale_stack = torch.stack(z_scale_pooled, dim=0)  # [4,B,D]
        scale_mean = F.normalize(scale_stack.mean(dim=1), dim=-1)  # [4,D]
        scale_sim = scale_mean @ scale_mean.t()
        # Average off-diagonal similarity.
        eye = torch.eye(4, device=scale_sim.device, dtype=scale_sim.dtype)
        stats['scale_expert_mean_pairwise_cosine'] = (
            (scale_sim * (1.0 - eye)).sum() / 12.0
        ).detach()
        # State expert similarity: compare mean Full vs Missing pooled over scales
        # using current-batch activated state outputs only (same state in batch-level alt).
        state_mean = F.normalize(
            torch.stack(z_state_pooled, dim=0).mean(dim=0).mean(dim=0), dim=-1
        )
        stats['state_private_output_unit_norm'] = state_mean.norm().detach()

        return StateScaleFactorizedTaskMoEOutput(
            features=out_features,
            aux_loss=aux_loss,
            stats=stats,
            role_context=role_context,
            z_shared_maps=z_shared_maps,
        )


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _smoke_test() -> None:
    torch.manual_seed(0)
    feats = [
        torch.randn(2, 64, 72, 72),
        torch.randn(2, 128, 36, 36),
        torch.randn(2, 320, 18, 18),
        torch.randn(2, 512, 9, 9),
    ]
    moe = StateScaleFactorizedTaskMoE(channels=(64, 128, 320, 512), private_rank=16)
    moe.train()
    out = moe(feats, route='full')
    for x, y in zip(feats, out.features):
        assert x.shape == y.shape
        assert float((x - y).abs().max().item()) == 0.0
    assert float(moe.effective_beta().abs().max().item()) == 0.0

    pet_available = torch.tensor([1, 0], dtype=torch.long)
    out_auto = moe(feats, route='auto', pet_available=pet_available)
    assert torch.isfinite(out_auto.features[0]).all()

    loss = sum(y.mean() for y in out_auto.features)
    loss.backward()
    # Shared expert must receive grads; missing and full state experts for mixed batch.
    assert moe.shared_expert.fc1.weight.grad is not None
    assert moe.state_experts[0].fc1.weight.grad is not None
    assert moe.state_experts[1].fc1.weight.grad is not None
    # Only S1 scale expert should get grads from a synthetic S1-only loss.
    moe.zero_grad(set_to_none=True)
    out2 = moe(feats, route='missing')
    out2.features[0].mean().backward()
    assert moe.scale_experts[0].fc1.weight.grad is not None
    assert moe.scale_experts[3].fc1.weight.grad is None
    assert moe.state_experts[0].fc1.weight.grad is not None
    assert moe.state_experts[1].fc1.weight.grad is None

    # FERS: private experts get grads even when beta=0 (bypasses residual).
    moe.zero_grad(set_to_none=True)
    out3 = moe(feats, route='full')
    assert float(out3.aux_loss.detach()) > 0.0
    out3.aux_loss.backward()
    assert moe.scale_experts[0].fc2.weight.grad is not None
    assert float(moe.scale_experts[0].fc2.weight.grad.abs().sum()) > 0
    assert moe.state_experts[1].fc2.weight.grad is not None
    assert moe.state_experts[0].fc2.weight.grad is None
    assert 'scale_role_acc' in out3.stats and 'state_role_acc' in out3.stats

    print('StateScaleFactorizedTaskMoE smoke test: PASS')
    print(f'params: {count_trainable_parameters(moe):,}')


if __name__ == '__main__':
    _smoke_test()
