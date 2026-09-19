# -*- coding: utf-8 -*-
"""
Independent Module-2 (MRFS option B) for CT/PET fusion.

Pipeline:
    CT/PET fixed-text modulation (DGNet T-KGM inspired)
    -> MRFS IGM-Att bidirectional cross-gating
    -> F_l = C_l^I + P_l^I

Project state convention:
    1 = Full / real PET
    0 = Missing / imputed PET

References:
- MRFS, CVPR 2024, official FMB/models/modules.py:
  ChannelAttention, SpatialAttention, MixAttention, IGMAVC.
- DGNet, ACM MM 2026, official model/dgnet/pwm.py:
  TargetKnowledgeGuidedModulation and language projection pattern.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
StateLike = Union[str, int, Tensor]

DEFAULT_CT_TEXT = (
    "A CT image showing the anatomical structure and boundaries of lung tumors."
)
DEFAULT_PET_TEXT = (
    "A PET image showing bright tumor regions in the lungs."
)


# =============================================================================
# DGNet-inspired text modulation
# =============================================================================

class TextProjection(nn.Module):
    """Linear(text_dim->C) -> LayerNorm -> GELU, following DGNet style."""

    def __init__(self, text_dim: int, channels: int) -> None:
        super().__init__()
        self.text_dim = int(text_dim)
        self.channels = int(channels)
        self.proj = nn.Sequential(
            nn.Linear(self.text_dim, self.channels),
            nn.LayerNorm(self.channels),
            nn.GELU(),
        )

    def forward(self, text_feature: Tensor) -> Tensor:
        if text_feature.ndim == 1:
            text_feature = text_feature.unsqueeze(0)
        if text_feature.ndim != 2:
            raise ValueError(
                f"text_feature must be [D] or [B,D], got {tuple(text_feature.shape)}"
            )
        if int(text_feature.shape[-1]) != self.text_dim:
            raise ValueError(
                f"text dim={int(text_feature.shape[-1])}, expected {self.text_dim}"
            )
        return self.proj(text_feature)


class TargetKnowledgeGuidedGate(nn.Module):
    """
    DGNet TargetKnowledgeGuidedModulation gate-generation path:

        l = conv(x * text)
        max_out = fc2(relu(fc1(GlobalMaxPool(l))))
        text_out = fc2(relu(fc1(text)))
        gate = sigmoid(max_out + text_out)
    """

    def __init__(self, channels: int, text_reduction: int = 16) -> None:
        super().__init__()
        channels = int(channels)
        text_reduction = int(text_reduction)
        if channels <= 0 or text_reduction <= 0:
            raise ValueError("channels/text_reduction must be positive")
        hidden = max(channels // text_reduction, 1)

        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, text_map: Tensor) -> Tensor:
        if x.ndim != 4 or text_map.ndim != 4:
            raise ValueError("x/text_map must be 4D")
        if int(x.shape[1]) != int(text_map.shape[1]):
            raise ValueError("visual/text channel mismatch")
        if tuple(text_map.shape[-2:]) != (1, 1):
            raise ValueError("text_map must have spatial size 1x1")

        l = self.conv(x * text_map)
        max_out = F.adaptive_max_pool2d(l, 1)
        max_out = self.fc2(self.relu1(self.fc1(max_out)))
        text_out = self.fc2(self.relu1(self.fc1(text_map)))
        return self.sigmoid(max_out + text_out)


class ModalityTextModulator(nn.Module):
    """
    Modality-specific text modulation.

    Project adaptation fixed by design:
        X^T = X + G(X,T) * X

    No extra alpha parameter.
    """

    def __init__(self, channels: int, text_dim: int, text_reduction: int = 16) -> None:
        super().__init__()
        self.channels = int(channels)
        self.text_proj = TextProjection(text_dim, self.channels)
        self.gate = TargetKnowledgeGuidedGate(self.channels, text_reduction)

    @staticmethod
    def _expand_batch(x: Tensor, batch_size: int) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError("condition must be [1,C] or [B,C]")
        if int(x.shape[0]) == batch_size:
            return x
        if int(x.shape[0]) == 1:
            return x.expand(batch_size, -1)
        raise ValueError(
            f"condition batch={int(x.shape[0])} cannot broadcast to B={batch_size}"
        )

    def forward(
        self,
        x: Tensor,
        text_feature: Tensor,
        state_condition: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 4:
            raise ValueError("x must be [B,C,H,W]")
        B, C, _, _ = x.shape
        if int(C) != self.channels:
            raise ValueError(f"expected C={self.channels}, got {int(C)}")

        condition = self._expand_batch(self.text_proj(text_feature), B)
        if state_condition is not None:
            state_condition = self._expand_batch(state_condition, B)
            if int(state_condition.shape[-1]) != self.channels:
                raise ValueError("state-condition channel mismatch")
            condition = condition + state_condition

        condition_map = condition.unsqueeze(-1).unsqueeze(-1)
        gate = self.gate(x, condition_map)
        x_modulated = x + gate * x
        return x_modulated, gate, condition_map


# =============================================================================
# MRFS IGM-Att: source-faithful components
# =============================================================================

class MRFSChannelAttention(nn.Module):
    """Source-faithful MRFS ChannelAttention."""

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        hidden = max(self.dim * 2 // int(reduction), 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.dim * 2),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        if x1.shape != x2.shape or x1.ndim != 4:
            raise ValueError("x1/x2 must be same [B,C,H,W]")
        if int(x1.shape[1]) != self.dim:
            raise ValueError(f"expected C={self.dim}")

        B = int(x1.shape[0])
        x = torch.cat((x1, x2), dim=1)
        avg_v = self.avg_pool(x).view(B, self.dim * 2)
        max_v = self.max_pool(x).view(B, self.dim * 2)
        avg_se = self.mlp(avg_v).view(B, self.dim * 2, 1)
        max_se = self.mlp(max_v).view(B, self.dim * 2, 1)
        stat_out = self.sigmoid(avg_se + max_se).view(B, self.dim * 2, 1)
        return (
            stat_out.reshape(B, 2, self.dim, 1, 1)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )


class MRFSSpatialAttention(nn.Module):
    """Source-faithful MRFS SpatialAttention."""

    def __init__(self, kernel_size: int = 1, reduction: int = 4) -> None:
        super().__init__()
        kernel_size = int(kernel_size)
        reduction = int(reduction)
        padding = 0 if kernel_size == 1 else kernel_size // 2
        self.mlp = nn.Sequential(
            nn.Conv2d(4, 4 * reduction, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 2, kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        if x1.shape != x2.shape or x1.ndim != 4:
            raise ValueError("x1/x2 must be same [B,C,H,W]")
        B, _, H, W = x1.shape
        x1_mean = torch.mean(x1, dim=1, keepdim=True)
        x1_max, _ = torch.max(x1, dim=1, keepdim=True)
        x2_mean = torch.mean(x2, dim=1, keepdim=True)
        x2_max, _ = torch.max(x2, dim=1, keepdim=True)
        x_cat = torch.cat((x1_mean, x1_max, x2_mean, x2_max), dim=1)
        return (
            self.mlp(x_cat)
            .reshape(B, 2, 1, H, W)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )


class MRFSMixAttention(nn.Module):
    """MRFS MixAttention = ChannelAttention * SpatialAttention."""

    def __init__(
        self,
        dim: int,
        channel_reduction: int = 4,
        spatial_reduction: int = 4,
        spatial_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        self.ca_gate = MRFSChannelAttention(dim, channel_reduction)
        self.sa_gate = MRFSSpatialAttention(
            kernel_size=spatial_kernel_size,
            reduction=spatial_reduction,
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        ca_out = self.ca_gate(x1, x2)  # [2,B,C,1,1]
        sa_out = self.sa_gate(x1, x2)  # [2,B,1,H,W]
        return ca_out.mul(sa_out)      # [2,B,C,H,W]


class MRFSInteractiveGatedMixedAttention(nn.Module):
    """
    Source-faithful MRFS IGMAVC extraction.

    Official structure preserved:
        gated_weight = gate(cat(x1_flat, x2_flat))
        mix_map = MA(x1,x2)
        Gated_attention_x1 = gated_weight * mix_map[0]
        Gated_attention_x2 = (1-gated_weight) * mix_map[1]
        out_x1 = x1 + Gated_attention_x2 * x2
        out_x2 = x2 + Gated_attention_x1 * x1
    """

    def __init__(
        self,
        dim: int,
        gate_reduction: int = 4,
        channel_reduction: int = 4,
        spatial_reduction: int = 4,
        spatial_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        hidden = max(self.dim * 2 // int(gate_reduction), 1)

        self.MA = MRFSMixAttention(
            dim=self.dim,
            channel_reduction=channel_reduction,
            spatial_reduction=spatial_reduction,
            spatial_kernel_size=spatial_kernel_size,
        )
        self.gate = nn.Sequential(
            nn.Linear(self.dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.dim),
            nn.Sigmoid(),
        )

        # Match official MRFS IGMAVC initialization.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1: Tensor, x2: Tensor, return_aux: bool = False):
        if x1.shape != x2.shape or x1.ndim != 4:
            raise ValueError("x1/x2 must be same [B,C,H,W]")
        B, C, H, W = x1.shape
        if int(C) != self.dim:
            raise ValueError(f"expected C={self.dim}, got {int(C)}")

        x1_flat = x1.flatten(2).transpose(1, 2)  # [B,HW,C]
        x2_flat = x2.flatten(2).transpose(1, 2)  # [B,HW,C]
        gated_weight = self.gate(torch.cat((x1_flat, x2_flat), dim=2))
        gated_weight = (
            gated_weight.reshape(B, H, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        mix_map = self.MA(x1, x2)
        gated_attention_x1 = gated_weight * mix_map[0]
        gated_attention_x2 = (1.0 - gated_weight) * mix_map[1]

        # Exact MRFS cross-residual form.
        out_x1 = x1 + gated_attention_x2 * x2
        out_x2 = x2 + gated_attention_x1 * x1

        if not return_aux:
            return out_x1, out_x2
        return out_x1, out_x2, {
            "interactive_gate": gated_weight,
            "mix_map_x1": mix_map[0],
            "mix_map_x2": mix_map[1],
            "cross_weight_x2_to_x1": gated_attention_x2,
            "cross_weight_x1_to_x2": gated_attention_x1,
        }


# =============================================================================
# One-scale stage
# =============================================================================

class TextModulatedMRFSStage(nn.Module):
    """CT/PET text modulation -> MRFS IGM-Att -> Add.

    When use_text_modulation=False, the DGNet T-KGM text path (including the
    PET Full/Missing state conditioning, which lives inside the PET text
    condition) is bypassed entirely: raw CT/PET features go straight into
    MRFS IGM-Att, then Add. This is the no-text IGM-Att-only ablation.
    """

    def __init__(
        self,
        channels: int,
        text_dim: int,
        state_dim: int,
        text_reduction: int = 16,
        igma_gate_reduction: int = 4,
        igma_channel_reduction: int = 4,
        igma_spatial_reduction: int = 4,
        igma_spatial_kernel_size: int = 1,
        use_text_modulation: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.use_text_modulation = bool(use_text_modulation)

        self.ct_modulator = ModalityTextModulator(
            self.channels, text_dim, text_reduction
        )
        self.pet_modulator = ModalityTextModulator(
            self.channels, text_dim, text_reduction
        )

        # Only per-scale projection. s_F / s_M themselves are global/shared.
        self.state_proj = nn.Sequential(
            nn.Linear(int(state_dim), self.channels),
            nn.LayerNorm(self.channels),
            nn.GELU(),
        )

        self.interaction = MRFSInteractiveGatedMixedAttention(
            dim=self.channels,
            gate_reduction=igma_gate_reduction,
            channel_reduction=igma_channel_reduction,
            spatial_reduction=igma_spatial_reduction,
            spatial_kernel_size=igma_spatial_kernel_size,
        )

    def forward(
        self,
        ct: Tensor,
        pet: Tensor,
        ct_text_feature: Optional[Tensor] = None,
        pet_text_feature: Optional[Tensor] = None,
        state_vector: Optional[Tensor] = None,
        return_aux: bool = False,
    ):
        if ct.shape != pet.shape or ct.ndim != 4:
            raise ValueError("CT/PET must be same [B,C,H,W]")
        if int(ct.shape[1]) != self.channels:
            raise ValueError(f"expected C={self.channels}")

        B = int(ct.shape[0])
        if state_vector is not None and state_vector.ndim == 1:
            state_vector = state_vector.unsqueeze(0)
        if self.use_text_modulation:
            if state_vector is None:
                raise ValueError("text modulation requires state_vector")
            if int(state_vector.shape[0]) == 1 and B > 1:
                state_vector = state_vector.expand(B, -1)
            elif int(state_vector.shape[0]) != B:
                raise ValueError("state-vector batch mismatch")
            if ct_text_feature is None or pet_text_feature is None:
                raise ValueError("text modulation requires text features")

            state_condition = self.state_proj(state_vector)

            # CT: fixed modality text only.
            ct_t, ct_text_gate, ct_condition = self.ct_modulator(
                ct, ct_text_feature, state_condition=None
            )

            # PET: fixed modality text + Full/Missing global state.
            pet_t, pet_text_gate, pet_condition = self.pet_modulator(
                pet, pet_text_feature, state_condition=state_condition
            )
        else:
            # No-text ablation: raw CT/PET straight into IGM-Att.
            # state_vector and text features are ignored here; the Full/Missing
            # routing (real vs compensated PET) is already done upstream.
            ct_t, pet_t = ct, pet
            ct_text_gate = pet_text_gate = None
            ct_condition = pet_condition = None

        if return_aux:
            ct_i, pet_i, interaction_aux = self.interaction(
                ct_t, pet_t, return_aux=True
            )
        else:
            ct_i, pet_i = self.interaction(ct_t, pet_t, return_aux=False)

        fused = ct_i + pet_i

        if not return_aux:
            return fused
        aux: Dict[str, Tensor] = {
            "ct_interacted": ct_i,
            "pet_interacted": pet_i,
            **interaction_aux,
        }
        if self.use_text_modulation:
            aux.update({
                "ct_text_gate": ct_text_gate,
                "pet_text_gate": pet_text_gate,
                "ct_text_condition": ct_condition,
                "pet_text_condition": pet_condition,
                "ct_text_modulated": ct_t,
                "pet_text_modulated": pet_t,
            })
        return fused, aux


# =============================================================================
# Full multi-scale Module-2
# =============================================================================

class TextModulatedMRFSFusion(nn.Module):
    """
    Complete independent MRFS-based second module.

    Default channels match the current PET/CT project:
        (64, 128, 320, 512)

    Hard constraints:
      - exactly two global learnable state vectors
      - 1 = Full / real PET
      - 0 = Missing / imputed PET
      - CT uses text only
      - PET uses text + state
      - no CT/PET text semantic fusion
      - no expert/router
      - no extra alpha
      - source-faithful MRFS IGM-Att
      - final F_l = C_l^I + P_l^I
      - use_text_modulation=False bypasses the whole DGNet text path
        (including PET s_F/s_M conditioning) for the no-text ablation;
        raw CT/PET go straight into MRFS IGM-Att
    """

    FULL = 1
    MISSING = 0

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        text_dim: int = 512,
        state_dim: int = 128,
        text_reduction: int = 16,
        igma_gate_reduction: int = 4,
        igma_channel_reduction: int = 4,
        igma_spatial_reduction: int = 4,
        igma_spatial_kernel_size: int = 1,
        use_text_modulation: bool = True,
        ct_text_prior: Optional[Tensor] = None,
        pet_text_prior: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.text_dim = int(text_dim)
        self.state_dim = int(state_dim)
        self.use_text_modulation = bool(use_text_modulation)

        if not self.channels:
            raise ValueError("channels cannot be empty")

        # Exactly two GLOBAL states, shared by all scales.
        # In no-text mode they are unused but kept for checkpoint compat.
        self.state_vectors = nn.Embedding(2, self.state_dim)
        nn.init.trunc_normal_(self.state_vectors.weight, std=0.02)

        self.stages = nn.ModuleList([
            TextModulatedMRFSStage(
                channels=c,
                text_dim=self.text_dim,
                state_dim=self.state_dim,
                text_reduction=text_reduction,
                igma_gate_reduction=igma_gate_reduction,
                igma_channel_reduction=igma_channel_reduction,
                igma_spatial_reduction=igma_spatial_reduction,
                igma_spatial_kernel_size=igma_spatial_kernel_size,
                use_text_modulation=self.use_text_modulation,
            )
            for c in self.channels
        ])

        self.register_buffer("ct_text_prior", torch.empty(0), persistent=True)
        self.register_buffer("pet_text_prior", torch.empty(0), persistent=True)

        if self.use_text_modulation:
            if ct_text_prior is not None or pet_text_prior is not None:
                if ct_text_prior is None or pet_text_prior is None:
                    raise ValueError("CT/PET text priors must be provided together")
                self.set_text_priors(ct_text_prior, pet_text_prior)
        elif ct_text_prior is not None or pet_text_prior is not None:
            raise ValueError(
                "use_text_modulation=False must not receive text priors"
            )

    @property
    def s_F(self) -> Tensor:
        return self.state_vectors.weight[self.FULL]

    @property
    def s_M(self) -> Tensor:
        return self.state_vectors.weight[self.MISSING]

    @torch.no_grad()
    def set_text_priors(
        self,
        ct_text_feature: Tensor,
        pet_text_feature: Tensor,
    ) -> None:
        self.ct_text_prior = self._canonicalize_text(ct_text_feature, "CT")
        self.pet_text_prior = self._canonicalize_text(pet_text_feature, "PET")

    def _canonicalize_text(self, x: Tensor, name: str) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or int(x.shape[0]) != 1:
            raise ValueError(
                f"{name} prior must be [D] or [1,D], got {tuple(x.shape)}"
            )
        if int(x.shape[-1]) != self.text_dim:
            raise ValueError(
                f"{name} text dim={int(x.shape[-1])}, expected {self.text_dim}"
            )
        return x.detach().clone().float()

    def has_text_priors(self) -> bool:
        return self.ct_text_prior.numel() > 0 and self.pet_text_prior.numel() > 0

    def _state_ids(
        self,
        state: StateLike,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if isinstance(state, str):
            aliases = {
                "full": self.FULL,
                "real": self.FULL,
                "present": self.FULL,
                "missing": self.MISSING,
                "imputed": self.MISSING,
            }
            key = state.strip().lower()
            if key not in aliases:
                raise ValueError(f"unknown state={state!r}")
            return torch.full(
                (batch_size,), aliases[key], dtype=torch.long, device=device
            )

        if isinstance(state, int):
            if state not in (self.MISSING, self.FULL):
                raise ValueError("state int must be 0 (Missing) or 1 (Full)")
            return torch.full(
                (batch_size,), state, dtype=torch.long, device=device
            )

        if torch.is_tensor(state):
            ids = state.to(device=device, dtype=torch.long)
            if ids.ndim == 0:
                ids = ids.repeat(batch_size)
            elif ids.ndim == 1 and ids.numel() == 1:
                ids = ids.repeat(batch_size)
            elif ids.ndim == 1 and ids.numel() == batch_size:
                pass
            else:
                raise ValueError("state tensor must be scalar, [1], or [B]")
            if torch.any((ids != self.MISSING) & (ids != self.FULL)):
                raise ValueError("state tensor may contain only 0 or 1")
            return ids

        raise TypeError(f"unsupported state type: {type(state)}")

    def forward(
        self,
        ct_feats: Sequence[Tensor],
        pet_feats: Sequence[Tensor],
        state: StateLike,
        ct_text_feature: Optional[Tensor] = None,
        pet_text_feature: Optional[Tensor] = None,
        return_aux: bool = False,
    ):
        if len(ct_feats) != len(self.stages) or len(pet_feats) != len(self.stages):
            raise ValueError(
                f"expected {len(self.stages)} CT/PET scales, got "
                f"{len(ct_feats)} and {len(pet_feats)}"
            )
        if len(ct_feats) == 0:
            raise ValueError("feature lists cannot be empty")

        B = int(ct_feats[0].shape[0])
        device = ct_feats[0].device

        for idx, (ct, pet, expected_c) in enumerate(
            zip(ct_feats, pet_feats, self.channels)
        ):
            if ct.shape != pet.shape:
                raise ValueError(
                    f"S{idx+1}: CT/PET mismatch {tuple(ct.shape)} vs {tuple(pet.shape)}"
                )
            if ct.ndim != 4:
                raise ValueError(f"S{idx+1}: features must be [B,C,H,W]")
            if int(ct.shape[0]) != B:
                raise ValueError(f"S{idx+1}: batch mismatch")
            if int(ct.shape[1]) != expected_c:
                raise ValueError(
                    f"S{idx+1}: expected C={expected_c}, got {int(ct.shape[1])}"
                )

        if ct_text_feature is None or pet_text_feature is None:
            if not self.use_text_modulation:
                ct_text_feature = None
                pet_text_feature = None
            elif not self.has_text_priors():
                raise RuntimeError(
                    "text priors missing: call set_text_priors(...) once or "
                    "pass both text features to forward()"
                )
            else:
                ct_text_feature = self.ct_text_prior
                pet_text_feature = self.pet_text_prior

        if self.use_text_modulation:
            ct_text_feature = ct_text_feature.to(
                device=device, dtype=ct_feats[0].dtype
            )
            pet_text_feature = pet_text_feature.to(
                device=device, dtype=pet_feats[0].dtype
            )
            state_ids = self._state_ids(state, B, device)
            state_vector = self.state_vectors(state_ids)
        else:
            # No-text ablation: no CLIP features, no s_F/s_M lookup.
            # Full/Missing routing is already encoded in pet_feats itself.
            ref = ct_feats[0]
            state_ids = ref.new_zeros((B,), dtype=torch.long)
            state_vector = None

        fused_feats: List[Tensor] = []
        aux_stages: List[Dict[str, Tensor]] = []

        for stage, ct, pet in zip(self.stages, ct_feats, pet_feats):
            if return_aux:
                fused, aux = stage(
                    ct=ct,
                    pet=pet,
                    ct_text_feature=ct_text_feature,
                    pet_text_feature=pet_text_feature,
                    state_vector=state_vector,
                    return_aux=True,
                )
                fused_feats.append(fused)
                aux_stages.append(aux)
            else:
                fused_feats.append(stage(
                    ct=ct,
                    pet=pet,
                    ct_text_feature=ct_text_feature,
                    pet_text_feature=pet_text_feature,
                    state_vector=state_vector,
                    return_aux=False,
                ))

        if return_aux:
            return fused_feats, {
                "state_ids": state_ids,
                "state_vector": state_vector,
                "use_text_modulation": self.use_text_modulation,
                "stages": aux_stages,
            }
        return fused_feats


# =============================================================================
# Optional one-time local CLIP helper
# =============================================================================

@torch.no_grad()
def build_local_clip_text_priors(
    clip_path: str,
    ct_text: str = DEFAULT_CT_TEXT,
    pet_text: str = DEFAULT_PET_TEXT,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Tensor, Tensor]:
    """
    Encode the two fixed modality descriptions ONCE with a local HF CLIP model.

    Recommended project path:
      /root/autodl-tmp/mkd-main/new-train/pretrained/clip-vit-base-patch32

    The returned [1,D] tensors are CPU float32. CLIP is not kept inside the
    fusion module and should not participate in training.
    """
    try:
        from transformers import CLIPModel, CLIPTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required only for build_local_clip_text_priors()"
        ) from exc

    device = torch.device(device)
    tokenizer = CLIPTokenizer.from_pretrained(
        clip_path,
        local_files_only=True,
    )
    clip_model = CLIPModel.from_pretrained(
        clip_path,
        local_files_only=True,
    )
    clip_model.eval().to(device)
    for p in clip_model.parameters():
        p.requires_grad_(False)

    tokens = tokenizer(
        [ct_text, pet_text],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    features = clip_model.get_text_features(**tokens).float().cpu()

    if features.ndim != 2 or int(features.shape[0]) != 2:
        raise RuntimeError(f"unexpected CLIP text shape: {tuple(features.shape)}")

    return features[0:1].contiguous(), features[1:2].contiguous()


# =============================================================================
# Minimal structural smoke test
# =============================================================================

def _smoke_test() -> None:
    torch.manual_seed(7)
    channels = (64, 128, 320, 512)
    spatial = (8, 4, 2, 1)
    B = 2
    text_dim = 512
    state_dim = 128

    model = TextModulatedMRFSFusion(
        channels=channels,
        text_dim=text_dim,
        state_dim=state_dim,
        text_reduction=16,
        igma_gate_reduction=4,
        igma_channel_reduction=4,
        igma_spatial_reduction=4,
        igma_spatial_kernel_size=1,
    )
    model.set_text_priors(
        torch.randn(1, text_dim),
        torch.randn(1, text_dim),
    )

    ct_feats = [
        torch.randn(B, c, s, s, requires_grad=True)
        for c, s in zip(channels, spatial)
    ]
    pet_feats = [
        torch.randn(B, c, s, s, requires_grad=True)
        for c, s in zip(channels, spatial)
    ]

    # Mixed batch: first Full, second Missing.
    state = torch.tensor([1, 0], dtype=torch.long)
    outputs, aux = model(
        ct_feats,
        pet_feats,
        state=state,
        return_aux=True,
    )

    for y, c, s in zip(outputs, channels, spatial):
        assert y.shape == (B, c, s, s)
        assert torch.isfinite(y).all()

    assert model.state_vectors.weight.shape == (2, state_dim)
    assert torch.equal(aux["state_ids"].cpu(), state)

    sum(y.float().mean() for y in outputs).backward()
    assert any(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )

    # No-text ablation: raw CT/PET straight into IGM-Att, no CLIP priors.
    model_no_text = TextModulatedMRFSFusion(
        channels=channels,
        text_dim=text_dim,
        state_dim=state_dim,
        text_reduction=16,
        igma_gate_reduction=4,
        igma_channel_reduction=4,
        igma_spatial_reduction=4,
        igma_spatial_kernel_size=1,
        use_text_modulation=False,
    )
    assert not model_no_text.has_text_priors()
    outputs_no_text = model_no_text(ct_feats, pet_feats, state=state)
    for y, c, s in zip(outputs_no_text, channels, spatial):
        assert y.shape == (B, c, s, s)
        assert torch.isfinite(y).all()

    print("[SMOKE] passed")
    print("[SMOKE] output shapes:", [tuple(x.shape) for x in outputs])
    print("[SMOKE] state table:", tuple(model.state_vectors.weight.shape))
    print("[SMOKE] convention: 1=Full, 0=Missing")


if __name__ == "__main__":
    _smoke_test()
