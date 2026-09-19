
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
StateLike = Union[str, int, Tensor]


class TwoLayerMLP(nn.Module):
    """GeminiFusion-style MLP_2."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TextProjection(nn.Module):
    """
    DGNet-style language projection:
        Linear(text_dim -> channels) -> LayerNorm -> GELU
    """
    def __init__(self, text_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )

    def forward(self, text_feature: Tensor) -> Tensor:
        if text_feature.ndim == 1:
            text_feature = text_feature.unsqueeze(0)
        if text_feature.ndim != 2:
            raise ValueError(
                f"text_feature must be [D] or [B,D], got {tuple(text_feature.shape)}"
            )
        return self.proj(text_feature)


class TargetKnowledgeGuidedGate(nn.Module):
    """
    DGNet T-KGM gate generation.

    Core path follows official DGNet:
        relation = Conv3x3(x * text)
        max_out = FC2(ReLU(FC1(GlobalMaxPool(relation))))
        text_out = FC2(ReLU(FC1(text)))
        gate = Sigmoid(max_out + text_out)
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, text_map: Tensor) -> Tensor:
        if x.ndim != 4 or text_map.ndim != 4:
            raise ValueError("x and text_map must be 4D tensors.")
        if x.shape[1] != text_map.shape[1]:
            raise ValueError(
                f"Channel mismatch: visual={x.shape[1]}, text={text_map.shape[1]}"
            )

        relation = self.conv(x * text_map)
        max_out = F.adaptive_max_pool2d(relation, 1)
        max_out = self.fc2(self.relu(self.fc1(max_out)))
        text_out = self.fc2(self.relu(self.fc1(text_map)))
        gate = self.sigmoid(max_out + text_out)
        return gate


class ModalityTextModulator(nn.Module):
    """
    Modality-specific text modulation.

    Agreed task-specific form:
        X^T = X + G(X,T) * X

    No extra alpha scaling parameter.
    """
    def __init__(
        self,
        channels: int,
        text_dim: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.text_proj = TextProjection(text_dim, channels)
        self.gate = TargetKnowledgeGuidedGate(channels, reduction)

    @staticmethod
    def _expand_batch(x: Tensor, batch_size: int) -> Tensor:
        if x.shape[0] == batch_size:
            return x
        if x.shape[0] == 1:
            return x.expand(batch_size, -1)
        raise ValueError(
            f"Condition batch {x.shape[0]} cannot broadcast to {batch_size}."
        )

    def forward(
        self,
        x: Tensor,
        text_feature: Tensor,
        state_condition: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B, C, _, _ = x.shape
        if C != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {C}.")

        condition = self.text_proj(text_feature)
        condition = self._expand_batch(condition, B)

        if state_condition is not None:
            if state_condition.ndim == 1:
                state_condition = state_condition.unsqueeze(0)
            state_condition = self._expand_batch(state_condition, B)
            condition = condition + state_condition

        condition_map = condition.unsqueeze(-1).unsqueeze(-1)
        gate = self.gate(x, condition_map)

        # No alpha: directly use DGNet-style gate in residual modulation.
        x_mod = x + gate * x
        return x_mod, gate, condition_map


class GeminiPixelWiseInteraction(nn.Module):
    """
    Standalone GeminiFusion-style aligned pixel-wise interaction.

    Retains the core official design:
      - separate modality LayerNorm
      - shared relation judger: MLP(2C -> C -> C) + Softmax
      - learnable K/V noise embeddings
      - two directional MultiheadAttention modules
      - per-pixel two-candidate attention
      - residual update
      - shared output projection

    B and N are flattened so every aligned pixel is still an independent
    MHA batch item, but without the original Python loop over B.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        relation_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.dim = dim

        self.norm_ct = nn.LayerNorm(dim, eps=1e-6)
        self.norm_pet = nn.LayerNorm(dim, eps=1e-6)

        self.relation_judger = nn.Sequential(
            TwoLayerMLP(dim * 2, dim, dim, relation_drop),
            nn.Softmax(dim=-1),
        )

        self.k_noise = nn.Embedding(2, dim)
        self.v_noise = nn.Embedding(2, dim)

        self.cross_attn_ct_from_pet = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=False
        )
        self.cross_attn_pet_from_ct = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=False
        )

        # Shared projection, matching GeminiFusion ModuleParallel weight sharing.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.normal_(self.k_noise.weight, std=0.02)
        nn.init.normal_(self.v_noise.weight, std=0.02)

    @staticmethod
    def _to_tokens(x: Tensor) -> Tensor:
        return x.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _to_map(x: Tensor, H: int, W: int) -> Tensor:
        B, N, C = x.shape
        if N != H * W:
            raise ValueError(f"N={N} != H*W={H*W}")
        return x.transpose(1, 2).reshape(B, C, H, W).contiguous()

    def _directional_interaction(
        self,
        query_tokens: Tensor,
        other_tokens: Tensor,
        direction_idx: int,
        mha: nn.MultiheadAttention,
    ) -> Tuple[Tensor, Tensor]:
        B, N, C = query_tokens.shape

        relation_input = torch.cat([query_tokens, other_tokens], dim=-1)
        relation_score = self.relation_judger(relation_input)

        # [1, B*N, C]: one query per aligned pixel.
        q = query_tokens.reshape(B * N, C).unsqueeze(0)
        relation = relation_score.reshape(B * N, C).unsqueeze(0)
        other = other_tokens.reshape(B * N, C).unsqueeze(0)

        # Official GeminiFusion candidate construction.
        noise_k = q + self.k_noise.weight[direction_idx].view(1, 1, C)
        noise_v = q + self.v_noise.weight[direction_idx].view(1, 1, C)

        k = torch.cat([noise_k, q * relation], dim=0)
        v = torch.cat([noise_v, other], dim=0)

        interacted, _ = mha(q, k, v, need_weights=False)
        interacted = interacted.squeeze(0).reshape(B, N, C)

        updated = query_tokens + interacted
        return updated, relation_score

    def forward(
        self,
        ct: Tensor,
        pet: Tensor,
        return_aux: bool = False,
    ):
        if ct.shape != pet.shape:
            raise ValueError(
                f"CT/PET shapes must match, got {ct.shape} vs {pet.shape}"
            )
        if ct.ndim != 4:
            raise ValueError("CT/PET features must be [B,C,H,W].")

        B, C, H, W = ct.shape
        if C != self.dim:
            raise ValueError(f"Expected C={self.dim}, got {C}")

        ct_tokens = self.norm_ct(self._to_tokens(ct))
        pet_tokens = self.norm_pet(self._to_tokens(pet))

        ct_i, rel_ct_from_pet = self._directional_interaction(
            ct_tokens,
            pet_tokens,
            direction_idx=0,
            mha=self.cross_attn_ct_from_pet,
        )
        pet_i, rel_pet_from_ct = self._directional_interaction(
            pet_tokens,
            ct_tokens,
            direction_idx=1,
            mha=self.cross_attn_pet_from_ct,
        )

        ct_i = self.proj_drop(self.proj(ct_i))
        pet_i = self.proj_drop(self.proj(pet_i))

        ct_i = self._to_map(ct_i, H, W)
        pet_i = self._to_map(pet_i, H, W)

        if not return_aux:
            return ct_i, pet_i

        return ct_i, pet_i, {
            "relation_ct_from_pet": rel_ct_from_pet,
            "relation_pet_from_ct": rel_pet_from_ct,
        }


class TextModulatedPixelInteractionStage(nn.Module):
    """
    One scale:
        CT -> CT text modulation ----------\
                                           Gemini pixel interaction -> CT^I
        PET -> PET text + state modulation /                         -> PET^I
        Final: F = CT^I + PET^I
    """
    def __init__(
        self,
        channels: int,
        text_dim: int,
        state_dim: int,
        num_heads: int,
        text_reduction: int = 16,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        relation_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.ct_modulator = ModalityTextModulator(
            channels, text_dim, text_reduction
        )
        self.pet_modulator = ModalityTextModulator(
            channels, text_dim, text_reduction
        )

        # Only a per-scale projection; the actual s_F/s_M are global/shared.
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )

        self.interaction = GeminiPixelWiseInteraction(
            dim=channels,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            relation_drop=relation_drop,
        )

    def forward(
        self,
        ct: Tensor,
        pet: Tensor,
        ct_text_feature: Tensor,
        pet_text_feature: Tensor,
        state_vector: Tensor,
        return_aux: bool = False,
    ):
        if ct.shape != pet.shape:
            raise ValueError(
                f"CT/PET shape mismatch: {ct.shape} vs {pet.shape}"
            )

        B = ct.shape[0]

        if state_vector.ndim == 1:
            state_vector = state_vector.unsqueeze(0)
        if state_vector.shape[0] == 1 and B > 1:
            state_vector = state_vector.expand(B, -1)

        state_condition = self.state_proj(state_vector)

        # CT: fixed CT text only.
        ct_t, ct_gate, ct_condition = self.ct_modulator(
            ct, ct_text_feature, state_condition=None
        )

        # PET: fixed PET text + global Full/Missing state.
        pet_t, pet_gate, pet_condition = self.pet_modulator(
            pet, pet_text_feature, state_condition=state_condition
        )

        if return_aux:
            ct_i, pet_i, interaction_aux = self.interaction(
                ct_t, pet_t, return_aux=True
            )
        else:
            ct_i, pet_i = self.interaction(
                ct_t, pet_t, return_aux=False
            )

        fused = ct_i + pet_i

        if not return_aux:
            return fused

        return fused, {
            "ct_text_gate": ct_gate,
            "pet_text_gate": pet_gate,
            "ct_text_condition": ct_condition,
            "pet_text_condition": pet_condition,
            "ct_text_modulated": ct_t,
            "pet_text_modulated": pet_t,
            "ct_interacted": ct_i,
            "pet_interacted": pet_i,
            **interaction_aux,
        }


class TextModulatedPixelInteractionFusion(nn.Module):
    """
    Complete independent second module.

    Default scales:
        S1:  64 channels
        S2: 128 channels
        S3: 320 channels
        S4: 512 channels

    Hard design constraints implemented:
      - exactly two global learnable state vectors s_F / s_M
      - state vectors shared across all scales
      - CT text modulates CT only
      - PET text + state modulates PET only
      - no CT/PET text semantic fusion
      - no expert / router
      - no alpha parameter
      - GeminiFusion-style aligned pixel-wise bidirectional interaction
      - final F_l = C_l^I + P_l^I
    """

    FULL = 0
    MISSING = 1

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        text_dim: int = 768,
        state_dim: int = 128,
        num_heads: Sequence[int] = (1, 2, 5, 8),
        text_reduction: int = 16,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        relation_drop: float = 0.0,
        ct_text_prior: Optional[Tensor] = None,
        pet_text_prior: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.channels = tuple(channels)
        self.text_dim = int(text_dim)
        self.state_dim = int(state_dim)
        self.num_heads = tuple(num_heads)

        if len(self.channels) != len(self.num_heads):
            raise ValueError("channels and num_heads must have same length.")

        for c, h in zip(self.channels, self.num_heads):
            if c % h != 0:
                raise ValueError(f"{c} must be divisible by {h} heads.")

        # Exactly two GLOBAL state vectors.
        self.state_vectors = nn.Embedding(2, self.state_dim)
        nn.init.trunc_normal_(self.state_vectors.weight, std=0.02)

        self.stages = nn.ModuleList([
            TextModulatedPixelInteractionStage(
                channels=c,
                text_dim=self.text_dim,
                state_dim=self.state_dim,
                num_heads=h,
                text_reduction=text_reduction,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                relation_drop=relation_drop,
            )
            for c, h in zip(self.channels, self.num_heads)
        ])

        self.register_buffer("ct_text_prior", torch.empty(0), persistent=True)
        self.register_buffer("pet_text_prior", torch.empty(0), persistent=True)

        if ct_text_prior is not None or pet_text_prior is not None:
            if ct_text_prior is None or pet_text_prior is None:
                raise ValueError(
                    "ct_text_prior and pet_text_prior must be provided together."
                )
            self.set_text_priors(ct_text_prior, pet_text_prior)

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
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(
                f"{name} prior must be [D] or [1,D], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.text_dim:
            raise ValueError(
                f"{name} text dim={x.shape[1]} != text_dim={self.text_dim}"
            )
        return x.detach().clone().float()

    def has_text_priors(self) -> bool:
        return (
            self.ct_text_prior.numel() > 0
            and self.pet_text_prior.numel() > 0
        )

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
            key = state.lower().strip()
            if key not in aliases:
                raise ValueError(f"Unknown state: {state}")
            return torch.full(
                (batch_size,),
                aliases[key],
                dtype=torch.long,
                device=device,
            )

        if isinstance(state, int):
            if state not in (self.FULL, self.MISSING):
                raise ValueError("state int must be 0 or 1")
            return torch.full(
                (batch_size,),
                state,
                dtype=torch.long,
                device=device,
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
                raise ValueError(
                    "state tensor must be scalar, [1], or [B]"
                )

            if torch.any((ids != 0) & (ids != 1)):
                raise ValueError("state ids must contain only 0 or 1")
            return ids

        raise TypeError(f"Unsupported state type: {type(state)}")

    @property
    def s_F(self) -> Tensor:
        return self.state_vectors.weight[self.FULL]

    @property
    def s_M(self) -> Tensor:
        return self.state_vectors.weight[self.MISSING]

    def forward(
        self,
        ct_feats: Sequence[Tensor],
        pet_feats: Sequence[Tensor],
        state: StateLike,
        ct_text_feature: Optional[Tensor] = None,
        pet_text_feature: Optional[Tensor] = None,
        return_aux: bool = False,
    ):
        if len(ct_feats) != len(self.stages):
            raise ValueError(
                f"Expected {len(self.stages)} CT scales, got {len(ct_feats)}"
            )
        if len(pet_feats) != len(self.stages):
            raise ValueError(
                f"Expected {len(self.stages)} PET scales, got {len(pet_feats)}"
            )

        B = ct_feats[0].shape[0]
        device = ct_feats[0].device

        for i, (ct, pet, c) in enumerate(
            zip(ct_feats, pet_feats, self.channels)
        ):
            if ct.shape != pet.shape:
                raise ValueError(
                    f"Scale {i}: CT/PET mismatch {ct.shape} vs {pet.shape}"
                )
            if ct.shape[1] != c:
                raise ValueError(
                    f"Scale {i}: expected {c} channels, got {ct.shape[1]}"
                )

        if ct_text_feature is None or pet_text_feature is None:
            if not self.has_text_priors():
                raise RuntimeError(
                    "Call set_text_priors(ct_text, pet_text) first, "
                    "or pass both text features into forward()."
                )
            ct_text_feature = self.ct_text_prior
            pet_text_feature = self.pet_text_prior

        ct_text_feature = ct_text_feature.to(
            device=device, dtype=ct_feats[0].dtype
        )
        pet_text_feature = pet_text_feature.to(
            device=device, dtype=pet_feats[0].dtype
        )

        state_ids = self._state_ids(state, B, device)
        state_vector = self.state_vectors(state_ids)

        fused_feats: List[Tensor] = []
        aux_stages: List[Dict[str, Tensor]] = []

        for stage, ct, pet in zip(self.stages, ct_feats, pet_feats):
            if return_aux:
                fused, aux = stage(
                    ct,
                    pet,
                    ct_text_feature,
                    pet_text_feature,
                    state_vector,
                    return_aux=True,
                )
                fused_feats.append(fused)
                aux_stages.append(aux)
            else:
                fused = stage(
                    ct,
                    pet,
                    ct_text_feature,
                    pet_text_feature,
                    state_vector,
                    return_aux=False,
                )
                fused_feats.append(fused)

        if return_aux:
            return fused_feats, {
                "state_ids": state_ids,
                "state_vector": state_vector,
                "stages": aux_stages,
            }

        return fused_feats


if __name__ == "__main__":
    # Minimal shape-only smoke test.
    torch.manual_seed(0)

    channels = (64, 128, 320, 512)
    spatial = (16, 8, 4, 2)
    B = 2
    text_dim = 768

    model = TextModulatedPixelInteractionFusion(
        channels=channels,
        text_dim=text_dim,
        state_dim=128,
        num_heads=(1, 2, 5, 8),
    )

    model.set_text_priors(
        torch.randn(text_dim),
        torch.randn(text_dim),
    )

    ct_feats = [
        torch.randn(B, c, s, s)
        for c, s in zip(channels, spatial)
    ]
    pet_feats = [
        torch.randn(B, c, s, s)
        for c, s in zip(channels, spatial)
    ]

    outputs = model(
        ct_feats,
        pet_feats,
        state="missing",
    )

    print([tuple(x.shape) for x in outputs])
    print("state table:", tuple(model.state_vectors.weight.shape))
