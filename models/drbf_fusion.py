"""
DRBF: Dominance-guided Bidirectional Residual Fusion

Designed for the second fusion stage of PET-CT missing-modality segmentation.

Per scale input:
    C : CT feature                      [B, C, H, W]
    E : PET evidence = alpha * P_cal   [B, C, H, W]

where P_cal is:
    Full    -> calibrated real PET feature
    Missing -> calibrated proxy/compensated PET feature

Core formula:
    [D_CT, D_PET] = RelativeDominance(C, E, text)
    [Delta_CT, Delta_PET] = TwoTokenMixer(C, E, text)

    F_out = C + E + D_PET * Delta_CT + D_CT * Delta_PET

Interpretation:
    Delta_CT  : PET -> CT residual enhancement
    Delta_PET : CT  -> PET residual enhancement

Important:
    - The Stage-1 fusion C + E is always preserved.
    - Delta output projections are zero-initialized, so at step 0:
          F_out == C + E
      exactly.
    - Cross-modal attention is only over 2 modality tokens at each location,
      so the attention matrix is always 2x2, never HW x HW.
    - The original feature channels are never replaced; only the residual
      interaction branch uses C//2 channels by default.
    - Fixed text prior is optional and expects precomputed external embeddings.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


Tensor = torch.Tensor


def _make_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _choose_heads(dim: int, max_heads: int = 4) -> int:
    heads = min(max_heads, dim)
    while dim % heads != 0 and heads > 1:
        heads -= 1
    return heads


class LocalContextProjector(nn.Module):
    """Local context in original C channels, then project residual branch to d."""

    def __init__(self, channels: int, interaction_dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.norm_c = _make_group_norm(channels)
        self.proj = nn.Conv2d(channels, interaction_dim, 1, bias=False)
        self.norm_d = _make_group_norm(interaction_dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm_c(x + self.dwconv(x))
        x = self.norm_d(self.proj(x))
        return self.act(x)


class RelativeDominanceEstimator(nn.Module):
    """
    Patch-wise CT/PET relative dominance.

    Returns:
        d_ct, d_pet: [B,1,H,W], and d_ct + d_pet = 1.
    """

    def __init__(self, interaction_dim: int):
        super().__init__()
        hidden = max(interaction_dim // 2, 8)

        def make_head():
            return nn.Sequential(
                nn.Conv2d(interaction_dim, hidden, 1, bias=False),
                _make_group_norm(hidden),
                nn.GELU(),
                nn.Conv2d(hidden, 1, 1, bias=True),
            )

        self.ct_head = make_head()
        self.pet_head = make_head()

    def forward(self, ct_ctx: Tensor, pet_ctx: Tensor) -> Tuple[Tensor, Tensor]:
        s_ct = self.ct_head(ct_ctx)
        s_pet = self.pet_head(pet_ctx)
        d = torch.softmax(torch.cat([s_ct, s_pet], dim=1), dim=1)
        return d[:, 0:1], d[:, 1:2]


class TwoTokenModalityMixer(nn.Module):
    """
    CT <-> PET interaction only at the same spatial location.

    For each (h,w), create two modality tokens:
        [CT_token, PET_token]

    Attention shape is [2,2], never [HW,HW].
    """

    def __init__(self, channels: int, interaction_dim: int, max_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.interaction_dim = interaction_dim
        self.num_heads = _choose_heads(interaction_dim, max_heads)
        self.head_dim = interaction_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(interaction_dim, interaction_dim, bias=False)
        self.k_proj = nn.Linear(interaction_dim, interaction_dim, bias=False)
        self.v_proj = nn.Linear(interaction_dim, interaction_dim, bias=False)
        self.out_norm = nn.LayerNorm(interaction_dim)

        # Directional residual projections.
        self.ct_out = nn.Linear(interaction_dim, channels, bias=True)
        self.pet_out = nn.Linear(interaction_dim, channels, bias=True)

        # Baseline-preserving initialization.
        nn.init.zeros_(self.ct_out.weight)
        nn.init.zeros_(self.ct_out.bias)
        nn.init.zeros_(self.pet_out.weight)
        nn.init.zeros_(self.pet_out.bias)

    def _to_heads(self, x: Tensor) -> Tensor:
        # [M,2,d] -> [M,h,2,dh]
        m, n, _ = x.shape
        x = x.view(m, n, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self, ct_ctx: Tensor, pet_ctx: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if ct_ctx.shape != pet_ctx.shape:
            raise ValueError(
                f"ct_ctx/pet_ctx shape mismatch: {ct_ctx.shape} vs {pet_ctx.shape}"
            )

        b, d, h, w = ct_ctx.shape

        ct_token = ct_ctx.permute(0, 2, 3, 1).reshape(-1, d)
        pet_token = pet_ctx.permute(0, 2, 3, 1).reshape(-1, d)
        x = torch.stack([ct_token, pet_token], dim=1)  # [BHW,2,d]

        q = self._to_heads(self.q_proj(x))
        k = self._to_heads(self.k_proj(x))
        v = self._to_heads(self.v_proj(x))

        # [BHW,heads,2,2]
        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * self.scale,
            dim=-1,
        )

        y = torch.matmul(attn, v)  # [BHW,heads,2,dh]
        y = y.permute(0, 2, 1, 3).contiguous().view(-1, 2, d)
        y = self.out_norm(y)

        # Token 0 is the PET-informed CT residual source.
        # Token 1 is the CT-informed PET residual source.
        delta_ct = self.ct_out(y[:, 0])
        delta_pet = self.pet_out(y[:, 1])

        delta_ct = delta_ct.view(b, h, w, self.channels).permute(0, 3, 1, 2)
        delta_pet = delta_pet.view(b, h, w, self.channels).permute(0, 3, 1, 2)

        return delta_ct.contiguous(), delta_pet.contiguous(), attn


class DRBFScale(nn.Module):
    """One scale of Dominance-guided Bidirectional Residual Fusion."""

    def __init__(
        self,
        channels: int,
        interaction_dim: Optional[int] = None,
        use_text_prior: bool = False,
        text_dim: Optional[int] = None,
        max_heads: int = 4,
    ):
        super().__init__()
        interaction_dim = interaction_dim or max(channels // 2, 1)

        self.channels = channels
        self.interaction_dim = interaction_dim
        self.use_text_prior = use_text_prior

        self.ct_context = LocalContextProjector(channels, interaction_dim)
        self.pet_context = LocalContextProjector(channels, interaction_dim)

        if use_text_prior:
            if text_dim is None:
                raise ValueError("text_dim is required when use_text_prior=True")
            self.text_proj = nn.Linear(text_dim, interaction_dim, bias=True)
            # Text starts as a no-op condition.
            nn.init.zeros_(self.text_proj.weight)
            nn.init.zeros_(self.text_proj.bias)
        else:
            self.text_proj = None

        self.dominance = RelativeDominanceEstimator(interaction_dim)
        self.mixer = TwoTokenModalityMixer(
            channels=channels,
            interaction_dim=interaction_dim,
            max_heads=max_heads,
        )

    def _condition_pet(
        self, pet_ctx: Tensor, text_embedding: Optional[Tensor]
    ) -> Tensor:
        if not self.use_text_prior:
            return pet_ctx
        if text_embedding is None:
            raise ValueError("text_embedding is required when text prior is enabled")
        shift = self.text_proj(text_embedding)[:, :, None, None]
        return pet_ctx + shift

    def forward(
        self,
        ct: Tensor,
        pet_evidence: Tensor,
        text_embedding: Optional[Tensor] = None,
        return_aux: bool = False,
    ):
        if ct.shape != pet_evidence.shape:
            raise ValueError(
                f"CT/PET evidence shape mismatch: {ct.shape} vs {pet_evidence.shape}"
            )
        if ct.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {ct.shape[1]}"
            )

        # Stage-1 information highway.
        baseline = ct + pet_evidence

        # Residual interaction branch.
        ct_ctx = self.ct_context(ct)
        pet_ctx = self.pet_context(pet_evidence)
        pet_ctx_cond = self._condition_pet(pet_ctx, text_embedding)

        # Local relative dominance.
        d_ct, d_pet = self.dominance(ct_ctx, pet_ctx_cond)

        # Same-location CT <-> PET interaction.
        delta_ct, delta_pet, attn = self.mixer(ct_ctx, pet_ctx_cond)

        # Dominance-guided bidirectional information flow.
        ct_plus = ct + d_pet * delta_ct
        pet_plus = pet_evidence + d_ct * delta_pet

        out = ct_plus + pet_plus

        if not torch.isfinite(out).all():
            raise RuntimeError("[DRBF] output contains NaN/Inf")

        if not return_aux:
            return out

        aux: Dict[str, Tensor] = {
            "baseline": baseline,
            "ct_ctx": ct_ctx,
            "pet_ctx": pet_ctx,
            "d_ct": d_ct,
            "d_pet": d_pet,
            "delta_ct": delta_ct,
            "delta_pet": delta_pet,
            # [BHW, heads, 2, 2]
            "modality_attention": attn,
        }
        return out, aux


class DRBFFusion(nn.Module):
    """
    Four-scale wrapper.

    Default input shapes for 512x512 images:
        S1: [B,  64, 128, 128]
        S2: [B, 128,  64,  64]
        S3: [B, 320,  32,  32]
        S4: [B, 512,  16,  16]

    `pet_evidence_feats` must already be:
        alpha_route,s * calibrated_pet_s
    """

    def __init__(
        self,
        channels: Sequence[int] = (64, 128, 320, 512),
        interaction_dims: Optional[Sequence[int]] = None,
        use_text_prior: bool = False,
        text_dim: Optional[int] = None,
        text_encoder=None,
        max_heads: int = 4,
        real_text_embedding: Optional[Tensor] = None,
        proxy_text_embedding: Optional[Tensor] = None,
    ):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.use_text_prior = use_text_prior
        self.text_dim = text_dim
        self.text_encoder = text_encoder

        if interaction_dims is None:
            interaction_dims = tuple(max(c // 2, 1) for c in self.channels)
        else:
            interaction_dims = tuple(int(d) for d in interaction_dims)

        if len(interaction_dims) != len(self.channels):
            raise ValueError("interaction_dims/channels length mismatch")

        self.register_buffer("real_text_embedding", torch.empty(0), persistent=True)
        self.register_buffer("proxy_text_embedding", torch.empty(0), persistent=True)

        if use_text_prior:
            if text_encoder is not None:
                text_encoder.ensure_ready()
                text_dim = int(text_encoder.text_dim)
                self.text_dim = text_dim
                real_text_embedding = text_encoder.real_embedding
                proxy_text_embedding = text_encoder.proxy_embedding
            elif real_text_embedding is not None:
                if text_dim is None:
                    text_dim = int(real_text_embedding.reshape(-1).numel())
                    self.text_dim = text_dim
            else:
                raise ValueError(
                    "text_encoder is required when use_text_prior=True"
                )

        if use_text_prior and text_dim is None:
            raise ValueError("text_dim is required when use_text_prior=True")

        self.scales = nn.ModuleList([
            DRBFScale(
                channels=c,
                interaction_dim=d,
                use_text_prior=use_text_prior,
                text_dim=text_dim,
                max_heads=max_heads,
            )
            for c, d in zip(self.channels, interaction_dims)
        ])

        if use_text_prior and (
            real_text_embedding is not None or proxy_text_embedding is not None
        ):
            if real_text_embedding is None or proxy_text_embedding is None:
                raise ValueError("Provide both real and proxy text embeddings")
            self.set_text_embeddings(real_text_embedding, proxy_text_embedding)

    @torch.no_grad()
    def set_text_embeddings(
        self, real_embedding: Tensor, proxy_embedding: Tensor
    ) -> None:
        if not self.use_text_prior:
            raise RuntimeError("use_text_prior=False")

        real = real_embedding.detach().float()
        proxy = proxy_embedding.detach().float()

        if real.ndim == 2 and real.shape[0] == 1:
            real = real.squeeze(0)
        if proxy.ndim == 2 and proxy.shape[0] == 1:
            proxy = proxy.squeeze(0)

        if real.ndim != 1 or proxy.ndim != 1:
            raise ValueError("Text embeddings must be [D] or [1,D]")
        if real.numel() != self.text_dim or proxy.numel() != self.text_dim:
            raise ValueError(
                f"Expected text_dim={self.text_dim}, got "
                f"real={real.numel()}, proxy={proxy.numel()}"
            )

        self.real_text_embedding = real
        self.proxy_text_embedding = proxy

    def _resolve_text(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        mode: str,
        text_embedding: Optional[Tensor],
        pet_available: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if not self.use_text_prior:
            return None

        # Explicit batch text embedding has highest priority.
        if text_embedding is not None:
            t = text_embedding
            if t.ndim == 1:
                t = t.unsqueeze(0)
            if t.ndim != 2:
                raise ValueError("text_embedding must be [D] or [B,D]")
            if t.shape[0] == 1 and batch_size > 1:
                t = t.expand(batch_size, -1)
            if t.shape[0] != batch_size:
                raise ValueError("text embedding batch size mismatch")
            return t.to(device=device, dtype=dtype)

        if self.real_text_embedding.numel() == 0:
            raise RuntimeError(
                "Text prior enabled but embeddings are not set. "
                "Initialize FixedPETSourceTextEncoder before building DRBF."
            )

        mode = str(mode).lower().strip()
        real = self.real_text_embedding.to(device=device, dtype=dtype)
        proxy = self.proxy_text_embedding.to(device=device, dtype=dtype)

        if mode == "full":
            return real.unsqueeze(0).expand(batch_size, -1)
        if mode == "missing":
            return proxy.unsqueeze(0).expand(batch_size, -1)
        if mode == "auto":
            if pet_available is None:
                raise ValueError("pet_available is required when mode='auto'")
            availability = pet_available.to(device=device).long().view(-1)
            if availability.numel() != batch_size:
                raise ValueError(
                    f"pet_available must have B={batch_size} entries, got {availability.numel()}"
                )
            if not torch.all((availability == 0) | (availability == 1)):
                raise ValueError("pet_available values must be 0 or 1")
            mask = availability.bool().unsqueeze(-1)
            real_b = real.unsqueeze(0).expand(batch_size, -1)
            proxy_b = proxy.unsqueeze(0).expand(batch_size, -1)
            return torch.where(mask, real_b, proxy_b)

        raise ValueError("mode must be 'full', 'missing', or 'auto'")

    def forward(
        self,
        ct_feats: Sequence[Tensor],
        pet_evidence_feats: Sequence[Tensor],
        mode: str = "full",
        pet_available: Optional[Tensor] = None,
        text_embedding: Optional[Tensor] = None,
        return_aux: bool = False,
    ):
        if len(ct_feats) != len(self.scales):
            raise ValueError(f"Expected {len(self.scales)} CT scales")
        if len(pet_evidence_feats) != len(self.scales):
            raise ValueError(f"Expected {len(self.scales)} PET scales")

        mode = str(mode).lower().strip()
        batch_size = ct_feats[0].shape[0]
        text = self._resolve_text(
            batch_size=batch_size,
            device=ct_feats[0].device,
            dtype=ct_feats[0].dtype,
            mode=mode,
            text_embedding=text_embedding,
            pet_available=pet_available,
        )

        outputs: List[Tensor] = []
        aux_list: List[Dict[str, Tensor]] = []

        for block, ct, pet in zip(self.scales, ct_feats, pet_evidence_feats):
            if return_aux:
                out, aux = block(
                    ct,
                    pet,
                    text_embedding=text,
                    return_aux=True,
                )
                outputs.append(out)
                aux_list.append(aux)
            else:
                outputs.append(
                    block(
                        ct,
                        pet,
                        text_embedding=text,
                        return_aux=False,
                    )
                )

        return (outputs, aux_list) if return_aux else outputs


def _demo() -> None:
    """Independent four-scale smoke test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DRBF demo] device={device}")

    channels = (64, 128, 320, 512)
    spatial = ((128, 128), (64, 64), (32, 32), (16, 16))
    b = 1

    model = DRBFFusion(
        channels=channels,
        interaction_dims=None,  # -> C//2
        use_text_prior=False,
    ).to(device)
    model.train()

    ct_feats = [
        torch.randn(b, c, h, w, device=device, requires_grad=True)
        for c, (h, w) in zip(channels, spatial)
    ]
    pet_evidence_feats = [
        torch.randn(b, c, h, w, device=device, requires_grad=True)
        for c, (h, w) in zip(channels, spatial)
    ]

    outputs, aux = model(
        ct_feats,
        pet_evidence_feats,
        mode="full",
        return_aux=True,
    )

    print("\n[Shapes + zero-step equivalence]")
    max_err = 0.0
    for i, (out, ct, pet, a) in enumerate(
        zip(outputs, ct_feats, pet_evidence_feats, aux), start=1
    ):
        err = (out - (ct + pet)).abs().max().item()
        max_err = max(max_err, err)
        print(
            f"S{i}: out={tuple(out.shape)}, "
            f"max|out-(ct+E)|={err:.8e}, "
            f"D_CT_mean={a['d_ct'].mean().item():.4f}, "
            f"D_PET_mean={a['d_pet'].mean().item():.4f}, "
            f"attn={tuple(a['modality_attention'].shape)}"
        )
        assert a["modality_attention"].shape[-2:] == (2, 2)
        assert torch.isfinite(out).all()

    if max_err != 0.0:
        raise RuntimeError(
            "Zero-step equivalence failed: DRBF must initially equal CT + E"
        )

    loss = sum(x.square().mean() for x in outputs)
    loss.backward()

    for i, (ct, pet) in enumerate(zip(ct_feats, pet_evidence_feats), start=1):
        assert ct.grad is not None and torch.isfinite(ct.grad).all(), f"bad CT grad S{i}"
        assert pet.grad is not None and torch.isfinite(pet.grad).all(), f"bad PET grad S{i}"

    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise RuntimeError(f"Non-finite gradient: {name}")

    print(f"\nloss={loss.item():.6f}")
    print("[PASS] DRBF standalone forward/backward test passed.")


if __name__ == "__main__":
    _demo()