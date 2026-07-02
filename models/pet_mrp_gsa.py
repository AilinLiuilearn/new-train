from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


PET_MRP_GSA_LOG_KEYS = (
    'pet_mrp_gsa_enabled',
    'pet_mrp_prior_skipped',
    'pet_mrp_active_stages',
)


def angle_transform(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    Follow DFormerv2-style rotary transform.
    x:   [B, heads, H, W, D]
    sin: [H, W, D]
    cos: [H, W, D]
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos[None, None, :, :, :] + x_rot * sin[None, None, :, :, :]


class DWConv2dBHWC(nn.Module):
    """
    Depthwise conv for BHWC tensor, same style as DFormerv2.
    """
    def __init__(self, dim: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,W,C]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class PETMetabolicPriorGen(nn.Module):
    """
    DFormerv2-inspired prior generator.

    Original DFormerv2:
        spatial distance + depth distance -> geometry prior

    PET-CT version:
        spatial distance + PET metabolic difference + PET co-activation
        -> metabolic relation prior

    The prior is converted to a negative attention mask by multiplying
    positive distances with per-head negative decay values.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        initial_value: float = 2.0,
        heads_range: float = 4.0,
        use_coactivation: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads
        assert head_dim % 2 == 0, "head_dim must be even for angle_transform."

        angle = 1.0 / (10000 ** torch.linspace(0, 1, head_dim // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer("angle", angle, persistent=False)

        # Same spirit as DFormerv2:
        # log(1 - 2^(-...)) is negative, so larger distance gives stronger suppression.
        decay = torch.log(
            1 - 2 ** (
                -initial_value
                - heads_range * torch.arange(num_heads, dtype=torch.float32) / num_heads
            )
        )
        self.register_buffer("decay", decay, persistent=False)

        self.use_coactivation = use_coactivation
        num_priors = 3 if use_coactivation else 2

        # DFormerv2 uses learnable memory weights to bridge depth and spatial priors.
        # Here we use the same form to bridge spatial/metabolic/co-activation priors.
        self.weight = nn.Parameter(torch.ones(num_priors, 1, 1, 1), requires_grad=True)

    @staticmethod
    def _prepare_pet(pet: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """
        pet: [B,1,H0,W0] or [B,3,H0,W0]
        return: [B,H,W] normalized to [0,1] per sample.
        """
        if pet.dim() != 4:
            raise ValueError(f"PET must be [B,C,H,W], got {pet.shape}")

        if pet.size(1) > 1:
            pet = pet[:, :1, :, :]

        pet = F.interpolate(pet, size=size, mode="bilinear", align_corners=False)
        b = pet.size(0)
        pet_flat = pet.flatten(1)
        pet_min = pet_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        pet_max = pet_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        pet = (pet - pet_min) / (pet_max - pet_min + 1e-6)
        return pet[:, 0, :, :].contiguous()

    def _sin_cos(self, H: int, W: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        index = torch.arange(H * W, device=device)
        sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(H, W, -1)
        cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(H, W, -1)
        return sin, cos

    def _pos_full(self, H: int, W: int, device) -> torch.Tensor:
        index_h = torch.arange(H, device=device)
        index_w = torch.arange(W, device=device)
        grid_h, grid_w = torch.meshgrid(index_h, index_w, indexing="ij")
        grid = torch.stack([grid_h, grid_w], dim=-1).reshape(H * W, 2)
        dist = (grid[:, None, :] - grid[None, :, :]).abs().sum(dim=-1).float()
        dist = dist / (dist.max() + 1e-6)
        return dist[None, :, :] * self.decay[:, None, None]  # [heads,N,N]

    def _metabolic_full(self, pet_hw: torch.Tensor):
        """
        pet_hw: [B,H,W]
        """
        B, H, W = pet_hw.shape
        m = pet_hw.reshape(B, H * W)

        diff = (m[:, :, None] - m[:, None, :]).abs()
        diff = diff / (diff.amax(dim=(1, 2), keepdim=True) + 1e-6)
        diff = diff[:, None, :, :] * self.decay[None, :, None, None]

        if not self.use_coactivation:
            return diff, None

        act_dist = 1.0 - (m[:, :, None] * m[:, None, :])
        act_dist = act_dist.clamp(0.0, 1.0)
        act_dist = act_dist[:, None, :, :] * self.decay[None, :, None, None]
        return diff, act_dist

    def _axis_pos(self, L: int, device) -> torch.Tensor:
        idx = torch.arange(L, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().float()
        dist = dist / (dist.max() + 1e-6)
        return dist[None, :, :] * self.decay[:, None, None]  # [heads,L,L]

    def _axis_metabolic_width(self, pet_hw: torch.Tensor):
        """
        width attention: each row has W tokens.
        return masks with shape [B,H,heads,W,W]
        """
        B, H, W = pet_hw.shape
        m = pet_hw  # [B,H,W]

        diff = (m[:, :, :, None] - m[:, :, None, :]).abs()
        diff = diff / (diff.amax(dim=(-1, -2), keepdim=True) + 1e-6)
        diff = diff[:, :, None, :, :] * self.decay[None, None, :, None, None]

        if not self.use_coactivation:
            return diff, None

        act_dist = 1.0 - (m[:, :, :, None] * m[:, :, None, :])
        act_dist = act_dist.clamp(0.0, 1.0)
        act_dist = act_dist[:, :, None, :, :] * self.decay[None, None, :, None, None]
        return diff, act_dist

    def _axis_metabolic_height(self, pet_hw: torch.Tensor):
        """
        height attention: each column has H tokens.
        return masks with shape [B,W,heads,H,H]
        """
        B, H, W = pet_hw.shape
        m = pet_hw.permute(0, 2, 1).contiguous()  # [B,W,H]

        diff = (m[:, :, :, None] - m[:, :, None, :]).abs()
        diff = diff / (diff.amax(dim=(-1, -2), keepdim=True) + 1e-6)
        diff = diff[:, :, None, :, :] * self.decay[None, None, :, None, None]

        if not self.use_coactivation:
            return diff, None

        act_dist = 1.0 - (m[:, :, :, None] * m[:, :, None, :])
        act_dist = act_dist.clamp(0.0, 1.0)
        act_dist = act_dist[:, :, None, :, :] * self.decay[None, None, :, None, None]
        return diff, act_dist

    def forward(self, hw: Tuple[int, int], pet: torch.Tensor, split_or_not: bool):
        H, W = hw
        device = pet.device
        pet_hw = self._prepare_pet(pet, size=(H, W))
        sin, cos = self._sin_cos(H, W, device=device)

        if split_or_not:
            pos_w = self._axis_pos(W, device=device)  # [heads,W,W]
            pos_h = self._axis_pos(H, device=device)  # [heads,H,H]

            diff_w, act_w = self._axis_metabolic_width(pet_hw)
            diff_h, act_h = self._axis_metabolic_height(pet_hw)

            # [B,H,heads,W,W]
            mask_w = self.weight[0] * pos_w[None, None, :, :, :] + self.weight[1] * diff_w
            # [B,W,heads,H,H]
            mask_h = self.weight[0] * pos_h[None, None, :, :, :] + self.weight[1] * diff_h

            if self.use_coactivation:
                mask_w = mask_w + self.weight[2] * act_w
                mask_h = mask_h + self.weight[2] * act_h

            return (sin, cos), (mask_h, mask_w)

        else:
            pos = self._pos_full(H, W, device=device)  # [heads,N,N]
            diff, act = self._metabolic_full(pet_hw)   # [B,heads,N,N]

            mask = self.weight[0] * pos[None, :, :, :] + self.weight[1] * diff
            if self.use_coactivation:
                mask = mask + self.weight[2] * act

            return (sin, cos), mask


class DecomposedPETGSA(nn.Module):
    """
    DFormerv2-style decomposed GSA.
    Used for high-resolution stages.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.lepe = DWConv2dBHWC(embed_dim, kernel_size=5, padding=2)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, rel_pos):
        """
        x: [B,H,W,C]
        mask_h: [B,W,heads,H,H]
        mask_w: [B,H,heads,W,W]
        """
        B, H, W, C = x.shape
        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling
        v = self.v_proj(x)
        lepe = self.lepe(v)

        q = q.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        v = v.view(B, H, W, self.num_heads, self.key_dim)

        q = angle_transform(q, sin, cos)
        k = angle_transform(k, sin, cos)

        # Width attention: [B,H,heads,W,D]
        q_w = q.permute(0, 2, 1, 3, 4)
        k_w = k.permute(0, 2, 1, 3, 4)
        v_w = v.permute(0, 1, 3, 2, 4)

        attn_w = q_w @ k_w.transpose(-1, -2)
        attn_w = attn_w + mask_w
        attn_w = torch.softmax(attn_w, dim=-1)

        out_w = attn_w @ v_w  # [B,H,heads,W,D]

        # Height attention: [B,W,heads,H,D]
        q_h = q.permute(0, 3, 1, 2, 4)
        k_h = k.permute(0, 3, 1, 2, 4)
        v_h = out_w.permute(0, 3, 2, 1, 4)

        attn_h = q_h @ k_h.transpose(-1, -2)
        attn_h = attn_h + mask_h
        attn_h = torch.softmax(attn_h, dim=-1)

        out = attn_h @ v_h  # [B,W,heads,H,D]
        out = out.permute(0, 3, 1, 2, 4).contiguous().view(B, H, W, C)
        out = out + lepe
        out = self.out_proj(out)
        return out


class FullPETGSA(nn.Module):
    """
    DFormerv2-style full GSA.
    Used for the lowest-resolution stage.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.lepe = DWConv2dBHWC(embed_dim, kernel_size=5, padding=2)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, rel_pos):
        """
        x: [B,H,W,C]
        mask: [B,heads,N,N]
        """
        B, H, W, C = x.shape
        N = H * W
        (sin, cos), mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling
        v = self.v_proj(x)
        lepe = self.lepe(v)

        q = q.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        v = v.view(B, H, W, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

        q = angle_transform(q, sin, cos).flatten(2, 3)
        k = angle_transform(k, sin, cos).flatten(2, 3)
        v = v.flatten(2, 3)

        attn = q @ k.transpose(-1, -2)
        attn = attn + mask
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, H, W, C)
        out = out + lepe
        out = self.out_proj(out)
        return out


class MLPFFN(nn.Module):
    """
    DFormerv2-style FFN with depthwise conv.
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dwconv = DWConv2dBHWC(hidden_dim, kernel_size=3, padding=1)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        residual = x
        x = self.dwconv(x)
        x = x + residual
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PETMRPGSABlock(nn.Module):
    """
    Lightweight PET-guided self-attention residual block.

    Only keeps the attention branch from DFormerv2-style RGB-D block:
        x = x + PET-GSA(LN(x), PET-prior)

    Block-level local DWConv and FFN are omitted to reduce interference
    with pretrained ConvNeXt stage features.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        split_or_not: bool,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        initial_value: float = 2.0,
        heads_range: float = 4.0,
        use_coactivation: bool = True,
    ):
        super().__init__()
        self.split_or_not = split_or_not
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.prior_gen = PETMetabolicPriorGen(
            embed_dim=dim,
            num_heads=num_heads,
            initial_value=initial_value,
            heads_range=heads_range,
            use_coactivation=use_coactivation,
        )

        if split_or_not:
            self.attn = DecomposedPETGSA(dim, num_heads)
        else:
            self.attn = FullPETGSA(dim, num_heads)

        try:
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        except Exception:
            self.drop_path = nn.Identity()

    @staticmethod
    def _resolve_pet_mask(pet_available, batch_size: int, device, dtype):
        if pet_available is None:
            return torch.ones(batch_size, 1, 1, 1, device=device, dtype=dtype)
        if isinstance(pet_available, bool):
            value = 1.0 if pet_available else 0.0
            return torch.full((batch_size, 1, 1, 1), value, device=device, dtype=dtype)
        mask = pet_available.to(device=device, dtype=dtype).view(batch_size)
        return mask.view(batch_size, 1, 1, 1)

    def _apply_guide(self, ct_feat: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        x = ct_feat.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        rel_pos = self.prior_gen((x.shape[1], x.shape[2]), pet, split_or_not=self.split_or_not)
        x = x + self.drop_path(self.attn(self.norm1(x), rel_pos))
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        ct_feat: torch.Tensor,
        pet: Optional[torch.Tensor] = None,
        pet_available: Optional[Union[bool, torch.Tensor]] = None,
    ):
        if pet is None:
            return ct_feat

        pet_mask = self._resolve_pet_mask(
            pet_available,
            batch_size=ct_feat.shape[0],
            device=ct_feat.device,
            dtype=ct_feat.dtype,
        )
        if float(pet_mask.sum().detach().cpu()) <= 0.0:
            return ct_feat

        guided = self._apply_guide(ct_feat, pet)
        if float(pet_mask.min().detach().cpu()) >= 1.0:
            return guided
        return ct_feat * (1.0 - pet_mask) + guided * pet_mask