"""
dual_prompt_pgfa.py

Standalone Dual-Prompt Prompt-Guided Feature Attention (DP-PGFA) module
for PET-CT fused feature refinement.

Design lineage
--------------
1) TG-ECNet / TaskMoE-style feature-conditioned learnable task prompt:
   CondNet -> GAP -> Softmax over a learnable prompt dictionary -> task prompt.

2) MP-HSIR / PGSSA-style feature processing:
   Window Spatial Self-Attention
     + Global transposed feature self-attention
     + Prompt-guided local feature attention
     + Gated MLP.

3) PET-CT-specific adaptation:
   - Input is ONLY the already-validated Stage-1 fused feature F_base.
   - No CT/PET/CPPI/alpha logic is changed inside this module.
   - "Spectral" attention is reinterpreted as generic latent feature attention.
   - A fixed biomedical text prompt (Full vs Missing) and a learnable prompt
     bank jointly guide local prompt selection.
   - A zero-initialized output projection makes the whole adapter identity-safe:
         F_out = F_base + ZeroProj(Core(F_base) - F_base)
     so F_out == F_base at initialization.

This file is self-contained for the core module (PyTorch only). Optional local
biomedical text encoding requires `transformers` only when a local text tower
is actually used.

References
----------
TG-ECNet / Task-Gated Multi-Expert Collaboration Network, ICML 2025.
MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image
Restoration, ICCV 2025 / arXiv:2503.09131.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


FULL_TEXT = (
    "This fused feature combines CT structural information with detailed, "
    "patient-specific metabolic information from real PET."
)
MISSING_TEXT = (
    "This fused feature combines CT structural information with smooth and "
    "coarse tumor-related information from compensated PET."
)


def _check_bchw(x: torch.Tensor, channels: Optional[int] = None, name: str = "x") -> None:
    if x.ndim != 4:
        raise ValueError(f"{name} must be BCHW, got shape={tuple(x.shape)}")
    if channels is not None and x.shape[1] != channels:
        raise ValueError(f"{name} expected C={channels}, got C={x.shape[1]}")
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """BHWC -> [B*nW, ws, ws, C]."""
    b, h, w, c = x.shape
    ws = int(window_size)
    if h % ws != 0 or w % ws != 0:
        raise ValueError(f"H={h}, W={w} must be divisible by ws={ws}")
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, c)


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    h: int,
    w: int,
    batch_size: int,
) -> torch.Tensor:
    """[B*nW, ws, ws, C] -> BHWC."""
    ws = int(window_size)
    c = windows.shape[-1]
    x = windows.view(batch_size, h // ws, w // ws, ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, h, w, c)


def pad_bhwc_to_window(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    b, h, w, c = x.shape
    ws = int(window_size)
    pad_h = (ws - h % ws) % ws
    pad_w = (ws - w % ws) % ws
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    x = x.permute(0, 3, 1, 2).contiguous()
    x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
    return x.permute(0, 2, 3, 1).contiguous(), pad_h, pad_w


# -----------------------------------------------------------------------------
# 1. TG-ECNet-style feature-conditioned learnable task prompt
# -----------------------------------------------------------------------------

class FeatureTaskPromptGenerator(nn.Module):
    """
    CondNet -> GAP -> prompt atom softmax -> weighted learnable dictionary.
    """

    def __init__(
        self,
        in_channels: int,
        atom_num: int = 32,
        atom_dim: int = 256,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.atom_num = int(atom_num)
        self.atom_dim = int(atom_dim)

        self.cond_net = nn.Sequential(
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
        self.atom_logits = nn.Linear(32, atom_num)
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim) * 0.02)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _check_bchw(x, self.in_channels, "FeatureTaskPromptGenerator.x")
        z = self.cond_net(x)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        atom_weights = F.softmax(self.atom_logits(z), dim=-1)
        prompt = self.act(atom_weights @ self.dictionary)
        return prompt, atom_weights


# -----------------------------------------------------------------------------
# 2. Optional fixed biomedical text encoding
# -----------------------------------------------------------------------------

@torch.no_grad()
def encode_fixed_biomedical_text_prompts(
    text_tower_path: str,
    full_text: str = FULL_TEXT,
    missing_text: str = MISSING_TEXT,
    max_length: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode Full/Missing texts once from a LOCAL HuggingFace text tower."""
    if not text_tower_path:
        raise ValueError("text_tower_path is required")
    if not os.path.isdir(text_tower_path):
        raise FileNotFoundError(f"Local text tower not found: {text_tower_path}")

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required only for local text encoding. "
            "Install it or pass precomputed embeddings."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(text_tower_path, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            text_tower_path, local_files_only=True, use_fast=False
        )

    model = AutoModel.from_pretrained(text_tower_path, local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokens = tokenizer(
        [full_text, missing_text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    outputs = model(**tokens)
    hidden = outputs.last_hidden_state.float()
    mask = tokens["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    pooled = F.normalize(pooled, p=2, dim=-1, eps=1e-6).cpu()
    full_embedding = pooled[0].contiguous()
    missing_embedding = pooled[1].contiguous()

    del model, tokenizer, outputs, hidden, mask, pooled, tokens
    gc.collect()
    return full_embedding, missing_embedding


# -----------------------------------------------------------------------------
# 3. MP-HSIR-style window spatial self-attention
# -----------------------------------------------------------------------------

class WindowSpatialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        ws = self.window_size
        table_size = (2 * ws - 1) * (2 * ws - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        try:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        except TypeError:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flat = torch.flatten(coords, 1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += ws - 1
        relative[:, :, 1] += ws - 1
        relative[:, :, 0] *= 2 * ws - 1
        self.register_buffer("relative_position_index", relative.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bwin, n, c = x.shape
        if c != self.dim or n != self.window_size * self.window_size:
            raise ValueError(f"unexpected window token shape {tuple(x.shape)}")

        qkv = self.qkv(x).reshape(
            bwin, n, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].view(n, n, self.num_heads).permute(2, 0, 1).contiguous()
        attn = attn + rel.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(bwin // nw, nw, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(bwin, self.num_heads, n, n)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(bwin, n, c)
        return self.proj_drop(self.proj(out))


# -----------------------------------------------------------------------------
# 4. MP-HSIR global spectral attention -> PET-CT global feature attention
# -----------------------------------------------------------------------------

class GlobalFeatureAttention(nn.Module):
    """
    Same transposed-attention computation as the MP-HSIR global branch,
    but latent channels are interpreted as learned feature channels, not spectra.
    """

    def __init__(self, dim: int, num_heads: int = 4, bias: bool = False) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_bchw(x, self.dim, "GlobalFeatureAttention.x")
        b, c, h, w = x.shape
        hc = c // self.num_heads
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.num_heads, hc, h * w)
        k = k.view(b, self.num_heads, hc, h * w)
        v = v.view(b, self.num_heads, hc, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.temperature, dim=-1)
        out = (attn @ v).view(b, c, h, w)
        return self.project_out(out)


# -----------------------------------------------------------------------------
# 5. Dual-prompt guided local feature attention
# -----------------------------------------------------------------------------

class DualPromptGuidedLocalFeatureAttention(nn.Module):
    """
    MP-HSIR PGSSA-style local prompt-guided attention with two extra prompt
    conditions for PET-CT:
      - TG-style feature task prompt
      - fixed Full/Missing biomedical text prompt

    Prompt selection:
      logits = local_logits + task_logits + text_logits
      weights = softmax(logits)
      selected_prompt = weighted learnable prompt bank

    Then, preserving the PGSSA local branch:
      selected_prompt -> Q
      local feature descriptor -> K,V
      reduced transposed attention -> window-wise channel modulation.
    """

    def __init__(
        self,
        dim: int,
        task_prompt_dim: int = 256,
        text_dim: int = 768,
        prompt_len: int = 128,
        compress_ratio: int = 8,
        bias: bool = False,
        use_task_prompt: bool = True,
        use_text_prompt: bool = True,
    ) -> None:
        super().__init__()
        reduced_dim = max(1, dim // compress_ratio)
        self.dim = int(dim)
        self.task_prompt_dim = int(task_prompt_dim)
        self.text_dim = int(text_dim)
        self.prompt_len = int(prompt_len)
        self.reduced_dim = int(reduced_dim)
        self.use_task_prompt = bool(use_task_prompt)
        self.use_text_prompt = bool(use_text_prompt)
        self.scale = self.reduced_dim ** -0.5

        self.linear_down = nn.Linear(dim, reduced_dim, bias=bias)
        self.linear_up = nn.Linear(reduced_dim, dim, bias=bias)
        self.local_to_prompt = nn.Linear(dim, prompt_len, bias=bias)
        self.prompt_bank = nn.Parameter(torch.rand(1, 1, prompt_len, reduced_dim))

        self.task_to_prompt = (
            nn.Linear(task_prompt_dim, prompt_len, bias=bias)
            if self.use_task_prompt else None
        )
        self.text_to_prompt = (
            nn.Linear(text_dim, prompt_len, bias=bias)
            if self.use_text_prompt else None
        )

        self.q = nn.Linear(reduced_dim, reduced_dim, bias=bias)
        self.kv = nn.Linear(reduced_dim, reduced_dim * 2, bias=bias)
        self.proj = nn.Linear(reduced_dim, reduced_dim)

    def forward(
        self,
        window_tokens: torch.Tensor,
        windows_per_image: int,
        task_prompt: Optional[torch.Tensor],
        text_prompt: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bwin, n, c = window_tokens.shape
        if c != self.dim or bwin % windows_per_image != 0:
            raise ValueError("invalid local feature attention input")
        b = bwin // windows_per_image
        shortcut = window_tokens
        pooled = window_tokens.mean(dim=1, keepdim=True)

        prompt_logits = self.local_to_prompt(pooled)

        if self.use_task_prompt:
            if task_prompt is None or task_prompt.shape != (b, self.task_prompt_dim):
                raise ValueError("invalid task_prompt")
            task_logits = self.task_to_prompt(task_prompt)
            task_logits = task_logits.repeat_interleave(windows_per_image, dim=0)
            prompt_logits = prompt_logits + task_logits.unsqueeze(1)

        if self.use_text_prompt:
            if text_prompt is None or text_prompt.shape != (b, self.text_dim):
                raise ValueError("invalid text_prompt")
            text_logits = self.text_to_prompt(text_prompt)
            text_logits = text_logits.repeat_interleave(windows_per_image, dim=0)
            prompt_logits = prompt_logits + text_logits.unsqueeze(1)

        prompt_weights = F.softmax(prompt_logits.float(), dim=-1)
        local_reduced = self.linear_down(pooled.float())
        bank = self.prompt_bank.float().expand(bwin, -1, -1, -1)
        selected = (prompt_weights.unsqueeze(-1) * bank).sum(dim=2)

        q = self.q(selected)
        k, v = self.kv(local_reduced).chunk(2, dim=-1)
        attn = F.softmax((q.transpose(-2, -1) @ k) * self.scale, dim=-1)
        out = attn @ v.transpose(-2, -1)
        out = out.transpose(-2, -1).contiguous()
        out = self.linear_up(self.proj(out))
        out = out.to(shortcut.dtype) * shortcut
        return out, prompt_weights.squeeze(1)


# -----------------------------------------------------------------------------
# 6. MP-HSIR-style gated MLP
# -----------------------------------------------------------------------------

class GatedMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_main, x_gate = self.fc1(x).chunk(2, dim=-1)
        x = x_main * F.gelu(x_gate)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


# -----------------------------------------------------------------------------
# 7. Adapted PGSSTB
# -----------------------------------------------------------------------------

class DualPromptPGSSTB(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.66,
        compress_ratio: int = 8,
        prompt_len: int = 128,
        task_prompt_dim: int = 256,
        text_dim: int = 768,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        bias: bool = False,
        use_task_prompt: bool = True,
        use_text_prompt: bool = True,
    ) -> None:
        super().__init__()
        if not (0 <= shift_size < window_size):
            raise ValueError("shift_size must be in [0, window_size)")
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.spatial_attn = WindowSpatialAttention(
            dim, window_size, num_heads, qkv_bias, attn_drop, drop
        )
        self.global_feature_attn = GlobalFeatureAttention(dim, num_heads, bias)
        self.local_prompt_attn = DualPromptGuidedLocalFeatureAttention(
            dim=dim,
            task_prompt_dim=task_prompt_dim,
            text_dim=text_dim,
            prompt_len=prompt_len,
            compress_ratio=compress_ratio,
            bias=bias,
            use_task_prompt=use_task_prompt,
            use_text_prompt=use_text_prompt,
        )
        hidden_dim = int(round(dim * mlp_ratio))
        self.mlp = GatedMlp(dim, hidden_dim, drop)

    def _shift_mask(self, h: int, w: int, device, dtype) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        ws, ss = self.window_size, self.shift_size
        img_mask = torch.zeros((1, h, w, 1), device=device, dtype=dtype)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        count = 0
        for hs in h_slices:
            for wslice in w_slices:
                img_mask[:, hs, wslice, :] = count
                count += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        task_prompt: Optional[torch.Tensor],
        text_prompt: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _check_bchw(x, self.dim, "DualPromptPGSSTB.x")
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        shortcut = tokens
        x_bhwc = self.norm1(tokens).view(b, h, w, c)
        x_bhwc, _, _ = pad_bhwc_to_window(x_bhwc, self.window_size)
        hp, wp = x_bhwc.shape[1:3]

        shifted = (
            torch.roll(x_bhwc, (-self.shift_size, -self.shift_size), (1, 2))
            if self.shift_size > 0 else x_bhwc
        )
        windows = window_partition(shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, c)
        mask = self._shift_mask(hp, wp, x.device, x.dtype)
        spatial_windows = self.spatial_attn(windows, mask)
        nwin = (hp // self.window_size) * (wp // self.window_size)

        local_windows, prompt_weights = self.local_prompt_attn(
            spatial_windows, nwin, task_prompt, text_prompt
        )

        spatial_map = window_reverse(
            spatial_windows.view(-1, self.window_size, self.window_size, c),
            self.window_size, hp, wp, b
        )
        local_map = window_reverse(
            local_windows.view(-1, self.window_size, self.window_size, c),
            self.window_size, hp, wp, b
        )

        if self.shift_size > 0:
            spatial_map = torch.roll(spatial_map, (self.shift_size, self.shift_size), (1, 2))
            local_map = torch.roll(local_map, (self.shift_size, self.shift_size), (1, 2))

        spatial_map = spatial_map[:, :h, :w, :]
        local_map = local_map[:, :h, :w, :]
        global_map = self.global_feature_attn(
            spatial_map.permute(0, 3, 1, 2).contiguous()
        )
        global_tokens = global_map.flatten(2).transpose(1, 2)
        local_tokens = local_map.reshape(b, h * w, c)

        mixed = local_tokens + global_tokens
        out_tokens = shortcut + self.drop_path(mixed)
        out_tokens = out_tokens + self.drop_path(self.mlp(self.norm2(out_tokens)))
        out = out_tokens.transpose(1, 2).view(b, c, h, w)

        pw = prompt_weights.float().clamp_min(1e-8)
        stats = {
            "prompt_weights_mean": prompt_weights.detach().mean(dim=0),
            "prompt_weights_entropy": (-(pw * pw.log()).sum(dim=-1).mean()).detach(),
        }
        return out, stats


# -----------------------------------------------------------------------------
# 8. MP-HSIR-style BaseBlock / stack
# -----------------------------------------------------------------------------

class DualPromptPGFABaseBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 2.66,
        compress_ratio: int = 8,
        prompt_len: int = 128,
        task_prompt_dim: int = 256,
        text_dim: int = 768,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        bias: bool = False,
        use_task_prompt: bool = True,
        use_text_prompt: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.blocks = nn.ModuleList([
            DualPromptPGSSTB(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                compress_ratio=compress_ratio,
                prompt_len=prompt_len,
                task_prompt_dim=task_prompt_dim,
                text_dim=text_dim,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                bias=bias,
                use_task_prompt=use_task_prompt,
                use_text_prompt=use_text_prompt,
            )
            for i in range(depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        task_prompt: Optional[torch.Tensor],
        text_prompt: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        shortcut = x
        entropies, means = [], []
        for block in self.blocks:
            x, stats = block(x, task_prompt, text_prompt)
            entropies.append(stats["prompt_weights_entropy"])
            means.append(stats["prompt_weights_mean"])
        out = x + shortcut
        return out, {
            "prompt_weights_entropy": torch.stack(entropies).mean(),
            "prompt_weights_mean": torch.stack(means).mean(dim=0),
        }


# -----------------------------------------------------------------------------
# 9. Final standalone Stage-2 adapter
# -----------------------------------------------------------------------------

@dataclass
class DualPromptPGFAOutput:
    feature: torch.Tensor
    stats: Dict[str, torch.Tensor]


class DualPromptPGFAAdapter(nn.Module):
    """
    Complete standalone Stage-2 adapter.

    Input:  Stage-1 fused feature F_base only.
    Output: F_out with identical BCHW shape.

    Identity-safe outer adapter:
        core = PGFA(F_base)
        residual = core - F_base
        delta = zero_init_1x1(residual)
        F_out = F_base + delta
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        window_size: int = 8,
        depth: int = 2,
        mlp_ratio: float = 2.66,
        compress_ratio: int = 8,
        prompt_len: int = 128,
        task_atom_num: int = 32,
        task_prompt_dim: int = 256,
        task_prompt_hidden: int = 64,
        text_dim: int = 768,
        use_task_prompt: bool = True,
        use_text_prompt: bool = True,
        full_text_embedding: Optional[torch.Tensor] = None,
        missing_text_embedding: Optional[torch.Tensor] = None,
        text_tower_path: Optional[str] = None,
        qkv_bias: bool = True,
        bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.text_dim = int(text_dim)
        self.use_task_prompt = bool(use_task_prompt)
        self.use_text_prompt = bool(use_text_prompt)
        self.prompt_len = int(prompt_len)

        self.task_prompt = (
            FeatureTaskPromptGenerator(
                in_channels, task_atom_num, task_prompt_dim, task_prompt_hidden
            ) if self.use_task_prompt else None
        )

        if self.use_text_prompt and text_tower_path is not None:
            full_text_embedding, missing_text_embedding = encode_fixed_biomedical_text_prompts(
                text_tower_path
            )
            inferred_dim = int(full_text_embedding.numel())
            if inferred_dim != self.text_dim:
                raise ValueError(
                    f"text_dim={self.text_dim} but local text tower returned {inferred_dim}. "
                    f"Instantiate with text_dim={inferred_dim}."
                )

        if self.use_text_prompt:
            if full_text_embedding is None or missing_text_embedding is None:
                self.register_buffer("full_text_embedding", torch.zeros(self.text_dim), persistent=True)
                self.register_buffer("missing_text_embedding", torch.zeros(self.text_dim), persistent=True)
                self.register_buffer("_text_ready", torch.tensor(False), persistent=True)
            else:
                full = self._prepare_embedding(full_text_embedding, "full_text_embedding")
                missing = self._prepare_embedding(missing_text_embedding, "missing_text_embedding")
                self.register_buffer("full_text_embedding", full, persistent=True)
                self.register_buffer("missing_text_embedding", missing, persistent=True)
                self.register_buffer("_text_ready", torch.tensor(True), persistent=True)
        else:
            self.register_buffer("full_text_embedding", torch.empty(0), persistent=False)
            self.register_buffer("missing_text_embedding", torch.empty(0), persistent=False)
            self.register_buffer("_text_ready", torch.tensor(False), persistent=False)

        self.core = DualPromptPGFABaseBlock(
            dim=in_channels,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            compress_ratio=compress_ratio,
            prompt_len=prompt_len,
            task_prompt_dim=task_prompt_dim,
            text_dim=text_dim,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            bias=bias,
            use_task_prompt=use_task_prompt,
            use_text_prompt=use_text_prompt,
        )

        self.out_proj = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def _prepare_embedding(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim == 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim != 1 or x.shape[0] != self.text_dim:
            raise ValueError(f"{name} must be [{self.text_dim}], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"{name} contains NaN/Inf")
        return x.detach().float().clone()

    @torch.no_grad()
    def set_fixed_text_embeddings(
        self,
        full_embedding: torch.Tensor,
        missing_embedding: torch.Tensor,
    ) -> None:
        if not self.use_text_prompt:
            raise RuntimeError("use_text_prompt=False")
        full = self._prepare_embedding(full_embedding, "full_embedding")
        missing = self._prepare_embedding(missing_embedding, "missing_embedding")
        self.full_text_embedding.copy_(full.to(self.full_text_embedding.device))
        self.missing_text_embedding.copy_(missing.to(self.missing_text_embedding.device))
        self._text_ready.fill_(True)

    def _select_text(
        self,
        b: int,
        route: str,
        pet_available: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.use_text_prompt:
            return None
        if not bool(self._text_ready.item()):
            raise RuntimeError(
                "Fixed text embeddings are not ready. Pass text_tower_path, "
                "precomputed embeddings, or call set_fixed_text_embeddings()."
            )

        route = str(route).lower().strip()
        full = self.full_text_embedding.to(device).view(1, -1)
        missing = self.missing_text_embedding.to(device).view(1, -1)

        if route == "full":
            return full.expand(b, -1)
        if route == "missing":
            return missing.expand(b, -1)
        if route == "auto":
            if pet_available is None:
                raise ValueError("pet_available is required for route='auto'")
            a = pet_available.to(device).long().view(-1)
            if a.numel() != b or not torch.all((a == 0) | (a == 1)):
                raise ValueError("pet_available must be B binary values")
            return torch.where(
                a[:, None].bool(),
                full.expand(b, -1),
                missing.expand(b, -1),
            )
        raise ValueError("route must be 'full', 'missing', or 'auto'")

    def forward(
        self,
        x: torch.Tensor,
        route: str = "full",
        pet_available: Optional[torch.Tensor] = None,
    ) -> DualPromptPGFAOutput:
        _check_bchw(x, self.in_channels, "DualPromptPGFAAdapter.x")
        b = x.shape[0]

        if self.use_task_prompt:
            task_prompt, atom_weights = self.task_prompt(x)
        else:
            task_prompt, atom_weights = None, None

        text_prompt = self._select_text(b, route, pet_available, x.device)
        core_out, core_stats = self.core(x, task_prompt, text_prompt)

        raw_residual = core_out - x
        delta = self.out_proj(raw_residual)
        out = x + delta

        eps = 1e-8
        base_norm = x.float().norm()
        stats = {
            "raw_residual_l2_ratio": (
                raw_residual.float().norm() / (base_norm + eps)
            ).detach(),
            "delta_l2_ratio": (
                delta.float().norm() / (base_norm + eps)
            ).detach(),
            "prompt_weights_entropy": core_stats["prompt_weights_entropy"],
            "prompt_weights_mean": core_stats["prompt_weights_mean"],
        }
        if atom_weights is not None:
            stats["task_atom_weights_mean"] = atom_weights.detach().mean(dim=0)
        if self.use_text_prompt:
            stats["full_missing_text_cosine"] = F.cosine_similarity(
                self.full_text_embedding.float().view(1, -1),
                self.missing_text_embedding.float().view(1, -1),
                dim=-1,
                eps=1e-6,
            ).squeeze(0).detach()

        return DualPromptPGFAOutput(feature=out, stats=stats)


# -----------------------------------------------------------------------------
# 10. Standalone smoke test
# -----------------------------------------------------------------------------


def _mock_text_embeddings(text_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1234)
    full = F.normalize(torch.randn(text_dim, generator=gen), dim=0)
    missing = F.normalize(torch.randn(text_dim, generator=gen), dim=0)
    return full, missing


def run_smoke_test(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    if args.text_tower_path:
        full, missing = encode_fixed_biomedical_text_prompts(args.text_tower_path)
        text_dim = int(full.numel())
        print(f"[TEXT] local text tower loaded, dim={text_dim}")
    else:
        text_dim = args.text_dim
        full, missing = _mock_text_embeddings(text_dim)
        print("[TEXT] using deterministic MOCK embeddings for architecture smoke test only")

    model = DualPromptPGFAAdapter(
        in_channels=args.channels,
        num_heads=args.heads,
        window_size=args.window_size,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        compress_ratio=args.compress_ratio,
        prompt_len=args.prompt_len,
        text_dim=text_dim,
        use_task_prompt=True,
        use_text_prompt=True,
        full_text_embedding=full,
        missing_text_embedding=missing,
        zero_init_output=True,
    )

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.randn(args.batch, args.channels, args.height, args.width, device=device)

    print(f"[MODULE] input={tuple(x.shape)}, device={device}")

    model.eval()
    with torch.no_grad():
        full_out = model(x, route="full")
        missing_out = model(x, route="missing")

    full_diff = (full_out.feature - x).abs().max().item()
    missing_diff = (missing_out.feature - x).abs().max().item()
    print(f"[FULL] zero-step diff={full_diff:.8e}")
    print(f"[MISSING] zero-step diff={missing_diff:.8e}")
    print(f"[TEXT] cosine={full_out.stats['full_missing_text_cosine'].item():.6f}")

    if args.batch >= 2:
        availability = torch.tensor(
            [1 if i % 2 == 0 else 0 for i in range(args.batch)], device=device
        )
        with torch.no_grad():
            auto_out = model(x, route="auto", pet_available=availability)
        auto_diff = (auto_out.feature - x).abs().max().item()
        print(f"[AUTO] pet_available={availability.tolist()}, diff={auto_diff:.8e}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    target = torch.randn_like(x)
    result = model(x, route="full")
    loss = F.mse_loss(result.feature, target)
    loss.backward()

    nonfinite = 0
    grad_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                nonfinite += 1
            grad_norm_sq += float(p.grad.float().norm()) ** 2
    grad_norm = math.sqrt(grad_norm_sq)
    optimizer.step()

    with torch.no_grad():
        post = model(x, route="full")
    post_diff = (post.feature - x).abs().max().item()

    print(f"[BACKWARD] loss={loss.item():.6f}, grad_norm={grad_norm:.6f}, nonfinite={nonfinite}")
    print(f"[BACKWARD] post-step diff={post_diff:.8e}")
    print(f"[PARAMS] trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    assert full_diff <= 1e-7
    assert missing_diff <= 1e-7
    assert nonfinite == 0
    assert post_diff > 0.0
    print("[SMOKE TEST] PASS")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone DP-PGFA smoke test")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--mlp_ratio", type=float, default=2.66)
    p.add_argument("--compress_ratio", type=int, default=8)
    p.add_argument("--prompt_len", type=int, default=128)
    p.add_argument("--text_dim", type=int, default=768)
    p.add_argument("--text_tower_path", type=str, default=None)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run_smoke_test(build_argparser().parse_args())
