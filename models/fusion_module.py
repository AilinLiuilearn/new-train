# -*- coding: utf-8 -*-
"""
Text-guided Cross-modal Privileged Fusion Module (TCPM-v2).

Improvements over the original TCPM:
  1. Parallel dual-branch query construction:
     - Channel branch: SE-based topk channel selection
     - Spatial branch: text-guided spatial attention
  2. Two branches summed to form enriched Q for cross-attention.
  3. Cleaner gate mechanism with learnable residual scaling.
"""

import json
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import open_clip
except ImportError as e:
    open_clip = None
    _OPEN_CLIP_ERROR = e

LOCAL_BIOMEDCLIP_DIR = (
    "/root/autodl-tmp/mkd-main/new-train/pretrained/biomedclip_model"
)
LOCAL_BIOMEDBERT_TEXT_TOWER_DIR = (
    "/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower"
)
BIOMEDCLIP_MODEL = "biomedclip_local"
BIOMEDCLIP_WEIGHTS = os.path.join(
    LOCAL_BIOMEDCLIP_DIR, "open_clip_pytorch_model.bin"
)
PROMPT = (
    "On this PET-CT lung image, the tumor areas are characterized "
    "by a focal bright hotspot indicating abnormal metabolic activity "
    "superimposed on a soft tissue mass."
)


# ---------------------------------------------------------------------------
# BiomedCLIP text encoder (frozen)
# ---------------------------------------------------------------------------

def _register_local_biomedclip_config():
    src_config = os.path.join(LOCAL_BIOMEDCLIP_DIR, "open_clip_config.json")
    if not os.path.exists(src_config):
        raise FileNotFoundError(f"Config not found: {src_config}")
    with open(src_config, "r") as f:
        cfg = json.load(f)
    cfg["model_cfg"]["text_cfg"]["hf_model_name"] = (
        LOCAL_BIOMEDBERT_TEXT_TOWER_DIR
    )
    cfg["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = (
        LOCAL_BIOMEDBERT_TEXT_TOWER_DIR
    )
    cfg_dir = os.path.join(
        tempfile.gettempdir(), "open_clip_local_biomedclip"
    )
    os.makedirs(cfg_dir, exist_ok=True)
    dst = os.path.join(cfg_dir, f"{BIOMEDCLIP_MODEL}.json")
    with open(dst, "w") as f:
        json.dump(cfg["model_cfg"], f)
    open_clip.add_model_config(cfg_dir)


class BiomedCLIPTextEncoder(nn.Module):
    def __init__(self, prompt=PROMPT, freeze=True):
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "pip install open_clip_torch"
            ) from _OPEN_CLIP_ERROR
        _register_local_biomedclip_config()
        self.model, _, _ = open_clip.create_model_and_transforms(
            BIOMEDCLIP_MODEL,
            pretrained=BIOMEDCLIP_WEIGHTS,
            pretrained_hf=False,
        )
        self.tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL)
        self.register_buffer(
            "tokens", self.tokenizer([prompt]), persistent=False
        )
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, batch_size, device):
        tokens = self.tokens.to(device).expand(batch_size, -1)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            text_code = self.model.encode_text(tokens).float()
            text_code = F.normalize(text_code, dim=-1)
        if was_training:
            self.model.train()
        return text_code


class PromptEmbedding(nn.Module):
    """Stores a fixed 512-d BiomedCLIP prompt embedding as a buffer.

    The heavy text encoder is loaded only once during initialization and is
    not kept inside the segmentation model. This avoids counting 195M frozen
    text-encoder parameters and removes text-encoder FLOPs during profiling.
    """

    def __init__(self, prompt=PROMPT):
        super().__init__()
        cache_path = os.path.join(
            LOCAL_BIOMEDCLIP_DIR, "tcpmprompt_embedding.pt"
        )
        if os.path.exists(cache_path):
            text_code = torch.load(cache_path, map_location="cpu")
        else:
            encoder = BiomedCLIPTextEncoder(prompt=prompt, freeze=True)
            with torch.no_grad():
                text_code = encoder(1, torch.device("cpu")).cpu()
            torch.save(text_code, cache_path)
            del encoder
        self.register_buffer("text_code", text_code.float(), persistent=True)

    def forward(self, batch_size, device):
        return self.text_code.to(device).expand(batch_size, -1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(
            x.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2).contiguous()


class LiteFeedForward(nn.Module):
    def __init__(self, c, expansion=0.5):
        super().__init__()
        h = max(8, int(c * expansion))
        self.net = nn.Sequential(
            nn.Conv2d(c, h, 1, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.Conv2d(h, h, 3, padding=1, groups=h, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.Conv2d(h, c, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# TopM Cross-Attention (Restormer-style, channel-dim with sparsity)
# ---------------------------------------------------------------------------

class TopMCrossAttention(nn.Module):
    def __init__(self, c, heads=4, keep_ratio=0.9, attn_ratio=0.5):
        super().__init__()
        self.heads = heads
        self.keep_ratio = keep_ratio
        attn_c = max(
            heads, (max(heads, int(c * attn_ratio)) // heads) * heads
        )
        self.attn_c = attn_c
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.q_reduce = nn.Conv2d(c, attn_c, 1, bias=False)
        self.kv_reduce = nn.Conv2d(c, attn_c * 2, 1, bias=False)
        self.q_dw = nn.Conv2d(
            attn_c, attn_c, 3, padding=1, groups=attn_c, bias=False
        )
        self.kv_dw = nn.Conv2d(
            attn_c * 2, attn_c * 2, 3, padding=1,
            groups=attn_c * 2, bias=False,
        )
        self.proj = nn.Conv2d(attn_c, c, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x_q, x_kv):
        _, _, h, w = x_q.shape
        q = self.q_dw(self.q_reduce(x_q))
        k, v = self.kv_dw(self.kv_reduce(x_kv)).chunk(2, dim=1)
        q = rearrange(q, "b (hd c) h w -> b hd c (h w)", hd=self.heads)
        k = rearrange(k, "b (hd c) h w -> b hd c (h w)", hd=self.heads)
        v = rearrange(v, "b (hd c) h w -> b hd c (h w)", hd=self.heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        keep = max(1, int(q.shape[2] * self.keep_ratio))
        idx = torch.topk(attn, keep, dim=-1).indices
        mask = torch.zeros_like(attn, dtype=torch.bool).scatter_(
            -1, idx, True
        )
        attn = torch.where(
            mask, attn,
            torch.full_like(attn, torch.finfo(attn.dtype).min),
        )
        out = F.softmax(attn, dim=-1) @ v
        out = rearrange(
            out * self.scale,
            "b hd c (h w) -> b (hd c) h w",
            hd=self.heads, h=h, w=w,
        )
        return self.proj(out)


# ---------------------------------------------------------------------------
# Text-guided Spatial Attention Branch (NEW)
# ---------------------------------------------------------------------------

class TextGuidedSpatialAttention(nn.Module):
    """Spatial attention guided by BiomedCLIP text embeddings.

    Text embedding modulates both channel-wise weighting and spatial
    attention to produce position-aware, semantically-guided features.
    """

    def __init__(self, ch_dim, text_dim=512, reduction=8):
        super().__init__()
        mid = max(8, ch_dim // reduction)
        self.text_to_channel_gate = nn.Sequential(
            nn.Linear(text_dim, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch_dim, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(ch_dim * 2, ch_dim, 1, groups=ch_dim, bias=False),
            nn.BatchNorm2d(ch_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_dim, ch_dim, 3, padding=1,
                      groups=ch_dim, bias=False),
            nn.BatchNorm2d(ch_dim),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(ch_dim, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1,
                      groups=mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=True),
        )
        nn.init.constant_(self.spatial_attn[-1].bias, 0.0)

    def forward(self, pet_feat, ct_feat, text_code):
        b, c, h, w = pet_feat.shape
        ch_gate = self.text_to_channel_gate(text_code).view(b, c, 1, 1)
        combined = self.spatial_conv(
            torch.cat([pet_feat, ct_feat], dim=1)
        )
        combined = combined * ch_gate
        sp_weight = torch.sigmoid(self.spatial_attn(combined))
        return combined * sp_weight


# ---------------------------------------------------------------------------
# Channel Selection Branch (cleaned from original TCPM)
# ---------------------------------------------------------------------------

class ChannelSelectionBranch(nn.Module):
    """SE topk channel selection + text-guided reordering."""

    def __init__(self, ch_dim, text_dim=512, topk_ratio=0.125,
                 se_ratio=32):
        super().__init__()
        self.dim = ch_dim
        self.topk = max(1, int(2 * ch_dim * topk_ratio))
        hidden = max(4, 2 * ch_dim // se_ratio)
        self.se_fc = nn.Sequential(
            nn.Linear(2 * ch_dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * ch_dim, bias=False),
            nn.Sigmoid(),
        )
        text_hidden = min(256, text_dim)
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, text_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(text_hidden, 2 * ch_dim, bias=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pick_conv = nn.Sequential(
            nn.Conv2d(
                self.topk, self.topk, 3, padding=1,
                groups=self.topk, bias=False,
            ),
            nn.Conv2d(self.topk, 2 * ch_dim, 1, bias=False),
        )
        self.out_conv = nn.Conv2d(2 * ch_dim, ch_dim, 1,
                                  groups=ch_dim, bias=False)

    def forward(self, pet_feat, ct_feat, text_code):
        b, _, h, w = pet_feat.shape
        both = torch.cat([pet_feat, ct_feat], dim=1)
        weight = self.se_fc(self.pool(both).flatten(1))
        idx = torch.topk(weight, self.topk, dim=1).indices
        idx_e = idx[:, :, None, None].expand(-1, -1, h, w)
        picked = self.pick_conv(both.gather(1, idx_e))
        tidx = torch.topk(
            self.text_fc(text_code), 2 * self.dim, dim=1
        ).indices
        picked = picked[
            torch.arange(b, device=picked.device)[:, None], tidx
        ]
        return self.out_conv(picked)


# ---------------------------------------------------------------------------
# Lightweight gated fusion for high-resolution stages
# ---------------------------------------------------------------------------

class LiteGatedFusion(nn.Module):
    """Lightweight text-guided gated fusion for high-res stages."""

    def __init__(self, ch_dim, text_dim=512):
        super().__init__()
        mid = max(8, ch_dim // 8)
        self.text_gate = nn.Sequential(
            nn.Linear(text_dim, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch_dim, bias=False),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(ch_dim * 2, ch_dim, 1, bias=False),
            nn.BatchNorm2d(ch_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_dim, ch_dim, 3, padding=1,
                      groups=ch_dim, bias=False),
            nn.BatchNorm2d(ch_dim),
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, pet_feat, ct_feat, text_code):
        b, c, _, _ = pet_feat.shape
        t_gate = self.text_gate(text_code).view(b, c, 1, 1)
        combined = self.fuse(torch.cat([pet_feat, ct_feat], dim=1))
        combined = combined * t_gate
        base = pet_feat + ct_feat
        g = torch.sigmoid(self.gate)
        return base + g * (combined - base), combined


# ---------------------------------------------------------------------------
# TCPMv2: Channel + Text-Spatial parallel → Q → CrossAttn
# ---------------------------------------------------------------------------

class TCPMv2(nn.Module):
    """Text-guided Cross-modal Privileged Fusion Module v2.

    ┌── ChannelSelectionBranch ── q_ch ──┐
    │                                    ├─ + → Q
    └── TextGuidedSpatialAttn ── q_sp ──┘
                                         ↓
                          TopMCrossAttn(Q, Ref)
                                         ↓
                                    FFN + residual
                                         ↓
                               gate·tcpm + (1-gate)·base
    """

    def __init__(self, ch_dim, num_heads=4, text_dim=512,
                 topk_ratio=0.125, se_ratio=32, attn_ratio=0.125,
                 gate_init=0.0):
        super().__init__()
        self.channel_branch = ChannelSelectionBranch(
            ch_dim, text_dim=text_dim,
            topk_ratio=topk_ratio, se_ratio=se_ratio,
        )
        self.spatial_branch = TextGuidedSpatialAttention(
            ch_dim, text_dim=text_dim, reduction=8,
        )
        self.ref_conv = nn.Conv2d(2 * ch_dim, ch_dim, 1,
                                  groups=ch_dim, bias=False)
        self.n1 = LayerNorm2d(ch_dim)
        self.n2 = LayerNorm2d(ch_dim)
        self.n3 = LayerNorm2d(ch_dim)
        self.attn = TopMCrossAttention(
            ch_dim, num_heads, attn_ratio=attn_ratio,
        )
        self.ffn = LiteFeedForward(ch_dim, expansion=0.25)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, pet_feat, ct_feat, text_code):
        q_ch = self.channel_branch(pet_feat, ct_feat, text_code)
        q_sp = self.spatial_branch(pet_feat, ct_feat, text_code)
        q = q_ch + q_sp

        ref = self.ref_conv(
            torch.cat([pet_feat, ct_feat], dim=1)
        )
        att = self.attn(self.n1(q), self.n2(ref))
        tcpm_out = att + self.ffn(self.n3(att))

        base = pet_feat + ct_feat
        g = torch.sigmoid(self.gate)
        fused = base + g * (tcpm_out - base)
        return fused, ref


# ---------------------------------------------------------------------------
# Multi-stage wrapper
# ---------------------------------------------------------------------------

class MultiStageTCPMv2Fusion(nn.Module):
    """Applies lightweight TCPMv2 at all encoder stages."""

    def __init__(self, encoder_channels, text_dim=512,
                 heads_per_stage=(1, 2, 4, 8)):
        super().__init__()
        self.text_encoder = PromptEmbedding(prompt=PROMPT)
        self.stages = nn.ModuleList([
            TCPMv2(
                ch_dim=ch,
                num_heads=h,
                text_dim=text_dim,
                topk_ratio=0.125,
                se_ratio=32,
                attn_ratio=0.125,
            )
            for ch, h in zip(encoder_channels, heads_per_stage)
        ])

    def forward(self, ct_feats, pet_feats):
        b = ct_feats[0].shape[0]
        device = ct_feats[0].device
        text_code = self.text_encoder(b, device)
        fused = []
        for stage, ct_f, pet_f in zip(
            self.stages, ct_feats, pet_feats
        ):
            f, _ = stage(pet_f, ct_f, text_code)
            fused.append(f)
        return fused
