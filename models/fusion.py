# -*- coding: utf-8 -*-
"""CUDM text-gated fusion module for PET-CT feature fusion.

The module follows a clean three-step design:
1. CUDM: disentangle common anatomical background and unique lesion residuals.
2. Text-gated Query: use a fixed BiomedCLIP text prior to clean tumor features.
3. Lightweight attention: clean Query attends to tumor-feature Key/Value.
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

LOCAL_BIOMEDCLIP_DIR = "/root/autodl-tmp/mkd-main/new-train/pretrained/biomedclip_model"
LOCAL_BIOMEDBERT_TEXT_TOWER_DIR = "/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower"
BIOMEDCLIP_MODEL = "biomedclip_local"
BIOMEDCLIP_WEIGHTS = os.path.join(LOCAL_BIOMEDCLIP_DIR, "open_clip_pytorch_model.bin")

TUMOR_PROMPT = (
    "A focal bright tumor hotspot with abnormal metabolic activity "
    "and clear lesion boundary."
)


def _register_local_biomedclip_config():
    src_config = os.path.join(LOCAL_BIOMEDCLIP_DIR, "open_clip_config.json")
    if not os.path.exists(src_config):
        raise FileNotFoundError(f"BiomedCLIP config not found: {src_config}")
    with open(src_config, "r") as f:
        cfg = json.load(f)
    cfg["model_cfg"]["text_cfg"]["hf_model_name"] = LOCAL_BIOMEDBERT_TEXT_TOWER_DIR
    cfg["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = LOCAL_BIOMEDBERT_TEXT_TOWER_DIR
    cfg_dir = os.path.join(tempfile.gettempdir(), "open_clip_local_biomedclip")
    os.makedirs(cfg_dir, exist_ok=True)
    dst = os.path.join(cfg_dir, f"{BIOMEDCLIP_MODEL}.json")
    with open(dst, "w") as f:
        json.dump(cfg["model_cfg"], f)
    open_clip.add_model_config(cfg_dir)


@torch.no_grad()
def _encode_prompt(prompt):
    if open_clip is None:
        raise ImportError("pip install open_clip_torch") from _OPEN_CLIP_ERROR
    _register_local_biomedclip_config()
    model, _, _ = open_clip.create_model_and_transforms(
        BIOMEDCLIP_MODEL,
        pretrained=BIOMEDCLIP_WEIGHTS,
        pretrained_hf=False,
    )
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL)
    model.eval()
    tokens = tokenizer([prompt])
    text = model.encode_text(tokens).float()
    text = F.normalize(text, dim=-1).cpu()
    del model
    return text


class FixedPromptEmbedding(nn.Module):
    def __init__(self, prompt=TUMOR_PROMPT):
        super().__init__()
        safe_name = "cudm_tumor_prompt_embedding.pt"
        cache_path = os.path.join(LOCAL_BIOMEDCLIP_DIR, safe_name)
        if os.path.exists(cache_path):
            text_code = torch.load(cache_path, map_location="cpu")
        else:
            text_code = _encode_prompt(prompt)
            torch.save(text_code, cache_path)
        self.register_buffer("text_code", text_code.float(), persistent=True)

    def forward(self, batch_size, device):
        return self.text_code.to(device).expand(batch_size, -1)


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class LiteFFN(nn.Module):
    def __init__(self, channels, expansion=0.25):
        super().__init__()
        hidden = max(8, int(channels * expansion))
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class LiteChannelAttention(nn.Module):
    """Restormer-style transposed channel attention with QKV + softmax."""

    def __init__(self, channels, heads=4, attn_ratio=0.125):
        super().__init__()
        self.heads = heads
        attn_c = max(heads, int(channels * attn_ratio))
        attn_c = max(heads, (attn_c // heads) * heads)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.q = nn.Conv2d(channels, attn_c, 1, bias=False)
        self.kv = nn.Conv2d(channels, attn_c * 2, 1, bias=False)
        self.q_dw = nn.Conv2d(attn_c, attn_c, 3, padding=1, groups=attn_c, bias=False)
        self.kv_dw = nn.Conv2d(attn_c * 2, attn_c * 2, 3, padding=1, groups=attn_c * 2, bias=False)
        self.proj = nn.Conv2d(attn_c, channels, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, query, kv_feature):
        _, _, h, w = query.shape
        q = self.q_dw(self.q(query))
        k, v = self.kv_dw(self.kv(kv_feature)).chunk(2, dim=1)
        q = rearrange(q, "b (hd c) h w -> b hd c (h w)", hd=self.heads)
        k = rearrange(k, "b (hd c) h w -> b hd c (h w)", hd=self.heads)
        v = rearrange(v, "b (hd c) h w -> b hd c (h w)", hd=self.heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.temperature, dim=-1)
        out = attn @ v
        out = rearrange(out * self.scale, "b hd c (h w) -> b (hd c) h w", hd=self.heads, h=h, w=w)
        return self.proj(out)


class TextGuidedChannelAttention(nn.Module):
    """Text-guided channel attention for cleaning tumor query features."""

    def __init__(self, channels, text_dim=512, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(channels * 2, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.fusion = nn.Sequential(
            nn.Linear(channels * 4, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, tumor_feature, text_code):
        b, c, _, _ = tumor_feature.shape
        avg_desc = tumor_feature.mean(dim=(2, 3))
        max_desc = tumor_feature.amax(dim=(2, 3))
        visual = self.visual_proj(torch.cat([avg_desc, max_desc], dim=1))
        text = self.text_proj(text_code)
        interaction = visual * text
        difference = torch.abs(visual - text)
        gate = self.fusion(torch.cat([visual, text, interaction, difference], dim=1))
        return tumor_feature * gate.view(b, c, 1, 1), gate


class MutualGate(nn.Module):
    """Estimate commonality via mutual channel gating between two modalities."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 4)
        self.ct_squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.pet_squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, ct_feature, pet_feature):
        ct_w = torch.sigmoid(self.pet_squeeze(pet_feature))
        pet_w = torch.sigmoid(self.ct_squeeze(ct_feature))
        mutual = ct_feature * ct_w + pet_feature * pet_w
        common = self.spatial(mutual)
        return common


class CUDMTextGate(nn.Module):
    """Commonality-Uniqueness Disentanglement with text-gated attention."""

    def __init__(self, channels, heads=4, text_dim=512, attn_ratio=0.125):
        super().__init__()
        self.mutual_gate = MutualGate(channels)
        self.unique_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.tgca = TextGuidedChannelAttention(
            channels, text_dim=text_dim, reduction=8,
        )
        self.nq = LayerNorm2d(channels)
        self.nkv = LayerNorm2d(channels)
        self.nf = LayerNorm2d(channels)
        self.attn = LiteChannelAttention(channels, heads=heads, attn_ratio=attn_ratio)
        self.ffn = LiteFFN(channels, expansion=0.25)
        self.out_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, pet_feature, ct_feature, text_code):
        common = self.mutual_gate(ct_feature, pet_feature)
        unique_pet = pet_feature - common
        unique_ct = ct_feature - common
        tumor = self.unique_fuse(torch.cat([unique_pet, unique_ct], dim=1))

        clean_query, _ = self.tgca(tumor, text_code)

        attn_out = self.attn(self.nq(clean_query), self.nkv(tumor))
        enhanced = attn_out + self.ffn(self.nf(attn_out))

        base = pet_feature + ct_feature
        gate = torch.sigmoid(self.out_gate)
        out = base * (1.0 - gate) + (base + enhanced) * gate
        return out, {"common": common, "tumor": tumor}


class MultiStageCUDMTextFusion(nn.Module):
    """Apply CUDMTextGate to all encoder stages."""

    def __init__(self, encoder_channels, text_dim=512, heads_per_stage=(1, 2, 4, 8)):
        super().__init__()
        self.text_encoder = FixedPromptEmbedding(prompt=TUMOR_PROMPT)
        self.stages = nn.ModuleList([
            CUDMTextGate(channels=ch, heads=head, text_dim=text_dim, attn_ratio=0.125)
            for ch, head in zip(encoder_channels, heads_per_stage)
        ])

    def forward(self, ct_feats, pet_feats):
        b = ct_feats[0].shape[0]
        device = ct_feats[0].device
        text_code = self.text_encoder(b, device)
        fused = []
        aux = []
        for stage, ct_f, pet_f in zip(self.stages, ct_feats, pet_feats):
            out, stage_aux = stage(pet_f, ct_f, text_code)
            fused.append(out)
            aux.append(stage_aux)
        return fused, aux