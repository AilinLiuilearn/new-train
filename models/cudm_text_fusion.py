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
    "A focal lesion region showing abnormal tissue density on CT "
    "with elevated metabolic uptake on PET imaging."
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
        import hashlib
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        safe_name = f"cudm_prompt_embedding_{prompt_hash}.pt"
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


class TextConditionedChannelGate(nn.Module):
    """Text-conditioned channel selection: text embedding directly produces channel weights."""

    def __init__(self, channels, text_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(text_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, tumor_feature, text_code):
        b, c, _, _ = tumor_feature.shape
        w = self.gate(text_code).view(b, c, 1, 1)
        return tumor_feature * w, w.squeeze(-1).squeeze(-1)


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
    """CUDM fusion on already-composed CT/PET features with text-conditioned query."""

    def __init__(self, channels, heads=4, text_dim=512, attn_ratio=0.125,
                 disable_text=False):
        super().__init__()
        self.disable_text = disable_text
        self.mutual_gate = MutualGate(channels)
        self.unique_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.tgca = None if disable_text else TextConditionedChannelGate(
            channels, text_dim=text_dim,
        )
        self.nq = LayerNorm2d(channels)
        self.nkv = LayerNorm2d(channels)
        self.nf = LayerNorm2d(channels)
        self.attn = LiteChannelAttention(channels, heads=heads, attn_ratio=attn_ratio)
        self.ffn = LiteFFN(channels, expansion=0.25)
        self.out_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, pet_feature, ct_feature, text_code):
        if pet_feature is None:
            pet_feature = ct_feature

        common = self.mutual_gate(ct_feature, pet_feature)
        unique_pet = pet_feature - common
        unique_ct = ct_feature - common
        tumor = self.unique_fuse(torch.cat([unique_pet, unique_ct], dim=1))

        if self.disable_text:
            clean_query = tumor
        else:
            clean_query, _ = self.tgca(tumor, text_code)

        attn_out = self.attn(self.nq(clean_query), self.nkv(tumor))
        enhanced = attn_out + self.ffn(self.nf(attn_out))

        base = pet_feature + ct_feature
        gate = torch.sigmoid(self.out_gate)
        out = base * (1.0 - gate) + (base + enhanced) * gate

        return out, {
            "common": common,
            "tumor": tumor,
            "unique_pet": unique_pet,
            "unique_ct": unique_ct,
        }


class StageDisentangleBlock(nn.Module):
    """Stage-wise ShaSpec-style shared/specific disentanglement for CT/PET features."""

    def __init__(self, channels):
        super().__init__()
        self.ct_common = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.pet_common = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.ct_specific = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.pet_specific = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.compose = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.reconstruct = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, ct_feature, pet_feature=None):
        pet_present = pet_feature is not None
        ct_common = self.ct_common(ct_feature)
        ct_specific = self.ct_specific(ct_feature)

        if pet_present:
            pet_common = self.pet_common(pet_feature)
            pet_specific = self.pet_specific(pet_feature)
        else:
            pet_common = ct_common
            pet_specific = torch.zeros_like(ct_specific)

        ct_residual = self.compose(torch.cat([ct_common, ct_specific], dim=1))
        pet_residual = self.compose(torch.cat([pet_common, pet_specific], dim=1))
        ct_composed = ct_common + ct_residual
        pet_composed = pet_common + pet_residual if pet_present else pet_common

        ct_recon = self.reconstruct(torch.cat([ct_common, ct_specific], dim=1))
        pet_recon = self.reconstruct(torch.cat([pet_common, pet_specific], dim=1)) if pet_present else None

        aux = {
            "ct_feature": ct_feature,
            "pet_feature": pet_feature,
            "ct_common": ct_common,
            "pet_common": pet_common,
            "ct_specific": ct_specific,
            "pet_specific": pet_specific,
            "ct_composed": ct_composed,
            "pet_composed": pet_composed,
            "ct_recon": ct_recon,
            "pet_recon": pet_recon,
            "pet_present": pet_present,
        }
        return ct_composed, pet_composed, aux


class MultiStageShaSpecDisentangle(nn.Module):
    """Apply ShaSpec-style disentanglement to all encoder stages."""

    def __init__(self, encoder_channels):
        super().__init__()
        self.stages = nn.ModuleList([
            StageDisentangleBlock(ch) for ch in encoder_channels
        ])

    def forward(self, ct_feats, pet_feats=None):
        pet_missing = pet_feats is None
        ct_out, pet_out, aux = [], [], []
        for i, (stage, ct_f) in enumerate(zip(self.stages, ct_feats)):
            pet_f = None if pet_missing else pet_feats[i]
            ct_c, pet_c, stage_aux = stage(ct_f, pet_f)
            ct_out.append(ct_c)
            pet_out.append(pet_c)
            aux.append(stage_aux)
        return ct_out, pet_out, aux


class MultiStageCUDMTextFusion(nn.Module):
    """Apply CUDMTextGate to all encoder stages."""

    def __init__(
        self,
        encoder_channels,
        text_dim=512,
        heads_per_stage=(1, 2, 4, 8),
        disable_text=False,
    ):
        super().__init__()
        self.disable_text = disable_text
        self.text_encoder = None if disable_text else FixedPromptEmbedding(prompt=TUMOR_PROMPT)
        self.stages = nn.ModuleList([
            CUDMTextGate(
                channels=ch,
                heads=head,
                text_dim=text_dim,
                attn_ratio=0.125,
                disable_text=disable_text,
            )
            for ch, head in zip(encoder_channels, heads_per_stage)
        ])

    def forward(self, ct_feats, pet_feats):
        b = ct_feats[0].shape[0]
        device = ct_feats[0].device
        text_code = None if self.disable_text else self.text_encoder(b, device)
        pet_missing = pet_feats is None
        fused = []
        aux = []
        for i, (stage, ct_f) in enumerate(zip(self.stages, ct_feats)):
            pet_f = None if pet_missing else pet_feats[i]
            out, stage_aux = stage(pet_f, ct_f, text_code)
            fused.append(out)
            aux.append(stage_aux)
        return fused, aux
